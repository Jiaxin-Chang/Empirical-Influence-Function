"""
Train-annotation viewer API.

Browse a compact graphsignal JSONL (input_ids + attention_edges + annotations).
Edits (add/delete edges) never rewrite the source file — they upsert into a
separate continue-train JSONL (ANNOTATION_CONTINUE_TRAIN_DATA).

Run:
  cd tools/annotation-viewer
  pip install -r server/requirements.txt
  python -m server.main --data ../../smoke_train_data.jsonl \\
    --continue-data ../../continue_annotated_subset.jsonl

Optional ALTI saliency (needs GPU + model weights):
  python -m server.main --data ... --model ../../../code-corr-annotation/models/Qwen2.5-Coder-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

REPO_ROOT = Path(__file__).resolve().parents[3]
VIEWER_ROOT = Path(__file__).resolve().parents[1]


def _load_dotenv(path: Path, *, override: bool = False) -> None:
    """Minimal .env loader (KEY=VALUE)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if not override and key in os.environ:
            continue
        os.environ[key] = value


# Prefer repo-root eif_api.env (shared with ttav_bundle_api); local .env is deprecated.
_load_dotenv(REPO_ROOT / "eif_api.env", override=True)
_load_dotenv(REPO_ROOT / ".env", override=False)
_load_dotenv(VIEWER_ROOT / ".env", override=False)


def _resolve_env_path(raw: str) -> Path:
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    for root in (REPO_ROOT, VIEWER_ROOT, Path.cwd()):
        cand = (root / p).resolve()
        if cand.exists():
            return cand
    return (REPO_ROOT / p).resolve()


def _default_data_path() -> Path:
    raw = (os.environ.get("ANNOTATION_TRAIN_DATA") or "").strip()
    if not raw:
        # Fall back to bank train if compact path not set.
        raw = (os.environ.get("EIF_TRAIN_DATA") or "").strip()
    if not raw:
        raise SystemExit(
            "ANNOTATION_TRAIN_DATA is not set. "
            "Put it in repo-root eif_api.env (or pass --data)."
        )
    return _resolve_env_path(raw)


def _default_continue_path() -> Path | None:
    """Writable small annotated subset used by continue-train (never the full bank)."""
    raw = (os.environ.get("ANNOTATION_CONTINUE_TRAIN_DATA") or "").strip()
    if not raw:
        return None
    return _resolve_env_path(raw)


def _default_corpus_path() -> Path | None:
    raw = (
        os.environ.get("EIF_LLM_TRAIN_CORPUS")
        or os.environ.get("EIF_TRAIN_CORPUS")
        or ""
    ).strip()
    if not raw:
        return None
    return _resolve_env_path(raw)


DEFAULT_DATA = _default_data_path()
DEFAULT_CONTINUE = _default_continue_path()
DEFAULT_CORPUS = _default_corpus_path()
# Tokenizer/decode: use base model path (no separate ANNOTATION_TOKENIZER).
_TOKENIZER_RAW = (os.environ.get("EIF_BASE_MODEL_PATH") or "").strip()
DEFAULT_TOKENIZER = Path(_TOKENIZER_RAW).expanduser() if _TOKENIZER_RAW else Path()

SUBTYPES = [
    "route",  # continue-train attention-routing (default for manual / LLM)
    "bracket",
    "defuse",
    "call",
    "return",
    "type",
    "dataflow",
    "semantic",
    "api",
]

app = FastAPI(title="Train Annotation Viewer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_state_lock = threading.RLock()
_data_path: Path | None = None
_offsets: list[int] = []  # byte offset of each non-empty line; last sentinel = file size
# Writable continue-train subset (add/delete upsert here; source stays read-only).
_continue_path: Path | None = None
_continue_offsets: list[int] = []
_continue_key_to_idx: dict[str, int] = {}
_corpus_path: Path | None = None
_corpus_offsets: list[int] | None = None
_tokenizer = None
_tokenizer_path: str | None = None
_model = None
_model_path: str | None = None
_saliency_cache_dir: Path | None = None
_saliency_cache: dict[tuple[int, int], list[dict[str, Any]]] = {}
# Cross-origin probe FIM payloads from correlation-report (sessionStorage cannot share).
_probe_focus_cache: dict[str, dict[str, Any]] = {}
_PROBE_FOCUS_CACHE_MAX = 64
# GraphSignal corpus auto-annotate previews (accept/reject before writing continue).
_gs_preview_cache: dict[str, dict[str, Any]] = {}
_GS_PREVIEW_CACHE_MAX = 32
_GS_PREVIEW_TTL_SEC = 3600
_llm_sem_preview_cache: dict[str, dict[str, Any]] = {}
# Corpus lines whose display was cleared: skip continue overlay; next persist appends.
_corpus_skip_overlay: set[int] = set()
# Context-hit → re-hollow MID: prep cache (cross-origin from correlation-report) +
# line binding so subsequent encode/edge ops keep the rewritten prompt/response.
_corpus_mid_rewrite_prep: dict[str, dict[str, Any]] = {}
_CORPUS_MID_REWRITE_PREP_MAX = 64
_corpus_mid_rewrite_by_line: dict[str, dict[str, Any]] = {}


def _disk_saliency_path(idx: int) -> Path | None:
    if _saliency_cache_dir is None:
        return None
    return _saliency_cache_dir / f"{idx}.json"


def _load_disk_saliency(idx: int, target: int, top_k: int) -> list[dict[str, Any]] | None:
    """Load precomputed top sources for (sample, target) from cache dir."""
    path = _disk_saliency_path(idx)
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    by_target = payload.get("by_target") or {}
    hits = by_target.get(str(target))
    if hits is None:
        return None
    return list(hits)[: max(1, min(top_k, 20))]


def _saliency_enabled() -> bool:
    return bool(_model_path) or (
        _saliency_cache_dir is not None and _saliency_cache_dir.exists()
    )


# ── Index / IO ────────────────────────────────────────────────────────────────

def _build_offsets(path: Path) -> list[int]:
    offsets: list[int] = []
    with path.open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
        offsets.append(f.tell())  # sentinel = EOF
    return offsets


def _read_sample(idx: int) -> dict[str, Any]:
    """Read raw row from the source (read-only) JSONL by 0-based index."""
    assert _data_path is not None
    if idx < 0 or idx >= len(_offsets) - 1:
        raise HTTPException(404, f"sample index {idx} out of range")
    with _data_path.open("rb") as f:
        f.seek(_offsets[idx])
        raw = f.readline()
    return json.loads(raw.decode("utf-8"))


def _sample_key(obj: dict[str, Any], source_idx: int | None = None) -> str:
    """Stable identity for upsert into the continue subset."""
    keys = _sample_keys(obj, source_idx)
    return keys[0]


def _sample_keys(obj: dict[str, Any], source_idx: int | None = None) -> list[str]:
    """All plausible identity keys for a sample (uid preferred, then fallbacks).

    Older continue rows may have been keyed only by ``source_idx`` while the
    source row also has ``uid`` — trying every key avoids silent overlay miss
    (looks like newly added edges vanished after reopen).
    """
    keys: list[str] = []
    uid = obj.get("uid")
    if isinstance(uid, str) and uid.strip():
        keys.append(f"uid:{uid.strip()}")
    raw_id = obj.get("raw_id")
    if isinstance(raw_id, str) and raw_id.strip():
        keys.append(f"raw_id:{raw_id.strip()}")
    corpus_line = obj.get("source_corpus_line")
    if corpus_line is not None:
        try:
            cl = int(corpus_line)
            if cl >= 0:
                keys.append(f"corpus_line:{cl}")
        except (TypeError, ValueError):
            pass
    task_id = obj.get("task_id")
    if isinstance(task_id, str) and task_id.strip():
        keys.append(f"task_id:{task_id.strip()}")
    stamped = obj.get("source_train_index")
    if isinstance(stamped, int) and stamped >= 0:
        keys.append(f"source_idx:{stamped}")
    if source_idx is not None and source_idx >= 0:
        k = f"source_idx:{source_idx}"
        if k not in keys:
            keys.append(k)
    if not keys:
        ids = obj.get("input_ids") or []
        keys.append(f"ids:{len(ids)}:{hash(tuple(int(x) for x in ids[:64]))}")
    return keys


def _rebuild_continue_index() -> None:
    global _continue_offsets, _continue_key_to_idx
    _continue_key_to_idx = {}
    if _continue_path is None or not _continue_path.is_file():
        _continue_offsets = []
        return
    _continue_offsets = _build_offsets(_continue_path)
    n = max(0, len(_continue_offsets) - 1)
    with _continue_path.open("rb") as f:
        for i in range(n):
            f.seek(_continue_offsets[i])
            raw = f.readline()
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            for key in _sample_keys(obj):
                _continue_key_to_idx[key] = i


def _read_continue_by_idx(idx: int) -> dict[str, Any]:
    assert _continue_path is not None
    if idx < 0 or idx >= len(_continue_offsets) - 1:
        raise HTTPException(404, f"continue sample index {idx} out of range")
    with _continue_path.open("rb") as f:
        f.seek(_continue_offsets[idx])
        raw = f.readline()
    return json.loads(raw.decode("utf-8"))


def _rewrite_jsonl_line(path: Path, offsets: list[int], idx: int, obj: dict[str, Any]) -> list[int]:
    """Replace one JSONL line via temp file; return rebuilt offsets."""
    if idx < 0 or idx >= len(offsets) - 1:
        raise HTTPException(404, f"sample index {idx} out of range")

    start = offsets[idx]
    end = offsets[idx + 1]
    new_line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")

    with path.open("rb") as src, tmp.open("wb") as dst:
        if start:
            dst.write(src.read(start))
        src.seek(end)
        dst.write(new_line)
        dst.write(src.read())

    try:
        os.replace(tmp, path)
    except PermissionError:
        import shutil

        with tmp.open("rb") as src, path.open("wb") as dst:
            shutil.copyfileobj(src, dst)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass

    return _build_offsets(path)


def _append_jsonl_line(path: Path, obj: dict[str, Any]) -> list[int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    with path.open("ab") as f:
        f.write(line)
    return _build_offsets(path)


def _ensure_continue_path() -> Path:
    if _continue_path is None:
        raise HTTPException(
            400,
            "No continue-train dataset configured. Set ANNOTATION_CONTINUE_TRAIN_DATA "
            "in repo-root eif_api.env (or pass --continue-data). "
            "Edits no longer write back to the source train JSONL.",
        )
    return _continue_path


def _lookup_continue(source_idx: int, source_obj: dict[str, Any]) -> tuple[str, int | None]:
    keys = _sample_keys(source_obj, source_idx)
    for key in keys:
        cont_idx = _continue_key_to_idx.get(key)
        if cont_idx is not None:
            return key, cont_idx
    return keys[0], None


def _effective_sample(idx: int) -> tuple[dict[str, Any], bool, str]:
    """Source row overlaid with continue-train edit if present.

    Display uses full viz edge list; continue JSONL ``attention_edges`` may be
    a subset (user-add / weight-bump / llm-auto only).

    Returns (obj, from_continue, sample_key).
    """
    source = _read_sample(idx)
    key, cont_idx = _lookup_continue(idx, source)
    if cont_idx is None:
        obj = _compose_display_obj(source, None, from_continue=False)
        return obj, False, key
    overlay = _read_continue_by_idx(cont_idx)
    obj = _compose_display_obj(source, overlay, from_continue=True)
    return obj, True, key


CONTINUE_CONTRIBS = frozenset({"user_add", "user_bump", "llm_auto"})


def _edge_key(e: dict[str, Any]) -> tuple[int, int, str]:
    return (int(e["src"]), int(e["dst"]), str(e.get("subtype") or ""))


def _normalize_edge(
    e: dict[str, Any],
    *,
    contrib: str | None = None,
    weight: float | None = None,
) -> dict[str, Any]:
    try:
        w = float(weight if weight is not None else e.get("weight", 1.0))
    except (TypeError, ValueError):
        w = 1.0
    if w <= 0:
        w = 1.0
    c = contrib if contrib is not None else str(e.get("contrib") or "source")
    out = {
        "src": int(e["src"]),
        "dst": int(e["dst"]),
        "subtype": str(e.get("subtype") or ""),
        "weight": max(1.0, w),
        "contrib": c,
    }
    if e.get("reason"):
        out["reason"] = e["reason"]
    if e.get("source"):
        out["source"] = e["source"]
    return out


def _is_continue_edge(e: dict[str, Any]) -> bool:
    return str(e.get("contrib") or "") in CONTINUE_CONTRIBS


def _source_baseline_edges(source: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in source.get("attention_edges") or []:
        if isinstance(e, dict) and "src" in e and "dst" in e:
            out.append(_normalize_edge(e, contrib="source"))
    return out


def _viz_and_continue_from_overlay(
    source: dict[str, Any],
    overlay: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (viz_edges, continue_edges) for a continue overlay row."""
    raw_viz = overlay.get("viz_attention_edges")
    raw_cont = overlay.get("attention_edges") or []

    if isinstance(raw_viz, list):
        viz = [_normalize_edge(e) for e in raw_viz if isinstance(e, dict) and "src" in e and "dst" in e]
        cont = [
            _normalize_edge(e)
            for e in raw_cont
            if isinstance(e, dict) and "src" in e and "dst" in e and _is_continue_edge(_normalize_edge(e))
        ]
        # If continue edges lack contrib tags (partial write), keep listed ones as user_add.
        if raw_cont and not cont:
            cont = [
                _normalize_edge(e, contrib=str(e.get("contrib") or "user_add"))
                for e in raw_cont
                if isinstance(e, dict) and "src" in e and "dst" in e
            ]
        return viz, cont

    # Legacy continue rows stored the full display list in attention_edges.
    # Treat them as viz; only keep edges already tagged for continue, else none
    # (force user to re-add / bump under the new policy).
    legacy = [
        _normalize_edge(e)
        for e in raw_cont
        if isinstance(e, dict) and "src" in e and "dst" in e
    ]
    cont = [e for e in legacy if _is_continue_edge(e)]
    if not cont and legacy:
        # Old files had no contrib field — do not silently continue-train on all
        # inherited edges; viz still shows the saved full list.
        viz = [_normalize_edge(e, contrib=str(e.get("contrib") or "source")) for e in legacy]
        return viz, []
    return legacy, cont


def _compose_display_obj(
    source: dict[str, Any],
    overlay: dict[str, Any] | None,
    *,
    from_continue: bool,
) -> dict[str, Any]:
    if overlay is None:
        obj = dict(source)
        obj["attention_edges"] = _source_baseline_edges(source)
        obj["_continue_edge_count"] = 0
        return obj
    viz, cont = _viz_and_continue_from_overlay(source, overlay)
    merged = dict(source)
    merged.update(overlay)
    merged["attention_edges"] = viz
    merged["_continue_edge_count"] = len(cont)
    merged["_from_continue"] = from_continue
    return merged


def _upsert_continue(source_idx: int, obj: dict[str, Any], *, force_insert: bool = False) -> dict[str, Any]:
    """Write edited sample into the continue subset (insert or replace by key)."""
    global _continue_offsets
    path = _ensure_continue_path()
    obj = dict(obj)
    if obj.get("source_corpus_line") is not None:
        obj.pop("source_train_index", None)
        if _corpus_path is not None and not obj.get("source_corpus_path"):
            obj["source_corpus_path"] = str(_corpus_path)
    else:
        obj["source_train_index"] = int(source_idx)
        if _data_path is not None:
            obj["source_train_path"] = str(_data_path)
    keys = _sample_keys(obj, source_idx)
    existing: int | None = None
    if not force_insert:
        for k in keys:
            if k in _continue_key_to_idx:
                existing = _continue_key_to_idx[k]
                break

    if existing is None or force_insert:
        _continue_offsets = _append_jsonl_line(path, obj)
        cont_idx = max(0, len(_continue_offsets) - 2)
        action = "inserted"
    else:
        _continue_offsets = _rewrite_jsonl_line(path, _continue_offsets, existing, obj)
        _rebuild_continue_index()
        cont_idx = _continue_key_to_idx.get(keys[0], existing)
        action = "updated"

    for k in keys:
        _continue_key_to_idx[k] = cont_idx

    _saliency_cache.clear()
    return {
        "ok": True,
        "action": action,
        "key": keys[0],
        "continue_path": str(path),
        "n_continue": max(0, len(_continue_offsets) - 1),
        "n_continue_edges": len(obj.get("attention_edges") or []),
    }


def _build_continue_row(
    source_idx: int,
    source: dict[str, Any],
    *,
    viz_edges: list[dict[str, Any]],
    continue_edges: list[dict[str, Any]],
    corpus_line: int | None = None,
    uid_override: str | None = None,
    duplicate_of: str | None = None,
) -> dict[str, Any]:
    """Build a continue JSONL row (not persisted)."""
    obj = dict(source)
    obj.pop("_continue_edge_count", None)
    obj.pop("_from_continue", None)
    if corpus_line is not None:
        obj["source_corpus_line"] = int(corpus_line)
        if _corpus_path is not None:
            obj["source_corpus_path"] = str(_corpus_path)
    if uid_override:
        obj["uid"] = uid_override
        obj["raw_id"] = uid_override
    if duplicate_of:
        obj["duplicate_of"] = duplicate_of
    viz_n = [_normalize_edge(e) for e in viz_edges]
    cont_n = [
        _normalize_edge(e)
        for e in continue_edges
        if _is_continue_edge(_normalize_edge(e))
    ]
    by_key = {_edge_key(e): e for e in viz_n}
    for e in cont_n:
        by_key[_edge_key(e)] = e
    viz_n = list(by_key.values())
    obj["viz_attention_edges"] = viz_n
    obj["attention_edges"] = cont_n
    anns = []
    for e in viz_n:
        anns.append({
            "token_i_idx": e["src"],
            "token_j_idx": e["dst"],
            "subtype": e["subtype"],
            "source": e.get("source") or e.get("contrib") or "Manual",
            "weight": e.get("weight", 1.0),
            "contrib": e.get("contrib"),
        })
    obj["annotations"] = anns
    _sync_meta(obj)
    meta = obj.get("annotation_meta")
    if isinstance(meta, dict):
        meta["continue_attention_edges"] = len(cont_n)
        meta["viz_attention_edges"] = len(viz_n)
    return obj


def _append_continue_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Append one or more rows to continue JSONL (always insert, never upsert)."""
    global _continue_offsets
    if not rows:
        raise HTTPException(400, "no rows to append")
    path = _ensure_continue_path()
    for obj in rows:
        _continue_offsets = _append_jsonl_line(path, obj)
    _rebuild_continue_index()
    _saliency_cache.clear()
    return {
        "ok": True,
        "action": "appended",
        "n_appended": len(rows),
        "continue_path": str(path),
        "n_continue": max(0, len(_continue_offsets) - 1),
    }


def _write_continue_payload(
    source_idx: int,
    source: dict[str, Any],
    *,
    viz_edges: list[dict[str, Any]],
    continue_edges: list[dict[str, Any]],
    corpus_line: int | None = None,
) -> dict[str, Any]:
    """Persist continue row: attention_edges = continue-only; viz_* = display."""
    force_insert = False
    uid_override = None
    if corpus_line is not None and int(corpus_line) in _corpus_skip_overlay:
        force_insert = True
        base_uid = str(source.get("uid") or source.get("task_id") or f"corpus_line_{corpus_line}")
        uid_override = f"{base_uid}::new{uuid.uuid4().hex[:8]}"
    obj = _build_continue_row(
        source_idx,
        source,
        viz_edges=viz_edges,
        continue_edges=continue_edges,
        corpus_line=corpus_line,
        uid_override=uid_override,
    )
    persist = _upsert_continue(source_idx, obj, force_insert=force_insert)
    if force_insert and corpus_line is not None:
        _corpus_skip_overlay.discard(int(corpus_line))
    return persist


def _current_viz_and_continue(idx: int) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (source_row, viz_edges, continue_edges)."""
    source = _read_sample(idx)
    key, cont_idx = _lookup_continue(idx, source)
    del key
    if cont_idx is None:
        viz = _source_baseline_edges(source)
        return source, viz, []
    overlay = _read_continue_by_idx(cont_idx)
    viz, cont = _viz_and_continue_from_overlay(source, overlay)
    return source, viz, cont


def _resolve_corpus_path(override: str | None = None) -> Path:
    global _corpus_path, _corpus_offsets
    if override:
        path = Path(override).expanduser().resolve()
    elif _corpus_path is not None:
        path = _corpus_path
    else:
        path = DEFAULT_CORPUS
    if path is None or not path.is_file():
        raise HTTPException(
            400,
            "No LLM train corpus configured. Set EIF_LLM_TRAIN_CORPUS in eif_api.env.",
        )
    if _corpus_path != path:
        _corpus_path = path
        _corpus_offsets = None
    return path


def _ensure_corpus_offsets() -> list[int]:
    global _corpus_offsets
    path = _resolve_corpus_path()
    if _corpus_offsets is None:
        print(f"[corpus] building line index for {path} …", flush=True)
        _corpus_offsets = _build_offsets(path)
        print(f"[corpus] indexed {max(0, len(_corpus_offsets) - 1)} lines", flush=True)
    return _corpus_offsets


def _read_corpus_raw_row(line: int, *, corpus_path: str | None = None) -> dict[str, Any]:
    path = _resolve_corpus_path(corpus_path)
    offsets = _ensure_corpus_offsets()
    if line < 0 or line >= len(offsets) - 1:
        raise HTTPException(404, f"corpus line {line} out of range (n={len(offsets) - 1})")
    with path.open("rb") as f:
        f.seek(offsets[line])
        raw = f.readline()
    row = json.loads(raw.decode("utf-8"))
    if not isinstance(row, dict):
        raise HTTPException(500, f"corpus line {line} is not a JSON object")
    return row


def _mid_rewrite_line_key(line: int, *, corpus_path: str | None = None) -> str:
    path = _resolve_corpus_path(corpus_path)
    return f"{path.resolve()}::{int(line)}"


def _bind_corpus_mid_rewrite(
    line: int,
    rewrite: dict[str, Any],
    *,
    corpus_path: str | None = None,
) -> None:
    key = _mid_rewrite_line_key(line, corpus_path=corpus_path)
    _corpus_mid_rewrite_by_line[key] = dict(rewrite)


def _get_bound_corpus_mid_rewrite(
    line: int,
    *,
    corpus_path: str | None = None,
) -> dict[str, Any] | None:
    return _corpus_mid_rewrite_by_line.get(_mid_rewrite_line_key(line, corpus_path=corpus_path))


def _compute_corpus_mid_rewrite(
    prompt: str,
    response: str,
    *,
    test_gold: str,
    expression: str = "",
) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from fim_mid_rewrite import rewrite_fim_mid  # type: ignore

    return rewrite_fim_mid(
        prompt,
        response,
        test_gold=test_gold,
        expression=expression or "",
    )


def _raw_row_with_bound_mid_rewrite(
    line: int,
    raw_row: dict[str, Any],
    *,
    corpus_path: str | None = None,
) -> dict[str, Any]:
    """Overlay rewritten prompt/response onto a corpus raw row when bound."""
    bound = _get_bound_corpus_mid_rewrite(line, corpus_path=corpus_path)
    if not bound or not bound.get("mode") or bound.get("mode") == "unchanged":
        return raw_row
    out = dict(raw_row)
    out["prompt"] = str(bound.get("prompt") or "")
    out["response"] = str(bound.get("response") or "")
    return out


def _encode_corpus_row(
    line: int,
    raw_row: dict[str, Any],
    *,
    corpus_path: str | None = None,
) -> dict[str, Any]:
    from server.corpus_encode import encode_prompt_response, extract_prompt_response

    prompt, response = extract_prompt_response(raw_row)
    if not prompt.strip():
        raise HTTPException(400, f"corpus line {line} missing prompt/input text")
    rewrite_meta: dict[str, Any] | None = None
    bound = _get_bound_corpus_mid_rewrite(line, corpus_path=corpus_path)
    if bound and bound.get("mode") and bound.get("mode") != "unchanged":
        prompt = str(bound.get("prompt") or prompt)
        response = str(bound.get("response") or response)
        rewrite_meta = {
            "mid_rewrite": True,
            "mid_rewrite_mode": bound.get("mode"),
            "mid_rewrite_locus": bound.get("dig_locus"),
            "mid_rewrite_geometry": bound.get("fim_geometry"),
            "mid_rewrite_reason": bound.get("reason"),
            "mid_rewrite_dig_preview": (str(bound.get("dig_text") or "")[:160]),
            "mid_rewrite_old_mid_preview": (str(bound.get("old_mid") or "")[:120]),
            "mid_rewrite_hash": bound.get("dig_hash"),
        }
    input_ids, labels = encode_prompt_response(_get_tokenizer(), prompt, response)
    task_id = str(raw_row.get("task_id") or f"line_{line}")
    uid = f"corpus:{task_id}"
    if rewrite_meta and rewrite_meta.get("mid_rewrite_hash"):
        uid = f"{uid}:midrw:{rewrite_meta['mid_rewrite_hash']}"
    meta: dict[str, Any] = {
        "corpus": True,
        "unannotated_source": True,
        "prompt_chars": len(prompt),
        "response_chars": len(response),
    }
    if rewrite_meta:
        meta.update(rewrite_meta)
    return {
        "input_ids": input_ids,
        "label": labels,
        "uid": uid,
        "task_id": task_id,
        "raw_id": task_id,
        "language": str(raw_row.get("language") or "go"),
        "source_corpus_line": int(line),
        "source_corpus_path": str(_corpus_path) if _corpus_path else "",
        "attention_edges": [],
        "annotation_meta": meta,
    }


def _effective_corpus_sample(line: int, *, corpus_path: str | None = None) -> tuple[dict[str, Any], bool, str]:
    raw_row = _read_corpus_raw_row(line, corpus_path=corpus_path)
    source = _encode_corpus_row(line, raw_row, corpus_path=corpus_path)
    key, cont_idx = _lookup_continue(-1, source)
    if int(line) in _corpus_skip_overlay:
        obj = _compose_display_obj(source, None, from_continue=False)
        return obj, False, key
    if cont_idx is None:
        obj = _compose_display_obj(source, None, from_continue=False)
        return obj, False, key
    overlay = _read_continue_by_idx(cont_idx)
    obj = _compose_display_obj(source, overlay, from_continue=True)
    return obj, True, key


def _current_viz_and_continue_corpus(
    line: int,
    *,
    corpus_path: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    raw_row = _read_corpus_raw_row(line, corpus_path=corpus_path)
    source = _encode_corpus_row(line, raw_row, corpus_path=corpus_path)
    key, cont_idx = _lookup_continue(-1, source)
    del key
    if int(line) in _corpus_skip_overlay or cont_idx is None:
        return source, [], []
    overlay = _read_continue_by_idx(cont_idx)
    viz, cont = _viz_and_continue_from_overlay(source, overlay)
    return source, viz, cont


def _write_continue_corpus_payload(
    line: int,
    source: dict[str, Any],
    *,
    viz_edges: list[dict[str, Any]],
    continue_edges: list[dict[str, Any]],
) -> dict[str, Any]:
    return _write_continue_payload(
        -1,
        source,
        viz_edges=viz_edges,
        continue_edges=continue_edges,
        corpus_line=line,
    )


def _sample_detail_from_obj(
    idx: int,
    obj: dict[str, Any],
    *,
    from_continue: bool,
    key: str,
) -> dict[str, Any]:
    input_ids = [int(x) for x in (obj.get("input_ids") or [])]
    labels = [int(x) for x in (obj.get("label") or obj.get("labels") or [])]
    tokens = _surface_tokens_from_obj(obj, input_ids)

    edges = []
    n_continue_edges = 0
    for e in obj.get("attention_edges") or []:
        if not isinstance(e, dict):
            continue
        try:
            w = float(e.get("weight", 1.0))
        except (TypeError, ValueError):
            w = 1.0
        if w <= 0:
            w = 1.0
        contrib = str(e.get("contrib") or "source")
        if contrib in CONTINUE_CONTRIBS:
            n_continue_edges += 1
        edges.append(
            {
                "src": int(e["src"]),
                "dst": int(e["dst"]),
                "subtype": str(e.get("subtype") or ""),
                "weight": w,
                "contrib": contrib,
            }
        )
    try:
        n_continue_edges = int(obj.get("_continue_edge_count") or n_continue_edges)
    except (TypeError, ValueError):
        pass

    return {
        "index": idx,
        "uid": obj.get("uid"),
        "language": obj.get("language"),
        "raw_id": obj.get("raw_id"),
        "tokens": tokens,
        "input_ids": input_ids,
        "answer_start": _answer_start(labels),
        "attention_edges": edges,
        "n_continue_edges": n_continue_edges,
        "annotation_meta": obj.get("annotation_meta") or {},
        "subtypes": SUBTYPES,
        "in_continue": from_continue,
        "sample_key": key,
        "continue_path": str(_continue_path) if _continue_path else None,
        "corpus_line": obj.get("source_corpus_line"),
        "corpus_path": obj.get("source_corpus_path"),
        "corpus_mode": bool((obj.get("annotation_meta") or {}).get("corpus")),
    }

def _resolve_tokenizer_path() -> str:
    if _tokenizer_path:
        return _tokenizer_path
    if _model_path:
        return _model_path
    if _TOKENIZER_RAW:
        return str(DEFAULT_TOKENIZER.expanduser().resolve())
    return ""


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is not None:
        return _tokenizer
    from transformers import AutoTokenizer

    path = _resolve_tokenizer_path()
    if not path or not Path(path).exists():
        raise HTTPException(
            500,
            "Tokenizer path not found. Set EIF_BASE_MODEL_PATH in "
            "repo-root eif_api.env or pass --tokenizer <path>.",
        )
    _tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return _tokenizer


def _decode_tokens(input_ids: list[int]) -> list[str]:
    tok = _get_tokenizer()
    # Per-id decode keeps alignment with attention_edges src/dst indices.
    return [tok.decode([int(i)], skip_special_tokens=False) for i in input_ids]


def _surface_tokens_from_obj(obj: dict[str, Any], input_ids: list[int]) -> list[str]:
    """Decode input_ids for display. Prefer live tokenizer; else JSONL qwen_tokens."""
    try:
        return _decode_tokens(input_ids)
    except HTTPException:
        pass

    # smoke / graphsignal JSONL already stores Qwen-aligned surfaces.
    qwen = obj.get("qwen_tokens") or []
    if isinstance(qwen, list) and len(qwen) == len(input_ids):
        out: list[str] = []
        for t in qwen:
            if isinstance(t, dict):
                out.append(str(t.get("surface") or ""))
            else:
                out.append(str(t))
        return out

    # Last resort: numeric placeholders (editing still works by index).
    return [f"<{tid}>" for tid in input_ids]


def _answer_start(labels: list[int]) -> int:
    for i, lab in enumerate(labels):
        if int(lab) != -100:
            return i
    return len(labels)


def _sync_meta(obj: dict[str, Any]) -> None:
    edges = obj.get("attention_edges") or []
    anns = obj.get("annotations") or []
    meta = obj.get("annotation_meta")
    if not isinstance(meta, dict):
        meta = {}
        obj["annotation_meta"] = meta
    meta["attention_edges"] = len(edges)
    meta["raw_annotations"] = len(anns)


def _ensure_model():
    global _model
    if _model is not None:
        return _model
    if not _model_path:
        raise HTTPException(
            501,
            "Saliency requires --model <Qwen path> when starting the server.",
        )
    import torch
    from transformers import AutoModelForCausalLM

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None
    _model = AutoModelForCausalLM.from_pretrained(
        _model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True,
        local_files_only=True,
    )
    _model.eval()
    return _model


# ── Schemas ───────────────────────────────────────────────────────────────────

class OpenDataBody(BaseModel):
    path: str


class DeleteEdgeBody(BaseModel):
    src: int
    dst: int
    subtype: str


class AddEdgeBody(BaseModel):
    src: int
    dst: int
    subtype: str = Field(..., description="annotation subtype")
    source: str = "Manual"
    weight: float = Field(1.0, description="positive edge weight (default 1)")


class BumpWeightBody(BaseModel):
    src: int
    dst: int
    subtype: str
    delta: float = Field(1.0, description="add this to weight (use -1 to decrease)")


class DuplicateContinueBody(BaseModel):
    copies: int = Field(
        1,
        ge=1,
        le=32,
        description="number of additional identical rows to append to continue JSONL",
    )


class GraphsignalAnnotateBody(BaseModel):
    use_llm: bool | None = None
    max_edges: int | None = Field(None, ge=1, le=256)


class GraphsignalPreviewActionBody(BaseModel):
    preview_id: str = Field(..., min_length=8)


class LlmSemanticAnnotateBody(BaseModel):
    max_sources_per_token: int | None = Field(None, ge=1, le=15)
    max_answer_tokens: int | None = Field(None, ge=1, le=256)

    class Config:
        extra = "ignore"


def _duplicate_continue_rows(
    source: dict[str, Any],
    viz: list[dict[str, Any]],
    cont: list[dict[str, Any]],
    *,
    copies: int,
    source_idx: int,
    corpus_line: int | None = None,
) -> dict[str, Any]:
    if not cont:
        raise HTTPException(
            400,
            "no continue-train edges yet — add at least one edge before duplicating",
        )
    import uuid

    base_uid = str(source.get("uid") or source.get("task_id") or f"sample_{source_idx}")
    rows: list[dict[str, Any]] = []
    for _ in range(int(copies)):
        uid = f"{base_uid}::dup{uuid.uuid4().hex[:10]}"
        rows.append(
            _build_continue_row(
                source_idx,
                source,
                viz_edges=viz,
                continue_edges=cont,
                corpus_line=corpus_line,
                uid_override=uid,
                duplicate_of=base_uid,
            )
        )
    return _append_continue_rows(rows)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    tok_path = _resolve_tokenizer_path()
    corpus_n = 0
    if _corpus_offsets is not None:
        corpus_n = max(0, len(_corpus_offsets) - 1)
    return {
        "ok": True,
        "data_path": str(_data_path) if _data_path else None,
        "n_samples": max(0, len(_offsets) - 1) if _offsets else 0,
        "continue_path": str(_continue_path) if _continue_path else None,
        "n_continue": max(0, len(_continue_offsets) - 1) if _continue_offsets else 0,
        "corpus_path": str(_corpus_path or DEFAULT_CORPUS) if (_corpus_path or DEFAULT_CORPUS) else None,
        "n_corpus": corpus_n,
        "write_mode": "continue_upsert",
        "subtypes": SUBTYPES,
        "tokenizer_path": tok_path,
        "tokenizer_ready": _tokenizer is not None,
        "saliency_available": _saliency_enabled(),
        "saliency_mode": (
            "model" if _model_path else ("disk_cache" if _saliency_cache_dir else "off")
        ),
        "saliency_cache_dir": str(_saliency_cache_dir) if _saliency_cache_dir else None,
        "model_loaded": _model is not None,
    }


@app.post("/api/open")
def open_data(body: OpenDataBody):
    global _data_path, _offsets
    path = Path(body.path).expanduser().resolve()
    if not path.exists():
        raise HTTPException(404, f"file not found: {path}")
    with _state_lock:
        _data_path = path
        _offsets = _build_offsets(path)
        _saliency_cache.clear()
    return {"path": str(path), "n_samples": len(_offsets) - 1}


@app.get("/api/samples")
def list_samples(q: str = "", offset: int = 0, limit: int = 50):
    if _data_path is None:
        raise HTTPException(400, "No data file open. POST /api/open first.")
    n = len(_offsets) - 1
    offset = max(0, offset)
    limit = max(1, min(limit, 500))
    q_norm = q.strip().lower()

    items: list[dict[str, Any]] = []
    # Linear scan is OK for 10k small headers; stop once page filled after filter.
    scanned = 0
    matched_before_offset = 0
    with _data_path.open("rb") as f:
        for i in range(n):
            f.seek(_offsets[i])
            raw = f.readline()
            # cheap uid extract without full json when possible
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            uid = str(obj.get("uid") or f"row_{i}")
            lang = str(obj.get("language") or "")
            raw_id = str(obj.get("raw_id") or "")
            hay = f"{uid} {lang} {raw_id} {i}".lower()
            if q_norm and q_norm not in hay:
                continue
            if matched_before_offset < offset:
                matched_before_offset += 1
                continue
            n_edges = len(obj.get("attention_edges") or [])
            key, cont_idx = _lookup_continue(i, obj)
            in_continue = cont_idx is not None
            if in_continue and cont_idx is not None:
                try:
                    cont = _read_continue_by_idx(cont_idx)
                    n_edges = len(cont.get("attention_edges") or [])
                except HTTPException:
                    pass
            items.append(
                {
                    "index": i,
                    "uid": uid,
                    "language": lang,
                    "raw_id": raw_id,
                    "length": int(obj.get("length") or len(obj.get("input_ids") or [])),
                    "n_edges": n_edges,
                    "in_continue": in_continue,
                    "sample_key": key,
                }
            )
            scanned += 1
            if scanned >= limit:
                break
    return {"offset": offset, "limit": limit, "items": items, "total_approx": n}


@app.get("/api/sample/{idx}")
def get_sample(idx: int):
    if _data_path is None:
        raise HTTPException(400, "No data file open.")
    with _state_lock:
        obj, from_continue, key = _effective_sample(idx)
    return _sample_detail_from_obj(idx, obj, from_continue=from_continue, key=key)


@app.get("/api/corpus/sample/{line}")
def get_corpus_sample(
    line: int,
    corpusPath: str = "",
    rewriteId: str = "",
):
    override = corpusPath.strip() or None
    rid = (rewriteId or "").strip()
    with _state_lock:
        if rid:
            prep = _corpus_mid_rewrite_prep.get(rid)
            if not prep:
                raise HTTPException(404, "mid-rewrite prep not found or expired")
            if int(prep.get("line", -1)) != int(line):
                raise HTTPException(400, "rewriteId line mismatch")
            prep_path = str(prep.get("corpus_path") or "").strip()
            if override and prep_path and prep_path != override:
                raise HTTPException(400, "rewriteId corpus path mismatch")
            if not override and prep_path:
                override = prep_path
            _bind_corpus_mid_rewrite(
                line,
                prep["rewrite"],
                corpus_path=override,
            )
        obj, from_continue, key = _effective_corpus_sample(line, corpus_path=override)
    return _sample_detail_from_obj(line, obj, from_continue=from_continue, key=key)


class CorpusMidRewritePrepBody(BaseModel):
    line: int
    corpusPath: str = ""
    testGold: str
    expression: str = ""


@app.post("/api/corpus/mid-rewrite-prep")
def corpus_mid_rewrite_prep(body: CorpusMidRewritePrepBody):
    """Precompute context→MID re-hollow for a corpus line (cross-origin open)."""
    override = (body.corpusPath or "").strip() or None
    test_gold = body.testGold or ""
    expression = (body.expression or "").strip()
    if not test_gold.strip() and not expression:
        raise HTTPException(400, "testGold or expression required")
    with _state_lock:
        raw_row = _read_corpus_raw_row(int(body.line), corpus_path=override)
        from server.corpus_encode import extract_prompt_response

        prompt, response = extract_prompt_response(raw_row)
        rewrite = _compute_corpus_mid_rewrite(
            prompt,
            response,
            test_gold=test_gold,
            expression=expression,
        )
        rewrite_id = uuid.uuid4().hex
        while len(_corpus_mid_rewrite_prep) >= _CORPUS_MID_REWRITE_PREP_MAX:
            _corpus_mid_rewrite_prep.pop(next(iter(_corpus_mid_rewrite_prep)))
        _corpus_mid_rewrite_prep[rewrite_id] = {
            "rewrite_id": rewrite_id,
            "line": int(body.line),
            "corpus_path": override or (str(_corpus_path) if _corpus_path else ""),
            "rewrite": rewrite,
            "created_at": time.time(),
        }
        # Bind immediately so a subsequent open without rewriteId still works
        # if the same viewer process loads the line.
        if rewrite.get("mode") and rewrite.get("mode") != "unchanged":
            _bind_corpus_mid_rewrite(int(body.line), rewrite, corpus_path=override)
    return {
        "ok": True,
        "rewrite_id": rewrite_id,
        "mode": rewrite.get("mode"),
        "reason": rewrite.get("reason"),
        "dig_preview": (str(rewrite.get("dig_text") or "")[:200]),
        "old_mid_preview": (str(rewrite.get("old_mid") or "")[:120]),
        "applied": bool(rewrite.get("mode") and rewrite.get("mode") != "unchanged"),
    }


@app.post("/api/corpus/sample/{line}/clear-display")
def clear_corpus_display(line: int, corpusPath: str = ""):
    """Clear on-screen edges only. Does not delete existing continue JSONL rows.

    The next add/accept for this line appends a new continue row on top of the file.
    """
    override = corpusPath.strip() or None
    with _state_lock:
        _corpus_skip_overlay.add(int(line))
        for cache in (_gs_preview_cache, _llm_sem_preview_cache):
            dead = [pid for pid, ent in cache.items() if int(ent.get("line", -1)) == int(line)]
            for pid in dead:
                cache.pop(pid, None)
        obj, from_continue, key = _effective_corpus_sample(line, corpus_path=override)
        sample = _sample_detail_from_obj(-1, obj, from_continue=from_continue, key=key)
    print(
        f"[corpus] clear-display line={line} (continue JSONL unchanged; next persist appends)",
        flush=True,
    )
    return {
        "ok": True,
        "sample": sample,
        "n_continue_edges": 0,
        "message": "已清空当前显示标注；续训文件未改。之后新增的边会追加一条新记录。",
    }


@app.post("/api/corpus/sample/{line}/edges/delete")
def delete_corpus_edge(line: int, body: DeleteEdgeBody, corpusPath: str = ""):
    if body.subtype not in SUBTYPES:
        raise HTTPException(400, f"unknown subtype {body.subtype}")
    override = corpusPath.strip() or None
    with _state_lock:
        source, viz, cont = _current_viz_and_continue_corpus(line, corpus_path=override)
        before = len(viz)
        viz = [
            e
            for e in viz
            if not (
                int(e.get("src", -1)) == body.src
                and int(e.get("dst", -1)) == body.dst
                and str(e.get("subtype", "")) == body.subtype
            )
        ]
        if len(viz) == before:
            raise HTTPException(404, "edge not found in attention_edges")
        cont = [
            e
            for e in cont
            if not (
                int(e.get("src", -1)) == body.src
                and int(e.get("dst", -1)) == body.dst
                and str(e.get("subtype", "")) == body.subtype
            )
        ]
        persist = _write_continue_corpus_payload(
            line, source, viz_edges=viz, continue_edges=cont,
        )
    return {
        "ok": True,
        "n_edges": len(viz),
        "n_continue_edges": len(cont),
        **persist,
    }


@app.post("/api/corpus/sample/{line}/edges/add")
def add_corpus_edge(line: int, body: AddEdgeBody, corpusPath: str = ""):
    if body.subtype not in SUBTYPES:
        raise HTTPException(400, f"unknown subtype {body.subtype}; choose from {SUBTYPES}")
    if body.src == body.dst:
        raise HTTPException(400, "src and dst must differ")
    override = corpusPath.strip() or None
    with _state_lock:
        source, viz, cont = _current_viz_and_continue_corpus(line, corpus_path=override)
        n = len(source.get("input_ids") or [])
        if not (0 <= body.src < n and 0 <= body.dst < n):
            raise HTTPException(400, f"src/dst out of range 0..{n-1}")
        for e in viz:
            if (
                int(e.get("src", -1)) == body.src
                and int(e.get("dst", -1)) == body.dst
                and str(e.get("subtype", "")) == body.subtype
            ):
                raise HTTPException(409, "edge already exists")
        edge = _normalize_edge(
            {
                "src": body.src,
                "dst": body.dst,
                "subtype": body.subtype,
                "source": body.source,
                "weight": body.weight,
            },
            contrib="user_add",
        )
        viz = list(viz) + [edge]
        cont = list(cont) + [edge]
        persist = _write_continue_corpus_payload(
            line, source, viz_edges=viz, continue_edges=cont,
        )
    return {
        "ok": True,
        "n_edges": len(viz),
        "n_continue_edges": len(cont),
        "edge": edge,
        **persist,
    }


@app.post("/api/corpus/sample/{line}/edges/bump-weight")
def bump_corpus_edge_weight(line: int, body: BumpWeightBody, corpusPath: str = ""):
    if body.subtype not in SUBTYPES:
        raise HTTPException(400, f"unknown subtype {body.subtype}; choose from {SUBTYPES}")
    delta = float(body.delta)
    if delta == 0:
        raise HTTPException(400, "delta must be non-zero")
    override = corpusPath.strip() or None
    with _state_lock:
        source, viz, cont = _current_viz_and_continue_corpus(line, corpus_path=override)
        found = None
        for e in viz:
            if (
                int(e.get("src", -1)) == body.src
                and int(e.get("dst", -1)) == body.dst
                and str(e.get("subtype", "")) == body.subtype
            ):
                found = e
                break
        if found is None:
            raise HTTPException(404, "edge not found — add it first, then bump weight")
        old_w = float(found.get("weight", 1.0))
        new_w = max(1.0, old_w + delta)
        bumped = _normalize_edge(found, contrib="user_bump", weight=new_w)
        viz_out: list[dict[str, Any]] = []
        for e in viz:
            viz_out.append(bumped if _edge_key(e) == _edge_key(found) else e)
        cont_out: list[dict[str, Any]] = []
        in_cont = False
        for e in cont:
            if _edge_key(e) == _edge_key(found):
                cont_out.append(bumped)
                in_cont = True
            else:
                cont_out.append(e)
        if not in_cont:
            cont_out.append(bumped)
        persist = _write_continue_corpus_payload(
            line, source, viz_edges=viz_out, continue_edges=cont_out,
        )
    return {
        "ok": True,
        "n_edges": len(viz_out),
        "n_continue_edges": len(cont_out),
        "edge": bumped,
        "old_weight": old_w,
        "new_weight": new_w,
        **persist,
    }


@app.get("/api/corpus/sample/{line}/saliency/{target}")
def get_corpus_saliency(line: int, target: int, top_k: int = 6, corpusPath: str = ""):
    override = corpusPath.strip() or None
    with _state_lock:
        raw_row = _read_corpus_raw_row(line, corpus_path=override)
        source = _encode_corpus_row(line, raw_row, corpus_path=override)
    input_ids = [int(x) for x in (source.get("input_ids") or [])]
    if target <= 0 or target >= len(input_ids):
        raise HTTPException(400, f"target {target} out of range")
    if not _model_path:
        return {
            "target": target,
            "top": [],
            "available": False,
            "message": "No live model for saliency on corpus samples.",
        }
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import torch
    from loss import compute_alti_saliency_vector  # type: ignore

    model = _ensure_model()
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(ids)
    batch = {"input_ids": ids, "attention_mask": attn}
    sal = compute_alti_saliency_vector(model, batch, target)
    scored = [
        (i, float(s))
        for i, s in enumerate(sal)
        if i < target and float(s) > 0
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [{"src": i, "score": sc} for i, sc in scored[: max(1, min(top_k, 20))]]
    return {"target": target, "top": top, "available": True, "cached": False, "source": "model"}


@app.post("/api/sample/{idx}/continue-duplicate")
def duplicate_train_continue(idx: int, body: DuplicateContinueBody):
    if _data_path is None:
        raise HTTPException(400, "No data file open.")
    with _state_lock:
        source, viz, cont = _current_viz_and_continue(idx)
        persist = _duplicate_continue_rows(
            source, viz, cont, copies=body.copies, source_idx=idx,
        )
    return {
        **persist,
        "n_continue_edges": len(cont),
    }


@app.post("/api/corpus/sample/{line}/continue-duplicate")
def duplicate_corpus_continue(line: int, body: DuplicateContinueBody, corpusPath: str = ""):
    override = corpusPath.strip() or None
    with _state_lock:
        source, viz, cont = _current_viz_and_continue_corpus(line, corpus_path=override)
        persist = _duplicate_continue_rows(
            source, viz, cont, copies=body.copies, source_idx=-1, corpus_line=line,
        )
    return {
        **persist,
        "n_continue_edges": len(cont),
    }


def _prune_gs_preview_cache() -> None:
    now = time.time()
    stale = [
        pid for pid, ent in _gs_preview_cache.items()
        if now - float(ent.get("created_at") or 0) > _GS_PREVIEW_TTL_SEC
    ]
    for pid in stale:
        _gs_preview_cache.pop(pid, None)
    while len(_gs_preview_cache) > _GS_PREVIEW_CACHE_MAX:
        oldest = min(
            _gs_preview_cache.items(),
            key=lambda kv: float(kv[1].get("created_at") or 0),
        )[0]
        _gs_preview_cache.pop(oldest, None)


def _graphsignal_llm_edges(raw_edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in raw_edges:
        if not isinstance(e, dict) or "src" not in e or "dst" not in e:
            continue
        out.append(
            _normalize_edge(
                {
                    "src": e["src"],
                    "dst": e["dst"],
                    "subtype": e.get("subtype") or "",
                    "weight": e.get("weight", 1.0),
                    "source": "GraphSignal",
                },
                contrib="llm_auto",
            )
        )
    return out


def _build_graphsignal_annotated_source(
    line: int,
    raw_row: dict[str, Any],
    annotated: dict[str, Any],
    *,
    use_llm: bool,
    corpus_path: str | None = None,
) -> dict[str, Any]:
    base = _encode_corpus_row(line, raw_row, corpus_path=corpus_path)
    base["input_ids"] = [int(x) for x in (annotated.get("input_ids") or [])]
    base["label"] = [int(x) for x in (annotated.get("label") or [])]
    raw_edges = annotated.get("attention_edges") or []
    base["attention_edges"] = [
        _normalize_edge(e, contrib="llm_auto")
        for e in raw_edges
        if isinstance(e, dict) and "src" in e and "dst" in e
    ]
    meta = dict(base.get("annotation_meta") or {})
    meta.update({
        "corpus": True,
        "graphsignal": True,
        "graphsignal_use_llm": bool(use_llm),
        "graphsignal_n_edges": len(base["attention_edges"]),
        "unannotated_source": False,
    })
    base["annotation_meta"] = meta
    return base


def _sample_detail_from_graphsignal_preview(
    line: int,
    annotated_source: dict[str, Any],
    *,
    from_continue: bool,
    key: str,
) -> dict[str, Any]:
    detail = _sample_detail_from_obj(-1, annotated_source, from_continue=from_continue, key=key)
    detail["corpus_line"] = int(line)
    detail["corpus_path"] = annotated_source.get("source_corpus_path")
    detail["corpus_mode"] = True
    detail["graphsignal_preview"] = True
    return detail


@app.post("/api/corpus/sample/{line}/graphsignal-annotate/preview")
def graphsignal_annotate_preview(line: int, body: GraphsignalAnnotateBody, corpusPath: str = ""):
    """Run GraphSignal (tree-sitter + optional LLM) on a corpus row; preview only."""
    override = corpusPath.strip() or None
    from server.graphsignal_annotate import annotate_corpus_row, default_use_llm

    use_llm = default_use_llm() if body.use_llm is None else bool(body.use_llm)
    with _state_lock:
        raw_row = _raw_row_with_bound_mid_rewrite(
            line,
            _read_corpus_raw_row(line, corpus_path=override),
            corpus_path=override,
        )
        source_enc = _encode_corpus_row(line, raw_row, corpus_path=override)
        key, _ = _lookup_continue(-1, source_enc)
        try:
            annotated = annotate_corpus_row(
                raw_row,
                _get_tokenizer(),
                max_teacher_edges=body.max_edges,
                use_llm=use_llm,
            )
        except RuntimeError as exc:
            raise HTTPException(501, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            print(f"[graphsignal] FAIL corpus line={line}: {exc}", flush=True)
            raise HTTPException(502, f"GraphSignal annotate failed: {exc}") from exc

        annotated_source = _build_graphsignal_annotated_source(
            line, raw_row, annotated, use_llm=use_llm, corpus_path=override,
        )
        preview_id = uuid.uuid4().hex
        _prune_gs_preview_cache()
        _gs_preview_cache[preview_id] = {
            "preview_id": preview_id,
            "line": int(line),
            "corpus_path": override or (str(_corpus_path) if _corpus_path else ""),
            "annotated_source": annotated_source,
            "raw_edges": list(annotated.get("attention_edges") or []),
            "use_llm": use_llm,
            "created_at": time.time(),
        }
        sample = _sample_detail_from_graphsignal_preview(
            line, annotated_source, from_continue=False, key=key,
        )

    n_edges = len(annotated_source.get("attention_edges") or [])
    print(
        f"[graphsignal] preview corpus line={line} preview_id={preview_id[:8]} "
        f"edges={n_edges} use_llm={use_llm}",
        flush=True,
    )
    return {
        "ok": True,
        "preview_id": preview_id,
        "n_edges": n_edges,
        "use_llm": use_llm,
        "sample": sample,
        "message": (
            f"GraphSignal 预览 {n_edges} 条边"
            f"（{'tree-sitter+LLM' if use_llm else '仅 tree-sitter'}）。"
            "接受后写入续训小集；拒绝则回退。"
        ),
    }


@app.post("/api/corpus/sample/{line}/graphsignal-annotate/accept")
def graphsignal_annotate_accept(line: int, body: GraphsignalPreviewActionBody, corpusPath: str = ""):
    override = corpusPath.strip() or None
    pid = str(body.preview_id or "").strip()
    with _state_lock:
        entry = _gs_preview_cache.get(pid)
        if not entry or int(entry.get("line", -1)) != int(line):
            raise HTTPException(404, "GraphSignal preview not found or expired")
        if override and str(entry.get("corpus_path") or "") not in ("", override):
            raise HTTPException(400, "preview corpus path mismatch")
        if not override and entry.get("corpus_path"):
            override = str(entry["corpus_path"]) or None

        annotated_source = dict(entry["annotated_source"])
        new_llm = _graphsignal_llm_edges(entry.get("raw_edges") or [])

        source_enc = _encode_corpus_row(
            line,
            _read_corpus_raw_row(line, corpus_path=override),
            corpus_path=override,
        )
        _, viz, cont = _current_viz_and_continue_corpus(line, corpus_path=override)
        cont_keep = [
            e for e in cont
            if str(e.get("contrib") or "") in ("user_add", "user_bump")
        ]
        viz_keep = [
            e for e in viz
            if str(e.get("contrib") or "") != "llm_auto"
        ]
        by_viz = {_edge_key(e): e for e in viz_keep}
        for e in cont_keep + new_llm:
            by_viz[_edge_key(e)] = e
        viz_out = list(by_viz.values())
        cont_out = cont_keep + new_llm
        cont_map = {_edge_key(e): e for e in cont_out}
        cont_out = list(cont_map.values())

        merged_source = dict(source_enc)
        merged_source["input_ids"] = annotated_source["input_ids"]
        merged_source["label"] = annotated_source["label"]
        meta = dict(merged_source.get("annotation_meta") or {})
        meta.update(annotated_source.get("annotation_meta") or {})
        merged_source["annotation_meta"] = meta

        persist = _write_continue_corpus_payload(
            line,
            merged_source,
            viz_edges=viz_out,
            continue_edges=cont_out,
        )
        _gs_preview_cache.pop(pid, None)
        _, cont_idx = _lookup_continue(-1, merged_source)
        obj, _, key2 = _effective_corpus_sample(line, corpus_path=override)
        sample = _sample_detail_from_obj(-1, obj, from_continue=cont_idx is not None, key=key2)

    print(
        f"[graphsignal] accept corpus line={line} llm_edges={len(new_llm)} "
        f"continue_edges={len(cont_out)} path={persist.get('continue_path') or '-'}",
        flush=True,
    )
    return {
        "ok": True,
        "preview_id": pid,
        "n_edges": len(viz_out),
        "n_continue_edges": len(cont_out),
        "proposed": new_llm,
        "sample": sample,
        **persist,
    }


@app.post("/api/corpus/sample/{line}/graphsignal-annotate/reject")
def graphsignal_annotate_reject(line: int, body: GraphsignalPreviewActionBody, corpusPath: str = ""):
    override = corpusPath.strip() or None
    pid = str(body.preview_id or "").strip()
    with _state_lock:
        entry = _gs_preview_cache.pop(pid, None)
        if entry and int(entry.get("line", -1)) != int(line):
            _gs_preview_cache[pid] = entry
            raise HTTPException(400, "preview line mismatch")
        obj, from_continue, key = _effective_corpus_sample(line, corpus_path=override)
        sample = _sample_detail_from_obj(-1, obj, from_continue=from_continue, key=key)
    return {"ok": True, "preview_id": pid, "sample": sample}


def _prune_llm_sem_preview_cache() -> None:
    now = time.time()
    stale = [
        pid for pid, ent in _llm_sem_preview_cache.items()
        if now - float(ent.get("created_at") or 0) > _GS_PREVIEW_TTL_SEC
    ]
    for pid in stale:
        _llm_sem_preview_cache.pop(pid, None)
    while len(_llm_sem_preview_cache) > _GS_PREVIEW_CACHE_MAX:
        oldest = min(
            _llm_sem_preview_cache.items(),
            key=lambda kv: float(kv[1].get("created_at") or 0),
        )[0]
        _llm_sem_preview_cache.pop(oldest, None)


def _llm_semantic_llm_edges(raw_edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in raw_edges:
        if not isinstance(e, dict) or "src" not in e or "dst" not in e:
            continue
        out.append(
            _normalize_edge(
                {
                    "src": e["src"],
                    "dst": e["dst"],
                    "subtype": str(e.get("subtype") or "semantic"),
                    "reason": e.get("reason") or "",
                    "weight": e.get("weight", 1.0),
                    "source": "LLMSemantic",
                },
                contrib="llm_auto",
            )
        )
    return out


def _build_llm_semantic_annotated_source(
    line: int,
    raw_row: dict[str, Any],
    annotated: dict[str, Any],
    *,
    corpus_path: str | None = None,
) -> dict[str, Any]:
    base = _encode_corpus_row(line, raw_row, corpus_path=corpus_path)
    base["input_ids"] = [int(x) for x in (annotated.get("input_ids") or [])]
    base["label"] = [int(x) for x in (annotated.get("label") or [])]
    raw_edges = annotated.get("attention_edges") or []
    base["attention_edges"] = [
        _normalize_edge(
            {
                "src": e["src"],
                "dst": e["dst"],
                "subtype": str(e.get("subtype") or "semantic"),
                "reason": e.get("reason") or "",
            },
            contrib="llm_auto",
        )
        for e in raw_edges
        if isinstance(e, dict) and "src" in e and "dst" in e
    ]
    meta = dict(base.get("annotation_meta") or {})
    sem_meta = annotated.get("_llm_semantic_meta") or {}
    meta.update({
        "corpus": True,
        "llm_semantic": True,
        "llm_semantic_n_edges": len(base["attention_edges"]),
        "llm_semantic_meta": sem_meta,
        "unannotated_source": False,
    })
    base["annotation_meta"] = meta
    return base


@app.post("/api/corpus/sample/{line}/llm-semantic-annotate/preview")
def llm_semantic_annotate_preview(
    line: int,
    body: LlmSemanticAnnotateBody | None = None,
    corpusPath: str = Query(""),
):
    """Per-token LLM attention-routing annotation; preview only."""
    override = corpusPath.strip() or None
    payload = body or LlmSemanticAnnotateBody()
    from server.corpus_encode import extract_prompt_response
    from server.llm_semantic_annotate import annotate_corpus_row_semantic

    with _state_lock:
        raw_row = _raw_row_with_bound_mid_rewrite(
            line,
            _read_corpus_raw_row(line, corpus_path=override),
            corpus_path=override,
        )
        source_enc = _encode_corpus_row(line, raw_row, corpus_path=override)
        key, _ = _lookup_continue(-1, source_enc)
        tokenizer = _get_tokenizer()
    prompt, response = extract_prompt_response(raw_row)
    print(
        f"[llm-semantic] preview start line={line} "
        f"prompt_chars={len(prompt)} response_chars={len(response)} "
        f"keys={sorted(str(k) for k in raw_row.keys())[:24]}",
        flush=True,
    )
    try:
        annotated = annotate_corpus_row_semantic(
            raw_row,
            tokenizer,
            max_sources_per_token=payload.max_sources_per_token,
            max_answer_tokens=payload.max_answer_tokens,
        )
    except ValueError as exc:
        print(f"[llm-semantic] FAIL corpus line={line}: {exc}", flush=True)
        raise HTTPException(400, str(exc)) from exc
    except Exception as exc:
        print(f"[llm-semantic] FAIL corpus line={line}: {exc}", flush=True)
        raise HTTPException(502, f"LLM semantic annotate failed: {exc}") from exc

    with _state_lock:
        annotated_source = _build_llm_semantic_annotated_source(
            line, raw_row, annotated, corpus_path=override,
        )
        preview_id = uuid.uuid4().hex
        _prune_llm_sem_preview_cache()
        _llm_sem_preview_cache[preview_id] = {
            "preview_id": preview_id,
            "line": int(line),
            "corpus_path": override or (str(_corpus_path) if _corpus_path else ""),
            "annotated_source": annotated_source,
            "raw_edges": list(annotated.get("attention_edges") or []),
            "meta": dict(annotated.get("_llm_semantic_meta") or {}),
            "created_at": time.time(),
        }
        sample = _sample_detail_from_graphsignal_preview(
            line, annotated_source, from_continue=False, key=key,
        )
        sample["llm_semantic_preview"] = True

    sem_meta = annotated.get("_llm_semantic_meta") or {}
    n_edges = len(annotated_source.get("attention_edges") or [])
    print(
        f"[llm-semantic] preview corpus line={line} preview_id={preview_id[:8]} "
        f"edges={n_edges} llm_calls={sem_meta.get('llm_calls')}",
        flush=True,
    )
    return {
        "ok": True,
        "preview_id": preview_id,
        "n_edges": n_edges,
        "llm_calls": sem_meta.get("llm_calls"),
        "answer_token_count": sem_meta.get("answer_token_count"),
        "sample": sample,
        "message": (
            f"LLM 语义标注预览 {n_edges} 条边"
            f"（{sem_meta.get('answer_token_count', '?')} 个答案 token ×"
            f" 最多 {sem_meta.get('max_sources_per_token', 15)} 源/token）。"
            "接受后写入续训小集；拒绝则回退。"
        ),
    }


@app.post("/api/corpus/sample/{line}/llm-semantic-annotate/accept")
def llm_semantic_annotate_accept(line: int, body: GraphsignalPreviewActionBody, corpusPath: str = ""):
    override = corpusPath.strip() or None
    pid = str(body.preview_id or "").strip()
    with _state_lock:
        entry = _llm_sem_preview_cache.get(pid)
        if not entry or int(entry.get("line", -1)) != int(line):
            raise HTTPException(404, "LLM semantic preview not found or expired")
        if override and str(entry.get("corpus_path") or "") not in ("", override):
            raise HTTPException(400, "preview corpus path mismatch")
        if not override and entry.get("corpus_path"):
            override = str(entry["corpus_path"]) or None

        annotated_source = dict(entry["annotated_source"])
        new_llm = _llm_semantic_llm_edges(entry.get("raw_edges") or [])

        source_enc = _encode_corpus_row(
            line,
            _read_corpus_raw_row(line, corpus_path=override),
            corpus_path=override,
        )
        _, viz, cont = _current_viz_and_continue_corpus(line, corpus_path=override)
        cont_keep = [
            e for e in cont
            if str(e.get("contrib") or "") in ("user_add", "user_bump")
        ]
        viz_keep = [
            e for e in viz
            if str(e.get("contrib") or "") != "llm_auto"
        ]
        by_viz = {_edge_key(e): e for e in viz_keep}
        for e in cont_keep + new_llm:
            by_viz[_edge_key(e)] = e
        viz_out = list(by_viz.values())
        cont_out = cont_keep + new_llm
        cont_map = {_edge_key(e): e for e in cont_out}
        cont_out = list(cont_map.values())

        merged_source = dict(source_enc)
        merged_source["input_ids"] = annotated_source["input_ids"]
        merged_source["label"] = annotated_source["label"]
        meta = dict(merged_source.get("annotation_meta") or {})
        meta.update(annotated_source.get("annotation_meta") or {})
        merged_source["annotation_meta"] = meta

        persist = _write_continue_corpus_payload(
            line,
            merged_source,
            viz_edges=viz_out,
            continue_edges=cont_out,
        )
        _llm_sem_preview_cache.pop(pid, None)
        _, cont_idx = _lookup_continue(-1, merged_source)
        obj, _, key2 = _effective_corpus_sample(line, corpus_path=override)
        sample = _sample_detail_from_obj(-1, obj, from_continue=cont_idx is not None, key=key2)

    print(
        f"[llm-semantic] accept corpus line={line} llm_edges={len(new_llm)} "
        f"continue_edges={len(cont_out)} path={persist.get('continue_path') or '-'}",
        flush=True,
    )
    return {
        "ok": True,
        "preview_id": pid,
        "n_edges": len(viz_out),
        "n_continue_edges": len(cont_out),
        "proposed": new_llm,
        "sample": sample,
        **persist,
    }


@app.post("/api/corpus/sample/{line}/llm-semantic-annotate/reject")
def llm_semantic_annotate_reject(line: int, body: GraphsignalPreviewActionBody, corpusPath: str = ""):
    override = corpusPath.strip() or None
    pid = str(body.preview_id or "").strip()
    with _state_lock:
        entry = _llm_sem_preview_cache.pop(pid, None)
        if entry and int(entry.get("line", -1)) != int(line):
            _llm_sem_preview_cache[pid] = entry
            raise HTTPException(400, "preview line mismatch")
        obj, from_continue, key = _effective_corpus_sample(line, corpus_path=override)
        sample = _sample_detail_from_obj(-1, obj, from_continue=from_continue, key=key)
    return {"ok": True, "preview_id": pid, "sample": sample}


@app.get("/api/sample/{idx}/saliency/{target}")
def get_saliency(idx: int, target: int, top_k: int = 6):
    """Return top-k ALTI saliency source indices for a target token position.

    Resolution order:
      1) in-memory cache
      2) on-disk precompute cache (--saliency-cache)
      3) live ALTI with --model
    """
    if _data_path is None:
        raise HTTPException(400, "No data file open.")
    cache_key = (idx, target)
    with _state_lock:
        if cache_key in _saliency_cache:
            return {
                "target": target,
                "top": _saliency_cache[cache_key],
                "cached": True,
                "available": True,
                "source": "memory",
            }

    disk_top = _load_disk_saliency(idx, target, top_k)
    if disk_top is not None:
        with _state_lock:
            _saliency_cache[cache_key] = disk_top
        return {
            "target": target,
            "top": disk_top,
            "cached": True,
            "available": True,
            "source": "disk",
        }

    if not _model_path:
        return {
            "target": target,
            "top": [],
            "available": False,
            "message": (
                "No live model and no disk cache hit. "
                "Either start with --model, or precompute on a GPU machine and "
                "pass --saliency-cache <dir> (see scripts/precompute_saliency_cache.py)."
            ),
        }

    with _state_lock:
        obj = _read_sample(idx)
    input_ids = [int(x) for x in (obj.get("input_ids") or [])]
    if target <= 0 or target >= len(input_ids):
        raise HTTPException(400, f"target {target} out of range")

    # Import from repo src/
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import torch
    from loss import compute_alti_saliency_vector  # type: ignore

    model = _ensure_model()
    device = next(model.parameters()).device
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(ids)
    batch = {"input_ids": ids, "attention_mask": attn}
    sal = compute_alti_saliency_vector(model, batch, target)

    # Rank sources excluding the target itself / trivial zeros.
    scored = [
        (i, float(s))
        for i, s in enumerate(sal)
        if i < target and float(s) > 0
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [{"src": i, "score": sc} for i, sc in scored[: max(1, min(top_k, 20))]]

    with _state_lock:
        _saliency_cache[cache_key] = top
    return {"target": target, "top": top, "available": True, "cached": False, "source": "model"}


@app.post("/api/sample/{idx}/edges/delete")
def delete_edge(idx: int, body: DeleteEdgeBody):
    if body.subtype not in SUBTYPES:
        raise HTTPException(400, f"unknown subtype {body.subtype}")
    if _data_path is None:
        raise HTTPException(400, "No data file open.")

    with _state_lock:
        source, viz, cont = _current_viz_and_continue(idx)
        before = len(viz)
        viz = [
            e
            for e in viz
            if not (
                int(e.get("src", -1)) == body.src
                and int(e.get("dst", -1)) == body.dst
                and str(e.get("subtype", "")) == body.subtype
            )
        ]
        if len(viz) == before:
            raise HTTPException(404, "edge not found in attention_edges")
        cont = [
            e
            for e in cont
            if not (
                int(e.get("src", -1)) == body.src
                and int(e.get("dst", -1)) == body.dst
                and str(e.get("subtype", "")) == body.subtype
            )
        ]
        persist = _write_continue_payload(
            idx, source, viz_edges=viz, continue_edges=cont,
        )

    return {
        "ok": True,
        "n_edges": len(viz),
        "n_continue_edges": len(cont),
        **persist,
    }


@app.post("/api/sample/{idx}/edges/add")
def add_edge(idx: int, body: AddEdgeBody):
    if body.subtype not in SUBTYPES:
        raise HTTPException(400, f"unknown subtype {body.subtype}; choose from {SUBTYPES}")
    if body.src == body.dst:
        raise HTTPException(400, "src and dst must differ")
    if _data_path is None:
        raise HTTPException(400, "No data file open.")

    with _state_lock:
        source, viz, cont = _current_viz_and_continue(idx)
        n = len(source.get("input_ids") or [])
        if not (0 <= body.src < n and 0 <= body.dst < n):
            raise HTTPException(400, f"src/dst out of range 0..{n-1}")

        for e in viz:
            if (
                int(e.get("src", -1)) == body.src
                and int(e.get("dst", -1)) == body.dst
                and str(e.get("subtype", "")) == body.subtype
            ):
                raise HTTPException(409, "edge already exists")

        edge = _normalize_edge(
            {
                "src": body.src,
                "dst": body.dst,
                "subtype": body.subtype,
                "source": body.source,
                "weight": body.weight,
            },
            contrib="user_add",
        )
        viz = list(viz) + [edge]
        cont = list(cont) + [edge]
        persist = _write_continue_payload(
            idx, source, viz_edges=viz, continue_edges=cont,
        )

    return {
        "ok": True,
        "n_edges": len(viz),
        "n_continue_edges": len(cont),
        "edge": edge,
        **persist,
    }


@app.post("/api/sample/{idx}/edges/bump-weight")
def bump_edge_weight(idx: int, body: BumpWeightBody):
    """Bump weight on an edge; marks it for continue-train (user_bump).

    Continue subset only stores user-added / bumped / llm-auto edges — not the
    full source annotation set. Visualization still shows the full viz list.
    """
    if body.subtype not in SUBTYPES:
        raise HTTPException(400, f"unknown subtype {body.subtype}; choose from {SUBTYPES}")
    if _data_path is None:
        raise HTTPException(400, "No data file open.")
    delta = float(body.delta)
    if delta == 0:
        raise HTTPException(400, "delta must be non-zero")

    with _state_lock:
        source, viz, cont = _current_viz_and_continue(idx)
        found = None
        for e in viz:
            if (
                int(e.get("src", -1)) == body.src
                and int(e.get("dst", -1)) == body.dst
                and str(e.get("subtype", "")) == body.subtype
            ):
                found = e
                break
        if found is None:
            raise HTTPException(404, "edge not found — add it first, then bump weight")

        try:
            old_w = float(found.get("weight", 1.0))
        except (TypeError, ValueError):
            old_w = 1.0
        new_w = max(1.0, old_w + delta)
        prev_contrib = str(found.get("contrib") or "source")
        # Keep user_add / llm_auto; promote source → user_bump for continue.
        new_contrib = prev_contrib if prev_contrib in CONTINUE_CONTRIBS else "user_bump"
        found["weight"] = new_w
        found["contrib"] = new_contrib

        cont_by = {_edge_key(e): e for e in cont}
        cont_by[_edge_key(found)] = _normalize_edge(found, contrib=new_contrib, weight=new_w)
        cont = list(cont_by.values())

        persist = _write_continue_payload(
            idx, source, viz_edges=viz, continue_edges=cont,
        )

    return {
        "ok": True,
        "n_edges": len(viz),
        "n_continue_edges": len(cont),
        "edge": {
            "src": body.src,
            "dst": body.dst,
            "subtype": body.subtype,
            "weight": new_w,
            "contrib": new_contrib,
        },
        "old_weight": old_w,
        "new_weight": new_w,
        **persist,
    }


class ProbeFocusCacheBody(BaseModel):
    probe_tokens: list[str]
    probe_answer_start: int = 0
    probe_focus_src: int
    probe_focus_dst: int
    probe_src_token: str = ""
    probe_dst_token: str = ""
    probe_mid_text: str | None = None
    query_mode: str = "manual"


@app.post("/api/probe-focus-cache")
def put_probe_focus_cache(body: ProbeFocusCacheBody):
    """Store probe/test FIM context for a later auto-annotate call (cross-origin)."""
    import uuid

    if not body.probe_tokens:
        raise HTTPException(400, "probe_tokens required")
    pid = uuid.uuid4().hex[:16]
    payload = body.model_dump()
    with _state_lock:
        while len(_probe_focus_cache) >= _PROBE_FOCUS_CACHE_MAX:
            # Drop oldest insertion order (Py3.7+ dict).
            _probe_focus_cache.pop(next(iter(_probe_focus_cache)))
        _probe_focus_cache[pid] = payload
    print(
        f"[auto-annotate] cached probe_id={pid} "
        f"tokens={len(body.probe_tokens)} "
        f"focus={body.probe_src_token!r}→{body.probe_dst_token!r} "
        f"mode={body.query_mode}",
        flush=True,
    )
    return {"ok": True, "probe_id": pid}


@app.get("/api/probe-focus-cache/{probe_id}")
def get_probe_focus_cache(probe_id: str):
    with _state_lock:
        payload = _probe_focus_cache.get(probe_id)
    if payload is None:
        raise HTTPException(404, "probe focus cache miss or expired")
    return {"ok": True, "probe": payload}


class AutoAnnotateBody(BaseModel):
    """Probe/test focus (full FIM context) → annotate edges on this train sample.

    Edges returned by the LLM are always indices on *this train sample*.
    Probe indices must never be written as train edges.
    """

    probe_src_token: str = ""
    probe_dst_token: str = ""
    focus_src_token: str = ""
    focus_dst_token: str = ""
    # Full probe/test sequence for FIM view (SOURCE/TARGET marked server-side).
    probe_tokens: list[str] | None = None
    probe_answer_start: int | None = None
    probe_focus_src: int | None = None
    probe_focus_dst: int | None = None
    # Predict: model completion as MID. Gold/manual: omit (use probe tokens' MID).
    probe_mid_text: str | None = None
    # Optional prebuilt probe FIM markdown (fallback if tokens omitted).
    probe_fim_view: str | None = None
    # Or load probe fields from server cache (set by correlation-report).
    probe_id: str | None = None
    # Viewer yellow highlight only (not injected into LLM prompt).
    hint_train_src: int | None = None
    hint_train_dst: int | None = None
    focus_src: int | None = None
    focus_dst: int | None = None
    query_mode: str = Field(
        "manual",
        description="predict | gold | manual — selects probe MID; not shown to the LLM",
    )
    # Train-sample MID override (rare); normally train gold completion.
    mid_text: str | None = None
    # When omitted, server uses ANNOTATE_MAX_EDGES from eif_api.env (default 8).
    max_edges: int | None = Field(None, ge=1, le=32)


def _annotate_max_edges(override: int | None) -> int:
    if override is not None:
        return max(1, min(32, int(override)))
    raw = (os.environ.get("ANNOTATE_MAX_EDGES") or os.environ.get("EIF_ANNOTATE_MAX_EDGES") or "8").strip()
    try:
        return max(1, min(32, int(raw)))
    except ValueError:
        return 8


@app.post("/api/sample/{idx}/auto-annotate")
def auto_annotate(idx: int, body: AutoAnnotateBody):
    """LLM-propose continue-train edges for a probe focus mechanism.

    Continuesubset ``attention_edges`` become the LLM edges (plus any prior
    user_add/user_bump). Source corpus labels are not copied into continue.
    Visualization merges source + continue contrib edges.
    """
    if _data_path is None:
        raise HTTPException(400, "No data file open.")

    with _state_lock:
        source, viz, cont = _current_viz_and_continue(idx)
        input_ids = [int(x) for x in (source.get("input_ids") or [])]
        labels_raw = source.get("labels", source.get("label")) or []
        labels = [int(x) for x in labels_raw] if isinstance(labels_raw, list) else []
        tokens = _surface_tokens_from_obj(source, input_ids)
        n = len(tokens)

        probe_src = (body.probe_src_token or body.focus_src_token or "").strip()
        probe_dst = (body.probe_dst_token or body.focus_dst_token or "").strip()

        hint_src = body.hint_train_src
        hint_dst = body.hint_train_dst
        if hint_src is None and body.focus_src is not None:
            hint_src = int(body.focus_src)
        if hint_dst is None and body.focus_dst is not None:
            hint_dst = int(body.focus_dst)
        if hint_src is not None and not (0 <= int(hint_src) < n):
            hint_src = None
        if hint_dst is not None and not (0 <= int(hint_dst) < n):
            hint_dst = None

        probe_tokens = (
            [str(t) for t in body.probe_tokens] if body.probe_tokens else None
        )
        probe_mid = str(body.probe_mid_text) if body.probe_mid_text else None
        probe_answer_start = body.probe_answer_start
        probe_focus_src = body.probe_focus_src
        probe_focus_dst = body.probe_focus_dst
        probe_fim_view = body.probe_fim_view

        if body.probe_id:
            with _state_lock:
                cached = _probe_focus_cache.get(str(body.probe_id))
            if cached:
                if not probe_tokens and cached.get("probe_tokens"):
                    probe_tokens = [str(t) for t in cached["probe_tokens"]]
                if probe_answer_start is None and cached.get("probe_answer_start") is not None:
                    probe_answer_start = int(cached["probe_answer_start"])
                if probe_focus_src is None and cached.get("probe_focus_src") is not None:
                    probe_focus_src = int(cached["probe_focus_src"])
                if probe_focus_dst is None and cached.get("probe_focus_dst") is not None:
                    probe_focus_dst = int(cached["probe_focus_dst"])
                if not probe_mid and cached.get("probe_mid_text"):
                    probe_mid = str(cached["probe_mid_text"])
                if not probe_src and cached.get("probe_src_token"):
                    probe_src = str(cached["probe_src_token"]).strip()
                if not probe_dst and cached.get("probe_dst_token"):
                    probe_dst = str(cached["probe_dst_token"]).strip()

        if not probe_src and probe_focus_src is not None and probe_tokens:
            i = int(probe_focus_src)
            if 0 <= i < len(probe_tokens):
                probe_src = str(probe_tokens[i])
        if not probe_dst and probe_focus_dst is not None and probe_tokens:
            i = int(probe_focus_dst)
            if 0 <= i < len(probe_tokens):
                probe_dst = str(probe_tokens[i])
        if not probe_src and hint_src is not None:
            probe_src = str(tokens[int(hint_src)])
        if not probe_dst and hint_dst is not None:
            probe_dst = str(tokens[int(hint_dst)])
        if not probe_src or not probe_dst:
            raise HTTPException(
                400,
                "probe_src_token and probe_dst_token required "
                "(or probe_focus_* / probe_id / hint_train_* fallback)",
            )

        if labels and len(labels) == n:
            answer_start = _answer_start(labels)
        else:
            answer_start = _answer_start([-100] * n)

        train_mid_override = str(body.mid_text) if body.mid_text else None

    max_edges_n = _annotate_max_edges(body.max_edges)
    uid = str(source.get("uid") or "") or None
    print(
        f"[auto-annotate] start train_sample={idx}"
        f"{f' uid={uid}' if uid else ''}"
        f" probe={probe_src!r}→{probe_dst!r}"
        f" probe_tokens={'yes' if probe_tokens else 'no'}"
        f" probe_id={body.probe_id or '-'}"
        f" max_edges={max_edges_n}"
        f" lang={source.get('language') or '-'}",
        flush=True,
    )

    try:
        from server.auto_annotate import call_llm_auto_annotate

        proposed, raw = call_llm_auto_annotate(
            tokens=tokens,
            probe_src_token=str(probe_src),
            probe_dst_token=str(probe_dst),
            language=str(source.get("language") or ""),
            max_edges=max_edges_n,
            answer_start=int(answer_start),
            mid_override=train_mid_override,
            sample_id=int(idx),
            sample_uid=uid,
            labels=labels if labels else None,
            probe_tokens=probe_tokens,
            probe_answer_start=probe_answer_start,
            probe_focus_src=probe_focus_src,
            probe_focus_dst=probe_focus_dst,
            probe_mid_override=probe_mid,
            probe_fim_view=probe_fim_view,
        )
    except Exception as exc:
        print(f"[auto-annotate] FAIL train_sample={idx}: {exc}", flush=True)
        raise HTTPException(502, f"auto-annotate LLM failed: {exc}") from exc

    print(
        f"[auto-annotate] LLM ok train_sample={idx} edges={len(proposed)} "
        f"raw_chars={len(raw or '')}",
        flush=True,
    )
    with _state_lock:
        source, viz, cont = _current_viz_and_continue(idx)
        cont_keep = [
            e for e in cont
            if str(e.get("contrib") or "") in ("user_add", "user_bump")
        ]
        viz_keep = [
            e for e in viz
            if str(e.get("contrib") or "") != "llm_auto"
        ]
        new_llm: list[dict[str, Any]] = []
        for p in proposed:
            edge = _normalize_edge(
                {
                    "src": p["src"],
                    "dst": p["dst"],
                    "subtype": p["subtype"],
                    "reason": p.get("reason") or "",
                    "source": "LLMAuto",
                    "weight": 1.0,
                },
                contrib="llm_auto",
            )
            new_llm.append(edge)

        by_viz = {_edge_key(e): e for e in viz_keep}
        for e in cont_keep + new_llm:
            by_viz[_edge_key(e)] = e
        viz_out = list(by_viz.values())
        cont_out = cont_keep + new_llm
        cont_map = {_edge_key(e): e for e in cont_out}
        cont_out = list(cont_map.values())

        persist = _write_continue_payload(
            idx, source, viz_edges=viz_out, continue_edges=cont_out,
        )

    print(
        f"[auto-annotate] done train_sample={idx} "
        f"llm_edges={len(new_llm)} continue_edges={len(cont_out)} "
        f"viz_edges={len(viz_out)} path={persist.get('continue_path') or '-'}",
        flush=True,
    )

    return {
        "ok": True,
        "n_edges": len(viz_out),
        "n_continue_edges": len(cont_out),
        "proposed": new_llm,
        "raw_preview": (raw or "")[:2000],
        **persist,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train annotation viewer server")
    parser.add_argument(
        "--data",
        type=str,
        default=str(DEFAULT_DATA),
        help="Source train JSONL (read-only browse). Default: ANNOTATION_TRAIN_DATA",
    )
    parser.add_argument(
        "--continue-data",
        type=str,
        default=str(DEFAULT_CONTINUE) if DEFAULT_CONTINUE else "",
        help=(
            "Writable continue-train subset JSONL for add/delete upserts. "
            "Default: ANNOTATION_CONTINUE_TRAIN_DATA from eif_api.env"
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="",
        help=(
            "Tokenizer-only path for decoding input_ids (no GPU / no live saliency). "
            "Default: EIF_BASE_MODEL_PATH from eif_api.env"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="Optional Qwen path for live ALTI saliency (also used as tokenizer if --tokenizer omitted)",
    )
    parser.add_argument(
        "--saliency-cache",
        type=str,
        default="",
        help="Directory of precomputed saliency JSON files ({idx}.json). "
             "Use this on a machine without GPU after copying from a GPU server.",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)

    global _data_path, _offsets, _continue_path, _continue_offsets, _continue_key_to_idx
    global _model_path, _tokenizer_path, _saliency_cache_dir, _corpus_path
    print(f"Source data (.env/CLI): {args.data}", flush=True)
    data = Path(args.data).expanduser().resolve()
    if not data.exists():
        print(f"[WARN] data file not found yet: {data}", flush=True)
    else:
        _data_path = data
        print(f"Indexing source {data} ...", flush=True)
        _offsets = _build_offsets(data)
        print(f"  {len(_offsets) - 1} samples (read-only)", flush=True)

    cont_raw = (args.continue_data or "").strip()
    if cont_raw:
        _continue_path = Path(cont_raw).expanduser().resolve()
        if _data_path is not None and _continue_path.resolve() == _data_path.resolve():
            raise SystemExit(
                "continue-data must differ from source --data. "
                "Edits must go to a separate small JSONL."
            )
        if not _continue_path.is_file():
            _continue_path.parent.mkdir(parents=True, exist_ok=True)
            _continue_path.write_text("", encoding="utf-8")
            print(
                f"Created empty continue-train JSONL: {_continue_path}",
                flush=True,
            )
        print(f"Indexing continue-train {_continue_path} ...", flush=True)
        _rebuild_continue_index()
        print(
            f"  {max(0, len(_continue_offsets) - 1)} samples "
            f"({len(_continue_key_to_idx)} keys)",
            flush=True,
        )
    else:
        print(
            "[WARN] No --continue-data / ANNOTATION_CONTINUE_TRAIN_DATA. "
            "Browse works; add/delete will refuse until configured.",
            flush=True,
        )

    if DEFAULT_CORPUS and DEFAULT_CORPUS.is_file():
        _corpus_path = DEFAULT_CORPUS
        print(f"LLM train corpus: {_corpus_path} (line index on first open)", flush=True)
    else:
        print(
            "[WARN] No EIF_LLM_TRAIN_CORPUS — corpus manual-annotation mode unavailable.",
            flush=True,
        )

    if args.saliency_cache:
        _saliency_cache_dir = Path(args.saliency_cache).expanduser().resolve()
        _saliency_cache_dir.mkdir(parents=True, exist_ok=True)
        n_files = sum(1 for _ in _saliency_cache_dir.glob("*.json"))
        print(f"Saliency disk cache: {_saliency_cache_dir} ({n_files} files)", flush=True)

    if args.tokenizer:
        _tokenizer_path = str(Path(args.tokenizer).expanduser().resolve())
        print(f"Tokenizer: {_tokenizer_path} (decode-only)", flush=True)
    if args.model:
        _model_path = str(Path(args.model).expanduser().resolve())
        print(f"Saliency model: {_model_path}", flush=True)
    elif not args.tokenizer:
        if _TOKENIZER_RAW and DEFAULT_TOKENIZER.expanduser().exists():
            print(
                f"Tokenizer from EIF_BASE_MODEL_PATH: {DEFAULT_TOKENIZER.expanduser().resolve()} "
                f"(live saliency {'disk_cache' if _saliency_cache_dir else 'off'})",
                flush=True,
            )
        else:
            print(
                "[WARN] No tokenizer configured. "
                "Set EIF_BASE_MODEL_PATH in eif_api.env or pass --tokenizer <path>.",
                flush=True,
            )

    # Eager-load tokenizer for faster first sample
    try:
        tok_path_str = _resolve_tokenizer_path()
        tok_path = Path(tok_path_str) if tok_path_str else Path()
        if tok_path_str and tok_path.exists():
            _get_tokenizer()
            print(f"Tokenizer ready: {tok_path}", flush=True)
        elif tok_path_str:
            print(f"[WARN] tokenizer path not found: {tok_path}", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] tokenizer not loaded: {exc}", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
