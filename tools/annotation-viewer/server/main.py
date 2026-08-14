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
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
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


DEFAULT_DATA = _default_data_path()
DEFAULT_CONTINUE = _default_continue_path()
# Tokenizer/decode: use base model path (no separate ANNOTATION_TOKENIZER).
_TOKENIZER_RAW = (os.environ.get("EIF_BASE_MODEL_PATH") or "").strip()
DEFAULT_TOKENIZER = Path(_TOKENIZER_RAW).expanduser() if _TOKENIZER_RAW else Path()

SUBTYPES = [
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
_tokenizer = None
_tokenizer_path: str | None = None
_model = None
_model_path: str | None = None
_saliency_cache_dir: Path | None = None
_saliency_cache: dict[tuple[int, int], list[dict[str, Any]]] = {}


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
    uid = obj.get("uid")
    if isinstance(uid, str) and uid.strip():
        return f"uid:{uid.strip()}"
    raw_id = obj.get("raw_id")
    if isinstance(raw_id, str) and raw_id.strip():
        return f"raw_id:{raw_id.strip()}"
    stamped = obj.get("source_train_index")
    if isinstance(stamped, int) and stamped >= 0:
        return f"source_idx:{stamped}"
    if source_idx is not None and source_idx >= 0:
        return f"source_idx:{source_idx}"
    # Last resort: content fingerprint (stable enough for interactive edits).
    ids = obj.get("input_ids") or []
    return f"ids:{len(ids)}:{hash(tuple(int(x) for x in ids[:64]))}"


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
            key = _sample_key(obj)
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
    key = _sample_key(source_obj, source_idx)
    return key, _continue_key_to_idx.get(key)


def _effective_sample(idx: int) -> tuple[dict[str, Any], bool, str]:
    """Source row overlaid with continue-train edit if present.

    Returns (obj, from_continue, sample_key).
    """
    source = _read_sample(idx)
    key, cont_idx = _lookup_continue(idx, source)
    if cont_idx is None:
        return source, False, key
    overlay = _read_continue_by_idx(cont_idx)
    # Prefer continue annotations; keep source fields if continue omitted them.
    merged = dict(source)
    merged.update(overlay)
    return merged, True, key


def _upsert_continue(source_idx: int, obj: dict[str, Any]) -> dict[str, Any]:
    """Write edited sample into the continue subset (insert or replace by key)."""
    global _continue_offsets
    path = _ensure_continue_path()
    obj = dict(obj)
    obj["source_train_index"] = int(source_idx)
    if _data_path is not None:
        obj["source_train_path"] = str(_data_path)
    key = _sample_key(obj, source_idx)
    existing = _continue_key_to_idx.get(key)

    if existing is None:
        _continue_offsets = _append_jsonl_line(path, obj)
        _continue_key_to_idx[key] = max(0, len(_continue_offsets) - 2)
        action = "inserted"
    else:
        _continue_offsets = _rewrite_jsonl_line(path, _continue_offsets, existing, obj)
        # Offsets changed; rebuild key map (line order preserved).
        _rebuild_continue_index()
        action = "updated"

    _saliency_cache.clear()
    return {
        "ok": True,
        "action": action,
        "key": key,
        "continue_path": str(path),
        "n_continue": max(0, len(_continue_offsets) - 1),
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


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    tok_path = _resolve_tokenizer_path()
    return {
        "ok": True,
        "data_path": str(_data_path) if _data_path else None,
        "n_samples": max(0, len(_offsets) - 1) if _offsets else 0,
        "continue_path": str(_continue_path) if _continue_path else None,
        "n_continue": max(0, len(_continue_offsets) - 1) if _continue_offsets else 0,
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
            key = _sample_key(obj, i)
            in_continue = key in _continue_key_to_idx
            if in_continue:
                try:
                    cont = _read_continue_by_idx(_continue_key_to_idx[key])
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
    input_ids = [int(x) for x in (obj.get("input_ids") or [])]
    labels = [int(x) for x in (obj.get("label") or [])]
    tokens = _surface_tokens_from_obj(obj, input_ids)

    edges = []
    for e in obj.get("attention_edges") or []:
        if not isinstance(e, dict):
            continue
        edges.append(
            {
                "src": int(e["src"]),
                "dst": int(e["dst"]),
                "subtype": str(e.get("subtype") or ""),
            }
        )

    return {
        "index": idx,
        "uid": obj.get("uid"),
        "language": obj.get("language"),
        "raw_id": obj.get("raw_id"),
        "tokens": tokens,
        "input_ids": input_ids,
        "answer_start": _answer_start(labels),
        "attention_edges": edges,
        "annotation_meta": obj.get("annotation_meta") or {},
        "subtypes": SUBTYPES,
        "in_continue": from_continue,
        "sample_key": key,
        "continue_path": str(_continue_path) if _continue_path else None,
    }


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
        obj, _, _ = _effective_sample(idx)
        edges = list(obj.get("attention_edges") or [])
        before = len(edges)
        edges = [
            e
            for e in edges
            if not (
                isinstance(e, dict)
                and int(e.get("src", -1)) == body.src
                and int(e.get("dst", -1)) == body.dst
                and str(e.get("subtype", "")) == body.subtype
            )
        ]
        if len(edges) == before:
            raise HTTPException(404, "edge not found in attention_edges")
        obj["attention_edges"] = edges

        # Best-effort: also drop a matching simple-token annotation if indices coincide.
        anns = list(obj.get("annotations") or [])
        obj["annotations"] = [
            a
            for a in anns
            if not (
                isinstance(a, dict)
                and int(a.get("token_i_idx", -1)) == body.src
                and int(a.get("token_j_idx", -1)) == body.dst
                and str(a.get("subtype", "")) == body.subtype
            )
        ]
        _sync_meta(obj)
        persist = _upsert_continue(idx, obj)

    return {"ok": True, "n_edges": len(edges), **persist}


@app.post("/api/sample/{idx}/edges/add")
def add_edge(idx: int, body: AddEdgeBody):
    if body.subtype not in SUBTYPES:
        raise HTTPException(400, f"unknown subtype {body.subtype}; choose from {SUBTYPES}")
    if body.src == body.dst:
        raise HTTPException(400, "src and dst must differ")
    if _data_path is None:
        raise HTTPException(400, "No data file open.")

    with _state_lock:
        obj, _, _ = _effective_sample(idx)
        n = len(obj.get("input_ids") or [])
        if not (0 <= body.src < n and 0 <= body.dst < n):
            raise HTTPException(400, f"src/dst out of range 0..{n-1}")

        edges = list(obj.get("attention_edges") or [])
        for e in edges:
            if (
                isinstance(e, dict)
                and int(e.get("src", -1)) == body.src
                and int(e.get("dst", -1)) == body.dst
                and str(e.get("subtype", "")) == body.subtype
            ):
                raise HTTPException(409, "edge already exists")

        edges.append({"src": body.src, "dst": body.dst, "subtype": body.subtype})
        obj["attention_edges"] = edges

        anns = list(obj.get("annotations") or [])
        anns.append(
            {
                "token_i_idx": body.src,
                "token_j_idx": body.dst,
                "subtype": body.subtype,
                "source": body.source,
            }
        )
        obj["annotations"] = anns
        _sync_meta(obj)
        persist = _upsert_continue(idx, obj)

    return {"ok": True, "n_edges": len(edges), "edge": body.model_dump(), **persist}

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
    global _model_path, _tokenizer_path, _saliency_cache_dir
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
