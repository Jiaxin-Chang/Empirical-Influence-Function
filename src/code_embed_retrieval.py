"""Full-text × one-line hole neighborhood embedding retrieval.

Train rows are embedded twice:

* full — ``prefix + middle + suffix``, first 4000 characters. This is the
  existing ``train_code_embeddings.npz`` and is not rebuilt when that file
  already matches the train set.
* gold — the line before the hole, the middle, and the line after the hole.
  Stored in ``train_gold_embeddings.npz`` with ``gold_window=line1``.

The rank score is ``cos(full) ** (1 - w) * cos(gold) ** w``. ``w`` is the
hole weight from the UI.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.fim_mid_rewrite import find_fim_span

REPO_ROOT = Path(__file__).resolve().parent.parent
EMBED_CHARS = 4000
GOLD_WINDOW = "line1"
DEFAULT_HOLE_WEIGHT = 0.4
_RESULTS_SERVER = Path(
    "/mnt/md124/jiaxin/Empirical-Influence-Function/mutation_attribution/results"
)
_LOCK = threading.Lock()
_PROGRESS_LOCK = threading.Lock()
_BANK: dict[str, Any] = {}
_BUILD_THREAD: threading.Thread | None = None
_PROGRESS: dict[str, Any] = {
    "status": "idle",
    "phase": "",
    "done": 0,
    "total": 0,
    "message": "",
    "error": "",
}


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def fim_parts(prompt: str, middle: str) -> tuple[str, str, str]:
    """Drop the template space after <PRE> and <SUF>. Middle is the gold fill."""
    span = find_fim_span(prompt)
    prefix = span.prefix[1:] if span.prefix.startswith(" ") else span.prefix
    suffix = span.suffix[1:] if span.suffix.startswith(" ") else span.suffix
    return prefix, middle, suffix


def full_text(prefix: str, middle: str, suffix: str) -> str:
    return f"{prefix}{middle}{suffix}".strip()[:EMBED_CHARS] or " "


def line_before(prefix: str) -> str:
    """The source line that ends where the hole starts."""
    if not prefix:
        return ""
    core = prefix[:-1] if prefix.endswith("\n") else prefix
    if not prefix.endswith("\n"):
        cut = core.rfind("\n")
        if cut < 0:
            return ""
        core = core[:cut]
    if not core:
        return ""
    cut = core.rfind("\n")
    return core[cut + 1 :] + "\n"


def line_after(middle: str, suffix: str) -> str:
    """The source line that starts where the hole ends."""
    rest = suffix
    if not middle.endswith("\n"):
        cut = rest.find("\n")
        if cut < 0:
            return ""
        rest = rest[cut + 1 :]
    if not rest:
        return ""
    cut = rest.find("\n")
    if cut < 0:
        return rest
    return rest[: cut + 1]


def gold_text(prefix: str, middle: str, suffix: str) -> str:
    return f"{line_before(prefix)}{middle}{line_after(middle, suffix)}".strip() or " "


def resolve_code_embed_train() -> Path:
    raw = _env("EIF_LLM_TRAIN_CORPUS") or _env("EIF_TRAIN_CORPUS")
    if raw:
        return Path(raw).expanduser()
    server = Path("/mnt/md124/jiaxin/training_code/data/csn_go_train_fim_10k.jsonl")
    local = REPO_ROOT / "csn_go_train_fim_10k.jsonl"
    if server.is_file():
        return server
    return local


def _results_dir() -> Path:
    if _RESULTS_SERVER.is_dir():
        return _RESULTS_SERVER
    local = REPO_ROOT / "mutation_attribution" / "results"
    local.mkdir(parents=True, exist_ok=True)
    return local


def gold_npz_path() -> Path:
    raw = _env("EIF_GOLD_EMBEDDINGS")
    if raw:
        return Path(raw).expanduser()
    return _results_dir() / "train_gold_embeddings.npz"


def full_npz_path() -> Path:
    raw = _env("EIF_CODE_EMBEDDINGS")
    if raw:
        return Path(raw).expanduser()
    return _results_dir() / "train_code_embeddings.npz"


def _cache_model(data) -> str:
    if "model" not in data.files:
        return ""
    raw = data["model"]
    return str(raw.item() if hasattr(raw, "item") else raw)


def _cache_kind(data) -> str:
    if "kind" not in data.files:
        return ""
    raw = data["kind"]
    return str(raw.item() if hasattr(raw, "item") else raw)


def _cache_window(data) -> str:
    if "gold_window" not in data.files:
        return ""
    raw = data["gold_window"]
    return str(raw.item() if hasattr(raw, "item") else raw)


def _api_model() -> str:
    from src.fim_semantic_index import embed_model_id

    model = embed_model_id()
    if not model or model.startswith("local:"):
        raise RuntimeError(
            "Set EIF_SEMANTIC_EMBED_MODEL and EIF_SEMANTIC_EMBED_BASE_URL "
            "for code embedding retrieval."
        )
    return model


def _cache_matches(data, kind: str, n_total: int, model: str) -> bool:
    if _cache_model(data) != model or int(data["n"]) != n_total:
        return False
    stored = _cache_kind(data)
    if kind == "full":
        return stored in ("", "full")
    if kind == "gold":
        return stored == "gold" and _cache_window(data) == GOLD_WINDOW
    return False


def _set_progress(**fields: Any) -> None:
    with _PROGRESS_LOCK:
        _PROGRESS.update(fields)


def _progress_copy() -> dict[str, Any]:
    with _PROGRESS_LOCK:
        return dict(_PROGRESS)


def _load_train_rows() -> list[dict[str, str]]:
    path = resolve_code_embed_train()
    if not path.is_file():
        raise FileNotFoundError(f"train file for the embedding cache not found: {path}")
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            rows.append({
                "task_id": str(row.get("task_id") or ""),
                "prefix": str(row.get("prefix") or ""),
                "middle": str(row.get("middle") or ""),
                "suffix": str(row.get("suffix") or ""),
            })
    if not rows:
        raise RuntimeError(f"train file is empty: {path}")
    return rows


def _row_text(row: dict[str, str], kind: str) -> str:
    if kind == "gold":
        return gold_text(row["prefix"], row["middle"], row["suffix"])
    return full_text(row["prefix"], row["middle"], row["suffix"])


def _save_npz(
    path: Path,
    mat: np.ndarray,
    model: str,
    n_total: int,
    n_done: int,
    normalized: bool,
    kind: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".npz.partial")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            vectors=np.asarray(mat[:n_done], dtype=np.float32),
            model=np.array(model),
            n=np.array(n_total),
            n_done=np.array(n_done),
            normalized=np.array(1 if normalized else 0),
            kind=np.array(kind),
            gold_window=np.array(GOLD_WINDOW if kind == "gold" else ""),
        )
    tmp.replace(path)


def _file_ready(path: Path, kind: str, n_total: int, model: str) -> bool:
    if not path.is_file():
        return False
    data = np.load(path)
    if not _cache_matches(data, kind, n_total, model):
        return False
    n_done = int(data["n_done"]) if "n_done" in data.files else int(data["vectors"].shape[0])
    normalized = int(data["normalized"]) if "normalized" in data.files else 0
    return n_done >= n_total and normalized == 1


def _embed_chunk(model: str, chunk: list[str]) -> list[list[float]]:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

    from src.fim_semantic_index import embed_texts

    wait = 2.0
    last: Exception | None = None
    for _attempt in range(10):
        try:
            return embed_texts(chunk, model=model, batch=len(chunk))
        except RateLimitError as exc:
            last = exc
        except APIStatusError as exc:
            last = exc
            if int(exc.status_code) not in (408, 409, 429, 500, 502, 503, 504):
                raise
        except (APIConnectionError, APITimeoutError) as exc:
            last = exc
        print(f"[code-embed] {type(last).__name__}, sleep {wait:.0f}s", flush=True)
        time.sleep(wait)
        wait = min(60.0, wait * 2)
    raise RuntimeError("embedding request failed after retries") from last


def _build_matrix(rows: list[dict[str, str]], kind: str, model: str) -> None:
    path = gold_npz_path() if kind == "gold" else full_npz_path()
    n_total = len(rows)
    if _file_ready(path, kind, n_total, model):
        print(f"[code-embed] {kind} cache hit {path}", flush=True)
        return
    texts = [_row_text(row, kind) for row in rows]
    mat = None
    start = 0
    if path.is_file():
        data = np.load(path)
        same = _cache_matches(data, kind, n_total, model)
        if same:
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            n_done = int(data["n_done"]) if "n_done" in data.files else int(vectors.shape[0])
            normalized = int(data["normalized"]) if "normalized" in data.files else 1
            if n_done >= n_total and normalized:
                return
            if n_done > 0 and not normalized:
                mat = np.zeros((n_total, vectors.shape[1]), dtype=np.float32)
                mat[:n_done] = vectors[:n_done]
                start = n_done
                print(f"[code-embed] {kind} resume {start}/{n_total}", flush=True)
    _set_progress(
        status="building",
        phase=kind,
        done=start,
        total=n_total,
        message=f"正在写入训练集 {kind} embedding {start}/{n_total}",
        error="",
    )
    # qwen3.7-text-embedding-flash 拒绝一次超过 25 条。20 是安全上限。
    # 限流由 _embed_chunk 退避，这里不再每批停一秒。
    step = 20
    pause = 0.0
    cursor = start
    while cursor < n_total:
        chunk = texts[cursor : cursor + step]
        block = np.asarray(_embed_chunk(model, chunk), dtype=np.float32)
        if mat is None:
            mat = np.zeros((n_total, block.shape[1]), dtype=np.float32)
        mat[cursor : cursor + len(chunk)] = block
        cursor += len(chunk)
        if cursor == n_total or cursor % 160 == 0:
            _save_npz(path, mat, model, n_total, cursor, False, kind)
        _set_progress(
            status="building",
            phase=kind,
            done=cursor,
            total=n_total,
            message=f"正在写入训练集 {kind} embedding {cursor}/{n_total}",
        )
        print(f"[code-embed] {kind} {cursor}/{n_total}", flush=True)
        if cursor < n_total and pause > 0:
            time.sleep(pause)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-12, None)
    _save_npz(path, mat, model, n_total, n_total, True, kind)
    print(f"[code-embed] {kind} saved {path}", flush=True)


def _build_all() -> None:
    try:
        model = _api_model()
        rows = _load_train_rows()
        _set_progress(
            status="building",
            phase="gold",
            done=0,
            total=len(rows),
            message="正在写入训练集 gold embedding",
            error="",
        )
        _build_matrix(rows, "full", model)
        _build_matrix(rows, "gold", model)
        _set_progress(
            status="ready",
            phase="",
            done=len(rows),
            total=len(rows),
            message="训练集全文与洞附近 embedding 已就绪",
            error="",
        )
    except Exception as exc:
        print(f"[code-embed] build failed: {exc}", flush=True)
        _set_progress(status="error", message=str(exc), error=str(exc))


def _ensure_started(*, restart: bool) -> dict[str, Any]:
    global _BUILD_THREAD
    model = _api_model()
    train_path = resolve_code_embed_train()
    n_total = 0
    if train_path.is_file():
        with train_path.open(encoding="utf-8") as handle:
            n_total = sum(1 for line in handle if line.strip())
    gold_path = gold_npz_path()
    full_path = full_npz_path()
    if n_total and _file_ready(gold_path, "gold", n_total, model) and _file_ready(full_path, "full", n_total, model):
        _set_progress(status="ready", error="", done=n_total, total=n_total, phase="")
        return {"ready": True}

    with _LOCK:
        if (
            n_total
            and _file_ready(gold_path, "gold", n_total, model)
            and _file_ready(full_path, "full", n_total, model)
        ):
            return {"ready": True}
        progress = _progress_copy()
        alive = _BUILD_THREAD is not None and _BUILD_THREAD.is_alive()
        if alive:
            return {"ready": False, **progress}
        if progress.get("status") == "error" and not restart:
            return {"ready": False, **progress}
        _set_progress(status="building", phase="gold", done=0, total=n_total, message="正在写入训练集 gold embedding", error="")
        _BUILD_THREAD = threading.Thread(target=_build_all, name="code-embed-cache", daemon=True)
        _BUILD_THREAD.start()
        return {"ready": False, **_progress_copy()}


def _load_matrix(path: Path, kind: str, n_total: int, model: str) -> np.ndarray:
    data = np.load(path)
    if not _cache_matches(data, kind, n_total, model):
        raise RuntimeError(f"{kind} embedding cache does not match the train file: {path}")
    vectors = np.asarray(data["vectors"], dtype=np.float32)
    n_done = int(data["n_done"]) if "n_done" in data.files else int(vectors.shape[0])
    if n_done < n_total:
        raise RuntimeError(f"{kind} embedding cache is incomplete: {n_done}/{n_total}")
    mat = vectors[:n_total]
    normalized = int(data["normalized"]) if "normalized" in data.files else 0
    if not normalized:
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / np.clip(norms, 1e-12, None)
    return np.ascontiguousarray(mat, dtype=np.float32)


def _load_bank() -> dict[str, Any]:
    model = _api_model()
    train_path = resolve_code_embed_train()
    gold_path = gold_npz_path()
    full_path = full_npz_path()
    key = (
        str(gold_path.resolve()),
        gold_path.stat().st_mtime_ns,
        str(full_path.resolve()),
        full_path.stat().st_mtime_ns,
        str(train_path.resolve()),
        train_path.stat().st_mtime_ns,
        model,
    )
    with _LOCK:
        if _BANK.get("key") == key:
            return _BANK["bank"]
        rows = _load_train_rows()
        n_total = len(rows)
        bank = {
            "full": _load_matrix(full_path, "full", n_total, model),
            "gold": _load_matrix(gold_path, "gold", n_total, model),
            "model": model,
            "task_ids": [row["task_id"] for row in rows],
            "previews": [
                " ".join(f"{row['prefix']}{row['middle']}{row['suffix']}".split())[:160]
                for row in rows
            ],
            "train_path": str(train_path),
            "gold_path": str(gold_path),
            "full_path": str(full_path),
        }
        _BANK["key"] = key
        _BANK["bank"] = bank
        return bank


def _as_unit(values: list[float]) -> np.ndarray:
    vec = np.asarray(values, dtype=np.float32)
    if vec.size == 0:
        raise RuntimeError("embedding API returned an empty vector")
    return vec / max(float(np.linalg.norm(vec)), 1e-12)


def _combine(context_cos: np.ndarray, gold_cos: np.ndarray, hole_weight: float) -> np.ndarray:
    weight = min(1.0, max(0.0, float(hole_weight)))
    ctx = np.clip(context_cos, 0.0, 1.0)
    gold = np.clip(gold_cos, 0.0, 1.0)
    if weight <= 0.0:
        return ctx
    if weight >= 1.0:
        return gold
    return np.exp(
        (1.0 - weight) * np.log(np.clip(ctx, 1e-8, 1.0))
        + weight * np.log(np.clip(gold, 1e-8, 1.0))
    )


def retrieve_code_embeddings(
    fim_prompt: str,
    gold_completion: str,
    *,
    top_k: int = 10,
    hole_weight: float = DEFAULT_HOLE_WEIGHT,
    restart: bool = False,
) -> dict[str, Any]:
    weight = min(1.0, max(0.0, float(hole_weight)))
    state = _ensure_started(restart=restart)
    if not state.get("ready"):
        status = "error" if state.get("status") == "error" else "building"
        return {
            "status": status,
            "method": "code_embed",
            "phase": state.get("phase") or "",
            "done": int(state.get("done") or 0),
            "total": int(state.get("total") or 0),
            "message": state.get("message") or state.get("error") or "",
            "hole_weight": weight,
            "hole_window": GOLD_WINDOW,
            "gold_cache_path": str(gold_npz_path()),
            "full_cache_path": str(full_npz_path()),
            "corpus_path": str(resolve_code_embed_train()),
        }

    prefix, middle, suffix = fim_parts(fim_prompt, gold_completion)
    whole = full_text(prefix, middle, suffix)
    hole = gold_text(prefix, middle, suffix)
    bank = _load_bank()
    model = str(bank["model"])
    embedded = _embed_chunk(model, [whole, hole])
    full_vec = _as_unit(embedded[0])
    gold_vec = _as_unit(embedded[1])
    full_cos = bank["full"] @ full_vec
    gold_cos = bank["gold"] @ gold_vec
    scores = _combine(full_cos, gold_cos, weight)
    k = max(1, min(int(top_k), int(scores.shape[0])))
    order = np.argsort(-scores)[:k]
    hits = []
    for rank, idx in enumerate(order, start=1):
        i = int(idx)
        hits.append({
            "rank": rank,
            "line": i,
            "task_id": bank["task_ids"][i],
            "score": round(float(scores[i]), 6),
            "embed_score": round(float(scores[i]), 6),
            "context_score": round(float(full_cos[i]), 6),
            "full_score": round(float(full_cos[i]), 6),
            "gold_score": round(float(gold_cos[i]), 6),
            "preview": bank["previews"][i],
        })
    return {
        "status": "success",
        "method": "code_embed",
        "embed_chars": EMBED_CHARS,
        "hole_window": GOLD_WINDOW,
        "hole_weight": weight,
        "context_chars": len(whole),
        "gold_chars": len(hole),
        "model": model,
        "cache_path": bank["full_path"],
        "gold_cache_path": bank["gold_path"],
        "full_cache_path": bank["full_path"],
        "corpus_path": bank["train_path"],
        "hits": hits,
    }
