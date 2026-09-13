"""Embedding recall + structured rerank for semantic FIM retrieval.

Stage-1: cosine over mechanism-weighted canonical text (FAISS if installed, else numpy).
Stage-2: relations/pattern/role rerank. Embedding is a 0.1 tie-break, not the ranker.
Near-duplicate AWS SDK clones are diversified by mechanism signature.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.fim_semantic_schema import (
    EMBED_TEXT_KIND,
    EMBED_TIE_WEIGHT,
    REL_COS_FLOOR,
    collect_relation_endpoints,
    flatten_semantic_text_for_embedding,
    lexical_endpoint_sim,
    mechanism_signature,
    normalize_semantic_repr,
    same_mechanism_cluster,
    semantic_combined_score,
)

_INDEX_CACHE: dict[str, Any] = {}
_PHRASE_VEC_CACHE: dict[tuple[str, str], Any] = {}
_PHRASE_CACHE_MAX = 20000


def embedding_npz_path(semantic_jsonl: str | Path) -> Path:
    path = Path(semantic_jsonl)
    return path.with_suffix(path.suffix + ".embeddings.npz")


def embedding_meta_path(semantic_jsonl: str | Path) -> Path:
    path = Path(semantic_jsonl)
    return path.with_suffix(path.suffix + ".embeddings.meta.json")


def _env(name: str, default: str = "") -> str:
    import os
    return (os.environ.get(name) or default).strip()


def embed_model_id() -> str:
    return (
        _env("EIF_SEMANTIC_EMBED_MODEL")
        or _env("EMBED_MODEL")
        or ""
    )


def recall_k_default(n: int) -> int:
    raw = _env("EIF_SEMANTIC_RECALL_K") or "500"
    try:
        k = int(raw)
    except ValueError:
        k = 500
    k = max(20, min(k, 2000))
    return max(1, min(k, n))


def embed_tie_weight() -> float:
    raw = _env("EIF_SEMANTIC_EMBED_WEIGHT") or str(EMBED_TIE_WEIGHT)
    try:
        w = float(raw)
    except ValueError:
        w = EMBED_TIE_WEIGHT
    return max(0.0, min(0.3, w))


def _build_embed_client():
    from openai import OpenAI
    from src.llm_train_retrieval import _build_openai_client

    base = _env("EIF_SEMANTIC_EMBED_BASE_URL")
    if not base:
        return _build_openai_client()
    key = (
        _env("EIF_SEMANTIC_EMBED_API_KEY")
        or _env("DASHSCOPE_API_KEY")
        or _env("OPENAI_API_KEY")
        or "dummy"
    )
    return OpenAI(api_key=key, base_url=base)


def embed_texts(texts: list[str], *, model: str, batch: int = 32) -> list[list[float]]:
    if not texts:
        return []
    client = _build_embed_client()
    out: list[list[float] | None] = [None] * len(texts)
    step = max(1, int(batch))
    for start in range(0, len(texts), step):
        chunk = texts[start : start + step]
        resp = client.embeddings.create(model=model, input=chunk)
        by_idx = {int(item.index): list(item.embedding) for item in resp.data}
        for i in range(len(chunk)):
            out[start + i] = by_idx[i]
    return [v if v is not None else [] for v in out]


def _remember_phrase_vec(model: str, phrase: str, vec: Any) -> None:
    if len(_PHRASE_VEC_CACHE) >= _PHRASE_CACHE_MAX:
        for i, key in enumerate(list(_PHRASE_VEC_CACHE)):
            if i % 2 == 0:
                _PHRASE_VEC_CACHE.pop(key, None)
    _PHRASE_VEC_CACHE[(model, phrase)] = vec


def embed_relation_phrases(phrases: list[str], model: str) -> dict[str, Any]:
    table: dict[str, Any] = {}
    missing: list[str] = []
    for phrase in phrases:
        cached = _PHRASE_VEC_CACHE.get((model, phrase))
        if cached is not None:
            table[phrase] = cached
        else:
            missing.append(phrase)
    if missing:
        vecs = embed_texts(missing, model=model)
        for phrase, raw in zip(missing, vecs):
            if not raw:
                continue
            arr = _l2_normalize(raw)
            _remember_phrase_vec(model, phrase, arr)
            table[phrase] = arr
    return table


def build_endpoint_sim(phrases: list[str], model: str):
    """Cosine of relation endpoints, with lexical backup and a 0.35 floor."""
    import numpy as np

    table = embed_relation_phrases(phrases, model)

    def sim(a: str, b: str) -> float:
        from src.fim_semantic_schema import _norm_term

        an, bn = _norm_term(a), _norm_term(b)
        lex = lexical_endpoint_sim(an, bn)
        va, vb = table.get(an), table.get(bn)
        if va is None or vb is None:
            return lex
        cos = float(np.dot(va, vb))
        if cos >= REL_COS_FLOOR:
            return max(lex, max(0.0, min(1.0, cos)))
        return lex

    return sim


def _l2_normalize(arr: Any) -> Any:
    import numpy as np

    x = np.asarray(arr, dtype="float32")
    if x.ndim == 1:
        n = float(np.linalg.norm(x))
        return x if n < 1e-8 else (x / n)
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, 1e-8)
    return x / n


def _search_ip(vectors: Any, query: Any, k: int, faiss_index: Any = None) -> tuple[Any, Any]:
    import numpy as np

    n = int(vectors.shape[0])
    k = max(1, min(int(k), n))
    q = np.asarray(query, dtype="float32").reshape(-1)
    if faiss_index is not None:
        scores, idxs = faiss_index.search(q.reshape(1, -1), k)
        return scores[0], idxs[0]
    sims = vectors @ q
    if k >= n:
        order = np.argsort(-sims)
    else:
        part = np.argpartition(-sims, k)[:k]
        order = part[np.argsort(-sims[part])]
    return sims[order], order


_JSONL_CACHE: dict[str, Any] = {}


def load_semantic_rows(jsonl_path: Path) -> dict[int, dict[str, Any]]:
    st = jsonl_path.stat()
    key = f"{jsonl_path}:{st.st_mtime}:{st.st_size}"
    cached = _JSONL_CACHE.get(key)
    if cached is not None:
        return cached
    by_line: dict[int, dict[str, Any]] = {}
    with jsonl_path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            try:
                source_line = int(row.get("source_line"))
            except (TypeError, ValueError):
                source_line = idx
            row = dict(row)
            row["_semantic_line"] = idx
            by_line[source_line] = row
    _JSONL_CACHE.clear()
    _JSONL_CACHE[key] = by_line
    return by_line


def load_embed_index(npz_path: Path) -> dict[str, Any]:
    import numpy as np

    key = f"{npz_path}:{npz_path.stat().st_mtime}:{npz_path.stat().st_size}"
    cached = _INDEX_CACHE.get(key)
    if cached is not None:
        return cached
    data = np.load(npz_path)
    vectors = _l2_normalize(data["vectors"])
    source_lines = np.asarray(data["source_lines"], dtype="int64")
    packed: dict[str, Any] = {
        "vectors": vectors,
        "source_lines": source_lines,
        "n": int(vectors.shape[0]),
        "dim": int(vectors.shape[1]) if vectors.ndim == 2 else 0,
        "path": str(npz_path),
        "faiss": None,
    }
    try:
        import faiss  # type: ignore

        faiss_index = faiss.IndexFlatIP(int(vectors.shape[1]))
        faiss_index.add(np.ascontiguousarray(vectors))
        packed["faiss"] = faiss_index
    except Exception:
        packed["faiss"] = None
    _INDEX_CACHE.clear()
    _INDEX_CACHE[key] = packed
    return packed


def _hit_from_row(
    row: dict[str, Any],
    *,
    score: float,
    struct_score: float,
    embed_score: float,
    recall_rank: int,
    parts: dict[str, float],
) -> dict[str, Any]:
    doc = normalize_semantic_repr(row)
    try:
        source_line = int(row.get("source_line"))
    except (TypeError, ValueError):
        source_line = int(row.get("_semantic_line") or -1)
    return {
        "line": source_line,
        "semantic_line": int(row.get("_semantic_line") or -1),
        "task_id": str(row.get("task_id") or row.get("id") or ""),
        "semantic_score": round(float(score), 3),
        "struct_score": round(float(struct_score), 3),
        "embed_score": round(float(embed_score), 3),
        "recall_rank": int(recall_rank),
        "match_region": "semantic",
        "prompt_preview": str(row.get("prompt_preview") or "")[:280],
        "response_preview": str(row.get("response_preview") or "")[:200],
        "summary": str(doc.get("summary") or "")[:240],
        "pattern": list(doc.get("pattern") or [])[:6],
        "relations": list(doc.get("relations") or [])[:6],
        "mechanism_sig": mechanism_signature(doc),
        "score_parts": {k: round(float(v), 3) for k, v in parts.items()},
    }


def diversify_mechanism_hits(
    items: list[tuple[float, dict[str, Any], dict[str, Any]]],
    top_k: int,
    endpoint_sim: Any,
    *,
    keep_per_cluster: int = 2,
) -> list[dict[str, Any]]:
    """Greedy cluster: keep 1-2 hits per semantic mechanism, not exact string signatures."""
    picked: list[dict[str, Any]] = []
    clusters: list[tuple[dict[str, Any], int]] = []
    for _score, hit, sem in items:
        placed = False
        for i, (rep, cnt) in enumerate(clusters):
            if same_mechanism_cluster(sem, rep, endpoint_sim):
                placed = True
                if cnt < keep_per_cluster:
                    picked.append(hit)
                    clusters[i] = (rep, cnt + 1)
                break
        if not placed:
            clusters.append((sem, 1))
            picked.append(hit)
        if len(picked) >= top_k:
            break
    return picked[:top_k]


def search_embed_then_rerank(
    corpus_path: str,
    query_sem: dict[str, Any] | None,
    *,
    top_k: int = 10,
    recall_k: int | None = None,
    embed_model: str | None = None,
) -> list[dict[str, Any]]:
    jsonl_path = Path(corpus_path)
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"semantic corpus not found: {corpus_path}")
    npz_path = embedding_npz_path(jsonl_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"semantic embeddings not found: {npz_path}")
    model = (embed_model or embed_model_id()).strip()
    if not model:
        raise RuntimeError("set EIF_SEMANTIC_EMBED_MODEL / EMBED_MODEL")

    q = normalize_semantic_repr(query_sem if isinstance(query_sem, dict) else {})
    q_text = flatten_semantic_text_for_embedding(q)
    q_vec = _l2_normalize(embed_texts([q_text], model=model)[0])

    index = load_embed_index(npz_path)
    rows = load_semantic_rows(jsonl_path)
    n = int(index["n"])
    k_recall = recall_k_default(n) if recall_k is None else max(1, min(int(recall_k), n))
    scores, idxs = _search_ip(
        index["vectors"], q_vec, k_recall, faiss_index=index.get("faiss"),
    )

    raw_cands: list[tuple[int, float, dict[str, Any]]] = []
    source_lines = index["source_lines"]
    for rank, (cos, ix) in enumerate(zip(scores, idxs), start=1):
        i = int(ix)
        if i < 0 or i >= n:
            continue
        source_line = int(source_lines[i])
        row = rows.get(source_line)
        if not isinstance(row, dict):
            continue
        raw_cands.append((rank, float(cos), row))

    rel_mode = "endpoint_embed"
    try:
        phrases = collect_relation_endpoints(q, *[row for _, _, row in raw_cands])
        endpoint_sim = build_endpoint_sim(phrases, model)
    except Exception as exc:
        print(f"[llm-semantic] relation endpoint embed failed ({exc}); lexical fallback", flush=True)
        endpoint_sim = lexical_endpoint_sim
        rel_mode = "lexical"

    w = embed_tie_weight()
    ranked_items: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
    for rank, embed_cos, row in raw_cands:
        combined, struct, parts = semantic_combined_score(
            q,
            row,
            embed_cos=embed_cos,
            embed_weight=w,
            endpoint_sim=endpoint_sim,
        )
        hit = _hit_from_row(
            row,
            score=combined,
            struct_score=struct,
            embed_score=max(0.0, embed_cos),
            recall_rank=rank,
            parts=parts,
        )
        ranked_items.append((combined, hit, row))

    ranked_items.sort(key=lambda x: (-x[0], x[1].get("line") or 0))
    hits = diversify_mechanism_hits(ranked_items, max(1, top_k), endpoint_sim)
    scanned = len(rows)
    for h in hits:
        h["scanned_rows"] = scanned
        h["recall_k"] = k_recall
        h["retrieval"] = "embed_recall+struct_rerank"
        h["embed_text_kind"] = EMBED_TEXT_KIND
        h["relation_align"] = rel_mode
    return hits
