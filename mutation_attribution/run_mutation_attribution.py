#!/usr/bin/env python3
"""Hit@1/5/10 and MRR of five attributors on the Go FIM mutation queries.

MRR is the mean reciprocal rank of the parent inside the top-10 list.
A parent outside that list contributes 0.

Train pool is ``csn_go_train_fim_10k.jsonl``. A hit is the parent row:
``ground_truth_task_id`` when both sides have one, otherwise
``ground_truth_line`` (0-based index in that file).

Methods
-------
bm25
    Okapi BM25 on identifier tokens of the whole function.
embed
    Cosine of raw-code embeddings from ``EIF_SEMANTIC_EMBED_MODEL``
    (OpenAI-compatible ``EIF_SEMANTIC_EMBED_BASE_URL``). Train vectors are
    cached. This is not the semantic method's relation rerank.
ast
    Mean of node-type and parent-edge histogram cosines (tree-sitter Go).
tracin
    Gold CE gradient of the query, sketched and cosined against the existing
    CE Q/K last-layer bank. Sent to the running 8766 ``/api/sample-attribution``
    so the checkpoint stays the one 8766 already loaded
    (``EIF_ADAPTER_PATH_CE`` = go-ce/checkpoint-5000).
semantic
    Current 8766 ``/api/llm-semantic-retrieve`` (top 10). Uses
    ``EIF_LLM_SEMANTIC_CORPUS``. Empty ``EIF_LLM_SEMANTIC_EMBEDDINGS`` keeps
    the full-corpus stage-2 path 8766 already uses.

No extra bank is built. TracIn reuses ``EIF_SALIENCY_BANK_PATH_CE``. That
file must be the CE Q/K bank for this checkpoint and this 10k order. Hits
prefer ``task_id``, so a bank indexed by the ids jsonl still counts when the
task id is the same parent.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for _path in (str(ROOT), str(HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

METHODS = ("bm25", "embed", "ast", "tracin", "semantic")
KS = (1, 5, 10)
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _hydrate() -> None:
    try:
        from src.gold_live_attribution import _hydrate_eif_env

        _hydrate_eif_env(force_file=True)
    except Exception as exc:
        print(f"[env] hydrate skipped: {exc}", flush=True)


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _post(url: str, body: dict, timeout: float) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"non-object response from {url}")
    return payload


def _full_code(row: dict) -> str:
    if row.get("prefix") or row.get("middle") or row.get("suffix"):
        return f"{row.get('prefix') or ''}{row.get('middle') or ''}{row.get('suffix') or ''}"
    return f"{row.get('prompt') or ''}\n{row.get('response') or row.get('label') or ''}"


def _tokens(text: str) -> list[str]:
    return [m.group(0).casefold() for m in _IDENT.finditer(text or "") if len(m.group(0)) > 1]


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _truth(row: dict) -> tuple[int, str]:
    return int(row["ground_truth_line"]), str(row.get("ground_truth_task_id") or "")


def _is_parent(line: int, task: str, truth_line: int, truth_task: str) -> bool:
    if truth_task and task and task == truth_task:
        return True
    return line == truth_line


def _parent_rank(ranked: list[tuple[int, str]], truth_line: int, truth_task: str) -> int:
    """1-based rank of the parent inside this list. 0 means it is absent."""
    for index, (line, task) in enumerate(ranked, start=1):
        if _is_parent(line, task, truth_line, truth_task):
            return index
    return 0


def _hits(ranked: list[tuple[int, str]], truth_line: int, truth_task: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for k in KS:
        found = 0
        for line, task in ranked[:k]:
            if _is_parent(line, task, truth_line, truth_task):
                found = 1
                break
        out[f"hit@{k}"] = found
    rank = _parent_rank(ranked, truth_line, truth_task)
    out["rank"] = rank
    out["rr"] = (1.0 / rank) if rank else 0.0
    return out


def _reciprocal(row: dict, method: str) -> float:
    """1/rank of the parent in the stored top-10. Missing parent → 0."""
    hits = (row.get("hits") or {}).get(method) or {}
    if "rr" in hits:
        return float(hits["rr"])
    truth_line = int(row.get("truth_line") or -1)
    truth_task = str(row.get("truth_task") or "")
    ranked = [
        (int(item.get("line")), str(item.get("task_id") or ""))
        for item in (row.get("ranks") or {}).get(method) or []
        if isinstance(item, dict)
    ]
    rank = _parent_rank(ranked, truth_line, truth_task)
    return (1.0 / rank) if rank else 0.0


def _rank_key(item: tuple[float, int, str]) -> tuple[float, int]:
    score, line, _task = item
    return (-score, line)


# ── BM25 ────────────────────────────────────────────────────────────────────


class BM25Index:
    def __init__(self, docs: list[list[str]]):
        self.docs = docs
        self.N = len(docs)
        self.dl = [len(doc) for doc in docs]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 1.0
        self.tf = [Counter(doc) for doc in docs]
        df: Counter[str] = Counter()
        for tf in self.tf:
            df.update(tf.keys())
        self.df = df
        self.k1 = 1.5
        self.b = 0.75

    def query(self, tokens: list[str], top_k: int) -> list[tuple[int, float]]:
        scores = [0.0] * self.N
        if not tokens or self.avgdl <= 0:
            return []
        for term in tokens:
            n = self.df.get(term, 0)
            if n == 0:
                continue
            idf = math.log(1.0 + (self.N - n + 0.5) / (n + 0.5))
            for i, tf in enumerate(self.tf):
                freq = tf.get(term, 0)
                if freq == 0:
                    continue
                denom = freq + self.k1 * (1.0 - self.b + self.b * self.dl[i] / self.avgdl)
                scores[i] += idf * (freq * (self.k1 + 1.0)) / denom
        order = sorted(range(self.N), key=lambda i: (-scores[i], i))[:top_k]
        return [(i, scores[i]) for i in order]


# ── AST ─────────────────────────────────────────────────────────────────────


def _ast_cache_path(train: Path, cache_dir: Path) -> Path:
    stamp = f"{train.stat().st_mtime_ns}_{train.stat().st_size}"
    return cache_dir / f"ast_{stamp}.pkl"


def _build_ast(train_rows: list[dict], cache: Path):
    from build_go_fim_mutations import _parser, ast_hists

    if cache.is_file():
        with cache.open("rb") as handle:
            return pickle.load(handle)
    parser = _parser()
    packed = []
    for i, row in enumerate(train_rows):
        types, edges, _bad = ast_hists(parser, _full_code(row))
        packed.append((types, edges))
        if (i + 1) % 1000 == 0:
            print(f"[ast] indexed {i + 1}/{len(train_rows)}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as handle:
        pickle.dump(packed, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return packed


def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
    from build_go_fim_mutations import cosine

    return cosine(a, b)


def _ast_query(parser, code: str, index, tasks: list[str], top_k: int) -> list[tuple[int, str, float]]:
    from build_go_fim_mutations import ast_hists

    q_types, q_edges, _bad = ast_hists(parser, code)
    scored = []
    for i, (types, edges) in enumerate(index):
        score = 0.5 * (_cosine(q_types, types) + _cosine(q_edges, edges))
        scored.append((score, i, tasks[i]))
    scored.sort(key=_rank_key)
    return [(i, task, score) for score, i, task in scored[:top_k]]


# ── Embedding ───────────────────────────────────────────────────────────────


def _embed_text(row: dict, limit: int) -> str:
    return _full_code(row).strip()[:limit] or " "


def _embed_model() -> str:
    from src.fim_semantic_index import embed_model_id

    model = embed_model_id()
    if not model or model.startswith("local:"):
        raise RuntimeError(
            "Set EIF_SEMANTIC_EMBED_MODEL and EIF_SEMANTIC_EMBED_BASE_URL "
            "for the embedding baseline."
        )
    return model


def _embed_client():
    from src.fim_semantic_index import _build_embed_client

    return _build_embed_client().with_options(max_retries=0, timeout=120.0)


def _embed_one_batch(client, model: str, chunk: list[str]) -> list[list[float]]:
    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

    wait = 2.0
    last: Exception | None = None
    for _attempt in range(10):
        try:
            resp = client.embeddings.create(model=model, input=chunk)
            ordered = sorted(resp.data, key=lambda item: int(item.index))
            return [list(item.embedding) for item in ordered]
        except RateLimitError as exc:
            last = exc
        except APIStatusError as exc:
            last = exc
            if int(exc.status_code) not in (408, 409, 429, 500, 502, 503, 504):
                raise
        except (APIConnectionError, APITimeoutError) as exc:
            last = exc
        print(f"[embed] {type(last).__name__}, sleep {wait:.0f}s", flush=True)
        time.sleep(wait)
        wait = min(60.0, wait * 2)
    raise RuntimeError("embedding request failed after retries") from last


def _cache_model(data) -> str:
    raw = data["model"]
    return str(raw.item() if hasattr(raw, "item") else raw)


def _save_embed_cache(cache: Path, mat, model: str, n_total: int, n_done: int, normalized: bool) -> None:
    import numpy as np

    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".npz.partial")
    with tmp.open("wb") as handle:
        np.savez(
            handle,
            vectors=np.asarray(mat[:n_done], dtype=np.float32),
            model=np.array(model),
            n=np.array(n_total),
            n_done=np.array(n_done),
            normalized=np.array(1 if normalized else 0),
        )
    tmp.replace(cache)


def _embed_matrix(texts: list[str], cache: Path, *, batch: int, pause: float):
    import numpy as np

    model = _embed_model()
    n_total = len(texts)
    mat = None
    start = 0
    if cache.is_file():
        data = np.load(cache)
        if _cache_model(data) == model and int(data["n"]) == n_total:
            vectors = np.asarray(data["vectors"], dtype=np.float32)
            n_done = int(data["n_done"]) if "n_done" in data.files else int(vectors.shape[0])
            normalized = int(data["normalized"]) if "normalized" in data.files else 1
            if n_done >= n_total and normalized:
                print(f"[embed] cache hit {cache}", flush=True)
                return vectors
            if n_done > 0 and not normalized:
                mat = np.zeros((n_total, vectors.shape[1]), dtype=np.float32)
                mat[:n_done] = vectors[:n_done]
                start = n_done
                print(f"[embed] resume {start}/{n_total}", flush=True)
    print(
        f"[embed] requesting {n_total - start}/{n_total} vectors "
        f"model={model} batch={batch} pause={pause}s",
        flush=True,
    )
    client = _embed_client()
    step = max(1, min(int(batch), 16))
    cursor = start
    while cursor < n_total:
        chunk = texts[cursor : cursor + step]
        rows = _embed_one_batch(client, model, chunk)
        block = np.asarray(rows, dtype=np.float32)
        if mat is None:
            mat = np.zeros((n_total, block.shape[1]), dtype=np.float32)
        mat[cursor : cursor + len(chunk)] = block
        cursor += len(chunk)
        if cursor == n_total or cursor % 160 == 0 or (start and cursor - start < step):
            _save_embed_cache(cache, mat, model, n_total, cursor, False)
        print(f"[embed] {cursor}/{n_total}", flush=True)
        if cursor < n_total and pause > 0:
            time.sleep(pause)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / np.clip(norms, 1e-12, None)
    _save_embed_cache(cache, mat, model, n_total, n_total, True)
    return mat


def _embed_query(text: str, matrix, tasks: list[str], top_k: int) -> list[tuple[int, str, float]]:
    import numpy as np

    vec = np.asarray(_embed_one_batch(_embed_client(), _embed_model(), [text])[0], dtype=np.float32)
    vec = vec / max(float(np.linalg.norm(vec)), 1e-12)
    scores = matrix @ vec
    order = np.argsort(-scores)[:top_k]
    return [(int(i), tasks[int(i)], float(scores[int(i)])) for i in order]


# ── 8766 ────────────────────────────────────────────────────────────────────


def _semantic_ranked(url: str, row: dict, top_k: int) -> list[tuple[int, str, float]]:
    prompt = str(row.get("prompt") or "")
    gold = str(row.get("middle") or row.get("response") or "")
    payload = _post(
        f"{url}/api/llm-semantic-retrieve",
        {
            "fimPrompt": prompt,
            "goldCompletion": gold,
            "topK": top_k,
            "runCorpusSearch": True,
            "language": "go",
        },
        timeout=600.0,
    )
    hits: list[dict] = []
    for block in payload.get("search_results") or []:
        if isinstance(block, dict):
            hits.extend(h for h in (block.get("corpus_hits") or []) if isinstance(h, dict))
    hits.sort(key=lambda h: -float(h.get("semantic_score") or 0))
    ranked = []
    for hit in hits[:top_k]:
        try:
            line = int(hit.get("line"))
        except (TypeError, ValueError):
            line = -1
        ranked.append((line, str(hit.get("task_id") or ""), float(hit.get("semantic_score") or 0)))
    return ranked


def _tracin_ranked(url: str, row: dict, tokenizer, top_k: int) -> list[tuple[int, str, float]]:
    from src.raw_eval_report import build_raw_eval_report

    report = build_raw_eval_report(
        {
            "prompt": row.get("prompt") or "",
            "label": row.get("middle") or row.get("response") or "",
            "predict": row.get("middle") or row.get("response") or "",
            "task_id": row.get("mutation_id") or "",
        },
        file_name="raw_ce/go_fim_mutations.jsonl",
        line_no=int(row.get("ground_truth_line") or 0),
        tokenizer=tokenizer,
    )
    payload = _post(
        f"{url}/api/sample-attribution",
        {
            "which": "gradient",
            "topK": top_k,
            "report": report,
            "reportFileName": "raw_ce/go_fim_mutations.jsonl",
        },
        timeout=600.0,
    )
    gradient = payload.get("gradient") or {}
    ranked = []
    for hit in (gradient.get("trains") or [])[:top_k]:
        ranked.append((
            int(hit.get("trainSampleId")),
            str(hit.get("taskId") or ""),
            float(hit.get("cos") or 0),
        ))
    return ranked


def _load_tokenizer():
    from src.raw_eval_report import _load_tokenizer as load

    return load()


# ── Driver ──────────────────────────────────────────────────────────────────


def _default_mutations() -> Path:
    server = Path("/mnt/md124/jiaxin/training_code/data/go_fim_mutations.jsonl")
    local = Path(__file__).resolve().parent / "go_fim_mutations.jsonl"
    if server.is_file():
        return server
    return local


def _default_train() -> Path:
    raw = _env("EIF_LLM_TRAIN_CORPUS")
    if raw:
        return Path(raw)
    return Path("/mnt/md124/jiaxin/training_code/data/csn_go_train_fim_10k.jsonl")


def _load_done(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    done = {}
    for row in _load_jsonl(path):
        mid = str(row.get("mutation_id") or "")
        if mid:
            done[mid] = row
    return done


def _summarize(rows: list[dict], methods: list[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(rows), "methods": {}}
    for method in methods:
        scored = [row for row in rows if method in (row.get("ranks") or {})]
        bucket = {"n": len(scored)}
        for k in KS:
            key = f"hit@{k}"
            hits = sum(int((row.get("hits") or {}).get(method, {}).get(key) or 0) for row in scored)
            bucket[key] = hits
            bucket[f"{key}_rate"] = (hits / len(scored)) if scored else 0.0
        bucket["mrr"] = (
            sum(_reciprocal(row, method) for row in scored) / len(scored)
        ) if scored else 0.0
        summary["methods"][method] = bucket
    return summary


def _quiet_embed_logs() -> None:
    import logging

    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> int:
    _hydrate()
    _quiet_embed_logs()
    parser = argparse.ArgumentParser(description="Mutation attribution Hit@1/5/10 and MRR")
    parser.add_argument("--mutations", type=Path, default=_default_mutations())
    parser.add_argument("--train", type=Path, default=_default_train())
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--methods", default=",".join(METHODS))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--eif-url", default=_env("EIF_URL", "http://127.0.0.1:8766"))
    parser.add_argument("--embed-chars", type=int, default=4000)
    parser.add_argument("--embed-batch", type=int, default=8)
    parser.add_argument("--embed-pause", type=float, default=1.0, help="seconds between embedding batches")
    parser.add_argument("--limit", type=int, default=0, help="debug: first N queries only")
    args = parser.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        print(f"unknown methods: {unknown}", file=sys.stderr)
        return 2
    if not args.mutations.is_file():
        print(f"mutations not found: {args.mutations}", file=sys.stderr)
        return 2
    if not args.train.is_file():
        print(f"train not found: {args.train}", file=sys.stderr)
        return 2

    queries = _load_jsonl(args.mutations)
    if args.limit > 0:
        queries = queries[: args.limit]
    train_rows = _load_jsonl(args.train)
    tasks = [str(row.get("task_id") or "") for row in train_rows]
    print(
        f"[run] queries={len(queries)} train={len(train_rows)} methods={methods} "
        f"eif={args.eif_url}",
        flush=True,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    result_path = args.out / "mutation_hits.jsonl"
    done = _load_done(result_path)

    bm25 = None
    ast_index = None
    ast_parser = None
    embed_mat = None
    tokenizer = None
    top_k = max(KS)

    if "bm25" in methods:
        print("[bm25] indexing", flush=True)
        bm25 = BM25Index([_tokens(_full_code(row)) for row in train_rows])
    if "ast" in methods:
        print("[ast] indexing", flush=True)
        ast_index = _build_ast(train_rows, _ast_cache_path(args.train, args.out))
        from build_go_fim_mutations import _parser

        ast_parser = _parser()
    if "embed" in methods:
        texts = [_embed_text(row, args.embed_chars) for row in train_rows]
        cache = args.out / "train_code_embeddings.npz"
        embed_mat = _embed_matrix(
            texts, cache, batch=args.embed_batch, pause=args.embed_pause
        )
    if "tracin" in methods:
        tokenizer = _load_tokenizer()

    for n, query in enumerate(queries, start=1):
        mid = str(query.get("mutation_id") or f"row{n}")
        rec = done.get(mid) or {
            "mutation_id": mid,
            "truth_line": int(query["ground_truth_line"]),
            "truth_task": str(query.get("ground_truth_task_id") or ""),
            "ranks": {},
            "hits": {},
        }
        truth_line, truth_task = _truth(query)
        pending = [m for m in methods if m not in rec["ranks"]]
        if not pending:
            print(f"[{n}/{len(queries)}] {mid} cached", flush=True)
            continue
        code = _full_code(query)
        print(f"[{n}/{len(queries)}] {mid} {pending}", flush=True)
        for method in pending:
            try:
                if method == "bm25":
                    ranked = [
                        (i, tasks[i], score)
                        for i, score in bm25.query(_tokens(code), top_k)
                    ]
                elif method == "embed":
                    ranked = _embed_query(_embed_text(query, args.embed_chars), embed_mat, tasks, top_k)
                elif method == "ast":
                    ranked = _ast_query(ast_parser, code, ast_index, tasks, top_k)
                elif method == "tracin":
                    ranked = _tracin_ranked(args.eif_url.rstrip("/"), query, tokenizer, top_k)
                else:
                    ranked = _semantic_ranked(args.eif_url.rstrip("/"), query, top_k)
            except (urllib.error.URLError, TimeoutError, RuntimeError, ValueError, OSError) as exc:
                rec.setdefault("errors", {})[method] = str(exc)
                print(f"  {method} failed: {exc}", flush=True)
                continue
            rec["ranks"][method] = [
                {"line": line, "task_id": task, "score": round(float(score), 6)}
                for line, task, score in ranked
            ]
            rec["hits"][method] = _hits([(line, task) for line, task, _s in ranked], truth_line, truth_task)
            print(f"  {method} {rec['hits'][method]}", flush=True)
        done[mid] = rec
        result_path.write_text(
            "".join(json.dumps(done[str(q.get("mutation_id"))], ensure_ascii=False) + "\n"
                    for q in queries if str(q.get("mutation_id")) in done),
            encoding="utf-8",
        )

    ordered = [done[str(q.get("mutation_id"))] for q in queries if str(q.get("mutation_id")) in done]
    summary = _summarize(ordered, methods)
    summary_path = args.out / "mutation_hit_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{'method':<12}{'n':<6}{'Hit@1':<10}{'Hit@5':<10}{'Hit@10':<10}{'MRR':<10}")
    for method in methods:
        bucket = summary["methods"][method]
        print(
            f"{method:<12}{bucket['n']:<6}"
            f"{bucket['hit@1_rate']:.3f}     {bucket['hit@5_rate']:.3f}     "
            f"{bucket['hit@10_rate']:.3f}     {bucket['mrr']:.3f}"
        )
    print(f"wrote {result_path}")
    print(f"wrote {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
