#!/usr/bin/env python3
"""Precompute structured semantic representations for a raw FIM train corpus.

Boolean retrieval still uses the **raw** FIM jsonl::

    EIF_LLM_TRAIN_CORPUS=/mnt/md124/jiaxin/training_code/data/csn_go_train_fim.jsonl

Semantic retrieval uses this script's output (schema JSON, optional embeddings)::

    python -m src.fim_semantic_preprocess \\
      -i /mnt/md124/jiaxin/training_code/data/csn_go_train_fim.jsonl \\
      -o /mnt/md124/jiaxin/training_code/data/csn_go_train_fim.semantic.jsonl \\
      --language go

    # Stage-1 embeddings (mechanism-weighted text). Re-run after more LLM rows.
    python -m src.fim_semantic_preprocess \\
      --embed-only \\
      -o /mnt/md124/jiaxin/training_code/data/csn_go_train_fim.semantic.jsonl \\
      --embed-model text-embedding-v3

Then in eif_api.env::

    EIF_LLM_SEMANTIC_CORPUS=/mnt/md124/jiaxin/training_code/data/csn_go_train_fim.semantic.jsonl

Do **not** point EIF_LLM_TRAIN_CORPUS at the preprocessed file. Boolean substring
search and MID rewrite need original prompt/response. The semantic jsonl only
stores schema + previews + source_line (index into the raw corpus).

Each output row::

    {
      "source_line": 123,
      "task_id": "...",
      "language": "go",
      "role": "propagate a credential into request metadata before request signing",
      "domain": [...],
      "pattern": [...],
      "entities": [...],
      "operations": [...],
      "conditions": [...],
      "relations": [{"source": "...", "target": "...", "type": "dataflow"}],
      "summary": "...",
      "semantic_flat_text": "Role: ...\\nDomain: ...",
      "prompt_preview": "...",
      "response_preview": "..."
    }
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _hydrate_env() -> None:
    try:
        from src.gold_live_attribution import _hydrate_eif_env
        _hydrate_eif_env(force_file=True)
    except Exception:
        env = REPO_ROOT / "eif_api.env"
        if not env.is_file():
            return
        for line in env.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, _, v = s.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


def _row_prompt(row: dict[str, Any]) -> str:
    for key in ("prompt", "input", "query"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _row_gold(row: dict[str, Any]) -> str:
    for key in ("response", "label", "output", "gold"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _row_task_id(row: dict[str, Any], source_line: int) -> str:
    for key in ("task_id", "id", "uid", "path"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return f"line_{source_line}"


def _already_done(out_path: Path) -> set[int]:
    done: set[int] = set()
    if not out_path.is_file():
        return done
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            try:
                done.add(int(row.get("source_line")))
            except (TypeError, ValueError):
                continue
    return done


def _write_row(out_path: Path, row: dict[str, Any]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _embed_only(out_path: Path, *, model: str, batch: int) -> int:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("pip install numpy") from exc

    from src.fim_semantic_index import embed_texts, embedding_meta_path, embedding_npz_path
    from src.fim_semantic_schema import EMBED_TEXT_KIND, flatten_semantic_text_for_embedding

    rows: list[dict[str, Any]] = []
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    texts = [flatten_semantic_text_for_embedding(r) for r in rows]
    vectors: list[list[float]] = []
    source_lines: list[int] = []
    for start in range(0, len(rows), max(1, batch)):
        chunk_rows = rows[start : start + batch]
        chunk_texts = texts[start : start + batch]
        vecs = embed_texts(chunk_texts, model=model)
        vectors.extend(vecs)
        for r in chunk_rows:
            try:
                source_lines.append(int(r.get("source_line")))
            except (TypeError, ValueError):
                source_lines.append(-1)
        print(f"  embedded {min(start + batch, len(rows))}/{len(rows)}", flush=True)

    arr = np.asarray(vectors, dtype="float32")
    npz_path = embedding_npz_path(out_path)
    np.savez_compressed(
        npz_path,
        vectors=arr,
        source_lines=np.asarray(source_lines, dtype="int64"),
    )
    meta_path = embedding_meta_path(out_path)
    meta_path.write_text(
        json.dumps(
            {
                "model": model,
                "rows": len(rows),
                "dim": int(arr.shape[1]) if arr.ndim == 2 else 0,
                "npz": str(npz_path),
                "semantic_jsonl": str(out_path),
                "text": EMBED_TEXT_KIND,
                "pipeline": "embed_recall+struct_rerank",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"wrote {npz_path} shape={arr.shape} text={EMBED_TEXT_KIND}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-i", "--input",
        default=os.environ.get("EIF_LLM_TRAIN_CORPUS") or "",
        help="raw FIM train jsonl (same file as EIF_LLM_TRAIN_CORPUS)",
    )
    parser.add_argument(
        "-o", "--output",
        default=os.environ.get("EIF_LLM_SEMANTIC_CORPUS") or "",
        help="output semantic jsonl (set EIF_LLM_SEMANTIC_CORPUS to this path)",
    )
    parser.add_argument("--language", default="go")
    parser.add_argument("--max-rows", type=int, default=0, help="0 = all remaining")
    parser.add_argument("--start-line", type=int, default=0)
    parser.add_argument(
        "--only-line",
        type=int,
        default=None,
        help="0-based line in the raw train jsonl; process only that sample",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="print structured JSON to stdout; do not append to the semantic corpus",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="reprocess even if source_line already exists in the output jsonl",
    )
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--embed-only", action="store_true",
                        help="embed an existing semantic jsonl; do not call the generative LLM")
    parser.add_argument("--embed", action="store_true",
                        help="after LLM pass, also write embeddings npz")
    parser.add_argument(
        "--embed-model",
        default=os.environ.get("EIF_SEMANTIC_EMBED_MODEL") or os.environ.get("EMBED_MODEL") or "",
        help="OpenAI-compatible embedding model id",
    )
    parser.add_argument("--embed-batch", type=int, default=16)
    args = parser.parse_args(argv)

    _hydrate_env()
    print_only = bool(args.print_only)
    out_raw = str(args.output or "").strip()
    out_path = Path(out_raw).expanduser() if out_raw else Path()
    if not print_only and not out_raw:
        print("need -o / EIF_LLM_SEMANTIC_CORPUS (or --print-only)", file=sys.stderr)
        return 2

    if args.embed_only:
        model = (args.embed_model or "").strip()
        if not model:
            print("need --embed-model / EIF_SEMANTIC_EMBED_MODEL", file=sys.stderr)
            return 2
        if not out_path.is_file():
            print(f"semantic jsonl not found: {out_path}", file=sys.stderr)
            return 2
        return _embed_only(out_path, model=model, batch=args.embed_batch)

    in_path = Path(args.input).expanduser()
    if not in_path.is_file():
        print(f"raw corpus not found: {in_path}", file=sys.stderr)
        print("EIF_LLM_TRAIN_CORPUS must stay the original FIM jsonl.", file=sys.stderr)
        return 2

    from src.fim_semantic_schema import flatten_semantic_text, normalize_semantic_repr
    from src.llm_semantic_retrieval import call_llm_semantic_analyze

    done = set() if args.force or print_only or not out_raw else _already_done(out_path)
    print(
        f"input={in_path}\noutput={'stdout' if print_only else out_path}\n"
        f"resume_skip={len(done)} language={args.language}"
        + (f" only_line={args.only_line}" if args.only_line is not None else ""),
        flush=True,
    )

    processed = 0
    failed = 0
    with in_path.open(encoding="utf-8") as fh:
        for source_line, line in enumerate(fh):
            if args.only_line is not None and source_line != args.only_line:
                if source_line > args.only_line:
                    break
                continue
            if source_line < args.start_line:
                continue
            if source_line in done:
                continue
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError:
                failed += 1
                continue
            if not isinstance(row, dict):
                continue
            prompt = _row_prompt(row)
            gold = _row_gold(row)
            if not prompt.strip() or not gold.strip():
                failed += 1
                print(f"  skip line={source_line}: missing prompt/gold", flush=True)
                continue
            t0 = time.perf_counter()
            try:
                llm_out = call_llm_semantic_analyze(
                    fim_prompt=prompt,
                    gold_completion=gold,
                    language=args.language,
                )
            except Exception as exc:
                failed += 1
                print(f"  fail line={source_line}: {exc}", flush=True)
                continue
            analysis = llm_out.get("semantic") if isinstance(llm_out.get("semantic"), dict) else {}
            sem = normalize_semantic_repr(analysis if analysis else llm_out)
            out_row = {
                "source_line": source_line,
                "task_id": _row_task_id(row, source_line),
                "language": args.language,
                **sem,
                "semantic_flat_text": flatten_semantic_text(sem),
                "prompt_preview": prompt[:280],
                "response_preview": gold[:200],
                "model": llm_out.get("model"),
            }
            if print_only:
                print(json.dumps(out_row, ensure_ascii=False, indent=2), flush=True)
            else:
                _write_row(out_path, out_row)
            processed += 1
            dt = time.perf_counter() - t0
            summary = (sem.get("summary") or "")[:120]
            print(
                f"  line={source_line} +{dt:.1f}s pattern={sem.get('pattern')!r} {summary}",
                flush=True,
            )
            if args.sleep > 0:
                time.sleep(args.sleep)
            if args.max_rows and processed >= args.max_rows:
                break

    print(f"done processed={processed} failed={failed} output={'stdout' if print_only else out_path}", flush=True)
    if print_only:
        return 0
    if args.embed:
        model = (args.embed_model or "").strip()
        if not model:
            print("skip embed: set --embed-model / EIF_SEMANTIC_EMBED_MODEL", flush=True)
            return 0
        return _embed_only(out_path, model=model, batch=args.embed_batch)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
