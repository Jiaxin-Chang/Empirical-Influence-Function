#!/usr/bin/env python3
"""LLM expression attribution → tight-to-loose corpus scan → MID → tree-sitter.

For each test row (``prompt`` + gold in ``label``/``response``; ignore ``predict``):

  1. Call LLM to summarize gold pattern and emit 2–5 boolean search expressions
     (LLM emits wide→narrow; this script **reverses** to tight→loose by default)
  2. For each expression, scan the train corpus; prefer gold-region hits
  3. Rewrite hits (keep ``<FIM>``): completion-hit → annotate as-is; context-hit →
     fill old hole, move ``<FIM>`` onto the matched span
  4. GraphSignal annotate with **tree-sitter only** (fill hole, parse full context)
  5. Write ``input_ids`` / ``label`` / ``attention_edges`` (× copies)

Needs: OpenAI-compatible LLM (vLLM / DashScope via ``eif_api.env``), tokenizer,
``tree-sitter`` / ``tree-sitter-cpp``. Train rows should contain ``<PRE>/<SUF>/<MID>``
for MID rewrite + annotate; non-FIM hits are skipped automatically.

Example::

    python -m src.llm_expr_treesitter \\
      -i /path/to/qwen3-8b-cpp_predictions.jsonl \\
      --corpus /mnt/md124/jiaxin/Empirical-Influence-Function/cpp_train_fixed.jsonl \\
      -o /path/to/cpp_llm_expr_ts.jsonl \\
      --tokenizer /mnt/md124/jiaxin/models/Qwen3-8B \\
      --language cpp --copies 10 --tight-first
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TEST = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/test/cpp_test_fixed.jsonl"
)
DEFAULT_CORPUS = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/cpp_train_fixed.jsonl"
)
DEFAULT_OUTPUT = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/cpp_llm_expr_treesitter.jsonl"
)

_FENCE_RE = re.compile(
    r"^\s*```[^\n]*\n([\s\S]*?)\n```\s*$",
    re.MULTILINE,
)

_MID_OK_WITHOUT_REWRITE = frozenset({
    "train_mid_already_is_gold",
    "same_as_original_mid",
    "keep_original_gold_response",
})


def _hydrate_env() -> None:
    try:
        from src.gold_live_attribution import _hydrate_eif_env
        _hydrate_eif_env(force_file=True)
    except Exception:
        env = REPO_ROOT / "eif_api.env"
        if not env.is_file():
            return
        import os
        for line in env.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, _, v = s.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k:
                os.environ[k] = v


def _strip_markdown_fence(text: str) -> str:
    t = (text or "").strip()
    m = _FENCE_RE.match(t)
    if m:
        return m.group(1).strip()
    if t.startswith("```"):
        nl = t.find("\n")
        if nl > 0:
            body = t[nl + 1 :]
            end = body.rfind("```")
            if end >= 0:
                return body[:end].strip()
    return t


def _clean_gold(raw: str) -> str:
    from src.llm_train_retrieval import clean_gold_mid_completion

    text = clean_gold_mid_completion(raw) or (raw or "").strip()
    return _strip_markdown_fence(text)


def _load_raw_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    from src.fast_gold_to_continue import _load_raw_rows as _load

    return _load(path)


def _test_fim_gold(row: dict[str, Any]) -> tuple[str, str]:
    prompt = str(row.get("prompt") or row.get("input") or "").strip()
    if isinstance(row.get("label"), list):
        gold = str(
            row.get("response") or row.get("gold") or row.get("target") or ""
        ).strip()
    else:
        gold = str(
            row.get("label")
            or row.get("response")
            or row.get("gold")
            or row.get("target")
            or ""
        ).strip()
    return prompt, gold


def _expr_items(analysis: dict[str, Any], *, tight_first: bool) -> list[dict[str, Any]]:
    exprs = analysis.get("corpus_search_expressions") or []
    if not isinstance(exprs, list):
        return []
    cleaned: list[dict[str, Any]] = []
    for i, item in enumerate(exprs):
        if not isinstance(item, dict):
            continue
        expression = str(item.get("expression") or "").strip()
        if not expression:
            continue
        cleaned.append({
            "name": str(item.get("name") or f"expr_{i}"),
            "expression": expression,
            "why": str(item.get("why") or ""),
            "llm_index": i,
        })
    # LLM emits wide→narrow; reverse for tight→loose.
    if tight_first:
        cleaned.reverse()
    return cleaned


def _rank_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {"gold": 0, "cross": 1, "context": 2}

    def key(h: dict[str, Any]) -> tuple[int, int]:
        region = str(h.get("match_region") or "").lower()
        raw = h.get("line")
        line = int(raw) if raw is not None else 10**12
        return (order.get(region, 9), line)

    return sorted(hits, key=key)


def _load_corpus_row(path: Path, line_idx: int) -> dict[str, Any] | None:
    with path.open(encoding="utf-8") as fh:
        for idx, line in enumerate(fh):
            if idx != line_idx:
                continue
            raw = line.strip()
            if not raw:
                return None
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
    return None


def _row_prompt_response(row: dict[str, Any]) -> tuple[str, str]:
    from src.fast_gold_to_continue import _row_prompt_response as _pr

    return _pr(row)


def _rewrite_train(
    prompt: str,
    response: str,
    *,
    test_gold: str,
    expression: str,
) -> tuple[str, str, dict[str, Any]]:
    from src.fim_mid_rewrite import rewrite_fim_mid

    out = rewrite_fim_mid(
        prompt,
        response,
        test_gold=test_gold,
        expression=expression or "",
    )
    mode = str(out.get("mode") or "unchanged")
    reason = str(out.get("reason") or "")
    # <FIM> path: keep format (keep hole or relocate onto context dig).
    if mode in ("angle_fim_keep", "relocate_angle_fim"):
        return str(out["prompt"]), str(out["response"]), out
    if mode != "unchanged" and reason == "ok":
        return str(out["prompt"]), str(out["response"]), out
    if reason in _MID_OK_WITHOUT_REWRITE or reason in (
        "train_mid_already_is_gold",
        "same_as_original_mid",
    ):
        return prompt, response, out
    raise RuntimeError(f"MID rewrite failed: reason={reason} detail={out.get('detail')}")


def _load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )


def _annotate_treesitter(
    *,
    tokenizer: Any,
    prompt: str,
    response: str,
    language: str,
    max_len: int,
    max_teacher_edges: int,
) -> dict[str, Any]:
    sys.path.insert(0, str(REPO_ROOT / "tools" / "annotation-viewer"))
    from server.graphsignal_annotate import annotate_corpus_row  # type: ignore

    return annotate_corpus_row(
        {"prompt": prompt, "response": response, "language": language},
        tokenizer,
        max_len=max_len,
        max_teacher_edges=max_teacher_edges,
        use_llm=False,
        default_language=language,
    )


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _edges_from_annotated(annotated: dict[str, Any]) -> list[Any]:
    edges = annotated.get("attention_edges")
    return edges if isinstance(edges, list) else []


def _build_output_rows(
    *,
    annotated: dict[str, Any],
    task_id: str,
    corpus_line: int,
    corpus_path: str,
    test_line: int,
    test_task_id: str,
    match_region: str,
    expression: str,
    expr_name: str,
    rewrite_meta: dict[str, Any],
    copies: int,
    language: str,
    ids_only: bool,
) -> list[dict[str, Any]]:
    input_ids = annotated["input_ids"]
    labels = annotated["label"]
    edges = _edges_from_annotated(annotated)
    dig_hash = str(rewrite_meta.get("dig_hash") or "")[:10]
    base_uid = f"corpus:{task_id}"
    if dig_hash:
        base_uid = f"{base_uid}:midrw:{dig_hash}"
    meta = {
        "corpus": True,
        "llm_expr": True,
        "treesitter_only": True,
        "use_llm_annotate": False,
        "test_line": int(test_line),
        "test_task_id": test_task_id,
        "match_region": match_region,
        "expression": expression,
        "expr_name": expr_name,
        "mid_rewrite": bool(
            rewrite_meta.get("mode") and rewrite_meta.get("mode") != "unchanged"
        ),
        "mid_rewrite_mode": rewrite_meta.get("mode"),
        "mid_rewrite_reason": rewrite_meta.get("reason"),
        "mid_rewrite_locus": rewrite_meta.get("dig_locus"),
        "mid_rewrite_hash": rewrite_meta.get("dig_hash"),
        "n_attention_edges": len(edges),
        "graphsignal": annotated.get("_graphsignal_meta") or {},
    }
    rows: list[dict[str, Any]] = []
    for i in range(max(1, copies)):
        if ids_only:
            rows.append({
                "input_ids": input_ids,
                "label": labels,
                "attention_edges": edges,
            })
            continue
        uid = base_uid if i == 0 else f"{base_uid}::dup{uuid.uuid4().hex[:10]}"
        row: dict[str, Any] = {
            "input_ids": input_ids,
            "label": labels,
            "attention_edges": edges,
            "uid": uid,
            "task_id": task_id,
            "raw_id": uid,
            "language": language,
            "source_corpus_line": int(corpus_line),
            "source_corpus_path": corpus_path,
            "annotation_meta": dict(meta),
        }
        if i > 0:
            row["duplicate_of"] = base_uid
        rows.append(row)
    return rows


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_file(raw: str, *, label: str) -> Path | None:
    text = (raw or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if path.is_file():
        return path.resolve()
    alt = REPO_ROOT / text
    if alt.is_file():
        return alt.resolve()
    print(f"{label} not found: {text!r}", file=sys.stderr)
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "LLM expr (tight→loose) → MID rewrite → tree-sitter-only annotate"
        ),
    )
    p.add_argument("-i", "--input", default=DEFAULT_TEST)
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    p.add_argument("--corpus", default=DEFAULT_CORPUS)
    p.add_argument("--copies", "-n", type=int, default=1)
    p.add_argument("--start-line", type=int, default=1)
    p.add_argument("--end-line", type=int, default=0)
    p.add_argument("--max-tests", type=int, default=0)
    p.add_argument("--max-scan", type=int, default=0, help="0 = full corpus per expr")
    p.add_argument("--top-k", type=int, default=20, help="hits kept per expression")
    p.add_argument("--state", default="")
    p.add_argument(
        "--skip-processed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--tight-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scan expressions tight→loose (default true; LLM emits wide→narrow)",
    )
    p.add_argument("--tokenizer", default="")
    p.add_argument("--language", default="cpp")
    p.add_argument("--model", default="", help="LLM retrieve model override")
    p.add_argument("--max-len", type=int, default=8192)
    p.add_argument("--max-teacher-edges", type=int, default=64)
    p.add_argument(
        "--ids-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--require-edges",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="skip hits that annotate to 0 structural edges",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import os

    args = parse_args(argv)
    _hydrate_env()

    input_path = _resolve_file(args.input, label="input")
    if input_path is None:
        return 2

    corpus_raw = (args.corpus or os.environ.get("EIF_LLM_TRAIN_CORPUS") or "").strip()
    corpus_path = _resolve_file(corpus_raw, label="corpus")
    if corpus_path is None:
        return 2

    out_raw = (args.output or "").strip()
    if not out_raw:
        print("set -o/--output", file=sys.stderr)
        return 2
    out_path = Path(out_raw).expanduser()
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()

    tok_path = (
        (args.tokenizer or "").strip()
        or (os.environ.get("EIF_BASE_MODEL_PATH") or "").strip()
    )
    if not tok_path or not Path(tok_path).is_dir():
        print(f"tokenizer/base model not found: {tok_path!r}", file=sys.stderr)
        return 2

    state_path = (
        Path(args.state).expanduser()
        if args.state
        else input_path.with_suffix(input_path.suffix + ".llm_expr_ts_state.json")
    )
    state: dict[str, Any] = {
        "ok_tests": [],
        "skipped": [],
        "processed_test_lines": [],
        "used_corpus_lines": [],
        "n_continue_rows": 0,
    }
    if state_path.is_file():
        try:
            state.update(json.loads(state_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass

    processed = set(int(x) for x in (state.get("processed_test_lines") or []))
    used_corpus = set(int(x) for x in (state.get("used_corpus_lines") or []))
    ok_tests = list(state.get("ok_tests") or [])

    from src.llm_train_retrieval import call_llm_train_retrieve, search_corpus_jsonl

    language = str(args.language or "cpp")
    print(f"[llm-expr-ts] input={input_path}", flush=True)
    print(f"[llm-expr-ts] corpus={corpus_path}", flush=True)
    print(f"[llm-expr-ts] output={out_path}", flush=True)
    print(
        f"[llm-expr-ts] tokenizer={tok_path} language={language} "
        f"tight_first={args.tight_first} copies={args.copies} use_llm_annotate=False",
        flush=True,
    )
    print("[llm-expr-ts] loading tokenizer…", flush=True)
    tokenizer = _load_tokenizer(tok_path)
    print("[llm-expr-ts] tokenizer ready", flush=True)

    rows = _load_raw_rows(input_path)
    end_line = int(args.end_line) or 10**12
    max_tests = int(args.max_tests) or 10**12
    max_scan = int(args.max_scan) if int(args.max_scan) > 0 else None
    top_k = max(1, int(args.top_k))
    copies = max(1, int(args.copies))
    ids_only = bool(args.ids_only)
    max_len = max(512, int(args.max_len))
    max_edges = max(1, int(args.max_teacher_edges))
    model = (args.model or "").strip() or None

    for test_line, row in rows:
        if test_line < int(args.start_line) or test_line > end_line:
            continue
        if len(ok_tests) >= max_tests:
            break
        if args.skip_processed and test_line in processed:
            print(f"[skip] test line={test_line} already processed", flush=True)
            continue

        task_id = str(row.get("task_id") or f"row_{test_line}")
        fim, gold_raw = _test_fim_gold(row)
        gold = _clean_gold(gold_raw)
        if not fim.strip() or not gold.strip():
            print(f"[test {test_line}] skip: empty prompt/gold", flush=True)
            processed.add(test_line)
            state["processed_test_lines"] = sorted(processed)
            state["skipped"] = (state.get("skipped") or [])[-200:] + [{
                "test_line": test_line, "reason": "empty_prompt_or_gold",
            }]
            _save_state(state_path, state)
            continue

        print(
            f"\n=== test line={test_line} task={task_id} "
            f"gold_chars={len(gold)} ok={len(ok_tests)} ===",
            flush=True,
        )
        t0 = time.perf_counter()
        try:
            retrieve = call_llm_train_retrieve(
                fim_prompt=fim,
                gold_completion=gold,
                model=model,
                language=language,
            )
        except Exception as exc:
            print(f"  LLM retrieve fail: {exc}", flush=True)
            processed.add(test_line)
            state["processed_test_lines"] = sorted(processed)
            state["skipped"] = (state.get("skipped") or [])[-200:] + [{
                "test_line": test_line,
                "reason": "retrieve_failed",
                "error": str(exc)[:300],
            }]
            _save_state(state_path, state)
            continue

        analysis = retrieve.get("analysis") or {}
        exprs = _expr_items(analysis, tight_first=bool(args.tight_first))
        summary = str(analysis.get("gold_pattern_summary") or "")[:160]
        if summary:
            print(f"  pattern: {summary}", flush=True)
        if not exprs:
            print("  no expressions from LLM", flush=True)
            processed.add(test_line)
            state["processed_test_lines"] = sorted(processed)
            state["skipped"] = (state.get("skipped") or [])[-200:] + [{
                "test_line": test_line, "reason": "no_expressions",
            }]
            _save_state(state_path, state)
            continue

        print(
            f"  exprs={len(exprs)} order="
            + ("tight→loose" if args.tight_first else "wide→narrow"),
            flush=True,
        )
        for ei, expr in enumerate(exprs):
            print(
                f"  [{ei + 1}/{len(exprs)}] {expr['name']}: "
                f"{expr['expression'][:140]}"
                + ("…" if len(expr["expression"]) > 140 else ""),
                flush=True,
            )

        chosen: tuple[
            int, dict[str, Any], str, dict[str, Any], dict[str, Any], str, str
        ] | None = None

        for expr in exprs:
            expression = expr["expression"]
            expr_name = str(expr["name"])
            try:
                hits = search_corpus_jsonl(
                    str(corpus_path),
                    expression,
                    top_k=top_k,
                    max_scan=max_scan,
                )
            except Exception as exc:
                print(f"    search fail ({expr_name}): {exc}", flush=True)
                continue
            hits = [
                h for h in _rank_hits(hits)
                if int(h.get("line", -1)) not in used_corpus
            ]
            print(f"    hits={len(hits)} ({expr_name})", flush=True)
            for hit in hits:
                cline = int(hit["line"])
                crow = _load_corpus_row(corpus_path, cline)
                if not crow:
                    continue
                prompt, response = _row_prompt_response(crow)
                region = str(hit.get("match_region") or "")
                print(
                    f"    try line={cline} region={region} "
                    f"task={crow.get('task_id')}",
                    flush=True,
                )
                try:
                    new_prompt, new_response, rw = _rewrite_train(
                        prompt,
                        response,
                        test_gold=gold,
                        expression=expression,
                    )
                except Exception as exc:
                    print(f"      MID fail → next: {exc}", flush=True)
                    continue
                print(
                    f"      rewrite ok reason={rw.get('reason')} "
                    f"mode={rw.get('mode')} locus={rw.get('dig_locus')}",
                    flush=True,
                )
                try:
                    annotated = _annotate_treesitter(
                        tokenizer=tokenizer,
                        prompt=new_prompt,
                        response=new_response,
                        language=language,
                        max_len=max_len,
                        max_teacher_edges=max_edges,
                    )
                except Exception as exc:
                    print(f"      annotate fail → next: {exc}", flush=True)
                    continue
                n_edges = len(_edges_from_annotated(annotated))
                if args.require_edges and n_edges <= 0:
                    print("      0 edges → next", flush=True)
                    continue
                chosen = (
                    cline, crow, region, annotated, rw, expression, expr_name,
                )
                break
            if chosen is not None:
                break

        search_ms = (time.perf_counter() - t0) * 1000
        if chosen is None:
            print(f"  no usable hit ({search_ms:.0f}ms)", flush=True)
            processed.add(test_line)
            state["processed_test_lines"] = sorted(processed)
            state["skipped"] = (state.get("skipped") or [])[-200:] + [{
                "test_line": test_line,
                "reason": "no_usable_hit",
                "n_exprs": len(exprs),
            }]
            _save_state(state_path, state)
            continue

        cline, crow, region, annotated, rw, expression, expr_name = chosen
        n_edges = len(_edges_from_annotated(annotated))
        train_task = str(crow.get("task_id") or f"line_{cline}")
        cont_rows = _build_output_rows(
            annotated=annotated,
            task_id=train_task,
            corpus_line=cline,
            corpus_path=str(corpus_path),
            test_line=test_line,
            test_task_id=task_id,
            match_region=region,
            expression=expression,
            expr_name=expr_name,
            rewrite_meta=rw,
            copies=copies,
            language=language,
            ids_only=ids_only,
        )
        _append_jsonl(out_path, cont_rows)

        used_corpus.add(cline)
        processed.add(test_line)
        rec = {
            "test_line": test_line,
            "task_id": task_id,
            "corpus_line": cline,
            "match_region": region,
            "expr_name": expr_name,
            "expression": expression,
            "continue_rows": len(cont_rows),
            "n_tokens": len(annotated.get("input_ids") or []),
            "n_edges": n_edges,
            "rewrite_reason": rw.get("reason"),
            "rewrite_mode": rw.get("mode"),
        }
        ok_tests.append(rec)
        state["ok_tests"] = ok_tests
        state["processed_test_lines"] = sorted(processed)
        state["used_corpus_lines"] = sorted(used_corpus)
        state["n_continue_rows"] = int(state.get("n_continue_rows") or 0) + len(cont_rows)
        _save_state(state_path, state)
        print(
            f"  wrote {len(cont_rows)} rows edges={n_edges} expr={expr_name} "
            f"-> {out_path} (total≈{state['n_continue_rows']}, {search_ms:.0f}ms)",
            flush=True,
        )

    print(
        f"\n=== done ok_tests={len(ok_tests)} "
        f"rows≈{state.get('n_continue_rows')} "
        f"out={out_path} state={state_path} ===",
        flush=True,
    )
    return 0 if ok_tests else 1


if __name__ == "__main__":
    raise SystemExit(main())
