#!/usr/bin/env python3
"""Go gold-only retrieval → classic PRE/SUF/MID rewrite → tree-sitter annotate.

Simplified Go pipeline (no LLM expr, no LLM annotate):

  1. Take gold from test ``label`` (fallback: response/gold)
  2. Substring-scan train corpus (prompt+response) for that gold
  3. ``rewrite_fim_mid`` (Go logic):
       - hit in train MID/response → keep sample
       - hit in FIM prefix/suffix → relocate ``<MID>``
       - hit in before → promote enclosing func to new FIM problem;
         dig becomes MID; above/below dig = PRE/SUF; old before tail + old
         FIM body move to after
  4. GraphSignal tree-sitter only → ChatML ``input_ids`` + ``label`` +
     ``attention_edges``

Example::

    python -m src.go_gold_treesitter \\
      -i /mnt/md124/jiaxin/Empirical-Influence-Function/processed_part1_imperfect.predictions.jsonl \\
      --corpus /mnt/md124/jiaxin/Empirical-Influence-Function/cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl \\
      -o /mnt/md124/jiaxin/Empirical-Influence-Function/go_gold_ts_ids.jsonl \\
      --tokenizer /mnt/md124/jiaxin/models/Qwen3-8B \\
      --copies 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_TEST = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/"
    "processed_part1_imperfect.predictions.jsonl"
)
DEFAULT_CORPUS = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/"
    "cloud_core_test_25.JunJunly_GoOnly_length_filter.jsonl"
)
DEFAULT_OUTPUT = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/go_gold_ts_ids.jsonl"
)

_MID_OK = frozenset({
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


def _clean_gold(raw: str) -> str:
    from src.llm_train_retrieval import clean_gold_mid_completion
    from src.fast_gold_treesitter import _strip_markdown_fence

    text = clean_gold_mid_completion(raw) or (raw or "").strip()
    return _strip_markdown_fence(text)


def _test_gold(row: dict[str, Any]) -> tuple[str, str]:
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


def _rewrite_train(prompt: str, response: str, *, test_gold: str):
    from src.fim_mid_rewrite import rewrite_fim_mid
    from src.fast_gold_to_continue import _gold_in_text

    out = rewrite_fim_mid(
        prompt, response, test_gold=test_gold, expression="",
    )
    mode = str(out.get("mode") or "unchanged")
    reason = str(out.get("reason") or "")
    if mode != "unchanged" and reason == "ok":
        return str(out["prompt"]), str(out["response"]), out
    if reason in _MID_OK:
        return prompt, response, out
    if test_gold.strip() and _gold_in_text(response, test_gold):
        return prompt, response, {**out, "reason": "keep_original_gold_response"}
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


def _build_rows(
    *,
    annotated: dict[str, Any],
    copies: int,
    ids_only: bool,
    meta: dict[str, Any],
) -> list[dict[str, Any]]:
    input_ids = annotated["input_ids"]
    labels = annotated["label"]
    edges = annotated.get("attention_edges") or []
    if not isinstance(edges, list):
        edges = []
    rows: list[dict[str, Any]] = []
    base_uid = str(meta.get("uid") or meta.get("task_id") or "go_gold")
    for i in range(max(1, copies)):
        if ids_only:
            rows.append({
                "input_ids": input_ids,
                "label": labels,
                "attention_edges": edges,
            })
            continue
        uid = base_uid if i == 0 else f"{base_uid}::dup{uuid.uuid4().hex[:10]}"
        row = {
            "input_ids": input_ids,
            "label": labels,
            "attention_edges": edges,
            "uid": uid,
            "language": "Go",
            "annotation_meta": {**meta, "n_edges": len(edges), "copy": i},
        }
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
        description="Go: gold substring search → MID rewrite → tree-sitter annotate",
    )
    p.add_argument("-i", "--input", default=DEFAULT_TEST, help="test/predictions JSONL")
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT, help="output JSONL")
    p.add_argument("--corpus", default=DEFAULT_CORPUS, help="Go train JSONL")
    p.add_argument("--copies", "-n", type=int, default=1)
    p.add_argument("--start-line", type=int, default=1)
    p.add_argument("--end-line", type=int, default=0)
    p.add_argument("--max-tests", type=int, default=0)
    p.add_argument("--max-scan", type=int, default=0)
    p.add_argument("--state", default="")
    p.add_argument(
        "--skip-processed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--tokenizer", default="")
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
        help="skip annotate results with 0 edges",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    import os

    args = parse_args(argv)
    _hydrate_env()

    input_path = _resolve_file(args.input, label="input")
    if input_path is None:
        return 2
    corpus_path = _resolve_file(args.corpus, label="corpus")
    if corpus_path is None:
        return 2

    out_path = Path(args.output).expanduser()
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()

    tok_path = (
        (args.tokenizer or "").strip()
        or (os.environ.get("EIF_BASE_MODEL_PATH") or "").strip()
    )
    if not tok_path or not Path(tok_path).is_dir():
        print(f"tokenizer not found: {tok_path!r}", file=sys.stderr)
        return 2

    state_path = (
        Path(args.state).expanduser()
        if args.state
        else input_path.with_suffix(input_path.suffix + ".go_gold_ts_state.json")
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

    from src.fast_gold_to_continue import (
        _iter_corpus_hits,
        _load_raw_rows,
        _row_prompt_response,
    )

    print(f"[go-gold-ts] input={input_path}", flush=True)
    print(f"[go-gold-ts] corpus={corpus_path}", flush=True)
    print(f"[go-gold-ts] output={out_path}", flush=True)
    print(
        f"[go-gold-ts] tokenizer={tok_path} copies={args.copies} "
        f"language=Go use_llm=False",
        flush=True,
    )
    tokenizer = _load_tokenizer(tok_path)
    print("[go-gold-ts] tokenizer ready", flush=True)

    rows = _load_raw_rows(input_path)
    end_line = int(args.end_line) or 10**12
    max_tests = int(args.max_tests) or 10**12
    max_scan = int(args.max_scan) if int(args.max_scan) > 0 else None
    copies = max(1, int(args.copies))
    ids_only = bool(args.ids_only)
    max_len = max(512, int(args.max_len))
    max_edges = max(1, int(args.max_teacher_edges))

    for test_line, row in rows:
        if test_line < int(args.start_line) or test_line > end_line:
            continue
        if len(ok_tests) >= max_tests:
            break
        if args.skip_processed and test_line in processed:
            print(f"[skip] test line={test_line}", flush=True)
            continue

        task_id = str(row.get("task_id") or f"row_{test_line}")
        _fim, gold_raw = _test_gold(row)
        gold = _clean_gold(gold_raw)
        if not gold.strip():
            print(f"[test {test_line}] skip: empty gold/label", flush=True)
            processed.add(test_line)
            state["processed_test_lines"] = sorted(processed)
            state["skipped"] = (state.get("skipped") or [])[-200:] + [{
                "test_line": test_line, "reason": "empty_gold",
            }]
            _save_state(state_path, state)
            continue

        print(
            f"\n=== test line={test_line} task={task_id} "
            f"gold_chars={len(gold)} ok={len(ok_tests)} ===",
            flush=True,
        )
        t0 = time.perf_counter()
        chosen = None
        tried = 0
        for cline, crow, region in _iter_corpus_hits(
            corpus_path, gold, used_lines=used_corpus, max_scan=max_scan,
        ):
            tried += 1
            prompt, response = _row_prompt_response(crow)
            print(
                f"  try corpus_line={cline} region={region} "
                f"task={crow.get('task_id')}",
                flush=True,
            )
            try:
                new_prompt, new_response, rw = _rewrite_train(
                    prompt, response, test_gold=gold,
                )
            except Exception as exc:
                print(f"    MID fail → next: {exc}", flush=True)
                continue
            print(
                f"    rewrite ok reason={rw.get('reason')} mode={rw.get('mode')} "
                f"locus={rw.get('dig_locus')}",
                flush=True,
            )
            try:
                annotated = _annotate_treesitter(
                    tokenizer=tokenizer,
                    prompt=new_prompt,
                    response=new_response,
                    language="Go",
                    max_len=max_len,
                    max_teacher_edges=max_edges,
                )
            except Exception as exc:
                print(f"    annotate fail → next: {exc}", flush=True)
                continue
            n_edges = len(annotated.get("attention_edges") or [])
            if args.require_edges and n_edges <= 0:
                print("    0 edges → next", flush=True)
                continue
            chosen = (cline, crow, region, annotated, rw, n_edges)
            break

        search_ms = (time.perf_counter() - t0) * 1000
        if chosen is None:
            print(f"  no usable hit (tried={tried}, {search_ms:.0f}ms)", flush=True)
            processed.add(test_line)
            state["processed_test_lines"] = sorted(processed)
            state["skipped"] = (state.get("skipped") or [])[-200:] + [{
                "test_line": test_line, "reason": "no_usable_hit", "tried": tried,
            }]
            _save_state(state_path, state)
            continue

        cline, crow, region, annotated, rw, n_edges = chosen
        train_task = str(crow.get("task_id") or f"line_{cline}")
        cont_rows = _build_rows(
            annotated=annotated,
            copies=copies,
            ids_only=ids_only,
            meta={
                "uid": f"corpus:{train_task}",
                "task_id": train_task,
                "test_line": test_line,
                "test_task_id": task_id,
                "corpus_line": cline,
                "match_region": region,
                "mid_rewrite_mode": rw.get("mode"),
                "mid_rewrite_reason": rw.get("reason"),
                "go_gold_treesitter": True,
                "use_llm": False,
            },
        )
        _append_jsonl(out_path, cont_rows)

        used_corpus.add(cline)
        processed.add(test_line)
        ok_tests.append({
            "test_line": test_line,
            "task_id": task_id,
            "corpus_line": cline,
            "match_region": region,
            "n_edges": n_edges,
            "continue_rows": len(cont_rows),
            "rewrite_mode": rw.get("mode"),
            "rewrite_reason": rw.get("reason"),
        })
        state["ok_tests"] = ok_tests
        state["processed_test_lines"] = sorted(processed)
        state["used_corpus_lines"] = sorted(used_corpus)
        state["n_continue_rows"] = int(state.get("n_continue_rows") or 0) + len(cont_rows)
        _save_state(state_path, state)
        print(
            f"  wrote {len(cont_rows)} rows edges={n_edges} -> {out_path} "
            f"(total≈{state['n_continue_rows']}, {search_ms:.0f}ms)",
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
