#!/usr/bin/env python3
"""Fast path: test gold → substring search in train corpus → MID rewrite → ids JSONL.

For each test row (``prompt`` + ``response`` as gold):

  1. Take gold from ``response`` (fallback: label/gold)
  2. Scan train corpus for a row whose prompt/response contains that gold
  3. Rewrite train FIM so the dig/MID teaches that gold (``rewrite_fim_mid``)
  4. Encode ChatML ``input_ids`` + ``label`` and append to the output JSONL

Default paths (Linux training box)::

    python -m src.fast_gold_to_continue \\
      -i /mnt/md124/jiaxin/Empirical-Influence-Function/test/cpp_test_fixed.jsonl \\
      --corpus /mnt/md124/jiaxin/Empirical-Influence-Function/cpp_train_fixed.jsonl \\
      -o /mnt/md124/jiaxin/Empirical-Influence-Function/cpp_fast_gold_ids.jsonl \\
      --copies 1 --ids-only
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
    "/mnt/md124/jiaxin/Empirical-Influence-Function/test/cpp_test_fixed.jsonl"
)
DEFAULT_CORPUS = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/cpp_train_fixed.jsonl"
)
DEFAULT_OUTPUT = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/cpp_fast_gold_ids.jsonl"
)


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


def _load_raw_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl" or path.name.lower().endswith(".jsonl"):
        out: list[tuple[int, dict[str, Any]]] = []
        for i, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append((i, obj))
        return out
    stripped = text.lstrip()
    if stripped.startswith("["):
        arr = json.loads(text)
        return [(i, o) for i, o in enumerate(arr, start=1) if isinstance(o, dict)]
    non_empty = [ln for ln in text.splitlines() if ln.strip()]
    if len(non_empty) == 1:
        obj = json.loads(non_empty[0])
        if isinstance(obj, dict):
            return [(1, obj)]
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            out.append((i, obj))
    return out


def _test_fim_gold(row: dict[str, Any]) -> tuple[str, str]:
    """Test sample: prompt + response(gold). Prefer response over label."""
    prompt = str(row.get("prompt") or row.get("input") or "").strip()
    gold = str(
        row.get("response")
        or row.get("label")
        or row.get("gold")
        or row.get("target")
        or ""
    ).strip()
    return prompt, gold


def _row_prompt_response(row: dict[str, Any]) -> tuple[str, str]:
    """Train/corpus row: prompt + response (completion)."""
    prompt = str(row.get("prompt") or row.get("input") or "").strip()
    response = str(
        row.get("response")
        or row.get("label")
        or row.get("gold")
        or row.get("target")
        or ""
    ).strip()
    # Compact token rows have list ``label`` — ignore those here.
    if isinstance(row.get("label"), list) and not response:
        response = ""
    if not prompt and not response:
        sys.path.insert(0, str(REPO_ROOT / "tools" / "annotation-viewer"))
        from server.corpus_encode import extract_prompt_response  # type: ignore
        return extract_prompt_response(row)
    return prompt, response


def _haystacks(row: dict[str, Any]) -> tuple[str, str, str]:
    prompt, response = _row_prompt_response(row)
    return prompt, response, prompt + "\n" + response


def _gold_in_text(hay: str, gold: str) -> bool:
    if not hay or not gold:
        return False
    if gold in hay:
        return True
    try:
        from src.fim_mid_rewrite import ws_flex_find
    except ImportError:
        return False
    return ws_flex_find(hay, gold) is not None


def _iter_corpus_hits(
    corpus_path: Path,
    gold: str,
    *,
    used_lines: set[int],
    max_scan: int | None,
    max_hits: int = 30,
):
    """Yield (0-based line, row, match_region) for rows containing gold."""
    scanned = 0
    n = 0
    with corpus_path.open(encoding="utf-8") as fh:
        for line_idx, line in enumerate(fh):
            scanned += 1
            if max_scan is not None and scanned > max_scan:
                break
            if line_idx in used_lines:
                continue
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            prompt, response, full = _haystacks(row)
            if _gold_in_text(response, gold):
                region = "gold"
            elif _gold_in_text(prompt, gold):
                region = "context"
            elif _gold_in_text(full, gold):
                region = "cross"
            else:
                continue
            yield line_idx, row, region
            n += 1
            if n >= max_hits:
                return


def _rewrite_train(
    prompt: str,
    response: str,
    *,
    test_gold: str,
) -> tuple[str, str, dict[str, Any]]:
    from src.fim_mid_rewrite import rewrite_fim_mid

    out = rewrite_fim_mid(
        prompt,
        response,
        test_gold=test_gold,
        expression="",
    )
    mode = str(out.get("mode") or "unchanged")
    reason = str(out.get("reason") or "")
    if mode != "unchanged" and reason == "ok":
        return str(out["prompt"]), str(out["response"]), out
    if reason in ("train_mid_already_is_gold", "same_as_original_mid"):
        return prompt, response, out
    if test_gold.strip() and _gold_in_text(response, test_gold):
        return prompt, response, {**out, "reason": "keep_original_gold_response"}
    raise RuntimeError(f"MID rewrite failed: reason={reason} detail={out.get('detail')}")


def _load_tokenizer(model_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True,
    )


def _encode(tokenizer, prompt: str, response: str) -> tuple[list[int], list[int]]:
    sys.path.insert(0, str(REPO_ROOT / "tools" / "annotation-viewer"))
    from server.corpus_encode import encode_prompt_response  # type: ignore

    return encode_prompt_response(tokenizer, prompt, response)


def _append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_output_rows(
    *,
    input_ids: list[int],
    labels: list[int],
    task_id: str,
    corpus_line: int,
    corpus_path: str,
    test_line: int,
    test_task_id: str,
    match_region: str,
    rewrite_meta: dict[str, Any],
    copies: int,
    language: str,
    ids_only: bool,
) -> list[dict[str, Any]]:
    dig_hash = str(rewrite_meta.get("dig_hash") or "")[:10]
    base_uid = f"corpus:{task_id}"
    if dig_hash:
        base_uid = f"{base_uid}:midrw:{dig_hash}"
    meta = {
        "corpus": True,
        "fast_gold": True,
        "unannotated_source": True,
        "test_line": int(test_line),
        "test_task_id": test_task_id,
        "match_region": match_region,
        "mid_rewrite": bool(
            rewrite_meta.get("mode") and rewrite_meta.get("mode") != "unchanged"
        ),
        "mid_rewrite_mode": rewrite_meta.get("mode"),
        "mid_rewrite_reason": rewrite_meta.get("reason"),
        "mid_rewrite_locus": rewrite_meta.get("dig_locus"),
        "mid_rewrite_hash": rewrite_meta.get("dig_hash"),
        "continue_attention_edges": 0,
        "viz_attention_edges": 0,
    }
    rows: list[dict[str, Any]] = []
    for i in range(max(1, copies)):
        if ids_only:
            rows.append({"input_ids": input_ids, "label": labels})
            continue
        uid = base_uid if i == 0 else f"{base_uid}::dup{uuid.uuid4().hex[:10]}"
        row: dict[str, Any] = {
            "input_ids": input_ids,
            "label": labels,
            "uid": uid,
            "task_id": task_id,
            "raw_id": uid,
            "language": language,
            "source_corpus_line": int(corpus_line),
            "source_corpus_path": corpus_path,
            "attention_edges": [],
            "viz_attention_edges": [],
            "annotations": [],
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
        description="test gold → corpus search → MID rewrite → input_ids+label JSONL",
    )
    p.add_argument(
        "-i", "--input",
        default=DEFAULT_TEST,
        help=f"test JSONL (prompt+response gold); default: {DEFAULT_TEST}",
    )
    p.add_argument(
        "-o", "--output",
        default=DEFAULT_OUTPUT,
        help=f"output JSONL; default: {DEFAULT_OUTPUT}",
    )
    p.add_argument(
        "--corpus",
        default=DEFAULT_CORPUS,
        help=f"train corpus JSONL (prompt+response); default: {DEFAULT_CORPUS}",
    )
    p.add_argument("--copies", "-n", type=int, default=1, help="rows per successful test")
    p.add_argument("--start-line", type=int, default=1)
    p.add_argument("--end-line", type=int, default=0, help="0 = until EOF")
    p.add_argument("--max-tests", type=int, default=0, help="0 = no limit on successes")
    p.add_argument("--max-scan", type=int, default=0, help="max corpus lines per search (0=all)")
    p.add_argument("--state", default="", help="checkpoint JSON path")
    p.add_argument(
        "--skip-processed",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--tokenizer",
        default="",
        help="tokenizer path (default: EIF_BASE_MODEL_PATH from eif_api.env)",
    )
    p.add_argument(
        "--language",
        default="cpp",
        help="language tag stored in full-meta rows (default: cpp)",
    )
    p.add_argument(
        "--ids-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write only input_ids+label (default: true); --no-ids-only keeps meta",
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

    out_raw = (
        (args.output or "").strip()
        or (os.environ.get("ANNOTATION_CONTINUE_TRAIN_DATA") or "").strip()
    )
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
        else input_path.with_suffix(input_path.suffix + ".fast_gold_state.json")
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

    from src.llm_train_retrieval import clean_gold_mid_completion

    print(f"[fast-gold] input={input_path}", flush=True)
    print(f"[fast-gold] corpus={corpus_path}", flush=True)
    print(f"[fast-gold] output={out_path}", flush=True)
    print(
        f"[fast-gold] tokenizer={tok_path} copies={args.copies} "
        f"ids_only={args.ids_only} language={args.language}",
        flush=True,
    )
    print("[fast-gold] loading tokenizer…", flush=True)
    tokenizer = _load_tokenizer(tok_path)
    print("[fast-gold] tokenizer ready", flush=True)

    rows = _load_raw_rows(input_path)
    end_line = int(args.end_line) or 10**12
    max_tests = int(args.max_tests) or 10**12
    max_scan = int(args.max_scan) if int(args.max_scan) > 0 else None
    copies = max(1, int(args.copies))
    language = str(args.language or "cpp")
    ids_only = bool(args.ids_only)

    for test_line, row in rows:
        if test_line < int(args.start_line) or test_line > end_line:
            continue
        if len(ok_tests) >= max_tests:
            break
        if args.skip_processed and test_line in processed:
            print(f"[skip] test line={test_line} already processed", flush=True)
            continue

        task_id = str(row.get("task_id") or f"row_{test_line}")
        _fim, gold_raw = _test_fim_gold(row)
        gold = clean_gold_mid_completion(gold_raw) or gold_raw
        if not gold.strip():
            print(f"[test {test_line}] skip: empty gold/response", flush=True)
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
        chosen: tuple[int, dict[str, Any], str, str, str, dict[str, Any]] | None = None
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
            chosen = (cline, crow, region, new_prompt, new_response, rw)
            break

        search_ms = (time.perf_counter() - t0) * 1000
        if chosen is None:
            print(f"  no usable hit (tried={tried}, {search_ms:.0f}ms)", flush=True)
            processed.add(test_line)
            state["processed_test_lines"] = sorted(processed)
            state["skipped"] = (state.get("skipped") or [])[-200:] + [{
                "test_line": test_line,
                "reason": "no_usable_hit",
                "tried": tried,
            }]
            _save_state(state_path, state)
            continue

        cline, crow, region, new_prompt, new_response, rw = chosen
        input_ids, labels = _encode(tokenizer, new_prompt, new_response)
        train_task = str(crow.get("task_id") or f"line_{cline}")
        cont_rows = _build_output_rows(
            input_ids=input_ids,
            labels=labels,
            task_id=train_task,
            corpus_line=cline,
            corpus_path=str(corpus_path),
            test_line=test_line,
            test_task_id=task_id,
            match_region=region,
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
            "continue_rows": len(cont_rows),
            "n_tokens": len(input_ids),
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
            f"  wrote {len(cont_rows)} rows -> {out_path} "
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
