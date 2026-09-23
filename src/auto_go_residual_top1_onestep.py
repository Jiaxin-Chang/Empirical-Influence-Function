#!/usr/bin/env python3
"""100 条错测：残差语义 → 全库 Top-1 → 标注 → 一步 CE+saliency，统计 gold CE 下降条数。

对 ``valid_imperfect_less.jsonl`` 每一条（prompt + label/completion + predict）：

  1. ``/api/llm-semantic-retrieve`` 用验证侧残差提示词（带上 predict）。
     不传 semanticCorpusPath，语料与是否有 embedding 都跟 ``eif_api.env``。
     当前 ``EIF_LLM_SEMANTIC_EMBEDDINGS`` 为空时，API 走全库 Stage-2。
  2. 只取 semantic_score 最高的 1 条训练样本。
  3. MID 对齐后，用这条测试的 4 字段 semantic 做 LLM 标注。
     续训读的是 compact ``input_ids``；没有边时 CE+saliency 会退回纯 CE。
  4. 只把这一条写进 trial（不复制 20 份、不叠加其它续训样本），
     ``maxSteps=1``、``lossMode=ce_saliency``。
  5. 看当前测试的 gold CE：``loss_after < loss_before`` 记为下降。
  6. ``/api/continue-adapter-recover``，下一条从原始 CE adapter 再出发。

viewer 的 continue JSONL 必须是本脚本的 scratch，标注落在那里，trial 另写。

Example::

    python -m src.auto_go_residual_top1_onestep

    python -m src.auto_go_residual_top1_onestep --max-tests 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from src.auto_go_semantic_verify_continue import (
    _annotate_hit,
    _loss_pair,
    _recover_adapter,
    _run_continue_train,
    _write_copies,
    _write_one_test,
)
from src.auto_raw_ce_to_continue import (
    PipelineClient,
    _collect_semantic_hits,
    _continue_paths_match,
    _gold_and_predict,
    _load_raw_rows,
    _row_language,
    _viewer_health,
)
from src.gold_live_attribution import _hydrate_eif_env

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/"
    "correlation_matching_results/raw_ce/valid_imperfect_less.jsonl"
)
DEFAULT_SCRATCH = REPO_ROOT / "go_residual_top1_scratch.jsonl"
DEFAULT_WORKDIR = REPO_ROOT / "go_residual_top1_work"


def _env(name: str) -> str:
    import os

    return (os.environ.get(name) or "").strip()


def _prompt_gold_pred(row: dict[str, Any]) -> tuple[str, str, str]:
    packed = dict(row)
    if not str(packed.get("label") or "").strip():
        packed["label"] = (
            row.get("completion") or row.get("response") or row.get("gold") or ""
        )
    gold, pred = _gold_and_predict(packed)
    prompt = str(row.get("prompt") or row.get("input") or "").strip()
    return prompt, gold, pred


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"processed_lines": [], "results": [], "n_decreased": 0}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"processed_lines": [], "results": [], "n_decreased": 0}
    if not isinstance(obj, dict):
        return {"processed_lines": [], "results": [], "n_decreased": 0}
    obj.setdefault("processed_lines", [])
    obj.setdefault("results", [])
    obj.setdefault("n_decreased", 0)
    return obj


def _record(
    state: dict[str, Any],
    state_path: Path,
    *,
    test_line: int,
    rec: dict[str, Any],
) -> None:
    state["results"].append(rec)
    state["processed_lines"].append(int(test_line))
    if rec.get("decreased"):
        state["n_decreased"] = int(state.get("n_decreased") or 0) + 1
    _save_state(state_path, state)


def _process_one(
    client: PipelineClient,
    *,
    test_line: int,
    row: dict[str, Any],
    language: str,
    scratch_path: Path,
    workdir: Path,
    max_steps: int,
    learning_rate: float,
    adapter_family: str,
    timeout: float,
    poll: float,
) -> dict[str, Any]:
    task_id = str(row.get("task_id") or f"row_{test_line}")
    fim, gold, pred = _prompt_gold_pred(row)
    rec: dict[str, Any] = {
        "test_line": test_line,
        "task_id": task_id,
        "decreased": False,
    }
    if not fim or not gold:
        rec["reason"] = "missing_prompt_or_gold"
        return rec
    if not pred:
        rec["reason"] = "missing_predict"
        return rec

    lang = _row_language(row, language)
    print(
        f"\n=== test line={test_line} task_id={task_id} "
        f"gold_chars={len(gold)} pred_chars={len(pred)} ===",
        flush=True,
    )
    try:
        retrieve = client.llm_semantic_retrieve(
            fim,
            gold,
            language=lang or None,
            model_prediction=pred,
        )
    except Exception as exc:
        rec["reason"] = f"semantic_retrieve_failed: {exc}"
        print(f"  [fail] {rec['reason']}", flush=True)
        return rec

    query = retrieve.get("query") if isinstance(retrieve.get("query"), dict) else {}
    rec["query_mode"] = query.get("query_mode")
    sem = retrieve.get("semantic") if isinstance(retrieve.get("semantic"), dict) else None
    print(f"  query_mode={rec.get('query_mode')}", flush=True)

    sr0 = (retrieve.get("search_results") or [{}])[0]
    if isinstance(sr0, dict) and sr0.get("corpus_error"):
        rec["reason"] = f"semantic_corpus_error: {sr0.get('corpus_error')}"
        print(f"  [fail] {rec['reason']}", flush=True)
        return rec

    hits = _collect_semantic_hits(retrieve)
    if not hits:
        rec["reason"] = "no_hits"
        print("  [fail] no semantic hits", flush=True)
        return rec
    hit = hits[0]
    rec["hit_line"] = hit.get("line")
    rec["hit_task_id"] = hit.get("task_id")
    rec["semantic_score"] = hit.get("semantic_score")
    rec["retrieval"] = hit.get("retrieval")
    rec["relation_align"] = hit.get("relation_align")
    print(
        f"  top1 line={hit.get('line')} score={hit.get('semantic_score')} "
        f"retrieval={hit.get('retrieval')} relation_align={hit.get('relation_align')} "
        f"task={hit.get('task_id') or '-'}",
        flush=True,
    )

    expression = str(retrieve.get("semantic_flat_text") or "")
    hit_corpus = str(retrieve.get("corpus_path") or "").strip() or client.corpus_path
    if client.dry_run:
        rec["reason"] = "dry_run"
        return rec

    row_ann, note = _annotate_hit(
        client,
        gold=gold,
        language=lang,
        hit=hit,
        expression=expression,
        hit_corpus=hit_corpus,
        target_semantic=sem,
        scratch_path=scratch_path,
    )
    if row_ann is None:
        rec["reason"] = note
        print(f"  [fail] annotate: {note}", flush=True)
        return rec
    rec["mid_note"] = note

    trial = workdir / "trials" / f"t{test_line}.jsonl"
    test_path = workdir / "tests" / f"t{test_line}.jsonl"
    _write_copies(row_ann, trial, 1)
    _write_one_test(test_path, task_id=task_id, prompt=fim, label=gold)
    current_test = {"taskId": task_id, "prompt": fim, "label": gold}
    try:
        result = _run_continue_train(
            client.eif_url,
            train_data=trial,
            test_data=test_path,
            output_dir=workdir / "runs" / f"t{test_line}",
            loss_mode="ce_saliency",
            current_test=current_test,
            max_steps=max_steps,
            learning_rate=learning_rate,
            adapter_family=adapter_family,
            timeout=timeout,
            poll=poll,
            predict_current=False,
        )
    except Exception as exc:
        rec["reason"] = f"continue_train_failed: {exc}"
        print(f"  [fail] {rec['reason']}", flush=True)
        return rec
    finally:
        _recover_adapter(client.eif_url)

    cur = result.get("currentTest") if isinstance(result, dict) else None
    before, after, delta = _loss_pair(cur if isinstance(cur, dict) else None)
    rec["loss_before"] = before
    rec["loss_after"] = after
    rec["loss_delta"] = delta
    rec["decreased"] = (
        before is not None and after is not None and after < before
    )
    if not rec["decreased"] and before is None:
        rec["reason"] = "missing_gold_ce"
    print(
        f"  gold CE {before} → {after} delta={delta} "
        f"decreased={rec['decreased']}",
        flush=True,
    )
    return rec


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Residual semantic Top-1 + one-step CE+saliency; count gold-CE drops.",
    )
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument(
        "--corpus-path",
        default="",
        help="raw FIM jsonl for MID (default: EIF_LLM_TRAIN_CORPUS). "
        "Semantic corpus stays the API env EIF_LLM_SEMANTIC_CORPUS.",
    )
    p.add_argument("--continue-path", default=str(DEFAULT_SCRATCH))
    p.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    p.add_argument("--eif-url", default="http://127.0.0.1:8766")
    p.add_argument("--viewer-url", default="http://127.0.0.1:8765")
    p.add_argument("--language", default="go")
    p.add_argument("--max-tests", type=int, default=0, help="0 = all rows")
    p.add_argument("--max-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--adapter-family", default="ce", choices=("ce", "saliency"))
    p.add_argument("--train-timeout", type=float, default=7200.0)
    p.add_argument("--poll", type=float, default=5.0)
    p.add_argument("--state", default="")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--fresh", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    _hydrate_eif_env(force_file=True)
    args = parse_args(argv)
    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        print(f"input not found: {input_path}", file=sys.stderr)
        return 2

    corpus = (args.corpus_path or _env("EIF_LLM_TRAIN_CORPUS")).strip()
    scratch_path = Path(args.continue_path).expanduser()
    workdir = Path(args.workdir).expanduser()
    workdir.mkdir(parents=True, exist_ok=True)
    state_path = (
        Path(args.state).expanduser() if args.state else workdir / "state.json"
    )
    if args.fresh and state_path.is_file():
        state_path.unlink()
    state = {"processed_lines": [], "results": [], "n_decreased": 0} if args.fresh else _load_state(state_path)
    processed = {int(x) for x in state.get("processed_lines") or []}

    client = PipelineClient(
        eif_url=args.eif_url,
        viewer_url=args.viewer_url,
        corpus_path=corpus or None,
        annotate="llm-semantic",
        top_k=1,
        max_scan=None,
        graphsignal_use_llm=None,
        dry_run=bool(args.dry_run),
        retrieve_mode="semantic",
        semantic_corpus_path=None,
    )
    client.ping()
    if not args.dry_run:
        try:
            health = _viewer_health(client.viewer_url)
        except Exception as exc:
            print(
                f"cannot read viewer /api/health at {client.viewer_url}: {exc}\n"
                f"start annotation-viewer with --continue-data {scratch_path}",
                file=sys.stderr,
            )
            return 2
        actual = str(health.get("continue_path") or "").strip()
        if not _continue_paths_match(str(scratch_path), actual):
            print(
                f"viewer continue_path mismatch.\n"
                f"  expected scratch: {scratch_path}\n"
                f"  actual:           {actual or '(unset)'}\n"
                f"Restart annotation-viewer:\n"
                f"  cd tools/annotation-viewer && python -m server.main "
                f"--continue-data {scratch_path}",
                file=sys.stderr,
            )
            return 2
        print(f"[scratch] viewer writes to {actual}", flush=True)

    rows = _load_raw_rows(input_path)
    print(
        f"[start] input={input_path} n={len(rows)} "
        f"raw_corpus={corpus or 'viewer-default'} "
        f"semantic_corpus=env top_k=1 steps={args.max_steps} "
        f"already_done={len(processed)}",
        flush=True,
    )
    ran = 0
    limit = int(args.max_tests) if int(args.max_tests) > 0 else len(rows)
    for test_line, row in rows:
        if ran >= limit:
            break
        if test_line in processed:
            print(f"[skip] line={test_line} already in state", flush=True)
            continue
        t0 = time.time()
        rec = _process_one(
            client,
            test_line=test_line,
            row=row,
            language=str(args.language or ""),
            scratch_path=scratch_path,
            workdir=workdir,
            max_steps=max(1, int(args.max_steps)),
            learning_rate=float(args.lr),
            adapter_family=str(args.adapter_family),
            timeout=float(args.train_timeout),
            poll=float(args.poll),
        )
        rec["elapsed_sec"] = round(time.time() - t0, 1)
        _record(state, state_path, test_line=test_line, rec=rec)
        ran += 1
        print(
            f"  progress decreased={state['n_decreased']}/{len(state['results'])}",
            flush=True,
        )

    n_dec = int(state.get("n_decreased") or 0)
    n_all = len(state.get("results") or [])
    print(
        f"\n=== loss_decreased={n_dec} / evaluated={n_all} "
        f"state={state_path} ===",
        flush=True,
    )
    print(n_dec, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
