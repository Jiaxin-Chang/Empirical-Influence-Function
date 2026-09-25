#!/usr/bin/env python3
"""已标注的 Top-5 召回：一步 CE+saliency，下降后再比一步纯 CE。

读 ``go_top5_rerank_continue_state.json`` 里每条 valid 留下的训练行号，
到续训 JSONL 里按 ``source_corpus_line`` 取最后一条标注，只拿这一条做
``maxSteps=1``。看的是这条 valid 的 gold CE：

  1. CE+saliency 之后 loss 必须下降。
  2. 下降了才跑一步纯 CE。纯 CE 的下降幅度必须更小。
  3. 两条都满足，这条召回算通过。

同一条训练样本被多道题选中时，文件里现在只有最后一次标注，几道题会共用它。
统计按 valid 输出：5 个召回里通过了几个。

全部 valid 都验完之后：

  * 写出每条 valid 哪些召回通过了。
  * 按 valid 计通过的召回：同一条训练行被不同 valid 通过，各算一条。
    每条各复制 10 份；复制后总行数仍少于 500 时改为各复制 20 份。
  * 追加到 csn_go_train_fim_10k_ids.jsonl 末尾，并启动 SAL_go_valid。

``--max-tests`` 只试跑，不追加、不开训。

Example::

    python -m src.auto_go_top5_onestep_verify --max-tests 1
    python -m src.auto_go_top5_onestep_verify
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

from src.auto_go_semantic_verify_continue import (
    _compact_ok,
    _recover_adapter,
    _run_continue_train,
    _verify_candidate,
    _write_copies,
    _write_one_test,
)
from src.auto_go_top5_rerank_continue import _prompt_gold_pred
from src.auto_raw_ce_to_continue import _load_raw_rows
from src.gold_live_attribution import _hydrate_eif_env

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_STATE = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/"
    "go_top5_rerank_continue_state.json"
)
DEFAULT_CONTINUE = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/"
    "new_continue_annotated_subset.jsonl"
)
DEFAULT_VALID = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/"
    "correlation_matching_results/raw_ce/valid_imperfect_less.jsonl"
)
DEFAULT_WORKDIR = REPO_ROOT / "go_top5_onestep_verify_work"
DEFAULT_OUT = REPO_ROOT / "go_top5_onestep_verify_state.json"
DEFAULT_PASSED = REPO_ROOT / "go_top5_onestep_verify_passed.json"
DEFAULT_TRAIN_IDS = (
    "/mnt/md124/jiaxin/training_code/data/csn_go_train_fim_10k_ids.jsonl"
)
DEFAULT_TRAIN_REPO = "/mnt/md124/jiaxin/training_code/Empirical-Influence-Function"
DEFAULT_BASE_MODEL = "/mnt/md124/jiaxin/models/Qwen3-8B"


def _index_continue(path: Path) -> dict[int, dict[str, Any]]:
    """Last compact row for each source_corpus_line."""
    found: dict[int, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or not _compact_ok(obj):
                continue
            try:
                line = int(obj.get("source_corpus_line"))
            except (TypeError, ValueError):
                continue
            found[line] = obj
    return found


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _save(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    n_tests = len(results)
    n_kept = 0
    n_pass = 0
    n_sal_drop = 0
    buckets = {i: 0 for i in range(6)}
    for rec in results:
        hits = rec.get("hits") or []
        passed = sum(1 for h in hits if h.get("passed"))
        n_kept += len(hits)
        n_pass += passed
        n_sal_drop += sum(1 for h in hits if (h.get("saliency_drop") or 0) > 0)
        buckets[min(5, passed)] = buckets.get(min(5, passed), 0) + 1
    return {
        "n_valid": n_tests,
        "n_recalls": n_kept,
        "n_pass": n_pass,
        "n_saliency_dropped": n_sal_drop,
        "pass_per_valid": {str(k): buckets[k] for k in range(6)},
    }


def _passed_record(results: list[dict[str, Any]]) -> dict[str, Any]:
    items = []
    for rec in results:
        hits = [h for h in (rec.get("hits") or []) if isinstance(h, dict)]
        passed = [
            {
                "rank": h.get("rank"),
                "line": h.get("line"),
                "saliency_drop": h.get("saliency_drop"),
                "ce_drop": h.get("ce_drop"),
            }
            for h in hits
            if h.get("passed")
        ]
        items.append({
            "test_line": rec.get("test_line"),
            "task_id": rec.get("task_id"),
            "n_kept": rec.get("n_kept", len(hits)),
            "n_pass": len(passed),
            "passed": passed,
        })
    return {"n_valid": len(items), "items": items}


def _passed_hits(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One entry per valid that passed a recall. Same train line on another valid stays."""
    hits: list[dict[str, Any]] = []
    for rec in results:
        test_line = rec.get("test_line")
        task_id = rec.get("task_id")
        for hit in rec.get("hits") or []:
            if not isinstance(hit, dict) or not hit.get("passed"):
                continue
            try:
                line = int(hit.get("line"))
            except (TypeError, ValueError):
                continue
            hits.append({
                "test_line": test_line,
                "task_id": task_id,
                "rank": hit.get("rank"),
                "line": line,
            })
    return hits


def _copies_for(n_hits: int) -> int:
    if n_hits <= 0:
        return 0
    if n_hits * 10 < 500:
        return 20
    return 10


def _append_passed(
    by_line: dict[int, dict[str, Any]],
    hits: list[dict[str, Any]],
    train_ids: Path,
    *,
    copies: int,
) -> int:
    rows: list[dict[str, Any]] = []
    missing = 0
    for hit in hits:
        line = int(hit["line"])
        ann = by_line.get(line)
        if ann is None:
            missing += 1
            continue
        base_uid = str(ann.get("uid") or ann.get("task_id") or f"line_{line}")
        tag = f"v{hit.get('test_line')}_r{hit.get('rank')}"
        for i in range(copies):
            obj = json.loads(json.dumps(ann, ensure_ascii=False))
            obj["uid"] = f"{base_uid}::salvalid_{tag}_{i:02d}_{uuid.uuid4().hex[:8]}"
            obj["raw_id"] = obj["uid"]
            obj["duplicate_of"] = base_uid
            obj["verify_test_line"] = hit.get("test_line")
            obj["verify_task_id"] = hit.get("task_id")
            rows.append(obj)
    if missing:
        print(f"[warn] {missing} passed recalls had no continue row", flush=True)
    if not rows:
        print("[warn] no passed rows to append", flush=True)
        return 0
    train_ids.parent.mkdir(parents=True, exist_ok=True)
    with train_ids.open("a", encoding="utf-8") as handle:
        for obj in rows:
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(
        f"[data] passed recalls={len(hits) - missing} copies={copies} "
        f"appended {len(rows)} rows onto {train_ids}",
        flush=True,
    )
    return len(rows)


def _launch_train(train_repo: Path, base_model: str, data_path: Path) -> None:
    log_path = train_repo / "SAL_go_valid.log"
    cmd = [
        sys.executable,
        "src/train/train.py",
        "--model_name_or_path", base_model,
        "--data_path", str(data_path),
        "--output_dir", "./outputs/SAL_go_valid",
        "--use_peft", "True",
        "--lora_r", "16",
        "--lora_alpha", "32",
        "--lora_dropout", "0.05",
        "--lora_target_modules", "q_proj,k_proj,v_proj,o_proj",
        "--loss_mode", "ce_saliency",
        "--saliency_loss_type", "contrastive",
        "--saliency_lambda", "1.5",
        "--saliency_alpha", "1.0",
        "--saliency_margin_plus", "2.0",
        "--saliency_layer", "-1",
        "--saliency_neg_sample_k", "64",
        "--num_train_epochs", "2",
        "--gradient_checkpointing", "True",
        "--ddp_find_unused_parameters", "False",
        "--enable_attn_viz", "False",
        "--saliency_detail_log_steps", "0",
        "--eval_codebleu_samples", "0",
        "--per_device_train_batch_size", "1",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "2e-5",
        "--lr_scheduler_type", "cosine",
        "--warmup_ratio", "0.03",
        "--max_grad_norm", "1.0",
        "--max_len", "2000",
        "--bf16", "True",
        "--use_flash_attention", "False",
        "--save_strategy", "steps",
        "--save_steps", "100",
        "--save_total_limit", "2",
        "--logging_steps", "1",
        "--dataloader_num_workers", "0",
        "--report_to", "none",
        "--run_name", "SAL_go_valid",
        "--remove_unused_columns", "False",
    ]
    log_handle = log_path.open("a", encoding="utf-8")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "3"
    proc = subprocess.Popen(
        cmd,
        cwd=str(train_repo),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    print(f"[train] pid={proc.pid} cuda=3 log={log_path}", flush=True)


def _print_summary(summary: dict[str, Any]) -> None:
    print(
        f"\n[summary] valid={summary['n_valid']} recalls={summary['n_recalls']} "
        f"pass={summary['n_pass']} saliency_dropped={summary['n_saliency_dropped']}",
        flush=True,
    )
    print("  每条 valid 的 5 个召回里，通过个数分布：", flush=True)
    for k in range(6):
        print(f"    {k}/5 : {summary['pass_per_valid'][str(k)]}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="One-step CE+saliency vs CE on saved Top-5 hits.")
    p.add_argument("--state", default=DEFAULT_STATE)
    p.add_argument("--continue-path", default=DEFAULT_CONTINUE)
    p.add_argument("--valid", default=DEFAULT_VALID)
    p.add_argument("--workdir", default=str(DEFAULT_WORKDIR))
    p.add_argument("--out", default=str(DEFAULT_OUT))
    p.add_argument("--passed-out", default=str(DEFAULT_PASSED))
    p.add_argument("--train-ids", default=os.environ.get("EIF_TRAIN_DATA") or DEFAULT_TRAIN_IDS)
    p.add_argument("--train-repo", default=DEFAULT_TRAIN_REPO)
    p.add_argument("--base-model", default=os.environ.get("EIF_BASE_MODEL_PATH") or DEFAULT_BASE_MODEL)
    p.add_argument("--eif-url", default="http://127.0.0.1:8766")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--max-tests", type=int, default=0, help="0 = all")
    p.add_argument("--max-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--min-drop", type=float, default=0.0)
    p.add_argument("--adapter-family", default="ce", choices=("ce", "saliency"))
    p.add_argument("--train-timeout", type=float, default=7200.0)
    p.add_argument("--poll", type=float, default=5.0)
    p.add_argument("--fresh", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _hydrate_eif_env()
    state_in = _load_json(Path(args.state))
    results_in = state_in.get("results") or []
    if not isinstance(results_in, list) or not results_in:
        print(f"[fail] no results in {args.state}", flush=True)
        return 2
    continue_path = Path(args.continue_path)
    if not continue_path.is_file():
        print(f"[fail] continue file missing: {continue_path}", flush=True)
        return 2
    by_line = _index_continue(continue_path)
    print(f"[data] continue rows by source line: {len(by_line)}", flush=True)
    valid_rows = {int(i): row for i, row in _load_raw_rows(Path(args.valid))}
    workdir = Path(args.workdir)
    out_path = Path(args.out)
    passed_path = Path(args.passed_out)
    out = {} if args.fresh else _load_json(out_path)
    done = {int(x) for x in (out.get("processed_lines") or [])}
    out.setdefault("results", [])
    out.setdefault("processed_lines", [])

    n_new = 0
    for rec_in in results_in:
        if not isinstance(rec_in, dict):
            continue
        test_line = int(rec_in.get("test_line") or 0)
        if test_line in done:
            continue
        if args.max_tests and n_new >= args.max_tests:
            break
        task_id = str(rec_in.get("task_id") or f"row_{test_line}")
        row = valid_rows.get(test_line)
        kept = [h for h in (rec_in.get("kept") or []) if isinstance(h, dict) and h.get("ok")]
        print(f"\n=== valid line={test_line} task={task_id} kept={len(kept)} ===", flush=True)
        rec: dict[str, Any] = {
            "test_line": test_line,
            "task_id": task_id,
            "hits": [],
        }
        if row is None:
            rec["reason"] = "valid_row_missing"
            out["results"].append(rec)
            out["processed_lines"].append(test_line)
            _save(out_path, out)
            n_new += 1
            continue
        fim, gold, _pred = _prompt_gold_pred(row)
        if not fim or not gold:
            rec["reason"] = "missing_prompt_or_gold"
            out["results"].append(rec)
            out["processed_lines"].append(test_line)
            _save(out_path, out)
            n_new += 1
            continue
        test_path = workdir / "tests" / f"t{test_line}.jsonl"
        _write_one_test(test_path, task_id=task_id, prompt=fim, label=gold)
        current_test = {"taskId": task_id, "prompt": fim, "label": gold}
        for rank, hit in enumerate(kept, start=1):
            try:
                line = int(hit.get("line"))
            except (TypeError, ValueError):
                rec["hits"].append({"rank": rank, "passed": False, "fail_reason": "bad_line"})
                continue
            ann = by_line.get(line)
            item: dict[str, Any] = {"rank": rank, "line": line, "passed": False}
            if ann is None:
                item["fail_reason"] = "continue_row_missing"
                print(f"  rank={rank} line={line} missing annotation", flush=True)
                rec["hits"].append(item)
                continue
            trial = workdir / "trials" / f"t{test_line}_r{rank}.jsonl"
            _write_copies(ann, trial, 1)
            print(f"  rank={rank} line={line} one-step verify", flush=True)
            try:
                passed, detail = _verify_candidate(
                    args.eif_url,
                    trial_path=trial,
                    test_path=test_path,
                    workdir=workdir,
                    test_line=test_line,
                    rank=rank,
                    current_test=current_test,
                    max_steps=args.max_steps,
                    learning_rate=args.lr,
                    adapter_family=args.adapter_family,
                    min_drop=args.min_drop,
                    timeout=args.train_timeout,
                    poll=args.poll,
                )
            except Exception as exc:
                _recover_adapter(args.eif_url)
                item["fail_reason"] = str(exc)
                rec["hits"].append(item)
                continue
            item.update(detail)
            item["passed"] = bool(passed)
            rec["hits"].append(item)
        n_pass = sum(1 for h in rec["hits"] if h.get("passed"))
        rec["n_pass"] = n_pass
        rec["n_kept"] = len(rec["hits"])
        print(f"  valid line={test_line} pass {n_pass}/{len(rec['hits'])}", flush=True)
        out["results"].append(rec)
        out["processed_lines"].append(test_line)
        out["summary"] = _summarize(out["results"])
        _save(out_path, out)
        _save(passed_path, _passed_record(out["results"]))
        n_new += 1

    summary = _summarize(out.get("results") or [])
    out["summary"] = summary
    passed = _passed_record(out.get("results") or [])
    _save(out_path, out)
    _save(passed_path, passed)
    _print_summary(summary)
    print(f"[out] {out_path}", flush=True)
    print(f"[passed] {passed_path}", flush=True)

    expected = {
        int(rec.get("test_line") or 0)
        for rec in results_in
        if isinstance(rec, dict)
    }
    finished = expected.issubset(set(out.get("processed_lines") or []))
    if not finished:
        print("[data] not all valids verified yet; skip append and train", flush=True)
        return 0
    if out.get("finalized"):
        print("[data] already appended and trained; skip", flush=True)
        return 0

    hits = _passed_hits(out.get("results") or [])
    copies = _copies_for(len(hits))
    print(
        f"[data] passed recalls={len(hits)} copies_each={copies} "
        f"total={len(hits) * copies}",
        flush=True,
    )
    n_appended = _append_passed(by_line, hits, Path(args.train_ids), copies=copies)
    out["finalized"] = True
    out["append"] = {
        "n_passed_recalls": len(hits),
        "copies_each": copies,
        "n_appended": n_appended,
        "train_ids": str(args.train_ids),
        "passed_hits": hits,
    }
    _save(out_path, out)
    if args.skip_train:
        print("[train] skipped", flush=True)
        return 0
    _launch_train(Path(args.train_repo), args.base_model, Path(args.train_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
