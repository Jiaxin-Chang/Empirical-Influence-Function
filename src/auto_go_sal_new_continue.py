#!/usr/bin/env python3
"""100 条错测：8767 语义召回 Top-10 → 文本相似度重排 → 标注 Top-1 → 每条复制成 10 份。

100 条成功则续训文件里是 1000 行。再取
``csn_go_train_fim_10k_ids.jsonl`` 的前 10000 行，接上这 1000 行，
以 CE+saliency 训练，输出目录 ``outputs/sal_new``。

不改写原始 10k 文件。标注后端必须已经用 ``new_3.jsonl`` 启动。

Example::

    python -m src.auto_go_sal_new_continue --max-tests 1 --skip-train
    python -m src.auto_go_sal_new_continue
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from src.auto_go_semantic_verify_continue import _annotate_hit
from src.auto_go_top5_rerank_continue import (
    _count_lines,
    _load_completions,
    _load_state,
    _prompt_gold_pred,
    _read_lines,
    _save_state,
)
from src.auto_raw_ce_to_continue import (
    PipelineClient,
    _collect_semantic_hits,
    _continue_paths_match,
    _load_raw_rows,
    _viewer_health,
)
from src.gold_live_attribution import _hydrate_eif_env
from src.structural_pair_retrieval import _cosine_counts, _text_hist

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/"
    "correlation_matching_results/raw_ce/valid_imperfect_less.jsonl"
)
DEFAULT_CONTINUE = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/new_3.jsonl"
)
DEFAULT_TRAIN_IDS = (
    "/mnt/md124/jiaxin/training_code/data/csn_go_train_fim_10k_ids.jsonl"
)
DEFAULT_MIXED = (
    "/mnt/md124/jiaxin/training_code/data/sal_new_train.jsonl"
)
DEFAULT_TRAIN_REPO = "/mnt/md124/jiaxin/training_code/Empirical-Influence-Function"
DEFAULT_BASE_MODEL = "/mnt/md124/jiaxin/models/Qwen3-8B"
DEFAULT_STATE = REPO_ROOT / "go_sal_new_continue_state.json"


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _code_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", text or "")


def _text_sim(a: str, b: str) -> float:
    return _cosine_counts(_text_hist(_code_tokens(a)), _text_hist(_code_tokens(b)))


def _rerank_text(
    gold: str,
    hits: list[dict[str, Any]],
    completions: dict[int, str],
) -> list[dict[str, Any]]:
    ranked: list[tuple[float, dict[str, Any]]] = []
    for hit in hits:
        try:
            line = int(hit.get("line"))
        except (TypeError, ValueError):
            line = -1
        mid = completions.get(line) or str(hit.get("response_preview") or "")
        text = _text_sim(gold, mid)
        sem = float(hit.get("semantic_score") or 0.0)
        row = dict(hit)
        row["text_sim"] = round(text, 4)
        row["rerank_score"] = round(text, 4)
        row["semantic_score"] = sem
        row["mid_preview"] = mid[:240]
        ranked.append((text, sem, row))
    ranked.sort(key=lambda item: (-item[0], -item[1]))
    return [row for _text, _sem, row in ranked]


def _append_copies(row: dict[str, Any], dest: Path, extra: int) -> int:
    """The accept path already wrote one row. Append ``extra`` uid-distinct copies."""
    if extra <= 0:
        return 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    base_uid = str(row.get("uid") or row.get("task_id") or row.get("raw_id") or "sample")
    n = 0
    with dest.open("a", encoding="utf-8") as handle:
        for i in range(extra):
            obj = json.loads(json.dumps(row, ensure_ascii=False))
            obj.pop("_verify_meta", None)
            obj["uid"] = f"{base_uid}::c{i:02d}_{uuid.uuid4().hex[:8]}"
            obj["raw_id"] = obj["uid"]
            obj["duplicate_of"] = base_uid
            handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    return n


def _head_lines(path: Path, n: int) -> list[str]:
    out: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            out.append(raw if raw.endswith("\n") else raw + "\n")
            if len(out) >= n:
                break
    return out


def _write_mixed(
    train_ids: Path,
    continue_path: Path,
    mixed_path: Path,
    *,
    head_n: int,
    start_lines: int,
) -> tuple[int, int]:
    head = _head_lines(train_ids, head_n)
    fresh = _read_lines(continue_path)[start_lines:]
    mixed_path.parent.mkdir(parents=True, exist_ok=True)
    with mixed_path.open("w", encoding="utf-8") as handle:
        handle.writelines(head)
        handle.writelines(fresh)
    print(
        f"[data] mixed {len(head)} train head + {len(fresh)} annotated "
        f"→ {mixed_path}",
        flush=True,
    )
    return len(head), len(fresh)


def _launch_train(train_repo: Path, base_model: str, data_path: Path) -> None:
    log_path = train_repo / "sal_new.log"
    cmd = [
        sys.executable,
        "src/train/train.py",
        "--model_name_or_path", base_model,
        "--data_path", str(data_path),
        "--output_dir", "./outputs/sal_new",
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
        "--run_name", "sal_new",
        "--remove_unused_columns", "False",
    ]
    log_handle = log_path.open("a", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        cwd=str(train_repo),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    print(f"[train] pid={proc.pid} log={log_path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    _hydrate_eif_env()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--continue-data", default=DEFAULT_CONTINUE)
    p.add_argument("--train-ids", default=_env("EIF_TRAIN_DATA") or DEFAULT_TRAIN_IDS)
    p.add_argument("--train-corpus", default=_env("EIF_LLM_TRAIN_CORPUS"))
    p.add_argument("--mixed-out", default=DEFAULT_MIXED)
    p.add_argument("--train-repo", default=DEFAULT_TRAIN_REPO)
    p.add_argument("--base-model", default=_env("EIF_BASE_MODEL_PATH") or DEFAULT_BASE_MODEL)
    p.add_argument("--state", default=str(DEFAULT_STATE))
    p.add_argument("--eif-url", default="http://127.0.0.1:8767")
    p.add_argument("--viewer-url", default="http://127.0.0.1:8765")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--copies", type=int, default=10, help="rows per successful top-1, including the accepted row")
    p.add_argument("--head-n", type=int, default=10000)
    p.add_argument("--max-tests", type=int, default=100)
    p.add_argument("--language", default="go")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args(argv)

    input_path = Path(args.input)
    continue_path = Path(args.continue_data)
    train_ids = Path(args.train_ids)
    mixed_path = Path(args.mixed_out)
    state_path = Path(args.state)
    copies = max(1, int(args.copies))
    if args.fresh and state_path.is_file():
        state_path.unlink()
    state = _load_state(state_path)
    if "continue_start_lines" not in state:
        state["continue_start_lines"] = _count_lines(continue_path)
        _save_state(state_path, state)

    health = _viewer_health(args.viewer_url)
    actual = str(health.get("continue_path") or health.get("continuePath") or "")
    if actual and not _continue_paths_match(str(continue_path), actual):
        print(
            f"[fail] viewer continue file is {actual}, want {continue_path}.\n"
            "Start it with:\n"
            "  cd tools/annotation-viewer && nohup python -m server.main "
            f"--port 8765 --continue-data {continue_path} "
            "> annotation_8765.log 2>&1 &",
            flush=True,
        )
        return 2

    rows = _load_raw_rows(input_path)
    completions = _load_completions(Path(args.train_corpus)) if args.train_corpus else {}
    client = PipelineClient(
        eif_url=args.eif_url,
        viewer_url=args.viewer_url,
        corpus_path=args.train_corpus or None,
        annotate="llm",
        top_k=args.top_k,
        max_scan=None,
        graphsignal_use_llm=None,
        dry_run=False,
        retrieve_mode="semantic",
    )
    done = {int(x) for x in state.get("processed_lines") or []}
    n_ok = 0
    limit = int(args.max_tests)
    for test_line, row in rows:
        if limit and n_ok >= limit:
            break
        if int(test_line) in done:
            n_ok += 1
            continue
        fim, gold, pred = _prompt_gold_pred(row)
        task_id = str(row.get("task_id") or f"row_{test_line}")
        rec: dict[str, Any] = {"test_line": test_line, "task_id": task_id}
        print(f"\n=== line={test_line} task={task_id} ===", flush=True)
        if not fim or not gold or not pred:
            rec["reason"] = "missing_prompt_gold_or_predict"
            state["results"].append(rec)
            state["processed_lines"].append(int(test_line))
            _save_state(state_path, state)
            n_ok += 1
            continue
        try:
            retrieve = client.llm_semantic_retrieve(
                fim, gold, language=args.language, model_prediction=pred,
            )
        except Exception as exc:
            rec["reason"] = f"retrieve_failed: {exc}"
            print(f"  [fail] {rec['reason']}", flush=True)
            state["results"].append(rec)
            state["processed_lines"].append(int(test_line))
            _save_state(state_path, state)
            n_ok += 1
            continue
        hits = _collect_semantic_hits(retrieve)[: args.top_k]
        if not hits:
            rec["reason"] = "no_hits"
            state["results"].append(rec)
            state["processed_lines"].append(int(test_line))
            _save_state(state_path, state)
            n_ok += 1
            continue
        ranked = _rerank_text(gold, hits, completions)
        top = ranked[0]
        print(
            f"  top1 line={top.get('line')} text={top.get('text_sim')} "
            f"sem={top.get('semantic_score')}",
            flush=True,
        )
        sem = retrieve.get("semantic") if isinstance(retrieve.get("semantic"), dict) else None
        expression = str(retrieve.get("semantic_flat_text") or "")
        hit_corpus = str(retrieve.get("corpus_path") or "").strip() or client.corpus_path
        row_ann, note = _annotate_hit(
            client,
            gold=gold,
            language=args.language,
            hit=top,
            expression=expression,
            hit_corpus=hit_corpus,
            target_semantic=sem,
            scratch_path=continue_path,
        )
        rec["top"] = {
            "line": top.get("line"),
            "text_sim": top.get("text_sim"),
            "semantic_score": top.get("semantic_score"),
            "note": note,
            "ok": row_ann is not None,
        }
        if row_ann is None:
            print(f"    [skip] {note}", flush=True)
            rec["reason"] = note
        else:
            extra = _append_copies(row_ann, continue_path, copies - 1)
            rec["copies"] = 1 + extra
            print(f"    annotated + {extra} copies → {1 + extra} rows", flush=True)
        state["results"].append(rec)
        state["processed_lines"].append(int(test_line))
        _save_state(state_path, state)
        n_ok += 1
        time.sleep(0.2)

    wanted = min(len(rows), limit) if limit else len(rows)
    processed = {int(x) for x in state.get("processed_lines") or []}
    seen = sum(1 for line, _row in rows[:wanted] if int(line) in processed)
    if seen < wanted:
        print("[stop] input not finished; not mixing or training", flush=True)
        return 0
    if state.get("finalized"):
        print("[done] already finalized", flush=True)
        return 0
    _write_mixed(
        train_ids,
        continue_path,
        mixed_path,
        head_n=int(args.head_n),
        start_lines=int(state.get("continue_start_lines") or 0),
    )
    state["finalized"] = True
    state["mixed_out"] = str(mixed_path)
    _save_state(state_path, state)
    if args.skip_train:
        print("[done] skip-train set", flush=True)
        return 0
    _launch_train(Path(args.train_repo), args.base_model, mixed_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
