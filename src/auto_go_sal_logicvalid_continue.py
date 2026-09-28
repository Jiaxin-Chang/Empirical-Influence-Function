#!/usr/bin/env python3
"""valid predictions：semantic top-10 → 逐条 LLM 标注 → 拼进 10k → 训练 → line-hit。

和正在跑的旧进程分开：旧进程写 ``new_4.jsonl``、训练 ``go-sal-logicvalid``、
标注后端 8765。本脚本默认写 ``go_sal_residual_continue.jsonl``、训练
``go-sal-residual``、标注后端 8774、GPU 1。8766 两边共用。

对 ``go_valid_prediction.jsonl`` 的每一条（含 prompt、label、predict）：

  1. 8766 ``/api/llm-semantic-retrieve`` 取 top-10 训练样本。
     predict 和 gold 去掉空白后不同时，四字段描述的是这个差，不是整段 gold。
  2. 按分数从高到低，用该条的四字段语义 JSON 做 LLM 语义标注，写入
     本脚本自己的 continue 文件。生成的边为空就跳过这一条 hit，继续下一条。

全部 prediction 行过完后，把本轮写入的续训行接在
``csn_go_train_fim_10k_ids.jsonl`` 前 10000 行后面。
没有 ``predict``、或 predict 与 gold 只差空白的行直接跳过。
8766 若没有返回 residual，脚本停掉，不继续标注。
原始 10k ids 不改。然后在 GPU 1 上训练 ``./outputs/go-sal-residual``，
结束后用该目录里编号最大的 checkpoint 跑 line-hit 评测。

8766 和本脚本的 annotation-viewer（8774）需要已经开着。

Example::

    python -m src.auto_go_sal_logicvalid_continue --max-tests 1 --skip-train
    python -m src.auto_go_sal_logicvalid_continue
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.auto_go_sal_new_continue import _head_lines, _write_mixed
from src.auto_go_semantic_verify_continue import _annotate_hit
from src.auto_go_top5_rerank_continue import (
    _count_lines,
    _load_state,
    _save_state,
)
from src.auto_raw_ce_to_continue import (
    PipelineClient,
    _collect_semantic_hits,
    _continue_paths_match,
    _fim_and_gold,
    _load_raw_rows,
    _row_language,
    _viewer_health,
)
from src.gold_live_attribution import _hydrate_eif_env
from src.llm_semantic_retrieval import _fills_equivalent

DEFAULT_INPUT = "/mnt/md124/jiaxin/training_code/data/go_valid_prediction.jsonl"
DEFAULT_RAW_TRAIN = "/mnt/md124/jiaxin/training_code/data/csn_go_train_fim_10k.jsonl"
DEFAULT_SEMANTIC = (
    "/mnt/md124/jiaxin/training_code/data/csn_go_train_fim_10k.semantic.vllm32b.jsonl"
)
DEFAULT_CONTINUE = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/go_sal_residual_continue.jsonl"
)
DEFAULT_TRAIN_IDS = (
    "/mnt/md124/jiaxin/training_code/data/csn_go_train_fim_10k_ids.jsonl"
)
DEFAULT_MIXED = "/mnt/md124/jiaxin/training_code/data/go_sal_residual_train.jsonl"
DEFAULT_TRAIN_REPO = "/mnt/md124/jiaxin/training_code/Empirical-Influence-Function"
DEFAULT_BASE_MODEL = "/mnt/md124/jiaxin/models/Qwen3-8B"
DEFAULT_EVAL_SCRIPT = (
    "/mnt/md124/jiaxin/AI4GoPlayGround-personal-h00613474-v3/eval_go_fim_line_hit.py"
)
DEFAULT_EVAL_INPUT = "/mnt/md124/jiaxin/training_code/data/csn_go_test_fim_logic.jsonl"
DEFAULT_EVAL_DIR = "/mnt/md124/jiaxin/training_code/SAL_residual_outputs"
DEFAULT_STATE_NAME = "go_sal_residual_state.json"
DEFAULT_GPU = "1"
DEFAULT_VIEWER = "http://127.0.0.1:8774"
TOP_K = 10
OUTPUT_NAME = "go-sal-residual"


def _gold_of(row: dict[str, Any]) -> tuple[str, str, str]:
    fim, gold = _fim_and_gold(row)
    if not gold:
        gold = str(row.get("middle") or "").strip()
    pred = str(row.get("predict") or row.get("prediction") or "").strip()
    return fim, gold, pred


def _has_predict_field(rows: list[tuple[int, dict[str, Any]]]) -> bool:
    return any("predict" in row or "prediction" in row for _line, row in rows)


def _query_mode(retrieve: dict[str, Any]) -> str:
    query = retrieve.get("query") if isinstance(retrieve.get("query"), dict) else {}
    return str(query.get("query_mode") or "")


def _latest_checkpoint(output_dir: Path) -> Path:
    found: list[tuple[int, Path]] = []
    if not output_dir.is_dir():
        raise FileNotFoundError(f"output dir missing: {output_dir}")
    for path in output_dir.glob("checkpoint-*"):
        if not path.is_dir():
            continue
        tail = path.name.split("-", 1)[-1]
        if tail.isdigit():
            found.append((int(tail), path))
    if not found:
        raise FileNotFoundError(f"no checkpoint-* under {output_dir}")
    found.sort()
    return found[-1][1]


def _train_cmd(base_model: str, data_path: Path) -> list[str]:
    return [
        sys.executable,
        "src/train/train.py",
        "--model_name_or_path", base_model,
        "--data_path", str(data_path),
        "--output_dir", f"./outputs/{OUTPUT_NAME}",
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
        "--run_name", OUTPUT_NAME,
        "--remove_unused_columns", "False",
    ]


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _launch_or_wait_train(
    train_repo: Path,
    base_model: str,
    data_path: Path,
    state: dict[str, Any],
    state_path: Path,
    gpu: str,
) -> int:
    output_dir = train_repo / "outputs" / OUTPUT_NAME
    log_path = train_repo / f"{OUTPUT_NAME}.log"
    pid = int(state.get("train_pid") or 0)
    if state.get("train_exit") is not None and not _pid_alive(pid):
        return int(state["train_exit"])
    if pid and _pid_alive(pid):
        print(f"[train] waiting for existing pid={pid}", flush=True)
    else:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("a", encoding="utf-8")
        proc = subprocess.Popen(
            _train_cmd(base_model, data_path),
            cwd=str(train_repo),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        pid = int(proc.pid)
        state["train_pid"] = pid
        state["train_log"] = str(log_path)
        _save_state(state_path, state)
        print(f"[train] pid={pid} gpu={gpu} log={log_path}", flush=True)
        code = int(proc.wait())
        state["train_exit"] = code
        _save_state(state_path, state)
        print(f"[train] exit={code}", flush=True)
        return code
    while _pid_alive(pid):
        time.sleep(30.0)
    code = int(state.get("train_exit") if state.get("train_exit") is not None else 0)
    if not list(output_dir.glob("checkpoint-*")):
        print(f"[train] pid {pid} ended without a checkpoint; see {log_path}", flush=True)
        return code or 1
    state["train_exit"] = code
    _save_state(state_path, state)
    return code


def _run_eval(
    adapter: Path,
    eval_script: Path,
    eval_input: Path,
    eval_dir: Path,
    base_model: str,
    gpu: str,
) -> int:
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(eval_script),
        "--input", str(eval_input),
        "--base-model", base_model,
        "--adapter", str(adapter),
        "--output-dir", str(eval_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    print("[eval] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env, check=False)
    print(f"[eval] exit={proc.returncode}", flush=True)
    return int(proc.returncode)


def _commit(state: dict[str, Any], rec: dict[str, Any]) -> None:
    line = int(rec["test_line"])
    kept: list[dict[str, Any]] = []
    for old in state.get("results") or []:
        try:
            if int(old.get("test_line")) == line:
                continue
        except (TypeError, ValueError):
            pass
        kept.append(old)
    kept.append(rec)
    state["results"] = kept
    processed: list[int] = []
    seen: set[int] = set()
    for item in list(state.get("processed_lines") or []) + [line]:
        try:
            ix = int(item)
        except (TypeError, ValueError):
            continue
        if ix in seen:
            continue
        seen.add(ix)
        processed.append(ix)
    state["processed_lines"] = processed


def _retrieve(client: PipelineClient, fim: str, gold: str, pred: str, language: str) -> dict[str, Any]:
    return client.llm_semantic_retrieve(
        fim,
        gold,
        language=language or None,
        model_prediction=pred or None,
    )


def main(argv: list[str] | None = None) -> int:
    _hydrate_eif_env()
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--raw-train", default=DEFAULT_RAW_TRAIN)
    parser.add_argument("--semantic-corpus", default=DEFAULT_SEMANTIC)
    parser.add_argument("--continue-data", default=DEFAULT_CONTINUE)
    parser.add_argument("--train-ids", default=os.environ.get("EIF_TRAIN_DATA") or DEFAULT_TRAIN_IDS)
    parser.add_argument("--mixed-out", default=DEFAULT_MIXED)
    parser.add_argument("--train-repo", default=DEFAULT_TRAIN_REPO)
    parser.add_argument("--base-model", default=os.environ.get("EIF_BASE_MODEL_PATH") or DEFAULT_BASE_MODEL)
    parser.add_argument("--eval-script", default=DEFAULT_EVAL_SCRIPT)
    parser.add_argument("--eval-input", default=DEFAULT_EVAL_INPUT)
    parser.add_argument("--eval-dir", default=DEFAULT_EVAL_DIR)
    parser.add_argument("--state", default=str(repo / DEFAULT_STATE_NAME))
    parser.add_argument("--eif-url", default="http://127.0.0.1:8766")
    parser.add_argument("--viewer-url", default=DEFAULT_VIEWER)
    parser.add_argument("--gpu", default=DEFAULT_GPU)
    parser.add_argument("--head-n", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-tests", type=int, default=0, help="0 = every row")
    parser.add_argument("--language", default="go")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)

    input_path = Path(args.input)
    raw_train = Path(args.raw_train)
    semantic_corpus = Path(args.semantic_corpus)
    continue_path = Path(args.continue_data)
    train_ids = Path(args.train_ids)
    mixed_path = Path(args.mixed_out)
    state_path = Path(args.state)
    if args.fresh and state_path.is_file():
        state_path.unlink()
    state = _load_state(state_path)
    recorded_input = str(state.get("input") or "")
    if state.get("processed_lines") and recorded_input != str(input_path):
        print(
            f"[fail] state {state_path} belongs to {recorded_input or 'an older input'}.\n"
            "That run attributed the FIM file with no predict. Pass --fresh "
            "(this does not delete the continue JSONL).",
            flush=True,
        )
        return 2
    for label, path in (("raw train", raw_train), ("semantic corpus", semantic_corpus)):
        if not path.is_file():
            print(f"[fail] {label} missing: {path}", flush=True)
            return 2

    health = _viewer_health(args.viewer_url)
    actual = str(health.get("continue_path") or health.get("continuePath") or "")
    if actual and not _continue_paths_match(str(continue_path), actual):
        print(
            f"[fail] viewer continue file is {actual}, want {continue_path}.\n"
            "Do not use port 8765 (the already-running job writes new_4.jsonl there).\n"
            "Start this run's viewer with:\n"
            "  cd tools/annotation-viewer && python -m server.main "
            f"--port 8774 --continue-data {continue_path}",
            flush=True,
        )
        return 2

    rows = _load_raw_rows(input_path)
    if not _has_predict_field(rows):
        print(
            f"[fail] {input_path} has no predict field. "
            "Semantic attribution would describe the whole gold. "
            "Use go_valid_prediction.jsonl.",
            flush=True,
        )
        return 2
    if "continue_start_lines" not in state:
        state["continue_start_lines"] = _count_lines(continue_path)
        state["input"] = str(input_path)
        _save_state(state_path, state)
        print(
            f"[data] continue already has {state['continue_start_lines']} rows; "
            "those are left out of the mixed train file",
            flush=True,
        )
    client = PipelineClient(
        eif_url=args.eif_url,
        viewer_url=args.viewer_url,
        corpus_path=str(raw_train),
        annotate="llm-semantic",
        top_k=max(1, int(args.top_k)),
        max_scan=None,
        graphsignal_use_llm=None,
        dry_run=False,
        retrieve_mode="semantic",
        semantic_corpus_path=str(semantic_corpus),
    )
    done = {int(x) for x in state.get("processed_lines") or []}
    limit = int(args.max_tests)
    seen = 0
    for test_line, row in rows:
        if limit and seen >= limit:
            break
        seen += 1
        if int(test_line) in done:
            continue
        fim, gold, pred = _gold_of(row)
        task_id = str(row.get("task_id") or f"row_{test_line}")
        lang = _row_language(row, args.language)
        rec: dict[str, Any] = {"test_line": test_line, "task_id": task_id, "annotated": []}
        print(f"\n=== line={test_line} task={task_id} ===", flush=True)
        if not fim or not gold:
            rec["reason"] = "missing_prompt_or_gold"
            _commit(state, rec)
            _save_state(state_path, state)
            continue
        if not pred:
            rec["reason"] = "missing_predict"
            print("  [skip] no predict; refusing full-span attribution", flush=True)
            _commit(state, rec)
            _save_state(state_path, state)
            continue
        if _fills_equivalent(gold, pred):
            rec["reason"] = "predict_matches_gold"
            print("  [skip] predict matches gold; no residual", flush=True)
            _commit(state, rec)
            _save_state(state_path, state)
            continue
        try:
            retrieve = _retrieve(client, fim, gold, pred, lang)
        except Exception as exc:
            rec["reason"] = f"semantic_failed: {exc}"
            print(f"  [fail] {rec['reason']}", flush=True)
            _commit(state, rec)
            _save_state(state_path, state)
            continue
        if str(retrieve.get("status") or "") not in ("success", "ok", ""):
            rec["reason"] = f"semantic_status: {retrieve.get('message') or retrieve.get('status')}"
            print(f"  [fail] {rec['reason']}", flush=True)
            _commit(state, rec)
            _save_state(state_path, state)
            continue
        mode = _query_mode(retrieve)
        if mode != "residual":
            print(
                f"[fail] 8766 query_mode={mode or '(empty)'} on a row whose predict differs from gold.\n"
                "Attribution would describe the whole gold. Restart 8766 from this repo.",
                flush=True,
            )
            return 2
        got_sem = str(retrieve.get("semantic_corpus_path") or "")
        if not _continue_paths_match(str(semantic_corpus), got_sem):
            print(
                f"[fail] 8766 searched {got_sem or '(empty)'}, want {semantic_corpus}.",
                flush=True,
            )
            return 2
        sr0 = (retrieve.get("search_results") or [{}])[0]
        if isinstance(sr0, dict) and sr0.get("corpus_error"):
            print(f"[fail] semantic corpus: {sr0.get('corpus_error')}", flush=True)
            return 2
        hits = _collect_semantic_hits(retrieve)[: max(1, int(args.top_k))]
        target_semantic = retrieve.get("semantic") if isinstance(retrieve.get("semantic"), dict) else None
        print(
            f"  mode={mode} semantic hits={len(hits)} "
            f"role={(target_semantic or {}).get('role', '')[:80]!r}",
            flush=True,
        )
        if not hits:
            rec["reason"] = "no_semantic_hits"
            _commit(state, rec)
            _save_state(state_path, state)
            continue
        for rank, hit in enumerate(hits, start=1):
            print(
                f"  rank={rank}/{len(hits)} line={hit.get('line')} "
                f"score={hit.get('semantic_score')}",
                flush=True,
            )
            row_ann, note = _annotate_hit(
                client,
                gold=gold,
                language=lang,
                hit=hit,
                expression="",
                hit_corpus=str(raw_train),
                target_semantic=target_semantic,
                scratch_path=continue_path,
            )
            item = {
                "rank": rank,
                "line": hit.get("line"),
                "task_id": hit.get("task_id"),
                "score": hit.get("semantic_score"),
                "ok": row_ann is not None,
                "note": note,
            }
            if row_ann is None:
                print(f"    [skip] {note}", flush=True)
            else:
                meta = row_ann.get("_verify_meta") if isinstance(row_ann.get("_verify_meta"), dict) else {}
                item["n_edges"] = int(meta.get("n_continue_edges") or 0)
                print(f"    annotated edges={item['n_edges']}", flush=True)
            rec["annotated"].append(item)
        rec["n_ok"] = sum(1 for item in rec["annotated"] if item.get("ok"))
        _commit(state, rec)
        _save_state(state_path, state)
        time.sleep(0.2)

    wanted = min(len(rows), limit) if limit else len(rows)
    processed = {int(x) for x in state.get("processed_lines") or []}
    finished = sum(1 for line, _row in rows[:wanted] if int(line) in processed)
    if finished < wanted:
        print("[stop] input not finished; not mixing or training", flush=True)
        return 0

    if not state.get("finalized"):
        head = _head_lines(train_ids, int(args.head_n))
        if len(head) < int(args.head_n):
            print(f"[fail] train ids has {len(head)} rows, need {args.head_n}", flush=True)
            return 2
        n_continue = _count_lines(continue_path)
        print(f"[data] continue file has {n_continue} rows", flush=True)
        start_lines = int(state.get("continue_start_lines") or 0)
        _write_mixed(
            train_ids,
            continue_path,
            mixed_path,
            head_n=int(args.head_n),
            start_lines=start_lines,
            copies=1,
        )
        state["finalized"] = True
        state["mixed_out"] = str(mixed_path)
        _save_state(state_path, state)
    if args.skip_train:
        print("[done] skip-train set", flush=True)
        return 0

    code = _launch_or_wait_train(
        Path(args.train_repo),
        args.base_model,
        mixed_path,
        state,
        state_path,
        str(args.gpu),
    )
    if code != 0:
        print(f"[fail] training exit {code}", flush=True)
        return code
    if args.skip_eval or state.get("eval_done"):
        print("[done] training finished", flush=True)
        return 0
    adapter = _latest_checkpoint(Path(args.train_repo) / "outputs" / OUTPUT_NAME)
    print(f"[eval] adapter={adapter}", flush=True)
    eval_code = _run_eval(
        adapter,
        Path(args.eval_script),
        Path(args.eval_input),
        Path(args.eval_dir),
        args.base_model,
        str(args.gpu),
    )
    state["eval_done"] = eval_code == 0
    state["eval_adapter"] = str(adapter)
    state["eval_exit"] = eval_code
    _save_state(state_path, state)
    return eval_code


if __name__ == "__main__":
    sys.exit(main())
