#!/usr/bin/env python3
"""175.jsonl：semantic 归因出语义 JSON → embedding top-10 → 用该 JSON 标注 → 续训 → 训练 → 评测。

对 ``raw_ce/175.jsonl`` 的每一条：

  1. 走 8766 的 semantic 归因，只拿四字段语义 JSON。不按这个 JSON 去搜训练集。
  2. 用全文 embedding 和「洞上一行 + middle + 洞下一行」的 embedding 取 top-10。
     全文直接用已有的 train_code_embeddings.npz。洞附近按行重算进
     train_gold_embeddings.npz。
  3. 把第 1 步的语义 JSON 作为 ``target_semantic``，按分数从高到低做 LLM 语义标注。
     一条没有边就换下一条，直到标上。10 条都失败才跳过。

全部结束后，每条成功标注复制成 10 份，接在
``csn_go_train_fim_10k_ids.jsonl`` 前 10000 行后面，写成单独的混合文件。
原始 10k ids 文件不改。然后在 GPU 2 上按 go-sal-175 训练，结束后用最后一个
checkpoint 跑 line-hit 评测。

8766 和 annotation-viewer 需要已经开着。viewer 的 continue 文件必须是本脚本的
``--continue-data``。

Example::

    python -m src.auto_go_sal_175_embed_continue --max-tests 1 --skip-train
    python -m src.auto_go_sal_175_embed_continue
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.auto_go_sal_new_continue import (
    _head_lines,
    _repeat_row,
    _write_mixed,
)
from src.auto_go_semantic_verify_continue import _annotate_hit
from src.auto_go_top5_rerank_continue import (
    _count_lines,
    _load_state,
    _prompt_gold_pred,
    _read_lines,
    _save_state,
)
from src.auto_raw_ce_to_continue import (
    PipelineClient,
    _continue_paths_match,
    _http_json,
    _load_raw_rows,
    _viewer_health,
)
from src.gold_live_attribution import _hydrate_eif_env

DEFAULT_INPUT = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/"
    "correlation_matching_results/raw_ce/175.jsonl"
)
DEFAULT_CONTINUE = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/go_sal_175_continue.jsonl"
)
DEFAULT_TRAIN_IDS = (
    "/mnt/md124/jiaxin/training_code/data/csn_go_train_fim_10k_ids.jsonl"
)
DEFAULT_MIXED = "/mnt/md124/jiaxin/training_code/data/go_sal_175_train.jsonl"
DEFAULT_TRAIN_REPO = "/mnt/md124/jiaxin/training_code/Empirical-Influence-Function"
DEFAULT_BASE_MODEL = "/mnt/md124/jiaxin/models/Qwen3-8B"
DEFAULT_EVAL_SCRIPT = (
    "/mnt/md124/jiaxin/AI4GoPlayGround-personal-h00613474-v3/eval_go_fim_line_hit.py"
)
DEFAULT_EVAL_INPUT = (
    "/mnt/md124/jiaxin/training_code/data/csn_go_test_fim_sample1000.jsonl"
)
DEFAULT_EVAL_DIR = "/mnt/md124/jiaxin/training_code/SAL_175_outputs"
DEFAULT_STATE_NAME = "go_sal_175_embed_continue_state.json"
EMBED_TOP_K = 10


def _query_semantic(
    eif_url: str,
    fim: str,
    gold: str,
    pred: str,
    language: str,
) -> dict[str, Any]:
    """Semantic attribution card for this test row. Corpus search stays off."""
    body = _http_json(
        "POST",
        f"{eif_url.rstrip('/')}/api/llm-semantic-retrieve",
        {
            "fimPrompt": fim,
            "goldCompletion": gold,
            "modelPrediction": pred,
            "topK": 1,
            "runCorpusSearch": False,
            "language": language,
        },
        timeout=600.0,
    )
    sem = body.get("semantic")
    if not isinstance(sem, dict) or not sem:
        raise RuntimeError("semantic card missing")
    return sem


def _embed_hits(
    eif_url: str,
    fim: str,
    gold: str,
    *,
    hole_weight: float,
    top_k: int,
) -> list[dict[str, Any]]:
    url = f"{eif_url.rstrip('/')}/api/code-embed-retrieve"
    restart = True
    while True:
        body = _http_json(
            "POST",
            url,
            {
                "fimPrompt": fim,
                "goldCompletion": gold,
                "topK": top_k,
                "holeWeight": hole_weight,
                "restart": restart,
            },
            timeout=600.0,
        )
        restart = False
        status = str(body.get("status") or "")
        if status == "building":
            print(
                f"  [embed] {body.get('message') or body.get('phase') or 'building'} "
                f"{body.get('done')}/{body.get('total')}",
                flush=True,
            )
            time.sleep(5.0)
            continue
        if status != "success":
            raise RuntimeError(str(body.get("message") or "code embed retrieve failed"))
        raw_hits = body.get("hits") or []
        if not raw_hits:
            raise RuntimeError("embedding retrieve returned no hit")
        corpus_path = body.get("corpus_path")
        hits: list[dict[str, Any]] = []
        for raw in raw_hits[:top_k]:
            hit = dict(raw)
            hit["corpus_path"] = corpus_path
            hits.append(hit)
        return hits


def _commit_rec(state: dict[str, Any], rec: dict[str, Any]) -> None:
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


def _expand_continue(path: Path, start_lines: int, copies: int) -> tuple[int, int]:
    rows = _read_lines(path)
    head = rows[:start_lines]
    fresh = rows[start_lines:]
    expanded: list[str] = []
    for raw in fresh:
        expanded.extend(_repeat_row(raw, copies))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.writelines(head)
        handle.writelines(expanded)
    tmp.replace(path)
    print(
        f"[data] continue {len(head)} kept + {len(fresh)} annotated × {copies} "
        f"= {len(expanded)} → {path}",
        flush=True,
    )
    return len(fresh), len(expanded)


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
        "--output_dir", "./outputs/go-sal-175",
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
        "--run_name", "go-sal-175",
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
) -> int:
    output_dir = train_repo / "outputs" / "go-sal-175"
    log_path = train_repo / "go-sal-175.log"
    pid = int(state.get("train_pid") or 0)
    if state.get("train_exit") is not None and not _pid_alive(pid):
        return int(state["train_exit"])
    if pid and _pid_alive(pid):
        print(f"[train] waiting for existing pid={pid}", flush=True)
    else:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "2"
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
        print(f"[train] pid={pid} gpu=2 log={log_path}", flush=True)
        code = int(proc.wait())
        state["train_exit"] = code
        _save_state(state_path, state)
        print(f"[train] exit={code}", flush=True)
        return code
    while _pid_alive(pid):
        time.sleep(30.0)
    code = int(state.get("train_exit") if state.get("train_exit") is not None else 0)
    if not (output_dir / "checkpoint-100").exists() and not list(output_dir.glob("checkpoint-*")):
        print(f"[train] pid {pid} ended without a checkpoint; see {log_path}", flush=True)
        return code or 1
    state["train_exit"] = code
    _save_state(state_path, state)
    return code


def _run_eval(adapter: Path, eval_script: Path, eval_input: Path, eval_dir: Path) -> int:
    eval_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(eval_script),
        "--input", str(eval_input),
        "--base-model", DEFAULT_BASE_MODEL,
        "--adapter", str(adapter),
        "--output-dir", str(eval_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "2"
    print("[eval] " + " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, env=env, check=False)
    print(f"[eval] exit={proc.returncode}", flush=True)
    return int(proc.returncode)


def main(argv: list[str] | None = None) -> int:
    _hydrate_eif_env()
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
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
    parser.add_argument("--viewer-url", default="http://127.0.0.1:8765")
    parser.add_argument("--hole-weight", type=float, default=0.4)
    parser.add_argument("--copies", type=int, default=10)
    parser.add_argument("--head-n", type=int, default=10000)
    parser.add_argument("--max-tests", type=int, default=0, help="0 = every row")
    parser.add_argument("--language", default="go")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)

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
            "  cd tools/annotation-viewer && python -m server.main "
            f"--port 8765 --continue-data {continue_path}",
            flush=True,
        )
        return 2

    rows = _load_raw_rows(input_path)
    client = PipelineClient(
        eif_url=args.eif_url,
        viewer_url=args.viewer_url,
        corpus_path=None,
        annotate="llm-semantic",
        top_k=1,
        max_scan=None,
        graphsignal_use_llm=None,
        dry_run=False,
        retrieve_mode="semantic",
    )
    retry_lines: set[int] = set()
    for old in state.get("results") or []:
        if not str(old.get("reason") or "").startswith("annotate_failed"):
            continue
        try:
            retry_lines.add(int(old["test_line"]))
        except (KeyError, TypeError, ValueError):
            pass
    done = {int(x) for x in state.get("processed_lines") or []} - retry_lines
    limit = int(args.max_tests)
    seen = 0
    for test_line, row in rows:
        if limit and seen >= limit:
            break
        seen += 1
        if int(test_line) in done:
            continue
        fim, gold, pred = _prompt_gold_pred(row)
        raw_gold = str(row.get("label") or row.get("response") or gold)
        task_id = str(row.get("task_id") or f"row_{test_line}")
        rec: dict[str, Any] = {"test_line": test_line, "task_id": task_id}
        print(f"\n=== line={test_line} task={task_id} ===", flush=True)
        if not fim or not str(raw_gold).strip():
            rec["reason"] = "missing_prompt_or_gold"
            _commit_rec(state, rec)
            _save_state(state_path, state)
            continue
        try:
            sem = _query_semantic(args.eif_url, fim, gold, pred, args.language)
        except Exception as exc:
            rec["reason"] = f"semantic_failed: {exc}"
            print(f"  [fail] {rec['reason']}", flush=True)
            _commit_rec(state, rec)
            _save_state(state_path, state)
            continue
        rec["semantic"] = sem
        print(
            f"  semantic role={(sem.get('role') or '')[:80]!r}",
            flush=True,
        )
        try:
            hits = _embed_hits(
                args.eif_url,
                fim,
                raw_gold,
                hole_weight=float(args.hole_weight),
                top_k=EMBED_TOP_K,
            )
        except Exception as exc:
            rec["reason"] = f"embed_failed: {exc}"
            print(f"  [fail] {rec['reason']}", flush=True)
            _commit_rec(state, rec)
            _save_state(state_path, state)
            continue
        row_ann = None
        note = "no embedding hit"
        hit: dict[str, Any] = {}
        rank = 0
        for rank, cand in enumerate(hits, start=1):
            print(
                f"  embed rank={rank}/{len(hits)} line={cand.get('line')} "
                f"score={cand.get('score')} "
                f"ctx={cand.get('context_score')} gold={cand.get('gold_score')}",
                flush=True,
            )
            hit_corpus = str(cand.get("corpus_path") or "").strip() or None
            row_ann, note = _annotate_hit(
                client,
                gold=gold,
                language=args.language,
                hit=cand,
                expression="",
                hit_corpus=hit_corpus,
                target_semantic=sem,
                scratch_path=continue_path,
            )
            hit = cand
            if row_ann is not None:
                break
            print(f"    [skip] rank={rank} {note}", flush=True)
        rec["embed_rank"] = rank
        rec["top"] = {
            "line": hit.get("line"),
            "task_id": hit.get("task_id"),
            "score": hit.get("score"),
            "context_score": hit.get("context_score"),
            "gold_score": hit.get("gold_score"),
            "rank": rank,
            "note": note,
            "ok": row_ann is not None,
        }
        if row_ann is None:
            print(f"    [skip] all {len(hits)} candidates failed", flush=True)
            rec["reason"] = note
        else:
            meta = row_ann.get("_verify_meta") if isinstance(row_ann.get("_verify_meta"), dict) else {}
            rec["n_edges"] = int(meta.get("n_continue_edges") or 0)
            print(
                f"    annotated rank={rank} line={hit.get('line')} edges={rec['n_edges']}",
                flush=True,
            )
        _commit_rec(state, rec)
        _save_state(state_path, state)
        time.sleep(0.2)

    wanted = min(len(rows), limit) if limit else len(rows)
    processed = {int(x) for x in state.get("processed_lines") or []}
    finished = sum(1 for line, _row in rows[:wanted] if int(line) in processed)
    if finished < wanted:
        print("[stop] input not finished; not mixing or training", flush=True)
        return 0

    start_lines = int(state.get("continue_start_lines") or 0)
    if not state.get("expanded"):
        _expand_continue(continue_path, start_lines, copies)
        state["expanded"] = True
        _save_state(state_path, state)
    if not state.get("finalized"):
        _head = _head_lines(train_ids, int(args.head_n))
        if len(_head) < int(args.head_n):
            print(f"[fail] train ids has {len(_head)} rows, need {args.head_n}", flush=True)
            return 2
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
    )
    if code != 0:
        print(f"[fail] training exit {code}", flush=True)
        return code
    if args.skip_eval or state.get("eval_done"):
        print("[done] training finished", flush=True)
        return 0
    adapter = _latest_checkpoint(Path(args.train_repo) / "outputs" / "go-sal-175")
    print(f"[eval] adapter={adapter}", flush=True)
    eval_code = _run_eval(
        adapter,
        Path(args.eval_script),
        Path(args.eval_input),
        Path(args.eval_dir),
    )
    state["eval_done"] = eval_code == 0
    state["eval_adapter"] = str(adapter)
    state["eval_exit"] = eval_code
    _save_state(state_path, state)
    return eval_code


if __name__ == "__main__":
    sys.exit(main())
