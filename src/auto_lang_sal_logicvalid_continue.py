#!/usr/bin/env python3
"""One-language logic-valid SAL: semantic top-10 → annotate → mix → train.

8766 is shared. Each language has its own annotation-viewer port and continue
JSONL, so rows are not appended into the Go ``new_4.jsonl`` on port 8765.

Semantic search is sent with that language's ``*.semantic.vllm32b.jsonl``.
Annotation reads the matching raw FIM jsonl (prompt + response), not the Go
corpus pinned in the running 8766 process.

The semantic jsonl must already exist (``python -m src.fim_semantic_preprocess``).
Line numbers in it are 0-based indexes into the raw train file.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.auto_go_sal_logicvalid_continue import _has_predict_field, _query_mode
from src.auto_go_sal_new_continue import _head_lines, _write_mixed
from src.auto_go_semantic_verify_continue import _annotate_hit
from src.auto_go_top5_rerank_continue import _count_lines, _load_state, _save_state
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

DATA = "/mnt/md124/jiaxin/training_code/data"
REPO_DATA = "/mnt/md124/jiaxin/Empirical-Influence-Function"
TRAIN_REPO = "/mnt/md124/jiaxin/training_code/Empirical-Influence-Function"
BASE_MODEL = "/mnt/md124/jiaxin/models/Qwen3-8B"
TOP_K = 10


@dataclass(frozen=True)
class LangSpec:
    key: str
    language: str
    predictions: str
    raw_train: str
    semantic_corpus: str
    train_ids: str
    continue_data: str
    mixed_out: str
    output_name: str
    viewer_port: int
    gpu: str
    state_name: str


def _spec(
    key: str,
    language: str,
    pred_name: str,
    stem: str,
    viewer_port: int,
    gpu: str,
) -> LangSpec:
    return LangSpec(
        key=key,
        language=language,
        predictions=f"{DATA}/{pred_name}",
        raw_train=f"{DATA}/{stem}.jsonl",
        semantic_corpus=f"{DATA}/{stem}.semantic.vllm32b.jsonl",
        train_ids=f"{DATA}/{stem}_ids.jsonl",
        continue_data=f"{REPO_DATA}/{key}_sal_logicvalid_continue.jsonl",
        mixed_out=f"{DATA}/{key}_sal_logicvalid_train.jsonl",
        output_name=f"{key}-sal",
        viewer_port=viewer_port,
        gpu=gpu,
        state_name=f"{key}_sal_logicvalid_state.json",
    )


# Cards follow the id-file order js, java, py → GPU 1, 2, 3.
# Ports stay off 8765 (Go continue) and 8766 (shared retrieve API).
SPECS: dict[str, LangSpec] = {
    "js": _spec(
        "js", "javascript", "js_valid_prediction.jsonl",
        "csn_js_train_fim_logic10000", 8771, "1",
    ),
    "java": _spec(
        "java", "java", "java_valid_prediction.jsonl",
        "csn_java_train_fim_logic10000", 8772, "2",
    ),
    "py": _spec(
        "py", "python", "python_valid_prediction.jsonl",
        "csn_py_train_fim_logic10000", 8773, "3",
    ),
}


def _viewer_start(spec: LangSpec) -> str:
    return (
        "cd /mnt/md124/jiaxin/Empirical-Influence-Function/tools/annotation-viewer\n"
        "nohup python -m server.main --host 127.0.0.1 "
        f"--port {spec.viewer_port} --continue-data {spec.continue_data} "
        f"> annotation_{spec.viewer_port}.log 2>&1 &"
    )


def _gold_of(row: dict[str, Any]) -> tuple[str, str, str]:
    fim, gold = _fim_and_gold(row)
    if not gold:
        gold = str(row.get("middle") or "").strip()
    pred = str(row.get("predict") or row.get("prediction") or "").strip()
    return fim, gold, pred


def _train_cmd(spec: LangSpec, base_model: str, data_path: Path) -> list[str]:
    name = spec.output_name
    return [
        sys.executable,
        "src/train/train.py",
        "--model_name_or_path", base_model,
        "--data_path", str(data_path),
        "--output_dir", f"./outputs/{name}",
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
        "--run_name", name,
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
    spec: LangSpec,
    train_repo: Path,
    base_model: str,
    data_path: Path,
    state: dict[str, Any],
    state_path: Path,
    gpu: str,
) -> int:
    output_dir = train_repo / "outputs" / spec.output_name
    log_path = train_repo / f"{spec.output_name}.log"
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
            _train_cmd(spec, base_model, data_path),
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


def _require_files(spec: LangSpec) -> str | None:
    needed = {
        "predictions": Path(spec.predictions),
        "raw_train": Path(spec.raw_train),
        "semantic_corpus": Path(spec.semantic_corpus),
        "train_ids": Path(spec.train_ids),
    }
    missing = [f"{name}={path}" for name, path in needed.items() if not path.is_file()]
    if not missing:
        return None
    preprocess = (
        "python -m src.fim_semantic_preprocess "
        f"-i {spec.raw_train} -o {spec.semantic_corpus} --language {spec.language}"
    )
    return "missing files:\n  " + "\n  ".join(missing) + f"\nBuild the semantic corpus with:\n  {preprocess}"


def main(spec: LangSpec, argv: list[str] | None = None) -> int:
    _hydrate_eif_env()
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description=f"{spec.language} valid predictions → semantic top-10 → annotate → {spec.output_name}",
    )
    parser.add_argument("--input", default=spec.predictions)
    parser.add_argument("--continue-data", default=spec.continue_data)
    parser.add_argument("--raw-train", default=spec.raw_train)
    parser.add_argument("--semantic-corpus", default=spec.semantic_corpus)
    parser.add_argument("--train-ids", default=spec.train_ids)
    parser.add_argument("--mixed-out", default=spec.mixed_out)
    parser.add_argument("--train-repo", default=TRAIN_REPO)
    parser.add_argument("--base-model", default=os.environ.get("EIF_BASE_MODEL_PATH") or BASE_MODEL)
    parser.add_argument("--state", default=str(repo / spec.state_name))
    parser.add_argument("--eif-url", default="http://127.0.0.1:8766")
    parser.add_argument("--viewer-url", default=f"http://127.0.0.1:{spec.viewer_port}")
    parser.add_argument("--gpu", default=spec.gpu)
    parser.add_argument("--head-n", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--max-tests", type=int, default=0, help="0 = every row")
    parser.add_argument("--language", default=spec.language)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args(argv)

    run_spec = LangSpec(
        key=spec.key,
        language=str(args.language),
        predictions=str(args.input),
        raw_train=str(args.raw_train),
        semantic_corpus=str(args.semantic_corpus),
        train_ids=str(args.train_ids),
        continue_data=str(args.continue_data),
        mixed_out=str(args.mixed_out),
        output_name=spec.output_name,
        viewer_port=spec.viewer_port,
        gpu=str(args.gpu),
        state_name=spec.state_name,
    )
    problem = _require_files(run_spec)
    if problem:
        print(f"[fail] {problem}", flush=True)
        return 2

    input_path = Path(args.input)
    continue_path = Path(args.continue_data)
    raw_train = Path(args.raw_train)
    semantic_corpus = Path(args.semantic_corpus)
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
            "Pass --fresh. This does not delete the continue JSONL.",
            flush=True,
        )
        return 2

    try:
        health = _viewer_health(args.viewer_url)
    except Exception as exc:
        print(
            f"[fail] annotation viewer {args.viewer_url} is not up ({exc}).\n"
            + _viewer_start(run_spec),
            flush=True,
        )
        return 2
    actual = str(health.get("continue_path") or health.get("continuePath") or "")
    if actual and not _continue_paths_match(str(continue_path), actual):
        print(
            f"[fail] viewer continue file is {actual}, want {continue_path}.\n"
            "Do not use port 8765 (that file is the Go continue set).\n"
            + _viewer_start(run_spec),
            flush=True,
        )
        return 2

    rows = _load_raw_rows(input_path)
    if not _has_predict_field(rows):
        print(
            f"[fail] {input_path} has no predict field. "
            "Semantic attribution would describe the whole gold. "
            f"Use the {run_spec.key} prediction jsonl.",
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
            retrieve = client.llm_semantic_retrieve(
                fim,
                gold,
                language=lang or None,
                model_prediction=pred or None,
            )
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
        sr0 = (retrieve.get("search_results") or [{}])[0]
        corpus_error = sr0.get("corpus_error") if isinstance(sr0, dict) else None
        if corpus_error:
            print(f"[fail] semantic corpus: {corpus_error}", flush=True)
            return 2
        got_sem = str(retrieve.get("semantic_corpus_path") or "")
        if not _continue_paths_match(str(semantic_corpus), got_sem):
            print(
                f"[fail] 8766 searched {got_sem or '(empty)'}, want {semantic_corpus}.\n"
                "The shared 8766 must honor semanticCorpusPath. Restart it from this repo.",
                flush=True,
            )
            return 2
        mode = _query_mode(retrieve)
        if mode != "residual":
            print(
                f"[fail] 8766 query_mode={mode or '(empty)'} on a row whose predict differs from gold.\n"
                "Attribution would describe the whole gold. Restart 8766 from this repo.",
                flush=True,
            )
            return 2
        hits = _collect_semantic_hits(retrieve)[: max(1, int(args.top_k))]
        target_semantic = retrieve.get("semantic") if isinstance(retrieve.get("semantic"), dict) else None
        print(
            f"  mode={mode} semantic hits={len(hits)} role={(target_semantic or {}).get('role', '')[:80]!r}",
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
        run_spec,
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
    print(f"[done] {run_spec.output_name} training finished", flush=True)
    return 0
