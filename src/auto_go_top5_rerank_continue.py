#!/usr/bin/env python3
"""100 条错测：语义召回 Top-10 → 补全结构重排 → 标注前 5 → 续训集加倍后接上训练集并开训。

对 valid_imperfect_less.jsonl 每一条：

  1. 现有验证端残差语义，全库 Stage-2 召回前 10。
  2. 用 gold 补全和命中补全的结构相似度 + API 名重合，叠上语义分，在这 10 条里重排。
  3. 前 5 条做带 target_semantic 的 LLM 语义标注，写入续训 JSONL。
  4. 100 条都结束后，把本轮新写入的续训行再复制一份（500 → 1000），
     追加到 csn_go_train_fim_10k_ids.jsonl 末尾。
  5. 在训练仓库里 nohup 启动 SAL_go 的 CE+saliency。

标注后端和 ttav 后端需要已经开着，且 annotation-viewer 的 continue 文件
就是 ANNOTATION_CONTINUE_TRAIN_DATA。

Example::

    python -m src.auto_go_top5_rerank_continue --max-tests 1 --skip-train
    python -m src.auto_go_top5_rerank_continue
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from src.auto_go_semantic_verify_continue import _annotate_hit
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
DEFAULT_CONTINUE = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/"
    "new_continue_annotated_subset.jsonl"
)
DEFAULT_TRAIN_IDS = (
    "/mnt/md124/jiaxin/training_code/data/csn_go_train_fim_10k_ids.jsonl"
)
DEFAULT_TRAIN_REPO = "/mnt/md124/jiaxin/training_code/Empirical-Influence-Function"
DEFAULT_BASE_MODEL = "/mnt/md124/jiaxin/models/Qwen3-8B"
DEFAULT_STATE = REPO_ROOT / "go_top5_rerank_continue_state.json"

_STOP = {
    "if", "else", "for", "return", "var", "func", "nil", "err", "true", "false",
    "range", "defer", "go", "struct", "map", "chan", "string", "int", "error",
    "default", "make", "new", "len", "append", "cap",
}


def _env(name: str) -> str:
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


def _load_completions(path: Path) -> dict[int, str]:
    """0-based source line → completion text from the raw train jsonl."""
    out: dict[int, str] = {}
    if not path.is_file():
        print(f"[warn] train corpus missing, MID text falls back to preview: {path}", flush=True)
        return out
    with path.open(encoding="utf-8") as handle:
        for i, raw in enumerate(handle):
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            text = str(
                row.get("completion") or row.get("response") or row.get("label") or ""
            )
            out[i] = text
    return out


def _strip_code(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//.*?$", " ", text, flags=re.M)
    text = re.sub(r"`[^`]*`", " STR ", text)
    text = re.sub(r"\"(?:\\.|[^\"\\])*\"", " STR ", text)
    return text


def _opcodes(text: str) -> list[str]:
    """Coarse statement shape. Order matters; names do not."""
    ops: list[str] = []
    for raw in _strip_code(text).splitlines():
        line = raw.strip()
        if not line or line in {"{", "}"}:
            continue
        low = line.lower()
        if low.startswith("defer "):
            ops.append("defer")
        elif low.startswith("if ") or low.startswith("if("):
            ops.append("if")
        elif low.startswith("for ") or low.startswith("for("):
            ops.append("for")
        elif low.startswith("return"):
            ops.append("return")
        elif "delete(" in low:
            ops.append("delete")
        elif re.search(r"\.\s*lock\s*\(", low):
            ops.append("lock")
        elif re.search(r"\.\s*unlock\s*\(", low):
            ops.append("unlock")
        elif ":=" in line or re.search(r"[^=]=[^=]", line):
            ops.append("assign")
        elif "(" in line:
            ops.append("call")
        else:
            ops.append("stmt")
    return ops


def _api_tokens(text: str) -> set[str]:
    found: set[str] = set()
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", _strip_code(text)):
        low = tok.lower()
        if low in _STOP or len(tok) <= 1:
            continue
        if tok[:1].isupper() or "_" in tok or len(tok) >= 6:
            found.add(low)
    return found


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _struct_sim(gold: str, other: str) -> float:
    ga, oa = _opcodes(gold), _opcodes(other)
    if not ga or not oa:
        return 0.0
    return float(SequenceMatcher(a=ga, b=oa).ratio())


def _rerank(gold: str, hits: list[dict[str, Any]], completions: dict[int, str]) -> list[dict[str, Any]]:
    """Semantic stays in the mix. Structure of the two MIDs is the larger extra term."""
    ranked: list[tuple[float, dict[str, Any]]] = []
    for hit in hits:
        try:
            line = int(hit.get("line"))
        except (TypeError, ValueError):
            line = -1
        mid = completions.get(line) or str(hit.get("response_preview") or "")
        struct = _struct_sim(gold, mid)
        api = _jaccard(_api_tokens(gold), _api_tokens(mid))
        sem = float(hit.get("semantic_score") or 0.0)
        score = 0.45 * struct + 0.25 * api + 0.30 * sem
        row = dict(hit)
        row["struct_sim"] = round(struct, 3)
        row["api_sim"] = round(api, 3)
        row["rerank_score"] = round(score, 3)
        row["mid_preview"] = mid[:240]
        ranked.append((score, row))
    ranked.sort(key=lambda item: -item[0])
    return [row for _score, row in ranked]


def _count_lines(path: Path) -> int:
    if not path.is_file():
        return 0
    n = 0
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                n += 1
    return n


def _read_lines(path: Path) -> list[str]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return [raw if raw.endswith("\n") else raw + "\n" for raw in handle if raw.strip()]


def _duplicate_and_append(
    continue_path: Path,
    train_ids: Path,
    *,
    start_lines: int,
) -> int:
    """Copy rows added this run once more, then append both copies onto the train ids file."""
    rows = _read_lines(continue_path)
    fresh = rows[start_lines:]
    if not fresh:
        print("[warn] no new continue rows to duplicate", flush=True)
        return 0
    with continue_path.open("a", encoding="utf-8") as handle:
        handle.writelines(fresh)
    train_ids.parent.mkdir(parents=True, exist_ok=True)
    with train_ids.open("a", encoding="utf-8") as handle:
        handle.writelines(fresh)
        handle.writelines(fresh)
    print(
        f"[data] continue {start_lines} + {len(fresh)} new, duplicated to {len(fresh) * 2}; "
        f"appended {len(fresh) * 2} rows onto {train_ids}",
        flush=True,
    )
    return len(fresh) * 2


def _launch_train(train_repo: Path, base_model: str, data_path: Path) -> None:
    log_path = train_repo / "SAL_go.log"
    cmd = [
        sys.executable,
        "src/train/train.py",
        "--model_name_or_path", base_model,
        "--data_path", str(data_path),
        "--output_dir", "./outputs/SAL_go",
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
        "--run_name", "SAL_go",
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


def _save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_state(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"processed_lines": [], "results": [], "finalized": False}
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"processed_lines": [], "results": [], "finalized": False}
    if not isinstance(obj, dict):
        return {"processed_lines": [], "results": [], "finalized": False}
    obj.setdefault("processed_lines", [])
    obj.setdefault("results", [])
    obj.setdefault("finalized", False)
    return obj


def main(argv: list[str] | None = None) -> int:
    _hydrate_eif_env()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", default=DEFAULT_INPUT)
    p.add_argument("--continue-data", default=_env("ANNOTATION_CONTINUE_TRAIN_DATA") or DEFAULT_CONTINUE)
    p.add_argument("--train-ids", default=_env("EIF_TRAIN_DATA") or DEFAULT_TRAIN_IDS)
    p.add_argument("--train-corpus", default=_env("EIF_LLM_TRAIN_CORPUS"))
    p.add_argument("--train-repo", default=DEFAULT_TRAIN_REPO)
    p.add_argument("--base-model", default=_env("EIF_BASE_MODEL_PATH") or DEFAULT_BASE_MODEL)
    p.add_argument("--state", default=str(DEFAULT_STATE))
    p.add_argument("--eif-url", default="http://127.0.0.1:8766")
    p.add_argument("--viewer-url", default="http://127.0.0.1:8765")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--keep", type=int, default=5)
    p.add_argument("--max-tests", type=int, default=0)
    p.add_argument("--language", default="go")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--fresh", action="store_true")
    args = p.parse_args(argv)

    input_path = Path(args.input)
    continue_path = Path(args.continue_data)
    train_ids = Path(args.train_ids)
    state_path = Path(args.state)
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
            f"[fail] viewer continue file is {actual}, want {continue_path}. "
            "Restart annotation-viewer with --continue-data pointing at that file.",
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
    for test_line, row in rows:
        if int(test_line) in done:
            continue
        if args.max_tests and n_ok >= args.max_tests:
            break
        fim, gold, pred = _prompt_gold_pred(row)
        task_id = str(row.get("task_id") or f"row_{test_line}")
        rec: dict[str, Any] = {"test_line": test_line, "task_id": task_id, "kept": []}
        print(f"\n=== line={test_line} task={task_id} ===", flush=True)
        if not fim or not gold or not pred:
            rec["reason"] = "missing_prompt_gold_or_predict"
            state["results"].append(rec)
            state["processed_lines"].append(int(test_line))
            _save_state(state_path, state)
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
            continue
        hits = _collect_semantic_hits(retrieve)[: args.top_k]
        if not hits:
            rec["reason"] = "no_hits"
            state["results"].append(rec)
            state["processed_lines"].append(int(test_line))
            _save_state(state_path, state)
            continue
        ranked = _rerank(gold, hits, completions)
        sem = retrieve.get("semantic") if isinstance(retrieve.get("semantic"), dict) else None
        expression = str(retrieve.get("semantic_flat_text") or "")
        hit_corpus = str(retrieve.get("corpus_path") or "").strip() or client.corpus_path
        for hit in ranked[: args.keep]:
            print(
                f"  keep line={hit.get('line')} rerank={hit.get('rerank_score')} "
                f"struct={hit.get('struct_sim')} api={hit.get('api_sim')} "
                f"sem={hit.get('semantic_score')}",
                flush=True,
            )
            row_ann, note = _annotate_hit(
                client,
                gold=gold,
                language=args.language,
                hit=hit,
                expression=expression,
                hit_corpus=hit_corpus,
                target_semantic=sem,
                scratch_path=continue_path,
            )
            rec["kept"].append({
                "line": hit.get("line"),
                "rerank_score": hit.get("rerank_score"),
                "note": note,
                "ok": row_ann is not None,
            })
            if row_ann is None:
                print(f"    [skip] {note}", flush=True)
        state["results"].append(rec)
        state["processed_lines"].append(int(test_line))
        _save_state(state_path, state)
        n_ok += 1
        time.sleep(0.2)

    processed = {int(x) for x in state.get("processed_lines") or []}
    if any(int(line) not in processed for line, _row in rows):
        print("[stop] input not finished; not duplicating or training", flush=True)
        return 0
    if state.get("finalized"):
        print("[done] already finalized", flush=True)
        return 0
    _duplicate_and_append(
        continue_path,
        train_ids,
        start_lines=int(state.get("continue_start_lines") or 0),
    )
    state["finalized"] = True
    _save_state(state_path, state)
    if args.skip_train:
        print("[done] skip-train set", flush=True)
        return 0
    _launch_train(Path(args.train_repo), args.base_model, train_ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
