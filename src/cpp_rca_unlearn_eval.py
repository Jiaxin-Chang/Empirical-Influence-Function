"""Batch root-cause eval: first predict/label mismatch → predict saliency → Stage3 → unlearn.

For each JSONL row (``prompt`` + ``label``/``response`` + ``predict``):

  1. Tokenize with the same ChatML eval template as continue-train generate.
  2. Find the first completion token where predict ≠ gold.
  3. Last-layer ALTI saliency on that predict token (``mode=predict``).
  4. Take the top source, skipping the first ``sink_prefix`` sequence tokens
     (attention sink). Use the first remaining context token with index < target.
  5. Stage2/3 retrieve the most related train saliency pair.
  6. One-step LoRA unlearn on that pair (weights restored afterwards).
  7. ``verdict == supports_causal`` counts as a correct root-cause call.

Precision = n_supports_causal / n_evaluated.

Does **not** use LLM boolean retrieval. Reads paths from repo-root ``eif_api.env``.

Example::

  python -m src.cpp_rca_unlearn_eval \\
    --predictions /mnt/md124/jiaxin/Empirical-Influence-Function/qwen3-8b-cpp-ce.predictions.jsonl \\
    --out outputs/cpp_rca_unlearn_eval.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from src.continue_train_eval import _render_eval_prompt, remove_leading_thinking
from src.gold_live_attribution import (
    REPO_ROOT,
    _ensure_session,
    _hydrate_eif_env,
    gold_retrieve_and_stage3,
    gold_saliency_top_k,
)
from src.unlearn_pair_probe import recover_pair_intervention, run_unlearn_pair_probe


DEFAULT_SINK_PREFIX = 3
SUCCESS_VERDICT = "supports_causal"


def _default_predictions_path() -> str:
    _hydrate_eif_env()
    return (
        (os.environ.get("EIF_CONTINUE_EVAL_BEFORE_CACHE") or "").strip()
        or str(REPO_ROOT / "correlation_matching_results" / "raw_ce" / "qwen3-8b-cpp-ce.predictions.jsonl")
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                continue
            obj["_source_line"] = line_no
            rows.append(obj)
    return rows


def _row_texts(row: dict[str, Any]) -> tuple[str, str, str]:
    prompt = str(row.get("prompt") or row.get("input") or "")
    gold = str(row.get("label") or row.get("response") or row.get("gold") or "")
    pred = str(row.get("predict") or row.get("prediction") or row.get("output") or "")
    gold, _ = remove_leading_thinking(gold)
    pred, _ = remove_leading_thinking(pred)
    return prompt, gold, pred


def _ids_to_tokens(tokenizer, ids: list[int]) -> list[str]:
    return [tokenizer.decode([int(i)], skip_special_tokens=False) for i in ids]


def tokenize_predict_and_gold(
    tokenizer,
    prompt: str,
    predict: str,
    gold: str,
) -> dict[str, Any]:
    rendered = _render_eval_prompt(tokenizer, prompt)
    prompt_ids = [int(x) for x in tokenizer.encode(rendered, add_special_tokens=False)]
    pred_ans = [int(x) for x in tokenizer.encode(predict, add_special_tokens=False)]
    gold_ans = [int(x) for x in tokenizer.encode(gold, add_special_tokens=False)]
    pred_ids = prompt_ids + pred_ans
    gold_ids = prompt_ids + gold_ans
    prompt_len = len(prompt_ids)
    return {
        "prompt_len": prompt_len,
        "pred_ids": pred_ids,
        "gold_ids": gold_ids,
        "pred_tokens": _ids_to_tokens(tokenizer, pred_ids),
        "gold_tokens": _ids_to_tokens(tokenizer, gold_ids),
        "rendered_prompt": rendered,
    }


def first_mismatch_target(prompt_len: int, pred_ids: list[int], gold_ids: list[int]) -> dict[str, Any]:
    """First completion index where predict token id ≠ gold token id.

    If predict is a proper prefix of gold, there is no predict token to attribute
    (the model stopped too early) → skip. If gold is a proper prefix of predict,
    the first extra predict token is the mismatch.
    """
    pred_ans = pred_ids[prompt_len:]
    gold_ans = gold_ids[prompt_len:]
    n = min(len(pred_ans), len(gold_ans))
    for i in range(n):
        if int(pred_ans[i]) != int(gold_ans[i]):
            return {
                "status": "mismatch",
                "target_index": prompt_len + i,
                "pred_token_id": int(pred_ans[i]),
                "gold_token_id": int(gold_ans[i]),
            }
    if len(pred_ans) == len(gold_ans):
        return {"status": "exact_match"}
    if len(pred_ans) > len(gold_ans):
        return {
            "status": "mismatch",
            "target_index": prompt_len + n,
            "pred_token_id": int(pred_ans[n]),
            "gold_token_id": None,
            "note": "predict longer than gold; attributing first extra token",
        }
    return {
        "status": "predict_prefix_of_gold",
        "note": "predict is a prefix of gold; no extra predict token to attribute",
    }


def pick_nonsink_source(
    top: list[dict[str, Any]],
    target_index: int,
    *,
    sink_prefix: int,
) -> dict[str, Any] | None:
    """First ranked source that is not in [0, sink_prefix) and is strictly before target."""
    sink_prefix = max(0, int(sink_prefix))
    for item in top:
        try:
            idx = int(item.get("source_token_index"))
        except (TypeError, ValueError):
            continue
        if idx < sink_prefix:
            continue
        if not (0 <= idx < int(target_index)):
            continue
        return item
    return None


def _minimal_report(
    *,
    task_id: str,
    pred_tokens: list[str],
    pred_ids: list[int],
    gold_tokens: list[str],
    gold_ids: list[int],
    prompt_len: int,
    predictions_path: str,
) -> dict[str, Any]:
    name = Path(predictions_path).name
    return {
        "experiment_meta": {
            "task_id": task_id,
            "fileName": f"raw_ce/{name}",
            "report_file": f"raw_ce/{name}",
            "report_family": "ce",
            "adapter_family": "ce",
            "raw_eval": True,
            "model_name": "qwen3-8b-cpp-ce",
        },
        "test_sample_baseline": {
            "full_tokens": pred_tokens,
            "full_token_ids": pred_ids,
            "prompt_len": int(prompt_len),
            "correct_full_tokens": gold_tokens,
            "correct_full_token_ids": gold_ids,
        },
    }


def _public_train_detail(detail: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(detail, dict):
        return None
    return {k: v for k, v in detail.items() if not str(k).startswith("_")}


def _load_done_task_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        tid = str(obj.get("task_id") or "").strip()
        if tid:
            done.add(tid)
    return done


def evaluate_one(
    row: dict[str, Any],
    *,
    tokenizer,
    predictions_path: str,
    sink_prefix: int,
    top_saliency: int,
    top_trains: int,
    unlearn_lr: float | None,
) -> dict[str, Any]:
    task_id = str(row.get("task_id") or f"line_{row.get('_source_line')}")
    prompt, gold, pred = _row_texts(row)
    base: dict[str, Any] = {
        "task_id": task_id,
        "source_line": row.get("_source_line") or row.get("source_line"),
        "correct": False,
    }
    if not prompt.strip() or not pred.strip():
        return {**base, "status": "skip_empty", "reason": "missing prompt or predict"}

    tok = tokenize_predict_and_gold(tokenizer, prompt, pred, gold)
    stored_pt = row.get("prompt_tokens")
    if stored_pt is not None:
        try:
            stored_n = int(stored_pt)
            if abs(stored_n - int(tok["prompt_len"])) > 8:
                base["prompt_len_warn"] = {
                    "jsonl": stored_n,
                    "tokenized": int(tok["prompt_len"]),
                }
        except (TypeError, ValueError):
            pass

    mm = first_mismatch_target(tok["prompt_len"], tok["pred_ids"], tok["gold_ids"])
    base["mismatch"] = mm
    base["prompt_len"] = tok["prompt_len"]
    if mm["status"] != "mismatch":
        return {**base, "status": mm["status"]}

    target_index = int(mm["target_index"])
    if target_index <= 0 or target_index >= len(tok["pred_ids"]):
        return {**base, "status": "skip_bad_target", "reason": f"target_index={target_index}"}

    report = _minimal_report(
        task_id=task_id,
        pred_tokens=tok["pred_tokens"],
        pred_ids=tok["pred_ids"],
        gold_tokens=tok["gold_tokens"],
        gold_ids=tok["gold_ids"],
        prompt_len=tok["prompt_len"],
        predictions_path=predictions_path,
    )

    sal = gold_saliency_top_k(
        report,
        target_index=target_index,
        top_k=max(8, int(top_saliency)),
        mode="predict",
    )
    top = list(sal.get("topCorrelations") or [])
    source = pick_nonsink_source(top, target_index, sink_prefix=sink_prefix)
    skipped_sink = [
        {
            "source_token_index": c.get("source_token_index"),
            "source_token": c.get("source_token"),
            "saliency_score": c.get("saliency_score"),
        }
        for c in top
        if int(c.get("source_token_index") or -1) < int(sink_prefix)
    ][:3]
    base["saliency"] = {
        "target_index": target_index,
        "target_token": sal.get("targetToken"),
        "picked": source,
        "skipped_sink_prefix": skipped_sink,
        "n_ranked": len(top),
    }
    if source is None:
        return {**base, "status": "skip_no_nonsink_source"}

    src_idx = int(source["source_token_index"])
    retrieved = gold_retrieve_and_stage3(
        report,
        source_index=src_idx,
        target_index=target_index,
        top_trains=int(top_trains),
        mode="predict",
    )
    pairs = list(retrieved.get("correlationPairs") or [])
    details = retrieved.get("trainSampleDetails") or {}
    if not pairs:
        return {**base, "status": "skip_no_train_pair", "relatedTrains": retrieved.get("relatedTrains")}

    pair = pairs[0]
    tr = pair.get("train_correlation") or {}
    train_id = int(pair["train_sample_id"])
    train_src = int(tr["source_token_index"])
    train_tgt = int(tr["target_token_index"])
    detail = _public_train_detail(details.get(str(train_id)))
    base["train_pair"] = {
        "train_sample_id": train_id,
        "cos_sim": pair.get("cos_sim"),
        "train_source_index": train_src,
        "train_target_index": train_tgt,
        "train_source_token": tr.get("source_token"),
        "train_target_token": tr.get("target_token"),
        "relatedTrains": retrieved.get("relatedTrains"),
    }

    unlearn = run_unlearn_pair_probe(
        report,
        train_sample_id=train_id,
        test_source_index=src_idx,
        test_target_index=target_index,
        train_source_index=train_src,
        train_target_index=train_tgt,
        pair_id=str(pair.get("id") or f"rca_{task_id}"),
        direction="unlearn",
        persist=False,
        completion_mode="predict",
        train_sample_detail=detail,
        recompute_saliency=True,
        unlearn_lr=unlearn_lr,
    )
    try:
        recover_pair_intervention()
    except Exception as exc:
        print(f"[rca] recover after unlearn: {exc}", flush=True)

    verdict = str(unlearn.get("verdict") or "")
    correct = verdict == SUCCESS_VERDICT
    return {
        **base,
        "status": "evaluated",
        "correct": correct,
        "verdict": verdict,
        "unlearn": {
            "before": unlearn.get("before"),
            "after": unlearn.get("after"),
            "delta": unlearn.get("delta"),
            "direction": unlearn.get("direction"),
        },
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    n_total = len(records)
    by_status: dict[str, int] = {}
    n_eval = 0
    n_ok = 0
    by_verdict: dict[str, int] = {}
    for rec in records:
        st = str(rec.get("status") or "unknown")
        by_status[st] = by_status.get(st, 0) + 1
        if st == "evaluated":
            n_eval += 1
            v = str(rec.get("verdict") or "")
            by_verdict[v] = by_verdict.get(v, 0) + 1
            if rec.get("correct"):
                n_ok += 1
    precision = (n_ok / n_eval) if n_eval else None
    return {
        "n_total": n_total,
        "n_evaluated": n_eval,
        "n_supports_causal": n_ok,
        "precision": precision,
        "precision_pct": None if precision is None else round(100.0 * precision, 2),
        "by_status": by_status,
        "by_verdict": by_verdict,
        "success_verdict": SUCCESS_VERDICT,
        "note": (
            "precision = n(verdict==supports_causal) / n(evaluated). "
            "exact_match / skip_* rows are excluded from the denominator."
        ),
    }


def main() -> None:
    _hydrate_eif_env()
    parser = argparse.ArgumentParser(description="C++ predict-mismatch RCA via saliency + unlearn")
    parser.add_argument(
        "--predictions",
        default=_default_predictions_path(),
        help="JSONL with prompt/label/predict (default: EIF_CONTINUE_EVAL_BEFORE_CACHE)",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "outputs" / "cpp_rca_unlearn_eval.json"),
        help="Summary JSON path; sidecar .jsonl stores per-sample rows",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows after offset")
    parser.add_argument("--sink-prefix", type=int, default=DEFAULT_SINK_PREFIX)
    parser.add_argument("--top-saliency", type=int, default=0, help="0 → EIF_GOLD_TOP_SALIENCY")
    parser.add_argument("--top-trains", type=int, default=0, help="0 → EIF_GOLD_TOP_TRAINS")
    parser.add_argument("--unlearn-lr", type=float, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip task_ids already in the sidecar jsonl")
    args = parser.parse_args()

    pred_path = Path(args.predictions).expanduser()
    if not pred_path.is_file():
        raise FileNotFoundError(f"predictions JSONL not found: {pred_path}")
    out_path = Path(args.out).expanduser()
    if not out_path.is_absolute():
        out_path = (REPO_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out_path.with_suffix(".jsonl")

    rows = _load_jsonl(pred_path)
    offset = max(0, int(args.offset))
    limit = int(args.limit)
    rows = rows[offset:]
    if limit > 0:
        rows = rows[:limit]

    done = _load_done_task_ids(sidecar) if args.resume else set()
    top_sal = int(args.top_saliency) or max(6, int(os.environ.get("EIF_GOLD_TOP_SALIENCY") or 6))
    top_tr = int(args.top_trains) or max(5, int(os.environ.get("EIF_GOLD_TOP_TRAINS") or 5))
    sink = max(0, int(args.sink_prefix))

    print(
        f"[rca] predictions={pred_path} n={len(rows)} sink_prefix={sink} "
        f"top_sal={top_sal} top_trains={top_tr} sidecar={sidecar}",
        flush=True,
    )

    tokenizer = None
    records: list[dict[str, Any]] = []
    if args.resume and sidecar.is_file():
        for line in sidecar.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

    n_run = 0
    started = time.time()
    with sidecar.open("a" if args.resume else "w", encoding="utf-8") as out_fh:
        for row in rows:
            task_id = str(row.get("task_id") or f"line_{row.get('_source_line')}")
            if task_id in done:
                continue
            t0 = time.time()
            print(f"[rca] start {task_id}", flush=True)
            try:
                if tokenizer is None:
                    dummy = _minimal_report(
                        task_id=task_id,
                        pred_tokens=["a", "b", "c", "d"],
                        pred_ids=[1, 2, 3, 4],
                        gold_tokens=["a", "b", "c", "d"],
                        gold_ids=[1, 2, 3, 4],
                        prompt_len=2,
                        predictions_path=str(pred_path),
                    )
                    tokenizer = _ensure_session(dummy)["tokenizer"]
                rec = evaluate_one(
                    row,
                    tokenizer=tokenizer,
                    predictions_path=str(pred_path),
                    sink_prefix=sink,
                    top_saliency=top_sal,
                    top_trains=top_tr,
                    unlearn_lr=args.unlearn_lr,
                )
            except Exception as exc:
                rec = {
                    "task_id": task_id,
                    "status": "error",
                    "correct": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc()[-2000:],
                    "source_line": row.get("_source_line"),
                }
                print(f"[rca] ERROR {task_id}: {exc}", flush=True)
                try:
                    recover_pair_intervention()
                except Exception:
                    pass

            rec["elapsed_sec"] = round(time.time() - t0, 2)
            records.append(rec)
            out_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_fh.flush()
            n_run += 1
            print(
                f"[rca] done {task_id} status={rec.get('status')} "
                f"verdict={rec.get('verdict')} correct={rec.get('correct')} "
                f"in {rec['elapsed_sec']}s",
                flush=True,
            )

    summary = summarize(records)
    summary.update({
        "predictions": str(pred_path),
        "sidecar": str(sidecar),
        "sink_prefix": sink,
        "n_this_run": n_run,
        "elapsed_sec": round(time.time() - started, 1),
        "adapter_hint": "EIF_ADAPTER_PATH_CE + EIF_TRAIN_DATA from eif_api.env",
    })
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["precision"] is not None:
        print(
            f"[rca] 根因分析查准率 = {summary['precision_pct']}% "
            f"({summary['n_supports_causal']}/{summary['n_evaluated']})",
            flush=True,
        )
    else:
        print("[rca] no evaluated samples (nothing to score)", flush=True)


if __name__ == "__main__":
    main()
