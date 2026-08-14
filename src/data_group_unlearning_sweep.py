from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from statistics import median

import torch
from accelerate import Accelerator
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, set_seed

from src.NIF import CustomCollator, build_train_dataset
from src.attribution_evaluation import (
    _apply_lm_head_ascent_update,
    _reduce_data_token_effects,
    _restore_lm_head_ascent_update,
    _single_train_batch_from_dataset,
    _target_token_losses,
    build_test_batch,
    parse_float_tuple,
    parse_int_tuple,
)
from src.intervention_experiment import (
    SEED,
    _clear_cuda_after_oom,
    _is_cuda_alloc_error,
    load_model_and_tokenizer,
    load_samples,
    load_train_samples,
)
from src.loss import compute_lm_head_ce_gradient_no_backward
from src.process_data import process_func_chatml


@dataclass(frozen=True)
class SweepConfig:
    name: str
    lr: float
    aggregation: str
    normalize: bool


def _parse_indices(raw: str | None, start_idx: int, end_idx: int | None, total: int) -> list[int]:
    if raw:
        indices = [int(x) for x in raw.replace(",", " ").split() if x]
    else:
        stop = total - 1 if end_idx is None else int(end_idx)
        indices = list(range(int(start_idx), stop + 1))
    for idx in indices:
        if idx < 0 or idx >= total:
            raise ValueError(f"test index out of range: {idx} (valid: 0..{total - 1})")
    return indices


def _parse_bool(raw: str) -> bool:
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "y", "normalize", "norm"}:
        return True
    if value in {"0", "false", "no", "n", "raw"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {raw}")


def _parse_configs(raw: str) -> list[SweepConfig]:
    configs: list[SweepConfig] = []
    for spec in [x.strip() for x in raw.split(",") if x.strip()]:
        parts = spec.split(":")
        if len(parts) != 4:
            raise ValueError(
                "--configs entries must be name:lr:aggregation:normalize, "
                f"got {spec!r}"
            )
        name, lr_raw, aggregation, normalize_raw = parts
        aggregation = aggregation.strip().lower()
        if aggregation not in {"sum", "mean"}:
            raise ValueError(f"Unsupported aggregation {aggregation!r}; expected sum or mean.")
        configs.append(
            SweepConfig(
                name=name.strip(),
                lr=float(lr_raw),
                aggregation=aggregation,
                normalize=_parse_bool(normalize_raw),
            )
        )
    if not configs:
        raise ValueError("--configs must contain at least one config.")
    names = [cfg.name for cfg in configs]
    if len(names) != len(set(names)):
        raise ValueError("--configs names must be unique.")
    return configs


def _source_json_path(source_results_dir: Path, test_index: int) -> Path:
    data_dir = source_results_dir / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot find source data directory: {data_dir}. "
            "--source-results-dir must point to the data_standard results root."
        )

    direct = data_dir / f"test{test_index}_data.json"
    if direct.exists():
        return direct

    matches = sorted(data_dir.glob(f"*{test_index}*_data.json"))
    if len(matches) == 1:
        return matches[0]

    indexed_matches: list[Path] = []
    for path in sorted(data_dir.glob("*_data.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if _payload_matches_test_index(payload, test_index):
            indexed_matches.append(path)

    if len(indexed_matches) == 1:
        return indexed_matches[0]
    if len(indexed_matches) > 1:
        raise FileExistsError(
            f"Found multiple source data JSON files for test index {test_index}: "
            + ", ".join(str(p) for p in indexed_matches[:5])
        )
    raise FileNotFoundError(f"Cannot find source data JSON for test index {test_index}.")


def _payload_matches_test_index(payload: dict, test_index: int) -> bool:
    raw_index = payload.get("experiment_meta", {}).get("test_sample_index")
    if raw_index is None:
        return False
    try:
        return int(raw_index) == int(test_index)
    except (TypeError, ValueError):
        return False


def _coarse_json_path(source_results_dir: Path, test_index: int) -> Path:
    coarse_dir = source_results_dir / "data_coarse"
    if not coarse_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot find source coarse directory: {coarse_dir}. "
            "--source-results-dir must point to the data_standard results root."
        )

    direct = coarse_dir / f"test{test_index}_data_coarse.json"
    if direct.exists():
        return direct

    matches = sorted(coarse_dir.glob(f"*{test_index}*_data_coarse.json"))
    if len(matches) == 1:
        return matches[0]

    indexed_matches: list[Path] = []
    for path in sorted(coarse_dir.glob("*_data_coarse.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if _payload_matches_test_index(payload, test_index):
            indexed_matches.append(path)

    if len(indexed_matches) == 1:
        return indexed_matches[0]
    if len(indexed_matches) > 1:
        raise FileExistsError(
            f"Found multiple source coarse JSON files for test index {test_index}: "
            + ", ".join(str(p) for p in indexed_matches[:5])
        )
    raise FileNotFoundError(f"Cannot find source coarse JSON for test index {test_index}.")


def _coarse_payload_as_source_payload(coarse_payload: dict) -> dict:
    meta = coarse_payload.get("experiment_meta", {})
    return {
        "experiment_meta": meta,
        "test_sample_baseline": {"response_source": None},
        "data_coarse_attribution": {
            "granularity": "sample_to_sample",
            "sample_to_token": [],
            "sample_to_sample": {
                "target_token_indices": meta.get("target_token_indices", []),
                "target_tokens": meta.get("target_tokens", []),
                "method_top": [],
            },
        },
    }


def _load_method_ranked(source_results_dir: Path, test_index: int, max_k: int) -> tuple[list[int], dict]:
    payload = None
    try:
        source_path = _source_json_path(source_results_dir, test_index)
        payload = json.loads(source_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        pass

    if payload is not None:
        sample_to_sample = payload["data_coarse_attribution"]["sample_to_sample"]
        method_top = sample_to_sample.get("method_top") or []
    else:
        method_top = []

    if method_top:
        ranked = [int(row["train_sample_id"]) for row in method_top[:max_k]]
    else:
        coarse_path = _coarse_json_path(source_results_dir, test_index)
        coarse_payload = json.loads(coarse_path.read_text(encoding="utf-8"))
        if payload is None:
            payload = _coarse_payload_as_source_payload(coarse_payload)
        ranked = [
            int(row["train_sample_id"])
            for row in coarse_payload.get("coarse_ranking", [])[:max_k]
        ]
    if not ranked:
        raise ValueError(f"No method ranking found for test index {test_index}.")
    if payload is None:
        raise ValueError(f"No source payload found for test index {test_index}.")
    return ranked, payload


def _summary(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    sorted_values = sorted(float(x) for x in values)
    n = len(sorted_values)

    def percentile(q: float) -> float:
        if n == 1:
            return sorted_values[0]
        pos = (n - 1) * q
        lo = int(pos)
        hi = min(lo + 1, n - 1)
        weight = pos - lo
        return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight

    return {
        "n": n,
        "mean": round(sum(sorted_values) / n, 6),
        "median": round(float(median(sorted_values)), 6),
        "p25": round(percentile(0.25), 6),
        "p75": round(percentile(0.75), 6),
        "min": round(sorted_values[0], 6),
        "max": round(sorted_values[-1], 6),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_markdown(path: Path, payload: dict) -> None:
    thresholds = payload["summary"]["thresholds"]
    lines = [
        "# Data Group Unlearning Sweep",
        "",
        f"- source_results_dir: `{payload['summary']['source_results_dir']}`",
        f"- sample_count: {payload['summary']['sample_count']}",
        f"- group_k_values: {payload['summary']['group_k_values']}",
        f"- effect_reduction: `{payload['summary']['effect_reduction']}`",
        "",
        "| config | k | lr | aggregation | normalize | mean | median | positive | "
        + " | ".join(f"eff@{tau:g}" for tau in thresholds)
        + " |",
        "|---|---:|---:|---|---|---:|---:|---:"
        + "|---:" * len(thresholds)
        + "|",
    ]
    for row in payload["rows"]:
        eff = " | ".join(f"{row[f'effectiveness@{tau:g}']:.4f}" for tau in thresholds)
        lines.append(
            f"| {row['config']} | {row['k']} | {row['lr']:.6g} | {row['aggregation']} | "
            f"{row['normalize']} | {row['mean']:.6f} | {row['median']:.6f} | "
            f"{row['positive_rate']:.4f} | {eff} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _evaluate_one_sample(
    *,
    model,
    tokenizer,
    test_batch: dict,
    train_ds,
    collator,
    idx_to_row: dict[int, int],
    method_ranked: list[int],
    target_positions: list[int],
    group_k_values: tuple[int, ...],
    configs: list[SweepConfig],
    device,
    effect_reduction: str,
) -> dict:
    requested_ks = sorted({int(k) for k in group_k_values if int(k) > 0})
    max_k = min(max(requested_ks), len(method_ranked))
    requested_set = {k for k in requested_ks if k <= len(method_ranked)}
    base_losses = _target_token_losses(model, test_batch, target_positions, device)
    group_grad = None
    selected_ids: list[int] = []
    skipped_ids: list[int] = []
    records: list[dict] = []

    for rank, train_idx in enumerate(method_ranked[:max_k], start=1):
        train_idx = int(train_idx)
        train_batch = None
        grad = None
        try:
            train_batch = _single_train_batch_from_dataset(
                train_ds,
                collator,
                train_idx,
                idx_to_row,
                device,
            )
            grad = compute_lm_head_ce_gradient_no_backward(
                model=model,
                batch=train_batch,
                device=device,
                ignored_token_ids=torch.tensor([], device=device),
            )
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not _is_cuda_alloc_error(exc):
                raise
            skipped_ids.append(train_idx)
            _clear_cuda_after_oom()
            del train_batch, grad
            continue

        if group_grad is None:
            group_grad = grad.detach()
        else:
            group_grad.add_(grad)
        selected_ids.append(train_idx)
        del train_batch, grad

        if rank not in requested_set or group_grad is None or not selected_ids:
            continue

        for cfg in configs:
            update_grad = group_grad
            if cfg.aggregation == "mean":
                update_grad = group_grad / max(1, len(selected_ids))
            delta = None
            try:
                delta = _apply_lm_head_ascent_update(
                    model,
                    update_grad,
                    lr=cfg.lr,
                    normalize=cfg.normalize,
                )
                changed_losses = _target_token_losses(model, test_batch, target_positions, device)
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if not _is_cuda_alloc_error(exc):
                    raise
                _clear_cuda_after_oom()
                continue
            finally:
                if delta is not None:
                    _restore_lm_head_ascent_update(model, delta)

            token_effects = changed_losses - base_losses
            effect = _reduce_data_token_effects(token_effects, effect_reduction)
            records.append(
                {
                    "config": cfg.name,
                    "k": int(rank),
                    "train_sample_ids": [int(x) for x in selected_ids],
                    "train_sample_count": len(selected_ids),
                    "requested_train_sample_count": int(rank),
                    "skipped_train_sample_ids": [int(x) for x in skipped_ids],
                    "lr": float(cfg.lr),
                    "aggregation": cfg.aggregation,
                    "normalize": bool(cfg.normalize),
                    "base_ce_loss": float(base_losses.mean().item()),
                    "perturbed_ce_loss": float(changed_losses.mean().item()),
                    "data_effect": float(effect),
                    "token_effects": [float(x) for x in token_effects.tolist()],
                }
            )
            del changed_losses, token_effects
            torch.cuda.empty_cache()

    del group_grad, base_losses
    torch.cuda.empty_cache()
    return {"group_effects": records}


def _build_summary(
    *,
    args,
    configs: list[SweepConfig],
    group_k_values: tuple[int, ...],
    thresholds: tuple[float, ...],
    sample_records: list[dict],
) -> dict:
    rows: list[dict] = []
    by_key: dict[tuple[str, int], list[float]] = {}
    for sample in sample_records:
        for record in sample.get("group_effects", []):
            by_key.setdefault((record["config"], int(record["k"])), []).append(float(record["data_effect"]))

    config_by_name = {cfg.name: cfg for cfg in configs}
    for cfg in configs:
        for k in group_k_values:
            effects = by_key.get((cfg.name, int(k)), [])
            stats = _summary(effects)
            row = {
                "config": cfg.name,
                "k": int(k),
                "lr": float(cfg.lr),
                "aggregation": cfg.aggregation,
                "normalize": bool(cfg.normalize),
                "n": int(stats.get("n", 0)),
                "mean": float(stats.get("mean", 0.0)),
                "median": float(stats.get("median", 0.0)),
                "p25": float(stats.get("p25", 0.0)),
                "p75": float(stats.get("p75", 0.0)),
                "min": float(stats.get("min", 0.0)),
                "max": float(stats.get("max", 0.0)),
                "positive_rate": (
                    round(sum(1 for x in effects if x > 0.0) / len(effects), 6)
                    if effects
                    else 0.0
                ),
            }
            for tau in thresholds:
                row[f"effectiveness@{tau:g}"] = (
                    round(sum(1 for x in effects if x >= tau) / len(effects), 6)
                    if effects
                    else 0.0
                )
            rows.append(row)

    best_by_tau = {}
    for tau in thresholds:
        key = f"effectiveness@{tau:g}"
        best_by_tau[key] = max(rows, key=lambda r: (r[key], r["mean"], -r["k"]))

    return {
        "summary": {
            "source_results_dir": str(args.source_results_dir),
            "output_dir": str(args.output_dir),
            "sample_count": len(sample_records),
            "group_k_values": [int(k) for k in group_k_values],
            "thresholds": [float(x) for x in thresholds],
            "effect_reduction": args.data_effect_reduction,
            "configs": [
                {
                    "name": cfg.name,
                    "lr": float(cfg.lr),
                    "aggregation": cfg.aggregation,
                    "normalize": bool(cfg.normalize),
                }
                for cfg in configs
            ],
        },
        "rows": rows,
        "best_by_tau": best_by_tau,
        "samples": sample_records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep data group unlearning updates using existing coarse rankings."
    )
    parser.add_argument("--source-results-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("attribution_results_data_group_sweep"))
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--train-data", type=str, default="sft_train.jsonl")
    parser.add_argument("--test-data", type=str, default="sft_test.jsonl")
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=20)
    parser.add_argument("--generation-limit", type=int, default=32)
    parser.add_argument("--use-ground-truth-response", action="store_true")
    parser.add_argument("--data-group-k-values", type=str, default="1,3,5,10")
    parser.add_argument(
        "--configs",
        type=str,
        default=(
            "norm0.01:0.01:sum:1,"
            "norm0.03:0.03:sum:1,"
            "norm0.1:0.1:sum:1,"
            "rawmean0.01:0.01:mean:0,"
            "rawsum0.003:0.003:sum:0,"
            "rawsum0.01:0.01:sum:0"
        ),
        help="Comma-separated entries of name:lr:aggregation:normalize.",
    )
    parser.add_argument("--thresholds", type=str, default="0.0005,0.001,0.002,0.005")
    parser.add_argument("--data-effect-reduction", choices=["mean", "sum", "max"], default="mean")
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--max-gpu-memory", type=str, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError("Sweep expects a single Accelerator process.")

    configs = _parse_configs(args.configs)
    group_k_values = parse_int_tuple(args.data_group_k_values)
    if not group_k_values:
        raise ValueError("--data-group-k-values must not be empty.")
    thresholds = parse_float_tuple(args.thresholds)
    if not thresholds:
        raise ValueError("--thresholds must not be empty.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "samples").mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        attn_implementation=args.attn_implementation,
        max_gpu_memory=args.max_gpu_memory,
    )
    convert_to_chatml = partial(process_func_chatml, tokenizer=tokenizer)
    train_samples = load_train_samples(args.train_data)
    test_samples = load_samples(args.test_data)
    if args.train_limit is not None:
        train_samples = train_samples[: int(args.train_limit)]
    test_indices = _parse_indices(args.indices, args.start_idx, args.end_idx, len(test_samples))

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    collator = CustomCollator(base_collator)
    train_ds = build_train_dataset(train_samples)
    idx_to_row = {int(train_ds[i]["sample_index"]): i for i in range(len(train_ds))}

    sample_records: list[dict] = []
    max_group_k = max(int(k) for k in group_k_values)
    for test_index in tqdm(test_indices, desc="Data group sweep"):
        source_ranked, source_payload = _load_method_ranked(
            args.source_results_dir,
            int(test_index),
            max_group_k,
        )
        sample = test_samples[int(test_index)]
        task_id = sample.get("task_id") or f"test{test_index}"
        test_batch, prompt_len, test_meta = build_test_batch(
            model,
            tokenizer,
            sample,
            convert_to_chatml,
            base_collator,
            accelerator,
            use_generated_response=not args.use_ground_truth_response,
            generation_limit=int(args.generation_limit),
        )
        saved_positions = (
            source_payload["data_coarse_attribution"]["sample_to_sample"]
            .get("target_token_indices", [])
        )
        target_positions = [
            int(pos)
            for pos in saved_positions
            if 0 < int(pos) < int(test_batch["input_ids"].size(1))
        ][: int(args.max_output_tokens)]
        if not target_positions:
            raise ValueError(f"No valid target positions for test index {test_index}.")

        result = _evaluate_one_sample(
            model=model,
            tokenizer=tokenizer,
            test_batch=test_batch,
            train_ds=train_ds,
            collator=collator,
            idx_to_row=idx_to_row,
            method_ranked=source_ranked,
            target_positions=target_positions,
            group_k_values=group_k_values,
            configs=configs,
            device=accelerator.device,
            effect_reduction=args.data_effect_reduction,
        )
        record = {
            "test_index": int(test_index),
            "task_id": task_id,
            "prompt_len": int(prompt_len),
            "target_token_indices": [int(x) for x in target_positions],
            "source_response": source_payload.get("test_sample_baseline", {}).get("response_source"),
            "current_response": test_meta.get("response_source"),
            "method_ranked_top": [int(x) for x in source_ranked],
            **result,
        }
        sample_records.append(record)
        _write_json(args.output_dir / "samples" / f"{task_id}_sweep.json", record)

    summary = _build_summary(
        args=args,
        configs=configs,
        group_k_values=group_k_values,
        thresholds=thresholds,
        sample_records=sample_records,
    )
    _write_json(args.output_dir / "summary.json", summary)
    _write_markdown(args.output_dir / "summary.md", summary)
    print(f"[data-group-sweep] wrote {args.output_dir / 'summary.json'}", flush=True)
    print(f"[data-group-sweep] wrote {args.output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
