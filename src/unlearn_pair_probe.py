"""One-step unlearning probe for a single train↔test saliency correlation pair.

Uses the same model loading path as ``intervention_experiment`` (full checkpoint
OR base + LoRA adapter). The update itself mirrors the data-attribution oracle:

    θ ← θ + η · normalize(∇_{lm_head} L_train)

then measures how the *test* target token's CE / log-prob and ALTI saliency
change. Weights are restored afterwards — this is a probe, not a permanent edit
of the adapter on disk.

The train loss is restricted to the pair's train *target* token (other labels
masked to -100), so the ascent is aimed at that saliency edge rather than the
whole sample.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from src.attribution_evaluation import (
    _apply_lm_head_ascent_update,
    _restore_lm_head_ascent_update,
    _target_token_losses,
)
from src.export_real_ttav_bundle import (
    convert_report_tokens_to_ids,
    validate_roundtrip_tokens,
)
from src.intervention_experiment import load_model_and_tokenizer
from src.loss import compute_alti_saliency_vector, compute_lm_head_ce_gradient_no_backward


# Reuse one loaded model across Unlearn clicks in the same API process.
_MODEL_CACHE: dict[tuple[str, str], tuple[object, object]] = {}


def _device_of(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_paths(
    report: dict[str, Any],
    model_path: str | None,
    base_model_path: str | None,
) -> tuple[str, str | None]:
    meta = report.get("experiment_meta") or {}
    resolved_model = (
        str(model_path or "").strip()
        or str(os.environ.get("EIF_ADAPTER_PATH") or os.environ.get("EIF_MODEL_PATH") or "").strip()
        or str(meta.get("model_path") or meta.get("adapter_path") or "").strip()
    )
    if not resolved_model:
        raise ValueError(
            "No adapter/model path. Pass modelPath in the request, set "
            "EIF_ADAPTER_PATH (or EIF_MODEL_PATH), or put model_path in the report meta."
        )

    resolved_base = (
        str(base_model_path or "").strip()
        or str(os.environ.get("EIF_BASE_MODEL_PATH") or "").strip()
        or str(meta.get("base_model_path") or "").strip()
        or None
    )
    return resolved_model, resolved_base


def _get_model(model_path: str, base_model_path: str | None):
    key = (os.path.abspath(model_path), os.path.abspath(base_model_path) if base_model_path else "")
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    model, tokenizer = load_model_and_tokenizer(
        model_path=model_path,
        base_model_path=base_model_path,
    )
    _MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def _batch_from_tokens(
    tokenizer,
    tokens: list[str],
    *,
    answer_start_index: int,
    labeled_positions: set[int] | None = None,
) -> dict[str, torch.Tensor]:
    """Build a single-row causal-LM batch from report token surfaces."""
    token_ids = convert_report_tokens_to_ids(tokenizer, tokens)
    validate_roundtrip_tokens(tokenizer, token_ids, tokens)
    input_ids = torch.tensor([token_ids], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)

    labels = input_ids.clone()
    if answer_start_index > 0:
        labels[0, :answer_start_index] = -100
    if labeled_positions is not None:
        keep = {int(p) for p in labeled_positions}
        for i in range(labels.size(1)):
            if i not in keep:
                labels[0, i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def _logprob_from_ce(ce: float) -> float:
    return float(-ce)


def run_unlearn_pair_probe(
    report: dict[str, Any],
    *,
    train_sample_id: int,
    test_source_index: int,
    test_target_index: int,
    train_source_index: int,
    train_target_index: int,
    pair_id: str | None = None,
    model_path: str | None = None,
    base_model_path: str | None = None,
    unlearn_lr: float = 1.0,
    normalize_grad: bool = True,
    recompute_saliency: bool = True,
) -> dict[str, Any]:
    """Run the probe and return before/after metrics (model weights restored)."""
    baseline = report.get("test_sample_baseline") or {}
    test_tokens = list(baseline.get("full_tokens") or [])
    if not test_tokens:
        raise ValueError("Report is missing test_sample_baseline.full_tokens.")

    train_detail = (report.get("train_sample_details") or {}).get(str(train_sample_id))
    if not isinstance(train_detail, dict):
        raise ValueError(f"Report has no train_sample_details[{train_sample_id}].")
    train_tokens = list(train_detail.get("full_tokens") or [])
    if not train_tokens:
        raise ValueError(f"Train sample {train_sample_id} has empty full_tokens.")

    answer_start = int(train_detail.get("answer_start_index") or 0)
    test_prompt_len = int(baseline.get("prompt_len") or 0)

    if not (0 <= test_source_index < test_target_index < len(test_tokens)):
        raise ValueError(
            f"Invalid test edge indices: source={test_source_index}, "
            f"target={test_target_index}, seq_len={len(test_tokens)}."
        )
    if not (0 <= train_source_index < train_target_index < len(train_tokens)):
        raise ValueError(
            f"Invalid train edge indices: source={train_source_index}, "
            f"target={train_target_index}, seq_len={len(train_tokens)}."
        )

    resolved_model, resolved_base = _resolve_paths(report, model_path, base_model_path)
    model, tokenizer = _get_model(resolved_model, resolved_base)
    device = _device_of(model)
    grad_space = getattr(model, "_eif_grad_space", "unknown")
    resolved_base_used = getattr(model, "_eif_base_model_path", resolved_base)

    test_batch = _batch_from_tokens(
        tokenizer,
        test_tokens,
        answer_start_index=test_prompt_len,
        labeled_positions=None,
    )
    train_batch = _batch_from_tokens(
        tokenizer,
        train_tokens,
        answer_start_index=answer_start,
        labeled_positions={train_target_index},
    )

    reported_saliency = None
    for corr in baseline.get("top_correlations") or []:
        if int(corr.get("source_token_index", -1)) == int(test_source_index) and int(
            corr.get("target_token_index", test_target_index)
        ) == int(test_target_index):
            reported_saliency = float(corr.get("saliency_score"))
            break
    if reported_saliency is None:
        for row in report.get("per_token_results") or []:
            if int(row.get("target_token_index", -1)) != int(test_target_index):
                continue
            for corr in row.get("top_correlations") or []:
                if int(corr.get("source_token_index", -1)) == int(test_source_index):
                    reported_saliency = float(corr.get("saliency_score"))
                    break
            break

    base_losses = _target_token_losses(model, test_batch, [test_target_index], device)
    if base_losses.numel() == 0:
        raise RuntimeError("Could not score CE at the test target position.")
    base_ce = float(base_losses[0].item())
    base_logprob = _logprob_from_ce(base_ce)

    base_saliency = None
    if recompute_saliency:
        sal_vec = compute_alti_saliency_vector(model, test_batch, test_target_index)
        if 0 <= test_source_index < len(sal_vec):
            base_saliency = float(sal_vec[test_source_index])
        del sal_vec

    grad = compute_lm_head_ce_gradient_no_backward(
        model=model,
        batch=train_batch,
        device=device,
        ignored_token_ids=torch.tensor([], device=device),
    )

    delta = None
    try:
        delta = _apply_lm_head_ascent_update(
            model,
            grad,
            lr=float(unlearn_lr),
            normalize=bool(normalize_grad),
        )
        after_losses = _target_token_losses(model, test_batch, [test_target_index], device)
        if after_losses.numel() == 0:
            raise RuntimeError("Could not score post-unlearn CE at the test target.")
        after_ce = float(after_losses[0].item())
        after_logprob = _logprob_from_ce(after_ce)

        after_saliency = None
        if recompute_saliency:
            sal_vec = compute_alti_saliency_vector(model, test_batch, test_target_index)
            if 0 <= test_source_index < len(sal_vec):
                after_saliency = float(sal_vec[test_source_index])
            del sal_vec
    finally:
        if delta is not None:
            _restore_lm_head_ascent_update(model, delta)
        del grad
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    delta_ce = after_ce - base_ce
    delta_logprob = after_logprob - base_logprob
    delta_saliency = (
        None if base_saliency is None or after_saliency is None
        else after_saliency - base_saliency
    )

    verdict = "inconclusive"
    if delta_ce > 1e-4 and (delta_saliency is None or delta_saliency < -1e-8):
        verdict = "supports_causal"
    elif abs(delta_ce) <= 1e-4:
        verdict = "no_effect"
    elif delta_ce < -1e-4:
        verdict = "opposite_effect"

    return {
        "status": "success",
        "pairId": pair_id,
        "trainSampleId": int(train_sample_id),
        "modelPath": resolved_model,
        "baseModelPath": resolved_base_used,
        "gradSpace": grad_space,
        "update": {
            "paramSpace": "lm_head.weight",
            "rule": (
                "theta <- theta + eta * normalized(grad_train_ce)"
                if normalize_grad
                else "theta <- theta + eta * grad_train_ce"
            ),
            "unlearnLr": float(unlearn_lr),
            "normalizeGrad": bool(normalize_grad),
            "steps": 1,
            "persistsToDisk": False,
            "trainLossScope": "train_target_token_only",
            "trainTargetIndex": int(train_target_index),
            "trainSourceIndex": int(train_source_index),
        },
        "testEdge": {
            "sourceIndex": int(test_source_index),
            "targetIndex": int(test_target_index),
            "sourceToken": test_tokens[test_source_index],
            "targetToken": test_tokens[test_target_index],
            "reportedSaliency": reported_saliency,
        },
        "before": {
            "ce": base_ce,
            "logprob": base_logprob,
            "saliency": base_saliency,
        },
        "after": {
            "ce": after_ce,
            "logprob": after_logprob,
            "saliency": after_saliency,
        },
        "delta": {
            "ce": delta_ce,
            "logprob": delta_logprob,
            "saliency": delta_saliency,
        },
        "verdict": verdict,
        "restored": True,
    }
