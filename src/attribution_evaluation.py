from __future__ import annotations

import argparse
import json
import math
import os
import random
from functools import partial
from heapq import nlargest
from typing import Iterable

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, set_seed

from src.NIF import (
    CustomCollator,
    DatasetWrapper,
    NewInferenceFunction,
    build_single_sample_dataset,
    build_train_dataset,
    round_floats,
)
from src.intervention_experiment import (
    ALTI_CHUNK_SIZE,
    SEED,
    _clear_cuda_after_oom,
    is_trivial_token,
    lm_head_filter,
    load_model_and_tokenizer,
    load_samples,
    top_nontrivial_saliency_sources,
    _gather_scores,
    _is_cuda_alloc_error,
    _screen_training_set,
)
from src.loss import (
    compute_alti_saliency_vector,
    compute_lm_head_ce_gradient_no_backward,
)
from src.process_data import process_func_chatml


DEFAULT_K_VALUES = (5, 10, 20)
DEFAULT_DATA_METHOD_K = (10, 50, 100)
DEFAULT_DATA_ORACLE_M = (10, 20, 50)


def _write_json(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(round_floats(payload, 6), f, indent=2, ensure_ascii=False)


def _tensor_batch_to_device(batch: dict, device) -> dict:
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _decode_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)])


def _valid_target_positions(tokenizer, input_ids_1d: torch.Tensor, prompt_len: int, max_tokens: int) -> list[int]:
    end = min(input_ids_1d.numel(), prompt_len + max_tokens)
    return [
        int(t)
        for t in range(prompt_len, end)
        if t > 0 and not is_trivial_token(tokenizer, int(input_ids_1d[t].item()))
    ]


@torch.no_grad()
def _target_token_losses(
    model,
    batch: dict,
    target_positions: Iterable[int],
    device,
) -> torch.Tensor:
    """Return CE loss for labels at absolute sequence positions target_positions."""
    positions = [int(p) for p in target_positions if int(p) > 0]
    if not positions:
        return torch.empty(0, dtype=torch.float32)

    input_ids = batch["input_ids"].to(device)
    attention_mask = batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    logits = outputs.logits.float()
    rows = []
    labels = []
    for pos in positions:
        if pos >= input_ids.size(1):
            continue
        rows.append(logits[0, pos - 1])
        labels.append(int(input_ids[0, pos].item()))

    if not rows:
        del outputs, logits, rows
        torch.cuda.empty_cache()
        return torch.empty(0, dtype=torch.float32)

    label_tensor = torch.tensor(labels, dtype=torch.long, device=logits.device)
    losses = F.cross_entropy(torch.stack(rows, dim=0), label_tensor, reduction="none")

    del outputs, logits, rows, label_tensor
    torch.cuda.empty_cache()
    return losses.detach().cpu()


def _target_batch_with_labels(batch: dict, target_positions: Iterable[int]) -> dict:
    labels = torch.full_like(batch["input_ids"], -100)
    for pos in target_positions:
        pos = int(pos)
        if 0 <= pos < labels.size(1):
            labels[0, pos] = batch["input_ids"][0, pos]
    return {
        "input_ids": batch["input_ids"],
        "attention_mask": batch["attention_mask"],
        "labels": labels,
    }


def _response_label_positions(batch: dict) -> list[int]:
    labels = batch.get("labels")
    if labels is None:
        raise KeyError("sample-to-sample data evaluation requires test_batch['labels'].")
    valid = torch.where(labels[0].ne(-100))[0]
    return [int(pos.item()) for pos in valid if int(pos.item()) > 0]


def _lm_head_grad_for_positions(model, batch: dict, target_positions: Iterable[int], device) -> torch.Tensor:
    labeled = _target_batch_with_labels(batch, target_positions)
    grad = compute_lm_head_ce_gradient_no_backward(
        model=model,
        batch=labeled,
        device=device,
        ignored_token_ids=torch.tensor([], device=device),
    )
    return grad


def _ranking_ndcg(ranked_ids: list[int], relevance: dict[int, float], k: int) -> float:
    def gain(item_id: int) -> float:
        # NDCG expects non-negative relevance. Negative intervention effects mean
        # the candidate did not support the target token, so they get zero gain.
        return max(0.0, float(relevance.get(int(item_id), 0.0)))

    top = ranked_ids[:k]
    dcg = 0.0
    for rank, item_id in enumerate(top, start=1):
        rel = gain(item_id)
        dcg += rel / math.log2(rank + 1)

    ideal_rels = sorted((max(0.0, float(v)) for v in relevance.values()), reverse=True)[:k]
    idcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(ideal_rels, start=1))
    return 0.0 if idcg <= 0 else dcg / idcg


def _recall_at(method_ranked: list[int], oracle_ranked: list[int], method_k: int, oracle_m: int) -> float:
    oracle_set = set(int(x) for x in oracle_ranked[:oracle_m])
    if not oracle_set:
        return 0.0
    method_set = set(int(x) for x in method_ranked[:method_k])
    return len(method_set & oracle_set) / len(oracle_set)


def _spearman_top_overlap(method_ranked: list[int], oracle_ranked: list[int], k: int) -> float:
    """Small dependency-free rank correlation over the union of both top-k lists."""
    items = list(dict.fromkeys(method_ranked[:k] + oracle_ranked[:k]))
    n = len(items)
    if n < 2:
        return 0.0

    fallback = n + 1
    method_pos = {item: rank for rank, item in enumerate(method_ranked[:k], start=1)}
    oracle_pos = {item: rank for rank, item in enumerate(oracle_ranked[:k], start=1)}
    xs = torch.tensor([method_pos.get(item, fallback) for item in items], dtype=torch.float32)
    ys = torch.tensor([oracle_pos.get(item, fallback) for item in items], dtype=torch.float32)
    xs = xs - xs.mean()
    ys = ys - ys.mean()
    denom = xs.norm() * ys.norm()
    return 0.0 if float(denom) == 0.0 else float(torch.dot(xs, ys) / denom)


def _perturb_sources(
    batch: dict,
    source_indices: Iterable[int],
    *,
    mode: str,
    replacement_token_id: int,
) -> dict:
    perturbed = {
        k: v.clone() if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }
    for idx in source_indices:
        idx = int(idx)
        if idx < 0 or idx >= perturbed["input_ids"].size(1):
            continue
        if mode == "replace":
            perturbed["input_ids"][0, idx] = int(replacement_token_id)
        elif mode == "zero_attention":
            if "attention_mask" not in perturbed:
                raise ValueError("zero_attention perturbation requires attention_mask.")
            perturbed["attention_mask"][0, idx] = 0
        else:
            raise ValueError(f"Unknown feature perturbation mode: {mode}")
    return perturbed


def _feature_aopc(
    model,
    tokenizer,
    batch: dict,
    target_idx: int,
    ranked_sources: list[int],
    base_loss: float,
    *,
    device,
    k: int,
    perturb_mode: str,
    replacement_token_id: int,
) -> float:
    effects = []
    selected = []
    for src_idx in ranked_sources[:k]:
        selected.append(int(src_idx))
        perturbed = _perturb_sources(
            batch,
            selected,
            mode=perturb_mode,
            replacement_token_id=replacement_token_id,
        )
        loss = _target_token_losses(model, perturbed, [target_idx], device)
        if loss.numel() == 0:
            continue
        effects.append(float(loss[0].item()) - base_loss)
        del perturbed, loss
    return 0.0 if not effects else sum(effects) / len(effects)


def evaluate_feature_attribution(
    model,
    tokenizer,
    test_batch: dict,
    target_positions: list[int],
    *,
    device,
    top_k_prompt_tokens: int,
    k_values: tuple[int, ...],
    perturb_mode: str,
    replacement_token_id: int,
    max_feature_sources: int | None,
) -> list[dict]:
    results = []
    ids_1d = test_batch["input_ids"][0]

    for target_idx in target_positions:
        target_idx = int(target_idx)
        target_text = _decode_token(tokenizer, int(ids_1d[target_idx].item()))
        print(f"\n[feature] target {target_idx}: {target_text!r}", flush=True)

        saliency = compute_alti_saliency_vector(
            model,
            test_batch,
            target_idx,
            chunk_size=ALTI_CHUNK_SIZE,
        )
        attr_pairs = [
            (int(idx), float(score))
            for idx, score in top_nontrivial_saliency_sources(
                tokenizer,
                ids_1d,
                saliency,
                k=max(target_idx, top_k_prompt_tokens),
            )
        ]
        if max_feature_sources is not None and max_feature_sources > 0:
            attr_pairs = attr_pairs[:max_feature_sources]

        method_ranked = [idx for idx, _ in attr_pairs]
        method_scores = {idx: score for idx, score in attr_pairs}

        base_loss = float(_target_token_losses(model, test_batch, [target_idx], device)[0].item())
        oracle_effects: dict[int, float] = {}
        for src_idx in tqdm(method_ranked, desc=f"Feature oracle t={target_idx}", leave=False):
            perturbed = _perturb_sources(
                test_batch,
                [src_idx],
                mode=perturb_mode,
                replacement_token_id=replacement_token_id,
            )
            perturbed_loss = _target_token_losses(model, perturbed, [target_idx], device)
            if perturbed_loss.numel() > 0:
                oracle_effects[int(src_idx)] = float(perturbed_loss[0].item()) - base_loss
            del perturbed, perturbed_loss

        oracle_ranked = sorted(oracle_effects, key=lambda x: oracle_effects[x], reverse=True)

        metrics = {}
        for k in k_values:
            kk = min(int(k), len(method_ranked), len(oracle_ranked))
            if kk <= 0:
                continue
            metrics[f"recall@{kk}"] = _recall_at(method_ranked, oracle_ranked, kk, kk)
            metrics[f"ndcg@{kk}"] = _ranking_ndcg(method_ranked, oracle_effects, kk)
            metrics[f"spearman_top{kk}"] = _spearman_top_overlap(method_ranked, oracle_ranked, kk)
            metrics[f"aopc@{kk}"] = _feature_aopc(
                model,
                tokenizer,
                test_batch,
                target_idx,
                method_ranked,
                base_loss,
                device=device,
                k=kk,
                perturb_mode=perturb_mode,
                replacement_token_id=replacement_token_id,
            )

        results.append({
            "target_token_index": target_idx,
            "target_token": target_text,
            "base_ce_loss": base_loss,
            "perturbation": {
                "mode": perturb_mode,
                "replacement_token_id": int(replacement_token_id),
                "replacement_token": _decode_token(tokenizer, int(replacement_token_id)),
            },
            "metrics": metrics,
            "method_top": [
                {
                    "rank": rank,
                    "source_token_index": int(idx),
                    "source_token": _decode_token(tokenizer, int(ids_1d[idx].item())),
                    "alti_saliency": float(method_scores[idx]),
                    "oracle_effect": float(oracle_effects.get(idx, 0.0)),
                }
                for rank, idx in enumerate(method_ranked[:max(k_values)], start=1)
            ],
            "oracle_top": [
                {
                    "rank": rank,
                    "source_token_index": int(idx),
                    "source_token": _decode_token(tokenizer, int(ids_1d[idx].item())),
                    "oracle_effect": float(oracle_effects[idx]),
                    "alti_saliency": float(method_scores.get(idx, 0.0)),
                }
                for rank, idx in enumerate(oracle_ranked[:max(k_values)], start=1)
            ],
        })

    return results


def _select_oracle_universe(
    coarse_ranked: list[tuple[int, float]],
    *,
    limit: int | None,
    include_method_top: int,
    seed: int,
) -> list[int]:
    all_ids = [int(idx) for idx, _ in coarse_ranked]
    if limit is None or limit <= 0 or limit >= len(all_ids):
        return all_ids

    limit = int(limit)
    include_method_top = max(0, min(int(include_method_top), limit, len(all_ids)))
    selected = list(dict.fromkeys(all_ids[:include_method_top]))
    remaining = [idx for idx in all_ids if idx not in set(selected)]
    rng = random.Random(seed)
    fill_n = max(0, limit - len(selected))
    if fill_n > 0:
        selected.extend(rng.sample(remaining, k=min(fill_n, len(remaining))))
    return selected


def _single_train_batch_from_dataset(train_ds, collator, train_idx: int, idx_to_row: dict[int, int], device) -> dict:
    row = idx_to_row[int(train_idx)]
    feature = dict(train_ds[row])
    batch = collator([feature])
    return _tensor_batch_to_device(batch, device)


@torch.no_grad()
def _apply_lm_head_ascent_update(
    model,
    grad: torch.Tensor,
    *,
    lr: float,
    normalize: bool,
) -> torch.Tensor:
    lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Model does not expose output embeddings.")

    delta = grad.to(device=lm_head.weight.device, dtype=torch.float32)
    if normalize:
        delta = delta / delta.norm().clamp_min(1e-12)
    delta = (lr * delta).to(dtype=lm_head.weight.dtype)
    lm_head.weight.data.add_(delta)
    return delta


@torch.no_grad()
def _restore_lm_head_ascent_update(model, delta: torch.Tensor) -> None:
    lm_head = model.get_output_embeddings()
    lm_head.weight.data.sub_(delta.to(device=lm_head.weight.device, dtype=lm_head.weight.dtype))


def _evaluate_data_coarse_unit(
    model,
    tokenizer,
    test_batch: dict,
    train_ds,
    train_loader,
    collator,
    target_positions: list[int],
    accelerator,
    *,
    unit_type: str,
    unit_name: str,
    prescreen_max_seq_len: int | None,
    method_top_ks: tuple[int, ...],
    oracle_top_ms: tuple[int, ...],
    oracle_limit: int | None,
    oracle_include_method_top: int,
    unlearn_lr: float,
    normalize_unlearn_grad: bool,
    effect_reduction: str,
    seed: int,
) -> dict:
    if accelerator.num_processes != 1:
        raise RuntimeError("Data oracle unlearning currently expects a single Accelerator process.")

    filtered_params = [p for n, p in model.named_parameters() if lm_head_filter(n, p)]
    lm_head_device = filtered_params[0].device if filtered_params else accelerator.device

    print(f"\n[data:{unit_name}] computing CE-gradient coarse ranking...", flush=True)
    test_grad = _lm_head_grad_for_positions(model, test_batch, target_positions, accelerator.device)
    test_grad_flat = test_grad.reshape(-1).to(lm_head_device).detach()
    local_scores = _screen_training_set(
        model,
        test_grad_flat,
        train_loader,
        accelerator.device,
        lm_head_device,
        desc=f"Data coarse scoring ({unit_name})",
        max_seq_len=prescreen_max_seq_len,
    )
    coarse_scores = _gather_scores(accelerator, local_scores, accelerator.device)
    coarse_ranked = nlargest(len(coarse_scores), coarse_scores, key=lambda x: x[1])
    method_score_by_id = {int(idx): float(score) for idx, score in coarse_ranked}

    oracle_universe = _select_oracle_universe(
        coarse_ranked,
        limit=oracle_limit,
        include_method_top=oracle_include_method_top,
        seed=seed,
    )
    idx_to_row = {int(train_ds[i]["sample_index"]): i for i in range(len(train_ds))}
    oracle_universe = [idx for idx in oracle_universe if idx in idx_to_row]

    base_losses = _target_token_losses(model, test_batch, target_positions, accelerator.device)
    if base_losses.numel() == 0:
        raise RuntimeError("No valid target positions for data attribution evaluation.")

    oracle_effects: dict[int, float] = {}
    oracle_token_effects: dict[int, list[float]] = {}
    skipped_oracle_oom: list[int] = []

    print(
        f"[data:{unit_name}] one-step unlearning oracle over "
        f"{len(oracle_universe)} train samples...",
        flush=True,
    )
    for train_idx in tqdm(oracle_universe, desc=f"Data unlearning oracle ({unit_name})", leave=False):
        train_batch = None
        grad = None
        try:
            train_batch = _single_train_batch_from_dataset(
                train_ds,
                collator,
                train_idx,
                idx_to_row,
                accelerator.device,
            )
            grad = compute_lm_head_ce_gradient_no_backward(
                model=model,
                batch=train_batch,
                device=accelerator.device,
                ignored_token_ids=torch.tensor([], device=accelerator.device),
            )
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not _is_cuda_alloc_error(exc):
                raise
            skipped_oracle_oom.append(int(train_idx))
            _clear_cuda_after_oom()
            print(
                f"[WARN] Data unlearning oracle ({unit_name}): "
                f"skipping train sample {int(train_idx)} after CUDA OOM while computing train gradient.",
                flush=True,
            )
            del train_batch, grad
            continue

        delta = None
        try:
            delta = _apply_lm_head_ascent_update(
                model,
                grad,
                lr=unlearn_lr,
                normalize=normalize_unlearn_grad,
            )
            changed_losses = _target_token_losses(model, test_batch, target_positions, accelerator.device)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not _is_cuda_alloc_error(exc):
                raise
            skipped_oracle_oom.append(int(train_idx))
            _clear_cuda_after_oom()
            print(
                f"[WARN] Data unlearning oracle ({unit_name}): "
                f"skipping train sample {int(train_idx)} after CUDA OOM while scoring test effect.",
                flush=True,
            )
            del train_batch, grad
            continue
        finally:
            if delta is not None:
                _restore_lm_head_ascent_update(model, delta)

        token_effects = changed_losses - base_losses
        if effect_reduction == "sum":
            effect = float(token_effects.sum().item())
        elif effect_reduction == "max":
            effect = float(token_effects.max().item())
        elif effect_reduction == "mean":
            effect = float(token_effects.mean().item())
        else:
            raise ValueError(f"Unsupported effect reduction: {effect_reduction}")

        oracle_effects[int(train_idx)] = effect
        oracle_token_effects[int(train_idx)] = [float(x) for x in token_effects.tolist()]

        del train_batch, grad, changed_losses, token_effects
        torch.cuda.empty_cache()

    if skipped_oracle_oom:
        print(
            f"[WARN] Data unlearning oracle ({unit_name}): skipped "
            f"{len(skipped_oracle_oom)} samples after CUDA OOM.",
            flush=True,
        )
    if not oracle_effects:
        raise RuntimeError(
            f"Data unlearning oracle ({unit_name}) produced no scores. "
            "Try freeing GPU memory, lowering --prescreen-max-seq-len, or setting --data-oracle-limit."
        )

    method_ranked = [
        int(idx)
        for idx, _ in coarse_ranked
        if int(idx) in oracle_effects
    ]
    oracle_ranked = sorted(oracle_effects, key=lambda x: oracle_effects[x], reverse=True)

    metrics = {}
    for method_k in method_top_ks:
        kk = min(int(method_k), len(method_ranked))
        if kk <= 0:
            continue
        metrics[f"ndcg@{kk}"] = _ranking_ndcg(method_ranked, oracle_effects, kk)
        for oracle_m in oracle_top_ms:
            mm = min(int(oracle_m), len(oracle_ranked))
            if mm <= 0:
                continue
            metrics[f"recall@{kk}_oracle_top{mm}"] = _recall_at(method_ranked, oracle_ranked, kk, mm)

    top_n = max(max(method_top_ks), max(oracle_top_ms))
    return {
        "unit_type": unit_type,
        "unit_name": unit_name,
        "target_token_indices": [int(x) for x in target_positions],
        "target_tokens": [
            _decode_token(tokenizer, int(test_batch["input_ids"][0, t].item()))
            for t in target_positions
        ],
        "oracle": {
            "unlearning_update": "theta <- theta + eta * normalized_grad_train"
            if normalize_unlearn_grad
            else "theta <- theta + eta * grad_train",
            "unlearn_lr": float(unlearn_lr),
            "normalize_train_gradient": bool(normalize_unlearn_grad),
            "effect_reduction": effect_reduction,
            "candidate_universe_size": len(oracle_universe),
            "scored_candidate_count": len(oracle_effects),
            "skipped_oom_count": len(skipped_oracle_oom),
            "skipped_oom_train_sample_ids": skipped_oracle_oom[:50],
            "loss_scope": (
                "single target token"
                if unit_type == "sample_to_token"
                else "all labeled response tokens in the test sample"
            ),
            "base_target_losses": [float(x) for x in base_losses.tolist()],
        },
        "metrics": metrics,
        "method_top": [
            {
                "rank": rank,
                "train_sample_id": int(idx),
                "coarse_cos_sim": float(method_score_by_id[idx]),
                "oracle_effect": float(oracle_effects.get(idx, 0.0)),
            }
            for rank, idx in enumerate(method_ranked[:top_n], start=1)
        ],
        "oracle_top": [
            {
                "rank": rank,
                "train_sample_id": int(idx),
                "oracle_effect": float(oracle_effects[idx]),
                "coarse_cos_sim": float(method_score_by_id.get(idx, 0.0)),
                "token_effects": oracle_token_effects[idx],
            }
            for rank, idx in enumerate(oracle_ranked[:top_n], start=1)
        ],
    }


def evaluate_data_coarse_attribution(
    model,
    tokenizer,
    test_batch: dict,
    train_ds,
    train_loader,
    collator,
    target_positions: list[int],
    accelerator,
    *,
    granularity: str,
    prescreen_max_seq_len: int | None,
    method_top_ks: tuple[int, ...],
    oracle_top_ms: tuple[int, ...],
    oracle_limit: int | None,
    oracle_include_method_top: int,
    unlearn_lr: float,
    normalize_unlearn_grad: bool,
    effect_reduction: str,
    seed: int,
) -> dict:
    if granularity not in {"sample_to_token", "sample_to_sample", "both"}:
        raise ValueError(
            "granularity must be one of: sample_to_token, sample_to_sample, both"
        )

    result = {
        "granularity": granularity,
        "sample_to_token": [],
        "sample_to_sample": None,
    }

    if granularity in {"sample_to_token", "both"}:
        for target_idx in target_positions:
            unit = _evaluate_data_coarse_unit(
                model,
                tokenizer,
                test_batch,
                train_ds,
                train_loader,
                collator,
                [int(target_idx)],
                accelerator,
                unit_type="sample_to_token",
                unit_name=f"token_{int(target_idx)}",
                prescreen_max_seq_len=prescreen_max_seq_len,
                method_top_ks=method_top_ks,
                oracle_top_ms=oracle_top_ms,
                oracle_limit=oracle_limit,
                oracle_include_method_top=oracle_include_method_top,
                unlearn_lr=unlearn_lr,
                normalize_unlearn_grad=normalize_unlearn_grad,
                effect_reduction=effect_reduction,
                seed=seed,
            )
            result["sample_to_token"].append(unit)

    if granularity in {"sample_to_sample", "both"}:
        response_positions = _response_label_positions(test_batch)
        if not response_positions:
            raise RuntimeError("No labeled response tokens found for sample-to-sample evaluation.")
        result["sample_to_sample"] = _evaluate_data_coarse_unit(
            model,
            tokenizer,
            test_batch,
            train_ds,
            train_loader,
            collator,
            response_positions,
            accelerator,
            unit_type="sample_to_sample",
            unit_name="full_response",
            prescreen_max_seq_len=prescreen_max_seq_len,
            method_top_ks=method_top_ks,
            oracle_top_ms=oracle_top_ms,
            oracle_limit=oracle_limit,
            oracle_include_method_top=oracle_include_method_top,
            unlearn_lr=unlearn_lr,
            normalize_unlearn_grad=normalize_unlearn_grad,
            effect_reduction=effect_reduction,
            seed=seed,
        )

    return result


def build_test_batch(
    model,
    tokenizer,
    test_sample: dict,
    convert_to_chatml,
    base_collator,
    accelerator,
    *,
    use_generated_response: bool,
    generation_limit: int,
) -> tuple[dict, int, dict]:
    test_ds = build_single_sample_dataset(test_sample, convert_to_chatml)
    raw_batch = base_collator([test_ds[0]])
    raw_batch = _tensor_batch_to_device(raw_batch, accelerator.device)

    infer = NewInferenceFunction(
        model=model,
        tokenizer=tokenizer,
        train_loader=None,
        accelerator=accelerator,
        param_filter_fn=lm_head_filter,
        top_k=20,
    )
    gen_result = infer.infer(raw_batch, gen_limit=generation_limit, skip_saliency=True)
    prompt_len = int(gen_result["target_idx"][0])

    if use_generated_response:
        prompt_ids = raw_batch["input_ids"][0, :prompt_len]
        pred_ids = torch.tensor(
            gen_result["pred_ids"][0],
            device=prompt_ids.device,
            dtype=prompt_ids.dtype,
        )
        input_ids = torch.cat([prompt_ids, pred_ids], dim=0).unsqueeze(0)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        labels[:, :prompt_len] = -100
        test_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        source = "generated_response"
    else:
        test_batch = raw_batch
        source = "ground_truth_response"

    meta = {
        "response_source": source,
        "prompt_len": prompt_len,
        "generated_full_tokens": gen_result["pred_full_tokens"][0],
        "ground_truth_full_tokens": gen_result["full_tokens"][0],
    }
    return test_batch, prompt_len, meta


def parse_int_tuple(raw: str) -> tuple[int, ...]:
    if not raw.strip():
        return ()
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def _default_output_path(task_id: str, target_positions: list[int]) -> str:
    token_part = f"tok{target_positions[0]}" if len(target_positions) == 1 else "tokens"
    return f"attribution_eval_{task_id}_{token_part}.json"


def _stage_output_path(output_path: str, stage_name: str) -> str:
    root, ext = os.path.splitext(output_path)
    return f"{root}_{stage_name}{ext or '.json'}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate feature and data attribution with intervention-based ranking metrics."
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--train-data", type=str, default="sft_train.jsonl")
    parser.add_argument("--test-data", type=str, default="sft_test.jsonl")
    parser.add_argument("--test-index", type=int, default=58)
    parser.add_argument("--token-index", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=20)
    parser.add_argument("--generation-limit", type=int, default=128)
    parser.add_argument("--use-ground-truth-response", action="store_true")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--prescreen-batch-size", type=int, default=1)
    parser.add_argument("--prescreen-max-seq-len", type=int, default=3000)
    parser.add_argument("--max-gpu-memory", type=str, default=None)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--skip-feature", action="store_true")
    parser.add_argument("--skip-data", action="store_true")
    parser.add_argument("--top-k-prompt-tokens", type=int, default=20)
    parser.add_argument("--feature-k-values", type=str, default="5,10,20")
    parser.add_argument("--feature-perturb-mode", choices=["replace", "zero_attention"], default="replace")
    parser.add_argument("--replacement-token-id", type=int, default=None)
    parser.add_argument("--max-feature-sources", type=int, default=None)
    parser.add_argument("--data-method-k-values", type=str, default="10,50,100")
    parser.add_argument("--data-oracle-m-values", type=str, default="10,20,50")
    parser.add_argument("--data-oracle-limit", type=int, default=None)
    parser.add_argument("--data-oracle-include-method-top", type=int, default=100)
    parser.add_argument(
        "--data-granularity",
        choices=["sample_to_token", "sample_to_sample", "both"],
        default="sample_to_sample",
        help=(
            "Data attribution evaluation scope. sample_to_token evaluates each target "
            "token separately; sample_to_sample evaluates the whole labeled response. "
            "Default: sample_to_sample."
        ),
    )
    parser.add_argument(
        "--unlearn-lr",
        type=float,
        default=1.0,
        help=(
            "Norm of the one-step unlearning update when train gradients are normalized. "
            "Use a smaller value if --no-normalize-unlearn-grad is set."
        ),
    )
    parser.add_argument("--no-normalize-unlearn-grad", action="store_true")
    parser.add_argument("--data-effect-reduction", choices=["mean", "sum", "max"], default="mean")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--feature-output",
        type=str,
        default=None,
        help="Optional path for the feature-attribution checkpoint report.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    accelerator = Accelerator()
    if accelerator.num_processes != 1 and not args.skip_data:
        raise RuntimeError("--skip-data is required for multi-process runs in this first evaluator.")

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        attn_implementation=args.attn_implementation,
        max_gpu_memory=args.max_gpu_memory,
    )
    convert_to_chatml = partial(process_func_chatml, tokenizer=tokenizer)

    train_samples = load_samples(args.train_data)
    test_samples = load_samples(args.test_data)
    if args.train_limit is not None:
        train_samples = train_samples[: int(args.train_limit)]
    test_sample = test_samples[int(args.test_index)]
    task_id = test_sample.get("task_id") or f"test{args.test_index}"

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    collator = CustomCollator(base_collator)

    print("[eval] building test batch...", flush=True)
    test_batch, prompt_len, test_meta = build_test_batch(
        model,
        tokenizer,
        test_sample,
        convert_to_chatml,
        base_collator,
        accelerator,
        use_generated_response=not args.use_ground_truth_response,
        generation_limit=args.generation_limit,
    )
    test_batch = _tensor_batch_to_device(test_batch, accelerator.device)

    if args.token_index is not None:
        target_positions = [int(args.token_index)]
    else:
        target_positions = _valid_target_positions(
            tokenizer,
            test_batch["input_ids"][0],
            prompt_len,
            int(args.max_output_tokens),
        )
    if not target_positions:
        raise RuntimeError("No target positions selected for evaluation.")

    train_ds = None
    train_loader = None

    replacement_token_id = args.replacement_token_id
    if replacement_token_id is None:
        replacement_token_id = tokenizer.pad_token_id
    if replacement_token_id is None:
        replacement_token_id = tokenizer.eos_token_id
    if replacement_token_id is None:
        raise RuntimeError("No replacement token id available; pass --replacement-token-id.")

    output_path = args.output or _default_output_path(task_id, target_positions)
    feature_output_path = args.feature_output or _stage_output_path(output_path, "feature")

    report = {
        "experiment_meta": {
            "test_sample_index": int(args.test_index),
            "task_id": task_id,
            "target_token_indices": [int(x) for x in target_positions],
            "target_tokens": [
                _decode_token(tokenizer, int(test_batch["input_ids"][0, t].item()))
                for t in target_positions
            ],
            "prompt_len": int(prompt_len),
            "train_size": len(train_samples),
            "evaluation_protocol": {
                "feature_oracle": "leave-one-source-token-out target CE increase",
                "data_oracle": "one-step gradient-ascent unlearning target CE increase",
                "data_sample_to_token": "rank train samples by effect on each target token CE",
                "data_sample_to_sample": "rank train samples by effect on whole response CE",
                "data_unlearning_param_space": "lm_head.weight",
                "data_unlearning_direction": "theta <- theta + eta * grad_train_loss",
            },
            "config": {
                "feature_k_values": parse_int_tuple(args.feature_k_values),
                "feature_perturb_mode": args.feature_perturb_mode,
                "max_feature_sources": args.max_feature_sources,
                "data_method_k_values": parse_int_tuple(args.data_method_k_values),
                "data_oracle_m_values": parse_int_tuple(args.data_oracle_m_values),
                "data_oracle_limit": args.data_oracle_limit,
                "data_oracle_include_method_top": args.data_oracle_include_method_top,
                "data_granularity": args.data_granularity,
                "unlearn_lr": args.unlearn_lr,
                "normalize_unlearn_grad": not args.no_normalize_unlearn_grad,
                "data_effect_reduction": args.data_effect_reduction,
                "prescreen_max_seq_len": args.prescreen_max_seq_len,
            },
        },
        "test_sample_baseline": test_meta,
    }

    if not args.skip_feature:
        report["feature_attribution"] = evaluate_feature_attribution(
            model,
            tokenizer,
            test_batch,
            target_positions,
            device=accelerator.device,
            top_k_prompt_tokens=max(1, int(args.top_k_prompt_tokens)),
            k_values=parse_int_tuple(args.feature_k_values) or DEFAULT_K_VALUES,
            perturb_mode=args.feature_perturb_mode,
            replacement_token_id=int(replacement_token_id),
            max_feature_sources=args.max_feature_sources,
        )
        feature_report = {
            "experiment_meta": {
                **report["experiment_meta"],
                "stage": "feature_attribution",
                "is_checkpoint": True,
            },
            "test_sample_baseline": report["test_sample_baseline"],
            "feature_attribution": report["feature_attribution"],
        }
        _write_json(feature_output_path, feature_report)
        print(f"\n[eval] wrote feature checkpoint {feature_output_path}", flush=True)

    if not args.skip_data:
        print("[eval] building train dataset...", flush=True)
        train_ds = build_train_dataset(train_samples, convert_to_chatml)
        train_loader = torch.utils.data.DataLoader(
            DatasetWrapper(train_ds),
            batch_size=max(1, int(args.prescreen_batch_size)),
            collate_fn=collator,
        )
        train_loader = accelerator.prepare(train_loader)

        report["data_coarse_attribution"] = evaluate_data_coarse_attribution(
            model,
            tokenizer,
            test_batch,
            train_ds,
            train_loader,
            collator,
            target_positions,
            accelerator,
            granularity=args.data_granularity,
            prescreen_max_seq_len=None if args.prescreen_max_seq_len <= 0 else int(args.prescreen_max_seq_len),
            method_top_ks=parse_int_tuple(args.data_method_k_values) or DEFAULT_DATA_METHOD_K,
            oracle_top_ms=parse_int_tuple(args.data_oracle_m_values) or DEFAULT_DATA_ORACLE_M,
            oracle_limit=args.data_oracle_limit,
            oracle_include_method_top=args.data_oracle_include_method_top,
            unlearn_lr=float(args.unlearn_lr),
            normalize_unlearn_grad=not args.no_normalize_unlearn_grad,
            effect_reduction=args.data_effect_reduction,
            seed=int(args.seed),
        )

    _write_json(output_path, report)
    print(f"\n[eval] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
