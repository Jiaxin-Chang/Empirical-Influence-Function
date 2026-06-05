from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass, field
from functools import partial
from heapq import nlargest
from typing import Iterable

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from transformers import DataCollatorForSeq2Seq, set_seed

from src.NIF import CustomCollator, DatasetWrapper, build_train_dataset
from src.attribution_batch_report import build_report, _write_json as write_report_json, _write_markdown
from src.attribution_evaluation import (
    DEFAULT_DATA_GROUP_K,
    DEFAULT_DATA_METHOD_K,
    DEFAULT_DATA_ORACLE_M,
    _apply_lm_head_ascent_update,
    _decode_token,
    _evaluate_data_group_unlearning_effects,
    _lm_head_grad_for_positions,
    _lm_head_sketch_for_positions,
    _ranking_ndcg,
    _recall_at,
    _response_label_positions,
    _restore_lm_head_ascent_update,
    _select_oracle_universe,
    _single_train_batch_from_dataset,
    _target_token_losses,
    _valid_target_positions,
    _write_json,
    build_test_batch,
    parse_int_tuple,
)
from src.intervention_experiment import (
    PRESCREEN_SKETCH_CACHE_DIR,
    PRESCREEN_SKETCH_DIM,
    PRESCREEN_SKETCH_SEED,
    SEED,
    _clear_cuda_after_oom,
    _gather_scores,
    _is_cuda_alloc_error,
    _load_or_build_prescreen_sketch_cache,
    _score_prescreen_sketch_cache,
    _screen_training_set,
    lm_head_filter,
    load_model_and_tokenizer,
    load_samples,
)
from src.loss import compute_lm_head_ce_gradient_no_backward
from src.process_data import process_func_chatml


@dataclass
class TestOracleState:
    test_index: int
    task_id: str
    output_path: str
    coarse_output_path: str
    test_batch: dict
    test_meta: dict
    prompt_len: int
    semantic_target_positions: list[int]
    response_positions: list[int]
    coarse_ranked: list[tuple[int, float]]
    coarse_scoring: str
    oracle_universe: list[int]
    method_score_by_id: dict[int, float]
    base_losses: torch.Tensor
    hidden_cpu: torch.Tensor | None = None
    labels_cpu: torch.Tensor | None = None
    oracle_effects: dict[int, float] = field(default_factory=dict)
    oracle_token_effects: dict[int, list[float]] = field(default_factory=dict)
    skipped_oracle_oom: list[int] = field(default_factory=list)


def _parse_indices(raw: str | None, start_idx: int, end_idx: int | None, total: int) -> list[int]:
    if raw:
        chunks = [x for x in raw.replace(",", " ").split() if x]
        indices = [int(x) for x in chunks]
    else:
        stop = total - 1 if end_idx is None else int(end_idx)
        indices = list(range(int(start_idx), stop + 1))

    for idx in indices:
        if idx < 0 or idx >= total:
            raise ValueError(f"test index out of range: {idx} (valid: 0..{total - 1})")
    return indices


def _chunks(items: list[int], size: int) -> Iterable[list[int]]:
    size = max(1, int(size))
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _lm_head_tied_to_input_embeddings(model) -> bool:
    output_embeddings = model.get_output_embeddings()
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    input_embeddings = get_input_embeddings() if callable(get_input_embeddings) else None
    if output_embeddings is None or input_embeddings is None:
        return False
    return output_embeddings.weight.data_ptr() == input_embeddings.weight.data_ptr()


@torch.no_grad()
def _cached_hidden_losses(
    model,
    hidden_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    *,
    delta: torch.Tensor | None,
    device,
    chunk_size: int,
) -> torch.Tensor:
    lm_head = model.get_output_embeddings()
    losses = []
    chunk_size = max(1, int(chunk_size))
    for start in range(0, hidden_cpu.size(0), chunk_size):
        end = min(start + chunk_size, hidden_cpu.size(0))
        hidden = hidden_cpu[start:end].to(device=device, dtype=lm_head.weight.dtype)
        labels = labels_cpu[start:end].to(device=device)
        logits = lm_head(hidden).float()
        if delta is not None:
            logits.add_(F.linear(hidden, delta.to(device=device, dtype=lm_head.weight.dtype)).float())
        losses.append(F.cross_entropy(logits, labels, reduction="none").detach().cpu())
        del hidden, labels, logits
    return torch.cat(losses, dim=0)


def _cached_hidden_losses_retry(
    model,
    hidden_cpu: torch.Tensor,
    labels_cpu: torch.Tensor,
    *,
    delta: torch.Tensor | None,
    device,
    chunk_size: int,
) -> torch.Tensor:
    cur_chunk = max(1, int(chunk_size))
    while True:
        try:
            return _cached_hidden_losses(
                model,
                hidden_cpu,
                labels_cpu,
                delta=delta,
                device=device,
                chunk_size=cur_chunk,
            )
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if not _is_cuda_alloc_error(exc) or cur_chunk == 1:
                raise
            _clear_cuda_after_oom()
            cur_chunk = max(1, cur_chunk // 2)
            print(
                f"[batched-data] cached hidden scoring OOM; retrying chunk_size={cur_chunk}.",
                flush=True,
            )


@torch.no_grad()
def _cache_response_hidden(
    model,
    test_batch: dict,
    target_positions: list[int],
    *,
    device,
    lm_head_device,
    logit_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = test_batch["input_ids"].to(device)
    attention_mask = test_batch.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    base_model = getattr(model, "model", None)
    if base_model is None:
        raise RuntimeError("Expected a HuggingFace causal LM with .model.")

    outputs = base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    )
    hidden = outputs.last_hidden_state
    row_positions = []
    labels = []
    for pos in target_positions:
        pos = int(pos)
        if 0 < pos < input_ids.size(1):
            row_positions.append(pos - 1)
            labels.append(int(input_ids[0, pos].item()))
    if not row_positions:
        raise RuntimeError("No valid response positions to cache.")

    rows = torch.tensor(row_positions, dtype=torch.long, device=hidden.device)
    hidden_rows = hidden[0].index_select(0, rows).to("cpu")
    labels_cpu = torch.tensor(labels, dtype=torch.long)
    base_losses = _cached_hidden_losses_retry(
        model,
        hidden_rows,
        labels_cpu,
        delta=None,
        device=lm_head_device,
        chunk_size=logit_chunk_size,
    )
    del outputs, hidden, rows
    torch.cuda.empty_cache()
    return hidden_rows, labels_cpu, base_losses


def _write_coarse_checkpoint(state: TestOracleState, tokenizer) -> None:
    checkpoint = {
        "experiment_meta": {
            "test_sample_index": int(state.test_index),
            "task_id": state.task_id,
            "stage": "data_coarse_attribution",
            "is_checkpoint": True,
            "unit_type": "sample_to_sample",
            "unit_name": "full_response",
            "target_token_indices": [int(x) for x in state.response_positions],
            "target_tokens": [
                _decode_token(tokenizer, int(state.test_batch["input_ids"][0, t].item()))
                for t in state.response_positions
            ],
            "coarse_scoring": state.coarse_scoring,
        },
        "coarse_ranking": [
            {
                "rank": rank,
                "train_sample_id": int(idx),
                "coarse_cos_sim": float(score),
            }
            for rank, (idx, score) in enumerate(state.coarse_ranked, start=1)
        ],
    }
    _write_json(state.coarse_output_path, checkpoint)


def _prepare_test_state(
    *,
    test_index: int,
    test_sample: dict,
    model,
    tokenizer,
    convert_to_chatml,
    base_collator,
    accelerator,
    train_loader,
    prescreen_sketch_cache,
    lm_head_device,
    idx_to_row: dict[int, int],
    output_dir: str,
    use_generated_response: bool,
    generation_limit: int,
    max_output_tokens: int,
    prescreen_max_seq_len: int | None,
    prescreen_sketch_dim: int,
    prescreen_sketch_seed: int,
    oracle_limit: int | None,
    oracle_include_method_top: int,
    seed: int,
    oracle_mode: str,
    logit_chunk_size: int,
) -> TestOracleState:
    task_id = test_sample.get("task_id") or f"test{test_index}"
    print(f"[batched-data] preparing test {test_index} ({task_id})", flush=True)
    test_batch, prompt_len, test_meta = build_test_batch(
        model,
        tokenizer,
        test_sample,
        convert_to_chatml,
        base_collator,
        accelerator,
        use_generated_response=use_generated_response,
        generation_limit=generation_limit,
    )
    test_batch = {
        k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
        for k, v in test_batch.items()
    }
    semantic_target_positions = _valid_target_positions(
        tokenizer,
        test_batch["input_ids"][0],
        prompt_len,
        int(max_output_tokens),
    )
    response_positions = _response_label_positions(test_batch)
    if not response_positions:
        raise RuntimeError(f"No labeled response tokens for test index {test_index}.")

    if prescreen_sketch_cache is not None:
        coarse_scoring = "cached_lm_head_tensor_sketch"
        test_sketch = _lm_head_sketch_for_positions(
            model,
            test_batch,
            response_positions,
            accelerator.device,
            sketch_dim=prescreen_sketch_dim,
            sketch_seed=prescreen_sketch_seed,
        )
        coarse_scores = _score_prescreen_sketch_cache(
            test_sketch,
            prescreen_sketch_cache,
            accelerator.device,
        )
        del test_sketch
    else:
        coarse_scoring = "exact_lm_head_gradient"
        test_grad = _lm_head_grad_for_positions(model, test_batch, response_positions, accelerator.device)
        test_grad_flat = test_grad.reshape(-1).to(lm_head_device).detach()
        local_scores = _screen_training_set(
            model,
            test_grad_flat,
            train_loader,
            accelerator.device,
            lm_head_device,
            desc=f"Data coarse scoring ({task_id})",
            max_seq_len=prescreen_max_seq_len,
        )
        coarse_scores = _gather_scores(accelerator, local_scores, accelerator.device)
        del test_grad, test_grad_flat
    if not coarse_scores:
        raise RuntimeError(f"Data coarse scoring produced no scores for test index {test_index}.")

    coarse_ranked = nlargest(len(coarse_scores), coarse_scores, key=lambda x: x[1])
    oracle_universe = _select_oracle_universe(
        coarse_ranked,
        limit=oracle_limit,
        include_method_top=oracle_include_method_top,
        seed=seed,
    )
    oracle_universe = [idx for idx in oracle_universe if idx in idx_to_row]

    data_dir = os.path.join(output_dir, "data")
    coarse_dir = os.path.join(output_dir, "data_coarse")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(coarse_dir, exist_ok=True)
    state = TestOracleState(
        test_index=int(test_index),
        task_id=str(task_id),
        output_path=os.path.join(data_dir, f"{task_id}_data.json"),
        coarse_output_path=os.path.join(coarse_dir, f"{task_id}_data_coarse.json"),
        test_batch=test_batch,
        test_meta=test_meta,
        prompt_len=int(prompt_len),
        semantic_target_positions=semantic_target_positions,
        response_positions=response_positions,
        coarse_ranked=coarse_ranked,
        coarse_scoring=coarse_scoring,
        oracle_universe=oracle_universe,
        method_score_by_id={int(idx): float(score) for idx, score in coarse_ranked},
        base_losses=torch.empty(0),
    )
    _write_coarse_checkpoint(state, tokenizer)
    print(f"[batched-data] wrote coarse checkpoint {state.coarse_output_path}", flush=True)

    if oracle_mode == "hidden":
        hidden_cpu, labels_cpu, base_losses = _cache_response_hidden(
            model,
            test_batch,
            response_positions,
            device=accelerator.device,
            lm_head_device=lm_head_device,
            logit_chunk_size=logit_chunk_size,
        )
        state.hidden_cpu = hidden_cpu
        state.labels_cpu = labels_cpu
        state.base_losses = base_losses
    else:
        state.base_losses = _target_token_losses(
            model,
            test_batch,
            response_positions,
            accelerator.device,
        )
    return state


def _record_effect(state: TestOracleState, train_idx: int, changed_losses: torch.Tensor, effect_reduction: str) -> None:
    token_effects = changed_losses - state.base_losses
    if effect_reduction == "sum":
        effect = float(token_effects.sum().item())
    elif effect_reduction == "max":
        effect = float(token_effects.max().item())
    elif effect_reduction == "mean":
        effect = float(token_effects.mean().item())
    else:
        raise ValueError(f"Unsupported effect reduction: {effect_reduction}")
    state.oracle_effects[int(train_idx)] = effect
    state.oracle_token_effects[int(train_idx)] = [float(x) for x in token_effects.tolist()]


def _run_batched_oracle(
    *,
    states: list[TestOracleState],
    model,
    train_ds,
    collator,
    idx_to_row: dict[int, int],
    accelerator,
    oracle_mode: str,
    logit_chunk_size: int,
    unlearn_lr: float,
    normalize_unlearn_grad: bool,
    effect_reduction: str,
) -> None:
    union = sorted({idx for state in states for idx in state.oracle_universe})
    print(
        f"[batched-data] one-step unlearning oracle over {len(union)} unique train samples "
        f"for {len(states)} test samples.",
        flush=True,
    )
    state_by_train: dict[int, list[TestOracleState]] = {}
    for state in states:
        for idx in state.oracle_universe:
            state_by_train.setdefault(int(idx), []).append(state)

    for train_idx in tqdm(union, desc="Batched data unlearning oracle", leave=False):
        affected_states = state_by_train.get(int(train_idx), [])
        train_batch = None
        grad = None
        try:
            train_batch = _single_train_batch_from_dataset(
                train_ds,
                collator,
                int(train_idx),
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
            _clear_cuda_after_oom()
            print(
                f"[WARN] Batched data oracle: skipping train sample {int(train_idx)} "
                "after CUDA OOM while computing train gradient.",
                flush=True,
            )
            for state in affected_states:
                state.skipped_oracle_oom.append(int(train_idx))
            del train_batch, grad
            continue

        if oracle_mode == "hidden":
            delta = None
            try:
                delta = _apply_lm_head_ascent_update(
                    model,
                    grad,
                    lr=unlearn_lr,
                    normalize=normalize_unlearn_grad,
                )
                for state in affected_states:
                    try:
                        changed_losses = _cached_hidden_losses_retry(
                            model,
                            state.hidden_cpu,
                            state.labels_cpu,
                            delta=None,
                            device=delta.device,
                            chunk_size=logit_chunk_size,
                        )
                    except (torch.OutOfMemoryError, RuntimeError) as exc:
                        if not _is_cuda_alloc_error(exc):
                            raise
                        _clear_cuda_after_oom()
                        state.skipped_oracle_oom.append(int(train_idx))
                        print(
                            f"[WARN] Batched data oracle: skipping train sample {int(train_idx)} "
                            f"for test {state.task_id} after CUDA OOM while scoring cached hidden effect.",
                            flush=True,
                        )
                        continue
                    _record_effect(state, int(train_idx), changed_losses, effect_reduction)
                    del changed_losses
            finally:
                if delta is not None:
                    _restore_lm_head_ascent_update(model, delta)
        else:
            delta = None
            try:
                delta = _apply_lm_head_ascent_update(
                    model,
                    grad,
                    lr=unlearn_lr,
                    normalize=normalize_unlearn_grad,
                )
                for state in affected_states:
                    try:
                        changed_losses = _target_token_losses(
                            model,
                            state.test_batch,
                            state.response_positions,
                            accelerator.device,
                        )
                    except (torch.OutOfMemoryError, RuntimeError) as exc:
                        if not _is_cuda_alloc_error(exc):
                            raise
                        _clear_cuda_after_oom()
                        state.skipped_oracle_oom.append(int(train_idx))
                        print(
                            f"[WARN] Batched data oracle: skipping train sample {int(train_idx)} "
                            f"for test {state.task_id} after CUDA OOM while scoring test effect.",
                            flush=True,
                        )
                        continue
                    _record_effect(state, int(train_idx), changed_losses, effect_reduction)
                    del changed_losses
            finally:
                if delta is not None:
                    _restore_lm_head_ascent_update(model, delta)

        del train_batch, grad
        torch.cuda.empty_cache()


def _finalize_state_report(
    *,
    state: TestOracleState,
    model,
    tokenizer,
    train_ds,
    collator,
    idx_to_row: dict[int, int],
    accelerator,
    train_size: int,
    method_top_ks: tuple[int, ...],
    group_k_values: tuple[int, ...],
    oracle_top_ms: tuple[int, ...],
    unlearn_lr: float,
    normalize_unlearn_grad: bool,
    effect_reduction: str,
    prescreen_max_seq_len: int | None,
    prescreen_sketch_dim: int,
    prescreen_sketch_seed: int,
    prescreen_sketch_cache_dir: str,
    oracle_mode: str,
) -> None:
    if not state.oracle_effects:
        raise RuntimeError(f"No oracle scores produced for {state.task_id}.")

    method_ranked = [
        int(idx)
        for idx, _ in state.coarse_ranked
        if int(idx) in state.oracle_effects
    ]
    oracle_ranked = sorted(state.oracle_effects, key=lambda x: state.oracle_effects[x], reverse=True)
    group_effects = _evaluate_data_group_unlearning_effects(
        model,
        state.test_batch,
        train_ds,
        collator,
        idx_to_row,
        state.response_positions,
        method_ranked,
        group_k_values,
        state.base_losses,
        accelerator.device,
        unlearn_lr=unlearn_lr,
        normalize_unlearn_grad=normalize_unlearn_grad,
        effect_reduction=effect_reduction,
    )
    metrics = {}
    for method_k in method_top_ks:
        kk = min(int(method_k), len(method_ranked))
        if kk <= 0:
            continue
        metrics[f"ndcg@{kk}"] = _ranking_ndcg(method_ranked, state.oracle_effects, kk)
        for oracle_m in oracle_top_ms:
            mm = min(int(oracle_m), len(oracle_ranked))
            if mm <= 0:
                continue
            metrics[f"recall@{kk}_oracle_top{mm}"] = _recall_at(method_ranked, oracle_ranked, kk, mm)
    for group in group_effects:
        k = int(group["k"])
        effect = float(group["data_effect"])
        metrics[f"group_data_effect@{k}"] = effect
        metrics[f"group_positive_data_effect@{k}"] = 1.0 if effect > 0.0 else 0.0

    top_n = max(max(method_top_ks), max(oracle_top_ms), max(group_k_values or (0,)))
    sample_to_sample = {
        "unit_type": "sample_to_sample",
        "unit_name": "full_response",
        "target_token_indices": [int(x) for x in state.response_positions],
        "target_tokens": [
            _decode_token(tokenizer, int(state.test_batch["input_ids"][0, t].item()))
            for t in state.response_positions
        ],
        "oracle": {
            "unlearning_update": "theta <- theta + eta * normalized_grad_train"
            if normalize_unlearn_grad
            else "theta <- theta + eta * grad_train",
            "unlearn_lr": float(unlearn_lr),
            "normalize_train_gradient": bool(normalize_unlearn_grad),
            "effect_reduction": effect_reduction,
            "candidate_universe_size": len(state.oracle_universe),
            "scored_candidate_count": len(state.oracle_effects),
            "skipped_oom_count": len(state.skipped_oracle_oom),
            "skipped_oom_train_sample_ids": state.skipped_oracle_oom[:50],
            "loss_scope": "all labeled response tokens in the test sample",
            "coarse_scoring": state.coarse_scoring,
            "oracle_mode": oracle_mode,
            "prescreen_sketch_dim": int(prescreen_sketch_dim)
            if state.coarse_scoring == "cached_lm_head_tensor_sketch"
            else 0,
            "prescreen_sketch_seed": int(prescreen_sketch_seed)
            if state.coarse_scoring == "cached_lm_head_tensor_sketch"
            else None,
            "base_target_losses": [float(x) for x in state.base_losses.tolist()],
        },
        "metrics": metrics,
        "group_effects": group_effects,
        "method_top": [
            {
                "rank": rank,
                "train_sample_id": int(idx),
                "coarse_cos_sim": float(state.method_score_by_id[idx]),
                "oracle_effect": float(state.oracle_effects.get(idx, 0.0)),
            }
            for rank, idx in enumerate(method_ranked[:top_n], start=1)
        ],
        "oracle_top": [
            {
                "rank": rank,
                "train_sample_id": int(idx),
                "oracle_effect": float(state.oracle_effects[idx]),
                "coarse_cos_sim": float(state.method_score_by_id.get(idx, 0.0)),
                "token_effects": state.oracle_token_effects[idx],
            }
            for rank, idx in enumerate(oracle_ranked[:top_n], start=1)
        ],
    }

    report = {
        "experiment_meta": {
            "test_sample_index": int(state.test_index),
            "task_id": state.task_id,
            "target_token_indices": [int(x) for x in state.semantic_target_positions],
            "target_tokens": [
                _decode_token(tokenizer, int(state.test_batch["input_ids"][0, t].item()))
                for t in state.semantic_target_positions
            ],
            "prompt_len": int(state.prompt_len),
            "train_size": int(train_size),
            "evaluation_protocol": {
                "data_oracle": "one-step gradient-ascent unlearning target CE increase",
                "data_sample_to_sample": "rank train samples by effect on whole response CE",
                "data_unlearning_param_space": "lm_head.weight",
                "data_unlearning_direction": "theta <- theta + eta * grad_train_loss",
                "batched_oracle": "one train unlearning update scored against multiple test samples",
            },
            "config": {
                "data_method_k_values": method_top_ks,
                "data_group_k_values": group_k_values,
                "data_oracle_m_values": oracle_top_ms,
                "data_oracle_limit": len(state.oracle_universe),
                "data_oracle_include_method_top": None,
                "data_granularity": "sample_to_sample",
                "unlearn_lr": float(unlearn_lr),
                "normalize_unlearn_grad": bool(normalize_unlearn_grad),
                "data_effect_reduction": effect_reduction,
                "prescreen_max_seq_len": prescreen_max_seq_len,
                "prescreen_sketch_dim": int(prescreen_sketch_dim),
                "prescreen_sketch_seed": int(prescreen_sketch_seed),
                "prescreen_sketch_cache_dir": prescreen_sketch_cache_dir,
                "oracle_mode": oracle_mode,
            },
        },
        "test_sample_baseline": state.test_meta,
        "data_coarse_attribution": {
            "granularity": "sample_to_sample",
            "sample_to_token": [],
            "sample_to_sample": sample_to_sample,
        },
    }
    _write_json(state.output_path, report)
    print(f"[batched-data] wrote {state.output_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch data-attribution oracle by reusing each train-sample unlearning update."
    )
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--train-data", type=str, default="sft_train.jsonl")
    parser.add_argument("--test-data", type=str, default="sft_test.jsonl")
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="attribution_results_batched_data")
    parser.add_argument("--test-batch-size", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=20)
    parser.add_argument("--generation-limit", type=int, default=128)
    parser.add_argument("--use-ground-truth-response", action="store_true")
    parser.add_argument("--prescreen-batch-size", type=int, default=1)
    parser.add_argument("--prescreen-max-seq-len", type=int, default=3000)
    parser.add_argument("--prescreen-sketch-dim", type=int, default=PRESCREEN_SKETCH_DIM)
    parser.add_argument("--prescreen-sketch-seed", type=int, default=PRESCREEN_SKETCH_SEED)
    parser.add_argument("--prescreen-sketch-cache-dir", type=str, default=PRESCREEN_SKETCH_CACHE_DIR)
    parser.add_argument("--no-prescreen-sketch-cache", action="store_true")
    parser.add_argument("--data-method-k-values", type=str, default="10,50,100")
    parser.add_argument(
        "--data-group-k-values",
        type=str,
        default="",
        help="Comma-separated method top-k prefixes to unlearn as a group, e.g. 1,3,5,10.",
    )
    parser.add_argument("--data-oracle-m-values", type=str, default="10,20,50")
    parser.add_argument("--data-oracle-limit", type=int, default=None)
    parser.add_argument("--data-oracle-include-method-top", type=int, default=100)
    parser.add_argument("--unlearn-lr", type=float, default=0.01)
    parser.add_argument("--no-normalize-unlearn-grad", action="store_true")
    parser.add_argument("--data-effect-reduction", choices=["mean", "sum", "max"], default="mean")
    parser.add_argument("--oracle-mode", choices=["auto", "hidden", "forward"], default="auto")
    parser.add_argument("--logit-chunk-size", type=int, default=16)
    parser.add_argument("--max-gpu-memory", type=str, default=None)
    parser.add_argument("--attn-implementation", type=str, default="eager")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise RuntimeError("Batched data attribution evaluator expects a single Accelerator process.")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "reports"), exist_ok=True)

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
    test_indices = _parse_indices(args.indices, args.start_idx, args.end_idx, len(test_samples))

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
    )
    collator = CustomCollator(base_collator)

    print("[batched-data] building train dataset...", flush=True)
    train_ds = build_train_dataset(train_samples, convert_to_chatml)
    idx_to_row = {int(train_ds[i]["sample_index"]): i for i in range(len(train_ds))}
    train_loader = torch.utils.data.DataLoader(
        DatasetWrapper(train_ds),
        batch_size=max(1, int(args.prescreen_batch_size)),
        collate_fn=collator,
    )
    train_loader = accelerator.prepare(train_loader)

    prescreen_max_seq_len = None if args.prescreen_max_seq_len <= 0 else int(args.prescreen_max_seq_len)
    prescreen_sketch_dim = 0 if args.no_prescreen_sketch_cache else int(args.prescreen_sketch_dim or 0)
    prescreen_sketch_cache = None
    if prescreen_sketch_dim > 0:
        prescreen_sketch_cache = _load_or_build_prescreen_sketch_cache(
            model,
            train_ds,
            train_loader,
            accelerator,
            max_seq_len=prescreen_max_seq_len,
            sketch_dim=prescreen_sketch_dim,
            sketch_seed=int(args.prescreen_sketch_seed),
            cache_dir=args.prescreen_sketch_cache_dir,
        )

    filtered_params = [p for n, p in model.named_parameters() if lm_head_filter(n, p)]
    lm_head_device = filtered_params[0].device if filtered_params else accelerator.device
    tied_embeddings = _lm_head_tied_to_input_embeddings(model)
    if args.oracle_mode == "auto":
        oracle_mode = "forward" if tied_embeddings else "hidden"
    else:
        oracle_mode = args.oracle_mode
    if oracle_mode == "hidden" and tied_embeddings:
        raise RuntimeError(
            "Cannot use --oracle-mode hidden because lm_head.weight is tied to input embeddings. "
            "Use --oracle-mode forward."
        )
    print(
        f"[batched-data] oracle_mode={oracle_mode}, tied_embeddings={tied_embeddings}, "
        f"lm_head_device={lm_head_device}",
        flush=True,
    )

    method_top_ks = parse_int_tuple(args.data_method_k_values) or DEFAULT_DATA_METHOD_K
    group_k_values = parse_int_tuple(args.data_group_k_values) or DEFAULT_DATA_GROUP_K
    oracle_top_ms = parse_int_tuple(args.data_oracle_m_values) or DEFAULT_DATA_ORACLE_M
    status_path = os.path.join(args.output_dir, "reports", "batch_status.tsv")
    with open(status_path, "w", encoding="utf-8") as f:
        f.write("sample_index\ttask_id\tstage\tstatus\texit_code\toutput_file\tlog_file\n")

    for group in _chunks(test_indices, int(args.test_batch_size)):
        states = []
        for test_index in group:
            test_sample = test_samples[int(test_index)]
            task_id = test_sample.get("task_id") or f"test{test_index}"
            try:
                state = _prepare_test_state(
                    test_index=int(test_index),
                    test_sample=test_sample,
                    model=model,
                    tokenizer=tokenizer,
                    convert_to_chatml=convert_to_chatml,
                    base_collator=base_collator,
                    accelerator=accelerator,
                    train_loader=train_loader,
                    prescreen_sketch_cache=prescreen_sketch_cache,
                    lm_head_device=lm_head_device,
                    idx_to_row=idx_to_row,
                    output_dir=args.output_dir,
                    use_generated_response=not args.use_ground_truth_response,
                    generation_limit=int(args.generation_limit),
                    max_output_tokens=int(args.max_output_tokens),
                    prescreen_max_seq_len=prescreen_max_seq_len,
                    prescreen_sketch_dim=prescreen_sketch_dim,
                    prescreen_sketch_seed=int(args.prescreen_sketch_seed),
                    oracle_limit=args.data_oracle_limit,
                    oracle_include_method_top=int(args.data_oracle_include_method_top),
                    seed=int(args.seed),
                    oracle_mode=oracle_mode,
                    logit_chunk_size=int(args.logit_chunk_size),
                )
            except ValueError as exc:
                message = str(exc)
                if (
                    "No labeled response tokens found" not in message
                    and "marker sequence not found" not in message
                ):
                    raise
                output_path = os.path.abspath(
                    os.path.join(args.output_dir, "data", f"{task_id}_data.json")
                )
                print(
                    f"[WARN] Skipping test {int(test_index)} ({task_id}): {message}",
                    flush=True,
                )
                with open(status_path, "a", encoding="utf-8") as status_f:
                    status_f.write(
                        f"{int(test_index)}\t{task_id}\tdata\tskipped\t0\t{output_path}\t\n"
                    )
                continue
            states.append(state)

        if not states:
            torch.cuda.empty_cache()
            continue

        _run_batched_oracle(
            states=states,
            model=model,
            train_ds=train_ds,
            collator=collator,
            idx_to_row=idx_to_row,
            accelerator=accelerator,
            oracle_mode=oracle_mode,
            logit_chunk_size=int(args.logit_chunk_size),
            unlearn_lr=float(args.unlearn_lr),
            normalize_unlearn_grad=not args.no_normalize_unlearn_grad,
            effect_reduction=args.data_effect_reduction,
        )

        with open(status_path, "a", encoding="utf-8") as status_f:
            for state in states:
                _finalize_state_report(
                    state=state,
                    model=model,
                    tokenizer=tokenizer,
                    train_ds=train_ds,
                    collator=collator,
                    idx_to_row=idx_to_row,
                    accelerator=accelerator,
                    train_size=len(train_samples),
                    method_top_ks=method_top_ks,
                    group_k_values=group_k_values,
                    oracle_top_ms=oracle_top_ms,
                    unlearn_lr=float(args.unlearn_lr),
                    normalize_unlearn_grad=not args.no_normalize_unlearn_grad,
                    effect_reduction=args.data_effect_reduction,
                    prescreen_max_seq_len=prescreen_max_seq_len,
                    prescreen_sketch_dim=prescreen_sketch_dim,
                    prescreen_sketch_seed=int(args.prescreen_sketch_seed),
                    prescreen_sketch_cache_dir=args.prescreen_sketch_cache_dir,
                    oracle_mode=oracle_mode,
                )
                status_f.write(
                    f"{state.test_index}\t{state.task_id}\tdata\tdone\t0\t"
                    f"{os.path.abspath(state.output_path)}\t\n"
                )

        del states
        torch.cuda.empty_cache()

    report_json = os.path.join(args.output_dir, "reports", "attribution_batch_report.json")
    report_md = os.path.join(args.output_dir, "reports", "attribution_batch_report.md")
    report = build_report(args.output_dir, status_path)
    write_report_json(report_json, report)
    _write_markdown(report_md, report)
    print(f"[batched-data] wrote {report_json}", flush=True)
    print(f"[batched-data] wrote {report_md}", flush=True)


if __name__ == "__main__":
    main()
