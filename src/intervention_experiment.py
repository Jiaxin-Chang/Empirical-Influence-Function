import os
import json
import torch
import torch.nn.functional as F
from functools import partial
from heapq import nlargest
from accelerate import Accelerator
from tqdm import tqdm

from src.NIF import (
    load_model_and_tokenizer,
    load_samples_from_formal_jsonl,
    build_train_dataset,
    build_single_sample_dataset,
    NewInferenceFunction,
    DatasetWrapper,
    CustomCollator,
    round_floats,
    _find_subseq_start,
)
from src.process_data import process_func_chatml
from src.loss import (
    compute_correlation_second_order_gradient,
    compute_full_saliency_vector,
    compute_gradients,
)
from transformers import DataCollatorForSeq2Seq, set_seed

# ====== CONFIGURATION ======
SEED = 42
SELECTED_TEST_SAMPLE_INDEX = 58
TOKEN_INDEX_TO_RETRIEVE = 703  # The "first wrong token" we are investigating (single-token mode)

TOP_K_PROMPT_TOKENS = 4        # How many test correlation features to extract
TOP_K_TRAIN_SAMPLES = 10       # How many top train samples from coarse screening
TOP_TARGETS = 3                # How many response tokens to scan per train sample
TOP_K_SOURCE_PER_TARGET = 3    # Top source tokens per target (includes response-internal tokens)
CONTEXT_WINDOW_SIZE = 3        # Tokens shown on each side of source/target for annotation

# All-tokens mode parameters
MAX_OUTPUT_TOKENS = 40         # Max response tokens to analyze in all-tokens mode
# Coarse-screening chunk size for all-tokens mode:
#   Test-token CE gradients are processed in groups of this size so that only
#   COARSE_CHUNK_SIZE gradient vectors (each ≈ lm_head size) live on GPU at once.
#   Training set is scanned ceil(N_tok / COARSE_CHUNK_SIZE) times.
#   Set to 1 for minimal GPU memory (sequential), or larger for fewer scans.
COARSE_CHUNK_SIZE = 4

# Training samples longer than this are skipped in all gradient-based screenings
# to prevent OOM during eager attention (O(N²) memory) on very long sequences.
SEQUENCE_LENGTH_LIMIT = 3000

# Token strings (after strip) that carry no semantic content and should be skipped
# in all-tokens mode. Single non-alphanumeric characters are also skipped.
_TRIVIAL_STRIPPED = {"{", "}", "(", ")", "[", "]", ",", ";"}


def lm_head_filter(name, param):
    """Only select the LM head weight — a single dense matrix that
    projects the last hidden state to vocabulary logits.
    This gives a compact, task-agnostic feature for every token prediction.
    """
    return name == "lm_head.weight"


def is_trivial_token(tokenizer, token_id: int) -> bool:
    """Return True for tokens that carry no semantic meaning.

    Skips: pure whitespace, lone punctuation ({, }, (, ), [, ], comma, semicolon),
    and any single non-alphanumeric/non-underscore character.
    """
    tok_str = tokenizer.decode([token_id])
    stripped = tok_str.strip()
    if not stripped:
        return True
    if stripped in _TRIVIAL_STRIPPED:
        return True
    if len(stripped) == 1 and not (stripped.isalnum() or stripped == "_"):
        return True
    return False


def find_first_valid_token_index(tokenizer, input_ids_tensor, start_idx):
    """
    Skip formatting characters like \\n, \\t, spaces, {, } to find the
    first token that carries actual semantic meaning.
    """
    valid_token_index = start_idx
    input_len = input_ids_tensor.size(1)
    while valid_token_index < input_len:
        tok_id = input_ids_tensor[0, valid_token_index].item()
        tok_str = tokenizer.decode([tok_id])
        if tok_str.strip() not in ["", "{", "}"]:
            break
        valid_token_index += 1
    return valid_token_index


def get_context_window(tokenizer, input_ids_1d, idx, window=CONTEXT_WINDOW_SIZE):
    """
    Return a list of token strings centered on `idx`.
    The focal token is wrapped in →[...]← for easy visual identification during annotation.
    """
    seq_len = input_ids_1d.size(0)
    tokens = []
    for i in range(max(0, idx - window), min(seq_len, idx + window + 1)):
        tok_str = tokenizer.decode([input_ids_1d[i].item()])
        tokens.append(f"→[{tok_str}]←" if i == idx else tok_str)
    return tokens


def _flat_grad_on_device(
    grads: list,
    filtered_params: list,
    target_device: torch.device,
) -> torch.Tensor:
    """Flatten and concatenate gradients onto `target_device` without moving to CPU."""
    return torch.cat([
        g.reshape(-1).to(target_device) if g is not None
        else torch.zeros(p.numel(), dtype=p.dtype, device=target_device)
        for g, p in zip(grads, filtered_params)
    ])


def _gather_scores(
    accelerator,
    local_scores: list[tuple[int, float]],
    device: torch.device,
) -> list[tuple[int, float]]:
    """Gather (train_idx, score) pairs from all processes into the main process.

    Handles variable-length lists per process by padding with sentinel -1 indices.
    In single-process mode this is a no-op.
    """
    if accelerator.num_processes == 1:
        return local_scores

    n = len(local_scores)
    all_lens = accelerator.gather(torch.tensor(n, device=device, dtype=torch.long))
    max_len = int(all_lens.max().item())

    padded_scores = torch.full((max_len,), float("nan"), device=device, dtype=torch.float32)
    padded_indices = torch.full((max_len,), -1, device=device, dtype=torch.long)
    if n > 0:
        padded_scores[:n] = torch.tensor([s for _, s in local_scores], device=device, dtype=torch.float32)
        padded_indices[:n] = torch.tensor([i for i, _ in local_scores], device=device, dtype=torch.long)

    all_scores = accelerator.gather(padded_scores)
    all_indices = accelerator.gather(padded_indices)

    valid = all_indices != -1
    return list(zip(
        all_indices[valid].cpu().tolist(),
        all_scores[valid].cpu().tolist(),
    ))


def _screen_training_set(
    model,
    test_ce_grad: torch.Tensor,   # flat tensor already on lm_head_device
    filtered_params: list,
    train_loader,
    accel_device: torch.device,   # accelerator.device for batch loading
    lm_head_device: torch.device, # device where lm_head lives (grad computed here)
    desc: str = "Scanning Train Samples",
    allowed_indices: set | None = None,
) -> list[tuple[int, float]]:
    """Compute cosine similarity on GPU (lm_head_device). Returns LOCAL scores for this process."""
    sample_scores: list[tuple[int, float]] = []
    empty_ignored = torch.tensor([], device=accel_device)

    for batch in tqdm(train_loader, desc=desc, leave=False):
        batch_device = {k: v.to(accel_device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        train_idx = int(batch_device["sample_index"].item())

        if allowed_indices is not None and train_idx not in allowed_indices:
            del batch_device
            continue

        # Skip sequences that would cause OOM during eager attention
        if batch_device["input_ids"].size(1) > SEQUENCE_LENGTH_LIMIT:
            del batch_device
            continue

        train_ce_grads = compute_gradients(
            model=model,
            batch=batch_device,
            param_filter_fn=lm_head_filter,
            device=accel_device,
            ignored_token_ids=empty_ignored,
        )
        # Keep on lm_head_device — no CPU round-trip
        flat_train_ce = _flat_grad_on_device(train_ce_grads, filtered_params, lm_head_device)
        score = F.cosine_similarity(test_ce_grad, flat_train_ce, dim=0).item()
        sample_scores.append((train_idx, score))

        del train_ce_grads, flat_train_ce, batch_device

    return sample_scores


def run_causal_intervention_experiment(all_tokens: bool = False):
    accelerator = Accelerator()
    set_seed(SEED)

    model, tokenizer = load_model_and_tokenizer()
    param_filter = lm_head_filter

    convert_to_chatml = partial(process_func_chatml, tokenizer=tokenizer)

    train_samples = load_samples_from_formal_jsonl("sft_train.jsonl")
    test_samples  = load_samples_from_formal_jsonl("sft_test.jsonl")

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding=True, label_pad_token_id=-100, return_tensors="pt",
    )
    collator = CustomCollator(base_collator)

    train_ds = build_train_dataset(train_samples, convert_to_chatml)
    train_loader = torch.utils.data.DataLoader(
        DatasetWrapper(train_ds), batch_size=1, collate_fn=collator,
    )
    train_loader = accelerator.prepare(train_loader)

    infer_fw = NewInferenceFunction(
        model=model, tokenizer=tokenizer,
        train_loader=train_loader, accelerator=accelerator,
        param_filter_fn=param_filter, top_k=20,
    )

    # lm_head might live on a different device under device_map="auto"
    filtered_params = [p for n, p in model.named_parameters() if param_filter(n, p)]
    lm_head_device  = filtered_params[0].device if filtered_params else accelerator.device

    marker_ids = tuple(tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))

    # ── Build full test sequence (prompt + generated response) ──────────────
    test_ds        = build_single_sample_dataset(test_samples[SELECTED_TEST_SAMPLE_INDEX], convert_to_chatml)
    raw_test_batch = base_collator([test_ds[0]])
    raw_test_batch = {k: v.to(accelerator.device) for k, v in raw_test_batch.items()}

    # ── Lightweight generate helper (for all-tokens mode) ───────────────────
    # infer_fw.infer() runs a heavy per-token saliency loop on the ground-truth
    # response (can take 100+ seconds) that all-tokens mode never consumes.
    # This helper skips that loop and generates only MAX_OUTPUT_TOKENS tokens,
    # with a guard against sequences longer than the model's position limit.
    MAX_POSITION_EMBEDDINGS = getattr(
        model.config, "max_position_embeddings", 32768
    )

    def _lightweight_gen(raw_batch) -> dict:
        """Generate response without expensive saliency computation."""
        input_ids      = raw_batch["input_ids"].to(accelerator.device)
        attention_mask = raw_batch["attention_mask"].to(accelerator.device)

        prompt_start = _find_subseq_start(input_ids[0], marker_ids) + 3

        # Truncate prompt to model's position limit to avoid RoPE overflow
        safe_prompt_end = min(int(prompt_start), MAX_POSITION_EMBEDDINGS - MAX_OUTPUT_TOKENS - 1)
        if safe_prompt_end < prompt_start:
            print(f"  [Warning] Prompt length {prompt_start} > safe limit "
                  f"{safe_prompt_end}; truncating for generation.")
        trim_ids  = input_ids[:, :safe_prompt_end]
        trim_mask = attention_mask[:, :safe_prompt_end]

        with torch.no_grad():
            gen_out = model.generate(
                input_ids=trim_ids,
                attention_mask=trim_mask,
                max_new_tokens=MAX_OUTPUT_TOKENS,
                do_sample=False,
                eos_token_id=[
                    tokenizer.eos_token_id,
                    tokenizer.pad_token_id,
                ],
                pad_token_id=tokenizer.pad_token_id,
            )

        gen_ids      = gen_out if isinstance(gen_out, torch.Tensor) else gen_out.sequences
        pred_tok_ids = gen_ids[0, safe_prompt_end:].tolist()
        full_gen_ids = gen_ids[0].tolist()
        gt_ids       = input_ids[0, :int(attention_mask[0].sum().item())].tolist()

        # Free GPU tensors before returning — only Python lists are needed downstream
        del trim_ids, trim_mask, gen_out, gen_ids
        torch.cuda.empty_cache()

        return {
            "target_idx":          [prompt_start],
            "pred_ids":            [pred_tok_ids],
            "pred_full_tokens":    [tokenizer.convert_ids_to_tokens(full_gen_ids)],
            "full_tokens":         [tokenizer.convert_ids_to_tokens(gt_ids)],
        }

    infer_fw.model.eval()

    # Always use the lightweight helper for this first call: the only fields consumed from
    # gen_result are target_idx / pred_ids / pred_full_tokens / full_tokens.
    # saliency_original is never read from gen_result in either mode — the single-token
    # mode's saliency is obtained from a separate infer_fw.infer(test_batch, ...) call later.
    gen_result = _lightweight_gen(raw_test_batch)

    prompt_len = int(gen_result["target_idx"][0])
    prompt_ids = raw_test_batch["input_ids"][0, :prompt_len]
    pred_ids   = torch.tensor(gen_result["pred_ids"][0], device=prompt_ids.device, dtype=prompt_ids.dtype)
    new_input_ids      = torch.cat([prompt_ids, pred_ids], dim=0).unsqueeze(0)
    new_attention_mask = torch.ones_like(new_input_ids)
    new_labels         = new_input_ids.clone()
    new_labels[:, :prompt_len] = -100

    test_batch = {
        "input_ids":      new_input_ids,
        "attention_mask": new_attention_mask,
        "labels":         new_labels,
    }
    seq_len = test_batch["input_ids"].size(1)

    # ════════════════════════════════════════════════════════════════════════════
    # Helper: Stage 3 processing for one train sample
    # Returns (train_sample_detail_dict, pair_records_list)
    # detail is None if the sample must be skipped (too long / no marker).
    # If the sample was already computed (cache hit), pass existing detail + candidate_pairs.
    # ════════════════════════════════════════════════════════════════════════════
    def _process_train_sample_stage3(
        train_idx: int,
        coarse_score: float,
        test_corr_features: dict,
        target_tok_text_for_record: str,
        target_tok_idx_for_record: int,
        pair_id_prefix: str,
        pair_id_start: int,
        cached_detail: dict | None = None,
    ) -> tuple[dict | None, list, int]:
        """Returns (detail_dict_or_None, pair_records, next_pair_id)."""
        nonlocal model, tokenizer, base_collator, accelerator, param_filter

        if cached_detail is None:
            tr_ds   = build_single_sample_dataset(train_samples[train_idx], convert_to_chatml)
            tr_batch = base_collator([tr_ds[0]])
            tr_batch = {k: v.to(accelerator.device) for k, v in tr_batch.items()}

            if tr_batch["input_ids"].size(1) > SEQUENCE_LENGTH_LIMIT:
                print(f"  Skipping train {train_idx}: sequence too long")
                return None, [], pair_id_start

            try:
                start_sys = _find_subseq_start(tr_batch["input_ids"][0], marker_ids) + 3
            except ValueError:
                print(f"  Skipping train {train_idx}: assistant marker not found")
                return None, [], pair_id_start

            response_start = find_first_valid_token_index(tokenizer, tr_batch["input_ids"], start_sys)
            tr_seq_len     = tr_batch["input_ids"].size(1)
            full_tokens    = tokenizer.convert_ids_to_tokens(tr_batch["input_ids"][0].tolist())
            target_saliencies: dict[int, list[float]] = {}
            candidate_pairs: list[tuple[float, int, int]] = []

            with torch.inference_mode(False):
                for t_offset in range(TOP_TARGETS):
                    t_tr = response_start + t_offset
                    if t_tr >= tr_seq_len:
                        break
                    sal_vec = compute_full_saliency_vector(model, tr_batch, t_tr)
                    target_saliencies[t_tr] = [round(float(s), 6) for s in sal_vec]
                    for s_idx, s_score in nlargest(TOP_K_SOURCE_PER_TARGET, enumerate(sal_vec), key=lambda x: x[1]):
                        candidate_pairs.append((float(s_score), t_tr, int(s_idx)))

            print(f"  Step A: {len(candidate_pairs)} candidate pairs "
                  f"({TOP_TARGETS}t × {TOP_K_SOURCE_PER_TARGET}s each)")

            cached_detail = {
                "full_tokens":        full_tokens,
                "answer_start_index": response_start,
                "coarse_cos_sim":     float(coarse_score),
                "saliencies_by_token": {str(k): v for k, v in target_saliencies.items()},
                "_candidate_pairs":   candidate_pairs,
                "_tr_batch_cpu":      {k: v.cpu() for k, v in tr_batch.items()},
            }
        else:
            # Update coarse score to the maximum seen across test tokens
            if coarse_score > cached_detail["coarse_cos_sim"]:
                cached_detail["coarse_cos_sim"] = float(coarse_score)
            candidate_pairs = cached_detail["_candidate_pairs"]

        # Move tr_batch to device for 2nd-order computation
        tr_batch_gpu = {k: v.to(accelerator.device) for k, v in cached_detail["_tr_batch_cpu"].items()}
        ids_1d       = tr_batch_gpu["input_ids"][0]
        response_start = cached_detail["answer_start_index"]
        pair_counter   = pair_id_start
        pair_records   = []

        with torch.inference_mode(False):
            for saliency_score, t_tr, s_idx in candidate_pairs:
                train_target_tok   = tokenizer.decode([ids_1d[t_tr].item()])
                train_source_tok   = tokenizer.decode([ids_1d[s_idx].item()])
                response_tok_offset = t_tr - response_start

                print(f"  Step B: '{train_source_tok}' -> '{train_target_tok}' "
                      f"(offset={response_tok_offset}, sal={saliency_score:.4f})")

                train_feat = compute_correlation_second_order_gradient(
                    model, tr_batch_gpu,
                    target_idx_in_seq=t_tr, source_idx_in_seq=s_idx,
                    param_filter_fn=param_filter,
                )
                source_ctx = get_context_window(tokenizer, ids_1d, s_idx)
                target_ctx = get_context_window(tokenizer, ids_1d, t_tr)

                for test_p_idx, (test_feat, test_src_text, test_saliency) in test_corr_features.items():
                    # Cosine similarity on lm_head_device (no CPU round-trip)
                    cos_sim = F.cosine_similarity(
                        test_feat.to(lm_head_device),
                        train_feat.to(lm_head_device),
                        dim=0,
                    ).item()

                    pair_records.append({
                        "id":            f"{pair_id_prefix}_{pair_counter:04d}",
                        "cos_sim":       float(cos_sim),
                        "coarse_cos_sim": float(coarse_score),
                        "train_sample_id": train_idx,
                        "test_correlation": {
                            "source_token":       test_src_text,
                            "source_token_index": test_p_idx,
                            "target_token":       target_tok_text_for_record,
                            "target_token_index": target_tok_idx_for_record,
                            "saliency_score":     float(test_saliency),
                        },
                        "train_correlation": {
                            "source_token":         train_source_tok,
                            "source_token_index":   s_idx,
                            "target_token":         train_target_tok,
                            "target_token_index":   t_tr,
                            "saliency_score":       saliency_score,
                            "response_token_offset": response_tok_offset,
                        },
                        "train_context": {
                            "source_context": source_ctx,
                            "target_context": target_ctx,
                        },
                        "annotation": None,
                    })
                    pair_counter += 1

                del train_feat  # free 2nd-order grad before next (t_tr, s_idx) pair

        del tr_batch_gpu
        torch.cuda.empty_cache()
        return cached_detail, pair_records, pair_counter

    # ════════════════════════════════════════════════════════════════════════════
    # Helper: compute test-side CE gradient for one token position
    # ════════════════════════════════════════════════════════════════════════════
    def _test_ce_grad_for_token(tok_idx: int) -> torch.Tensor:
        ce_labels = torch.full_like(test_batch["input_ids"], -100)
        ce_labels[0, tok_idx] = test_batch["input_ids"][0, tok_idx]
        single_tok_batch = {
            "input_ids":      test_batch["input_ids"],
            "attention_mask": test_batch["attention_mask"],
            "labels":         ce_labels,
        }
        grads = compute_gradients(
            model=model, batch=single_tok_batch,
            param_filter_fn=param_filter, device=accelerator.device,
            ignored_token_ids=torch.tensor([], device=accelerator.device),
        )
        return _flat_grad_on_device(grads, filtered_params, lm_head_device).detach()

    # ════════════════════════════════════════════════════════════════════════════
    # ── SINGLE-TOKEN MODE ────────────────────────────────────────────────────
    # ════════════════════════════════════════════════════════════════════════════
    if not all_tokens:
        print("\n=== Stage 1: Extracting test correlation features ===")

        target_idx_tensor = torch.tensor([TOKEN_INDEX_TO_RETRIEVE], device=accelerator.device)
        baseline_res      = infer_fw.infer(test_batch, target_idx=target_idx_tensor)

        target_tok_id   = test_batch["input_ids"][0, TOKEN_INDEX_TO_RETRIEVE].item()
        target_tok_text = tokenizer.decode([target_tok_id])
        baseline_saliency = baseline_res["saliency_original"][0][0]["saliency"]

        top_test_corr = nlargest(TOP_K_PROMPT_TOKENS, enumerate(baseline_saliency), key=lambda x: x[1])
        top_test_correlations = [
            {
                "source_token_index": idx,
                "source_token":       tokenizer.decode([test_batch["input_ids"][0, idx].item()]),
                "target_token_index": TOKEN_INDEX_TO_RETRIEVE,
                "target_token":       target_tok_text,
                "saliency_score":     float(score),
            }
            for idx, score in top_test_corr
        ]

        print("Computing 2nd-order test features...")
        test_corr_features: dict = {}
        with torch.inference_mode(False):
            for item in top_test_correlations:
                p_idx = item["source_token_index"]
                print(f"  '{item['source_token']}' -> '{item['target_token']}' (sal={item['saliency_score']:.4f})")
                feat = compute_correlation_second_order_gradient(
                    model=model, batch=test_batch,
                    target_idx_in_seq=TOKEN_INDEX_TO_RETRIEVE, source_idx_in_seq=p_idx,
                    param_filter_fn=param_filter,
                )
                test_corr_features[p_idx] = (feat.to(lm_head_device), item["source_token"], item["saliency_score"])

        print(f"\n=== Stage 2: Coarse screening (single token {TOKEN_INDEX_TO_RETRIEVE}) ===")
        test_ce_grad  = _test_ce_grad_for_token(TOKEN_INDEX_TO_RETRIEVE)
        local_scores  = _screen_training_set(
            model, test_ce_grad, filtered_params, train_loader,
            accelerator.device, lm_head_device, desc="Stage 2",
        )
        sample_scores  = _gather_scores(accelerator, local_scores, accelerator.device)
        related_samples = nlargest(TOP_K_TRAIN_SAMPLES, sample_scores, key=lambda x: x[1])
        print(f"  Top-{TOP_K_TRAIN_SAMPLES} selected: {[(i, round(s,4)) for i,s in related_samples]}")

        print("\n=== Stage 3: Fine-grained correlation matching ===")
        all_pair_records: list = []
        train_sample_details: dict = {}
        pair_id_counter = 0

        for rank, (train_idx, coarse_score) in enumerate(related_samples):
            print(f"\n--- Train {train_idx}  (rank={rank+1}, coarse={coarse_score:.4f}) ---")
            detail, pairs, pair_id_counter = _process_train_sample_stage3(
                train_idx, coarse_score, test_corr_features,
                target_tok_text, TOKEN_INDEX_TO_RETRIEVE,
                "pair", pair_id_counter,
                cached_detail=train_sample_details.get(str(train_idx)),
            )
            if detail is not None:
                train_sample_details[str(train_idx)] = detail
                all_pair_records.extend(pairs)

        all_pair_records.sort(key=lambda x: x["cos_sim"], reverse=True)

        report_json = {
            "experiment_meta": {
                "test_sample_index":  SELECTED_TEST_SAMPLE_INDEX,
                "target_token_index": TOKEN_INDEX_TO_RETRIEVE,
                "mode":               "single_token",
                "config": {
                    "TOP_K_PROMPT_TOKENS":    TOP_K_PROMPT_TOKENS,
                    "TOP_K_TRAIN_SAMPLES":    TOP_K_TRAIN_SAMPLES,
                    "TOP_TARGETS":            TOP_TARGETS,
                    "TOP_K_SOURCE_PER_TARGET": TOP_K_SOURCE_PER_TARGET,
                    "CONTEXT_WINDOW_SIZE":    CONTEXT_WINDOW_SIZE,
                },
            },
            "test_sample_baseline": {
                "target_token":        target_tok_text,
                "target_token_index":  TOKEN_INDEX_TO_RETRIEVE,
                "full_tokens":         baseline_res["full_tokens"][0],  # prompt + model output (test_batch input)
                "correct_full_tokens": gen_result["full_tokens"][0],    # prompt + ground truth answer
                "top_correlations":    top_test_correlations,
            },
            "correlation_pairs": all_pair_records,
            "train_sample_details": {
                k: {fk: fv for fk, fv in v.items() if not fk.startswith("_")}
                for k, v in train_sample_details.items()
            },
        }

        total = len(all_pair_records)
        print(f"\nTotal pairs: {total}  (top-10 by cos_sim below)")
        for r in all_pair_records[:10]:
            print(f"  [{r['id']}] train#{r['train_sample_id']:3d} "
                  f"'{r['train_correlation']['source_token']}'->"
                  f"'{r['train_correlation']['target_token']}' "
                  f"| test '{r['test_correlation']['source_token']}'->"
                  f"'{r['test_correlation']['target_token']}' "
                  f"| cos_sim={r['cos_sim']:.4f}")

        report_filename = f"correlation_matching_results_test{SELECTED_TEST_SAMPLE_INDEX}_tok{TOKEN_INDEX_TO_RETRIEVE}.json"

    # ════════════════════════════════════════════════════════════════════════════
    # ── ALL-TOKENS MODE ──────────────────────────────────────────────────────
    # ════════════════════════════════════════════════════════════════════════════
    else:
        max_end = min(prompt_len + MAX_OUTPUT_TOKENS, seq_len)

        # ── Debug: show every generated token so we can see why some are trivial ──
        print(f"\n[Debug] Generated {max_end - prompt_len} response tokens:")
        for _t in range(prompt_len, max_end):
            _tok_id  = int(test_batch["input_ids"][0, _t].item())
            _tok_str = tokenizer.decode([_tok_id])
            _trivial = is_trivial_token(tokenizer, _tok_id)
            print(f"  pos={_t}  id={_tok_id}  repr={repr(_tok_str)}  trivial={_trivial}")

        valid_test_tokens = [
            t for t in range(prompt_len, max_end)
            if not is_trivial_token(tokenizer, int(test_batch["input_ids"][0, t].item()))
        ]
        n_test_toks = len(valid_test_tokens)
        print(f"\nAll-tokens mode: {n_test_toks} semantic tokens "
              f"(from {max_end - prompt_len} response tokens, trivial skipped)")

        # ── Phase 1: Chunked coarse screening ───────────────────────────────────
        # Process test-token CE grads in chunks of COARSE_CHUNK_SIZE:
        #   • Only K gradient vectors (≈ K × lm_head size) live on GPU at once.
        #   • Training set is scanned ceil(N_tok / K) times.
        #   • Each training sample's gradient is computed once per chunk scan, then freed.
        # This avoids both storing all N_tok grads simultaneously AND rescanning
        # the training set N_tok times.
        n_chunks = (n_test_toks + COARSE_CHUNK_SIZE - 1) // COARSE_CHUNK_SIZE
        print(f"\n=== Stage 2: Chunked coarse screening "
              f"({n_test_toks} tokens, chunk_size={COARSE_CHUNK_SIZE}, "
              f"→ {n_chunks} training-set scan(s)) ===")

        # Accumulate scores keyed by tok position index (not tok_idx) for Phase 2 lookup
        per_token_local_scores: dict[int, list[tuple[int, float]]] = {
            t: [] for t in valid_test_tokens
        }
        empty_ignored = torch.tensor([], device=accelerator.device)

        for chunk_start in range(0, n_test_toks, COARSE_CHUNK_SIZE):
            chunk_tokens = valid_test_tokens[chunk_start : chunk_start + COARSE_CHUNK_SIZE]
            chunk_size   = len(chunk_tokens)

            # Compute CE grads for this chunk and stack onto lm_head_device: [K, D]
            chunk_grads = []
            for t in chunk_tokens:
                chunk_grads.append(_test_ce_grad_for_token(t))   # [D] on lm_head_device
            chunk_grad_matrix = torch.stack(chunk_grads, dim=0)   # [K, D]
            del chunk_grads

            # One scan of the full training set for this chunk
            for batch in tqdm(
                train_loader,
                desc=f"Stage 2 chunk {chunk_start // COARSE_CHUNK_SIZE + 1}/{n_chunks}",
                leave=False,
            ):
                batch_device = {k: v.to(accelerator.device) for k, v in batch.items()
                                if isinstance(v, torch.Tensor)}
                train_idx = int(batch_device["sample_index"].item())

                # Skip sequences that would cause OOM during eager attention
                if batch_device["input_ids"].size(1) > SEQUENCE_LENGTH_LIMIT:
                    del batch_device
                    continue

                train_ce_grads = compute_gradients(
                    model=model, batch=batch_device,
                    param_filter_fn=lm_head_filter, device=accelerator.device,
                    ignored_token_ids=empty_ignored,
                )
                flat_train = _flat_grad_on_device(
                    train_ce_grads, filtered_params, lm_head_device
                )  # [D]

                # Batched cosine: [K, D] × [1, D] → [K]
                cos_sims = F.cosine_similarity(
                    chunk_grad_matrix,       # [K, D]
                    flat_train.unsqueeze(0), # [1, D]
                    dim=1,
                )

                for t, cos_sim in zip(chunk_tokens, cos_sims.tolist()):
                    per_token_local_scores[t].append((train_idx, float(cos_sim)))

                del train_ce_grads, flat_train, batch_device

            # Free chunk grads before loading the next chunk
            del chunk_grad_matrix
            torch.cuda.empty_cache()

        # Gather across GPUs (no-op in single-process mode) and select top-K per token
        per_token_related: dict[int, list[tuple[int, float]]] = {}
        for t in valid_test_tokens:
            all_s = _gather_scores(accelerator, per_token_local_scores[t], accelerator.device)
            per_token_related[t] = nlargest(TOP_K_TRAIN_SAMPLES, all_s, key=lambda x: x[1])
        del per_token_local_scores

        # ── Phase 2: Per-token Stage 1 (saliency + 2nd-order) + Stage 3 ─────────
        per_token_results: list  = []
        train_sample_cache: dict = {}   # str(train_idx) → detail dict (with _candidate_pairs, _tr_batch_cpu)
        pair_id_counter = 0

        for t in tqdm(valid_test_tokens, desc="Test tokens",
                       disable=not accelerator.is_local_main_process):
            target_tok_id   = int(test_batch["input_ids"][0, t].item())
            target_tok_text = tokenizer.decode([target_tok_id])
            print(f"\n=== Token {t}: '{target_tok_text}' ===")

            # Stage 1a: saliency at t (cheap, one forward+backward)
            with torch.inference_mode(False):
                sal_vec = compute_full_saliency_vector(model, test_batch, t)
            top_test_corr = nlargest(TOP_K_PROMPT_TOKENS, enumerate(sal_vec), key=lambda x: x[1])
            del sal_vec  # no longer needed after top-K selection
            top_test_correlations = [
                {
                    "source_token_index": idx,
                    "source_token":       tokenizer.decode([int(test_batch["input_ids"][0, idx].item())]),
                    "target_token_index": t,
                    "target_token":       target_tok_text,
                    "saliency_score":     float(score),
                }
                for idx, score in top_test_corr
            ]

            # Stage 1b: 2nd-order test features (kept on lm_head_device, freed after this token)
            print(f"  Computing {len(top_test_correlations)} test 2nd-order features...")
            test_corr_features: dict = {}
            with torch.inference_mode(False):
                for item in top_test_correlations:
                    p_idx = item["source_token_index"]
                    feat  = compute_correlation_second_order_gradient(
                        model=model, batch=test_batch,
                        target_idx_in_seq=t, source_idx_in_seq=p_idx,
                        param_filter_fn=param_filter,
                    )
                    test_corr_features[p_idx] = (feat.to(lm_head_device), item["source_token"], item["saliency_score"])

            # Use pre-computed Stage 2 results — no additional training-set scan needed
            related_samples = per_token_related[t]
            print(f"  Top-{TOP_K_TRAIN_SAMPLES} from Stage 2: "
                  f"{[(i, round(s,4)) for i,s in related_samples]}")

            # Stage 3: fine-grained matching for this token's top train samples
            token_pair_records: list = []
            for rank, (train_idx, coarse_score) in enumerate(related_samples):
                print(f"\n  --- Train {train_idx} (rank={rank+1}, coarse={coarse_score:.4f}) ---")
                cached = train_sample_cache.get(str(train_idx))
                detail, pairs, pair_id_counter = _process_train_sample_stage3(
                    train_idx, coarse_score, test_corr_features,
                    target_tok_text, t,
                    f"t{t}", pair_id_counter,
                    cached_detail=cached,
                )
                if detail is not None:
                    train_sample_cache[str(train_idx)] = detail
                    token_pair_records.extend(pairs)

            token_pair_records.sort(key=lambda x: x["cos_sim"], reverse=True)
            per_token_results.append({
                "target_token_index": t,
                "target_token":       target_tok_text,
                "top_correlations":   top_test_correlations,
                "correlation_pairs":  token_pair_records,
            })

            # Free test-side features before next token
            del test_corr_features
            torch.cuda.empty_cache()

        # Clean internal cache keys before saving
        train_sample_details = {
            k: {fk: fv for fk, fv in v.items() if not fk.startswith("_")}
            for k, v in train_sample_cache.items()
        }

        report_json = {
            "experiment_meta": {
                "test_sample_index":  SELECTED_TEST_SAMPLE_INDEX,
                "mode":               "all_tokens",
                "max_output_tokens":  MAX_OUTPUT_TOKENS,
                "tokens_analyzed":    len(per_token_results),
                "screening":          "single_pass_all_tokens",
                "config": {
                    "TOP_K_PROMPT_TOKENS":    TOP_K_PROMPT_TOKENS,
                    "TOP_K_TRAIN_SAMPLES":    TOP_K_TRAIN_SAMPLES,
                    "TOP_TARGETS":            TOP_TARGETS,
                    "TOP_K_SOURCE_PER_TARGET": TOP_K_SOURCE_PER_TARGET,
                    "CONTEXT_WINDOW_SIZE":    CONTEXT_WINDOW_SIZE,
                },
            },
            "test_sample_baseline": {
                "full_tokens":         gen_result["pred_full_tokens"][0],  # prompt + model output (clickable)
                "correct_full_tokens": gen_result["full_tokens"][0],       # prompt + ground truth answer
                "prompt_len":          prompt_len,
            },
            "per_token_results":    per_token_results,
            "train_sample_details": train_sample_details,
        }

        report_filename = f"correlation_matching_results_test{SELECTED_TEST_SAMPLE_INDEX}_all_tokens.json"

    # ── Save (main process only in multi-GPU) ────────────────────────────────
    if accelerator.is_main_process:
        report_json = round_floats(report_json, 5)
        base_dir    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        report_path = os.path.join(base_dir, report_filename)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report_json, f, indent=2, ensure_ascii=False)
        print(f"\nExperiment completed. Results → {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run causal intervention experiment for a single test sample + token."
    )
    parser.add_argument(
        "--test-index", type=int, default=None,
        help="Index of the test sample to analyse (overrides SELECTED_TEST_SAMPLE_INDEX).",
    )
    parser.add_argument(
        "--token-index", type=int, default=None,
        help="Token position to investigate (overrides TOKEN_INDEX_TO_RETRIEVE). Single-token mode only.",
    )
    parser.add_argument(
        "--all-tokens", action="store_true",
        help="Attribute all output tokens (up to MAX_OUTPUT_TOKENS) instead of a single token.",
    )
    args = parser.parse_args()

    if args.test_index is not None:
        SELECTED_TEST_SAMPLE_INDEX = args.test_index
    if args.token_index is not None:
        TOKEN_INDEX_TO_RETRIEVE = args.token_index

    mode_str = "all_tokens" if args.all_tokens else f"single_token tok={TOKEN_INDEX_TO_RETRIEVE}"
    print(f"[intervention] test_index={SELECTED_TEST_SAMPLE_INDEX}  mode={mode_str}")
    run_causal_intervention_experiment(all_tokens=args.all_tokens)

