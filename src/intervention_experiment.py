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
TOKEN_INDEX_TO_RETRIEVE = 703  # The "first wrong token" we are investigating

TOP_K_PROMPT_TOKENS = 4        # How many test correlation features to extract
TOP_K_TRAIN_SAMPLES = 10       # How many top train samples from coarse screening
TOP_TARGETS = 5                # How many response tokens to scan per train sample
TOP_K_SOURCE_PER_TARGET = 4    # Top source tokens per target (includes response-internal tokens)
CONTEXT_WINDOW_SIZE = 3        # Tokens shown on each side of source/target for annotation


def lm_head_filter(name, param):
    """Only select the LM head weight — a single dense matrix that
    projects the last hidden state to vocabulary logits.
    This gives a compact, task-agnostic feature for every token prediction.
    """
    return name == "lm_head.weight"


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


def run_causal_intervention_experiment():
    accelerator = Accelerator()
    set_seed(SEED)

    model, tokenizer = load_model_and_tokenizer()
    param_filter = lm_head_filter

    convert_to_chatml = partial(process_func_chatml, tokenizer=tokenizer)

    train_samples = load_samples_from_formal_jsonl("sft_train.jsonl")
    test_samples = load_samples_from_formal_jsonl("sft_test.jsonl")

    SEQUENCE_LENGTH_LIMIT = 3000

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt"
    )
    collator = CustomCollator(base_collator)

    train_ds = build_train_dataset(train_samples, convert_to_chatml)
    train_loader = torch.utils.data.DataLoader(
        DatasetWrapper(train_ds),
        batch_size=1, collate_fn=collator
    )
    train_loader = accelerator.prepare(train_loader)

    infer_fw = NewInferenceFunction(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        accelerator=accelerator,
        param_filter_fn=param_filter,
        top_k=20,
    )

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 1: Test Sample — Saliency + Correlation Feature Extraction
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 1: Extracting test correlation features ===")

    test_ds = build_single_sample_dataset(test_samples[SELECTED_TEST_SAMPLE_INDEX], convert_to_chatml)
    raw_test_batch = base_collator([test_ds[0]])
    raw_test_batch = {k: v.to(accelerator.device) for k, v in raw_test_batch.items()}

    # Generate response to build the full sequence (prompt + predicted tokens)
    infer_fw.model.eval()
    gen_result = infer_fw.infer(raw_test_batch)
    prompt_len = int(gen_result["target_idx"][0])
    prompt_ids = raw_test_batch["input_ids"][0, :prompt_len]
    pred_ids = torch.tensor(
        gen_result["pred_ids"][0],
        device=prompt_ids.device,
        dtype=prompt_ids.dtype
    )
    new_input_ids = torch.cat([prompt_ids, pred_ids], dim=0).unsqueeze(0)
    new_attention_mask = torch.ones_like(new_input_ids)
    new_labels = new_input_ids.clone()
    new_labels[:, :prompt_len] = -100

    test_batch = {
        "input_ids": new_input_ids,
        "attention_mask": new_attention_mask,
        "labels": new_labels
    }

    target_idx_tensor = torch.tensor([TOKEN_INDEX_TO_RETRIEVE], device=accelerator.device)

    # Get test saliency via infer (needed for full_tokens and baseline info)
    infer_fw.model.eval()
    baseline_res = infer_fw.infer(test_batch, target_idx=target_idx_tensor)

    target_tok_id = test_batch["input_ids"][0, TOKEN_INDEX_TO_RETRIEVE].item()
    target_tok_text = tokenizer.decode([target_tok_id])
    baseline_saliency = baseline_res["saliency_original"][0][0]["saliency"]

    # Select top-K test correlations
    top_test_corr = nlargest(TOP_K_PROMPT_TOKENS, enumerate(baseline_saliency), key=lambda x: x[1])
    top_test_correlations = [
        {
            "source_token_index": idx,
            "source_token": tokenizer.decode([test_batch["input_ids"][0, idx].item()]),
            "target_token_index": TOKEN_INDEX_TO_RETRIEVE,
            "target_token": target_tok_text,
            "saliency_score": float(score)
        }
        for idx, score in top_test_corr
    ]

    report_json = {
        "experiment_meta": {
            "test_sample_index": SELECTED_TEST_SAMPLE_INDEX,
            "target_token_index": TOKEN_INDEX_TO_RETRIEVE,
            "config": {
                "TOP_K_PROMPT_TOKENS": TOP_K_PROMPT_TOKENS,
                "TOP_K_TRAIN_SAMPLES": TOP_K_TRAIN_SAMPLES,
                "TOP_TARGETS": TOP_TARGETS,
                "TOP_K_SOURCE_PER_TARGET": TOP_K_SOURCE_PER_TARGET,
                "CONTEXT_WINDOW_SIZE": CONTEXT_WINDOW_SIZE,
            }
        },
        "test_sample_baseline": {
            "target_token": target_tok_text,
            "target_token_index": TOKEN_INDEX_TO_RETRIEVE,
            "full_tokens": baseline_res["full_tokens"][0],
            "top_correlations": top_test_correlations
        },
        "correlation_pairs": []
    }

    # Compute second-order gradient feature for each test correlation
    # test_corr_features: { source_token_index -> (feat_tensor, source_token_text, saliency_score) }
    print("Computing second-order gradient features for test correlations...")
    test_corr_features = {}
    with torch.inference_mode(False):
        for item in top_test_correlations:
            p_idx = item["source_token_index"]
            print(f"  -> '{item['source_token']}' -> '{item['target_token']}' (saliency={item['saliency_score']:.4f})")
            feat = compute_correlation_second_order_gradient(
                model=model,
                batch=test_batch,
                target_idx_in_seq=TOKEN_INDEX_TO_RETRIEVE,
                source_idx_in_seq=p_idx,
                param_filter_fn=param_filter
            )
            test_corr_features[p_idx] = (feat, item["source_token"], item["saliency_score"])

    # ── Compute test-side CE gradient at token 703 (for Stage 2 coarse screening) ──
    # This is the token-targeted influence function query vector:
    #   g_test = ∂CE(test, token_703) / ∂lm_head
    # Semantics: "what gradient direction at lm_head would have corrected this specific token?"
    # Training samples whose CE gradient aligns with this are the most influential for token 703.
    print("Computing token-targeted CE gradient for coarse screening (token 703)...")
    filtered_params = [p for n, p in model.named_parameters() if param_filter(n, p)]

    # Build a single-token label batch: only token 703 has a valid label, rest are masked
    ce_labels = torch.full_like(test_batch["input_ids"], -100)
    ce_labels[0, TOKEN_INDEX_TO_RETRIEVE] = test_batch["input_ids"][0, TOKEN_INDEX_TO_RETRIEVE]
    single_token_batch = {
        "input_ids": test_batch["input_ids"],
        "attention_mask": test_batch["attention_mask"],
        "labels": ce_labels,
    }
    test_ce_grads = compute_gradients(
        model=model,
        batch=single_token_batch,
        param_filter_fn=param_filter,
        device=accelerator.device,
        ignored_token_ids=infer_fw.ignored_token_ids,
    )
    test_ce_grad = torch.cat([
        g.reshape(-1).cpu() if g is not None else torch.zeros(p.numel(), dtype=p.dtype)
        for g, p in zip(test_ce_grads, filtered_params)
    ]).detach()
    del test_ce_grads
    print(f"  test_ce_grad shape: {test_ce_grad.shape}, norm: {test_ce_grad.norm().item():.4f}")

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 2: Coarse Screening — token-targeted CE gradient cosine similarity
    #   Both sides use first-order CE gradients w.r.t. lm_head — consistent types.
    #   Test : g_test = ∂CE(test, token_703) / ∂lm_head
    #   Train: g_train = ∂CE(train, full_response) / ∂lm_head
    #   Score = cosine_sim(g_test, g_train)  [standard TracIn coarse screening]
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 2: Coarse screening over full training set ===")
    print("  Method: token-targeted CE gradient cosine similarity (TracIn variant)")
    print(f"  Test query: ∂CE(token_{TOKEN_INDEX_TO_RETRIEVE})/∂lm_head  "
          f"vs  Train: ∂CE(full_response)/∂lm_head")

    sample_scores = []

    for batch in tqdm(train_loader, desc="Scanning Train Samples"):
        batch_device = {k: v.to(accelerator.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        train_idx = int(batch_device["sample_index"].item())

        train_ce_grads = compute_gradients(
            model=model,
            batch=batch_device,
            param_filter_fn=param_filter,
            device=accelerator.device,
            ignored_token_ids=infer_fw.ignored_token_ids
        )

        flat_train_ce = torch.cat([
            g.reshape(-1).cpu() if g is not None else torch.zeros(p.numel(), dtype=p.dtype)
            for g, p in zip(train_ce_grads, filtered_params)
        ])

        # Token-targeted TracIn: cosine similarity between test token-703 CE gradient
        # and this training sample's full-response CE gradient.
        score = F.cosine_similarity(test_ce_grad, flat_train_ce, dim=0).item()
        sample_scores.append((train_idx, score))

        del train_ce_grads, flat_train_ce, batch_device

    related_samples = nlargest(TOP_K_TRAIN_SAMPLES, sample_scores, key=lambda x: x[1])
    print(f"  Top-{TOP_K_TRAIN_SAMPLES} train samples selected.")
    for rank, (tidx, sc) in enumerate(related_samples):
        print(f"    [{rank+1}] train_idx={tidx}  coarse_score={sc:.4f}")

    marker_ids = tuple(tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))

    # ═══════════════════════════════════════════════════════════════════════════
    # STAGE 3: Fine-grained Correlation Matching
    #   Step A — cheap saliency scan to enumerate (target, source) candidate pairs
    #   Step B — full second-order gradient computation + cosine matching for all pairs
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n=== Stage 3: Fine-grained correlation matching ===")

    all_pair_records = []
    pair_id_counter = 0
    train_sample_details = {}  # str(train_idx) -> full token + saliency data for display

    for rank, (train_idx, coarse_score) in enumerate(related_samples):
        print(f"\n--- Train Sample {train_idx}  (rank={rank+1}, coarse_score={coarse_score:.4f}) ---")

        tr_ds = build_single_sample_dataset(train_samples[train_idx], convert_to_chatml)
        tr_batch = base_collator([tr_ds[0]])
        tr_batch = {k: v.to(accelerator.device) for k, v in tr_batch.items()}

        if tr_batch["input_ids"].size(1) > SEQUENCE_LENGTH_LIMIT:
            print(f"  Skipping: sequence too long ({tr_batch['input_ids'].size(1)} tokens)")
            continue

        try:
            start_sys = _find_subseq_start(tr_batch["input_ids"][0], marker_ids) + 3
        except ValueError:
            print(f"  Skipping: assistant marker not found")
            continue

        response_start = find_first_valid_token_index(tokenizer, tr_batch["input_ids"], start_sys)
        seq_len = tr_batch["input_ids"].size(1)

        # Full token list in vocab-level encoding (Ġ=space, Ċ=newline) for SwitchTokenCodeBlock
        full_tokens = tokenizer.convert_ids_to_tokens(tr_batch["input_ids"][0].tolist())
        target_saliencies: dict[int, list[float]] = {}  # target_token_idx -> saliency_list

        # ── Step A: Cheap saliency scan ────────────────────────────────────────
        # For each of the first TOP_TARGETS response tokens, compute full saliency
        # vector and select the top TOP_K_SOURCE_PER_TARGET source tokens.
        # Source search range: ALL tokens before target t (prompt + earlier response),
        # since intra-response attention paths are also learned correlations.
        candidate_pairs = []  # list of (saliency_score, target_t, source_s_idx)

        with torch.inference_mode(False):
            for t_offset in range(TOP_TARGETS):
                t = response_start + t_offset
                if t >= seq_len:
                    break

                saliency_vec = compute_full_saliency_vector(model, tr_batch, t)
                target_saliencies[t] = [round(float(s), 6) for s in saliency_vec]

                # nlargest over all preceding tokens (0..t-1)
                top_sources = nlargest(
                    TOP_K_SOURCE_PER_TARGET,
                    enumerate(saliency_vec),
                    key=lambda x: x[1]
                )
                for s_idx, s_score in top_sources:
                    candidate_pairs.append((float(s_score), t, int(s_idx)))

        print(f"  Step A: {len(candidate_pairs)} candidate (target, source) pairs found "
              f"({TOP_TARGETS} targets × {TOP_K_SOURCE_PER_TARGET} sources each)")

        # Save full token + saliency data for this train sample (used by frontend visualization)
        train_sample_details[str(train_idx)] = {
            "full_tokens": full_tokens,
            "answer_start_index": response_start,
            "coarse_cos_sim": float(coarse_score),
            "saliencies_by_token": {str(k): v for k, v in target_saliencies.items()},
        }

        # ── Step B: Full second-order gradient matching ────────────────────────
        # Compute one second-order gradient feature per candidate pair, then
        # compare against every test correlation feature. Keep all scores.
        with torch.inference_mode(False):
            for saliency_score, t, s_idx in candidate_pairs:
                ids_1d = tr_batch["input_ids"][0]
                train_target_tok = tokenizer.decode([ids_1d[t].item()])
                train_source_tok = tokenizer.decode([ids_1d[s_idx].item()])
                response_token_offset = t - response_start

                print(f"  Step B: 2nd-order feat  '{train_source_tok}' -> '{train_target_tok}'"
                      f"  (offset={response_token_offset}, saliency={saliency_score:.4f})")

                train_feat = compute_correlation_second_order_gradient(
                    model, tr_batch,
                    target_idx_in_seq=t,
                    source_idx_in_seq=s_idx,
                    param_filter_fn=param_filter
                )

                source_context = get_context_window(tokenizer, ids_1d, s_idx)
                target_context = get_context_window(tokenizer, ids_1d, t)

                # Compare this single train_feat against ALL test correlation features
                for test_p_idx, (test_feat, test_src_text, test_saliency) in test_corr_features.items():
                    cos_sim = F.cosine_similarity(
                        test_feat.cpu(), train_feat.cpu(), dim=0
                    ).item()

                    record = {
                        "id": f"pair_{pair_id_counter:04d}",
                        "cos_sim": float(cos_sim),
                        "coarse_cos_sim": float(coarse_score),
                        "train_sample_id": train_idx,

                        "test_correlation": {
                            "source_token": test_src_text,
                            "source_token_index": test_p_idx,
                            "target_token": target_tok_text,
                            "target_token_index": TOKEN_INDEX_TO_RETRIEVE,
                            "saliency_score": float(test_saliency)
                        },

                        "train_correlation": {
                            "source_token": train_source_tok,
                            "source_token_index": s_idx,
                            "target_token": train_target_tok,
                            "target_token_index": t,
                            "saliency_score": saliency_score,
                            "response_token_offset": response_token_offset
                        },

                        "train_context": {
                            "source_context": source_context,
                            "target_context": target_context
                        },

                        # Human annotation field — to be filled in:
                        #   "correct"    : the (source -> target) association is semantically valid
                        #                  within this training sample's logic
                        #   "incorrect"  : spurious / wrong reasoning path
                        #   "ambiguous"  : unclear
                        "annotation": None
                    }
                    all_pair_records.append(record)
                    pair_id_counter += 1

        torch.cuda.empty_cache()

    # Sort all records by cos_sim descending — ready for threshold-based annotation
    all_pair_records.sort(key=lambda x: x["cos_sim"], reverse=True)
    report_json["correlation_pairs"] = all_pair_records
    report_json["train_sample_details"] = train_sample_details

    # Quick summary
    total = len(all_pair_records)
    print(f"\n{'='*60}")
    print(f"Total correlation pair records: {total}")
    print(f"  (≤ {TOP_K_TRAIN_SAMPLES} train × {TOP_TARGETS}×{TOP_K_SOURCE_PER_TARGET} pairs × {TOP_K_PROMPT_TOKENS} test corr"
          f" = {TOP_K_TRAIN_SAMPLES * TOP_TARGETS * TOP_K_SOURCE_PER_TARGET * TOP_K_PROMPT_TOKENS} max)")
    print(f"\nTop-10 pairs by cos_sim:")
    for r in all_pair_records[:10]:
        print(f"  [{r['id']}] train#{r['train_sample_id']:3d} "
              f"'{r['train_correlation']['source_token']}'->"
              f"'{r['train_correlation']['target_token']}' "
              f"| test '{r['test_correlation']['source_token']}'->"
              f"'{r['test_correlation']['target_token']}' "
              f"| cos_sim={r['cos_sim']:.4f}")
    print('='*60)

    report_json = round_floats(report_json, 5)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # filename encodes the experiment parameters so multiple runs don't overwrite each other
    # this mirrors the saliency_test{N}_tok{tok}.json naming convention
    report_filename = f'correlation_matching_results_test{SELECTED_TEST_SAMPLE_INDEX}_tok{TOKEN_INDEX_TO_RETRIEVE}.json'
    report_path = os.path.join(base_dir, report_filename)
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2, ensure_ascii=False)

    print(f"\nExperiment completed. Results → {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run causal intervention experiment for a single test sample + token."
    )
    parser.add_argument(
        "--test-index", type=int, default=None,
        help="Index of the test sample to analyse (overrides SELECTED_TEST_SAMPLE_INDEX)."
    )
    parser.add_argument(
        "--token-index", type=int, default=None,
        help="Token position to investigate (overrides TOKEN_INDEX_TO_RETRIEVE)."
    )
    args = parser.parse_args()

    if args.test_index is not None:
        SELECTED_TEST_SAMPLE_INDEX = args.test_index
    if args.token_index is not None:
        TOKEN_INDEX_TO_RETRIEVE = args.token_index

    print(f"[intervention] test_index={SELECTED_TEST_SAMPLE_INDEX}  token_index={TOKEN_INDEX_TO_RETRIEVE}")
    run_causal_intervention_experiment()

