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
from src.loss import compute_correlation_second_order_gradient
from transformers import DataCollatorForSeq2Seq, set_seed

# ====== CONFIGURATION ======
SEED = 42
SELECTED_TEST_SAMPLE_INDEX = 58
TOKEN_INDEX_TO_RETRIEVE = 703  # This is the "first wrong token" we are investigating
TOP_K_TRAIN_SAMPLES = 10       # How many top train samples to evaluate
TOP_K_PROMPT_TOKENS = 4        # How many correlation tokens to extract/compare

def qk_last_quarter_filter(name, param, num_layers=28):
    import re
    match = re.search(r'layers\.(\d+)\.', name)
    if match:
        layer_idx = int(match.group(1))
        if layer_idx >= num_layers * 3 // 4:
            if 'q_proj' in name or 'k_proj' in name:
                return True
    return False

def find_first_valid_token_index(tokenizer, input_ids_tensor, start_idx):
    """
    Skip formatting characters like \n, \t, spaces, {, } to find the 
    first token that carries actual semantic meaning.
    """
    valid_token_index = start_idx
    ignore_tokens = ["\n", "\t", " ", "{", "}", "\r\n"]
    
    input_len = input_ids_tensor.size(1)
    while valid_token_index < input_len:
        tok_id = input_ids_tensor[0, valid_token_index].item()
        tok_str = tokenizer.decode([tok_id])
        if tok_str.strip() not in ["", "{", "}"]:
            break
        valid_token_index += 1
        
    return valid_token_index

def run_causal_intervention_experiment():
    accelerator = Accelerator()
    set_seed(SEED)

    model, tokenizer = load_model_and_tokenizer()
    # Find num_layers for filter 
    num_layers = len(model.model.layers)
    param_filter = partial(qk_last_quarter_filter, num_layers=num_layers)

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

    # Create full train loader for our 30-min scan!
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

    # 1. SETUP TEST SAMPLE & GET BASELINE
    test_ds = build_single_sample_dataset(test_samples[SELECTED_TEST_SAMPLE_INDEX], convert_to_chatml)
    raw_test_batch = base_collator([test_ds[0]])
    raw_test_batch = {k: v.to(accelerator.device) for k, v in raw_test_batch.items()}
    
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
    
    # Baseline run to get test Saliency
    infer_fw.model.eval()
    baseline_res = infer_fw.infer(test_batch, target_idx=target_idx_tensor)
    
    target_tok_id = test_batch["input_ids"][0, TOKEN_INDEX_TO_RETRIEVE].item()
    baseline_saliency = baseline_res["saliency_original"][0][0]["saliency"]
    
    # Top correlation pairs for test sample: each is a full (source -> target) pair
    target_tok_text = tokenizer.decode([target_tok_id])
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
            "target_token_index": TOKEN_INDEX_TO_RETRIEVE
        },
        "test_sample_baseline": {
            "target_token": target_tok_text,
            "target_token_index": TOKEN_INDEX_TO_RETRIEVE,
            "full_tokens": baseline_res["full_tokens"][0],
            "top_correlations": top_test_correlations
        },
        "interventions": []
    }

    # Extract Correlation Gradient Features for TEST SAMPLE
    print("Extracting test correlation features...")
    # test_corr_features: { source_token_index -> (feat_tensor, source_token_text) }
    test_corr_features = {}
    with torch.inference_mode(False):
        for item in top_test_correlations:
            p_idx = item["source_token_index"]
            print(f"  -> test correlation: '{item['source_token']}' -> '{item['target_token']}'")
            feat = compute_correlation_second_order_gradient(
                model=model, 
                batch=test_batch, 
                target_idx_in_seq=TOKEN_INDEX_TO_RETRIEVE,
                source_idx_in_seq=p_idx,
                param_filter_fn=param_filter
            )
            test_corr_features[p_idx] = (feat, item["source_token"])

    # 2. IDENTIFY TOP CORRELATED TRAIN SAMPLES (Coarse-Grained Screening)
    print("\nStage 2: Scanning full training set CE Gradients against Test Saliency Query (est. 30 mins) ...")
    
    from src.loss import compute_gradients
    
    # Use the highest-saliency test correlation as primary query for coarse screening
    primary_test_query = list(test_corr_features.values())[0][0]
    
    sample_scores = []
    
    with torch.inference_mode(False):
        for batch in tqdm(train_loader, desc="Scanning Train Samples"):
            batch_device = {k: v.to(accelerator.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            train_idx = int(batch_device["sample_index"].item())
            
            # Extract CE gradient signature for this train sample
            train_ce_grads = compute_gradients(
                model=model,
                batch=batch_device,
                param_filter_fn=param_filter,
                device=accelerator.device,
                ignored_token_ids=infer_fw.ignored_token_ids
            )
            
            # Flatten to 1D vector and compute cosine similarity
            flat_train_ce = torch.cat([
                g.reshape(-1) if g is not None else torch.zeros_like(p).reshape(-1) 
                for g, p in zip(train_ce_grads, [p for n,p in model.named_parameters() if param_filter(n,p)])
            ])
            
            # Score how well this train sample's CE gradient aligns with our microscopic Query!
            cos_sim = F.cosine_similarity(primary_test_query, flat_train_ce, dim=0).item()
            sample_scores.append((train_idx, cos_sim))
            
            # Prevent OOM during loop
            del train_ce_grads, flat_train_ce, batch_device
            
    related_samples = nlargest(TOP_K_TRAIN_SAMPLES, sample_scores, key=lambda x: x[1])

    marker_ids = tuple(tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))

    # 3. CORRELATION MATCHING LOOP
    for rank, (train_idx, score) in enumerate(related_samples):
        print(f"\n--- Processing Correlations: Top {rank+1} Train Sample (ID {train_idx}, Score {score:.4f}) ---")
        
        tr_ds = build_single_sample_dataset(train_samples[train_idx], convert_to_chatml)
        tr_batch = base_collator([tr_ds[0]])
        tr_batch = {k: v.to(accelerator.device) for k, v in tr_batch.items()}
        
        if tr_batch["input_ids"].size(1) > SEQUENCE_LENGTH_LIMIT:
            print(f"Train sample {train_idx} is too long ({tr_batch['input_ids'].size(1)}), skipping.")
            continue
            
        try:
            start_sys = _find_subseq_start(tr_batch["input_ids"][0], marker_ids) + 3
        except ValueError:
            print(f"Marker not found in train sample {train_idx}, skipping.")
            continue
            
        valid_tok_idx = find_first_valid_token_index(tokenizer, tr_batch["input_ids"], start_sys)
        valid_tok_id = tr_batch["input_ids"][0, valid_tok_idx].item()
        first_valid_token_text = tokenizer.decode([valid_tok_id])
        
        # Get Train Sample Saliency
        with torch.inference_mode(False):
            infer_fw.model.eval()
            tr_res = infer_fw.infer(tr_batch, target_idx=torch.tensor([valid_tok_idx], device=accelerator.device))
            train_saliency = tr_res["saliency_original"][0][0]["saliency"]
            train_full_tokens = tr_res["full_tokens"][0]
        
        top_train_corr = nlargest(TOP_K_PROMPT_TOKENS, enumerate(train_saliency), key=lambda x: x[1])

        # For each train's top source token, find its best match against EACH test correlation
        # Store correlation_matches: list of per-test-correlation match results
        correlation_matches = []  # one entry per test correlation

        with torch.inference_mode(False):
            for test_p_idx, (test_feat, test_src_text) in test_corr_features.items():
                # Find which train source token best matches this particular test correlation
                best_train_cos = -1.0
                best_train_src_text = ""
                best_train_src_idx = -1

                for p_idx, p_score in top_train_corr:
                    train_source_text = tokenizer.decode([tr_batch["input_ids"][0, p_idx].item()])
                    print(f"  -> [{test_src_text} -> {target_tok_text}] vs [{train_source_text} -> {first_valid_token_text}]")

                    train_feat = compute_correlation_second_order_gradient(
                        model, tr_batch, valid_tok_idx, p_idx, param_filter
                    )
                    cos_sim = F.cosine_similarity(test_feat, train_feat, dim=0).item()

                    if cos_sim > best_train_cos:
                        best_train_cos = cos_sim
                        best_train_src_text = train_source_text
                        best_train_src_idx = p_idx

                correlation_matches.append({
                    "test_correlation": {
                        "source_token": test_src_text,
                        "source_token_index": test_p_idx,
                        "target_token": target_tok_text,
                        "target_token_index": TOKEN_INDEX_TO_RETRIEVE
                    },
                    "train_correlation": {
                        "source_token": best_train_src_text,
                        "source_token_index": best_train_src_idx,
                        "target_token": first_valid_token_text,
                        "target_token_index": valid_tok_idx
                    },
                    "cos_sim": float(best_train_cos)
                })

        torch.cuda.empty_cache()
        
        print(f"Correlation matches for Train Sample {train_idx}:")
        for m in correlation_matches:
            print(f"  [{m['test_correlation']['source_token']} -> {m['test_correlation']['target_token']}]"
                  f" matched [{m['train_correlation']['source_token']} -> {m['train_correlation']['target_token']}]"
                  f" cos_sim={m['cos_sim']:.4f}")
        
        intervention_record = {
            "train_sample_id": train_idx,
            "coarse_cos_sim": float(score),  # Stage 2 coarse screening score
            "train_context": {
                "target_token": first_valid_token_text,
                "target_token_index": valid_tok_idx,
                "full_tokens": train_full_tokens,
                "saliency_list": train_saliency
            },
            "correlation_matches": correlation_matches
        }
        
        report_json["interventions"].append(intervention_record)

    report_json = round_floats(report_json, 5)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_path = os.path.join(base_dir, 'correlation_matching_results.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2)
        
    print(f"\nExperiment completed successfully! Results written to {report_path}")

if __name__ == "__main__":
    run_causal_intervention_experiment()
