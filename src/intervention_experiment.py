import os
import json
import torch
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
    finetune_on_sample,
    CustomCollator,
    round_floats,
    _find_subseq_start,
    _apply_freeze_strategy,
)
from src.process_data import process_func_chatml
from transformers import DataCollatorForSeq2Seq, set_seed

# ====== CONFIGURATION ======
SEED = 42
SELECTED_TEST_SAMPLE_INDEX = 58
TOKEN_INDEX_TO_RETRIEVE = 703  # This is the "first wrong token" we are investigating
INTERVENTION_EPOCHS = 3        # Low epochs for mild intervention
BOOST_COEF = 10.0              # To manually amplify the correlation
TOP_K_TRAIN_SAMPLES = 10       # How many top train samples to evaluate
TOP_K_PROMPT_TOKENS = 8         # How many correlation tokens to extract/compare

def get_gradient_related_samples(test_idx, target_tok_idx):
    """
    Loads the cached gradient similarity results.
    We assume main_compute_gradient_related_samples in NIF.py has already 
    saved this JSON.
    """
    grad_file = f'test_{test_idx}_{target_tok_idx}_result.json'
    # The file path in NIF.py is an uplevel folder: '../test_58_703_result.json'
    # Actually, in NIF.py it's: os.path.join(os.path.dirname(__file__), f'../{grad_file}')
    # Let's check where it really is.
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    grad_path = os.path.join(base_dir, grad_file)
    
    if os.path.exists(grad_path):
        with open(grad_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data["result"]
    else:
        raise FileNotFoundError(f"Gradient cached file not found at {grad_path}. Please run gradient influence first!")

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
    convert_to_chatml = partial(process_func_chatml, tokenizer=tokenizer)

    train_samples = load_samples_from_formal_jsonl("sft_train.jsonl")
    test_samples = load_samples_from_formal_jsonl("sft_test.jsonl")

    # Limit train dataset length just like in NIF.py
    SEQUENCE_LENGTH_LIMIT = 3000
    train_ds = build_train_dataset(train_samples, convert_to_chatml)

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt"
    )
    collator = CustomCollator(base_collator)

    # We mock a small train loader just to instantiate NewInferenceFunction 
    # (since it requires a Datloader, though we do manual retrieval here)
    train_loader = torch.utils.data.DataLoader(
        DatasetWrapper(train_ds.select(range(10))), 
        batch_size=1, collate_fn=collator
    )
    train_loader = accelerator.prepare(train_loader)

    infer_fw = NewInferenceFunction(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        accelerator=accelerator,
        param_filter_fn=None,
        top_k=20,
    )

    # 1. SETUP TEST SAMPLE & GET BASELINE
    test_ds = build_single_sample_dataset(test_samples[SELECTED_TEST_SAMPLE_INDEX], convert_to_chatml)
    raw_test_batch = base_collator([test_ds[0]])
    raw_test_batch = {k: v.to(accelerator.device) for k, v in raw_test_batch.items()}
    
    # We must evaluate the target token from the model's PREDICTION, not the ground truth label!
    # So we execute inference once to get its prediction, then bind prompt & prediction together as our test benchmark.
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
    
    # Baseline run
    infer_fw.model.eval()
    baseline_res = infer_fw.infer(test_batch, target_idx=target_idx_tensor)
    
    # Calculate initial target token probability
    logits_baseline = baseline_res["logits"][0] # [seq_len, vocab_size]
    # Prediction of TOKEN_INDEX_TO_RETRIEVE happens at position TOKEN_INDEX_TO_RETRIEVE - 1
    baseline_probs = torch.softmax(logits_baseline[TOKEN_INDEX_TO_RETRIEVE - 1], dim=-1)
    target_tok_id = test_batch["input_ids"][0, TOKEN_INDEX_TO_RETRIEVE].item()
    target_tok_prob_baseline = baseline_probs[target_tok_id].item()
    
    # saliency_original[batch][target_token] = {"index": t, "saliency": [float per prompt token]}
    # We want saliency scores over prompt tokens for the single target token we're probing.
    baseline_saliency = baseline_res["saliency_original"][0][0]["saliency"]
    
    # Top correlation prompt tokens for the test sample
    top_test_corr = nlargest(TOP_K_PROMPT_TOKENS, enumerate(baseline_saliency), key=lambda x: x[1])
    top_test_prompt_tokens = [
        {
            "index": idx,
            "token": tokenizer.decode([test_batch["input_ids"][0, idx].item()]),
            "saliency_score": float(score)
        }
        for idx, score in top_test_corr
    ]

    report_json = {
        "experiment_meta": {
            "test_sample_index": SELECTED_TEST_SAMPLE_INDEX,
            "target_token_index": TOKEN_INDEX_TO_RETRIEVE,
            "intervention_epochs": INTERVENTION_EPOCHS,
            "boost_coef": BOOST_COEF
        },
        "test_sample_baseline": {
            "target_token": tokenizer.decode([target_tok_id]),
            "target_token_prob": target_tok_prob_baseline,
            "full_tokens": baseline_res["full_tokens"][0],
            "saliency_list": baseline_saliency,
            "top_correlated_prompt_tokens": top_test_prompt_tokens
        },
        "interventions": []
    }

    # 2. IDENTIFY TOP CORRELATED TRAIN SAMPLES 
    print("Loading cached gradient similarities...")
    grad_results = get_gradient_related_samples(SELECTED_TEST_SAMPLE_INDEX, TOKEN_INDEX_TO_RETRIEVE)
    # The highest positive score indicates the strongest positive correlation
    related_samples = nlargest(TOP_K_TRAIN_SAMPLES, grad_results, key=lambda x: x[1])

    marker_ids = tuple(tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))

    # 3. INTERVENTION LOOP
    for rank, (train_idx, score) in enumerate(related_samples):
        print(f"\n--- Processing Interventions: Top {rank+1} Train Sample (ID {train_idx}, Score {score:.4f}) ---")
        
        # Build Train Sample
        tr_ds = build_single_sample_dataset(train_samples[train_idx], convert_to_chatml)
        tr_batch = base_collator([tr_ds[0]])
        tr_batch = {k: v.to(accelerator.device) for k, v in tr_batch.items()}
        
        if tr_batch["input_ids"].size(1) > SEQUENCE_LENGTH_LIMIT:
            print(f"Train sample {train_idx} is too long ({tr_batch['input_ids'].size(1)}), skipping.")
            continue
            
        # Find start of assistant generation
        try:
            start_sys = _find_subseq_start(tr_batch["input_ids"][0], marker_ids) + 3
        except ValueError:
            print(f"Marker not found in train sample {train_idx}, skipping.")
            continue
            
        # Find first valid token
        valid_tok_idx = find_first_valid_token_index(tokenizer, tr_batch["input_ids"], start_sys)
        valid_tok_id = tr_batch["input_ids"][0, valid_tok_idx].item()
        first_valid_token_text = tokenizer.decode([valid_tok_id])
        
        # Get Train Sample Saliency
        infer_fw.model.eval()
        tr_res = infer_fw.infer(tr_batch, target_idx=torch.tensor([valid_tok_idx], device=accelerator.device))
        train_saliency = tr_res["saliency_original"][0][0]["saliency"]
        
        top_train_corr = nlargest(TOP_K_PROMPT_TOKENS, enumerate(train_saliency), key=lambda x: x[1])
        boost_indices = [idx for idx, _ in top_train_corr]
        boost_tokens_text = [tokenizer.decode([tr_batch["input_ids"][0, idx].item()]) for idx in boost_indices]
        
        print(f"Targeting Valid Token [{valid_tok_idx}]: '{first_valid_token_text}'")
        print(f"Found Train Correlated Prompt Tokens: {boost_tokens_text}")
        
        # --- CAUSAL INTERVENTION ---
        # 1. Apply the same freeze strategy that finetune_on_sample will use,
        #    so save/restore operate on the exact same set of parameters.
        _apply_freeze_strategy(infer_fw.model, "qk_last_quarter")
        infer_fw.save_model_params(to="original")
        
        # 2. Overfit with Attention Boost (Force correlation)
        train_labels = tr_batch["input_ids"].clone()
        # Ensure we mask out prompt for pure finetuning (start_sys instead of valid_tok_idx, meaning we fine-tune the whole answer)
        train_labels[0, :start_sys] = -100 
        
        finetune_on_sample(
            infer_fw.model,
            tokenizer,
            epochs=INTERVENTION_EPOCHS,
            input_ids=tr_batch["input_ids"],
            labels=train_labels,
            boost_indices=boost_indices,
            boost_coef=BOOST_COEF,
            # query position predicting valid_tok_idx is `valid_tok_idx - 1`
            first_gen_pos=valid_tok_idx - 1
        )
        
        # 3. Observe Test Sample Changes
        infer_fw.model.eval()
        after_res = infer_fw.infer(test_batch, target_idx=target_idx_tensor)
        after_logits = after_res["logits"][0]
        after_probs = torch.softmax(after_logits[TOKEN_INDEX_TO_RETRIEVE - 1], dim=-1)
        target_tok_prob_after = after_probs[target_tok_id].item()
        
        after_saliency = after_res["saliency_original"][0][0]["saliency"]
        
        correlation_shifts = []
        
        sum_delta = 0.0
        for baseline_item in top_test_prompt_tokens:
            p_idx = baseline_item["index"]
            b_score = baseline_item["saliency_score"]
            a_score = after_saliency[p_idx]
            delta = a_score - b_score
            sum_delta += delta
            
            correlation_shifts.append({
                "prompt_token_index": p_idx,
                "prompt_token": baseline_item["token"],
                "saliency_before": b_score,
                "saliency_after": float(a_score),
                "delta": float(delta)
            })
            
        # We classify as positive if the overall attention to the key tokens increased 
        # (aggregate delta > 0) OR if the probability of the wrong token increased.
        is_positive_correlated = False
        prob_diff = target_tok_prob_after - target_tok_prob_baseline
        if prob_diff > 0.001 or sum_delta > 0.01:
            is_positive_correlated = True
            
        # 4. Restore original weights, zero grad, and free GPU memory
        infer_fw.restore_model_params()
        infer_fw.model.zero_grad(set_to_none=True)
        # Store full tokens strings before deleting tr_res
        train_full_tokens = tr_res["full_tokens"][0]
        del after_res, after_logits, after_probs, tr_res, tr_batch, train_labels
        torch.cuda.empty_cache()
        
        # Record everything
        conclusion = "POSITIVE_CORRELATION" if is_positive_correlated else "NEGATIVE_OR_UNRELATED"
        print(f"Result -> Prob {target_tok_prob_baseline:.4f} => {target_tok_prob_after:.4f} | Conclusion: {conclusion}")
        
        intervention_record = {
            "train_sample_id": train_idx,
            "gradient_influence_score": float(score),
            "train_context": {
                "first_valid_token_index": valid_tok_idx,
                "first_valid_token": first_valid_token_text,
                "boost_indices": boost_indices,
                "boost_tokens_text": boost_tokens_text,
                "full_tokens": train_full_tokens,
                "saliency_list": train_saliency
            },
            "test_after_intervention": {
                "target_token_prob": float(target_tok_prob_after),
                "saliency_list": after_saliency,
                "correlation_shifts": correlation_shifts,
                "conclusion": conclusion
            }
        }
        
        report_json["interventions"].append(intervention_record)

    # Compress floats to keep JSON size manageable
    report_json = round_floats(report_json, 5)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    report_path = os.path.join(base_dir, 'intervention_results.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_json, f, indent=2)
        
    print(f"\\nExperiment completed successfully! Results written to {report_path}")

if __name__ == "__main__":
    run_causal_intervention_experiment()
