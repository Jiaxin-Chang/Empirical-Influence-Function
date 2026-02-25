import os
import json
import torch
from functools import partial
from transformers import DataCollatorForSeq2Seq
from torch.utils.data import DataLoader
from accelerate import Accelerator

# Import necessary functions/classes from NIF
from src.NIF import (
    load_model_and_tokenizer,
    process_func_chatml,
    load_samples_from_formal_jsonl,
    build_train_dataset,
    build_single_sample_dataset,
    CustomCollator,
    DatasetWrapper,
    NewInferenceFunction,
    set_seed,
    SEED,
    SEQUENCE_LENGTH_LIMIT
)

def evaluate_ground_truth_influence():
    # 1. Initialize NIF dependencies
    accelerator = Accelerator()
    set_seed(SEED)

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    convert_to_chatml_with_tokenizer = partial(process_func_chatml, tokenizer=tokenizer)

    # 2. Load background (real) training data + Ground Truth (synthetic)
    # We load 100 samples from the real training set as 'noise' or 'background'
    real_train_samples = load_samples_from_formal_jsonl("sft_train.jsonl")[:100]
    
    gt_file = "ground_truth_demo.jsonl"
    if not os.path.exists(gt_file):
        raise FileNotFoundError(f"{gt_file} not found. Generate GT first.")

    gt_samples = []
    with open(gt_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            
            # Robust extraction of system, input, output
            sys_text = obj.get("system", "")
            inp_text = obj.get("input", "")
            outp_text = obj.get("output", "")
            
            if not sys_text or not inp_text or not outp_text:
                if "messages" in obj and len(obj["messages"]) >= 3:
                    sys_text = obj["messages"][0]["content"]
                    inp_text = obj["messages"][1]["content"]
                    outp_text = obj["messages"][2]["content"]
                    
            parsed_obj = {
                'system': sys_text,
                'input':  inp_text,
                'output': outp_text,
                'label': obj.get("label", "unknown")
            }
            gt_samples.append(parsed_obj)
            
    print(f"Loaded {len(real_train_samples)} real training samples for background noise.")
    print(f"Loaded {len(gt_samples)} ground truth variants.")
    
    # Combine them
    mixed_train_samples = real_train_samples + gt_samples
    
    # Keep track of where gt_samples are (the last 6 items)
    gt_indices = set(range(len(real_train_samples), len(mixed_train_samples)))
    
    # 3. Build train DataLoader
    print("Building dataset and DataLoader...")
    train_ds = build_train_dataset(mixed_train_samples, convert_to_chatml_with_tokenizer)
    
    # Filter length just like in NIF
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) <= SEQUENCE_LENGTH_LIMIT)
    
    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )
    collator = CustomCollator(base_collator)
    
    train_loader = DataLoader(
        DatasetWrapper(train_ds),
        batch_size=1,
        shuffle=False,
        collate_fn=collator
    )
    train_loader = accelerator.prepare(train_loader)
    
    # Custom filter function to strictly calculate gradients ONLY for the lm_head layer
    def filter_lm_head(name: str, param: torch.nn.Parameter):
        return "lm_head.weight" in name and param.requires_grad
        
    for name, param in model.named_parameters():
        if "lm_head.weight" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    # Initialize the IF engine
    inference_function = NewInferenceFunction(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        accelerator=accelerator,
        param_filter_fn=filter_lm_head,  # Passing the filter function here
        top_k=20,
    )
    
    # 4. Load the Target Test Sample
    TEST_SAMPLE_INDEX = 0 # According to construct_ground_truth.py, we used the first sample
    print(f"\nLoading Test Sample {TEST_SAMPLE_INDEX}...")
    test_samples = load_samples_from_formal_jsonl("sft_test.jsonl")
    temp_ds = build_single_sample_dataset(test_samples[TEST_SAMPLE_INDEX], convert_to_chatml_with_tokenizer)
    
    query_batch = base_collator([temp_ds[0]])
    for k, v in query_batch.items():
        query_batch[k] = v.to(accelerator.device)
        
    # Get the inference result to find the exact target idx and generation
    result = inference_function.infer(query_batch)
    prompt_len = int(result["target_idx"][0])
    
    # Let's say we want to compute influence on the very first generated token.
    # It means we check if the training data influences the model's first step in the generation.
    target_token_index = prompt_len
    print(f"Prompt length: {prompt_len}. Target Token Index for Attribution: {target_token_index}")
    
    # 5. Use the *ORIGINAL CORRECT ANSWER* from the test set as the target, NOT the model's wrong prediction.
    # If the model gets it wrong, its generated sequence gradients will naturally be deeply orthogonal
    # or strongly negative to the gradients of the training examples that teach the *correct* logic.
    # We want to measure who supports the *correct* answer.
    eval_batch = query_batch.copy()
    
    # Calculate loss on the *ENTIRE* correct response!
    # NIF's influence_gradient_single will automatically mask labels before target_token_index for us.
    
    # 6. Compute Influence Score
    print("\nComputing influence gradient scores over the mixed dataset... (This might take a minute)")
    scores, indices = inference_function.influence_gradient_single(
        query_batch=eval_batch,
        target_idx=target_token_index
    )
    
    # Combine scores and indices
    results = list(zip(indices, scores))
    # Sort by descending score (higher positive dot-product = more beneficial influence)
    results.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n========== 🎯 INFLUENCE BENCHMARK RESULTS 🎯 ==========")
    print(f"Total Evaluated: {len(results)} samples (100 background + 6 ground truth)")
    
    print("\n--- 🎖️ TOP 5 MOST POSITIVE (BENEFICIAL) SAMPLES ---")
    for idx, score in results[:5]:
        if idx in gt_indices:
            label = mixed_train_samples[idx]["label"]
            print(f"Index: {idx:<4} | Score: {score:.6f} | [GROUND TRUTH: {label}] ✅")
        else:
            print(f"Index: {idx:<4} | Score: {score:.6f} | [Noise/Real Data]")
            
    print("\n--- 💀 TOP 5 MOST NEGATIVE (DETRIMENTAL) SAMPLES ---")
    for idx, score in reversed(results[-5:]):
        if idx in gt_indices:
            label = mixed_train_samples[idx]["label"]
            print(f"Index: {idx:<4} | Score: {score:.6f} | [GROUND TRUTH: {label}] ❌")
        else:
            print(f"Index: {idx:<4} | Score: {score:.6f} | [Noise/Real Data]")
            
    print("\n--- 📊 GROUND TRUTH OVERALL RANKING ---")
    out_of = len(results)
    for rank, (idx, score) in enumerate(results):
        if idx in gt_indices:
            label = mixed_train_samples[idx]["label"]
            print(f"Rank: {rank+1:<3}/{out_of} | Index: {idx:<4} | Score: {score:>9.6f} | Variant: {label}")

if __name__ == "__main__":
    evaluate_ground_truth_influence()
