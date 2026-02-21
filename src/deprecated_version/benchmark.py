"""
Benchmark: Original streaming vs. Cached fast retrieval using CountSketch.

Uses CountSketch for projection to support high dimensionality (65536)
without OOM (requires only ~5MB memory for hash tables vs 64GB for matrix).

Usage: python -m src.benchmark
"""

import sys
import time
import torch
from tqdm import tqdm
from functools import partial
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, set_seed

from src.NIF import (
    load_model_and_tokenizer,
    load_samples_from_formal_jsonl,
    build_train_dataset,
    build_single_sample_dataset,
    CustomCollator,
    process_func_chatml,
    DatasetWrapper,
    NewInferenceFunction,
    SEED,
    SELECTED_TEST_SAMPLE_INDEX,
    TOKEN_INDEX_TO_RETRIEVE,
    SEQUENCE_LENGTH_LIMIT,
)
from src.loss import compute_gradients

# Experiment settings
NUM_TRAIN_SAMPLES = 100
PROJECTION_DIM = 262144  # 65536 * 4  # High dimension for better precision
CHUNK_SIZE = 524288


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


def project_gradient_countsketch(grads, hash_indices, hash_signs, device, chunk_size, proj_dim):
    """
    Project gradients using CountSketch (scatter_add_).
    Memory efficient: O(P) runtime, O(1) extra memory.
    """
    model_dtype = hash_signs.dtype
    sketch = torch.zeros(proj_dim, device=device, dtype=model_dtype)
    
    # Flatten all gradients to reduce Python loop overhead
    g_parts = [g.detach().reshape(-1).to(device=device) for g in grads if g is not None]
    if not g_parts:
        return sketch
        
    g_flat = torch.cat(g_parts)
    n = g_flat.numel()
    
    # Process in chunks to keep memory low
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        length = end - start
        
        segment = g_flat[start:end]
        h_idx = hash_indices[:length]
        h_sign = hash_signs[:length]
        
        # sketch[h_idx] += segment * h_sign
        vals = segment * h_sign
        sketch.scatter_add_(0, h_idx, vals)
        
    return sketch


def main():
    accelerator = Accelerator()
    set_seed(SEED)

    log("Loading model, tokenizer, data...")
    model, tokenizer = load_model_and_tokenizer()
    device = accelerator.device
    model_dtype = next(model.parameters()).dtype

    convert_fn = partial(process_func_chatml, tokenizer=tokenizer)
    train_samples = load_samples_from_formal_jsonl("sft_train.jsonl")
    test_samples = load_samples_from_formal_jsonl("sft_test.jsonl")

    train_ds = build_train_dataset(train_samples, convert_fn)
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) <= SEQUENCE_LENGTH_LIMIT)
    train_ds_small = train_ds.select(range(min(NUM_TRAIN_SAMPLES, len(train_ds))))

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
    )
    collator = CustomCollator(base_collator)

    train_loader = DataLoader(
        DatasetWrapper(train_ds_small), batch_size=1, shuffle=False, collate_fn=collator,
    )
    train_loader = accelerator.prepare(train_loader)

    inference_function = NewInferenceFunction(
        model=model, tokenizer=tokenizer, train_loader=train_loader,
        accelerator=accelerator, param_filter_fn=None, top_k=20,
    )

    log(f"Building query for test sample {SELECTED_TEST_SAMPLE_INDEX}...")
    temp_ds = build_single_sample_dataset(
        test_samples[SELECTED_TEST_SAMPLE_INDEX], convert_fn
    )
    query_batch = base_collator([temp_ds[0]])
    for k, v in query_batch.items():
        query_batch[k] = v.to(device)

    result = inference_function.infer(query_batch)
    prompt_len = int(result["target_idx"][0])
    prompt_ids = query_batch["input_ids"][0, :prompt_len]
    pred_ids = torch.tensor(result["pred_ids"][0], device=prompt_ids.device, dtype=prompt_ids.dtype)

    new_input_ids = torch.cat([prompt_ids, pred_ids], dim=0).unsqueeze(0)
    new_attention_mask = torch.ones_like(new_input_ids)
    new_labels = new_input_ids.clone()
    new_labels[:, :prompt_len] = -100
    new_labels[:, (TOKEN_INDEX_TO_RETRIEVE + 1):] = -100

    grad_batch = {
        "input_ids": new_input_ids,
        "attention_mask": new_attention_mask,
        "labels": new_labels,
    }

    log("")
    log("=" * 60)
    log(f"BENCHMARK: {NUM_TRAIN_SAMPLES} samples, Dim={PROJECTION_DIM}, Method=CountSketch")
    log("=" * 60)

    # Method A
    log("")
    log("─── Method A: Original Streaming ───")
    t_a_start = time.time()
    scores_a, indices_a = inference_function.influence_gradient_single(
        query_batch=grad_batch,
        target_idx=TOKEN_INDEX_TO_RETRIEVE,
    )
    t_a_total = time.time() - t_a_start
    log(f"  ★ Method A total: {t_a_total:.1f}s")

    # Method B (CountSketch)
    log("")
    log("─── Method B: Cache (CountSketch) + Fast Retrieval ───")

    # B1: Init Hash Tables (Tiny Memory!)
    t_b_start = time.time()
    torch.manual_seed(12345)
    # Generate hash indices in [0, PROJECTION_DIM)
    hash_indices = torch.randint(0, PROJECTION_DIM, (CHUNK_SIZE,), device=device)
    # Generate signs {-1, 1}
    hash_signs = (torch.randint(0, 2, (CHUNK_SIZE,), device=device) * 2 - 1).to(model_dtype)
    
    log(f"  Hash tables size: {hash_indices.numel() * 4 / 1e6:.1f} MB (vs Matrix 64GB)")

    # B2: Cache Build
    log("  Building cache (one-time cost)...")
    t_cache_start = time.time()
    ignored_token_ids = torch.tensor([], device=device)
    cached_grads = []
    cached_indices = []

    for i, batch in enumerate(tqdm(train_loader, desc="  Caching")):
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        sample_idx = batch["sample_index"].item()

        grads = compute_gradients(model, batch, None, device, ignored_token_ids)
        sketch = project_gradient_countsketch(grads, hash_indices, hash_signs, device, CHUNK_SIZE, PROJECTION_DIM)

        cached_grads.append(sketch)
        cached_indices.append(sample_idx)
        del grads

    train_grads_matrix = torch.stack(cached_grads)
    t_cache = time.time() - t_cache_start
    log(f"  Cache built in {t_cache:.1f}s")

    # B3: Retrieval
    log("  Running fast retrieval...")
    t_retrieval_start = time.time()

    grad_batch_b, _ = inference_function._mask_labels_before_target(
        {k: v.clone() for k, v in grad_batch.items()}, TOKEN_INDEX_TO_RETRIEVE
    )
    query_grads_b = compute_gradients(
        model, grad_batch_b, None, device, inference_function.ignored_token_ids
    )
    query_vec_b = project_gradient_countsketch(query_grads_b, hash_indices, hash_signs, device, CHUNK_SIZE, PROJECTION_DIM)

    train_norms = train_grads_matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    train_normalized = train_grads_matrix / train_norms
    query_norm = query_vec_b.norm().clamp_min(1e-12)
    query_normalized = query_vec_b / query_norm
    scores_b_all = torch.mv(train_normalized, query_normalized)

    t_retrieval = time.time() - t_retrieval_start
    
    log(f"  ★ Fast retrieval only: {t_retrieval:.2f}s")

    # Comparison
    log("")
    log("=" * 60)
    log("RESULTS")
    log("=" * 60)
    log(f"  Method A (Original):               {t_a_total:.1f}s")
    log(f"  Method B (Retrieval):              {t_retrieval:.2f}s")
    log(f"  Speedup:                           {t_a_total / t_retrieval:.0f}x")
    log("")

    results_a = sorted(zip(indices_a, scores_a), key=lambda x: -x[1])[:20]
    results_b = sorted(
        zip(cached_indices, scores_b_all.cpu().tolist()),
        key=lambda x: -x[1]
    )[:20]

    top10_a = set(idx for idx, _ in results_a[:10])
    top10_b = set(idx for idx, _ in results_b[:10])
    overlap = len(top10_a & top10_b)

    log(f"  Top-10 overlap: {overlap}/10")
    log("")
    log("  Method A Top-10:                    Method B Top-10:")
    for rank in range(10):
        idx_a, sc_a = results_a[rank] if rank < len(results_a) else (-1, 0)
        idx_b, sc_b = results_b[rank] if rank < len(results_b) else (-1, 0)
        log(f"    {rank+1:2d}. idx={idx_a:5d} score={sc_a:+.6f}    "
            f"{rank+1:2d}. idx={idx_b:5d} score={sc_b:+.6f}")


if __name__ == "__main__":
    main()
