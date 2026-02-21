"""
Pre-compute projected gradients for all training samples.

Uses batched matrix multiplication for projection: reshape gradient into
a matrix and do ONE matmul, instead of 3400 vector-matrix multiplies in a loop.

Usage: python -m src.precompute
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
    CustomCollator,
    process_func_chatml,
    DatasetWrapper,
)
from src.loss import compute_gradients

# Constants
PROJECTION_DIM = 65536
SEED = 42
SEQUENCE_LENGTH_LIMIT = 3000
CACHE_FILE = "train_grads_cache.pt"
CHUNK_SIZE = 524288  # 512K params per chunk


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")
    sys.stdout.flush()


def project_gradient(grads, shared_R, device, chunk_size):
    """
    Project gradients using batched matmul (single GPU op, R read only once).
    
    Instead of 3400 separate vector-matrix multiplies in a Python loop,
    we flatten all gradients into a matrix [num_chunks, chunk_size] and
    do ONE matmul: [num_chunks, chunk_size] @ [chunk_size, proj_dim] = [num_chunks, proj_dim],
    then sum to get [proj_dim].
    """
    # 1. Flatten all grads to one vector on the target device
    g_parts = []
    for g in grads:
        if g is not None:
            g_parts.append(g.detach().reshape(-1).to(device=device))
    g_flat = torch.cat(g_parts)
    
    n = g_flat.numel()
    
    # 2. Pad to multiple of chunk_size
    remainder = n % chunk_size
    if remainder != 0:
        g_flat = torch.nn.functional.pad(g_flat, (0, chunk_size - remainder))
    
    # 3. Reshape to matrix and do ONE matmul
    num_chunks = g_flat.numel() // chunk_size
    g_matrix = g_flat.reshape(num_chunks, chunk_size)       # [num_chunks, chunk_size]
    sketches = g_matrix @ shared_R                           # [num_chunks, proj_dim]
    sketch = sketches.sum(dim=0)                             # [proj_dim]
    
    return sketch


def main():
    accelerator = Accelerator()
    set_seed(SEED)

    log("Step 1/5: Loading model and tokenizer...")
    t0 = time.time()
    model, tokenizer = load_model_and_tokenizer()
    device = accelerator.device
    log(f"  Done in {time.time()-t0:.1f}s. Device={device}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_dtype = next(model.parameters()).dtype
    num_chunks = (total_params + CHUNK_SIZE - 1) // CHUNK_SIZE
    log(f"  Total trainable params: {total_params:,}, dtype={model_dtype}")
    log(f"  Chunk size: {CHUNK_SIZE:,}, num_chunks: {num_chunks}")
    log(f"  Projection: ONE matmul [{num_chunks}, {CHUNK_SIZE}] @ [{CHUNK_SIZE}, {PROJECTION_DIM}]")

    log("Step 2/5: Loading training data...")
    t0 = time.time()
    convert_fn = partial(process_func_chatml, tokenizer=tokenizer)
    train_samples = load_samples_from_formal_jsonl("sft_train.jsonl")
    train_ds = build_train_dataset(train_samples, convert_fn)
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) <= SEQUENCE_LENGTH_LIMIT)
    log(f"  Done in {time.time()-t0:.1f}s. Samples: {len(train_ds)}")

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, padding=True, label_pad_token_id=-100
    )
    collator = CustomCollator(base_collator)
    train_loader = DataLoader(
        DatasetWrapper(train_ds), batch_size=1, shuffle=False, collate_fn=collator,
    )

    log("  Calling accelerator.prepare(train_loader)...")
    t0 = time.time()
    train_loader = accelerator.prepare(train_loader)
    log(f"  Done in {time.time()-t0:.1f}s")

    # Projection matrix: [chunk_size, PROJECTION_DIM]
    log(f"Step 3/5: Generating R matrix [{CHUNK_SIZE} x {PROJECTION_DIM}]...")
    t0 = time.time()
    torch.manual_seed(12345)
    shared_R = torch.randn(CHUNK_SIZE, PROJECTION_DIM, device=device, dtype=model_dtype)
    shared_R /= (PROJECTION_DIM ** 0.5)
    mem_gb = shared_R.nelement() * shared_R.element_size() / 1e9
    log(f"  Done in {time.time()-t0:.1f}s. R memory: {mem_gb:.2f} GB")

    log("Step 4/5: Computing gradients and projecting...")
    model.eval()
    ignored_token_ids = torch.tensor([], device=device)
    projected_grads = []
    indices_list = []

    for i, batch in enumerate(tqdm(train_loader, desc="Pre-computing")):
        t_sample = time.time()

        # Move batch to device
        t1 = time.time()
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        sample_idx = batch["sample_index"].item()
        t_move = time.time() - t1

        # Compute gradients (same as influence_gradient_single)
        t1 = time.time()
        grads = compute_gradients(
            model, batch, param_filter_fn=None, device=device,
            ignored_token_ids=ignored_token_ids
        )
        t_grad = time.time() - t1

        # Project: ONE matmul (R is read only once by GPU)
        t1 = time.time()
        sketch = project_gradient(grads, shared_R, device, CHUNK_SIZE)
        t_proj = time.time() - t1

        projected_grads.append(sketch.cpu())
        indices_list.append(sample_idx)
        del grads

        # Print timing for first 5 samples, then every 100
        if i < 5 or i % 100 == 0:
            log(f"  Sample {i} (idx={sample_idx}): "
                f"move={t_move:.2f}s, grad={t_grad:.2f}s, proj={t_proj:.2f}s, "
                f"total={time.time()-t_sample:.2f}s")

    log("Step 5/5: Saving cache...")
    t0 = time.time()
    cache = {
        "projected_grads": torch.stack(projected_grads),
        "indices": torch.tensor(indices_list),
        "projection_matrix": shared_R.cpu(),
        "chunk_size": CHUNK_SIZE,
    }
    torch.save(cache, CACHE_FILE)
    log(f"  Done in {time.time()-t0:.1f}s. File: {CACHE_FILE}")
    log(f"  Total samples: {len(projected_grads)}")


if __name__ == "__main__":
    main()
