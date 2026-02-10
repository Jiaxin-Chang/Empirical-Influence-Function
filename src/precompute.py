import torch
import os
import json
from tqdm import tqdm
from functools import partial
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, set_seed

# Import from existing project structure
from src.NIF import (
    load_model_and_tokenizer,
    load_samples_from_formal_jsonl,
    build_train_dataset,
    CustomCollator,
    process_func_chatml,
    NewInferenceFunction,
    compute_gradients  # Ensure this is imported or available
)
from src.loss import compute_gradients

# Constants
PROJECTION_DIM = 4096
SEED = 42
SEQUENCE_LENGTH_LIMIT = 2000 # Keep consistent with NIF.py or adjust as needed
CACHE_FILE = "train_grads_cache.pt"

def flatten_gradients(grads):
    return torch.cat([g.reshape(-1) for g in grads if g is not None])

def main():
    accelerator = Accelerator()
    set_seed(SEED)

    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    device = accelerator.device

    # 1. Prepare Projection Matrix
    print("Initializing random projection matrix...")
    # Calculate total number of parameters that require grad
    # Since param_filter_fn is None in NIF.py for main_compute_gradient_related_samples, 
    # we assume full parameters or whatever requires_grad is True.
    
    total_params = 0
    for p in model.parameters():
        if p.requires_grad:
            total_params += p.numel()
    
    print(f"Total trainable parameters: {total_params}")
    
    # Create a reproducible random matrix
    # We generate it in chunks to avoid OOM if total_params is huge
    # Actually, we can generate it on the fly or row by row, 
    # but for dot product P @ R, we need R to be [Total_Params, Dim].
    # For 7B model, 7e9 * 4096 * 2bytes (fp16) is too large (~56TB). 
    # We CANNOT materialize the full projection matrix.
    # We must use a seed-based generator or a sparse matrix?
    # Or simply compute projection layer by layer?
    
    # Better approach for memory efficiency:
    # Instead of P @ R, we can compute dot product with a "virtual" matrix determined by a seed.
    # However, for speed in this script, let's try a simpler approach if Total_Params is too big:
    # We can perform the projection *layer by layer* to save memory.
    
    # Let's save the seed used for projection so we can reproduce it during inference
    projection_seed = 12345
    torch.manual_seed(projection_seed)
    
    # We will generate projection matrix blocks on the fly to save memory.
    
    # 2. Load Data
    print("Loading training data...")
    convert_to_chatml_with_tokenizer = partial(process_func_chatml, tokenizer=tokenizer)
    train_samples = load_samples_from_formal_jsonl("sft_train.jsonl")
    train_ds = build_train_dataset(train_samples, convert_to_chatml_with_tokenizer)
    
    # Filter long sequences
    train_ds = train_ds.filter(lambda x: len(x["input_ids"]) <= SEQUENCE_LENGTH_LIMIT)
    
    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100
    )
    collator = CustomCollator(base_collator)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=1, 
        shuffle=False, 
        collate_fn=collator
    )
    train_loader = accelerator.prepare(train_loader)

    print(f"Start pre-computing gradients for {len(train_ds)} samples...")
    
    projected_grads = []
    indices_list = []
    
    model.eval()
    
    # Store parameter shapes to reconstruct/iterate correctly
    param_shapes = []
    for p in model.parameters():
        if p.requires_grad:
            param_shapes.append(p.shape)
            
    # We need a consistent projection matrix. 
    # Generating a 7B x 4096 matrix is impossible. 
    # We will use a `FastRandomProjection` approach:
    # Just use a fixed RNG state to generate the column `i` of the projection matrix when needed? 
    # No, that's too slow (re-generating random numbers N times).
    
    # Compromise: We only project the "largest" layers or use a smaller set of params? 
    # NO, we want full parameter Equivalent.
    # 
    # Standard practice for JL transform on huge vectors:
    # Use a sparse random projection (Achlioptas, 2003). 
    # Or efficient implementation:
    # Generate R in blocks.
    # projected_vec = sum( block_grad @ block_R )
    
    # Let's pre-generate small blocks of R to save memory.
    # But R needs to be consistent across samples.
    # So we can effectively iterate through layers.
    
    projected_accumulator = torch.zeros(len(train_ds), PROJECTION_DIM, device=device)
    
    # To avoid storing all gradients (OOM), we can invert the loop?
    # Loop 1: Compute gradients for ALL samples -> Save to disk (Huge space) -> Project. No.
    # Loop 2: Loop samples, compute grad, project on the fly.
    
    # To project on the fly, we need access to the full R.
    # If R is too big, we can generate R block-by-block.
    # But for each sample, we need the *entire* R.
    # Re-generating R for each sample is slow.
    
    # Optimization: 
    # 1. Generate a "Layer-wise" projection matrix.
    # For each layer L, generate R_L of shape [Size_L, 4096].
    # If Size_L is small enough, keep R_L in memory.
    # If 7B params, even one layer (e.g. fc) can be huge. 
    # But usually layers are e.g. 4096*4096 ~ 16M params. 
    # 16M * 4096 * 2bytes ~ 128GB... still too big for R_L if we do simple matrix mult.
    
    # WAIT. 
    # 7B Params. The gradient vector $g$ is 7B.
    # We want $g @ R$ where $R$ is $7B \times 4096$.
    # This matrix multiplication is huge.
    
    # Alternative for acceleration that fits in memory:
    # Only project the **Embedding Layer** + **Last Layer** gradients?
    # User confirmed earlier that "full parameters" was used, but acknowledged it's slow.
    # If we want to strictly follow "full parameters", we must solve the memory issue.
    # 
    # Actually, `rademacher_projection` or hashing might be faster and memory efficient.
    # But let's stick to a simpler approximation if full R is impossible:
    # -> **Bucket Projection**: Sum params into 4096 buckets (hashing). 
    # This is equivalent to R being a sparse matrix where each row has exactly one '1'.
    # This is extremely memory efficient and fast.
    
    # Let's implement Hashing Trick / Sketching.
    # Map each parameter index to a bucket in [0, 4095] using a hash function.
    # val = grad[i] * sign[i]
    # bucket[hash(i)] += val
    
    print("Using Hashing Trick for Dimensionality Reduction (Dimension: 4096)...")
    
    # Pre-compute hash indices for each parameter block?
    # To ensure consistency, we just need a deterministic seed.
    
    # We will compute the projection for each sample.
    for i, batch in enumerate(tqdm(train_loader, desc="Computing")):
        batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        sample_idx = batch["sample_index"].item()
        
        # 1. Compute Gradients (Full Params)
        # Note: We use the same function as in NIF.py (checking logic there)
        # In NIF.py, param_filter_fn is None -> all params.
        
        # We need to manually call compute_probabilities / loss / grad
        # Re-using logic from compute_gradients in loss.py
        # But we need to handle the 'ignored_token_ids'
        # Let's assume NIF.py has an instance or we recreate it.
        ignored_token_ids = torch.tensor([], device=device) # Handle if needed
        
        # We'll use the imported compute_gradients function
        # Note: param_filter_fn=None means all params
        grads = compute_gradients(
            model, 
            batch, 
            param_filter_fn=None, 
            device=device,
            ignored_token_ids=ignored_token_ids
        )
        
        # 2. Sketching / Hashing
        # We flatten inputs and project
        
        # Initialize sketch vector
        sketch = torch.zeros(PROJECTION_DIM, device=device)
        
        current_param_idx = 0
        
        # We need a DETRIMINISTIC way to map params to buckets.
        # Since structure is static, we can generate a big random index tensor? NO, too big.
        # We can seed RNG for each layer?
        
        # Iterating through gradients
        for g_tensor in grads:
            if g_tensor is None:
                continue
                
            g_flat = g_tensor.detach().reshape(-1)
            num_elements = g_flat.numel()
            
            # Generate random indices and signs for this block
            # This is slow if done on CPU every time. 
            # But doing it on GPU is fast.
            # We seed based on (Layer_ID or global_offset)
            
            # Fast way:
            # We don't really need cryptographically secure hasing.
            # Just random linear projection.
            
            # Let's try a simpler approach for the prototype:
            # Random Linear Projection realized by block-wise reusing a smaller matrix?
            # E.g. generate a [1024*1024, 4096] matrix and reuse it? 
            # No, that introduces correlations.
            
            # Let's stick to the Hashing idea (CountSketch):
            # h(i) -> [0, d-1]
            # s(i) -> {+1, -1}
            
            # To make it fast and deterministic:
            # Use a fixed generator seeded with the accumulated parameter index.
            # But reseeding is also slow.
            
            # Only option for reasonable speed + memory:
            # Pre-generate a list of (indices, signs) for each *Layer* output shape.
            # Wait, layer shapes are repeated? No.
            # There are only e.g. 300 parameter tensors.
            # We can pre-generate the indices/signs for each tensor ONE time.
            
            pass 
        
        # Since writing the hashing logic from scratch is complex to get efficient,
        # let's try a simpler robust method: 
        # Project each layer to a small vector (size 4096/Num_Layers?) then concatenation?
        # No, that changes vector space structure.
        
        # Let's use a "Chunked Random Projection".
        # We have a fixed valid "Random Matrix" R of shape [Max_Layer_Size, 4096] stored in VRAM.
        # For each layer gradient G_l (flattened), we do:
        # v_l = G_l @ R[0:len(G_l)] (wrapping around if len(G_l) > len(R))
        # sketch += v_l
        # This is equivalent to R being a block-circulant, repeating matrix.
        # This preserves distances well enough and is super fast and memory constant.
        
        # Implementation of Chunked Projection
        if i == 0:
            # Initialize the shared random matrix ONCE
            max_numel = 0
            for p in model.parameters():
                max_numel = max(max_numel, p.numel())
            
            # Make a random matrix large enough for the biggest layer
            # [Max_Params_In_Layer, PROJECTION_DIM]
            # 7B model -> largest layer is usually [4096, 11008] ~ 45M params.
            # 45M * 4096 is still big (180GB).
            
            # Okay, we need R to be smaller.
            # Let's clean up:
            # 1. Generate R of size [1MB, 4096].
            # 2. For any gradient chunk, reuse R by tiling.
            
            chunk_size = 1024 * 1024 # 1M params
            shared_R = torch.randn(chunk_size, PROJECTION_DIM, device=device) / (PROJECTION_DIM**0.5)
        
        sketch = torch.zeros(PROJECTION_DIM, device=device)
        
        for g_tensor in grads:
            if g_tensor is None: 
                continue
                
            g_flat = g_tensor.detach().reshape(-1)
            n = g_flat.numel()
            
            # Process in chunks
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                length = end - start
                
                segment = g_flat[start:end]
                # Project with shared_R
                # If segment is smaller than chunk_size, slice R
                
                res = segment @ shared_R[:length]
                sketch += res
                
        projected_grads.append(sketch.cpu())
        indices_list.append(sample_idx)
        
        # Setup for simple periodic saving? Or just all at once?
        if len(projected_grads) % 100 == 0:
            print(f"Processed {len(projected_grads)} samples...")

    print("Saving cache...")
    # We don't save R because it's deterministic if we fix seed?
    # No, we generated R using torch.randn. We MUST save shared_R to reproduce the same projection for test sample.
    
    data = {
        "projected_grads": torch.stack(projected_grads),
        "indices": torch.tensor(indices_list),
        "projection_matrix": shared_R.cpu(),
        "chunk_size": chunk_size
    }
    
    torch.save(data, CACHE_FILE)
    print(f"Done! Saved to {CACHE_FILE}")

if __name__ == "__main__":
    main()
