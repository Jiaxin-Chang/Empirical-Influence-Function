import os
import json
import torch
import torch.nn.functional as F
from functools import partial
from heapq import nlargest
from accelerate import Accelerator
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    set_seed,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.NIF import (
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
    compute_lm_head_ce_gradient_no_backward,
    compute_lm_head_ce_gradient_scores_no_backward,
    compute_saliency_feature_proxy,
)

# ====== DATA LOADING (auto-detects format) ======

def load_samples(jsonl_path: str) -> list[dict]:
    """Load samples from a JSONL file, auto-detecting the format.

    Supported formats:

    **Format A – messages array (original):**
    ``{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."},
                    {"role": "assistant", "content": "..."}]}``

    **Format B – flat fields (new):**
    ``{"prompt": "...", "response": "...", "task_id": "...", "system": "..."}``
    ``system`` is optional; ``task_id`` is preserved for output file naming.

    Both formats are normalised to ``{system, input, output, task_id}``.
    """
    samples: list[dict] = []
    seen_inputs: set[str] = set()

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)

            # ── Format A: messages array ──────────────────────────────────
            if "messages" in obj:
                msgs = obj["messages"]
                if len(msgs) < 3 or not msgs[2]["content"]:
                    continue
                system = msgs[0]["content"]
                inp    = msgs[1]["content"]
                output = msgs[2]["content"]
                task_id = obj.get("task_id", "")

            # ── Format B: flat prompt / response ──────────────────────────
            elif "prompt" in obj and "response" in obj:
                system  = obj.get("system", "")
                inp     = obj["prompt"]
                output  = obj["response"]
                task_id = str(obj.get("task_id", ""))
                if not output:
                    continue

            else:
                continue  # unrecognised format, skip

            if inp in seen_inputs:
                continue
            seen_inputs.add(inp)

            samples.append({
                "system":  system,
                "input":   inp,
                "output":  output,
                "task_id": task_id,
            })

    return samples


# ====== MODEL LOADING (generic — supports Qwen2, Qwen3, Qwen3-MoE, etc.) ======

def _patch_model_with_attn_hook(model: torch.nn.Module) -> torch.nn.Module:
    """Replace the Qwen2-specific subclass approach with a generic forward hook.

    After patching, any call to ``model(..., save_last_attention=True)`` will:
      1. Register a hook on the *last* self-attention module before the forward pass.
      2. Capture the attention weight tensor returned by that module.
      3. Remove the hook and return a :class:`CausalLMOutputWithPast` whose
         ``attentions`` field holds the captured weight (matching the interface
         expected by ``compute_answer_only_saliency_masked_loss``).

    Works for dense models (Qwen2, Qwen3) **and** MoE models (Qwen3.5 35B A3B)
    because it only touches the attention module output, not the MoE routing.

    Requires ``attn_implementation="eager"`` so that the attention module actually
    computes and returns weight tensors (flash-attention variants return ``None``).
    """
    model.last_attention = None
    _orig_forward = model.forward

    def _patched_forward(*args, save_last_attention: bool = False, **kwargs):
        model.last_attention = None
        handle = None

        if save_last_attention:
            last_attn = model.model.layers[-1].self_attn

            def _hook(_module, _inputs, output):
                # Attention modules return (attn_output, attn_weights[, past_kv, ...]).
                # attn_weights is a Tensor when attn_implementation="eager",
                # and None for flash/sdpa variants.
                if isinstance(output, tuple) and len(output) >= 2:
                    model.last_attention = output[1]

            handle = last_attn.register_forward_hook(_hook)

        try:
            outputs = _orig_forward(*args, **kwargs)
        finally:
            if handle is not None:
                handle.remove()

        if save_last_attention:
            return CausalLMOutputWithPast(
                loss=outputs.loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=(model.last_attention,) if model.last_attention is not None else None,
            )
        return outputs

    model.forward = _patched_forward
    return model


def load_model_and_tokenizer(
    model_path: str | None = None,
    attn_implementation: str = "eager",
    max_gpu_memory: str | None = None,
):
    """Load any AutoModelForCausalLM checkpoint and patch it with the generic
    attention hook. ``save_last_attention=True`` only returns attention weights
    when the backend materializes them (for example ``eager``).

    Args:
        model_path: Absolute path to the model checkpoint directory.
                    Defaults to the standard NIF checkpoint location.
    """
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__), "sft", "scripts", "nif-checkpoints", "checkpoint-full"
        )
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    print(f"Using attention implementation: {attn_implementation}")
    config = AutoConfig.from_pretrained(
        model_path,
        attn_implementation=attn_implementation,
        output_attentions=False,
        use_cache=False,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        device_map="auto",
        max_memory=(
            {i: max_gpu_memory for i in range(torch.cuda.device_count())}
            if max_gpu_memory and torch.cuda.is_available()
            else None
        ),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model.eval()

    model = _patch_model_with_attn_hook(model)
    return model, tokenizer


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
SEQUENCE_LENGTH_LIMIT = 3000   # Default guard for gradient-heavy train sample scans
# Global pre-screen pool size. The full training set is scanned ONCE with the full-response
# CE gradient to obtain this pool, then each per-token re-ranking only scans the pool
# (COARSE_POOL_SIZE samples) instead of the full training set.
# Cost: 1 × N_train (global) + N_tokens × COARSE_POOL_SIZE (per-token re-rank)
COARSE_POOL_SIZE = 100

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
    train_loader,
    accel_device: torch.device,   # accelerator.device for batch loading
    lm_head_device: torch.device, # device where lm_head lives (grad computed here)
    desc: str = "Scanning Train Samples",
    allowed_indices: set | None = None,
    max_seq_len: int | None = None,
) -> list[tuple[int, float]]:
    """Compute cosine similarity on GPU (lm_head_device). Returns LOCAL scores for this process."""
    sample_scores: list[tuple[int, float]] = []
    empty_ignored = torch.tensor([], device=accel_device)
    skipped_long = 0

    for batch in tqdm(train_loader, desc=desc, leave=False):
        train_indices = batch["sample_index"].view(-1).tolist()
        keep_rows = []
        for row, train_idx in enumerate(train_indices):
            if allowed_indices is not None and train_idx not in allowed_indices:
                continue
            if max_seq_len is not None:
                seq_len = int(batch["attention_mask"][row].sum().item())
                if seq_len > max_seq_len:
                    skipped_long += 1
                    continue
            keep_rows.append(row)

        if not keep_rows:
            continue

        rows = torch.tensor(keep_rows, dtype=torch.long, device=batch["input_ids"].device)
        batch_kept = {
            k: v.index_select(0, rows).to(accel_device)
            for k, v in batch.items()
            if isinstance(v, torch.Tensor) and k != "sample_index"
        }
        scores = compute_lm_head_ce_gradient_scores_no_backward(
            model=model,
            batch=batch_kept,
            device=accel_device,
            ignored_token_ids=empty_ignored,
            test_ce_grad=test_ce_grad,
            score_device=lm_head_device,
        )
        for row, score in zip(keep_rows, scores):
            sample_scores.append((int(train_indices[row]), float(score)))

        del batch_kept, scores, rows

    if skipped_long:
        print(f"[DEBUG] {desc}: skipped {skipped_long} samples longer than {max_seq_len} tokens.", flush=True)

    return sample_scores


def run_causal_intervention_experiment(
    all_tokens: bool = False,
    model_path: str | None = None,
    train_data: str = "sft_train.jsonl",
    test_data: str = "sft_test.jsonl",
    train_limit: int | None = None,
    attn_implementation: str | None = None,
    prescreen_max_seq_len: int | None = SEQUENCE_LENGTH_LIMIT,
    max_gpu_memory: str | None = None,
    corr_feature_mode: str = "auto",
    prescreen_batch_size: int = 1,
):
    import sys; sys.stdout.reconfigure(line_buffering=True)
    print("[DEBUG] Initializing Accelerator...", flush=True)
    accelerator = Accelerator()
    print(f"[DEBUG] Accelerator ready. num_processes={accelerator.num_processes}, device={accelerator.device}", flush=True)
    set_seed(SEED)

    if attn_implementation is None:
        # All-tokens mode uses embedding-gradient saliency and CE gradients; it
        # does not need materialized attention weights. SDPA avoids the large
        # eager attention matrix that can OOM during prescreening.
        attn_implementation = "sdpa" if all_tokens else "eager"
    if prescreen_max_seq_len is not None and prescreen_max_seq_len <= 0:
        prescreen_max_seq_len = None
    prescreen_batch_size = max(1, int(prescreen_batch_size))
    model, tokenizer = load_model_and_tokenizer(
        model_path,
        attn_implementation=attn_implementation,
        max_gpu_memory=max_gpu_memory,
    )
    if corr_feature_mode == "auto":
        model_type = str(getattr(model.config, "model_type", "")).lower()
        class_name = model.__class__.__name__.lower()
        corr_feature_mode = "saliency_proxy" if "moe" in model_type or "moe" in class_name else "second_order"
    print(f"[DEBUG] correlation feature mode: {corr_feature_mode}", flush=True)
    print("[DEBUG] Model and tokenizer loaded.", flush=True)
    param_filter = lm_head_filter

    convert_to_chatml = partial(process_func_chatml, tokenizer=tokenizer)

    print("[DEBUG] Loading train/test samples...", flush=True)
    train_samples = load_samples(train_data)
    test_samples  = load_samples(test_data)
    if train_limit is not None and train_limit < len(train_samples):
        print(f"[DEBUG] Limiting train samples: {len(train_samples)} -> {train_limit}", flush=True)
        train_samples = train_samples[:train_limit]
    print(f"[DEBUG] Loaded {len(train_samples)} train, {len(test_samples)} test samples.", flush=True)

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding=True, label_pad_token_id=-100, return_tensors="pt",
    )
    collator = CustomCollator(base_collator)

    print("[DEBUG] Building train dataset...", flush=True)
    train_ds = build_train_dataset(train_samples, convert_to_chatml)
    print(f"[DEBUG] Train dataset built: {len(train_ds)} samples.", flush=True)
    train_loader = torch.utils.data.DataLoader(
        DatasetWrapper(train_ds), batch_size=prescreen_batch_size, collate_fn=collator,
    )
    print("[DEBUG] Calling accelerator.prepare(train_loader)...", flush=True)
    train_loader = accelerator.prepare(train_loader)
    print("[DEBUG] train_loader prepared.", flush=True)

    infer_fw = NewInferenceFunction(
        model=model, tokenizer=tokenizer,
        train_loader=train_loader, accelerator=accelerator,
        param_filter_fn=param_filter, top_k=20,
    )
    print("[DEBUG] NewInferenceFunction created.", flush=True)

    # lm_head might live on a different device under device_map="auto"
    filtered_params = [p for n, p in model.named_parameters() if param_filter(n, p)]
    lm_head_device  = filtered_params[0].device if filtered_params else accelerator.device
    print(f"[DEBUG] lm_head_device={lm_head_device}", flush=True)

    marker_ids = tuple(tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))

    # ── Build full test sequence (prompt + generated response) ──────────────
    _cur_test     = test_samples[SELECTED_TEST_SAMPLE_INDEX]
    # Use task_id for output file naming; fall back to index if absent.
    _task_id      = _cur_test.get("task_id") or f"test{SELECTED_TEST_SAMPLE_INDEX}"
    test_ds       = build_single_sample_dataset(_cur_test, convert_to_chatml)
    raw_test_batch = base_collator([test_ds[0]])
    print(f"[DEBUG] Moving test batch to device {accelerator.device}...", flush=True)
    raw_test_batch = {k: v.to(accelerator.device) for k, v in raw_test_batch.items()}
    print(f"[DEBUG] Test batch on device. input_ids shape={raw_test_batch['input_ids'].shape}", flush=True)

    infer_fw.model.eval()
    print("[DEBUG] Starting infer_fw.infer(raw_test_batch)...", flush=True)
    gen_result = infer_fw.infer(raw_test_batch, skip_saliency=True)
    print("[DEBUG] Inference done.", flush=True)
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

    def _compute_correlation_feature(batch, target_idx_in_seq: int, source_idx_in_seq: int) -> torch.Tensor:
        if corr_feature_mode == "second_order":
            return compute_correlation_second_order_gradient(
                model=model,
                batch=batch,
                target_idx_in_seq=target_idx_in_seq,
                source_idx_in_seq=source_idx_in_seq,
                param_filter_fn=param_filter,
            )
        if corr_feature_mode == "saliency_proxy":
            return compute_saliency_feature_proxy(
                model=model,
                batch=batch,
                target_idx_in_seq=target_idx_in_seq,
                source_idx_in_seq=source_idx_in_seq,
            )
        raise ValueError(f"Unsupported corr_feature_mode: {corr_feature_mode}")

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
                start_sys = _find_subseq_start(tr_batch["input_ids"][0], marker_ids) + len(marker_ids)
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

        # Move tr_batch to device for correlation feature computation
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

                train_feat = _compute_correlation_feature(tr_batch_gpu, t_tr, s_idx)
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
        grad = compute_lm_head_ce_gradient_no_backward(
            model=model,
            batch=single_tok_batch,
            device=accelerator.device,
            ignored_token_ids=torch.tensor([], device=accelerator.device),
        )
        return grad.reshape(-1).to(lm_head_device).detach()

    # ════════════════════════════════════════════════════════════════════════════
    # Helper: compute test-side CE gradient aggregated over ALL response tokens
    # Used for the global one-shot coarse pre-screen in all_tokens mode.
    # test_batch["labels"] already masks prompt positions with -100, so
    # The analytic lm_head gradient naturally aggregates CE loss across all
    # response tokens.
    # ════════════════════════════════════════════════════════════════════════════
    def _test_ce_grad_full_response() -> torch.Tensor:
        grad = compute_lm_head_ce_gradient_no_backward(
            model=model,
            batch=test_batch,
            device=accelerator.device,
            ignored_token_ids=torch.tensor([], device=accelerator.device),
        )
        return grad.reshape(-1).to(lm_head_device).detach()

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

        print(f"Computing test correlation features ({corr_feature_mode})...")
        test_corr_features: dict = {}
        with torch.inference_mode(False):
            for item in top_test_correlations:
                p_idx = item["source_token_index"]
                print(f"  '{item['source_token']}' -> '{item['target_token']}' (sal={item['saliency_score']:.4f})")
                feat = _compute_correlation_feature(test_batch, TOKEN_INDEX_TO_RETRIEVE, p_idx)
                test_corr_features[p_idx] = (feat.to(lm_head_device), item["source_token"], item["saliency_score"])

        print(f"\n=== Stage 2: Coarse screening (single token {TOKEN_INDEX_TO_RETRIEVE}) ===")
        test_ce_grad  = _test_ce_grad_for_token(TOKEN_INDEX_TO_RETRIEVE)
        local_scores  = _screen_training_set(
            model, test_ce_grad, train_loader,
            accelerator.device, lm_head_device, desc="Stage 2",
            max_seq_len=prescreen_max_seq_len,
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

        report_filename = f"correlation_matching_results_{_task_id}_tok{TOKEN_INDEX_TO_RETRIEVE}.json"

    # ════════════════════════════════════════════════════════════════════════════
    # ── ALL-TOKENS MODE ──────────────────────────────────────────────────────
    # ════════════════════════════════════════════════════════════════════════════
    else:
        max_end = min(prompt_len + MAX_OUTPUT_TOKENS, seq_len)
        valid_test_tokens = [
            t for t in range(prompt_len, max_end)
            if not is_trivial_token(tokenizer, int(test_batch["input_ids"][0, t].item()))
        ]
        print(f"\nAll-tokens mode: {len(valid_test_tokens)} semantic tokens "
              f"(from {max_end - prompt_len} response tokens, trivial skipped)")

        # ── Stage 0: Global coarse pre-screen (one pass over the full training set) ──
        # Aggregate CE loss over all response tokens → single gradient vector G_test.
        # Cost: 1 × N_train  (vs. N_tokens × N_train in the old per-token approach)
        print(f"\n=== Global Pre-Screen: full training set → Top-{COARSE_POOL_SIZE} pool ===")
        full_response_ce_grad = _test_ce_grad_full_response()
        local_pool_scores = _screen_training_set(
            model, full_response_ce_grad, train_loader,
            accelerator.device, lm_head_device, desc="Global Pre-Screen",
            max_seq_len=prescreen_max_seq_len,
        )
        all_pool_scores = _gather_scores(accelerator, local_pool_scores, accelerator.device)
        coarse_pool: set[int] = {
            idx for idx, _ in nlargest(COARSE_POOL_SIZE, all_pool_scores, key=lambda x: x[1])
        }
        del full_response_ce_grad
        print(f"  Coarse pool ({len(coarse_pool)} samples): {sorted(coarse_pool)}")

        # Materialise pool as a dedicated DataLoader so per-token Stage 2 truly iterates
        # only COARSE_POOL_SIZE times (not N_train times with skip logic).
        idx_to_row  = {int(train_ds[i]["sample_index"]): i for i in range(len(train_ds))}
        pool_rows   = sorted(idx_to_row[idx] for idx in coarse_pool if idx in idx_to_row)
        pool_ds     = train_ds.select(pool_rows)
        pool_loader = torch.utils.data.DataLoader(
            DatasetWrapper(pool_ds), batch_size=prescreen_batch_size, collate_fn=collator,
        )
        pool_loader = accelerator.prepare(pool_loader)
        print(f"  pool_loader built: {len(pool_rows)} samples")

        # ── Per-token loop: re-rank within pool_loader, then run Stage 3 ──
        per_token_results: list  = []
        train_sample_cache: dict = {}   # str(train_idx) → detail dict (with _candidate_pairs, _tr_batch_cpu)
        pair_id_counter = 0

        for t in tqdm(valid_test_tokens, desc="Test tokens",
                      disable=not accelerator.is_local_main_process):
            target_tok_id   = int(test_batch["input_ids"][0, t].item())
            target_tok_text = tokenizer.decode([target_tok_id])
            print(f"\n=== Token {t}: '{target_tok_text}' ===")

            # Stage 1a: cheap saliency at t
            with torch.inference_mode(False):
                sal_vec = compute_full_saliency_vector(model, test_batch, t)
            top_test_corr = nlargest(TOP_K_PROMPT_TOKENS, enumerate(sal_vec), key=lambda x: x[1])
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

            # Stage 1b: test correlation features (kept on lm_head_device, freed after this token)
            print(f"  Computing {len(top_test_correlations)} test correlation features ({corr_feature_mode})...")
            test_corr_features: dict = {}
            with torch.inference_mode(False):
                for item in top_test_correlations:
                    p_idx = item["source_token_index"]
                    feat = _compute_correlation_feature(test_batch, t, p_idx)
                    test_corr_features[p_idx] = (feat.to(lm_head_device), item["source_token"], item["saliency_score"])

            # Stage 2: per-token CE grad → re-rank within pool_loader only
            # pool_loader contains exactly COARSE_POOL_SIZE samples — no skip logic needed.
            token_ce_grad  = _test_ce_grad_for_token(t)
            local_scores   = _screen_training_set(
                model, token_ce_grad, pool_loader,
                accelerator.device, lm_head_device,
                desc=f"Stage2 t={t}",
                max_seq_len=prescreen_max_seq_len,
            )
            all_scores     = _gather_scores(accelerator, local_scores, accelerator.device)
            related_samples = nlargest(TOP_K_TRAIN_SAMPLES, all_scores, key=lambda x: x[1])
            del token_ce_grad
            print(f"  Top-{TOP_K_TRAIN_SAMPLES} from pool: "
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
                "screening":          "global_pool_then_per_token_rerank",
                "config": {
                    "TOP_K_PROMPT_TOKENS":     TOP_K_PROMPT_TOKENS,
                    "TOP_K_TRAIN_SAMPLES":     TOP_K_TRAIN_SAMPLES,
                    "COARSE_POOL_SIZE":        COARSE_POOL_SIZE,
                    "TOP_TARGETS":             TOP_TARGETS,
                    "TOP_K_SOURCE_PER_TARGET": TOP_K_SOURCE_PER_TARGET,
                    "CONTEXT_WINDOW_SIZE":     CONTEXT_WINDOW_SIZE,
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

        report_filename = f"correlation_matching_results_{_task_id}_all_tokens.json"

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
    parser.add_argument(
        "--model-path", type=str, default=None,
        help=(
            "Path to the model checkpoint directory. "
            "Supports any AutoModelForCausalLM architecture "
            "(Qwen2, Qwen3, Qwen3-MoE, etc.). "
            "Defaults to src/sft/scripts/nif-checkpoints/checkpoint-full."
        ),
    )
    parser.add_argument(
        "--train-data", type=str, default="sft_train.jsonl",
        help="Path to the training JSONL file (default: sft_train.jsonl).",
    )
    parser.add_argument(
        "--test-data", type=str, default="sft_test.jsonl",
        help="Path to the test JSONL file containing error samples (default: sft_test.jsonl).",
    )
    parser.add_argument(
        "--train-limit", type=int, default=None,
        help="Limit the number of training samples to process (default: all). Useful for debugging.",
    )
    parser.add_argument(
        "--attn-implementation", type=str, default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help=(
            "Attention backend for model loading. Defaults to sdpa in all-tokens "
            "mode and eager in single-token mode."
        ),
    )
    parser.add_argument(
        "--prescreen-max-seq-len", type=int, default=SEQUENCE_LENGTH_LIMIT,
        help=(
            "Skip training samples longer than this during prescreen/rerank. "
            "Use 0 to disable this guard."
        ),
    )
    parser.add_argument(
        "--max-gpu-memory", type=str, default=None,
        help=(
            "Per-GPU max_memory passed to from_pretrained, e.g. 26GiB. "
            "Useful for leaving activation headroom with device_map=auto."
        ),
    )
    parser.add_argument(
        "--corr-feature-mode", type=str, default="auto",
        choices=["auto", "second_order", "saliency_proxy"],
        help=(
            "Feature used for fine correlation matching. auto uses saliency_proxy "
            "for MoE models because their grouped-mm kernels do not support 2nd-order AD."
        ),
    )
    parser.add_argument(
        "--prescreen-batch-size", type=int, default=1,
        help="Batch size for global prescreen and pool rerank forwards.",
    )
    args = parser.parse_args()

    if args.test_index is not None:
        SELECTED_TEST_SAMPLE_INDEX = args.test_index
    if args.token_index is not None:
        TOKEN_INDEX_TO_RETRIEVE = args.token_index

    mode_str = "all_tokens" if args.all_tokens else f"single_token tok={TOKEN_INDEX_TO_RETRIEVE}"
    print(f"[intervention] test_index={SELECTED_TEST_SAMPLE_INDEX}  mode={mode_str}")
    run_causal_intervention_experiment(
        all_tokens=args.all_tokens,
        model_path=args.model_path,
        train_data=args.train_data,
        test_data=args.test_data,
        train_limit=args.train_limit,
        attn_implementation=args.attn_implementation,
        prescreen_max_seq_len=args.prescreen_max_seq_len,
        max_gpu_memory=args.max_gpu_memory,
        corr_feature_mode=args.corr_feature_mode,
        prescreen_batch_size=args.prescreen_batch_size,
    )
