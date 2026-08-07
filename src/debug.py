"""
Quick diagnostic script: checks model generation output for a given test sample
and shows exactly which tokens are generated + whether they pass trivial filtering.

Usage (on the server):
    CUDA_VISIBLE_DEVICES=6,7 python debug_inference.py --test-index 58
"""

import argparse
import torch
from functools import partial
from transformers import DataCollatorForSeq2Seq

from src.NIF import (
    load_model_and_tokenizer,
    load_samples_from_formal_jsonl,
    build_single_sample_dataset,
    _find_subseq_start,
)
from src.process_data import process_func_chatml

# ── mirrors intervention_experiment.py ──────────────────────────────────────
_TRIVIAL_STRIPPED = {"{", "}", "(", ")", "[", "]", ",", ";"}
MAX_OUTPUT_TOKENS = 40


def is_trivial_token(tokenizer, token_id: int) -> bool:
    tok_str = tokenizer.decode([token_id])
    stripped = tok_str.strip()
    if not stripped:
        return True
    if stripped in _TRIVIAL_STRIPPED:
        return True
    if len(stripped) == 1 and not (stripped.isalnum() or stripped == "_"):
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-index", type=int, default=58)
    parser.add_argument("--max-new-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer()
    convert_to_chatml = partial(process_func_chatml, tokenizer=tokenizer)

    test_samples = load_samples_from_formal_jsonl("sft_test.jsonl")
    sample = test_samples[args.test_index]
    print(f"\n[Test sample {args.test_index}]")
    print(f"  Input (first 200 chars): {sample['input'][:200]!r}")
    print(f"  GT output (first 200 chars): {sample['output'][:200]!r}")

    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model,
        padding=True, label_pad_token_id=-100, return_tensors="pt",
    )
    test_ds = build_single_sample_dataset(sample, convert_to_chatml)
    batch = base_collator([test_ds[0]])

    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)

    # ── Find prompt length (response start) ────────────────────────────────
    marker_ids = tuple(tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False))
    prompt_len = _find_subseq_start(input_ids[0], marker_ids) + 3
    print(f"\n  prompt_len = {prompt_len}  (total input tokens = {input_ids.size(1)})")

    # Trim to prompt only for generation
    trimmed_ids  = input_ids[:, :prompt_len]
    trimmed_mask = attention_mask[:, :prompt_len]

    print(f"\n[Generating up to {args.max_new_tokens} new tokens...]")
    model.eval()
    with torch.no_grad():
        gen_out = model.generate(
            input_ids=trimmed_ids,
            attention_mask=trimmed_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            eos_token_id=[tokenizer.eos_token_id, tokenizer.pad_token_id],
            pad_token_id=tokenizer.pad_token_id,
        )

    new_ids = gen_out[0, prompt_len:].tolist()
    print(f"  Generated {len(new_ids)} token(s)\n")

    # ── Token-by-token breakdown ────────────────────────────────────────────
    trivial_count = 0
    semantic_count = 0
    print(f"  {'idx':>5}  {'id':>7}  {'trivial':>7}  repr")
    print(f"  {'-'*5}  {'-'*7}  {'-'*7}  {'-'*40}")
    for offset, tok_id in enumerate(new_ids):
        abs_idx = prompt_len + offset
        trivial = is_trivial_token(tokenizer, tok_id)
        tok_repr = repr(tokenizer.decode([tok_id]))
        marker = "  <-- TRIVIAL" if trivial else ""
        print(f"  {abs_idx:>5}  {tok_id:>7}  {str(trivial):>7}  {tok_repr}{marker}")
        if trivial:
            trivial_count += 1
        else:
            semantic_count += 1

    print(f"\n  Summary: {semantic_count} semantic, {trivial_count} trivial out of {len(new_ids)} total tokens")
    print(f"\n  Full decoded response:\n{tokenizer.decode(new_ids)!r}")


if __name__ == "__main__":
    main()
