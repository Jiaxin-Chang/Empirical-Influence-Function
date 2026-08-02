"""
Inject ``annotations_by_target`` into an existing all_tokens report JSON.

Reads ``train_sample_details`` from the report, looks up each train sample in
the training JSONL, extracts ``attention_edges`` (grouped by dst / target token
index), and writes the augmented report.

Usage:
    python tools/inject_annotations_to_report.py \\
        --report correlation_matching_results_ce_only_..._all_tokens.json \\
        --train-data huawei_train_chatml.jsonl \\
        --output correlation_matching_results_ce_only_..._all_tokens_with_ann.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def load_train_line(jsonl_path: str, line_no: int) -> dict | None:
    """Read line ``line_no`` (0-based) from a JSONL file, skipping empty lines."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        current = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            if current == line_no:
                return json.loads(line)
            current += 1
    return None


# ChatML structural tokens that should never be annotation sources.
_CHATML_NOISE = frozenset({
    "<|im_start|>", "<|im_end|>", "system", "user", "assistant",
    "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>",
    "<|repo_name|>", "<|file_sep|>", "<|endoftext|>",
})


def build_annotations_by_target(sample: dict, tokenizer=None) -> dict[str, list[dict]]:
    """Group attention_edges by dst token index, filtering ChatML template noise.

    Result: ``{"49": [{"src": 44, "subtype": "bracket"}, ...], ...}``
    """
    edges = sample.get("attention_edges") or sample.get("edges") or []
    input_ids = sample.get("input_ids") or []

    ann: dict[str, list[dict]] = defaultdict(list)
    skipped = 0
    for e in edges:
        if isinstance(e, dict):
            src = int(e["src"])
            dst = int(e["dst"])

            # Decode and filter ChatML template noise from sources
            if tokenizer is not None and src < len(input_ids):
                src_tok = tokenizer.decode([input_ids[src]]).strip()
                if src_tok in _CHATML_NOISE:
                    skipped += 1
                    continue

            dst_key = str(dst)
            ann[dst_key].append({
                "src": src,
                "subtype": str(e.get("subtype", "")),
            })
        elif isinstance(e, (list, tuple)) and len(e) >= 2:
            dst = str(int(e[1]))
            src = int(e[0])
            if tokenizer is not None and src < len(input_ids):
                src_tok = tokenizer.decode([input_ids[src]]).strip()
                if src_tok in _CHATML_NOISE:
                    skipped += 1
                    continue
            ann[dst].append({
                "src": src,
                "subtype": "",
            })
    if skipped:
        print(f"    [filter] skipped {skipped} ChatML-template source edges")
    return dict(ann)


def main():
    parser = argparse.ArgumentParser(
        description="Inject annotations_by_target into an existing all_tokens report."
    )
    parser.add_argument("--report", required=True, help="Path to the *_all_tokens.json report.")
    parser.add_argument("--train-data", required=True, help="Path to the training JSONL.")
    parser.add_argument("--output", required=True, help="Path for the augmented output JSON.")
    parser.add_argument("--tokenizer-path", default=None,
                        help="Optional: tokenizer path for filtering ChatML template noise from annotations.")
    args = parser.parse_args()

    # ── Load tokenizer (optional, for filtering) ──────────────────────────
    tokenizer = None
    if args.tokenizer_path:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
        print(f"Tokenizer loaded from {args.tokenizer_path} (ChatML noise filter enabled)")

    # ── Load report ────────────────────────────────────────────────────────
    print(f"Loading report: {args.report}")
    with open(args.report, "r", encoding="utf-8") as f:
        report = json.load(f)

    tsd = report.get("train_sample_details")
    if not tsd:
        print("[error] Report has no train_sample_details. Nothing to inject.")
        return 1

    train_ids = sorted(int(k) for k in tsd.keys())
    print(f"  train_sample_details: {len(train_ids)} entries (IDs {train_ids[0]}..{train_ids[-1]})")

    # ── Inject annotations ─────────────────────────────────────────────────
    n_injected = 0
    n_skipped = 0
    for tid_str, detail in tsd.items():
        tid = int(tid_str)
        sample = load_train_line(args.train_data, tid)
        if sample is None:
            print(f"  [warn] train_sample_id={tid} not found in {args.train_data}")
            n_skipped += 1
            continue

        ann = build_annotations_by_target(sample, tokenizer=tokenizer)
        detail["annotations_by_target"] = ann
        if ann:
            n_injected += 1
        else:
            n_skipped += 1

    print(f"  injected: {n_injected} samples with annotations")
    print(f"  skipped:  {n_skipped} samples (not found or no edges)")

    # ── Write augmented report ─────────────────────────────────────────────
    print(f"Writing: {args.output}")
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
