#!/usr/bin/env python3
"""Java [MASK] train JSONL → ChatML → input_ids + label + empty attention_edges.

Input row shape (jfreechart)::

    {
      "prompt": "... * Incomplete Code:\\n... [MASK] ...",
      "response": "return this.notify;",
      "language": "java",
      ...
    }

Output row::

    {"input_ids": [...], "label": [...], "attention_edges": []}

Uses the same ChatML packer as annotation-viewer ``corpus_encode`` /
``raw_fim_to_train_ids`` (system/user = -100, assistant tokens trained).
No tree-sitter / no LLM — edges are always ``[]``.

Example::

    python -m src.java_mask_to_ids \\
      -i /mnt/md124/jiaxin/Empirical-Influence-Function/jfreechart_fim_train.jsonl \\
      -o /mnt/md124/jiaxin/Empirical-Influence-Function/jfreechart_fim_train_ids.jsonl \\
      -t /mnt/md124/jiaxin/models/Qwen3-8B
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.raw_fim_to_train_ids import convert_file

DEFAULT_INPUT = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/jfreechart_fim_train.jsonl"
)
DEFAULT_OUTPUT = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/jfreechart_fim_train_ids.jsonl"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", default=DEFAULT_INPUT)
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT)
    p.add_argument(
        "-t",
        "--tokenizer",
        required=True,
        help="tokenizer / model dir (e.g. Qwen3-8B)",
    )
    p.add_argument(
        "--max-rows",
        "--head",
        type=int,
        default=0,
        dest="max_rows",
        help="only convert the first N rows (sequential; 0 = all)",
    )
    p.add_argument(
        "--sample",
        type=int,
        default=0,
        help="randomly sample N non-empty rows then convert",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for --sample")
    p.add_argument("--max-seq-len", type=int, default=0)
    p.add_argument(
        "--keep-text",
        action="store_true",
        help="also keep prompt/response in output",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats = convert_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        tokenizer_path=args.tokenizer,
        max_rows=args.max_rows,
        sample_n=int(args.sample or 0),
        sample_seed=int(args.seed),
        max_seq_len=args.max_seq_len,
        keep_chatml_preview=bool(args.keep_text),
        ids_only=True,
        language="java",
    )
    print(
        f"[java-mask-to-ids] seen={stats['seen']} written={stats['written']} "
        f"skip_empty={stats['skip_empty']} skip_long={stats['skip_long']} "
        f"candidates={stats.get('candidates', 0)} sample_n={stats.get('sample_n', 0)} "
        f"-> {args.output}",
        flush=True,
    )
    return 0 if stats["written"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
