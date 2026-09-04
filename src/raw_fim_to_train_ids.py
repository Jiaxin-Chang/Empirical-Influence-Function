#!/usr/bin/env python3
"""Standalone: raw FIM JSONL (prompt + response) → ChatML training ids.

Copy **this one file** only. Needs: ``pip install transformers`` + a local
tokenizer/model directory.

Output per row::

    {"input_ids": [...], "label": [...], "attention_edges": []}

ChatML packing (Qwen-style): system/user labels = -100, assistant tokens trained.
``attention_edges`` is always [].

Example (Go)::

    python raw_fim_to_train_ids.py \\
      -i go_train_raw.jsonl \\
      -o go_train_ids.jsonl \\
      -t /path/to/Qwen2.5-Coder-7B-Instruct \\
      --language go --ids-only
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

DEFAULT_SYSTEM = "You are a helpful assistant."


def _load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=True,
    )


def encode_prompt_response(
    tokenizer,
    prompt: str,
    response: str,
    *,
    system: str = DEFAULT_SYSTEM,
) -> tuple[list[int], list[int]]:
    """Qwen ChatML: train assistant content only."""
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_tokens = tokenizer.encode("\n", add_special_tokens=False)

    def build_turn(role: str, content: str, *, train: bool) -> tuple[list[int], list[int]]:
        role_ids = (
            [im_start_id]
            + tokenizer.encode(role, add_special_tokens=False)
            + nl_tokens
        )
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        footer_ids = [im_end_id] + nl_tokens
        full_ids = role_ids + content_ids + footer_ids
        if train:
            labels = [-100] * len(role_ids) + content_ids + footer_ids
        else:
            labels = [-100] * len(full_ids)
        return full_ids, labels

    input_ids: list[int] = []
    labels: list[int] = []
    for role, content, train in (
        ("system", system, False),
        ("user", prompt, False),
        ("assistant", response, True),
    ):
        ids, labs = build_turn(role, content, train=train)
        input_ids.extend(ids)
        labels.extend(labs)
    return input_ids, labels


def _prompt_response(row: dict[str, Any]) -> tuple[str, str]:
    prompt = str(row.get("prompt") or row.get("input") or "").strip()
    response = ""
    for key in ("response", "label", "gold", "target"):
        v = row.get(key)
        if isinstance(v, str) and v.strip():
            response = v.strip()
            break
    return prompt, response


def convert_file(
    *,
    input_path: Path,
    output_path: Path,
    tokenizer_path: str,
    max_rows: int = 0,
    sample_n: int = 0,
    sample_seed: int = 42,
    max_seq_len: int = 0,
    keep_chatml_preview: bool = False,
    ids_only: bool = True,
    language: str = "go",
) -> dict[str, int]:
    tokenizer = _load_tokenizer(tokenizer_path)
    stats = {
        "seen": 0,
        "written": 0,
        "skip_empty": 0,
        "skip_long": 0,
        "candidates": 0,
        "sample_n": 0,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)

    selected_offsets: list[int] | None = None
    if sample_n and sample_n > 0:
        offsets: list[int] = []
        with input_path.open("rb") as fb:
            while True:
                pos = fb.tell()
                raw_b = fb.readline()
                if not raw_b:
                    break
                if raw_b.strip():
                    offsets.append(pos)
        stats["candidates"] = len(offsets)
        n = min(int(sample_n), len(offsets))
        stats["sample_n"] = n
        if n <= 0:
            return stats
        rng = random.Random(int(sample_seed))
        picked = rng.sample(offsets, n)
        picked.sort()
        selected_offsets = picked
        print(
            f"[sample] candidates={len(offsets)} pick={n} seed={sample_seed}",
            flush=True,
        )

    def _write_one(fout, line_no: int, row: dict[str, Any]) -> None:
        prompt, response = _prompt_response(row)
        if not prompt or not response:
            stats["skip_empty"] += 1
            return
        input_ids, labels = encode_prompt_response(tokenizer, prompt, response)
        if max_seq_len > 0 and len(input_ids) > max_seq_len:
            stats["skip_long"] += 1
            return
        if ids_only:
            out: dict[str, Any] = {
                "input_ids": input_ids,
                "label": labels,
                "attention_edges": [],
            }
        else:
            out = {
                "input_ids": input_ids,
                "label": labels,
                "attention_edges": [],
                "uid": row.get("uid") or row.get("task_id") or f"line_{line_no}",
                "task_id": row.get("task_id"),
                "language": row.get("language") or language,
                "length": len(input_ids),
                "source_file": str(input_path),
                "source_line": line_no,
            }
        if keep_chatml_preview:
            out["prompt"] = prompt
            out["response"] = response
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        stats["written"] += 1

    with output_path.open("w", encoding="utf-8") as fout:
        if selected_offsets is not None:
            with input_path.open("rb") as fb:
                for pos in selected_offsets:
                    if max_rows > 0 and stats["written"] >= max_rows:
                        break
                    fb.seek(pos)
                    raw_b = fb.readline()
                    raw = raw_b.decode("utf-8", errors="replace").strip()
                    if not raw:
                        stats["skip_empty"] += 1
                        continue
                    stats["seen"] += 1
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        stats["skip_empty"] += 1
                        continue
                    if not isinstance(row, dict):
                        stats["skip_empty"] += 1
                        continue
                    _write_one(fout, stats["seen"], row)
        else:
            with input_path.open(encoding="utf-8") as fin:
                for line_no, line in enumerate(fin, start=1):
                    raw = line.strip()
                    if not raw:
                        continue
                    stats["seen"] += 1
                    if max_rows > 0 and stats["written"] >= max_rows:
                        break
                    try:
                        row = json.loads(raw)
                    except json.JSONDecodeError:
                        stats["skip_empty"] += 1
                        continue
                    if not isinstance(row, dict):
                        stats["skip_empty"] += 1
                        continue
                    _write_one(fout, line_no, row)

    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True, help="raw FIM JSONL (prompt+response)")
    p.add_argument("-o", "--output", required=True, help="output JSONL")
    p.add_argument(
        "-t",
        "--tokenizer",
        required=True,
        help="tokenizer / model directory (local)",
    )
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    p.add_argument("--sample", type=int, default=0, help="random sample N rows (0=off)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for --sample")
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="skip rows longer than this (0 = no limit)",
    )
    p.add_argument("--language", default="go", help="language tag when not ids-only")
    p.add_argument(
        "--ids-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="write only input_ids+label+attention_edges=[] (default true)",
    )
    p.add_argument(
        "--keep-text",
        action="store_true",
        help="also keep prompt/response text in output rows",
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
        ids_only=bool(args.ids_only),
        language=str(args.language or "go"),
    )
    print(
        f"done: seen={stats['seen']} written={stats['written']} "
        f"skip_empty={stats['skip_empty']} skip_long={stats['skip_long']} "
        f"candidates={stats.get('candidates', 0)} sample_n={stats.get('sample_n', 0)} "
        f"-> {args.output}",
        flush=True,
    )
    return 0 if stats["written"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
