#!/usr/bin/env python3
"""Convert raw FIM JSONL (prompt + label/response) → ChatML → input_ids + label.

Uses the same ChatML packing as annotation-viewer ``corpus_encode`` /
``process_func_chatml``: system/user masked with -100, assistant trained.

Example::

    python -m src.raw_fim_to_train_ids \\
        --input test.jsonl \\
        --output test_train_ids.jsonl \\
        --tokenizer D:/AAAworks/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tokenizer(path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, local_files_only=True,
    )


def _encode(tokenizer, prompt: str, response: str) -> tuple[list[int], list[int]]:
    sys.path.insert(0, str(REPO_ROOT / "tools" / "annotation-viewer"))
    from server.corpus_encode import encode_prompt_response  # type: ignore

    return encode_prompt_response(tokenizer, prompt, response)


def _prompt_response(row: dict[str, Any]) -> tuple[str, str]:
    prompt = str(row.get("prompt") or row.get("input") or "").strip()
    # Prefer gold label over model predict for training targets.
    response = str(
        row.get("label")
        or row.get("response")
        or row.get("gold")
        or row.get("target")
        or ""
    ).strip()
    return prompt, response


def convert_file(
    *,
    input_path: Path,
    output_path: Path,
    tokenizer_path: str,
    max_rows: int = 0,
    max_seq_len: int = 0,
    keep_chatml_preview: bool = False,
) -> dict[str, int]:
    tokenizer = _load_tokenizer(tokenizer_path)
    stats = {"seen": 0, "written": 0, "skip_empty": 0, "skip_long": 0}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open(encoding="utf-8") as fin, output_path.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            raw = line.strip()
            if not raw:
                continue
            stats["seen"] += 1
            if max_rows > 0 and stats["written"] >= max_rows:
                break
            row = json.loads(raw)
            if not isinstance(row, dict):
                stats["skip_empty"] += 1
                continue
            prompt, response = _prompt_response(row)
            if not prompt or not response:
                stats["skip_empty"] += 1
                continue

            input_ids, labels = _encode(tokenizer, prompt, response)
            if max_seq_len > 0 and len(input_ids) > max_seq_len:
                stats["skip_long"] += 1
                continue

            out: dict[str, Any] = {
                "input_ids": input_ids,
                "label": labels,
                "uid": row.get("uid") or row.get("task_id") or f"line_{line_no}",
                "task_id": row.get("task_id"),
                "language": row.get("language") or "Go",
                "length": len(input_ids),
                "source_file": str(input_path),
                "source_line": line_no,
            }
            if keep_chatml_preview:
                out["prompt"] = prompt
                out["response"] = response
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            stats["written"] += 1

    return stats


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-i", "--input", required=True, help="raw FIM JSONL (prompt+label)")
    p.add_argument("-o", "--output", required=True, help="output JSONL (input_ids+label)")
    p.add_argument(
        "-t",
        "--tokenizer",
        required=True,
        help="tokenizer / model directory (local)",
    )
    p.add_argument("--max-rows", type=int, default=0, help="0 = all rows")
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="skip rows longer than this (0 = no limit)",
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
        max_seq_len=args.max_seq_len,
        keep_chatml_preview=bool(args.keep_text),
    )
    print(
        f"done: seen={stats['seen']} written={stats['written']} "
        f"skip_empty={stats['skip_empty']} skip_long={stats['skip_long']} "
        f"-> {args.output}",
        flush=True,
    )
    return 0 if stats["written"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
