"""Remove prompt↔prompt attention_edges from compact graphsignal JSONL.

Keeps edges where at least one endpoint is in the response region
(label != -100). Writes a new file; does not overwrite the source.

Usage:
  python tools/filter_prompt_prompt_edges.py \\
    --input go_single_train_v2_graphsignal_10k_compact.json.bak \\
    --output go_single_train_v2_graphsignal_10k_compact_resp_edges.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def answer_start(labels: list) -> int:
    for i, lab in enumerate(labels):
        if int(lab) != -100:
            return i
    return len(labels)


def is_prompt_prompt(src: int, dst: int, ans: int) -> bool:
    return src < ans and dst < ans


def filter_obj(obj: dict) -> tuple[dict, int, int]:
    labels = obj.get("label") or []
    ans = answer_start(labels)
    edges = obj.get("attention_edges") or []
    kept = []
    dropped = 0
    for e in edges:
        if not isinstance(e, dict):
            continue
        try:
            src = int(e["src"])
            dst = int(e["dst"])
        except (KeyError, TypeError, ValueError):
            continue
        if is_prompt_prompt(src, dst, ans):
            dropped += 1
            continue
        kept.append(e)
    obj = dict(obj)
    obj["attention_edges"] = kept
    meta = dict(obj.get("annotation_meta") or {})
    meta["attention_edges"] = len(kept)
    meta["filtered_prompt_prompt_edges"] = dropped
    meta["edge_filter"] = "drop_prompt_prompt"
    obj["annotation_meta"] = meta
    return obj, len(kept), dropped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("go_single_train_v2_graphsignal_10k_compact.json.bak"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("go_single_train_v2_graphsignal_10k_compact_resp_edges.jsonl"),
    )
    args = parser.parse_args()

    inp: Path = args.input
    out: Path = args.output
    if not inp.exists():
        raise SystemExit(f"input not found: {inp}")

    n_samples = 0
    total_before = 0
    total_after = 0
    total_dropped = 0
    empty_after = 0

    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            before = len(obj.get("attention_edges") or [])
            obj2, after, dropped = filter_obj(obj)
            fout.write(json.dumps(obj2, ensure_ascii=False) + "\n")
            n_samples += 1
            total_before += before
            total_after += after
            total_dropped += dropped
            if after == 0:
                empty_after += 1
            if n_samples % 1000 == 0:
                print(f"  processed {n_samples} ...", flush=True)

    print("=" * 60)
    print(f"input:   {inp}")
    print(f"output:  {out}")
    print(f"samples: {n_samples}")
    print(f"edges before: {total_before}")
    print(f"edges after:  {total_after}")
    print(f"dropped prompt-prompt: {total_dropped} "
          f"({100.0 * total_dropped / max(total_before, 1):.1f}%)")
    print(f"samples with 0 edges left: {empty_after}")
    print("=" * 60)


if __name__ == "__main__":
    main()
