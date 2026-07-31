#!/usr/bin/env python3
"""Precompute ALTI top-k saliency for annotation-viewer (run on a GPU machine).

Writes one JSON file per sample:
  <out_dir>/<sample_index>.json
  {
    "sample_index": 0,
    "uid": "...",
    "top_k": 6,
    "by_target": {
      "112": [{"src": 10, "score": 0.12}, ...]
    }
  }

Then copy the directory to your laptop and start the viewer WITHOUT --model:
  python -m server.main --data ... --saliency-cache ./saliency_cache

Examples:
  # first 20 samples, all answer-region targets
  python tools/annotation-viewer/scripts/precompute_saliency_cache.py \\
    --data go_single_train_v2_graphsignal_10k_compact_resp_edges.jsonl \\
    --model /path/to/Qwen2.5-Coder-7B-Instruct \\
    --out saliency_cache --indices 0-19 --targets answer

  # only annotation edge destinations (faster)
  python ... --indices 0,5,59 --targets edge_dst
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_indices(spec: str, n_samples: int) -> list[int]:
    """Parse '0-10,15,20-22' into sorted unique indices."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            if lo > hi:
                lo, hi = hi, lo
            for i in range(lo, hi + 1):
                if 0 <= i < n_samples:
                    out.add(i)
        else:
            i = int(part)
            if 0 <= i < n_samples:
                out.add(i)
    return sorted(out)


def answer_start(labels: list[int]) -> int:
    for i, lab in enumerate(labels):
        if int(lab) != -100:
            return i
    return len(labels)


def select_targets(obj: dict, mode: str) -> list[int]:
    input_ids = obj.get("input_ids") or []
    labels = [int(x) for x in (obj.get("label") or obj.get("labels") or [])]
    n = len(input_ids)
    ans = answer_start(labels) if labels else 0
    targets: set[int] = set()
    if mode in ("answer", "both"):
        for i in range(max(1, ans), n):
            targets.add(i)
    if mode in ("edge_dst", "both"):
        for e in obj.get("attention_edges") or []:
            if isinstance(e, dict):
                dst = int(e.get("dst", -1))
                if 0 < dst < n:
                    targets.add(dst)
    return sorted(targets)


def build_offsets(path: Path) -> list[int]:
    offsets: list[int] = []
    with path.open("rb") as f:
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                offsets.append(pos)
        offsets.append(f.tell())
    return offsets


def read_sample(path: Path, offsets: list[int], idx: int) -> dict:
    with path.open("rb") as f:
        f.seek(offsets[idx])
        return json.loads(f.readline().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--out", type=Path, required=True, help="cache directory")
    ap.add_argument("--indices", type=str, default="0-9",
                    help="sample indices, e.g. 0-19 or 0,5,10")
    ap.add_argument("--targets", choices=["answer", "edge_dst", "both"], default="answer",
                    help="which target positions to precompute")
    ap.add_argument("--top-k", type=int, default=6)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[3]  # tools/annotation-viewer/scripts -> repo
    sys.path.insert(0, str(repo / "src"))

    import torch
    from transformers import AutoModelForCausalLM
    from loss import compute_alti_saliency_vector  # type: ignore

    data = args.data.expanduser().resolve()
    out_dir = args.out.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Indexing {data} ...", flush=True)
    offsets = build_offsets(data)
    n_samples = len(offsets) - 1
    indices = parse_indices(args.indices, n_samples)
    print(f"  samples={n_samples}, will compute {len(indices)}: {indices[:20]}{'...' if len(indices)>20 else ''}", flush=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"Loading model {args.model} dtype={dtype} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        local_files_only=True,
    )
    model.eval()
    device = next(model.parameters()).device

    for si, idx in enumerate(indices):
        obj = read_sample(data, offsets, idx)
        targets = select_targets(obj, args.targets)
        input_ids = [int(x) for x in (obj.get("input_ids") or [])]
        by_target: dict[str, list[dict]] = {}
        print(f"[{si+1}/{len(indices)}] sample {idx} uid={obj.get('uid')} targets={len(targets)}", flush=True)

        for t in targets:
            if t <= 0 or t >= len(input_ids):
                continue
            ids = torch.tensor([input_ids], dtype=torch.long, device=device)
            batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            sal = compute_alti_saliency_vector(model, batch, t)
            scored = [
                (i, float(s))
                for i, s in enumerate(sal)
                if i < t and float(s) > 0
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            top = [{"src": i, "score": sc} for i, sc in scored[: args.top_k]]
            by_target[str(t)] = top

        payload = {
            "sample_index": idx,
            "uid": obj.get("uid"),
            "top_k": args.top_k,
            "targets_mode": args.targets,
            "by_target": by_target,
        }
        (out_dir / f"{idx}.json").write_text(
            json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    print(f"Done. Wrote {len(indices)} files under {out_dir}", flush=True)
    print("Copy this directory to your laptop, then:", flush=True)
    print(f"  python -m server.main --data <same.jsonl> --saliency-cache {out_dir.name}", flush=True)


if __name__ == "__main__":
    main()
