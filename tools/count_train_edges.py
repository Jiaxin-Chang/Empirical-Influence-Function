"""Count attention_edges in the train JSONL used by degradation retrieve.

No GPU. Reads EIF_TRAIN_DATA from eif_api.env (or --jsonl).

  python tools/count_train_edges.py
  python tools/count_train_edges.py --jsonl path/to/train.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SEQ_LIMIT = 3000  # src/intervention_experiment.SEQUENCE_LENGTH_LIMIT


def _load_env_train_path() -> Path | None:
    env_path = REPO_ROOT / "eif_api.env"
    if not env_path.is_file():
        return None
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != "EIF_TRAIN_DATA":
            continue
        p = Path(value.strip().strip('"').strip("'"))
        return p
    return None


def _parse_edges(edges) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for e in edges or []:
        try:
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                src, dst = int(e[0]), int(e[1])
                subtype = str(e[3]) if len(e) >= 4 else ""
            elif isinstance(e, dict):
                src = int(e.get("src", e.get("source", -1)))
                dst = int(e.get("dst", e.get("target", -1)))
                subtype = str(e.get("subtype") or "")
            else:
                continue
        except (TypeError, ValueError, AttributeError):
            continue
        if src < 0 or dst < 0 or src == dst:
            continue
        out.append((src, dst, subtype or "(none)"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Count train attention_edges (CPU only).")
    parser.add_argument("--jsonl", type=str, default="", help="Override train JSONL path")
    parser.add_argument("--top", type=int, default=20, help="Print this many samples")
    args = parser.parse_args()

    candidates: list[Path] = []
    if args.jsonl.strip():
        candidates.append(Path(args.jsonl.strip()))
    env_p = _load_env_train_path()
    if env_p is not None:
        candidates.append(env_p)
    local = REPO_ROOT / "smoke_train_data.jsonl"
    if local.is_file():
        candidates.append(local)

    jsonl = next((p for p in candidates if p.is_file()), None)
    if jsonl is None:
        tried = ", ".join(str(p) for p in candidates) or "(none)"
        raise SystemExit(f"Train JSONL not found. Tried: {tried}")

    n_rows = 0
    n_compact = 0
    n_with_edges = 0
    n_edges = 0
    n_in_range = 0
    n_skip_seq = 0
    n_skip_seq_edges = 0
    subtypes: Counter[str] = Counter()
    per_sample: list[tuple[int, str, int, int, int, bool]] = []

    with jsonl.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            n_rows += 1
            obj = json.loads(line)
            ids = obj.get("input_ids")
            labs = obj.get("label", obj.get("labels"))
            if not isinstance(ids, list) or not ids:
                continue
            if not isinstance(labs, list) or len(labs) != len(ids):
                continue
            n_compact += 1
            seq = len(ids)
            too_long = seq > SEQ_LIMIT
            parsed = _parse_edges(obj.get("attention_edges") or obj.get("edges"))
            in_range = [(s, d, t) for s, d, t in parsed if s < seq and d < seq]
            if parsed:
                n_with_edges += 1
            n_edges += len(parsed)
            if too_long:
                n_skip_seq += 1
                n_skip_seq_edges += len(in_range)
            else:
                n_in_range += len(in_range)
            for _, _, sub in in_range:
                subtypes[sub] += 1
            task = str(obj.get("task_id") or obj.get("uid") or f"row_{line_no}")
            per_sample.append((n_compact - 1, task, seq, len(parsed), len(in_range), too_long))

    scored = n_in_range  # retrieve_degradation loop: skip seq>3000, then in-range edges
    print(f"file              : {jsonl}")
    print(f"jsonl rows        : {n_rows}")
    print(f"compact samples   : {n_compact}")
    print(f"samples w/ edges  : {n_with_edges}")
    print(f"parsed edges      : {n_edges}   (src!=dst, indices >= 0)")
    print(f"in-range edges    : {n_in_range + n_skip_seq_edges}")
    print(f"seq > {SEQ_LIMIT:<5}     : {n_skip_seq} samples, {n_skip_seq_edges} edges skipped")
    print(f"would be scored   : {scored}   (in-range and seq <= {SEQ_LIMIT})")
    print("subtypes (in-range, including too-long samples):")
    for name, c in subtypes.most_common():
        print(f"  {c:6d}  {name}")
    print()
    print(f"{'idx':>5}  {'seq':>6}  {'edges':>6}  {'ok':>6}  skip  task")
    show = per_sample[: max(0, args.top)]
    for idx, task, seq, n_e, n_ok, too_long in show:
        flag = "YES" if too_long else "   "
        print(f"{idx:5d}  {seq:6d}  {n_e:6d}  {n_ok:6d}  {flag:4s}  {task[:60]}")
    if len(per_sample) > len(show):
        print(f"  … {len(per_sample) - len(show)} more samples")
    print()
    print("Degrade retrieve caches per-edge sketched L_sal grads under")
    print(".cache/degrade_sal_edge_grads/ (default sketch dim 8192).")
    if scored:
        lo, hi = scored * 5 / 60, scored * 15 / 60
        print(f"If ~5-15s/edge on 8B last-layer LoRA, {scored} edges ~ {lo:.0f}-{hi:.0f} min (first fill).")


if __name__ == "__main__":
    main()
