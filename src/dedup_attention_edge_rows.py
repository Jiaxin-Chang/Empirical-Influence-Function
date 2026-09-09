"""Keep unique compact-train rows that have non-empty attention_edges.

Continue-train copies (~20×) share ``input_ids`` even when ``uid`` is
``…::dup…``. Default key is ``input_ids`` (not ``duplicate_of`` / uid prefix,
which can collapse different samples into one group).

Example::

    python -m src.dedup_attention_edge_rows \\
      --input /mnt/md124/jiaxin/Empirical-Influence-Function/cpp_train_ids_20000.jsonl \\
      --out /mnt/md124/jiaxin/Empirical-Influence-Function/cpp_train_ids_edges_unique.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _has_edges(obj: dict[str, Any]) -> bool:
    edges = obj.get("attention_edges")
    if edges is None:
        edges = obj.get("edges")
    return bool(edges)


def _n_edges(obj: dict[str, Any]) -> int:
    edges = obj.get("attention_edges")
    if edges is None:
        edges = obj.get("edges")
    return len(edges) if isinstance(edges, list) else (1 if edges else 0)


def _sha(obj: Any) -> str:
    blob = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def _keys(obj: dict[str, Any]) -> dict[str, str]:
    ids = obj.get("input_ids")
    uid = str(obj.get("uid") or obj.get("task_id") or "").strip()
    dup = str(obj.get("duplicate_of") or "").strip()
    corpus = obj.get("source_corpus_line")
    return {
        "input_ids": f"ids:{_sha(ids)}" if isinstance(ids, list) and ids else f"row:{_sha(obj)}",
        "duplicate_of": f"dup:{dup}" if dup else "",
        "uid_prefix": f"uid:{uid.split('::dup', 1)[0]}" if uid else "",
        "corpus_line": f"line:{int(corpus)}" if corpus is not None and str(corpus).strip() != "" else "",
    }


def _mult_line(counts: Counter[str]) -> str:
    mult = Counter(v for v in counts.values() if v)
    parts = [f"{k}×→{mult[k]}" for k in sorted(mult)]
    return ", ".join(parts) if parts else "(none)"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write unique nonempty-attention_edges rows (drop copies)."
    )
    parser.add_argument(
        "--input",
        default="/mnt/md124/jiaxin/Empirical-Influence-Function/cpp_train_ids_20000.jsonl",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Default: <input stem>_edges_unique.jsonl beside the input file",
    )
    parser.add_argument(
        "--key",
        choices=("input_ids", "duplicate_of", "uid_prefix", "corpus_line"),
        default="input_ids",
        help="Dedup field (default: input_ids)",
    )
    args = parser.parse_args()

    src = Path(args.input).expanduser()
    if not src.is_file():
        raise FileNotFoundError(src)
    out = Path(args.out).expanduser() if args.out else src.with_name(f"{src.stem}_edges_unique.jsonl")

    best: dict[str, tuple[int, int, str]] = {}
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    n_total = 0
    n_edges = 0
    n_missing_key = 0

    with src.open(encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            n_total += 1
            obj = json.loads(raw)
            if not _has_edges(obj):
                continue
            n_edges += 1
            ks = _keys(obj)
            for kind, key in ks.items():
                if key:
                    by_kind[kind][key] += 1
            key = ks.get(args.key) or ""
            if not key:
                n_missing_key += 1
                key = ks["input_ids"]
            score = _n_edges(obj)
            prev = best.get(key)
            if prev is None or score > prev[0]:
                best[key] = (score, line_no, raw)

    rows = [item[2] for _, item in sorted(best.items(), key=lambda kv: kv[1][1])]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for raw in rows:
            fh.write(raw)
            fh.write("\n")

    write_counts = by_kind.get(args.key) or Counter(best)
    if args.key != "input_ids" and n_missing_key:
        write_counts = Counter()
        # written rows keyed as used
        write_counts.update({k: 1 for k in best})

    print(f"input={src}")
    print(f"out={out}")
    print(f"total_rows={n_total}")
    print(f"nonempty_attention_edges={n_edges}")
    print(f"dedup_key={args.key}")
    print(f"unique_written={len(rows)}")
    print(f"dropped_copies={n_edges - len(rows)}")
    if n_missing_key:
        print(f"missing_{args.key}_fell_back_to_input_ids={n_missing_key}")
    print("alt unique counts among nonempty-edge rows:")
    for kind in ("input_ids", "duplicate_of", "uid_prefix", "corpus_line"):
        c = by_kind.get(kind) or Counter()
        n_unique = len(c)
        covered = sum(c.values())
        print(f"  {kind}: unique={n_unique} covered={covered}/{n_edges}  multiplicity: {_mult_line(c)}")
    print(f"write multiplicity ({args.key}): {_mult_line(by_kind.get(args.key) or Counter())}")


if __name__ == "__main__":
    main()
