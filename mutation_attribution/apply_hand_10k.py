#!/usr/bin/env python3
"""Replace mutation queries with hand-written rewrites and rescore them.

Each spec keeps the hole's mechanism and uses a different statement skeleton
plus a disjoint identifier vocabulary. m01 and m05 stay as already written.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_go_fim_mutations import _parser, content_tokens, full_code, make_prompt  # noqa: E402

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _alias(word: str) -> str:
    if word[:1].isupper():
        return "Alt" + word
    return "alt" + word[:1].upper() + word[1:]


def disjoin(text: str, banned: set[str]) -> str:
    """Rename identifiers that also appear in the parent, keeping the stem readable."""
    for _ in range(6):
        shared = {tok for tok in content_tokens(text) if tok in banned}
        if not shared:
            return text

        def repl(match: re.Match) -> str:
            word = match.group(0)
            if word.casefold() not in shared:
                return word
            return _alias(word)

        text = _IDENT.sub(repl, text)
    return text
from expand_to_100 import _pool, _score  # noqa: E402
from hand_specs import SPECS  # noqa: E402

OUT = HERE / "go_fim_mutations.jsonl"


def main() -> None:
    rows = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_id = {row["mutation_id"]: row for row in rows}
    missing = sorted(set(SPECS) - set(by_id))
    if missing:
        raise SystemExit(f"unknown ids {missing}")
    for mid, spec in SPECS.items():
        row = by_id[mid]
        parent_code = full_code(
            {
                "prefix": row.get("parent_prefix"),
                "middle": row.get("parent_middle"),
                "suffix": row.get("parent_suffix"),
            }
        )
        banned = set(content_tokens(parent_code))
        row["prefix"] = disjoin(spec["prefix"], banned)
        row["middle"] = disjoin(spec["middle"], banned)
        row["suffix"] = disjoin(spec["suffix"], banned)
        row["response"] = row["middle"]
        row["prompt"] = make_prompt(row["prefix"], row["suffix"])
        row["rewrite_note"] = spec["note"]
        row["shared_semantic"] = spec["semantic"]
        row["origin"] = "hand"
        row.pop("prior_mutation_id", None)
    for row in rows:
        if row["mutation_id"] in ("m01", "m05"):
            row["origin"] = "hand"

    parser = _parser()
    used = {int(row["ground_truth_line"]) for row in rows}
    pool_tokens, pool_ast = _pool(parser, used)
    print(f"{'id':<6}{'parse':<8}{'shared':<8}{'astR':<8}{'tokR':<8}func")
    for row in rows:
        parent = full_code(
            {
                "prefix": row.get("parent_prefix"),
                "middle": row.get("parent_middle"),
                "suffix": row.get("parent_suffix"),
            }
        )
        mutant = row["prefix"] + row["middle"] + row["suffix"]
        row["gap"] = _score(mutant, parent, parser, pool_tokens, pool_ast)
        gap = row["gap"]
        flag = "ERR" if gap["parse_error"] else "ok"
        print(
            f"{row['mutation_id']:<6}{flag:<8}{len(gap['shared_content_tokens']):<8}"
            f"{gap['ast_rank_in_251']:<8}{gap['token_rank_in_251']:<8}{row.get('ground_truth_func')}"
        )
        if gap["shared_content_tokens"]:
            print("       ", ", ".join(gap["shared_content_tokens"][:16]))

    OUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    n_err = sum(1 for row in rows if row["gap"]["parse_error"])
    n_share = sum(1 for row in rows if row["gap"]["shared_content_tokens"])
    n_ast = sum(1 for row in rows if row["gap"]["ast_rank_in_251"] <= 5)
    n_hand = sum(1 for row in rows if row.get("origin") == "hand")
    print(f"wrote {OUT} hand={n_hand} parse_err={n_err} shared={n_share} ast_top5={n_ast}")


if __name__ == "__main__":
    main()
