#!/usr/bin/env python3
"""Rewrite the 100 queries so whole-function embedding is less of a paraphrase.

The hole stays one call whose arguments name the semantic relation. The rest
of the parent checklist is not copied. A short off-topic prelude dominates
the raw text the embedding model sees; the missing span is still the relation.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build_go_fim_mutations import content_tokens, full_code, make_prompt  # noqa: E402
from apply_hand_10k import disjoin  # noqa: E402
from expand_to_100 import _pool, _score  # noqa: E402
from build_go_fim_mutations import _parser  # noqa: E402

OUT = HERE / "go_fim_mutations.jsonl"

_WORD = re.compile(r"[A-Za-z]+")
_SKIP = {
    "the", "a", "an", "of", "to", "and", "then", "from", "before", "after",
    "its", "their", "this", "that", "into", "for", "with", "when", "once",
    "already", "than", "over", "under", "per", "via", "onto", "upon",
}

# Unrelated surface topics. They are not the hole; they shift the raw embedding.
_TOPICS = [
    ("kiln", "glaze", "shard"),
    ("sail", "boom", "tack"),
    ("loaf", "yeast", "crust"),
    ("orbit", "apsis", "epoch"),
    ("yarn", "skein", "ply"),
    ("cedar", "sap", "ring"),
    ("canoe", "paddle", "wake"),
    ("mosaic", "grout", "tile"),
    ("violin", "peg", "bow"),
    ("harbor", "buoy", "pier"),
    ("quartz", "facet", "vein"),
    ("linen", "warp", "weft"),
    ("cider", "press", "pulp"),
    ("falcon", "jess", "hood"),
    ("copper", "patina", "seam"),
    ("meadow", "clover", "pollen"),
    ("anchor", "chain", "fluke"),
    ("ivory", "inlay", "grain"),
    ("pepper", "pod", "seed"),
    ("marble", "vein", "chisel"),
    ("lantern", "wick", "soot"),
    ("willow", "bark", "twig"),
    ("coral", "polyp", "reef"),
    ("bronze", "mold", "pour"),
    ("thistle", "down", "spine"),
    ("kettle", "steam", "spout"),
    ("linen", "hem", "stitch"),
    ("granite", "grain", "fault"),
    ("orchid", "petal", "lip"),
    ("canyon", "ledge", "scree"),
    ("amber", "resin", "speck"),
    ("bamboo", "node", "culm"),
    ("velvet", "pile", "nap"),
    ("pewter", "rim", "dent"),
    ("moss", "spore", "mat"),
    ("indigo", "vat", "dip"),
    ("rye", "awn", "sheaf"),
    ("obsidian", "flake", "core"),
    ("tulip", "bulb", "stem"),
    ("fir", "cone", "resin"),
    ("clay", "slip", "kiln"),
    ("reed", "pith", "stalk"),
    ("opal", "fire", "base"),
    ("wool", "fleece", "card"),
    ("birch", "bark", "sap"),
    ("plum", "pit", "skin"),
    ("dune", "crest", "slip"),
    ("oak", "gall", "acorn"),
    ("silk", "cocoon", "strand"),
    ("basalt", "flow", "joint"),
    ("honey", "comb", "wax"),
    ("spruce", "pitch", "knot"),
    ("pearl", "nacre", "grit"),
    ("flax", "rett", "fiber"),
    ("slate", "cleat", "bed"),
    ("ginger", "knob", "fiber"),
    ("alder", "catkin", "leaf"),
    ("cobalt", "bloom", "ore"),
    ("iris", "rhizome", "fall"),
    ("chalk", "dust", "seam"),
    ("hemp", "hurd", "tow"),
    ("jasper", "band", "mottle"),
    ("poplar", "fluff", "bud"),
    ("saffron", "thread", "crocus"),
    ("nickel", "matte", "slag"),
    ("lavender", "bud", "spike"),
    ("gneiss", "band", "foliation"),
    ("cotton", "boll", "lint"),
    ("topaz", "crystal", "axis"),
    ("maple", "sap", "tap"),
    ("cumin", "seed", "ridge"),
    ("schist", "sheet", "mica"),
    ("jute", "rett", "strand"),
    ("garnet", "crystal", "face"),
    ("cypress", "knee", "bark"),
    ("anise", "seed", "pod"),
    ("pumice", "pore", "float"),
    ("ramie", "fiber", "degum"),
    ("spinel", "octahedron", "hue"),
    ("hickory", "nut", "husk"),
    ("clove", "bud", "nail"),
    ("tuff", "ash", "weld"),
    ("sisal", "leaf", "fiber"),
    ("zircon", "grain", "halo"),
    ("sequoia", "bark", "bur"),
    ("nutmeg", "seed", "mace"),
    ("marl", "lime", "clay"),
    ("coir", "husk", "fiber"),
    ("tourmaline", "prism", "striation"),
    ("redwood", "burl", "ring"),
    ("fennel", "frond", "bulb"),
    ("loess", "silt", "cliff"),
    ("kapok", "pod", "floss"),
    ("peridot", "grain", "olive"),
    ("larch", "needle", "cone"),
    ("caraway", "seed", "ridge"),
    ("till", "clay", "boulder"),
    ("abaca", "leaf", "pulp"),
    ("alexandrite", "twin", "hue"),
]


def _stem(phrase: str) -> str:
    words = [w.lower() for w in _WORD.findall(phrase or "")]
    words = [w for w in words if w not in _SKIP and len(w) > 2]
    if not words:
        words = ["item"]
    words = words[-2:]
    head, *rest = words
    return head + "".join(part.capitalize() for part in rest)


def _prelude(i: int) -> str:
    a, b, c = _TOPICS[i % len(_TOPICS)]
    scale = 3 + (i % 5)
    bias = 1 + (i % 4)
    return (
        f"\t{a} := box.{b}*{scale} + box.{c}\n"
        f"\tif {a} > box.limit+{bias} {{\n"
        f"\t\tbox.{b} = {a} / {scale}\n"
        f"\t}} else {{\n"
        f"\t\tbox.{c} = {a} + {bias}\n"
        f"\t}}\n"
        f"\tbox.tally += {a}\n"
    )


def _rewrite(row: dict, i: int) -> tuple[str, str, str]:
    sem = row.get("shared_semantic") or {}
    rels = sem.get("relations") or []
    if rels:
        src = _stem(str(rels[0].get("source") or "input"))
        dst = _stem(str(rels[0].get("target") or "result"))
    else:
        src, dst = _stem(str(sem.get("role") or "input")), "result"
    extra = ""
    if len(rels) > 1:
        extra = ", box." + _stem(str(rels[1].get("target") or "later"))
    prefix = f"func apply{i+1}(box *carrier) status {{\n" + _prelude(i)
    middle = f"\treturn box.bind{dst[:1].upper() + dst[1:]}(box.{src}{extra})\n"
    suffix = "}\n"
    return prefix, middle, suffix


def main() -> None:
    rows = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    parser = _parser()
    used = {int(row["ground_truth_line"]) for row in rows}
    print("building pool", flush=True)
    pool_tokens, pool_ast = _pool(parser, used)
    ast_top5 = 0
    shared_n = 0
    parse_n = 0
    for i, row in enumerate(rows):
        parent = full_code(
            {
                "prefix": row.get("parent_prefix"),
                "middle": row.get("parent_middle"),
                "suffix": row.get("parent_suffix"),
            }
        )
        banned = set(content_tokens(parent))
        prefix, middle, suffix = _rewrite(row, i)
        row["prefix"] = disjoin(prefix, banned)
        row["middle"] = disjoin(middle, banned)
        row["suffix"] = disjoin(suffix, banned)
        row["response"] = row["middle"]
        row["prompt"] = make_prompt(row["prefix"], row["suffix"])
        row["rewrite_note"] = (
            "hole is one bind of the relation; prelude is an unrelated tally "
            "so the whole function is not a paraphrase of the parent"
        )
        row["origin"] = "embed-gap"
        gap = _score(full_code(row), parent, parser, pool_tokens, pool_ast)
        row["gap"] = gap
        if gap["parse_error"]:
            parse_n += 1
        if gap["shared_content_tokens"]:
            shared_n += 1
        if gap["ast_rank_in_251"] <= 5:
            ast_top5 += 1
        print(
            f"{row['mutation_id']} parse={gap['parse_error']} "
            f"shared={len(gap['shared_content_tokens'])} ast={gap['ast_rank_in_251']}",
            flush=True,
        )
    OUT.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    print(f"n={len(rows)} parse_err={parse_n} shared_rows={shared_n} ast_top5={ast_top5}")


if __name__ == "__main__":
    main()
