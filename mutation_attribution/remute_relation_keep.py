#!/usr/bin/env python3
"""Rewrite the same 100 parents.

Relation lines keep their shape. Each identifier, and each quoted string that
contains a word, is replaced with a whole word drawn from a pool of 200
generated words. The same literal always maps to the same word inside one query.
Unrelated statements are removed. In their place, before the closing brace,
each query gets 3-5 semantic blocks taken from other 10k training functions.
Borrowed blocks keep their original strings.
"""

from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from build_go_fim_mutations import (  # noqa: E402
    STOP,
    TRAIN,
    _IDENT,
    _parser,
    content_tokens,
    full_code,
    make_prompt,
)
from expand_to_100 import _pool, _score  # noqa: E402

OUT = HERE / "go_fim_mutations.jsonl"
SEM = ROOT / "csn_go_train_fim_10k.semantic.vllm32b.jsonl"

STMT = {
    "short_var_declaration",
    "var_declaration",
    "const_declaration",
    "type_declaration",
    "assignment_statement",
    "inc_statement",
    "dec_statement",
    "expression_statement",
    "send_statement",
    "return_statement",
    "if_statement",
    "for_statement",
    "expression_switch_statement",
    "type_switch_statement",
    "select_statement",
    "go_statement",
    "defer_statement",
    "labeled_statement",
    "break_statement",
    "continue_statement",
    "fallthrough_statement",
    "goto_statement",
    "block",
}
BLOCKISH = {
    "block",
    "expression_case",
    "default_case",
    "type_case",
    "communication_case",
}

_TOPICS = [
    ("kiln", "glaze", "shard"),
    ("sail", "boom", "tack"),
    ("loaf", "yeast", "crust"),
    ("orbit", "apsis", "epoch"),
    ("yarn", "skein", "ply"),
    ("cedar", "sapwood", "ring"),
    ("canoe", "paddle", "wake"),
    ("mosaic", "grout", "tessera"),
    ("violin", "pegbox", "bowhair"),
    ("harbor", "buoy", "pier"),
    ("quartz", "facet", "vein"),
    ("linen", "warp", "weft"),
    ("cider", "press", "pulp"),
    ("falcon", "jess", "hood"),
    ("copper", "patina", "seam"),
    ("meadow", "clover", "pollen"),
    ("anchor", "fluke", "shank"),
    ("ivory", "inlay", "grain"),
    ("pepper", "pod", "kernel"),
    ("marble", "chisel", "veining"),
    ("lantern", "wick", "soot"),
    ("willow", "twig", "catkin"),
    ("coral", "polyp", "reef"),
    ("bronze", "mold", "pour"),
    ("thistle", "spine", "down"),
    ("kettle", "spout", "steam"),
    ("granite", "fault", "grain"),
    ("orchid", "petal", "sepal"),
    ("canyon", "ledge", "scree"),
    ("amber", "resin", "speck"),
    ("bamboo", "culm", "node"),
    ("velvet", "pile", "nap"),
    ("pewter", "dent", "rim"),
    ("moss", "spore", "mat"),
    ("indigo", "vat", "dip"),
    ("rye", "awn", "sheaf"),
    ("obsidian", "flake", "core"),
    ("tulip", "bulb", "stem"),
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
    ("jasper", "mottle", "band"),
    ("poplar", "fluff", "bud"),
    ("saffron", "crocus", "thread"),
    ("nickel", "matte", "slag"),
    ("lavender", "spike", "bud"),
    ("gneiss", "foliation", "band"),
    ("cotton", "boll", "lint"),
    ("topaz", "axis", "crystal"),
    ("maple", "tap", "sapwood"),
    ("cumin", "ridge", "seed"),
    ("schist", "mica", "sheet"),
    ("jute", "strand", "rett"),
    ("garnet", "face", "crystal"),
    ("cypress", "knee", "bark"),
    ("anise", "pod", "seed"),
    ("pumice", "pore", "float"),
    ("ramie", "degum", "fiber"),
    ("spinel", "hue", "octa"),
    ("hickory", "husk", "nut"),
    ("clove", "nail", "bud"),
    ("tuff", "weld", "ash"),
    ("sisal", "blade", "fiber"),
    ("zircon", "halo", "grain"),
    ("sequoia", "bur", "bark"),
    ("nutmeg", "mace", "seed"),
    ("marl", "lime", "claybed"),
    ("coir", "husk", "fiber"),
    ("tourmaline", "prism", "stripe"),
    ("balsa", "grain", "float"),
    ("fennel", "frond", "bulb"),
    ("malachite", "band", "polish"),
    ("linden", "bast", "bloom"),
    ("paprika", "pod", "seed"),
    ("dolomite", "rhombo", "grain"),
    ("kenaf", "bast", "core"),
    ("peridot", "crystal", "facet"),
    ("redwood", "bur", "bark"),
    ("cardamom", "pod", "seed"),
    ("gneissrock", "band", "mica"),
    ("raffia", "strip", "palm"),
    ("iolite", "pleochro", "gem"),
    ("aspen", "catkin", "leaf"),
    ("caraway", "ridge", "seed"),
    ("travertine", "band", "pore"),
    ("abaca", "fiber", "leaf"),
    ("kunzite", "crystal", "hue"),
    ("larch", "cone", "needle"),
    ("dill", "frond", "seed"),
    ("scoria", "vesicle", "flow"),
    ("piassava", "fiber", "palm"),
    ("tanzanite", "pleoch", "gem"),
    ("beech", "mast", "bark"),
    ("fennelseed", "ridge", "pod"),
]


def _pieces(text: str) -> list[tuple[bool, str]]:
    """Split Go source into code vs string/comment pieces."""
    pieces: list[tuple[bool, str]] = []
    buf: list[str] = []
    i = 0
    n = len(text)

    def flush() -> None:
        if buf:
            pieces.append((True, "".join(buf)))
            buf.clear()

    while i < n:
        c = text[i]
        if c == '"':
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    j += 1
                    break
                j += 1
            flush()
            pieces.append((False, text[i:j]))
            i = j
            continue
        if c == "`":
            j = text.find("`", i + 1)
            j = n if j < 0 else j + 1
            flush()
            pieces.append((False, text[i:j]))
            i = j
            continue
        if c == "'":
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "'":
                    j += 1
                    break
                j += 1
            flush()
            pieces.append((False, text[i:j]))
            i = j
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            j = n if j < 0 else j
            flush()
            pieces.append((False, text[i:j]))
            i = j
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            flush()
            pieces.append((False, text[i:j]))
            i = j
            continue
        buf.append(c)
        i += 1
    flush()
    return pieces


_WORD_IN_STRING = re.compile(r"[A-Za-z]{2,}")


def _closed_string(piece: str) -> bool:
    if len(piece) < 2:
        return False
    quote = piece[0]
    return quote in ('"', "`") and piece.endswith(quote)


def _rename_text(
    text: str,
    mapping: dict[str, str],
    strings: dict[str, str] | None = None,
    drop_comments: bool = False,
) -> str:
    strings = strings or {}
    out: list[str] = []
    for is_code, piece in _pieces(text):
        if not is_code:
            if drop_comments and (piece.startswith("//") or piece.startswith("/*")):
                continue
            if piece in strings:
                quote = piece[0]
                piece = quote + strings[piece] + quote
            out.append(piece)
            continue

        def repl(match):
            word = match.group(0)
            return mapping.get(word, word)

        out.append(_IDENT.sub(repl, piece))
    return "".join(out)


def _mask_code(text: str) -> str:
    out: list[str] = []
    for is_code, piece in _pieces(text):
        if is_code:
            out.append(_IDENT.sub("ID", piece))
        elif _closed_string(piece):
            out.append(piece[0] + "STR" + piece[0])
        else:
            out.append(piece)
    return "".join(out)


def _fresh(banned: set[str], idx: int, slot: int) -> tuple[str, str, str]:
    topic = _TOPICS[(idx * 3 + slot) % len(_TOPICS)]
    names: list[str] = []
    for word in topic:
        name = word
        n = 2
        while name.casefold() in banned or name in names:
            name = f"{word}{n}"
            n += 1
        names.append(name)
    return names[0], names[1], names[2]


def _snippet(names: tuple[str, str, str], indent: str, kind: int) -> str:
    a, b, c = names
    i2 = indent + "\t"
    i3 = indent + "\t\t"
    i4 = indent + "\t\t\t"
    if kind % 4 == 0:
        return (
            f"{indent}{a} := 0\n"
            f"{indent}for {b} := 1; {b} < 6; {b}++ {{\n"
            f"{i2}switch {b} % 3 {{\n"
            f"{i2}case 0:\n"
            f"{i3}{a} += {b} + 1\n"
            f"{i2}case 1:\n"
            f"{i3}{a} -= 2\n"
            f"{i2}default:\n"
            f"{i3}if {a} < 11 {{\n"
            f"{i4}{a} = {a} * 2\n"
            f"{i3}}}\n"
            f"{i2}}}\n"
            f"{indent}}}\n"
            f"{indent}_ = {a}\n"
        )
    if kind % 4 == 1:
        return (
            f"{indent}{a} := 3\n"
            f"{indent}select {{\n"
            f"{indent}default:\n"
            f"{i2}if {a}%2 == 0 {{\n"
            f"{i3}{a} += 4\n"
            f"{i2}}} else {{\n"
            f"{i3}{a} -= 1\n"
            f"{i2}}}\n"
            f"{indent}}}\n"
            f"{indent}defer func({b} int) {{\n"
            f"{i2}{b}++\n"
            f"{indent}}}({a})\n"
            f"{indent}_ = {a}\n"
        )
    if kind % 4 == 2:
        return (
            f"{indent}{a} := []int{{1, 2, 4, 8}}\n"
            f"{indent}{b} := 0\n"
            f"{indent}for _, {c} := range {a} {{\n"
            f"{i2}if {c} > 3 {{\n"
            f"{i3}{b} += {c} - 1\n"
            f"{i2}}} else {{\n"
            f"{i3}{b} += 1\n"
            f"{i2}}}\n"
            f"{indent}}}\n"
            f"{indent}_ = {b}\n"
        )
    return (
        f"{indent}{a} := 2\n"
        f"{indent}switch {{\n"
        f"{indent}case {a} < 0:\n"
        f"{i2}{a} = 1\n"
        f"{indent}case {a} > 9:\n"
        f"{i2}for {b} := 0; {b} < 3; {b}++ {{\n"
        f"{i3}{a} -= {b}\n"
        f"{i2}}}\n"
        f"{indent}default:\n"
        f"{i2}{a} = {a}*3 + 1\n"
        f"{indent}}}\n"
        f"{indent}_ = {a}\n"
    )


def _indent_at(raw: bytes, byte_pos: int) -> str:
    line_start = raw.rfind(b"\n", 0, byte_pos) + 1
    i = line_start
    while i < len(raw) and raw[i] in (9, 32):
        i += 1
    return raw[line_start:i].decode("utf-8")


def _direct_bodies(node):
    found = []

    def walk(n) -> None:
        for ch in n.children:
            if ch.type in BLOCKISH:
                found.append(ch)
                continue
            walk(ch)

    walk(node)
    return found


def _lhs_names(node, raw: bytes) -> set[str]:
    if node.type == "expression_list":
        names: set[str] = set()
        for ch in node.children:
            if ch.type != ",":
                names |= _lhs_names(ch, raw)
        return names
    if node.type == "identifier":
        return {raw[node.start_byte : node.end_byte].decode("utf-8")}
    if node.type == "selector_expression":
        for ch in reversed(node.children):
            if ch.type == "field_identifier":
                return {raw[ch.start_byte : ch.end_byte].decode("utf-8")}
        return set()
    if node.type in ("index_expression", "slice_expression", "parenthesized_expression"):
        for ch in node.children:
            if ch.type not in ("[", "]", "(", ")", ":"):
                return _lhs_names(ch, raw)
        return set()
    return set()


def _defined_names(node, raw: bytes) -> set[str]:
    names: set[str] = set()
    if node.type in ("assignment_statement", "short_var_declaration"):
        for ch in node.children:
            if ch.type in ("=", ":="):
                break
            if ch.type == ",":
                continue
            names |= _lhs_names(ch, raw)
        return names
    if node.type in ("inc_statement", "dec_statement"):
        for ch in node.children:
            if ch.type not in ("++", "--"):
                names |= _lhs_names(ch, raw)
        return names
    if node.type == "var_declaration":
        for ch in node.children:
            if ch.type != "var_spec":
                continue
            for part in ch.children:
                if part.type in ("=", "type_identifier", "qualified_type", "pointer_type", "slice_type", "map_type", "array_type", "struct_type", "interface_type", "function_type", "channel_type"):
                    break
                if part.type == "identifier":
                    names.add(raw[part.start_byte : part.end_byte].decode("utf-8"))
        return names
    if node.type == "for_statement":
        for ch in node.children:
            if ch.type == "range_clause":
                for part in ch.children:
                    if part.type in ("=", ":="):
                        break
                    if part.type == "identifier":
                        names.add(raw[part.start_byte : part.end_byte].decode("utf-8"))
            if ch.type == "for_clause":
                for part in ch.children:
                    if part.type == "short_var_declaration":
                        names |= _defined_names(part, raw)
        return names
    return names


def _overlaps(node, a: int, b: int) -> bool:
    return node.start_byte < b and node.end_byte > a


def _plan(tree, raw: bytes, mid0: int, mid1: int):
    """Return (replace_spans, keep_byte_count) in original coordinates."""
    def find_body(node):
        if node.type in ("function_declaration", "method_declaration") and _overlaps(node, mid0, mid1):
            for ch in node.children:
                if ch.type == "block":
                    return ch
        for ch in node.children:
            found = find_body(ch)
            if found is not None:
                return found
        return None

    body = find_body(tree.root_node)
    if body is None:
        return [], 0, None

    hole_text = raw[mid0:mid1].decode("utf-8")
    defs_in_hole: set[str] = set()

    def note_defs(node) -> None:
        for name in _defined_names(node, raw):
            folded = name.casefold()
            if folded in content_tokens(hole_text) or name in hole_text:
                defs_in_hole.add(folded)

    def scan_block(block) -> None:
        for ch in block.children:
            if ch.type in STMT:
                scan_statement(ch)

    def scan_statement(node) -> None:
        bodies = _direct_bodies(node)
        if _overlaps(node, mid0, mid1) and any(_overlaps(b, mid0, mid1) for b in bodies):
            for b in bodies:
                scan_block(b)
            return
        if _overlaps(node, mid0, mid1):
            note_defs(node)

    scan_block(body)
    # `err` is assigned all over a Go function. It is not a forward seed,
    # or every later error check would be treated as the relation.
    defs_forward = defs_in_hole - {"err"}
    uses_in_hole = set(content_tokens(hole_text)) - defs_in_hole

    replaces: list[tuple[int, int]] = []
    keep_bytes = 0

    def touches(node) -> bool:
        defined = {n.casefold() for n in _defined_names(node, raw)}
        if defined & uses_in_hole:
            return True
        used = set(content_tokens(raw[node.start_byte : node.end_byte].decode("utf-8")))
        if used & defs_forward:
            return True
        return False

    def handle_block(block) -> None:
        kids = [ch for ch in block.children if ch.type in STMT]
        for i, ch in enumerate(kids):
            prev_is_hole = i > 0 and _overlaps(kids[i - 1], mid0, mid1)
            handle_statement(
                ch,
                i == len(kids) - 1 and block.type == "block" and block == body,
                prev_is_hole,
            )

    def handle_statement(node, is_final_body: bool, follow_hole: bool) -> None:
        nonlocal keep_bytes
        bodies = _direct_bodies(node)
        # The hole includes this control's opening brace, so the branch
        # itself is the relation. Keep its direct statements; still rewrite
        # an else the hole does not open.
        if _overlaps(node, mid0, mid1) and bodies:
            owned = [b for b in bodies if mid0 <= b.start_byte < mid1]
            if owned:
                for b in owned:
                    for ch in b.children:
                        if ch.type in STMT:
                            keep_bytes += ch.end_byte - ch.start_byte
                for b in bodies:
                    if b not in owned:
                        handle_block(b)
                keep_bytes += _header_bytes(node, bodies)
                return
        if _overlaps(node, mid0, mid1) and any(_overlaps(b, mid0, mid1) for b in bodies):
            for b in bodies:
                handle_block(b)
            keep_bytes += _header_bytes(node, bodies)
            return
        if _overlaps(node, mid0, mid1):
            keep_bytes += node.end_byte - node.start_byte
            return
        text = raw[node.start_byte : node.end_byte].decode("utf-8").strip()
        if is_final_body and text.startswith("return"):
            keep_bytes += node.end_byte - node.start_byte
            return
        if follow_hole and text.startswith("if") and "err" in text and "nil" in text and len(text) < 120:
            keep_bytes += node.end_byte - node.start_byte
            return
        related = touches(node)
        if related and bodies and (node.end_byte - node.start_byte) > 180:
            for b in bodies:
                handle_block(b)
            keep_bytes += _header_bytes(node, bodies)
            return
        if related:
            keep_bytes += node.end_byte - node.start_byte
            return
        replaces.append((node.start_byte, node.end_byte))

    handle_block(body)
    return replaces, keep_bytes, body


def _header_bytes(node, bodies) -> int:
    covered = 0
    for b in bodies:
        covered += b.end_byte - b.start_byte
    return max(0, (node.end_byte - node.start_byte) - covered)


def _word_pool(n: int = 200) -> list[str]:
    """200 generated words. Assignment per query is shuffled, not a fixed glossary."""
    rng = random.Random(20260327)
    consonants = "bcdfghjklmnpqrstvwxyz"
    vowels = "aeiou"
    words: list[str] = []
    seen: set[str] = set()
    while len(words) < n:
        parts = []
        for _ in range(rng.randint(2, 3)):
            parts.append(rng.choice(consonants) + rng.choice(vowels) + rng.choice(consonants))
        word = "".join(parts)
        if word in seen or word in STOP:
            continue
        seen.add(word)
        words.append(word)
    return words


_WORDS = _word_pool()


def _build_mapping(segments: list[str], banned: set[str], rng: random.Random) -> dict[str, str]:
    bag = _WORDS[:]
    rng.shuffle(bag)
    cursor = 0
    used: set[str] = set()
    mapping: dict[str, str] = {}

    def take(upper: bool) -> str:
        nonlocal cursor
        while cursor < len(bag):
            cand = bag[cursor]
            cursor += 1
            if cand in banned or cand in used or cand in STOP:
                continue
            used.add(cand)
            if upper:
                return cand[:1].upper() + cand[1:]
            return cand
        extra = f"z{rng.randrange(1000, 9999)}{len(used)}"
        used.add(extra)
        return extra[:1].upper() + extra[1:] if upper else extra

    for text in segments:
        for is_code, piece in _pieces(text):
            if not is_code:
                continue
            for match in _IDENT.finditer(piece):
                word = match.group(0)
                if word in mapping or word.casefold() in STOP or word == "_":
                    mapping.setdefault(word, word)
                    continue
                mapping[word] = take(word[:1].isupper())
    return mapping


def _build_string_mapping(
    segments: list[str],
    banned: set[str],
    taken: set[str],
    rng: random.Random,
) -> dict[str, str]:
    """Map each distinct word-bearing literal to one pool word."""
    bag = _WORDS[:]
    rng.shuffle(bag)
    cursor = 0
    used = set(taken)
    mapping: dict[str, str] = {}

    def take() -> str:
        nonlocal cursor
        while cursor < len(bag):
            cand = bag[cursor]
            cursor += 1
            if cand in banned or cand in used or cand in STOP:
                continue
            used.add(cand)
            return cand
        extra = f"z{rng.randrange(1000, 9999)}{len(used)}"
        used.add(extra)
        return extra

    for text in segments:
        for is_code, piece in _pieces(text):
            if is_code or piece in mapping or not _closed_string(piece):
                continue
            if not _WORD_IN_STRING.search(piece[1:-1]):
                continue
            mapping[piece] = take()
    return mapping


def _apply(
    raw: bytes,
    start: int,
    end: int,
    edits: list[tuple[int, int, str]],
    mapping: dict[str, str],
    strings: dict[str, str],
) -> str:
    """Apply edits whose spans lie inside [start, end), then rename the gaps."""
    local = [(s - start, e - start, text) for s, e, text in edits if start <= s and e <= end]
    local.sort(key=lambda item: (item[0], item[1]))
    chunk = raw[start:end]
    out: list[str] = []
    cursor = 0
    for s, e, text in local:
        if s < cursor:
            continue
        out.append(_rename_text(chunk[cursor:s].decode("utf-8"), mapping, strings, drop_comments=True))
        out.append(text)
        cursor = e
    out.append(_rename_text(chunk[cursor:].decode("utf-8"), mapping, strings, drop_comments=True))
    return "".join(out)


def _load_semantic() -> dict[int, dict]:
    found: dict[int, dict] = {}
    with SEM.open(encoding="utf-8") as handle:
        for i, line in enumerate(handle):
            rec = json.loads(line)
            found[int(rec.get("source_line", i))] = rec
    return found


def _card(sem: dict) -> dict:
    return {
        "role": sem.get("role") or "",
        "pattern": sem.get("pattern") or [],
        "operations": sem.get("operations") or [],
        "relations": sem.get("relations") or [],
    }


def _reindent(text: str) -> str:
    lines = text.strip("\n").splitlines()
    filled = [line for line in lines if line.strip()]
    if not filled:
        return ""
    cut = min(len(line) - len(line.lstrip(" \t")) for line in filled)
    out = []
    for line in lines:
        if not line.strip():
            out.append("")
            continue
        out.append("\t" + line[cut:])
    return "\n".join(out).rstrip() + "\n"


def _function_body(parser, code: str):
    raw = code.encode("utf-8")
    tree = parser.parse(raw)

    def walk(node):
        if node.type in ("function_declaration", "method_declaration"):
            for ch in node.children:
                if ch.type == "block":
                    return ch
        for ch in node.children:
            found = walk(ch)
            if found is not None:
                return found
        return None

    return raw, walk(tree.root_node)


_BLOCK_TYPES = {
    "if_statement",
    "for_statement",
    "expression_switch_statement",
    "type_switch_statement",
    "select_statement",
}


def _mine_blocks(parser, banned_lines: set[int]) -> list[tuple[int, str]]:
    """One control-flow block per other training function."""
    found: list[tuple[int, str]] = []
    with TRAIN.open(encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx in banned_lines or not line.strip():
                continue
            code = full_code(json.loads(line))
            raw, body = _function_body(parser, code)
            if body is None:
                continue
            best = ""
            grouped: list[str] = []
            grouped_len = 0
            for ch in body.children:
                if ch.type not in STMT:
                    continue
                text = raw[ch.start_byte : ch.end_byte].decode("utf-8").strip()
                if not text:
                    continue
                size = len(text)
                if ch.type in _BLOCK_TYPES and 80 <= size <= 700 and size > len(best):
                    best = text
                if size <= 240 and grouped_len < 500:
                    grouped.append(text)
                    grouped_len += size + 1
            if not best and grouped_len >= 80:
                best = "\n".join(grouped)
            if best:
                found.append((idx, best))
    return found


def _pick_blocks(
    blocks: list[tuple[int, str]],
    rng: random.Random,
    parent_line: int,
) -> list[tuple[int, str]]:
    pool = [item for item in blocks if item[0] != parent_line]
    count = rng.choice((3, 4, 5))
    if len(pool) <= count:
        return pool
    return rng.sample(pool, count)


def rewrite_one(
    row: dict,
    parser,
    sem_by_line: dict[int, dict],
    serial: int,
    blocks: list[tuple[int, str]],
) -> dict:
    parent_pre = row["parent_prefix"]
    parent_mid = row["parent_middle"]
    parent_suf = row["parent_suffix"]
    parent = parent_pre + parent_mid + parent_suf
    raw = parent.encode("utf-8")
    mid0 = len(parent_pre.encode("utf-8"))
    mid1 = mid0 + len(parent_mid.encode("utf-8"))
    tree = parser.parse(raw)
    replaces, _keep_bytes, body = _plan(tree, raw, mid0, mid1)

    rng = random.Random(serial * 9973 + 17)
    edits: list[tuple[int, int, str]] = []
    for start, end in replaces:
        edits.append((start, end, "\n"))

    chosen = _pick_blocks(blocks, rng, int(row["ground_truth_line"]))
    borrowed = "\n" + "".join(_reindent(text) for _line, text in chosen)
    if body is not None and chosen:
        insert_at = body.end_byte - 1
        if insert_at < mid1:
            insert_at = len(raw)
        edits.append((insert_at, insert_at, borrowed))

    banned = set(content_tokens(parent))
    segments = [parent_pre, parent_mid, parent_suf]
    mapping = _build_mapping(segments, banned, rng)
    strings = _build_string_mapping(
        segments,
        banned,
        {word.casefold() for word in mapping.values()},
        rng,
    )

    new_pre = _apply(raw, 0, mid0, edits, mapping, strings)
    new_mid = _rename_text(parent_mid, mapping, strings)
    new_suf = _apply(raw, mid1, len(raw), edits, mapping, strings)
    if _mask_code(new_mid) != _mask_code(parent_mid):
        raise RuntimeError(f"{row['mutation_id']} middle shape changed")

    line = int(row["ground_truth_line"])
    sem = sem_by_line.get(line)
    if sem and sem.get("task_id") != row["ground_truth_task_id"]:
        raise RuntimeError(f"{row['mutation_id']} semantic task_id mismatch at line {line}")

    out = dict(row)
    out["prefix"] = new_pre
    out["middle"] = new_mid
    out["suffix"] = new_suf
    out["response"] = new_mid
    out["prompt"] = make_prompt(new_pre, new_suf)
    out["origin"] = "relation-rand"
    out["donor_lines"] = [src for src, _text in chosen]
    out["rewrite_note"] = (
        "kept relation lines replace each identifier and each word-bearing string "
        "with a random pool word; other statements are removed and 3-5 blocks "
        "from other train functions are inserted before the closing brace"
    )
    if sem:
        out["shared_semantic"] = _card(sem)
    out["gap"] = {}
    return out


def main() -> None:
    rows = [json.loads(line) for line in OUT.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 100:
        raise SystemExit(f"expected 100 mutations, found {len(rows)}")
    sem_by_line = _load_semantic()
    parser = _parser()
    banned = {int(row["ground_truth_line"]) for row in rows}
    print(f"[mine] scanning train blocks, skip {len(banned)} parents", flush=True)
    blocks = _mine_blocks(parser, banned)
    print(f"[mine] blocks={len(blocks)}", flush=True)
    rewritten = []
    for serial, row in enumerate(rows, start=1):
        rewritten.append(rewrite_one(row, parser, sem_by_line, serial, blocks))

    used = {int(row["ground_truth_line"]) for row in rewritten}
    pool_tokens, pool_ast = _pool(parser, used)
    parse_bad = []
    for row in rewritten:
        parent = full_code(
            {"prefix": row["parent_prefix"], "middle": row["parent_middle"], "suffix": row["parent_suffix"]}
        )
        mutant = full_code(row)
        row["gap"] = _score(mutant, parent, parser, pool_tokens, pool_ast)
        if row["gap"]["parse_error"]:
            parse_bad.append(row["mutation_id"])

    with OUT.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rewritten:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    n = len(rewritten)
    shared_rows = sum(1 for row in rewritten if row["gap"]["shared_content_tokens"])
    ast_top5 = sum(1 for row in rewritten if row["gap"]["ast_rank_in_251"] <= 5)
    ast_top1 = sum(1 for row in rewritten if row["gap"]["ast_rank_in_251"] == 1)
    print(
        f"n={n} parse_bad={len(parse_bad)} shared_rows={shared_rows} "
        f"ast_top1={ast_top1} ast_top5={ast_top5}"
    )
    if parse_bad:
        print("parse", ",".join(parse_bad))
    for row in rewritten:
        g = row["gap"]
        print(
            f"{row['mutation_id']} ast={g['ast_rank_in_251']} tok={g['token_rank_in_251']} "
            f"cos={g['token_cosine_to_parent']} shared={len(g['shared_content_tokens'])}"
        )


if __name__ == "__main__":
    main()
