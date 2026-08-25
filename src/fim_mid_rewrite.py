"""Re-hollow FIM training samples when a corpus search hits context, not gold.

Goal: teach a MID pattern that currently only appears in train *context*.
Fill the original hole with the old MID (so it becomes context), then dig a
new ``<MID>`` over the span that matches the test gold (or a best proxy).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

PRE = "<PRE>"
SUF = "<SUF>"
MID = "<MID>"

# Rich Go FIM prompt sections (outside the <PRE>/<SUF>/<MID> hole).
BEFORE_HEADER = "The code snippets before the function is:"
AFTER_HEADER = "The code snippets after the function is:"
FUNCTION_HEADER = "And here is the function you are asked to complete:"

_STR_LIT_RE = re.compile(r'"((?:\\.|[^"\\])*)"|\'((?:\\.|[^\'\\])*)\'')
_FUNC_LINE_RE = re.compile(r"(?m)^[ \t]*func\b")


def _skip_string_or_comment(text: str, i: int) -> int | None:
    """If ``text[i]`` starts a string/comment, return index after it; else None."""
    n = len(text)
    if i >= n:
        return None
    ch = text[i]
    # Line comment
    if ch == "/" and i + 1 < n and text[i + 1] == "/":
        nl = text.find("\n", i + 2)
        return n if nl < 0 else nl
    # Block comment
    if ch == "/" and i + 1 < n and text[i + 1] == "*":
        end = text.find("*/", i + 2)
        return n if end < 0 else end + 2
    # Rune / string / raw string
    if ch == "`":
        end = text.find("`", i + 1)
        return n if end < 0 else end + 1
    if ch in "\"'":
        j = i + 1
        while j < n:
            if text[j] == "\\":
                j += 2
                continue
            if text[j] == ch:
                return j + 1
            j += 1
        return n
    return None


def _match_brace_block_end(text: str, open_idx: int) -> int:
    """Return index *after* the matching ``}`` for ``text[open_idx] == '{'``."""
    if open_idx < 0 or open_idx >= len(text) or text[open_idx] != "{":
        return -1
    depth = 0
    i = open_idx
    n = len(text)
    while i < n:
        skipped = _skip_string_or_comment(text, i)
        if skipped is not None:
            i = skipped
            continue
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return -1


def find_enclosing_go_func(
    text: str,
    dig_start: int,
    dig_end: int,
    *,
    clamp: tuple[int, int] | None = None,
) -> tuple[int, int] | None:
    """Find ``[func_start, func_end)`` of the Go func that contains ``[dig_start, dig_end)``.

    ``func_start`` is the line start of ``func ...``; ``func_end`` is past the
    closing ``}`` of that function body.
    """
    if dig_start < 0 or dig_end > len(text) or dig_start >= dig_end:
        return None
    lo, hi = (0, len(text)) if clamp is None else clamp
    lo = max(0, lo)
    hi = min(len(text), hi)
    if not (lo <= dig_start and dig_end <= hi):
        return None

    # Candidate func lines before dig, nearest first.
    region = text[lo:dig_start]
    starts = [lo + m.start() for m in _FUNC_LINE_RE.finditer(region)]
    for func_start in reversed(starts):
        # Opening brace of the function body (first '{' after signature).
        brace = -1
        i = func_start
        while i < hi:
            skipped = _skip_string_or_comment(text, i)
            if skipped is not None:
                i = skipped
                continue
            if text[i] == "{":
                brace = i
                break
            # Don't walk into a later top-level func.
            if i > func_start and _FUNC_LINE_RE.match(text, i):
                break
            i += 1
        if brace < 0:
            continue
        func_end = _match_brace_block_end(text, brace)
        if func_end < 0 or func_end > hi:
            continue
        if func_start <= dig_start and dig_end <= func_end:
            return func_start, func_end
    return None


def build_relocated_fim(
    doc: str,
    dig_start: int,
    dig_end: int,
    *,
    clamp: tuple[int, int] | None = None,
) -> tuple[str, str, str, str]:
    """Build prompt with FIM hole at dig; PRE/SUF span the enclosing function when possible.

    Returns ``(new_prompt, new_prefix, new_suffix, geometry)`` where geometry is
    ``func`` (PRE=func→MID, SUF=MID→func_end) or ``point`` (empty PRE body).
    """
    dig = doc[dig_start:dig_end]
    func = find_enclosing_go_func(doc, dig_start, dig_end, clamp=clamp)
    if func is not None:
        fs, fe = func
        new_prefix = doc[fs:dig_start]
        new_suffix = doc[dig_end:fe]
        new_prompt = doc[:fs] + PRE + new_prefix + SUF + new_suffix + MID + doc[fe:]
        return new_prompt, new_prefix, new_suffix, "func"
    # Fallback: point hole (everything left of dig is outside PRE).
    new_prompt = doc[:dig_start] + PRE + SUF + doc[dig_end:] + MID
    return new_prompt, "", doc[dig_end:], "point"


@dataclass(frozen=True)
class FimSpan:
    pre_i: int
    suf_i: int
    mid_i: int
    mid_end: int
    prefix: str
    suffix: str


def find_fim_span(prompt: str) -> FimSpan:
    pre_i = prompt.rfind(PRE)
    if pre_i < 0:
        raise ValueError("prompt missing <PRE> marker")
    suf_i = prompt.find(SUF, pre_i + len(PRE))
    mid_i = prompt.find(MID, suf_i + len(SUF)) if suf_i >= 0 else -1
    if suf_i < 0 or mid_i < 0:
        raise ValueError("prompt missing ordered <PRE>/<SUF>/<MID> markers")
    return FimSpan(
        pre_i=pre_i,
        suf_i=suf_i,
        mid_i=mid_i,
        mid_end=mid_i + len(MID),
        prefix=prompt[pre_i + len(PRE) : suf_i],
        suffix=prompt[suf_i + len(SUF) : mid_i],
    )


def has_fim_markers(prompt: str) -> bool:
    try:
        find_fim_span(prompt)
        return True
    except ValueError:
        return False


def _eval_boolean_expression(text: str, expression: str) -> bool:
    try:
        from llm_train_retrieval import eval_boolean_expression
    except ImportError:  # pragma: no cover
        from src.llm_train_retrieval import eval_boolean_expression
    return eval_boolean_expression(text, expression)


def classify_match_region(prompt: str, response: str, expression: str) -> str:
    """``gold`` if expression hits response; else ``context`` if it hits prompt."""
    expr = (expression or "").strip()
    if not expr:
        return "none"
    if _eval_boolean_expression(response or "", expr):
        return "gold"
    if _eval_boolean_expression(prompt or "", expr):
        return "context"
    return "none"


def _strip_outer_parens(expr: str) -> str:
    t = (expr or "").strip()
    while t.startswith("(") and t.endswith(")"):
        depth = 0
        in_str = False
        balanced = True
        for i, ch in enumerate(t):
            if ch == '"' and (i == 0 or t[i - 1] != "\\"):
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0 and i != len(t) - 1:
                    balanced = False
                    break
                if depth < 0:
                    balanced = False
                    break
        if not balanced or depth != 0:
            break
        t = t[1:-1].strip()
    return t


def _split_top_level(expr: str, op: str) -> list[str]:
    t = (expr or "").strip()
    if not t:
        return []
    op_re = re.compile(rf"\s+{re.escape(op)}\s+", re.IGNORECASE)
    parts: list[str] = []
    depth = 0
    in_str = False
    start = 0
    i = 0
    while i < len(t):
        ch = t[i]
        if ch == '"' and (i == 0 or t[i - 1] != "\\"):
            in_str = not in_str
            i += 1
            continue
        if in_str:
            i += 1
            continue
        if ch == "(":
            depth += 1
            i += 1
            continue
        if ch == ")":
            depth = max(0, depth - 1)
            i += 1
            continue
        if depth == 0:
            m = op_re.match(t, i)
            if m:
                parts.append(t[start:i].strip())
                i = m.end()
                start = i
                continue
        i += 1
    parts.append(t[start:].strip())
    return [p for p in parts if p]


def expression_literals(expression: str) -> list[str]:
    """Collect leaf string literals from an AND/OR expression (deduped, long-first)."""
    out: list[str] = []

    def walk(term: str) -> None:
        t = _strip_outer_parens(term)
        and_parts = _split_top_level(t, "AND")
        if len(and_parts) > 1:
            for p in and_parts:
                walk(p)
            return
        or_parts = _split_top_level(t, "OR")
        if len(or_parts) > 1:
            for p in or_parts:
                walk(p)
            return
        for m in _STR_LIT_RE.finditer(t):
            lit = m.group(1) if m.group(1) is not None else m.group(2)
            if lit is None:
                continue
            lit = lit.replace('\\"', '"').replace("\\'", "'").replace("\\\\", "\\")
            if lit.strip():
                out.append(lit)
            return
        raw = t.strip().strip('"').strip("'")
        if raw:
            out.append(raw)

    walk(expression or "")
    # Prefer longer needles when covering windows.
    dedup: list[str] = []
    seen: set[str] = set()
    for lit in sorted(out, key=len, reverse=True):
        if lit in seen:
            continue
        seen.add(lit)
        dedup.append(lit)
    return dedup


def _snap_to_lines(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand [start, end) to whole lines (keep trailing newline out of dig)."""
    if start < 0 or end > len(text) or start >= end:
        return start, end
    line_start = text.rfind("\n", 0, start) + 1
    nl = text.find("\n", end - 1 if end > 0 else 0)
    line_end = nl if nl >= end - 1 and nl >= 0 else len(text)
    # If dig already ends at newline, keep it; else include through line end without forcing NL.
    if end > start and text[end - 1] == "\n":
        line_end = end
    return line_start, max(line_end, end)


def _find_preferring_context(
    filled: str,
    needle: str,
    *,
    mid_start: int,
    mid_end: int,
) -> int:
    """Index of ``needle`` in ``filled``, preferring occurrences outside old MID."""
    if not needle:
        return -1
    best = -1
    start = 0
    while True:
        i = filled.find(needle, start)
        if i < 0:
            break
        dig_end = i + len(needle)
        entirely_in_mid = i >= mid_start and dig_end <= mid_end
        if not entirely_in_mid:
            return i
        if best < 0:
            best = i
        start = i + 1
    return best


def longest_needle_substr_in_hay(
    needle: str,
    hay: str,
    *,
    min_len: int,
) -> str | None:
    """Longest contiguous substring of ``needle`` that appears in ``hay``."""
    needle = needle or ""
    hay = hay or ""
    if not needle or not hay:
        return None
    min_len = max(1, int(min_len))

    lines = [ln for ln in needle.splitlines() if ln.strip()]
    best = ""
    for i in range(len(lines)):
        for j in range(len(lines), i, -1):
            block = "\n".join(lines[i:j])
            if len(block) < min_len:
                break
            if block in hay and len(block) > len(best):
                best = block
            # j decreases; first hit at this i is longest for this start
            if block in hay:
                break
    if best:
        return best

    hi = min(len(needle), len(hay))
    lo = min_len
    if hi < lo:
        return None
    found: str | None = None
    while lo <= hi:
        mid = (lo + hi) // 2
        hit: str | None = None
        limit = len(needle) - mid + 1
        # Cap character search for huge needles.
        step = 1 if limit <= 4000 else max(1, limit // 2000)
        for i in range(0, limit, step):
            sub = needle[i : i + mid]
            if sub in hay:
                hit = sub
                break
        if hit is None and step > 1:
            for i in range(0, limit):
                sub = needle[i : i + mid]
                if sub in hay:
                    hit = sub
                    break
        if hit is not None:
            found = hit
            lo = mid + 1
        else:
            hi = mid - 1
    return found


def _expression_cover_span(hay: str, expression: str) -> tuple[int, int] | None:
    """Minimal span in ``hay`` covering one hit of each AND-required literal."""
    if not hay or not (expression or "").strip():
        return None
    if not _eval_boolean_expression(hay, expression):
        return None

    lits = expression_literals(expression)
    if not lits:
        return None

    # AND: cover all literals that appear; OR-only leaves may over-cover — OK.
    present = [lit for lit in lits if lit in hay]
    if not present:
        return None

    positions: list[tuple[int, int]] = []
    for lit in present:
        i = hay.find(lit)
        if i < 0:
            continue
        positions.append((i, i + len(lit)))
    if not positions:
        return None
    start = min(p[0] for p in positions)
    end = max(p[1] for p in positions)
    # Cap runaway covers (expression terms far apart).
    if end - start > max(400, 3 * max(len(x) for x in present)):
        # Fall back to longest single literal span.
        lit = max(present, key=len)
        i = hay.find(lit)
        return i, i + len(lit)
    return start, end


def _section_ranges(doc: str) -> dict[str, tuple[int, int]]:
    """Byte ranges for before / function / after bodies inside a filled prompt."""
    text = doc or ""
    headers = [
        ("before", text.find(BEFORE_HEADER)),
        ("function", text.find(FUNCTION_HEADER)),
        ("after", text.find(AFTER_HEADER)),
    ]
    present = [(name, idx) for name, idx in headers if idx >= 0]
    present.sort(key=lambda x: x[1])
    out: dict[str, tuple[int, int]] = {}
    for i, (name, start) in enumerate(present):
        # Skip the header line itself; body starts after the header.
        body_start = start + len(
            BEFORE_HEADER if name == "before"
            else FUNCTION_HEADER if name == "function"
            else AFTER_HEADER
        )
        end = present[i + 1][1] if i + 1 < len(present) else len(text)
        if body_start < end:
            out[name] = (body_start, end)
    return out


def _locus_at(doc: str, pos: int) -> str:
    for name, (a, b) in _section_ranges(doc).items():
        if a <= pos < b:
            return name
    return "other"


def ws_flex_find(hay: str, needle: str) -> tuple[int, int] | None:
    """Find ``needle`` in ``hay`` allowing tab/space/newline run differences.

    Returns ``[start, end)`` into ``hay`` (original characters), or None.
    """
    parts = [p for p in re.split(r"\s+", (needle or "").strip()) if p]
    if not parts or not hay:
        return None
    # Prefer longer matches: require all tokens in order with flexible whitespace.
    pat = r"\s+".join(re.escape(p) for p in parts)
    try:
        m = re.search(pat, hay)
    except re.error:
        return None
    if not m:
        return None
    return m.start(), m.end()


def _choose_in_hay(hay: str, test_gold: str, expression: str) -> tuple[str, str]:
    """Return ``(dig_text, mode)`` from one haystack, or empty.

    ``dig_text`` is always a verbatim slice of ``hay`` (preserves tabs).
    """
    gold = (test_gold or "").strip("\n")
    hay = hay or ""
    if not hay.strip():
        return "", "unchanged"

    if gold and gold in hay:
        return gold, "exact_test_gold"

    if gold:
        span = ws_flex_find(hay, gold)
        if span is not None:
            a, b = span
            # Prefer snapping to whole lines when the match ate most of the gold.
            a2, b2 = _snap_to_lines(hay, a, b)
            dig = hay[a2:b2].strip("\n")
            if dig.strip():
                return dig, "ws_exact_test_gold"

    min_lcs = max(12, min(40, int(0.35 * len(gold)))) if gold else 12
    if gold:
        lines = [ln for ln in gold.splitlines() if ln.strip()]
        best_dig = ""
        best_mode = ""
        for i in range(len(lines)):
            for j in range(len(lines), i, -1):
                block = "\n".join(lines[i:j])
                if len(re.sub(r"\s+", "", block)) < min_lcs:
                    break
                span = ws_flex_find(hay, block)
                if span is None:
                    continue
                a, b = _snap_to_lines(hay, span[0], span[1])
                dig = hay[a:b].strip("\n")
                if len(dig) > len(best_dig):
                    best_dig = dig
                    best_mode = "ws_lcs_test_gold"
                break
        if best_dig:
            return best_dig, best_mode

        # Character LCS fallback (verbatim substring of gold that appears in hay).
        lcs = longest_needle_substr_in_hay(gold, hay, min_len=min_lcs)
        if lcs:
            return lcs, "lcs_test_gold"
        # Also try whitespace-flex on the longest gold line.
        long_line = max(lines, key=len) if lines else ""
        if long_line and len(long_line.strip()) >= 12:
            span = ws_flex_find(hay, long_line)
            if span is not None:
                a, b = _snap_to_lines(hay, span[0], span[1])
                dig = hay[a:b].strip("\n")
                if dig.strip():
                    return dig, "ws_line_test_gold"

    cover = _expression_cover_span(hay, expression)
    if cover is not None:
        a, b = _snap_to_lines(hay, cover[0], cover[1])
        dig = hay[a:b].strip("\n")
        if dig.strip():
            return dig, "expression_cover"

    return "", "unchanged"


def choose_dig_text(
    *,
    doc: str,
    test_gold: str,
    expression: str,
) -> tuple[str, str]:
    """Prefer before → after → full doc (old MID already filled into context)."""
    sections = _section_ranges(doc)
    # Hits in before/after are the common failure mode this rewrite targets.
    for name in ("before", "after"):
        if name not in sections:
            continue
        a, b = sections[name]
        dig, mode = _choose_in_hay(doc[a:b], test_gold, expression)
        if dig:
            return dig, f"{mode}_{name}"

    dig, mode = _choose_in_hay(doc, test_gold, expression)
    if dig:
        return dig, mode
    return "", "unchanged"


def rewrite_fim_mid(
    prompt: str,
    response: str,
    *,
    test_gold: str,
    expression: str = "",
) -> dict[str, Any]:
    """Fill old MID into context, dig a new MID for the test-gold-like span.

    Works for spans inside ``<PRE>/<SUF>`` **or** in before/after code sections:
    the original hole is filled first, then FIM markers are relocated to the dig.

    Returns a dict with ``prompt``, ``response``, ``mode``, and metadata.
    On failure / skip, returns the originals with ``mode='unchanged'``.
    """
    prompt = prompt or ""
    old_mid = response or ""
    try:
        from llm_train_retrieval import clean_gold_mid_completion
    except ImportError:  # pragma: no cover
        try:
            from src.llm_train_retrieval import clean_gold_mid_completion
        except ImportError:
            clean_gold_mid_completion = lambda s: (s or "").strip()  # type: ignore
    test_gold = clean_gold_mid_completion(test_gold or "")

    base: dict[str, Any] = {
        "prompt": prompt,
        "response": old_mid,
        "mode": "unchanged",
        "dig_text": "",
        "old_mid": old_mid,
        "reason": "",
    }
    if not has_fim_markers(prompt):
        base["reason"] = "no_fim_markers"
        return base

    # Train MID already is the teaching target → keep original sample.
    if test_gold.strip():
        if test_gold in old_mid or (
            old_mid.strip() and old_mid.strip() in test_gold and len(old_mid.strip()) >= 12
        ):
            base["reason"] = "train_mid_already_is_gold"
            return base
        mid_hit = ws_flex_find(old_mid, test_gold)
        if mid_hit is not None:
            covered = mid_hit[1] - mid_hit[0]
            if covered >= max(12, int(0.8 * len(re.sub(r"\s+", "", test_gold)))):
                base["reason"] = "train_mid_already_is_gold"
                return base

    span = find_fim_span(prompt)
    # Full user text with the original hole filled and markers removed.
    doc = (
        prompt[: span.pre_i]
        + span.prefix
        + old_mid
        + span.suffix
        + prompt[span.mid_end :]
    )
    old_mid_start = span.pre_i + len(span.prefix)
    old_mid_end = old_mid_start + len(old_mid)

    dig, mode = choose_dig_text(
        doc=doc,
        test_gold=test_gold,
        expression=expression,
    )
    if not dig or mode == "unchanged":
        base["reason"] = "no_dig_span"
        return base

    prefer_range: tuple[int, int] | None = None
    sections = _section_ranges(doc)
    if mode.endswith("_before") and "before" in sections:
        prefer_range = sections["before"]
    elif mode.endswith("_after") and "after" in sections:
        prefer_range = sections["after"]

    pos = -1
    if prefer_range is not None:
        a, b = prefer_range
        local = doc[a:b]
        rel = local.find(dig)
        if rel >= 0:
            pos = a + rel
    if pos < 0:
        pos = _find_preferring_context(
            doc, dig, mid_start=old_mid_start, mid_end=old_mid_end,
        )
    if pos < 0:
        base["reason"] = "dig_not_found_in_filled"
        return base

    # Relocate FIM hole to dig. Prefer PRE/SUF = enclosing Go function
    # (func start → MID, MID → func end), especially for before/after snippets.
    if dig == old_mid and pos == old_mid_start:
        base["reason"] = "same_as_original_mid"
        return base

    dig_end = pos + len(dig)
    clamp = prefer_range
    # When dig is in the main function section, still prefer enclosing-func geometry.
    if clamp is None and "function" in sections:
        clamp = sections["function"]

    new_prompt, new_prefix, new_suffix, geometry = build_relocated_fim(
        doc, pos, dig_end, clamp=clamp,
    )
    locus = _locus_at(doc, pos)
    dig_hash = hashlib.sha1(dig.encode("utf-8")).hexdigest()[:10]
    return {
        "prompt": new_prompt,
        "response": dig,
        "mode": mode,
        "dig_locus": locus,
        "fim_geometry": geometry,
        "dig_text": dig,
        "old_mid": old_mid,
        "new_prefix_preview": new_prefix[:120],
        "new_suffix_preview": new_suffix[:80],
        "dig_hash": dig_hash,
        "reason": "ok",
        "dig_start_in_filled": pos,
        "dig_end_in_filled": dig_end,
    }
