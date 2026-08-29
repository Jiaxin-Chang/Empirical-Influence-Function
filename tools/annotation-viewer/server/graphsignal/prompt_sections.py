#!/usr/bin/env python3
"""Parse rich FIM prompts (PRE/SUF/MID + struct/before/after) for annotation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


PRE = "<PRE>"
SUF = "<SUF>"
MID = "<MID>"
MASK = "[MASK]"
ANGLE_FIM = "<FIM>"

LANG_ALIASES = {
    "go": "Go",
    "golang": "Go",
    "python": "Python",
    "py": "Python",
    "java": "Java",
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "php": "PHP",
    "ruby": "Ruby",
    "c": "C",
    "cpp": "C++",
    "c++": "C++",
    "csharp": "C#",
    "c#": "C#",
}

FENCE_LANG_RE = re.compile(r"```([A-Za-z0-9+#]+)\b")
TASK_LANG_RE = re.compile(
    r"This is a\s+([A-Za-z0-9+#]+)\s+programming task",
    re.IGNORECASE,
)
TRAILING_FENCE_RE = re.compile(r"```[A-Za-z0-9+#]*\s*$")
RESPONSE_CUE_RE = re.compile(
    r"(?:\n+Ensure that only missing codes.*)?\n*### Response:\s*$",
    re.DOTALL,
)
INCOMPLETE_CODE_RE = re.compile(r"\* Incomplete Code:\n(.*?)(?:\n+Please fill|$)", re.DOTALL)

# Section headers in the rich prompt template (order matters for slicing).
BEFORE_HEADER = "The code snippets before the function is:"
AFTER_HEADER = "The code snippets after the function is:"
FUNCTION_HEADER = "And here is the function you are asked to complete:"
PACKAGE_HEADER = "Below is the package path:"
STRUCT_HEADERS = (
    "The receiver struct definitions of the function is:",
    "The parameter struct definitions or not exist of the function is:",
    "The return value struct definitions or not exist of the function is:",
)

FENCE_BODY_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


@dataclass
class EdgeableSpan:
    """A contiguous code span inside the filled user string."""

    kind: str  # before | function | after
    start: int
    end: int
    text: str


@dataclass
class PreparedPrompt:
    language: str
    prefix: str
    suffix: str
    target: str
    before_code: str
    after_code: str
    reference_text: str
    user_content: str
    messages: list[dict[str, str]]


def detect_language(prompt: str, default: str = "Go") -> str:
    m = TASK_LANG_RE.search(prompt)
    if m:
        return LANG_ALIASES.get(m.group(1).lower(), m.group(1).capitalize())
    m = FENCE_LANG_RE.search(prompt)
    if m:
        return LANG_ALIASES.get(m.group(1).lower(), m.group(1).capitalize())
    return default


def normalize_language(value: Any, default: str = "Go") -> str:
    text = str(value or default)
    return LANG_ALIASES.get(text.lower(), text)


@dataclass
class FimSpan:
    """Exact FIM marker geometry inside a prompt / user string."""

    pre_i: int
    suf_i: int
    mid_i: int
    mid_end: int
    prefix: str  # exact bytes between <PRE> and <SUF>
    suffix: str  # exact bytes between <SUF> and <MID>


@dataclass
class HoleFill:
    """Result of inserting the completion into the user hole for annotation."""

    mode: str  # "fim" | "mask"
    filled: str
    completion_start: int
    completion_end: int
    prefix: str
    suffix: str
    # fim geometry in the *user* string
    pre_i: int = -1
    prefix_start: int = -1
    suf_i: int = -1
    suffix_start: int = -1
    mid_i: int = -1
    mid_end: int = -1
    # mask geometry in the *user* string
    mask_pos: int = -1


def find_fim_span(prompt: str) -> FimSpan:
    """Locate the FIM hole via the last ``<PRE>`` (instruction text may mention ``<MID>``)."""
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


def extract_fim_span(prompt: str) -> tuple[int, int, str, str]:
    """Return (pre_start, mid_end, prefix, suffix) for the FIM hole."""
    span = find_fim_span(prompt)
    return span.pre_i, span.mid_end, span.prefix, span.suffix


def has_fim_markers(text: str) -> bool:
    try:
        find_fim_span(text)
        return True
    except ValueError:
        return False


def find_angle_fim_index(text: str) -> int:
    """Hole ``<FIM>`` index (last occurrence; earlier mentions are often instructional)."""
    return (text or "").rfind(ANGLE_FIM)


def has_angle_fim(text: str) -> bool:
    return find_angle_fim_index(text) >= 0


def has_any_hole_marker(text: str) -> bool:
    return has_fim_markers(text) or has_angle_fim(text) or (MASK in (text or ""))


def build_hole_fill(user_content: str, target: str) -> HoleFill:
    """Insert ``target`` into the user hole for annotation tokenization.

    Prefers ``<PRE>/<SUF>/<MID>``; then single ``<FIM>``; then ``[MASK]``.
    """
    if has_fim_markers(user_content):
        span = find_fim_span(user_content)
        filled = (
            user_content[: span.pre_i]
            + span.prefix
            + target
            + span.suffix
            + user_content[span.mid_end :]
        )
        completion_start = span.pre_i + len(span.prefix)
        return HoleFill(
            mode="fim",
            filled=filled,
            completion_start=completion_start,
            completion_end=completion_start + len(target),
            prefix=span.prefix,
            suffix=span.suffix,
            pre_i=span.pre_i,
            prefix_start=span.pre_i + len(PRE),
            suf_i=span.suf_i,
            suffix_start=span.suf_i + len(SUF),
            mid_i=span.mid_i,
            mid_end=span.mid_end,
        )

    fim_i = find_angle_fim_index(user_content)
    if fim_i >= 0:
        filled = (
            user_content[:fim_i]
            + target
            + user_content[fim_i + len(ANGLE_FIM) :]
        )
        return HoleFill(
            mode="angle_fim",
            filled=filled,
            completion_start=fim_i,
            completion_end=fim_i + len(target),
            prefix="",
            suffix="",
            mask_pos=fim_i,
        )

    marker = "* Incomplete Code:\n"
    marker_pos = user_content.find(marker)
    if marker_pos >= 0:
        code_start = marker_pos + len(marker)
        mask_pos = user_content.find(MASK, code_start)
    else:
        mask_pos = -1
    if mask_pos < 0:
        mask_pos = user_content.rfind(MASK)
    if mask_pos < 0:
        raise ValueError(
            "user content has neither <PRE>/<SUF>/<MID>, <FIM>, nor [MASK]"
        )
    filled = user_content[:mask_pos] + target + user_content[mask_pos + len(MASK) :]
    return HoleFill(
        mode="mask",
        filled=filled,
        completion_start=mask_pos,
        completion_end=mask_pos + len(target),
        prefix="",
        suffix="",
        mask_pos=mask_pos,
    )


def remap_hole_offset_to_chatml(
    offset: int,
    *,
    user_content_start: int,
    assistant_content_start: int,
    hole: HoleFill,
) -> int:
    """Map a char offset in ``hole.filled`` to an offset in the ChatML string."""
    if hole.mode in ("mask", "angle_fim"):
        target_len = hole.completion_end - hole.completion_start
        mask_pos = hole.mask_pos
        marker_len = len(ANGLE_FIM) if hole.mode == "angle_fim" else len(MASK)
        target_end = mask_pos + target_len
        if offset < mask_pos:
            return user_content_start + offset
        if offset < target_end:
            return assistant_content_start + (offset - mask_pos)
        shrink = target_len - marker_len
        return user_content_start + offset - shrink

    # filled: head + prefix + target + suffix + tail
    # user:   head + <PRE> + prefix + <SUF> + suffix + <MID> + tail
    head_len = hole.pre_i
    cs, ce = hole.completion_start, hole.completion_end
    if offset < head_len:
        return user_content_start + offset
    if offset < cs:
        return user_content_start + hole.prefix_start + (offset - head_len)
    if offset < ce:
        return assistant_content_start + (offset - cs)
    rel = offset - ce
    if rel < len(hole.suffix):
        return user_content_start + hole.suffix_start + rel
    return user_content_start + hole.mid_end + (rel - len(hole.suffix))


def fim_marker_spans_in_user(user_content: str) -> list[tuple[int, int]]:
    """Char spans of the FIM hole markers (not instructional mentions)."""
    if not has_fim_markers(user_content):
        return []
    span = find_fim_span(user_content)
    return [
        (span.pre_i, span.pre_i + len(PRE)),
        (span.suf_i, span.suf_i + len(SUF)),
        (span.mid_i, span.mid_end),
    ]


def _slice_until_headers(text: str, start: int, stop_headers: tuple[str, ...]) -> str:
    end = len(text)
    for header in stop_headers:
        idx = text.find(header, start)
        if idx >= 0:
            end = min(end, idx)
    return text[start:end]


def _first_fence_body(section_text: str) -> str:
    m = FENCE_BODY_RE.search(section_text)
    if not m:
        return ""
    return m.group(1).rstrip("\n")


def extract_named_section(prompt: str, header: str, stop_headers: tuple[str, ...]) -> str:
    idx = prompt.find(header)
    if idx < 0:
        return ""
    body_start = idx + len(header)
    return _slice_until_headers(prompt, body_start, stop_headers)


def extract_before_after_code(prompt: str) -> tuple[str, str]:
    stop_for_before = (AFTER_HEADER, FUNCTION_HEADER)
    stop_for_after = (FUNCTION_HEADER, BEFORE_HEADER)
    before_sec = extract_named_section(prompt, BEFORE_HEADER, stop_for_before)
    after_sec = extract_named_section(prompt, AFTER_HEADER, stop_for_after)
    return _first_fence_body(before_sec), _first_fence_body(after_sec)


def extract_reference_text(prompt: str) -> str:
    """Task text + package path + struct blocks (not before/after/function code)."""
    cut_headers = (BEFORE_HEADER, AFTER_HEADER, FUNCTION_HEADER, "* Incomplete Code:")
    cut = len(prompt)
    for header in cut_headers:
        idx = prompt.find(header)
        if idx >= 0:
            cut = min(cut, idx)
    return prompt[:cut].rstrip()


def get_target(row: dict[str, Any]) -> str:
    for key in ("target", "fim_completion", "response", "label"):
        if key in row and row[key] is not None:
            return str(row[key])
    raise ValueError("row has no target/fim_completion/response/label")


def build_user_content_from_raw(prompt: str) -> str:
    """Keep the original prompt, including ``<PRE>/<SUF>/<MID>``; drop response cue."""
    return RESPONSE_CUE_RE.sub("", prompt).rstrip()


def prepare_from_raw_prompt(
    prompt: str,
    target: str,
    *,
    language: str | None = None,
    default_language: str = "Go",
) -> PreparedPrompt:
    lang = normalize_language(language or detect_language(prompt, default_language), default_language)
    before_code, after_code = extract_before_after_code(prompt)
    user = build_user_content_from_raw(prompt)
    reference_text = extract_reference_text(user)

    if has_fim_markers(prompt):
        span = find_fim_span(prompt)
        prefix, suffix = span.prefix, span.suffix
    elif has_angle_fim(prompt):
        prefix, suffix = "", ""
    else:
        raise ValueError("prompt missing <PRE>/<SUF>/<MID> or <FIM> hole")

    messages = [
        {"role": "system", "content": f"You are a {lang} code completion assistant."},
        {"role": "user", "content": user},
        {"role": "assistant", "content": target},
    ]
    return PreparedPrompt(
        language=lang,
        prefix=prefix,
        suffix=suffix,
        target=target,
        before_code=before_code,
        after_code=after_code,
        reference_text=reference_text,
        user_content=user,
        messages=messages,
    )


def _enrich_chatml_row(row: dict[str, Any], *, default_language: str = "Go") -> dict[str, Any]:
    out = dict(row)
    user = get_user_content_from_row(out)
    language = normalize_language(out.get("language") or detect_language(user, default_language))
    out["language"] = language
    target = str(out.get("target") or out.get("fim_completion") or "")
    if not target:
        for msg in out.get("messages") or []:
            if msg.get("role") == "assistant":
                target = str(msg.get("content", ""))
                break
    out["target"] = target
    if "reference_text" not in out:
        out["reference_text"] = extract_reference_text(user)
    if "before_code" not in out or "after_code" not in out:
        before_code, after_code = extract_before_after_code(user)
        out.setdefault("before_code", before_code)
        out.setdefault("after_code", after_code)
    if "prefix" not in out or "suffix" not in out:
        if has_fim_markers(user):
            span = find_fim_span(user)
            out.setdefault("prefix", span.prefix)
            out.setdefault("suffix", span.suffix)
        elif has_angle_fim(user):
            out.setdefault("prefix", "")
            out.setdefault("suffix", "")
        else:
            m = INCOMPLETE_CODE_RE.search(user)
            if m and MASK in m.group(1):
                code = m.group(1)
                mi = code.find(MASK)
                out.setdefault("prefix", code[:mi])
                out.setdefault("suffix", code[mi + len(MASK) :])
    if "uid" not in out and out.get("task_id") is not None:
        out["uid"] = str(out["task_id"])
    out["only_last_turn_loss"] = True
    if has_fim_markers(user):
        out["hole_mode"] = "fim"
    elif has_angle_fim(user):
        out["hole_mode"] = "angle_fim"
    elif MASK in user:
        out["hole_mode"] = "mask"
    else:
        out["hole_mode"] = "unknown"
    return out


def prepare_row(row: dict[str, Any], *, default_language: str = "Go") -> dict[str, Any]:
    """Normalize a raw or ChatML row into annotate-ready ChatML + section fields.

    Raw prompts keep ``<PRE>/<SUF>/<MID>`` or single ``<FIM>`` in the user message.
    Legacy ``[MASK]`` ChatML rows remain supported.
    """
    if row.get("messages"):
        user = str(get_user_content_from_row(row))
        if (
            has_fim_markers(user)
            or has_angle_fim(user)
            or MASK in user
            or row.get("prefix") is not None
        ):
            return _enrich_chatml_row(row, default_language=default_language)

    prompt = str(row.get("prompt", ""))
    if not prompt:
        raise ValueError("row missing prompt/messages")
    target = get_target(row)
    prepared = prepare_from_raw_prompt(
        prompt,
        target,
        language=str(row["language"]) if row.get("language") else None,
        default_language=default_language,
    )
    if has_fim_markers(prompt):
        hole_mode = "fim"
    elif has_angle_fim(prompt):
        hole_mode = "angle_fim"
    else:
        hole_mode = "unknown"
    out = {
        "language": prepared.language,
        "prefix": prepared.prefix,
        "suffix": prepared.suffix,
        "target": prepared.target,
        "before_code": prepared.before_code,
        "after_code": prepared.after_code,
        "reference_text": prepared.reference_text,
        "messages": prepared.messages,
        "only_last_turn_loss": True,
        "hole_mode": hole_mode,
    }
    for key in ("task_id", "uid", "source_file", "source_line", "raw_id"):
        if key in row:
            out[key] = row[key]
    if "task_id" in row and "uid" not in out:
        out["uid"] = str(row["task_id"])
    return out


def get_user_content_from_row(row: dict[str, Any]) -> str:
    for msg in row.get("messages") or []:
        if msg.get("role") == "user":
            return str(msg.get("content", ""))
    return str(row.get("instruction", row.get("prompt", "")))


def _find_span(filled: str, text: str, *, prefer_after: int = 0) -> EdgeableSpan | None:
    if not text or not text.strip():
        return None
    idx = filled.find(text, prefer_after)
    if idx < 0:
        idx = filled.find(text)
    if idx < 0:
        # Fallback: search stripped variants for minor whitespace drift.
        stripped = text.strip("\n")
        idx = filled.find(stripped, prefer_after)
        if idx < 0:
            return None
        text = stripped
    return EdgeableSpan(kind="", start=idx, end=idx + len(text), text=filled[idx : idx + len(text)])


def _fence_span_containing(filled: str, pos: int) -> tuple[int, int] | None:
    """Return ``[start, end)`` of the ``` fence body that contains ``pos``, if any."""
    if pos < 0 or pos > len(filled):
        return None
    # Find opening fence before pos.
    open_idx = filled.rfind("```", 0, pos)
    if open_idx < 0:
        return None
    nl = filled.find("\n", open_idx)
    if nl < 0 or nl >= pos:
        return None
    body_start = nl + 1
    close_idx = filled.find("```", pos)
    if close_idx < 0:
        return None
    return body_start, close_idx


def locate_edgeable_spans(
    filled: str,
    *,
    before_code: str = "",
    after_code: str = "",
    hole: HoleFill | None = None,
) -> list[EdgeableSpan]:
    """Locate before / filled-function / after spans inside filled user text.

    Prompt order is before → after → function, but edgeable source order is
    before → function → after. Spans are returned in source order.
    """
    spans: list[EdgeableSpan] = []

    func_span: EdgeableSpan | None = None
    if hole is not None:
        func_start = hole.completion_start - len(hole.prefix)
        func_end = hole.completion_end + len(hole.suffix)
        if hole.mode in ("mask", "angle_fim"):
            m = INCOMPLETE_CODE_RE.search(filled)
            if m:
                func_start, func_end = m.start(1), m.end(1)
            elif hole.mode == "angle_fim":
                # Prefer the fenced code block that contains the filled hole.
                fence = _fence_span_containing(filled, hole.completion_start)
                if fence is not None:
                    func_start, func_end = fence
                else:
                    # Fallback: local window around the completion.
                    pad = 800
                    func_start = max(0, hole.completion_start - pad)
                    func_end = min(len(filled), hole.completion_end + pad)
        if 0 <= func_start < func_end <= len(filled):
            func_span = EdgeableSpan("function", func_start, func_end, filled[func_start:func_end])
    if func_span is None:
        m = INCOMPLETE_CODE_RE.search(filled)
        if m:
            func_span = EdgeableSpan("function", m.start(1), m.end(1), m.group(1))
        elif has_fim_markers(filled):
            span = find_fim_span(filled)
            func_span = EdgeableSpan("function", span.pre_i, span.mid_end, filled[span.pre_i : span.mid_end])
        elif has_angle_fim(filled):
            i = find_angle_fim_index(filled)
            fence = _fence_span_containing(filled, i)
            if fence is not None:
                func_span = EdgeableSpan("function", fence[0], fence[1], filled[fence[0]:fence[1]])

    before_span = _find_span(filled, before_code) if before_code else None
    if before_span:
        before_span.kind = "before"

    # Prefer the after block that appears before the incomplete function when possible.
    prefer_after = 0
    if before_span:
        prefer_after = before_span.end
    after_span = _find_span(filled, after_code, prefer_after=prefer_after) if after_code else None
    if after_span and func_span and after_span.start >= func_span.start:
        # Avoid matching code that was copied into the function body.
        after_span = _find_span(filled, after_code, prefer_after=0)
        if after_span and after_span.start >= func_span.start:
            after_span = None
    if after_span:
        after_span.kind = "after"

    for span in (before_span, func_span, after_span):
        if span is not None and span.end > span.start:
            spans.append(span)
    return spans


def edgeable_filled_view(spans: list[EdgeableSpan]) -> str:
    """Concatenate before/function/after in source order for LLM reference."""
    parts = [s.text for s in spans if s.text.strip()]
    return "\n\n".join(parts)
