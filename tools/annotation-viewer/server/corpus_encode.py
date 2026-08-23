"""Encode prompt+response corpus rows into compact ChatML input_ids + labels."""

from __future__ import annotations

from typing import Any

DEFAULT_SYSTEM = "You are a helpful assistant."


def _as_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return ""


def extract_prompt_response(row: dict[str, Any]) -> tuple[str, str]:
    """Best-effort prompt/completion from raw corpus / ChatML / FIM rows."""
    prompt = (
        _as_text(row.get("prompt"))
        or _as_text(row.get("input"))
        or _as_text(row.get("instruction"))
        or _as_text(row.get("user"))
    )
    response = (
        _as_text(row.get("response"))
        or _as_text(row.get("output"))
        or _as_text(row.get("completion"))
        or _as_text(row.get("target"))
        or _as_text(row.get("fim_completion"))
        or _as_text(row.get("gold"))
        or _as_text(row.get("answer"))
        or _as_text(row.get("canonical_solution"))
        or _as_text(row.get("mid"))
        or _as_text(row.get("code"))
    )
    # Compact rows store token ids in ``label``; only use it when it is text.
    if not response:
        response = _as_text(row.get("label"))
    resp_list = row.get("responses")
    if not response and isinstance(resp_list, list) and resp_list:
        response = _as_text(resp_list[0])

    for msg in row.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role") or "")
        content = _as_text(msg.get("content"))
        if not prompt and role == "user":
            prompt = content
        if not response and role == "assistant":
            response = content

    return prompt, response


def encode_prompt_response(
    tokenizer,
    prompt: str,
    response: str,
    *,
    system: str = DEFAULT_SYSTEM,
) -> tuple[list[int], list[int]]:
    """Mirror ``src.process_data.process_func_chatml`` (no truncation)."""
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    nl_tokens = tokenizer.encode("\n", add_special_tokens=False)

    def build_turn(role: str, content: str, *, train: bool) -> tuple[list[int], list[int]]:
        role_ids = [im_start_id] + tokenizer.encode(role, add_special_tokens=False) + nl_tokens
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        footer_ids = [im_end_id] + nl_tokens
        full_ids = role_ids + content_ids + footer_ids
        if train:
            labels = [-100] * len(role_ids) + content_ids + footer_ids
        else:
            labels = [-100] * len(full_ids)
        return full_ids, labels

    input_ids: list[int] = []
    labels: list[int] = []
    for role, content, train in (
        ("system", system, False),
        ("user", prompt, False),
        ("assistant", response, True),
    ):
        ids, labs = build_turn(role, content, train=train)
        input_ids.extend(ids)
        labels.extend(labs)
    return input_ids, labels
