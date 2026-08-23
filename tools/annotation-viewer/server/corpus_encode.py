"""Encode prompt+response corpus rows into compact ChatML input_ids + labels."""

from __future__ import annotations

DEFAULT_SYSTEM = "You are a helpful assistant."


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
