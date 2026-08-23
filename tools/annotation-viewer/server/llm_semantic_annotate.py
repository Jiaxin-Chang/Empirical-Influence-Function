"""Per-token LLM semantic attention annotation for corpus FIM rows.

Unlike GraphSignal (tree-sitter structure), this asks the LLM to infer which
context tokens a model should attend to when correctly predicting each answer
token — one LLM call per completion token, up to ``max_sources_per_token`` edges.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from server.auto_annotate import (
    _annotate_backend,
    _build_client,
    _env,
    _extra_body,
    _is_junk_surface,
    _surface,
)
from server.corpus_encode import encode_prompt_response, extract_prompt_response

FIM_PRE = "<PRE>"
FIM_SUF = "<SUF>"
FIM_MID = "<MID>"
SUBTYPE = "semantic"


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def default_max_sources_per_token() -> int:
    return max(1, min(15, _env_int("ANNOTATE_SEMANTIC_MAX_SOURCES", 15)))


def default_max_answer_tokens() -> int:
    return max(1, _env_int("ANNOTATE_SEMANTIC_MAX_ANSWER_TOKENS", 48))


def _decode_tokens(tokenizer: Any, input_ids: list[int]) -> list[str]:
    return [tokenizer.decode([int(i)], skip_special_tokens=False) for i in input_ids]


def _answer_indices(labels: list[int], answer_start: int) -> list[int]:
    out: list[int] = []
    for i in range(answer_start, len(labels)):
        try:
            if int(labels[i]) != -100:
                out.append(i)
        except (TypeError, ValueError):
            continue
    return out


def _extract_fim_view(prompt: str, response: str) -> dict[str, str]:
    """Best-effort FIM decomposition for the LLM prompt.

    Supports both common layouts:
      ``<PRE> prefix <SUF> suffix <MID>``  (Code Llama / many jsonl dumps)
      ``<PRE> prefix <MID> hole <SUF> suffix``
    """
    p = str(prompt or "")
    r = str(response or "")
    pre = suf = hole = ""
    if FIM_PRE in p:
        after_pre = p.split(FIM_PRE, 1)[1]
        i_suf = after_pre.find(FIM_SUF)
        i_mid = after_pre.find(FIM_MID)
        if i_suf >= 0 and i_mid >= 0:
            if i_mid < i_suf:
                pre, rest = after_pre.split(FIM_MID, 1)
                parts = rest.split(FIM_SUF, 1)
                hole = parts[0]
                suf = parts[1] if len(parts) > 1 else ""
            else:
                pre, rest = after_pre.split(FIM_SUF, 1)
                parts = rest.split(FIM_MID, 1)
                suf = parts[0]
                hole = parts[1] if len(parts) > 1 else ""
        elif i_mid >= 0:
            pre, hole = after_pre.split(FIM_MID, 1)
        elif i_suf >= 0:
            pre, suf = after_pre.split(FIM_SUF, 1)
    return {
        "prefix": pre.strip(),
        "suffix": suf.strip(),
        "hole_placeholder": hole.strip(),
        "correct_mid": r,
        "full_user_prompt": p.strip(),
    }


def _completion_text(tokens: list[str], start: int, end: int) -> str:
    parts: list[str] = []
    for i in range(start, end):
        parts.append(_surface(tokens[i]))
    return "".join(parts).replace("Ġ", " ").replace("▁", " ")


def _select_context_indices(
    dst: int,
    answer_start: int,
    *,
    max_context: int = 420,
) -> list[int]:
    """Pick context indices ``< dst`` for the LLM (prompt tail + completion prefix)."""
    if dst <= 0:
        return []
    all_ctx = list(range(0, dst))
    if len(all_ctx) <= max_context:
        return all_ctx

    prompt_ctx = list(range(0, min(answer_start, dst)))
    answer_ctx = list(range(max(answer_start, 0), dst))
    picked: list[int] = []
    # Completion prefix is usually small and critical.
    picked.extend(answer_ctx)
    budget = max_context - len(picked)
    if budget <= 0:
        return sorted(set(picked))[:max_context]

    if len(prompt_ctx) <= budget:
        picked.extend(prompt_ctx)
        return sorted(set(picked))

    # Prompt is long: keep head (system/user start) + tail (FIM neighborhood).
    head_n = min(60, budget // 3)
    tail_n = budget - head_n
    picked.extend(prompt_ctx[:head_n])
    picked.extend(prompt_ctx[-tail_n:])
    return sorted(set(i for i in picked if 0 <= i < dst))[:max_context]


def _build_per_token_messages(
    *,
    language: str,
    fim: dict[str, str],
    tokens: list[str],
    answer_indices: list[int],
    dst: int,
    context_indices: list[int],
    max_sources: int,
) -> list[dict[str, str]]:
    pos_in_ans = answer_indices.index(dst) if dst in answer_indices else -1
    already = _completion_text(tokens, answer_indices[0], dst) if answer_indices else ""
    target_surf = _surface(tokens[dst]) if 0 <= dst < len(tokens) else ""

    completion_ref = [
        [int(i), _surface(tokens[i])]
        for i in answer_indices
    ]
    context_list = [
        [int(i), _surface(tokens[i])]
        for i in context_indices
        if not _is_junk_surface(_surface(tokens[i]))
    ]

    system = (
        "You infer causal attention routing for a code Fill-in-the-Middle (FIM) "
        "training sample.\n\n"
        "Setup: a causal LM predicts the assistant completion token-by-token (left "
        "to right). For the TARGET completion token at index `target_index`, list "
        "which earlier CONTEXT token indices (strictly less than `target_index`) "
        "the model should attend to in order to emit TARGET correctly.\n\n"
        "This is NOT syntactic bracket matching. Think like saliency / attention "
        "routing: which prior tokens provide the information needed to write TARGET "
        "now?\n\n"
        "Prioritize:\n"
        "1. FIM prefix/suffix code that constrains the hole (types, names, calls, "
        "control flow near <MID>)\n"
        "2. Identifiers, literals, or operators that TARGET copies or depends on\n"
        "3. Delimiters / keywords that govern TARGET's syntax\n"
        "4. Earlier completion tokens if TARGET continues them (e.g. partial "
        "identifier, open bracket)\n\n"
        "Rules:\n"
        f"- Return at most {max_sources} source indices\n"
        "- Every source index must appear in `context_tokens` and be < `target_index`\n"
        "- Do NOT pick TARGET itself or any later token\n"
        "- Avoid ChatML specials, pure whitespace, or meaningless punctuation unless "
        "structurally required\n"
        "- Precision over recall; fewer high-confidence sources beat noisy lists\n"
        "- Return JSON only, no markdown"
    )

    user_obj = {
        "language": language,
        "fim": {
            "user_prompt": fim.get("full_user_prompt", ""),
            "prefix_before_mid": fim.get("prefix", ""),
            "suffix_after_mid": fim.get("suffix", ""),
            "correct_completion_text": fim.get("correct_mid", ""),
        },
        "target_index": int(dst),
        "target_surface": target_surf,
        "position_in_completion": int(pos_in_ans),
        "already_generated_completion_text": already,
        "full_correct_completion_tokens": completion_ref,
        "context_tokens": context_list,
        "max_sources": int(max_sources),
        "output_schema": {
            "target_index": "int (repeat)",
            "sources": [
                {"index": "int", "reason": "short clause"}
            ],
        },
    }

    user = (
        "Infer attention sources for the TARGET token below.\n"
        f"{json.dumps(user_obj, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_sources_json(text: str, *, dst: int, allowed: set[int]) -> list[tuple[int, str]]:
    text = (text or "").strip()
    if not text:
        return []
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            return []
        parsed = json.loads(text[start : end + 1])

    sources_raw: list[Any] = []
    if isinstance(parsed, dict):
        if isinstance(parsed.get("sources"), list):
            sources_raw = parsed["sources"]
        elif isinstance(parsed.get("source_indices"), list):
            sources_raw = [{"index": x} for x in parsed["source_indices"]]
        elif isinstance(parsed.get("indices"), list):
            sources_raw = [{"index": x} for x in parsed["indices"]]
    elif isinstance(parsed, list):
        sources_raw = parsed

    out: list[tuple[int, str]] = []
    seen: set[int] = set()
    for item in sources_raw:
        idx: int | None = None
        reason = ""
        if isinstance(item, int):
            idx = int(item)
        elif isinstance(item, dict):
            for key in ("index", "src", "i", "token_i_idx"):
                if key in item:
                    try:
                        idx = int(item[key])
                        break
                    except (TypeError, ValueError):
                        continue
            reason = str(item.get("reason") or item.get("rationale") or "")
        if idx is None:
            continue
        if idx in seen:
            continue
        if idx not in allowed or idx >= dst:
            continue
        seen.add(idx)
        out.append((idx, reason))
    return out


def _call_llm_for_target(
    messages: list[dict[str, str]],
    *,
    dst: int,
    allowed: set[int],
    max_sources: int,
) -> tuple[list[tuple[int, str]], str]:
    backend = _annotate_backend()
    default_model = "qwen3.8-max" if backend == "api" else "qwen3-8b"
    model = (
        _env("ANNOTATE_MODEL")
        or _env("EIF_ANNOTATE_MODEL")
        or _env("OPENAI_MODEL")
        or default_model
    )
    max_tokens = int(_env("ANNOTATE_MAX_TOKENS", "2048") or "2048")
    client = _build_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    extra = _extra_body()
    if extra:
        kwargs["extra_body"] = extra

    raw = ""
    try:
        resp = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content or ""
    except Exception:
        resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""

    sources = _parse_sources_json(raw, dst=dst, allowed=allowed)[:max_sources]
    return sources, raw


def annotate_corpus_row_semantic(
    raw_row: dict[str, Any],
    tokenizer: Any,
    *,
    max_sources_per_token: int | None = None,
    max_answer_tokens: int | None = None,
) -> dict[str, Any]:
    prompt, response = extract_prompt_response(raw_row)
    if not prompt.strip():
        raise ValueError("corpus row missing prompt/input text")
    if not response.strip():
        raise ValueError(
            "corpus row missing completion text "
            "(expected string fields: response / output / completion / target, "
            "or messages[].assistant). Token-id `label` lists are not used."
        )

    language = str(raw_row.get("language") or "go")
    max_src = default_max_sources_per_token() if max_sources_per_token is None else max(
        1, min(15, int(max_sources_per_token)),
    )
    max_ans = default_max_answer_tokens() if max_answer_tokens is None else max(
        1, int(max_answer_tokens),
    )

    input_ids, labels = encode_prompt_response(tokenizer, prompt, response)
    tokens = _decode_tokens(tokenizer, input_ids)
    n = len(tokens)
    answer_start = next((i for i, lab in enumerate(labels) if int(lab) != -100), n)
    answer_indices = _answer_indices(labels, answer_start)[:max_ans]
    if not answer_indices:
        raise ValueError("no trainable answer tokens found in labels")

    fim = _extract_fim_view(prompt, response)
    edges: list[dict[str, Any]] = []
    edge_seen: set[tuple[int, int]] = set()
    llm_calls = 0

    print(
        f"[llm-semantic] start answer_tokens={len(answer_indices)} "
        f"max_sources={max_src} seq_len={n}",
        flush=True,
    )

    for dst in answer_indices:
        ctx_idx = _select_context_indices(dst, answer_start)
        allowed = set(ctx_idx)
        messages = _build_per_token_messages(
            language=language,
            fim=fim,
            tokens=tokens,
            answer_indices=answer_indices,
            dst=dst,
            context_indices=ctx_idx,
            max_sources=max_src,
        )
        t0 = time.perf_counter()
        try:
            sources, _raw = _call_llm_for_target(
                messages,
                dst=dst,
                allowed=allowed,
                max_sources=max_src,
            )
        except Exception as exc:
            print(f"[llm-semantic] dst={dst} LLM failed: {exc}", flush=True)
            continue
        llm_calls += 1
        kept = 0
        for src, reason in sources:
            if src >= dst or not (0 <= src < n):
                continue
            if _is_junk_surface(_surface(tokens[src])):
                continue
            key = (src, dst)
            if key in edge_seen:
                continue
            edge_seen.add(key)
            edge = {"src": src, "dst": dst, "subtype": SUBTYPE}
            if reason:
                edge["reason"] = reason
            edges.append(edge)
            kept += 1
        print(
            f"[llm-semantic] dst={dst} surf={_surface(tokens[dst])!r} "
            f"sources={kept} elapsed={time.perf_counter() - t0:.1f}s",
            flush=True,
        )

    if not edges:
        raise ValueError(
            "LLM semantic annotate produced no edges "
            f"({llm_calls} LLM calls on {len(answer_indices)} answer tokens)"
        )

    print(
        f"[llm-semantic] done edges={len(edges)} llm_calls={llm_calls}",
        flush=True,
    )
    return {
        "input_ids": input_ids,
        "label": labels,
        "attention_edges": edges,
        "_llm_semantic_meta": {
            "max_sources_per_token": max_src,
            "max_answer_tokens": max_ans,
            "answer_token_count": len(answer_indices),
            "llm_calls": llm_calls,
            "language": language,
            "annotate_model": _env("ANNOTATE_MODEL") or "",
        },
    }
