"""Full-sample LLM semantic attention annotation for corpus FIM rows.

Unlike GraphSignal (tree-sitter structure), this asks the LLM once for the whole
completion: which earlier context tokens a model should attend to when correctly
predicting each answer token. Caps total edges and per-dst sources.
"""

from __future__ import annotations

import json
import os
import re
import signal
import threading
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

_abort_event = threading.Event()
_sigint_count = 0
_prev_sigint_handler = None


class LlmSemanticCancelled(Exception):
    """Raised when semantic annotate is aborted (user cancel or server shutdown)."""


def request_llm_semantic_abort() -> None:
    _abort_event.set()


def clear_llm_semantic_abort() -> None:
    _abort_event.clear()


def llm_semantic_abort_requested() -> bool:
    return _abort_event.is_set()


def install_llm_semantic_signal_handlers() -> None:
    """First Ctrl+C requests abort; second forces immediate process exit."""
    global _prev_sigint_handler

    def _handler(signum, frame):  # noqa: ANN001
        global _sigint_count
        request_llm_semantic_abort()
        _sigint_count += 1
        if _sigint_count >= 2:
            print("[llm-semantic] second Ctrl+C — force exit", flush=True)
            os._exit(130)
        print(
            "[llm-semantic] interrupt: stopping after current step "
            "(Ctrl+C again to force quit)",
            flush=True,
        )
        if callable(_prev_sigint_handler) and _prev_sigint_handler not in (
            signal.SIG_DFL,
            signal.SIG_IGN,
        ):
            _prev_sigint_handler(signum, frame)

    try:
        _prev_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, _handler)
    except (ValueError, OSError):
        pass  # not main thread / unsupported platform


def _env_int(name: str, default: int) -> int:
    raw = _env(name, str(default))
    try:
        return int(raw)
    except ValueError:
        return default


def _optional_env_int(name: str) -> int | None:
    raw = _env(name, "")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def default_max_sources_per_token() -> int:
    return max(1, min(15, _env_int("ANNOTATE_SEMANTIC_MAX_SOURCES", 15)))


def default_max_answer_tokens() -> int:
    return max(1, _env_int("ANNOTATE_SEMANTIC_MAX_ANSWER_TOKENS", 48))


def resolve_max_edges(
    n_answer_tokens: int,
    max_sources_per_token: int,
    override: int | None = None,
) -> int:
    """Cap = answer tokens × sources/token, unless an explicit ceiling is set.

    Unset ``ANNOTATE_SEMANTIC_MAX_EDGES`` → ``n * 15`` (or whatever sources/token is).
    Env / API override is an extra ceiling, never above that budget.
    """
    budget = max(1, int(n_answer_tokens) * max(1, int(max_sources_per_token)))
    cap = override if override is not None else _optional_env_int("ANNOTATE_SEMANTIC_MAX_EDGES")
    if cap is None:
        return budget
    return max(1, min(budget, int(cap)))


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


def default_max_context() -> int:
    return max(80, min(2000, _env_int("ANNOTATE_SEMANTIC_MAX_CONTEXT", 800)))


def _build_full_sample_messages(
    *,
    language: str,
    fim: dict[str, str],
    tokens: list[str],
    answer_indices: list[int],
    context_indices: list[int],
    max_edges: int,
    max_sources_per_dst: int,
    prompt_bundle: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    from server.semantic_prompt import DEFAULT_SYSTEM, format_few_shots_block, load_active

    bundle = prompt_bundle if isinstance(prompt_bundle, dict) else load_active()
    system = str(bundle.get("system") or DEFAULT_SYSTEM).strip() + "\n"
    shots = format_few_shots_block(list(bundle.get("few_shots") or []))
    if shots:
        system += "\n" + shots

    completion_ref = [
        [int(i), _surface(tokens[i])]
        for i in answer_indices
        if 0 <= i < len(tokens)
    ]
    context_list = [
        [int(i), _surface(tokens[i])]
        for i in context_indices
        if not _is_junk_surface(_surface(tokens[i]))
    ]
    user_prompt = str(fim.get("full_user_prompt") or "")
    if len(user_prompt) > 6000:
        user_prompt = user_prompt[:3000] + "\n…\n" + user_prompt[-2500:]

    user_obj = {
        "language": language,
        "fim": {
            "user_prompt": user_prompt,
            "prefix_before_mid": fim.get("prefix", ""),
            "suffix_after_mid": fim.get("suffix", ""),
            "correct_completion_text": fim.get("correct_mid", ""),
        },
        "completion_tokens": completion_ref,
        "context_tokens": context_list,
        "max_edges": int(max_edges),
        "max_sources_per_completion_token": int(max_sources_per_dst),
        "output_schema": {
            "edges": [
                {"src": "int", "dst": "int", "reason": "short clause"}
            ],
        },
    }
    user = (
        "Annotate the whole completion in one shot. Return every high-confidence "
        "src→dst routing edge (dst in completion_tokens).\n"
        f"{json.dumps(user_obj, ensure_ascii=False, indent=2)}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_full_edges_json(
    text: str,
    *,
    allowed_src: set[int],
    allowed_dst: set[int],
    max_edges: int,
    max_sources_per_dst: int,
) -> list[tuple[int, int, str]]:
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
            start, end = text.find("["), text.rfind("]")
        if start < 0 or end <= start:
            return []
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            return []

    raw_edges: list[Any] = []
    if isinstance(parsed, dict):
        if isinstance(parsed.get("edges"), list):
            raw_edges = parsed["edges"]
        elif isinstance(parsed.get("pairs"), list):
            raw_edges = parsed["pairs"]
        elif isinstance(parsed.get("by_target"), dict):
            for dst_s, sources in parsed["by_target"].items():
                try:
                    dst = int(dst_s)
                except (TypeError, ValueError):
                    continue
                if isinstance(sources, list):
                    for item in sources:
                        if isinstance(item, dict):
                            raw_edges.append({**item, "dst": dst})
                        else:
                            raw_edges.append({"src": item, "dst": dst})
    elif isinstance(parsed, list):
        raw_edges = parsed

    per_dst: dict[int, int] = {}
    out: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int]] = set()
    for item in raw_edges:
        src = dst = None
        reason = ""
        if isinstance(item, dict):
            for sk, dk in (("src", "dst"), ("i", "j"), ("token_i_idx", "token_j_idx")):
                if sk in item and dk in item:
                    try:
                        src, dst = int(item[sk]), int(item[dk])
                    except (TypeError, ValueError):
                        src = dst = None
                    break
            reason = str(item.get("reason") or item.get("rationale") or "")
        if src is None or dst is None:
            continue
        if dst not in allowed_dst or src not in allowed_src or src >= dst:
            continue
        key = (src, dst)
        if key in seen:
            continue
        if per_dst.get(dst, 0) >= max_sources_per_dst:
            continue
        seen.add(key)
        per_dst[dst] = per_dst.get(dst, 0) + 1
        out.append((src, dst, reason))
        if len(out) >= max_edges:
            break
    return out


def _call_llm_full_sample(
    messages: list[dict[str, str]],
    *,
    allowed_src: set[int],
    allowed_dst: set[int],
    max_edges: int,
    max_sources_per_dst: int,
) -> tuple[list[tuple[int, int, str]], str]:
    backend = _annotate_backend()
    default_model = "qwen3.8-max" if backend == "api" else "qwen3-8b"
    model = (
        _env("ANNOTATE_MODEL")
        or _env("EIF_ANNOTATE_MODEL")
        or _env("OPENAI_MODEL")
        or default_model
    )
    max_tokens = int(_env("ANNOTATE_MAX_TOKENS", "4096") or "4096")
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

    edges = _parse_full_edges_json(
        raw,
        allowed_src=allowed_src,
        allowed_dst=allowed_dst,
        max_edges=max_edges,
        max_sources_per_dst=max_sources_per_dst,
    )
    return edges, raw


def _call_llm_full_sample_interruptible(
    messages: list[dict[str, str]],
    *,
    allowed_src: set[int],
    allowed_dst: set[int],
    max_edges: int,
    max_sources_per_dst: int,
) -> tuple[list[tuple[int, int, str]], str]:
    """Run the single full-sample LLM call in a side thread so abort can break waits."""
    if llm_semantic_abort_requested():
        raise LlmSemanticCancelled("aborted before LLM call")

    result: dict[str, Any] = {}
    error: dict[str, BaseException] = {}

    def _worker() -> None:
        try:
            result["pair"] = _call_llm_full_sample(
                messages,
                allowed_src=allowed_src,
                allowed_dst=allowed_dst,
                max_edges=max_edges,
                max_sources_per_dst=max_sources_per_dst,
            )
        except BaseException as exc:
            error["exc"] = exc

    t = threading.Thread(target=_worker, name="llm-semantic-full", daemon=True)
    t.start()
    while t.is_alive():
        if llm_semantic_abort_requested():
            raise LlmSemanticCancelled("aborted during LLM call")
        t.join(timeout=0.4)
    if "exc" in error:
        raise error["exc"]
    return result["pair"]


def annotate_prompt_response_semantic(
    prompt: str,
    response: str,
    tokenizer: Any,
    *,
    language: str = "go",
    max_sources_per_token: int | None = None,
    max_answer_tokens: int | None = None,
    max_edges: int | None = None,
    prompt_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One LLM call → compact src→dst edge table for a FIM prompt+completion."""
    from server.semantic_prompt import load_active

    if not str(prompt or "").strip():
        raise ValueError("missing prompt/input text")
    if not str(response or "").strip():
        raise ValueError(
            "missing completion text "
            "(expected string fields: response / output / completion / target, "
            "or messages[].assistant). Token-id `label` lists are not used."
        )

    max_src = default_max_sources_per_token() if max_sources_per_token is None else max(
        1, min(15, int(max_sources_per_token)),
    )
    max_ans = default_max_answer_tokens() if max_answer_tokens is None else max(
        1, int(max_answer_tokens),
    )
    bundle = prompt_bundle if isinstance(prompt_bundle, dict) else load_active()
    vid = str(bundle.get("id") or "default")

    input_ids, labels = encode_prompt_response(tokenizer, prompt, response)
    tokens = _decode_tokens(tokenizer, input_ids)
    n = len(tokens)
    answer_start = next((i for i, lab in enumerate(labels) if int(lab) != -100), n)
    answer_indices = _answer_indices(labels, answer_start)[:max_ans]
    if not answer_indices:
        raise ValueError("no trainable answer tokens found in labels")
    max_e = resolve_max_edges(len(answer_indices), max_src, max_edges)

    last_dst = int(answer_indices[-1])
    ctx_idx = _select_context_indices(
        last_dst, answer_start, max_context=default_max_context(),
    )
    allowed_src = {i for i in ctx_idx if 0 <= i < n}
    allowed_dst = {i for i in answer_indices if 0 <= i < n}
    fim = _extract_fim_view(prompt, response)
    messages = _build_full_sample_messages(
        language=str(language or "go"),
        fim=fim,
        tokens=tokens,
        answer_indices=answer_indices,
        context_indices=ctx_idx,
        max_edges=max_e,
        max_sources_per_dst=max_src,
        prompt_bundle=bundle,
    )

    clear_llm_semantic_abort()
    print(
        f"[llm-semantic] start mode=full answer_tokens={len(answer_indices)} "
        f"max_edges={max_e} max_sources={max_src} seq_len={n} prompt={vid}",
        flush=True,
    )
    t0 = time.perf_counter()
    pairs, _raw = _call_llm_full_sample_interruptible(
        messages,
        allowed_src=allowed_src,
        allowed_dst=allowed_dst,
        max_edges=max_e,
        max_sources_per_dst=max_src,
    )
    elapsed = time.perf_counter() - t0

    edges: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for src, dst, reason in pairs:
        if src >= dst or not (0 <= src < n and 0 <= dst < n):
            continue
        if _is_junk_surface(_surface(tokens[src])):
            continue
        key = (src, dst)
        if key in seen:
            continue
        seen.add(key)
        edge = {"src": src, "dst": dst, "subtype": SUBTYPE}
        if reason:
            edge["reason"] = reason
        edges.append(edge)

    print(
        f"[llm-semantic] done mode=full edges={len(edges)} llm_calls=1 "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )
    if not edges:
        raise ValueError(
            "LLM semantic annotate produced no edges "
            f"(1 LLM call on {len(answer_indices)} answer tokens)"
        )
    return {
        "input_ids": input_ids,
        "label": labels,
        "attention_edges": edges,
        "tokens": tokens,
        "_llm_semantic_meta": {
            "mode": "full",
            "max_sources_per_token": max_src,
            "max_answer_tokens": max_ans,
            "max_edges": max_e,
            "answer_token_count": len(answer_indices),
            "llm_calls": 1,
            "language": str(language or "go"),
            "annotate_model": _env("ANNOTATE_MODEL") or "",
            "prompt_version": vid,
            "elapsed_sec": round(elapsed, 2),
        },
    }


def annotate_corpus_row_semantic(
    raw_row: dict[str, Any],
    tokenizer: Any,
    *,
    language: str | None = None,
    max_sources_per_token: int | None = None,
    max_answer_tokens: int | None = None,
    max_edges: int | None = None,
    prompt_bundle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    prompt, response = extract_prompt_response(raw_row)
    return annotate_prompt_response_semantic(
        prompt,
        response,
        tokenizer,
        language=str(language or raw_row.get("language") or "go"),
        max_sources_per_token=max_sources_per_token,
        max_answer_tokens=max_answer_tokens,
        max_edges=max_edges,
        prompt_bundle=prompt_bundle,
    )
