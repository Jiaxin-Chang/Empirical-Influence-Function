"""LLM auto-annotate for continue-train (OpenAI-compatible endpoint).

Probe/test FIM (with SOURCE/TARGET marked) gives the focus mechanism.
Train FIM + indexed tokens is what gets annotated. Never mix indices.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

CONTINUE_CONTRIBS = frozenset({"user_add", "user_bump", "llm_auto"})
ROUTE_SUBTYPE = "route"

FIM_PRE = "<PRE>"
FIM_SUF = "<SUF>"
FIM_MID = "<MID>"
_IM_END = "<|im_end|>"


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _annotate_backend() -> str:
    raw = _env("ANNOTATE_BACKEND", "").lower()
    if raw in ("api", "dashscope", "remote", "cloud"):
        return "api"
    if raw in ("vllm", "local"):
        return "vllm"
    if _env("DASHSCOPE_API_KEY"):
        return "api"
    return "vllm"


def _build_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "openai package required for auto-annotate. "
            "pip install openai"
        ) from exc

    backend = _annotate_backend()
    kwargs: dict[str, Any] = {}

    if backend == "api":
        api_key = (
            _env("DASHSCOPE_API_KEY")
            or _env("OPENAI_API_KEY")
            or _env("ANNOTATE_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "ANNOTATE_BACKEND=api requires DASHSCOPE_API_KEY or OPENAI_API_KEY "
                "in repo-root eif_api.env"
            )
        kwargs["api_key"] = api_key
        kwargs["base_url"] = (
            _env("OPENAI_BASE_URL")
            or _env("ANNOTATE_BASE_URL")
            or "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
    else:
        kwargs["api_key"] = (
            _env("OPENAI_API_KEY")
            or _env("ANNOTATE_API_KEY")
            or "dummy"
        )
        base = _env("OPENAI_BASE_URL") or _env("ANNOTATE_BASE_URL") or "http://127.0.0.1:8000/v1"
        kwargs["base_url"] = base
        try:
            import httpx
            kwargs["http_client"] = httpx.Client(
                transport=httpx.HTTPTransport(proxy=None, verify=True),
            )
        except Exception:
            pass

    return OpenAI(**kwargs)


def _extra_body() -> dict[str, Any]:
    body: dict[str, Any] = {}
    raw = _env("ANNOTATE_EXTRA_BODY_JSON")
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                body.update(parsed)
        except json.JSONDecodeError:
            pass
    think = _env("ANNOTATE_ENABLE_THINKING", _env("ENABLE_THINKING", "")).lower()
    if think in ("1", "true", "yes", "on"):
        body.setdefault("enable_thinking", True)
    elif think in ("0", "false", "no", "off"):
        body["enable_thinking"] = False
    return body


def _surface(tok: Any) -> str:
    if tok is None:
        return ""
    if isinstance(tok, str):
        return tok
    return str(tok)


def _norm_surf(tok: Any) -> str:
    return _surface(tok).replace("Ġ", " ").replace("▁", " ")


def _join_tokens(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = 0
    for t in tokens:
        s = _norm_surf(t)
        parts.append(s)
        spans.append((pos, pos + len(s)))
        pos += len(s)
    return "".join(parts), spans


def _answer_start_from_labels(labels: list[int] | None, n: int) -> int:
    if isinstance(labels, list) and len(labels) == n:
        for i, lab in enumerate(labels):
            try:
                if int(lab) != -100:
                    return i
            except (TypeError, ValueError):
                continue
    return max(1, (2 * n) // 3) if n else 0


def _mid_token_range(tokens: list[str], answer_start: int) -> tuple[int, int]:
    n = len(tokens)
    if answer_start < 0 or answer_start >= n:
        return 0, 0
    end = n
    for j in range(answer_start, n):
        if _IM_END in _surface(tokens[j]):
            end = j
            break
    return answer_start, max(answer_start, end)


def _find_fim_bodies(text: str) -> tuple[str, str] | None:
    pre_i = text.rfind(FIM_PRE)
    if pre_i < 0:
        return None
    suf_i = text.find(FIM_SUF, pre_i + len(FIM_PRE))
    if suf_i < 0:
        return None
    mid_i = text.find(FIM_MID, suf_i + len(FIM_SUF))
    if mid_i < 0:
        return None
    prefix = text[pre_i + len(FIM_PRE) : suf_i]
    suffix = text[suf_i + len(FIM_SUF) : mid_i]
    return prefix, suffix


def _mark_token(surf: str, role: str) -> str:
    s = surf if surf else "·"
    return f"**{s}**【{role}】"


def _or_empty(text: str) -> str:
    """Keep FIM section headers visible when a region has no body tokens."""
    if text is None:
        return "(empty)"
    if text.strip() == "":
        return "(empty)"
    return text


def build_fim_marked_view(
    tokens: list[str],
    *,
    answer_start: int,
    mark_src: int | None = None,
    mark_dst: int | None = None,
    src_role: str = "SOURCE",
    dst_role: str = "TARGET",
    mid_override: str | None = None,
    include_indexed: bool = False,
) -> dict[str, Any]:
    """Readable PRE+MID+SUF with optional marked token pair.

    ``mid_override`` replaces MID text (e.g. predict completion). Marking still
    uses ``mark_*`` indices on ``tokens`` when they fall in PRE/SUF/MID ranges;
    if MID is overridden and a mark sits in the completion span, a footnote is
    appended so the focus is still visible.
    """
    text, spans = _join_tokens(tokens)
    mid_lo, mid_hi = _mid_token_range(tokens, int(answer_start))
    gold_mid = "".join(_norm_surf(tokens[i]) for i in range(mid_lo, mid_hi))
    mid_text = mid_override if mid_override is not None else gold_mid

    fim = _find_fim_bodies(text)
    if fim is not None:
        prefix, suffix = fim
    else:
        prefix = "".join(_norm_surf(tokens[i]) for i in range(0, mid_lo))
        suffix = ""

    n = len(tokens)
    ms = int(mark_src) if mark_src is not None else -1
    md = int(mark_dst) if mark_dst is not None else -1
    if not (0 <= ms < n):
        ms = -1
    if not (0 <= md < n):
        md = -1

    def _render_region(indices: list[int]) -> str:
        chunks: list[str] = []
        for i in indices:
            surf = _norm_surf(tokens[i])
            if i == ms:
                chunks.append(_mark_token(surf, src_role))
            elif i == md:
                chunks.append(_mark_token(surf, dst_role))
            else:
                chunks.append(surf)
        return "".join(chunks)

    pre_idxs: list[int] = []
    suf_idxs: list[int] = []
    if fim is not None:
        pre_i = text.rfind(FIM_PRE)
        suf_i = text.find(FIM_SUF, pre_i + len(FIM_PRE))
        mid_m = text.find(FIM_MID, suf_i + len(FIM_SUF))
        pre_body_lo, pre_body_hi = pre_i + len(FIM_PRE), suf_i
        suf_body_lo, suf_body_hi = suf_i + len(FIM_SUF), mid_m
        for ti, (a, b) in enumerate(spans):
            if b > pre_body_lo and a < pre_body_hi:
                pre_idxs.append(ti)
            if b > suf_body_lo and a < suf_body_hi:
                suf_idxs.append(ti)
    else:
        pre_idxs = list(range(0, mid_lo))

    mid_idxs = list(range(mid_lo, mid_hi))
    pre_view = _render_region(pre_idxs) if pre_idxs else prefix
    suf_view = _render_region(suf_idxs) if suf_idxs else suffix

    if mid_override is not None:
        mid_view = mid_text
        notes: list[str] = []
        if mid_lo <= ms < mid_hi:
            notes.append(
                f"SOURCE idx={ms} surface={_surface(tokens[ms])!r} "
                f"(in train/probe completion span; MID text overridden)"
            )
        if mid_lo <= md < mid_hi:
            notes.append(
                f"TARGET idx={md} surface={_surface(tokens[md])!r} "
                f"(in train/probe completion span; MID text overridden)"
            )
        if notes:
            mid_view = mid_text + "\n\n[" + "; ".join(notes) + "]"
    else:
        mid_view = _render_region(mid_idxs) if mid_idxs else mid_text

    recon = (
        f"<PRE>\n{_or_empty(pre_view)}\n"
        f"<SUF>\n{_or_empty(suf_view)}\n"
        f"<MID>\n{_or_empty(mid_view)}\n"
        f"</reconstructed as prefix+mid+suffix for reading>\n\n"
        f"=== Reconstructed code (prefix + mid + suffix) ===\n"
        f"{pre_view}{mid_view}{suf_view}\n"
    )
    out: dict[str, Any] = {
        "fim_view": recon,
        "n_tokens": n,
        "mark_src": ms if ms >= 0 else None,
        "mark_dst": md if md >= 0 else None,
    }
    if include_indexed:
        out["indexed_tokens"] = [{"i": i, "t": _surface(tokens[i])} for i in range(n)]
    return out


_SPECIAL_SURFACES = frozenset({
    "<|im_start|>",
    "<|im_end|>",
    "<PRE>",
    "<SUF>",
    "<MID>",
    "user",
    "assistant",
    "system",
})


def _is_junk_surface(surf: str) -> bool:
    s = (surf or "").strip()
    if not s:
        return True
    if s in _SPECIAL_SURFACES:
        return True
    if re.fullmatch(r"[\W_]+", s, flags=re.UNICODE):
        return True
    return False


def _is_identish(surf: str) -> bool:
    s = (surf or "").strip().lstrip("Ġ▁")
    if not s or _is_junk_surface(s):
        return False
    return bool(re.search(r"[A-Za-z_\u4e00-\u9fff]", s))


def _compact_indexed_tokens(
    tokens: list[str],
    *,
    answer_start: int,
    max_pre_idents: int = 180,
    mid_ctx: int = 40,
) -> list[dict[str, Any]]:
    """Keep idents in PRE + window around MID; indices stay global."""
    n = len(tokens)
    ans = max(0, min(int(answer_start), n))
    mid_lo = max(0, ans - mid_ctx)
    keep: set[int] = set(range(mid_lo, n))
    pre_idents: list[int] = []
    for i in range(0, ans):
        if _is_identish(_surface(tokens[i])):
            pre_idents.append(i)
    if len(pre_idents) > max_pre_idents:
        pre_idents = pre_idents[-max_pre_idents:]
    keep.update(pre_idents)
    return [{"i": i, "t": _surface(tokens[i])} for i in sorted(keep)]


def build_auto_annotate_messages(
    *,
    train_tokens: list[str],
    probe_src_token: str,
    probe_dst_token: str,
    language: str | None = None,
    max_edges: int = 8,
    train_answer_start: int = 0,
    train_mid_override: str | None = None,
    sample_id: int | None = None,
    sample_uid: str | None = None,
    probe_tokens: list[str] | None = None,
    probe_answer_start: int | None = None,
    probe_focus_src: int | None = None,
    probe_focus_dst: int | None = None,
    probe_mid_override: str | None = None,
    probe_fim_view: str | None = None,
) -> list[dict[str, str]]:
    lang = (language or "code").strip() or "code"
    train_view = build_fim_marked_view(
        train_tokens,
        answer_start=train_answer_start,
        mid_override=train_mid_override,
        include_indexed=False,
    )
    indexed = _compact_indexed_tokens(
        train_tokens, answer_start=int(train_answer_start),
    )
    sid = f"train_sample_id={sample_id}" if sample_id is not None else "this train sample"
    if sample_uid:
        sid += f" uid={sample_uid}"

    src_tok = (probe_src_token or "").strip() or "·"
    dst_tok = (probe_dst_token or "").strip() or "·"
    short_dst = len(dst_tok.strip()) <= 2

    # Build probe FIM with SOURCE/TARGET marks when tokens are available.
    probe_block = (probe_fim_view or "").strip()
    if not probe_block and probe_tokens:
        ans = int(probe_answer_start) if probe_answer_start is not None else 0
        if ans <= 0:
            ans = max(1, (2 * len(probe_tokens)) // 3)
        probe_built = build_fim_marked_view(
            [str(t) for t in probe_tokens],
            answer_start=ans,
            mark_src=probe_focus_src,
            mark_dst=probe_focus_dst,
            src_role="SOURCE",
            dst_role="TARGET",
            mid_override=probe_mid_override,
            include_indexed=False,
        )
        probe_block = str(probe_built["fim_view"])

    if not probe_block:
        probe_block = (
            f"(No full probe FIM provided.) FOCUS tokens only: "
            f"SOURCE {src_tok!r} → TARGET {dst_tok!r}\n"
        )

    short_note = ""
    if short_dst:
        short_note = (
            f"\nNote: probe TARGET surface {dst_tok!r} is a short subword "
            "(often the first piece of an identifier). Do NOT match other tokens "
            "just because they share that letter. Recover the full identifier from "
            "neighboring tokens in the probe FIM, then find the analogous name / "
            "binding / API use on the train sample.\n"
        )

    system = (
        f"You are helping continue-train a {lang} code LLM with a saliency objective.\n\n"
        "Why we annotate\n"
        "--------------\n"
        "Our training loss is CE + λ · saliency_loss on directed token edges.\n"
        "For each annotated edge s→t, saliency_loss pushes the model to attribute "
        "more contribution / attention from source token s when predicting or "
        "representing target token t (and away from unrelated causal tokens).\n"
        "Annotations are training signals for useful associations — NOT nearby "
        "keyword dumps, NOT AST taxonomy labels.\n\n"
        "Focus association (from probe / test attribution)\n"
        "------------------------------------------------\n"
        "The user selected a FOCUS on a *probe/test* example "
        f"(SOURCE {src_tok!r} → TARGET {dst_tok!r}).\n"
        "Read the probe FIM with SOURCE/TARGET marked and understand the "
        "*mechanism* (copy a name, bind a type, route an API, close a bracket, …).\n"
        f"{short_note}"
        "Propose a small set of directed edges on the *train* sample that teach "
        "the same mechanism. Probe indices are invalid on train.\n\n"
        f"Annotate THIS train sample: {sid}\n"
        "User message: (1) probe FIM (2) train FIM (3) compact indexed tokens "
        "(identifier-like PRE tokens + window around MID). "
        "Every edge src/dst must be from that list.\n"
        f"Every edge subtype must be \"{ROUTE_SUBTYPE}\".\n\n"
        "Return JSON only:\n"
        f'{{ "edges": [ {{"src": <int>, "dst": <int>, "subtype": "{ROUTE_SUBTYPE}", '
        '"reason": <one short clause naming the concrete link>}} ] }}\n'
        "Hard rules (precision ≫ recall):\n"
        "- Prefer: same-name / identifier-span copy, def→use, type→name, "
        "callee→'(', API/string routing, bracket open→close — matching the "
        "probe focus mechanism.\n"
        "- Each reason must name a concrete link. Vague reasons are invalid.\n"
        "- Forbid edges that only share a letter/substring with no shared "
        "identifier/binding/call/dataflow role "
        "(e.g. random Exist/IP/! → first letter of a variable).\n"
        "- Forbid ChatML/FIM markers, whitespace-only, or punctuation-only "
        "as src or dst.\n"
        "- If dst is in MID/completion, require src < dst (causal).\n"
        f"- At most {max(1, int(max_edges))} edges; 1–3 good edges beat 8 junk; "
        "empty is better than junk.\n"
        "- Indices must appear in the compact list "
        f"(valid range 0..{max(0, len(train_tokens) - 1)}).\n"
    )

    user = (
        "=== PROBE / TEST sample (context only; SOURCE/TARGET marked) ===\n"
        f"{probe_block}\n\n"
        "=== TRAIN sample to annotate (PRE + MID + SUF) ===\n"
        f"{train_view['fim_view']}\n\n"
        "=== Compact indexed tokens of TRAIN "
        "(idents in PRE + window around MID; use only these indices) ===\n"
        f"{json.dumps(indexed, ensure_ascii=False)}\n"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _parse_edges_json(text: str) -> list[dict[str, Any]]:
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
            raise
        parsed = json.loads(text[start : end + 1])

    if isinstance(parsed, dict):
        edges = parsed.get("edges") or parsed.get("pairs") or parsed.get("annotations") or []
    elif isinstance(parsed, list):
        edges = parsed
    else:
        edges = []
    if not isinstance(edges, list):
        return []

    out: list[dict[str, Any]] = []
    for e in edges:
        if not isinstance(e, dict):
            continue
        try:
            if "src" in e and "dst" in e:
                src, dst = int(e["src"]), int(e["dst"])
            elif "token_i_idx" in e and "token_j_idx" in e:
                src, dst = int(e["token_i_idx"]), int(e["token_j_idx"])
            elif "i" in e and "j" in e:
                src, dst = int(e["i"]), int(e["j"])
            else:
                continue
        except (TypeError, ValueError):
            continue
        reason = str(e.get("reason") or "")
        out.append({"src": src, "dst": dst, "subtype": ROUTE_SUBTYPE, "reason": reason})
    return out


def call_llm_auto_annotate(
    *,
    tokens: list[str],
    probe_src_token: str,
    probe_dst_token: str,
    language: str | None = None,
    max_edges: int = 8,
    answer_start: int = 0,
    mid_override: str | None = None,
    sample_id: int | None = None,
    sample_uid: str | None = None,
    labels: list[int] | None = None,
    probe_tokens: list[str] | None = None,
    probe_answer_start: int | None = None,
    probe_focus_src: int | None = None,
    probe_focus_dst: int | None = None,
    probe_mid_override: str | None = None,
    probe_fim_view: str | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return (edges on train sample, raw_model_text)."""
    backend = _annotate_backend()
    default_model = "qwen3.8-max" if backend == "api" else "qwen3-8b"
    model = (
        _env("ANNOTATE_MODEL")
        or _env("EIF_ANNOTATE_MODEL")
        or _env("OPENAI_MODEL")
        or default_model
    )
    max_tokens = int(_env("ANNOTATE_MAX_TOKENS", "2048") or "2048")
    max_edges = max(1, min(32, int(max_edges)))
    ans = int(answer_start)
    if ans <= 0 and labels is not None:
        ans = _answer_start_from_labels(labels, len(tokens))

    messages = build_auto_annotate_messages(
        train_tokens=tokens,
        probe_src_token=probe_src_token,
        probe_dst_token=probe_dst_token,
        language=language,
        max_edges=max_edges,
        train_answer_start=ans,
        train_mid_override=mid_override,
        sample_id=sample_id,
        sample_uid=sample_uid,
        probe_tokens=probe_tokens,
        probe_answer_start=probe_answer_start,
        probe_focus_src=probe_focus_src,
        probe_focus_dst=probe_focus_dst,
        probe_mid_override=probe_mid_override,
        probe_fim_view=probe_fim_view,
    )
    sys_chars = len(messages[0]["content"]) if messages else 0
    user_chars = len(messages[1]["content"]) if len(messages) > 1 else 0
    print(
        f"[auto-annotate] calling LLM backend={backend} model={model} "
        f"max_tokens={max_tokens} prompt_chars=system:{sys_chars}+user:{user_chars} "
        f"train_tokens={len(tokens)} probe_tokens={len(probe_tokens) if probe_tokens else 0}",
        flush=True,
    )
    t0 = time.perf_counter()
    client = _build_client()
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": max_tokens,
    }
    extra = _extra_body()
    if extra:
        kwargs["extra_body"] = extra

    try:
        resp = client.chat.completions.create(
            **kwargs,
            response_format={"type": "json_object"},
        )
    except Exception as exc:
        print(
            f"[auto-annotate] json_object response_format failed ({exc}); retry plain",
            flush=True,
        )
        resp = client.chat.completions.create(**kwargs)
    elapsed = time.perf_counter() - t0
    raw = ""
    try:
        raw = resp.choices[0].message.content or ""
    except Exception:
        raw = str(resp)
    edges = _parse_edges_json(raw)
    n = len(tokens)
    cleaned: list[dict[str, Any]] = []
    seen: set[tuple[int, int, str]] = set()
    dropped = 0
    dropped_reasons: list[str] = []
    for e in edges:
        src, dst, sub = int(e["src"]), int(e["dst"]), str(e["subtype"])
        if src == dst:
            dropped += 1
            dropped_reasons.append(f"{src}→{dst}:same")
            continue
        if not (0 <= src < n and 0 <= dst < n):
            dropped += 1
            dropped_reasons.append(f"{src}→{dst}:oor")
            continue
        if ans > 0 and dst >= ans and not (src < dst):
            dropped += 1
            dropped_reasons.append(f"{src}→{dst}:noncausal")
            continue
        src_s = _surface(tokens[src])
        dst_s = _surface(tokens[dst])
        if _is_junk_surface(src_s) or _is_junk_surface(dst_s):
            dropped += 1
            dropped_reasons.append(f"{src}→{dst}:junk({src_s!r}/{dst_s!r})")
            continue
        key = (src, dst, sub)
        if key in seen:
            dropped += 1
            dropped_reasons.append(f"{src}→{dst}:dup")
            continue
        seen.add(key)
        cleaned.append(e)
        if len(cleaned) >= max(1, int(max_edges)):
            break
    print(
        f"[auto-annotate] LLM returned in {elapsed:.1f}s "
        f"parsed={len(edges)} kept={len(cleaned)} dropped={dropped}"
        + (f" ({', '.join(dropped_reasons[:8])})" if dropped_reasons else ""),
        flush=True,
    )
    return cleaned, raw
