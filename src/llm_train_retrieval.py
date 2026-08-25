"""LLM-based train-sample retrieval for a whole test FIM.

Two-stage analysis (no attention-edge / annotation design):
  1) summarize the gold <MID> completion's code pattern
  2) describe ideal train-sample traits and emit dual boolean expressions
     (gold_expr on response + context_expr on prompt); retrieval requires both.

Uses OpenAI-compatible API from repo-root ``eif_api.env``.
The server evaluates expressions and returns matching rows.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

_IM_START_RE = re.compile(r"<\|im_start\|>", re.IGNORECASE)
_IM_END_RE = re.compile(r"<\|redacted_im_end\|>", re.IGNORECASE)
_THINKING_RE = re.compile(
    r"<\s*redacted_thinking\s*>.*?</\s*redacted_thinking\s*>",
    re.DOTALL | re.IGNORECASE,
)
_USER_BLOCK_RE = re.compile(
    r"<\|im_start\|>\s*user\s*\n(.*?)<\|redacted_im_end\|>",
    re.DOTALL | re.IGNORECASE,
)
_TASK_START_MARKERS = (
    "This is a go programming task",
    "### Given Task:",
    "Below is the package path:",
    "And here is the function you are asked to complete:",
)


def _strip_thinking_blocks(text: str) -> str:
    out = _THINKING_RE.sub("", text or "")
    # Unclosed thinking at start
    out = re.sub(
        r"^\s*<\s*redacted_thinking\s*>[\s\S]*$",
        "",
        out,
        count=1,
        flags=re.IGNORECASE,
    )
    return out


def extract_fim_problem_surface(raw: str) -> str:
    """Strip ChatML / assistant wrappers; keep only the FIM task body (题面)."""
    text = (raw or "").replace("\r\n", "\n")
    text = _strip_thinking_blocks(text)

    user_blocks = [m.group(1).strip() for m in _USER_BLOCK_RE.finditer(text)]
    if user_blocks:
        body = user_blocks[-1]
    else:
        body = text
        body = _IM_START_RE.sub("", body)
        body = _IM_END_RE.sub("", body)
        body = re.sub(
            r"^\s*(system|user|assistant)\s*\n+",
            "",
            body,
            flags=re.IGNORECASE | re.MULTILINE,
        )

    for cut in (
        "### Response:",
        "<|im_start|>assistant",
        "\n<|im_start|>assistant",
    ):
        idx = body.find(cut)
        if idx >= 0:
            body = body[:idx]

    body = re.sub(r"\n/no_think\s*$", "", body.strip(), flags=re.IGNORECASE)

    # Prefer task section if present inside a longer blob.
    for marker in _TASK_START_MARKERS:
        pos = body.find(marker)
        if pos >= 0:
            body = body[pos:]
            break

    return body.strip()


def clean_gold_mid_completion(raw: str) -> str:
    """Gold = <MID> fill only; strip thinking / ChatML noise."""
    text = _strip_thinking_blocks(raw or "")
    text = _IM_START_RE.sub("", text)
    text = _IM_END_RE.sub("", text)
    text = re.sub(r"^\s*(assistant|user|system)\s*\n+", "", text, flags=re.IGNORECASE)
    return text.strip()


def prepare_llm_train_query(fim_prompt: str, gold_completion: str) -> dict[str, str]:
    raw_prompt = (fim_prompt or "").strip()
    raw_gold = (gold_completion or "").strip()
    surface = extract_fim_problem_surface(raw_prompt)
    gold = clean_gold_mid_completion(raw_gold)
    if not surface:
        surface = raw_prompt
    if not gold:
        gold = raw_gold
    return {
        "fim_problem_surface": surface,
        "gold_mid_completion": gold,
        "raw_fim_prompt": raw_prompt,
        "raw_gold_completion": raw_gold,
    }


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _build_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("pip install openai") from exc

    api_key = (
        _env("DASHSCOPE_API_KEY")
        or _env("OPENAI_API_KEY")
        or _env("ANNOTATE_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Set DASHSCOPE_API_KEY or OPENAI_API_KEY in eif_api.env"
        )
    base_url = (
        _env("OPENAI_BASE_URL")
        or _env("ANNOTATE_BASE_URL")
        or "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return OpenAI(api_key=api_key, base_url=base_url)


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


def _strip_code_fence(text: str) -> str:
    t = (text or "").strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return t


def _parse_json_object(text: str) -> dict[str, Any]:
    raw = _strip_code_fence(text)
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(raw[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    raise ValueError("LLM response is not valid JSON object")


def build_llm_train_retrieve_messages(
    *,
    fim_prompt: str,
    gold_completion: str,
    example_expression: str | None = None,
) -> list[dict[str, str]]:
    """Assemble system + user messages: pattern summary then dual corpus search."""
    prepared = prepare_llm_train_query(fim_prompt, gold_completion)
    problem = prepared["fim_problem_surface"]
    gold = prepared["gold_mid_completion"]
    example_gold = example_expression or (
        '("if err :=" OR "if err !=") AND "err != nil" AND "return" AND "Wrap(err"'
    )
    example_ctx = (
        '"func (" AND ("HandleEvent" OR "Execute") AND ("transdsl" OR "TransactionInfo")'
    )
    system = (
        "你是 Go 代码补全训练数据检索助手。\n"
        "用户会给出一条 **Go FIM 测试题**（Fill-in-the-Middle，中间缺失处标记为 <MID>）"
        "及其 gold 补全（仅 <MID> 处应填写的代码片段）。\n"
        "题面已去除 ChatML 对话包装（不是 system/user/assistant 聊天消息）。\n"
        "本阶段只做「相关代码模式 → 检索训练样本」，不要设计、不要提及 "
        "attention_edges、标注、saliency、subtype 或 source→target 边。\n\n"
        "请分两段思考，并输出**严格 JSON**（不要 markdown 包裹）：\n"
        "第一段：概括这条 FIM 的 gold 回答是什么样的代码格式/模式。\n"
        "第二段：为了让模型学会这种模式，理想训练样本应具备哪些特征；"
        "并据此给出 2-5 组检索式，从宽到窄。"
        "每组必须同时约束 **gold（补全）** 与 **context（题面/前后文）**，"
        "检索时两边都要命中才会入选。\n\n"
        "JSON 字段：\n"
        "{\n"
        '  "gold_pattern_summary": "概括 gold 的代码格式与模式（中文，2-5句）",\n'
        '  "ideal_train_sample_traits": ["理想训练样本特征1", "特征2"],\n'
        '  "corpus_search_expressions": [\n'
        "    {\n"
        '      "name": "简短英文名",\n'
        '      "gold_expr": "只描述 <MID>/response 补全形态的布尔子串式",\n'
        '      "context_expr": "只描述题面/前后文场景的布尔子串式",\n'
        '      "why": "为何这对表达式能同时约束题型与补全内容、宽还是窄"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "表达式语法（gold_expr / context_expr 相同）：\n"
        '- 字面量用双引号，如 "err != nil {"\n'
        '- OR 连接备选，如 ("if err :=" OR "if err !=")\n'
        '- AND 连接必须同时出现，如 A AND B AND C\n'
        "- 不要给整条 AND 链再包一层最外层括号；括号只用于 OR 分组。\n"
        f"- gold_expr 示例（只谈补全）：{example_gold}\n"
        f"- context_expr 示例（只谈场景）：{example_ctx}\n"
        "硬性规则：\n"
        "- gold_expr 必须能在训练行的 response/label（补全）里单独命中；"
        "不要把只出现在上下文里的符号写进 gold_expr。\n"
        "- context_expr 必须能在训练行的 prompt/input（题面）里单独命中；"
        "可用包路径、receiver、邻近 API、FIM 前后片段特征等。\n"
        "- 禁止只用一个 expression 字段；必须同时给出 gold_expr 与 context_expr。\n"
        "- 给出 2-5 组，按从宽到窄排序。"
    )
    user = (
        "【题目类型】Go 代码 FIM 补全测试题（非对话；已去除 ChatML 包装）\n\n"
        "【题面】\n"
        f"{problem}\n\n"
        "【Gold】<MID> 处应填写的正确代码：\n"
        f"{gold}\n\n"
        "请按两段回答：\n"
        "1）这条 gold 是什么样的代码格式/模式？\n"
        "2）为了训练模型学会该模式，理想训练样本应有哪些特征？"
        "给出 2-5 组从宽到窄的检索式，每组都要有 gold_expr（约束补全）"
        "和 context_expr（约束题面场景）。不要讨论标注。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _strip_outer_parens(expr: str) -> str:
    """Strip layers of balanced wrapping parentheses (ignoring parens inside quotes)."""
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
    """Split on `` AND `` / `` OR `` only outside parentheses and string literals."""
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


def _eval_literal(text: str, lit: str) -> bool:
    s = lit.strip().strip('"').strip("'")
    return bool(s) and s in text


def _eval_or_term(text: str, term: str) -> bool:
    t = _strip_outer_parens(term)
    # Parenthesized AND-group: recurse as full expression.
    and_parts = _split_top_level(t, "AND")
    if len(and_parts) > 1:
        return all(_eval_or_term(text, p) for p in and_parts)
    or_parts = _split_top_level(t, "OR")
    if len(or_parts) > 1:
        return any(_eval_or_term(text, p) for p in or_parts)
    return _eval_literal(text, t)


def eval_boolean_expression(text: str, expression: str) -> bool:
    """Evaluate ``(A OR B) AND C`` style substring expression.

    Supports an outer wrapping ``(...)`` around the whole AND-chain (common in
    LLM outputs). AND/OR / parentheses inside ``"..."`` literals are ignored
    for structure (so ``"func ("`` does not break parsing).
    """
    expr = _strip_outer_parens(expression or "")
    if not expr:
        return False
    parts = _split_top_level(expr, "AND")
    if not parts:
        return False
    return all(_eval_or_term(text, p) for p in parts)

def _row_context_text(row: dict[str, Any]) -> str:
    for key in ("prompt", "input", "query"):
        v = row.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


def _row_gold_text(row: dict[str, Any]) -> str:
    for key in ("response", "label", "output", "gold"):
        v = row.get(key)
        if isinstance(v, str) and v:
            return v
    return ""


def _sample_haystack(row: dict[str, Any]) -> str:
    """Legacy combined haystack (context + gold). Prefer dual-field match."""
    chunks = [c for c in (_row_context_text(row), _row_gold_text(row)) if c]
    if not chunks and isinstance(row.get("input_ids"), list):
        chunks.append(f"compact_n_tokens={len(row['input_ids'])}")
    return "\n".join(chunks)


def normalize_dual_expr_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one corpus_search_expressions entry to gold_expr + context_expr.

    Legacy single ``expression`` is treated as gold_expr only (context empty →
    that side is skipped so old LLM replies still searchable, but new prompts
    require both).
    """
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "expr").strip() or "expr"
    why = str(item.get("why") or "").strip()
    gold_expr = str(
        item.get("gold_expr")
        or item.get("gold_expression")
        or item.get("expression")
        or ""
    ).strip()
    context_expr = str(
        item.get("context_expr")
        or item.get("context_expression")
        or ""
    ).strip()
    if not gold_expr and not context_expr:
        return None
    return {
        "name": name,
        "gold_expr": gold_expr,
        "context_expr": context_expr,
        # Keep a combined display string for older UI snippets.
        "expression": (
            f"[gold] {gold_expr}"
            + (f"  AND  [context] {context_expr}" if context_expr else "")
        ).strip(),
        "why": why,
    }


def row_matches_dual_expr(
    row: dict[str, Any],
    *,
    gold_expr: str = "",
    context_expr: str = "",
    legacy_expression: str = "",
) -> bool:
    """Match gold on response/label and context on prompt/input.

    If only ``legacy_expression`` is set (no dual fields), fall back to the
    old combined haystack behavior.
    """
    gold_expr = (gold_expr or "").strip()
    context_expr = (context_expr or "").strip()
    legacy_expression = (legacy_expression or "").strip()

    if gold_expr or context_expr:
        if gold_expr:
            gold_text = _row_gold_text(row)
            if not gold_text or not eval_boolean_expression(gold_text, gold_expr):
                return False
        if context_expr:
            ctx_text = _row_context_text(row)
            if not ctx_text or not eval_boolean_expression(ctx_text, context_expr):
                return False
        return bool(gold_expr or context_expr)

    if legacy_expression:
        return eval_boolean_expression(_sample_haystack(row), legacy_expression)
    return False


def search_corpus_jsonl(
    corpus_path: str,
    expression: str = "",
    *,
    gold_expr: str = "",
    context_expr: str = "",
    top_k: int = 20,
    max_scan: int | None = None,
    stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    path = Path(corpus_path)
    if not path.is_file():
        raise FileNotFoundError(f"corpus not found: {corpus_path}")
    hits: list[dict[str, Any]] = []
    stop_reason = "eof"
    scanned = 0
    with path.open(encoding="utf-8") as fh:
        for line_idx, line in enumerate(fh):
            scanned = line_idx + 1
            if max_scan is not None and line_idx >= max_scan:
                stop_reason = "max_scan"
                break
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            if not row_matches_dual_expr(
                row,
                gold_expr=gold_expr,
                context_expr=context_expr,
                legacy_expression=expression,
            ):
                continue
            ctx = _row_context_text(row)
            resp = _row_gold_text(row)
            hits.append({
                "line": line_idx,
                "task_id": row.get("task_id"),
                "prompt_preview": (ctx[:280] + "…") if len(ctx) > 280 else ctx,
                "response_preview": (resp[:200] + "…") if len(resp) > 200 else resp,
            })
            if len(hits) >= top_k:
                stop_reason = "top_k"
                break
    if stats is not None:
        stats["scanned_lines"] = scanned
        stats["stop_reason"] = stop_reason
        stats["top_k"] = top_k
        stats["cached"] = False
        stats["mode"] = (
            "dual"
            if (gold_expr or "").strip() or (context_expr or "").strip()
            else "legacy"
        )
    return hits


def search_local_train_bank(
    expression: str = "",
    *,
    gold_expr: str = "",
    context_expr: str = "",
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Search compact ``EIF_TRAIN_DATA`` rows (smoke bank) by line index."""
    train_path = _env("EIF_TRAIN_DATA") or _env("ANNOTATION_TRAIN_DATA")
    if not train_path or not Path(train_path).is_file():
        return []
    hits: list[dict[str, Any]] = []
    with Path(train_path).open(encoding="utf-8") as fh:
        for train_idx, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            edges = row.get("attention_edges") or row.get("edges") or []
            n_edges = len(edges) if isinstance(edges, list) else 0
            hay = _sample_haystack(row)
            if not hay.strip() and isinstance(row.get("input_ids"), list):
                continue
            if not row_matches_dual_expr(
                row,
                gold_expr=gold_expr,
                context_expr=context_expr,
                legacy_expression=expression,
            ):
                continue
            subtypes: set[str] = set()
            if isinstance(edges, list):
                for e in edges:
                    if isinstance(e, dict):
                        st = str(e.get("subtype") or "").strip()
                        if st:
                            subtypes.add(st)
            hits.append({
                "train_sample_id": train_idx,
                "n_attention_edges": n_edges,
                "edge_subtypes": sorted(subtypes)[:12],
                "preview": hay[:320] + ("…" if len(hay) > 320 else ""),
            })
            if len(hits) >= top_k:
                break
    return hits


def call_llm_train_retrieve(
    *,
    fim_prompt: str,
    gold_completion: str,
    model: str | None = None,
    max_tokens: int | None = None,
) -> dict[str, Any]:
    client = _build_openai_client()
    model_name = model or _env("ANNOTATE_MODEL") or _env("LLM_RETRIEVE_MODEL") or "qwen-plus"
    mt = max_tokens or int(_env("ANNOTATE_MAX_TOKENS") or "4096")
    messages = build_llm_train_retrieve_messages(
        fim_prompt=fim_prompt,
        gold_completion=gold_completion,
    )
    kwargs: dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": 0.2,
        "max_tokens": mt,
    }
    extra = _extra_body()
    if extra:
        kwargs["extra_body"] = extra
    print(
        f"[llm-train] model={model_name} surface_chars={len(fim_prompt)} "
        f"gold_chars={len(gold_completion)}",
        flush=True,
    )
    resp = client.chat.completions.create(**kwargs)
    raw = (resp.choices[0].message.content or "").strip()
    parsed = _parse_json_object(raw)
    # Normalize dual expressions in-place for downstream + UI.
    raw_exprs = parsed.get("corpus_search_expressions")
    if isinstance(raw_exprs, list):
        normalized: list[dict[str, Any]] = []
        for item in raw_exprs:
            if not isinstance(item, dict):
                continue
            n = normalize_dual_expr_item(item)
            if n:
                normalized.append(n)
        parsed["corpus_search_expressions"] = normalized
    return {
        "model": model_name,
        "raw": raw,
        "analysis": parsed,
        "messages": messages,
    }


def retrieve_llm_train_samples(
    *,
    fim_prompt: str,
    gold_completion: str,
    corpus_path: str | None = None,
    top_k: int = 15,
    max_corpus_scan: int | None = None,
    run_corpus_search: bool = True,
    search_local_bank: bool = True,
) -> dict[str, Any]:
    if not (fim_prompt or "").strip():
        raise ValueError("fim_prompt is required")
    if not (gold_completion or "").strip():
        raise ValueError("gold_completion is required")

    prepared = prepare_llm_train_query(fim_prompt, gold_completion)
    problem = prepared["fim_problem_surface"]
    gold = prepared["gold_mid_completion"]

    llm_out = call_llm_train_retrieve(
        fim_prompt=problem,
        gold_completion=gold,
    )
    analysis = llm_out.get("analysis") or {}
    exprs = analysis.get("corpus_search_expressions") or []
    if not isinstance(exprs, list):
        exprs = []

    corpus = corpus_path or _env("EIF_LLM_TRAIN_CORPUS") or _env("EIF_TRAIN_CORPUS") or ""
    search_results: list[dict[str, Any]] = []

    for item in exprs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "expr")
        gold_expr = str(item.get("gold_expr") or "").strip()
        context_expr = str(item.get("context_expr") or "").strip()
        expression = str(item.get("expression") or "").strip()
        why = str(item.get("why") or "")
        if not gold_expr and not context_expr and not expression:
            continue
        entry: dict[str, Any] = {
            "name": name,
            "gold_expr": gold_expr,
            "context_expr": context_expr,
            "expression": expression or (
                f"[gold] {gold_expr}"
                + (f"  AND  [context] {context_expr}" if context_expr else "")
            ),
            "why": why,
            "corpus_hits": [],
            "local_bank_hits": [],
        }
        if run_corpus_search and corpus and Path(corpus).is_file():
            try:
                entry["corpus_hits"] = search_corpus_jsonl(
                    corpus,
                    expression="" if (gold_expr or context_expr) else expression,
                    gold_expr=gold_expr,
                    context_expr=context_expr,
                    top_k=top_k,
                    max_scan=max_corpus_scan,
                )
                entry["corpus_path"] = corpus
            except Exception as exc:
                entry["corpus_error"] = str(exc)
        if search_local_bank:
            entry["local_bank_hits"] = search_local_train_bank(
                expression="" if (gold_expr or context_expr) else expression,
                gold_expr=gold_expr,
                context_expr=context_expr,
                top_k=top_k,
            )
        search_results.append(entry)

    return {
        "status": "success",
        "query": {
            "fim_prompt_chars": len(fim_prompt),
            "fim_surface_chars": len(prepared["fim_problem_surface"]),
            "gold_completion_chars": len(gold_completion),
            "gold_preview": prepared["gold_mid_completion"][:400],
            "fim_surface_preview": prepared["fim_problem_surface"][:600],
            "stripped_chatml": prepared["fim_problem_surface"] != prepared["raw_fim_prompt"],
        },
        "prepared": prepared,
        "llm": llm_out,
        "analysis": analysis,
        "search_results": search_results,
        "corpus_path": corpus or None,
        "local_bank_path": _env("EIF_TRAIN_DATA") or _env("ANNOTATION_TRAIN_DATA") or None,
    }
