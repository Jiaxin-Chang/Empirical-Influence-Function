"""LLM-based train-sample retrieval for a whole test FIM.

Two-stage analysis (no attention-edge / annotation design):
  1) summarize the gold <MID> completion's code pattern
  2) describe ideal train-sample traits and emit boolean corpus search expressions

Search matches each expression against the full training row text
(prompt/input + response/label).

Uses OpenAI-compatible API from repo-root ``eif_api.env``.
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


def _annotate_backend() -> str:
    raw = _env("ANNOTATE_BACKEND", "").lower()
    if raw in ("api", "dashscope", "remote", "cloud"):
        return "api"
    if raw in ("vllm", "local"):
        return "vllm"
    if _env("DASHSCOPE_API_KEY"):
        return "api"
    return "vllm"


def _build_openai_client():
    """OpenAI-compatible client for DashScope API or local vLLM."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("pip install openai") from exc

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
        kwargs["base_url"] = (
            _env("OPENAI_BASE_URL")
            or _env("ANNOTATE_BASE_URL")
            or "http://127.0.0.1:8000/v1"
        )
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
    """Assemble system + user messages: pattern summary then corpus search."""
    prepared = prepare_llm_train_query(fim_prompt, gold_completion)
    problem = prepared["fim_problem_surface"]
    gold = prepared["gold_mid_completion"]
    example_expr = example_expression or (
        '("if err :=" OR "if err !=") AND "err != nil {" AND "return" AND "Wrap(err"'
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
        "并据此给出 2-5 条布尔检索式，从宽到窄，用于在大规模 Go 训练 JSONL"
        "（每行完整 prompt+response 文本）里找出同类样本。\n\n"
        "JSON 字段：\n"
        "{\n"
        '  "gold_pattern_summary": "概括 gold 的代码格式与模式（中文，2-5句）",\n'
        '  "ideal_train_sample_traits": ["理想训练样本特征1", "特征2"],\n'
        '  "corpus_search_expressions": [\n'
        "    {\n"
        '      "name": "简短英文名",\n'
        '      "expression": "布尔子串表达式",\n'
        '      "why": "这条式子对应哪种代码模式、宽还是窄"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "corpus_search_expressions 的 expression 语法：\n"
        '- 字面量用双引号，如 "err != nil {"\n'
        '- OR 连接备选，如 ("if err :=" OR "if err !=")\n'
        '- AND 连接必须同时出现，如 A AND B AND C\n'
        "- 不要给整条 AND 链再包一层最外层括号；括号只用于 OR 分组。\n"
        f"- 示例：{example_expr}\n"
        "必须给出 2-5 条 expression，按从宽到窄排序；"
        "检索会在训练行的完整文本（题面 prompt + 补全 response）上匹配；"
        "只检索代码文本模式，不要检索标注字段。"
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
        "给出 2-5 条从宽到窄的布尔表达式，用于在训练集完整文本里找同类代码样本。"
        "不要讨论标注。"
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
    """Full training-row text: prompt/input + response/label."""
    chunks = [c for c in (_row_context_text(row), _row_gold_text(row)) if c]
    if not chunks and isinstance(row.get("input_ids"), list):
        chunks.append(f"compact_n_tokens={len(row['input_ids'])}")
    return "\n".join(chunks)


def normalize_expr_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one corpus_search_expressions entry to a single expression."""
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "expr").strip() or "expr"
    why = str(item.get("why") or "").strip()
    expression = str(
        item.get("expression")
        or item.get("gold_expr")
        or item.get("gold_expression")
        or ""
    ).strip()
    if not expression:
        return None
    return {
        "name": name,
        "expression": expression,
        "why": why,
    }


def search_corpus_jsonl(
    corpus_path: str,
    expression: str,
    *,
    top_k: int = 20,
    max_scan: int | None = None,
    stats: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Match ``expression`` against full row text (prompt + response)."""
    path = Path(corpus_path)
    if not path.is_file():
        raise FileNotFoundError(f"corpus not found: {corpus_path}")
    expr = (expression or "").strip()
    if not expr:
        return []
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
            hay = _sample_haystack(row)
            if not eval_boolean_expression(hay, expr):
                continue
            ctx = _row_context_text(row)
            resp = _row_gold_text(row)
            # Prefer gold: if response matches, keep original MID (no rewrite).
            if eval_boolean_expression(resp, expr):
                match_region = "gold"
            elif eval_boolean_expression(ctx, expr):
                match_region = "context"
            else:
                match_region = "cross"
            hits.append({
                "line": line_idx,
                "task_id": row.get("task_id"),
                "match_region": match_region,
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
        stats["mode"] = "full_prompt"
    return hits


def search_local_train_bank(
    expression: str,
    *,
    top_k: int = 20,
) -> list[dict[str, Any]]:
    """Search compact ``EIF_TRAIN_DATA`` rows (smoke bank) by line index."""
    train_path = _env("EIF_TRAIN_DATA") or _env("ANNOTATION_TRAIN_DATA")
    if not train_path or not Path(train_path).is_file():
        return []
    expr = (expression or "").strip()
    if not expr:
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
            if not eval_boolean_expression(hay, expr):
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
    raw_exprs = parsed.get("corpus_search_expressions")
    if isinstance(raw_exprs, list):
        normalized: list[dict[str, Any]] = []
        for item in raw_exprs:
            if not isinstance(item, dict):
                continue
            n = normalize_expr_item(item)
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
        expression = str(item.get("expression") or "").strip()
        why = str(item.get("why") or "")
        if not expression:
            continue
        entry: dict[str, Any] = {
            "name": name,
            "expression": expression,
            "why": why,
            "corpus_hits": [],
            "local_bank_hits": [],
        }
        if run_corpus_search and corpus and Path(corpus).is_file():
            try:
                entry["corpus_hits"] = search_corpus_jsonl(
                    corpus,
                    expression,
                    top_k=top_k,
                    max_scan=max_corpus_scan,
                )
                entry["corpus_path"] = corpus
            except Exception as exc:
                entry["corpus_error"] = str(exc)
        if search_local_bank:
            entry["local_bank_hits"] = search_local_train_bank(expression, top_k=top_k)
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
