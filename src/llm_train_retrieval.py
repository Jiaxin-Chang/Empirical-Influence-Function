"""LLM Boolean train-sample retrieval for a whole test FIM.

Boolean substring queries only. Structured semantic retrieval lives in
``src/llm_semantic_retrieval.py`` and uses ``EIF_LLM_SEMANTIC_CORPUS``.

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
    "This is a c/cpp programming task",
    "This is a cpp programming task",
    "You are a Java code completion assistant.",
    "Fill the [MASK] in the Java function",
    "* Incomplete Code:",
    "### Given Task:",
    "Below is the package path:",
    "Here is the file path where the current code is located.",
    "And here is the function you are asked to complete:",
    "And here is the code snippet you are asked to complete:",
)

_LANG_PROFILES: dict[str, dict[str, str]] = {
    "go": {
        "name": "Go",
        "example": (
            '("if err :=" OR "if err !=") AND "err != nil {" AND "return" AND "Wrap(err"'
        ),
        "boolean_extra": (
            "Go copy_from_context：宽式子不要把 gold 独有标识符当 AND 必选项；"
            "至少一条 match_in=response。"
        ),
    },
    "cpp": {
        "name": "C/C++",
        "example": (
            '("TEST_F(" OR "EXPECT_EQ(") AND "SCM_" AND "VOS_OK"'
        ),
        "boolean_extra": (
            "C/C++ copy_from_context：宽式子不要把 gold 独有宏/标识符当 AND 必选项；"
            "至少一条 match_in=response。"
        ),
    },
    "c": {
        "name": "C/C++",
        "example": (
            '("TEST_F(" OR "EXPECT_EQ(") AND "SCM_" AND "VOS_OK"'
        ),
        "boolean_extra": (
            "C/C++ copy_from_context：宽式子不要把 gold 独有宏/标识符当 AND 必选项；"
            "至少一条 match_in=response。"
        ),
    },
    "java": {
        "name": "Java",
        "example": (
            '("yFormat.format" OR "xFormat.format" OR "numberFormat.format") '
            'AND "else" AND "result["'
        ),
        "boolean_extra": (
            "Java 镜像复用（copy_from_context）Boolean 基线：\n"
            "1) sibling_line 必须引用题面里被抄的那一行（不要编造）。\n"
            "2) 宽式子不要把 gold 独有标识符（如 xFormat、result[1]）当作 AND 必选项；"
            "那些只允许出现在最窄的一条。\n"
            "3) 至少一条 expression 设 match_in=response。\n"
            "4) 禁止只用 result[ 或 Copies. 当区分特征。\n"
            "示例（宽→窄）：\n"
            '  full: "DateFormat" AND ".format(" AND "else" AND "result["\n'
            '  full: ("xFormat" OR "yFormat" OR "zFormat" OR "numberFormat") '
            'AND ".format(" AND "result["\n'
            '  response: ".format(" AND "result["'
        ),
    },
}


def _lang_profile(language: str | None) -> dict[str, str]:
    key = (language or "go").strip().lower().replace("c++", "cpp")
    if key in ("cplusplus", "cxx"):
        key = "cpp"
    return _LANG_PROFILES.get(key, _LANG_PROFILES["go"])


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
    raw = _strip_code_fence(_strip_thinking_blocks(text))
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


_JSON_RETRY_USER = (
    "Your previous reply was not a valid JSON object. "
    "Return STRICT JSON only: no markdown, no commentary, no code fences, "
    "no thinking tags. The entire reply must start with { and end with }."
)


def _json_max_attempts() -> int:
    raw = _env("LLM_JSON_MAX_ATTEMPTS") or "3"
    try:
        n = int(raw)
    except ValueError:
        n = 3
    return max(1, min(n, 8))


def _chat_complete_json(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    extra_body: dict[str, Any] | None = None,
    temperature: float = 0.2,
    max_attempts: int | None = None,
    log_prefix: str = "[llm]",
) -> tuple[str, dict[str, Any]]:
    """Call chat.completions and parse a JSON object; retry on invalid JSON."""
    attempts = _json_max_attempts() if max_attempts is None else max(1, min(int(max_attempts), 8))
    convo = list(messages)
    last_err: Exception | None = None
    last_raw = ""
    for i in range(attempts):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": convo,
            "temperature": temperature if i == 0 else min(0.7, temperature + 0.2 * i),
            "max_tokens": max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        print(f"{log_prefix} json_attempt={i + 1}/{attempts}", flush=True)
        try:
            resp = client.chat.completions.create(**kwargs)
            last_raw = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            last_err = exc
            print(f"{log_prefix} api error attempt={i + 1}/{attempts}: {exc}", flush=True)
            continue
        try:
            parsed = _parse_json_object(last_raw)
            if i:
                print(f"{log_prefix} recovered JSON on attempt={i + 1}", flush=True)
            return last_raw, parsed
        except ValueError as exc:
            last_err = exc
            preview = last_raw.replace("\n", " ")[:240] or "(empty)"
            print(
                f"{log_prefix} invalid JSON attempt={i + 1}/{attempts}: {preview}",
                flush=True,
            )
            convo = list(convo) + [
                {"role": "assistant", "content": last_raw or "(empty)"},
                {"role": "user", "content": _JSON_RETRY_USER},
            ]
    raise ValueError(
        f"LLM response is not valid JSON object after {attempts} attempts"
    ) from last_err


def build_llm_boolean_retrieve_messages(
    *,
    fim_prompt: str,
    gold_completion: str,
    example_expression: str | None = None,
    language: str | None = None,
) -> list[dict[str, str]]:
    """Boolean-only: pattern summary + corpus substring expressions."""
    prepared = prepare_llm_train_query(fim_prompt, gold_completion)
    problem = prepared["fim_problem_surface"]
    gold = prepared["gold_mid_completion"]
    profile = _lang_profile(language)
    lang_name = profile["name"]
    example_expr = example_expression or profile["example"]
    boolean_extra = (profile.get("boolean_extra") or "").strip()
    system = (
        f"你是 {lang_name} 代码补全训练数据检索助手。\n"
        f"用户会给出一条 **{lang_name} FIM 测试题**（Fill-in-the-Middle，"
        "中间缺失处可能标记为 <MID>、<FIM> 或 [MASK]）"
        "及其 gold 补全（仅挖空处应填写的代码片段）。\n"
        "题面已去除 ChatML 对话包装（不是 system/user/assistant 聊天消息）。\n"
        "本阶段只做「相关代码模式 → 布尔子串检索」，不要输出 semantic / domain / "
        "pattern / relations 等结构化语义字段，也不要提及 attention_edges、"
        "标注、saliency、subtype 或 source→target 边。\n\n"
        "请分两段思考，并输出**严格 JSON**（不要 markdown 包裹）：\n"
        "第一段：概括这条 FIM 的 gold 回答是什么样的代码格式/模式。\n"
        "第二段：为了让模型学会这种模式，理想训练样本应具备哪些特征；"
        f"并据此给出 2-5 条布尔检索式，从宽到窄，用于在大规模 {lang_name} 训练 JSONL"
        "（每行完整 prompt+response 文本）里找出同类样本。\n\n"
        "JSON 字段：\n"
        "{\n"
        '  "hole_relation": "copy_from_context | compose_local | other",\n'
        '  "sibling_line": "若是镜像抄写，原样引用题面里被抄的那一行；否则空字符串",\n'
        '  "gold_pattern_summary": "概括 gold 的代码格式与模式（中文，2-5句）",\n'
        '  "ideal_train_sample_traits": ["理想训练样本特征1", "特征2"],\n'
        '  "corpus_search_expressions": [\n'
        "    {\n"
        '      "name": "简短英文名",\n'
        '      "expression": "布尔子串表达式",\n'
        '      "match_in": "full | response | prompt",\n'
        '      "why": "这条式子对应哪种代码模式、宽还是窄"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "先判断挖空与题面的关系 hole_relation：\n"
        "- copy_from_context：gold 是把题面 prefix/suffix 里已经出现的某一行（或对称分支）"
        "改名/换槽位抄过来。例如 suffix 已有 result[2] = this.yFormat.format(y)，"
        "gold 是 result[1] = this.xFormat.format(x)。\n"
        "- compose_local：gold 组合了题面 API，但不是镜像抄写。\n"
        "- other：其它。\n\n"
        "若 hole_relation=copy_from_context：\n"
        "1) sibling_line 必须引用题面里被抄的那一行（不要编造）。\n"
        "2) 检索目标是「同样在做平行槽位/对称分支复用」的训练题，不是 gold 的字面标识符。\n"
        "3) 宽式子不要把 gold 独有标识符当作 AND 必选项；那些只允许出现在最窄的一条。\n"
        "4) 至少一条 expression 设 match_in=response。\n"
        "5) 禁止只用 result[ 或 Copies. 当区分特征。\n"
        + (f"{boolean_extra}\n\n" if boolean_extra else "\n")
        + "corpus_search_expressions 的 expression 语法：\n"
        '- 字面量用双引号，如 "err != nil {"\n'
        '- OR 连接备选，如 ("if err :=" OR "if err !=")\n'
        "- AND 连接必须同时出现，如 A AND B AND C\n"
        "- 不要给整条 AND 链再包一层最外层括号；括号只用于 OR 分组。\n"
        "match_in：full=整行 prompt+response（默认）；response=只匹配补全；prompt=只匹配题面。\n"
        f"- 示例：{example_expr}\n"
        "必须给出 2-5 条 expression，按从宽到窄排序；"
        "只检索代码文本模式，不要检索标注字段。"
    )
    user = (
        f"【题目类型】{lang_name} 代码 FIM 补全测试题（非对话；已去除 ChatML 包装）\n\n"
        "【题面】\n"
        f"{problem}\n\n"
        "【Gold】挖空处应填写的正确代码：\n"
        f"{gold}\n\n"
        "请按两段回答：\n"
        "1）这条 gold 是什么样的代码格式/模式？它是不是在抄题面里已经出现的对称行？\n"
        "2）为了训练模型学会该模式，理想训练样本应有哪些特征？"
        "给出 2-5 条从宽到窄的布尔表达式。copy_from_context 时不要用 gold 字面当宽检索，"
        "并至少一条 match_in=response。"
        "不要输出结构化 semantic 字段，不要讨论标注。"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_llm_train_retrieve_messages(
    *,
    fim_prompt: str,
    gold_completion: str,
    example_expression: str | None = None,
    language: str | None = None,
    mode: str | None = None,
) -> list[dict[str, str]]:
    """Boolean-only. ``mode`` is ignored (kept for old callers)."""
    return build_llm_boolean_retrieve_messages(
        fim_prompt=fim_prompt,
        gold_completion=gold_completion,
        example_expression=example_expression,
        language=language,
    )


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


def _normalize_match_in(value: Any) -> str:
    raw = str(value or "full").strip().lower()
    if raw in ("response", "gold", "label", "completion"):
        return "response"
    if raw in ("prompt", "context", "input"):
        return "prompt"
    return "full"


def _sample_haystack(row: dict[str, Any]) -> str:
    """Full training-row text: prompt/input + response/label."""
    chunks = [c for c in (_row_context_text(row), _row_gold_text(row)) if c]
    if not chunks and isinstance(row.get("input_ids"), list):
        chunks.append(f"compact_n_tokens={len(row['input_ids'])}")
    return "\n".join(chunks)


def _haystack_for_match_in(row: dict[str, Any], match_in: str) -> str:
    if match_in == "response":
        return _row_gold_text(row)
    if match_in == "prompt":
        return _row_context_text(row)
    return _sample_haystack(row)


def normalize_expr_item(item: dict[str, Any]) -> dict[str, Any] | None:
    """Normalize one corpus_search_expressions entry to a single expression."""
    if not isinstance(item, dict):
        return None
    name = str(item.get("name") or "expr").strip() or "expr"
    why = str(item.get("why") or "").strip()
    match_in = _normalize_match_in(item.get("match_in") or item.get("where") or "full")
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
        "match_in": match_in,
    }


def search_corpus_jsonl(
    corpus_path: str,
    expression: str,
    *,
    top_k: int = 20,
    max_scan: int | None = None,
    stats: dict[str, Any] | None = None,
    match_in: str = "full",
) -> list[dict[str, Any]]:
    """Match ``expression`` against prompt, response, or full row text."""
    path = Path(corpus_path)
    if not path.is_file():
        raise FileNotFoundError(f"corpus not found: {corpus_path}")
    expr = (expression or "").strip()
    if not expr:
        return []
    match_in = _normalize_match_in(match_in)
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
            hay = _haystack_for_match_in(row, match_in)
            if not eval_boolean_expression(hay, expr):
                continue
            ctx = _row_context_text(row)
            resp = _row_gold_text(row)
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
                "match_in": match_in,
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
        stats["mode"] = f"{match_in}_prompt" if match_in != "full" else "full_prompt"
        stats["match_in"] = match_in
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
    language: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    del mode  # boolean-only; semantic is src.llm_semantic_retrieval
    client = _build_openai_client()
    model_name = model or _env("ANNOTATE_MODEL") or _env("LLM_RETRIEVE_MODEL") or "qwen-plus"
    mt = max_tokens or int(_env("ANNOTATE_MAX_TOKENS") or "4096")
    messages = build_llm_train_retrieve_messages(
        fim_prompt=fim_prompt,
        gold_completion=gold_completion,
        language=language,
    )
    extra = _extra_body()
    print(
        f"[llm-train] boolean model={model_name} "
        f"surface_chars={len(fim_prompt)} gold_chars={len(gold_completion)}",
        flush=True,
    )
    raw, parsed = _chat_complete_json(
        client,
        model=model_name,
        messages=messages,
        max_tokens=mt,
        extra_body=extra or None,
        temperature=0.2,
        log_prefix="[llm-train]",
    )
    parsed.pop("semantic", None)
    parsed.pop("semantic_flat_text", None)
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
        "retrieve_mode": "boolean",
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
    language: str | None = None,
    mode: str | None = None,
) -> dict[str, Any]:
    del mode
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
        language=language,
    )
    analysis = llm_out.get("analysis") or {}
    if not isinstance(analysis, dict):
        analysis = {}
    exprs = analysis.get("corpus_search_expressions") or []
    if not isinstance(exprs, list):
        exprs = []

    raw_corpus = corpus_path or _env("EIF_LLM_TRAIN_CORPUS") or _env("EIF_TRAIN_CORPUS") or ""
    search_results: list[dict[str, Any]] = []

    for item in exprs:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "expr")
        expression = str(item.get("expression") or "").strip()
        why = str(item.get("why") or "")
        match_in = _normalize_match_in(item.get("match_in"))
        if not expression:
            continue
        entry = {
            "name": name,
            "expression": expression,
            "why": why,
            "match_in": match_in,
            "corpus_hits": [],
            "local_bank_hits": [],
        }
        if run_corpus_search and raw_corpus and Path(raw_corpus).is_file():
            try:
                entry["corpus_hits"] = search_corpus_jsonl(
                    raw_corpus,
                    expression,
                    top_k=top_k,
                    max_scan=max_corpus_scan,
                    match_in=match_in,
                )
                entry["corpus_path"] = raw_corpus
                entry["retrieval"] = "boolean"
            except Exception as exc:
                entry["corpus_error"] = str(exc)
        if search_local_bank:
            entry["local_bank_hits"] = search_local_train_bank(expression, top_k=top_k)
        search_results.append(entry)

    return {
        "status": "success",
        "retrieve_mode": "boolean",
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
        "corpus_path": raw_corpus or None,
        "local_bank_path": _env("EIF_TRAIN_DATA") or _env("ANNOTATION_TRAIN_DATA") or None,
    }
