"""GraphSignal (tree-sitter + optional LLM) annotation for corpus rows."""

from __future__ import annotations

import os
from typing import Any

from server.graphsignal.run import annotate_row


def _sync_llm_env_from_eif() -> None:
    """Mirror eif_api.env LLM vars used by auto_annotate / GraphSignal."""
    if not os.environ.get("OPENAI_API_KEY") and os.environ.get("DASHSCOPE_API_KEY"):
        os.environ["OPENAI_API_KEY"] = os.environ["DASHSCOPE_API_KEY"]


def default_use_llm() -> bool:
    raw = (os.environ.get("ANNOTATE_USE_LLM") or os.environ.get("GRAPHSIGNAL_USE_LLM") or "1").strip().lower()
    return raw not in ("0", "false", "no", "off")


def default_max_len() -> int:
    raw = (
        os.environ.get("EIF_MODEL_MAX_LENGTH")
        or os.environ.get("ANNOTATE_MODEL_MAX_LENGTH")
        or "8192"
    ).strip()
    try:
        return max(512, int(raw))
    except ValueError:
        return 8192


def default_max_teacher_edges() -> int:
    raw = (
        os.environ.get("ANNOTATE_MAX_EDGES")
        or os.environ.get("ANNOTATE_MAX_TEACHER_EDGES")
        or "64"
    ).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        return 64


def annotate_corpus_row(
    raw_row: dict[str, Any],
    tokenizer: Any,
    *,
    max_len: int | None = None,
    max_teacher_edges: int | None = None,
    use_llm: bool | None = None,
    default_language: str = "Go",
) -> dict[str, Any]:
    """Run GraphSignal ``annotate_row`` on a corpus ``{prompt, response, ...}`` row."""
    _sync_llm_env_from_eif()

    row = dict(raw_row)
    lang = str(row.get("language") or default_language or "Go")
    row["language"] = lang
    if "response" not in row and row.get("label") is not None:
        row["response"] = row["label"]

    ml = default_max_len() if max_len is None else max_len
    mte = default_max_teacher_edges() if max_teacher_edges is None else max_teacher_edges
    llm = default_use_llm() if use_llm is None else bool(use_llm)

    result = annotate_row(row, tokenizer, ml, mte, use_llm=llm)
    if result is None:
        raise ValueError(
            "GraphSignal annotate failed: row is not FIM-annotatable "
            "(need <PRE>/<SUF>/<MID> in prompt and a non-empty response)."
        )
    return {
        **result,
        "_graphsignal_meta": {
            "use_llm": llm,
            "max_len": ml,
            "max_teacher_edges": mte,
            "language": lang,
            "annotate_model": os.environ.get("ANNOTATE_MODEL") or "",
        },
    }
