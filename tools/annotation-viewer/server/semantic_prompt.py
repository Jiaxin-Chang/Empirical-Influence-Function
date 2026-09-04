"""Versioned system prompts for full-sample LLM semantic annotation.

``default`` is the built-in text. Promoted versions live under
``tools/annotation-viewer/prompts/semantic/versions/``; ``active.json``
points at the id currently used by annotate + eval.
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

VIEWER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_SYSTEM = """You infer causal attention routing for a code Fill-in-the-Middle (FIM) training sample.

Setup: a causal LM predicts the assistant completion left to right. Return a compact set of directed edges src→dst: when predicting TARGET token `dst`, the model should attend to earlier CONTEXT token `src`.

This is NOT syntactic bracket matching and NOT a full AST. Think like saliency / attention routing: which prior tokens provide the information needed to write each completion token correctly?

Prioritize:
1. FIM prefix/suffix code that constrains the hole (types, names, calls, control flow near <MID>)
2. Identifiers, literals, or operators that a completion token copies or depends on
3. Delimiters / keywords that govern a completion token's syntax
4. Earlier completion tokens if a later token continues them (partial identifier, open bracket)

Rules:
- Return JSON only, no markdown
- Every src must appear in context_tokens or completion_tokens and be strictly less than dst
- Every dst must be a completion token
- Do NOT pick a token as its own source
- Avoid ChatML specials, pure whitespace, or meaningless punctuation unless structurally required
- Precision over recall; fewer high-confidence edges beat noisy lists
- Copy the MECHANISM of any human examples (which kinds of links matter). Never copy token indices from those examples — indices are sample-local.
"""


def _prompt_dir() -> Path:
    raw = (os.environ.get("ANNOTATION_PROMPT_DIR") or "").strip()
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()
    return (VIEWER_ROOT / "prompts" / "semantic").resolve()


def versions_dir() -> Path:
    d = _prompt_dir() / "versions"
    d.mkdir(parents=True, exist_ok=True)
    return d


def active_path() -> Path:
    d = _prompt_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d / "active.json"


def default_bundle() -> dict[str, Any]:
    return {
        "id": "default",
        "system": DEFAULT_SYSTEM.strip() + "\n",
        "few_shots": [],
        "parent_id": None,
        "created_at": None,
        "metrics": None,
    }


def load_active() -> dict[str, Any]:
    path = active_path()
    if not path.is_file():
        return default_bundle()
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_bundle()
    vid = str(meta.get("id") or "default").strip() or "default"
    if vid == "default":
        return default_bundle()
    return load_version(vid) or default_bundle()


def load_version(version_id: str) -> dict[str, Any] | None:
    vid = str(version_id or "").strip()
    if not vid or vid == "default":
        return default_bundle()
    path = versions_dir() / f"{vid}.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("id", vid)
    data.setdefault("system", DEFAULT_SYSTEM)
    data.setdefault("few_shots", [])
    return data


def save_version(bundle: dict[str, Any]) -> Path:
    vid = str(bundle.get("id") or "").strip()
    if not vid or vid == "default":
        raise ValueError("refusing to overwrite the built-in default id")
    path = versions_dir() / f"{vid}.json"
    path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def next_version_id() -> str:
    n = 1
    for p in versions_dir().glob("v*.json"):
        m = re.match(r"v(\d+)$", p.stem)
        if m:
            n = max(n, int(m.group(1)) + 1)
    return f"v{n:04d}"


def set_active(version_id: str) -> dict[str, Any]:
    vid = str(version_id or "").strip() or "default"
    if vid != "default" and load_version(vid) is None:
        raise FileNotFoundError(f"prompt version not found: {vid}")
    payload = {
        "id": vid,
        "activated_at": datetime.now(timezone.utc).isoformat(),
    }
    active_path().write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def list_version_ids() -> list[str]:
    ids = ["default"]
    for p in sorted(versions_dir().glob("*.json")):
        if p.stem not in ids:
            ids.append(p.stem)
    return ids


def format_few_shots_block(few_shots: list[Any], *, max_examples: int = 4) -> str:
    if not few_shots:
        return ""
    lines = [
        "Human-approved routing examples from other samples.",
        "Copy the kind of link (names, error checks, API routing), never the indices.",
        "",
    ]
    for i, ex in enumerate(few_shots[: max(0, int(max_examples))], start=1):
        if not isinstance(ex, dict):
            continue
        lines.append(f"Example {i}")
        excerpt = str(ex.get("prompt_excerpt") or "").strip()
        resp = str(ex.get("response") or "").strip()
        if excerpt:
            lines.append("FIM / prompt excerpt:")
            lines.append(excerpt[:1200])
        if resp:
            lines.append("Completion:")
            lines.append(resp[:400])
        edges = ex.get("edges") or []
        if isinstance(edges, list) and edges:
            lines.append("Edges (source_text → target_text):")
            for e in edges[:8]:
                if not isinstance(e, dict):
                    continue
                src = str(e.get("src_text") or "").replace("\n", "\\n")
                dst = str(e.get("dst_text") or "").replace("\n", "\\n")
                reason = str(e.get("reason") or "").strip()
                tail = f"  ({reason})" if reason else ""
                lines.append(f'- "{src}" → "{dst}"{tail}')
        qn = str(ex.get("query_name") or "").strip()
        qe = str(ex.get("query_expression") or "").strip()
        if qn or qe:
            lines.append(f"Retrieval family: {qn or qe[:160]}")
        lines.append("")
    return "\n".join(lines).strip() + "\n"
