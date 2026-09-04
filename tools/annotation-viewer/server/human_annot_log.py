"""Append-only log of *human* annotation edits (not tree-sitter / LLM auto).

Each event stores the full sample text (prompt + response) plus the edge as
token surfaces. Indices are kept only so the same sample can be replayed;
they must not be copied across samples into prompts.

Optional ``query_expression`` / ``query_name`` record which corpus-search
boolean retrieved this row — used later to group few-shots by retrieval family.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from server.semantic_prompt import load_active

VIEWER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]

_log_lock = threading.Lock()


def log_path() -> Path:
    raw = (os.environ.get("ANNOTATION_HUMAN_LOG") or "").strip()
    if raw:
        p = Path(raw).expanduser()
        return p if p.is_absolute() else (REPO_ROOT / p).resolve()
    return (VIEWER_ROOT / "human_annot_log.jsonl").resolve()


def _norm_surf(tok: Any) -> str:
    s = "" if tok is None else str(tok)
    return s.replace("Ġ", " ").replace("▁", " ")


def token_window(tokens: list[str], idx: int, *, radius: int = 3) -> str:
    if not tokens:
        return ""
    i = max(0, min(int(idx), len(tokens) - 1))
    lo = max(0, i - radius)
    hi = min(len(tokens), i + radius + 1)
    return "".join(_norm_surf(t) for t in tokens[lo:hi])


def append_human_event(event: dict[str, Any]) -> None:
    row = dict(event)
    row.setdefault("ts", datetime.now(timezone.utc).isoformat())
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(row, ensure_ascii=False)
    with _log_lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def load_events(*, limit: int | None = None) -> list[dict[str, Any]]:
    path = log_path()
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    if limit is not None:
        return out[-max(0, int(limit)) :]
    return out


def sample_key(event: dict[str, Any]) -> str:
    corpus = str(event.get("corpus_path") or "").strip()
    line = event.get("line")
    if corpus and line is not None:
        return f"corpus:{corpus}::{int(line)}"
    idx = event.get("train_index")
    if idx is not None:
        return f"train:{int(idx)}"
    uid = str(event.get("uid") or event.get("task_id") or "").strip()
    return uid or "unknown"


def gold_edges_for_sample(events: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """Net human add-minus-delete edges as (src_text, dst_text)."""
    gold: set[tuple[str, str]] = set()
    for ev in events:
        action = str(ev.get("action") or "")
        edge = ev.get("edge") if isinstance(ev.get("edge"), dict) else {}
        src = _norm_surf(edge.get("src_text") or "").strip()
        dst = _norm_surf(edge.get("dst_text") or "").strip()
        if not src or not dst:
            continue
        pair = (src, dst)
        if action in ("add", "bump"):
            gold.add(pair)
        elif action == "delete":
            gold.discard(pair)
    return gold


def gold_index_edges_for_sample(events: list[dict[str, Any]]) -> set[tuple[int, int]]:
    gold: set[tuple[int, int]] = set()
    for ev in events:
        action = str(ev.get("action") or "")
        edge = ev.get("edge") if isinstance(ev.get("edge"), dict) else {}
        try:
            src = int(edge.get("src_idx"))
            dst = int(edge.get("dst_idx"))
        except (TypeError, ValueError):
            continue
        pair = (src, dst)
        if action in ("add", "bump"):
            gold.add(pair)
        elif action == "delete":
            gold.discard(pair)
    return gold


def group_by_sample(events: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for ev in events:
        grouped.setdefault(sample_key(ev), []).append(ev)
    return grouped


def build_event(
    *,
    action: str,
    tokens: list[str],
    src: int,
    dst: int,
    subtype: str,
    prompt: str = "",
    response: str = "",
    language: str = "",
    task_id: str = "",
    uid: str = "",
    corpus_path: str = "",
    line: int | None = None,
    train_index: int | None = None,
    query_expression: str = "",
    query_name: str = "",
    weight: float | None = None,
    contrib: str = "user_add",
) -> dict[str, Any]:
    n = len(tokens)
    src_i = int(src)
    dst_i = int(dst)
    src_txt = _norm_surf(tokens[src_i] if 0 <= src_i < n else "")
    dst_txt = _norm_surf(tokens[dst_i] if 0 <= dst_i < n else "")
    active = load_active()
    edge: dict[str, Any] = {
        "src_idx": src_i,
        "dst_idx": dst_i,
        "src_text": src_txt,
        "dst_text": dst_txt,
        "src_context": token_window(tokens, src_i),
        "dst_context": token_window(tokens, dst_i),
        "subtype": str(subtype or ""),
    }
    if weight is not None:
        edge["weight"] = float(weight)
    return {
        "action": str(action),
        "contrib": str(contrib or "user_add"),
        "n_tokens": n,
        "prompt": str(prompt or ""),
        "response": str(response or ""),
        "language": str(language or ""),
        "task_id": str(task_id or ""),
        "uid": str(uid or ""),
        "corpus_path": str(corpus_path or ""),
        "line": None if line is None else int(line),
        "train_index": None if train_index is None else int(train_index),
        "query_expression": str(query_expression or ""),
        "query_name": str(query_name or ""),
        "prompt_version": str(active.get("id") or "default"),
        "edge": edge,
    }
