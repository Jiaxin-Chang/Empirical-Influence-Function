#!/usr/bin/env python3
"""Find train holes shaped like SequenceEntropy: special-case assign 0, else assigns otherwise.

Scans ``EIF_LLM_TRAIN_CORPUS`` as text. No LLM and no embeddings.

A hit is a gold fill that only assigns 0, sitting in an ``if`` whose suffix
opens ``else`` and assigns that same name to something other than 0.

Example::

    python -m src.search_manual_semantic
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

_ZERO = re.compile(
    r"""^\s*
        (?P<name>[A-Za-z_][\w.]*)
        \s*(?::=|=)
        \s*
        (?:float64|float32|int|int64|int32|uint|uint64|uint32|byte|rune)?
        \(?
        \s*0(?:\.0+)?\s*
        \)?
        \s*(?://.*)?$
    """,
    re.VERBOSE,
)
_ASSIGN = re.compile(
    r"""^\s*
        (?P<name>[A-Za-z_][\w.]*)
        \s*(?::=|=)
        \s*
        (?P<rhs>\S.*?)\s*$
    """,
    re.VERBOSE,
)
_ELSE_OPEN = re.compile(r"^\}\s*else(?:\s+if\b[^{]*)?\s*\{")


def _hydrate() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "eif_api.env"
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key, val = key.strip(), val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def _gold(row: dict) -> str:
    for key in ("completion", "response", "label", "output", "gold"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _prompt(row: dict) -> str:
    for key in ("prompt", "input", "query"):
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _zero_names(gold: str) -> list[str]:
    names: list[str] = []
    for raw in gold.splitlines():
        line = raw.strip()
        if not line or line.startswith("//"):
            continue
        match = _ZERO.match(line)
        if not match:
            return []
        names.append(match.group("name"))
    return names


def _pre_suf(prompt: str) -> tuple[str, str]:
    pre_at = prompt.rfind("<PRE>")
    suf_at = prompt.find("<SUF>", pre_at + 1 if pre_at >= 0 else 0)
    if pre_at < 0 or suf_at < 0:
        return "", ""
    prefix = prompt[pre_at + len("<PRE>") : suf_at]
    suffix = prompt[suf_at + len("<SUF>") :]
    mid_at = suffix.find("<MID>")
    if mid_at >= 0:
        suffix = suffix[:mid_at]
    return prefix, suffix


def _else_rhs(suffix: str, name: str) -> str:
    lines = suffix.splitlines()
    opened = False
    for raw in lines[:24]:
        line = raw.strip()
        if not opened:
            if _ELSE_OPEN.match(line) or _ELSE_OPEN.match(line.lstrip()):
                opened = True
            continue
        if not line or line.startswith("//"):
            continue
        match = _ASSIGN.match(line)
        if match and match.group("name") == name and not _ZERO.match(line):
            return line.strip()
        if line.startswith("}"):
            break
    return ""


def _if_tail(prefix: str) -> str:
    kept = [ln.rstrip() for ln in prefix.splitlines() if ln.strip()]
    tail = kept[-4:]
    blob = "\n".join(tail)
    if not re.search(r"\bif\b", blob):
        return ""
    if not blob.rstrip().endswith("{"):
        return ""
    return blob


def main() -> None:
    _hydrate()
    train = (os.environ.get("EIF_LLM_TRAIN_CORPUS") or "").strip()
    if not train:
        raise SystemExit("set EIF_LLM_TRAIN_CORPUS")
    path = Path(train)
    if not path.is_file():
        raise SystemExit(f"train corpus not found: {path}")
    hits: list[tuple[int, str, str, str, str]] = []
    with path.open(encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle):
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            names = _zero_names(_gold(row))
            if not names:
                continue
            prefix, suffix = _pre_suf(_prompt(row))
            if_tail = _if_tail(prefix)
            if not if_tail:
                continue
            rhs = ""
            picked = ""
            for name in dict.fromkeys(names):
                rhs = _else_rhs(suffix, name)
                if rhs:
                    picked = name
                    break
            if not rhs:
                continue
            hits.append((line_no, picked, _gold(row).strip(), if_tail, rhs))
    print(f"train={path} hits={len(hits)}", flush=True)
    for line_no, name, gold, if_tail, rhs in hits:
        print(f"\n#line={line_no} name={name}", flush=True)
        print("  if:", flush=True)
        for ln in if_tail.splitlines():
            print(f"    {ln}", flush=True)
        print(f"  gold: {gold}", flush=True)
        print(f"  else: {rhs}", flush=True)


if __name__ == "__main__":
    main()
