#!/usr/bin/env python3
"""Semantic-attribute each imperfect valid and bucket relation types.

Reads ``valid_imperfect_less.jsonl``. Each row is sent to the already-running
8766 API (``POST /api/llm-semantic-retrieve``), the same call the report
frontend uses. Corpus search is off; only the residual semantic JSON is kept.
Records whether any relation is ``control`` (also ``control_flow`` /
``controlflow``) or ``semantic_dependency``.

Writes one JSON object per row, and a summary of the two buckets.
Already-ok lines in the output file are skipped, so a stopped run can resume.

Example::

    python -m src.scan_valid_relation_types
    python -m src.scan_valid_relation_types --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.auto_go_top5_rerank_continue import _prompt_gold_pred
from src.auto_raw_ce_to_continue import _http_json
from src.fim_semantic_schema import canonicalize_relation_type

DEFAULT_INPUT = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/"
    "correlation_matching_results/raw_ce/valid_imperfect_less.jsonl"
)
DEFAULT_OUT = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/"
    "valid_imperfect_relation_types.jsonl"
)
DEFAULT_EIF = "http://127.0.0.1:8766"

_CONTROL = frozenset({"control", "control_flow", "controlflow"})
_SEMDEP = frozenset({"semantic_dependency"})


def _bucket(type_name: str) -> str:
    canon = canonicalize_relation_type(type_name)
    if canon in _CONTROL:
        return "control"
    if canon in _SEMDEP:
        return "semantic_dependency"
    return ""


def _load_done(path: Path) -> dict[int, dict[str, Any]]:
    done: dict[int, dict[str, Any]] = {}
    if not path.is_file():
        return done
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict) or row.get("status") != "ok":
                continue
            line = row.get("line")
            if isinstance(line, int):
                done[line] = row
    return done


def _relations(sem: dict[str, Any]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for item in sem.get("relations") or []:
        if not isinstance(item, dict):
            continue
        src = str(item.get("source") or "").strip()
        tgt = str(item.get("target") or "").strip()
        typ = str(item.get("type") or "").strip()
        if src and tgt and typ:
            out.append({"source": src, "target": tgt, "type": typ})
    return out


def _attribute(eif_url: str, prompt: str, gold: str, pred: str, language: str) -> dict[str, Any]:
    """Same endpoint as the report UI. 8766 already has eif_api.env loaded."""
    url = eif_url.rstrip("/") + "/api/llm-semantic-retrieve"
    result = _http_json(
        "POST",
        url,
        {
            "fimPrompt": prompt,
            "goldCompletion": gold,
            "modelPrediction": pred,
            "language": language,
            "runCorpusSearch": False,
        },
        timeout=600.0,
    )
    if result.get("status") != "success":
        raise RuntimeError(str(result.get("message") or result))
    sem = result.get("semantic")
    if not isinstance(sem, dict):
        raise RuntimeError("8766 response missing semantic")
    return sem


def _scan(input_path: Path, out_path: Path, *, eif_url: str, limit: int) -> None:
    done = _load_done(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seen = 0
    new_ok = 0
    with input_path.open(encoding="utf-8") as src, out_path.open("a", encoding="utf-8") as dst:
        for line_no, raw in enumerate(src, start=1):
            if limit and seen >= limit:
                break
            if not raw.strip():
                continue
            seen += 1
            if line_no in done:
                print(f"[skip] line={line_no} already ok", flush=True)
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                rec = {"line": line_no, "status": "bad_json", "error": str(exc)}
                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
                dst.flush()
                print(f"[bad_json] line={line_no} {exc}", flush=True)
                continue
            if not isinstance(row, dict):
                continue
            prompt, gold, pred = _prompt_gold_pred(row)
            task_id = str(row.get("task_id") or "")
            language = str(row.get("language") or "go")
            if not prompt.strip() or not gold.strip():
                rec = {
                    "line": line_no,
                    "task_id": task_id,
                    "status": "skip",
                    "error": "missing prompt or gold",
                }
                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
                dst.flush()
                continue
            try:
                sem = _attribute(eif_url, prompt, gold, pred, language)
            except Exception as exc:
                rec = {
                    "line": line_no,
                    "task_id": task_id,
                    "status": "error",
                    "error": str(exc),
                }
                dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
                dst.flush()
                print(f"[error] line={line_no} {task_id} {exc}", flush=True)
                continue
            rels = _relations(sem)
            buckets = sorted({b for rel in rels if (b := _bucket(rel["type"]))})
            rec = {
                "line": line_no,
                "task_id": task_id,
                "source_line": row.get("source_line"),
                "status": "ok",
                "buckets": buckets,
                "relations": rels,
            }
            dst.write(json.dumps(rec, ensure_ascii=False) + "\n")
            dst.flush()
            done[line_no] = rec
            new_ok += 1
            print(
                f"[ok] line={line_no} task={task_id} buckets={buckets or '-'} "
                f"relations={len(rels)}",
                flush=True,
            )
    _write_summary(out_path, done)
    print(f"[done] new_ok={new_ok} total_ok={len(done)} out={out_path}", flush=True)


def _write_summary(out_path: Path, done: dict[int, dict[str, Any]]) -> None:
    groups = {"control": [], "semantic_dependency": []}
    for line_no in sorted(done):
        rec = done[line_no]
        hit_rels = {
            "control": [],
            "semantic_dependency": [],
        }
        for rel in rec.get("relations") or []:
            bucket = _bucket(str(rel.get("type") or ""))
            if bucket:
                hit_rels[bucket].append(rel)
        for bucket, rels in hit_rels.items():
            if not rels:
                continue
            groups[bucket].append({
                "line": line_no,
                "task_id": rec.get("task_id") or "",
                "source_line": rec.get("source_line"),
                "relations": rels,
            })
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(groups, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for bucket, rows in groups.items():
        print(f"\n## {bucket} ({len(rows)})", flush=True)
        for row in rows:
            rels = "; ".join(
                f"{r['source']} -> {r['target']} [{r['type']}]" for r in row["relations"]
            )
            print(
                f"  line={row['line']} task={row['task_id']} {rels}",
                flush=True,
            )
    print(f"\n[summary] {summary_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--eif-url", default=DEFAULT_EIF)
    parser.add_argument("--limit", type=int, default=0, help="0 = all rows")
    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.is_file():
        print(f"input not found: {input_path}", file=sys.stderr)
        sys.exit(2)
    _scan(
        input_path,
        Path(args.out),
        eif_url=str(args.eif_url).rstrip("/"),
        limit=max(0, int(args.limit)),
    )


if __name__ == "__main__":
    main()
