#!/usr/bin/env python3
"""Automate: raw_ce → LLM 表达式归因 → 从紧到松搜语料 → MID 改写 → 标注 → 写入续训小集.

Mirrors the manual UI flow in correlation-report + annotation-viewer, but
runs headless over HTTP.

Prerequisites (both must be running, with eif_api.env loaded)::

    # ttav / EIF API (LLM retrieve + corpus search) — default :8766
    python -m src.ttav_bundle_api

    # annotation-viewer API (MID rewrite + annotate → continue JSONL) — default :8765
    cd tools/annotation-viewer && python -m server.main

Per test sample (default)::

    1 successful corpus hit → GraphSignal annotate once → duplicate to 20
    continue-train rows → next test.

Example::

    python -m src.auto_raw_ce_to_continue \\
        --input correlation_matching_results/raw_ce/foo.jsonl \\
        --copies 20 \\
        --annotate graphsignal \\
        --tight-first

Expression order: LLM returns 从宽到窄; ``--tight-first`` (default) reverses
to try the narrowest expression first. MID rewrite failures skip that hit
and try the next corpus line / next looser expression.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# MID prep reasons that still allow annotating the *original* train sample.
_MID_OK_WITHOUT_REWRITE = frozenset({
    "train_mid_already_is_gold",
    "same_as_original_mid",
})

# Explicit MID failures → skip hit (try next).
_MID_SKIP_REASONS = frozenset({
    "no_dig_span",
    "section_rewrite_failed",
    "dig_not_found_in_filled",
    "no_fim_markers",
})


@dataclass
class RunState:
    """``annotated`` = one record per *successful test* (not per continue row)."""
    copies_per_hit: int
    max_tests: int  # 0 = no limit
    annotated: list[dict[str, Any]] = field(default_factory=list)
    skipped: list[dict[str, Any]] = field(default_factory=list)
    processed_test_lines: list[int] = field(default_factory=list)
    used_corpus_lines: set[int] = field(default_factory=set)

    @property
    def n_ok_tests(self) -> int:
        return len(self.annotated)

    @property
    def n_continue_rows(self) -> int:
        return sum(int(r.get("continue_rows") or 0) for r in self.annotated)

    def done(self) -> bool:
        return self.max_tests > 0 and self.n_ok_tests >= self.max_tests


def _http_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    *,
    timeout: float = 600.0,
) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            if not raw.strip():
                return {"ok": True, "_http_status": resp.status}
            out = json.loads(raw)
            if isinstance(out, dict):
                out["_http_status"] = resp.status
                return out
            return {"ok": True, "data": out, "_http_status": resp.status}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        detail: Any
        try:
            detail = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            detail = {"message": raw[:500]}
        msg = ""
        if isinstance(detail, dict):
            msg = str(detail.get("message") or detail.get("detail") or "")
        raise RuntimeError(
            f"HTTP {exc.code} {method} {url}: {msg or raw[:300]}"
        ) from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Cannot reach {url}: {exc}") from exc


def _load_raw_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    """Return ``(1-based line_no, row)`` for JSONL or a JSON list/object.

    ``*.jsonl`` is always parsed line-by-line (one JSON object per line).
    """
    text = path.read_text(encoding="utf-8")
    # Prefer JSONL for .jsonl even if the first char is '{' (every line is an object).
    if path.suffix.lower() == ".jsonl" or path.name.lower().endswith(".jsonl"):
        out: list[tuple[int, dict[str, Any]]] = []
        for i, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append((i, obj))
        return out

    stripped = text.lstrip()
    if stripped.startswith("["):
        arr = json.loads(text)
        if not isinstance(arr, list):
            raise ValueError(f"{path}: expected JSON array")
        out = []
        for i, obj in enumerate(arr, start=1):
            if isinstance(obj, dict):
                out.append((i, obj))
        return out

    # Single-object .json (no newline between braces as multiple records)
    non_empty = [ln for ln in text.splitlines() if ln.strip()]
    if len(non_empty) == 1 and non_empty[0].lstrip().startswith("{"):
        obj = json.loads(non_empty[0])
        if not isinstance(obj, dict):
            raise ValueError(f"{path}: expected JSON object")
        return [(1, obj)]

    # Fallback: treat as JSONL
    out = []
    for i, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            out.append((i, obj))
    return out


def _fim_and_gold(row: dict[str, Any]) -> tuple[str, str]:
    prompt = str(row.get("prompt") or row.get("input") or "").strip()
    gold = str(
        row.get("label") or row.get("response") or row.get("gold") or ""
    ).strip()
    return prompt, gold


def _expr_items(retrieve: dict[str, Any], *, tight_first: bool) -> list[dict[str, Any]]:
    analysis = retrieve.get("analysis") or {}
    exprs = analysis.get("corpus_search_expressions") or []
    if not isinstance(exprs, list):
        exprs = []
    cleaned: list[dict[str, Any]] = []
    for i, item in enumerate(exprs):
        if not isinstance(item, dict):
            continue
        expression = str(item.get("expression") or "").strip()
        if not expression:
            continue
        cleaned.append({
            "name": str(item.get("name") or f"expr_{i}"),
            "expression": expression,
            "why": str(item.get("why") or ""),
            "llm_index": i,  # 0 = widest
        })
    if tight_first:
        cleaned.reverse()
    return cleaned


def _rank_hits(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Prefer gold match_region, then cross, then context."""
    order = {"gold": 0, "cross": 1, "context": 2}

    def key(h: dict[str, Any]) -> tuple[int, int]:
        region = str(h.get("match_region") or "").lower()
        return (order.get(region, 9), int(h.get("line") or 10**12))

    return sorted(hits, key=key)


def _mid_decision(prep: dict[str, Any], match_region: str) -> tuple[bool, str | None, str]:
    """Return (accept_hit, rewrite_id_or_none, note)."""
    reason = str(prep.get("reason") or prep.get("mode") or "")
    rewrite_id = str(prep.get("rewrite_id") or "").strip() or None
    region = (match_region or "").lower()

    if prep.get("applied"):
        return True, rewrite_id, f"rewritten ({prep.get('mode')}@{prep.get('dig_locus') or '?'})"

    if reason in _MID_OK_WITHOUT_REWRITE:
        return True, None, f"keep original ({reason})"

    if region == "gold" and reason in ("unchanged", ""):
        # Gold hit but dig already aligned / nothing to move — still usable.
        return True, None, "gold hit, keep original MID"

    if reason in _MID_SKIP_REASONS or reason == "unchanged":
        detail = str(prep.get("detail") or "")
        return False, None, f"MID skip ({reason}{': ' + detail if detail else ''})"

    # Unknown failure → skip to be safe
    return False, None, f"MID skip (unexpected reason={reason!r})"


def _save_state(path: Path | None, state: RunState) -> None:
    if path is None:
        return
    payload = {
        "copies_per_hit": state.copies_per_hit,
        "max_tests": state.max_tests,
        "n_ok_tests": state.n_ok_tests,
        "n_continue_rows": state.n_continue_rows,
        "annotated": state.annotated,
        "skipped": state.skipped[-200:],
        "processed_test_lines": state.processed_test_lines,
        "used_corpus_lines": sorted(state.used_corpus_lines),
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_state(
    path: Path | None,
    *,
    copies_per_hit: int,
    max_tests: int,
) -> RunState:
    state = RunState(copies_per_hit=copies_per_hit, max_tests=max_tests)
    if path is None or not path.is_file():
        return state
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return state
    for row in obj.get("annotated") or []:
        if isinstance(row, dict):
            state.annotated.append(row)
            line = row.get("corpus_line")
            if line is not None:
                state.used_corpus_lines.add(int(line))
    for line in obj.get("processed_test_lines") or []:
        try:
            state.processed_test_lines.append(int(line))
        except (TypeError, ValueError):
            pass
    for line in obj.get("used_corpus_lines") or []:
        try:
            state.used_corpus_lines.add(int(line))
        except (TypeError, ValueError):
            pass
    print(
        f"[resume] loaded ok_tests={state.n_ok_tests} "
        f"continue_rows≈{state.n_continue_rows} from {path}",
        flush=True,
    )
    return state


class PipelineClient:
    def __init__(
        self,
        *,
        eif_url: str,
        viewer_url: str,
        corpus_path: str | None,
        annotate: str,
        top_k: int,
        max_scan: int | None,
        graphsignal_use_llm: bool | None,
        dry_run: bool,
    ):
        self.eif_url = eif_url.rstrip("/")
        self.viewer_url = viewer_url.rstrip("/")
        self.corpus_path = (corpus_path or "").strip() or None
        self.annotate = annotate
        self.top_k = top_k
        self.max_scan = max_scan
        self.graphsignal_use_llm = graphsignal_use_llm
        self.dry_run = dry_run

    def ping(self) -> None:
        # Soft checks — endpoints may 404 on GET; connection errors are hard fail.
        for name, base in (("eif", self.eif_url), ("viewer", self.viewer_url)):
            try:
                urllib.request.urlopen(base + "/", timeout=5)
            except urllib.error.HTTPError:
                pass
            except Exception as exc:
                print(f"[warn] {name} API {base} may be down: {exc}", flush=True)

    def llm_retrieve(self, fim_prompt: str, gold: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "fimPrompt": fim_prompt,
            "goldCompletion": gold,
            "topK": self.top_k,
            # Expressions only here; we search ourselves tight→loose.
            "runCorpusSearch": False,
            "searchLocalBank": False,
        }
        if self.corpus_path:
            body["corpusPath"] = self.corpus_path
        return _http_json(
            "POST",
            f"{self.eif_url}/api/llm-train-retrieve",
            body,
            timeout=300.0,
        )

    def corpus_search(self, expression: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "expression": expression,
            "topK": self.top_k,
        }
        if self.corpus_path:
            body["corpusPath"] = self.corpus_path
        if self.max_scan is not None:
            body["maxCorpusScan"] = self.max_scan
        return _http_json(
            "POST",
            f"{self.eif_url}/api/llm-corpus-search",
            body,
            timeout=600.0,
        )

    def mid_rewrite_prep(
        self,
        *,
        line: int,
        gold: str,
        expression: str,
        corpus_path: str | None,
    ) -> dict[str, Any]:
        return _http_json(
            "POST",
            f"{self.viewer_url}/api/corpus/mid-rewrite-prep",
            {
                "line": int(line),
                "corpusPath": corpus_path or self.corpus_path or "",
                "testGold": gold,
                "expression": expression or "",
            },
            timeout=120.0,
        )

    def bind_rewrite(
        self,
        *,
        line: int,
        rewrite_id: str | None,
        corpus_path: str | None,
    ) -> None:
        if not rewrite_id:
            return
        q = {"corpusPath": corpus_path or self.corpus_path or ""}
        q["rewriteId"] = rewrite_id
        qs = urllib.parse.urlencode({k: v for k, v in q.items() if v})
        _http_json(
            "GET",
            f"{self.viewer_url}/api/corpus/sample/{int(line)}?{qs}",
            timeout=120.0,
        )

    def annotate_and_accept(
        self,
        *,
        line: int,
        corpus_path: str | None,
    ) -> dict[str, Any]:
        if self.dry_run:
            return {"ok": True, "dry_run": True, "n_continue_edges": 0}

        cp = corpus_path or self.corpus_path or ""
        qs = urllib.parse.urlencode({"corpusPath": cp}) if cp else ""
        base = f"{self.viewer_url}/api/corpus/sample/{int(line)}"

        if self.annotate == "none":
            return {"ok": True, "skipped_annotate": True}

        if self.annotate == "llm-semantic":
            preview = _http_json(
                "POST",
                f"{base}/llm-semantic-annotate/preview" + (f"?{qs}" if qs else ""),
                {},
                timeout=1800.0,
            )
            pid = str(preview.get("preview_id") or "")
            if not pid:
                raise RuntimeError("llm-semantic preview missing preview_id")
            return _http_json(
                "POST",
                f"{base}/llm-semantic-annotate/accept" + (f"?{qs}" if qs else ""),
                {"preview_id": pid},
                timeout=120.0,
            )

        # default: graphsignal
        body: dict[str, Any] = {}
        if self.graphsignal_use_llm is not None:
            body["use_llm"] = bool(self.graphsignal_use_llm)
        preview = _http_json(
            "POST",
            f"{base}/graphsignal-annotate/preview" + (f"?{qs}" if qs else ""),
            body,
            timeout=600.0,
        )
        pid = str(preview.get("preview_id") or "")
        if not pid:
            raise RuntimeError("graphsignal preview missing preview_id")
        n_edges = int(preview.get("n_edges") or 0)
        if n_edges <= 0:
            raise RuntimeError("graphsignal produced 0 edges")
        return _http_json(
            "POST",
            f"{base}/graphsignal-annotate/accept" + (f"?{qs}" if qs else ""),
            {"preview_id": pid},
            timeout=120.0,
        )

    def continue_duplicate(
        self,
        *,
        line: int,
        copies: int,
        corpus_path: str | None,
    ) -> dict[str, Any]:
        cp = corpus_path or self.corpus_path or ""
        qs = urllib.parse.urlencode({"corpusPath": cp}) if cp else ""
        return _http_json(
            "POST",
            f"{self.viewer_url}/api/corpus/sample/{int(line)}/continue-duplicate"
            + (f"?{qs}" if qs else ""),
            {"copies": int(copies)},
            timeout=60.0,
        )


def _duplicate_to_total(
    client: PipelineClient,
    *,
    line: int,
    corpus_path: str | None,
    total_copies: int,
) -> int:
    """After 1 accept row exists, append more until ``total_copies`` rows.

    API ``copies`` = *additional* rows (max 32 per call).
    Returns number of continue rows written for this hit (including the original).
    """
    if client.dry_run or client.annotate == "none":
        return 1
    want = max(1, int(total_copies))
    extra = want - 1
    written_extra = 0
    while written_extra < extra:
        batch = min(32, extra - written_extra)
        client.continue_duplicate(
            line=line,
            copies=batch,
            corpus_path=corpus_path,
        )
        written_extra += batch
    return 1 + written_extra


def process_test_row(
    client: PipelineClient,
    state: RunState,
    *,
    test_line: int,
    row: dict[str, Any],
    tight_first: bool,
    state_path: Path | None,
) -> None:
    """Find one usable train hit → annotate once → duplicate ``copies_per_hit`` → done."""
    if state.done():
        return

    task_id = str(row.get("task_id") or f"row_{test_line}")
    fim, gold = _fim_and_gold(row)
    if not fim or not gold:
        print(f"[test {test_line}] skip: missing prompt/gold ({task_id})", flush=True)
        state.skipped.append({
            "test_line": test_line,
            "task_id": task_id,
            "reason": "missing_prompt_or_gold",
        })
        state.processed_test_lines.append(test_line)
        return

    print(
        f"\n=== test line={test_line} task_id={task_id} "
        f"(ok_tests={state.n_ok_tests}, continue_rows≈{state.n_continue_rows}) ===",
        flush=True,
    )
    print(f"  gold_chars={len(gold)} prompt_chars={len(fim)}", flush=True)

    try:
        retrieve = client.llm_retrieve(fim, gold)
    except Exception as exc:
        print(f"  [fail] llm-train-retrieve: {exc}", flush=True)
        state.skipped.append({
            "test_line": test_line,
            "task_id": task_id,
            "reason": f"retrieve_failed: {exc}",
        })
        state.processed_test_lines.append(test_line)
        return

    if str(retrieve.get("status") or "") not in ("success", "ok", ""):
        if not (retrieve.get("analysis") or {}).get("corpus_search_expressions"):
            print(
                f"  [fail] retrieve status={retrieve.get('status')}: "
                f"{retrieve.get('message')}",
                flush=True,
            )
            state.skipped.append({
                "test_line": test_line,
                "task_id": task_id,
                "reason": "retrieve_bad_status",
                "detail": retrieve.get("message"),
            })
            state.processed_test_lines.append(test_line)
            return

    exprs = _expr_items(retrieve, tight_first=tight_first)
    if not exprs:
        print("  [fail] no corpus_search_expressions from LLM", flush=True)
        state.skipped.append({
            "test_line": test_line,
            "task_id": task_id,
            "reason": "no_expressions",
        })
        state.processed_test_lines.append(test_line)
        return

    order_label = "紧→松" if tight_first else "宽→窄"
    print(f"  expressions ({len(exprs)}, {order_label}):", flush=True)
    for i, e in enumerate(exprs):
        print(
            f"    [{i}] (llm#{e['llm_index']}) {e['name']}: {e['expression'][:120]}",
            flush=True,
        )

    corpus_path = (
        str(retrieve.get("corpus_path") or "").strip()
        or client.corpus_path
    )

    for ei, expr in enumerate(exprs):
        expression = expr["expression"]
        print(f"  -- expr[{ei}] {expr['name']} search…", flush=True)
        try:
            search = client.corpus_search(expression)
        except Exception as exc:
            print(f"     search failed: {exc}", flush=True)
            continue

        hits = _rank_hits(list(search.get("hits") or []))
        n_hits = len(hits)
        stats = search.get("search_stats") or {}
        print(
            f"     hits={n_hits} scanned={stats.get('scanned_lines')} "
            f"stop={stats.get('stop_reason')} "
            f"path={search.get('corpus_path') or corpus_path}",
            flush=True,
        )
        if n_hits == 0:
            print("     no hits → try next (looser) expression", flush=True)
            continue

        hit_corpus = str(search.get("corpus_path") or corpus_path or "") or None

        for hit in hits:
            cline = hit.get("line")
            if cline is None:
                continue
            cline = int(cline)
            if cline in state.used_corpus_lines:
                continue
            region = str(hit.get("match_region") or "")
            print(
                f"     try corpus line={cline} region={region} "
                f"task={hit.get('task_id') or '-'}",
                flush=True,
            )

            try:
                prep = client.mid_rewrite_prep(
                    line=cline,
                    gold=gold,
                    expression=expression,
                    corpus_path=hit_corpus,
                )
            except Exception as exc:
                print(f"       MID prep error → skip: {exc}", flush=True)
                state.skipped.append({
                    "test_line": test_line,
                    "corpus_line": cline,
                    "reason": f"mid_prep_error: {exc}",
                })
                continue

            ok, rewrite_id, note = _mid_decision(prep, region)
            if not ok:
                print(f"       {note} → next hit", flush=True)
                state.skipped.append({
                    "test_line": test_line,
                    "corpus_line": cline,
                    "reason": note,
                    "mid_reason": prep.get("reason"),
                    "detail": prep.get("detail"),
                })
                continue
            print(f"       MID ok: {note}", flush=True)

            try:
                client.bind_rewrite(
                    line=cline,
                    rewrite_id=rewrite_id,
                    corpus_path=hit_corpus,
                )
                accept = client.annotate_and_accept(
                    line=cline,
                    corpus_path=hit_corpus,
                )
                continue_rows = _duplicate_to_total(
                    client,
                    line=cline,
                    corpus_path=hit_corpus,
                    total_copies=state.copies_per_hit,
                )
            except Exception as exc:
                print(f"       annotate/duplicate failed → next hit: {exc}", flush=True)
                state.skipped.append({
                    "test_line": test_line,
                    "corpus_line": cline,
                    "reason": f"annotate_or_dup_failed: {exc}",
                })
                continue

            state.used_corpus_lines.add(cline)
            rec = {
                "test_line": test_line,
                "task_id": task_id,
                "corpus_line": cline,
                "corpus_path": hit_corpus,
                "match_region": region,
                "expression": expression,
                "expr_name": expr["name"],
                "mid_note": note,
                "rewrite_id": rewrite_id,
                "n_continue_edges": accept.get("n_continue_edges"),
                "continue_path": accept.get("continue_path"),
                "continue_rows": continue_rows,
                "dry_run": bool(accept.get("dry_run")),
            }
            state.annotated.append(rec)
            state.processed_test_lines.append(test_line)
            print(
                f"       ✓ 1 hit annotated + duplicated → {continue_rows} continue rows "
                f"(ok_tests={state.n_ok_tests}, continue_rows≈{state.n_continue_rows}) "
                f"path={accept.get('continue_path') or '-'}",
                flush=True,
            )
            print(f"  [done] test line={test_line} → next test", flush=True)
            _save_state(state_path, state)
            return

        print(
            "     all hits failed MID/annotate → try next (looser) expression",
            flush=True,
        )

    state.processed_test_lines.append(test_line)
    print(f"  [done] no usable train sample for test line={test_line}", flush=True)
    _save_state(state_path, state)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Automate raw_ce → LLM expressions → corpus search (tight→loose) "
            "→ MID rewrite → annotate 1 hit → duplicate N rows → next test"
        ),
    )
    p.add_argument(
        "--input", "-i",
        required=True,
        help="raw_ce JSONL (or JSON array) path",
    )
    p.add_argument(
        "--copies", "-n",
        type=int,
        default=20,
        help=(
            "per successful test: write this many identical continue-train rows "
            "(1 annotate + N-1 duplicates; default 20)"
        ),
    )
    p.add_argument(
        "--max-tests",
        type=int,
        default=0,
        help="stop after this many successful tests (0 = process all)",
    )
    p.add_argument(
        "--start-line",
        type=int,
        default=1,
        help="1-based first raw row to process",
    )
    p.add_argument(
        "--end-line",
        type=int,
        default=0,
        help="1-based last raw row (0 = until EOF)",
    )
    p.add_argument(
        "--eif-url",
        default="http://127.0.0.1:8766",
        help="ttav_bundle_api base URL",
    )
    p.add_argument(
        "--viewer-url",
        default="http://127.0.0.1:8765",
        help="annotation-viewer API base URL (not the Vite 5275 UI)",
    )
    p.add_argument(
        "--corpus-path",
        default="",
        help="override EIF_LLM_TRAIN_CORPUS for search / MID prep",
    )
    p.add_argument(
        "--annotate",
        choices=("graphsignal", "llm-semantic", "none"),
        default="graphsignal",
        help="annotation backend after MID rewrite (default: graphsignal)",
    )
    p.add_argument(
        "--graphsignal-use-llm",
        choices=("auto", "1", "0"),
        default="auto",
        help="force GraphSignal LLM on/off (default: server auto)",
    )
    p.add_argument(
        "--tight-first",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="try LLM expressions from narrowest to widest (default: true)",
    )
    p.add_argument("--top-k", type=int, default=15, help="corpus hits per expression")
    p.add_argument("--max-scan", type=int, default=0, help="max corpus lines to scan (0=all)")
    p.add_argument(
        "--state",
        default="",
        help="JSON checkpoint path for resume (default: beside input)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="run retrieve/search/MID only; do not write continue JSONL",
    )
    p.add_argument(
        "--skip-processed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="skip test lines listed in state file (default: true)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = Path(args.input).expanduser()
    if not input_path.is_file():
        alt = REPO_ROOT / args.input
        if alt.is_file():
            input_path = alt
        else:
            print(f"input not found: {args.input}", file=sys.stderr)
            return 2

    if args.state:
        state_path = Path(args.state).expanduser()
        if not state_path.is_absolute():
            cand = REPO_ROOT / state_path
            state_path = cand if not state_path.exists() else state_path
    else:
        state_path = input_path.with_suffix(input_path.suffix + ".auto_pipeline_state.json")

    gs_llm: bool | None
    if args.graphsignal_use_llm == "auto":
        gs_llm = None
    else:
        gs_llm = args.graphsignal_use_llm == "1"

    copies = max(1, int(args.copies))
    client = PipelineClient(
        eif_url=args.eif_url,
        viewer_url=args.viewer_url,
        corpus_path=args.corpus_path or None,
        annotate=args.annotate,
        top_k=max(1, int(args.top_k)),
        max_scan=int(args.max_scan) if int(args.max_scan) > 0 else None,
        graphsignal_use_llm=gs_llm,
        dry_run=bool(args.dry_run),
    )
    client.ping()

    rows = _load_raw_rows(input_path)
    print(
        f"[input] {input_path} rows={len(rows)} "
        f"copies_per_hit={copies} max_tests={args.max_tests or 'all'}",
        flush=True,
    )
    print(f"[apis] eif={client.eif_url} viewer={client.viewer_url}", flush=True)
    print(
        f"[opts] annotate={args.annotate} tight_first={args.tight_first} "
        f"dry_run={args.dry_run} state={state_path}",
        flush=True,
    )

    state = _load_state(
        state_path if state_path else None,
        copies_per_hit=copies,
        max_tests=int(args.max_tests),
    )
    processed = set(state.processed_test_lines) if args.skip_processed else set()

    end_line = int(args.end_line) or 10**12
    for test_line, row in rows:
        if test_line < int(args.start_line) or test_line > end_line:
            continue
        if state.done():
            break
        if test_line in processed:
            print(f"[skip] test line={test_line} already in state", flush=True)
            continue
        process_test_row(
            client,
            state,
            test_line=test_line,
            row=row,
            tight_first=bool(args.tight_first),
            state_path=state_path,
        )

    _save_state(state_path, state)
    print(
        f"\n=== finished: ok_tests={state.n_ok_tests} "
        f"continue_rows≈{state.n_continue_rows} "
        f"skipped_events={len(state.skipped)} state={state_path} ===",
        flush=True,
    )
    return 0 if state.n_ok_tests > 0 or args.dry_run else 1


if __name__ == "__main__":
    raise SystemExit(main())
