#!/usr/bin/env python3
"""Stage-2 ablations on the 100 Go FIM mutation queries.

Does not call 8766. Each query is annotated once with the same vLLM client
the semantic retriever uses. Relation / pattern / role phrases are embedded
once. The four field scores against the 10k semantic corpus are cached, then
every weight vector is a local dot product.

Two experiments, in this order
------------------------------
weights
    Fix w_rel in {0.2, 0.4, 0.6, 0.8}. Grid w_pat, w_role, w_op on the
    simplex w_pat + w_role + w_op = 1 - w_rel, step 0.1 by default.
    The best point is the one with the highest Hit@1, then Hit@5, Hit@10, MRR.
field
    Start from that best weight vector. Full keeps all four. Each other row
    drops one of relations / pattern / role / operations and renormalizes
    what remains so the weights sum to 1.

Hit@k and MRR use the parent inside the top-10, same rule as
run_mutation_attribution.py. A parent outside the top-10 contributes 0 to MRR.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIELDS = ("relation", "pattern", "role", "operations")
KS = (1, 5, 10)
DEFAULT_REL = (0.2, 0.4, 0.6, 0.8)


def _hydrate() -> None:
    try:
        from src.gold_live_attribution import _hydrate_eif_env

        _hydrate_eif_env(force_file=True)
    except Exception as exc:
        print(f"[env] hydrate skipped: {exc}", flush=True)


def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _fingerprint(row: dict) -> str:
    blob = (
        str(row.get("prompt") or "")
        + "\n"
        + str(row.get("middle") or row.get("response") or "")
    )
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _default_mutations() -> Path:
    server = Path("/mnt/md124/jiaxin/training_code/data/go_fim_mutations.jsonl")
    local = HERE / "go_fim_mutations.jsonl"
    if server.is_file():
        return server
    return local


def _default_semantic() -> Path:
    raw = _env("EIF_LLM_SEMANTIC_CORPUS")
    if raw:
        return Path(raw)
    return Path(
        "/mnt/md124/jiaxin/training_code/data/csn_go_train_fim_10k.semantic.vllm32b.jsonl"
    )


def _is_parent(line: int, task: str, truth_line: int, truth_task: str) -> bool:
    if truth_task and task and task == truth_task:
        return True
    return line == truth_line


def _normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        raise ValueError(f"weights sum to 0: {weights}")
    return {key: max(0.0, float(weights.get(key, 0.0))) / total for key in FIELDS}


def _drop_from(weights: dict[str, float], field: str) -> dict[str, float]:
    kept = {key: float(weights.get(key, 0.0)) for key in FIELDS if key != field}
    return _normalize(kept)


def _simplex(total: float, step: float) -> list[tuple[float, float, float]]:
    """Non-negative (pat, role, op) triples that sum to ``total`` on ``step``."""
    if step <= 0:
        raise ValueError("grid step must be positive")
    units = int(round(total / step))
    if abs(units * step - total) > 1e-6:
        raise ValueError(f"{total} is not a multiple of step {step}")
    points = []
    for pat_u in range(units + 1):
        for role_u in range(units - pat_u + 1):
            op_u = units - pat_u - role_u
            points.append((pat_u * step, role_u * step, op_u * step))
    return points


def _annotate_queries(queries: list[dict], cache_path: Path) -> list[dict]:
    from src.fim_semantic_schema import semantic_export_repr
    from src.llm_semantic_retrieval import (
        call_llm_semantic_analyze,
        prepare_llm_train_query,
    )

    cached: dict[str, dict] = {}
    if cache_path.is_file():
        for row in _load_jsonl(cache_path):
            mid = str(row.get("mutation_id") or "")
            if mid:
                cached[mid] = row

    cards = []
    for index, query in enumerate(queries, start=1):
        mid = str(query.get("mutation_id") or f"row{index}")
        fingerprint = _fingerprint(query)
        previous = cached.get(mid)
        if previous and previous.get("fingerprint") == fingerprint and previous.get("semantic"):
            cards.append(previous)
            print(f"[annotate] {index}/{len(queries)} {mid} cached", flush=True)
            continue
        prompt = str(query.get("prompt") or "")
        gold = str(query.get("middle") or query.get("response") or "")
        prepared = prepare_llm_train_query(prompt, gold)
        last_error = ""
        semantic = None
        for attempt in range(1, 4):
            try:
                llm_out = call_llm_semantic_analyze(
                    fim_prompt=prepared["fim_problem_surface"],
                    gold_completion=prepared["gold_mid_completion"],
                    language="go",
                )
                semantic = semantic_export_repr(llm_out.get("semantic") or {})
                break
            except Exception as exc:
                last_error = str(exc)
                print(f"[annotate] {mid} attempt {attempt} failed: {exc}", flush=True)
                time.sleep(min(8, 2 ** attempt))
        if semantic is None:
            raise SystemExit(f"annotate failed for {mid}: {last_error}")
        card = {
            "mutation_id": mid,
            "fingerprint": fingerprint,
            "ground_truth_line": int(query["ground_truth_line"]),
            "ground_truth_task_id": str(query.get("ground_truth_task_id") or ""),
            "semantic": semantic,
        }
        cached[mid] = card
        cards.append(card)
        _write_jsonl(cache_path, [cached[str(q.get("mutation_id") or f"row{i}")]
                                  for i, q in enumerate(queries, start=1)
                                  if str(q.get("mutation_id") or f"row{i}") in cached])
        print(f"[annotate] {index}/{len(queries)} {mid} ok", flush=True)
    return cards


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_corpus(path: Path) -> tuple[list[dict], Any, list[str]]:
    import numpy as np

    from src.fim_semantic_schema import normalize_semantic_repr

    docs = []
    lines = []
    tasks = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            try:
                source_line = int(row.get("source_line"))
            except (TypeError, ValueError):
                source_line = index
            docs.append(normalize_semantic_repr(row))
            lines.append(source_line)
            tasks.append(str(row.get("task_id") or ""))
    return docs, np.asarray(lines, dtype=np.int32), tasks


def _endpoint_sim(queries: list[dict], docs: list[dict]):
    from src.fim_semantic_index import build_endpoint_sim, embed_model_id
    from src.fim_semantic_schema import (
        collect_semantic_embed_phrases,
        lexical_endpoint_sim,
    )

    seen: set[str] = set()
    phrases: list[str] = []
    for sem in list(queries) + docs:
        for phrase in collect_semantic_embed_phrases(sem):
            if phrase not in seen:
                seen.add(phrase)
                phrases.append(phrase)
    model = embed_model_id()
    print(f"[embed] phrases={len(phrases)} model={model or 'lexical'}", flush=True)
    if not model:
        return lexical_endpoint_sim, "lexical"
    try:
        sim = build_endpoint_sim(phrases, model)
    except Exception as exc:
        print(f"[embed] endpoint embed failed ({exc}); lexical endpoints", flush=True)
        return lexical_endpoint_sim, "lexical"
    return sim, "relation_embed"


def _components(
    cards: list[dict],
    docs: list[dict],
    sim,
) -> Any:
    import numpy as np

    from src.fim_semantic_schema import normalize_semantic_repr, semantic_struct_components

    parts = np.zeros((len(cards), len(docs), len(FIELDS)), dtype=np.float32)
    queries = [normalize_semantic_repr(card["semantic"]) for card in cards]
    total = len(cards)
    for qi, query in enumerate(queries):
        for di, doc in enumerate(docs):
            comp = semantic_struct_components(query, doc, endpoint_sim=sim)
            for fi, field in enumerate(FIELDS):
                parts[qi, di, fi] = float(comp[field])
        if (qi + 1) % 5 == 0 or qi + 1 == total:
            print(f"[score] {qi + 1}/{total}", flush=True)
    return parts


def _load_or_score(
    cards: list[dict],
    docs: list[dict],
    lines,
    cache_path: Path,
):
    import numpy as np

    fingerprints = [str(card["fingerprint"]) for card in cards]
    if cache_path.is_file():
        blob = np.load(cache_path, allow_pickle=False)
        cached_fp = [str(item) for item in blob["fingerprints"].tolist()]
        same_lines = np.array_equal(blob["lines"], lines)
        if (
            cached_fp == fingerprints
            and blob["parts"].shape[:2] == (len(cards), len(docs))
            and same_lines
        ):
            print(f"[score] cache hit {cache_path}", flush=True)
            align = blob["align"]
            align_text = str(align.item() if getattr(align, "shape", ()) == () else align)
            return blob["parts"], align_text
    sim, align = _endpoint_sim([card["semantic"] for card in cards], docs)
    parts = _components(cards, docs, sim)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        parts=parts,
        lines=lines,
        fingerprints=np.asarray(fingerprints),
        align=np.asarray(align),
    )
    print(f"[score] wrote {cache_path} align={align}", flush=True)
    return parts, align


def _metrics(scores, lines, tasks: list[str], truths: list[tuple[int, str]]) -> dict[str, float]:
    import numpy as np

    n = int(scores.shape[0])
    hits = {k: 0 for k in KS}
    rr_sum = 0.0
    for i in range(n):
        order = np.lexsort((lines, -scores[i]))
        truth_line, truth_task = truths[i]
        rank = 0
        for pos, doc_i in enumerate(order[:10], start=1):
            doc_i = int(doc_i)
            if _is_parent(int(lines[doc_i]), tasks[doc_i], truth_line, truth_task):
                rank = pos
                break
        for k in KS:
            if rank and rank <= k:
                hits[k] += 1
        if rank:
            rr_sum += 1.0 / rank
    return {
        "n": n,
        "hit@1": hits[1] / n if n else 0.0,
        "hit@5": hits[5] / n if n else 0.0,
        "hit@10": hits[10] / n if n else 0.0,
        "mrr": rr_sum / n if n else 0.0,
        "hit@1_count": hits[1],
        "hit@5_count": hits[5],
        "hit@10_count": hits[10],
    }


def _apply(parts, weights: dict[str, float]):
    import numpy as np

    vector = np.asarray([weights[field] for field in FIELDS], dtype=np.float32)
    return parts @ vector


def _round_weights(weights: dict[str, float]) -> dict[str, float]:
    return {key: round(float(weights[key]), 6) for key in FIELDS}


def _best_weight_row(rows: list[dict]) -> dict:
    """Highest Hit@1, then Hit@5, Hit@10, then MRR."""
    return max(rows, key=lambda row: (row["hit@1"], row["hit@5"], row["hit@10"], row["mrr"]))


def _field_rows(parts, lines, tasks, truths, base: dict[str, float]) -> list[dict]:
    specs = [("full", None, _normalize(base))]
    for field in FIELDS:
        specs.append((f"no_{field}", field, _drop_from(base, field)))
    rows = []
    for name, dropped, weights in specs:
        metrics = _metrics(_apply(parts, weights), lines, tasks, truths)
        rows.append({
            "variant": name,
            "dropped": dropped,
            "weights": _round_weights(weights),
            **metrics,
        })
    return rows


def _weight_rows(parts, lines, tasks, truths, rels: tuple[float, ...], step: float) -> list[dict]:
    rows = []
    for rel in rels:
        rest = round(1.0 - float(rel), 6)
        for pat, role, op in _simplex(rest, step):
            weights = _normalize({
                "relation": float(rel),
                "pattern": pat,
                "role": role,
                "operations": op,
            })
            metrics = _metrics(_apply(parts, weights), lines, tasks, truths)
            rows.append({
                "w_rel": round(float(rel), 6),
                "w_pat": round(pat, 6),
                "w_role": round(role, 6),
                "w_op": round(op, 6),
                "weights": _round_weights(weights),
                **metrics,
            })
        print(f"[grid] w_rel={rel:.1f} done", flush=True)
    return rows


def _print_field(rows: list[dict]) -> None:
    print(f"{'variant':<16}{'n':<6}{'Hit@1':<10}{'Hit@5':<10}{'Hit@10':<10}{'MRR':<10}")
    for row in rows:
        print(
            f"{row['variant']:<16}{row['n']:<6}"
            f"{row['hit@1']:.3f}     {row['hit@5']:.3f}     "
            f"{row['hit@10']:.3f}     {row['mrr']:.3f}"
        )


def _print_weight_peaks(rows: list[dict]) -> None:
    rels = []
    for row in rows:
        if row["w_rel"] not in rels:
            rels.append(row["w_rel"])
    print(f"{'w_rel':<8}{'metric':<10}{'best':<10}{'w_pat':<8}{'w_role':<8}{'w_op':<8}{'near':<12}")
    for rel in rels:
        slice_rows = [row for row in rows if row["w_rel"] == rel]
        for metric in ("hit@1", "hit@5", "hit@10", "mrr"):
            best = max(slice_rows, key=lambda row: (row[metric], -row["w_pat"], -row["w_role"]))
            near = sum(1 for row in slice_rows if best[metric] - row[metric] <= 0.02)
            print(
                f"{rel:<8.1f}{metric:<10}{best[metric]:<10.3f}"
                f"{best['w_pat']:<8.2f}{best['w_role']:<8.2f}{best['w_op']:<8.2f}"
                f"{near}/{len(slice_rows)}"
            )


def _write_ternary(rows: list[dict], path: Path) -> None:
    """Four triangles per metric. Vertices are pattern, role, operations."""
    import math

    rels: list[float] = []
    for row in rows:
        if row["w_rel"] not in rels:
            rels.append(row["w_rel"])
    metrics = ("hit@1", "hit@5", "hit@10", "mrr")
    side = 220
    height = int(side * math.sqrt(3) / 2)
    pad = 46
    panel_w = side + pad * 2
    panel_h = height + pad * 2 + 18
    width = panel_w * max(1, len(rels)) + 16
    svg_h = panel_h * len(metrics) + 16
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{svg_h}" '
        f'viewBox="0 0 {width} {svg_h}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for row_i, metric in enumerate(metrics):
        for col_i, rel in enumerate(rels):
            slice_rows = [item for item in rows if item["w_rel"] == rel]
            values = [float(item[metric]) for item in slice_rows]
            lo = min(values)
            hi = max(values)
            origin_x = 8 + col_i * panel_w + pad
            origin_y = 8 + row_i * panel_h + pad
            ax, ay = origin_x, origin_y + height
            bx, by = origin_x + side, origin_y + height
            cx, cy = origin_x + side / 2.0, origin_y
            parts.append(
                f'<polygon points="{ax:.1f},{ay:.1f} {bx:.1f},{by:.1f} {cx:.1f},{cy:.1f}" '
                f'fill="#f3f3f3" stroke="#222" stroke-width="1"/>'
            )
            span = hi - lo
            for item in slice_rows:
                total = item["w_pat"] + item["w_role"] + item["w_op"]
                if total <= 0:
                    continue
                x = (
                    ax * (item["w_pat"] / total)
                    + bx * (item["w_role"] / total)
                    + cx * (item["w_op"] / total)
                )
                y = (
                    ay * (item["w_pat"] / total)
                    + by * (item["w_role"] / total)
                    + cy * (item["w_op"] / total)
                )
                t = 0.5 if span < 1e-9 else (float(item[metric]) - lo) / span
                red = int(36 + 200 * t)
                blue = int(170 - 130 * t)
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" '
                    f'fill="rgb({red},70,{blue})" stroke="#fff" stroke-width="0.5">'
                    f"<title>{metric}={item[metric]:.3f} "
                    f"pat={item['w_pat']:.2f} role={item['w_role']:.2f} "
                    f"op={item['w_op']:.2f}</title></circle>"
                )
            parts.append(f'<text x="{ax - 8}" y="{ay + 16}" font-size="12">pattern</text>')
            parts.append(f'<text x="{bx - 18}" y="{by + 16}" font-size="12">role</text>')
            parts.append(f'<text x="{cx - 36}" y="{cy - 6}" font-size="12">operations</text>')
            parts.append(
                f'<text x="{origin_x - 8}" y="{origin_y - 14}" font-size="13">'
                f"{metric}  w_rel={rel:.1f}  {lo:.3f}–{hi:.3f}</text>"
            )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> int:
    _hydrate()
    parser = argparse.ArgumentParser(description="Stage-2 field and weight ablations")
    parser.add_argument("--mutations", type=Path, default=_default_mutations())
    parser.add_argument("--semantic-corpus", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=HERE / "results")
    parser.add_argument("--grid-step", type=float, default=0.1)
    parser.add_argument("--rel-weights", default=",".join(str(v) for v in DEFAULT_REL))
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    if args.semantic_corpus is None:
        args.semantic_corpus = _default_semantic()

    rels = tuple(float(part) for part in args.rel_weights.split(",") if part.strip())
    for rel in rels:
        if not 0.0 < rel < 1.0:
            print(f"w_rel must be strictly between 0 and 1, got {rel}", file=sys.stderr)
            return 2
        if abs(round((1.0 - rel) / args.grid_step) * args.grid_step - (1.0 - rel)) > 1e-6:
            print(
                f"1 - {rel} is not a multiple of grid step {args.grid_step}",
                file=sys.stderr,
            )
            return 2
    if not args.mutations.is_file():
        print(f"mutations not found: {args.mutations}", file=sys.stderr)
        return 2
    if not args.semantic_corpus.is_file():
        print(f"semantic corpus not found: {args.semantic_corpus}", file=sys.stderr)
        return 2

    queries = _load_jsonl(args.mutations)
    if args.limit > 0:
        queries = queries[: args.limit]
    print(
        f"[run] queries={len(queries)} corpus={args.semantic_corpus} "
        f"rel={list(rels)} step={args.grid_step}",
        flush=True,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    cards = _annotate_queries(queries, args.out / "ablation_query_cards.jsonl")
    docs, lines, tasks = _load_corpus(args.semantic_corpus)
    print(f"[corpus] docs={len(docs)}", flush=True)
    missing = [
        card["mutation_id"]
        for card in cards
        if card["ground_truth_task_id"] and card["ground_truth_task_id"] not in set(tasks)
    ]
    if missing:
        print(f"[corpus] parent task_id missing for {missing}", flush=True)

    os.environ["EIF_LLM_SEMANTIC_CORPUS"] = str(args.semantic_corpus)
    parts, align = _load_or_score(cards, docs, lines, args.out / "ablation_components.npz")
    truths = [
        (int(card["ground_truth_line"]), str(card["ground_truth_task_id"] or ""))
        for card in cards
    ]
    weight_rows = _weight_rows(parts, lines, tasks, truths, rels, args.grid_step)
    best = _best_weight_row(weight_rows)
    field_rows = _field_rows(parts, lines, tasks, truths, best["weights"])

    field_path = args.out / "ablation_field.json"
    weight_path = args.out / "ablation_weights.jsonl"
    field_path.write_text(
        json.dumps(
            {
                "align": align,
                "n": len(cards),
                "selected_by": ["hit@1", "hit@5", "hit@10", "mrr"],
                "selected": {
                    "w_rel": best["w_rel"],
                    "w_pat": best["w_pat"],
                    "w_role": best["w_role"],
                    "w_op": best["w_op"],
                    "hit@1": best["hit@1"],
                    "hit@5": best["hit@5"],
                    "hit@10": best["hit@10"],
                    "mrr": best["mrr"],
                },
                "note": (
                    "Grid runs first. Field ablation starts from the best grid "
                    "point and renormalizes after each field is removed. "
                    "MRR is mean reciprocal rank inside top-10."
                ),
                "rows": field_rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_jsonl(weight_path, weight_rows)
    print("[weights] peak and how many grid points sit within 0.02 of that peak")
    _print_weight_peaks(weight_rows)
    print(
        "[weights] selected "
        f"w_rel={best['w_rel']:.2f} w_pat={best['w_pat']:.2f} "
        f"w_role={best['w_role']:.2f} w_op={best['w_op']:.2f} "
        f"Hit@1={best['hit@1']:.3f} Hit@5={best['hit@5']:.3f} "
        f"Hit@10={best['hit@10']:.3f} MRR={best['mrr']:.3f}",
        flush=True,
    )
    print("[field] drop one field from the selected weights")
    _print_field(field_rows)
    ternary_path = args.out / "ablation_ternary.svg"
    _write_ternary(weight_rows, ternary_path)
    print(f"wrote {field_path}")
    print(f"wrote {weight_path}")
    print(f"wrote {ternary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
