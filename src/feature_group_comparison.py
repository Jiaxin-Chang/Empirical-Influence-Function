"""Compare feature group perturbation runs.

This script summarizes per-target group perturbation effects across result
directories and computes paired deltas on shared sample/target keys.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


TARGET_KEY = tuple[int, int]


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _sample_index(path: Path) -> int | None:
    match = re.search(r"test(\d+)_feature$", path.stem)
    return int(match.group(1)) if match else None


def _group_candidate_name(label: str, group: dict[str, Any]) -> str | None:
    if "k" in group:
        return f"{label}:group@{group['k']}"
    if "mass_threshold" in group:
        threshold = float(group["mass_threshold"])
        return f"{label}:mass@{threshold:g}"
    return None


def _group_record(
    sample_index: int,
    target: dict[str, Any],
    group: dict[str, Any],
) -> dict[str, Any] | None:
    drop = _as_float(group.get("logprob_drop"))
    if drop is None:
        return None
    target_index = target.get("target_token_index")
    if not isinstance(target_index, int):
        return None
    token_count = group.get("source_token_count")
    if token_count is None:
        token_count = len(group.get("source_token_indices") or [])
    unit_count = group.get("source_unit_count", token_count)
    return {
        "sample_index": sample_index,
        "target_token_index": target_index,
        "target_token": target.get("target_token"),
        "logprob_drop": drop,
        "prob_drop": _as_float(group.get("prob_drop")),
        "source_token_count": _as_float(token_count),
        "source_unit_count": _as_float(unit_count),
        "saliency_mass_coverage": _as_float(group.get("saliency_mass_coverage")),
        "source_tokens": group.get("source_tokens") or [],
        "source_token_indices": group.get("source_token_indices") or [],
    }


def load_run(label: str, results_dir: Path) -> dict[str, dict[TARGET_KEY, dict[str, Any]]]:
    feature_dir = results_dir / "feature"
    if not feature_dir.exists():
        raise FileNotFoundError(f"missing feature dir: {feature_dir}")

    candidates: dict[str, dict[TARGET_KEY, dict[str, Any]]] = {}
    for path in sorted(feature_dir.glob("*_feature.json")):
        sample_index = _sample_index(path)
        if sample_index is None:
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        targets = payload.get("feature_attribution") or []
        for target in targets:
            if not isinstance(target, dict):
                continue
            groups = list(target.get("group_effects") or [])
            groups.extend(target.get("saliency_mass_group_effects") or [])
            for group in groups:
                name = _group_candidate_name(label, group)
                record = _group_record(sample_index, target, group)
                if name is None or record is None:
                    continue
                key = (record["sample_index"], record["target_token_index"])
                candidates.setdefault(name, {})[key] = record
    return candidates


def summarize_candidate(name: str, rows: dict[TARGET_KEY, dict[str, Any]]) -> dict[str, Any]:
    drops = [row["logprob_drop"] for row in rows.values()]
    tokens = [
        row["source_token_count"]
        for row in rows.values()
        if row.get("source_token_count") is not None
    ]
    units = [
        row["source_unit_count"]
        for row in rows.values()
        if row.get("source_unit_count") is not None
    ]
    coverages = [
        row["saliency_mass_coverage"]
        for row in rows.values()
        if row.get("saliency_mass_coverage") is not None
    ]
    return {
        "candidate": name,
        "n": len(drops),
        "mean_logprob_drop": _mean(drops),
        "median_logprob_drop": _median(drops),
        "positive_rate": _mean([1.0 if value > 0 else 0.0 for value in drops]),
        "reverse_rate": _mean([1.0 if value < 0 else 0.0 for value in drops]),
        "effect_rate_tau0.1": _mean([1.0 if value >= 0.1 else 0.0 for value in drops]),
        "effect_rate_tau0.2": _mean([1.0 if value >= 0.2 else 0.0 for value in drops]),
        "effect_rate_tau0.5": _mean([1.0 if value >= 0.5 else 0.0 for value in drops]),
        "mean_source_token_count": _mean(tokens),
        "median_source_token_count": _median(tokens),
        "mean_source_unit_count": _mean(units),
        "mean_saliency_mass_coverage": _mean(coverages),
    }


def paired_summary(
    baseline_name: str,
    baseline_rows: dict[TARGET_KEY, dict[str, Any]],
    candidate_name: str,
    candidate_rows: dict[TARGET_KEY, dict[str, Any]],
) -> dict[str, Any]:
    keys = sorted(set(baseline_rows) & set(candidate_rows))
    base_drops = [baseline_rows[key]["logprob_drop"] for key in keys]
    candidate_drops = [candidate_rows[key]["logprob_drop"] for key in keys]
    deltas = [cand - base for cand, base in zip(candidate_drops, base_drops)]
    return {
        "baseline": baseline_name,
        "candidate": candidate_name,
        "n": len(keys),
        "baseline_mean_logprob_drop": _mean(base_drops),
        "candidate_mean_logprob_drop": _mean(candidate_drops),
        "mean_delta_logprob_drop": _mean(deltas),
        "median_delta_logprob_drop": _median(deltas),
        "win_rate": _mean([1.0 if delta > 0 else 0.0 for delta in deltas]),
        "tie_rate": _mean([1.0 if delta == 0 else 0.0 for delta in deltas]),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def _candidate_sort_key(name: str) -> tuple[str, int, float]:
    label, _, suffix = name.partition(":")
    if suffix.startswith("group@"):
        return (label, 0, float(suffix.split("@", 1)[1]))
    if suffix.startswith("mass@"):
        return (label, 1, float(suffix.split("@", 1)[1]))
    return (label, 2, 0.0)


def render_markdown(
    summaries: list[dict[str, Any]],
    paired: list[dict[str, Any]],
    nearest_budget: list[dict[str, Any]],
) -> str:
    summary_columns = [
        "candidate",
        "n",
        "mean_logprob_drop",
        "median_logprob_drop",
        "positive_rate",
        "reverse_rate",
        "effect_rate_tau0.1",
        "effect_rate_tau0.2",
        "effect_rate_tau0.5",
        "mean_source_token_count",
        "mean_saliency_mass_coverage",
    ]
    paired_columns = [
        "baseline",
        "candidate",
        "n",
        "baseline_mean_logprob_drop",
        "candidate_mean_logprob_drop",
        "mean_delta_logprob_drop",
        "median_delta_logprob_drop",
        "win_rate",
    ]
    nearest_columns = [
        "candidate",
        "candidate_mean_tokens",
        "nearest_baseline",
        "baseline_mean_tokens",
        "token_gap",
        "candidate_mean_logprob_drop",
        "baseline_mean_logprob_drop",
        "mean_delta_logprob_drop",
        "win_rate",
    ]
    parts = [
        "# Feature Group Comparison",
        "",
        "## Candidate Summary",
        "",
        _markdown_table(summaries, summary_columns),
    ]
    if nearest_budget:
        parts.extend(["", "## Nearest Baseline By Mean Source Tokens", "", _markdown_table(nearest_budget, nearest_columns)])
    if paired:
        parts.extend(["", "## Paired Comparisons", "", _markdown_table(paired, paired_columns)])
    return "\n".join(parts) + "\n"


def build_nearest_budget(
    summaries: list[dict[str, Any]],
    paired_by_key: dict[tuple[str, str], dict[str, Any]],
    baseline_candidates: list[str],
) -> list[dict[str, Any]]:
    summary_by_name = {row["candidate"]: row for row in summaries}
    baseline_summaries = [
        summary_by_name[name]
        for name in baseline_candidates
        if name in summary_by_name and summary_by_name[name].get("mean_source_token_count") is not None
    ]
    rows: list[dict[str, Any]] = []
    for summary in summaries:
        candidate = summary["candidate"]
        if candidate in baseline_candidates or summary.get("mean_source_token_count") is None:
            continue
        nearest = min(
            baseline_summaries,
            key=lambda base: abs(
                base["mean_source_token_count"] - summary["mean_source_token_count"]
            ),
            default=None,
        )
        if nearest is None:
            continue
        pair = paired_by_key.get((nearest["candidate"], candidate), {})
        rows.append(
            {
                "candidate": candidate,
                "candidate_mean_tokens": summary.get("mean_source_token_count"),
                "nearest_baseline": nearest["candidate"],
                "baseline_mean_tokens": nearest.get("mean_source_token_count"),
                "token_gap": abs(
                    nearest["mean_source_token_count"] - summary["mean_source_token_count"]
                ),
                "candidate_mean_logprob_drop": summary.get("mean_logprob_drop"),
                "baseline_mean_logprob_drop": nearest.get("mean_logprob_drop"),
                "mean_delta_logprob_drop": pair.get("mean_delta_logprob_drop"),
                "win_rate": pair.get("win_rate"),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="Run to load as LABEL=RESULTS_DIR. Can be repeated.",
    )
    parser.add_argument(
        "--baseline-candidate",
        action="append",
        default=[],
        help="Candidate name used for paired comparison, e.g. baseline:group@10.",
    )
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    all_candidates: dict[str, dict[TARGET_KEY, dict[str, Any]]] = {}
    for item in args.run:
        if "=" not in item:
            raise SystemExit(f"--run must be LABEL=RESULTS_DIR, got {item!r}")
        label, raw_dir = item.split("=", 1)
        loaded = load_run(label, Path(raw_dir))
        overlap = set(all_candidates) & set(loaded)
        if overlap:
            raise SystemExit(f"duplicate candidate names: {sorted(overlap)}")
        all_candidates.update(loaded)

    summaries = [
        summarize_candidate(name, rows)
        for name, rows in sorted(all_candidates.items(), key=lambda item: _candidate_sort_key(item[0]))
    ]
    paired: list[dict[str, Any]] = []
    for baseline_name in args.baseline_candidate:
        baseline_rows = all_candidates.get(baseline_name)
        if baseline_rows is None:
            raise SystemExit(f"missing baseline candidate: {baseline_name}")
        for candidate_name, candidate_rows in sorted(all_candidates.items(), key=lambda item: _candidate_sort_key(item[0])):
            if candidate_name == baseline_name:
                continue
            paired.append(
                paired_summary(
                    baseline_name,
                    baseline_rows,
                    candidate_name,
                    candidate_rows,
                )
            )

    paired_by_key = {
        (row["baseline"], row["candidate"]): row
        for row in paired
    }
    nearest_budget = build_nearest_budget(summaries, paired_by_key, args.baseline_candidate)
    payload = {
        "summaries": summaries,
        "nearest_budget": nearest_budget,
        "paired": paired,
    }

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[feature-group-comparison] wrote {args.output_json}")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(summaries, paired, nearest_budget), encoding="utf-8")
        print(f"[feature-group-comparison] wrote {args.output_md}")
    if not args.output_json and not args.output_md:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
