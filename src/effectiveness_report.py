from __future__ import annotations

import argparse
import json
import math
import os
from glob import glob
from statistics import mean, median
from typing import Any


def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_round(payload), f, indent=2, ensure_ascii=False)


def _round(obj: Any, digits: int = 6) -> Any:
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, list):
        return [_round(x, digits) for x in obj]
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    return obj


def _parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def _parse_floats(raw: str) -> tuple[float, ...]:
    return tuple(float(x.strip()) for x in raw.split(",") if x.strip())


def _fmt_tau(value: float) -> str:
    return f"{float(value):g}"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return values[lo]
    return values[lo] * (hi - pos) + values[hi] * (pos - lo)


def _summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "p25": _percentile(values, 0.25),
        "p75": _percentile(values, 0.75),
        "min": min(values),
        "max": max(values),
    }


def _metric_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for key, value in (row.get("metrics") or {}).items():
            if isinstance(value, (int, float)):
                values.setdefault(key, []).append(float(value))
    return {key: mean(vals) for key, vals in sorted(values.items()) if vals}


def _sample_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    index = row.get("sample_index")
    if isinstance(index, int):
        return (index, str(row.get("task_id", "")))
    if isinstance(index, str) and index.isdigit():
        return (int(index), str(row.get("task_id", "")))
    return (10**12, str(row.get("task_id", "")))


def _summarize_feature(results_dir: str) -> dict[str, Any]:
    files = sorted(glob(os.path.join(results_dir, "feature", "*_feature.json")))
    target_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    for path in files:
        payload = _read_json(path)
        meta = payload.get("experiment_meta") or {}
        rows = payload.get("feature_attribution") or []
        sample_rows.append({
            "sample_index": meta.get("test_sample_index"),
            "task_id": meta.get("task_id"),
            "target_count": len(rows),
            "metric_means": _metric_means(rows),
            "file": path,
        })
        for row in rows:
            target_rows.append({
                "sample_index": meta.get("test_sample_index"),
                "task_id": meta.get("task_id"),
                "target_token_index": row.get("target_token_index"),
                "target_token": row.get("target_token"),
                "base_ce_loss": row.get("base_ce_loss"),
                "base_prob": row.get("base_prob"),
                "metrics": row.get("metrics") or {},
            })

    metric_values: dict[str, list[float]] = {}
    for row in target_rows:
        for key, value in row["metrics"].items():
            if isinstance(value, (int, float)):
                metric_values.setdefault(key, []).append(float(value))
    sample_rows.sort(key=_sample_sort_key)
    return {
        "file_count": len(files),
        "sample_count": len(sample_rows),
        "target_count": len(target_rows),
        "target_metric_summary": {
            key: _summarize(vals)
            for key, vals in sorted(metric_values.items())
        },
        "samples": sample_rows,
    }


def _data_effectiveness_for_sample(
    method_top: list[dict[str, Any]],
    group_effects: list[dict[str, Any]],
    *,
    k_values: tuple[int, ...],
    thresholds: tuple[float, ...],
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    effects = [float(row.get("oracle_effect", 0.0)) for row in method_top]
    for k in k_values:
        kk = min(int(k), len(effects))
        if kk <= 0:
            continue
        top_effects = effects[:kk]
        metrics[f"mean_data_effect@{kk}"] = mean(top_effects)
        metrics[f"positive_rate_data_effect@{kk}"] = (
            sum(1 for value in top_effects if value > 0.0) / kk
        )
        for threshold in thresholds:
            suffix = _fmt_tau(threshold)
            count = sum(1 for value in top_effects if value >= float(threshold))
            metrics[f"data_effectiveness@{kk}_tau{suffix}"] = count / kk
            metrics[f"data_hit@{kk}_tau{suffix}"] = 1.0 if count > 0 else 0.0
    for group in group_effects:
        k = group.get("k")
        effect = group.get("data_effect")
        if not isinstance(k, int) or not isinstance(effect, (int, float)):
            continue
        effect = float(effect)
        metrics[f"data_group_effect@{k}"] = effect
        metrics[f"data_group_positive@{k}"] = 1.0 if effect > 0.0 else 0.0
        for threshold in thresholds:
            suffix = _fmt_tau(threshold)
            metrics[f"data_group_effectiveness@{k}_tau{suffix}"] = (
                1.0 if effect >= float(threshold) else 0.0
            )
    return metrics


def _summarize_data(
    results_dir: str,
    *,
    k_values: tuple[int, ...],
    thresholds: tuple[float, ...],
) -> dict[str, Any]:
    files = sorted(glob(os.path.join(results_dir, "data", "*_data.json")))
    sample_rows: list[dict[str, Any]] = []
    for path in files:
        payload = _read_json(path)
        meta = payload.get("experiment_meta") or {}
        sample_to_sample = (
            (payload.get("data_coarse_attribution") or {}).get("sample_to_sample") or {}
        )
        method_top = sample_to_sample.get("method_top") or []
        group_effects = sample_to_sample.get("group_effects") or []
        oracle = sample_to_sample.get("oracle") or {}
        metrics = _data_effectiveness_for_sample(
            method_top,
            group_effects,
            k_values=k_values,
            thresholds=thresholds,
        )
        sample_rows.append({
            "sample_index": meta.get("test_sample_index"),
            "task_id": meta.get("task_id"),
            "method_top_count": len(method_top),
            "group_effect_count": len(group_effects),
            "oracle": {
                "candidate_universe_size": oracle.get("candidate_universe_size"),
                "scored_candidate_count": oracle.get("scored_candidate_count"),
                "skipped_oom_count": oracle.get("skipped_oom_count"),
                "coarse_scoring": oracle.get("coarse_scoring"),
                "effect_reduction": oracle.get("effect_reduction"),
            },
            "metrics": metrics,
            "file": path,
        })

    metric_values: dict[str, list[float]] = {}
    for row in sample_rows:
        for key, value in row["metrics"].items():
            metric_values.setdefault(key, []).append(float(value))
    sample_rows.sort(key=_sample_sort_key)
    return {
        "file_count": len(files),
        "sample_count": len(sample_rows),
        "sample_metric_summary": {
            key: _summarize(vals)
            for key, vals in sorted(metric_values.items())
        },
        "samples": sample_rows,
    }


def _write_markdown(path: str, report: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    feature_summary = report["feature"].get("target_metric_summary") or {}
    data_summary = report["data"].get("sample_metric_summary") or {}

    def get(summary: dict[str, Any], key: str) -> str:
        value = (summary.get(key) or {}).get("mean")
        return "" if value is None else f"{float(value):.4f}"

    def get_first(summary: dict[str, Any], prefix: str) -> str:
        for key in sorted(summary):
            if key.startswith(prefix):
                value = (summary.get(key) or {}).get("mean")
                return "" if value is None else f"{float(value):.4f}"
        return ""

    def metric_first(metrics: dict[str, Any], prefix: str) -> float:
        for key in sorted(metrics):
            if key.startswith(prefix):
                value = metrics.get(key)
                return float(value) if isinstance(value, (int, float)) else 0.0
        return 0.0

    lines = [
        "# Effectiveness Attribution Report",
        "",
        f"- Feature files: {report['feature']['file_count']}",
        f"- Feature targets: {report['feature']['target_count']}",
        f"- Data files: {report['data']['file_count']}",
        "",
        "## Key Metrics",
        "",
        "| metric | mean |",
        "| --- | ---: |",
        f"| feature group effectiveness@5 tau0.5 | {get(feature_summary, 'group_effectiveness_logprob_drop@5_tau0.5')} |",
        f"| feature group effectiveness@10 tau1 | {get(feature_summary, 'group_effectiveness_logprob_drop@10_tau1')} |",
        f"| feature group logprob drop@5 | {get(feature_summary, 'group_logprob_drop@5')} |",
        f"| feature group reverse@5 | {get(feature_summary, 'group_reverse_logprob_drop@5')} |",
        f"| feature group source tokens@5 | {get(feature_summary, 'group_source_token_count@5')} |",
        f"| feature mass group logprob drop | {get_first(feature_summary, 'group_mass_logprob_drop@')} |",
        f"| feature mass group reverse | {get_first(feature_summary, 'group_mass_reverse_logprob_drop@')} |",
        f"| feature mass group source tokens | {get_first(feature_summary, 'group_mass_source_token_count@')} |",
        f"| data effectiveness@10 tau0.001 | {get(data_summary, 'data_effectiveness@10_tau0.001')} |",
        f"| data hit@10 tau0.001 | {get(data_summary, 'data_hit@10_tau0.001')} |",
        f"| data mean effect@10 | {get(data_summary, 'mean_data_effect@10')} |",
        f"| data group effectiveness@10 tau0.001 | {get(data_summary, 'data_group_effectiveness@10_tau0.001')} |",
        f"| data group effect@10 | {get(data_summary, 'data_group_effect@10')} |",
        "",
        "## Feature Samples",
        "",
        "| index | task_id | targets | group eff@5 tau0.5 | group eff@10 tau1 | group drop@5 | group reverse@5 | group tokens@5 | mass drop | mass tokens |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["feature"].get("samples") or []:
        metrics = row.get("metric_means") or {}
        lines.append(
            "| "
            + " | ".join([
                str(row.get("sample_index", "")),
                str(row.get("task_id", "")),
                str(row.get("target_count", "")),
                f"{float(metrics.get('group_effectiveness_logprob_drop@5_tau0.5', 0.0)):.4f}",
                f"{float(metrics.get('group_effectiveness_logprob_drop@10_tau1', 0.0)):.4f}",
                f"{float(metrics.get('group_logprob_drop@5', 0.0)):.4f}",
                f"{float(metrics.get('group_reverse_logprob_drop@5', 0.0)):.4f}",
                f"{float(metrics.get('group_source_token_count@5', 0.0)):.2f}",
                f"{metric_first(metrics, 'group_mass_logprob_drop@'):.4f}",
                f"{metric_first(metrics, 'group_mass_source_token_count@'):.2f}",
            ])
            + " |"
        )
    lines.extend([
        "",
        "## Data Samples",
        "",
        "| index | task_id | top count | groups | data eff@10 tau0.001 | data hit@10 tau0.001 | mean effect@10 | group eff@10 tau0.001 | group effect@10 |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in report["data"].get("samples") or []:
        metrics = row.get("metrics") or {}
        lines.append(
            "| "
            + " | ".join([
                str(row.get("sample_index", "")),
                str(row.get("task_id", "")),
                str(row.get("method_top_count", "")),
                str(row.get("group_effect_count", "")),
                f"{float(metrics.get('data_effectiveness@10_tau0.001', 0.0)):.4f}",
                f"{float(metrics.get('data_hit@10_tau0.001', 0.0)):.4f}",
                f"{float(metrics.get('mean_data_effect@10', 0.0)):.6f}",
                f"{float(metrics.get('data_group_effectiveness@10_tau0.001', 0.0)):.4f}",
                f"{float(metrics.get('data_group_effect@10', 0.0)):.6f}",
            ])
            + " |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    data_k_values = _parse_ints(args.data_k_values)
    data_thresholds = _parse_floats(args.data_thresholds)
    return {
        "summary": {
            "results_dir": os.path.abspath(args.results_dir),
            "data_k_values": data_k_values,
            "data_thresholds": data_thresholds,
        },
        "feature": _summarize_feature(args.results_dir),
        "data": _summarize_data(
            args.results_dir,
            k_values=data_k_values,
            thresholds=data_thresholds,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize feature/data effectiveness results.")
    parser.add_argument("--results-dir", default="attribution_results_effectiveness")
    parser.add_argument("--data-k-values", default="10,50,100")
    parser.add_argument("--data-thresholds", default="0.0005,0.001,0.002,0.005")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    report = build_report(args)
    output_json = args.output_json or os.path.join(
        args.results_dir,
        "reports",
        "effectiveness_report.json",
    )
    output_md = args.output_md or os.path.join(
        args.results_dir,
        "reports",
        "effectiveness_report.md",
    )
    _write_json(output_json, report)
    _write_markdown(output_md, report)
    print(f"[effectiveness-report] wrote {output_json}")
    print(f"[effectiveness-report] wrote {output_md}")


if __name__ == "__main__":
    main()
