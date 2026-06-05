from __future__ import annotations

import argparse
import csv
import json
import os
from glob import glob
from statistics import mean
from typing import Any

def _read_json(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _round_floats(obj: Any, digits: int = 6) -> Any:
    if isinstance(obj, float):
        return round(obj, digits)
    if isinstance(obj, list):
        return [_round_floats(x, digits) for x in obj]
    if isinstance(obj, dict):
        return {k: _round_floats(v, digits) for k, v in obj.items()}
    return obj


def _write_json(path: str, payload: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_round_floats(payload, 6), f, indent=2, ensure_ascii=False)


def _metric_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    values: dict[str, list[float]] = {}
    for row in rows:
        for name, value in (row.get("metrics") or {}).items():
            if isinstance(value, (int, float)):
                values.setdefault(name, []).append(float(value))
    return {name: mean(vals) for name, vals in sorted(values.items()) if vals}


def _experiment_key(payload: dict[str, Any], fallback_task_id: str) -> tuple[int | None, str]:
    meta = payload.get("experiment_meta") or {}
    sample_index = meta.get("test_sample_index")
    if isinstance(sample_index, int):
        idx = sample_index
    else:
        idx = None
    task_id = str(meta.get("task_id") or fallback_task_id)
    return idx, task_id


def _task_id_from_filename(path: str, suffix: str) -> str:
    name = os.path.basename(path)
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return os.path.splitext(name)[0]


def _summarize_feature_file(path: str) -> dict[str, Any]:
    payload = _read_json(path)
    fallback_task_id = _task_id_from_filename(path, "_feature.json")
    sample_index, task_id = _experiment_key(payload, fallback_task_id)
    rows = payload.get("feature_attribution") or []
    return {
        "sample_index": sample_index,
        "task_id": task_id,
        "file": path,
        "target_count": len(rows),
        "metric_means": _metric_means(rows),
        "targets": [
            {
                "target_token_index": row.get("target_token_index"),
                "target_token": row.get("target_token"),
                "metrics": row.get("metrics") or {},
            }
            for row in rows
        ],
    }


def _summarize_data_file(path: str) -> dict[str, Any]:
    payload = _read_json(path)
    fallback_task_id = _task_id_from_filename(path, "_data.json")
    sample_index, task_id = _experiment_key(payload, fallback_task_id)
    data = payload.get("data_coarse_attribution") or {}
    sample_to_sample = data.get("sample_to_sample") or {}
    sample_to_token = data.get("sample_to_token") or []
    oracle = sample_to_sample.get("oracle") or {}
    return {
        "sample_index": sample_index,
        "task_id": task_id,
        "file": path,
        "granularity": data.get("granularity"),
        "sample_to_sample_metrics": sample_to_sample.get("metrics") or {},
        "sample_to_sample_oracle": {
            "candidate_universe_size": oracle.get("candidate_universe_size"),
            "scored_candidate_count": oracle.get("scored_candidate_count"),
            "skipped_oom_count": oracle.get("skipped_oom_count"),
            "coarse_scoring": oracle.get("coarse_scoring"),
            "effect_reduction": oracle.get("effect_reduction"),
        },
        "sample_to_token_metric_means": _metric_means(sample_to_token),
        "sample_to_token_count": len(sample_to_token),
    }


def _read_status(path: str | None) -> list[dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return [dict(row) for row in reader]


def _first_metric(metrics: dict[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in metrics:
            return metrics[name]
    return None


def _first_metric_with_prefix(metrics: dict[str, Any], prefix: str) -> Any:
    for name in sorted(metrics):
        if name.startswith(prefix):
            return metrics[name]
    return None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _write_markdown(path: str, report: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    samples = report["samples"]
    lines = [
        "# Attribution Batch Report",
        "",
        f"- Feature files: {report['summary']['feature_file_count']}",
        f"- Data files: {report['summary']['data_file_count']}",
        f"- Status records: {report['summary']['status_record_count']}",
        "",
        "| index | task_id | feature targets | group eff@3 | group logdrop@3 | group reverse@5 | group tokens@5 | mass logdrop | mass reverse | mass tokens | feature eff@10 | feature logdrop@10 | feature reverse@10 | feature ndcg@10 | feature recall@10 | feature aopc@10 | data ndcg@100 | data recall@100/top50 | data group effect@10 | data group positive@10 | data scored | OOM skipped |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for sample in samples:
        feature = sample.get("feature") or {}
        data = sample.get("data") or {}
        feature_metrics = feature.get("metric_means") or {}
        data_metrics = data.get("sample_to_sample_metrics") or {}
        oracle = data.get("sample_to_sample_oracle") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(sample.get("sample_index")),
                    _fmt(sample.get("task_id")),
                    _fmt(feature.get("target_count")),
                    _fmt(
                        _first_metric(
                            feature_metrics,
                            (
                                "group_effectiveness_logprob_drop@3_tau0.5",
                                "group_effectiveness_logprob_drop@3_tau0.2",
                                "group_effectiveness_prob_drop@3_tau0.2",
                                "group_effectiveness_logprob_drop@5_tau0.5",
                            ),
                        )
                    ),
                    _fmt(_first_metric(feature_metrics, ("group_logprob_drop@3", "group_logprob_drop@5"))),
                    _fmt(_first_metric(feature_metrics, ("group_reverse_logprob_drop@5", "group_reverse_logprob_drop@10"))),
                    _fmt(_first_metric(feature_metrics, ("group_source_token_count@5", "group_source_token_count@10"))),
                    _fmt(_first_metric_with_prefix(feature_metrics, "group_mass_logprob_drop@")),
                    _fmt(_first_metric_with_prefix(feature_metrics, "group_mass_reverse_logprob_drop@")),
                    _fmt(_first_metric_with_prefix(feature_metrics, "group_mass_source_token_count@")),
                    _fmt(
                        _first_metric(
                            feature_metrics,
                            (
                                "effectiveness_logprob_drop@10_tau0.2",
                                "effectiveness_prob_drop@10_tau0.2",
                                "effectiveness_logprob_drop@10_tau0.1",
                                "effectiveness_logprob_drop@5_tau0.2",
                            ),
                        )
                    ),
                    _fmt(_first_metric(feature_metrics, ("mean_logprob_drop@10", "mean_logprob_drop@5"))),
                    _fmt(_first_metric(feature_metrics, ("reverse_rate_logprob_drop@10", "reverse_rate_logprob_drop@5"))),
                    _fmt(_first_metric(feature_metrics, ("ndcg@10", "ndcg@20", "ndcg@5"))),
                    _fmt(_first_metric(feature_metrics, ("recall@10", "recall@20", "recall@5"))),
                    _fmt(_first_metric(feature_metrics, ("aopc@10", "aopc@20", "aopc@5"))),
                    _fmt(_first_metric(data_metrics, ("ndcg@100", "ndcg@50", "ndcg@10"))),
                    _fmt(
                        _first_metric(
                            data_metrics,
                            (
                                "recall@100_oracle_top50",
                                "recall@50_oracle_top50",
                                "recall@100_oracle_top20",
                                "recall@50_oracle_top20",
                            ),
                        )
                    ),
                    _fmt(_first_metric(data_metrics, ("group_data_effect@10", "group_data_effect@5", "group_data_effect@3"))),
                    _fmt(_first_metric(data_metrics, ("group_positive_data_effect@10", "group_positive_data_effect@5", "group_positive_data_effect@3"))),
                    _fmt(oracle.get("scored_candidate_count")),
                    _fmt(oracle.get("skipped_oom_count")),
                ]
            )
            + " |"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def build_report(results_dir: str, status_tsv: str | None) -> dict[str, Any]:
    feature_files = sorted(glob(os.path.join(results_dir, "feature", "*_feature.json")))
    data_files = sorted(glob(os.path.join(results_dir, "data", "*_data.json")))
    status_records = _read_status(status_tsv)

    by_task: dict[str, dict[str, Any]] = {}
    for path in feature_files:
        feature = _summarize_feature_file(path)
        task_id = feature["task_id"]
        row = by_task.setdefault(task_id, {"task_id": task_id})
        row["sample_index"] = feature.get("sample_index")
        row["feature"] = feature

    for path in data_files:
        data = _summarize_data_file(path)
        task_id = data["task_id"]
        row = by_task.setdefault(task_id, {"task_id": task_id})
        if row.get("sample_index") is None:
            row["sample_index"] = data.get("sample_index")
        row["data"] = data

    samples = sorted(
        by_task.values(),
        key=lambda row: (
            row.get("sample_index") is None,
            row.get("sample_index") if row.get("sample_index") is not None else 10**12,
            row.get("task_id") or "",
        ),
    )
    return {
        "summary": {
            "results_dir": os.path.abspath(results_dir),
            "feature_file_count": len(feature_files),
            "data_file_count": len(data_files),
            "status_record_count": len(status_records),
            "sample_count": len(samples),
        },
        "status_records": status_records,
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize batch attribution evaluation outputs.")
    parser.add_argument("--results-dir", default="attribution_results")
    parser.add_argument("--status-tsv", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    report = build_report(args.results_dir, args.status_tsv)
    output_json = args.output_json or os.path.join(
        args.results_dir,
        "reports",
        "attribution_batch_report.json",
    )
    output_md = args.output_md or os.path.join(
        args.results_dir,
        "reports",
        "attribution_batch_report.md",
    )
    _write_json(output_json, report)
    _write_markdown(output_md, report)
    print(f"[batch-report] wrote {output_json}")
    print(f"[batch-report] wrote {output_md}")


if __name__ == "__main__":
    main()
