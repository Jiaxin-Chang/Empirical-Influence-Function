from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter, defaultdict
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
        if math.isnan(obj) or math.isinf(obj):
            return obj
        return round(obj, digits)
    if isinstance(obj, list):
        return [_round(x, digits) for x in obj]
    if isinstance(obj, dict):
        return {k: _round(v, digits) for k, v in obj.items()}
    return obj


def _summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0}
    sorted_values = sorted(values)
    return {
        "n": len(values),
        "mean": mean(values),
        "median": median(values),
        "min": sorted_values[0],
        "max": sorted_values[-1],
        "p25": sorted_values[int(0.25 * (len(sorted_values) - 1))],
        "p75": sorted_values[int(0.75 * (len(sorted_values) - 1))],
    }


_GO_KEYWORDS = {
    "break",
    "case",
    "chan",
    "const",
    "continue",
    "default",
    "defer",
    "else",
    "fallthrough",
    "for",
    "func",
    "go",
    "goto",
    "if",
    "import",
    "interface",
    "map",
    "new",
    "package",
    "range",
    "return",
    "select",
    "struct",
    "switch",
    "type",
    "var",
}


def _token_category(token: str) -> str:
    stripped = token.strip()
    if not stripped:
        return "whitespace"
    if stripped in _GO_KEYWORDS:
        return "go_keyword"
    if re.fullmatch(r"[{}()\[\],.;:]+", stripped):
        return "punctuation"
    if re.fullmatch(r"[-+*/%=!<>|&^~.]+", stripped):
        return "operator"
    if re.fullmatch(r"""["'`]+""", stripped):
        return "quote"
    if re.fullmatch(r"\d+(?:\.\d+)?", stripped):
        return "number"
    if any("\u4e00" <= ch <= "\u9fff" for ch in stripped):
        return "cjk"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", stripped):
        if token and token[0].isspace():
            return "identifier_word"
        return "identifier_fragment"
    if re.fullmatch(r"[A-Za-z0-9_./:-]+", stripped):
        return "mixed_fragment"
    return "other"


def _feature_files(results_dir: str) -> list[str]:
    direct = sorted(glob(os.path.join(results_dir, "*_feature.json")))
    nested = sorted(glob(os.path.join(results_dir, "feature", "*_feature.json")))
    return direct or nested


def _sample_key(meta: dict[str, Any], path: str) -> str:
    task_id = meta.get("task_id")
    if task_id:
        return str(task_id)
    idx = meta.get("test_sample_index")
    if idx is not None:
        return f"test{idx}"
    return os.path.basename(path)


def _baseline_tokens(payload: dict[str, Any]) -> list[str]:
    baseline = payload.get("test_sample_baseline") or {}
    for key in ("generated_full_tokens", "full_tokens", "ground_truth_full_tokens", "correct_full_tokens"):
        tokens = baseline.get(key)
        if isinstance(tokens, list) and tokens:
            return [str(token) for token in tokens]
    return []


def _context(tokens: list[str], index: Any, radius: int = 4) -> list[str]:
    if not isinstance(index, int):
        try:
            index = int(index)
        except (TypeError, ValueError):
            return []
    if index < 0 or index >= len(tokens):
        return []
    start = max(0, index - radius)
    end = min(len(tokens), index + radius + 1)
    context = []
    for pos in range(start, end):
        token = tokens[pos]
        context.append(f"->[{token}]<-" if pos == index else token)
    return context


def _top_examples(rows: list[dict[str, Any]], limit: int, *, reverse: bool = False) -> list[dict[str, Any]]:
    key = (lambda row: float(row.get("logprob_drop", 0.0)))
    return sorted(rows, key=key, reverse=reverse)[:limit]


def analyze_feature_results(
    results_dir: str,
    *,
    method_k: int,
    oracle_m: int,
    effect_eps: float,
    example_limit: int,
) -> dict[str, Any]:
    files = _feature_files(results_dir)
    target_count = 0
    configs = Counter()
    base_losses: list[float] = []
    group_effects_by_k: dict[int, list[float]] = defaultdict(list)
    individual_effects_by_rank: dict[int, list[float]] = defaultdict(list)
    reverse_individual: list[dict[str, Any]] = []
    reverse_groups: list[dict[str, Any]] = []
    missed_oracle: list[dict[str, Any]] = []
    hit_oracle: list[dict[str, Any]] = []
    token_category_counts: dict[str, Counter] = {
        "reverse_individual": Counter(),
        "reverse_group_sources": Counter(),
        "missed_oracle": Counter(),
        "hit_oracle": Counter(),
    }
    rows_with_individual_effects = 0
    rows_with_full_oracle = 0

    for path in files:
        payload = _read_json(path)
        meta = payload.get("experiment_meta") or {}
        tokens = _baseline_tokens(payload)
        config = meta.get("config") or {}
        config_key = tuple(
            sorted(
                (str(k), str(config.get(k)))
                for k in (
                    "feature_evaluation_mode",
                    "feature_group_only",
                    "feature_perturb_mode",
                    "feature_k_values",
                    "feature_random_trials",
                    "feature_ranking_mode",
                    "feature_source_unit",
                    "feature_span_score",
                )
            )
        )
        configs[config_key] += 1
        sample = _sample_key(meta, path)

        for row in payload.get("feature_attribution") or []:
            target_count += 1
            target = row.get("target_token")
            target_idx = row.get("target_token_index")
            base_ce = row.get("base_ce_loss")
            if isinstance(base_ce, (int, float)):
                base_losses.append(float(base_ce))

            method_top = row.get("method_top") or []
            method_ids = {
                int(item.get("source_token_index"))
                for item in method_top[:method_k]
                if item.get("source_token_index") is not None
            }

            has_individual = any(
                abs(float(item.get("logprob_drop", 0.0))) > 0.0
                for item in method_top
                if isinstance(item.get("logprob_drop"), (int, float))
            )
            if has_individual:
                rows_with_individual_effects += 1

            for item in method_top:
                rank = int(item.get("rank", 0) or 0)
                effect = item.get("logprob_drop")
                if rank <= 0 or not isinstance(effect, (int, float)):
                    continue
                effect = float(effect)
                individual_effects_by_rank[rank].append(effect)
                if rank <= method_k and effect < -effect_eps:
                    record = {
                        "file": os.path.basename(path),
                        "sample": sample,
                        "target_token_index": target_idx,
                        "target_token": target,
                        "target_context": _context(tokens, target_idx),
                        "target_category": _token_category(str(target or "")),
                        "base_ce_loss": base_ce,
                        "rank": rank,
                        "source_token_index": item.get("source_token_index"),
                        "source_token": item.get("source_token"),
                        "source_context": _context(tokens, item.get("source_token_index")),
                        "source_category": _token_category(str(item.get("source_token") or "")),
                        "alti_saliency": item.get("alti_saliency"),
                        "logprob_drop": effect,
                        "prob_drop": item.get("prob_drop"),
                    }
                    reverse_individual.append(record)
                    token_category_counts["reverse_individual"][record["source_category"]] += 1

            for group in row.get("group_effects") or []:
                k = int(group.get("k", 0) or 0)
                effect = group.get("logprob_drop")
                if k <= 0 or not isinstance(effect, (int, float)):
                    continue
                effect = float(effect)
                group_effects_by_k[k].append(effect)
                if k == method_k and effect < -effect_eps:
                    source_tokens = group.get("source_tokens") or []
                    for token in source_tokens:
                        token_category_counts["reverse_group_sources"][_token_category(str(token))] += 1
                    reverse_groups.append(
                        {
                            "file": os.path.basename(path),
                            "sample": sample,
                            "target_token_index": target_idx,
                            "target_token": target,
                            "target_context": _context(tokens, target_idx),
                            "target_category": _token_category(str(target or "")),
                            "base_ce_loss": base_ce,
                            "k": k,
                            "source_token_indices": group.get("source_token_indices"),
                            "source_tokens": source_tokens,
                            "source_contexts": [
                                _context(tokens, source_idx)
                                for source_idx in (group.get("source_token_indices") or [])
                            ],
                            "source_categories": [_token_category(str(token)) for token in source_tokens],
                            "logprob_drop": effect,
                            "prob_drop": group.get("prob_drop"),
                        }
                    )

            oracle_top = row.get("oracle_top") or []
            if oracle_top:
                rows_with_full_oracle += 1
                for item in oracle_top[:oracle_m]:
                    source_idx = item.get("source_token_index")
                    if source_idx is None:
                        continue
                    effect = item.get("oracle_effect", item.get("logprob_drop"))
                    record = {
                        "file": os.path.basename(path),
                        "sample": sample,
                        "target_token_index": target_idx,
                        "target_token": target,
                        "target_context": _context(tokens, target_idx),
                        "target_category": _token_category(str(target or "")),
                        "base_ce_loss": base_ce,
                        "oracle_rank": item.get("rank"),
                        "source_token_index": source_idx,
                        "source_token": item.get("source_token"),
                        "source_context": _context(tokens, source_idx),
                        "source_category": _token_category(str(item.get("source_token") or "")),
                        "oracle_effect": effect,
                        "alti_saliency": item.get("alti_saliency"),
                        "in_method_top_k": int(source_idx) in method_ids,
                    }
                    if int(source_idx) in method_ids:
                        hit_oracle.append(record)
                        token_category_counts["hit_oracle"][record["source_category"]] += 1
                    else:
                        missed_oracle.append(record)
                        token_category_counts["missed_oracle"][record["source_category"]] += 1

    top_rank_effects = {
        f"rank<={k}": _summarize(
            [
                value
                for rank, values in individual_effects_by_rank.items()
                if rank <= k
                for value in values
            ]
        )
        for k in (1, 3, 5, 10)
    }
    group_summaries = {
        f"group@{k}": {
            **_summarize(values),
            "positive_rate": (
                sum(1 for value in values if value > effect_eps) / len(values)
                if values else 0.0
            ),
            "reverse_rate": (
                sum(1 for value in values if value < -effect_eps) / len(values)
                if values else 0.0
            ),
        }
        for k, values in sorted(group_effects_by_k.items())
    }

    return {
        "summary": {
            "results_dir": os.path.abspath(results_dir),
            "file_count": len(files),
            "target_count": target_count,
            "method_k": int(method_k),
            "oracle_m": int(oracle_m),
            "effect_eps": float(effect_eps),
            "rows_with_individual_effects": rows_with_individual_effects,
            "rows_with_full_oracle": rows_with_full_oracle,
            "base_ce_loss": _summarize(base_losses),
            "configs": [
                {"count": count, "config": dict(key)}
                for key, count in configs.most_common()
            ],
        },
        "effect_summaries": {
            "individual_by_rank": top_rank_effects,
            "group_by_k": group_summaries,
        },
        "category_counts": {
            name: dict(counter.most_common())
            for name, counter in token_category_counts.items()
        },
        "reverse_individual_examples": _top_examples(reverse_individual, example_limit),
        "reverse_group_examples": _top_examples(reverse_groups, example_limit),
        "missed_oracle_examples": sorted(
            missed_oracle,
            key=lambda row: (
                -float(row.get("oracle_effect") or 0.0),
                int(row.get("oracle_rank") or 10**9),
            ),
        )[:example_limit],
        "hit_oracle_examples": sorted(
            hit_oracle,
            key=lambda row: (
                -float(row.get("oracle_effect") or 0.0),
                int(row.get("oracle_rank") or 10**9),
            ),
        )[:example_limit],
    }


def _fmt_pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _compact_context(tokens: list[str], limit: int = 80) -> str:
    text = "".join(tokens)
    text = text.replace("\n", "\\n").replace("\t", "\\t")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _write_markdown(path: str, report: dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    summary = report["summary"]
    lines = [
        "# Feature Error Analysis",
        "",
        "## Coverage",
        "",
        f"- Results: `{summary['results_dir']}`",
        f"- Files: {summary['file_count']}",
        f"- Targets: {summary['target_count']}",
        f"- Method top-k: {summary['method_k']}",
        f"- Oracle top-m: {summary['oracle_m']}",
        f"- Direction epsilon: {summary['effect_eps']}",
        f"- Rows with individual effects: {summary['rows_with_individual_effects']}",
        f"- Rows with full oracle ranking: {summary['rows_with_full_oracle']}",
        "",
        "## Group Effects",
        "",
        "| metric | n | mean drop | median drop | positive | reverse |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key, stats in report["effect_summaries"]["group_by_k"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(stats.get("n", 0)),
                    f"{float(stats.get('mean', 0.0)):.4f}",
                    f"{float(stats.get('median', 0.0)):.4f}",
                    _fmt_pct(float(stats.get("positive_rate", 0.0))),
                    _fmt_pct(float(stats.get("reverse_rate", 0.0))),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Category Counts",
            "",
            "### Reverse Individual Sources",
            "",
        ]
    )
    for category, count in report["category_counts"]["reverse_individual"].items():
        lines.append(f"- {category}: {count}")
    lines.extend(["", "### Missed Oracle Sources", ""])
    for category, count in report["category_counts"]["missed_oracle"].items():
        lines.append(f"- {category}: {count}")

    def add_examples(title: str, rows: list[dict[str, Any]], effect_key: str) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| sample | target | base CE | source | category | rank | effect | source context |",
                "| --- | --- | ---: | --- | --- | ---: | ---: | --- |",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.get("sample", "")),
                        repr(row.get("target_token", "")),
                        f"{float(row.get('base_ce_loss') or 0.0):.4f}",
                        repr(row.get("source_token", "")),
                        str(row.get("source_category", "")),
                        str(row.get("rank") or row.get("oracle_rank") or ""),
                        f"{float(row.get(effect_key) or 0.0):.4f}",
                        _compact_context(row.get("source_context") or []),
                    ]
                )
                + " |"
            )

    add_examples("Reverse Individual Examples", report["reverse_individual_examples"], "logprob_drop")
    lines.extend(
        [
            "",
            "## Reverse Group Examples",
            "",
            "| sample | target | base CE | k | effect | source tokens | categories |",
            "| --- | --- | ---: | ---: | ---: | --- | --- |",
        ]
    )
    for row in report["reverse_group_examples"]:
        source_tokens = ", ".join(repr(token) for token in row.get("source_tokens") or [])
        categories = ", ".join(str(cat) for cat in row.get("source_categories") or [])
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("sample", "")),
                    repr(row.get("target_token", "")),
                    f"{float(row.get('base_ce_loss') or 0.0):.4f}",
                    str(row.get("k", "")),
                    f"{float(row.get('logprob_drop') or 0.0):.4f}",
                    source_tokens,
                    categories,
                ]
            )
            + " |"
        )
    add_examples("Missed Oracle Examples", report["missed_oracle_examples"], "oracle_effect")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose feature attribution misses and reverse effects.")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--method-k", type=int, default=10)
    parser.add_argument("--oracle-m", type=int, default=10)
    parser.add_argument("--effect-eps", type=float, default=1e-4)
    parser.add_argument("--example-limit", type=int, default=30)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    args = parser.parse_args()

    report = analyze_feature_results(
        args.results_dir,
        method_k=args.method_k,
        oracle_m=args.oracle_m,
        effect_eps=args.effect_eps,
        example_limit=args.example_limit,
    )
    output_json = args.output_json or os.path.join(
        args.results_dir,
        "reports",
        "feature_error_analysis.json",
    )
    output_md = args.output_md or os.path.join(
        args.results_dir,
        "reports",
        "feature_error_analysis.md",
    )
    _write_json(output_json, report)
    _write_markdown(output_md, report)
    print(f"[feature-error-analysis] wrote {output_json}")
    print(f"[feature-error-analysis] wrote {output_md}")


if __name__ == "__main__":
    main()
