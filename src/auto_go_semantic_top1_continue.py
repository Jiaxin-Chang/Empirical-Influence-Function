#!/usr/bin/env python3
"""Go imperfect CE: 全部测试行 → Semantic 测试最高分召回 → LLM 语义标注 → 续训 JSONL.

对 ``qwen3-8b-go-ce.imperfect.predictions.jsonl`` 每一行走与报告页
「Semantic 测试」相同的 ``/api/llm-semantic-retrieve``（当前回退后的检索逻辑，
不做 polarity 惩罚），按 ``semantic_score`` 从高到低取 hit，MID 对齐后走
annotation-viewer 的 ``llm-semantic-annotate`` preview+accept。

写入（viewer 启动时必须指向同一文件）::

    /mnt/md124/jiaxin/Empirical-Influence-Function/go_continue_annotated_subset.jsonl

Prerequisites（``eif_api.env`` 已加载，两端 API 已起）::

    python -m src.ttav_bundle_api

    cd tools/annotation-viewer && python -m server.main \\
      --continue-data /mnt/md124/jiaxin/Empirical-Influence-Function/go_continue_annotated_subset.jsonl

Env（Linux 示例）::

    EIF_LLM_TRAIN_CORPUS=/mnt/md124/jiaxin/training_code/data/csn_go_train_fim.jsonl
    EIF_LLM_SEMANTIC_CORPUS=/mnt/md124/jiaxin/training_code/data/csn_go_train_fim.semantic.jsonl
    ANNOTATION_CONTINUE_TRAIN_DATA=/mnt/md124/jiaxin/Empirical-Influence-Function/go_continue_annotated_subset.jsonl

Example::

    python -m src.auto_go_semantic_top1_continue

    python -m src.auto_go_semantic_top1_continue --max-tests 3 --dry-run

    python -m src.auto_go_semantic_top1_continue --fresh

    nohup python -m src.auto_go_semantic_top1_continue \\
      > logs/go_semantic_top1_continue.log 2>&1 &
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.auto_raw_ce_to_continue import main as pipeline_main

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PREDICTIONS = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/"
    "qwen3-8b-go-ce.imperfect.predictions.jsonl"
)
DEFAULT_CORPUS = "/mnt/md124/jiaxin/training_code/data/csn_go_train_fim.jsonl"
DEFAULT_SEMANTIC_CORPUS = (
    "/mnt/md124/jiaxin/training_code/data/csn_go_train_fim.semantic.jsonl"
)
DEFAULT_CONTINUE = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/"
    "go_continue_annotated_subset.jsonl"
)
DEFAULT_STATE = str(
    REPO_ROOT / "qwen3-8b-go-ce.imperfect.predictions.semantic_top1_state.json"
)


def main(argv: list[str] | None = None) -> int:
    extra = list(sys.argv[1:] if argv is None else argv)
    forwarded: list[str] = [
        "--input", DEFAULT_PREDICTIONS,
        "--corpus-path", DEFAULT_CORPUS,
        "--semantic-corpus-path", DEFAULT_SEMANTIC_CORPUS,
        "--continue-path", DEFAULT_CONTINUE,
        "--retrieve-mode", "semantic",
        "--annotate", "llm-semantic",
        "--language", "go",
        "--copies", "1",
        "--top-k", "10",
        "--state", DEFAULT_STATE,
    ]
    return pipeline_main(forwarded + extra)


if __name__ == "__main__":
    raise SystemExit(main())
