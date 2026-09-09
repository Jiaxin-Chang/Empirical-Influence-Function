#!/usr/bin/env python3
"""Go CSN: mismatch tests → LLM boolean expressions → corpus search → LLM semantic annotate.

Only rows with ``label != predict`` (after stripping thinking / whitespace) are
processed. Each mismatch: LLM writes corpus search expressions, search
``csn_go_train_fim.jsonl`` tight→loose, MID-align one hit, then the current
full-sample LLM semantic annotator; accept writes continue-train JSONL.

Prerequisites (both must be running, ``eif_api.env`` loaded)::

    python -m src.ttav_bundle_api

    cd tools/annotation-viewer && python -m server.main

Env (Linux example)::

    EIF_LLM_TRAIN_CORPUS=/mnt/md124/jiaxin/training_code/data/csn_go_train_fim.jsonl
    ANNOTATION_CONTINUE_TRAIN_DATA=/mnt/md124/jiaxin/Empirical-Influence-Function/go_csn_continue_semantic.jsonl
    ANNOTATION_TRAIN_DATA=...   # viewer still needs a compact source; can be a small jsonl
    DASHSCOPE_API_KEY / OPENAI_BASE_URL / ANNOTATE_MODEL   # same as semantic annotate

Example::

    python -m src.auto_go_csn_mismatch_semantic

    python -m src.auto_go_csn_mismatch_semantic --copies 1 --max-tests 5 --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.auto_raw_ce_to_continue import main as pipeline_main

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PREDICTIONS = "/mnt/md124/jiaxin/go_csn_ce_outputs/qwen3-8b-go.predictions.jsonl"
DEFAULT_CORPUS = "/mnt/md124/jiaxin/training_code/data/csn_go_train_fim.jsonl"


def main(argv: list[str] | None = None) -> int:
    extra = list(sys.argv[1:] if argv is None else argv)
    forwarded: list[str] = [
        "--input", DEFAULT_PREDICTIONS,
        "--corpus-path", DEFAULT_CORPUS,
        "--annotate", "llm-semantic",
        "--mismatches-only",
        "--language", "go",
        "--copies", "1",
        "--tight-first",
    ]
    # Caller flags override the defaults above (argparse last-wins for store
    # actions; BooleanOptionalAction needs the flag present).
    # Put defaults first, then user args.
    return pipeline_main(forwarded + extra)


if __name__ == "__main__":
    raise SystemExit(main())
