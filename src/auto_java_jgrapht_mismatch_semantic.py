#!/usr/bin/env python3
"""JGraphT Java: mismatch tests → LLM boolean expressions → corpus search → LLM semantic annotate.

Only rows with ``label != predict`` (after stripping thinking / whitespace) are
processed. Each mismatch: LLM writes corpus search expressions, search
``jgrapht_fim_train.jsonl`` tight→loose, [MASK]/MID-align one hit, then the
full-sample LLM semantic annotator; accept writes continue-train JSONL.

Writes to (viewer must already be started with this continue file)::

    /mnt/md124/jiaxin/Empirical-Influence-Function/continue_annotated.jsonl

Prerequisites (both must be running, ``eif_api.env`` loaded)::

    python -m src.ttav_bundle_api

    cd tools/annotation-viewer && python -m server.main \\
      --continue-data /mnt/md124/jiaxin/Empirical-Influence-Function/continue_annotated.jsonl

If a Go/C++ job is already using 8765, start a second viewer on another port
and pass ``--viewer-url http://127.0.0.1:<port>``. Do not edit the running
stack's ``eif_api.env``.

Example::

    python -m src.auto_java_jgrapht_mismatch_semantic

    python -m src.auto_java_jgrapht_mismatch_semantic --copies 1 --max-tests 5 --dry-run
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.auto_raw_ce_to_continue import main as pipeline_main

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PREDICTIONS = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/jgrapht/"
    "jgrapht_fim_test.predictions.jsonl"
)
DEFAULT_CORPUS = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/jgrapht/"
    "jgrapht_fim_train.jsonl"
)
DEFAULT_CONTINUE = (
    "/mnt/md124/jiaxin/Empirical-Influence-Function/continue_annotated.jsonl"
)


def main(argv: list[str] | None = None) -> int:
    extra = list(sys.argv[1:] if argv is None else argv)
    forwarded: list[str] = [
        "--input", DEFAULT_PREDICTIONS,
        "--corpus-path", DEFAULT_CORPUS,
        "--continue-path", DEFAULT_CONTINUE,
        "--annotate", "llm-semantic",
        "--mismatches-only",
        "--language", "java",
        "--copies", "1",
        "--tight-first",
    ]
    # Caller flags override the defaults above (argparse last-wins for store
    # actions; BooleanOptionalAction needs the flag present).
    return pipeline_main(forwarded + extra)


if __name__ == "__main__":
    raise SystemExit(main())
