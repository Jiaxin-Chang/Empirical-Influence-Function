#!/usr/bin/env bash
# =============================================================================
# run_batch_experiments.sh
#
# Batch runner for all-token correlation matching experiments.
# For each test_index, runs:
#   intervention_experiment.py --all-tokens
#     → correlation_matching_results_test{N}_all_tokens.json
#
# Usage:
#   bash run_batch_experiments.sh
#
# Edit the EXPERIMENTS array below to add / remove cases.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# ★  EDIT THIS LIST to add your test indices  ★
# ---------------------------------------------------------------------------
# Format: "test_index". Old "test_index:token_index" entries are also accepted;
# the token part is ignored in all-tokens mode.
EXPERIMENTS=(
    "58"
    "23"
    # add more test indices here ...
)
# ---------------------------------------------------------------------------

# Python interpreter — adjust if you use a venv / conda env
PYTHON="${PYTHON:-python}"

# Optional extra args, e.g.
#   IE_EXTRA_ARGS="--fine-match-proj qk --alti-grad-chunk-size 32 --top-targets 8" bash run_batch_experiments.sh
IE_EXTRA_ARGS="${IE_EXTRA_ARGS:-}"

# Project root (directory containing this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Log directory
LOG_DIR="${ROOT_DIR}/logs/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "================================================================="
echo "  Batch Experiment Runner"
echo "  Mode      : intervention --all-tokens"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "  Log dir   : ${LOG_DIR}"
echo "================================================================="

TOTAL=${#EXPERIMENTS[@]}
IDX=0

for PAIR in "${EXPERIMENTS[@]}"; do
    IDX=$((IDX + 1))
    TEST_IDX="${PAIR%%:*}"

    echo ""
    echo "-----------------------------------------------------------------"
    echo "  [${IDX}/${TOTAL}]  test=${TEST_IDX}  all_tokens"
    echo "-----------------------------------------------------------------"

    CORR_OUT="${ROOT_DIR}/correlation_matching_results_test${TEST_IDX}_all_tokens.json"

    if [[ -f "${CORR_OUT}" ]]; then
        echo "  [skip IE]   ${CORR_OUT} already exists."
    else
        IE_LOG="${LOG_DIR}/intervention_test${TEST_IDX}_all_tokens.log"
        echo "  [IE]   Running all-tokens...  (log: ${IE_LOG})"
        "${PYTHON}" -m src.intervention_experiment \
            --test-index "${TEST_IDX}" \
            --all-tokens \
            ${IE_EXTRA_ARGS} \
            2>&1 | tee "${IE_LOG}"
        echo "  [IE]   Done → ${CORR_OUT}"
    fi
done

echo ""
echo "================================================================="
echo "  All ${TOTAL} experiment(s) finished."
echo "  Results are in: ${ROOT_DIR}"
echo "================================================================="
