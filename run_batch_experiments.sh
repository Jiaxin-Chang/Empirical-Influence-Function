#!/usr/bin/env bash
# =============================================================================
# run_batch_experiments.sh
#
# Batch runner for Empirical-Influence-Function experiments.
# For each (test_index, token_index) pair, runs:
#   1) NIF.py   → saliency_test{N}_tok{tok}.json
#   2) intervention_experiment.py → correlation_matching_results_test{N}_tok{tok}.json
#
# Usage:
#   bash run_batch_experiments.sh [--nif-only | --intervention-only]
#
# Edit the EXPERIMENTS array below to add / remove cases.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# ★  EDIT THIS LIST to add your (test_index, token_index) pairs  ★
# ---------------------------------------------------------------------------
# Format: "test_index:token_index"
EXPERIMENTS=(
    "58:703"
    "58:750"
    "23:412"
    # add more pairs here ...
)
# ---------------------------------------------------------------------------

# Which stages to run: "both" | "nif" | "intervention"
MODE="both"
if [[ "${1:-}" == "--nif-only" ]];          then MODE="nif"; fi
if [[ "${1:-}" == "--intervention-only" ]]; then MODE="intervention"; fi

# Python interpreter — adjust if you use a venv / conda env
PYTHON="${PYTHON:-python}"

# Project root (directory containing this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Log directory
LOG_DIR="${ROOT_DIR}/logs/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "================================================================="
echo "  Batch Experiment Runner"
echo "  Mode      : ${MODE}"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "  Log dir   : ${LOG_DIR}"
echo "================================================================="

TOTAL=${#EXPERIMENTS[@]}
IDX=0

for PAIR in "${EXPERIMENTS[@]}"; do
    IDX=$((IDX + 1))
    TEST_IDX="${PAIR%%:*}"
    TOK_IDX="${PAIR##*:}"

    echo ""
    echo "-----------------------------------------------------------------"
    echo "  [${IDX}/${TOTAL}]  test=${TEST_IDX}  tok=${TOK_IDX}"
    echo "-----------------------------------------------------------------"

    # ------------------------------------------------------------------
    # Stage 1: NIF.py  (saliency)
    # ------------------------------------------------------------------
    SALIENCY_OUT="${ROOT_DIR}/saliency_test${TEST_IDX}_tok${TOK_IDX}.json"

    if [[ "${MODE}" == "both" || "${MODE}" == "nif" ]]; then
        if [[ -f "${SALIENCY_OUT}" ]]; then
            echo "  [skip NIF]  ${SALIENCY_OUT} already exists."
        else
            NIF_LOG="${LOG_DIR}/nif_test${TEST_IDX}_tok${TOK_IDX}.log"
            echo "  [NIF]  Running...  (log: ${NIF_LOG})"
            "${PYTHON}" -m src.NIF \
                --test-index  "${TEST_IDX}" \
                --token-index "${TOK_IDX}" \
                2>&1 | tee "${NIF_LOG}"
            echo "  [NIF]  Done → ${SALIENCY_OUT}"
        fi
    fi

    # ------------------------------------------------------------------
    # Stage 2: intervention_experiment.py  (correlation matching)
    # ------------------------------------------------------------------
    CORR_OUT="${ROOT_DIR}/correlation_matching_results_test${TEST_IDX}_tok${TOK_IDX}.json"

    if [[ "${MODE}" == "both" || "${MODE}" == "intervention" ]]; then
        if [[ -f "${CORR_OUT}" ]]; then
            echo "  [skip IE]   ${CORR_OUT} already exists."
        else
            IE_LOG="${LOG_DIR}/intervention_test${TEST_IDX}_tok${TOK_IDX}.log"
            echo "  [IE]   Running...  (log: ${IE_LOG})"
            "${PYTHON}" -m src.intervention_experiment \
                --test-index  "${TEST_IDX}" \
                --token-index "${TOK_IDX}" \
                2>&1 | tee "${IE_LOG}"
            echo "  [IE]   Done → ${CORR_OUT}"
        fi
    fi
done

echo ""
echo "================================================================="
echo "  All ${TOTAL} experiment(s) finished."
echo "  Results are in: ${ROOT_DIR}"
echo "================================================================="
