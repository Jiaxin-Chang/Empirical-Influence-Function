#!/usr/bin/env bash
# =============================================================================
# run_batch_experiments.sh
#
# Batch runner for all-token correlation matching experiments.
# For each test_index, runs:
#   intervention_experiment.py --all-tokens
#     → correlation_matching_results_test{N}_all_tokens_new.json
#
# Usage:
#   bash run_batch_experiments.sh [--parallel-jobs 4]
#
# Edit the EXPERIMENTS array below to add / remove cases.
# =============================================================================

set -uo pipefail

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
PARALLEL_JOBS="${PARALLEL_JOBS:-4}"
RESULT_SUFFIX="${RESULT_SUFFIX-new}"

# Optional extra args, e.g.
#   IE_EXTRA_ARGS="--prescreen-sketch-dim 8192 --fine-match-proj qk --alti-grad-chunk-size 32 --top-targets 8" bash run_batch_experiments.sh
IE_EXTRA_ARGS="${IE_EXTRA_ARGS:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel-jobs)
            PARALLEL_JOBS="$2"
            shift 2
            ;;
        --result-suffix)
            RESULT_SUFFIX="$2"
            shift 2
            ;;
        *)
            echo "[batch] Unknown option: $1"
            exit 1
            ;;
    esac
done

if (( PARALLEL_JOBS < 1 )); then
    PARALLEL_JOBS=1
fi

if (( BASH_VERSINFO[0] > 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] >= 3) )); then
    WAIT_N_SUPPORTED=1
else
    WAIT_N_SUPPORTED=0
fi

# Project root (directory containing this script)
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Log directory
LOG_DIR="${ROOT_DIR}/logs/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "================================================================="
echo "  Batch Experiment Runner"
echo "  Mode      : intervention --all-tokens"
echo "  Experiments: ${#EXPERIMENTS[@]}"
echo "  Parallel jobs: ${PARALLEL_JOBS}"
echo "  Result suffix: ${RESULT_SUFFIX:-none}"
echo "  Log dir   : ${LOG_DIR}"
echo "================================================================="

TOTAL=${#EXPERIMENTS[@]}
IDX=0
N_SKIP=0
N_FAIL=0

cleanup_children() {
    local pids
    pids="$(jobs -rp)"
    if [[ -n "${pids}" ]]; then
        kill ${pids} 2>/dev/null || true
    fi
}
trap cleanup_children INT TERM

run_one_experiment() {
    local display_idx="$1"
    local pair="$2"
    local test_idx="${pair%%:*}"
    local suffix_part=""
    if [[ -n "${RESULT_SUFFIX}" ]]; then
        suffix_part="_${RESULT_SUFFIX#_}"
    fi
    local corr_out="${ROOT_DIR}/correlation_matching_results_test${test_idx}_all_tokens${suffix_part}.json"
    local ie_log="${LOG_DIR}/intervention_test${test_idx}_all_tokens${suffix_part}.log"

    echo ""
    echo "-----------------------------------------------------------------"
    echo "  [${display_idx}/${TOTAL}]  test=${test_idx}  all_tokens"
    echo "-----------------------------------------------------------------"

    if [[ -f "${corr_out}" ]]; then
        echo "  [skip IE]   ${corr_out} already exists."
        return 0
    fi

    echo "  [IE]   Running all-tokens...  (log: ${ie_log})"
    "${PYTHON}" -m src.intervention_experiment \
        --test-index "${test_idx}" \
        --all-tokens \
        ${RESULT_SUFFIX:+--output-suffix "${RESULT_SUFFIX}"} \
        ${IE_EXTRA_ARGS} \
        2>&1 | tee "${ie_log}"
    local exit_code=${PIPESTATUS[0]}

    if [[ ${exit_code} -eq 0 ]]; then
        echo "  [IE]   Done → ${corr_out}"
    else
        echo "  [IE]   FAILED test=${test_idx} exit=${exit_code}  (log: ${ie_log})"
    fi
    return "${exit_code}"
}

wait_for_one_job() {
    local exit_code=0
    if [[ "${WAIT_N_SUPPORTED}" -eq 1 ]]; then
        wait -n
        exit_code=$?
    else
        local pid status
        for pid in $(jobs -rp); do
            wait "${pid}"
            status=$?
            if [[ ${status} -ne 0 ]]; then
                exit_code=${status}
            fi
        done
    fi
    if [[ ${exit_code} -ne 0 ]]; then
        N_FAIL=$((N_FAIL + 1))
    fi
}

for PAIR in "${EXPERIMENTS[@]}"; do
    IDX=$((IDX + 1))
    TEST_IDX="${PAIR%%:*}"
    SUFFIX_PART=""
    if [[ -n "${RESULT_SUFFIX}" ]]; then
        SUFFIX_PART="_${RESULT_SUFFIX#_}"
    fi
    CORR_OUT="${ROOT_DIR}/correlation_matching_results_test${TEST_IDX}_all_tokens${SUFFIX_PART}.json"

    if [[ -f "${CORR_OUT}" ]]; then
        echo "  [skip IE]   ${CORR_OUT} already exists."
        N_SKIP=$((N_SKIP + 1))
    else
        while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "${PARALLEL_JOBS}" ]]; do
            wait_for_one_job
        done
        run_one_experiment "${IDX}" "${PAIR}" &
    fi
done

while [[ "$(jobs -rp | wc -l | tr -d ' ')" -gt 0 ]]; do
    wait_for_one_job
done

echo ""
echo "================================================================="
echo "  All ${TOTAL} experiment(s) finished."
echo "  Skipped: ${N_SKIP}"
echo "  Failed : ${N_FAIL}"
echo "  Results are in: ${ROOT_DIR}"
echo "================================================================="

if [[ "${N_FAIL}" -ne 0 ]]; then
    exit 1
fi
