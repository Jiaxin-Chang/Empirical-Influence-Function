#!/usr/bin/env bash
# =============================================================================
# run_batch_experiments.sh
#
# Scan the entire test set in all-tokens mode.
# Every sample in TEST_DATA is treated as a wrong-prediction case to analyse.
#
# Usage:
#   bash run_batch_experiments.sh [options]
#
# Options (can also be set as env vars):
#   --model-path  PATH   Path to the model checkpoint directory
#   --train-data  FILE   Training JSONL  (default: sft_train.jsonl)
#   --test-data   FILE   Test JSONL      (default: sft_test.jsonl)
#   --start-idx   N      Start from sample index N (default: 0, for resume)
#   --end-idx     N      Stop after sample index N (inclusive, default: last)
#
# Examples:
#   # Full run with a new model
#   bash run_batch_experiments.sh \
#       --model-path /data/models/Qwen3-Coder-30B-Instruct \
#       --train-data /data/my_train.jsonl \
#       --test-data  /data/my_test.jsonl
#
#   # Resume from sample 10 after a crash
#   bash run_batch_experiments.sh \
#       --model-path /data/models/Qwen3-Coder-30B-Instruct \
#       --start-idx 10
# =============================================================================

set -uo pipefail   # -e intentionally omitted: one sample failing won't stop the batch

# ---------------------------------------------------------------------------
# Default configuration (override via flags or env vars)
# ---------------------------------------------------------------------------
MODEL_PATH="${MODEL_PATH:-}"
TRAIN_DATA="${TRAIN_DATA:-sft_train.jsonl}"
TEST_DATA="${TEST_DATA:-sft_test.jsonl}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-}"          # empty = auto-detect from file
PYTHON="${PYTHON:-python}"

# ---------------------------------------------------------------------------
# Parse command-line arguments
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2";  shift 2 ;;
        --train-data) TRAIN_DATA="$2";  shift 2 ;;
        --test-data)  TEST_DATA="$2";   shift 2 ;;
        --start-idx)  START_IDX="$2";   shift 2 ;;
        --end-idx)    END_IDX="$2";     shift 2 ;;
        *) echo "[batch] Unknown option: $1"; exit 1 ;;
    esac
done

# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------
if [[ -z "$MODEL_PATH" ]]; then
    echo "[batch] ERROR: --model-path is required."
    echo "        Example: bash run_batch_experiments.sh --model-path /data/models/Qwen3-Coder-30B-Instruct"
    exit 1
fi

if [[ ! -f "$TEST_DATA" ]]; then
    echo "[batch] ERROR: test data file not found: $TEST_DATA"
    exit 1
fi

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "[batch] ERROR: train data file not found: $TRAIN_DATA"
    exit 1
fi

# ---------------------------------------------------------------------------
# Count samples (each non-empty line = one sample)
# ---------------------------------------------------------------------------
TOTAL=$(grep -c . "$TEST_DATA" || true)
if [[ -z "$END_IDX" ]]; then
    END_IDX=$((TOTAL - 1))
fi

# ---------------------------------------------------------------------------
# Setup log directory
# ---------------------------------------------------------------------------
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${ROOT_DIR}/logs/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${LOG_DIR}"

echo "================================================================="
echo "  Batch Experiment Runner  (all-tokens mode)"
echo "  model     : ${MODEL_PATH}"
echo "  train data: ${TRAIN_DATA}"
echo "  test data : ${TEST_DATA}  (${TOTAL} samples)"
echo "  range     : [${START_IDX}, ${END_IDX}]"
echo "  log dir   : ${LOG_DIR}"
echo "================================================================="

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
N_DONE=0
N_SKIP=0
N_FAIL=0
FAILED_INDICES=()

for IDX in $(seq "$START_IDX" "$END_IDX"); do
    RESULT_FILE="${ROOT_DIR}/correlation_matching_results_test${IDX}_all_tokens.json"

    # Resume: skip if result already exists
    if [[ -f "$RESULT_FILE" ]]; then
        echo "[$(date +%H:%M:%S)] [${IDX}/${END_IDX}] SKIP  (result exists)"
        N_SKIP=$((N_SKIP + 1))
        continue
    fi

    echo ""
    echo "[$(date +%H:%M:%S)] [${IDX}/${END_IDX}] START  test_index=${IDX}"

    LOG_FILE="${LOG_DIR}/test_${IDX}.log"

    "${PYTHON}" -m src.intervention_experiment \
        --model-path  "${MODEL_PATH}" \
        --train-data  "${TRAIN_DATA}" \
        --test-data   "${TEST_DATA}"  \
        --test-index  "${IDX}"        \
        --all-tokens  \
        2>&1 | tee "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] [${IDX}/${END_IDX}] DONE  → ${RESULT_FILE}"
        N_DONE=$((N_DONE + 1))
    else
        echo "[$(date +%H:%M:%S)] [${IDX}/${END_IDX}] FAIL  (exit=${EXIT_CODE}, log: ${LOG_FILE})"
        N_FAIL=$((N_FAIL + 1))
        FAILED_INDICES+=("$IDX")
    fi
done

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "================================================================="
echo "  Batch finished"
echo "  Done  : ${N_DONE}"
echo "  Skip  : ${N_SKIP}  (already existed)"
echo "  Failed: ${N_FAIL}"
if [[ ${#FAILED_INDICES[@]} -gt 0 ]]; then
    echo "  Failed indices: ${FAILED_INDICES[*]}"
    echo ""
    echo "  To retry failed samples:"
    echo "    for IDX in ${FAILED_INDICES[*]}; do"
    echo "      bash run_batch_experiments.sh --model-path '${MODEL_PATH}' \\"
    echo "          --train-data '${TRAIN_DATA}' --test-data '${TEST_DATA}' \\"
    echo "          --start-idx \$IDX --end-idx \$IDX"
    echo "    done"
fi
echo "  Results : ${ROOT_DIR}/correlation_matching_results_test*_all_tokens.json"
echo "  Logs    : ${LOG_DIR}/"
echo "================================================================="
