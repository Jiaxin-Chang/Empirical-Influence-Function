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
TRAIN_LIMIT="${TRAIN_LIMIT:-}"  # empty = use all training samples
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"  # empty = intervention default
PRESCREEN_MAX_SEQ_LEN="${PRESCREEN_MAX_SEQ_LEN:-}"  # empty = intervention default; 0 disables guard
MAX_GPU_MEMORY="${MAX_GPU_MEMORY:-}"  # e.g. 26GiB to leave activation headroom
CORR_FEATURE_MODE="${CORR_FEATURE_MODE:-}"  # auto, second_order, or saliency_proxy
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
        --train-limit) TRAIN_LIMIT="$2"; shift 2 ;;
        --attn-implementation) ATTN_IMPLEMENTATION="$2"; shift 2 ;;
        --prescreen-max-seq-len) PRESCREEN_MAX_SEQ_LEN="$2"; shift 2 ;;
        --max-gpu-memory) MAX_GPU_MEMORY="$2"; shift 2 ;;
        --corr-feature-mode) CORR_FEATURE_MODE="$2"; shift 2 ;;
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
# Read task_ids from test JSONL (one per non-empty line).
# Falls back to sequential index if a line has no task_id field.
# ---------------------------------------------------------------------------
mapfile -t TASK_IDS < <(python3 - "$TEST_DATA" <<'PYEOF'
import sys, json
path = sys.argv[1]
with open(path, encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        print(obj.get("task_id") or f"test{i}")
PYEOF
)

TOTAL=${#TASK_IDS[@]}
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
echo "  train_limit: ${TRAIN_LIMIT:-all}"
echo "  attention : ${ATTN_IMPLEMENTATION:-auto}"
echo "  prescreen max seq len: ${PRESCREEN_MAX_SEQ_LEN:-default}"
echo "  max gpu memory: ${MAX_GPU_MEMORY:-auto}"
echo "  corr feature mode: ${CORR_FEATURE_MODE:-auto}"
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
    TASK_ID="${TASK_IDS[$IDX]}"
    RESULT_FILE="${ROOT_DIR}/correlation_matching_results_${TASK_ID}_all_tokens.json"

    # Resume: skip if result already exists
    if [[ -f "$RESULT_FILE" ]]; then
        echo "[$(date +%H:%M:%S)] [${IDX}/${END_IDX}] SKIP  ${TASK_ID}  (result exists)"
        N_SKIP=$((N_SKIP + 1))
        continue
    fi

    echo ""
    echo "[$(date +%H:%M:%S)] [${IDX}/${END_IDX}] START  task_id=${TASK_ID}"

    LOG_FILE="${LOG_DIR}/${TASK_ID}.log"

    "${PYTHON}" -m src.intervention_experiment \
        --model-path  "${MODEL_PATH}" \
        --train-data  "${TRAIN_DATA}" \
        --test-data   "${TEST_DATA}"  \
        --test-index  "${IDX}"        \
        --all-tokens  \
        ${TRAIN_LIMIT:+--train-limit "${TRAIN_LIMIT}"} \
        ${ATTN_IMPLEMENTATION:+--attn-implementation "${ATTN_IMPLEMENTATION}"} \
        ${PRESCREEN_MAX_SEQ_LEN:+--prescreen-max-seq-len "${PRESCREEN_MAX_SEQ_LEN}"} \
        ${MAX_GPU_MEMORY:+--max-gpu-memory "${MAX_GPU_MEMORY}"} \
        ${CORR_FEATURE_MODE:+--corr-feature-mode "${CORR_FEATURE_MODE}"} \
        2>&1 | tee "${LOG_FILE}"

    EXIT_CODE=${PIPESTATUS[0]}

    if [[ $EXIT_CODE -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] [${IDX}/${END_IDX}] DONE  → ${RESULT_FILE}"
        N_DONE=$((N_DONE + 1))
    else
        echo "[$(date +%H:%M:%S)] [${IDX}/${END_IDX}] FAIL  ${TASK_ID}  (exit=${EXIT_CODE}, log: ${LOG_FILE})"
        N_FAIL=$((N_FAIL + 1))
        FAILED_INDICES+=("$TASK_ID")
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
    echo "  Failed task_ids: ${FAILED_INDICES[*]}"
    echo ""
    echo "  Logs for failed samples are in: ${LOG_DIR}/"
fi
echo "  Results : ${ROOT_DIR}/correlation_matching_results_test*_all_tokens.json"
echo "  Logs    : ${LOG_DIR}/"
echo "================================================================="
