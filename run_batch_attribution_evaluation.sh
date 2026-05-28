#!/usr/bin/env bash
# =============================================================================
# run_batch_attribution_evaluation.sh
#
# Batch-run feature attribution and/or data attribution evaluation.
# All outputs are written under attribution_results/ by default.
#
# Usage:
#   bash run_batch_attribution_evaluation.sh [options]
#
# Common options:
#   --model-path  PATH   Path to the model checkpoint directory
#   --train-data  FILE   Training JSONL  (default: sft_train.jsonl)
#   --test-data   FILE   Test JSONL      (default: sft_test.jsonl)
#   --indices     LIST   Comma/space-separated sample indices, e.g. 3,17,58
#   --start-idx   N      Start sample index for range mode (default: 0)
#   --end-idx     N      End sample index for range mode, inclusive
#   --stages      NAME   feature, data, or both (default: both)
#   --output-dir  DIR    Output directory (default: attribution_results)
#
# Example:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 \
#   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#   bash run_batch_attribution_evaluation.sh \
#       --model-path /data/models/Qwen3-1.5B \
#       --indices 3,17,58 \
#       --data-oracle-limit 300
# =============================================================================

set -uo pipefail

MODEL_PATH="${MODEL_PATH:-}"
TRAIN_DATA="${TRAIN_DATA:-sft_train.jsonl}"
TEST_DATA="${TEST_DATA:-sft_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-attribution_results}"
STAGES="${STAGES:-both}"
INDICES="${INDICES:-}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-}"
TRAIN_LIMIT="${TRAIN_LIMIT:-}"
MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-}"
GENERATION_LIMIT="${GENERATION_LIMIT:-}"
USE_GROUND_TRUTH_RESPONSE="${USE_GROUND_TRUTH_RESPONSE:-0}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-}"
MAX_GPU_MEMORY="${MAX_GPU_MEMORY:-}"
SEED="${SEED:-}"

TOP_K_PROMPT_TOKENS="${TOP_K_PROMPT_TOKENS:-}"
FEATURE_K_VALUES="${FEATURE_K_VALUES:-}"
FEATURE_PERTURB_MODE="${FEATURE_PERTURB_MODE:-}"
REPLACEMENT_TOKEN_ID="${REPLACEMENT_TOKEN_ID:-}"
MAX_FEATURE_SOURCES="${MAX_FEATURE_SOURCES:-}"

PRESCREEN_BATCH_SIZE="${PRESCREEN_BATCH_SIZE:-}"
PRESCREEN_MAX_SEQ_LEN="${PRESCREEN_MAX_SEQ_LEN:-}"
PRESCREEN_SKETCH_DIM="${PRESCREEN_SKETCH_DIM:-}"
PRESCREEN_SKETCH_SEED="${PRESCREEN_SKETCH_SEED:-}"
PRESCREEN_SKETCH_CACHE_DIR="${PRESCREEN_SKETCH_CACHE_DIR:-}"
NO_PRESCREEN_SKETCH_CACHE="${NO_PRESCREEN_SKETCH_CACHE:-0}"
DATA_GRANULARITY="${DATA_GRANULARITY:-}"
DATA_METHOD_K_VALUES="${DATA_METHOD_K_VALUES:-}"
DATA_ORACLE_M_VALUES="${DATA_ORACLE_M_VALUES:-}"
DATA_ORACLE_LIMIT="${DATA_ORACLE_LIMIT:-}"
DATA_ORACLE_INCLUDE_METHOD_TOP="${DATA_ORACLE_INCLUDE_METHOD_TOP:-}"
UNLEARN_LR="${UNLEARN_LR:-}"
NO_NORMALIZE_UNLEARN_GRAD="${NO_NORMALIZE_UNLEARN_GRAD:-0}"
DATA_EFFECT_REDUCTION="${DATA_EFFECT_REDUCTION:-}"

PYTHON="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-path) MODEL_PATH="$2"; shift 2 ;;
        --train-data) TRAIN_DATA="$2"; shift 2 ;;
        --test-data) TEST_DATA="$2"; shift 2 ;;
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --stages|--stage) STAGES="$2"; shift 2 ;;
        --indices) INDICES="$2"; shift 2 ;;
        --start-idx) START_IDX="$2"; shift 2 ;;
        --end-idx) END_IDX="$2"; shift 2 ;;
        --train-limit) TRAIN_LIMIT="$2"; shift 2 ;;
        --max-output-tokens) MAX_OUTPUT_TOKENS="$2"; shift 2 ;;
        --generation-limit) GENERATION_LIMIT="$2"; shift 2 ;;
        --use-ground-truth-response) USE_GROUND_TRUTH_RESPONSE=1; shift ;;
        --attn-implementation) ATTN_IMPLEMENTATION="$2"; shift 2 ;;
        --max-gpu-memory) MAX_GPU_MEMORY="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --top-k-prompt-tokens) TOP_K_PROMPT_TOKENS="$2"; shift 2 ;;
        --feature-k-values) FEATURE_K_VALUES="$2"; shift 2 ;;
        --feature-perturb-mode) FEATURE_PERTURB_MODE="$2"; shift 2 ;;
        --replacement-token-id) REPLACEMENT_TOKEN_ID="$2"; shift 2 ;;
        --max-feature-sources) MAX_FEATURE_SOURCES="$2"; shift 2 ;;
        --prescreen-batch-size) PRESCREEN_BATCH_SIZE="$2"; shift 2 ;;
        --prescreen-max-seq-len) PRESCREEN_MAX_SEQ_LEN="$2"; shift 2 ;;
        --prescreen-sketch-dim) PRESCREEN_SKETCH_DIM="$2"; shift 2 ;;
        --prescreen-sketch-seed) PRESCREEN_SKETCH_SEED="$2"; shift 2 ;;
        --prescreen-sketch-cache-dir) PRESCREEN_SKETCH_CACHE_DIR="$2"; shift 2 ;;
        --no-prescreen-sketch-cache) NO_PRESCREEN_SKETCH_CACHE=1; shift ;;
        --data-granularity) DATA_GRANULARITY="$2"; shift 2 ;;
        --data-method-k-values) DATA_METHOD_K_VALUES="$2"; shift 2 ;;
        --data-oracle-m-values) DATA_ORACLE_M_VALUES="$2"; shift 2 ;;
        --data-oracle-limit) DATA_ORACLE_LIMIT="$2"; shift 2 ;;
        --data-oracle-include-method-top) DATA_ORACLE_INCLUDE_METHOD_TOP="$2"; shift 2 ;;
        --unlearn-lr) UNLEARN_LR="$2"; shift 2 ;;
        --no-normalize-unlearn-grad) NO_NORMALIZE_UNLEARN_GRAD=1; shift ;;
        --data-effect-reduction) DATA_EFFECT_REDUCTION="$2"; shift 2 ;;
        *) echo "[batch-attribution] Unknown option: $1"; exit 1 ;;
    esac
done

case "$STAGES" in
    feature|data|both) ;;
    *) echo "[batch-attribution] ERROR: --stages must be feature, data, or both."; exit 1 ;;
esac

if [[ ! -f "$TEST_DATA" ]]; then
    echo "[batch-attribution] ERROR: test data file not found: $TEST_DATA"
    exit 1
fi

if [[ "$STAGES" != "feature" && ! -f "$TRAIN_DATA" ]]; then
    echo "[batch-attribution] ERROR: train data file not found: $TRAIN_DATA"
    exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR_ABS="$(cd "$ROOT_DIR" && mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"
FEATURE_DIR="${OUTPUT_DIR_ABS}/feature"
DATA_DIR="${OUTPUT_DIR_ABS}/data"
DATA_COARSE_DIR="${OUTPUT_DIR_ABS}/data_coarse"
REPORT_DIR="${OUTPUT_DIR_ABS}/reports"
LOG_DIR="${OUTPUT_DIR_ABS}/logs/batch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$FEATURE_DIR" "$DATA_DIR" "$DATA_COARSE_DIR" "$REPORT_DIR" "$LOG_DIR"

STATUS_TSV="${REPORT_DIR}/batch_status.tsv"
printf "sample_index\ttask_id\tstage\tstatus\texit_code\toutput_file\tlog_file\n" > "$STATUS_TSV"

mapfile -t TASK_IDS < <("${PYTHON}" - "$TEST_DATA" <<'PYEOF'
import json
import sys

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
if [[ "$TOTAL" -eq 0 ]]; then
    echo "[batch-attribution] ERROR: no test samples found in $TEST_DATA"
    exit 1
fi

if [[ -z "$END_IDX" ]]; then
    END_IDX=$((TOTAL - 1))
fi

if [[ -n "$INDICES" ]]; then
    IFS=', ' read -r -a RUN_INDICES <<< "$INDICES"
else
    mapfile -t RUN_INDICES < <(seq "$START_IDX" "$END_IDX")
fi

VALID_RUN_INDICES=()
for IDX in "${RUN_INDICES[@]}"; do
    [[ -z "$IDX" ]] && continue
    if ! [[ "$IDX" =~ ^[0-9]+$ ]]; then
        echo "[batch-attribution] ERROR: invalid sample index: $IDX"
        exit 1
    fi
    if (( IDX < 0 || IDX >= TOTAL )); then
        echo "[batch-attribution] ERROR: sample index out of range: $IDX (valid: 0..$((TOTAL - 1)))"
        exit 1
    fi
    VALID_RUN_INDICES+=("$IDX")
done

if [[ ${#VALID_RUN_INDICES[@]} -eq 0 ]]; then
    echo "[batch-attribution] ERROR: no sample indices selected."
    exit 1
fi

COMMON_ARGS=(--train-data "$TRAIN_DATA" --test-data "$TEST_DATA")
[[ -n "$MODEL_PATH" ]] && COMMON_ARGS+=(--model-path "$MODEL_PATH")
[[ -n "$TRAIN_LIMIT" ]] && COMMON_ARGS+=(--train-limit "$TRAIN_LIMIT")
[[ -n "$MAX_OUTPUT_TOKENS" ]] && COMMON_ARGS+=(--max-output-tokens "$MAX_OUTPUT_TOKENS")
[[ -n "$GENERATION_LIMIT" ]] && COMMON_ARGS+=(--generation-limit "$GENERATION_LIMIT")
[[ "$USE_GROUND_TRUTH_RESPONSE" == "1" ]] && COMMON_ARGS+=(--use-ground-truth-response)
[[ -n "$ATTN_IMPLEMENTATION" ]] && COMMON_ARGS+=(--attn-implementation "$ATTN_IMPLEMENTATION")
[[ -n "$MAX_GPU_MEMORY" ]] && COMMON_ARGS+=(--max-gpu-memory "$MAX_GPU_MEMORY")
[[ -n "$SEED" ]] && COMMON_ARGS+=(--seed "$SEED")

FEATURE_ARGS=()
[[ -n "$TOP_K_PROMPT_TOKENS" ]] && FEATURE_ARGS+=(--top-k-prompt-tokens "$TOP_K_PROMPT_TOKENS")
[[ -n "$FEATURE_K_VALUES" ]] && FEATURE_ARGS+=(--feature-k-values "$FEATURE_K_VALUES")
[[ -n "$FEATURE_PERTURB_MODE" ]] && FEATURE_ARGS+=(--feature-perturb-mode "$FEATURE_PERTURB_MODE")
[[ -n "$REPLACEMENT_TOKEN_ID" ]] && FEATURE_ARGS+=(--replacement-token-id "$REPLACEMENT_TOKEN_ID")
[[ -n "$MAX_FEATURE_SOURCES" ]] && FEATURE_ARGS+=(--max-feature-sources "$MAX_FEATURE_SOURCES")

DATA_ARGS=()
[[ -n "$PRESCREEN_BATCH_SIZE" ]] && DATA_ARGS+=(--prescreen-batch-size "$PRESCREEN_BATCH_SIZE")
[[ -n "$PRESCREEN_MAX_SEQ_LEN" ]] && DATA_ARGS+=(--prescreen-max-seq-len "$PRESCREEN_MAX_SEQ_LEN")
[[ -n "$PRESCREEN_SKETCH_DIM" ]] && DATA_ARGS+=(--prescreen-sketch-dim "$PRESCREEN_SKETCH_DIM")
[[ -n "$PRESCREEN_SKETCH_SEED" ]] && DATA_ARGS+=(--prescreen-sketch-seed "$PRESCREEN_SKETCH_SEED")
[[ -n "$PRESCREEN_SKETCH_CACHE_DIR" ]] && DATA_ARGS+=(--prescreen-sketch-cache-dir "$PRESCREEN_SKETCH_CACHE_DIR")
[[ "$NO_PRESCREEN_SKETCH_CACHE" == "1" ]] && DATA_ARGS+=(--no-prescreen-sketch-cache)
[[ -n "$DATA_GRANULARITY" ]] && DATA_ARGS+=(--data-granularity "$DATA_GRANULARITY")
[[ -n "$DATA_METHOD_K_VALUES" ]] && DATA_ARGS+=(--data-method-k-values "$DATA_METHOD_K_VALUES")
[[ -n "$DATA_ORACLE_M_VALUES" ]] && DATA_ARGS+=(--data-oracle-m-values "$DATA_ORACLE_M_VALUES")
[[ -n "$DATA_ORACLE_LIMIT" ]] && DATA_ARGS+=(--data-oracle-limit "$DATA_ORACLE_LIMIT")
[[ -n "$DATA_ORACLE_INCLUDE_METHOD_TOP" ]] && DATA_ARGS+=(--data-oracle-include-method-top "$DATA_ORACLE_INCLUDE_METHOD_TOP")
[[ -n "$UNLEARN_LR" ]] && DATA_ARGS+=(--unlearn-lr "$UNLEARN_LR")
[[ "$NO_NORMALIZE_UNLEARN_GRAD" == "1" ]] && DATA_ARGS+=(--no-normalize-unlearn-grad)
[[ -n "$DATA_EFFECT_REDUCTION" ]] && DATA_ARGS+=(--data-effect-reduction "$DATA_EFFECT_REDUCTION")

echo "================================================================="
echo "  Batch Attribution Evaluation"
echo "  stages    : ${STAGES}"
echo "  model     : ${MODEL_PATH:-default}"
echo "  train data: ${TRAIN_DATA}"
echo "  test data : ${TEST_DATA} (${TOTAL} samples)"
echo "  output dir: ${OUTPUT_DIR_ABS}"
echo "  log dir   : ${LOG_DIR}"
if [[ -n "$INDICES" ]]; then
    echo "  indices   : ${VALID_RUN_INDICES[*]}"
else
    echo "  range     : [${START_IDX}, ${END_IDX}]"
fi
echo "================================================================="

N_DONE=0
N_SKIP=0
N_FAIL=0
FAILED=()

record_status() {
    local idx="$1"
    local task_id="$2"
    local stage="$3"
    local status="$4"
    local exit_code="$5"
    local output_file="$6"
    local log_file="$7"
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$idx" "$task_id" "$stage" "$status" "$exit_code" "$output_file" "$log_file" >> "$STATUS_TSV"
}

run_feature() {
    local idx="$1"
    local task_id="$2"
    local output_file="${FEATURE_DIR}/${task_id}_feature.json"
    local log_file="${LOG_DIR}/${task_id}_feature.log"

    if [[ -f "$output_file" ]]; then
        echo "[$(date +%H:%M:%S)] [${idx}] SKIP feature ${task_id}"
        N_SKIP=$((N_SKIP + 1))
        record_status "$idx" "$task_id" "feature" "skipped" "0" "$output_file" "$log_file"
        return
    fi

    echo "[$(date +%H:%M:%S)] [${idx}] START feature ${task_id}"
    "${PYTHON}" -m src.feature_attribution_evaluation \
        "${COMMON_ARGS[@]}" \
        "${FEATURE_ARGS[@]}" \
        --test-index "$idx" \
        --output "$output_file" \
        --feature-output "$output_file" \
        2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    if [[ $exit_code -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] [${idx}] DONE feature ${task_id}"
        N_DONE=$((N_DONE + 1))
        record_status "$idx" "$task_id" "feature" "done" "$exit_code" "$output_file" "$log_file"
    else
        echo "[$(date +%H:%M:%S)] [${idx}] FAIL feature ${task_id} (exit=${exit_code})"
        N_FAIL=$((N_FAIL + 1))
        FAILED+=("${task_id}:feature")
        record_status "$idx" "$task_id" "feature" "failed" "$exit_code" "$output_file" "$log_file"
    fi
}

run_data() {
    local idx="$1"
    local task_id="$2"
    local output_file="${DATA_DIR}/${task_id}_data.json"
    local coarse_file="${DATA_COARSE_DIR}/${task_id}_data_coarse.json"
    local log_file="${LOG_DIR}/${task_id}_data.log"

    if [[ -f "$output_file" ]]; then
        echo "[$(date +%H:%M:%S)] [${idx}] SKIP data ${task_id}"
        N_SKIP=$((N_SKIP + 1))
        record_status "$idx" "$task_id" "data" "skipped" "0" "$output_file" "$log_file"
        return
    fi

    echo "[$(date +%H:%M:%S)] [${idx}] START data ${task_id}"
    "${PYTHON}" -m src.data_attribution_evaluation \
        "${COMMON_ARGS[@]}" \
        "${DATA_ARGS[@]}" \
        --test-index "$idx" \
        --output "$output_file" \
        --data-coarse-output "$coarse_file" \
        2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    if [[ $exit_code -eq 0 ]]; then
        echo "[$(date +%H:%M:%S)] [${idx}] DONE data ${task_id}"
        N_DONE=$((N_DONE + 1))
        record_status "$idx" "$task_id" "data" "done" "$exit_code" "$output_file" "$log_file"
    else
        echo "[$(date +%H:%M:%S)] [${idx}] FAIL data ${task_id} (exit=${exit_code})"
        N_FAIL=$((N_FAIL + 1))
        FAILED+=("${task_id}:data")
        record_status "$idx" "$task_id" "data" "failed" "$exit_code" "$output_file" "$log_file"
    fi
}

for IDX in "${VALID_RUN_INDICES[@]}"; do
    TASK_ID="${TASK_IDS[$IDX]}"
    echo ""
    echo "-----------------------------------------------------------------"
    echo "Sample ${IDX}: ${TASK_ID}"
    echo "-----------------------------------------------------------------"

    if [[ "$STAGES" == "feature" || "$STAGES" == "both" ]]; then
        run_feature "$IDX" "$TASK_ID"
    fi
    if [[ "$STAGES" == "data" || "$STAGES" == "both" ]]; then
        run_data "$IDX" "$TASK_ID"
    fi
done

REPORT_JSON="${REPORT_DIR}/attribution_batch_report.json"
REPORT_MD="${REPORT_DIR}/attribution_batch_report.md"
"${PYTHON}" -m src.attribution_batch_report \
    --results-dir "$OUTPUT_DIR_ABS" \
    --status-tsv "$STATUS_TSV" \
    --output-json "$REPORT_JSON" \
    --output-md "$REPORT_MD"

echo ""
echo "================================================================="
echo "  Batch finished"
echo "  Done   : ${N_DONE}"
echo "  Skipped: ${N_SKIP}"
echo "  Failed : ${N_FAIL}"
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo "  Failed : ${FAILED[*]}"
fi
echo "  Results: ${OUTPUT_DIR_ABS}"
echo "  Report : ${REPORT_JSON}"
echo "           ${REPORT_MD}"
echo "  Logs   : ${LOG_DIR}"
echo "================================================================="
