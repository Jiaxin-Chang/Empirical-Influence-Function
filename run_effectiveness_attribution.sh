#!/usr/bin/env bash
# Batch-run effectiveness-only feature/data attribution evaluation.
#
# Feature stage:
#   - Uses feature_evaluation_mode=effectiveness.
#   - Perturbs only method top-k source tokens plus grouped top-k perturbations.
#
# Data stage:
#   - Uses the cached/batched data evaluator.
#   - Restricts oracle unlearning to method top DATA_TOP_K train samples.
#   - This avoids the old full 7906-candidate oracle.
#   - Defaults to one test sample per oracle loop because top-K train sets
#     are usually different across test samples; increase DATA_TEST_BATCH_SIZE
#     only when you want to try union reuse.

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-}"
TRAIN_DATA="${TRAIN_DATA:-sft_train.jsonl}"
TEST_DATA="${TEST_DATA:-sft_test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-attribution_results_effectiveness}"
STAGES="${STAGES:-both}"
INDICES="${INDICES:-}"
START_IDX="${START_IDX:-0}"
END_IDX="${END_IDX:-}"
PYTHON="${PYTHON:-python}"

MAX_OUTPUT_TOKENS="${MAX_OUTPUT_TOKENS:-20}"
GENERATION_LIMIT="${GENERATION_LIMIT:-128}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-eager}"
MAX_GPU_MEMORY="${MAX_GPU_MEMORY:-}"

FEATURE_K_VALUES="${FEATURE_K_VALUES:-1,3,5,10}"
FEATURE_THRESHOLDS="${FEATURE_THRESHOLDS:-0.2,0.5,1.0}"
FEATURE_EFFECT_METRIC="${FEATURE_EFFECT_METRIC:-logprob_drop}"

DATA_K_VALUES="${DATA_K_VALUES:-10,50,100}"
DATA_TOP_K="${DATA_TOP_K:-100}"
DATA_THRESHOLDS="${DATA_THRESHOLDS:-0.0005,0.001,0.002,0.005}"
DATA_TEST_BATCH_SIZE="${DATA_TEST_BATCH_SIZE:-1}"
UNLEARN_LR="${UNLEARN_LR:-0.01}"
PRESCREEN_MAX_SEQ_LEN="${PRESCREEN_MAX_SEQ_LEN:-3000}"
PRESCREEN_SKETCH_DIM="${PRESCREEN_SKETCH_DIM:-8192}"
PRESCREEN_SKETCH_SEED="${PRESCREEN_SKETCH_SEED:-42}"
PRESCREEN_SKETCH_CACHE_DIR="${PRESCREEN_SKETCH_CACHE_DIR:-.cache/prescreen_sketch}"

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
        --max-output-tokens) MAX_OUTPUT_TOKENS="$2"; shift 2 ;;
        --generation-limit) GENERATION_LIMIT="$2"; shift 2 ;;
        --attn-implementation) ATTN_IMPLEMENTATION="$2"; shift 2 ;;
        --max-gpu-memory) MAX_GPU_MEMORY="$2"; shift 2 ;;
        --feature-k-values) FEATURE_K_VALUES="$2"; shift 2 ;;
        --feature-thresholds|--feature-effect-thresholds) FEATURE_THRESHOLDS="$2"; shift 2 ;;
        --feature-effect-metric) FEATURE_EFFECT_METRIC="$2"; shift 2 ;;
        --data-k-values|--data-method-k-values) DATA_K_VALUES="$2"; shift 2 ;;
        --data-top-k) DATA_TOP_K="$2"; shift 2 ;;
        --data-thresholds|--data-effect-thresholds) DATA_THRESHOLDS="$2"; shift 2 ;;
        --data-test-batch-size|--test-batch-size) DATA_TEST_BATCH_SIZE="$2"; shift 2 ;;
        --unlearn-lr) UNLEARN_LR="$2"; shift 2 ;;
        --prescreen-max-seq-len) PRESCREEN_MAX_SEQ_LEN="$2"; shift 2 ;;
        --prescreen-sketch-dim) PRESCREEN_SKETCH_DIM="$2"; shift 2 ;;
        --prescreen-sketch-seed) PRESCREEN_SKETCH_SEED="$2"; shift 2 ;;
        --prescreen-sketch-cache-dir) PRESCREEN_SKETCH_CACHE_DIR="$2"; shift 2 ;;
        *) echo "[effectiveness-runner] Unknown option: $1"; exit 1 ;;
    esac
done

case "$STAGES" in
    feature|data|both) ;;
    *) echo "[effectiveness-runner] ERROR: --stages must be feature, data, or both."; exit 1 ;;
esac

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR_ABS="$(cd "$ROOT_DIR" && mkdir -p "$OUTPUT_DIR" && cd "$OUTPUT_DIR" && pwd)"
mkdir -p "$OUTPUT_DIR_ABS/reports"

COMMON_RANGE_ARGS=()
if [[ -n "$INDICES" ]]; then
    COMMON_RANGE_ARGS+=(--indices "$INDICES")
else
    COMMON_RANGE_ARGS+=(--start-idx "$START_IDX")
    [[ -n "$END_IDX" ]] && COMMON_RANGE_ARGS+=(--end-idx "$END_IDX")
fi

COMMON_MODEL_ARGS=()
[[ -n "$MODEL_PATH" ]] && COMMON_MODEL_ARGS+=(--model-path "$MODEL_PATH")
[[ -n "$ATTN_IMPLEMENTATION" ]] && COMMON_MODEL_ARGS+=(--attn-implementation "$ATTN_IMPLEMENTATION")
[[ -n "$MAX_GPU_MEMORY" ]] && COMMON_MODEL_ARGS+=(--max-gpu-memory "$MAX_GPU_MEMORY")

echo "================================================================="
echo "  Effectiveness Attribution Evaluation"
echo "  stages    : ${STAGES}"
echo "  output dir: ${OUTPUT_DIR_ABS}"
echo "  feature k : ${FEATURE_K_VALUES}, thresholds=${FEATURE_THRESHOLDS}"
echo "  data k    : ${DATA_K_VALUES}, top-k-unlearn=${DATA_TOP_K}, test-batch-size=${DATA_TEST_BATCH_SIZE}, thresholds=${DATA_THRESHOLDS}"
echo "================================================================="

if [[ "$STAGES" == "feature" || "$STAGES" == "both" ]]; then
    bash "$ROOT_DIR/run_batch_attribution_evaluation.sh" \
        --stages feature \
        --train-data "$TRAIN_DATA" \
        --test-data "$TEST_DATA" \
        --output-dir "$OUTPUT_DIR_ABS" \
        "${COMMON_RANGE_ARGS[@]}" \
        "${COMMON_MODEL_ARGS[@]}" \
        --max-output-tokens "$MAX_OUTPUT_TOKENS" \
        --generation-limit "$GENERATION_LIMIT" \
        --feature-k-values "$FEATURE_K_VALUES" \
        --feature-evaluation-mode effectiveness \
        --feature-effect-thresholds "$FEATURE_THRESHOLDS" \
        --feature-effect-metric "$FEATURE_EFFECT_METRIC"
fi

if [[ "$STAGES" == "data" || "$STAGES" == "both" ]]; then
    DATA_ARGS=(
        --train-data "$TRAIN_DATA"
        --test-data "$TEST_DATA"
        --output-dir "$OUTPUT_DIR_ABS"
        "${COMMON_RANGE_ARGS[@]}"
        "${COMMON_MODEL_ARGS[@]}"
        --max-output-tokens "$MAX_OUTPUT_TOKENS"
        --generation-limit "$GENERATION_LIMIT"
        --test-batch-size "$DATA_TEST_BATCH_SIZE"
        --data-method-k-values "$DATA_K_VALUES"
        --data-oracle-m-values "$DATA_K_VALUES"
        --data-oracle-limit "$DATA_TOP_K"
        --data-oracle-include-method-top "$DATA_TOP_K"
        --unlearn-lr "$UNLEARN_LR"
        --prescreen-max-seq-len "$PRESCREEN_MAX_SEQ_LEN"
        --prescreen-sketch-dim "$PRESCREEN_SKETCH_DIM"
        --prescreen-sketch-seed "$PRESCREEN_SKETCH_SEED"
        --prescreen-sketch-cache-dir "$PRESCREEN_SKETCH_CACHE_DIR"
    )
    "$PYTHON" -m src.batched_data_attribution_evaluation "${DATA_ARGS[@]}"
fi

"$PYTHON" -m src.effectiveness_report \
    --results-dir "$OUTPUT_DIR_ABS" \
    --data-k-values "$DATA_K_VALUES" \
    --data-thresholds "$DATA_THRESHOLDS" \
    --output-json "$OUTPUT_DIR_ABS/reports/effectiveness_report.json" \
    --output-md "$OUTPUT_DIR_ABS/reports/effectiveness_report.md"

echo "================================================================="
echo "  Done"
echo "  Results: ${OUTPUT_DIR_ABS}"
echo "  Report : ${OUTPUT_DIR_ABS}/reports/effectiveness_report.json"
echo "           ${OUTPUT_DIR_ABS}/reports/effectiveness_report.md"
echo "================================================================="
