#!/bin/bash

# ==============================================================================
# AlpacaEval: Sequential Generation + Judging (single node)
#
# This script runs AlpacaEval in two sequential phases to avoid GPU memory
# conflicts between the generator model and the 70B judge model:
#
#   Phase 1: Generate completions using vLLM (offline batch), then exit Python
#            to free all GPU memory.
#   Phase 2: Start the judge (Llama-3.3-70B-Instruct) vLLM server, then run
#            `alpaca_eval evaluate` on the pre-generated outputs.
#
# This allows both large generator models (up to 70B) and the 70B judge to
# run on the same node without OOM issues.
# ==============================================================================

# ==============================================================================
# CONFIGURATION
# ==============================================================================
PROJECT_DIR="${SCRATCH}/ActiveUltraFeedback"
MODELS_DIR=${MODELS_DIR:-${PROJECT_DIR}/models}

# Model & results paths
MODEL_PATH="${MODEL_PATH:-${MODELS_DIR}/dpo/b27bst06/0mbq0vu7}"
RESULTS_DIR="${RESULTS_DIR:-${MODEL_PATH}/results/alpaca_eval}"
HF_HOME="${HF_HOME:-${SCRATCH}/huggingface}"

ALPACA_EVAL_DIR="${ALPACA_EVAL_DIR:-${PROJECT_DIR}/resources/alpaca_eval}"
LOGS_DIR="${LOGS_DIR:-${PROJECT_DIR}/logs/alpaca_eval}"

# Generator vLLM configuration (Phase 1 - offline batch)
GENERATOR_GPU_MEM_UTILIZATION="${GENERATOR_GPU_MEM_UTILIZATION:-0.90}"
GENERATOR_TENSOR_PARALLEL_SIZE="${GENERATOR_TENSOR_PARALLEL_SIZE:-4}"
GENERATOR_MAX_MODEL_LEN="${GENERATOR_MAX_MODEL_LEN:-4096}"
GENERATOR_MAX_NEW_TOKENS="${GENERATOR_MAX_NEW_TOKENS:-2048}"
GENERATOR_TEMPERATURE="${GENERATOR_TEMPERATURE:-0.0}"

# Annotator/Judge vLLM server configuration (Phase 2)
ANNOTATOR_NAME="${ANNOTATOR_NAME:-activeuf}"
ANNOTATOR_PORT="${ANNOTATOR_PORT:-25125}"
ANNOTATOR_API_KEY="${ANNOTATOR_API_KEY:-token-abc123}"
ANNOTATOR_GPU_MEM_UTILIZATION="${ANNOTATOR_GPU_MEM_UTILIZATION:-0.90}"
ANNOTATOR_TENSOR_PARALLEL_SIZE="${ANNOTATOR_TENSOR_PARALLEL_SIZE:-4}"

# ==============================================================================
# ARGUMENT PARSING
# ==============================================================================
help_function() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Required Options:"
    echo "  --model_path <path>                        Model path to evaluate"
    echo "  --results_dir <path>                       Directory for evaluation results"
    echo ""
    echo "Optional Options:"
    echo "  --generator_gpu_mem_utilization <float>    GPU memory for generator (default: 0.90)"
    echo "  --generator_tensor_parallel_size <int>     TP size for generator (default: 4)"
    echo "  --generator_max_model_len <int>            Max model len for generator (default: 4096)"
    echo "  --annotator_gpu_mem_utilization <float>    GPU memory for judge (default: 0.90)"
    echo "  --annotator_tensor_parallel_size <int>     TP size for judge (default: 4)"
    echo "  -h, --help                                 Show this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --results_dir)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --generator_gpu_mem_utilization)
            GENERATOR_GPU_MEM_UTILIZATION="$2"
            shift 2
            ;;
        --generator_tensor_parallel_size)
            GENERATOR_TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        --generator_max_model_len)
            GENERATOR_MAX_MODEL_LEN="$2"
            shift 2
            ;;
        --annotator_gpu_mem_utilization)
            ANNOTATOR_GPU_MEM_UTILIZATION="$2"
            shift 2
            ;;
        --annotator_tensor_parallel_size)
            ANNOTATOR_TENSOR_PARALLEL_SIZE="$2"
            shift 2
            ;;
        -h|--help)
            help_function
            ;;
        *)
            echo "Unknown argument: $1"
            help_function
            ;;
    esac
done

# ==============================================================================
# VALIDATION
# ==============================================================================
missing_args=0

if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required."
    missing_args=1
fi
if [ -z "$RESULTS_DIR" ]; then
    echo "Error: --results_dir is required."
    missing_args=1
fi

if [ "$missing_args" -eq 1 ]; then
    echo "----------------------------------------"
    help_function
fi

# ==============================================================================
# SETUP
# ==============================================================================
set -e  # Exit on error

EVAL_EXIT_CODE=1

mkdir -p "${RESULTS_DIR}"
mkdir -p "${LOGS_DIR}"

# Derive model/generator name relative to MODELS_DIR
MODEL_NAME="${MODEL_PATH#${MODELS_DIR}/}"
MODEL_NAME="${MODEL_NAME#/}"
# Fallback: if MODEL_PATH is not under MODELS_DIR, use basename
if [ "$MODEL_NAME" = "$MODEL_PATH" ]; then
    MODEL_NAME="$(basename "$MODEL_PATH")"
fi

MODEL_OUTPUTS_FILE="${RESULTS_DIR}/model_outputs.json"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "========================================"
echo "CONFIGURATION"
echo "========================================"
echo "MODEL_PATH:                         ${MODEL_PATH}"
echo "MODEL_NAME:                         ${MODEL_NAME}"
echo "RESULTS_DIR:                        ${RESULTS_DIR}"
echo "MODEL_OUTPUTS_FILE:                 ${MODEL_OUTPUTS_FILE}"
echo "GENERATOR_GPU_MEM_UTILIZATION:      ${GENERATOR_GPU_MEM_UTILIZATION}"
echo "GENERATOR_TENSOR_PARALLEL_SIZE:     ${GENERATOR_TENSOR_PARALLEL_SIZE}"
echo "GENERATOR_MAX_MODEL_LEN:            ${GENERATOR_MAX_MODEL_LEN}"
echo "ANNOTATOR_GPU_MEM_UTILIZATION:      ${ANNOTATOR_GPU_MEM_UTILIZATION}"
echo "ANNOTATOR_TENSOR_PARALLEL_SIZE:     ${ANNOTATOR_TENSOR_PARALLEL_SIZE}"
echo "HF_HOME:                            ${HF_HOME}"
echo "========================================"

cd "${PROJECT_DIR}"

# Delete old results if they exist
if [ -d "${RESULTS_DIR}" ]; then
    echo "Erasing existing results at ${RESULTS_DIR}"
    rm -rf "${RESULTS_DIR}"
fi
mkdir -p "${RESULTS_DIR}"

# ==============================================================================
# STEP 0: Install alpaca_eval
# ==============================================================================
if [ ! -d "${ALPACA_EVAL_DIR}" ]; then
    echo "Cloning alpaca_eval repo..."
    git clone https://github.com/tatsu-lab/alpaca_eval.git "${ALPACA_EVAL_DIR}"
fi

echo "Installing alpaca_eval..."
cd "${ALPACA_EVAL_DIR}"
python -m pip install -e . --quiet
cd "${PROJECT_DIR}"

# ==============================================================================
# PHASE 1: Generate completions (offline vLLM batch inference)
# ==============================================================================
echo ""
echo "========================================"
echo "PHASE 1: Generating completions"
echo "========================================"
echo "Generator model gets full GPU memory (${GENERATOR_GPU_MEM_UTILIZATION})"
echo ""

python "${SCRIPT_DIR}/generate_alpaca_completions.py" \
    --model_path "${MODEL_PATH}" \
    --output_path "${MODEL_OUTPUTS_FILE}" \
    --generator_name "${MODEL_NAME}" \
    --tensor_parallel_size "${GENERATOR_TENSOR_PARALLEL_SIZE}" \
    --gpu_memory_utilization "${GENERATOR_GPU_MEM_UTILIZATION}" \
    --max_model_len "${GENERATOR_MAX_MODEL_LEN}" \
    --max_new_tokens "${GENERATOR_MAX_NEW_TOKENS}" \
    --temperature "${GENERATOR_TEMPERATURE}"

if [ $? -ne 0 ]; then
    echo "ERROR: Generation failed!"
    exit 1
fi

if [ ! -f "${MODEL_OUTPUTS_FILE}" ]; then
    echo "ERROR: model_outputs.json was not created!"
    exit 1
fi

NUM_OUTPUTS=$(python -c "import json; print(len(json.load(open('${MODEL_OUTPUTS_FILE}'))))")
echo "Phase 1 complete: ${NUM_OUTPUTS} completions saved to ${MODEL_OUTPUTS_FILE}"

# ==============================================================================
# PHASE 2: Judge/Annotate with Llama-3.3-70B-Instruct
# ==============================================================================
echo ""
echo "========================================"
echo "PHASE 2: Running annotation (judge)"
echo "========================================"
echo "Judge gets full GPU memory (${ANNOTATOR_GPU_MEM_UTILIZATION})"
echo ""

# ---- Create annotator config ----
echo "Creating annotator config..."
ANNOTATOR_CONFIG_DIR="${ALPACA_EVAL_DIR}/src/alpaca_eval/evaluators_configs/${ANNOTATOR_NAME}"
mkdir -p "${ANNOTATOR_CONFIG_DIR}"
ANNOTATOR_CONFIG_FILE="${ANNOTATOR_CONFIG_DIR}/configs.yaml"
if [ ! -f "${ANNOTATOR_CONFIG_FILE}" ]; then
    cat > "${ANNOTATOR_CONFIG_FILE}" <<EOF
${ANNOTATOR_NAME}:
    prompt_template: "alpaca_eval_clf_gpt4_turbo/alpaca_eval_clf.txt"
    fn_completions: "openai_completions"
    completions_kwargs:
        model_name: "meta-llama/Llama-3.3-70B-Instruct"
        max_tokens: 1
        temperature: 1
        logprobs: true
        top_logprobs: 5
        requires_chatml: true
    fn_completion_parser: "logprob_parser"
    completion_parser_kwargs:
        numerator_token: "m"
        denominator_tokens: ["m", "M"]
        is_binarize: false
    completion_key: "completions_all"
    batch_size: 1
EOF
    echo "Annotator config created at: ${ANNOTATOR_CONFIG_FILE}"
else
    echo "Annotator config already exists at: ${ANNOTATOR_CONFIG_FILE}"
fi

# ---- Start vLLM judge server ----
echo "Starting vLLM judge server (Llama-3.3-70B-Instruct)..."
ANNOTATOR_LOG="${RESULTS_DIR}/annotator_server.log"
vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --gpu-memory-utilization "${ANNOTATOR_GPU_MEM_UTILIZATION}" \
    --swap-space 1 \
    --tensor-parallel-size "${ANNOTATOR_TENSOR_PARALLEL_SIZE}" \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --dtype bfloat16 \
    --port "${ANNOTATOR_PORT}" \
    --api-key "${ANNOTATOR_API_KEY}" \
    --download-dir "${HF_HOME}" \
    > "${ANNOTATOR_LOG}" 2>&1 &
ANNOTATOR_PID=$!
echo "vLLM judge server started with PID: ${ANNOTATOR_PID}"
echo "Log file: ${ANNOTATOR_LOG}"

# ---- Set up environment for OpenAI-compatible annotator ----
unset SSL_CERT_FILE 2>/dev/null || true
export OPENAI_API_BASE="http://localhost:${ANNOTATOR_PORT}/v1"
export OPENAI_API_KEY="${ANNOTATOR_API_KEY}"

# ---- Wait for judge server to be ready ----
echo "Waiting for vLLM judge server to be ready..."
MAX_WAIT=600
WAIT_INTERVAL=10
ELAPSED=0
SERVER_READY=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if grep -q "Application startup complete" "${ANNOTATOR_LOG}" 2>/dev/null; then
        echo "vLLM judge server is ready!"
        SERVER_READY=1
        break
    fi
    if curl -s "http://localhost:${ANNOTATOR_PORT}/health" >/dev/null 2>&1; then
        echo "vLLM judge server is responding!"
        SERVER_READY=1
        break
    fi
    echo "Waiting... (${ELAPSED}s / ${MAX_WAIT}s)"
    sleep ${WAIT_INTERVAL}
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [ $SERVER_READY -eq 0 ]; then
    echo "ERROR: vLLM judge server did not become ready within ${MAX_WAIT} seconds"
    echo "Last 50 lines of annotator log:"
    tail -n 50 "${ANNOTATOR_LOG}"
    kill ${ANNOTATOR_PID} 2>/dev/null || true
    exit 1
fi

# ---- Run annotation on pre-generated outputs ----
echo "Running alpaca_eval evaluate on pre-generated outputs..."

alpaca_eval evaluate \
    --model_outputs "${MODEL_OUTPUTS_FILE}" \
    --annotators_config "${ANNOTATOR_NAME}" \
    --output_path "${RESULTS_DIR}" \
    --caching_path "${RESULTS_DIR}/annotations_cache.json"

EVAL_EXIT_CODE=$?

# ==============================================================================
# CLEANUP
# ==============================================================================
if [ -n "${ANNOTATOR_PID}" ]; then
    echo "Stopping vLLM judge server..."
    kill ${ANNOTATOR_PID} 2>/dev/null || true
    wait ${ANNOTATOR_PID} 2>/dev/null || true
fi

# ==============================================================================
# EXIT
# ==============================================================================
if [ $EVAL_EXIT_CODE -eq 0 ]; then
    echo "=========================================="
    echo "Evaluation completed successfully!"
    echo "Results saved to: ${RESULTS_DIR}"
    echo "=========================================="
else
    echo "=========================================="
    echo "Evaluation failed with exit code: ${EVAL_EXIT_CODE}"
    echo "=========================================="
fi

exit ${EVAL_EXIT_CODE}