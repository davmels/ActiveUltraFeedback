#!/bin/bash
OUTPUT_DIR="./datasets"
MODEL_POOL=(
    "Qwen/Qwen3-0.6B"
    "Qwen/Qwen3-1.7B"
    "Qwen/Qwen3-4B"
)
JUDGE_MODEL="Qwen/Qwen3-8B"

# Exit on error, unset variable, or pipe failure
set -euo pipefail

# ===== DATASET PRE-PROCESSING =====
# Download the dataset and take the first 1024 rows for example purposes
# Note: allenai/ultrafeedback_binarized_cleaned is supported without further pre-processing
python <<PY
import os
from datasets import load_dataset

dataset = load_dataset("allenai/ultrafeedback_binarized_cleaned", split="train_prefs")
dataset = dataset.select(range(1024))
dataset.save_to_disk("${OUTPUT_DIR}/0_pre_processed")
PY

# ===== RESPONSE GENERATION =====
# Generate the completions for each model individually (can be run in parallel)
for MODEL in "${MODEL_POOL[@]}"; do
    MODEL_NAME=${MODEL##*/}  # Get the model name from the path, e.g Qwen/Qwen3-0.6B -> Qwen3-0.6B
    python -m activeuf.completions.generate_completions \
        --dataset_path ${OUTPUT_DIR}/0_pre_processed \
        --model_name ${MODEL} \
        --model_class vllm \
        --output_path ${OUTPUT_DIR}/1_individual_completions/${MODEL_NAME}
done

# Merge the individual completions into a single dataset
python -m activeuf.completions.merge_completions \
    --datasets_path ${OUTPUT_DIR}/1_individual_completions \
    --output_path ${OUTPUT_DIR}/2_merged_completions

# ===== RESPONSE ANNOTATION =====
# Pre-compute the judge scores for all responses (can be run in parallel)
for MODEL in "${MODEL_POOL[@]}"; do
    MODEL_NAME=${MODEL##*/}  # Get the model name from the path, e.g Qwen/Qwen3-0.6B -> Qwen3-0.6B
    python -m activeuf.oracle.get_raw_annotations \
        --model_name ${JUDGE_MODEL} \
        --model_to_annotate ${MODEL} \
        --dataset_path ${OUTPUT_DIR}/2_merged_completions \
        --model_class vllm \
        --output_path ${OUTPUT_DIR}/3_annotated_completions/
done

# Merge the annotated completions into a single dataset
python -m activeuf.oracle.combine_annotated_completions \
    --annotations_folder ${OUTPUT_DIR}/3_annotated_completions \
    --completions_folder ${OUTPUT_DIR}/1_individual_completions \
    --output_folder ${OUTPUT_DIR}/4_merged_annotations

# ===== MAIN LOOP =====
python -m activeuf.loop.run --config_path configs/example_loop.yaml