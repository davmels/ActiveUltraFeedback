#!/bin/bash

# List of full paths to your datasets
# Update these paths to point to your own dataset locations
DATASETS=(
    "/path/to/datasets/baselines/delta_qwen"
    "/path/to/datasets/baselines/maxmin"
    "/path/to/datasets/baselines/random"
    "/path/to/datasets/baselines/ultrafeedback"
)

SBATCH_SCRIPT="activeuf/cpo/training.sbatch"

echo "Staging Dataset Sweep..."

USE_LORA=true

for dataset in "${DATASETS[@]}"; do
    
    # Extract dataset name for logging (optional)
    ds_name=$(basename "$dataset")
    
    echo "Submitting job for: $ds_name"
    
    # Submit job, passing the path variable
    sbatch \
        --export=ALL,MyLossType="ipo",MyLR="5.0e-6",MyDatasetPath="$dataset",MyBeta="0.01",MyGamma="1.2",MyUseLora="$USE_LORA",MyRank="64",MyAlpha="16",MyOutputDir="${MODELS_DIR:-/path/to/models}/cpo_ipo_baseline" \
        "$SBATCH_SCRIPT"
        
    sleep 1
done

echo "All jobs submitted."