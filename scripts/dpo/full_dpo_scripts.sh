#!/bin/bash

STRATEGIES=("drts" "deltaucb" "dts" "maxminlcb" "infomax")

BASE_OUTPUT_LOCATION="$SCRATCH/models/dpo/new_era"

SBATCH_SCRIPT="activeuf/dpo/training.sbatch"

for STRAT in "${STRATEGIES[@]}"; do
    DATASET_PATH="ActiveUltraFeedback/active_tulu3_dpo_${STRAT}"
    
    OUTPUT_DIR="${BASE_OUTPUT_LOCATION}/${STRAT}N1" ## ADJUST THIS FOR EXPERIMENTS

    echo "Submitting job for strategy: ${STRAT}"
    echo "  Dataset: ${DATASET_PATH}"
    echo "  Output:  ${OUTPUT_DIR}"

    sbatch \
        --export=ALL,MyDatasetPath="${DATASET_PATH}",MyOutputDir="${OUTPUT_DIR}" \
        "$SBATCH_SCRIPT" #would be good to have kwargs here, and pass it directly to the py script from the sbatch script
    
    echo "-----------------------------------"
done