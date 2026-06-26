#!/bin/bash
set -euo pipefail

DATA_ROOT="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/dolci/prompted_partitions/parts2"
OUTPUT_ROOT="$SCRATCH/models/dpo_new/parts2704"

for group_dir in "$DATA_ROOT"/*/; do
    # group_name=$(basename "$group_dir")
    # for split_dir in "$group_dir"*/; do
    #     split_name=$(basename "$split_dir")
        dataset_path="$group_dir"
        group_name=$(basename "$group_dir")
        output_dir="$OUTPUT_ROOT/$group_name"

        echo "============================================"
        echo "Submitting: $group_dir"
        echo "  Dataset:  $dataset_path"
        echo "  Output:   $output_dir"
        echo "============================================"

        mkdir -p "$output_dir"

        sbatch --export=ALL,MyDatasetPath="$dataset_path",MyOutputDir="$output_dir" \
            /iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/activeuf/dpo/training.sbatch
    # done
done
