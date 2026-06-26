#!/bin/bash
set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir> [--sizes 5000 10000 ...]"
    echo "  input_dir:  directory containing multiple dataset folders"
    echo "  output_dir: directory where partitioned outputs will be saved"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PARTITION_SCRIPT="${SCRIPT_DIR}/partition_active_dataset.py"

EXTRA_ARGS=("$@")

for dataset_path in "${INPUT_DIR}"/*/; do
    dataset_name="$(basename "$dataset_path")"
    dataset_output_dir="${OUTPUT_DIR}/${dataset_name}"

    echo "=== Processing: ${dataset_name} ==="
    echo "  Input:  ${dataset_path}"
    echo "  Output: ${dataset_output_dir}"

    python "${PARTITION_SCRIPT}" \
        --dataset_path "${dataset_path}" \
        --output_dir "${dataset_output_dir}" \
        "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"

    echo ""
done

echo "All datasets processed."
