#!/bin/bash
# Submit one 4-node Active-DPO run per threshold C.
# Usage (from repo root):  bash activeuf/loop/launch_dpo.sh
# Override the list:        THRESHOLDS="0.5 1.5 3.0" bash activeuf/loop/launch_dpo.sh
set -euo pipefail

THRESHOLDS="${THRESHOLDS:-0.1 0.25 0.5}"
CONFIG="${CONFIG:-configs/loop_dpo.yaml}"

mkdir -p logs/active_dpo   # Slurm won't create the --output/--error dir itself

for t in $THRESHOLDS; do
    echo "Submitting Active-DPO run for threshold C=$t (config=$CONFIG)"
    THRESHOLD="$t" CONFIG="$CONFIG" sbatch \
        --job-name="active_dpo_C${t}" \
        activeuf/loop/run_dpo.sbatch
done
