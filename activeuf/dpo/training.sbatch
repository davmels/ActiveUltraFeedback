#!/bin/bash

#SBATCH -A infra01
#SBATCH --job-name=dpo
#SBATCH --output=logs/dpo/%j-%x.out
#SBATCH --error=logs/dpo/%j-%x.err
#SBATCH --nodes=8
#SBATCH --reservation=SD-69241-apertus-1-5-0
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --exclude=nid006076

# --- ARGUMENT PARSING ---
# 1. Dataset Path
if [[ -n "$MyDatasetPath" ]]; then
    export DATASET_PATH="$MyDatasetPath"
    echo "Using Custom Dataset Path: $DATASET_PATH"
else
    echo "Error: 'MyDatasetPath' environment variable is required."
    exit 1
fi

# 2. Output Directory
if [[ -n "$MyOutputDir" ]]; then
    export OUTPUT_DIR="$MyOutputDir"
    echo "Using Custom Output Dir: $OUTPUT_DIR"
else
    export OUTPUT_DIR="$SCRATCH/models/dpo"
    echo "Using Default Output Dir: $OUTPUT_DIR"
fi

# Append jobid + timestamp so each run gets a unique output dir (no overwrites).
# Computed once on the host -> identical across all srun tasks/nodes.
export OUTPUT_DIR="${OUTPUT_DIR}_${SLURM_JOB_ID}_$(date +%Y%m%d-%H%M%S)"
echo "Unique Output Dir: $OUTPUT_DIR"

# 3. Seed
if [[ -n "$MySeed" ]]; then
    export SEED="$MySeed"
    echo "Using Custom Seed: $SEED"
else
    export SEED="42"
    echo "Using Default Seed: $SEED"
fi

# 4. Debug mode (optional): set MyDebug=1 to trim the dataset to 100 samples
DEBUG_ARG=""
if [[ -n "$MyDebug" ]]; then
    DEBUG_ARG="--debug_mode True"
    echo "DEBUG MODE ON: trimming dataset to 100 samples"
fi

echo "Dataset Path: $DATASET_PATH"
echo "Seed:         $SEED"
echo "Output Dir: $OUTPUT_DIR"
# ------------------------

BASE_CACHE_DIR="$SCRATCH"
export HF_HOME=$BASE_CACHE_DIR/hf_home
export VLLM_CACHE_DIR=$BASE_CACHE_DIR/vllm_cache
export WANDB_PROJECT=DPO
export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"

# below is based on https://docs.csc.fi/support/tutorials/ml-multi/accelerate
MAIN_PROCESS_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MAIN_PROCESS_PORT=29500
NUM_PROCESSES=$(expr $SLURM_NNODES \* $SLURM_GPUS_ON_NODE)

# ------------------------
# SETUP SYNC BARRIER
# ------------------------
INSTALL_FLAG="$SCRATCH/.install_flag_${SLURM_JOB_ID}"

# We pack everything into a single multiline string.
# Variables WITHOUT a backslash evaluate on the head node immediately.
# Variables WITH a backslash (\$SLURM_NODEID) evaluate on the worker nodes during srun.
CMD="
if [ \"\$SLURM_NODEID\" -eq 0 ]; then
    echo 'Node 0: Installing transformers version 4.57.1...'
    pip install transformers==4.57.1 --upgrade
    echo 'Node 0: Installation complete. Signaling other nodes.'
    touch $INSTALL_FLAG
else
    echo \"Node \$SLURM_NODEID: Waiting for Node 0 to finish pip install...\"
    while [ ! -f $INSTALL_FLAG ]; do
        sleep 5
    done
fi

accelerate launch \
    --config_file=$SCRATCH/ActiveUltraFeedback/configs/accelerate/deepspeed2.yaml \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_NNODES \
    --machine_rank \$SLURM_NODEID \
    --main_process_ip $MAIN_PROCESS_IP \
    --main_process_port $MAIN_PROCESS_PORT \
    -m activeuf.dpo.training \
    --config_path $SCRATCH/ActiveUltraFeedback/configs/dpo_training.yaml \
    --output_dir $OUTPUT_DIR \
    --dataset_path $DATASET_PATH \
    --seed $SEED \
    $DEBUG_ARG
"

echo "Executing command with sync barrier:"
# Added quotes around "$CMD" so it prints with proper newlines in your log
echo "$CMD"

START=$(date +%s)

srun --environment=activeuf bash -c "$CMD"

END=$(date +%s)
DURATION=$(( END - START ))

# Clean up the flag file after the job finishes
rm -f "$INSTALL_FLAG"

echo "Job ended at: $(date)"
echo "Total execution time: $DURATION seconds"