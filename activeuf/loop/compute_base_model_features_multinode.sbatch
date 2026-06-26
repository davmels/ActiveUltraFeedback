#!/bin/bash

#SBATCH --job-name=compute_base_model_features
#SBATCH -D .
#SBATCH -A infra01
#SBATCH --output=./logs/O-%x.%j
#SBATCH --error=./logs/E-%x.%j
#SBATCH --time=12:00:00
#SBATCH --partition=normal
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4

# NOTE: the batch script runs on the HOST (no #SBATCH --environment), so that
# `srun` can load Slurm's host-side Slingshot plugin. The container is applied
# per-task via `srun --environment=...` (the CSCS-recommended pattern).

# rendezvous on the first node of the allocation
MAIN_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
NUM_PROCESSES=$((SLURM_NNODES * 4))
echo "Master: $MAIN_ADDR | nodes: $SLURM_NNODES | total GPUs/procs: $NUM_PROCESSES"

# one srun task per node; accelerate spawns 4 local processes (one per GPU).
# \$SLURM_NODEID is escaped so it expands per-task (the per-node machine rank).
srun --environment=activeuf_thesis bash -c "accelerate launch \
    --multi_gpu \
    --mixed_precision=bf16 \
    --num_machines=$SLURM_NNODES \
    --num_processes=$NUM_PROCESSES \
    --machine_rank=\$SLURM_NODEID \
    --main_process_ip=$MAIN_ADDR \
    --main_process_port=14014 \
    -m activeuf.loop.compute_base_model_features --config_path configs/compute_base_model_features.yaml"
