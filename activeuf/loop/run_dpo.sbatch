#!/bin/bash

#SBATCH --job-name=active_dpo
#SBATCH -D .
#SBATCH -A infra01
#SBATCH --output=./logs/active_dpo/%j-%x.out
#SBATCH --error=./logs/active_dpo/%j-%x.err
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
##SBATCH --reservation=SD-69241-apertus-1-5-0

# The batch script runs on the HOST (no #SBATCH --environment) so srun can load
# Slurm's host-side Slingshot plugin; the container is applied per-task via
# `srun --environment=...` below (the CSCS-recommended pattern).

export GPUS_PER_NODE=4
export HF_HOME=$SCRATCH/huggingface
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export TOKENIZERS_PARALLELISM=false

# Required: threshold C. Optional: CONFIG path override.
THRESHOLD="${THRESHOLD:?ERROR: THRESHOLD not set (e.g. THRESHOLD=2.5 sbatch activeuf/loop/run_dpo.sbatch)}"
CONFIG="${CONFIG:-configs/loop_dpo.yaml}"

# Optional explicit run id (else auto = activedpo_C{threshold}_{timestamp})
RUN_ID_ARG=""
[[ -n "${RUN_ID:-}" ]] && RUN_ID_ARG="--run_id=$RUN_ID"

head_node_ip=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# One accelerate launcher per node (ntasks-per-node=1). Each task sets its own
# machine_rank from SLURM_NODEID; accelerate then spawns GPUS_PER_NODE procs/node.
# \$SLURM_NODEID is escaped so it is evaluated per-task inside srun, not here.
export CMD="accelerate launch \
    --config_file=configs/accelerate/deepspeed2.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --machine_rank \$SLURM_NODEID \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    -m activeuf.loop.run_dpo \
    --config_path $CONFIG \
    --threshold=$THRESHOLD \
    $RUN_ID_ARG \
    --debug=${DEBUG:-false} \
    --debug_n=${DEBUG_N:-1001}"

srun --environment=activeuf_thesis bash -c "$CMD"
