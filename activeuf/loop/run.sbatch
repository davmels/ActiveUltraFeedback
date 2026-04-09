#!/bin/bash

#SBATCH --job-name=multinode_${acquisition_function}_${annotator_model}_loop
#SBATCH -D .
#SBATCH -A ${SLURM_ACCOUNT:-your-slurm-account}
#SBATCH --output=./logs/loopman.out
#SBATCH --error=./logs/loopman.err
#SBATCH --nodes=4                   # number of nodes
#SBATCH --ntasks-per-node=1         # number of MP tasks
#SBATCH --gres=gpu:4                # number of GPUs per node
#SBATCH --cpus-per-task=288         # number of cores per tasks
#SBATCH --time=12:00:00             # maximum execution time (HH:MM:SS)
#SBATCH --environment=activeuf      # using compressed docker image as an environment

export GPUS_PER_NODE=4
export HF_HOME=${HF_HOME:-/path/to/hf_cache}
######################

echo $ACCELERATE_DIR

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

export LAUNCHER="accelerate launch \
    --config_file=configs/accelerate/multi_node.yaml \
    --num_processes $((SLURM_NNODES * GPUS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    -m
    "

export ACCELERATE_DIR="${ACCELERATE_DIR:-/accelerate}"
export PYTHON_FILE="activeuf.active_learning_loop"
export SCRIPT_ARGS=" \
   --completions_dataset_path \${SCRATCH}/datasets/combined_annotations_${annotator_model}/ \
   --output_path=\${DATA_DIR:-/path/to/datasets}/preference_new_${acquisition_function}_${annotator_model}_8/ \
    --acquisition_function_type=${acquisition_function} \
    --regularization_weight_decay_type=linear \
    --exponential_decay_base=0.95 \
    --regularization_towards_initial_weights=100 \
   "
        
# This step is necessary because accelerate launch does not handle multiline arguments properly
export CMD="\$LAUNCHER \$PYTHON_FILE \$SCRIPT_ARGS" 

export DS_ACCELERATOR_CACHE_DIR="\${CACHE_DIR:-/tmp}/deepspeed_cache_\$SLURM_JOB_ID"
export TRANSFORMERS_CACHE="\${CACHE_DIR:-/tmp}/hf_cache_\$SLURM_JOB_ID"
mkdir -p "\$DS_ACCELERATOR_CACHE_DIR" "\$TRANSFORMERS_CACHE"

START=\$(date +%s)

cd \${PROJECT_ROOT:-$(pwd)}/
export PYTHONPATH="\${PROJECT_ROOT:-$(pwd)}:\$PYTHONPATH"

export CUDA_VISIBLE_DEVICES=0,1,2,3

srun --chdir=\${PROJECT_ROOT:-$(pwd)} \$CMD

END=\$(date +%s)
DURATION=\$(( END - START ))

echo "Job ended at: \$(date)"
echo "Total execution time: \$DURATION seconds"