#!/bin/bash
#SBATCH --job-name=rm_evals
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=01:30:00
#SBATCH -A ${SLURM_ACCOUNT:-your-slurm-account}
#SBATCH --output=./checkpoint_processing/O-%x.%j
#SBATCH --error=./checkpoint_processing/E-%x.%j
#SBATCH --environment=activeuf_dev      # using compressed docker image as an environment

# Activate your environment if needed
# source /path/to/your/env/bin/activate

suffix=random

# Step 1: Run Python processing
python ${PROJECT_ROOT:-$(pwd)}/scripts/reward_model/checkpoint_processing.py --checkpoints_dir=${MODELS_DIR:-/path/to/models}/reward_models/preference_random_llama_checkpoints_good --output_txt=${PROJECT_ROOT:-$(pwd)}/processed_checkpoints_$suffix.txt

pip install ./resources/reward-bench

# Step 2: Run reward bench for all processed checkpoints
bash ${PROJECT_ROOT:-$(pwd)}/scripts/reward_model/run_reward_bench_2_from_file.sh ${PROJECT_ROOT:-$(pwd)}/processed_checkpoints_$suffix.txt