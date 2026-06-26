#!/bin/bash

#SBATCH --job-name=compute_base_model_features
#SBATCH -D .
#SBATCH -A infra01
#SBATCH --output=./logs/O-%x.%j
#SBATCH --error=./logs/E-%x.%j
#SBATCH --time=12:00:00
#SBATCH --environment=activeuf_thesis
#SBATCH --partition=normal

python -m pip install -e resources/rewarduq

accelerate launch \
    --config_file=configs/accelerate/single_node.yaml \
    -m activeuf.loop.compute_base_model_features --config_path configs/compute_base_model_features.yaml