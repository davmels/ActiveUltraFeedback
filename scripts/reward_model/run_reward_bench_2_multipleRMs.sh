#!/bin/bash

# Array of reward model paths
# Update these paths to point to your own trained models
MODELS=(
    "/path/to/models/reward_models/dts_1"
    "/path/to/models/reward_models/dts_2"
    "/path/to/models/reward_models/dts_3"
    "/path/to/models/reward_models/dts_4"
    "/path/to/models/reward_models/dts_5"
    "/path/to/models/reward_models/skywork_10000"
    "/path/to/models/reward_models/skywork_20000"
    "/path/to/models/reward_models/skywork_30000"
    "/path/to/models/reward_models/skywork_50000"
    "/path/to/models/reward_models/skywork_60000"
    "/path/to/models/reward_models/skywork_70000"
    "/path/to/models/reward_models/skywork_78000"
)

for MODEL in "${MODELS[@]}"; do
    echo "Running reward bench for model: $MODEL"
    bash activeuf/reward_model/run_reward_bench_2.sh "$MODEL"
done