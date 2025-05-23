#!/bin/bash

# Define arrays for model names and their corresponding work directories
models=(
    # "lmdeploy_sft_0502_v1"
    # "lmdeploy_sft_0502_v2_rlvr5k"
    # "lmdeploy_sft_0502_v3_rlvr10k"
    # "lmdeploy_sft_0502_v4"
    # "lmdeploy_sft_0502_v5"
    # "lmdeploy_sft_0502_v6"
    # "lmdeploy_sft_0502_v7"
    # "lmdeploy_sft_0502_v8"
    # "lmdeploy_sft_0502_v2"
    # "lmdeploy_sft_0502_v3"
    # "lmdeploy_sft_pub85k_tiku25k"
    # "lmdeploy_rl_pub8k_wllm_coldstart_shuffle"
    # "lmdeploy_sft_pub25k"
    # "lmdeploy_sft_pub25k_tiku25k"
    # "lmdeploy_sft_pub140k_tiku25k"
    # "lmdeploy_sft_0502_v8_rlvr50k"
    # "lmdeploy_sft_0503_v9"
    # "lmdeploy_sft_0503_v10"
    # "lmdeploy_sft_0503_v11"
    # "lmdeploy_sft_0503_v12"
    "lmdeploy_sft_0503_v10_tmp0"
    "lmdeploy_sft_0503_v11_tmp0"
    "lmdeploy_sft_0503_v12_tmp0"
)

work_dirs=(
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_sft_0502_v1"
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_sft_0502_v2_rlvr5k"
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_sft_0502_v3_rlvr10k"
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_sft_0502_v4"
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_sft_0502_v5"
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_sft_0502_v6"
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_sft_0502_v7"
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_sft_0502_v8"
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_sft_0502_v2"
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_sft_0502_v3"
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_pub85k_tiku25k"
    # "outputs/qwen2_5_vl_7b_mmr1_pub8k_wllm_coldstart_shuffle"
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_pub25k"
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_pub25k_tiku25k"
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_pub140k_tiku25k"
    "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_sft_0502_v8_rlvr50k"
    "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_sft_0503_v9"
    "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_sft_0503_v10"
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_sft_0503_v11"
    # "outputs/mmo1-math-qwen2.5_vl_7b-sft_mmr1_sft_0503_v12"
)

# Run all commands in parallel
for i in "${!models[@]}"; do
    python run.py --data MMMU_DEV_VAL MathVista_MINI --model "${models[$i]}" --verbose --api-nproc 16 --work-dir "${work_dirs[$i]}" &
    echo "Started evaluation for ${models[$i]}"
done

# Wait for all background processes to finish
wait

echo "All evaluations completed!" 