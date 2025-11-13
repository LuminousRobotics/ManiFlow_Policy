#!/bin/bash
# bash run_multitask.sh <gpu_id>

# Define tasks array
# task_list=("pick_apple_messy" "block_hammer_beat" "diverse_bottles_pick" "dual_bottles_pick_easy" "dual_bottles_pick_hard" "dual_shoes_place" "empty_cup_place" "mug_hanging_easy" "mug_hanging_hard" "pick_apple_messy" "shoe_place" "tool_adjust")
# task_list=("pick_apple_messy" "block_hammer_beat" "block_handover" "blocks_stack_easy" "blocks_stack_hard" "bottle_adjust" "container_place" "diverse_bottles_pick" "dual_bottles_pick_easy" "dual_bottles_pick_hard" "dual_shoes_place" "empty_cup_place" "mug_hanging_easy" "mug_hanging_hard" "pick_apple_messy" "shoe_place" "tool_adjust" "put_apple_cabinet")
task_list=("pick_apple_messy" "block_hammer_beat" "diverse_bottles_pick" "dual_bottles_pick_easy" "dual_bottles_pick_hard")

gpu_id=${1}

# Set GPU
export CUDA_VISIBLE_DEVICES=${gpu_id}

# Run each task
for task in "${task_list[@]}"; do
    echo "Running task: ${task}"
    echo ${task} | python script/run_task.py
done

echo "All tasks completed"