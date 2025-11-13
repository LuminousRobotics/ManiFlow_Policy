#!/bin/bash
# bash run_task.sh pick_apple_messy 4
# bash run_task.sh diverse_bottles_pick 4
# bash run_task.sh dual_bottles_pick_hard 4
# bash run_task.sh empty_cup_place 4
# bash run_task.sh mug_hanging_hard 4
# bash run_task.sh container_place 4
# bash run_task.sh shoe_place 4
# bash run_task.sh dual_shoes_place 4
# bash run_task.sh block_hammer_beat 4

task_name=${1}
gpu_id=${2}

camera_type="D435"  # Camera type, e.g., D435
num_demos=50  # Number of demonstrations to generate

export CUDA_VISIBLE_DEVICES=${gpu_id}
echo ${task_name} | python script/run_task.py ${task_name} ${camera_type} ${num_demos}