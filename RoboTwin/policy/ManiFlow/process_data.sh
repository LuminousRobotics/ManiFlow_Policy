#!/bin/bash
# bash process_data.sh ${task_name} ${task_config} ${expert_data_num}
# Example: 
# bash process_data.sh lift_pot demo_randomized 50
# bash process_data.sh pick_dual_bottles demo_randomized 50
# bash process_data.sh put_object_cabinet demo_randomized 50

task_name=${1}
task_config=${2}
expert_data_num=${3}

python scripts/process_data.py $task_name $task_config $expert_data_num