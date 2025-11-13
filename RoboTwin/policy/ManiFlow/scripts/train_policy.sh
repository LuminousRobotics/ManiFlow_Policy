DEBUG=False
save_ckpt=True
train=True

alg_name=${1}
task_name=${2}
setting=${3}
expert_data_num=${4}
config_name=${alg_name}
addition_info=${5}
seed=${6}
gpu_id=${7}

exp_name=${task_name}-${alg_name}-${addition_info}
run_dir="data/outputs/${exp_name}_seed${seed}"



# Environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export VK_ICD_FILENAMES="${SCRIPT_DIR}/nvidia_icd.json"
export TOKENIZERS_PARALLELISM=false
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=${gpu_id}


# Set wandb mode based on debug flag
if [ $DEBUG = True ]; then
    wandb_mode=offline
    echo -e "\033[33m=== DEBUG MODE ===\033[0m"
else
    wandb_mode=online
    echo -e "\033[33m=== TRAINING MODE ===\033[0m"
fi

# Change to workspace directory
cd ManiFlow/maniflow/workspace

# Training phase
echo -e "\033[32m=== Starting Training ===\033[0m"
python train_maniflow_robotwin2_workspace.py \
    --config-name=${config_name}.yaml \
    task_name=${task_name} \
    hydra.run.dir=${run_dir} \
    training.debug=$DEBUG \
    training.seed=${seed} \
    training.device="cuda:0" \
    exp_name=${exp_name} \
    logging.mode=${wandb_mode} \
    checkpoint.save_ckpt=${save_ckpt} \
    expert_data_num=${expert_data_num} \
    setting=${setting}
    
   