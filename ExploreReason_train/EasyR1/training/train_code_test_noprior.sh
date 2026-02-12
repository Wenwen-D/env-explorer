#!/bin/bash


set -e
set -x


export WANDB_CONFIG_DIR=TODO_PATH_TO_WANDB_CONFIG_DIR
export NETRC=TODO_PATH_TO_NETRC_PATH
export WANDB_ENTITY=TODO_WANDB_ENTITY_NAME


export VLLM_USE_V1=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTHONUNBUFFERED=1
export HF_HOME=TODO_PATH_TO_HF_HOME
export CURATOR_DISABLE_CACHE=1



MODEL_PATH=Qwen/Qwen3-8B

# Feature Flags
USE_PRIOR=false      # true = with_prior, false = no_prior
USE_THINKING=true  # true = with_thinking, false = no_thinking

# ============================================
# Auto-generated Names and Paths
# ============================================

# Generate model name from path (replace / with _)
MODEL_NAME="${MODEL_PATH//\//_}"

# Build experiment name
if [ "$USE_PRIOR" = true ]; then
    PRIOR_STR="with_prior"
else
    PRIOR_STR="no_prior"
fi

if [ "$USE_THINKING" = true ]; then
    THINKING_STR="with_thinking"
    ENABLE_THINKING_FLAG=true
else
    THINKING_STR="no_thinking"
    ENABLE_THINKING_FLAG=false
fi

# Select data files based on USE_PRIOR
if [ "$USE_PRIOR" = true ]; then
    TRAIN_FILES="PATH_TO_ExploreReason_train/data_generation_diverse/code_test_explorer_0.5_1_2_4_balanced_rl_with_prior@train"
    VAL_FILES="PATH_TO_ExploreReason_train/data_generation_diverse/code_test_explorer_0.5_1_2_4_balanced_rl_with_prior@val"
else
    TRAIN_FILES="PATH_TO_ExploreReason_train/data_generation_diverse/code_test_explorer_0.5_1_2_4_balanced_rl_no_prior@train"
    VAL_FILES="PATH_TO_ExploreReason_train/data_generation_diverse/code_test_explorer_0.5_1_2_4_balanced_rl_no_prior@val"
fi

# Set length parameters based on USE_THINKING
if [ "$USE_THINKING" = true ]; then
    MAX_PROMPT_LENGTH=11200
    MAX_RESPONSE_LENGTH=6400
    SINGLE_TURN_RESPONSE_LENGTH=2048
else
    MAX_PROMPT_LENGTH=1024
    MAX_RESPONSE_LENGTH=256
    SINGLE_TURN_RESPONSE_LENGTH=128
fi


ALL_TASK_BASE_PATH=TODO_PATH_TO/code_explorer_balanced_data
# Construct experiment name
BASE_EXPERIMENT_NAME="code_test_explorer_${MODEL_NAME}_${PRIOR_STR}_${THINKING_STR}_balanced_nvidia"
BASE_SAVE_PATH="TODO_PATH_TO_SAVE_DIR/code_test_explorer/model_ckpts/${BASE_EXPERIMENT_NAME}"

TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
FULL_EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${TIMESTAMP}"
FULL_SAVE_PATH="${BASE_SAVE_PATH}_${TIMESTAMP}"

LOG_BASE_DIR="TODO_PATH_TO_LOG_DIR/_runs"
LOG_PATH="${LOG_BASE_DIR}/${FULL_EXPERIMENT_NAME}"
mkdir -p "$LOG_PATH"

echo "============================================"
echo "Experiment Configuration:"
echo "============================================"
echo "Model: ${MODEL_PATH} (${MODEL_NAME})"
echo "Prior: ${PRIOR_STR}"
echo "Thinking: ${THINKING_STR}"
echo "Experiment Name: ${FULL_EXPERIMENT_NAME}"
echo "Save Path: ${FULL_SAVE_PATH}"
echo "Train Files: ${TRAIN_FILES}"
echo "Val Files: ${VAL_FILES}"
echo "Log Path: ${LOG_PATH}"
echo "Model Save Path: ${FULL_SAVE_PATH}"
echo "============================================"




python -m EasyR1.verl.trainer.main \
   config=EasyR1/training/config_explorer_multi_turn_code_test_explore.yaml \
   data.train_files=${TRAIN_FILES} \
   data.val_files=${VAL_FILES} \
   worker.actor.model.model_path=${MODEL_PATH} \
   worker.rollout.tensor_parallel_size=1 \
   worker.rollout.all_task_base_path=${ALL_TASK_BASE_PATH} \
   trainer.project_name=EasyR1_code_test_explorer \
   trainer.experiment_name=${FULL_EXPERIMENT_NAME} \
   trainer.nnodes=1 \
   trainer.n_gpus_per_node=8 \
   trainer.save_checkpoint_path=${FULL_SAVE_PATH} \
   data.rollout_batch_size=16 \
   worker.actor.global_batch_size=16 \
   worker.actor.micro_batch_size_per_device_for_update=2 \
   worker.actor.micro_batch_size_per_device_for_experience=4 \
   worker.rollout.n=8 \
   worker.rollout.enable_thinking=${ENABLE_THINKING_FLAG} \
   trainer.total_epochs=2 \
   worker.rollout.temperature=1.0 \
   worker.actor.optim.lr=1e-6 \
   data.max_prompt_length=${MAX_PROMPT_LENGTH} \
   data.max_response_length=${MAX_RESPONSE_LENGTH} \
   worker.rollout.single_turn_response_length=${SINGLE_TURN_RESPONSE_LENGTH} \
   algorithm.overlong_filtering=False \
   algorithm.void_trace_filtering=False \
   algorithm.disable_kl=False \
   algorithm.use_kl_loss=True \
   algorithm.kl_coef=0.01 \
   data.shuffle=True \
   worker.reward.reward_function_kwargs.log_dir=$LOG_PATH \
   worker.reward.reward_function_kwargs.max_turns=6 \
   worker.rollout.max_turns=6 \
   worker.rollout.gpu_memory_utilization=0.6 \
   worker.rollout.prompt_length=${MAX_PROMPT_LENGTH} \
   worker.rollout.single_turn_response_length=${SINGLE_TURN_RESPONSE_LENGTH} \
   worker.rollout.response_length=${MAX_RESPONSE_LENGTH} \
   worker.rollout.max_model_len=40960 \
   2>&1 | tee "${LOG_PATH}/run.log"
