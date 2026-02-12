#!/bin/bash
#SBATCH -J tb-vllm
#SBATCH -o sbatch_logs/tb-vllm.%j.out
#SBATCH -e sbatch_logs/tb-vllm.%j.err
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 03:00:00
#SBATCH -p gh                     # uncomment/adjust partition if needed
#SBATCH -A CCR24067              # CCR24067, CGAI24022
##SBATCH --mem=64G                # add memory if your site requires explicit mem

set -euo pipefail
cd /work/10672/wenxuand/vista/code-explorer/code_explorer/src/models
# ---- USER VARS: update these paths/values for your env ----
HF_HOME="/scratch/10672/wenxuand/hf_home/"
# CONDA_ENV="/scratch/08002/gsmyrnis/miniconda3/envs/dcft"
# CONDA_ENV="/work/10672/wenxuand/anaconda3/envs/vllm"
# CONDA_ENV="/work/10672/wenxuand/anaconda3/envs/code-explorer"
CONDA_ENV="/work/10672/wenxuand/anaconda3/envs/vllm-92"

VLLM_PORT=8000
VLLM_HOST="127.0.0.1"            # connect via loopback; server binds 0.0.0.0
# MODEL_NAME="Qwen/Qwen3-8B"
# MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
MODEL_NAME="Qwen/Qwen3-8B"
MODEL_NAME="wenwenD/qwen3-8b-codeexp_grpo_with_prior_think_step50_2026-01-27_03-19-15_nvidia_balanced"
MODEL_NAME="wenwenD/qwen3-8b-codeexp_grpo_with_prior_think_step200_2026-01-27_03-19-15_nvidia_balanced"
# MODEL_NAME="wenwenD/qwen3-8b-codeexp_grpo_with_prior_think_step150_2026-01-27_03-19-15_nvidia_balanced"
# MODEL_NAME="wenwenD/qwen3-8b-codeexp_grpo_with_prior_think_step100_2026-01-27_03-19-15_nvidia_balanced"
# MODEL_NAME="wenwenD/qwen3-8b-codeexp_grpo_with_prior_think_step300_2026-01-27_03-19-15_nvidia_balanced"
MODEL_NAME="wenwenD/qwen3-8b-codeexp_grpo_with_prior_think_step350_2026-01-27_03-19-15_nvidia_balanced"



# MODEL_NAME="wenwenD/qwen3-8B_csv_format_sft_10epcs"
# MODEL_NAME="wenwenD/qwen3-1.7b-csv_explorer_grpo_step100"
# MODEL_NAME="Qwen/Qwen3-1.7B"
# MODEL_NAME="Qwen/Qwen3-8B"
# MODEL_NAME="wenwenD/qwen3-4b-codeexp_grpo_w_prior_think_step280_2026-01-25_06-28-54_nvidia_balanced"
# MODEL_NAME="wenwenD/qwen3-4b-codeexp_grpo_no_prior_think_step280_2026-01-25_06-29-13_nvidia_balanced"

# MODEL_NAME="Qwen/Qwen3-0.6B"
# MODEL_NAME="Qwen/Qwen3-1.7B"

GPU_MEM_UTIL="0.85"

# Remote Docker VM SSH
# SSH_KEY="/scratch/10672/wenxuand/dc-agent-shared/.ssh/docker_vm_key"
# # SSH_KEY="/home/08134/negin/.ssh/docker_vm_key"
# SSH_USER_HOST="negin_raoof@34.42.235.32"
# DOCKER_REMOTE_TCP="127.0.0.1:23750"
# DOCKER_LOCAL_TCP="127.0.0.1:23751"

# Terminal-Bench repo dir (inside the batch job working dir)
# TB_DIR="$PWD/terminal-bench"

# If your TB wrapper isn't patched to always include max_tokens & ignore response_format,
# keep this; vLLM requires max_tokens.
# TB_EXTRA_ARGS=( "--agent-kwarg" "max_tokens=512" "--agent-kwarg" "temperature=0.7" )
# -----------------------------------------------------------

echo "[INFO] Node: $HOSTNAME"
date

module load cuda
export CC=gcc
export CXX=g++
# source /scratch/08002/gsmyrnis/miniconda3/etc/profile.d/conda.sh
source /work/10672/wenxuand/anaconda3/etc/profile.d/conda.sh 

conda activate "$CONDA_ENV"

# --- Start vLLM server in background ---
 # set to mmddhhmm if not running in SLURM
# SLURM_JOB_ID=${SLURM_JOB_ID:-$(date +%m%d%H%M)}
SLURM_JOB_ID=${SLURM_JOB_ID:-$(date +%m%d%H%M)}
LOG_DIR="$PWD/runs/_job_${SLURM_JOB_ID}"
mkdir -p "$LOG_DIR"
VLLM_LOG="$LOG_DIR/vllm.log"
echo "[INFO] Logs in $LOG_DIR"

echo "[INFO] Starting vLLM..."
# export LD_LIBRARY_PATH=/work/10672/wenxuand/anaconda3/envs/vllm/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"


nohup env HF_HOME="/scratch/10672/wenxuand/hf_home/" \
         HF_HUB_CACHE="/scratch/10672/wenxuand/hf_home/" \
         HUGGINGFACE_HUB_CACHE="/scratch/10672/wenxuand/hf_home/" \
         TRANSFORMERS_CACHE="/scratch/10672/wenxuand/hf_home/" \
         python -m vllm.entrypoints.openai.api_server \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port "$VLLM_PORT" \
  --model "$MODEL_NAME" \
  --served-model-name "$MODEL_NAME" \
  --device cuda \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  >"$VLLM_LOG" 2>&1 &

  # --reasoning-parser deepseek_r1 \
  # --enable-reasoning \
VLLM_PID=$!

LD_PRELOAD="/usr/lib64/libssl.so.3.0.7:/usr/lib64/libcrypto.so.3.0.7"

cleanup() {
  echo "[CLEANUP] Killing vLLM (PID $VLLM_PID) and SSH tunnel (if any)..."
  if kill -0 "$VLLM_PID" 2>/dev/null; then kill "$VLLM_PID" || true; fi
  if [[ -f "$LOG_DIR/ssh_tun.pid" ]]; then
    TUN_PID=$(cat "$LOG_DIR/ssh_tun.pid" || true)
    if [[ -n "${TUN_PID:-}" ]] && kill -0 "$TUN_PID" 2>/dev/null; then kill "$TUN_PID" || true; fi
  fi
}
trap cleanup EXIT

# --- Wait for vLLM readiness ---
echo "[INFO] Waiting for vLLM on http://${VLLM_HOST}:${VLLM_PORT}/v1/models ..."
for i in {1..60}; do
  if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null; then
    echo "[INFO] vLLM is up."
    break
  fi
  sleep 2
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[ERROR] vLLM died during startup. Last 50 lines:"
    tail -n 50 "$VLLM_LOG" || true
    exit 1
  fi
  if [[ $i -eq 60 ]]; then
    echo "[ERROR] vLLM did not become ready. Log tail:"
    tail -n 100 "$VLLM_LOG" || true
    exit 1
  fi
done

# --- Bring up SSH tunnel to remote Docker daemon ---
echo "[INFO] Starting SSH tunnel to $SSH_USER_HOST (Docker $DOCKER_REMOTE_TCP -> local $DOCKER_LOCAL_TCP)..."
# Ensure key perms are strict
chmod 600 "$SSH_KEY"
# ExitOnForwardFailure fails fast if bind/listen fails
ssh -fN -o ExitOnForwardFailure=yes -i "$SSH_KEY" \
  -L "${DOCKER_LOCAL_TCP}:${DOCKER_REMOTE_TCP}" \
  "$SSH_USER_HOST"
echo $! > "$LOG_DIR/ssh_tun.pid"
chmod 777 "$SSH_KEY"

# # Verify Docker proxy reachable
# echo "[INFO] Verifying Docker API via tunnel..."
# for i in {1..10}; do
#   if curl -s "http://${DOCKER_LOCAL_TCP}/version" | grep -q '"ApiVersion"'; then
#     echo "[INFO] Docker tunnel OK."
#     break
#   fi
#   sleep 1
#   if [[ $i -eq 10 ]]; then
#     echo "[ERROR] Docker tunnel verification failed."
#     exit 1
#   fi
# done

conda activate vllm-92
export HF_HOME=/scratch/10672/wenxuand/hf_home/
export CC=gcc
# reasoning: always do reasoning, the reasoning is handled by litellm generate_response_url, "chat_template_kwargs": {"enable_thinking": false}"
vllm serve Qwen/Qwen3-8B --enable-reasoning #--reasoning-parser qwen3
# VLLM_USE_V1=0
# # no reasoning:
# vllm serve Qwen/Qwen3-8B #--reasoning-parser qwen3