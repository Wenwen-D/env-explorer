#!/bin/bash

# ============================================
# vLLM Server Script for NVIDIA A100 Machine
# ============================================

set -e
set -x

cd /home/nvidia/wenxuand/code-explorer/work/10672/wenxuand/vista/code-explorer/code_explorer/src/models

# ---- USER VARS: update these paths/values for your env ----
HF_HOME="/home/nvidia/wenxuand/code-explorer/scratch/10672/wenxuand/hf_home"
CONDA_ENV="/mnt/users/lytang/miniconda3/envs/code_explorer"

# GPU Configuration for A100
# Specify which GPUs to use (e.g., "0" for GPU 0, "0,1" for GPUs 0 and 1, etc.)
export CUDA_VISIBLE_DEVICES=7

# vLLM Server Configuration
VLLM_PORT=8000
VLLM_HOST="127.0.0.1"            # connect via loopback; server binds 0.0.0.0

# Model Selection - uncomment the one you want to use
# MODEL_NAME="Qwen/Qwen3-8B"
# MODEL_NAME="Qwen/Qwen3-4B-Thinking-2507"
MODEL_NAME="Qwen/Qwen3-8B"
# MODEL_NAME="Qwen/Qwen3-1.7B"
# MODEL_NAME="Qwen/Qwen3-0.6B"
# MODEL_NAME="wenwenD/qwen3-8B_csv_format_sft_10epcs"
# MODEL_NAME="wenwenD/qwen3-1.7b-csv_explorer_grpo_step100"

# A100 optimized settings
GPU_MEM_UTIL="0.85"
TENSOR_PARALLEL_SIZE=1  # Set to >1 if using multiple GPUs

# Reasoning mode (set to true to enable reasoning/thinking)
ENABLE_REASONING=false

# -----------------------------------------------------------

echo "[INFO] Node: $HOSTNAME"
echo "[INFO] GPUs: $CUDA_VISIBLE_DEVICES"
date

# No module load needed on NVIDIA machine - CUDA already available
export CC=gcc
export CXX=g++

# Activate conda environment
source /mnt/users/lytang/miniconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# --- Start vLLM server in background ---
JOB_ID=$(date +%m%d%H%M)
LOG_DIR="$PWD/runs/_job_${JOB_ID}"
mkdir -p "$LOG_DIR"
VLLM_LOG="$LOG_DIR/vllm.log"
echo "[INFO] Logs in $LOG_DIR"

echo "[INFO] Starting vLLM server..."
echo "[INFO] Model: $MODEL_NAME"
echo "[INFO] GPU Memory Utilization: $GPU_MEM_UTIL"
echo "[INFO] Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "[INFO] Reasoning enabled: $ENABLE_REASONING"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export HF_HOME="$HF_HOME"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"

# Build vLLM command
VLLM_CMD="vllm serve $MODEL_NAME \
  --host 0.0.0.0 \
  --port $VLLM_PORT \
  --gpu-memory-utilization $GPU_MEM_UTIL \
  --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
  --trust-remote-code"

# Add reasoning flags if enabled
if [ "$ENABLE_REASONING" = true ]; then
  VLLM_CMD="$VLLM_CMD --enable-reasoning"
  # Uncomment to add specific reasoning parser:
  # VLLM_CMD="$VLLM_CMD --reasoning-parser qwen3"
fi

# Start vLLM in background
nohup bash -c "$VLLM_CMD" > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
echo "[INFO] vLLM PID: $VLLM_PID"

# Cleanup function
cleanup() {
  echo "[CLEANUP] Killing vLLM (PID $VLLM_PID)..."
  if kill -0 "$VLLM_PID" 2>/dev/null; then
    kill "$VLLM_PID" || true
    sleep 2
    # Force kill if still running
    kill -9 "$VLLM_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# --- Wait for vLLM readiness ---
echo "[INFO] Waiting for vLLM on http://${VLLM_HOST}:${VLLM_PORT}/v1/models ..."
for i in {1..60}; do
  if curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "[INFO] vLLM is up and ready!"
    curl -s "http://${VLLM_HOST}:${VLLM_PORT}/v1/models" | head -20
    break
  fi
  sleep 3
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[ERROR] vLLM died during startup. Last 50 lines:"
    tail -n 50 "$VLLM_LOG" || true
    exit 1
  fi
  if [[ $i -eq 60 ]]; then
    echo "[ERROR] vLLM did not become ready in 3 minutes. Log tail:"
    tail -n 100 "$VLLM_LOG" || true
    exit 1
  fi
  echo "[INFO] Still waiting... ($i/60)"
done

echo ""
echo "==============================================="
echo "vLLM Server is running!"
echo "==============================================="
echo "API endpoint: http://${VLLM_HOST}:${VLLM_PORT}/v1"
echo "Model: $MODEL_NAME"
echo "Log file: $VLLM_LOG"
echo "PID: $VLLM_PID"
echo ""
echo "To test the server, run:"
echo "  curl http://${VLLM_HOST}:${VLLM_PORT}/v1/models"
echo ""
echo "To stop the server, press Ctrl+C or kill PID $VLLM_PID"
echo "==============================================="

# Keep script running (server is in background)
echo "[INFO] Tailing log file (Ctrl+C to exit)..."
tail -f "$VLLM_LOG"
