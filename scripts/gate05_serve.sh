#!/usr/bin/env bash
# Gate 0.5 — serve ONE base model with TWO LoRA adapters concurrently under vLLM.
#
# Run this on the rented GPU box, not locally. It blocks; run the probe in a second shell.
#
# The two adapters are deliberately of different quality: the trained intent adapter and
# the M11 under-trained checkpoint. If vLLM silently served one adapter for both names,
# their outputs would match — and the gate would look passed when it was not. The quality
# gap between them is the discriminator.
set -euo pipefail

BASE="${BASE:-Qwen/Qwen2.5-1.5B-Instruct}"
HF_USER="${HF_USER:-Tanny03}"
PORT="${PORT:-8000}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"   # must be >= TrainConfig.lora_r
MAX_LORAS="${MAX_LORAS:-2}"            # adapters resident in a single batch

echo "vllm version:"; python -c "import vllm; print(' ', vllm.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  gpu: /'

exec vllm serve "$BASE" \
  --port "$PORT" \
  --enable-lora \
  --max-loras "$MAX_LORAS" \
  --max-lora-rank "$MAX_LORA_RANK" \
  --lora-modules \
      "intent=${HF_USER}/adapterops-intent" \
      "intent-undertrained=${HF_USER}/adapterops-intent-undertrained-m11" \
  --max-model-len 1024 \
  --gpu-memory-utilization 0.90 \
  --disable-log-requests
