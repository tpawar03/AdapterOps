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

# Install into a venv, never --user. Lambda Stack (and most ML images) ship system
# packages compiled against numpy 1.x; vLLM pulls numpy 2, and every compiled system
# package then fails on an ABI mismatch — scipy, scikit-learn, ml_dtypes, one after
# another. A venv does not inherit /usr/lib/python3/dist-packages, so the problem
# cannot arise:
#     python3 -m venv ~/venv && source ~/venv/bin/activate
#     pip install vllm pandas pyarrow requests

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
  --gpu-memory-utilization 0.90
