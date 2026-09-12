#!/usr/bin/env bash
# Serve the base model with ALL FOUR pinned adapters concurrently (F5, M1).
#
# Run on the rented GPU box. Blocks; run the generation client in a second shell:
#     uv run python -m adapterops.router.generate
#
# Adapters are resolved through manifests/adapters.json and downloaded at their pinned
# revision, so what is served does not depend on where `main` points today. Verify the
# pins before spending anything:
#     uv run adapterops verify-pins
#
# Venv, never --user — Lambda Stack's system packages are built against numpy 1.x and
# vLLM pulls numpy 2 (GATE-0.5.md):
#     python3 -m venv ~/venv && source ~/venv/bin/activate
#     pip install vllm pandas pyarrow requests huggingface_hub
set -euo pipefail

cd "$(dirname "$0")/.."
export PYTHONPATH=src

PORT="${PORT:-8000}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"   # >= TrainConfig.lora_r
MAX_LORAS="${MAX_LORAS:-4}"            # all four resident in one batch — this is M1
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1536}" # PII prompts reach ~320 tok; drafting generates ~400

# PINS=manifests/candidates/<name>.json serves a candidate set (M7, M11) under the same
# adapter names; without it, manifests/adapters.json is served.
eval "$(python -m adapterops.serve.launch --print-args ${PINS:+--pins "$PINS"})"
echo "  base:    $BASE @ ${BASE_REVISION:0:8}"
echo "  pin set:  $PIN_SET"
echo "  adapters: $LORA_MODULES"

python -c "import vllm; print('vllm', vllm.__version__)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sed 's/^/  gpu: /'

# shellcheck disable=SC2086
exec vllm serve "$BASE" \
  --revision "$BASE_REVISION" \
  --port "$PORT" \
  --enable-lora \
  --max-loras "$MAX_LORAS" \
  --max-lora-rank "$MAX_LORA_RANK" \
  --lora-modules $LORA_MODULES \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization 0.90
