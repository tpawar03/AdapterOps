#!/usr/bin/env bash
# One GPU session: training variance for the provisional gates, and GPT-4o-mini under load.
#
# Run on the rented A10 from the repository root, set up as for gpu_load_session.sh
# (uv sync --extra gpu, uv pip install vllm==0.29.0 "flashinfer-cubin==0.6.18", .env with the key):
#     source .venv/bin/activate && bash scripts/gpu_training_variance_session.sh
#
# What it does, in order, stopping at the first failure. Cheap results come first, so a failure in
# the long training step still leaves them on disk.
#   1. verify-pins; refuse any vLLM but 0.29.0.
#   2. vLLM with the served adapters; a regression run of them ($NAME-original).
#   3. GPT-4o-mini under load: serve-api with the frontier on, FRONTIER_SECONDS at concurrency 32.
#      ~43 pairs/s with 20% escalated is ~520 calls a minute — at or past the account's per-minute
#      limit — so this measures GPT-4o-mini's latency and how rate-limit errors land (the pair keeps
#      its local answer and the error is counted). ~2,600 calls: a quarter of the daily cap, ~$0.35.
#   4. Stop vLLM and wait for the GPU to be free.
#   5. A second training run of each task in $TASKS at the configuration that produced the served
#      adapter — same seed, rows and epochs — to checkpoints/<task>-rerun. Nothing is pushed to the
#      Hub. Shortest first: drafting ~23 min, urgency ~35, PII ~61 (8,000 rows, 3 epochs — set by
#      override when it was trained, so repeated by override here).
#   6. vLLM with those checkpoints swapped in (a local pin set), and a regression run ($NAME-rerun).
#
# About 2½–3 hours. Afterwards, on the laptop, where the judge lives:
#     uv run adapterops judge-score --run runs/regression__$NAME-original.json
#     uv run adapterops judge-score --run runs/regression__$NAME-rerun.json
#     uv run adapterops training-variance --original runs/regression__$NAME-original.json \
#         --rerun runs/regression__$NAME-rerun.json --tasks urgency pii drafting
#     uv run adapterops derive-thresholds --run-a runs/regression__v1-baseline-1.json \
#         --run-b runs/regression__v1-baseline-2.json --force
#
# Bring back (scp -i <key> ...):
#     runs/regression__$NAME-original.json  and its __predictions.parquet
#     runs/regression__$NAME-rerun.json     and its __predictions.parquet
#     runs/request_path__$NAME-frontier-load.json  and its __pairs.parquet
#     runs/urgency-rerun__train.json  runs/pii-rerun__train.json  runs/drafting-rerun__train.json
#     runs/logs/
set -euo pipefail

cd "$(dirname "$0")/.."
export UV_NO_SYNC=1
export PYTHONPATH=src
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
NAME="${NAME:-a10-v4}"
TASKS="${TASKS:-drafting urgency pii}"
VLLM_PORT="${PORT:-8000}"
API_PORT="${API_PORT:-8080}"
FRONTIER_SECONDS="${FRONTIER_SECONDS:-300}"
FRONTIER_CONCURRENCY="${FRONTIER_CONCURRENCY:-32}"
RERUN_PINS="_adapters/rerun-pins.json"
mkdir -p runs/logs

if ! grep -qs '^OPENAI_API_KEY=' .env && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "no OPENAI_API_KEY in the environment or .env — step 3 needs it" >&2
  exit 1
fi

installed=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo none)
if [[ "$installed" != "$VLLM_VERSION" ]]; then
  echo "vllm is $installed, the recorded runs used $VLLM_VERSION:" >&2
  echo "  uv pip install vllm==$VLLM_VERSION   (or VLLM_VERSION=$installed to accept the difference)" >&2
  exit 1
fi
python -c "import bitsandbytes, peft" || { echo "training needs the gpu extra: uv sync --extra gpu" >&2; exit 1; }

# Each server runs in its own process group, so stopping it takes vLLM's engine workers with it.
groups=()
cleanup() { for g in "${groups[@]}"; do kill -- -"$g" 2>/dev/null || true; done; }
trap cleanup EXIT

start_group() {  # log, command...
  local log=$1; shift
  setsid "$@" > "$log" 2>&1 &
  groups+=($!)
  LAST_GROUP=$!
}

stop_group() {
  kill -- -"$1" 2>/dev/null || true
  local remaining=()
  for g in "${groups[@]}"; do [[ "$g" != "$1" ]] && remaining+=("$g"); done
  groups=("${remaining[@]}")
}

wait_for() {  # url, what, seconds
  local url=$1 what=$2 deadline=$((SECONDS + $3))
  until curl -sf "$url" >/dev/null; do
    if (( SECONDS > deadline )); then echo "$what did not come up; see runs/logs/" >&2; exit 1; fi
    sleep 5
  done
  echo "  $what is up"
}

wait_gpu_free() {
  local deadline=$((SECONDS + 300)) used
  while true; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    (( used < 1500 )) && { echo "  GPU free (${used} MiB in use)"; return; }
    (( SECONDS > deadline )) && { echo "GPU still holds ${used} MiB after stopping vLLM" >&2; exit 1; }
    sleep 5
  done
}

echo "== 1. pins"
uv run adapterops verify-pins

echo "== 2. served adapters: vLLM and a regression run ($NAME-original)"
start_group runs/logs/vllm-original.log env PORT="$VLLM_PORT" bash scripts/phase2_serve.sh
VLLM_GROUP=$LAST_GROUP
wait_for "http://localhost:$VLLM_PORT/v1/models" "vLLM" 900
uv run adapterops regress --base-url "http://localhost:$VLLM_PORT" --name "$NAME-original" \
  --baseline runs/regression__v1-baseline-1.json --save-predictions 2>&1 | tee runs/logs/regress-original.log

echo "== 3. GPT-4o-mini under load: ${FRONTIER_SECONDS}s at concurrency $FRONTIER_CONCURRENCY"
start_group runs/logs/serve-api-frontier-load.log uv run adapterops serve-api --backend vllm \
  --base-url "http://localhost:$VLLM_PORT" --policy confidence --no-judge --trace none --port "$API_PORT"
API_GROUP=$LAST_GROUP
wait_for "http://127.0.0.1:$API_PORT/healthz" "serve-api with GPT-4o-mini" 300
uv run adapterops request-path-run --url "http://127.0.0.1:$API_PORT" --name "$NAME-frontier-load" \
  --concurrency "$FRONTIER_CONCURRENCY" --duration "$FRONTIER_SECONDS" --window 30 \
  2>&1 | tee runs/logs/request-path-frontier-load.log

echo "== 4. stopping serving to free the GPU for training"
stop_group "$API_GROUP"
stop_group "$VLLM_GROUP"
wait_gpu_free

echo "== 5. second training runs: $TASKS"
for task in $TASKS; do
  extra=()
  [[ "$task" == "pii" ]] && extra=(--subsample 8000 --epochs 3)
  echo "---- $task ${extra[*]}"
  start=$SECONDS
  python -m adapterops.train.cli_train --task "$task" --variant rerun "${extra[@]}" \
    2>&1 | tee "runs/logs/train-$task-rerun.log"
  test -f "checkpoints/$task-rerun/adapter_model.safetensors" \
    || { echo "no adapter written for $task" >&2; exit 1; }
  echo "---- $task finished in $(( (SECONDS - start) / 60 )) min"
done

echo "== 6. retrained adapters: vLLM from local pins and a regression run ($NAME-rerun)"
# shellcheck disable=SC2086
python - "$RERUN_PINS" $TASKS <<'PY'
import json
import sys
from pathlib import Path

out, tasks = Path(sys.argv[1]), sys.argv[2:]
components = json.loads(Path("manifests/system.json").read_text())["components"]
pins = {"base_model": components["base_model"], **components["adapters"]}
for task in tasks:
    pins[task] = {"path": f"checkpoints/{task}-rerun"}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "purpose": "Training-variance rerun: local checkpoints swapped in, never published.",
    "candidate": "rerun", "replaces": tasks, "components": pins}, indent=2) + "\n")
print(f"  pins: {out} — {', '.join(f'{t}=local' for t in tasks)}")
PY
start_group runs/logs/vllm-rerun.log env PORT="$VLLM_PORT" PINS="$RERUN_PINS" bash scripts/phase2_serve.sh
wait_for "http://localhost:$VLLM_PORT/v1/models" "vLLM with the retrained adapters" 900
uv run adapterops regress --base-url "http://localhost:$VLLM_PORT" --name "$NAME-rerun" \
  --baseline "runs/regression__$NAME-original.json" --save-predictions 2>&1 | tee runs/logs/regress-rerun.log

echo "== done — copy back:"
ls -1 runs/regression__"$NAME"-original* runs/regression__"$NAME"-rerun* \
  runs/request_path__"$NAME"-frontier-load* runs/*-rerun__train.json
