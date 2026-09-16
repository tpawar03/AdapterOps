#!/usr/bin/env bash
# One GPU session: retrain every task at further seeds and regression-score each set beside the served
# adapters, so each gate threshold rests on a variance estimate rather than a single range (F33, D37;
# TODO.md §2).
#
# Why seeds. The recorded spreads come from two runs at one seed, which differ only by nondeterminism and
# library drift. The gate's question is whether an equivalent retrain would be blocked, and an equivalent
# retrain is the same configuration at another seed — everything else here stays fixed.
#
# Setup on the rented A10 (repository root): uv sync --extra gpu, then
#     UV_NO_SYNC=1 uv pip install vllm==0.29.0     # pins torch==2.13.0, torchvision==0.28.0
#     UV_NO_SYNC=1 uv pip uninstall flashinfer-cubin   # no release matches flashinfer 0.6.18
# Keep UV_NO_SYNC=1 set: a bare `uv run` re-syncs the lockfile and breaks the pinned torch.
#     source .venv/bin/activate && bash scripts/gpu_variance_seeds_session.sh
#
# SEEDS="11" trains one extra run per task (~3.5 h); the default two give a three-value estimate with a
# standard deviation (~7 h). TASKS can be narrowed. SKIP_TRAIN=1 resumes after a failed serving step.
#
# Afterwards, on the laptop (drafting's metric needs the judge, which lives here):
#     uv run adapterops judge-score --run runs/regression__$NAME-served.json
#     uv run adapterops judge-score --run runs/regression__$NAME-seed11.json      # and seed22
#     uv run adapterops training-variance --runs runs/regression__$NAME-served.json \
#         runs/regression__$NAME-seed11.json runs/regression__$NAME-seed22.json \
#         --tasks intent urgency pii drafting --force
#     uv run adapterops derive-thresholds --run-a runs/regression__v1-baseline-1.json \
#         --run-b runs/regression__v1-baseline-2.json --force
#
# Bring back: runs/regression__$NAME-*.json (and predictions), runs/*-seed*__train.json, runs/logs/
set -euo pipefail

cd "$(dirname "$0")/.."
export UV_NO_SYNC=1
export PYTHONPATH=src
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
NAME="${NAME:-a10-v8-variance}"
VLLM_PORT="${PORT:-8000}"
SEEDS="${SEEDS:-11 22}"
TASKS="${TASKS:-intent urgency pii drafting}"
mkdir -p runs/logs _adapters

installed=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo none)
if [[ "$installed" != "$VLLM_VERSION" ]]; then
  echo "vllm is $installed, the recorded runs used $VLLM_VERSION:" >&2
  echo "  UV_NO_SYNC=1 uv pip install vllm==$VLLM_VERSION" >&2
  exit 1
fi
python -c "import bitsandbytes, peft" || { echo "training needs the gpu extra: uv sync --extra gpu" >&2; exit 1; }

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

regress_with_pins() {  # label, pins-file-or-empty
  local label=$1 pins=${2:-}
  start_group "runs/logs/vllm-$label.log" env PORT="$VLLM_PORT" ${pins:+PINS="$pins"} bash scripts/phase2_serve.sh
  local group=$LAST_GROUP
  wait_for "http://localhost:$VLLM_PORT/v1/models" "vLLM ($label)" 900
  uv run adapterops regress --base-url "http://localhost:$VLLM_PORT" --name "$NAME-$label" \
    --baseline runs/regression__v1-baseline-1.json --save-predictions 2>&1 | tee "runs/logs/regress-$label.log"
  stop_group "$group"
  wait_gpu_free
}

echo "== 1. pins"
uv run adapterops verify-pins

echo "== 2. training: seeds [$SEEDS] x tasks [$TASKS]"
if [[ -n "${SKIP_TRAIN:-}" ]]; then
  echo "  skipped: using the checkpoints already in checkpoints/"
else
  for seed in $SEEDS; do
    for task in $TASKS; do
      if [[ -f "checkpoints/$task-seed$seed/adapter_model.safetensors" ]]; then
        echo "  $task seed $seed already trained"
        continue
      fi
      start=$SECONDS
      python -m adapterops.train.cli_train --task "$task" --seed "$seed" --variant "seed$seed" \
        2>&1 | tee "runs/logs/train-$task-seed$seed.log"
      test -f "checkpoints/$task-seed$seed/adapter_model.safetensors" || { echo "no adapter for $task seed $seed" >&2; exit 1; }
      echo "---- $task seed $seed trained in $(( (SECONDS - start) / 60 )) min"
    done
  done
fi
wait_gpu_free

echo "== 3. served adapters ($NAME-served)"
regress_with_pins served

for seed in $SEEDS; do
  echo "== 4. seed $seed adapters ($NAME-seed$seed)"
  python - "$seed" "_adapters/variance-seed$seed-pins.json" "$TASKS" <<'PY'
import json
import sys
from pathlib import Path

seed, out, tasks = sys.argv[1], Path(sys.argv[2]), sys.argv[3].split()
components = json.loads(Path("manifests/system.json").read_text())["components"]
pins = {"base_model": components["base_model"], **components["adapters"]}
for task in tasks:
    pins[task] = {"path": f"checkpoints/{task}-seed{seed}"}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"purpose": f"Variance run: every task retrained at seed {seed}, local checkpoints.",
                           "candidate": f"variance-seed{seed}", "components": pins}, indent=2) + "\n")
print(f"  pins: {out} — {', '.join(tasks)} local")
PY
  regress_with_pins "seed$seed" "_adapters/variance-seed$seed-pins.json"
done

echo "== done — copy back:"
ls -1 runs/regression__"$NAME"-*.json runs/*-seed*__train.json
echo "then score drafting with the judge on the laptop and run training-variance --runs ... --force"
