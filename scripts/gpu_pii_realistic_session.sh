#!/usr/bin/env bash
# One GPU session: retrain PII with Nemotron-PII's realistic formats added, and regression-score it beside the
# served adapter (TODO.md §2; the split and decision rule are frozen in data/pii/split_train_realistic.json
# before this runs).
#
# Commit and push data/pii/split_train_realistic.parquet and its .json first, then on the rented A10 from the
# repository root, set up as for gpu_pii_negatives_session.sh (uv sync --extra gpu, uv pip install vllm==0.29.0):
#     source .venv/bin/activate && bash scripts/gpu_pii_realistic_session.sh
#
# What it does, in order, stopping at the first failure:
#   1. Refuse to start unless the frozen split matches its recorded sha256, vLLM is 0.29.0 and the
#      training libraries are installed.
#   2. Train PII on data/pii/split_train_realistic.parquet — the served adapter's 13,910 rows plus 4,000
#      Nemotron-PII documents — same seed, 3 epochs, no row cap. Nothing is pushed to the Hub. ~2–2.5 h.
#   3. vLLM with the served adapters, and a regression run ($NAME-served).
#   4. vLLM with the retrained PII adapter swapped in from a local pin set, and a regression run
#      ($NAME-realistic). Both runs in one session: the same adapters moved by up to 0.0081 between sessions.
#
# No OpenAI key is needed. Afterwards, copy the checkpoint back and, on the laptop:
#     uv run adapterops pii-realistic --adapter-dir checkpoints/pii-realistic --name realistic
#     uv run adapterops pii-realistic-compare --candidate realistic
#     uv run adapterops pii-false-positives --adapter-dir checkpoints/pii-realistic --name realistic
#     uv run adapterops pii-false-positives --set ai4privacy_val --adapter-dir checkpoints/pii-realistic \
#         --name realistic-val
#
# Bring back (scp -i <key> -r ...):
#     runs/regression__$NAME-served.json    runs/regression__$NAME-realistic.json  (and predictions)
#     runs/pii-realistic__train.json        checkpoints/pii-realistic/  (adapter files only)
#     runs/logs/
set -euo pipefail

cd "$(dirname "$0")/.."
export UV_NO_SYNC=1
export PYTHONPATH=src
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
NAME="${NAME:-a10-v7-pii}"
VLLM_PORT="${PORT:-8000}"
PINS="_adapters/pii-realistic-pins.json"
mkdir -p runs/logs

installed=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo none)
if [[ "$installed" != "$VLLM_VERSION" ]]; then
  echo "vllm is $installed, the recorded runs used $VLLM_VERSION:" >&2
  echo "  uv pip install vllm==$VLLM_VERSION   (or VLLM_VERSION=$installed to accept the difference)" >&2
  exit 1
fi
python -c "import bitsandbytes, peft" || { echo "training needs the gpu extra: uv sync --extra gpu" >&2; exit 1; }
python - <<'PY'
import hashlib
import json
from pathlib import Path

record = json.loads(Path("data/pii/split_train_realistic.json").read_text())
actual = hashlib.sha256(Path(record["file"]).read_bytes()).hexdigest()
if actual != record["sha256"]:
    raise SystemExit(f"{record['file']} is not the frozen split ({actual[:12]} vs {record['sha256'][:12]})")
print(f"  split verified: {record['rows']}")
PY

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

echo "== 2. training PII with realistic formats"
start=$SECONDS
python -m adapterops.train.cli_train --task pii --train-split train_realistic --variant realistic \
  --subsample 100000 --epochs 3 2>&1 | tee runs/logs/train-pii-realistic.log
test -f checkpoints/pii-realistic/adapter_model.safetensors || { echo "no adapter written" >&2; exit 1; }
echo "---- trained in $(( (SECONDS - start) / 60 )) min"
wait_gpu_free

echo "== 3. served adapters: vLLM and a regression run ($NAME-served)"
start_group runs/logs/vllm-pii-served.log env PORT="$VLLM_PORT" bash scripts/phase2_serve.sh
VLLM_GROUP=$LAST_GROUP
wait_for "http://localhost:$VLLM_PORT/v1/models" "vLLM" 900
uv run adapterops regress --base-url "http://localhost:$VLLM_PORT" --name "$NAME-served" \
  --baseline runs/regression__v1-baseline-1.json --save-predictions 2>&1 | tee runs/logs/regress-pii-served.log
stop_group "$VLLM_GROUP"
wait_gpu_free

echo "== 4. retrained PII adapter: vLLM from local pins and a regression run ($NAME-realistic)"
python - "$PINS" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
components = json.loads(Path("manifests/system.json").read_text())["components"]
pins = {"base_model": components["base_model"], **components["adapters"],
        "pii": {"path": "checkpoints/pii-realistic"}}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"purpose": "PII retrained with realistic formats: local checkpoint, never published.",
                           "candidate": "pii-realistic", "replaces": "pii", "components": pins},
                          indent=2) + "\n")
print(f"  pins: {out} — pii=local")
PY
start_group runs/logs/vllm-pii-realistic.log env PORT="$VLLM_PORT" PINS="$PINS" bash scripts/phase2_serve.sh
wait_for "http://localhost:$VLLM_PORT/v1/models" "vLLM with the retrained PII adapter" 900
uv run adapterops regress --base-url "http://localhost:$VLLM_PORT" --name "$NAME-realistic" \
  --baseline "runs/regression__$NAME-served.json" --save-predictions 2>&1 | tee runs/logs/regress-pii-realistic.log

echo "== done — copy back:"
ls -1 runs/regression__"$NAME"-served* runs/regression__"$NAME"-realistic* runs/pii-realistic__train.json
ls -1 checkpoints/pii-realistic/adapter_model.safetensors checkpoints/pii-realistic/adapter_config.json
