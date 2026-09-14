#!/usr/bin/env bash
# One GPU session: the request path under load, and one regression run (PRD §7, §8, F17).
#
# Run on the rented A10 from the repository root, after `uv sync --extra gpu`:
#     bash scripts/gpu_request_path_session.sh
#
# What it does, in order, stopping at the first failure:
#   1. verify-pins — refuse to spend GPU time on adapters that moved on the Hub.
#   2. vLLM with the four adapters manifests/system.json pins (scripts/phase2_serve.sh).
#   3. One regression run against it. Nothing else is sending load, so its wall-clock is the
#      §7 "< 25 min" figure and its fallback rate is F17's, both written into the run file.
#   4. serve-api in front of vLLM: confidence routing at the committed 20% operating point,
#      GPT-4o-mini on escalation and fallback, JSONL traces without ticket text.
#   5. request-path-run: the curve's 392 recorded pairs, one full pass at each concurrency.
#
# Spend: GPT-4o-mini sees about a fifth of each pass (~80 calls); the service stops itself at
# $1.00. The GPU hour is the cost. Set NO_FRONTIER=1 to run without an OpenAI key.
#
# Bring back (scp to the laptop, then `adapterops judge-score --run runs/regression__$NAME.json`
# there, where the judge checkpoint lives):
#     runs/regression__$NAME.json  runs/regression__${NAME}__predictions.parquet
#     runs/request_path__$NAME.json  runs/request_path__${NAME}__pairs.parquet
#     runs/traces/requests.jsonl  runs/logs/
set -euo pipefail

cd "$(dirname "$0")/.."
# `uv run` would otherwise re-sync to uv.lock, which pins vllm 0.22.1 and drops the gpu extra.
export UV_NO_SYNC=1
# The recorded curve, M1 and the regression baselines were served on vllm 0.29.0 (runs/gate05.json).
# A different version can move log-probabilities, and with them every routing decision compared.
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
NAME="${NAME:-a10-v4}"
BASELINE="${BASELINE:-runs/regression__v1-baseline-1.json}"
CONCURRENCY="${CONCURRENCY:-1 4 16 32}"
VLLM_PORT="${PORT:-8000}"
API_PORT="${API_PORT:-8080}"
mkdir -p runs/logs

if [[ -z "${NO_FRONTIER:-}" ]] && ! grep -qs '^OPENAI_API_KEY=' .env && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "no OPENAI_API_KEY in the environment or .env — set it, or NO_FRONTIER=1" >&2
  exit 1
fi

pids=()
cleanup() { for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT

wait_for() {  # url, what, seconds
  local url=$1 what=$2 deadline=$((SECONDS + $3))
  until curl -sf "$url" >/dev/null; do
    if (( SECONDS > deadline )); then echo "$what did not come up; see runs/logs/" >&2; exit 1; fi
    sleep 5
  done
  echo "  $what is up"
}

installed=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo none)
if [[ "$installed" != "$VLLM_VERSION" ]]; then
  echo "vllm is $installed, the recorded runs used $VLLM_VERSION:" >&2
  echo "  uv pip install vllm==$VLLM_VERSION   (or VLLM_VERSION=$installed to accept the difference)" >&2
  exit 1
fi

echo "== 1. pins"
uv run adapterops verify-pins

echo "== 2. vLLM"
PORT="$VLLM_PORT" bash scripts/phase2_serve.sh > runs/logs/vllm.log 2>&1 &
pids+=($!)
wait_for "http://localhost:$VLLM_PORT/v1/models" "vLLM" 900

echo "== 3. regression run ($NAME, against $BASELINE)"
uv run adapterops regress --base-url "http://localhost:$VLLM_PORT" --name "$NAME" \
  --baseline "$BASELINE" --save-predictions 2>&1 | tee runs/logs/regress.log

echo "== 4. request path"
flags=(--backend vllm --base-url "http://localhost:$VLLM_PORT" --policy confidence
       --trace jsonl --port "$API_PORT")
[[ -n "${NO_FRONTIER:-}" ]] && flags+=(--no-frontier)
# The judge checkpoint is not in git; without it on this box, drafts go unscored.
judge_file=$(uv run python -c "import json; c=json.load(open('manifests/system.json'))['components']['judge']; print(next(v['file'] for k, v in c.items() if k.endswith('model.safetensors')))")
[[ -f "$judge_file" ]] || { echo "  no judge at $judge_file — drafts unscored"; flags+=(--no-judge); }
uv run adapterops serve-api "${flags[@]}" > runs/logs/serve-api.log 2>&1 &
pids+=($!)
wait_for "http://127.0.0.1:$API_PORT/healthz" "serve-api" 300

echo "== 5. recorded pairs through the request path, concurrency $CONCURRENCY"
# shellcheck disable=SC2086
uv run adapterops request-path-run --url "http://127.0.0.1:$API_PORT" --name "$NAME" \
  --concurrency $CONCURRENCY 2>&1 | tee runs/logs/request-path-run.log

echo "== done — copy back:"
ls -1 runs/regression__"$NAME"* runs/request_path__"$NAME"* runs/traces/requests.jsonl
