#!/usr/bin/env bash
# One GPU session: the request path past a burst (PRD §7, §8, §11).
#
# Run on the rented A10 from the repository root, set up as for gpu_request_path_session.sh
# (uv sync --extra gpu, uv pip install vllm==0.29.0, matching flashinfer-cubin, .env with the key):
#     source .venv/bin/activate && bash scripts/gpu_load_session.sh
#
# What it does, in order, stopping at the first failure:
#   1. verify-pins, and refuse any vLLM but the one the recorded runs used.
#   2. vLLM with the four adapters manifests/system.json pins.
#   3. Two request paths in front of it. :8080 has GPT-4o-mini on, for the shifted population, where
#      served quality needs the frontier's answers. :8081 has it off, and tracing off, for load:
#      a pair past the threshold is still counted as escalated, but answered locally — so the
#      escalation rate is measured without spending ~20,000 frontier calls against an account
#      capped at 10,000 requests a day, and latency is the local path's own.
#   4. The shifted population (1,047 pairs) once through :8080 — the population routing mattered
#      most on offline, never run live.
#   5. A saturation sweep: the in-distribution pairs for SWEEP_SECONDS at each concurrency, first
#      straight at vLLM (its ceiling), then through :8081 (the request path's).
#   6. Sustained load: SUSTAIN_SECONDS at SUSTAIN_CONCURRENCY through :8081, summarised per minute.
#
# About 35 minutes after vLLM is up. Spend: ~200 GPT-4o-mini calls in step 4, a few cents.
#
# Bring back (scp -i <key> "ubuntu@<ip>:AdapterOps/runs/*a10-v4-*" runs/):
#     runs/request_path__$NAME-shift.json         and its __pairs.parquet
#     runs/request_path__$NAME-sweep-vllm.json    and its __pairs.parquet
#     runs/request_path__$NAME-sweep.json         and its __pairs.parquet
#     runs/request_path__$NAME-sustained.json     and its __pairs.parquet
#     runs/logs/
set -euo pipefail

cd "$(dirname "$0")/.."
export UV_NO_SYNC=1
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
NAME="${NAME:-a10-v4}"
VLLM_PORT="${PORT:-8000}"
API_PORT="${API_PORT:-8080}"
LOAD_PORT="${LOAD_PORT:-8081}"
SWEEP="${SWEEP:-16 32 64 128 256}"
SWEEP_SECONDS="${SWEEP_SECONDS:-60}"
SUSTAIN_CONCURRENCY="${SUSTAIN_CONCURRENCY:-32}"
SUSTAIN_SECONDS="${SUSTAIN_SECONDS:-900}"
mkdir -p runs/logs

if ! grep -qs '^OPENAI_API_KEY=' .env && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "no OPENAI_API_KEY in the environment or .env — the shifted-population step needs it" >&2
  exit 1
fi

installed=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo none)
if [[ "$installed" != "$VLLM_VERSION" ]]; then
  echo "vllm is $installed, the recorded runs used $VLLM_VERSION:" >&2
  echo "  uv pip install vllm==$VLLM_VERSION   (or VLLM_VERSION=$installed to accept the difference)" >&2
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

echo "== 1. pins"
uv run adapterops verify-pins

echo "== 2. vLLM"
PORT="$VLLM_PORT" bash scripts/phase2_serve.sh > runs/logs/vllm-load.log 2>&1 &
pids+=($!)
wait_for "http://localhost:$VLLM_PORT/v1/models" "vLLM" 900

echo "== 3. request paths"
uv run adapterops serve-api --backend vllm --base-url "http://localhost:$VLLM_PORT" \
  --policy confidence --no-judge --trace jsonl --port "$API_PORT" \
  > runs/logs/serve-api-frontier.log 2>&1 &
pids+=($!)
uv run adapterops serve-api --backend vllm --base-url "http://localhost:$VLLM_PORT" \
  --policy confidence --no-judge --no-frontier --trace none --port "$LOAD_PORT" \
  > runs/logs/serve-api-load.log 2>&1 &
pids+=($!)
wait_for "http://127.0.0.1:$API_PORT/healthz" "serve-api with GPT-4o-mini" 300
wait_for "http://127.0.0.1:$LOAD_PORT/healthz" "serve-api for load" 300

echo "== 4. shifted population through the request path"
uv run adapterops request-path-run --url "http://127.0.0.1:$API_PORT" --name "$NAME-shift" \
  --population router_shift --concurrency 16 2>&1 | tee runs/logs/request-path-shift.log

echo "== 5a. saturation sweep, straight at vLLM, ${SWEEP_SECONDS}s per level: $SWEEP"
# shellcheck disable=SC2086
uv run adapterops request-path-run --direct-vllm "http://localhost:$VLLM_PORT" \
  --name "$NAME-sweep-vllm" --concurrency $SWEEP --duration "$SWEEP_SECONDS" --window 20 \
  2>&1 | tee runs/logs/request-path-sweep-vllm.log

echo "== 5b. saturation sweep, through the request path, ${SWEEP_SECONDS}s per level: $SWEEP"
# shellcheck disable=SC2086
uv run adapterops request-path-run --url "http://127.0.0.1:$LOAD_PORT" \
  --name "$NAME-sweep" --concurrency $SWEEP --duration "$SWEEP_SECONDS" --window 20 \
  2>&1 | tee runs/logs/request-path-sweep.log

echo "== 6. sustained load: ${SUSTAIN_SECONDS}s at concurrency $SUSTAIN_CONCURRENCY"
uv run adapterops request-path-run --url "http://127.0.0.1:$LOAD_PORT" \
  --name "$NAME-sustained" --concurrency "$SUSTAIN_CONCURRENCY" --duration "$SUSTAIN_SECONDS" \
  --window 60 2>&1 | tee runs/logs/request-path-sustained.log

echo "== done — copy back:"
ls -1 runs/request_path__"$NAME"-*
