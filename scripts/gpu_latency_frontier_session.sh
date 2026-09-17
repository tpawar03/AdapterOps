#!/usr/bin/env bash
# One GPU session: the laptop-only PII numbers on vLLM, where P95 goes, and GPT-4o-mini under sustained
# load with the domain gate on (TODO.md §2). Serves manifest v11; nothing is trained or pushed.
#
# Setup on the rented A10 (repository root): uv sync --extra gpu, then
#     UV_NO_SYNC=1 uv pip install vllm==0.29.0         # pins torch==2.13.0, torchvision==0.28.0
#     UV_NO_SYNC=1 uv pip uninstall flashinfer-cubin   # no release matches flashinfer 0.6.18
# and .env with OPENAI_API_KEY. Keep UV_NO_SYNC=1 set: a bare `uv run` re-syncs and breaks torch.
#     source .venv/bin/activate && bash scripts/gpu_latency_frontier_session.sh
#
# What it does, in order, stopping at the first failure; cheap results first:
#   1. verify-pins; refuse any vLLM but 0.29.0.
#   2. vLLM with the pinned adapters (urgency is TF-IDF, so three LoRAs).
#   3. latency-profile: time to first token vs decode per adapter, streamed, concurrency 1 and 16.
#   4. PII false positives on vLLM: the 928 screened golden texts and the 491 validation sentences
#      (laptop: 4/928 and 4/491 for these weights).
#   5. PII routing re-check on vLLM: every PII pair of both curve populations straight at vLLM, which
#      records each pair's live score and the decision at the operating threshold (laptop:
#      runs/router__rescore__pii.json).
#   6. GPT-4o-mini under sustained load: serve-api with the frontier and the domain gate on,
#      SUSTAIN_SECONDS at concurrency 32. At ~43 pairs/s with a fifth escalated the path wants ~500
#      calls a minute; FRONTIER_PER_MINUTE caps it, and FRONTIER_PER_DAY ends frontier answers partway
#      through, so the run also shows the budget running out. At most FRONTIER_PER_DAY calls, ~$1.
#
# About an hour after vLLM is up. Bring back (scp -i <key> -r "ubuntu@<ip>:AdapterOps/runs/..." runs/):
#     runs/latency__$NAME.json
#     runs/pii__false_positives__$NAME.json  runs/pii__false_positives__$NAME-val.json  (and __outputs)
#     runs/request_path__$NAME-pii-direct*.json  (and __pairs.parquet)
#     runs/request_path__$NAME-frontier-sustained.json  (and __pairs.parquet)
#     runs/logs/
set -euo pipefail

cd "$(dirname "$0")/.."
export UV_NO_SYNC=1
export PYTHONPATH=src
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"
NAME="${NAME:-a10-v11}"
VLLM_PORT="${PORT:-8000}"
API_PORT="${API_PORT:-8080}"
SUSTAIN_SECONDS="${SUSTAIN_SECONDS:-1800}"
SUSTAIN_CONCURRENCY="${SUSTAIN_CONCURRENCY:-32}"
FRONTIER_PER_MINUTE="${FRONTIER_PER_MINUTE:-300}"
FRONTIER_PER_DAY="${FRONTIER_PER_DAY:-7000}"
mkdir -p runs/logs

if ! grep -qs '^OPENAI_API_KEY=' .env && [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "no OPENAI_API_KEY in the environment or .env — step 6 needs it" >&2
  exit 1
fi

installed=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo none)
if [[ "$installed" != "$VLLM_VERSION" ]]; then
  echo "vllm is $installed, the recorded runs used $VLLM_VERSION:" >&2
  echo "  UV_NO_SYNC=1 uv pip install vllm==$VLLM_VERSION" >&2
  exit 1
fi

# Each server runs in its own process group, so stopping it takes vLLM's engine workers with it.
groups=()
cleanup() { for g in "${groups[@]}"; do kill -- -"$g" 2>/dev/null || true; done; }
trap cleanup EXIT

start_group() {  # log, command...
  local log=$1; shift
  setsid "$@" > "$log" 2>&1 &
  groups+=($!)
}

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
start_group runs/logs/vllm-latency-frontier.log env PORT="$VLLM_PORT" bash scripts/phase2_serve.sh
wait_for "http://localhost:$VLLM_PORT/v1/models" "vLLM" 900

echo "== 3. latency profile"
uv run adapterops latency-profile --base-url "http://localhost:$VLLM_PORT" --name "$NAME" \
  --concurrency 1 16 2>&1 | tee runs/logs/latency-profile.log

echo "== 4. PII false positives on vLLM"
uv run adapterops pii-false-positives --base-url "http://localhost:$VLLM_PORT" --name "$NAME" \
  2>&1 | tee runs/logs/pii-false-positives-vllm.log
uv run adapterops pii-false-positives --base-url "http://localhost:$VLLM_PORT" --name "$NAME-val" \
  --set ai4privacy_val 2>&1 | tee runs/logs/pii-false-positives-vllm-val.log

echo "== 5. PII routing re-check on vLLM"
for population in router_in_distribution router_shift; do
  suffix=""; [[ "$population" == router_shift ]] && suffix="-shift"
  uv run adapterops request-path-run --direct-vllm "http://localhost:$VLLM_PORT" \
    --name "$NAME-pii-direct$suffix" --population "$population" --tasks pii --concurrency 16 \
    2>&1 | tee "runs/logs/request-path-pii-direct$suffix.log"
done

echo "== 6. GPT-4o-mini under sustained load: ${SUSTAIN_SECONDS}s at concurrency $SUSTAIN_CONCURRENCY"
start_group runs/logs/serve-api-frontier-sustained.log uv run adapterops serve-api --backend vllm \
  --base-url "http://localhost:$VLLM_PORT" --policy confidence --no-judge --trace none --port "$API_PORT" \
  --frontier-per-minute "$FRONTIER_PER_MINUTE" --frontier-per-day "$FRONTIER_PER_DAY"
wait_for "http://127.0.0.1:$API_PORT/healthz" "serve-api with GPT-4o-mini and the domain gate" 300
curl -s "http://127.0.0.1:$API_PORT/v1/service" | python -m json.tool | grep -E '"(frontier|domain_gate|policy)"' || true
uv run adapterops request-path-run --url "http://127.0.0.1:$API_PORT" --name "$NAME-frontier-sustained" \
  --concurrency "$SUSTAIN_CONCURRENCY" --duration "$SUSTAIN_SECONDS" --window 60 \
  2>&1 | tee runs/logs/request-path-frontier-sustained.log

echo "== done — copy back:"
ls -1 runs/latency__"$NAME".json runs/pii__false_positives__"$NAME"* runs/request_path__"$NAME"-*
