#!/usr/bin/env bash
# Finish the frontier arm (F8) whenever the daily request quota allows.
#
# The account's ceiling is 10,000 requests/day, and it binds long before any dollar cap —
# $0.21 of tokens has already exhausted it twice. Whether the window is fixed or rolling
# is not documented, so this polls with a single cheap request rather than guessing a
# reset time, and resumes from the per-pair cache each time. Nothing already cached is
# re-billed, and `adapterops frontier` takes a lock, so this cannot collide with a run
# started by hand.
#
#   ./scripts/frontier_until_done.sh                 # poll every 20 min, up to 20 h
#   INTERVAL=600 MAX_HOURS=4 ./scripts/frontier_until_done.sh
set -uo pipefail
cd "$(dirname "$0")/.."

INTERVAL="${INTERVAL:-1200}"
MAX_HOURS="${MAX_HOURS:-20}"
deadline=$(( $(date +%s) + MAX_HOURS * 3600 ))

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

remaining() {
  uv run python -c "
import json, pandas as pd
cached = {json.loads(l)['pair_id'] for l in open('data/router/frontier_cache.jsonl') if l.strip()}
pool = pd.read_parquet('data/router/pool.parquet')
print(int((~pool.pair_id.isin(cached)).sum()))"
}

# 0 = quota available, 1 = daily quota exhausted, 2 = some other failure.
quota_ok() {
  uv run python -c "
import sys
from dotenv import load_dotenv; load_dotenv('.env')
from openai import OpenAI
try:
    OpenAI(timeout=20.0, max_retries=0).chat.completions.create(
        model='gpt-4o-mini', messages=[{'role':'user','content':'OK'}], max_completion_tokens=1)
except Exception as exc:
    sys.exit(1 if 'requests per day' in str(exc) else 2)" 2>/dev/null
}

echo "[$(stamp)] starting · $(remaining) pairs left · polling every ${INTERVAL}s"

while :; do
  left=$(remaining)
  if [ "$left" -eq 0 ]; then
    echo "[$(stamp)] DONE — all 5,800 pairs cached"
    exit 0
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[$(stamp)] gave up after ${MAX_HOURS}h with $left pairs left"
    exit 1
  fi

  quota_ok; probe=$?
  if [ $probe -eq 0 ]; then
    echo "[$(stamp)] quota available · $left left · running the router slice first"
    uv run adapterops frontier --purpose router --workers 4 2>&1 | sed 's/^/    /'
    uv run adapterops frontier --workers 4 2>&1 | sed 's/^/    /'
  elif [ $probe -eq 1 ]; then
    echo "[$(stamp)] daily quota still exhausted · $left left · sleeping ${INTERVAL}s"
  else
    echo "[$(stamp)] probe failed for a non-quota reason · $left left · sleeping ${INTERVAL}s"
  fi
  sleep "$INTERVAL"
done
