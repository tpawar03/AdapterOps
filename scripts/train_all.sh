#!/usr/bin/env bash
# Train every ready adapter in one rented-GPU session, pushing each to the Hub as it
# finishes. Run on the box, inside a venv (see GATE-0.5.md for why --user breaks).
#
#   export HF_TOKEN=hf_...        # Write scope
#   export HF_USER=Tanny03
#   ./scripts/train_all.sh
#
# Override with e.g. TASKS="pii drafting" to retrain others.
set -euo pipefail

TASKS="${TASKS:-urgency}"
: "${HF_USER:?set HF_USER}"
: "${HF_TOKEN:?set HF_TOKEN (Write scope)}"

cd "$(dirname "$0")/.."
export PYTHONPATH=src
export HF_TOKEN

# Check the GPU *and* the token before spending any time. A bad HF_TOKEN used to
# surface 20 minutes in, on the first Hub call inside training.
python -c "
import torch; assert torch.cuda.is_available(); print('gpu:', torch.cuda.get_device_name(0))
from huggingface_hub import HfApi
print('hf user:', HfApi().whoami()['name'])"

for task in $TASKS; do
  echo ""
  echo "================ $task ================"
  start=$SECONDS
  # Pushes from inside train(), so a recycled box cannot lose the weights.
  python -m adapterops.train.cli_train --task "$task" --hub-repo "${HF_USER}/adapterops-${task}"
  echo "---- $task finished in $(( (SECONDS-start)/60 )) min ----"
done

echo ""
echo "All done. Adapters pushed under ${HF_USER}/adapterops-<task>."
echo "Commit runs/*.json back to the repo, then TERMINATE the instance."
