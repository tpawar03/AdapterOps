"""End-to-end smoke of F14 judge training on synthetic grades, with every path redirected.

Its first version passed a judge that had learned nothing — Spearman -0.45 — because it
checked that a correlation existed rather than that the judge tracked the signal. It now
plants a strong signal and requires Spearman above 0.5.

    uv run python scripts/smoke_judge_training.py
"""
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from adapterops.judge import train as jt
from adapterops.judge.label import plan_items

tmp = Path(tempfile.mkdtemp(prefix="smoke-judge-"))
jt.REPO_ROOT, jt.JUDGE_DIR = tmp, tmp
jt.LABELS_FILE, jt.RUN_FILE = tmp / "judgments.parquet", tmp / "runs" / "judge__train.json"

items = plan_items(False)
train = items[items.split == "train"].head(160)
cal = items[items.split == "calibration"].head(40)
frame = pd.concat([train, cal]).reset_index(drop=True)
# A learnable signal, so a working path shows a positive correlation: longer replies score higher.
rank = frame.reply.str.len().rank(pct=True).to_numpy()
noise = np.random.default_rng(0).normal(0, 0.3, len(frame))
frame["score"] = np.clip(np.rint(1 + 4 * rank + noise), 1, 5).astype(int)
frame["parsed_ok"] = True
frame.to_parquet(jt.LABELS_FILE)

real_before = {p.name for p in (ROOT / "data" / "judge").glob("calibration_scored*")} \
    | {p.name for p in (ROOT / "runs").glob("judge__train*")}
summary = jt.train(jt.JudgeConfig(epochs=4.0, max_length=128, min_fit=10, learning_rate=5e-5,
                                  output_dir=str(tmp / "ck"), tag="smoke"))
real_after = {p.name for p in (ROOT / "data" / "judge").glob("calibration_scored*")} \
    | {p.name for p in (ROOT / "runs").glob("judge__train*")}

out = tmp / "ck"
checks = {
    "rows": summary["rows"],
    "calibration": summary["calibration"],
    "intermediate_checkpoints_left": sorted(p.name for p in out.glob("checkpoint-*")),
    "final_model_present": (out / "model.safetensors").exists(),
    "real_repo_untouched": real_before == real_after,
}
print(json.dumps(checks, indent=1))
# The first version passed with Spearman -0.45: it checked a correlation existed, not
# that the judge learned. On a planted, strong signal it must now actually track it.
ok = ((summary["calibration"]["spearman"] or 0) > 0.5 and not checks["intermediate_checkpoints_left"]
      and checks["final_model_present"] and checks["real_repo_untouched"])
shutil.rmtree(tmp, ignore_errors=True)
print("JUDGE SMOKE", "PASS" if ok else "FAIL")
