"""Rehearse the whole post-GPU chain on a synthetic scored pool.

`router-dataset` -> `router-train` -> `router-report` all run for the first time on the far
side of a paid GPU session, which is the worst place to discover that any of them is
broken. This synthesises what the GPU run would write — at the *measured* per-task failure
rates, so class balance is realistic rather than a convenient 50/50 — and runs the real
chain against it in a temp directory with every module's paths redirected. The repo is
never touched.

It has already paid for itself three times: an oracle that was not an upper bound (D32), an
MPS OOM in router training, and confirmation that the intent hard-cases cap does not bind.

    uv run python scripts/rehearse_post_gpu_chain.py
"""
import json, shutil, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp(prefix="rehearse-"))
(tmp / "runs").mkdir()
for f in ("pool.parquet", "shift.parquet", "frontier.parquet"):
    shutil.copy(ROOT / "data/router" / f, tmp / f)

# --- synthesise what the GPU run would write -------------------------------------
from adapterops.router.scoring import label, score_pair

pool = pd.read_parquet(tmp / "pool.parquet")
rng = np.random.default_rng(11)
# Per-task failure rates taken from the real golden-set scores, so the chain meets
# realistic class balance rather than a convenient 50/50.
FAIL = {"intent": 0.07, "urgency": 0.53, "pii": 0.45, "drafting": 0.50}
rows = []
for _, r in pool.iterrows():
    fails = rng.random() < FAIL[r.task]
    if r.task in ("intent", "urgency"):
        pred = "WRONG_LABEL" if fails else r.gold
    elif r.task == "pii":
        lines = r.gold.split("\n")
        pred = "\n".join(lines[:-1]) if (fails and len(lines) > 1) else r.gold
    else:
        pred = "unrelated filler text" if fails else r.gold
    rows.append({"pair_id": r.pair_id, "prediction": pred,
                 "mean_logprob": float(rng.normal(-3.0 if fails else -0.4, 0.6)),
                 "min_logprob": -5.0, "n_tokens": 20, "latency_s": 0.08,
                 "finish_reason": "stop", "error": None})
frame = pool.merge(pd.DataFrame(rows), on="pair_id")
comp = pd.DataFrame([score_pair(r.task, r.text, r.gold, r.prediction)
                     for _, r in frame.iterrows()])
scored = label(pd.concat([frame.reset_index(drop=True), comp], axis=1))
scored.to_parquet(tmp / "scored.parquet")
print(f"synthetic scored pool: {len(scored):,} pairs")
print(scored.groupby(['purpose','task']).success.mean().round(3).to_string())

# --- redirect and run the chain ---------------------------------------------------
from adapterops.router import dataset, report
from adapterops.router import train as rtrain

for mod in (dataset, rtrain, report):
    mod.DATA_DIR = tmp
    mod.REPO_ROOT = tmp
dataset.SCORED_FILE = tmp / "scored.parquet"
dataset.SHIFT_FILE = tmp / "shift.parquet"
dataset.OUT_DIR = tmp
dataset.MANIFEST = tmp / "ROUTER_DATASET.json"
rtrain.RUN_FILE = tmp / "runs/router__train.json"
report.OUT_JSON = tmp / "runs/router__operating_curve.json"
report.OUT_CSV = tmp / "runs/router__operating_curve.csv"

print("\n=== router-dataset ===")
assert dataset.main() == 0
print(json.dumps(json.loads((tmp / "ROUTER_DATASET.json").read_text())["hard_candidates"],
                 indent=1, default=str)[:700])

print("\n=== router-train ===")
cfg = rtrain.RouterConfig(epochs=2.0, output_dir=str(tmp / "ck"))
summary = rtrain.train(cfg)
print(json.dumps(summary["splits"]["eval"], indent=1))

print("\n=== router-report ===")
assert report.main() == 0
print(f"\nREHEARSAL OK · artifacts in {tmp}")
