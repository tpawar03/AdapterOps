"""Hard-cases adjudication screen (F30, M10) — which mined failures are label noise.

A split built from "items the adapter got wrong" is enriched for items whose gold label is
wrong, because a competent model disagrees with bad labels. The screen exists to separate
those from genuinely hard items before the split is allowed to gate anything.

**The PRD's literal rule is confounded by how the items were chosen.** §9 says to quarantine
a mined item when the frontier model also disagrees with the gold label. Measured on the
mined candidates, that rule would quarantine 64% of intent's hard cases. The flaw is
selection: an item is a hard case *because* the adapter failed on it, so the set is
disproportionately difficult, and a second model missing it too is what difficulty
predicts. It is not evidence the label is wrong.

**The rule used instead (D36): independent agreement against gold.** An item is quarantined
when an independent model gives *the same alternative answer* as the adapter. Two models
converging on one specific answer the gold rejects is the pattern label noise produces and
difficulty does not — on a 77-class problem, two independent errors coincide by chance
roughly once in 76. The output reports that uniform chance rate beside the observed rate. It
is a floor, not the true chance rate: confusable neighbouring classes raise it.

**Applicability is stated per task, not assumed:**

- *intent* (77 classes) — applied.
- *urgency* (3 classes) — measured and reported, **not applied**. Once both models are wrong,
  two classes remain, so coinciding is close to a coin flip and says little about any one
  label.
- *pii* — not applicable. The frontier gets almost none of these documents fully right;
  disagreeing with gold here measures ai4privacy's span conventions, not label quality.
- *drafting* — not applicable. The gold is a reference reply; there is no label to dispute.

It costs nothing: the frontier's independent answers on the mining slice were already paid
for by the escalation arm.

    uv run adapterops hard-cases
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from adapterops.train.qlora import COLUMNS

REPO_ROOT = Path(__file__).resolve().parents[3]
CANDIDATES = REPO_ROOT / "data" / "router" / "router_hard_candidates.parquet"
FRONTIER = REPO_ROOT / "data" / "router" / "frontier.parquet"
HARD_DIR = REPO_ROOT / "evals" / "hard"
HARD_FILE = HARD_DIR / "hard_cases.parquet"
QUARANTINE_FILE = HARD_DIR / "quarantine.parquet"
MANIFEST = REPO_ROOT / "evals" / "HARD_CASES.json"

CLASSIFICATION = ("intent", "urgency")
APPLICABILITY = {
    "intent": "applied",
    "urgency": "reported_not_applied",
    "pii": "not_applicable",
    "drafting": "not_applicable",
}
REASONS = {
    "intent": "77 classes: two independent errors coincide by chance about once in 76",
    "urgency": "3 classes: once both models are wrong, coinciding is near a coin flip",
    "pii": "span conventions dominate — frontier disagreement measures convention, not labels",
    "drafting": "gold is a reference reply; there is no label to dispute",
}


def n_classes(task: str) -> int | None:
    if task not in CLASSIFICATION:
        return None
    _, label_col = COLUMNS[task]
    return int(pd.read_parquet(REPO_ROOT / "data" / task / "split_train.parquet")[label_col]
               .nunique())


def uniform_chance_given_both_wrong(k: int | None) -> float | None:
    """P(two wrong answers coincide) if wrong answers were uniform over the other k-1 classes."""
    return None if not k or k < 2 else 1.0 / (k - 1)


def adjudicate(candidates: pd.DataFrame, frontier: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in ("pair_id", "frontier_pred_clean", "frontier_span_f1")
            if c in frontier.columns]
    m = candidates.merge(frontier[keep], on="pair_id", how="left", validate="one_to_one")
    has_answer = m.frontier_pred_clean.notna()
    frontier_answer = m.frontier_pred_clean.fillna("").astype(str).str.strip()
    adapter_answer = m.pred_clean.fillna("").astype(str).str.strip()
    gold = m.gold.astype(str).str.strip()
    classification = m.task.isin(CLASSIFICATION)

    m["has_frontier_answer"] = has_answer
    m["frontier_rejects_gold"] = classification & has_answer & (frontier_answer != gold)
    m["independent_agreement_against_gold"] = (
        classification & has_answer & (frontier_answer == adapter_answer)
        & (adapter_answer != gold))
    m["applicability"] = m.task.map(APPLICABILITY)
    m["quarantine"] = m.independent_agreement_against_gold & (m.applicability == "applied")
    return m


def per_task(m: pd.DataFrame, classes: dict[str, int | None] | None = None) -> dict:
    classes = classes if classes is not None else {t: n_classes(t) for t in m.task.unique()}
    out = {}
    for task, g in m.groupby("task"):
        answered = g[g.has_frontier_answer]
        rejects = int(answered.frontier_rejects_gold.sum())
        agrees = int(answered.independent_agreement_against_gold.sum())
        classification = task in CLASSIFICATION
        row = {
            "candidates": len(g),
            "with_frontier_answer": len(answered),
            "applicability": APPLICABILITY[task],
            "reason": REASONS[task],
            "quarantined": int(g.quarantine.sum()),
            "retained": int((~g.quarantine).sum()),
            "quarantine_rate": round(float(g.quarantine.mean()), 4),
        }
        if classification and len(answered):
            k = classes.get(task)
            row |= {
                "classes": k,
                "prd_literal_rule_rate": round(rejects / len(answered), 4),
                "independent_agreement_rate": round(agrees / len(answered), 4),
                "agreement_given_both_wrong": round(agrees / rejects, 4) if rejects else None,
                "uniform_chance_given_both_wrong":
                    (round(c, 4) if (c := uniform_chance_given_both_wrong(k)) else None),
            }
        elif task == "pii" and "frontier_span_f1" in answered and len(answered):
            row["frontier_documents_fully_right"] = round(
                float((answered.frontier_span_f1 >= 1.0).mean()), 4)
        out[task] = row
    return out


def main(force: bool = False) -> int:
    if MANIFEST.exists() and not force:
        print(f"  {MANIFEST.relative_to(REPO_ROOT)} exists — the hard-cases split is frozen.")
        print("  Re-screening moves the bar the hard split gates on. --force if you mean it.")
        return 1

    m = adjudicate(pd.read_parquet(CANDIDATES), pd.read_parquet(FRONTIER))
    retained, quarantined = m[~m.quarantine], m[m.quarantine]
    if len(retained) + len(quarantined) != len(m) or \
            set(retained.pair_id) & set(quarantined.pair_id):
        msg = "retained and quarantined do not partition the candidates"
        raise RuntimeError(msg)

    HARD_DIR.mkdir(parents=True, exist_ok=True)
    retained.reset_index(drop=True).to_parquet(HARD_FILE)
    quarantined.reset_index(drop=True).to_parquet(QUARANTINE_FILE)
    report = per_task(m)

    def pin(path: Path) -> dict:
        return {"file": str(path.relative_to(REPO_ROOT)), "rows": len(pd.read_parquet(path)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    MANIFEST.write_text(json.dumps({
        "purpose": "Hard-cases split (F31) after the adjudication screen (F30). Mined from "
                   "adapter failures in the mining slice only (D29); never router data.",
        "regenerate": "uv run adapterops hard-cases --force",
        "rule": "quarantine when an independent model gives the same alternative answer as "
                "the adapter against the gold label, on tasks where that carries signal (D36)",
        "rejected_rule": "PRD §9 literal — quarantine when the frontier also rejects gold. "
                         "Confounded by selection: hard cases are chosen for difficulty, so a "
                         "second miss is expected rather than evidence of a bad label.",
        "cost": "none — reuses the escalation arm's frontier answers on the mining slice",
        "hard_cases": pin(HARD_FILE),
        "quarantine": pin(QUARANTINE_FILE),
        "per_task": report,
    }, indent=2) + "\n", encoding="utf-8")

    for task, r in report.items():
        extra = ""
        if "independent_agreement_rate" in r:
            extra = (f" · PRD rule {r['prd_literal_rule_rate']:.1%} · agreement "
                     f"{r['independent_agreement_rate']:.1%} · given both wrong "
                     f"{r['agreement_given_both_wrong']:.1%} vs chance "
                     f"{r['uniform_chance_given_both_wrong']:.1%}")
        elif "frontier_documents_fully_right" in r:
            extra = f" · frontier fully right on {r['frontier_documents_fully_right']:.1%}"
        print(f"  {task:9s} {r['candidates']:>3} → retained {r['retained']:>3} · quarantined "
              f"{r['quarantined']:>2} ({r['applicability']}){extra}")
    print(f"\n  froze {HARD_FILE.relative_to(REPO_ROOT)} ({len(retained)}) and "
          f"{QUARANTINE_FILE.relative_to(REPO_ROOT)} ({len(quarantined)})")
    return 0
