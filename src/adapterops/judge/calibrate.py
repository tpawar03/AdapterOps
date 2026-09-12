"""Judge calibration (F15, M5), and the two comparisons the judge exists to make possible.

**Calibration** is the M5 number: how closely the distilled judge tracks its GPT-4o teacher
on 150 held-out items. Spearman and Pearson are both reported because they fail
differently, and neither sees a consistent offset — a student that ranks perfectly but runs
a full point generous scores 1.0 on both. So mean bias is reported beside them, along with
agreement within ±1, which on a 1–5 scale is the reading a person would actually use. PRD
§4 sets no pass threshold; N3 (correlation ≥ 0.80) is a nice-to-hit.

A continuous student score is rounded to the nearest integer on the scale for the agreement
figures (half-to-even, numpy's rule) and left continuous for the correlations.

**Proxy versus judge** is the commitment D28 made before any of this existed: when the real
drafting metric arrived, measure how much the token-F1 proxy agreed with it. Reported as a
rank correlation and as agreement on the binary success label, with Cohen's kappa beside
the raw rate, because on a roughly 50/50 label raw agreement is inflated by chance — two
unrelated coin flips agree half the time.

**The same-family check** is the §10 caveat made as measurable as it can be. GPT-4o grades
both the adapter's replies and GPT-4o-mini's, on the same requests. A gap favouring the
frontier could be real quality or family preference, and without human labels the two
cannot be separated — so this reports the paired gap and says exactly that.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import cohen_kappa_score

SCALE = (1, 5)


def _correlations(a: np.ndarray, b: np.ndarray) -> tuple[float | None, float | None]:
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None, None
    return float(stats.spearmanr(a, b).statistic), float(stats.pearsonr(a, b).statistic)


def agreement(teacher: Sequence[float], student: Sequence[float], within: int = 1) -> dict:
    """Student judge against teacher labels on the calibration set (F15)."""
    t = np.asarray(teacher, dtype=float)
    s = np.asarray(student, dtype=float)
    if t.shape != s.shape:
        msg = f"{len(t)} teacher scores but {len(s)} student scores"
        raise ValueError(msg)
    if len(t) < 3:
        msg = "calibration needs at least 3 paired scores"
        raise ValueError(msg)

    rounded = np.clip(np.rint(s), *SCALE)
    spearman, pearson = _correlations(t, s)
    out = {
        "n": len(t),
        "spearman": None if spearman is None else round(spearman, 4),
        "pearson": None if pearson is None else round(pearson, 4),
        "exact_agreement": round(float(np.mean(rounded == t)), 4),
        f"within_{within}_agreement": round(float(np.mean(np.abs(rounded - t) <= within)), 4),
        "mean_absolute_error": round(float(np.mean(np.abs(s - t))), 4),
        "mean_bias_student_minus_teacher": round(float(np.mean(s - t)), 4),
    }
    if spearman is None:
        out["note"] = "a constant score on one side — correlation is undefined, not zero"
    return out


def proxy_vs_judge(proxy: Sequence[float], judge_scores: Sequence[float], *,
                   proxy_cut: float, judge_success_min: int = 4) -> dict:
    """How far the D28 token-F1 proxy agrees with the real drafting metric.

    The proxy's success rule is reproduced exactly (non-zero and at or above the cut).
    `judge_success_min` — the judge score that counts as a successful draft — is a stated
    choice, not a derived one, and is recorded in the output so it can be argued with.
    """
    p = np.asarray(proxy, dtype=float)
    j = np.asarray(judge_scores, dtype=float)
    if p.shape != j.shape:
        msg = f"{len(p)} proxy scores but {len(j)} judge scores"
        raise ValueError(msg)

    proxy_success = (p > 0) & (p >= proxy_cut)
    judge_success = j >= judge_success_min
    spearman, _ = _correlations(p, j)
    both_vary = proxy_success.any() and (~proxy_success).any() \
        and judge_success.any() and (~judge_success).any()
    return {
        "n": len(p),
        "spearman": None if spearman is None else round(spearman, 4),
        "binary_agreement": round(float(np.mean(proxy_success == judge_success)), 4),
        "cohen_kappa": (round(float(cohen_kappa_score(proxy_success, judge_success)), 4)
                        if both_vary else None),
        "proxy_success_rate": round(float(proxy_success.mean()), 4),
        "judge_success_rate": round(float(judge_success.mean()), 4),
        "proxy_cut": float(proxy_cut),
        "judge_success_min": judge_success_min,
        "note": "judge_success_min is a stated choice; kappa is the chance-corrected reading",
    }


def same_family_check(local_scores: pd.Series, frontier_scores: pd.Series) -> dict:
    """Paired by pair_id: one request, answered by the adapter and by GPT-4o-mini."""
    joined = pd.concat({"local": local_scores, "frontier": frontier_scores}, axis=1).dropna()
    caveat = ("GPT-4o grades both arms. A gap favouring the frontier may be real quality or "
              "family preference, and the two cannot be separated without human labels.")
    if len(joined) < 10:
        return {"n": len(joined), "note": "too few paired items to compare",
                "caveat": caveat}

    diff = joined.frontier - joined.local
    nonzero = diff[diff != 0]
    return {
        "n": len(joined),
        "mean_local": round(float(joined.local.mean()), 4),
        "mean_frontier": round(float(joined.frontier.mean()), 4),
        "mean_difference_frontier_minus_local": round(float(diff.mean()), 4),
        "frontier_higher": round(float((diff > 0).mean()), 4),
        "local_higher": round(float((diff < 0).mean()), 4),
        "tied": round(float((diff == 0).mean()), 4),
        "wilcoxon_p": (round(float(stats.wilcoxon(nonzero).pvalue), 6)
                       if len(nonzero) >= 10 else None),
        "caveat": caveat,
    }
