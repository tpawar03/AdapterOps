"""Static Hugging Face Space: recorded outputs, routing decisions, results and the failure demo (M8, D43).

Hugging Face no longer hosts Gradio Spaces on its free CPU tier — creating one returned 402, PRO
required — and only static Spaces stay free. So the public demo is one static page built from
committed runs, and the live Gradio app stays in `demo/app.py` for anyone who clones the repository.

**Recorded, and it says so.** Every output on the page was saved during a run whose score is already
in the repository: the adapters' golden-set predictions from regression run v1-baseline-1 (vLLM on an
A10, greedy, pinned revisions) — except PII's, which come from a10-v5-pii-negatives, the adapter manifest
v6 serves (D50) — GPT-4o-mini's from the golden-set frontier reference, the prompted
base model's drafting replies from its M2 run, and GPT-4o's grades and one-line reasons for all three
drafting sides.

**Routing decisions are §5's step 3, recorded (F22).** No model runs on a static page, so the page
cannot route a pasted ticket. What it shows instead is every routing decision the published operating
curve counted: the 392 held-out (ticket, task) pairs, each with the adapter's answer, GPT-4o-mini's,
the score each policy ranked it by, whether it escalated at the 20% operating point, and whether that
was a misroute (F37). Pasting a ticket needs the live app.

**The page cannot contradict the scores.** Before writing anything, the build recomputes each system's
score from the exact rows it is about to publish and fails if any differs from the recorded run — and
the routing decisions must reproduce the published curve's quality at the operating point. A browser
full of examples that did not add up to the published number would be the demo disagreeing with the
dashboard.

**Text is never interpreted as markup.** Dataset text and model output reach the page through
`textContent`; the embedded JSON has `</` escaped so no string can close its script tag; the dashboard
is rendered from Markdown with raw HTML disabled.

    uv run python demo/build_static.py            # writes demo/_static/
    uv run python demo/build_static.py --push     # creates or updates the static Space
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval.spans import parse_model_output, score_spans, spans_from_values

OUT = HERE / "_static"
TEMPLATE = HERE / "static_template.html"
SPACE_ID = "Tanny03/adapterops-demo"
FAILURE_HEADING = "## Failure demo"
TOLERANCE = 1e-4
"""Recorded scores are rounded to four places; a recomputation within this is the same number."""

SOURCES = {
    "adapter": "runs/regression__v1-baseline-1__predictions.parquet",
    "adapter_run": "runs/regression__v1-baseline-1.json",
    "adapter_pii": "runs/regression__a10-v5-pii-negatives__predictions.parquet",
    "adapter_pii_run": "runs/regression__a10-v5-pii-negatives.json",
    "pii_false_positives": "runs/pii__false_positives__negatives.json",
    "pii_false_positives_previous": "runs/pii__false_positives.json",
    "frontier": "runs/frontier__golden__predictions.parquet",
    "frontier_run": "runs/frontier__golden.json",
    "prompted_drafting": "runs/drafting__prompted-fewshot__predictions.parquet",
    "grades": "data/judge/m2_golden.parquet",
    "grades_frontier": "data/judge/m2_golden__frontier.parquet",
    "m2": "runs/drafting__m2_gpt4o.json",
    "m2_frontier": "runs/drafting__m2_gpt4o__frontier.json",
    "prompted_intent": "runs/intent__prompted-per-class.json",
    "prompted_urgency": "runs/urgency__prompted-fewshot.json",
    "prompted_pii": "runs/pii__prompted-fewshot.json",
    "urgency_adapter": "runs/urgency__adapter.json",
    "curve": "runs/router__operating_curve__judged.json",
    "frontier_pool": "data/router/frontier.parquet",
    "dashboard": "runs/DASHBOARD.md",
}

CARD = """---
title: AdapterOps
emoji: 🧩
colorFrom: indigo
colorTo: gray
sdk: static
pinned: false
license: other
short_description: Recorded outputs, results and failure demo for AdapterOps
---

Recorded demo for [AdapterOps](https://github.com/tpawar03/AdapterOps): saved outputs from the four
adapters and GPT-4o-mini on the frozen golden sets, the recorded routing decisions behind the operating
curve, the six-metric results dashboard and the detect → block → rollback failure demo. Not live
inference — the live Gradio app is `demo/app.py` in the repository.

Data: Banking77 (MIT); `Tobi-Bueck/customer-support-tickets` (CC BY-NC 4.0 — non-commercial, attribution
required); `ai4privacy/pii-masking-openpii-1m` (CC BY 4.0, synthetic); Bitext customer-support dataset
(CDLA-Sharing-1.0).
"""


def first_line(text: str) -> str:
    """The classification rule every system is scored by (`eval.regression.score`)."""
    return str(text).strip().split("\n")[0].strip()


def check(name: str, got: float, recorded: float) -> None:
    if abs(got - recorded) > TOLERANCE:
        msg = (f"{name}: the published rows give {got:.4f}, the recorded run says {recorded:.4f} — "
               f"the page would contradict the score")
        raise ValueError(msg)


def _read(key: str):
    path = ROOT / SOURCES[key]
    return pd.read_parquet(path) if path.suffix == ".parquet" else json.loads(path.read_text())


def _golden(frame: pd.DataFrame, task: str) -> pd.DataFrame:
    return frame[(frame.task == task) & (frame.split == "random")].reset_index(drop=True)


def _aligned(task: str, adapter: pd.DataFrame, frontier: pd.DataFrame) -> None:
    if len(adapter) != len(frontier) or (adapter.text.to_numpy() != frontier.text.to_numpy()).any():
        msg = f"{task}: adapter and GPT-4o-mini rows are not the same golden items in the same order"
        raise ValueError(msg)


def classification(task: str, adapter: pd.DataFrame, frontier: pd.DataFrame) -> list[dict]:
    _aligned(task, adapter, frontier)
    items = []
    for i, (a, f) in enumerate(zip(adapter.itertuples(), frontier.itertuples(), strict=True)):
        gold = str(a.gold).strip()
        items.append({"id": i, "text": a.text, "gold": gold,
                      "adapter": first_line(a.prediction), "frontier": first_line(f.prediction),
                      "adapter_ok": first_line(a.prediction) == gold,
                      "frontier_ok": first_line(f.prediction) == gold})
    return items


def pii(adapter: pd.DataFrame, frontier: pd.DataFrame) -> tuple[list[dict], float, float]:
    _aligned("pii", adapter, frontier)
    items, gold_all, a_all, f_all = [], [], [], []
    for i, (a, f) in enumerate(zip(adapter.itertuples(), frontier.itertuples(), strict=True)):
        gold_pairs = parse_model_output(a.gold)
        a_pairs, f_pairs = parse_model_output(a.prediction), parse_model_output(f.prediction)
        g_sp = spans_from_values(a.text, gold_pairs)
        a_sp, f_sp = spans_from_values(a.text, a_pairs), spans_from_values(a.text, f_pairs)
        gold_all.append(g_sp)
        a_all.append(a_sp)
        f_all.append(f_sp)
        items.append({
            "id": i, "text": a.text,
            "gold": [list(p) for p in gold_pairs],
            "adapter": [list(p) for p in a_pairs], "frontier": [list(p) for p in f_pairs],
            "adapter_f1": round(float(score_spans([g_sp], [a_sp], strict=True)["f1"]), 3),
            "frontier_f1": round(float(score_spans([g_sp], [f_sp], strict=True)["f1"]), 3),
        })
    pooled_a = float(score_spans(gold_all, a_all, strict=True)["f1"])
    pooled_f = float(score_spans(gold_all, f_all, strict=True)["f1"])
    return items, pooled_a, pooled_f


def _side(grades: pd.DataFrame, source: str) -> pd.DataFrame:
    side = grades[grades.source == source].copy()
    side["pair"] = side.pair.astype(int)
    return side.sort_values("pair").reset_index(drop=True)


def _reason(raw: str) -> str:
    try:
        return str(json.loads(raw).get("reason", ""))
    except (TypeError, ValueError):
        return ""


def drafting(adapter: pd.DataFrame, grades: pd.DataFrame) -> tuple[list[dict], dict[str, float]]:
    sides = {s: _side(grades, s) for s in ("adapter", "prompted", "frontier")}
    for name, side in sides.items():
        if len(side) != len(adapter) or (side.instruction.to_numpy() != adapter.text.to_numpy()).any():
            msg = f"drafting: the {name} grades are not for the golden requests in golden order"
            raise ValueError(msg)
    items = []
    for i, a in enumerate(adapter.itertuples()):
        items.append({
            "id": i, "text": a.text, "reference": str(a.gold),
            "sides": {name: {"reply": str(side.reply[i]).strip(), "grade": int(side.score[i]),
                             "reason": _reason(side.raw[i])} for name, side in sides.items()},
        })
    means = {name: float(side.score.astype(float).mean()) for name, side in sides.items()}
    return items, means


def routing() -> dict:
    """Every routing decision the published operating curve counted, for both policies (F22, F37).

    The decisions come from the curve's own ranking code, and each policy's quality at the operating
    point must match the published curve before anything is written (`misroute.check`).
    """
    from adapterops.router.baselines import escalation_scores
    from adapterops.router.misroute import BUDGET, frames, summarise, tag
    from adapterops.router.misroute import check as check_decisions

    population = "router_in_distribution"
    block = _read("curve")["populations"][population]["all_tasks"]
    frame = frames()[population]
    frontier = _read("frontier_pool").set_index("pair_id").prediction

    tagged, policies = {}, {}
    for policy, key in (("confidence", "confidence"), ("router_p_fail", "router")):
        t = tag(frame, escalation_scores(frame, policy))
        s = summarise(t)
        row = next(r for r in block["curve"] if r["policy"] == policy and r["budget"] == BUDGET)
        check_decisions(s, row["quality"], population, policy)
        tagged[key] = t.set_index("pair_id")
        policies[key] = {k: s[k] for k in ("escalated", "rescued", "harmful_escalation",
                                           "wasted_escalation", "missed_rescue", "quality")}
        policies[key]["threshold"] = row["threshold"]

    items = []
    for i, pair_id in enumerate(tagged["confidence"].index):
        c, r = tagged["confidence"].loc[pair_id], tagged["router"].loc[pair_id]
        items.append({
            "id": i, "task": c.task, "text": str(c.text),
            "adapter": str(c.prediction).strip(), "frontier": str(frontier[pair_id]).strip(),
            "adapter_ok": bool(c.success), "frontier_ok": bool(c.frontier_success),
            "confidence": {"score": round(float(c.escalation_score), 4),
                           "escalated": bool(c.escalated), "misroute": c.failure_type},
            "router": {"score": round(float(r.escalation_score), 4),
                       "escalated": bool(r.escalated), "misroute": r.failure_type},
        })
    return {
        "population": population, "pairs": len(items), "budget": BUDGET,
        "local_quality": block["local_quality"], "frontier_quality": block["frontier_quality"],
        "policies": policies,
        "caption": (f"{len(items)} held-out (ticket, task) pairs from the router's in-distribution "
                    f"eval split — every decision the published operating curve counted. Each policy "
                    f"escalates the {BUDGET:.0%} of pairs it ranks highest. Drafting success is a "
                    f"GPT-4o grade of 4 or more on both sides; the other tasks are exact match."),
        "items": items,
    }


def build_payload() -> dict:
    adapter_all, frontier_all = _read("adapter"), _read("frontier")
    adapter_run, frontier_run = _read("adapter_run")["per_split"], _read("frontier_run")["per_task"]
    tasks: dict = {}

    for task, title, metric in (("intent", "Intent", "micro_accuracy"),
                                ("urgency", "Urgency", "macro_f1")):
        a, f = _golden(adapter_all, task), _golden(frontier_all, task)
        items = classification(task, a, f)
        check(f"{task} adapter accuracy", sum(i["adapter_ok"] for i in items) / len(items),
              adapter_run[task]["random"]["micro_accuracy"])
        check(f"{task} GPT-4o-mini accuracy", sum(i["frontier_ok"] for i in items) / len(items),
              frontier_run[task]["micro_accuracy"])
        prompted = _read(f"prompted_{task}")["metrics"][metric]
        summary = [
            {"label": f"Adapter · {metric}", "value": f"{adapter_run[task]['random'][metric]:.4f}"},
            {"label": f"GPT-4o-mini · {metric}", "value": f"{frontier_run[task][metric]:.4f}"},
            {"label": f"Prompted base · {metric}", "value": f"{prompted:.4f}",
             "note": "score only — outputs not saved"},
        ]
        caption = (f"{len(items)} golden items. A row is correct when the first line of the output is "
                   "exactly the gold label — the rule every system is scored by.")
        if task == "urgency":
            tfidf = _read("urgency_adapter")["context"]["tfidf_logreg_baseline"]["macro_f1"]
            summary.append({"label": "TF-IDF + logistic regression", "value": f"{tfidf:.4f}",
                            "note": "beats every model here"})
            caption += (" Negative result: no model here beats a TF-IDF baseline, and the adapter's "
                        "margin over prompting is within its own run-to-run noise.")
        tasks[task] = {"title": title, "summary": summary, "caption": caption, "items": items}

    # PII shows the adapter manifest v6 serves (D50), not v1-baseline-1's: publishing the replaced
    # adapter's outputs under "the adapter" would show a model nobody serves.
    a, f = _golden(_read("adapter_pii"), "pii"), _golden(frontier_all, "pii")
    items, pooled_a, pooled_f = pii(a, f)
    check("pii adapter strict span F1", pooled_a,
          _read("adapter_pii_run")["per_split"]["pii"]["random"]["span_f1_strict"])
    check("pii GPT-4o-mini strict span F1", pooled_f, frontier_run["pii"]["span_f1_strict"])
    flagged = _read("pii_false_positives")["adapter"]["all"]
    flagged_before = _read("pii_false_positives_previous")["adapter"]["all"]
    tasks["pii"] = {
        "title": "PII",
        "summary": [
            {"label": "Adapter · strict span F1", "value": f"{pooled_a:.4f}"},
            {"label": "GPT-4o-mini · strict span F1", "value": f"{pooled_f:.4f}"},
            {"label": "Prompted base · strict span F1",
             "value": f"{_read('prompted_pii')['metrics']['span_f1_strict']:.4f}",
             "note": "score only — outputs not saved"},
            {"label": "Adapter · PII-free texts it flags",
             "value": f"{flagged['texts_with_any_line']} of {flagged['texts']}",
             "note": (f"the previous adapter flagged {flagged_before['texts_with_any_line']} of "
                      f"{flagged_before['texts']}")},
        ],
        "caption": ("Outputs from the PII adapter manifest v6 serves — retrained with PII-free sentences so "
                    "it can answer nothing — recorded in regression run a10-v5-pii-negatives; the other "
                    f"tasks' outputs are from v1-baseline-1. {len(items)} golden documents of synthetic "
                    "personal data. Per-item badges are each document's strict span F1; the summary pools "
                    "every span, as the gate does."),
        "items": items,
    }

    grades = pd.concat([_read("grades"), _read("grades_frontier")], ignore_index=True)
    items, means = drafting(_golden(adapter_all, "drafting"), grades)
    m2, m2f = _read("m2")["sides"], _read("m2_frontier")["sides"]
    check("drafting adapter GPT-4o mean", means["adapter"], m2["adapter"]["gpt4o_mean"])
    check("drafting prompted GPT-4o mean", means["prompted"], m2["prompted"]["gpt4o_mean"])
    check("drafting GPT-4o-mini GPT-4o mean", means["frontier"], m2f["frontier"]["gpt4o_mean"])
    tasks["drafting"] = {
        "title": "Drafting",
        "summary": [
            {"label": "Adapter · GPT-4o grade", "value": f"{means['adapter']:.2f}"},
            {"label": "Prompted base · GPT-4o grade", "value": f"{means['prompted']:.2f}",
             "note": "77% of replies cut off at the token cap"},
            {"label": "GPT-4o-mini · GPT-4o grade", "value": f"{means['frontier']:.2f}",
             "note": "graded by its own model family"},
        ],
        "caption": (f"{len(items)} golden requests, each answered by all three systems and graded 1–5 "
                    "by GPT-4o with a one-line reason. GPT-4o-mini leads here; GPT-4o grading its own "
                    "family is a preference that cannot be separated from quality without human labels."),
        "items": items,
    }
    return {"tasks": tasks, "routing": routing(), "sources": SOURCES}


def render_dashboard() -> tuple[str, str]:
    from markdown_it import MarkdownIt

    md = MarkdownIt("commonmark", {"html": False}).enable("table")
    text = (ROOT / SOURCES["dashboard"]).read_text()
    results, _, failure = text.partition(FAILURE_HEADING)

    def html(src: str) -> str:
        return (md.render(src).replace("<table>", '<div class="scroll"><table>')
                .replace("</table>", "</table></div>"))

    return html(results), html(FAILURE_HEADING + failure)


def embed(payload: dict) -> str:
    """JSON safe inside a script element: `</` is escaped so no string can end the tag early."""
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def page(payload: dict) -> str:
    results, failure = render_dashboard()
    template = TEMPLATE.read_text()
    for token in ("%%DATA%%", "%%RESULTS%%", "%%FAILURE%%"):
        if template.count(token) != 1:
            msg = f"template must contain {token} exactly once"
            raise ValueError(msg)
    return (template.replace("%%RESULTS%%", results).replace("%%FAILURE%%", failure)
            .replace("%%DATA%%", embed(payload)))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python demo/build_static.py")
    ap.add_argument("--push", action="store_true", help=f"create or update {SPACE_ID}")
    args = ap.parse_args(argv)

    html = page(build_payload())
    OUT.mkdir(exist_ok=True)
    (OUT / "index.html").write_text(html)
    (OUT / "README.md").write_text(CARD)
    print(f"wrote {OUT.relative_to(ROOT)}/index.html ({len(html.encode()) / 1e6:.2f} MB), "
          "every published score reproduced")

    if args.push:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(SPACE_ID, repo_type="space", space_sdk="static", private=False,
                        exist_ok=True)
        info = api.upload_folder(repo_id=SPACE_ID, repo_type="space", folder_path=str(OUT),
                                 commit_message="Recorded demo (demo/build_static.py)")
        print(f"pushed {SPACE_ID} -> {info.oid[:8]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
