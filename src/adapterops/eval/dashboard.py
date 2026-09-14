"""Six-metric results dashboard (F17), rendered from committed runs — quality rows never blended (F32).

Every number in `runs/DASHBOARD.md` is read from a file already in the repo; none is typed by hand,
so the dashboard cannot drift from the evidence it summarises. The six rows are PRD §11's:

    quality retained (random set) · quality retained (hard cases) · frontier-call rate
    cost per 1K requests · P95 latency · fallback rate

A row the project never measured in the form §11 describes says so in place, rather than being
filled with the nearest available number. The frontier-call rate is the main case: no router runs
in the serving path, so what exists is the offline operating curve, and it is labelled as that.

The failure-demo section (M7, M11) sits below a fixed heading so the Gradio demo can show it on its
own tab.

    uv run adapterops economics      # first — the cost row reads runs/economics.json
    uv run adapterops dashboard
"""

from __future__ import annotations

import json
from pathlib import Path

from adapterops.eval.regression import GATED, TASKS

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_FILE = REPO_ROOT / "runs" / "DASHBOARD.md"
FAILURE_HEADING = "## Failure demo"

INPUTS = {
    "baseline_a": "runs/regression__v1-baseline-1.json",
    "baseline_b": "runs/regression__v1-baseline-2.json",
    "thresholds": "evals/GATE_THRESHOLDS.json",
    "serving": "runs/m1_serving.json",
    "latency": "runs/router__latency.json",
    "economics": "runs/economics.json",
    "curve": "runs/router__operating_curve__judged.json",
    "router_v2": "runs/router__v2.json",
    "rules": "runs/router__rules.json",
    "misroutes": "runs/router__misroutes.json",
    "judge_cost": "runs/judge__cost.json",
    "ceiling": "runs/frontier__golden.json",
    "ceiling_drafting": "runs/drafting__m2_gpt4o__frontier.json",
    "m2": "runs/drafting__m2_gpt4o.json",
    "shuffled": "runs/regression__intent-shuffled.json",
    "m11": "runs/regression__all-m11.json",
    "system": "manifests/system.json",
    "live": "runs/request_path__a10-v4.json",
    "regression_live": "runs/regression__a10-v4.json",
    "shift": "runs/request_path__a10-v4-shift.json",
    "sweep": "runs/request_path__a10-v4-sweep.json",
    "sweep_vllm": "runs/request_path__a10-v4-sweep-vllm.json",
    "sustained": "runs/request_path__a10-v4-sustained.json",
    "pii_negatives": "runs/pii__false_positives.json",
}
HISTORY_GLOB = "manifests/history/system-*.json"
OPERATING_BUDGET = 0.2


def load(root: Path = REPO_ROOT) -> dict:
    data = {k: json.loads((root / p).read_text()) for k, p in INPUTS.items()}
    data["history"] = [json.loads(p.read_text()) for p in sorted(root.glob(HISTORY_GLOB))]
    return data


def fmt(v: float | None, dp: int = 4) -> str:
    return "—" if v is None else f"{v:.{dp}f}"


def gain_captured(quality: float, local: float, oracle: float) -> float | None:
    """Share of the gain available at a budget that a policy realises: 0 is no better than never
    escalating, 1 is the oracle, negative is worse than never escalating."""
    return None if oracle == local else (quality - local) / (oracle - local)


def at_budget(curve: list[dict], policy: str, budget: float) -> dict:
    return next(r for r in curve if r["policy"] == policy and r["budget"] == budget)


def gate_state(row: dict) -> str:
    if row["threshold"] is None:
        return "report-only — floor 0, no threshold"
    if row["provisional"]:
        return "provisional, not enforced"
    return f"**enforced** ({row['bound_by']} variance)"


def quality_rows(data: dict, split: str) -> list[str]:
    derivation = data["thresholds"]["derivation"]["per_task"]
    lines = [("| task | gated metric | baseline run 1 | baseline run 2 | inference spread | "
             "training spread | threshold | gate |"), "|---|---|---|---|---|---|---|---|"]
    for task in TASKS:
        metric = GATED[task]
        a = data["baseline_a"]["per_split"][task][split]
        b = data["baseline_b"]["per_split"][task][split]
        row = derivation[task][split]
        # The derivation gives the hard split thresholds once training spreads exist; the gate never
        # reads them (M11), so the table must not call them enforced.
        state = ("report-only — the hard split never gates (M11)"
                 if split == "hard" and row["threshold"] is not None else gate_state(row))
        lines.append(
            f"| {task} | {metric} (n={a['n']}) | {fmt(a[metric])} | {fmt(b[metric])} | "
            f"{fmt(row['inference_spread'])} | {fmt(row['training_spread'])} | "
            f"{fmt(row['threshold'])} | {state} |")
    return lines


def routing_rows(data: dict) -> list[str]:
    lines = [("| population | never escalate | GPT-4o-mini alone | policy | escalated | quality | "
             "share of available gain |"), "|---|---|---|---|---|---|---|"]
    for name, pop in data["curve"]["populations"].items():
        pooled = pop["all_tasks"]
        oracle = at_budget(pooled["curve"], "oracle", OPERATING_BUDGET)["quality"]
        for policy in ("confidence", "router_p_fail", "random"):
            r = at_budget(pooled["curve"], policy, OPERATING_BUDGET)
            gain = gain_captured(r["quality"], pooled["local_quality"], oracle)
            lines.append(
                f"| {name} | {fmt(pooled['local_quality'], 3)} | "
                f"{fmt(pooled['frontier_quality'], 3)} | {policy} | "
                f"{r['escalation_rate']:.1%} | {fmt(r['quality'], 3)} | "
                f"{'—' if gain is None else f'{gain:.0%}'} |")
    return lines


def router_v2_rows(data: dict) -> list[str]:
    v2 = data["router_v2"]
    pops = v2["populations"]
    lines = [
        ("**Pre-registered retry (D42).** Three fixes for the learned router, frozen before training. "
         "Quality difference from confidence at a 20% budget, with 95% paired bootstrap intervals:"),
        "",
        "| variant | in-distribution | shift | verdict |", "|---|---|---|---|",
    ]
    for name, ind in pops["router_in_distribution"]["vs_confidence_at_0.20"].items():
        shift = pops["router_shift"]["vs_confidence_at_0.20"][name]
        if name.startswith("gain_text_s"):
            verdict = v2["verdicts"]["gain_text"]["per_seed"][name.rsplit("_s", 1)[1]]
        else:
            verdict = v2["verdicts"][name]
        lines.append(
            f"| {name} | {ind['difference']:+.3f} [{ind['ci95'][0]:+.3f}, {ind['ci95'][1]:+.3f}] | "
            f"{shift['difference']:+.3f} [{shift['ci95'][0]:+.3f}, {shift['ci95'][1]:+.3f}] | "
            f"{verdict} |")
    return [*lines, ""]


def frontier_rows(data: dict) -> list[str]:
    ceiling = data["ceiling"]["per_task"]
    lines = [
        ("**Frontier reference, not a gate.** GPT-4o-mini with the escalation arm's prompts, on the "
         "same golden items. Drafting is compared on GPT-4o grades, which also grade GPT-4o-mini."),
        "",
        "| task | this adapter | GPT-4o-mini |", "|---|---|---|",
    ]
    for task in ("intent", "urgency", "pii"):
        metric = GATED[task]
        lines.append(f"| {task} ({metric}) | "
                     f"{data['baseline_a']['per_split'][task]['random'][metric]:.4f} | "
                     f"{ceiling[task][metric]:.4f} |")
    lines.append(f"| drafting (GPT-4o grade) | {data['m2']['sides']['adapter']['gpt4o_mean']:.4f} | "
                 f"{data['ceiling_drafting']['sides']['frontier']['gpt4o_mean']:.4f} |")
    negatives = data["pii_negatives"]
    pii = negatives["adapter"]["all"]
    lines += [
        "",
        (f"**PII's span F1 is conditional on the input containing PII.** On {pii['texts']:,} texts "
         "with nothing any PII label could point at, the adapter reported personal data in "
         f"**{pii['texts_with_any_line']:,} ({pii['false_positive_rate']:.0%})**: invented values in "
         f"{pii['texts_with_invented_value']:,}, and a real word — mostly \"I\", \"Can\", \"My\" as a "
         f"name — in {pii['texts_with_grounded_span']:,}. The baseline "
         f"({negatives['baseline']['name']}) flagged "
         f"{negatives['baseline']['all']['texts_with_any_line']}. "
         "Every training and golden document contained PII (D21), so the adapter never learned an "
         "empty answer. `runs/pii__false_positives.json`"),
    ]
    return [*lines, ""]


def rules_rows(data: dict) -> list[str]:
    pops = data["rules"]["populations"]
    ind, shift = pops["router_in_distribution"], pops["router_shift"]
    lines = [
        ("**Rules-based baseline (F26), reference only.** Written after the D42 allocation table was "
         "read; the task order comes from the train split. At a 20% budget:"),
        "",
        "| rule | gain captured (in-dist / shift) | vs confidence, in-dist | vs confidence, shift |",
        "|---|---|---|---|",
    ]
    for rule in ("rules_length", "rules_task_length"):
        a, b = ind["vs_confidence_at_0.20"][rule], shift["vs_confidence_at_0.20"][rule]
        lines.append(
            f"| {rule} | {ind['headroom_captured']['0.2'][rule]:.0%} / "
            f"{shift['headroom_captured']['0.2'][rule]:.0%} | "
            f"{a['difference']:+.3f} [{a['ci95'][0]:+.3f}, {a['ci95'][1]:+.3f}] | "
            f"{b['difference']:+.3f} [{b['ci95'][0]:+.3f}, {b['ci95'][1]:+.3f}] |")
    return [*lines, ""]


def misroute_rows(data: dict) -> list[str]:
    m = data["misroutes"]
    lines = [
        (f"**Router-misroute diagnostic (F37), reported, never gated.** Each decision at a "
         f"{m['budget']:.0%} budget, against the gain from escalating it. Records: "
         f"`{m['records']['file']}` ({m['records']['rows']:,} rows)."),
        "",
        ("| population | policy | escalated | rescued | harmful escalation | wasted escalation | "
         "missed rescue |"),
        "|---|---|---|---|---|---|---|",
    ]
    for population, by_policy in m["populations"].items():
        for policy, s in by_policy.items():
            lines.append(f"| {population} | {policy} | {s['escalated']:,} | {s['rescued']:,} | "
                         f"{s['harmful_escalation']:,} | {s['wasted_escalation']:,} | "
                         f"{s['missed_rescue']:,} |")
    return [*lines, "",
            ("*Harmful*: escalation broke a pair the adapter got right. *Wasted*: escalation "
             "changed nothing but cost. *Missed rescue*: kept local where GPT-4o-mini would have "
             "succeeded."), ""]


def failure_demo(data: dict) -> list[str]:
    derivation = data["thresholds"]["derivation"]["per_task"]
    intent = data["shuffled"]["comparison"]["per_task"]["intent"]
    enforced = data["system"]["gate"]["thresholds"]["intent"]
    out = [
        FAILURE_HEADING, "",
        "### M7 — detect → block → rollback, on real serving", "",
        ("A shuffled-label intent adapter (F21) was served *as* `intent` and scored against the "
        "last known-good manifest."), "",
        "| step | record |", "|---|---|",
        (f"| detect | intent micro-accuracy {fmt(intent['random']['candidate'])} vs baseline "
        f"{fmt(intent['random']['baseline'])} — drop **{fmt(intent['random']['drop'])}** "
        f"against an enforced threshold of {fmt(enforced)} |"),
    ]
    for m in data["history"]:
        despite = m.get("promoted_despite")
        if despite:
            out.append(f"| block → force | promotion refused for: `{'; '.join(despite)}` — "
                       f"forced as v{m['version']} with the override recorded in the manifest |")
    out += [
        (f"| rollback | current manifest is v{data['system']['version']}, rolled back from "
        f"v{data['system'].get('rolled_back_from', '—')} |"), "",
        "Manifest history:", "",
        "| version | note | gate | promoted despite |", "|---|---|---|---|",
    ]
    for m in data["history"]:
        despite = "; ".join(m.get("promoted_despite") or []) or "—"
        out.append(f"| v{m['version']} | {m['note']} | {m['gate']['state']} | {despite} |")

    out += [
        "", "### M11 — does the hard split catch what the random set misses?", "",
        ("All four under-trained checkpoints (15% of training steps) scored on both splits against "
        "the v1 baseline. A positive drop is a regression."), "",
        "| task | random drop | random threshold | hard drop | hard threshold |",
        "|---|---|---|---|---|",
    ]
    for task in TASKS:
        c = data["m11"]["comparison"]["per_task"][task]
        r, h = derivation[task]["random"], derivation[task]["hard"]
        out.append(f"| {task} | **{fmt(c['random']['drop'])}** | {fmt(r['threshold'])} "
                   f"({gate_state(r)}) | **{fmt(c['hard']['drop'])}** | {fmt(h['threshold'])} "
                   f"({gate_state(h)}) |")
    base_hard = data["baseline_a"]["per_split"]
    out += [
        "",
        ("**Answer: no.** The random set flags every checkpoint; the hard split flags only PII and "
        "*improves* on the others. The hard split was mined from the v1 adapter's own failures, so "
        f"v1 scores {fmt(base_hard['intent']['hard'][GATED['intent']])} on intent's hard cases and "
        f"{fmt(base_hard['urgency']['hard'][GATED['urgency']])} on urgency's by construction. Any "
        "model whose errors differ from v1's scores higher there. A split mined from one model's "
        "failures measures difference from that model, not difficulty — it stays report-only."),
    ]
    return out


ADAPTER_P95_MS = 500.0


def live_rows(data: dict) -> list[str]:
    """The request path's measured frontier-call rate, one row per client concurrency, beside the
    rate the offline curve gives the same pairs."""
    live = data["live"]
    svc = live["service"]
    lines = [
        (f"**Measured live, on the curve's own pairs.** `adapterops request-path-run` sent the "
         f"{live['pairs']} in-distribution pairs the curve counted through the request path — "
         f"{svc['local_backend']}, manifest v{svc['manifest_version']}, {svc['policy']} at "
         f"{svc['threshold']}, {svc['frontier']} on — one full pass per client concurrency. Served "
         "quality covers intent, urgency and PII; drafting is routed but not graded here."),
        "",
        ("| concurrency | pairs/s | frontier-call rate | curve, same pairs | same decision as curve "
         "| fallback rate | served quality | curve quality |"),
        "|---|---|---|---|---|---|---|---|",
    ]
    for level in live["levels"]:
        per_task = level["per_task"]
        pairs = sum(t["pairs"] for t in per_task.values())
        curve = sum(t["curve_escalation_rate"] * t["pairs"] for t in per_task.values()) / pairs
        graded, agree = level["graded_tasks"], level["agreement_with_curve"]
        lines.append(
            f"| {level['concurrency']} | {level['throughput_pairs_per_s']} | "
            f"{level['escalation_rate']:.1%} | {curve:.1%} | {agree['same_decision']:.1%} | "
            f"{level['fallback_rate']:.2%} | {graded['served_success']:.3f} | "
            f"{graded['curve_success_at_operating_point']:.3f} |")
    shift = data["shift"]["levels"][0]
    per_task = shift["per_task"]
    pairs = sum(t["pairs"] for t in per_task.values())
    at_threshold = sum(t["threshold_escalation_rate"] * t["pairs"] for t in per_task.values()) / pairs
    lines += [
        "",
        (f"**Shifted population** — the {data['shift']['pairs']:,} pairs the curve measured under shift, "
         f"concurrency {shift['concurrency']}, GPT-4o-mini on: **{shift['escalation_rate']:.1%}** "
         f"escalated against {at_threshold:.1%} on the recorded scores at the same threshold, the "
         f"recorded decision on {shift['agreement_with_curve']['same_decision']:.1%} of pairs, served "
         f"quality {shift['graded_tasks']['served_success']:.3f} against "
         f"{shift['graded_tasks']['recorded_success_as_served']:.3f} recorded for the side that served "
         "each pair."),
    ]
    return [*lines, ""]


def load_rows(data: dict) -> list[str]:
    """The A10's throughput ceiling, request path beside vLLM direct, sustained load, and the local
    cost per 1K derived at each measured rate."""
    econ = data["economics"]
    gpu = econ["assumptions"]["gpu_usd_per_hour"]
    frontier = econ["same_workload"]["frontier_usd_per_1k"]
    direct = {level["concurrency"]: level for level in data["sweep_vllm"]["levels"]}
    lines = [
        "### Throughput ceiling and sustained load — A10", "",
        (f"The curve's in-distribution pairs, cycled for {data['sweep']['duration_s']:.0f} s at each "
         "client concurrency, through the request path with GPT-4o-mini off — escalations counted, "
         "answered locally — and straight at vLLM. Local cost per 1K is derived from the "
         f"${gpu}/h assumption at the measured rate; GPT-4o-mini is ${frontier:.4f}."), "",
        ("| concurrency | request path pairs/s | vLLM direct pairs/s | request path P95 ms | "
         "escalated | derived local $ per 1K | GPT-4o-mini ÷ local |"),
        "|---|---|---|---|---|---|---|",
    ]
    for level in data["sweep"]["levels"]:
        c, rate = level["concurrency"], level["throughput_pairs_per_s"]
        per_1k = gpu / 3600 / rate * 1000
        lines.append(f"| {c} | {rate} | {direct[c]['throughput_pairs_per_s']} | "
                     f"{level['client_p95_ms']:,.0f} | {level['escalation_rate']:.1%} | "
                     f"${per_1k:.4f} | {frontier / per_1k:.1f}× |")
    sustained = data["sustained"]["levels"][0]
    windows = sustained["windows"]
    rates = [w["throughput_per_s"] for w in windows]
    p95 = [w["client_p95_ms"] for w in windows]
    escalated = [w["escalation_rate"] for w in windows]
    lines += [
        "",
        (f"**Sustained:** {sustained['requests']:,} requests over "
         f"{sustained['duration_s'] / 60:.0f} minutes at concurrency {sustained['concurrency']} — "
         f"{min(rates)}–{max(rates)} pairs/s in every minute, P95 {min(p95):,.0f}–{max(p95):,.0f} ms, "
         f"escalated {min(escalated):.1%}–{max(escalated):.1%}, {sustained['http_errors']} errors."),
        "",
    ]
    return lines


def targets_rows(data: dict) -> list[str]:
    """PRD §7's non-functional targets, each read against the run that measured it."""
    serving, lat = data["serving"], data["latency"]
    lines = [
        "## 7 · PRD §7 targets", "",
        ("Each non-functional target read against the run that measured it. A miss is shown as a "
         "miss, and a target nothing measured says so."), "",
        "| target | scope | measured | verdict |", "|---|---|---|---|",
    ]
    for task, r in serving["per_adapter"].items():
        verdict = "met" if r["p95_ms"] < ADAPTER_P95_MS else "**missed**"
        lines.append(f"| adapter P95 < {ADAPTER_P95_MS:,.0f} ms | {task} on the A10, M1, "
                     f"{r['mean_generated_tokens']} tokens generated on average | "
                     f"{r['p95_ms']:,.0f} ms | {verdict} |")
    for run in lat["runs"].values():
        one, four = run["batch_1_per_pair"], run["batch_4_per_ticket"]
        threads = f"{run['torch_threads']} CPU thread{'s' if run['torch_threads'] > 1 else ''}"
        lines.append(f"| router decision < {one['target_ms']:.0f} ms | one pair, {threads} | "
                     f"P95 {one['p95_ms']:.1f} ms | "
                     f"{'met' if one['p95_within_target'] else '**missed**'} |")
        per_ticket = ("within it per ticket" if four["p95_within_target"]
                      else f"met per pair ({four['p95_ms'] / 4:.1f} ms), over it per ticket")
        lines.append(f"| router decision < {four['target_ms']:.0f} ms | one ticket's four pairs "
                     f"batched, {threads} | P95 {four['p95_ms']:.1f} ms | {per_ticket} |")
    rps = serving["throughput_rps"]
    top = max(data["live"]["levels"], key=lambda level: level["concurrency"])
    reg = data["regression_live"]
    # The judge runs where its checkpoint lives, so its share is derived from the committed timing
    # rather than measured inside the GPU run: seconds per 1K drafts, plus one model load.
    judge = data["judge_cost"]["distilled"]
    drafts = sum(reg["per_split"]["drafting"][split]["n"] for split in ("random", "hard"))
    judge_seconds = judge["seconds_per_1k"] * drafts / 1000 + judge["load_seconds"]
    total_seconds = reg["wall_seconds"] + judge_seconds
    lines += [
        (f"| sustain 5–10 req/s briefly | four adapters at concurrency {serving['concurrency']}, M1 | "
         f"{rps} req/s for {serving['wall_seconds']:.0f} s | {'met' if rps >= 5 else '**missed**'} |"),
        (f"| sustain 5–10 req/s briefly | request path, routing and GPT-4o-mini included, "
         f"concurrency {top['concurrency']} | {top['throughput_pairs_per_s']} pairs/s for "
         f"{top['wall_seconds']:.0f} s | {'met' if top['throughput_pairs_per_s'] >= 5 else '**missed**'} |"),
        (f"| regression run < 25 min | both splits, four tasks, `{reg['name']}` on the A10, judge on "
         f"{judge['host']} CPU | {reg['wall_seconds']:.0f} s generating {reg['fallback']['requests']:,} "
         f"requests + ~{judge_seconds:.0f} s judging {drafts} drafts at the committed "
         f"{judge['seconds_per_1k']} s per 1K = ~{total_seconds / 60:.1f} min | "
         f"{'met' if total_seconds < 25 * 60 else '**missed**'} |"), "",
        (f"Router timed on {lat['host']} against the manifest's pinned checkpoint, reproducing its "
         f"committed scores. §7 set one latency budget for every adapter with no allowance for "
         f"output length; the two misses are the two tasks that generate long outputs."), "",
    ]
    return lines


def render(data: dict) -> str:
    econ = data["economics"]
    work = econ["same_workload"]
    serving = data["serving"]
    lines = [
        "# Results dashboard", "",
        ("Generated by `uv run adapterops dashboard` from committed runs — no number here is typed "
        "by hand. Six metrics (PRD §11). **The two quality rows are separate tables and are never "
        "averaged together** (F32)."), "",
        "Sources: " + " · ".join(f"`{p}`" for p in INPUTS.values()), "",
        "## 1 · Quality retained — random golden set", "",
        ("Two runs of the unchanged v1 manifest. A threshold is 3× the larger of inference and "
         "training spread (D37), and is enforced once the task's training spread is measured — "
         "enforced now for " + ", ".join(sorted(data["thresholds"]["gate"]["thresholds"])) +
         ". Intent's training spread comes from `runs/intent__adapter.json`; the others' from "
         "`runs/training_variance.json`, one same-configuration retrain each (D49). "
         "`manifests/system.json` carries a gate from its last promotion."), "",
        *quality_rows(data, "random"), "",
        *frontier_rows(data),
        "## 2 · Quality retained — hard cases", "",
        ("Mined from v1's failures and adjudicated for label noise. **Scores near 0 are expected "
        "by construction**, which is why this split cannot gate — see M11 below."), "",
        *quality_rows(data, "hard"), "",
        "## 3 · Frontier-call rate", "",
        *live_rows(data),
        ("**The offline operating curve** over the router pool, read at a "
         f"{OPERATING_BUDGET:.0%} escalation budget — what the live threshold was taken from. "
         "Drafting is graded by GPT-4o on both arms. The live pairs are this curve's in-distribution "
         "eval split, so the shifted population has not been measured live."), "",
        *routing_rows(data), "",
        ("Chart of the full curves, both populations: `runs/router__operating_curve__judged.png` "
         "(`uv run adapterops router-plot`)."), "",
        *router_v2_rows(data),
        *rules_rows(data),
        *misroute_rows(data),
        ("The adapter's own confidence is the only policy that pays. The learned router is worse "
        "than never escalating, and GPT-4o-mini on its own scores below the local adapters."), "",
        "## 4 · Cost per 1K requests — derived, not billed", "",
        "| | value |", "|---|---|",
        (f"| GPT-4o-mini, same {work['pairs']:,} pairs, list price | "
        f"${work['frontier_usd_per_1k']:.4f} |"),
        (f"| local A10 at M1 throughput ({econ['assumptions']['local_throughput_rps']} rps), "
        f"${econ['assumptions']['gpu_usd_per_hour']}/h assumed | "
        f"${work['local_usd_per_1k']:.4f} |"),
        (f"| **break-even sustained load** | **{work['break_even_rps']} req/s** "
        f"({work['break_even_requests_per_hour']:,} req/h) |"),
        f"| ratio, only if the GPU is never idle | {work['ratio_if_gpu_fully_busy']}× |",
        (f"| weights on disk: one base + {len(econ['footprint']['adapter_bytes'])} adapters vs a "
        f"full copy per task | {econ['footprint']['one_base_plus_adapters_bytes'] / 1e9:.2f} GB "
        f"vs {econ['footprint']['full_copy_per_task_bytes'] / 1e9:.2f} GB |"),
        (f"| judging 1K drafting replies: GPT-4o, token-derived over "
         f"{data['judge_cost']['gpt4o']['overall']['grades']:,} grades | "
         f"${data['judge_cost']['gpt4o']['overall']['usd_per_1k']:.2f} |"),
        (f"| judging 1K drafting replies: distilled judge, "
         f"{data['judge_cost']['distilled']['host']} CPU, no API | "
         f"{data['judge_cost']['distilled']['seconds_per_1k']:.0f} s |"), "",
        "Not measured: " + "; ".join(f"{k.replace('_', ' ')} ({v})"
                                     for k, v in econ["not_measured"].items()) + ".", "",
        ("## 5 · P95 latency — M1, A10, four adapters at once, concurrency "
        f"{serving['concurrency']}"), "",
        "| adapter | requests | P50 ms | P95 ms | max new tokens | mean generated |",
        "|---|---|---|---|---|---|",
    ]
    for task, r in serving["per_adapter"].items():
        lines.append(f"| {task} | {r['requests']:,} | {r['p50_ms']:,.0f} | {r['p95_ms']:,.0f} | "
                     f"{r['max_new_tokens']} | {r['mean_generated_tokens']} |")
    reg, levels = data["regression_live"], data["live"]["levels"]
    live_pairs = sum(level["pairs"] for level in levels)
    live_fallbacks = sum(round(level["fallback_rate"] * level["pairs"]) for level in levels)
    lines += [
        "", *load_rows(data), "## 6 · Fallback rate", "",
        "| run | fallbacks | requests | rate |", "|---|---|---|---|",
        (f"| M1, four adapters, concurrency {serving['concurrency']} | {serving['errors']} | "
         f"{serving['requests']:,} | {serving['fallback_rate']:.2%} |"),
        (f"| regression run `{reg['name']}`, requests local serving failed | "
         f"{reg['fallback']['errors']} | {reg['fallback']['requests']:,} | "
         f"{reg['fallback']['fallback_rate']:.2%} |"),
        (f"| request path, {len(levels)} passes, errors or unusable output | {live_fallbacks} | "
         f"{live_pairs:,} | {live_fallbacks / live_pairs:.2%} |"), "",
        ("A request-path fallback is local output the task cannot use — here one intent label "
         "outside the label set, the same pair on every pass — answered by GPT-4o-mini and counted "
         "apart from escalation."), "",
        *targets_rows(data),
        *failure_demo(data), "",
    ]
    return "\n".join(lines)


def main() -> int:
    OUT_FILE.write_text(render(load()))
    print(f"wrote {OUT_FILE.relative_to(REPO_ROOT)}")
    return 0
