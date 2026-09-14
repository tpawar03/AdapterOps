"""AdapterOps demo (F22, M8) — route a ticket, try each adapter, and see what was measured.

**This is not the benchmark path (D5, D40).** The benchmark numbers come from vLLM on a rented A10.
This app runs the same pinned adapters with `transformers` + `peft` on whatever hardware hosts it —
a free CPU Space, or a laptop — so its timings describe that hardware and nothing else.

**Route a ticket** is PRD §5's primary flow: one ticket through the request path
(`adapterops.serve.pipeline`), with each task's routing decision, the score it was made on, the
output, and per-request latency and cost. Routing uses the confidence policy at the committed 20%
operating point. GPT-4o-mini is called only if `OPENAI_API_KEY` is set; without it, a pair that
would escalate is answered locally and labelled as such.

Same prompts as training, same pinned revisions as serving (`manifests/system.json`), greedy
decoding. Drafting is capped below the benchmark's 448 tokens to keep a CPU reply short, and a reply
that hits the cap is marked as cut off.

    uv run python demo/app.py                 # from the repo
    python app.py                             # from a bundle built by demo/build_space.py
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src" / "adapterops").exists() else HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from adapterops.serve import pipeline as pl

DASHBOARD = ROOT / "runs" / "DASHBOARD.md"
FAILURE_HEADING = "## Failure demo"
TASKS = pl.TASKS

CAVEATS = {
    "intent": "Banking77, 77 labels. The adapter clearly beats a prompted base model.",
    "urgency": "**Negative result:** this adapter loses to a TF-IDF baseline and is not "
               "distinguishable from a prompted base model. Trained on CC BY-NC 4.0 data — "
               "non-commercial use only.",
    "pii": "Trained on synthetic spans only, every one of which contained personal data — it can "
           "invent spans on text that has none. **Do not enter real personal data.**",
    "drafting": "Graded by GPT-4o, the adapter beats a prompted base model. The distilled judge "
                "alone had it the other way round — see Results.",
}

EXAMPLES = [
    ["intent", "My card payment was declined at the supermarket this morning."],
    ["urgency", "Our payment page has been down for an hour and customers cannot check out."],
    ["pii", ("Hi, this is Dana Whitfield, born 14/03/1991. Reach me at dana.w@example.org "
             "or 555-0142.")],
    ["drafting", "I want to cancel my order, it has not shipped yet."],
]

TICKETS = [
    ["My card still hasn't arrived and it's been three weeks. This is unacceptable."],
    [("Hi, I'm Dana Whitfield (dana.w@example.org). I was charged twice for the same transfer "
      "yesterday — please refund one of them urgently.")],
]


@lru_cache(maxsize=1)
def backend() -> pl.TransformersBackend:
    return pl.TransformersBackend(pl.adapter_components(pl.load_manifest()))


@lru_cache(maxsize=1)
def service() -> tuple[pl.Service, str]:
    """The request path with the demo's backend. Returns the service and a note on what is live."""
    manifest = pl.load_manifest()
    vocab = pl.load_vocab()
    notes = []
    frontier = None
    if os.environ.get("OPENAI_API_KEY"):
        frontier = pl.OpenAIFrontier(vocab["intent"])
    else:
        notes.append("GPT-4o-mini is not configured here, so pairs that would escalate are "
                     "answered locally and marked.")
    judge = None
    try:
        checkpoint = pl.pinned_checkpoint(manifest, "judge")
        if checkpoint is not None:
            from adapterops.judge.score import load_judge

            judge = load_judge(checkpoint)
    except (FileNotFoundError, ValueError):
        notes.append("The distilled judge's weights are not on this host, so drafting is not "
                     "scored.")
    svc = pl.Service(local=backend(), policy=pl.make_policy("confidence", manifest),
                     frontier=frontier, judge=judge, vocab=vocab, manifest=manifest)
    return svc, " ".join(notes)


def first_line(text: str) -> str:
    return text.strip().split("\n")[0].strip()


def run(task: str, text: str) -> str:
    if not text.strip():
        return "Enter some text first."
    g = backend().generate(task, text.strip())
    if task in ("intent", "urgency"):
        body = f"### `{first_line(g.text)}`"
    elif task == "pii":
        spans = pl.parse_output("pii", g.text)["spans"]
        body = ("| label | value |\n|---|---|\n"
                + "\n".join(f"| {s['label']} | {s['value']} |" for s in spans)
                if spans else "_No personal information found._")
    else:
        body = g.text.strip()
    cap = pl.TransformersBackend.MAX_NEW_TOKENS[task]
    note = f"\n\n_Cut off at the demo's {cap}-token cap._" if g.finish_reason == "length" else ""
    return (f"{body}{note}\n\n---\n{CAVEATS[task]}\n\n"
            f"<sub>{g.n_tokens} tokens in {g.latency_s:.1f} s on this host — not the "
            f"benchmark.</sub>")


def _cell(text: str, limit: int = 80) -> str:
    text = " ".join(str(text).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[:limit - 1] + "…"


def _output(task: str, pair: dict) -> str:
    out = pair["output"]
    if out is None:
        return "—"
    if task in ("intent", "urgency"):
        return f"`{out['label']}`"
    if task == "pii":
        return _cell(", ".join(f"{s['label']}: {s['value']}" for s in out["spans"]) or "none found")
    return _cell(out["reply"])


def route(text: str) -> str:
    if not text.strip():
        return "Enter a ticket first."
    svc, note = service()
    result = svc.handle(text.strip())
    info = svc.describe()
    rows = []
    for task in TASKS:
        p = result["pairs"][task]
        score = "—" if p["score"] is None else f"{p['score']:.3f} vs {p['threshold']:.3f}"
        route_cell = {"local": "local", "escalated": "**escalate**",
                      "fallback": "**fallback**"}[p["route"]]
        probability = "—" if p["mean_token_probability"] is None else p["mean_token_probability"]
        judged = "—" if p["judge_score"] is None else p["judge_score"]
        rows.append(f"| {task} | {route_cell} | {p['served_by']} | {score} | {probability} | "
                    f"{_output(task, p)} | {judged} | {p['latency_ms'] / 1000:.1f} s | "
                    f"${p['cost_usd']:.6f} |")
    reply = result["pairs"]["drafting"]["output"]
    notes = sorted({n for p in result["pairs"].values() for n in p["notes"]})
    header = (f"**Manifest v{info['manifest_version']}** · policy `{info['policy']}` · escalate "
              f"when −mean log-prob ≥ {info['threshold']:.4f} (the committed 20% operating point)")
    footer = (f"<sub>Ticket total {result['latency_ms'] / 1000:.1f} s · "
              f"${result['cost_usd']:.6f} (derived: A10 hour over M1 throughput, GPT-4o-mini at "
              f"list price). Decisions near the threshold can differ from vLLM's: the threshold "
              f"was measured on its log-probabilities. {note}</sub>")
    table_head = ("| task | route | answered by | score vs threshold | mean token prob | output | "
                  "judge | latency | cost |")
    return "\n".join([
        header, "", table_head, "|---|---|---|---|---|---|---|---|---|", *rows, "",
        f"**Drafted reply:** {reply['reply'] if reply else '—'}", "",
        *(f"_{n}_" for n in notes),
        footer,
    ])


def dashboard_parts() -> tuple[str, str]:
    if not DASHBOARD.exists():
        missing = "`runs/DASHBOARD.md` is missing — run `uv run adapterops dashboard`."
        return missing, missing
    text = DASHBOARD.read_text()
    results, _, failure = text.partition(FAILURE_HEADING)
    return results, FAILURE_HEADING + failure


def build() -> gr.Blocks:
    results, failure = dashboard_parts()
    with gr.Blocks(title="AdapterOps") as ui:
        gr.Markdown(
            "# AdapterOps\n"
            "Four LoRA adapters over one Qwen2.5-1.5B base, a regression gate proven by breaking "
            "an adapter on purpose, and the negative results kept. Portfolio project — no real "
            "users or customer data.")
        with gr.Tab("Route a ticket"):
            gr.Markdown("One ticket, four (ticket, task) pairs, each routed on the adapter's own "
                        "confidence. The first request loads the base model and four adapters, "
                        "which takes a minute or two on CPU.")
            ticket = gr.Textbox(lines=4, label="Support ticket")
            go_route = gr.Button("Route", variant="primary")
            routed = gr.Markdown()
            gr.Examples(TICKETS, inputs=[ticket])
            go_route.click(route, inputs=[ticket], outputs=routed)
        with gr.Tab("Try one adapter"):
            task = gr.Radio(list(TASKS), value="intent", label="Adapter")
            text = gr.Textbox(lines=4, label="Input")
            go = gr.Button("Run", variant="primary")
            output = gr.Markdown()
            gr.Examples(EXAMPLES, inputs=[task, text])
            go.click(run, inputs=[task, text], outputs=output)
        with gr.Tab("Results"):
            gr.Markdown(results)
        with gr.Tab("Failure demo"):
            gr.Markdown(failure)
    return ui


if __name__ == "__main__":
    build().launch()
