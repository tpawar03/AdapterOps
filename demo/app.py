"""AdapterOps demo (F22, M8) — the four adapters, the results dashboard, and the failure demo.

**This is not the benchmark path (D5, D40).** The benchmark numbers come from vLLM on a rented A10.
This app runs the same pinned adapters with `transformers` + `peft` on whatever hardware hosts it —
a free CPU Space, or a laptop — so its timings describe that hardware and nothing else. It exists
to show what each adapter does; the Results and Failure demo tabs show what was measured.

Same prompts as training (`adapterops.train.qlora.PROMPTS`), same pinned revisions as serving
(`manifests/adapters.json`), greedy decoding. Drafting is capped below the benchmark's 448 tokens
to keep a CPU reply under a minute, and a reply that hits the cap is marked as cut off.

    uv run python demo/app.py                 # from the repo
    python app.py                             # from a bundle built by demo/build_space.py
"""

from __future__ import annotations

import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "src" / "adapterops").exists() else HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from adapterops.eval.spans import parse_model_output
from adapterops.train.qlora import PROMPTS

PINS = ROOT / "manifests" / "adapters.json"
DASHBOARD = ROOT / "runs" / "DASHBOARD.md"
FAILURE_HEADING = "## Failure demo"
TASKS = ("intent", "urgency", "pii", "drafting")

MAX_NEW_TOKENS = {"intent": 12, "urgency": 6, "pii": 384, "drafting": 200}
"""Benchmark caps (router.generate.MAX_TOKENS), except drafting: 448 there, 200 here for CPU."""

CAVEATS = {
    "intent": "Banking77, 77 labels. The adapter clearly beats a prompted base model.",
    "urgency": "**Negative result:** this adapter loses to a TF-IDF baseline and is not "
               "distinguishable from a prompted base model. Trained on CC BY-NC 4.0 data — "
               "non-commercial use only.",
    "pii": "Trained on synthetic spans only. **Do not enter real personal data.**",
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


def first_line(text: str) -> str:
    """The classification rule the scorer applies (router.scoring)."""
    return text.strip().split("\n")[0].strip()


@lru_cache(maxsize=1)
def load_model():
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    components = json.loads(PINS.read_text())["components"]
    base = components["base_model"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32

    tok = AutoTokenizer.from_pretrained(base["repo"], revision=base["revision"])
    model = AutoModelForCausalLM.from_pretrained(base["repo"], revision=base["revision"],
                                                 dtype=dtype)
    for i, task in enumerate(TASKS):
        entry = components[task]
        path = snapshot_download(entry["repo"], revision=entry["revision"],
                                 allow_patterns=["adapter_model.safetensors",
                                                 "adapter_config.json"])
        if i == 0:
            model = PeftModel.from_pretrained(model, path, adapter_name=task)
        else:
            model.load_adapter(path, adapter_name=task)
    model.to(device).eval()
    return tok, model, device


def run(task: str, text: str) -> str:
    import torch

    if not text.strip():
        return "Enter some text first."
    tok, model, device = load_model()
    model.set_adapter(task)
    ids = tok(PROMPTS[task].format(text=text.strip()), return_tensors="pt",
              add_special_tokens=False).to(device)
    started = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=MAX_NEW_TOKENS[task], do_sample=False,
                             pad_token_id=tok.eos_token_id)
    seconds = time.perf_counter() - started
    new = out[0, ids["input_ids"].shape[1]:]
    reply = tok.decode(new, skip_special_tokens=True)
    cut_off = len(new) >= MAX_NEW_TOKENS[task] and int(new[-1]) != tok.eos_token_id

    if task in ("intent", "urgency"):
        body = f"### `{first_line(reply)}`"
    elif task == "pii":
        pairs = parse_model_output(reply)
        body = ("| label | value |\n|---|---|\n"
                + "\n".join(f"| {label} | {value} |" for label, value in pairs)
                if pairs else "_No personal information found._")
    else:
        body = reply.strip()
    note = (f"\n\n_Cut off at the demo's {MAX_NEW_TOKENS[task]}-token cap._" if cut_off else "")
    return (f"{body}{note}\n\n---\n{CAVEATS[task]}\n\n"
            f"<sub>{len(new)} tokens in {seconds:.1f} s on {device} — this host, not the "
            f"benchmark.</sub>")


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
        with gr.Tab("Try the adapters"):
            gr.Markdown("The first request loads the base model and four adapters, which takes "
                        "a minute or two on CPU.")
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
