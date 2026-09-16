"""Re-score one task's routing pairs with a replaced adapter (D48, D50).

The confidence router escalates a pair when the adapter's `-mean_logprob` is at or above the committed
operating threshold, and that threshold was calibrated on scores the served adapters produced. Manifest v6
replaced the PII adapter with one trained to give empty answers, so its confidence on real PII may not be
the old adapter's — and at the old adapter's scores PII escalated nothing.

This scores every PII pair in both router populations with **both** adapters on one machine, the same way,
so a difference is the adapter's and not the hardware's: generation follows the model's own config (as vLLM
does), and the score is the mean raw log-probability of the generated tokens up to and including the first
end-of-text token (D48). The old adapter's laptop scores are checked against the recorded vLLM scores
before anything is read from the new one.

Reported per population: how many PII pairs each adapter escalates at the live threshold, how many sit just
under it, whether each adapter answers correctly by the curve's rule, the PII quality the routing decision
yields, and the whole population's escalation count with only PII's scores replaced, against the curve's
20% budget.

    uv run adapterops router-rescore --task pii
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RUNS_DIR = REPO_ROOT / "runs"
HISTORY = REPO_ROOT / "manifests" / "history"
NEAR = 0.05
"""How far under the threshold counts as close enough for a small shift to flip the decision."""


def mean_logprobs(scores, generated, eos_ids: Sequence[int]) -> list[tuple[float | None, int]]:
    """Per sequence: the mean of `scores` over generated tokens up to and including the first end-of-text
    token, and that token count. Padding after a finished sequence is never averaged in."""
    import torch

    eos = torch.tensor(list(eos_ids), device=generated.device)
    out = []
    for row_scores, row_tokens in zip(scores, generated, strict=True):
        is_eos = torch.isin(row_tokens, eos).nonzero()
        length = int(is_eos[0]) + 1 if len(is_eos) else len(row_tokens)
        values = row_scores[:length].float()
        values = values[torch.isfinite(values)]
        out.append((float(values.mean()) if len(values) else None, length))
    return out


def load_two(task: str, old: dict, new: dict):
    """One base model with the replaced and the replacing adapter loaded side by side."""
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    def path_of(pin: dict) -> str:
        return pin.get("path") or snapshot_download(
            pin["repo"], revision=pin["revision"],
            allow_patterns=["adapter_model.safetensors", "adapter_config.json"])

    base = json.loads((REPO_ROOT / "manifests" / "system.json").read_text())["components"]["base_model"]
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "mps" else torch.float32
    tok = AutoTokenizer.from_pretrained(base["repo"], revision=base["revision"])
    tok.padding_side = "left"
    tok.pad_token = tok.pad_token or tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(base["repo"], revision=base["revision"], dtype=dtype)
    model = PeftModel.from_pretrained(model, path_of(old), adapter_name="old")
    model.load_adapter(path_of(new), adapter_name="new")
    return tok, model.to(device).eval(), device


SCORE_CHUNK = 64
"""Generated positions scored at once. `output_logits` keeps batch × steps × a 151,936-token vocabulary —
37 GB for 16 PII answers at the 384-token cap — so the scores come from a second, chunked pass instead."""


def token_logprobs(model, sequences, attention_mask, prompt_len: int):
    """Raw log-probability of each generated token, teacher-forced over the generated sequence.

    The hidden states come from one forward pass of the adapter-carrying model with position ids that
    match generation's under left padding; the vocabulary projection runs only over generated positions,
    `SCORE_CHUNK` at a time, so the full logits tensor is never materialised."""
    import torch

    inner = model.get_base_model()
    positions = (attention_mask.cumsum(-1) - 1).clamp(min=0)
    hidden = inner.model(input_ids=sequences, attention_mask=attention_mask,
                         position_ids=positions).last_hidden_state
    steps = sequences.shape[1] - prompt_len
    out = []
    for start in range(0, steps, SCORE_CHUNK):
        stop = min(start + SCORE_CHUNK, steps)
        logits = inner.lm_head(hidden[:, prompt_len - 1 + start:prompt_len - 1 + stop]).float()
        chosen = sequences[:, prompt_len + start:prompt_len + stop].unsqueeze(-1)
        out.append(torch.log_softmax(logits, dim=-1).gather(-1, chosen).squeeze(-1))
    return torch.cat(out, dim=1)


def generate(tok, model, device: str, adapter: str, task: str, texts: Sequence[str],
             batch: int) -> list[dict]:
    import torch

    from adapterops.router.generate import MAX_TOKENS
    from adapterops.train.qlora import PROMPTS

    model.set_adapter(adapter)
    eos_ids = model.generation_config.eos_token_id
    eos_ids = eos_ids if isinstance(eos_ids, list) else [eos_ids]
    rows = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        enc = tok([PROMPTS[task].format(text=t) for t in chunk], return_tensors="pt", padding=True,
                  add_special_tokens=False).to(device)
        prompt_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            sequences = model.generate(**enc, max_new_tokens=MAX_TOKENS[task], do_sample=False,
                                       pad_token_id=tok.pad_token_id)
            generated = sequences[:, prompt_len:]
            mask = torch.cat([enc["attention_mask"], torch.ones_like(generated)], dim=1)
            scores = token_logprobs(model, sequences, mask, prompt_len)
        for tokens, (mean, length) in zip(generated, mean_logprobs(scores, generated, eos_ids),
                                          strict=True):
            rows.append({"prediction": tok.decode(tokens[:length], skip_special_tokens=True),
                         "mean_logprob": mean, "n_tokens": length})
        del sequences, generated, mask, scores
        if device == "mps":
            # MPS keeps freed blocks cached, so memory climbs batch by batch; a full PII run on a 24 GB
            # machine reached 24 GB before this. Releasing the cache each batch keeps it flat.
            torch.mps.empty_cache()
        # One line per batch, so a run piped through grep or tail still shows where it is.
        print(f"  progress {adapter}: {min(i + batch, len(texts))}/{len(texts)}", flush=True)
    return rows


def success(task: str, text: str, gold: str, prediction: str) -> bool:
    from adapterops.router.scoring import score_pair

    scored = score_pair(task, text, gold, prediction)
    return bool(scored["span_f1"] >= 1.0) if task == "pii" else bool(scored.get("exact") == 1.0)


def summarise(population: pd.DataFrame, task_rows: pd.DataFrame, threshold: float,
              budget_escalations: int) -> dict:
    """`population` is every pair with its recorded score; `task_rows` the re-scored pairs of one task."""

    def block(score: pd.Series, ok: pd.Series) -> dict:
        esc = score >= threshold
        quality = np.where(esc, task_rows.recorded_frontier_success, ok)
        return {
            "score_median": round(float(score.median()), 5),
            "score_p95": round(float(score.quantile(0.95)), 5),
            "score_max": round(float(score.max()), 5),
            "escalated_at_threshold": int(esc.sum()),
            "within_near_below_threshold": int(((score < threshold) & (score >= threshold - NEAR)).sum()),
            "local_success": round(float(ok.mean()), 4),
            "quality_at_threshold": round(float(np.mean(quality)), 4),
        }

    recorded = -task_rows.recorded_mean_logprob
    old, new = -task_rows.old_mean_logprob, -task_rows.new_mean_logprob
    others = population[~population.pair_id.isin(task_rows.pair_id)]
    other_escalations = int((-others.recorded_mean_logprob >= threshold).sum())
    return {
        "pairs": len(task_rows),
        "population_pairs": len(population),
        "threshold": threshold,
        "near": NEAR,
        "recorded_vllm_old_adapter": block(recorded, task_rows.recorded_local_success.astype(bool)),
        "laptop_old_adapter": block(old, task_rows.old_success),
        "laptop_new_adapter": block(new, task_rows.new_success),
        "laptop_old_vs_recorded": {
            "spearman": round(float(old.rank().corr(recorded.rank())), 4),
            "median_abs_diff": round(float((old - recorded).abs().median()), 5),
            "same_decision": round(float(((old >= threshold) == (recorded >= threshold)).mean()), 4),
        },
        "new_vs_old_same_decision": round(float(((new >= threshold) == (old >= threshold)).mean()), 4),
        "population_escalations": {
            "budget_20pct": budget_escalations,
            "recorded": other_escalations + int((recorded >= threshold).sum()),
            "with_new_adapter_scores": other_escalations + int((new >= threshold).sum()),
        },
    }


def main(task: str = "pii", batch: int = 8, limit: int | None = None) -> int:
    from adapterops.serve.pipeline import operating_threshold
    from adapterops.serve.request_run import POPULATIONS, request_set

    threshold = operating_threshold("confidence")
    current = json.loads((REPO_ROOT / "manifests" / "system.json").read_text())
    new_pin = current["components"]["adapters"][task]
    replaces = json.loads((REPO_ROOT / "manifests" / "adapters.json").read_text())["components"][task].get(
        "replaces", {}).get("revision")
    old_manifest = next((json.loads(p.read_text()) for p in sorted(HISTORY.glob("system-*.json"), reverse=True)
                         if json.loads(p.read_text())["components"]["adapters"][task]["revision"] == replaces),
                        None)
    if old_manifest is None:
        print(f"  no archived manifest pins the replaced {task} revision {replaces}")
        return 2
    old_pin = old_manifest["components"]["adapters"][task]

    tok, model, device = load_two(task, old_pin, new_pin)
    result, frames = {"task": task, "device": device, "batch": batch,
                      "old_adapter": {"revision": old_pin["revision"], "manifest": old_manifest["version"]},
                      "new_adapter": {"revision": new_pin["revision"], "manifest": current["version"]},
                      "populations": {}}, []
    started = time.perf_counter()
    for population in POPULATIONS:
        pairs = request_set(population)
        rows = pairs[pairs.task == task].reset_index(drop=True)
        if limit:
            rows = rows.head(limit)
        texts = list(rows.text)
        for name in ("old", "new"):
            generated = pd.DataFrame(generate(tok, model, device, name, task, texts, batch))
            rows[f"{name}_mean_logprob"] = generated.mean_logprob
            rows[f"{name}_prediction"] = generated.prediction
            rows[f"{name}_success"] = [success(task, t, g, p) for t, g, p in
                                       zip(rows.text, rows.gold, generated.prediction, strict=True)]
        budget = int(pairs.recorded_escalated.sum())
        result["populations"][population] = summarise(pairs, rows, threshold, budget)
        frames.append(rows.drop(columns=["text", "gold"]).assign(population=population))
    result["seconds"] = round(time.perf_counter() - started, 1)
    RUNS_DIR.mkdir(exist_ok=True)
    out = RUNS_DIR / f"router__rescore__{task}.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    pd.concat(frames, ignore_index=True).to_parquet(RUNS_DIR / f"router__rescore__{task}__pairs.parquet",
                                                    index=False)
    for population, s in result["populations"].items():
        print(f"  {population}: escalated at threshold — recorded {s['recorded_vllm_old_adapter']['escalated_at_threshold']}"
              f", old {s['laptop_old_adapter']['escalated_at_threshold']}, new "
              f"{s['laptop_new_adapter']['escalated_at_threshold']} of {s['pairs']} · population "
              f"{s['population_escalations']}")
    print(f"  wrote {out.relative_to(REPO_ROOT)}")
    return 0
