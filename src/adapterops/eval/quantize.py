"""int8 quantization comparison on one adapter (F27, N2) — int8 on CPU, intent golden set.

**What this measures, and what it does not.** N2 asks for an int8 comparison on one adapter. The
paid version is int8 serving on the A10, where latency is reportable (§7). This one spends nothing:
the pinned intent adapter merged into the pinned base model, run on this machine's CPU in fp32 and in
three int8 variants. **Quality is comparable across the variants**: same host, prompts, greedy decoding
and golden items, so a changed label is the quantization's doing. **Speed and size are this CPU's**,
not a serving figure, and are recorded as such.

**Three int8 variants, because one would confuse a recipe with the idea.**

- `int8_per_tensor` — PyTorch dynamic quantization, default qconfig: int8 weights with one scale per
  matrix, and **activations quantized to int8 per batch** as they flow through each Linear layer.
- `int8_per_channel` — the same, with one weight scale per output row.
- `int8_weight_only_simulated` — every Linear weight rounded to per-channel symmetric int8 and back,
  activations left in fp32. Its quality is what weight-only int8 inference computes; its speed is
  fp32's, because it runs fp32 kernels, so no speed is reported for it.

The first two were run first, and both lost ~30 points of accuracy. Dynamic quantization changes
weights *and* activations, and small decoder models carry large activation outliers that an 8-bit
per-tensor activation range clips — so the weight-only variant is what separates "int8 weights hurt"
from "8-bit activations hurt".

Intent is the adapter chosen because its gated metric is exact-match accuracy — a quantization error
shows up as a changed label rather than a shift in a judge score — and its outputs are a few tokens,
so several full passes fit in a CPU session.

**The fp32 pass is also checked against the benchmark.** The same weights scored 0.9286 through vLLM
on the A10. A CPU fp32 figure far from that would mean this harness is not the one the gate used, and
the int8 deltas beside it would mean nothing; the difference is recorded.

    uv run adapterops quantize-compare             # 770 items, four passes, CPU
    uv run adapterops quantize-compare --limit 154 # 2 per class, a quick look
"""

from __future__ import annotations

import json
import os
import platform
import tempfile
import time
from pathlib import Path

import pandas as pd

from adapterops.eval.regression import score
from adapterops.serve.pipeline import adapter_components, first_line, load_manifest, load_vocab
from adapterops.train.qlora import COLUMNS, PROMPTS

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN = REPO_ROOT / "evals" / "golden" / "intent.parquet"
BENCHMARK_RUN = REPO_ROOT / "runs" / "regression__v1-baseline-1.json"
OUT_FILE = REPO_ROOT / "runs" / "intent__int8.json"
TASK = "intent"
BATCH = 16
MAX_NEW_TOKENS = 12
DYNAMIC = ("int8_per_tensor", "int8_per_channel")
WEIGHT_ONLY = "int8_weight_only_simulated"
RECIPES = (*DYNAMIC, WEIGHT_ONLY)


def balanced_sample(frame: pd.DataFrame, label: str, per_class: int) -> pd.DataFrame:
    """The first `per_class` items of every class, so a quick run keeps the golden set's balance."""
    return frame.groupby(label, group_keys=False, sort=False).head(per_class).reset_index(drop=True)


def agreement(a: list[str], b: list[str]) -> float:
    return sum(first_line(x) == first_line(y) for x, y in zip(a, b, strict=True)) / len(a)


def generate_all(tok, model, texts: list[str]) -> tuple[list[str], float]:
    import torch

    outputs: list[str] = []
    started = time.perf_counter()
    for i in range(0, len(texts), BATCH):
        prompts = [PROMPTS[TASK].format(text=t) for t in texts[i:i + BATCH]]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        with torch.inference_mode():
            generated = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                       pad_token_id=tok.pad_token_id or tok.eos_token_id)
        outputs += tok.batch_decode(generated[:, enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True)
    return outputs, time.perf_counter() - started


def parameter_bytes(model) -> int:
    """fp32 size from tensor storage, counting tied weights once — Qwen2.5-1.5B ties its input
    embedding to its output head, and the state dict lists that one matrix under both names."""
    seen, total = set(), 0
    for t in model.state_dict().values():
        if not hasattr(t, "untyped_storage"):
            continue
        key = t.untyped_storage().data_ptr()
        if key not in seen:
            seen.add(key)
            total += t.numel() * t.element_size()
    return total


def serialized_bytes(model) -> int:
    """int8 size as saved. Quantized Linear layers hold packed parameters, not plain tensors."""
    import torch

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "model.pt"
        torch.save(model.state_dict(), path)
        return path.stat().st_size


def round_trip_int8(weight):
    """Per-channel symmetric int8: one scale per output row, values in [-127, 127]."""
    scale = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127
    return (weight / scale).round().clamp(-127, 127) * scale


def simulate_weight_only_int8(model) -> int:
    """Replace every Linear weight with its int8 round trip; return the int8 model's size.

    Each weight becomes a new tensor, so the output head's tie to the input embedding breaks and the
    embedding lookup keeps full precision — as dynamic quantization leaves it. Size is analytic:
    one byte per Linear weight plus a fp32 scale per row, and everything else at fp32.
    """
    import torch

    storages = {t.untyped_storage().data_ptr() for t in model.state_dict().values()
                if hasattr(t, "untyped_storage")}
    total = parameter_bytes(model)
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                w = module.weight
                shared = sum(1 for t in model.state_dict().values()
                             if hasattr(t, "untyped_storage")
                             and t.untyped_storage().data_ptr() == w.untyped_storage().data_ptr()) > 1
                if not shared and w.untyped_storage().data_ptr() in storages:
                    total -= w.numel() * w.element_size()        # its fp32 copy goes away
                total += w.numel() + w.shape[0] * 4               # int8 values + a scale per row
                module.weight = torch.nn.Parameter(round_trip_int8(w), requires_grad=False)
    return total


def main(limit: int | None = None) -> int:
    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from torch.ao.quantization import (
        default_dynamic_qconfig,
        per_channel_dynamic_qconfig,
        quantize_dynamic,
    )
    from transformers import AutoModelForCausalLM, AutoTokenizer

    manifest = load_manifest()
    components = adapter_components(manifest)
    base, adapter = components["base_model"], components[TASK]
    text_col, gold_col = COLUMNS[TASK]
    frame = pd.read_parquet(GOLDEN)
    if limit:
        frame = balanced_sample(frame, gold_col, max(1, limit // frame[gold_col].nunique()))
    texts = frame[text_col].astype(str).tolist()
    gold = frame[gold_col].astype(str).tolist()
    vocab = load_vocab()[TASK]

    tok = AutoTokenizer.from_pretrained(base["repo"], revision=base["revision"],
                                        padding_side="left")
    adapter_path = snapshot_download(adapter["repo"], revision=adapter["revision"],
                                     allow_patterns=["adapter_model.safetensors",
                                                     "adapter_config.json"])

    def merged():
        model = AutoModelForCausalLM.from_pretrained(base["repo"], revision=base["revision"],
                                                     dtype=torch.float32)
        return PeftModel.from_pretrained(model, adapter_path).merge_and_unload().eval()

    engines = torch.backends.quantized.supported_engines
    if platform.machine() in ("arm64", "aarch64") and "qnnpack" in engines:
        torch.backends.quantized.engine = "qnnpack"
    qconfigs = {"int8_per_tensor": default_dynamic_qconfig,
                "int8_per_channel": per_channel_dynamic_qconfig}

    def summary(pred: list[str], seconds: float | None, size: int, size_method: str) -> dict:
        return {**score(TASK, texts, gold, pred),
                "exact_label_rate": round(sum(first_line(p) in vocab for p in pred) / len(pred), 4),
                "seconds": None if seconds is None else round(seconds, 1),
                "seconds_per_item": None if seconds is None else round(seconds / len(pred), 3),
                "weights_bytes": size, "weights_bytes_method": size_method}

    print(f"  {len(texts)} items, batch {BATCH}, {torch.get_num_threads()} threads", flush=True)
    model = merged()
    fp32_bytes = parameter_bytes(model)
    fp32_pred, fp32_seconds = generate_all(tok, model, texts)
    variants = {"fp32": summary(fp32_pred, fp32_seconds, fp32_bytes,
                                "tensor storage, tied weights once")}
    print(f"  fp32 {variants['fp32']['micro_accuracy']} in {fp32_seconds:.0f} s", flush=True)
    predictions = {"fp32": fp32_pred}

    for recipe in RECIPES:
        model = merged()
        if recipe in DYNAMIC:
            quantize_dynamic(model, {torch.nn.Linear: qconfigs[recipe]}, inplace=True)
            pred, seconds = generate_all(tok, model, texts)
            variants[recipe] = summary(pred, seconds, serialized_bytes(model),
                                       "serialised state_dict")
        else:
            size = simulate_weight_only_int8(model)
            pred, _ = generate_all(tok, model, texts)
            variants[recipe] = summary(pred, None, size,
                                       "analytic: int8 Linear weights + per-row scales, rest fp32")
        predictions[recipe] = pred
        print(f"  {recipe} {variants[recipe]['micro_accuracy']}", flush=True)
        del model

    benchmark = json.loads(BENCHMARK_RUN.read_text())["per_split"][TASK]["random"]
    fp32 = variants["fp32"]
    out = {
        "prd": "F27 / N2 — int8 quantization comparison on one adapter",
        "adapter": {"task": TASK, "repo": adapter["repo"], "revision": adapter["revision"],
                    "manifest_version": manifest["version"]},
        "method": {
            "common": (f"adapter merged into the base; every nn.Linear including lm_head; greedy, "
                       f"{MAX_NEW_TOKENS} new tokens, batch {BATCH}"),
            "int8_per_tensor": (f"torch dynamic quantization ({torch.backends.quantized.engine}), "
                                "default qconfig — int8 weights per tensor, int8 activations per batch"),
            "int8_per_channel": "as above with per_channel_dynamic_qconfig — one weight scale per row",
            WEIGHT_ONLY: ("weights rounded to per-channel symmetric int8 and back; activations fp32. "
                          "Quality of weight-only int8; runs fp32 kernels, so speed is not measured"),
        },
        "items": len(texts),
        "subset": None if not limit else f"first {len(texts) // 77} per class",
        "host": f"{platform.system()} {platform.machine()}, CPU, {torch.get_num_threads()} threads",
        "torch": torch.__version__,
        "variants": variants,
        "vs_fp32": {
            recipe: {
                "micro_accuracy": round(variants[recipe]["micro_accuracy"] - fp32["micro_accuracy"], 4),
                "macro_f1": round(variants[recipe]["macro_f1"] - fp32["macro_f1"], 4),
                "label_agreement": round(agreement(predictions["fp32"], predictions[recipe]), 4),
                "cpu_time_ratio": (None if variants[recipe]["seconds"] is None
                                   else round(variants[recipe]["seconds"] / fp32["seconds"], 2)),
                "size_ratio_fp32_over_int8": round(fp32["weights_bytes"]
                                                   / variants[recipe]["weights_bytes"], 2),
            }
            for recipe in RECIPES
        },
        "benchmark_check": {
            "vllm_a10_micro_accuracy": benchmark["micro_accuracy"],
            "cpu_fp32_micro_accuracy": fp32["micro_accuracy"],
            "comparable": limit is None,
            "note": "the same weights through vLLM on the A10; only a full 770-item run compares",
        },
        "not_reportable": "speed and size are this CPU's (PRD §7: latency only on a rented GPU)",
    }
    OUT_FILE.write_text(json.dumps(out, indent=2) + "\n")
    for recipe, d in out["vs_fp32"].items():
        ratio = "—" if d["cpu_time_ratio"] is None else f"x{d['cpu_time_ratio']}"
        print(f"  {recipe}: {d['micro_accuracy']:+.4f} accuracy vs fp32 · labels agree "
              f"{d['label_agreement']:.1%} · CPU time {ratio} · weights /{d['size_ratio_fp32_over_int8']}")
    print(f"  wrote {OUT_FILE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    raise SystemExit(main())
