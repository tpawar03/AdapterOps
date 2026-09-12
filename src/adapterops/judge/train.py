"""Train the distilled drafting judge (F14) — DeBERTa-v3-base regressing GPT-4o's 1–5 grade.

The base is DeBERTa-v3-base by §15's rule, measured rather than assumed (D35): it trains
faster than the alternative that could actually run on the available compute.

**The calibration holdout never touches training — not even checkpoint selection.**
`load_best_model_at_end` picks a checkpoint by evaluation loss, and whichever split that
loss is computed on has, in effect, been trained against. Selecting on the 150 calibration
items would make M5's correlation a number the model was tuned to produce. So a separate
validation split is carved from the training items, and the calibration set is scored
exactly once, after training ends.

**Only the adapter's replies train it.** GPT-4o-mini's replies are graded for the fair
escalation comparison and the §10 same-family check; letting them into training would change
the distribution the judge is calibrated on and blur the one caveat it exists to measure.

The same guards as the router, for the same reasons: weights are loaded in fp32 explicitly
and asserted, CUDA is used when present and CPU otherwise (MPS skipped), checkpoints keep
weights only, and intermediate checkpoints are removed once the best model is written.

    uv run adapterops judge-train
"""

from __future__ import annotations

import inspect
import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from adapterops.judge.calibrate import agreement

REPO_ROOT = Path(__file__).resolve().parents[3]
JUDGE_DIR = REPO_ROOT / "data" / "judge"
LABELS_FILE = JUDGE_DIR / "judgments.parquet"
RUN_FILE = REPO_ROOT / "runs" / "judge__train.json"
OUT_DIR = REPO_ROOT / "checkpoints" / "judge"

SEED = 20260909

SCORE_CENTER = 3.0
SCORE_SCALE = 2.0
"""Grades are trained as (score - 3) / 2, a [-1, 1] target, and mapped back afterwards.

Found by a smoke run, not by design. A regression head initialises near zero while grades
run 1-5, so training starts with a mean squared error around 9 and spends its first steps
learning the average grade. On the smoke run that was *all* it learned: calibration
Spearman came out -0.45 with a bias of -2.38, and a pass criterion that only checked a
correlation existed reported it as a pass. Centring the target removes the offset the head
would otherwise have to discover."""


def to_target(score):
    return (score - SCORE_CENTER) / SCORE_SCALE


def from_target(prediction):
    return prediction * SCORE_SCALE + SCORE_CENTER


@dataclass
class JudgeConfig:
    base_model: str = "microsoft/deberta-v3-base"
    max_length: int = 512
    """Covers every judge input measured: instruction plus reply runs p99 380 / max 411
    DeBERTa tokens, so nothing is truncated."""
    epochs: float = 3.0
    batch_size: int = 8
    grad_accum: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    val_fraction: float = 0.1
    min_fit: int = 200
    """Refuse to train on fewer labelled items than this — a half-finished grading run
    would otherwise produce a judge and a calibration number that look complete."""
    seed: int = SEED
    use_cpu: bool | None = None
    tag: str = ""
    output_dir: str = str(OUT_DIR)


def to_text(frame: pd.DataFrame) -> list[str]:
    """The judge sees exactly what the GPT-4o teacher saw: request and reply, no reference."""
    return [f"REQUEST:\n{i}\n\nREPLY:\n{r}"
            for i, r in zip(frame.instruction, frame.reply, strict=True)]


def load_labelled(path: Path | None = None) -> pd.DataFrame:
    return pd.read_parquet(path or LABELS_FILE)


def split_for_training(labelled: pd.DataFrame, val_fraction: float = 0.1,
                       seed: int = SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(fit, validation, calibration). Validation is carved from training items only."""
    local = labelled[(labelled.source == "local") & labelled.parsed_ok.astype(bool)]
    calibration = local[local.split == "calibration"]
    pool = local[local.split == "train"]
    validation = pool.sample(frac=val_fraction, random_state=seed)
    fit = pool.drop(index=validation.index)

    ids = {n: set(f.item_id) for n, f in (("fit", fit), ("val", validation),
                                           ("calibration", calibration))}
    overlap = (ids["fit"] & ids["val"]) | (ids["fit"] & ids["calibration"]) \
        | (ids["val"] & ids["calibration"])
    if overlap:
        msg = f"judge splits overlap on {len(overlap)} items"
        raise RuntimeError(msg)
    return fit, validation, calibration


def train(cfg: JudgeConfig | None = None) -> dict:
    cfg = cfg or JudgeConfig()
    fit, validation, calibration = split_for_training(
        load_labelled(LABELS_FILE), cfg.val_fraction, cfg.seed)
    if len(fit) < cfg.min_fit:
        msg = (f"only {len(fit)} labelled training items (need {cfg.min_fit}). Finish the "
               f"grading run before training, or the judge and its calibration number will "
               f"look complete and not be.")
        raise ValueError(msg)
    if len(calibration) < 3:
        msg = f"only {len(calibration)} graded calibration items — M5 cannot be computed"
        raise ValueError(msg)

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tok = AutoTokenizer.from_pretrained(cfg.base_model)

    def encode(frame: pd.DataFrame) -> Dataset:
        ds = Dataset.from_dict({"text": to_text(frame),
                                "labels": to_target(frame.score.astype(float)).tolist()})
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=cfg.max_length),
                      batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=1, dtype=torch.float32)
    non_fp32 = {str(p.dtype) for p in model.parameters() if p.dtype != torch.float32}
    if non_fp32:
        msg = f"judge weights must be fp32 to train with AdamW — got {non_fp32}"
        raise TypeError(msg)

    use_cpu = (not torch.cuda.is_available()) if cfg.use_cpu is None else cfg.use_cpu
    total_steps = math.ceil(len(fit) / (cfg.batch_size * cfg.grad_accum)) * cfg.epochs
    ta_kwargs = {
        "output_dir": cfg.output_dir,
        "num_train_epochs": cfg.epochs,
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "warmup_steps": max(10, int(cfg.warmup_fraction * total_steps)),
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": 2,
        "save_only_model": True,
        "seed": cfg.seed,
        "report_to": [],
        "logging_steps": 25,
        "use_cpu": use_cpu,
    }
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    unsupported = sorted(set(ta_kwargs) - supported)
    if unsupported:
        msg = f"TrainingArguments does not accept {unsupported} in this transformers version"
        raise TypeError(msg)

    trainer = Trainer(model=model, args=TrainingArguments(**ta_kwargs),
                      train_dataset=encode(fit), eval_dataset=encode(validation),
                      processing_class=tok)
    trainer.train()

    predicted = from_target(
        np.asarray(trainer.predict(encode(calibration)).predictions).reshape(-1))
    if not np.isfinite(predicted).all():
        msg = "judge produced non-finite scores on the calibration set"
        raise ValueError(msg)

    suffix = f"__{cfg.tag}" if cfg.tag else ""
    JUDGE_DIR.mkdir(parents=True, exist_ok=True)
    calibration.assign(judge_score=predicted).to_parquet(
        JUDGE_DIR / f"calibration_scored{suffix}.parquet")

    summary = {
        "config": asdict(cfg),
        "device": "cpu" if use_cpu else "cuda",
        "rows": {"fit": len(fit), "validation": len(validation),
                 "calibration": len(calibration)},
        "selection": "checkpoint chosen on a validation split carved from training items; "
                     "the calibration holdout is scored once, after training",
        "target_scaling": "trained on (score - 3) / 2; predictions mapped back to the 1-5 scale",
        "calibration": agreement(calibration.score.astype(float).to_numpy(), predicted),
    }

    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    for intermediate in Path(cfg.output_dir).glob("checkpoint-*"):
        shutil.rmtree(intermediate, ignore_errors=True)

    run_file = RUN_FILE if not cfg.tag else RUN_FILE.with_name(f"judge__train__{cfg.tag}.json")
    run_file.parent.mkdir(parents=True, exist_ok=True)
    run_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    if not LABELS_FILE.exists():
        print("  no data/judge/judgments.parquet — run `uv run adapterops judge-label` first.")
        return 2
    summary = train()
    print(json.dumps(summary["calibration"], indent=2))
    return 0
