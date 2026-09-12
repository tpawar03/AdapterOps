"""Train the router (F10) — DeBERTa-v3-small over (ticket, task) pairs.

The router predicts one thing: **will the local adapter fail on this pair?** Its output is
a probability, and `baselines.py` turns that into a policy by escalating the highest-scoring
pairs up to a budget. Nothing here picks a threshold; that decision belongs to the operating
curve, where it can be read against the alternatives.

**Task identity is an input feature, not a separate model per task** (F10, D2). Four
per-task models would be the obvious alternative and would answer a different question —
the router exists because a single deployed component has to handle every task, and whether
one model can carry all four is precisely what is being measured. It is also the design that
makes the shift test valid (D11): a router that has never seen `task=drafting` cannot be
evaluated on drafting under shift.

The task is prepended as text rather than encoded as an embedding, so the encoder sees it
the same way it sees everything else and no new parameters are introduced for four values.

**The floor to beat is not 50%.** Predicting "always succeeds" scores whatever the base
success rate is — around 0.93 on intent — so accuracy is close to useless here. What is
reported is the ranking quality (ROC-AUC and average precision on the failure class),
because ranking is what the budget sweep consumes.

    uv run adapterops router-train
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data" / "router"
OUT_DIR = REPO_ROOT / "checkpoints" / "router"
RUN_FILE = REPO_ROOT / "runs" / "router__train.json"

SEED = 20260909


@dataclass
class RouterConfig:
    base_model: str = "microsoft/deberta-v3-small"
    max_length: int = 256
    epochs: float = 3.0
    batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    seed: int = SEED
    output_dir: str = str(OUT_DIR)


def to_text(frame: pd.DataFrame) -> list[str]:
    """`task: <name> | <ticket text>` — the task value seen as text, like everything else."""
    return [f"task: {t} | {x}" for t, x in zip(frame.task, frame.text, strict=True)]


def ranking_report(y_true: np.ndarray, p_fail: np.ndarray, tasks: np.ndarray) -> dict:
    """Ranking quality, overall and per task. Accuracy is deliberately absent.

    With intent failing ~7% of the time, "always succeeds" is 93% accurate and useless as a
    router. The budget sweep consumes an ordering, so an ordering is what gets scored.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    def one(mask: np.ndarray) -> dict:
        y, p = y_true[mask], p_fail[mask]
        if len(set(y)) < 2:
            return {"n": int(mask.sum()), "failure_rate": round(float(y.mean()), 4),
                    "roc_auc": None, "average_precision": None,
                    "note": "one class only — ranking is undefined"}
        return {
            "n": int(mask.sum()),
            "failure_rate": round(float(y.mean()), 4),
            "roc_auc": round(float(roc_auc_score(y, p)), 4),
            "average_precision": round(float(average_precision_score(y, p)), 4),
            "baseline_average_precision": round(float(y.mean()), 4),
        }

    report = {"overall": one(np.ones(len(y_true), dtype=bool))}
    report["per_task"] = {t: one(tasks == t) for t in sorted(set(tasks))}
    return report


def train(cfg: RouterConfig | None = None) -> dict:
    """Fine-tune the encoder. Imports are deferred so this module stays importable on CPU."""
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    cfg = cfg or RouterConfig()
    splits = {name: pd.read_parquet(DATA_DIR / f"router_{name}.parquet")
              for name in ("train", "eval", "shift_eval")}

    tok = AutoTokenizer.from_pretrained(cfg.base_model)

    def encode(frame: pd.DataFrame) -> Dataset:
        ds = Dataset.from_dict({
            "text": to_text(frame),
            # label 1 = the adapter failed = escalate. Framing the positive class as the
            # failure keeps average precision reading as "how well are failures ranked".
            "labels": (~frame.success.astype(bool)).astype(int).tolist(),
        })
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=cfg.max_length),
                      batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(cfg.base_model, num_labels=2)
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size * 2,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_total_limit=1,
        seed=cfg.seed,
        report_to=[],
        logging_steps=25,
    )
    trainer = Trainer(model=model, args=args,
                      train_dataset=encode(splits["train"]),
                      eval_dataset=encode(splits["eval"]),
                      processing_class=tok)
    trainer.train()

    summary: dict = {"config": asdict(cfg), "splits": {}}
    for name in ("eval", "shift_eval"):
        frame = splits[name]
        logits = trainer.predict(encode(frame)).predictions
        p_fail = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        frame = frame.assign(router_p_fail=p_fail)
        frame.to_parquet(DATA_DIR / f"router_{name}_scored.parquet")
        summary["splits"][name] = ranking_report(
            (~frame.success.astype(bool)).to_numpy(), p_fail, frame.task.to_numpy())

    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    RUN_FILE.parent.mkdir(exist_ok=True)
    RUN_FILE.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    if not (DATA_DIR / "router_train.parquet").exists():
        print("  no router dataset — run `uv run adapterops router-dataset` first,")
        print("  which needs the scored pool from a GPU session.")
        return 2
    summary = train()
    print(json.dumps(summary["splits"], indent=2))
    return 0
