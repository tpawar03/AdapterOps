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

**Loaded in fp32, explicitly.** `microsoft/deberta-v3-small` ships fp16 weights, and
transformers 5 honours a checkpoint's dtype instead of upcasting the way earlier versions
did. Training the result with AdamW diverges on the *first* optimizer step: gradients are
finite, but `exp_avg_sq = (1-β₂)·g²` underflows fp16 and the resulting division overflows
it, so 105 of ~200 parameters come back non-finite and every subsequent loss is NaN. The
symptom is a run that completes normally and reports `nan`. Asserted below rather than
trusted, because the next transformers release could flip the default back.

**The floor to beat is not 50%.** Predicting "always succeeds" scores whatever the base
success rate is — around 0.93 on intent — so accuracy is close to useless here. What is
reported is the ranking quality (ROC-AUC and average precision on the failure class),
because ranking is what the budget sweep consumes.

    uv run adapterops router-train
"""

from __future__ import annotations

import inspect
import json
import math
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
    batch_size: int = 8
    """Micro-batch. 16 at `max_length` 256 OOM'd the laptop's MPS allocator during a full
    rehearsal — DeBERTa's disentangled attention holds two extra attention matrices per
    layer, so its activation memory is well above a same-sized BERT. Effective batch stays
    16 via accumulation."""
    grad_accum: int = 2
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_fraction: float = 0.1
    """Converted to `warmup_steps` below. `warmup_ratio` was removed from
    TrainingArguments in transformers 5 — qlora.py already hit this and the guard it
    added is reused here, because the failure mode is a TypeError partway into a run."""
    seed: int = SEED
    include_task: bool = True
    """Prepend `task: <name>` to the text (F10). Set False for the shortcut ablation.

    Task identity is a *shortcut feature*: the four tasks fail at 10%, 27%, 49% and 49%,
    so a model that reads only the task name already ranks failures well across the pooled
    set. The first trained router scored 0.7064 ROC-AUC and correlated **0.996** with a
    pure task-prior lookup scoring 0.6883 — it took the shortcut. This flag is how that is
    demonstrated rather than asserted."""
    tag: str = ""
    """Names this variant's outputs. Empty is the default run.

    An ablation once wrote its predictions over the default run's, because the experiment
    redirected the run file but not the data directory — the two variants shared
    `router_eval_scored.parquet` and the second silently won. Tagging makes collisions
    impossible rather than careful."""
    use_cpu: bool | None = None
    """None selects: CUDA when present, otherwise CPU. **MPS is skipped deliberately.**

    A full rehearsal OOM'd the MPS allocator twice on this 24 GB machine — at micro-batch
    16 and again at 8 — with ~12 GB of system memory free. MPS shares unified memory and
    its watermark is above physical RAM, so "other allocations: 23.99 GiB" is the machine,
    not this process. DeBERTa's disentangled attention carries two extra attention matrices
    per layer, which makes it a bad fit for a memory-constrained shared allocator.

    CPU costs minutes for a 141M model over ~200 steps, is deterministic, and matches the
    deployment target: PRD §7 budgets the router at <50 ms on CPU. Set explicitly to
    override."""
    output_dir: str = str(OUT_DIR)


def to_text(frame: pd.DataFrame, include_task: bool = True) -> list[str]:
    """`task: <name> | <ticket text>` — the task value seen as text, like everything else."""
    if not include_task:
        return frame.text.astype(str).tolist()
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


def shortcut_check(train_frame: pd.DataFrame, eval_frame: pd.DataFrame,
                   summary: dict) -> dict:
    """How much of the router's ranking is explained by the task name alone.

    The first trained router scored 0.7064 ROC-AUC and looked like a working component.
    A lookup table holding nothing but each task's failure rate scored 0.6883, and the
    router's predictions correlated 0.996 with it. Reported on every run from now on,
    because the pooled AUC cannot distinguish the two on its own.
    """
    from sklearn.metrics import roc_auc_score

    prior = 1 - train_frame.groupby("task").success.mean()
    y = (~eval_frame.success.astype(bool)).astype(int)
    task_only = eval_frame.task.map(prior).to_numpy()
    return {
        "task_prior_only_roc_auc": round(float(roc_auc_score(y, task_only)), 4),
        "router_roc_auc": summary["splits"]["eval"]["overall"]["roc_auc"],
        "per_task_failure_rates": {k: round(float(v), 4) for k, v in prior.items()},
        "note": "if these two are close, the pooled AUC is the task prior, not the text",
    }


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
    use_cpu = (not torch.cuda.is_available()) if cfg.use_cpu is None else cfg.use_cpu
    splits = {name: pd.read_parquet(DATA_DIR / f"router_{name}.parquet")
              for name in ("train", "eval", "shift_eval")}

    tok = AutoTokenizer.from_pretrained(cfg.base_model)

    def encode(frame: pd.DataFrame) -> Dataset:
        ds = Dataset.from_dict({
            "text": to_text(frame, cfg.include_task),
            # label 1 = the adapter failed = escalate. Framing the positive class as the
            # failure keeps average precision reading as "how well are failures ranked".
            "labels": (~frame.success.astype(bool)).astype(int).tolist(),
        })
        return ds.map(lambda b: tok(b["text"], truncation=True, max_length=cfg.max_length),
                      batched=True, remove_columns=["text"])

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model, num_labels=2, dtype=torch.float32)
    non_fp32 = {n: str(p.dtype) for n, p in model.named_parameters()
                if p.dtype != torch.float32}
    if non_fp32:
        msg = (f"router weights must be fp32 to train with AdamW — got {set(non_fp32.values())}. "
               f"In fp16 the first optimizer step returns NaN for most parameters and the "
               f"run completes reporting nan.")
        raise TypeError(msg)

    total_steps = math.ceil(
        len(splits["train"]) / (cfg.batch_size * cfg.grad_accum)) * cfg.epochs
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
        "save_total_limit": 2,   # >=2 so the best checkpoint survives pruning
        "seed": cfg.seed,
        "report_to": [],
        "logging_steps": 25,
        "use_cpu": use_cpu,
    }
    # Same guard as qlora.py, for the same reason: surface an unsupported argument as a
    # readable error rather than a TypeError partway into training.
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    unsupported = sorted(set(ta_kwargs) - supported)
    if unsupported:
        msg = (f"TrainingArguments does not accept {unsupported} in this transformers "
               f"version. Update router/train.py.")
        raise TypeError(msg)
    args = TrainingArguments(**ta_kwargs)
    trainer = Trainer(model=model, args=args,
                      train_dataset=encode(splits["train"]),
                      eval_dataset=encode(splits["eval"]),
                      processing_class=tok)
    trainer.train()

    summary: dict = {"config": asdict(cfg), "device": "cpu" if use_cpu else "cuda",
                     "splits": {}}
    for name in ("eval", "shift_eval"):
        frame = splits[name]
        logits = trainer.predict(encode(frame)).predictions
        p_fail = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
        if not np.isfinite(p_fail).all():
            msg = ("router produced non-finite probabilities — check the fp32 guard above; "
                   "in fp16 AdamW returns NaN from the first step and the run still "
                   "completes")
            raise ValueError(msg)
        frame = frame.assign(router_p_fail=p_fail)
        suffix = f"__{cfg.tag}" if cfg.tag else ""
        frame.to_parquet(DATA_DIR / f"router_{name}_scored{suffix}.parquet")
        summary["splits"][name] = ranking_report(
            (~frame.success.astype(bool)).to_numpy(), p_fail, frame.task.to_numpy())

    summary["shortcut_check"] = shortcut_check(splits["train"], splits["eval"], summary)
    trainer.save_model(cfg.output_dir)
    tok.save_pretrained(cfg.output_dir)
    run_file = (RUN_FILE if not cfg.tag
                else RUN_FILE.with_name(f"router__train__{cfg.tag}.json"))
    run_file.parent.mkdir(exist_ok=True)
    run_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    if not (DATA_DIR / "router_train.parquet").exists():
        print("  no router dataset — run `uv run adapterops router-dataset` first,")
        print("  which needs the scored pool from a GPU session.")
        return 2
    summary = train()
    print(json.dumps(summary["splits"], indent=2))
    return 0
