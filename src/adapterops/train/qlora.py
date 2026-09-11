"""QLoRA training for the task adapters (F1-F4).

One entrypoint, task as an argument — BUILD-PLAN Phase 0. Reads the *frozen* splits
(`data/<task>/split_train.parquet`), never the raw mirror, so training can never
accidentally see a golden-set row.

CUDA-only imports (`bitsandbytes`) are deferred into `train()` so this module stays
importable on macOS for formatting and inspection.

**M11 side-effect, deliberate:** an early checkpoint is saved partway through training and
kept. That under-trained adapter is the subtle regression the Phase 5 gate-sensitivity test
needs (PRD M11). It costs nothing now and a full retrain in week 9.
"""

from __future__ import annotations

import inspect
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
SEED = 20260909

PROMPT = (
    "Classify the customer's banking request into one intent label.\n"
    "Request: {text}\n"
    "Intent:"
)

PROMPTS = {
    "intent": PROMPT,
    "urgency": (
        "Classify the urgency of this support ticket.\n"
        "Ticket: {text}\n"
        "Urgency:"
    ),
    "pii": (
        "List every piece of personal information in the text, one per line, "
        "as LABEL: value.\n"
        "Text: {text}\n"
        "Found:\n"
    ),
}

COLUMNS = {
    "intent": ("text", "label_text"),
    "urgency": ("text", "label"),
    "pii": ("source_text", "target"),
}


@dataclass
class TrainConfig:
    task: str = "intent"
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_seq_length: int = 128
    """A truncation cap only; padding is dynamic. Per-task values are in TASK_CONFIGS —
    PII sequences run 5x longer than intent's and would be cut mid-span at 128."""
    epochs: float = 2.0
    """2, not 3. Validation loss bottomed at epoch 2 in every intent run, under two
    different loss definitions, and epoch 3 overfitted each time."""
    train_subsample: int | None = None
    """Cap on training rows, applied after the frozen split.

    PII starts at 3,000 of its 17,000 — not to satisfy PRD §9, whose figure assumed
    free-tier-only training, but because 3,000 documents is already ~21,000 span examples
    over a fixed 19-label vocabulary, and it is unknown whether more helps. 3,000 costs
    30 free minutes and produces the number that decides. If the result sits well below
    the 0.9942 scorer ceiling AND train loss is still falling at the end, the task is
    data-limited and the full 17,000 is worth ~$1 of rented A10. If it lands near the
    ceiling, the extra rows would buy a longer run and nothing else."""
    batch_size: int = 8
    """Micro-batch. Peak memory is dominated by the loss over Qwen's 151,936-token
    vocabulary: logits are [batch x seq x 151936] and cross_entropy upcasts them to fp32,
    so batch 16 with a 110-token sequence peaks around 2.1 GB for the loss alone and OOMs
    a 16 GB T4. Effective batch stays 32 via grad_accum."""
    grad_accum: int = 4
    learning_rate: float = 2e-4
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    ])
    early_checkpoint_fraction: float = 0.15
    """Where to snapshot the under-trained adapter for M11. 15% of steps is deliberately
    too few to have converged — that is the point."""
    output_dir: str = "checkpoints/intent"
    hub_repo: str | None = None          # e.g. "tpawar03/adapterops-intent"


def label_column(task: str) -> str:
    return COLUMNS[task][1]


def text_column(task: str) -> str:
    return COLUMNS[task][0]


def load_split(task: str, split: str, subsample: int | None = None) -> pd.DataFrame:
    path = REPO_ROOT / f"data/{task}/split_{split}.parquet"
    if not path.exists():
        msg = f"{path} missing — run `uv run adapterops splits` first"
        raise FileNotFoundError(msg)
    df = pd.read_parquet(path)
    if subsample is not None and len(df) > subsample:
        df = df.sample(n=subsample, random_state=SEED).reset_index(drop=True)
    return df


TASK_CONFIGS: dict[str, dict] = {
    "intent": {"max_seq_length": 128, "batch_size": 8, "grad_accum": 4},
    "urgency": {"max_seq_length": 128, "batch_size": 8, "grad_accum": 4},
    "pii": {
        # Sequences run median 160 / p95 429 / max 802 tokens against intent's 34.
        # Peak loss memory is batch x seq x 151,936 vocab, upcast to fp32 with a gradient:
        # batch 4 at p95 is ~2.1 GB, which is what OOM'd the T4 on intent. Batch 2 is
        # ~1.0 GB. Effective batch stays 32.
        "max_seq_length": 512,
        "batch_size": 2,
        "grad_accum": 16,
        # Start here and let the measurement decide; see TrainConfig.train_subsample.
        "train_subsample": 3000,
    },
}


def config_for(task: str, **overrides) -> TrainConfig:
    """Per-task defaults, overridable. One entrypoint, four tasks."""
    base = {"task": task, "output_dir": f"checkpoints/{task}"}
    return TrainConfig(**{**base, **TASK_CONFIGS.get(task, {}), **overrides})


def format_examples(df: pd.DataFrame, task: str) -> list[dict[str, str]]:
    """One prompt/completion pair per row. The completion is the bare label, so scoring
    is an exact-match check rather than parsing free text."""
    tcol, lcol = text_column(task), label_column(task)
    return [
        {"prompt": PROMPTS[task].format(text=r[tcol]), "completion": " " + str(r[lcol])}
        for _, r in df.iterrows()
    ]


def train(cfg: TrainConfig) -> dict:
    from importlib.metadata import version

    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainerCallback,
        TrainingArguments,
    )

    if not torch.cuda.is_available():
        msg = "QLoRA needs CUDA — run this on Colab/Kaggle or the rented GPU, not locally"
        raise RuntimeError(msg)

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    tok.pad_token = tok.pad_token or tok.eos_token

    def encode(batch: dict) -> dict:
        """Tokenise prompt and completion separately so the prompt can be masked out of
        the loss with -100. Without this the model spends most of its capacity learning
        to reproduce the customer's request, and the reported loss is not comparable
        across prompt formats. Tokenising the two halves separately (rather than slicing
        a joined sequence by character length) keeps the boundary exact under BPE.

        No padding here — the collator pads each batch to its own longest sequence.
        """
        input_ids, labels, attention = [], [], []
        for prompt, completion in zip(batch["prompt"], batch["completion"], strict=True):
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            c_ids = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids = (p_ids + c_ids)[: cfg.max_seq_length]
            lab = ([-100] * len(p_ids) + c_ids)[: cfg.max_seq_length]
            input_ids.append(ids)
            labels.append(lab)
            attention.append([1] * len(ids))
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}

    train_ds = Dataset.from_list(
        format_examples(load_split(cfg.task, "train", cfg.train_subsample), cfg.task))
    val_ds = Dataset.from_list(format_examples(load_split(cfg.task, "val"), cfg.task))
    train_ds = train_ds.map(encode, batched=True, remove_columns=["prompt", "completion"])
    val_ds = val_ds.map(encode, batched=True, remove_columns=["prompt", "completion"])

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(
                torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
            ),
            bnb_4bit_use_double_quant=True,
        ),
        device_map="auto",
    )
    # Gradient checkpointing recomputes activations to save memory, at roughly a third of
    # throughput. At 1.5B in 4-bit with ~50-token sequences there is ample headroom on a
    # 16 GB T4, so it is off. If this OOMs, set use_gradient_checkpointing=True and/or
    # halve TrainConfig.batch_size.
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules, task_type="CAUSAL_LM", bias="none"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    out_dir = REPO_ROOT / cfg.output_dir
    early_dir = out_dir.parent / f"{Path(cfg.output_dir).name}-undertrained-m11"

    class EarlyCheckpoint(TrainerCallback):
        """Snapshot an under-trained adapter for the M11 gate-sensitivity test."""

        def __init__(self) -> None:
            self.saved = False

        def on_step_end(self, args, state, control, **kw):
            if self.saved or not state.max_steps:
                return control
            if state.global_step >= max(1, int(state.max_steps * cfg.early_checkpoint_fraction)):
                kw["model"].save_pretrained(early_dir)
                self.saved = True
                print(f"  [M11] under-trained checkpoint saved at step {state.global_step} -> {early_dir}")
            return control

    # transformers 5 removed `warmup_ratio` and kept `warmup_steps`, so compute the steps
    # ourselves — `warmup_steps` exists in both 4.x and 5.x.
    # T4 (Turing, SM 7.5) has no bf16 tensor cores; asking for bf16 there runs an
    # emulated path several times slower than fp16. Do NOT use
    # torch.cuda.is_bf16_supported() — its `including_emulation` argument defaults to
    # True, so it answers True on Turing. Real bf16 starts at Ampere (SM 8.0).
    bf16_ok = torch.cuda.get_device_capability(0)[0] >= 8
    precision = {"bf16": True} if bf16_ok else {"fp16": True}
    print(f"  precision: {'bf16' if bf16_ok else 'fp16'} "
          f"(device {torch.cuda.get_device_name(0)}, "
          f"capability {torch.cuda.get_device_capability(0)})")

    steps_per_epoch = math.ceil(len(train_ds) / (cfg.batch_size * cfg.grad_accum))
    total_steps = int(steps_per_epoch * cfg.epochs)
    ta_kwargs = {
        "output_dir": str(out_dir),
        "num_train_epochs": cfg.epochs,
        "per_device_train_batch_size": cfg.batch_size,
        "per_device_eval_batch_size": cfg.batch_size,
        "gradient_accumulation_steps": cfg.grad_accum,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": "cosine",
        "warmup_steps": max(10, int(0.03 * total_steps)),
        "logging_steps": 10,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,   # >=2 so the best checkpoint survives pruning
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        **precision,
        "optim": "paged_adamw_8bit",
        "report_to": [],
        "seed": SEED,
    }
    # Fail with something actionable rather than a bare TypeError halfway through a
    # rented GPU session, the way `warmup_ratio` did.
    supported = set(inspect.signature(TrainingArguments.__init__).parameters)
    unsupported = sorted(set(ta_kwargs) - supported)
    if unsupported:
        msg = (f"TrainingArguments in transformers {version('transformers')} does not accept "
               f"{unsupported}. Update train/qlora.py for this version.")
        raise TypeError(msg)
    args = TrainingArguments(**ta_kwargs)
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        # Seq2Seq, not ForLanguageModeling: the LM collator rebuilds labels from
        # input_ids and would discard the prompt masking above.
        data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100),
        callbacks=[EarlyCheckpoint()],
    )
    result = trainer.train()
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    if early_dir.is_dir():
        tok.save_pretrained(early_dir)

    summary = {
        "task": cfg.task,
        "base_model": cfg.base_model,
        "train_rows": len(train_ds),
        "val_rows": len(val_ds),
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "train_loss": round(float(result.training_loss), 4),
        "adapter_dir": str(out_dir.relative_to(REPO_ROOT)),
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_eval_loss": trainer.state.best_metric,
        "prompt_tokens_masked": True,
        "m11_undertrained_dir": str(early_dir.relative_to(REPO_ROOT)),
        "seed": SEED,
    }
    (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    # Push here, not in a later cell. A Colab VM can be recycled between cells; this
    # adapter has been lost twice that way, once with the whole clone. Uploading inside
    # train() means the weights are durable the moment they exist, and a failure here is
    # loud rather than discovered 20 minutes later.
    if cfg.hub_repo:
        from huggingface_hub import HfApi

        api = HfApi()
        pushed = {}
        for local, repo in (
            (out_dir, cfg.hub_repo),
            (early_dir, f"{cfg.hub_repo}-undertrained-m11"),
        ):
            if not local.is_dir():
                print(f"  WARNING: {local} missing, not pushed")
                continue
            api.create_repo(repo, exist_ok=True)
            api.upload_folder(folder_path=str(local), repo_id=repo)
            pushed[repo] = str(local)
            print(f"  pushed {repo}")
        summary["hub_repos"] = pushed

    return summary
