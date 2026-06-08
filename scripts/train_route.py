#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from healthqa_ft.config import load_yaml, route_config
from healthqa_ft.data import CausalCollator, CausalQADataset, Seq2SeqQADataset, load_subset_csv
from healthqa_ft.utils import free_memory, is_oom_error, latest_checkpoint, log, seed_everything, set_hf_runtime_env, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one routed Health QA LoRA adapter")
    parser.add_argument("--config", default=str(ROOT / "configs" / "routes.yaml"))
    parser.add_argument("--route", required=True, help="Route key, e.g. amh, swa, lug, aka, eng")
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--max_train_rows", type=int, default=None)
    parser.add_argument("--max_eval_rows", type=int, default=None)
    return parser.parse_args()


def quantization_config(train_cfg: dict[str, Any]) -> BitsAndBytesConfig | None:
    if not train_cfg.get("load_in_4bit", True):
        return None
    dtype_name = str(train_cfg.get("bnb_4bit_compute_dtype", "float16"))
    dtype = torch.bfloat16 if dtype_name == "bfloat16" else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=train_cfg.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=bool(train_cfg.get("bnb_4bit_use_double_quant", True)),
    )


def load_tokenizer(route: dict[str, Any]):
    tokenizer = AutoTokenizer.from_pretrained(
        route["model_name"],
        trust_remote_code=bool(route.get("trust_remote_code", False)),
        use_fast=True,
    )
    if route["architecture"] == "causal":
        tokenizer.padding_side = "right"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_base_model(route: dict[str, Any], train_cfg: dict[str, Any]):
    kwargs = {
        "device_map": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": bool(route.get("trust_remote_code", False)),
    }
    qcfg = quantization_config(train_cfg)
    if qcfg is not None:
        kwargs["quantization_config"] = qcfg
    else:
        kwargs["torch_dtype"] = torch.bfloat16 if train_cfg.get("bf16") else torch.float16

    if route["architecture"] == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(route["model_name"], **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(route["model_name"], **kwargs)

    if train_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    if train_cfg.get("load_in_4bit", True):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=train_cfg.get("gradient_checkpointing", True))
    return model


def attach_lora(model, route: dict[str, Any]):
    lora_cfg = route["lora"]
    task_type = TaskType.SEQ_2_SEQ_LM if route["architecture"] == "seq2seq" else TaskType.CAUSAL_LM
    peft_cfg = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=list(lora_cfg.get("target_modules", [])),
        bias="none",
        task_type=task_type,
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()
    return model


def build_datasets(cfg: dict[str, Any], route: dict[str, Any], tokenizer, train_csv: str, val_csv: str | None, train_cfg: dict[str, Any]):
    data_cfg = cfg.get("data", {})
    q_col = data_cfg.get("question_column", "input")
    a_col = data_cfg.get("answer_column", "output")
    s_col = data_cfg.get("subset_column", "subset")
    id_col = data_cfg.get("id_column", "ID")
    max_train_rows = train_cfg.get("max_train_rows")
    max_eval_rows = train_cfg.get("max_eval_rows")

    train_df = load_subset_csv(
        train_csv,
        route["subsets"],
        question_col=q_col,
        answer_col=a_col,
        subset_col=s_col,
        id_col=id_col,
        max_rows=max_train_rows,
        seed=int(cfg.get("seed", 42)),
    )
    eval_df = None
    if val_csv and Path(val_csv).exists():
        eval_df = load_subset_csv(
            val_csv,
            route["subsets"],
            question_col=q_col,
            answer_col=a_col,
            subset_col=s_col,
            id_col=id_col,
            max_rows=max_eval_rows,
            seed=int(cfg.get("seed", 42)),
        )

    if route["architecture"] == "seq2seq":
        train_ds = Seq2SeqQADataset(train_df, tokenizer, route["language_name"], q_col, a_col, int(train_cfg["max_seq_length"]))
        eval_ds = Seq2SeqQADataset(eval_df, tokenizer, route["language_name"], q_col, a_col, int(train_cfg["max_seq_length"])) if eval_df is not None and len(eval_df) else None
        collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=None, label_pad_token_id=-100)
    else:
        train_ds = CausalQADataset(
            train_df,
            tokenizer,
            route["language_name"],
            q_col,
            a_col,
            int(train_cfg["max_seq_length"]),
            bool(train_cfg.get("answer_only_loss", True)),
        )
        eval_ds = CausalQADataset(
            eval_df,
            tokenizer,
            route["language_name"],
            q_col,
            a_col,
            int(train_cfg["max_seq_length"]),
            bool(train_cfg.get("answer_only_loss", True)),
        ) if eval_df is not None and len(eval_df) else None
        collator = CausalCollator(tokenizer)
    return train_ds, eval_ds, collator


def trainer_args(route_dir: Path, train_cfg: dict[str, Any], has_eval: bool) -> TrainingArguments:
    eval_strategy = train_cfg.get("evaluation_strategy", "epoch") if has_eval else "no"
    return TrainingArguments(
        output_dir=str(route_dir / "checkpoints"),
        overwrite_output_dir=False,
        num_train_epochs=float(train_cfg.get("num_train_epochs", 3)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        per_device_eval_batch_size=int(train_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 16)),
        learning_rate=float(train_cfg.get("learning_rate", 1.5e-4)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        optim=train_cfg.get("optim", "paged_adamw_8bit"),
        logging_steps=int(train_cfg.get("logging_steps", 20)),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        eval_strategy=eval_strategy,
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.3)),
        fp16=bool(train_cfg.get("fp16", True)),
        bf16=bool(train_cfg.get("bf16", False)),
        report_to=[],
        remove_unused_columns=False,
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        dataloader_num_workers=2,
    )


def train_once(cfg: dict[str, Any], route: dict[str, Any], train_csv: str, val_csv: str | None, output_root: str, resume: bool) -> dict[str, Any]:
    route_dir = Path(output_root) / route["route_name"]
    route_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = route["training"]

    log(f"Route {route['route_name']}: {route['display_name']}")
    log(f"Model: {route['model_name']}")
    log(f"Subsets: {route['subsets']}")
    log(f"Training config: batch={train_cfg['per_device_train_batch_size']} grad_acc={train_cfg['gradient_accumulation_steps']} max_seq={train_cfg['max_seq_length']}")

    tokenizer = load_tokenizer(route)
    model = load_base_model(route, train_cfg)
    model = attach_lora(model, route)
    train_ds, eval_ds, collator = build_datasets(cfg, route, tokenizer, train_csv, val_csv, train_cfg)

    args = trainer_args(route_dir, train_cfg, has_eval=eval_ds is not None)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds, data_collator=collator, tokenizer=tokenizer)
    resume_checkpoint = latest_checkpoint(args.output_dir) if resume else None
    if resume_checkpoint:
        log(f"Resuming from checkpoint: {resume_checkpoint}")
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)

    final_dir = route_dir / "final_adapter"
    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    status = {
        "route": route["route_name"],
        "model_name": route["model_name"],
        "subsets": route["subsets"],
        "final_adapter": str(final_dir),
        "train_metrics": result.metrics,
        "training_config": train_cfg,
    }
    write_json(route_dir / "train_status.json", status)
    return status


def train_with_oom_retries(cfg: dict[str, Any], base_route: dict[str, Any], train_csv: str, val_csv: str | None, output_root: str, resume: bool) -> dict[str, Any]:
    retry_cfg = cfg.get("oom_retry", {})
    attempts = retry_cfg.get("attempts") if retry_cfg.get("enabled", True) else [{}]
    attempts = attempts or [{}]
    errors = []
    for idx, override in enumerate(attempts, start=1):
        route = dict(base_route)
        route["training"] = dict(base_route["training"])
        route["training"].update(override or {})
        try:
            log(f"OOM retry attempt {idx}/{len(attempts)}")
            return train_once(cfg, route, train_csv, val_csv, output_root, resume)
        except Exception as exc:
            free_memory()
            if not is_oom_error(exc):
                raise
            errors.append(str(exc))
            log(f"OOM on attempt {idx}; trying smaller memory profile.")
    raise RuntimeError("All OOM retry attempts failed:\n" + "\n".join(errors))


def main() -> None:
    args = parse_args()
    set_hf_runtime_env()
    cfg = load_yaml(args.config)
    seed_everything(int(cfg.get("seed", 42)))
    route = route_config(cfg, args.route)
    if args.max_train_rows is not None:
        route["training"]["max_train_rows"] = args.max_train_rows
    if args.max_eval_rows is not None:
        route["training"]["max_eval_rows"] = args.max_eval_rows

    data_cfg = cfg.get("data", {})
    train_csv = args.train_csv or data_cfg.get("train_csv")
    val_csv = args.val_csv or data_cfg.get("val_csv")
    output_root = args.output_root or data_cfg.get("output_root")
    if not train_csv:
        raise ValueError("train_csv is required")
    if not output_root:
        raise ValueError("output_root is required")

    status = train_with_oom_retries(cfg, route, train_csv, val_csv, output_root, resume=not args.no_resume)
    log(f"Done: {status['final_adapter']}")


if __name__ == "__main__":
    main()

