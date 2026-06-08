#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from healthqa_ft.config import load_yaml, route_config, routes_to_train
from healthqa_ft.prompts import plain_causal_prompt, seq2seq_input
from healthqa_ft.utils import clean_text, free_memory, log, set_hf_runtime_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate routed adapters with ROUGE")
    parser.add_argument("--config", default=str(ROOT / "configs" / "routes.yaml"))
    parser.add_argument("--routes", nargs="*", default=None)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--run_root", default=None)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--max_rows_per_route", type=int, default=None)
    return parser.parse_args()


def qconfig() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def load_model_and_tokenizer(route: dict[str, Any], adapter_dir: Path):
    tokenizer = AutoTokenizer.from_pretrained(adapter_dir, trust_remote_code=bool(route.get("trust_remote_code", False)))
    if route["architecture"] == "causal":
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            route["model_name"],
            device_map="auto",
            quantization_config=qconfig(),
            low_cpu_mem_usage=True,
            trust_remote_code=bool(route.get("trust_remote_code", False)),
        )
    else:
        base = AutoModelForSeq2SeqLM.from_pretrained(
            route["model_name"],
            device_map="auto",
            quantization_config=qconfig(),
            low_cpu_mem_usage=True,
            trust_remote_code=bool(route.get("trust_remote_code", False)),
        )
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tokenizer


def render_prompt(route: dict[str, Any], tokenizer, question: str) -> str:
    lang = route["language_name"]
    if route["architecture"] == "seq2seq":
        return seq2seq_input(question, lang)
    messages = [
        {"role": "system", "content": f"You are a careful health question-answering assistant. Answer only in {lang}."},
        {"role": "user", "content": f"Answer this health question in {lang}:\n{question}"},
    ]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    except Exception:
        pass
    return plain_causal_prompt(question, lang)


@torch.inference_mode()
def generate_answer(route: dict[str, Any], model, tokenizer, question: str, gen_cfg: dict[str, Any]) -> str:
    prompt = render_prompt(route, tokenizer, question)
    enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    num_beams = int(gen_cfg.get("num_beams_seq2seq" if route["architecture"] == "seq2seq" else "num_beams_causal", 1))
    kwargs = {
        "max_new_tokens": int(gen_cfg.get("max_new_tokens", 256)),
        "do_sample": bool(gen_cfg.get("do_sample", False)),
        "num_beams": num_beams,
        "repetition_penalty": float(gen_cfg.get("repetition_penalty", 1.05)),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if kwargs["do_sample"]:
        kwargs["temperature"] = float(gen_cfg.get("temperature", 0.7))
        kwargs["top_p"] = float(gen_cfg.get("top_p", 0.9))
    gen = model.generate(**enc, **kwargs)
    if route["architecture"] == "causal":
        new_tokens = gen[0][enc["input_ids"].shape[1] :]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    else:
        text = tokenizer.decode(gen[0], skip_special_tokens=True)
    return clean_text(text)


def main() -> None:
    set_hf_runtime_env()
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    val_csv = args.val_csv or data_cfg.get("val_csv")
    run_root = Path(args.run_root or data_cfg.get("output_root") or "/workspace/healthqa_multimodel_runs")
    out_csv = Path(args.out_csv or (run_root / "rouge_by_route.csv"))
    routes = routes_to_train(cfg, args.routes)
    df_val = pd.read_csv(val_csv)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

    records = []
    for route_name in routes:
        route = route_config(cfg, route_name)
        adapter_dir = run_root / route_name / "final_adapter"
        if not adapter_dir.exists():
            log(f"Skipping {route_name}: adapter missing at {adapter_dir}")
            continue
        sub = df_val[df_val[data_cfg.get("subset_column", "subset")].isin(route["subsets"])].copy()
        if args.max_rows_per_route and len(sub) > args.max_rows_per_route:
            sub = sub.sample(n=args.max_rows_per_route, random_state=int(cfg.get("seed", 42)))
        if len(sub) == 0:
            continue

        log(f"Evaluating route={route_name} rows={len(sub):,}")
        model, tokenizer = load_model_and_tokenizer(route, adapter_dir)
        try:
            rows = []
            for _, row in tqdm(sub.iterrows(), total=len(sub), desc=f"eval {route_name}"):
                pred = generate_answer(route, model, tokenizer, clean_text(row[data_cfg.get("question_column", "input")]), cfg.get("generation", {}))
                ref = clean_text(row[data_cfg.get("answer_column", "output")])
                scores = scorer.score(ref, pred)
                rows.append(
                    {
                        "route": route_name,
                        "subset": row[data_cfg.get("subset_column", "subset")],
                        "id": row.get(data_cfg.get("id_column", "ID"), ""),
                        "rouge1": scores["rouge1"].fmeasure,
                        "rougeL": scores["rougeL"].fmeasure,
                        "prediction": pred,
                        "reference": ref,
                    }
                )
            route_df = pd.DataFrame(rows)
            records.append(route_df)
            summary = route_df.groupby(["route", "subset"])[["rouge1", "rougeL"]].mean().reset_index()
            log("\n" + summary.to_string(index=False))
        finally:
            del model
            del tokenizer
            free_memory()

    if records:
        out = pd.concat(records, ignore_index=True)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        summary_csv = out_csv.with_name(out_csv.stem + "_summary.csv")
        out.groupby(["route", "subset"])[["rouge1", "rougeL"]].mean().reset_index().to_csv(summary_csv, index=False)
        log(f"Wrote predictions: {out_csv}")
        log(f"Wrote summary: {summary_csv}")


if __name__ == "__main__":
    main()

