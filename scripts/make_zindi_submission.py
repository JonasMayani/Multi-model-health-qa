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
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, BitsAndBytesConfig

from healthqa_ft.config import load_yaml, route_config, routes_to_train
from healthqa_ft.prompts import plain_causal_prompt, seq2seq_input
from healthqa_ft.utils import clean_text, free_memory, log, set_hf_runtime_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate routed Zindi submission from trained adapters")
    parser.add_argument("--config", default=str(ROOT / "configs" / "routes.yaml"))
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--run_root", default=None)
    parser.add_argument("--out_csv", default=None)
    parser.add_argument("--routes", nargs="*", default=None)
    parser.add_argument("--best_summary_csv", default=None, help="Optional *_summary.csv from evaluate_routes.py; picks best route per subset by rougeL")
    parser.add_argument("--max_rows", type=int, default=None, help="Smoke-test limit")
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


def fixed_subset_to_route(cfg: dict[str, Any], routes: list[str]) -> dict[str, str]:
    mapping = {}
    for route_name in routes:
        route = route_config(cfg, route_name)
        for subset in route["subsets"]:
            mapping.setdefault(subset, route_name)
    return mapping


def best_subset_to_route(summary_csv: str, fallback: dict[str, str]) -> dict[str, str]:
    df = pd.read_csv(summary_csv)
    required = {"route", "subset", "rougeL"}
    if not required.issubset(df.columns):
        raise ValueError(f"{summary_csv} must contain columns: {sorted(required)}")
    best = df.sort_values("rougeL", ascending=False).drop_duplicates("subset")
    mapping = dict(zip(best["subset"], best["route"]))
    merged = dict(fallback)
    merged.update(mapping)
    return merged


def zindi_submission(ids: list[str], predictions: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": ids,
            "TargetRLF1": predictions,
            "TargetR1F1": predictions,
            "TargetLLM": predictions,
        }
    )


def main() -> None:
    set_hf_runtime_env()
    args = parse_args()
    cfg = load_yaml(args.config)
    data_cfg = cfg.get("data", {})
    run_root = Path(args.run_root or data_cfg.get("output_root") or "/workspace/healthqa_multimodel_runs")
    test_csv = args.test_csv or data_cfg.get("test_csv")
    if not test_csv:
        raise ValueError("test_csv is required")
    out_csv = Path(args.out_csv or (Path(data_cfg.get("submission_dir", run_root / "submissions")) / "zindi_submission.csv"))
    routes = routes_to_train(cfg, args.routes)
    subset_to_route = fixed_subset_to_route(cfg, routes)
    if args.best_summary_csv:
        subset_to_route = best_subset_to_route(args.best_summary_csv, subset_to_route)

    id_col = data_cfg.get("id_column", "ID")
    q_col = data_cfg.get("question_column", "input")
    subset_col = data_cfg.get("subset_column", "subset")
    df = pd.read_csv(test_csv)
    if args.max_rows:
        df = df.head(args.max_rows).copy()
    for col in [id_col, q_col, subset_col]:
        if col not in df.columns:
            raise ValueError(f"{test_csv} missing required column: {col}")

    predictions = pd.Series(index=df.index, dtype=object)
    route_used = pd.Series(index=df.index, dtype=object)
    for route_name in routes:
        route = route_config(cfg, route_name)
        routed_subsets = [s for s, r in subset_to_route.items() if r == route_name]
        sub_idx = df.index[df[subset_col].isin(routed_subsets)].tolist()
        if not sub_idx:
            continue
        adapter_dir = run_root / route_name / "final_adapter"
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter missing for route={route_name}: {adapter_dir}")
        log(f"Generating route={route_name} rows={len(sub_idx):,} subsets={routed_subsets}")
        model, tokenizer = load_model_and_tokenizer(route, adapter_dir)
        try:
            for idx in tqdm(sub_idx, desc=f"submit {route_name}"):
                pred = generate_answer(route, model, tokenizer, clean_text(df.at[idx, q_col]), cfg.get("generation", {}))
                predictions.at[idx] = pred
                route_used.at[idx] = route_name
        finally:
            del model
            del tokenizer
            free_memory()

    missing = predictions.isna()
    if missing.any():
        missing_counts = df.loc[missing, subset_col].value_counts().to_dict()
        raise RuntimeError(f"Missing predictions for subsets: {missing_counts}")

    sub = zindi_submission(df[id_col].astype(str).tolist(), predictions.astype(str).tolist())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_csv, index=False)
    detail = df[[id_col, q_col, subset_col]].copy()
    detail["route"] = route_used.values
    detail["prediction"] = predictions.values
    detail.to_csv(out_csv.with_name(out_csv.stem + "_with_routes.csv"), index=False)
    log(f"Wrote Zindi submission: {out_csv}")
    log(f"Wrote route details: {out_csv.with_name(out_csv.stem + '_with_routes.csv')}")


if __name__ == "__main__":
    main()

