from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

from .prompts import plain_causal_prompt, seq2seq_input, system_prompt, user_prompt
from .utils import clean_text, log


def load_subset_csv(
    csv_path: str,
    subsets: list[str],
    question_col: str = "input",
    answer_col: str = "output",
    subset_col: str = "subset",
    id_col: str = "ID",
    max_rows: int | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = [question_col, answer_col, subset_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} missing columns: {missing}")
    if id_col not in df.columns:
        df[id_col] = [f"row_{i:08d}" for i in range(len(df))]

    df = df[df[subset_col].isin(subsets)].copy()
    df[question_col] = df[question_col].map(clean_text)
    df[answer_col] = df[answer_col].map(clean_text)
    df = df[(df[question_col].str.len() > 0) & (df[answer_col].str.len() > 0)].copy()
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=seed)
    df = df.reset_index(drop=True)
    log(f"Loaded {len(df):,} rows from {csv_path} for subsets={subsets}")
    return df


def render_chat_or_plain(tokenizer: Any, question: str, answer: str | None, language_name: str, add_answer: bool) -> list[int]:
    messages = [
        {"role": "system", "content": system_prompt(language_name)},
        {"role": "user", "content": user_prompt(question, language_name)},
    ]
    if add_answer and answer is not None:
        messages.append({"role": "assistant", "content": answer})

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=not add_answer,
            enable_thinking=False,
        )
    except TypeError:
        try:
            return tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=not add_answer)
        except Exception:
            pass
    except Exception:
        pass

    if add_answer and answer is not None:
        text = f"{plain_causal_prompt(question, language_name)} {answer}{tokenizer.eos_token or ''}"
    else:
        text = plain_causal_prompt(question, language_name)
    return tokenizer(text, add_special_tokens=True)["input_ids"]


class CausalQADataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        language_name: str,
        question_col: str = "input",
        answer_col: str = "output",
        max_length: int = 768,
        answer_only_loss: bool = True,
    ):
        self.rows = df.to_dict("records")
        self.tokenizer = tokenizer
        self.language_name = language_name
        self.question_col = question_col
        self.answer_col = answer_col
        self.max_length = max_length
        self.answer_only_loss = answer_only_loss

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        row = self.rows[idx]
        question = clean_text(row[self.question_col])
        answer = clean_text(row[self.answer_col])
        prompt_ids = render_chat_or_plain(self.tokenizer, question, None, self.language_name, add_answer=False)
        full_ids = render_chat_or_plain(self.tokenizer, question, answer, self.language_name, add_answer=True)
        eos = self.tokenizer.eos_token_id
        if eos is not None and (not full_ids or full_ids[-1] != eos):
            full_ids = full_ids + [eos]
        full_ids = full_ids[: self.max_length]
        attention_mask = [1] * len(full_ids)
        labels = list(full_ids)
        if self.answer_only_loss:
            prompt_len = min(len(prompt_ids), len(labels))
            labels[:prompt_len] = [-100] * prompt_len
        return {"input_ids": full_ids, "attention_mask": attention_mask, "labels": labels}


@dataclass
class CausalCollator:
    tokenizer: Any
    label_pad_token_id: int = -100

    def __call__(self, features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in features)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in features:
            pad = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * pad)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad)
            batch["labels"].append(item["labels"] + [self.label_pad_token_id] * pad)
        return {k: torch.tensor(v, dtype=torch.long) for k, v in batch.items()}


class Seq2SeqQADataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: Any,
        language_name: str,
        question_col: str = "input",
        answer_col: str = "output",
        max_length: int = 512,
    ):
        self.rows = df.to_dict("records")
        self.tokenizer = tokenizer
        self.language_name = language_name
        self.question_col = question_col
        self.answer_col = answer_col
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        row = self.rows[idx]
        model_input = seq2seq_input(clean_text(row[self.question_col]), self.language_name)
        target = clean_text(row[self.answer_col])
        enc = self.tokenizer(model_input, truncation=True, max_length=self.max_length)
        labels = self.tokenizer(text_target=target, truncation=True, max_length=self.max_length)
        enc["labels"] = labels["input_ids"]
        return enc

