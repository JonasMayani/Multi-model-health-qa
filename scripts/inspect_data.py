#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--subset_col", default="subset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.csv)
    print(f"Rows: {len(df):,}")
    print(df[args.subset_col].value_counts().sort_index().to_string())
    print()
    print(df.head(3).to_string())


if __name__ == "__main__":
    main()

