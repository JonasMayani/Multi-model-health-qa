#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate routed adapters and create Zindi submission")
    parser.add_argument("--config", default=str(ROOT / "configs" / "routes.yaml"))
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--test_csv", default=None)
    parser.add_argument("--run_root", default=None)
    parser.add_argument("--submission_csv", default=None)
    parser.add_argument("--routes", nargs="*", default=None)
    parser.add_argument("--max_eval_rows_per_route", type=int, default=None)
    parser.add_argument("--max_test_rows", type=int, default=None)
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root or "/workspace/healthqa_multimodel_runs")
    eval_csv = run_root / "rouge_by_route.csv"
    summary_csv = run_root / "rouge_by_route_summary.csv"
    submission_csv = Path(args.submission_csv or (run_root / "submissions" / "zindi_submission.csv"))

    eval_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "evaluate_routes.py"),
        "--config",
        args.config,
        "--run_root",
        str(run_root),
        "--out_csv",
        str(eval_csv),
    ]
    if args.val_csv:
        eval_cmd.extend(["--val_csv", args.val_csv])
    if args.routes:
        eval_cmd.extend(["--routes", *args.routes])
    if args.max_eval_rows_per_route:
        eval_cmd.extend(["--max_rows_per_route", str(args.max_eval_rows_per_route)])
    run(eval_cmd)

    submit_cmd = [
        sys.executable,
        str(ROOT / "scripts" / "make_zindi_submission.py"),
        "--config",
        args.config,
        "--run_root",
        str(run_root),
        "--best_summary_csv",
        str(summary_csv),
        "--out_csv",
        str(submission_csv),
    ]
    if args.test_csv:
        submit_cmd.extend(["--test_csv", args.test_csv])
    if args.routes:
        submit_cmd.extend(["--routes", *args.routes])
    if args.max_test_rows:
        submit_cmd.extend(["--max_rows", str(args.max_test_rows)])
    run(submit_cmd)


if __name__ == "__main__":
    main()

