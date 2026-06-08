#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from healthqa_ft.config import load_yaml, routes_to_train
from healthqa_ft.utils import log, set_hf_runtime_env, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train all routed Health QA LoRA adapters sequentially")
    parser.add_argument("--config", default=str(ROOT / "configs" / "routes.yaml"))
    parser.add_argument("--routes", nargs="*", default=None, help="Optional route list, e.g. amh swa lug")
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--output_root", default=None)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--stop_on_error", action="store_true")
    return parser.parse_args()


def main() -> None:
    set_hf_runtime_env()
    args = parse_args()
    cfg = load_yaml(args.config)
    route_names = routes_to_train(cfg, args.routes)
    data_cfg = cfg.get("data", {})
    output_root = args.output_root or data_cfg.get("output_root") or "/workspace/healthqa_multimodel_runs"
    Path(output_root).mkdir(parents=True, exist_ok=True)

    summary = []
    for route in route_names:
        log(f"Starting route: {route}")
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "train_route.py"),
            "--config",
            args.config,
            "--route",
            route,
            "--output_root",
            output_root,
        ]
        if args.train_csv:
            cmd.extend(["--train_csv", args.train_csv])
        if args.val_csv:
            cmd.extend(["--val_csv", args.val_csv])
        if args.no_resume:
            cmd.append("--no_resume")

        proc = subprocess.run(cmd)
        item = {"route": route, "returncode": proc.returncode}
        summary.append(item)
        write_json(Path(output_root) / "route_training_summary.json", summary)
        if proc.returncode != 0:
            log(f"Route failed: {route} returncode={proc.returncode}")
            if args.stop_on_error:
                raise SystemExit(proc.returncode)
        else:
            log(f"Route complete: {route}")

    log("All requested routes finished.")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

