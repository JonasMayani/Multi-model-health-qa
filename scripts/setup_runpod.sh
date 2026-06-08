#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

mkdir -p /workspace/data
mkdir -p /workspace/healthqa_multimodel_runs

echo "Setup complete."
echo "Put train/validation CSVs in /workspace/data or pass explicit paths to train_all.py."

