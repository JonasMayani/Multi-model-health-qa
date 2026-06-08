# Multimodel Health QA Fine-Tuning Pipeline

RunPod A100 80GB pipeline for routed multilingual Health QA fine-tuning.

Target routes:

| Dataset subset | Base model | Adapter |
|---|---|---|
| `Amh_Eth` | `McGill-NLP/AfriqueQwen-14B` | Amharic LoRA |
| `Swa_Ken` | `McGill-NLP/AfriqueQwen-14B` | Swahili LoRA |
| `Lug_Uga` | `Sunbird/Sunflower-14B` | Luganda LoRA |
| `Aka_Gha` | `CohereLabs/aya-101` | Akan/Twi LoRA |
| `Eng_*` | `Qwen/Qwen3-14B` | shared English LoRA |

The pipeline trains one route at a time, loads each base model in 4-bit QLoRA, saves checkpoints every epoch, and retries with smaller memory settings if an OOM occurs.

## GitHub Repository

This folder is GitHub-ready. It includes `.gitignore` rules that exclude data, checkpoints, model weights, logs, and generated submissions.

Create and push a GitHub repository with GitHub CLI:

```bash
cd /workspace/Mult-model
bash scripts/create_github_repo.sh multimodel-healthqa private
```

Or push manually:

```bash
git init
git branch -M main
git add .
git commit -m "Initial multimodel Health QA fine-tuning pipeline"
git remote add origin git@github.com:<YOUR_USER>/multimodel-healthqa.git
git push -u origin main
```

## RunPod Setup

```bash
cd /workspace/Mult-model
bash scripts/setup_runpod.sh
```

Put data here:

```text
/workspace/data/train_augmented.csv
/workspace/data/val_clean_quality.csv
/workspace/data/Test.csv
```

Expected columns:

```text
ID,input,output,subset
```

## Train All Routes

```bash
python scripts/train_all.py \
  --config configs/routes.yaml \
  --train_csv /workspace/data/train_augmented.csv \
  --val_csv /workspace/data/val_clean_quality.csv \
  --output_root /workspace/healthqa_multimodel_runs
```

Train one route only:

```bash
python scripts/train_route.py \
  --config configs/routes.yaml \
  --route amh \
  --train_csv /workspace/data/train_augmented.csv \
  --val_csv /workspace/data/val_clean_quality.csv \
  --output_root /workspace/healthqa_multimodel_runs
```

Resume is automatic when checkpoints exist. Use `--no_resume` only when you intentionally want to restart a route.

## Evaluate ROUGE

```bash
python scripts/evaluate_routes.py \
  --config configs/routes.yaml \
  --val_csv /workspace/data/val_clean_quality.csv \
  --run_root /workspace/healthqa_multimodel_runs \
  --out_csv /workspace/healthqa_multimodel_runs/rouge_by_route.csv
```

## Create Zindi Submission

After training, generate predictions on `Test.csv` and write the required Zindi format:

```text
ID,TargetRLF1,TargetR1F1,TargetLLM
```

Run evaluation and submission together:

```bash
python scripts/evaluate_and_submit.py \
  --config configs/routes.yaml \
  --val_csv /workspace/data/val_clean_quality.csv \
  --test_csv /workspace/data/Test.csv \
  --run_root /workspace/healthqa_multimodel_runs \
  --submission_csv /workspace/healthqa_multimodel_runs/submissions/zindi_submission.csv
```

Or create submission only:

```bash
python scripts/make_zindi_submission.py \
  --config configs/routes.yaml \
  --test_csv /workspace/data/Test.csv \
  --run_root /workspace/healthqa_multimodel_runs \
  --out_csv /workspace/healthqa_multimodel_runs/submissions/zindi_submission.csv
```

The submission script routes rows by `subset`:

```text
Amh_Eth -> amh adapter
Swa_Ken -> swa adapter
Lug_Uga -> lug adapter
Aka_Gha -> aka adapter
Eng_*   -> eng adapter
```

If you pass `--best_summary_csv`, it can choose the best route per subset based on validation `rougeL`.

## Practical Notes

- Start with `max_seq_length: 768`. If outputs truncate badly, raise to `1024` on A100.
- Keep deterministic decoding for ROUGE evaluation: no sampling, low max tokens, optional beams for Aya-101.
- If the route trainer hits OOM, it automatically tries smaller batch and sequence settings.
- For a quick smoke test, set `max_train_rows` and `max_eval_rows` in `configs/routes.yaml`.
- Keep English rows as anchors, but do not let English dominate African-language route training.

## Output Layout

```text
/workspace/healthqa_multimodel_runs/
  amh/
    checkpoints/
    final_adapter/
    train_status.json
  swa/
  lug/
  aka/
  eng/
  submissions/
    zindi_submission.csv
    zindi_submission_with_routes.csv
  route_training_summary.json
```
