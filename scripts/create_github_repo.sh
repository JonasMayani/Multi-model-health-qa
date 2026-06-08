#!/usr/bin/env bash
set -euo pipefail

REPO_NAME="${1:-multimodel-healthqa}"
VISIBILITY="${2:-private}"

if ! command -v git >/dev/null 2>&1; then
  echo "git is required" >&2
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI (gh) is required to create/push automatically." >&2
  echo "Install gh or create a GitHub repo manually, then run:" >&2
  echo "  git remote add origin git@github.com:<USER>/${REPO_NAME}.git" >&2
  echo "  git push -u origin main" >&2
  exit 1
fi

git init
git branch -M main
git add .
git commit -m "Initial multimodel Health QA fine-tuning pipeline" || true
gh repo create "$REPO_NAME" "--$VISIBILITY" --source=. --remote=origin --push

echo "Repository pushed to GitHub: $REPO_NAME"

