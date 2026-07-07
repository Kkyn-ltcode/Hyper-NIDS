#!/bin/bash
# Run from your project root

OUT="code_summary.txt"
> "$OUT"

# Folders to skip entirely — add any others you have (data/, checkpoints/, wandb/, etc.)
EXCLUDES="-not -path '*/__pycache__/*' \
  -not -path '*/.git/*' \
  -not -path '*/venv/*' \
  -not -path '*/.venv/*' \
  -not -path '*/env/*' \
  -not -path '*/node_modules/*' \
  -not -path '*/wandb/*' \
  -not -path '*/checkpoints/*' \
  -not -path '*/.ipynb_checkpoints/*'"

# 1. Full file tree, however deep, minus junk
echo "=== PROJECT STRUCTURE ===" >> "$OUT"
eval find . -name "*.py" $EXCLUDES | sort >> "$OUT"
echo "" >> "$OUT"

# 2. Per-file skeleton: docstring + classes/functions, preserving relative path
echo "=== FILE SKELETONS ===" >> "$OUT"
eval find . -name "*.py" $EXCLUDES | sort | while read -r f; do
    echo "--- $f ---" >> "$OUT"
    head -5 "$f" >> "$OUT"
    grep -n "^class \|^def \|^    def " "$f" >> "$OUT"
    echo "" >> "$OUT"
done

echo "Done. See $OUT"