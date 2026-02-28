#!/usr/bin/env bash
set -e

PYTHONPATH=src python -m prostate_fluence.run_ablation \
  --data-root "/home/uj/Prostate/data" \
  --ckpt-dir "./outputs" \
  --epochs 1 \
  --batch-size 2 \
  --smoke-test