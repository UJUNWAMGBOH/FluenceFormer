# FluenceFormer (Prostate Fluence) – Ablation + Prediction Saving

This repository contains:
1) Backbone × Loss ablation (metrics + CSV)
2) Clean training + inference pipeline (FAR loss) that saves predictions per patient with alignment audit trail.

## Backbones included (paper-reported)
- SwinUNETR (MONAI)
- UNETR (MONAI)
- nnFormer 
- MedFormer

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

Expected data layout

DATA_ROOT/
  train/
    ct/          {id}_ct.npy
    contour/     {id}_contoursCT.npy
    dose/        {id}_dose.npy
    fluences/    {id}_fluences.npy
  val/
    ...
  test/
    ...

For my dataset, each fluences file must contain 9 beam maps per patient.


1) Train (FAR) + Inference + Save Predictions

This pipeline:
  Trains each backbone using FAR loss
  Saves model checkpoints
  Runs inference on the test set
  Aligns predicted beams to ground truth using Hungarian matching
  Saves prediction stacks, alignment mappings, and visualizations

Run Training + Inference
PYTHONPATH=src python -m prostate_fluence.run_train_and_infer \
  --data-root "/path/to/DATA_ROOT" \
  --out-dir "./train_infer_outputs" \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --save-per-beam

train_infer_outputs/
├─ checkpoints/
│   ├─ SwinUNETR_FAR.pth
│   ├─ UNETR_FAR.pth
│   ├─ nnFormer_FAR.pth
│   └─ MedFormer_FAR.pth
│
├─ SwinUNETR/
│   └─ patient_<pid>/
│        ├─ gt_9hw.npy
│        ├─ pred_9hw_raw.npy
│        ├─ pred_9hw_aligned.npy
│        ├─ alignment_mapping.npy
│        ├─ <pid>_viz.png
│        └─ beams/ (optional)
│
├─ UNETR/
├─ nnFormer/
└─ MedFormer/


2) Backbone × Loss Ablation

This script performs the full ablation experiment.
For each backbone:
  Retrains from scratch under each loss configuration:
  Baseline (MSE)
  MSE + Correlation
  MSE + Energy
  MSE + Gradient
  Proposed FAR
  Evaluates on the validation set

Computes:
  MAE, Energy %, PSNR, SSIM, Performs Wilcoxon signed-rank test (FAR vs Baseline)
  Writes results to CSV

Run Ablation
PYTHONPATH=src python -m prostate_fluence.run_ablation \
  --data-root "/path/to/DATA_ROOT" \
  --ckpt-dir "./ablation_outputs" \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --retrain-per-loss



Citation
If you use this code in your research, please cite the FluenceFormer paper.

