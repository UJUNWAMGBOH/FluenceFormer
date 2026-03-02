# Instructions

## Expected Data Layout

Your dataset directory must follow this structure:

```
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
```

### Important

- For this project, each `fluences` file ** contains 9 beam maps per patient**.
- File naming must remain consistent across `ct`, `contour`, `dose`, and `fluences`.

---

# 1) Train (FAR) + Inference + Save Predictions

This pipeline:

- Trains each backbone using the proposed **FAR loss**
- Saves model checkpoints
- Runs inference on the test set
- Aligns predicted beams to ground truth using Hungarian matching
- Saves prediction stacks, alignment mappings, and visualizations

---

## Run Training + Inference

```bash
PYTHONPATH=src python -m prostate_fluence.run_train_and_infer \
  --data-root "/path/to/DATA_ROOT" \
  --out-dir "./train_infer_outputs" \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --save-per-beam
```

---

## Output Structure

```
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
```

---

# 2) Backbone × Loss Ablation

This script performs the full ablation experiment.

For each backbone, it:

- Retrains from scratch under multiple loss configurations:
  - Baseline (MSE)
  - MSE + Correlation
  - MSE + Energy
  - MSE + Gradient
  - Proposed FAR
- Evaluates on the validation set

---

## Metrics Computed

- MAE  
- Energy Error (%)  
- PSNR  
- SSIM  
- Wilcoxon signed-rank test (FAR vs Baseline)

Results are written to a CSV file.

---

## Run Ablation

```bash
PYTHONPATH=src python -m prostate_fluence.run_ablation \
  --data-root "/path/to/DATA_ROOT" \
  --ckpt-dir "./ablation_outputs" \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --retrain-per-loss
```

---

## Ablation Output

```
ablation_outputs/
└─ all_ablation_results.csv
```