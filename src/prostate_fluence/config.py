import os
import numpy as np

# =========================================================
# CONFIGURATION
# =========================================================

H = 128
W = 128

EPOCHS_S1 = 50
EPOCHS_S2 = 50

FLU_SCALING_FACTOR = 100.0
BEAM_ANGLES = np.linspace(0, 360, 9, endpoint=False)

# Default paths (override via CLI)
DATA_ROOT = "/home/uj/Prostate/data"
CT_PATH   = "{root}/{task}/ct/{id}_ct.npy"
CNT_PATH  = "{root}/{task}/contour/{id}_contoursCT.npy"
DOSE_PATH = "{root}/{task}/dose/{id}_dose.npy"
FLU_PATH  = "{root}/{task}/fluences/{id}_fluences.npy"

CKPT_DIR = "/home/uj/Prostate/visualization/2staged/full_ablation_study"
RESULTS_FILE = os.path.join(CKPT_DIR, "all_ablation_results.csv")

# Ablation configurations:
# (Name, alpha (MSE), beta (Grad), gamma (Corr), delta (Energy))
LOSS_CONFIGS = [
    ("Baseline (MSE)", 1.0, 0.0, 0.0, 0.0),
    ("MSE + Corr",     1.0, 0.0, 0.3, 0.0),
    ("MSE + Energy",   1.0, 0.0, 0.0, 0.2),
    ("MSE + Grad",     1.0, 0.5, 0.0, 0.0),
    ("Proposed FAR",   1.0, 0.5, 0.3, 0.2),
]