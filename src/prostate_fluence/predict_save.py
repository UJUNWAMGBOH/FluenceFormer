import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

from .config import H, W, BEAM_ANGLES
from .utils import deg2vec

def predict_patient_9_fluences(model, device, ct, cnt, dos):
    """
    Predicts fluence for the mid-slice only.
    Inputs:
      ct,cnt,dos: [D,1,H,W]
    Output:
      pred_9hw: [9,H,W]
    """
    model.eval()
    model.to(device)

    D = ct.shape[0]
    mid = D // 2
    ct_s, cnt_s, dos_s = ct[mid, 0], cnt[mid, 0], dos[mid, 0]

    anatomy = np.stack([ct_s, cnt_s, dos_s], axis=0).astype(np.float32)  # [3,H,W]
    anatomy_t = torch.from_numpy(anatomy).to(device)

    pred_9hw = np.zeros((9, H, W), dtype=np.float32)

    with torch.no_grad():
        for b in range(9):
            sin_v, cos_v = deg2vec(BEAM_ANGLES[b])
            geo = torch.zeros((2, H, W), device=device, dtype=torch.float32)
            geo[0] = sin_v
            geo[1] = cos_v

            x = torch.cat([anatomy_t, geo], dim=0).unsqueeze(0)  # [1,5,H,W]
            pred_9hw[b] = F.relu(model(x))[0, 0].detach().cpu().numpy()

    return pred_9hw

def align_pred_to_gt(pred_9hw: np.ndarray, gt_9hw: np.ndarray):
    """
    Hungarian alignment by MSE between beams.
    Returns:
      pred_aligned: [9,H,W]
      mapping: [9] where mapping[i] is the original pred beam index assigned to GT beam i.
    """
    C = pred_9hw.shape[0]
    cost = np.zeros((C, C), dtype=np.float64)
    for i in range(C):
        gi = gt_9hw[i]
        for j in range(C):
            pj = pred_9hw[j]
            cost[i, j] = np.mean((gi - pj) ** 2)

    row_ind, col_ind = linear_sum_assignment(cost)
    mapping = col_ind.astype(np.int64)
    pred_aligned = pred_9hw[mapping]
    return pred_aligned, mapping

def save_patient_png_9x3(gt_9hw: np.ndarray, pred_9hw: np.ndarray, patient_dir: str, filename: str):
    """
    Save 9x3: GT | Pred | |Diff|
    """
    os.makedirs(patient_dir, exist_ok=True)
    diff_9hw = np.abs(gt_9hw - pred_9hw)

    fig, axes = plt.subplots(9, 3, figsize=(12, 36))
    for i in range(9):
        vmax = float(gt_9hw[i].max())
        if vmax == 0:
            vmax = 1.0
        vmin = 0.0

        axes[i, 0].imshow(gt_9hw[i], cmap="jet", vmin=vmin, vmax=vmax)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_9hw[i], cmap="jet", vmin=vmin, vmax=vmax)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(diff_9hw[i], cmap="jet")
        axes[i, 2].axis("off")

    plt.tight_layout()
    out_path = os.path.join(patient_dir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def save_patient_predictions(
    patient_dir: str,
    gt_9hw: np.ndarray,
    pred_9hw_raw: np.ndarray,
    pred_9hw_aligned: np.ndarray,
    mapping: np.ndarray,
    save_per_beam: bool = True,
):
    """
    Saves:
      patient_dir/
        gt_9hw.npy
        pred_9hw_raw.npy
        pred_9hw_aligned.npy
        alignment_mapping.npy
        beams/pred_beam_1.npy ... pred_beam_9.npy (optional; aligned)
    """
    os.makedirs(patient_dir, exist_ok=True)

    np.save(os.path.join(patient_dir, "gt_9hw.npy"), gt_9hw.astype(np.float32))
    np.save(os.path.join(patient_dir, "pred_9hw_raw.npy"), pred_9hw_raw.astype(np.float32))
    np.save(os.path.join(patient_dir, "pred_9hw_aligned.npy"), pred_9hw_aligned.astype(np.float32))
    np.save(os.path.join(patient_dir, "alignment_mapping.npy"), mapping.astype(np.int64))

    if save_per_beam:
        beams_dir = os.path.join(patient_dir, "beams")
        os.makedirs(beams_dir, exist_ok=True)
        for b in range(9):
            np.save(os.path.join(beams_dir, f"pred_beam_{b+1}.npy"), pred_9hw_aligned[b].astype(np.float32))