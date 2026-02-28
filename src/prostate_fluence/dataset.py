import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from .config import FLU_SCALING_FACTOR, BEAM_ANGLES, H, W
from .utils import deg2vec, ensure_D1HW, normalize_global, resize_D1HW_numpy, upsample_chw_numpy

class Stage2BeamDataset(Dataset):
    """
    Per-slice, per-beam samples:
      x: [5, H, W] = (CT, Contour, Dose, sin(angle), cos(angle))
      y: [1, H, W] = beam fluence
    """
    def __init__(self, root, task, ct_path, contour_path, dose_path, flu_path, H=128, W=128):
        self.H, self.W = H, W
        self.items = []

        for ct_fp in glob.glob(ct_path.format(root=root, task=task, id="*")):
            pid = os.path.basename(ct_fp).split("_")[0]
            cnt_fp = contour_path.format(root=root, task=task, id=pid)
            dos_fp = dose_path.format(root=root, task=task, id=pid)
            flu_fp = flu_path.format(root=root, task=task, id=pid)

            if not (os.path.exists(cnt_fp) and os.path.exists(dos_fp) and os.path.exists(flu_fp)):
                continue

            ct  = ensure_D1HW(np.load(ct_fp).astype(np.float32))
            cnt = ensure_D1HW(np.load(cnt_fp).astype(np.float32))
            dos = ensure_D1HW(np.load(dos_fp).astype(np.float32))

            flu = np.load(flu_fp).astype(np.float32)
            if flu.ndim == 4:
                flu = np.squeeze(flu, -1)  # [B,H,W]

            if ct.shape[2] != H:
                ct, cnt, dos = [resize_D1HW_numpy(x, H, W) for x in (ct, cnt, dos)]
            if flu.shape[1] != H:
                flu = upsample_chw_numpy(flu, H, W)

            ct  = np.clip(normalize_global(ct, -1000, 3000), 0, 1)
            cnt = np.clip(cnt, 0, 1)
            dos = dos / 1e6
            flu = flu * FLU_SCALING_FACTOR

            D = ct.shape[0]
            for i in range(D):
                base = np.concatenate([ct[i], cnt[i], dos[i]], axis=0)  # [3,H,W]
                for b in range(len(BEAM_ANGLES)):
                    gt_b = flu[b:b+1]  # [1,H,W]
                    self.items.append((base, gt_b, b))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        base, gt, ang_idx = self.items[i]
        sin_v, cos_v = deg2vec(BEAM_ANGLES[ang_idx])
        geo = np.zeros((2, self.H, self.W), dtype=np.float32)
        geo[0] = sin_v
        geo[1] = cos_v
        x = np.concatenate([base, geo], axis=0)  # [5,H,W]
        return torch.tensor(x).float(), torch.tensor(gt).float()