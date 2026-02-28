import numpy as np
import torch
import torch.nn.functional as F

def deg2vec(deg: float):
    rad = np.deg2rad(deg)
    return np.sin(rad), np.cos(rad)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def normalize_global(x, vmin, vmax):
    return (x - vmin) / (vmax - vmin + 1e-8)

def ensure_D1HW(vol):
    """
    Ensure volume is shaped [D, 1, H, W] (D slices, 1 channel).
    """
    vol = np.asarray(vol)
    if vol.ndim == 4:
        if vol.shape[1] == 1:
            return vol
        if vol.shape[-1] == 1:
            return np.moveaxis(vol, -1, 1)
        if vol.shape[0] == 1:
            return np.moveaxis(vol, 0, 1)
        return np.moveaxis(vol, -1, 1)
    elif vol.ndim == 3:
        return vol[:, None, ...]
    raise ValueError(f"Unsupported shape: {vol.shape}")

def resize_D1HW_numpy(vol_D1HW, H, W):
    t = torch.from_numpy(vol_D1HW).float()
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t.cpu().numpy()

def upsample_chw_numpy(arr_chw, H, W):
    t = torch.from_numpy(arr_chw[None, ...]).float()
    t = F.interpolate(t, size=(H, W), mode="bilinear", align_corners=False)
    return t[0].cpu().numpy()

def list_patient_ids(data_root, task="test"):
    import os, glob
    test_ct_files = glob.glob(os.path.join(data_root, f"{task}/ct/*.npy"))
    return sorted([os.path.basename(f).split("_")[0] for f in test_ct_files])