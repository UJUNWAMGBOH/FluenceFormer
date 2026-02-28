import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.optim import AdamW

from .config import (
    H, W, FLU_SCALING_FACTOR,
    DATA_ROOT, CT_PATH, CNT_PATH, DOSE_PATH, FLU_PATH
)
from .utils import get_device, ensure_D1HW, normalize_global, resize_D1HW_numpy, upsample_chw_numpy, list_patient_ids
from .dataset import Stage2BeamDataset
from .models import MODEL_BUILDERS
from .losses import pafr_loss
from .predict_save import (
    predict_patient_9_fluences,
    align_pred_to_gt,
    save_patient_png_9x3,
    save_patient_predictions,
)

def parse_args():
    p = argparse.ArgumentParser(description="Train (FAR) then infer on test and save predictions.")
    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--out-dir", default="./train_infer_outputs")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-per-beam", action="store_true", help="Also save 9 per-beam NPY files.")
    return p.parse_args()

def load_patient_arrays(data_root, pid, task):
    ct_fp  = CT_PATH.format(root=data_root, task=task, id=pid)
    cnt_fp = CNT_PATH.format(root=data_root, task=task, id=pid)
    dos_fp = DOSE_PATH.format(root=data_root, task=task, id=pid)
    flu_fp = FLU_PATH.format(root=data_root, task=task, id=pid)

    if not (os.path.exists(ct_fp) and os.path.exists(cnt_fp) and os.path.exists(dos_fp) and os.path.exists(flu_fp)):
        return None

    ct  = ensure_D1HW(np.load(ct_fp).astype(np.float32))
    cnt = ensure_D1HW(np.load(cnt_fp).astype(np.float32))
    dos = ensure_D1HW(np.load(dos_fp).astype(np.float32))
    flu = np.load(flu_fp).astype(np.float32)
    if flu.ndim == 4:
        flu = np.squeeze(flu, -1)

    if ct.shape[2] != H:
        ct, cnt, dos = [resize_D1HW_numpy(x, H, W) for x in (ct, cnt, dos)]
    if flu.shape[1] != H:
        flu = upsample_chw_numpy(flu, H, W)

    ct  = np.clip(normalize_global(ct, -1000, 3000), 0, 1)
    cnt = np.clip(cnt, 0, 1)
    dos = dos / 1e6
    flu = flu * FLU_SCALING_FACTOR
    return ct, cnt, dos, flu

def train_far(model, loader, device, epochs, lr):
    """
    Trains with FAR weights (alpha=1.0, beta=0.5, gamma=0.3, delta=0.2)
    """
    model.train()
    opt = AdamW(model.parameters(), lr=lr)
    a, b, g, d = 1.0, 0.5, 0.3, 0.2

    for ep in range(epochs):
        total = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            pred = F.relu(model(x))
            loss, _ = pafr_loss(pred, y, a, b, g, d)
            loss.backward()
            opt.step()
            total += float(loss.item())
        print(f"  Epoch {ep+1}/{epochs} | Loss: {total/len(loader):.4f}")

    return model

def main():
    args = parse_args()
    device = get_device()
    torch.cuda.empty_cache()

    out_dir = args.out_dir
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Train set
    ds_train = Stage2BeamDataset(args.data_root, "train", CT_PATH, CNT_PATH, DOSE_PATH, FLU_PATH, H, W)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Test patient list
    patient_ids = list_patient_ids(args.data_root, task="test")
    print(f"Found {len(patient_ids)} test patients.")

    for mname, builder in MODEL_BUILDERS:
        print(f"\n>>> Processing {mname} (FAR loss)...")

        # A) Train
        print(f"  Training {mname} from scratch...")
        model = builder().to(device)
        model = train_far(model, dl_train, device, args.epochs, args.lr)

        # Save weights
        ckpt_path = os.path.join(ckpt_dir, f"{mname}_FAR.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Saved weights to {ckpt_path}")

        # B) Inference & saving
        model_out_dir = os.path.join(out_dir, mname)
        os.makedirs(model_out_dir, exist_ok=True)
        print(f"  Running inference on {len(patient_ids)} patients...")

        for pid in patient_ids:
            arrays = load_patient_arrays(args.data_root, pid, "test")
            if arrays is None:
                continue
            ct, cnt, dos, flu_gt = arrays

            pred_9hw_raw = predict_patient_9_fluences(model, device, ct, cnt, dos)
            pred_9hw_aligned, mapping = align_pred_to_gt(pred_9hw_raw, flu_gt)

            patient_dir = os.path.join(model_out_dir, f"patient_{pid}")
            os.makedirs(patient_dir, exist_ok=True)

            save_patient_png_9x3(flu_gt, pred_9hw_aligned, patient_dir, f"{pid}_viz.png")
            save_patient_predictions(
                patient_dir=patient_dir,
                gt_9hw=flu_gt,
                pred_9hw_raw=pred_9hw_raw,
                pred_9hw_aligned=pred_9hw_aligned,
                mapping=mapping,
                save_per_beam=args.save_per_beam,
            )

        print(f"  Saved outputs to {model_out_dir}")

    print("\nDone.")

if __name__ == "__main__":
    main()