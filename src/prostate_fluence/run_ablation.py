import os
import argparse
import torch

from torch.utils.data import DataLoader
from torch.optim import AdamW
from scipy.stats import wilcoxon

from .config import (
    EPOCHS_S2, LOSS_CONFIGS,
    DATA_ROOT, CT_PATH, CNT_PATH, DOSE_PATH, FLU_PATH
)
from .utils import get_device
from .dataset import Stage2BeamDataset
from .models import MODEL_BUILDERS
from .train_eval import train_epoch, evaluate_metrics, print_result_block, format_mean_std


def parse_args():
    p = argparse.ArgumentParser(description="FluenceFormer backbone × loss ablation.")

    p.add_argument("--data-root", default=DATA_ROOT)
    p.add_argument("--ckpt-dir", default="./ablation_outputs")
    p.add_argument("--epochs", type=int, default=EPOCHS_S2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)

    # 🔥 Explicit retraining flag
    p.add_argument("--retrain-per-loss", action="store_true",
                   help="Force rebuild + retrain model separately for each loss config.")

    # Speed controls
    p.add_argument("--max-train-batches", type=int, default=None)
    p.add_argument("--max-val-batches", type=int, default=None)

    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    torch.cuda.empty_cache()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    results_file = os.path.join(args.ckpt_dir, "all_ablation_results.csv")

    # Data
    ds_tr = Stage2BeamDataset(args.data_root, "train", CT_PATH, CNT_PATH, DOSE_PATH, FLU_PATH)
    ds_va = Stage2BeamDataset(args.data_root, "val",   CT_PATH, CNT_PATH, DOSE_PATH, FLU_PATH)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size,
                       shuffle=True, num_workers=args.num_workers)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size,
                       shuffle=False, num_workers=args.num_workers)

    print("\nStarting Backbone × Loss Ablation...")

    with open(results_file, "w") as f:
        f.write("Backbone,Loss Config,MAE,Energy%,PSNR,SSIM,PValue\n")

    for backbone_name, builder in MODEL_BUILDERS:

        print(f"\n==============================")
        print(f">>> Backbone: {backbone_name}")
        print(f"==============================")

        baseline_energy = []
        proposed_energy = []

        for config_name, a, b, g, d in LOSS_CONFIGS:

            print(f"\n--- Training with Loss: {config_name} ---")

            # 🔥 Force new model each time if flag is set
            if args.retrain_per_loss:
                model = builder().to(device)
            else:
                model = builder().to(device)

            opt = AdamW(model.parameters(), lr=args.lr)

            # Train
            for ep in range(args.epochs):
                train_epoch(
                    model, dl_tr, opt, device,
                    a, b, g, d,
                    max_batches=args.max_train_batches
                )
                print(f"  Epoch {ep+1}/{args.epochs} completed.")

            # Evaluate
            res = evaluate_metrics(
                model, device, dl_va,
                max_batches=args.max_val_batches
            )

            print_result_block(backbone_name, config_name, res)

            mae_s  = format_mean_std(res["mae"],   fmt="{:.2f}")
            en_s   = format_mean_std(res["energy"], fmt="{:.2f}")
            psnr_s = format_mean_std(res["psnr"],  fmt="{:.2f}")
            ssim_s = format_mean_std(res["ssim"],  fmt="{:.4f}")

            with open(results_file, "a") as f:
                f.write(f"{backbone_name},{config_name},{mae_s},{en_s},{psnr_s},{ssim_s},-\n")

            if config_name == "Baseline (MSE)":
                baseline_energy = res["energy"]
            elif config_name == "Proposed FAR":
                proposed_energy = res["energy"]

        # Wilcoxon significance test
        if len(proposed_energy) > 0 and len(baseline_energy) > 0:
            _, p_val = wilcoxon(proposed_energy, baseline_energy, alternative="less")
            p_val_str = f"{p_val:.4e}"
            print(f"\n[Stats] {backbone_name} FAR vs Baseline p-value: {p_val_str}")

            with open(results_file, "a") as f:
                f.write(f"{backbone_name},Significance Test,-,-,-,-,{p_val_str}\n")

    print(f"\nDone. Results saved to {results_file}")


if __name__ == "__main__":
    main()