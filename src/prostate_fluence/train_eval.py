import numpy as np
import torch
import torch.nn.functional as F

from .losses import pafr_loss

try:
    from skimage.metrics import structural_similarity as ssim
except ImportError:
    from skimage.measure import compare_ssim as ssim

def train_epoch(model, loader, opt, device, a, b, g, d, max_batches=None):
    model.train()
    for step, (x, y) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        p = F.relu(model(x))
        loss, _ = pafr_loss(p, y, a, b, g, d)
        loss.backward()
        opt.step()

@torch.no_grad()
def evaluate_metrics(model, device, loader, max_batches=None):
    model.eval()
    mae_list, en_list, psnr_list, ssim_list = [], [], [], []

    for step, (x, y) in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            break

        x, y = x.to(device), y.to(device)
        p = F.relu(model(x))

        p_np, y_np = p.cpu().numpy(), y.cpu().numpy()

        mae = torch.abs(p - y).mean().item()
        en = torch.abs(p.sum() - y.sum()) / (y.sum() + 1e-6) * 100.0

        mse = ((p_np - y_np) ** 2).mean()
        vmax = y_np.max() if y_np.max() > 0 else 1.0
        psnr = 10 * np.log10(vmax ** 2 / (mse + 1e-12))

        dr = float(y_np.max()) if float(y_np.max()) > 0 else 1.0
        batch_ssim = 0.0
        for i in range(p_np.shape[0]):
            batch_ssim += ssim(y_np[i, 0], p_np[i, 0], data_range=dr)

        mae_list.append(float(mae))
        en_list.append(float(en.item()))
        psnr_list.append(float(psnr))
        ssim_list.append(float(batch_ssim / p_np.shape[0]))

    return {"mae": mae_list, "energy": en_list, "psnr": psnr_list, "ssim": ssim_list}

def format_mean_std(lst, fmt="{:.2f}"):
    return f"{fmt.format(np.mean(lst))} +/- {fmt.format(np.std(lst))}"

def print_result_block(backbone, loss_name, res):
    print(f"[{backbone} - {loss_name}]")
    print(f"  MAE:        {np.mean(res['mae']):.4f} +/- {np.std(res['mae']):.4f}")
    print(f"  Energy Err: {np.mean(res['energy']):.2f}% +/- {np.std(res['energy']):.2f}%")
    print(f"  PSNR:       {np.mean(res['psnr']):.2f} +/- {np.std(res['psnr']):.2f}")
    print(f"  SSIM:       {np.mean(res['ssim']):.4f} +/- {np.std(res['ssim']):.4f}")
    print("-" * 40)