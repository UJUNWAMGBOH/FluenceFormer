import torch
import torch.nn.functional as F

def corr_loss(pred, target, eps=1e-8):
    B = pred.shape[0]
    p_flat = pred.view(B, -1)
    t_flat = target.view(B, -1)
    p_mean = p_flat.mean(1, keepdim=True)
    t_mean = t_flat.mean(1, keepdim=True)

    num = ((p_flat - p_mean) * (t_flat - t_mean)).sum(1)
    den = torch.sqrt(((p_flat - p_mean) ** 2).sum(1) * ((t_flat - t_mean) ** 2).sum(1) + eps)
    return 1.0 - (num / (den + eps)).mean()

def pafr_loss(pred, target, alpha, beta, gamma, delta):
    l_mse = F.mse_loss(pred, target)

    g_x = torch.abs(pred[:, :, 1:] - pred[:, :, :-1]) - torch.abs(target[:, :, 1:] - target[:, :, :-1])
    g_y = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]) - torch.abs(target[:, :, :, 1:] - target[:, :, :, :-1])
    l_grad = g_x.abs().mean() + g_y.abs().mean()

    l_corr = corr_loss(pred, target)

    l_energy = torch.abs(pred.flatten(1).sum(1) - target.flatten(1).sum(1)).mean() / (128 * 128)

    total = alpha * l_mse + beta * l_grad + gamma * l_corr + delta * l_energy
    return total, {"mse": l_mse.item(), "total": total.item()}