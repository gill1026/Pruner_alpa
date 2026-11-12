"""Reconstruction loss from FastDriveVLA: L = α·L_fore + (1−α)·L_back, each λ(1−SSIM) + (1−λ)MSE.

SSIM settings are not given in the paper; we use the standard 11x11 Gaussian window, σ=1.5, data range 1.
"""
import torch
import torch.nn.functional as F


def _gauss(win, sigma, device, dtype):
    x = torch.arange(win, device=device, dtype=dtype) - win // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None, :])[None, None].expand(3, 1, win, win).contiguous()


def ssim(x, y, win=11, sigma=1.5):
    """x, y: (N, 3, H, W) in [0, 1]. Returns per-image SSIM (N,)."""
    k = _gauss(win, sigma, x.device, x.dtype)
    mu_x, mu_y = F.conv2d(x, k, groups=3), F.conv2d(y, k, groups=3)
    sxx = F.conv2d(x * x, k, groups=3) - mu_x ** 2
    syy = F.conv2d(y * y, k, groups=3) - mu_y ** 2
    sxy = F.conv2d(x * y, k, groups=3) - mu_x * mu_y
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2))
    return s.mean(dim=(1, 2, 3))


def recon_loss(pred, gt, lam=0.2):
    """Per-image loss and its parts. pred, gt: (N, 3, H, W)."""
    pred, gt = pred.float(), gt.float()
    mse = ((pred - gt) ** 2).mean(dim=(1, 2, 3))
    s = ssim(pred, gt)
    return lam * (1 - s) + (1 - lam) * mse, mse, s


def total_loss(pred_fore, pred_back, gt_fore, gt_back, valid, alpha=0.5, lam=0.2):
    """valid: (N,) bool, images excluded from the loss (e.g. no road detected) are False."""
    lf, mse_f, ssim_f = recon_loss(pred_fore, gt_fore, lam)
    lb, mse_b, ssim_b = recon_loss(pred_back, gt_back, lam)
    w = valid.float()
    n = w.sum().clamp(min=1)
    avg = lambda t: (t * w).sum() / n
    loss = alpha * avg(lf) + (1 - alpha) * avg(lb)
    return loss, dict(loss=loss.item(), l_fore=avg(lf).item(), l_back=avg(lb).item(),
                      mse_fore=avg(mse_f).item(), mse_back=avg(mse_b).item(),
                      ssim_fore=avg(ssim_f).item(), ssim_back=avg(ssim_b).item())
