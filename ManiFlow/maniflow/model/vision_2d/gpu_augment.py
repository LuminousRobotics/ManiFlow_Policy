"""GPU-side batch augmentation for the Lumi place policy (TRAIN ONLY).

Runs the label-consistent SO(2) optical-axis roll + photometric appearance aug on the
CUDA batch inside the training loop (before the policy normalizer), so the L40S (idle ~95%
under CPU-aug runs) does the work and the DataLoader workers stop starving it. Applied to the
RAW batch: head_cam in [0,1], and the camera-frame 6-D labels action / goal_cam / prev_action.

SIGN SAFETY: the SO(2) rotation uses the SAME primitive as the validated CPU path
(torchvision TF.rotate) and the SAME label co-rotation R_z(SO2_SIGN*phi) with SO2_SIGN=-1
(pinned + unit-tested in tests/test_so2_rotation_aug.py). A single phi PER BATCH (not
per-sample) keeps it a one-shot TF.rotate + one matmul — a small aug-diversity trade for a
zero-recalibration guarantee that the labels stay consistent with the rotated pixels.
Photometric ops are label-free and applied PER-SAMPLE.

NB: this deliberately does NOT re-implement rotation via grid_sample/kornia — those have
different center/direction conventions that would silently break the pinned sign.
"""
import math
import numpy as np
import torch
import torchvision.transforms.functional as TF

SO2_SIGN = -1  # matches lumi_place_image_dataset._augment_so2 (TF.rotate(+phi) == camera R_z(-phi))
_LUMA = (0.299, 0.587, 0.114)


def _rz_transpose(phi_deg, device, dtype):
    """R_z(SO2_SIGN*phi).T so that v' = v @ Rz_T co-rotates row-vectors (matches CPU _co)."""
    a = math.radians(SO2_SIGN * phi_deg)
    ca, sa = math.cos(a), math.sin(a)
    return torch.tensor([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]],
                        device=device, dtype=dtype)


def _corotate(x, rz_t):
    """x[..., :6] camera-frame [pos(3)|rotvec(3)] -> co-rotate both triplets by R_z."""
    out = x.clone()
    out[..., :3] = x[..., :3] @ rz_t
    out[..., 3:6] = x[..., 3:6] @ rz_t
    return out


def _luma(img):
    """(...,3,H,W) -> (...,1,H,W) Rec.601 luma (matches torchvision grayscale weights)."""
    w = torch.tensor(_LUMA, device=img.device, dtype=img.dtype).view(3, 1, 1)
    return (img * w).sum(dim=-3, keepdim=True)


@torch.no_grad()
def gpu_augment(batch, cfg, rng=None):
    """Augment a TRAIN batch in place on its current device. `cfg` is a dict with the same
    keys as the dataset augmentation block: rot_aug_deg, photo_brightness/contrast/saturation/
    hue, photo_blur_p, photo_noise, photo_erase_p. Returns the batch."""
    obs = batch['obs']
    hc = obs['head_cam']                       # (B,To,3,H,W) float in [0,1]
    device, dtype = hc.device, hc.dtype
    B, To = hc.shape[:2]
    C, H, W = hc.shape[-3:]
    g = rng if rng is not None else torch.Generator(device='cpu')

    def _u(lo, hi, shape=()):                  # uniform sample on CPU gen -> device (reproducible-ish)
        return (torch.rand(shape, generator=g).to(device) * (hi - lo) + lo)

    # ---- SO(2) optical-axis roll: one phi per batch (validated sign) ----
    rot = float(cfg.get('rot_aug_deg', 0.0) or 0.0)
    if rot > 0.0:
        phi = float(_u(-rot, rot).item())
        if abs(phi) >= 1e-3:
            flat = hc.reshape(B * To, C, H, W)
            flat = TF.rotate(flat, angle=phi, interpolation=TF.InterpolationMode.BILINEAR)
            hc = flat.reshape(B, To, C, H, W)
            rz_t = _rz_transpose(phi, device, dtype)
            if 'action' in batch:
                batch['action'] = _corotate(batch['action'].to(device), rz_t)
            if 'goal_cam' in batch:
                batch['goal_cam'] = _corotate(batch['goal_cam'].to(device), rz_t)
            if 'prev_action' in obs:
                obs['prev_action'] = _corotate(obs['prev_action'].to(device), rz_t)

    # ---- photometric (label-free), per-sample factor shared across the To frames ----
    def _f(name):                              # per-sample factor (B,1,1,1,1) in [1-a,1+a]
        a = float(cfg.get(name, 0.0) or 0.0)
        if a <= 0.0:
            return None
        return _u(1.0 - a, 1.0 + a, (B,)).view(B, 1, 1, 1, 1)

    fb = _f('photo_brightness')
    if fb is not None:
        hc = hc * fb
    fc = _f('photo_contrast')
    if fc is not None:
        mean = _luma(hc).mean(dim=(-1, -2), keepdim=True)   # (B,To,1,1,1) per-image gray mean
        hc = (hc - mean) * fc + mean
    fs = _f('photo_saturation')
    if fs is not None:
        gray = _luma(hc)                                    # (B,To,1,H,W) broadcast over channels
        hc = (hc - gray) * fs + gray

    # hue: per-batch (HSV conversion is the pricey op; one shift for the batch is enough)
    hue = float(cfg.get('photo_hue', 0.0) or 0.0)
    if hue > 0.0:
        h = float(_u(-hue, hue).item())
        h = max(-0.5, min(0.5, h))
        hc = TF.adjust_hue(hc.reshape(B * To, C, H, W).clamp(0.0, 1.0), h).reshape(B, To, C, H, W)

    # gaussian blur: per-batch, probabilistic
    blur_p = float(cfg.get('photo_blur_p', 0.0) or 0.0)
    if blur_p > 0.0 and float(_u(0.0, 1.0).item()) < blur_p:
        sigma = float(_u(0.4, 1.2).item())
        hc = TF.gaussian_blur(hc.reshape(B * To, C, H, W), kernel_size=5, sigma=sigma).reshape(B, To, C, H, W)

    # gaussian pixel noise, per-sample
    noise = float(cfg.get('photo_noise', 0.0) or 0.0)
    if noise > 0.0:
        hc = hc + torch.randn(hc.shape, generator=g).to(device) * noise

    hc = hc.clamp(0.0, 1.0)

    # random-erase: up to 2 boxes per sample, filled with a random gray (occlusion/glare)
    erase_p = float(cfg.get('photo_erase_p', 0.0) or 0.0)
    if erase_p > 0.0:
        for b in range(B):
            for _ in range(2):
                if float(_u(0.0, 1.0).item()) >= erase_p:
                    continue
                eh = int(H * float(_u(0.05, 0.20).item()))
                ew = int(W * float(_u(0.05, 0.20).item()))
                if eh < 1 or ew < 1:
                    continue
                y0 = int(torch.randint(0, max(1, H - eh + 1), (1,), generator=g).item())
                x0 = int(torch.randint(0, max(1, W - ew + 1), (1,), generator=g).item())
                hc[b, :, :, y0:y0 + eh, x0:x0 + ew] = float(_u(0.0, 1.0).item())

    obs['head_cam'] = hc
    return batch
