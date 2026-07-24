"""GPU-side batch augmentation for the Lumi place policy (TRAIN ONLY).

Runs the label-consistent SO(2) optical-axis roll + photometric (RGB) + sensor perturbation
(depth) on the CUDA batch inside the training loop (before the policy normalizer), so the L40S
(idle ~95% under CPU-aug runs) does the work and the DataLoader workers stop starving it.

Co-rotated together under ONE per-batch angle phi (so nothing desyncs):
  - head_cam (RGB, BILINEAR), depth (NEAREST)                         [images]
  - action / goal_cam / prev_action  via R_z(SO2_SIGN*phi)           [6-D camera-frame labels]
  - arm_kpts_cam (3-D) via R_z; arm_kpts_uv RE-PROJECTED from the rotated 3-D using cam_k_norm
    (2-D keypoints inherit the validated 3-D sign + handle anisotropic fx!=fy / W!=H correctly)

SIGN SAFETY: SO(2) uses torchvision TF.rotate (== validated CPU path) + R_z(SO2_SIGN*phi),
SO2_SIGN=-1 (tests/test_so2_rotation_aug.py, test_gpu_so2_aug.py). Photometric (RGB) + sensor
perturbation (depth) are label-free and per-sample. Does NOT use grid_sample/kornia (different
rotation conventions would silently break the pinned sign).
"""
import math
import numpy as np
import torch
import torchvision.transforms.functional as TF

SO2_SIGN = -1
_LUMA = (0.299, 0.587, 0.114)


def _rz_transpose(phi_deg, device, dtype):
    a = math.radians(SO2_SIGN * phi_deg)
    ca, sa = math.cos(a), math.sin(a)
    return torch.tensor([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]],
                        device=device, dtype=dtype)


def _corotate6(x, rz_t):
    """x[...,:6] = [pos(3)|rotvec(3)] -> co-rotate both triplets by R_z."""
    out = x.clone()
    out[..., :3] = x[..., :3] @ rz_t
    out[..., 3:6] = x[..., 3:6] @ rz_t
    return out


def _project_norm(cam3d, k):
    """cam3d (...,3) optical-frame metres + k=(fx_n,fy_n,cx_n,cy_n) normalized intrinsics ->
    (...,3) = [u/W, v/H, visible]. visible := in front (Z>0) AND inside the normalized frame."""
    X, Y, Z = cam3d[..., 0], cam3d[..., 1], cam3d[..., 2]
    fxn, fyn, cxn, cyn = k
    Zc = Z.clamp(min=1e-6)
    un = fxn * X / Zc + cxn
    vn = fyn * Y / Zc + cyn
    vis = ((Z > 1e-6) & (un >= 0) & (un <= 1) & (vn >= 0) & (vn <= 1)).to(cam3d.dtype)
    return torch.stack([un.clamp(0, 1), vn.clamp(0, 1), vis], dim=-1)


def _luma(img):
    w = torch.tensor(_LUMA, device=img.device, dtype=img.dtype).view(3, 1, 1)
    return (img * w).sum(dim=-3, keepdim=True)


@torch.no_grad()
def gpu_augment(batch, cfg, rng=None):
    """Augment a TRAIN batch in place on its current device. `cfg` keys mirror the dataset
    augmentation block: rot_aug_deg; photo_brightness/contrast/saturation/hue/blur_p/noise/erase_p;
    depth_rel_noise/depth_px_dropout/depth_flying_px; cam_k_norm=[fx/W,fy/H,cx/W,cy/H] (needed to
    re-project rotated keypoints — required iff arm_kpts_* present and rot_aug_deg>0)."""
    obs = batch['obs']
    hc = obs['head_cam']                       # (B,To,3,H,W) float [0,1]
    device, dtype = hc.device, hc.dtype
    B, To = hc.shape[:2]
    C, H, W = hc.shape[-3:]
    g = rng if rng is not None else torch.Generator(device='cpu')

    def _u(lo, hi, shape=()):
        return (torch.rand(shape, generator=g).to(device) * (hi - lo) + lo)

    depth = obs.get('depth', None)             # raw depth (B,To,1,H,W) if present (F-model)

    # ---- SO(2) optical-axis roll: one phi per batch (validated sign), co-rotates everything ----
    rot = float(cfg.get('rot_aug_deg', 0.0) or 0.0)
    if rot > 0.0:
        phi = float(_u(-rot, rot).item())
        if abs(phi) >= 1e-3:
            hc = TF.rotate(hc.reshape(B * To, C, H, W), angle=phi,
                           interpolation=TF.InterpolationMode.BILINEAR).reshape(B, To, C, H, W)
            if depth is not None:              # NEAREST: never interpolate across depth edges/invalid
                dC = depth.shape[-3]
                depth = TF.rotate(depth.reshape(B * To, dC, H, W), angle=phi,
                                  interpolation=TF.InterpolationMode.NEAREST).reshape(B, To, dC, H, W)
                obs['depth'] = depth
            rz_t = _rz_transpose(phi, device, dtype)
            if 'action' in batch:
                batch['action'] = _corotate6(batch['action'].to(device), rz_t)
            if 'goal_cam' in batch:
                batch['goal_cam'] = _corotate6(batch['goal_cam'].to(device), rz_t)
            if 'prev_action' in obs:
                obs['prev_action'] = _corotate6(obs['prev_action'].to(device), rz_t)
            # keypoints: rotate the 3-D targets by R_z, RE-PROJECT the 2-D (inherits the sign)
            for cam_key, uv_key in (('arm_kpts_cam', 'arm_kpts_uv'),
                                    ('side_kpts_cam', 'side_kpts_uv')):
                if cam_key in batch:
                    c = batch[cam_key].to(device)
                    c_rot = c @ rz_t                                   # (...,3) rotate
                    batch[cam_key] = c_rot
                    k = cfg.get('cam_k_norm', None)
                    if uv_key in batch and k is not None:
                        # keep a point invisible if it was invisible pre-rotation (occlusion GT)
                        prev_vis = batch[uv_key].to(device)[..., 2:3]
                        uv = _project_norm(c_rot, k)
                        uv[..., 2:3] = uv[..., 2:3] * prev_vis
                        batch[uv_key] = uv

    # ---- RGB photometric (label-free), per-sample factor shared across the To frames ----
    def _f(name):
        a = float(cfg.get(name, 0.0) or 0.0)
        if a <= 0.0:
            return None
        return _u(1.0 - a, 1.0 + a, (B,)).view(B, 1, 1, 1, 1)

    fb = _f('photo_brightness')
    if fb is not None:
        hc = hc * fb
    fc = _f('photo_contrast')
    if fc is not None:
        mean = _luma(hc).mean(dim=(-1, -2), keepdim=True)
        hc = (hc - mean) * fc + mean
    fs = _f('photo_saturation')
    if fs is not None:
        gray = _luma(hc)
        hc = (hc - gray) * fs + gray
    hue = float(cfg.get('photo_hue', 0.0) or 0.0)
    if hue > 0.0:
        h = max(-0.5, min(0.5, float(_u(-hue, hue).item())))
        hc = TF.adjust_hue(hc.reshape(B * To, C, H, W).clamp(0.0, 1.0), h).reshape(B, To, C, H, W)
    blur_p = float(cfg.get('photo_blur_p', 0.0) or 0.0)
    if blur_p > 0.0 and float(_u(0.0, 1.0).item()) < blur_p:
        sigma = float(_u(0.4, 1.2).item())
        hc = TF.gaussian_blur(hc.reshape(B * To, C, H, W), kernel_size=5, sigma=sigma).reshape(B, To, C, H, W)
    noise = float(cfg.get('photo_noise', 0.0) or 0.0)
    if noise > 0.0:
        hc = hc + torch.randn(hc.shape, generator=g).to(device) * noise
    hc = hc.clamp(0.0, 1.0)
    # random-erase: up to 2 boxes/sample, VECTORIZED (no per-sample Python loop / .item() syncs,
    # which serialize the GPU and were the throughput bottleneck). Per-sample box params -> a
    # broadcasted mask -> torch.where. All on-device, one shot.
    erase_p = float(cfg.get('photo_erase_p', 0.0) or 0.0)
    if erase_p > 0.0:
        yy = torch.arange(H, device=device).view(1, H, 1)
        xx = torch.arange(W, device=device).view(1, 1, W)
        for _ in range(2):
            do = (_u(0.0, 1.0, (B,)) < erase_p)                       # (B,)
            eh = (H * _u(0.05, 0.20, (B,))).clamp(min=1).long()       # (B,)
            ew = (W * _u(0.05, 0.20, (B,))).clamp(min=1).long()
            y0 = (_u(0.0, 1.0, (B,)) * (H - eh).clamp(min=1).float()).long()
            x0 = (_u(0.0, 1.0, (B,)) * (W - ew).clamp(min=1).float()).long()
            ymask = (yy >= y0.view(B, 1, 1)) & (yy < (y0 + eh).view(B, 1, 1))   # (B,H,1)
            xmask = (xx >= x0.view(B, 1, 1)) & (xx < (x0 + ew).view(B, 1, 1))   # (B,1,W)
            box = (ymask & xmask).view(B, 1, 1, H, W) & do.view(B, 1, 1, 1, 1)
            fill = _u(0.0, 1.0, (B,)).view(B, 1, 1, 1, 1).expand_as(hc)
            hc = torch.where(box, fill, hc)
    obs['head_cam'] = hc

    # ---- DEPTH sensor perturbation (label-free; scale-free so it works in mm or m) ----
    # Models real Gemini-336 noise on the matte rail/tube cloud: relative axial noise (grows with
    # range), random pixel dropout, and flying pixels. Invalid (==0) pixels stay invalid.
    if depth is not None:
        d = obs['depth']
        valid = d > 0
        rel = float(cfg.get('depth_rel_noise', 0.0) or 0.0)
        if rel > 0.0:                          # multiplicative axial noise ~ N(1, rel)
            d = d * (1.0 + torch.randn(d.shape, generator=g).to(device) * rel)
        dp = float(cfg.get('depth_px_dropout', 0.0) or 0.0)
        if dp > 0.0:
            d = torch.where(torch.rand(d.shape, generator=g).to(device) < dp, torch.zeros_like(d), d)
        fp = float(cfg.get('depth_flying_px', 0.0) or 0.0)
        if fp > 0.0 and valid.any():
            dmax = float(d[valid].max())
            fly = torch.rand(d.shape, generator=g).to(device) < fp
            d = torch.where(fly, _u(0.03 * dmax, dmax, d.shape), d)
        d = torch.where(valid, d.clamp(min=0.0), torch.zeros_like(d))  # keep invalid==0
        obs['depth'] = d

    return batch
