"""High-resolution heatmap keypoint head (SimpleBaseline deconv, Xiao 2018 arXiv:1804.06208)
+ per-keypoint metric log-z + backprojection to camera-frame 3-D points.

Replaces the 7x7 soft-argmax head (v0.1.16/19): the measured 34-58px keypoint error was the
7x7 representational ceiling (183px/cell @1280), not a soft-argmax defect. This head deconvs
the encoder's final map up to 56x56 (stride-4-equivalent), supervises with a Gaussian-target
cross-entropy (the ViTPose/SimpleBaseline recipe), and decodes with a WINDOWED soft-argmax
(expectation over the 5x5 around the argmax) which removes the background-mass bias of full-map
soft-argmax (Gu ICCV 2021). Range comes from a per-keypoint log-z MLP on features sampled at
the predicted (u,v) — supervised by the projected-GT camera-frame Z (arm_kpts_cam) — because
rail keypoints are near-collinear and uv-spread ranging is PnP-degenerate. All ops (deconv,
softmax, argmax->mask expectation, grid_sample, exp) are ONNX-exportable.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeatmapKeypointHead(nn.Module):
    """(B,D,g,g) encoder map -> N keypoints: normalized uv in [0,1], presence logit,
    metric z (metres) and camera-frame 3-D points via intrinsics backprojection."""

    def __init__(self, in_dim: int, n_kpts: int, hidden: int = 128, n_deconv: int = 3,
                 window: int = 5, z_max_m: float = 4.0):
        super().__init__()
        self.n_kpts = n_kpts
        self.window = window
        self.z_max_m = z_max_m
        layers, d = [], in_dim
        for _ in range(n_deconv):                       # 7 -> 14 -> 28 -> 56
            layers += [nn.ConvTranspose2d(d, hidden, 4, stride=2, padding=1),
                       nn.GroupNorm(8, hidden), nn.SiLU()]
            d = hidden
        self.deconv = nn.Sequential(*layers)
        self.heat = nn.Conv2d(hidden, n_kpts, 1)
        self.presence = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden, n_kpts))
        # per-keypoint log-z from decoder features sampled at the predicted uv
        # input: hidden feat + (u,v) + [depth_sample, depth_valid] (zeros when no depth map)
        self.z_mlp = nn.Sequential(nn.Linear(hidden + 4, 64), nn.SiLU(), nn.Linear(64, 1))

    def forward(self, feat: torch.Tensor, cam_k_norm: torch.Tensor = None,
                depth_z: torch.Tensor = None):
        """feat (B,D,g,g); cam_k_norm (4,) [fx,fy,cx,cy] normalized by image size;
        depth_z optional (B,2,H,W) [z_metres, valid] point-map channels for z fusion.

        Returns dict: uv (B,N,2) in [0,1], conf (B,N) logits, heat_logits (B,N,Hh,Wh),
        z (B,N) metres, pts3d (B,N,3) camera-frame metres (zeros if cam_k_norm None)."""
        B = feat.shape[0]
        dec = self.deconv(feat)                                      # (B,hidden,56,56)
        logits = self.heat(dec)                                      # (B,N,56,56)
        Hh, Wh = logits.shape[-2:]
        prob = logits.reshape(B, self.n_kpts, -1).softmax(-1).reshape(B, self.n_kpts, Hh, Wh)

        # ---- windowed soft-argmax (5x5 around argmax, renormalized expectation) ----
        flat_idx = prob.reshape(B, self.n_kpts, -1).argmax(-1)       # (B,N)
        iy = (flat_idx // Wh).float(); ix = (flat_idx % Wh).float()
        gy = torch.arange(Hh, device=feat.device, dtype=prob.dtype).view(1, 1, Hh, 1)
        gx = torch.arange(Wh, device=feat.device, dtype=prob.dtype).view(1, 1, 1, Wh)
        r = (self.window - 1) // 2
        mask = ((gy - iy.view(B, self.n_kpts, 1, 1)).abs() <= r) & \
               ((gx - ix.view(B, self.n_kpts, 1, 1)).abs() <= r)
        pw = prob * mask
        pw = pw / pw.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)
        u = (pw.sum(2) * ((torch.arange(Wh, device=feat.device, dtype=prob.dtype) + 0.5) / Wh)).sum(-1)
        v = (pw.sum(3) * ((torch.arange(Hh, device=feat.device, dtype=prob.dtype) + 0.5) / Hh)).sum(-1)
        uv = torch.stack([u, v], dim=-1)                             # (B,N,2) in [0,1]
        conf = self.presence(dec)                                    # (B,N)

        # ---- per-keypoint feature sample at uv -> log-z ----
        grid = (uv * 2.0 - 1.0).unsqueeze(2)                         # (B,N,1,2) for grid_sample
        fs = F.grid_sample(dec, grid, mode="bilinear", align_corners=False)  # (B,hidden,N,1)
        fs = fs.squeeze(-1).transpose(1, 2)                          # (B,N,hidden)
        if depth_z is not None:                                      # median-free but VALID-gated
            ds = F.grid_sample(depth_z, grid, mode="bilinear", align_corners=False)
            ds = ds.squeeze(-1).transpose(1, 2)                      # (B,N,2) [z, valid]
        else:
            ds = torch.zeros(B, self.n_kpts, 2, device=feat.device, dtype=feat.dtype)
        z = torch.exp(self.z_mlp(torch.cat([fs, uv, ds], dim=-1)).squeeze(-1))  # (B,N) metres
        z = z.clamp(max=self.z_max_m)

        if cam_k_norm is not None:
            k = cam_k_norm.reshape(-1, 4).to(feat.dtype)             # (1,4) static or (B,4) per-sample
            fx, fy, cx, cy = k[:, 0:1], k[:, 1:2], k[:, 2:3], k[:, 3:4]   # broadcast over N
            X = (uv[..., 0] - cx) / fx.clamp(min=1e-6) * z
            Y = (uv[..., 1] - cy) / fy.clamp(min=1e-6) * z
            pts3d = torch.stack([X, Y, z], dim=-1)                   # (B,N,3) camera frame, metres
        else:
            pts3d = torch.zeros(B, self.n_kpts, 3, device=feat.device, dtype=feat.dtype)
        return {"uv": uv, "conf": conf, "heat_logits": logits, "z": z, "pts3d": pts3d}


def gaussian_heatmap_targets(gt_uv: torch.Tensor, H: int, W: int, sigma_cells: float = 1.5):
    """gt_uv (B,N,2) normalized -> (B,N,H,W) Gaussian target distributions (sum=1 per kpt)."""
    B, N = gt_uv.shape[:2]
    gy = (torch.arange(H, device=gt_uv.device, dtype=gt_uv.dtype) + 0.5) / H
    gx = (torch.arange(W, device=gt_uv.device, dtype=gt_uv.dtype) + 0.5) / W
    dy = (gy.view(1, 1, H, 1) - gt_uv[..., 1].view(B, N, 1, 1)) * H
    dx = (gx.view(1, 1, 1, W) - gt_uv[..., 0].view(B, N, 1, 1)) * W
    t = torch.exp(-(dx * dx + dy * dy) / (2.0 * sigma_cells ** 2))
    return t / t.sum(dim=(-2, -1), keepdim=True).clamp(min=1e-8)


def heatmap_kpt_loss(out: dict, gt_uv: torch.Tensor, gt_vis: torch.Tensor,
                     gt_z: torch.Tensor = None, sigma_cells: float = 1.5):
    """Gaussian-target cross-entropy on heatmaps (vis-masked) + presence BCE + log-z smooth-L1.
    out = HeatmapKeypointHead forward dict; gt_uv (B,N,2); gt_vis (B,N) {0,1}; gt_z (B,N) metres.
    Returns (loss_heat, loss_pres, loss_z)."""
    logits = out["heat_logits"]
    B, N, H, W = logits.shape
    logp = logits.reshape(B, N, -1).log_softmax(-1)
    tgt = gaussian_heatmap_targets(gt_uv, H, W, sigma_cells).reshape(B, N, -1)
    ce = -(tgt * logp).sum(-1)                                       # (B,N)
    loss_heat = (ce * gt_vis).sum() / gt_vis.sum().clamp(min=1e-6)
    loss_pres = F.binary_cross_entropy_with_logits(out["conf"], gt_vis)
    if gt_z is not None:
        vz = gt_vis * (gt_z > 1e-3).to(gt_vis.dtype)                 # z supervision needs valid GT
        lz = F.smooth_l1_loss(torch.log(out["z"].clamp(min=1e-3)),
                              torch.log(gt_z.clamp(min=1e-3)), reduction="none")
        loss_z = (lz * vz).sum() / vz.sum().clamp(min=1e-6)
    else:
        loss_z = logits.new_zeros(())
    return loss_heat, loss_pres, loss_z
