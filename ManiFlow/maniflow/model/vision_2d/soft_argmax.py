"""Spatial soft-argmax keypoint head (integral regression; Sun et al. 2018, arXiv:1711.08229).

Takes a dense feature map (B,D,H,W) -> N keypoints as NORMALIZED (u,v) in [0,1] via a per-keypoint
heatmap + soft-argmax (probability-weighted spatial expectation). Sub-pixel and resolution-
INDEPENDENT (the expectation interpolates between cells), which is exactly the property our pooled
goal head lacked (the ~15 mm plateau). Also emits a per-keypoint presence logit.

Trained against projected-GT keypoints (converter arm_kpts_uv: [u/W, v/H, visible]), visibility-
masked. Runs on raw pixels at inference — no external module. The place pose is derived from the
keypoints downstream (goal head), so this is the learned "find the tube/hatrail in RGB".
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftArgmaxKeypointHead(nn.Module):
    def __init__(self, in_dim: int, n_kpts: int, hidden: int = 128, temperature: float = 1.0):
        super().__init__()
        self.n_kpts = n_kpts
        self.temperature = temperature
        self.heat = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=1), nn.GELU(),
            nn.Conv2d(hidden, n_kpts, kernel_size=1))
        # presence logit per keypoint (is this keypoint visible in this frame?)
        self.presence = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(hidden, n_kpts))

    def forward(self, feat: torch.Tensor):
        """feat (B,D,H,W) -> uv (B,N,2) normalized [0,1], conf (B,N) presence logits."""
        B, D, H, W = feat.shape
        heat = self.heat(feat) / self.temperature            # (B,N,H,W)
        prob = heat.view(B, self.n_kpts, H * W).softmax(dim=-1).view(B, self.n_kpts, H, W)
        xs = (torch.arange(W, device=feat.device, dtype=feat.dtype) + 0.5) / W   # (W,) cell centers
        ys = (torch.arange(H, device=feat.device, dtype=feat.dtype) + 0.5) / H
        u = (prob.sum(dim=2) * xs).sum(dim=-1)               # sum over H -> (B,N,W) -> weight x -> (B,N)
        v = (prob.sum(dim=3) * ys).sum(dim=-1)               # sum over W -> (B,N,H) -> weight y -> (B,N)
        uv = torch.stack([u, v], dim=-1)                     # (B,N,2)
        conf = self.presence(feat)                           # (B,N)
        return uv, conf


def keypoint_loss(uv_pred, conf_logit, gt_uv, gt_vis, eps: float = 1e-6):
    """uv_pred (B,N,2), conf_logit (B,N); gt_uv (B,N,2), gt_vis (B,N) in {0,1}. Position loss is
    visibility-masked (only supervise where the GT keypoint is in-frame); presence is BCE over all."""
    pos = F.smooth_l1_loss(uv_pred, gt_uv, reduction='none').sum(-1)      # (B,N)
    loss_pos = (pos * gt_vis).sum() / gt_vis.sum().clamp(min=eps)
    loss_pres = F.binary_cross_entropy_with_logits(conf_logit, gt_vis)
    return loss_pos, loss_pres
