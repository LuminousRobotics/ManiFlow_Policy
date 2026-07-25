"""Lightweight PointNet for the F1-depth branch (DP3-style native 3-D).

Consumes the camera-frame point map the dataset already produces (xyz unprojection of depth;
lumi_place_image_dataset._encode_depth "xyz" -> (X,Y,Z,valid) per pixel) and encodes it as a
POINT CLOUD via a shared per-point MLP + masked max-pool -> one global 3-D token. This is the
NATIVE way (DP3 / 3D Diffusion Policy): explicit metric geometry, NOT depth-as-2D-image-channels
(which our C2 xyz-through-the-2D-CNN run proved the model ignores).

Invalid pixels (valid==0) are excluded from the max-pool. Sub-sampled (stride) for speed:
~4k points is plenty for the rail/tube geometry and keeps it fast + ONNX-friendly.
"""
import torch
import torch.nn as nn


class PointNetEncoder(nn.Module):
    def __init__(self, out_dim: int, in_dim: int = 3, hidden: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(),
            nn.Linear(64, hidden), nn.GELU(),
            nn.Linear(hidden, out_dim))
        self.out_dim = out_dim

    def forward(self, pts: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """pts (B,P,C) camera-frame points, valid (B,P) in {0,1} -> (B,out_dim) global feature.
        Invalid points are excluded from the max-pool; an all-invalid sample yields zeros."""
        feat = self.mlp(pts)                                   # (B,P,out)
        m = valid.unsqueeze(-1) > 0.5                          # (B,P,1)
        feat = feat.masked_fill(~m, float('-inf'))
        pooled = feat.max(dim=1).values                        # (B,out)
        # all-invalid rows -> -inf -> replace with 0 (no valid geometry this frame)
        pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        return pooled


def cloud_from_pointmap(depth_points: torch.Tensor, stride: int = 4):
    """depth_points (B,C,H,W) with C>=4 = [X,Y,Z,valid,...] (camera frame, normalized) ->
    (pts (B,P,3), valid (B,P)) sub-sampled by `stride`. P = ceil(H/stride)*ceil(W/stride)."""
    dp = depth_points[:, :, ::stride, ::stride]                # (B,C,h,w)
    B, C, h, w = dp.shape
    dp = dp.reshape(B, C, h * w).transpose(1, 2)               # (B,P,C)
    pts = dp[..., :3]
    valid = dp[..., 3]
    return pts, valid
