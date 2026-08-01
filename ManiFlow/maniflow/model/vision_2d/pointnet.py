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

    def forward_tokens(self, pts: torch.Tensor, valid: torch.Tensor,
                       hw, grid: int = 4) -> torch.Tensor:
        """H (contract §4.2): K = grid*grid TOKENS instead of one global vector.

        pts (B,P,C) / valid (B,P) come from `cloud_from_pointmap` in ROW-MAJOR (h,w) order;
        `hw` = (h,w) of that sub-sampled map. The map is split into a fixed grid x grid SPATIAL
        partition and the masked max-pool runs PER GROUP -> (B, grid*grid, out_dim).

        Why a fixed spatial grid and not FPS: pytorch3d is not in the training image (and FPS
        is data-dependent, i.e. a nightmare for a static ONNX graph). A 4x4 image-space
        partition is deterministic, ONNX-exportable, and — because the cloud IS a point MAP —
        already a coarse 3-D partition. One global token collapsed the whole scene into a
        single vector (the v1 1-token lesson); 16 gives the cross-attention something to
        localize with. An all-invalid group yields the zero vector (existing behaviour).
        """
        h, w = int(hw[0]), int(hw[1])
        B, P, _ = pts.shape
        assert P == h * w, f"pts {P} != h*w {h}*{w}"
        assert h % grid == 0 and w % grid == 0, (
            f"point-map {h}x{w} must divide the {grid}x{grid} token grid; adjust the PointNet "
            f"stride or pointnet_tokens")
        gh, gw = h // grid, w // grid
        feat = self.mlp(pts)                                   # (B,P,out)
        D = feat.shape[-1]
        m = (valid > 0.5).unsqueeze(-1)                        # (B,P,1)
        feat = feat.masked_fill(~m, float('-inf'))
        # (B,h,w,D) -> (B,grid,gh,grid,gw,D) -> (B,grid*grid,gh*gw,D) -> max over the group
        feat = feat.reshape(B, grid, gh, grid, gw, D).permute(0, 1, 3, 2, 4, 5)
        feat = feat.reshape(B, grid * grid, gh * gw, D)
        pooled = feat.max(dim=2).values                        # (B,K,D)
        return torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))


def cloud_from_pointmap(depth_points: torch.Tensor, stride: int = 4, return_hw: bool = False):
    """depth_points (B,C,H,W) with C>=4 = [X,Y,Z,valid,...] (camera frame, normalized) ->
    (pts (B,P,3), valid (B,P)) sub-sampled by `stride`, ROW-MAJOR over the (h,w) sub-map.
    P = ceil(H/stride)*ceil(W/stride). `return_hw` additionally yields (h,w) — needed by
    `forward_tokens` to recover the spatial layout for the grid pooling."""
    dp = depth_points[:, :, ::stride, ::stride]                # (B,C,h,w)
    B, C, h, w = dp.shape
    dp = dp.reshape(B, C, h * w).transpose(1, 2)               # (B,P,C)
    pts = dp[..., :3]
    valid = dp[..., 3]
    if return_hw:
        return pts, valid, (h, w)
    return pts, valid
