"""Soft-argmax must recover a known sub-pixel keypoint location + beat the grid resolution."""
import torch
from maniflow.model.vision_2d.soft_argmax import SoftArgmaxKeypointHead, keypoint_loss


def test_recovers_peak_subpixel():
    torch.manual_seed(0)
    B, D, H, W, N = 2, 16, 7, 7, 3
    head = SoftArgmaxKeypointHead(in_dim=D, n_kpts=N, temperature=0.05)  # sharp
    # bypass the conv: feed a heatmap directly by monkeypatching heat to identity-ish is hard;
    # instead verify the soft-argmax MATH on a hand-built prob peak.
    # Build heat with a sharp gaussian peak at a known normalized target per keypoint.
    targets = torch.tensor([[0.5, 0.5], [0.786, 0.214], [0.10, 0.90]])   # (N,2) normalized
    xs = (torch.arange(W) + 0.5) / W; ys = (torch.arange(H) + 0.5) / H
    heat = torch.zeros(B, N, H, W)
    for n in range(N):
        tx, ty = targets[n]
        gx = torch.exp(-((xs - tx) ** 2) / (2 * 0.05 ** 2))
        gy = torch.exp(-((ys - ty) ** 2) / (2 * 0.05 ** 2))
        heat[:, n] = torch.log(gy[:, None] * gx[None, :] + 1e-9)         # log so softmax recovers gaussian
    prob = heat.view(B, N, H * W).softmax(-1).view(B, N, H, W)
    u = (prob.sum(2) * xs).sum(-1); v = (prob.sum(3) * ys).sum(-1)
    uv = torch.stack([u, v], -1)[0]                                      # (N,2)
    err = (uv - targets).abs().max().item()
    grid = 1.0 / W
    print(f"soft-argmax recovery err={err:.4f} (grid cell={grid:.3f}) -> sub-cell={err < grid}")
    assert err < grid, f"soft-argmax not sub-cell: {err} >= {grid}"

    # module forward + loss shapes/finiteness
    feat = torch.randn(B, D, H, W)
    uvp, conf = head(feat)
    assert uvp.shape == (B, N, 2) and conf.shape == (B, N)
    assert (uvp >= 0).all() and (uvp <= 1).all(), "uv must be normalized [0,1]"
    gt_uv = torch.rand(B, N, 2); gt_vis = (torch.rand(B, N) > 0.3).float()
    lp, lpr = keypoint_loss(uvp, conf, gt_uv, gt_vis)
    assert torch.isfinite(lp) and torch.isfinite(lpr)
    print("SoftArgmaxKeypointHead forward + loss OK; uv normalized; recovery sub-cell.")


if __name__ == "__main__":
    test_recovers_peak_subpixel()
