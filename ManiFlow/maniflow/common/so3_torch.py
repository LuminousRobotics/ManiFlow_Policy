"""Batched, differentiable SO(3) helpers for the H-series goal-frame machinery.

Used ONLY on the training path (goal-frame scheduled sampling, the goal-coincidence integral
constraint and the goal-consistency loss) — the exported ONNX graph never touches these. They
are nevertheless written branch-free (no data-dependent Python control flow) so they stay
trace-safe if a future arm needs them at inference.

Convention parity: these mirror `maniflow.dataset.lumi_place_image_dataset.rotvec_to_rotmat` /
`rotmat_to_rotvec` (themselves byte-parity twins of the converter's transforms.py). The numpy
twins are the reference; `tests/test_h_geometry_parity.py` pins torch against them.
"""
import torch


def rotvec_to_rotmat(rv: torch.Tensor) -> torch.Tensor:
    """(...,3) rotation vector -> (...,3,3) rotation matrix (Rodrigues).

    Small angles: the sin(a)/a and (1-cos a)/a^2 forms are used so the a -> 0 limit is exact and
    the gradient stays finite (the naive axis = rv/|rv| formulation has a 0/0 there)."""
    a2 = (rv * rv).sum(-1, keepdim=True).clamp(min=1e-24)
    a = a2.sqrt()                                            # (...,1)
    s = torch.sin(a) / a                                     # -> 1
    c = (1.0 - torch.cos(a)) / a2                            # -> 1/2
    x, y, z = rv[..., 0], rv[..., 1], rv[..., 2]
    zero = torch.zeros_like(x)
    K = torch.stack([torch.stack([zero, -z, y], -1),
                     torch.stack([z, zero, -x], -1),
                     torch.stack([-y, x, zero], -1)], -2)     # (...,3,3) skew
    eye = torch.eye(3, dtype=rv.dtype, device=rv.device).expand_as(K)
    return eye + s.unsqueeze(-1) * K + c.unsqueeze(-1) * (K @ K)


def rotmat_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """(...,3,3) rotation matrix -> (...,3) rotation vector.

    Uses the skew part scaled by angle/(2 sin angle). Valid for |angle| < pi - eps, which every
    H quantity satisfies by construction (post-yaw-skip total rotation is ~9 deg, and the
    per-sample goal-frame noise is 1.5 deg). The trace is clamped so acos never sees >1 from
    fp round-off, and sin is clamped away from 0 so the small-angle limit degrades to the
    first-order skew form rather than NaN."""
    tr = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1.0) * 0.5
    angle = torch.acos(tr.clamp(-1.0 + 1e-7, 1.0 - 1e-7))
    skew = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                        R[..., 0, 2] - R[..., 2, 0],
                        R[..., 1, 0] - R[..., 0, 1]], dim=-1)
    scale = angle / (2.0 * torch.sin(angle).clamp(min=1e-7))
    # angle -> 0: scale -> 1/2, which is exactly the first-order skew form.
    scale = torch.where(angle < 1e-5, torch.full_like(scale, 0.5), scale)
    return skew * scale.unsqueeze(-1)


def compose_rotvecs(rotvecs: torch.Tensor) -> torch.Tensor:
    """(...,K,3) rotation vectors applied IN ORDER -> (...,3) composed rotation vector.

    Angular-velocity rows integrate by COMPOSITION (prod exp(w_k dt)), not summation. K
    sequential matmuls; K = 15 for H so the chain is cheap. Mirrors the numpy twin."""
    Rs = rotvec_to_rotmat(rotvecs)                            # (...,K,3,3)
    acc = Rs[..., 0, :, :]
    for k in range(1, Rs.shape[-3]):
        acc = Rs[..., k, :, :] @ acc
    return rotmat_to_rotvec(acc)
