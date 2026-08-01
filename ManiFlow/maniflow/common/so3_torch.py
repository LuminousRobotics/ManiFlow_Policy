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


# Angle band (radians below pi) where the SKEW part stops carrying a usable axis and the
# symmetric part takes over. Matches the numpy twin's `angle > np.pi - 1e-6` exactly, so the
# two implementations switch branches on the same inputs and `tests/test_h_geometry_parity.py`
# compares like with like.
_NEAR_PI_BAND = 1e-6


def _axis_near_pi(R: torch.Tensor, skew: torch.Tensor) -> torch.Tensor:
    """Unit rotation axis recovered from the SYMMETRIC part, for angles at/near pi.

    At angle == pi, `R = 2 a aT - I`, so `(R + I)/2 = a aT`: column k of it is `a_k * a`, and
    the best-conditioned column is the one with the largest diagonal `a_k^2`. The skew part is
    `2 sin(angle) * a`, which VANISHES at pi — that is exactly why it cannot supply the axis
    there, and why the plain skew formula collapsed to |out| ~ 0.7 instead of ~pi.
    Written with gather/argmax rather than python indexing so the function stays trace-safe."""
    eye = torch.eye(3, dtype=R.dtype, device=R.device).expand_as(R)
    A = 0.5 * (R + eye)                                       # -> a aT
    d = torch.diagonal(A, dim1=-2, dim2=-1)                   # (...,3) = a_k^2
    k = d.argmax(dim=-1, keepdim=True)                        # (...,1) best-conditioned column
    col = torch.gather(A, -1, k.unsqueeze(-2).expand(*A.shape[:-1], 1)).squeeze(-1)
    axis = col / col.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    # +a*pi and -a*pi are the SAME rotation at exactly pi, but just below pi the skew still
    # carries the sign — follow it so the function is continuous across the branch switch.
    s = torch.sign((axis * skew).sum(-1, keepdim=True))
    return axis * torch.where(s == 0, torch.ones_like(s), s)


def rotmat_to_rotvec(R: torch.Tensor) -> torch.Tensor:
    """(...,3,3) rotation matrix -> (...,3) rotation vector. Valid over the FULL [0, pi] domain.

    The angle comes from `atan2(|skew|/2, (tr-1)/2)` == `atan2(sin a, cos a)`, NOT from
    `acos(cos a)`. This matters: `acos` is ill-conditioned wherever |cos a| -> 1, i.e. at BOTH
    ends of the domain, and the previous `clamp(tr, -1+1e-7, ...)` capped the angle at
    `pi - 4.5e-4` — a matrix 1e-4 from pi came back with |out| = 0.70 and a geodesic error of
    ~pi. `atan2` has no such cap and is exact at 0 and pi.

    The AXIS still comes from the skew part (`2 sin a * a`) everywhere except the last
    `_NEAR_PI_BAND` radians, where sin a -> 0 destroys it and the symmetric part takes over
    (`_axis_near_pi`). In-domain numerics are unchanged in kind: for H's actual quantities
    (post-yaw-skip rotation <= ~9 deg, goal-frame noise 1.5 deg) `atan2` and `acos` agree to
    fp round-off, and the parity test pins that against the numpy twin.

    Reachable near-pi inputs are rare but real: the goal-consistency target composes
    `F_hatT (R_G R_0T) F_hat` from the PLACE HEAD's output, which early in training is garbage
    and can be a half turn from the anchor. A silent collapse there feeds a wrong gradient into
    the very head the loss exists to supervise."""
    skew = torch.stack([R[..., 2, 1] - R[..., 1, 2],
                        R[..., 0, 2] - R[..., 2, 0],
                        R[..., 1, 0] - R[..., 0, 1]], dim=-1)  # = 2 sin(angle) * axis
    sin_a = 0.5 * skew.norm(dim=-1)                            # |sin(angle)|, >= 0
    cos_a = ((R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]) - 1.0) * 0.5
    angle = torch.atan2(sin_a, cos_a)                          # in [0, pi], stable at both ends
    scale = angle / (2.0 * sin_a.clamp(min=1e-7))
    # angle -> 0: scale -> 1/2, which is exactly the first-order skew form.
    scale = torch.where(angle < 1e-5, torch.full_like(scale, 0.5), scale)
    out = skew * scale.unsqueeze(-1)
    near_pi = (angle > (torch.pi - _NEAR_PI_BAND)).unsqueeze(-1)
    return torch.where(near_pi, _axis_near_pi(R, skew) * angle.unsqueeze(-1), out)


def compose_rotvecs(rotvecs: torch.Tensor) -> torch.Tensor:
    """(...,K,3) rotation vectors applied IN ORDER -> (...,3) composed rotation vector.

    Angular-velocity rows integrate by COMPOSITION (prod exp(w_k dt)), not summation. K
    sequential matmuls; K = 15 for H so the chain is cheap. Mirrors the numpy twin."""
    Rs = rotvec_to_rotmat(rotvecs)                            # (...,K,3,3)
    acc = Rs[..., 0, :, :]
    for k in range(1, Rs.shape[-3]):
        acc = Rs[..., k, :, :] @ acc
    return rotmat_to_rotvec(acc)
