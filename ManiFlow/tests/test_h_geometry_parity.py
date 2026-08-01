"""H-series geometry parity: the fork's numpy twins vs the converter's `transforms.py`.

The fork ships into the training container as a COPY of `lumi_place_image_dataset.py`; the
converter runs in a different container. There is no shared package, so the ONLY guarantee that
the labels the fork builds are the labels the converter documented is this test. It is the pin.

Run:
    python tests/test_h_geometry_parity.py
    LUMI_PIPELINE_TRANSFORMS=/path/to/src/converter/transforms.py python tests/test_h_geometry_parity.py

If the converter source is not reachable the geometry-parity half SKIPS LOUDLY (it does not
silently pass) — the torch-vs-numpy half still runs.
"""
import importlib.util
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maniflow.common import so3_torch as st                                   # noqa: E402
from maniflow.dataset import lumi_place_image_dataset as fk                   # noqa: E402

TOL = 1e-9
SEED = 20260731

DEFAULT_PIPELINE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "..", "aws_sage_maker", "policy-training-pipeline", "src", "converter", "transforms.py")


def _load_pipeline_transforms():
    path = os.environ.get("LUMI_PIPELINE_TRANSFORMS", DEFAULT_PIPELINE)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        return None, path
    spec = importlib.util.spec_from_file_location("pipeline_transforms", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, path


def _rand_rot(rng):
    q = rng.normal(size=4)
    return fk.quat_wxyz_to_rotmat(q / np.linalg.norm(q))


def test_twin_parity(pl):
    """Every H function the contract names, on a SHARED random seed, to 1e-9."""
    rng = np.random.default_rng(SEED)
    checks = 0

    assert pl.EE_TCP_Y_OFFSET_M == fk.EE_TCP_Y_OFFSET_M, "EE_TCP_Y_OFFSET_M differs"
    checks += 1

    for _ in range(50):
        a = float(rng.uniform(-np.pi, np.pi))
        assert np.abs(pl.rotz(a) - fk.rotz(a)).max() < TOL, "rotz"
        checks += 1

        p_tcp = rng.normal(size=3)
        R_tcp = _rand_rot(rng)
        q7 = float(rng.uniform(-2.0, 2.0))
        pe0, Re0 = pl.ee_from_tcp(p_tcp, R_tcp, q7)
        pe1, Re1 = fk.ee_from_tcp(p_tcp, R_tcp, q7)
        assert np.abs(pe0 - pe1).max() < TOL and np.abs(Re0 - Re1).max() < TOL, "ee_from_tcp"
        pt0, Rt0 = pl.tcp_from_ee(pe0, Re0, q7)
        pt1, Rt1 = fk.tcp_from_ee(pe1, Re1, q7)
        assert np.abs(pt0 - pt1).max() < TOL and np.abs(Rt0 - Rt1).max() < TOL, "tcp_from_ee"
        # exact round trip (the ee<->tcp ambiguity that H exists to remove)
        assert np.abs(pt1 - p_tcp).max() < 1e-12 and np.abs(Rt1 - R_tcp).max() < 1e-12
        checks += 3

        R_cv0 = _rand_rot(rng)
        grv = rng.normal(scale=0.2, size=3)
        assert np.abs(pl.goal_frame_rotation(R_cv0, grv)
                      - fk.goal_frame_rotation(R_cv0, grv)).max() < TOL, "goal_frame_rotation"
        checks += 1

        rvs = rng.normal(scale=0.05, size=(7, 3))
        assert np.abs(pl.compose_rotvecs(rvs) - fk.compose_rotvecs(rvs)).max() < TOL, \
            "compose_rotvecs"
        checks += 1

        Hn = 16
        p_seq = np.cumsum(rng.normal(scale=0.01, size=(Hn, 3)), axis=0)
        R_seq = np.stack([fk.rotvec_to_rotmat(rv) for rv in
                          np.cumsum(rng.normal(scale=0.01, size=(Hn, 3)), axis=0)])
        R_frame = _rand_rot(rng)
        dt = 0.1
        assert np.abs(pl.twist_rows_from_poses(p_seq, R_seq, R_frame, dt)
                      - fk.twist_rows_from_poses(p_seq, R_seq, R_frame, dt)).max() < TOL, \
            "twist_rows_from_poses"
        checks += 1

        p_goal, R_goal = rng.normal(size=3), _rand_rot(rng)
        u0 = pl.remaining_to_goal(p_seq[3], R_seq[3], p_goal, R_goal, R_frame)
        u1 = fk.remaining_to_goal(p_seq[3], R_seq[3], p_goal, R_goal, R_frame)
        assert np.abs(u0 - u1).max() < TOL, "remaining_to_goal"
        q0 = pl.pose_from_remaining(p_goal, R_goal, u0, R_frame)
        q1 = fk.pose_from_remaining(p_goal, R_goal, u1, R_frame)
        assert np.abs(q0[0] - q1[0]).max() < TOL and np.abs(q0[1] - q1[1]).max() < TOL, \
            "pose_from_remaining"
        # ... and it inverts remaining_to_goal exactly. Checked with a REALISTIC goal-vs-pose
        # rotation (H's post-yaw-skip remaining rotation is <= ~9 deg): a fully random pair can
        # land within fp-eps of a pi rotation, where the rotvec axis is genuinely ill-conditioned
        # (both twins agree there — that is what the parity check above measures — but neither
        # can round-trip it to 1e-11, and H never produces such a rotation).
        R_goal_near = fk.rotvec_to_rotmat(rng.normal(scale=0.1, size=3)) @ R_seq[3]
        u_near = fk.remaining_to_goal(p_seq[3], R_seq[3], p_goal, R_goal_near, R_frame)
        pn, Rn = fk.pose_from_remaining(p_goal, R_goal_near, u_near, R_frame)
        assert np.abs(pn - p_seq[3]).max() < 1e-11
        assert np.abs(Rn - R_seq[3]).max() < 1e-11
        checks += 2

        a0 = pl.anchored_delta(p_seq[0], R_seq[0], p_seq[5], R_seq[5], R_cv0)
        a1 = fk.anchored_delta(p_seq[0], R_seq[0], p_seq[5], R_seq[5], R_cv0)
        assert np.abs(a0 - a1).max() < TOL, "anchored_delta"
        checks += 1

    print(f"  numpy twin parity: {checks} comparisons, all < {TOL:g}")


def test_torch_vs_numpy():
    """The torch SO(3) helpers used by the goal-frame machinery must agree with the numpy
    twins (they build the same quantities on the training path)."""
    rng = np.random.default_rng(SEED + 1)
    rv = rng.normal(scale=0.4, size=(64, 3))
    Rt = st.rotvec_to_rotmat(torch.from_numpy(rv)).numpy()
    Rn = np.stack([fk.rotvec_to_rotmat(r) for r in rv])
    e1 = np.abs(Rt - Rn).max()
    back = st.rotmat_to_rotvec(torch.from_numpy(Rn)).numpy()
    e2 = np.abs(back - np.stack([fk.rotmat_to_rotvec(m) for m in Rn])).max()
    seq = rng.normal(scale=0.05, size=(8, 15, 3))
    ct = st.compose_rotvecs(torch.from_numpy(seq)).numpy()
    cn = np.stack([fk.compose_rotvecs(s) for s in seq])
    e3 = np.abs(ct - cn).max()
    # small-angle limit must stay finite and exact
    tiny = torch.zeros(3, 3, dtype=torch.float64)
    assert torch.isfinite(st.rotvec_to_rotmat(tiny)).all()
    assert np.abs(st.rotvec_to_rotmat(tiny).numpy()
                  - np.eye(3)[None]).max() < 1e-12
    assert max(e1, e2, e3) < 1e-9, (e1, e2, e3)
    print(f"  torch vs numpy: rotvec->R {e1:.2e}, R->rotvec {e2:.2e}, compose {e3:.2e}")


if __name__ == "__main__":
    print("test_h_geometry_parity")
    test_torch_vs_numpy()
    pl, path = _load_pipeline_transforms()
    if pl is None:
        print(f"  !! SKIPPED numpy twin parity: converter transforms not found at {path}\n"
              f"     set LUMI_PIPELINE_TRANSFORMS to run it. THIS IS NOT A PASS.")
        sys.exit(3)
    test_twin_parity(pl)
    print("PASS")
