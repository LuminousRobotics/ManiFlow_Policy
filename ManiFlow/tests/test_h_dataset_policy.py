"""H-series CPU checks: label round-trips, all four modes forward/backward, ONNX export.

No GPU and no 100 GB dataset needed — a tiny synthetic v10 zarr fixture is generated here that
is GEOMETRICALLY CONSISTENT (goal_*_cam / rail_a_cam / ee_goal_* / grasp_offset all derived from
one pose story), which is what makes the dataset's `_probe_grasp_offset_convention` and the
round-trip assertions meaningful rather than tautological.

Run:  python tests/test_h_dataset_policy.py            (add --onnx to include the export check)
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maniflow.dataset import lumi_place_image_dataset as fk                    # noqa: E402
from maniflow.dataset.lumi_place_image_dataset import (                        # noqa: E402
    LumiPlaceImageDataset, anchored_delta, build_h_action_rows, h_action_row_k_start,
    h_action_rows, quat_wxyz_to_rotmat, rotvec_to_rotmat, pose_from_remaining, rotz)

SEED = 7
S = 64            # fixture image size (keeps the resnet18 trunk + 2x2 token grid cheap on CPU)
T_EP = 40         # frames per episode
N_EP = 3
POSE_H = 16       # contract H
CONTROL_HZ = 10.0
SETTLE = 6        # last N frames hold exactly at the goal (terminal-settle rows must be 0)


# --------------------------------------------------------------------------- fixture builder
def _quat(R):
    """(3,3) -> wxyz (uses the fork's own rotvec path so conventions cannot drift)."""
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s,
                      (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    else:
        i = int(np.argmax(np.diagonal(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(max(0.0, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2
        q = np.empty(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = q / np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _episode(rng, ep_idx=0, n_eps=1):
    """One geometrically-consistent episode. Returns a dict of per-frame arrays."""
    # --- static rail_a + the per-episode PANEL GOAL, expressed in the rail frame ---------
    # Contract §1.2b, reproduced faithfully because it is the whole point: in the rail frame the
    # panel goal has x fixed at module_width/2 but y FREE over ~0.19 m (which slot along the rail
    # this panel takes — set by the neighbour edge) and z jittering ~24 mm. An episode is
    # therefore NOT reconstructible from rail_a alone, so the identifiability test below and the
    # dataset's convention probe are measuring something real rather than a tautology.
    p_railA = np.array([0.6, 0.05, 0.35]) + rng.normal(scale=0.05, size=3)
    R_railA = rotvec_to_rotmat(rng.normal(scale=0.15, size=3))
    # y is swept DETERMINISTICALLY across the fixture's episodes rather than sampled, so a
    # 3-episode fixture still spans the full 0.1913 m and the identifiability test cannot pass
    # or fail on RNG luck.
    y_free = -0.09565 + (0.1913 * ep_idx / max(1, n_eps - 1))
    rail_to_panel = np.array([0.600,                                # module_width/2, CONSTANT
                              y_free,                               # FREE: 0.1913 m along-rail
                              float(rng.uniform(-0.012, 0.012))])   # 24 mm hover jitter
    p_panel_goal = p_railA + R_railA @ rail_to_panel
    R_panel_goal = R_railA                                          # rail-aligned by construction
    # --- grasp = T_tcp_panel, ~180 deg like the real data, so the "a 180 deg rotation is its
    # own inverse -> orientation cannot discriminate the convention" trap (§1.2a) is present.
    p_off = rng.normal(scale=0.12, size=3)
    _ax = rng.normal(size=3)
    R_off = rotvec_to_rotmat(_ax / np.linalg.norm(_ax) * (np.pi + rng.normal(scale=0.02)))
    # --- contract §1.2a invert=True: T_tcp_goal = T_panel_goal @ inv(T_tcp_panel) ---------
    R_gt = R_panel_goal @ R_off.T
    p_gt = p_panel_goal - R_gt @ p_off
    q7_goal = float(rng.uniform(-0.4, 0.4))

    # --- trajectory: descend toward the tcp goal, then HOLD for SETTLE frames ----------
    p_start = p_gt + np.array([0.0, 0.0, 0.18]) + rng.normal(scale=0.02, size=3)
    rv_start = rng.normal(scale=0.08, size=3)                    # rotation offset at t=0
    n_move = T_EP - SETTLE
    s = np.concatenate([np.linspace(0.0, 1.0, n_move), np.ones(SETTLE)])
    p_tcp = p_start[None] * (1 - s[:, None]) + p_gt[None] * s[:, None]
    R_tcp = np.stack([rotvec_to_rotmat(rv_start * (1.0 - si)) @ R_gt for si in s])
    q7 = q7_goal + (rng.uniform(-0.3, 0.3)) * (1.0 - s)

    # --- ee_link (the action/proprio frame) --------------------------------------------
    p_ee = np.stack([fk.ee_from_tcp(p_tcp[t], R_tcp[t], q7[t])[0] for t in range(T_EP)])
    R_ee = np.stack([fk.ee_from_tcp(p_tcp[t], R_tcp[t], q7[t])[1] for t in range(T_EP)])
    p_ee_goal, R_ee_goal = fk.ee_from_tcp(p_gt, R_gt, q7_goal)

    # --- camera (wrist): rigid offset from ee_link, so it moves with the arm -----------
    R_off_cam = rotvec_to_rotmat(np.array([0.1, -0.2, 0.05]))
    p_cam = p_ee + np.einsum('tij,j->ti', R_ee, np.array([0.02, -0.05, -0.12]))
    R_cv = np.einsum('tij,jk->tik', R_ee, R_off_cam)

    def ad(pa, Ra, pb, Rb, Rf):
        return anchored_delta(pa, Ra, pb, Rb, Rf)

    goal_cam = np.stack([ad(p_tcp[t], R_tcp[t], p_gt, R_gt, R_cv[t]) for t in range(T_EP)])
    # rail_a_cam: TCP-ANCHORED DELTA (aux target)
    rail_a_cam = np.stack([ad(p_tcp[t], R_tcp[t], p_railA, R_railA, R_cv[t])
                           for t in range(T_EP)])
    # panel_goal_cam: camera-frame-ABSOLUTE (primary target) — a DIFFERENT convention (§1.2)
    panel_goal_cam = np.stack([
        np.concatenate([R_cv[t].T @ (p_panel_goal - p_cam[t]),
                        fk.rotmat_to_rotvec(R_cv[t].T @ R_panel_goal)])
        for t in range(T_EP)])
    ego = np.zeros((T_EP, 6), dtype=np.float32)
    for t in range(1, T_EP):
        ego[t] = ad(p_ee[t - 1], R_ee[t - 1], p_ee[t], R_ee[t], R_cv[t - 1])
    twist = np.zeros((T_EP, 6), dtype=np.float32)
    dt = 1.0 / CONTROL_HZ
    for t in range(1, T_EP):
        twist[t, :3] = R_cv[t].T @ (p_ee[t] - p_ee[t - 1]) / dt
        twist[t, 3:] = fk.rotmat_to_rotvec(R_cv[t].T @ (R_ee[t] @ R_ee[t - 1].T) @ R_cv[t]) / dt
    twist_hist = np.repeat(twist[:, None, :], 5, axis=1).astype(np.float32)

    agent_pos = np.concatenate(
        [rng.normal(scale=0.3, size=(T_EP, 6)).astype(np.float32), q7[:, None]], axis=1)
    phase = np.where(s < 0.5, 0, 1).astype(np.int8).reshape(-1, 1)
    done = np.zeros((T_EP, 1), dtype=np.float32)
    done[-2:] = 1.0
    grasp = np.tile(np.concatenate([p_off, _quat(R_off)]), (T_EP, 1)).astype(np.float32)

    return {
        'head_camera': rng.integers(0, 256, (T_EP, 3, S, S), dtype=np.uint8),
        'depth': rng.integers(300, 1500, (T_EP, 1, S, S)).astype(np.uint16),
        'tcp_pos_w': p_tcp.astype(np.float32),
        'tcp_quat_w': np.stack([_quat(R) for R in R_tcp]).astype(np.float32),
        'cam_pos_w': p_cam.astype(np.float32),
        'cam_quat_cv': np.stack([_quat(R) for R in R_cv]).astype(np.float32),
        'agent_pos': agent_pos,
        'task': rng.integers(0, 2, (T_EP, 1)).astype(np.float32),
        'arm_kpts_uv': np.concatenate(
            [rng.uniform(0.1, 0.9, (T_EP, 7, 2)),
             np.ones((T_EP, 7, 1))], axis=-1).astype(np.float32),
        'arm_kpts_cam': np.concatenate(
            [rng.uniform(-0.3, 0.3, (T_EP, 7, 2)),
             rng.uniform(0.4, 1.2, (T_EP, 7, 1))], axis=-1).astype(np.float32),
        'goal_pos_cam': goal_cam[:, :3].astype(np.float32),
        'goal_rot_cam': goal_cam[:, 3:].astype(np.float32),
        'ee_pos_w': p_ee.astype(np.float32),
        'ee_quat_w': np.stack([_quat(R) for R in R_ee]).astype(np.float32),
        'q7': q7.reshape(-1, 1).astype(np.float32),
        'ee_goal_pos_w': np.tile(p_ee_goal, (T_EP, 1)).astype(np.float32),
        'ee_goal_quat_w': np.tile(_quat(R_ee_goal), (T_EP, 1)).astype(np.float32),
        'q7_goal': np.full((T_EP, 1), q7_goal, dtype=np.float32),
        'panel_goal_cam': panel_goal_cam.astype(np.float32),
        'rail_a_cam': rail_a_cam.astype(np.float32),
        'grasp_offset': grasp,
        'tcp_egomotion': ego,
        'tcp_twist': twist,
        'twist_hist': twist_hist,
        'phase_id': phase,
        'done': done,
        # v10 RETAINS these for normalizer/eval compat even though H does not consume them —
        # keeping them in the fixture lets the LEGACY (pre-H) path be regression-tested here.
        'prev_action': ego.copy(),
        'gravity_cam': rng.normal(scale=0.1, size=(T_EP, 3)).astype(np.float32),
    }


def build_fixture(path):
    import zarr
    rng = np.random.default_rng(SEED)
    eps = [_episode(rng, e, N_EP) for e in range(N_EP)]
    root = zarr.open(path, mode='w')
    data = root.create_group('data')
    for key in eps[0]:
        arr = np.concatenate([e[key] for e in eps], axis=0)
        data.create_dataset(key, data=arr, chunks=(min(64, len(arr)),) + arr.shape[1:])
    meta = root.create_group('meta')
    meta.create_dataset('episode_ends',
                        data=np.cumsum([T_EP] * N_EP).astype(np.int64))
    # camera_K at the STORED resolution
    meta.create_dataset('camera_K', data=np.array(
        [S * 0.9, S * 0.9, S / 2.0, S / 2.0], dtype=np.float32))
    return path


# --------------------------------------------------------------------------------- tests
def test_round_trips():
    """For each mode, labels built from a KNOWN ee pose sequence must reconstruct it, and the
    goal modes must contract to zero at the goal."""
    rng = np.random.default_rng(11)
    Hn = POSE_H
    dt = 1.0 / CONTROL_HZ
    # a pose sequence that ENDS exactly at the goal (so terminal rows must vanish)
    p_goal = np.array([0.55, 0.02, 0.30])
    R_goal = rotvec_to_rotmat(np.array([0.05, -0.1, 0.02]))
    q7_goal = 0.17
    # reach the goal by k=H-3 and HOLD: the twist terminal row is (p_15-p_14)/dt, so it only
    # vanishes for a settled tail — which is exactly what the sampler's frame-repeat padding
    # produces at a segment end, and what `terminal_zero_weight` supervises.
    s = np.concatenate([np.linspace(0.0, 1.0, Hn - 2), np.ones(2)])
    p0 = p_goal + np.array([0.0, 0.0, 0.12])
    rv0 = np.array([0.03, 0.02, -0.04])
    p_ee = p0[None] * (1 - s[:, None]) + p_goal[None] * s[:, None]
    R_ee = np.stack([rotvec_to_rotmat(rv0 * (1 - si)) @ R_goal for si in s])
    q7 = q7_goal + 0.2 * (1 - s)
    R_cv0 = rotvec_to_rotmat(rng.normal(scale=0.4, size=3))
    goal_rot_cam = anchored_delta(p_ee[0], R_ee[0], p_goal, R_goal, R_cv0)[3:]
    R_g = fk.goal_frame_rotation(R_cv0, goal_rot_cam)

    for param in ("twist", "delta"):
        for frame in ("cam0", "goal"):
            R_frame = R_cv0 if frame == "cam0" else R_g
            rows = build_h_action_rows(p_ee, R_ee, q7, p_goal, R_goal, q7_goal,
                                       R_frame, param, frame, dt)
            assert rows.shape == (h_action_rows(param, frame, Hn), 7), rows.shape

            if param == "twist":
                # integrate: position exactly (single fixed frame), rotation by composition
                acc_p = p_ee[0].copy()
                acc_R = R_ee[0].copy()
                ep, eR = 0.0, 0.0
                for k in range(rows.shape[0]):
                    acc_p = acc_p + R_frame @ rows[k, :3] * dt
                    dRw = R_frame @ rotvec_to_rotmat(rows[k, 3:6] * dt) @ R_frame.T
                    acc_R = dRw @ acc_R
                    ep = max(ep, np.abs(acc_p - p_ee[k + 1]).max())
                    eR = max(eR, np.abs(acc_R - R_ee[k + 1]).max())
            elif frame == "cam0":
                ep, eR = 0.0, 0.0
                for k in range(rows.shape[0]):
                    p = p_ee[0] + R_cv0 @ rows[k, :3]
                    R = (R_cv0 @ rotvec_to_rotmat(rows[k, 3:6]) @ R_cv0.T) @ R_ee[0]
                    ep = max(ep, np.abs(p - p_ee[k + 1]).max())
                    eR = max(eR, np.abs(R - R_ee[k + 1]).max())
            else:
                ep, eR = 0.0, 0.0
                for k in range(rows.shape[0]):
                    p, R = pose_from_remaining(p_goal, R_goal, rows[k, :6], R_g)
                    ep = max(ep, np.abs(p - p_ee[k]).max())
                    eR = max(eR, np.abs(R - R_ee[k]).max())
            assert ep < 1e-9 and eR < 1e-9, (param, frame, ep, eR)

            # dJ7 semantics (contract §3): positional in every mode, frame-dependent
            if frame == "cam0":
                ks = np.arange(1, Hn)
                assert np.abs(rows[:, 6] - (q7[ks] - q7[0])).max() < 1e-12
            else:
                ks = np.arange(1, Hn) if param == "twist" else np.arange(0, Hn)
                assert np.abs(rows[:, 6] - (q7_goal - q7[ks])).max() < 1e-12

            # terminal property
            if frame == "goal":
                assert np.abs(rows[-1]).max() < 1e-9, (param, frame, rows[-1])
            print(f"  round-trip {param:5s}/{frame:4s}: rows={rows.shape[0]:2d} "
                  f"pos_err={ep:.2e} rot_err={eR:.2e} terminal="
                  f"{np.abs(rows[-1]).max():.2e}")

    # delta+cam0's k=0 row is identically zero — that is WHY it is neither predicted nor shipped
    # (contract §3 FINAL). The check lives HERE, as a test, instead of as a runtime anchor gate:
    # it is a property of the parameterization, invariant over samples and checkpoints, so a
    # per-inference gate on it could only ever reject healthy output (the G1 26% lesson).
    a0 = anchored_delta(p_ee[0], R_ee[0], p_ee[0], R_ee[0], R_cv0)
    assert np.abs(a0).max() < 1e-15, a0
    rows_dc = build_h_action_rows(p_ee, R_ee, q7, p_goal, R_goal, q7_goal,
                                  R_cv0, "delta", "cam0", dt)
    assert rows_dc.shape[0] == 15 and h_action_row_k_start("delta", "cam0") == 1
    assert h_action_row_k_start("delta", "goal") == 0
    # ... and the first SHIPPED row is k=1, i.e. genuinely non-zero (nothing was prepended)
    assert np.abs(rows_dc[0, :6]).max() > 1e-4, rows_dc[0]
    print(f"  row policy: k=0 row |{np.abs(a0).max():.1e}| (not shipped); first shipped row "
          f"k=1 |{np.abs(rows_dc[0, :6]).max():.3e}|; k_start 1/1/1/0")


def test_tcpcam_formula():
    """`tcpcam` must match the NORMATIVE contract §4 formula exactly, and (as documentation of
    why the anchored_delta form the glue agent may use is equivalent) agree with it."""
    rng = np.random.default_rng(23)
    for _ in range(200):
        R_cv = rotvec_to_rotmat(rng.normal(scale=1.0, size=3))
        R_ee = rotvec_to_rotmat(rng.normal(scale=1.0, size=3))
        p_cam, p_ee = rng.normal(size=3), rng.normal(size=3)
        spec = np.concatenate([R_cv.T @ (p_ee - p_cam),
                               fk.rotmat_to_rotvec(R_cv.T @ R_ee)])
        ad = anchored_delta(p_cam, R_cv, p_ee, R_ee, R_cv)
        assert np.abs(spec - ad).max() < 1e-9, (spec, ad)
    print("  tcpcam: contract formula == anchored_delta(cam, ee, R_cv) to <1e-9 (200 draws)")


def test_goal_composition(zarr_path):
    """Contract §1.2a/§1.2b: the panel-goal composition reproduces the stored ee_link goal, the
    WRONG convention is hundreds of mm away, and — the trap — ORIENTATION cannot tell them apart.
    Also checks the rail-based composition is grossly wrong (the landmark, not the sign)."""
    import zarr as _zarr
    ds = LumiPlaceImageDataset(
        zarr_path=zarr_path, horizon=POSE_H, pad_before=1, pad_after=7, seed=SEED,
        val_ratio=0.0, use_depth=True, depth_input='xyz', n_obs_steps=2, control_hz=CONTROL_HZ,
        action_param='twist', action_frame='cam0', augmentation=dict(enable=False))
    assert ds.grasp_offset_invert is fk.GRASP_OFFSET_INVERT is True
    idxs = list(range(0, T_EP * N_EP, 7))
    r_ok = [ds._goal_compose_residual(i, True) for i in idxs]
    r_bad = [ds._goal_compose_residual(i, False) for i in idxs]
    pos_ok, rot_ok = max(p for _, p in r_ok), max(r for r, _ in r_ok)
    pos_bad, rot_bad = max(p for _, p in r_bad), max(r for r, _ in r_bad)
    assert pos_ok < 1e-6, pos_ok                          # fixture uses the rigid form exactly
    assert pos_bad > 0.05, pos_bad                        # wrong convention: far outside the gate
    # THE TRAP: the wrong convention barely moves the angle (~180 deg grasp is self-inverse), so
    # an orientation-only check would have passed it. Assert that explicitly, so nobody "simplifies"
    # the probe into a rotation comparison later.
    assert np.degrees(rot_bad) < 5.0, (
        f"fixture grasp is not ~180 deg: wrong-convention rot err {np.degrees(rot_bad):.2f} deg; "
        f"the discrimination trap is not being exercised")
    assert pos_bad / max(pos_ok, 1e-9) > 1e3
    print(f"  goal composition: invert=True {pos_ok * 1000:.4f} mm / "
          f"{np.degrees(rot_ok):.5f} deg | invert=False {pos_bad * 1000:.1f} mm / "
          f"{np.degrees(rot_bad):.3f} deg  <- position discriminates, angle does not")

    # ---- §1.2b identifiability: rail_a + grasp_offset does NOT determine the goal ----------
    root = _zarr.open(str(zarr_path), mode='r')['data']
    ends = np.cumsum([T_EP] * N_EP)
    ys = []
    for e, end in enumerate(ends):
        i = int(end) - 1
        R_cv = quat_wxyz_to_rotmat(root['cam_quat_cv'][i])
        R_tcp = quat_wxyz_to_rotmat(root['tcp_quat_w'][i])
        p_tcp = np.asarray(root['tcp_pos_w'][i], np.float64)
        p_cam = np.asarray(root['cam_pos_w'][i], np.float64)
        rail = np.asarray(root['rail_a_cam'][i], np.float64)
        pg = np.asarray(root['panel_goal_cam'][i], np.float64)
        # invert the TCP-anchored delta: R_railA = (R_cv @ exp(drv) @ R_cv.T) @ R_tcp
        R_railA = (R_cv @ rotvec_to_rotmat(rail[3:]) @ R_cv.T) @ R_tcp
        p_railA = p_tcp + R_cv @ rail[:3]
        p_panel = p_cam + R_cv @ pg[:3]
        ys.append((R_railA.T @ (p_panel - p_railA)))                     # panel goal IN RAIL FRAME
    ys = np.stack(ys)
    spread = ys.max(axis=0) - ys.min(axis=0)
    assert spread[0] < 1e-6, f"x should be constant (module_width/2): spread {spread[0]}"
    assert spread[1] > 0.15, f"y must be FREE — this fixture is not exercising §1.2b: {spread[1]}"
    print(f"  identifiability: panel goal in rail frame — x spread {spread[0] * 1000:.4f} mm "
          f"(constant), y spread {spread[1] * 1000:.1f} mm (FREE), z spread "
          f"{spread[2] * 1000:.1f} mm => rail_a alone cannot determine the goal")


def test_conventions_not_interchangeable(zarr_path):
    """`panel_goal_cam` (camera-absolute) and `rail_a_cam` (TCP-anchored delta) are different
    conventions. Assert they are NOT numerically interchangeable, and that the documented
    conversion through `tcpcam` is exact — so a future reader cannot 'unify' them."""
    import zarr as _zarr
    root = _zarr.open(str(zarr_path), mode='r')['data']
    gap, worst = 0.0, 0.0
    for i in range(0, T_EP * N_EP, 5):
        rail = np.asarray(root['rail_a_cam'][i], np.float64)
        pg = np.asarray(root['panel_goal_cam'][i], np.float64)
        gap = max(gap, float(np.linalg.norm(rail[:3] - pg[:3])))
        # conversion: TCP-anchored delta -> camera-absolute, using ONLY tcpcam-derivable terms
        R_cv = quat_wxyz_to_rotmat(root['cam_quat_cv'][i])
        R_ee = quat_wxyz_to_rotmat(root['ee_quat_w'][i])
        q7 = float(np.asarray(root['q7'][i]).reshape(-1)[0])
        tcpcam = np.concatenate([R_cv.T @ (np.asarray(root['ee_pos_w'][i], np.float64)
                                           - np.asarray(root['cam_pos_w'][i], np.float64)),
                                 fk.rotmat_to_rotvec(R_cv.T @ R_ee)])
        R_ee_c = rotvec_to_rotmat(tcpcam[3:6])
        R_tcp_c = R_ee_c @ rotz(q7)                                   # contract §0 inverse
        p_tcp_c = tcpcam[:3] - R_tcp_c @ np.array([0.0, fk.EE_TCP_Y_OFFSET_M, 0.0])
        rail_abs = rail[:3] + p_tcp_c                                 # camera-absolute rail pos
        want = R_cv.T @ (np.asarray(root['tcp_pos_w'][i], np.float64)
                         + R_cv @ rail[:3] - np.asarray(root['cam_pos_w'][i], np.float64))
        worst = max(worst, float(np.abs(rail_abs - want).max()))
    assert gap > 0.05, f"the two targets are suspiciously close ({gap:.3f} m) — check the fixture"
    assert worst < 1e-5, worst
    print(f"  conventions: rail_a_cam vs panel_goal_cam differ by up to {gap * 1000:.0f} mm "
          f"(NOT interchangeable); tcpcam-mediated conversion exact to {worst:.2e}")


def test_tcpcam_dataset(zarr_path):
    """What the DATASET emits for `tcpcam` must be the contract §4 formula evaluated on the
    zarr rows at the proprio indices — the guard against the dataset and the spec drifting."""
    ds = LumiPlaceImageDataset(
        zarr_path=zarr_path, horizon=POSE_H, pad_before=1, pad_after=7, seed=SEED,
        val_ratio=0.0, use_depth=True, depth_input='xyz', n_obs_steps=2,
        control_hz=CONTROL_HZ, action_param='twist', action_frame='cam0',
        augmentation=dict(enable=False))     # clean => proprio idx = [anchor-1, anchor], no DR
    rb = ds.replay_buffer
    worst = 0.0
    for idx in (5, 9, 17, 33):
        item = ds[idx]
        # window index w maps to buffer index buffer_start_idx + (w - sample_start_idx);
        # skip any window whose anchor-1 falls in the front-padded (frame-repeated) region.
        b_start, _, s_start, _ = (int(v) for v in ds.sampler.indices[idx])
        if ds.pad_before - 1 < s_start:
            continue
        anchor_abs = b_start + (ds.pad_before - s_start)
        for j, k in enumerate((anchor_abs - 1, anchor_abs)):
            R_cv = quat_wxyz_to_rotmat(np.asarray(rb['cam_quat_cv'][k], np.float64))
            R_ee = quat_wxyz_to_rotmat(np.asarray(rb['ee_quat_w'][k], np.float64))
            want = np.concatenate([
                R_cv.T @ (np.asarray(rb['ee_pos_w'][k], np.float64)
                          - np.asarray(rb['cam_pos_w'][k], np.float64)),
                fk.rotmat_to_rotvec(R_cv.T @ R_ee),
                np.asarray(rb['q7'][k], np.float64).reshape(1)])
            worst = max(worst, float(np.abs(item['obs']['tcpcam'][j].numpy() - want).max()))
    assert worst < 1e-5, worst        # float32 storage
    print(f"  tcpcam: dataset output == contract formula on the zarr, max err {worst:.2e}")


def _make_policy(action_param, action_frame, shape_meta, n_obs_steps=2):
    from maniflow.model.vision_2d.timm_obs_encoder import TimmObsEncoder
    from maniflow.policy.maniflow_image_policy import ManiFlowTransformerImagePolicy
    enc = TimmObsEncoder(
        shape_meta=shape_meta, model_name='resnet18', pretrained=False, frozen=False,
        global_pool='', transforms=None, use_group_norm=True, share_rgb_model=False,
        imagenet_norm=True, feature_aggregation=None, downsample_ratio=32,
        token_output=True, lowdim_as_tokens=True)
    return ManiFlowTransformerImagePolicy(
        shape_meta=shape_meta, horizon=POSE_H, n_action_steps=8, n_obs_steps=n_obs_steps,
        num_inference_steps=2, obs_encoder=enc, visual_cond_len=64,
        n_layer=1, n_head=4, n_emb=128, block_type="DiTX",
        flow_batch_ratio=0.75, consistency_batch_ratio=0.25,
        endpoint_loss_weight=0.2,
        kpt_loss_weight=1.0, place_loss_weight=1.0, n_keypoints=7, kpt_head_hires=True,
        pointnet_dim=1, pointnet_tokens=16, rgb3d_pos_enc=True,
        proprio_mask_p=0.25,
        action_param=action_param, action_frame=action_frame, control_hz=CONTROL_HZ,
        goal_integral_weight=0.5, terminal_zero_weight=0.5, goal_consistency_weight=0.5,
        action_rate_weight=0.05, goal_frame_self_p_max=0.5, goal_frame_anneal_epochs=4,
        phase_loss_weight=0.05, done_loss_weight=0.05)


def _shape_meta(action_dim=7):
    # DictConfig, exactly as hydra hands it to the policy (plain dicts would make
    # dict_apply recurse into the per-key attr dicts).
    from omegaconf import OmegaConf
    return OmegaConf.create({
        'obs': {
            'head_cam': {'shape': [3, S, S], 'type': 'rgb', 'horizon': 2},
            'agent_pos': {'shape': [7], 'type': 'low_dim', 'horizon': 2},
            'task': {'shape': [1], 'type': 'low_dim', 'horizon': 2},
            'grasp_off': {'shape': [7], 'type': 'low_dim', 'horizon': 2},
            'ego': {'shape': [6], 'type': 'low_dim', 'horizon': 2},
            'twist': {'shape': [6], 'type': 'low_dim', 'horizon': 2},
            'tcpcam': {'shape': [7], 'type': 'low_dim', 'horizon': 2},
            'dt': {'shape': [1], 'type': 'low_dim', 'horizon': 2},
            'twist_hist': {'shape': [5, 6], 'type': 'low_dim', 'horizon': 2},
        },
        'action': {'shape': [action_dim], 'horizon': POSE_H},
    })


def test_dataset_and_policy(zarr_path, do_onnx=False):
    from torch.utils.data import DataLoader
    results = {}
    for action_param in ("twist", "delta"):
        for action_frame in ("cam0", "goal"):
            ds = LumiPlaceImageDataset(
                zarr_path=zarr_path, horizon=POSE_H, pad_before=1, pad_after=7,
                seed=SEED, val_ratio=0.34, use_depth=True, depth_input='xyz',
                n_obs_steps=2, control_hz=CONTROL_HZ,
                action_param=action_param, action_frame=action_frame,
                augmentation=dict(
                    enable=True, gpu_offload=True, depth_episode_dropout=0.1,
                    obs_gap_probs=[0.1, 0.7, 0.2], image_lag_probs=[0.6, 0.4],
                    proprio_noise_joint_deg=0.7, proprio_noise_twist_frac=0.05,
                    proprio_noise_ego_mm=1.0,
                    goal_frame_noise_mm=15.0, goal_frame_noise_deg=1.5))
            expect_rows = h_action_rows(action_param, action_frame, POSE_H)
            assert ds.action_rows == expect_rows
            assert ds.seq_len == ds.pad_before + POSE_H and ds.pad_before == 3

            sm = _shape_meta()
            policy = _make_policy(action_param, action_frame, sm)
            policy.set_normalizer(ds.get_normalizer())
            policy.set_epoch(4)                      # scheduled sampling fully ramped
            policy.train()

            loader = DataLoader(ds, batch_size=4, shuffle=False, drop_last=True)
            batch = next(iter(loader))
            assert batch['action'].shape[1:] == (expect_rows, 7), batch['action'].shape
            for k, want in (('head_cam', (2, 3, S, S)), ('depth_cam', (2, 4, S, S)),
                            ('twist_hist', (2, 5, 6)), ('dt', (2, 1)),
                            ('tcpcam', (2, 7)), ('grasp_off', (2, 7))):
                assert tuple(batch['obs'][k].shape[1:]) == want, (k, batch['obs'][k].shape)
            # dt reports the TRUE gap and stays inside the contract support
            assert float(batch['obs']['dt'].min()) >= 0.0
            assert float(batch['obs']['dt'].max()) <= 0.2 + 1e-6

            import copy as _copy
            ema = _copy.deepcopy(policy)
            loss, log = policy.compute_loss(batch, ema_model=ema)
            assert torch.isfinite(loss), (action_param, action_frame, loss)
            loss.backward()
            gnorm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), 1e9))
            assert np.isfinite(gnorm) and gnorm > 0.0, gnorm
            # the zero-init 3D-RGB projection must receive gradient (it is on the actor path)
            assert policy.rgb3d_proj.weight.grad is not None
            assert float(policy.rgb3d_proj.weight.grad.abs().max()) > 0.0
            # §1.2b: the place head targets panel_goal (PRIMARY) and rail_a is a separate AUX
            # head that must actually be trained, not silently dead.
            assert 'panel_goal' in policy.normalizer.params_dict
            assert 'rail_a' in policy.normalizer.params_dict
            assert policy.rail_aux_head is not None
            assert float(policy.rail_aux_head[1].weight.grad.abs().max()) > 0.0
            assert 'loss_rail_aux' in log and np.isfinite(log['loss_rail_aux'])
            # ... and the two targets must NOT share a normalizer (different conventions)
            assert not torch.allclose(
                policy.normalizer.params_dict['panel_goal']['scale'],
                policy.normalizer.params_dict['rail_a']['scale'])

            policy.eval()
            with torch.no_grad():
                out = policy.predict_action({k: v[:1] for k, v in batch['obs'].items()})
            # contract §3 FINAL: nothing is prepended, so the predicted count IS the shipped
            # count for every mode (15/15/15/16), and tensor row j is chunk step k_start + j.
            assert tuple(out['action_pred'].shape) == (1, expect_rows, 7), out['action_pred'].shape
            assert policy.action_row_k_start == (
                0 if (action_param, action_frame) == ("delta", "goal") else 1)
            assert policy.k0_row_structurally_zero == (
                (action_param, action_frame) == ("delta", "cam0"))
            assert not hasattr(policy, 'row0_zero'), "row0_zero must be gone (§3 FINAL)"
            assert tuple(out['place_pred'].shape) == (1, 6)
            assert tuple(out['kpt_uv'].shape) == (1, 7, 3)
            assert tuple(out['kpt_cam'].shape) == (1, 7, 3)
            assert tuple(out['phase'].shape) == (1, 2)
            assert tuple(out['done'].shape) == (1, 1)

            # val split must be clean AND nominal-timed (gap 1, lag 0)
            vds = ds.get_validation_dataset()
            vb = next(iter(DataLoader(vds, batch_size=2, shuffle=False)))
            assert abs(float(vb['obs']['dt'].min()) - 0.1) < 1e-6
            assert abs(float(vb['obs']['dt'].max()) - 0.1) < 1e-6

            results[(action_param, action_frame)] = dict(
                loss=float(loss.item()), rows=expect_rows, gnorm=gnorm,
                keys=sorted(k for k in log if k.startswith(('loss_', 'goal', 'phase', 'self')))
            )
            print(f"  {action_param:5s}/{action_frame:4s}: rows={expect_rows:2d} "
                  f"loss={float(loss.item()):.4f} |g|={gnorm:.3g} "
                  f"self_frame={log.get('self_frame_frac', 0.0):.2f} "
                  f"place_mm={log.get('place_mm', 0.0):.1f}")

            if do_onnx:
                _onnx_check(policy, ds, action_param, action_frame)
    return results


def _onnx_check(policy, ds, action_param, action_frame):
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "..", "aws_sage_maker", "policy-training-pipeline", "src"))
    import export_onnx as ex
    import onnx
    policy.eval()
    # same two prep steps build_wrapper_from_ckpt does (we bypass it: no checkpoint here)
    ex._patch_rmsnorm_for_export()
    ex._make_transforms_deterministic(policy)
    # DiT-X's final layer is zero-init (AdaLN-Zero discipline), so an untrained model returns
    # velocity == 0 and `action` would be exactly `unnormalize(noise)` — a parity check that
    # never touches the transformer. Perturb it so the flow unroll is genuinely exercised.
    torch.nn.init.normal_(policy.model.final_layer.ffn_final.fc2.weight, std=0.05)
    torch.nn.init.normal_(policy.model.final_layer.ffn_final.fc2.bias, std=0.05)
    policy.cam_k_norm_buf.copy_(torch.as_tensor(ds.cam_k_norm, dtype=torch.float32))
    bundle = {"wrapper": ex._build_wrapper(policy)[0],
              "obs_keys": list(ex.H_INPUT_ORDER), "policy": policy,
              "cfg": None, "weights_used": "test", "deploy_inference_steps": 2}
    tmp = tempfile.mkdtemp()
    out = os.path.join(tmp, "policy.onnx")
    try:
        rep = ex.export_onnx("test.ckpt", out, image_hw=(S, S), prebuilt=bundle)
        md = {p.key: p.value for p in onnx.load(out).metadata_props}
        for req in ("contract_version", "action_param", "action_frame", "action_dim",
                    "action_rows", "action_row_k_start", "tcp_link", "quat_order", "phase_order",
                    "action_frame_note", "tcp_from_ee", "camera_K", "control_hz",
                    "train_dt_range_s", "driver_limits", "inference_steps",
                    "anchor_residual_p99_mm", "j7_channel", "camera_frame", "resize_policy",
                    "units", "goal_composition", "place_pred_meaning", "keypoint_order"):
                assert req in md and md[req] != "", f"metadata_props missing {req}: {md}"
        assert md["contract_version"] == "h1"
        # §1.2a/§1.2b must travel INSIDE the model: the composition the deploy node applies and
        # what place_pred actually means (panel goal, not rail_a).
        assert "inv(T_tcp_panel)" in md["goal_composition"], md["goal_composition"]
        assert "panel_goal_cam" in md["place_pred_meaning"]
        assert "edge_n,edge_f" in md["keypoint_order"]
        assert md["action_param"] == action_param and md["action_frame"] == action_frame
        assert md["action_dim"] == "7" and md["tcp_link"] == "ee_link"
        # §3 FINAL: row counts 15/15/15/16, k_start 1/1/1/0, and NO row0_zero key at all
        assert "row0_zero" not in md, "row0_zero must not be exported (§3 FINAL)"
        assert int(md["action_rows"]) == {("twist", "cam0"): 15, ("twist", "goal"): 15,
                                         ("delta", "cam0"): 15, ("delta", "goal"): 16}[
            (action_param, action_frame)], md["action_rows"]
        assert int(md["action_row_k_start"]) == (
            0 if (action_param, action_frame) == ("delta", "goal") else 1)
        assert int(md["action_rows"]) == list(rep["outputs"]["action"]["shape"])[1]
        assert int(md["action_rows"]) == policy.action_rows
        assert rep["parity"]["passed"], rep["parity"]
        print(f"  onnx  {action_param:5s}/{action_frame:4s}: rows={md['action_rows']} "
              f"k_start={md['action_row_k_start']} "
              f"max_abs={rep['parity']['max_abs_diff']:.2e} "
              f"per_output={ {k: round(v, 9) for k, v in rep['parity']['per_output_max_abs'].items()} }")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_legacy_regression(zarr_path):
    """The pre-H (G1) path must still work bit-compatibly in structure: action_param=None keeps
    the v3 anchored-TCP dataset, the last-frame-only kpt tokens and the single PointNet token."""
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from maniflow.model.vision_2d.timm_obs_encoder import TimmObsEncoder
    from maniflow.policy.maniflow_image_policy import ManiFlowTransformerImagePolicy
    ds = LumiPlaceImageDataset(
        zarr_path=zarr_path, horizon=POSE_H, pad_before=1, pad_after=7, seed=SEED,
        val_ratio=0.34, use_depth=True, depth_input='xyz',
        augmentation=dict(enable=True, gpu_offload=True, depth_episode_dropout=0.1))
    assert not ds.h_mode and ds.action_rows == POSE_H and ds.seq_len == POSE_H
    sm = OmegaConf.create({
        'obs': {
            'head_cam': {'shape': [3, S, S], 'type': 'rgb', 'horizon': 2},
            'task': {'shape': [1], 'type': 'low_dim', 'horizon': 2},
            'agent_pos': {'shape': [7], 'type': 'low_dim', 'horizon': 2},
            'goal_prior': {'shape': [7], 'type': 'low_dim', 'horizon': 2},
        },
        'action': {'shape': [6], 'horizon': POSE_H}})
    enc = TimmObsEncoder(
        shape_meta=sm, model_name='resnet18', pretrained=False, frozen=False, global_pool='',
        transforms=None, use_group_norm=True, share_rgb_model=False, imagenet_norm=True,
        feature_aggregation=None, downsample_ratio=32, token_output=True, lowdim_as_tokens=True)
    policy = ManiFlowTransformerImagePolicy(
        shape_meta=sm, horizon=POSE_H, n_action_steps=8, n_obs_steps=2,
        num_inference_steps=2, obs_encoder=enc, visual_cond_len=64,
        n_layer=1, n_head=4, n_emb=128, block_type="DiTX",
        endpoint_loss_weight=0.2, kpt_loss_weight=1.0, place_loss_weight=1.0,
        n_keypoints=7, kpt_head_hires=True, pointnet_dim=1)
    assert not policy.h_mode and policy.horizon == POSE_H and policy.action_rows == POSE_H
    assert policy.phase_head is None and policy.rgb3d_proj is None
    assert not policy.lowdim_typed_tokens
    policy.set_normalizer(ds.get_normalizer())
    policy.train()
    batch = next(iter(DataLoader(ds, batch_size=4, shuffle=False, drop_last=True)))
    import copy as _copy
    loss, log = policy.compute_loss(batch, ema_model=_copy.deepcopy(policy))
    assert torch.isfinite(loss), loss
    loss.backward()
    print(f"  legacy (action_param=None): loss={float(loss.item()):.4f} "
          f"kpt_px={log['kpt_px']:.1f} place_mm={log['place_mm']:.1f}")


if __name__ == "__main__":
    torch.manual_seed(0)
    do_onnx = "--onnx" in sys.argv
    print("test_h_dataset_policy")
    print(" round-trips:")
    test_round_trips()
    test_tcpcam_formula()
    tmpd = tempfile.mkdtemp()
    try:
        zp = build_fixture(os.path.join(tmpd, "train.zarr"))
        print(" goal identifiability + convention (§1.2a/§1.2b):")
        test_goal_composition(zp)
        test_conventions_not_interchangeable(zp)
        print(" obs contract:")
        test_tcpcam_dataset(zp)
        print(" legacy regression:")
        test_legacy_regression(zp)
        print(" dataset + policy (4 modes, CPU fwd/bwd):")
        test_dataset_and_policy(zp, do_onnx=do_onnx)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    print("PASS")
