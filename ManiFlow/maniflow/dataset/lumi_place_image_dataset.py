"""Lumi place-policy dataset for ManiFlow (pickplace_v4 -> train.zarr v6 contract).

v6 (ANCHORED) — see the pipeline's src/converter (ConversionVersion v6) + the v3 plan:
- buffer keys: head_camera, depth(u16 mm), tcp_pos_w, tcp_quat_w, cam_pos_w,
  cam_quat_cv, prev_action, gravity_cam, goal_pos_cam, goal_rot_cam, task.
- ACTIONS are ANCHORED-to-chunk-start (spec §3.4), built PER SAMPLE from the stored
  absolute poses: row k = TCP pose at t0+k relative to the pose at t0, expressed in
  the camera frame at t0 (cam_quat_cv[anchor]). No step-chaining -> no in-chunk
  compounding, and the LAST row is the chunk endpoint.
- obs dict keys: head_cam, depth_cam (u16 mm -> tiled OR H3DP-lite 3-band masks),
  prev_action (6), task (1). goal_cam (6) rides ALONGSIDE action (supervision only;
  never enters the obs / ONNX contract).
- Augmentation (train split only, zarr clean): depth (u16-mm domain) axial noise +
  dropout + flying px + episode dropout; prev_action gaussian; per-stream 1-tick
  latency. IMU augmentation removed (no IMU input in v3).

This file is COPY'd into the ManiFlow clone at docker build.
"""

from typing import Dict
import copy

import numpy as np
import torch
import zarr
from termcolor import cprint

from maniflow.common.pytorch_util import dict_apply
from maniflow.common.replay_buffer import ReplayBuffer
from maniflow.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from maniflow.dataset.base_dataset import BaseDataset

# depth stored as uint16 millimeters (0 = invalid). "tiled" mode normalizes to [0,1] over this max.
DEPTH_MM_MAX = 3000.0
# v3 near-focused H3DP band EDGES (meters), data-driven (docs/v3_depth_design.md):
# 8 fine bins over the 0.3-1.5m action zone (matte tube ~0.4-0.6m + hatrail posts, near-mode
# median ~0.81m) + 1 mid "gap" bin (1.5-3.0m) + 1 far "background" bin (3.0-6.0m; >6m clipped
# into it). 11 edges -> 10 channels. Verified: each near bin holds 7.5-18.7% of near-pixel mass.
DEPTH_BAND_EDGES_M = (0.30, 0.45, 0.60, 0.75, 0.90, 1.05, 1.20, 1.35, 1.50, 3.00, 6.00)
DEPTH_BAND_SOFT = 0.05   # meters of smooth falloff at each band edge
# C2 "xyz" point-map mode: one metric scale for X, Y and Z (isotropy preserved so the
# trunk sees true relative distances). Z is clipped to this range; the sub-mm u16 source
# resolution is preserved (the 0.15m band bins threw it away). MUST match eval_policy.
XYZ_SCALE_M = 2.5

DEFAULT_AUGMENTATION = {
    "enable": True,
    "depth_noise_sigma0": 0.005,     # axial sigma = sigma0 * d^2  [m]
    "depth_px_dropout": 0.02,        # fraction of valid px zeroed per frame
    "depth_flying_px": 0.005,        # fraction of px given random depth per frame
    "depth_episode_dropout": 0.4,    # fraction of episodes with depth fully zeroed
    "prev_action_noise_mm": 0.5,     # gaussian on prev_action translation [mm]
    "prev_action_noise_deg": 0.05,   # gaussian on prev_action rotvec [deg]
    # G1 proprio + goal-prior latch aug (all 0 => off). Noise at DEPLOYMENT-error scale
    # (Adapt-Your-Body/DART calibration), not sensor scale; masks fight the copycat.
    "agent_pos_noise_deg": 0.0,      # gaussian on joint positions [deg]
    "agent_pos_mask_p": 0.0,         # per-sample prob of zeroing the whole agent_pos block
    "goal_prior_noise_mm": 0.0,      # gaussian on the goal_prior position [mm]
    "goal_prior_noise_deg": 0.0,     # gaussian on the goal_prior rotvec [deg]
    "goal_prior_drop_p": 0.0,        # per-sample prob the latch is absent (zeros + conf 0)
    "fov_dropout_p": 0.0,            # GPU-aug: per-sample prob of blanking the recent RGB frame
    "latency_shift_prob": 0.5,       # per-episode prob a stream lags 1 control tick
    "rot_aug_deg": 0.0,              # v7 Rewire: SO(2) optical-axis roll aug half-range [deg]; 0 => off
    "gpu_offload": False,            # v7 GPU-aug: skip CPU SO(2)+photometric here (workspace does it on
                                     # the CUDA batch instead); prev_action noise + latency lag stay on CPU
    # v7 Rewire: label-safe RGB appearance aug (train-only; all 0 => off). Photometric only —
    # no geometry change, so no label co-rotation needed. Per-sample, same across the To frames.
    "photo_brightness": 0.0,         # brightness jitter fraction (torchvision adjust_brightness)
    "photo_contrast": 0.0,           # contrast jitter fraction
    "photo_saturation": 0.0,         # saturation jitter fraction
    "photo_hue": 0.0,                # hue jitter (+/-, in [0,0.5])
    "photo_blur_p": 0.0,             # prob of a mild gaussian blur
    "photo_noise": 0.0,              # gaussian pixel noise std (image in [0,1])
    "photo_erase_p": 0.0,            # prob of a random-erasing box (up to 2 boxes)
    # ---- H-series (v10) timing augmentation (contract §5). Defaults = NOMINAL (no jitter),
    # so a config that does not set them reproduces the un-augmented 100 ms / no-lag timing.
    # `obs_gap_frames ~ obs_gap_probs` over {0,1,2}: how many control frames back the PREVIOUS
    # obs image is taken. 0 reproduces the observed duplicate-camera-frame failure; 1 = nominal
    # 100 ms; 2 = 200 ms. `image_lag_frames ~ image_lag_probs` over {0,1}: both obs images are
    # taken that many frames BEFORE the action anchor while proprio stays at the anchor (the
    # measured 100-150 ms observation staleness — joint states arrive fast, images do not).
    # `dt` reports the TRUE gap, so the jitter is in-distribution instead of a silent lie.
    "obs_gap_probs": None,           # None => [0,1,0] (always nominal 1 frame)
    "image_lag_probs": None,         # None => [1,0]   (always 0 lag)
    # ---- H-series per-key proprio DR (contract §4.3). Noise is PHYSICAL and lives here;
    # per-key block MASKING lives in the policy (it needs a learned mask token).
    "proprio_noise_joint_deg": 0.0,  # gaussian on agent_pos joints [deg]
    "proprio_noise_twist_frac": 0.0, # multiplicative gaussian on twist / twist_hist [fraction]
    "proprio_noise_ego_mm": 0.0,     # gaussian on tcp_egomotion translation [mm]
    # ---- H1 scheduled sampling on the OUTPUT frame (contract §3.3). The dataset supplies the
    # GT goal + noise; the POLICY does the anneal toward its own (detached) place-head goal.
    "goal_frame_noise_mm": 0.0,
    "goal_frame_noise_deg": 0.0,
}


# --------------------------------------------------------------------- anchored math
# NOTE: kept byte-parity with the pipeline's src/converter/transforms.py (anchored_delta).
# A shared numeric fixture pins this (test at bottom of this module).

def _quat_wxyz_to_rotmat(q):
    q = np.asarray(q, dtype=np.float64)
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def _rotmat_to_rotvec(R):
    R = np.asarray(R, dtype=np.float64)
    tr = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(tr)
    if angle < 1e-8:
        return np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2.0
    if angle > np.pi - 1e-6:
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diagonal(A), 0.0))
        k = int(np.argmax(axis))
        axis = A[:, k] / axis[k] if axis[k] > 0 else np.array([1.0, 0.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        skew = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
        if np.dot(axis, skew) < 0:
            axis = -axis
        return axis * angle
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2.0 * np.sin(angle))
    return axis * angle


def build_anchored_chunk(tcp_pos, tcp_quat, cam_quat_cv, anchor_idx):
    """(T,3),(T,4 wxyz),(T,4 wxyz) + anchor -> (T,6) anchored action rows.
    a_k = [R_cv0^T (p_k - p0); rotvec(R_cv0^T (R_k R0^T) R_cv0)]."""
    p = np.asarray(tcp_pos, dtype=np.float64)
    R = _quat_wxyz_to_rotmat(tcp_quat)
    R_cv0 = _quat_wxyz_to_rotmat(cam_quat_cv[anchor_idx])
    p0, R0 = p[anchor_idx], R[anchor_idx]
    out = np.empty((len(p), 6), dtype=np.float32)
    RcT = R_cv0.T
    for k in range(len(p)):
        out[k, :3] = RcT @ (p[k] - p0)
        out[k, 3:] = _rotmat_to_rotvec(RcT @ (R[k] @ R0.T) @ R_cv0)
    return out


def _rv_to_R(rv):
    a = float(np.linalg.norm(rv))
    if a < 1e-12:
        return np.eye(3)
    ax = rv / a
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * (K @ K)


# ======================================================================================
# H-series (v10) geometry — BYTE-PARITY TWINS of the pipeline's
# `policy-training-pipeline/src/converter/transforms.py`.
#
# WHY twins and not an import: the fork ships into the training container as a COPY of
# this file; the converter runs in a different container. There is no shared package, so
# the only way to guarantee the labels the fork builds are the labels the converter
# documented is to duplicate the math and PIN IT WITH A TEST. `tests/test_h_geometry_parity.py`
# asserts agreement to 1e-9 on a shared random seed against the pipeline source. If you
# touch anything below, touch the converter too, and run that test.
#
# Naming is deliberately identical to the converter's so a diff is trivial to eyeball.
# ======================================================================================

# Public aliases so the twins below read exactly like the converter's module.
quat_wxyz_to_rotmat = _quat_wxyz_to_rotmat
rotmat_to_rotvec = _rotmat_to_rotvec
rotvec_to_rotmat = _rv_to_R

# ee_link is +Y of tcp_link in the link_7/tcp frame (URDF A-000022.xacro, verified to 2e-7 m).
# ee_link is J7-INVARIANT (fixed w.r.t. gripper_frame), which is what makes a 6-DoF ee_link
# delta/twist executable by J1..J6 alone; the wrist camera is on link_6_extension and is
# therefore also J7-invariant, so ee_link motion expressed in the camera frame is fully
# decoupled from J7. J7 becomes a separate POSITIONAL channel. Ground truth (goals, place
# error, keypoints) stays in tcp_link — ee_link is ONLY the action/proprio frame.
EE_TCP_Y_OFFSET_M = 0.0595178643762819


def rotz(angle) -> np.ndarray:
    """(3,3) rotation about Z by `angle` radians."""
    c, s = np.cos(float(angle)), np.sin(float(angle))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ee_from_tcp(p_tcp, R_tcp, q7):
    """(tcp_link pose, joint-7 angle) -> ee_link pose. Exact inverse: `tcp_from_ee`."""
    R_tcp = np.asarray(R_tcp, dtype=np.float64)
    p_tcp = np.asarray(p_tcp, dtype=np.float64)
    p_ee = p_tcp + R_tcp @ np.array([0.0, EE_TCP_Y_OFFSET_M, 0.0])
    R_ee = R_tcp @ rotz(-float(q7))
    return p_ee, R_ee


def tcp_from_ee(p_ee, R_ee, q7):
    """ee_link pose + joint-7 angle -> tcp_link pose (exact inverse of `ee_from_tcp`).
    This is the transform the DEPLOY node needs: the policy acts on ee_link, but the
    suction cup (and therefore the place goal) lives at tcp_link."""
    R_ee = np.asarray(R_ee, dtype=np.float64)
    p_ee = np.asarray(p_ee, dtype=np.float64)
    R_tcp = R_ee @ rotz(float(q7))
    p_tcp = p_ee - R_tcp @ np.array([0.0, EE_TCP_Y_OFFSET_M, 0.0])
    return p_tcp, R_tcp


def goal_frame_rotation(R_cv0, goal_rotvec_cam) -> np.ndarray:
    """H1 output frame: the camera frame at the anchor, rotated by the goal's orientation.
    Actions expressed in this frame rotate WITH the goal estimate -> goal equivariance."""
    return np.asarray(R_cv0, dtype=np.float64) @ rotvec_to_rotmat(goal_rotvec_cam)


def compose_rotvecs(rotvecs) -> np.ndarray:
    """Compose a sequence of rotation vectors (applied in order) -> single rotation vector.
    Angular-velocity rows integrate by COMPOSITION (prod exp(w_k dt)), not summation."""
    R = np.eye(3)
    for rv in rotvecs:
        R = rotvec_to_rotmat(rv) @ R
    return rotmat_to_rotvec(R)


def anchored_delta(p0, R0, p_k, R_k, R_cv0) -> np.ndarray:
    """Pose at step k RELATIVE to the anchor pose, expressed in the frame `R_cv0`.
        dp = R_cv0.T @ (p_k - p0);  dR = R_cv0.T @ (R_k @ R0.T) @ R_cv0
    Inverse: p_k = p0 + R_cv0 @ dp; R_k = (R_cv0 @ rotvec_to_rotmat(drv) @ R_cv0.T) @ R0."""
    dp = np.asarray(R_cv0, np.float64).T @ (
        np.asarray(p_k, np.float64) - np.asarray(p0, np.float64))
    dR_cam = np.asarray(R_cv0, np.float64).T @ (
        np.asarray(R_k, np.float64) @ np.asarray(R0, np.float64).T) @ np.asarray(R_cv0, np.float64)
    return np.concatenate([dp, rotmat_to_rotvec(dR_cam)])


def twist_rows_from_poses(p_seq, R_seq, R_frame, dt: float) -> np.ndarray:
    """Per-interval twist rows for a pose sequence, expressed in ONE fixed frame.

    Returns (H-1, 6) float64 rows `[v(3) m/s, omega(3) rad/s]` where
        v_k     = R_frame.T @ (p_k - p_{k-1}) / dt
        omega_k = rotvec( R_frame.T @ (R_k @ R_{k-1}.T) @ R_frame ) / dt
    Position integrates EXACTLY (single fixed frame): sum(v_k*dt) == R_frame.T @ (p_last - p_0).
    Rotation integrates by composition (see `compose_rotvecs`)."""
    p_seq = np.asarray(p_seq, dtype=np.float64)
    R_seq = np.asarray(R_seq, dtype=np.float64)
    R_frame = np.asarray(R_frame, dtype=np.float64)
    rows = np.empty((len(p_seq) - 1, 6), dtype=np.float64)
    for k in range(1, len(p_seq)):
        rows[k - 1, :3] = R_frame.T @ (p_seq[k] - p_seq[k - 1]) / dt
        dR = R_frame.T @ (R_seq[k] @ R_seq[k - 1].T) @ R_frame
        rows[k - 1, 3:] = rotmat_to_rotvec(dR) / dt
    return rows


def remaining_to_goal(p_k, R_k, p_goal, R_goal, R_frame) -> np.ndarray:
    """H1 `delta`+`goal` row: the transform from pose k TO the goal, in `R_frame`.
    Contracts to zero at the goal by construction — the whole point of the goal-frame
    parameterization. Inverse: `pose_from_remaining`."""
    return anchored_delta(p_k, R_k, p_goal, R_goal, R_frame)


def pose_from_remaining(p_goal, R_goal, u, R_frame):
    """Inverse of `remaining_to_goal` — reconstruct pose k from the goal and the row."""
    u = np.asarray(u, dtype=np.float64)
    R_frame = np.asarray(R_frame, dtype=np.float64)
    p_k = np.asarray(p_goal, dtype=np.float64) - R_frame @ u[:3]
    R_k = (R_frame @ rotvec_to_rotmat(-u[3:]) @ R_frame.T) @ np.asarray(R_goal, dtype=np.float64)
    return p_k, R_k


# ---------------------------------------------------------------- H action space (contract §3)
# `action_dim = 7` always: [6-DoF ee_link part | dJ7]. Two orthogonal knobs give the locked 2x2.
# Row COUNT is per-mode (contract §3, FINAL 2026-07-31). The rule that fixes it:
# A ROW EXISTS IFF THE MODEL PREDICTS IT AND IT CARRIES INFORMATION.
#   twist+cam0 (H0)       15 rows  w_k, k=1..15   (no zero row exists by construction)
#   twist+goal (H1)       15 rows  w_k, k=1..15   (w_15 -> 0 settle + integral constraint)
#   delta+cam0 (H-delta)  15 rows  a_k, k=1..15   (a_0 == 0 identically -> NOT a row at all;
#                                                  nothing is ever prepended anywhere)
#   delta+goal (H1-delta) 16 rows  u_k, k=0..15   (u_15 -> 0; u_0 IS predicted, see h_action_rows)
H_ACTION_PARAMS = ("twist", "delta")
H_ACTION_FRAMES = ("cam0", "goal")
H_ACTION_DIM = 7

# The H observation dict (contract §4). NOTE what is absent and why:
#   - no `prev_action`  : the free-run collapse verdict (chained deltas are the copycat channel)
#   - no `goal_prior`   : G0 verdict (an input goal latch gets ignored / shortcut)
#   - no `suction`      : always-on during place => zero variance => dead dimension
#   - no goal FRAME     : the goal frame is an OUTPUT coordinate system, never an input token
H_OBS_KEYS = ('head_cam', 'depth_cam', 'agent_pos', 'task', 'grasp_off',
              'ego', 'twist', 'tcpcam', 'dt', 'twist_hist')

# Zarr keys the H MODEL consumes. Fail-loud list: the F-series lost a whole run because
# `arm_kpts_*` were silently absent from buffer_keys (the heads trained against nothing).
# Never make this soft. `prev_action` is deliberately NOT here — v10 retains it in the zarr
# for eval/normalizer compat but it is not a model input (free-run collapse verdict).
H_REQUIRED_ZARR_KEYS = (
    'head_camera', 'depth',
    'tcp_pos_w', 'tcp_quat_w', 'cam_pos_w', 'cam_quat_cv',
    'agent_pos', 'task',
    'arm_kpts_uv', 'arm_kpts_cam',
    'goal_pos_cam', 'goal_rot_cam',
    'ee_pos_w', 'ee_quat_w', 'q7',
    'ee_goal_pos_w', 'ee_goal_quat_w', 'q7_goal',
    'panel_goal_cam', 'rail_a_cam', 'grasp_offset',
    'tcp_egomotion', 'tcp_twist', 'twist_hist',
    'phase_id', 'done',
)

# Anchors whose horizon does NOT reach the segment end make the goal-coincidence integral
# constraint (contract §3.2) mathematically FALSE: sum(v_k*dt) == p_15 - p_0, which only equals
# the remaining transform p_G - p_0 when p_15 == p_G. Applying it anyway would inject an error
# equal to the un-covered distance (>150mm early in a descent). We therefore ship a per-sample
# `h_goal_inreach` mask and apply the integral loss only where the constraint holds. Deliberate
# addition to §3.2 (no new config knob) — documented in the H report.
#
# THE MASK IS EVALUATED AGAINST THE **CLEAN GT GOAL** (contract §9.2). It is a statement about
# LABEL GEOMETRY — "does this window's horizon actually reach the place goal?" — and label
# geometry does not move when scheduled sampling perturbs the OUTPUT FRAME. Measuring it against
# the noised goal instead compares a 5 mm tolerance against a 15 mm-sigma 3-D perturbation:
# P(pass) ~ 0.4%, which is not a mask but an off switch. Measured before the fix: in-reach
# fraction 0.55 clean -> 0.000 noised (0/80 samples), and the flagship arm's signature loss
# logged 0.0 — i.e. "perfect" — on every batch with nothing supervised behind it.
H_GOAL_INTEGRAL_TOL_M = 0.005      # 5 mm == the terminal precision target

# Goal-frame normalizer coverage (contract §9.5): how many fixed-seed goal-frame-noise draws are
# appended to each window's CLEAN rows when fitting the action normalizer. 0 reproduces the old
# clean-only fit. 2 costs one extra pass over the pose windows (no image IO) at startup and
# covers the +-3 sigma tail of a 15 mm / 1.5 deg perturbation on the near-terminal rows.
H_NORM_NOISE_DRAWS = 2


def h_action_rows(action_param: str, action_frame: str, pose_horizon: int) -> int:
    """Number of rows for a mode (contract §3 table, FINAL). `pose_horizon` = H poses.

    15 for twist+cam0 / twist+goal / delta+cam0 (k = 1..H-1), 16 for delta+goal (k = 0..H-1).

    Why the counts are NOT uniform. `delta`+`cam0`'s k=0 row is `ad(P_0, P_0, R_cv0)` — the zero
    vector, always, for every sample and every checkpoint. It is therefore not predicted and not
    shipped: a constant the exporter inserts itself carries no information, and a runtime gate on
    it cannot distinguish "mis-anchored" from "wants to move" (the G1 field run rejected 26% of
    HEALTHY inferences exactly that way). Mis-anchoring is a train/export bug class — a fixed
    property of a checkpoint — so it is checked once at publish/desk-gate time, not 15x/s.

    `delta`+`goal`'s k=0 row is the opposite case and IS shipped. `u_0 = ad(P_0, G, R_g)` is the
    remaining transform from the anchor pose to the goal: at deploy `G` is the model's OWN
    prediction, so `u_0` is that self-predicted goal re-expressed in action space. The same
    estimate is simultaneously the output coordinate system and a predicted output, which is the
    flagship's structural coupling made explicit and measurable — pinned from one side by the
    goal-coincidence integral constraint and from the other by `goal_consistency_weight`. It is
    load-bearing, not decorative. See `ManiFlowTransformerImagePolicy._h_goal_integral`.

    `action_row_k_start` (1, 1, 1, 0) travels in the ONNX `metadata_props` next to `action_rows`;
    consumers MUST read both rather than assume a count.
    """
    assert action_param in H_ACTION_PARAMS, action_param
    assert action_frame in H_ACTION_FRAMES, action_frame
    if action_param == "delta" and action_frame == "goal":
        return int(pose_horizon)            # k = 0..H-1
    return int(pose_horizon) - 1            # k = 1..H-1


def h_action_row_k_start(action_param: str, action_frame: str) -> int:
    """Chunk-step index `k` of the FIRST row (contract §3): 0 for delta+goal, else 1."""
    assert action_param in H_ACTION_PARAMS, action_param
    assert action_frame in H_ACTION_FRAMES, action_frame
    return 0 if (action_param == "delta" and action_frame == "goal") else 1


def build_h_action_rows(p_ee, R_ee, q7, p_goal, R_goal, q7_goal,
                        R_frame, action_param, action_frame, dt):
    """Contract §3 action rows for ONE sample. All poses are **ee_link**, world frame.

    p_ee (H,3), R_ee (H,3,3), q7 (H,) on the control grid starting AT THE ANCHOR (index 0 ==
    the sample's obs time). (p_goal, R_goal, q7_goal) = the episode-constant ee_link goal.
    `R_frame` = the expression frame: R_cv0 for `cam0`, R_g (goal frame) for `goal`.

    Returns (rows, 7) float64: [..6-DoF.., dJ7].

    dJ7 is POSITIONAL in every mode (J7's Maxon is on a position service — a "J7 velocity"
    would just be re-integrated downstream). Per contract §3 it depends on the FRAME only:
        cam0: dj7_k = q7[k] - q7[0]          (progress from the anchor)
        goal: dj7_k = q7_goal - q7[k]        (remaining to the goal, contracts to 0)
    """
    p_ee = np.asarray(p_ee, dtype=np.float64)
    R_ee = np.asarray(R_ee, dtype=np.float64)
    q7 = np.asarray(q7, dtype=np.float64).reshape(-1)
    H = len(p_ee)
    assert R_ee.shape == (H, 3, 3) and q7.shape == (H,), (R_ee.shape, q7.shape)

    if action_param == "twist":
        six = twist_rows_from_poses(p_ee, R_ee, R_frame, dt)                    # (H-1,6)
        ks = np.arange(1, H)
    elif action_frame == "cam0":                                               # delta + cam0
        six = np.stack([anchored_delta(p_ee[0], R_ee[0], p_ee[k], R_ee[k], R_frame)
                        for k in range(1, H)])                                  # (H-1,6)
        ks = np.arange(1, H)
    else:                                                                      # delta + goal
        six = np.stack([remaining_to_goal(p_ee[k], R_ee[k], p_goal, R_goal, R_frame)
                        for k in range(H)])                                     # (H,6)
        ks = np.arange(0, H)

    if action_frame == "cam0":
        dj7 = q7[ks] - q7[0]
    else:
        dj7 = float(q7_goal) - q7[ks]
    return np.concatenate([six, dj7.reshape(-1, 1)], axis=1)


# `grasp_offset` composition convention — SETTLED EMPIRICALLY (contract §1.2a, 2026-07-31).
# `grasp_offset` stores T_tcp_panel (the panel pose IN THE TCP FRAME), NOT pre-inverted, so the
# TCP goal is recovered by composing with its INVERSE. Measured residuals against the stored goal:
#     T_panel_goal @ inv(T_tcp_panel)   0.28-0.90 mm   <- THIS ONE
#     T_panel_goal @      T_tcp_panel   274.6 mm
#     T_railA      @ inv(T_tcp_panel)   620-1020 mm    (wrong LANDMARK, see §1.2b)
#     T_railA      @      T_tcp_panel   879-1019 mm
GRASP_OFFSET_INVERT = True

# THE MILLIMETRE TERM IS THE DISCRIMINATOR, NOT THE ANGLE. This grasp is a ~180 deg rotation and a
# 180 deg rotation is its own inverse, so a wrong (non-inverted) convention shifts the reconstructed
# ORIENTATION by only ~0.07 deg while shifting POSITION by 275 mm. Any check that looks at
# orientation alone silently passes the wrong convention. Gate on position.
#
# Tolerance is 50 mm, matching the converter's `grasp_goal_max_mm`, and NOT because the geometry is
# sloppy: the stored TCP goal is currently built as the conjugation
# `T_tcp_end @ inv(T_panel_end) @ T_panel_goal`, which differs from the rigid-grasp composition by
# `(R_grasp.T - I) @ R_tcp_end.T @ (p_panel_goal - p_panel_end)` — bounded by 2*place_err, i.e.
# <=~40 mm at the 20 mm place_err QA gate and 0.28-0.90 mm in practice. The converter owner is
# switching to the rigid form, after which this residual becomes ~0; expect the logged number to
# DROP sub-millimetre and do not read that as a regression. A wrong convention is 275-1020 mm out,
# so 50 mm still separates the failure class by 5-20x.
GRASP_PROBE_MAX_POS_M = 0.050
GRASP_PROBE_MAX_ROT_RAD = np.radians(1.0)


def panel_to_ee_goal_maps(R_tcp0_cam, cam_minus_ee_cam, grasp_offset, q7_goal,
                          R_c0_from_cimg=None,
                          invert_offset: bool = GRASP_OFFSET_INVERT):
    """Precompute the (tiny) constant maps that turn a PREDICTED `panel_goal_cam` into the ee_link
    goal + the goal frame, all expressed in the camera frame at the anchor ("C0").

    WHY THE PANEL GOAL AND NOT `rail_a` (contract §1.2b, measured on real episodes): in the rail
    frame the panel goal has x fixed at exactly module_width/2 but **y free over 0.19 m** and z
    jittering 24 mm. `goal = f(rail_a, grasp_offset)` is therefore UNDER-DETERMINED by one
    translational DoF — a panel is placed flush against the previously installed panel, so the
    along-rail DoF comes from the NEIGHBOUR EDGE (keypoints 5-6), not from the rail. The place head
    consequently regresses the panel goal directly; `rail_a_cam` survives only as an aux landmark.

    WHY this lives in the dataset: the policy only sees NORMALIZED tokens and has no access to the
    raw camera/TCP pose the composition needs. These four constants make the policy-side
    composition three matmuls, exactly and differentiably, with no new obs input.

    Derivation. `panel_goal_cam` is camera-frame-ABSOLUTE (unlike `rail_a_cam`), and per contract
    §9.3 the place head is supervised in the frame of the camera that captured the MOST RECENT
    USED IMAGE ("Cimg"), which under the §5 timing augmentation is `image_lag_frames` control
    frames BEFORE the anchor. So with Y = exp(panel_goal_cam[3:6]) and c_p = panel_goal_cam[0:3]
    taken at THAT frame:
        R_cimg.T @ R_panel_goal = Y       and     R_cimg.T @ (p_panel_goal - p_cam_img) = c_p,
    and `A` := R_c0_from_cimg = R_cv0.T @ R_cimg carries Cimg quantities into C0 (A == I whenever
    the image frame IS the anchor, i.e. lag 0 — which is every eval/export sample).
    With Q := inv(T_tcp_panel) (invert_offset=True; §1.2a) = (Rq, pq) = (R_off.T, -R_off.T @ p_off),
        T_tcp_goal = T_panel_goal @ Q  =>  R_gt = R_panel_goal @ Rq,
                                           p_gt = p_panel_goal + R_panel_goal @ pq
    then ee_link (contract §0): R_ge = R_gt @ Rz(-q7_goal), p_ge = p_gt + R_gt @ [0, +Y_OFF, 0].
    Pushing into C0 and re-anchoring the position at p_ee0:
        p_goal_ee (in C0, rel. p_ee0) = A @ (c_p + Y @ tvec) + cam_minus_ee_cam
        R_goal_ee (in C0)             = A @ Y @ R_ee
        goal FRAME (in C0)            = A @ Y @ R_frame    [ == exp(goal_rot_cam) ]
    `cam_minus_ee_cam` = R_cv0.T @ (p_cam_img - p_ee0) re-anchors from the camera THAT TOOK THE
    IMAGE (where `panel_goal_cam` lives) to p_ee0 (where the action rows are anchored). Note both
    `A` and this offset use the IMAGE camera; `R_tcp0_cam` uses the ANCHOR tcp pose, because the
    goal frame is defined by the anchor's TCP orientation (§3).

    `invert_offset=False` selects the WRONG convention and exists only so
    `_probe_grasp_offset_convention` can measure the discrimination margin. Do not set it.
    """
    R_tcp0_cam = np.asarray(R_tcp0_cam, dtype=np.float64)
    cam_minus_ee_cam = np.asarray(cam_minus_ee_cam, dtype=np.float64).reshape(3)
    A = (np.eye(3) if R_c0_from_cimg is None
         else np.asarray(R_c0_from_cimg, dtype=np.float64).reshape(3, 3))
    go = np.asarray(grasp_offset, dtype=np.float64).reshape(7)
    p_off, R_off = go[:3], quat_wxyz_to_rotmat(go[3:])
    if invert_offset:                       # Q = inv(T_tcp_panel)   <- contract §1.2a
        Rq, pq = R_off.T, -(R_off.T @ p_off)
    else:                                   # Q = T_tcp_panel        (diagnostic only)
        Rq, pq = R_off, p_off
    tvec = pq + Rq @ np.array([0.0, EE_TCP_Y_OFFSET_M, 0.0])
    return {
        "tvec": tvec.astype(np.float32),                            # (3,)
        "toff": cam_minus_ee_cam.astype(np.float32),                # (3,)
        "A": A.astype(np.float32),                                  # (3,3) Cimg -> C0
        "R_ee": (Rq @ rotz(-float(q7_goal))).astype(np.float32),    # (3,3)
        "R_frame": (Rq @ R_tcp0_cam.T).astype(np.float32),          # (3,3)
    }


class LumiPlaceImageDataset(BaseDataset):
    def __init__(self,
                 zarr_path,
                 horizon=1,
                 pad_before=0,
                 pad_after=0,
                 seed=42,
                 val_ratio=0.0,
                 max_train_episodes=None,
                 task_name=None,
                 augmentation=None,
                 depth_input="tiled",        # "tiled" | "bands" (H3DP) | "xyz" (point-map)
                 depth_band_edges=None,      # meters; N edges -> N-1 channels (bands mode)
                 depth_band_soft=DEPTH_BAND_SOFT,
                 use_depth=True,             # C2-rgb control arm: no depth stream at all
                 # ---- H-series (v10) action space + timing (contract §3/§4/§5).
                 # action_param=None keeps the v3 ANCHORED-TCP behaviour byte-for-byte.
                 action_param=None,          # None (v3 legacy) | "twist" | "delta"
                 action_frame="cam0",        # "cam0" | "goal"
                 n_obs_steps=2,              # obs frames; needed to derive the timing padding
                 control_hz=10.0,            # control-grid rate -> Delta t for twist rows
                 **kwargs):
        super().__init__()
        self.task_name = task_name
        self.use_depth = bool(use_depth)
        self.depth_input = depth_input
        self.depth_band_edges = tuple(depth_band_edges) if depth_band_edges else DEPTH_BAND_EDGES_M
        self.depth_band_soft = float(depth_band_soft)
        if depth_input == "bands":
            self.depth_channels = len(self.depth_band_edges) - 1
        elif depth_input == "xyz":
            self.depth_channels = 4          # X, Y, Z (camera frame, /XYZ_SCALE_M) + valid
        else:
            self.depth_channels = 3
        _mode_label = (f'H-series ({action_param}+{action_frame})' if action_param is not None
                       else 'v6 anchored')
        cprint(f'Loading LumiPlaceImageDataset ({_mode_label}, depth='
               f'{depth_input if self.use_depth else "OFF"}, '
               f'{self.depth_channels}ch) from {zarr_path}', 'green')

        # --------------------------------------------------------------- H-series mode switch
        self.action_param = action_param
        self.action_frame = str(action_frame)
        self.h_mode = action_param is not None
        self.n_obs_steps = int(n_obs_steps)
        self.control_hz = float(control_hz)
        self.h_dt = 1.0 / self.control_hz
        if self.h_mode:
            assert action_param in H_ACTION_PARAMS, f"action_param={action_param!r}"
            assert self.action_frame in H_ACTION_FRAMES, f"action_frame={action_frame!r}"
            assert self.use_depth and depth_input == "xyz", (
                "H requires the xyz point-map (PointNet K tokens + 3D-aware RGB tokens + "
                f"kpt z-fusion all read it); got use_depth={use_depth} depth_input={depth_input}")

        _zk = set(zarr.open(str(zarr_path), mode='r')['data'].keys())
        if self.h_mode:
            # FAIL LOUD on a missing key. The F-series lost an entire run to `arm_kpts_*`
            # silently absent from buffer_keys (the heads trained against nothing and the
            # loss looked "fine"). Never soften this into a hasattr/back-compat loop.
            missing = [k for k in H_REQUIRED_ZARR_KEYS if k not in _zk]
            if missing:
                raise KeyError(
                    f"H-series (action_param={action_param}) requires converter v10 "
                    f"(CONVERTER_VERSION 2.4.0) keys; zarr at {zarr_path} is MISSING {missing}. "
                    f"Present: {sorted(_zk)}")
            buffer_keys = list(H_REQUIRED_ZARR_KEYS)
            self.has_agent_pos = True
        else:
            buffer_keys = ['head_camera', 'tcp_pos_w', 'tcp_quat_w', 'cam_pos_w',
                           'cam_quat_cv', 'prev_action', 'gravity_cam', 'goal_pos_cam',
                           'goal_rot_cam', 'task']
            if self.use_depth:
                buffer_keys.insert(1, 'depth')
            # F-series: load rail/tube keypoint targets if present (v8+; back-compat with v7).
            for _k in ('arm_kpts_uv', 'arm_kpts_cam', 'agent_pos'):
                if _k in _zk:
                    buffer_keys.append(_k)
            self.has_agent_pos = 'agent_pos' in _zk
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=buffer_keys)

        # C2 "xyz" point-map mode: unproject each depth pixel to metric camera-frame
        # coordinates using the stored intrinsics — the goal's 3D position becomes a
        # READOUT from the input rather than a monocular-scale inference (the v0.1.4
        # eval showed a ~24%-of-distance fractional ranging error = RGB scale cues).
        # Values are crop-safe: a pixel's (X,Y,Z) does not depend on where the crop
        # places it. camera_K is [fx, fy, cx, cy] at the STORED resolution.
        if self.use_depth and depth_input == "xyz":
            K = np.asarray(zarr.open(str(zarr_path), mode='r')['meta']['camera_K'][:],
                           dtype=np.float64).reshape(-1)
            fx, fy, cx, cy = float(K[0]), float(K[1]), float(K[2]), float(K[3])
            S = int(self.replay_buffer['depth'].shape[-1])
            us = np.arange(S, dtype=np.float32)
            vs = np.arange(S, dtype=np.float32)
            uu, vv = np.meshgrid(us, vs)                 # (S,S), uu = x-pixel, vv = y-pixel
            self._xmap = ((uu - cx) / fx).astype(np.float32)
            self._ymap = ((vv - cy) / fy).astype(np.float32)

        # F-series: normalized intrinsics [fx/W, fy/H, cx/W, cy/H] for re-projecting rotated
        # keypoints under GPU SO(2) aug. camera_K is stored at resolution S and the converter
        # normalized arm_kpts_uv by full-res (u/W == u_stored/S), so cam_k_norm = camera_K / S.
        self.cam_k_norm = None
        try:
            _K = np.asarray(zarr.open(str(zarr_path), mode='r')['meta']['camera_K'][:],
                            dtype=np.float64).reshape(-1)
            _S = float(self.replay_buffer['head_camera'].shape[-1])
            self.cam_k_norm = [float(_K[0] / _S), float(_K[1] / _S),
                               float(_K[2] / _S), float(_K[3] / _S)]
        except Exception:
            self.cam_k_norm = None

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        if max_train_episodes is None:
            max_train_episodes = self.replay_buffer.n_episodes - np.sum(val_mask)
        cprint(f'Maximum training episodes: {max_train_episodes}', 'yellow')
        cprint(f'Validation ratio: {val_ratio}', 'yellow')
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        # ------------------------------------------------- H window geometry (contract §3/§5)
        # v3 semantics: the sampler window IS the action chunk (`horizon` rows, anchor at
        # pad_before). H semantics differ: `horizon` is the number of POSES starting AT the
        # anchor (H=16 -> 15 executable rows), and the timing augmentation needs image frames
        # BEFORE the anchor. So the window is longer than the chunk:
        #     pad_before_eff = max_image_lag + max_obs_gap * (n_obs_steps - 1)   (= 3 for To=2)
        #     sequence_length = pad_before_eff + pose_horizon                    (= 19 for H=16)
        # The support of the timing jitter is FIXED by contract §5 ({0,1,2} gap, {0,1} lag), so
        # the padding does not depend on the probability values -> train and val windows are
        # identical geometry regardless of augmentation settings (the normalizer must be fit
        # over the exact rows training sees).
        self.h_max_gap = 2
        self.h_max_lag = 1
        self.pose_horizon = int(horizon)
        if self.h_mode:
            self.action_rows = h_action_rows(
                self.action_param, self.action_frame, self.pose_horizon)
            h_obs_pad = self.h_max_lag + self.h_max_gap * (self.n_obs_steps - 1)
            pad_before = max(int(pad_before), h_obs_pad)
            # Let EVERY frame be an anchor: the terminal frames are exactly where the settle
            # rows (twist -> 0 / u -> 0) and done==1 live, and dropping them would delete the
            # terminal-precision supervision. SequenceSampler pads by repeating the last frame,
            # which IS the settle semantics (a stationary tool tip).
            pad_after = max(int(pad_after), self.pose_horizon - 1)
            seq_len = pad_before + self.pose_horizon
            cprint(f"[H] action={self.action_param}/{self.action_frame} rows={self.action_rows} "
                   f"dim={H_ACTION_DIM} | poses={self.pose_horizon} window={seq_len} "
                   f"anchor={pad_before} pad_after={pad_after} dt={self.h_dt:.3f}s", "green")
        else:
            self.action_rows = self.pose_horizon
            seq_len = int(horizon)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=seq_len,
            pad_before=pad_before, pad_after=pad_after, episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.seq_len = seq_len
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed
        # ---- epoch counter for the per-sample augmentation RNG (contract §9.4) -------------
        # WHY A SHARED-MEMORY TENSOR AND NOT AN INT: the train DataLoader runs with
        # `persistent_workers: True`, so each worker holds a FORKED COPY of this dataset for the
        # whole run. A plain `self._epoch = 0` written in the parent would never reach them and
        # `set_epoch` would be a silent no-op — the exact bug class §9.4 exists to kill. A
        # shared-memory tensor is visible to the forked children, mirroring the policy-side
        # `h_epoch` buffer pattern (and, like it, resume-safe: the workspace passes its restored
        # `self.epoch`, so a resumed run continues the augmentation schedule instead of replaying
        # epoch 0's draws).
        self._epoch = torch.zeros((), dtype=torch.long).share_memory_()

        aug = dict(DEFAULT_AUGMENTATION)
        if augmentation:
            aug.update({k: v for k, v in dict(augmentation).items() if v is not None})
        self.aug = aug
        self.augment = bool(aug.get("enable", True))
        self.rot_aug_deg = float(aug.get("rot_aug_deg", 0.0))
        self.photo_brightness = float(aug.get("photo_brightness", 0.0))
        self.photo_contrast = float(aug.get("photo_contrast", 0.0))
        self.photo_saturation = float(aug.get("photo_saturation", 0.0))
        self.photo_hue = float(aug.get("photo_hue", 0.0))
        self.photo_blur_p = float(aug.get("photo_blur_p", 0.0))
        self.photo_noise = float(aug.get("photo_noise", 0.0))
        self.photo_erase_p = float(aug.get("photo_erase_p", 0.0))
        # v7 GPU-aug: when true, SO(2) roll + photometric are done on the CUDA batch in the
        # training loop (maniflow.model.vision_2d.gpu_augment), NOT here per-sample on the CPU.
        self.gpu_offload = bool(aug.get("gpu_offload", False))
        self._episode_ends = self.replay_buffer.episode_ends[:]

        self.zarr_path = zarr_path
        self.train_episodes_num = np.sum(train_mask)
        self.val_episodes_num = np.sum(val_mask)

        # H timing-aug distributions, validated once here (a bad prob vector must not surface
        # as a per-sample numpy error inside a DataLoader worker).
        self.obs_gap_probs = self._h_probs(aug.get("obs_gap_probs"), 3, [0.0, 1.0, 0.0])
        self.image_lag_probs = self._h_probs(aug.get("image_lag_probs"), 2, [1.0, 0.0])
        if self.h_mode:
            cprint(f"[H] timing aug: obs_gap_probs={self.obs_gap_probs} "
                   f"image_lag_probs={self.image_lag_probs} (eval/export = gap 1, lag 0)", "green")
            self.grasp_offset_invert = self._probe_grasp_offset_convention()

    def set_epoch(self, epoch: int):
        """Advance the per-sample augmentation RNG (contract §9.4). Called by the workspace once
        per epoch, BEFORE the epoch's iteration starts.

        Without this the seed `[seed, 7777777, ep_idx, sample_idx]` is a pure function of the
        sample index, so every sample got the SAME timing gap/lag, the SAME goal-frame noise and
        the SAME proprio DR offsets for the entire run: scheduled sampling degenerates into a
        fixed per-sample bias the network can memorise, and the timing augmentation stops being
        augmentation at all (it becomes a fixed re-labelling of the dataset). The epoch enters
        the seed rather than driving a stateful generator so a sample stays DETERMINISTIC within
        an epoch (reproducible, worker-count-independent) while differing across epochs."""
        self._epoch.fill_(int(epoch))

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    @staticmethod
    def _h_probs(value, n, default):
        """Validate an H timing-aug probability vector; None -> the nominal default."""
        if value is None:
            return list(default)
        p = [float(x) for x in list(value)]
        if len(p) != n or min(p) < 0.0 or abs(sum(p) - 1.0) > 1e-6:
            raise ValueError(f"H timing probs must be {n} non-negative values summing to 1; got {p}")
        return p

    def _goal_compose_residual(self, idx: int, invert: bool):
        """Reconstruct the ee_link goal from the STORED `panel_goal_cam` + `grasp_offset` at one
        frame and return (rotation error [rad], position error [m]) vs the stored goal."""
        rb = self.replay_buffer
        R_cv0 = quat_wxyz_to_rotmat(rb['cam_quat_cv'][idx])
        R_tcp0 = quat_wxyz_to_rotmat(rb['tcp_quat_w'][idx])
        p_cam0 = np.asarray(rb['cam_pos_w'][idx], np.float64)
        p_ee0 = np.asarray(rb['ee_pos_w'][idx], np.float64)
        m = panel_to_ee_goal_maps(
            R_cv0.T @ R_tcp0, R_cv0.T @ (p_cam0 - p_ee0),
            rb['grasp_offset'][idx], float(np.asarray(rb['q7_goal'][idx]).reshape(-1)[0]),
            invert_offset=invert)
        pg = np.asarray(rb['panel_goal_cam'][idx], np.float64).reshape(6)
        Y = rotvec_to_rotmat(pg[3:])
        # A == I here (this probe evaluates label and anchor at the SAME frame), but the formula
        # is written with it so it stays a line-for-line twin of the policy's `_h_reframe`.
        A = np.asarray(m["A"], np.float64)
        # (a) goal FRAME: A @ Y @ R_frame must equal exp(goal_rot_cam)
        F_pred = A @ Y @ np.asarray(m["R_frame"], np.float64)
        F_gt = rotvec_to_rotmat(np.asarray(rb['goal_rot_cam'][idx], np.float64).reshape(3))
        e_rot = float(np.abs(rotmat_to_rotvec(F_pred @ F_gt.T)).max())
        # (b) ee goal POSITION in C0 relative to p_ee0 — the discriminating term
        p_pred = A @ (pg[:3] + Y @ np.asarray(m["tvec"], np.float64)) \
            + np.asarray(m["toff"], np.float64)
        p_gt = R_cv0.T @ (np.asarray(rb['ee_goal_pos_w'][idx], np.float64) - p_ee0)
        e_pos = float(np.linalg.norm(p_pred - p_gt))
        return e_rot, e_pos

    def _probe_grasp_offset_convention(self, n_probe: int = 16) -> bool:
        """Verify the `grasp_offset` composition against the STORED ground truth. FAIL LOUD.

        Contract §1.2a settled the convention empirically (`invert=True`, i.e. `T_tcp_goal =
        T_panel_goal @ inv(T_tcp_panel)`) and says consumers must not *choose* at runtime. This is
        not a chooser — it is a guard, and it stays because the earlier version of this probe
        would have hard-failed on BOTH of its branches and that firing WOULD HAVE BEEN CORRECT:
        it was composing from `rail_a`, and the landmark is the PANEL GOAL (§1.2b). A cheap check
        that catches an architecture error is worth keeping even once the convention is known.

        Two traps it is built to avoid:
          1. ORIENTATION CANNOT DISCRIMINATE. The grasp is ~180 deg and a 180 deg rotation is its
             own inverse, so the wrong convention moves orientation by ~0.07 deg and position by
             275 mm. The gate is therefore on POSITION (rotation is reported, and gated loosely
             only to catch a gross frame error).
          2. The stored goal currently carries the converter's conjugation-vs-rigid-grasp
             difference (<= ~2*place_err). Hence GRASP_PROBE_MAX_POS_M = 50 mm, not 1 mm — see the
             note on that constant.

        Returns `GRASP_OFFSET_INVERT`; raises if the settled convention does not reproduce the
        stored goal."""
        rb = self.replay_buffer
        n = int(rb['panel_goal_cam'].shape[0])
        rng = np.random.default_rng([self.seed, 20260731])
        idxs = [int(i) for i in rng.choice(n, size=min(n_probe, n), replace=False)]
        errs = {}
        for invert in (True, False):
            rs, ps = zip(*(self._goal_compose_residual(i, invert) for i in idxs))
            errs[invert] = (max(rs), max(ps))
        e_rot, e_pos = errs[GRASP_OFFSET_INVERT]
        e_pos_other = errs[not GRASP_OFFSET_INVERT][1]
        if e_pos > GRASP_PROBE_MAX_POS_M or e_rot > GRASP_PROBE_MAX_ROT_RAD:
            raise ValueError(
                "grasp_offset composition does NOT reproduce the stored ee_link goal "
                f"(contract §1.2a says invert={GRASP_OFFSET_INVERT}). "
                f"invert=True err(rot_rad, pos_m)={errs[True]}, invert=False={errs[False]}; "
                f"gates {GRASP_PROBE_MAX_ROT_RAD:.4f} rad / {GRASP_PROBE_MAX_POS_M} m.\n"
                "Read the POSITION column, not the angle — a ~180 deg grasp is its own inverse, "
                "so a wrong convention barely moves the angle.\n"
                "If BOTH branches are ~0.6-1 m out the LANDMARK is wrong, not the sign: the goal "
                "is the PANEL GOAL frame, not rail_a (§1.2b). Do NOT train through this.")
        # A healthy dataset must also show the failure class is far away; if the two branches are
        # indistinguishable the probe is not actually testing anything (e.g. a degenerate grasp).
        if e_pos_other < 10.0 * max(e_pos, 1e-4):
            cprint(f"[H] WARNING: grasp convention probe has a weak margin "
                   f"(correct {e_pos * 1000:.2f} mm vs wrong {e_pos_other * 1000:.2f} mm) — it "
                   f"would not catch an inverted store on this data.", "yellow")
        cprint(f"[H] grasp_offset composition verified from panel_goal_cam: "
               f"invert={GRASP_OFFSET_INVERT}, residual {e_pos * 1000:.3f} mm / "
               f"{np.degrees(e_rot):.4f} deg (gate {GRASP_PROBE_MAX_POS_M * 1000:.0f} mm; "
               f"wrong-convention branch {e_pos_other * 1000:.1f} mm)", "green")
        return GRASP_OFFSET_INVERT

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=self.seq_len,
            pad_before=self.pad_before, pad_after=self.pad_after,
            episode_mask=~self.train_mask)
        val_set.train_mask = ~self.train_mask
        val_set.augment = False  # validation runs clean
        return val_set

    def _all_anchored_actions(self):
        """Anchored chunks for EVERY train-sampler window — used to fit the action
        normalizer over the exact tensors training sees (not the raw absolute poses).
        Uses a pose-only sampler so image arrays are never sliced here (fitting over
        ~2-3k windows would otherwise churn GBs of head_camera/depth copies)."""
        pose_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=self.horizon,
            pad_before=self.pad_before, pad_after=self.pad_after,
            keys=['tcp_pos_w', 'tcp_quat_w', 'cam_quat_cv'],
            episode_mask=self.train_mask)
        chunks = []
        for idx in range(len(pose_sampler)):
            s = pose_sampler.sample_sequence(idx)
            anchor = min(self.pad_before, len(s['tcp_pos_w']) - 1)
            chunks.append(build_anchored_chunk(
                s['tcp_pos_w'], s['tcp_quat_w'], s['cam_quat_cv'], anchor))
        return np.concatenate(chunks, axis=0)

    # ------------------------------------------------------------------ H row construction
    def _h_window_geometry(self, s, anchor):
        """Extract the H pose geometry from a sampled window `s` at `anchor`.

        Everything the row builder + the policy-side goal composition needs, in ONE place so
        the normalizer pass and __getitem__ cannot drift apart. Returns a dict of float64.
        """
        H = self.pose_horizon
        sl = slice(anchor, anchor + H)
        p_ee = np.asarray(s['ee_pos_w'][sl], np.float64)
        R_ee = quat_wxyz_to_rotmat(np.asarray(s['ee_quat_w'][sl], np.float64))
        q7 = np.asarray(s['q7'][sl], np.float64).reshape(-1)
        assert len(p_ee) == H, f"window too short: {len(p_ee)} < pose_horizon {H}"
        R_cv0 = quat_wxyz_to_rotmat(np.asarray(s['cam_quat_cv'][anchor], np.float64))
        # goal keys are episode-constant and tiled -> the anchor row is the whole episode
        p_goal = np.asarray(s['ee_goal_pos_w'][anchor], np.float64)
        R_goal = quat_wxyz_to_rotmat(np.asarray(s['ee_goal_quat_w'][anchor], np.float64))
        q7_goal = float(np.asarray(s['q7_goal'][anchor]).reshape(-1)[0])
        goal_rot_cam = np.asarray(s['goal_rot_cam'][anchor], np.float64).reshape(3)
        return dict(p_ee=p_ee, R_ee=R_ee, q7=q7, R_cv0=R_cv0,
                    p_goal=p_goal, R_goal=R_goal, q7_goal=q7_goal, goal_rot_cam=goal_rot_cam)

    def _h_goal_perturbation(self, sample_rng):
        """Scheduled-sampling noise on the goal ESTIMATE (contract §3.3). Returns (dR, dp) in
        WORLD coordinates. dR left-multiplies both the goal rotation and the goal FRAME, which
        is exactly equivalent to perturbing `goal_rot_cam` (R_g' = dR @ R_g); dp shifts the goal
        position, which only the `delta`+`goal` rows depend on. Zero knobs => identity."""
        s_mm = float(self.aug.get("goal_frame_noise_mm", 0.0) or 0.0)
        s_deg = float(self.aug.get("goal_frame_noise_deg", 0.0) or 0.0)
        dR = np.eye(3)
        dp = np.zeros(3)
        if s_deg > 0.0:
            dR = rotvec_to_rotmat(sample_rng.normal(scale=np.radians(s_deg), size=3))
        if s_mm > 0.0:
            dp = sample_rng.normal(scale=s_mm / 1000.0, size=3)
        return dR, dp

    def _h_build_rows(self, g, dR=None, dp=None):
        """Build the (rows,7) action tensor + the goal-frame bookkeeping the policy needs.

        `g` comes from `_h_window_geometry`; (dR, dp) is the optional goal perturbation.
        Returns (action, extras) where extras carries the C0-frame tensors used by the
        goal-frame scheduled sampling and the goal-coincidence losses."""
        dR = np.eye(3) if dR is None else dR
        dp = np.zeros(3) if dp is None else dp
        R_cv0 = g['R_cv0']
        p_goal = g['p_goal'] + dp
        R_goal = dR @ g['R_goal']
        # F = the goal frame expressed in the anchor camera frame ("C0"); R_g = R_cv0 @ F.
        # Perturbing the goal rotation by dR (world) maps to R_g' = dR @ R_g exactly, i.e.
        # F' = R_cv0.T @ dR @ R_cv0 @ F. The policy only ever needs F (R_cv0 cancels in the
        # frame-change rotation M = F_hat.T @ F), so C0 is the natural currency.
        F = R_cv0.T @ dR @ goal_frame_rotation(R_cv0, g['goal_rot_cam'])
        R_frame = R_cv0 if self.action_frame == "cam0" else (R_cv0 @ F)
        action = build_h_action_rows(
            g['p_ee'], g['R_ee'], g['q7'], p_goal, R_goal, g['q7_goal'],
            R_frame, self.action_param, self.action_frame, self.h_dt)
        # goal-coincidence target: the remaining transform at the anchor, in the output frame.
        remaining = remaining_to_goal(g['p_ee'][0], g['R_ee'][0], p_goal, R_goal, R_frame)
        # ... valid ONLY where the horizon actually reaches the goal (see H_GOAL_INTEGRAL_TOL_M).
        # Measured against the CLEAN GT goal `g['p_goal']`, NEVER the noised `p_goal` (§9.2):
        # this asks whether the LABELS satisfy the constraint, which is a property of the
        # trajectory, not of the scheduled-sampling draw. Against the noised goal a 15 mm
        # perturbation vs a 5 mm tolerance zeroes the mask on ~99.6% of samples and silently
        # deletes the loss.
        inreach = float(np.linalg.norm(g['p_ee'][-1] - g['p_goal']) <= H_GOAL_INTEGRAL_TOL_M)
        extras = dict(
            h_goal_frame=F.astype(np.float32),                       # (3,3) F (with noise)
            h_goal_remaining=remaining.astype(np.float32),            # (6,)
            h_goal_inreach=np.array([inreach], dtype=np.float32),     # (1,)
            h_ee0_rot=(R_cv0.T @ g['R_ee'][0]).astype(np.float32),   # (3,3) R_ee[0] in C0
        )
        return action.astype(np.float32), extras

    def _h_all_action_rows(self):
        """Action rows for EVERY train window — the normalizer must be fit over the exact
        tensors training sees (mirrors `_all_anchored_actions`). Pose-only sampler so the
        image arrays are never sliced.

        GOAL FRAMES ALSO FIT OVER **NOISED** ROWS (contract §9.5). The clean-only fit was wrong
        by an amount that grows as the rows contract: a `delta`+`goal` row near the goal is
        `p_goal - p_k`, a few mm, so the 15 mm scheduled-sampling perturbation DOMINATES it and
        the terminal rows leave the fitted limits entirely — measured max |normalized train
        action| = 2.48 in limits mode. `H_NORM_NOISE_DRAWS` fixed-seed draws per window are
        appended to the clean rows, so the fit covers what training actually consumes and stays
        deterministic (the seed is derived from the dataset seed and the window index; it does
        NOT touch the per-sample augmentation RNG). `cam0` frames are untouched: their rows do
        not depend on the goal at all."""
        keys = ['ee_pos_w', 'ee_quat_w', 'q7', 'cam_quat_cv',
                'ee_goal_pos_w', 'ee_goal_quat_w', 'q7_goal', 'goal_rot_cam']
        pose_sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=self.seq_len,
            pad_before=self.pad_before, pad_after=self.pad_after,
            keys=keys, episode_mask=self.train_mask)
        # Gated on `self.augment` as well as on the knobs: a run with augmentation disabled never
        # perturbs the goal frame, so widening its normalizer would be fitting noise nobody sees.
        noisy = (self.action_frame == "goal" and H_NORM_NOISE_DRAWS > 0 and self.augment
                 and (float(self.aug.get("goal_frame_noise_mm", 0.0) or 0.0) > 0.0
                      or float(self.aug.get("goal_frame_noise_deg", 0.0) or 0.0) > 0.0))
        rows = []
        for idx in range(len(pose_sampler)):
            s = pose_sampler.sample_sequence(idx)
            g = self._h_window_geometry(s, self.pad_before)
            rows.append(self._h_build_rows(g)[0])
            if noisy:
                rng = np.random.default_rng([self.seed, 424242, idx])
                for _ in range(H_NORM_NOISE_DRAWS):
                    rows.append(self._h_build_rows(g, *self._h_goal_perturbation(rng))[0])
        out = np.concatenate(rows, axis=0)
        cprint(f"[H] action normalizer fit over {len(pose_sampler)} windows x "
               f"{self.action_rows} rows"
               + (f" (+{H_NORM_NOISE_DRAWS} goal-noise draws each, §9.5)" if noisy else "")
               + f" -> {out.shape}", "green")
        return out

    def _h_all_tcpcam(self):
        """`tcpcam` over the whole buffer (for the normalizer), vectorized.

        MUST stay identical to the per-sample construction in `_h_sample_to_data` and to the
        normative contract §4 formula `[R_cv.T @ (p_ee - p_cam), rotvec(R_cv.T @ R_ee), q7]` —
        a normalizer fit on a different quantity than the model sees is a silent scale bug."""
        rb = self.replay_buffer
        R_cv = quat_wxyz_to_rotmat(np.asarray(rb['cam_quat_cv'][:], np.float64))       # (T,3,3)
        R_ee = quat_wxyz_to_rotmat(np.asarray(rb['ee_quat_w'][:], np.float64))
        dp = np.asarray(rb['ee_pos_w'][:], np.float64) - np.asarray(rb['cam_pos_w'][:], np.float64)
        pos = np.einsum('tji,tj->ti', R_cv, dp)                     # R_cv.T @ dp per frame
        rel = np.einsum('tji,tjk->tik', R_cv, R_ee)                 # R_cv.T @ R_ee
        rv = np.stack([rotmat_to_rotvec(m) for m in rel])
        q7 = np.asarray(rb['q7'][:], np.float64).reshape(-1, 1)
        return np.concatenate([pos, rv, q7], axis=1).astype(np.float32)   # (T,7)

    def get_h_normalizer(self, mode='limits', **kwargs):
        """H normalizer (contract §4 obs dict + §3 action). Every field the model consumes is
        fit here; a missing field would silently pass through UNNORMALIZED (LinearNormalizer
        returns the input for unknown keys) — so assert the set afterwards."""
        rb = self.replay_buffer
        # `dt` is the TRUE image gap in seconds. Fit over the full contract support {0, .1, .2}
        # rather than the observed values, so that turning the timing aug on/off cannot change
        # the normalization (and a nominal-only val split is not a degenerate fit).
        dt_support = (np.arange(self.h_max_gap + 1, dtype=np.float32)
                      / self.control_hz).reshape(-1, 1)
        data = {
            'action': self._h_all_action_rows(),
            # PRIMARY place-head target (§1.2b): the panel's target pose, camera-frame-ABSOLUTE.
            'panel_goal': np.asarray(rb['panel_goal_cam'][:], np.float32),
            # AUX landmark target only, and a DIFFERENT CONVENTION (TCP-anchored delta). Fit
            # separately — sharing a normalizer between an absolute pose and a delta would be
            # a silent scale error.
            'rail_a': np.asarray(rb['rail_a_cam'][:], np.float32),
            'task': np.asarray(rb['task'][:], np.float32),
            'agent_pos': np.asarray(rb['agent_pos'][:], np.float32),
            'grasp_off': np.asarray(rb['grasp_offset'][:], np.float32),
            'ego': np.asarray(rb['tcp_egomotion'][:], np.float32),
            'twist': np.asarray(rb['tcp_twist'][:], np.float32),
            'tcpcam': self._h_all_tcpcam(),
            'dt': dt_support,
            'twist_hist': np.asarray(rb['twist_hist'][:], np.float32).reshape(-1, 6),
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['head_cam'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['depth_cam'] = SingleFieldLinearNormalizer.create_identity()
        missing = [k for k in H_OBS_KEYS + ('action', 'panel_goal', 'rail_a')
                   if k not in normalizer.params_dict]
        assert not missing, f"H normalizer missing fields {missing} (would pass through raw)"
        return normalizer

    def get_normalizer(self, mode='limits', **kwargs):
        if self.h_mode:
            return self.get_h_normalizer(mode=mode, **kwargs)
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                'action': self._all_anchored_actions(),
                'goal_cam': np.concatenate(
                    [self.replay_buffer['goal_pos_cam'], self.replay_buffer['goal_rot_cam']],
                    axis=1),
                'prev_action': self.replay_buffer['prev_action'],
                'task': self.replay_buffer['task'],
                # G1: latch = [goal(6), conf(1)]; conf column fit over [0,1] (dropout emits 0s
                # at train — a constant-1 fit would make the limits normalizer degenerate)
                'goal_prior': np.concatenate(
                    [self.replay_buffer['goal_pos_cam'], self.replay_buffer['goal_rot_cam'],
                     np.concatenate([np.zeros((1, 1), dtype=np.float32),
                                     np.ones((len(self.replay_buffer['goal_pos_cam']) - 1, 1),
                                             dtype=np.float32)])], axis=1),
                **({'agent_pos': self.replay_buffer['agent_pos']}
                   if self.has_agent_pos else {}),
            },
            last_n_dims=1, mode=mode, **kwargs)
        normalizer['head_cam'] = SingleFieldLinearNormalizer.create_identity()
        if self.use_depth:
            normalizer['depth_cam'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    # ------------------------------------------------------------------ augmentation

    def _episode_of(self, idx: int) -> int:
        buffer_start_idx = int(self.sampler.indices[idx][0])
        return int(np.searchsorted(self._episode_ends, buffer_start_idx, side='right'))

    def _augment_depth_mm(self, depth_mm, ep_rng, sample_rng):
        a = self.aug
        if ep_rng.random() < a["depth_episode_dropout"]:
            return np.zeros_like(depth_mm)
        d = depth_mm.astype(np.float32)
        valid = d > 0
        meters = d / 1000.0
        meters = meters + sample_rng.normal(size=d.shape).astype(np.float32) \
            * (a["depth_noise_sigma0"] * meters * meters)
        drop = sample_rng.random(d.shape) < a["depth_px_dropout"]
        fly = sample_rng.random(d.shape) < a["depth_flying_px"]
        meters[fly] = sample_rng.uniform(DEPTH_MM_MAX * 0.03 / 1000.0, DEPTH_MM_MAX / 1000.0,
                                         size=int(fly.sum()))
        out = np.clip(meters * 1000.0, 0.0, 65535.0)
        out[~valid | drop] = 0.0
        return out.astype(np.float32)  # keep float mm for the encode step

    def _augment_photometric(self, head_cam, rng):
        """v7 Rewire: label-safe RGB appearance aug (train-only). Photometric only (no geometry
        change => no label co-rotation). Per-sample, SAME factor/box across the To frames (a
        single physical scene). Order: color jitter (brightness/contrast/saturation/hue) ->
        gaussian blur -> gaussian noise -> random erasing. head_cam is (T,3,S,S) in [0,1].
        Targets lighting/reflectance/optical robustness — the specular RGB-variation the model
        must generalize over."""
        import torchvision.transforms.functional as TF
        t = torch.from_numpy(head_cam)                                   # (T,3,S,S) in [0,1]
        # ---- color jitter (torchvision), random factor once, applied to all frames ----
        if self.photo_brightness > 0.0:
            t = TF.adjust_brightness(t, float(rng.uniform(1 - self.photo_brightness, 1 + self.photo_brightness)))
        if self.photo_contrast > 0.0:
            t = TF.adjust_contrast(t, float(rng.uniform(1 - self.photo_contrast, 1 + self.photo_contrast)))
        if self.photo_saturation > 0.0:
            t = TF.adjust_saturation(t, float(rng.uniform(1 - self.photo_saturation, 1 + self.photo_saturation)))
        if self.photo_hue > 0.0:
            h = float(rng.uniform(-self.photo_hue, self.photo_hue))
            t = TF.adjust_hue(t.clamp(0.0, 1.0), max(-0.5, min(0.5, h)))
        # ---- mild gaussian blur (optical defocus / motion) ----
        if self.photo_blur_p > 0.0 and rng.random() < self.photo_blur_p:
            sigma = float(rng.uniform(0.4, 1.2))
            t = TF.gaussian_blur(t, kernel_size=5, sigma=sigma)
        hc = t.numpy()
        # ---- gaussian pixel noise (sensor) ----
        if self.photo_noise > 0.0:
            hc = hc + rng.normal(0.0, self.photo_noise, hc.shape).astype(np.float32)
        hc = np.clip(hc, 0.0, 1.0)
        # ---- random erasing (occlusion / glare patch): up to 2 boxes, filled with random gray ----
        if self.photo_erase_p > 0.0:
            hc = hc.copy()
            S = hc.shape[-1]
            for _ in range(2):
                if rng.random() >= self.photo_erase_p:
                    continue
                eh = int(S * rng.uniform(0.05, 0.20)); ew = int(S * rng.uniform(0.05, 0.20))
                y0 = int(rng.integers(0, max(1, S - eh + 1))); x0 = int(rng.integers(0, max(1, S - ew + 1)))
                hc[..., y0:y0 + eh, x0:x0 + ew] = float(rng.uniform(0.0, 1.0))
        return hc.astype(np.float32)

    def _augment_so2(self, head_cam, depth_mm, action, goal_cam, prev_action, rng):
        """v7 Rewire: label-consistent SO(2) optical-axis roll. Rotate the wrist image
        (+depth if present) in-plane by phi about center, and CO-ROTATE the camera-frame
        6-D labels [pos|rotvec] by R_z(SO2_SIGN*phi). Multiplies the manifold along the
        task's real roll symmetry -> anti-overfit + RGB-roll robustness. SO2_SIGN=-1 was
        pinned empirically: torchvision TF.rotate(+phi_deg) == camera R_z(-phi) for a
        centered pinhole (scratchpad/so2_sign_test.py, sub-pixel match)."""
        import torchvision.transforms.functional as TF
        phi_deg = float(rng.uniform(-self.rot_aug_deg, self.rot_aug_deg))
        if abs(phi_deg) < 1e-3:
            return head_cam, depth_mm, action, goal_cam, prev_action
        # images: rotate the last two dims (H,W); leading (T,C) treated as batch. Bilinear
        # for RGB; nearest for depth (avoid interpolating across the invalid/glass boundary).
        hc = TF.rotate(torch.from_numpy(head_cam), angle=phi_deg,
                       interpolation=TF.InterpolationMode.BILINEAR).numpy()
        dm = depth_mm
        if depth_mm is not None:
            dm = TF.rotate(torch.from_numpy(depth_mm), angle=phi_deg,
                           interpolation=TF.InterpolationMode.NEAREST).numpy()
        # labels: v' = R_z(-phi) @ v  for BOTH pos (rows :3) and rotvec (rows 3:)
        a = np.radians(-phi_deg)     # SO2_SIGN = -1
        ca, sa = np.cos(a), np.sin(a)
        Rz_T = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)  # R_z(a).T

        def _co(arr):                # arr: (...,6) -> co-rotate pos & rotvec by R_z(a) (== arr @ R_z(a).T)
            out = arr.copy()
            out[..., :3] = arr[..., :3] @ Rz_T
            out[..., 3:] = arr[..., 3:] @ Rz_T
            return out
        return hc, dm, _co(action), _co(goal_cam), _co(prev_action)

    def _augment_prev_action(self, pa, sample_rng):
        pa = pa.copy()
        pa[:, :3] += sample_rng.normal(scale=self.aug["prev_action_noise_mm"] / 1000.0,
                                       size=pa[:, :3].shape).astype(np.float32)
        pa[:, 3:] += sample_rng.normal(scale=np.radians(self.aug["prev_action_noise_deg"]),
                                       size=pa[:, 3:].shape).astype(np.float32)
        return pa

    @staticmethod
    def _lag_one_tick(arr):
        lagged = np.empty_like(arr)
        lagged[1:] = arr[:-1]
        lagged[0] = arr[0]
        return lagged

    def _encode_depth(self, depth_mm_float):
        """(T,1,S,S) float mm -> (T,C,S,S) float32.
        xyz:   C=4 ordered POINT-MAP — each pixel's metric camera-frame (X,Y,Z)/XYZ_SCALE_M
               + validity. Continuous (per-mm gradient everywhere; the band encoding had
               ~50mm dead zones), crop-safe, and hands the model metric ranging directly.
        bands: C=(len(edges)-1) near-focused soft H3DP band masks (see docs/v3_depth_design.md);
               invalid pixels -> all channels 0. tiled: normalized depth repeated x3.
        The far edge acts as a CLIP: depth beyond edges[-1] still lights the last band (so
        background collapses into ch -1 rather than vanishing)."""
        m = depth_mm_float / 1000.0                      # meters, 0 = invalid
        valid = depth_mm_float > 0
        if self.depth_input == "xyz":
            vf = valid.astype(np.float32)
            z = np.clip(m, 0.0, XYZ_SCALE_M) * vf        # (T,1,S,S) meters, clipped
            x = self._xmap[None, None] * z               # unproject: X=(u-cx)/fx * Z
            y = self._ymap[None, None] * z
            # This camera's HFOV is 93.4deg (>90), so xmap reaches ~+-1.06 at the frame
            # edge and x/XYZ_SCALE_M can exceed 1.0 for far right-edge pixels. Clip the
            # normalized X,Y to [-1,1] so the map stays in a sane range (the encoder's u8
            # /255 heuristic is separately gated off for depth keys, but keep the domain tight).
            return np.clip(np.concatenate(
                [x / XYZ_SCALE_M, y / XYZ_SCALE_M, z / XYZ_SCALE_M, vf],
                axis=1), -1.0, 1.0).astype(np.float32)   # (T,4,S,S)
        if self.depth_input == "bands":
            s = self.depth_band_soft
            edges = self.depth_band_edges
            chans = []
            n = len(edges) - 1
            for i in range(n):
                lo, hi = edges[i], edges[i + 1]
                up = np.clip((m - (lo - s)) / s, 0.0, 1.0)
                if i == n - 1:
                    down = np.ones_like(m)               # last band: no upper cutoff (clip tail in)
                else:
                    down = np.clip(((hi + s) - m) / s, 0.0, 1.0)
                chans.append(np.minimum(up, down) * valid)
            return np.concatenate(chans, axis=1).astype(np.float32)   # (T,C,S,S)
        norm = np.clip(m / (DEPTH_MM_MAX / 1000.0), 0.0, 1.0) * valid
        return np.repeat(norm, 3, axis=1).astype(np.float32)

    # ------------------------------------------------------------------ sample assembly

    def _check_depth_channels(self, depth_cam):
        """Fail loud if the emitted depth channel count != shape_meta expectation (a
        depth_input/shape_meta mismatch would otherwise surface as a cryptic conv error)."""
        if depth_cam.shape[1] != self.depth_channels:
            raise ValueError(
                f"depth_cam emitted {depth_cam.shape[1]} channels but depth_channels="
                f"{self.depth_channels} (depth_input={self.depth_input}). Align shape_meta "
                f"depth_cam / robotwin_task.depth_channels / depth_input.")

    # ------------------------------------------------------------------ H sample assembly

    def _h_frame_indices(self, anchor, sample_rng, pad_start=0):
        """Timing augmentation (contract §5) -> (image indices, proprio indices, dt seconds).

        `obs_gap_frames ~ obs_gap_probs` over {0,1,2}: spacing of the obs IMAGES. 0 reproduces
        the observed duplicate-camera-frame failure, 1 = nominal 100 ms, 2 = 200 ms.
        `image_lag_frames ~ image_lag_probs` over {0,1}: both images are taken that many frames
        BEFORE the action anchor while PROPRIO stays at the anchor — the measured 100-150 ms
        observation staleness (joint states arrive fast, images do not).
        `dt` reports the TRUE image gap so the jitter is in-distribution, never a silent lie.
        Sub-100 ms jitter is NOT representable (images exist only at 10 Hz) — data-request item,
        do not fake it. Eval/export (augment off) => gap 1, lag 0 = nominal.

        `pad_start` = the sampler's `sample_start_idx`: window slots BELOW it are PADDING that
        `SequenceSampler` fills by repeating the episode's first frame. dt is therefore computed
        from the EFFECTIVE indices `max(i, pad_start)` (contract §9.4): at an episode-start
        anchor the two "obs images" are byte-identical duplicates, and reporting the nominal
        100 ms there would be exactly the silent lie the key exists to prevent — the more so
        because gap = 0 is a MODELLED failure mode (the observed duplicate-camera-frame bug), so
        a lie here trains the model to expect parallax that a real stalled camera will not
        deliver. Clipping also fixes the intermediate case honestly: gap 2 clipped to one real
        frame reports 100 ms, not 200.
        """
        To = self.n_obs_steps
        if self.augment:
            gap = int(sample_rng.choice(self.h_max_gap + 1, p=self.obs_gap_probs))
            lag = int(sample_rng.choice(self.h_max_lag + 1, p=self.image_lag_probs))
        else:
            gap, lag = 1, 0
        cur = anchor - lag
        img_idx = [cur - gap * (To - 1 - i) for i in range(To)]
        # Proprio is NOT lagged and keeps its native 1-frame spacing: `ego`/`twist`/`twist_hist`
        # are stored per control frame with those exact semantics (contract §1.2), so re-spacing
        # them would silently redefine the quantity.
        pro_idx = [anchor - (To - 1 - i) for i in range(To)]
        assert min(img_idx + pro_idx) >= 0, (
            f"H window underflow: img={img_idx} pro={pro_idx} anchor={anchor} "
            f"(pad_before={self.pad_before})")
        eff = [max(int(i), int(pad_start)) for i in img_idx]
        dt_true = float(eff[-1] - eff[0]) / max(1, To - 1) / self.control_hz
        return img_idx, pro_idx, dt_true

    def _h_sample_to_data(self, sample, ep_idx=None, sample_idx=0, pad_start=0):
        """Build one H training sample: contract §4 obs dict + §3 action rows (+ supervision)."""
        anchor = self.pad_before
        # EPOCH IS PART OF THE SEED (contract §9.4). Without it every sample's timing draw, goal
        # noise and proprio DR is a fixed constant for the whole run — see `set_epoch`. Within an
        # epoch the draw stays a pure function of (epoch, episode, sample), so it is reproducible
        # and independent of worker count / shuffling.
        ep = self.epoch
        # ep_rng stays EPISODE-scoped (it drives `depth_episode_dropout`, whose semantics are
        # "this episode's depth is unusable") but is no longer frozen: a fixed 10% of episodes
        # being permanently depth-blind is memorisable, a resampled 10% is domain randomisation.
        ep_rng = (np.random.default_rng([self.seed, 1000003, ep, ep_idx])
                  if ep_idx is not None else np.random.default_rng([self.seed, 1000003, ep]))
        sample_rng = np.random.default_rng([self.seed, 7777777, ep, ep_idx or 0, sample_idx])

        img_idx, pro_idx, dt_true = self._h_frame_indices(anchor, sample_rng, pad_start=pad_start)
        ii = np.asarray(img_idx)
        pi = np.asarray(pro_idx)

        # ---------------- images (RGB + xyz point-map), taken at the IMAGE indices ----------
        head_cam = sample['head_camera'][ii].astype(np.float32) / 255.0
        depth_mm = sample['depth'][ii].astype(np.float32)

        # ---------------- typed proprio, taken at the PROPRIO indices ----------------------
        agent_pos = sample['agent_pos'][pi].astype(np.float32)          # (To,7) joints [rad]
        task = sample['task'][pi].astype(np.float32)                    # (To,1)
        grasp_off = sample['grasp_offset'][pi].astype(np.float32)       # (To,7) [xyz,quat wxyz]
        ego = sample['tcp_egomotion'][pi].astype(np.float32)            # (To,6)
        twist = sample['tcp_twist'][pi].astype(np.float32)              # (To,6)
        twist_hist = sample['twist_hist'][pi].astype(np.float32)        # (To,5,6)
        dt = np.full((self.n_obs_steps, 1), dt_true, dtype=np.float32)  # (To,1) TRUE image gap
        # ---------------------------------------------------------------------------------
        # `dt` and `ego` describe DIFFERENT INTERVALS, BY DESIGN. Do not "reconcile" them.
        #   dt  = the TRUE gap between the two obs IMAGES (0 / 100 / 200 ms under the §5 timing
        #         aug). It exists so the model knows how much visual parallax to expect.
        #   ego = the stored `tcp_egomotion`, which is by its §1.2 definition the ee_link
        #         displacement over ONE control interval (always 100 ms), in the camera frame at
        #         the previous frame. It is a measured sensor-path quantity.
        # Re-deriving `ego` over the augmented image gap would silently redefine a key the
        # converter owns and the deploy node computes from /joint_states — the two clocks really
        # are different (joint states arrive fast, images do not), and that difference is exactly
        # what the model must learn to tolerate. Contract §4 is normative on the source.
        # ---------------------------------------------------------------------------------
        # `tcpcam` (contract §4, NORMATIVE):
        #     [ R_cv.T @ (p_ee - p_cam) , rotvec(R_cv.T @ R_ee) , q7 ]
        # Written out literally so it diffs against the contract text line-for-line. It is
        # CONFIGURATION, not motion: where the tool tip sits in camera coordinates, so that the
        # servo error (goal_cam (-) tcp_cam) is expressible at all; it varies with q7 because the
        # camera rides on link_6_extension. (This is algebraically identical to
        # anchored_delta(cam_pose, ee_pose, R_cv) — the anchor rotation IS R_cv, so its
        # R_cv.T (R_ee R_cv.T) R_cv collapses to R_cv.T R_ee — and the test asserts the identity;
        # the explicit form is preferred here to keep one fewer matmul between us and the spec.)
        tcpcam = np.empty((self.n_obs_steps, 7), dtype=np.float32)
        for j, k in enumerate(pi):
            R_cv = quat_wxyz_to_rotmat(np.asarray(sample['cam_quat_cv'][k], np.float64))
            R_ee = quat_wxyz_to_rotmat(np.asarray(sample['ee_quat_w'][k], np.float64))
            dp = np.asarray(sample['ee_pos_w'][k], np.float64) \
                - np.asarray(sample['cam_pos_w'][k], np.float64)
            tcpcam[j, :3] = R_cv.T @ dp
            tcpcam[j, 3:6] = rotmat_to_rotvec(R_cv.T @ R_ee)
            tcpcam[j, 6] = float(np.asarray(sample['q7'][k]).reshape(-1)[0])

        # ---------------- action rows + goal bookkeeping ------------------------------------
        g = self._h_window_geometry(sample, anchor)
        dR, dp = ((None, None) if not self.augment
                  else self._h_goal_perturbation(sample_rng))
        action, extras = self._h_build_rows(g, dR, dp)

        # Constant maps that let the POLICY compose the ee_link goal (and the goal FRAME) from its
        # own predicted `panel_goal_cam` (+) `grasp_offset`, in C0 — see `panel_to_ee_goal_maps`.
        # Training-only: never an obs input, never in the ONNX graph.
        #
        # The place head's output lives in the camera frame of the MOST RECENT USED IMAGE (§9.3,
        # index `i_img`), which under `image_lag_frames > 0` is NOT the anchor. The maps therefore
        # carry `A = R_cv0.T @ R_cimg` and re-anchor the position from that camera; with lag 0
        # (every eval/export sample) A is the identity and this reduces exactly to the anchor-frame
        # form. Getting this wrong would rotate the composed goal frame by the camera's egomotion
        # over one control step — the same error class §9.3 removes from the labels, re-introduced
        # through the back door.
        i_img = int(ii[-1])
        R_cimg = quat_wxyz_to_rotmat(np.asarray(sample['cam_quat_cv'][i_img], np.float64))
        gc = panel_to_ee_goal_maps(
            g['R_cv0'].T @ quat_wxyz_to_rotmat(np.asarray(sample['tcp_quat_w'][anchor], np.float64)),
            g['R_cv0'].T @ (np.asarray(sample['cam_pos_w'][i_img], np.float64) - g['p_ee'][0]),
            sample['grasp_offset'][anchor], g['q7_goal'],
            R_c0_from_cimg=g['R_cv0'].T @ R_cimg,
            invert_offset=getattr(self, 'grasp_offset_invert', GRASP_OFFSET_INVERT))

        # ---------------- augmentation ------------------------------------------------------
        if self.augment and ep_idx is not None:
            depth_mm = self._augment_depth_mm(depth_mm, ep_rng, sample_rng)
            # Per-key proprio DR (contract §4.3). Noise only — the per-key BLOCK MASK lives in
            # the policy because it substitutes a LEARNED mask token (zero is an in-distribution
            # value the net can detect and ignore; v7 lesson).
            nj = float(self.aug.get("proprio_noise_joint_deg", 0.0) or 0.0)
            if nj > 0.0:
                agent_pos = agent_pos + sample_rng.normal(
                    scale=np.radians(nj), size=agent_pos.shape).astype(np.float32)
                tcpcam[:, 6] += sample_rng.normal(
                    scale=np.radians(nj), size=self.n_obs_steps).astype(np.float32)
            nt = float(self.aug.get("proprio_noise_twist_frac", 0.0) or 0.0)
            if nt > 0.0:                                  # multiplicative: scale-free on v and w
                twist = twist * (1.0 + sample_rng.normal(
                    scale=nt, size=twist.shape).astype(np.float32))
                twist_hist = twist_hist * (1.0 + sample_rng.normal(
                    scale=nt, size=twist_hist.shape).astype(np.float32))
            ne = float(self.aug.get("proprio_noise_ego_mm", 0.0) or 0.0)
            if ne > 0.0:
                ego = ego.copy()
                ego[:, :3] += sample_rng.normal(
                    scale=ne / 1000.0, size=ego[:, :3].shape).astype(np.float32)
            if not self.gpu_offload and any(
                    v > 0.0 for v in (self.photo_brightness, self.photo_contrast,
                                      self.photo_saturation, self.photo_hue, self.photo_blur_p,
                                      self.photo_noise, self.photo_erase_p)):
                head_cam = self._augment_photometric(head_cam, sample_rng)
            # SO(2) roll is NOT applied in H: co-rotating the xyz point-map (per-pixel 3-D) and
            # every camera-frame label/goal-frame tensor here is not implemented, and a partial
            # co-rotation would silently de-register the labels. Keep rot_aug_deg=0 for H.
            assert self.rot_aug_deg == 0.0, (
                "rot_aug_deg>0 is unsupported in H mode (xyz point-map + goal-frame tensors "
                "would need co-rotation); set augmentation.rot_aug_deg=0")

        depth_cam = self._encode_depth(depth_mm)
        self._check_depth_channels(depth_cam)

        obs = {
            'head_cam': head_cam, 'depth_cam': depth_cam,
            'agent_pos': agent_pos, 'task': task, 'grasp_off': grasp_off,
            'ego': ego, 'twist': twist, 'tcpcam': tcpcam, 'dt': dt,
            'twist_hist': twist_hist,
        }
        assert set(obs) == set(H_OBS_KEYS), f"H obs keys drifted: {sorted(obs)}"
        # ---------------- supervision ------------------------------------------------------
        # PERCEPTION LABELS FOLLOW THE PIXELS (contract §9.3): every target of a head that reads
        # ONLY image features is indexed at the IMAGE frames `ii`, never at the anchor/proprio
        # indices. Under the live §5 augmentation (lag {0,1} @ [0.6,0.4]) ~40% of train samples
        # otherwise supervise the heatmap/place/rail heads with labels one control frame AHEAD of
        # the pixels they see — irreducible, camera-egomotion-scale label noise injected straight
        # into the head that IS deploy's goal source. Those heads have no proprio path, so they
        # cannot even in principle compensate for it.
        #
        # ACTION labels stay anchored AT THE ANCHOR (`anchor`), unchanged and bit-identical across
        # every gap/lag draw — the action rows describe motion from the obs time forward, and
        # re-indexing them would redefine the chunk. `phase_id`/`done` are anchor-time facts about
        # the CONTROL state (which phase am I in, am I finished), not about the image, so they
        # stay at the anchor too. Tests pin both invariants. (`i_img` == ii[-1], newest image.)
        out = {
            'obs': obs,
            'action': action,                                                    # (rows,7)
            # PRIMARY place-head target (§1.2b): the PANEL's target pose, camera-frame-ABSOLUTE.
            # It is grasp-offset-free (a scene property) AND it is identifiable, which `rail_a`
            # is not — the along-rail DoF is free by 0.19 m relative to the rail.
            # FRAME (§9.3): the camera that captured the MOST RECENT USED image, i.e. `ii[-1]`.
            'panel_goal': sample['panel_goal_cam'][i_img].astype(np.float32),    # (6,)
            # AUX landmark grounding ONLY, and note the DIFFERENT CONVENTION: a TCP-anchored
            # delta, not a camera-absolute pose. Inter-convertible via `tcpcam`, never equal.
            'rail_a': sample['rail_a_cam'][i_img].astype(np.float32),            # (6,)
            'arm_kpts_uv': sample['arm_kpts_uv'][ii].astype(np.float32),         # (To,N,3)
            'arm_kpts_cam': sample['arm_kpts_cam'][ii].astype(np.float32),       # (To,N,3)
            'phase_id': sample['phase_id'][anchor].astype(np.int64).reshape(1),  # (1,)
            'done': sample['done'][anchor].astype(np.float32).reshape(1),        # (1,)
            'h_gc_tvec': gc['tvec'], 'h_gc_toff': gc['toff'], 'h_gc_A': gc['A'],
            'h_gc_Ree': gc['R_ee'], 'h_gc_Rframe': gc['R_frame'],
        }
        out.update(extras)
        return out

    def _sample_to_data(self, sample, ep_idx=None, sample_idx=0):
        task = sample['task'][:, ].astype(np.float32)
        prev_action = sample['prev_action'][:, ].astype(np.float32)
        # G1: ManiFlow-faithful joint proprio (v9 zarr) + the goal-prior LATCH channel.
        # goal_prior = per-frame [goal_pos_cam, goal_rot_cam, conf]. Because the rail is
        # STATIC and FK is exact, the deploy-time FK-propagated latch equals the current
        # true label + estimate noise — so scheduled sampling is just label+noise+dropout
        # (Swift/Nature-2023 architecture, no pose math needed).
        agent_pos = (sample['agent_pos'][:, ].astype(np.float32)
                     if 'agent_pos' in sample else None)
        goal_prior = np.concatenate(
            [sample['goal_pos_cam'], sample['goal_rot_cam'],
             np.ones((len(sample['goal_pos_cam']), 1))], axis=1).astype(np.float32)
        # G0 STRUCTURAL control: drop_p>=1 disables the latch in ALL splits (train AND val),
        # so checkpoint selection never scores the model under an input it wasn't trained on
        # (review 2026-07-26: the augment-gated drop alone contaminated G0's val/selection).
        if float(self.aug.get("goal_prior_drop_p", 0.0) or 0.0) >= 1.0:
            goal_prior = np.zeros_like(goal_prior)
        depth_mm = (sample['depth'][:, ].astype(np.float32)   # (T,1,S,S) mm (float for aug)
                    if self.use_depth else None)

        anchor = min(self.pad_before, len(sample['tcp_pos_w']) - 1)
        action = build_anchored_chunk(
            sample['tcp_pos_w'], sample['tcp_quat_w'], sample['cam_quat_cv'], anchor)
        goal_cam = np.concatenate(
            [sample['goal_pos_cam'][anchor], sample['goal_rot_cam'][anchor]]).astype(np.float32)
        head_cam = sample['head_camera'][:, ].astype(np.float32) / 255.0

        if self.augment and ep_idx is not None:
            ep_rng = np.random.default_rng([self.seed, 1000003, ep_idx])
            sample_rng = np.random.default_rng([self.seed, 7777777, ep_idx, sample_idx])
            if depth_mm is not None:
                depth_mm = self._augment_depth_mm(depth_mm, ep_rng, sample_rng)
            prev_action = self._augment_prev_action(prev_action, sample_rng)
            # G1 proprio noise + mask (anti-copycat; deployment-error scale)
            if agent_pos is not None and self.aug.get("agent_pos_noise_deg", 0.0) > 0.0:
                agent_pos = agent_pos + sample_rng.normal(
                    scale=np.radians(self.aug["agent_pos_noise_deg"]),
                    size=agent_pos.shape).astype(np.float32)
            if agent_pos is not None and sample_rng.random() < self.aug.get("agent_pos_mask_p", 0.0):
                agent_pos = np.zeros_like(agent_pos)
            # G1 goal-prior latch: scheduled-sampling noise + dropout (independent guards)
            if self.aug.get("goal_prior_noise_mm", 0.0) > 0.0:
                goal_prior[:, :3] += sample_rng.normal(
                    scale=self.aug["goal_prior_noise_mm"] / 1000.0,
                    size=goal_prior[:, :3].shape).astype(np.float32)
            if self.aug.get("goal_prior_noise_deg", 0.0) > 0.0:
                goal_prior[:, 3:6] += sample_rng.normal(
                    scale=np.radians(self.aug["goal_prior_noise_deg"]),
                    size=goal_prior[:, 3:6].shape).astype(np.float32)
            if sample_rng.random() < self.aug.get("goal_prior_drop_p", 0.0):
                goal_prior = np.zeros_like(goal_prior)
            lat_rng = np.random.default_rng([self.seed, 424243, ep_idx])
            p = self.aug["latency_shift_prob"]
            # C2: depth is the SAME physical camera as head_cam — never lag it independently
            # of RGB (the v3 independent lag de-synced the streams on ~50% of samples and
            # taught the model depth was temporally unreliable). Only prev_action (a
            # different sensor path with real latency) keeps the lag augmentation.
            if lat_rng.random() < p:
                prev_action = self._lag_one_tick(prev_action)
            # v7 Rewire: SO(2) optical-axis roll co-rotates image + camera-frame labels.
            # When gpu_offload is set, the workspace does SO(2)+photometric on the CUDA batch
            # instead (this per-sample CPU path is skipped to avoid double-augmenting).
            if not self.gpu_offload:
                if self.rot_aug_deg > 0.0:
                    head_cam, depth_mm, action, goal_cam, prev_action = self._augment_so2(
                        head_cam, depth_mm, action, goal_cam, prev_action, sample_rng)
                # v7 Rewire: RGB appearance aug (train-only; keeps eval obs clean)
                if any(v > 0.0 for v in (self.photo_brightness, self.photo_contrast,
                                         self.photo_saturation, self.photo_hue, self.photo_blur_p,
                                         self.photo_noise, self.photo_erase_p)):
                    head_cam = self._augment_photometric(head_cam, sample_rng)

        obs = {
            'head_cam': head_cam,
            'goal_prior': goal_prior,
            'prev_action': prev_action,
            'task': task,
        }
        if agent_pos is not None:
            obs['agent_pos'] = agent_pos
        if depth_mm is not None:
            depth_cam = self._encode_depth(depth_mm)
            self._check_depth_channels(depth_cam)
            obs['depth_cam'] = depth_cam

        out = {
            'obs': obs,
            'action': action,
            'goal_cam': goal_cam,       # supervision only; NOT in obs / ONNX
        }
        # F-series: rail/tube keypoints (v8 zarr) for the soft-argmax head. Full window; the
        # policy uses the most-recent obs frame. NOT in obs/ONNX (targets + goal-token source).
        if 'arm_kpts_uv' in sample:
            out['arm_kpts_uv'] = sample['arm_kpts_uv'][:].astype(np.float32)   # (T,N,3) [u/W,v/H,vis]
            out['arm_kpts_cam'] = sample['arm_kpts_cam'][:].astype(np.float32)  # (T,N,3) [Xc,Yc,Zc]
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        if self.h_mode:
            # `sample_start_idx`: window slots below it are the sampler's front PADDING (frame 0
            # repeated), which the dt-honesty rule needs to see (contract §9.4 / `_h_frame_indices`).
            data = self._h_sample_to_data(
                sample, ep_idx=self._episode_of(idx), sample_idx=idx,
                pad_start=int(self.sampler.indices[idx][2]))
        else:
            data = self._sample_to_data(sample, ep_idx=self._episode_of(idx), sample_idx=idx)
        return dict_apply(data, torch.from_numpy)
