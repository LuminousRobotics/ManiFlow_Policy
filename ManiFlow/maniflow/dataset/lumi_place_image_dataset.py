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
    "latency_shift_prob": 0.5,       # per-episode prob a stream lags 1 control tick
    "rot_aug_deg": 0.0,              # v7 Rewire: SO(2) optical-axis roll aug half-range [deg]; 0 => off
    # v7 Rewire: label-safe RGB appearance aug (train-only; all 0 => off). Photometric only —
    # no geometry change, so no label co-rotation needed. Per-sample, same across the To frames.
    "photo_brightness": 0.0,         # brightness jitter fraction (torchvision adjust_brightness)
    "photo_contrast": 0.0,           # contrast jitter fraction
    "photo_saturation": 0.0,         # saturation jitter fraction
    "photo_hue": 0.0,                # hue jitter (+/-, in [0,0.5])
    "photo_blur_p": 0.0,             # prob of a mild gaussian blur
    "photo_noise": 0.0,              # gaussian pixel noise std (image in [0,1])
    "photo_erase_p": 0.0,            # prob of a random-erasing box (up to 2 boxes)
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
        cprint(f'Loading LumiPlaceImageDataset (v6 anchored, depth='
               f'{depth_input if self.use_depth else "OFF"}, '
               f'{self.depth_channels}ch) from {zarr_path}', 'green')

        buffer_keys = ['head_camera', 'tcp_pos_w', 'tcp_quat_w', 'cam_pos_w',
                       'cam_quat_cv', 'prev_action', 'gravity_cam', 'goal_pos_cam',
                       'goal_rot_cam', 'task']
        if self.use_depth:
            buffer_keys.insert(1, 'depth')
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

        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed)
        train_mask = ~val_mask
        if max_train_episodes is None:
            max_train_episodes = self.replay_buffer.n_episodes - np.sum(val_mask)
        cprint(f'Maximum training episodes: {max_train_episodes}', 'yellow')
        cprint(f'Validation ratio: {val_ratio}', 'yellow')
        train_mask = downsample_mask(mask=train_mask, max_n=max_train_episodes, seed=seed)

        self.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=horizon,
            pad_before=pad_before, pad_after=pad_after, episode_mask=train_mask)
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.seed = seed

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
        self._episode_ends = self.replay_buffer.episode_ends[:]

        self.zarr_path = zarr_path
        self.train_episodes_num = np.sum(train_mask)
        self.val_episodes_num = np.sum(val_mask)

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer, sequence_length=self.horizon,
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

    def get_normalizer(self, mode='limits', **kwargs):
        normalizer = LinearNormalizer()
        normalizer.fit(
            data={
                'action': self._all_anchored_actions(),
                'goal_cam': np.concatenate(
                    [self.replay_buffer['goal_pos_cam'], self.replay_buffer['goal_rot_cam']],
                    axis=1),
                'prev_action': self.replay_buffer['prev_action'],
                'task': self.replay_buffer['task'],
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

    def _sample_to_data(self, sample, ep_idx=None, sample_idx=0):
        task = sample['task'][:, ].astype(np.float32)
        prev_action = sample['prev_action'][:, ].astype(np.float32)
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
            lat_rng = np.random.default_rng([self.seed, 424243, ep_idx])
            p = self.aug["latency_shift_prob"]
            # C2: depth is the SAME physical camera as head_cam — never lag it independently
            # of RGB (the v3 independent lag de-synced the streams on ~50% of samples and
            # taught the model depth was temporally unreliable). Only prev_action (a
            # different sensor path with real latency) keeps the lag augmentation.
            if lat_rng.random() < p:
                prev_action = self._lag_one_tick(prev_action)
            # v7 Rewire: SO(2) optical-axis roll co-rotates image + camera-frame labels
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
            'prev_action': prev_action,
            'task': task,
        }
        if depth_mm is not None:
            depth_cam = self._encode_depth(depth_mm)
            self._check_depth_channels(depth_cam)
            obs['depth_cam'] = depth_cam

        return {
            'obs': obs,
            'action': action,
            'goal_cam': goal_cam,       # supervision only; NOT in obs / ONNX
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample, ep_idx=self._episode_of(idx), sample_idx=idx)
        return dict_apply(data, torch.from_numpy)
