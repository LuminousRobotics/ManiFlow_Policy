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
from termcolor import cprint

from maniflow.common.pytorch_util import dict_apply
from maniflow.common.replay_buffer import ReplayBuffer
from maniflow.common.sampler import SequenceSampler, get_val_mask, downsample_mask
from maniflow.model.common.normalizer import LinearNormalizer, SingleFieldLinearNormalizer
from maniflow.dataset.base_dataset import BaseDataset

# depth stored as uint16 millimeters (0 = invalid); normalized to [0,1] over this range.
DEPTH_MM_MAX = 3000.0
# H3DP-lite depth bands (meters): near / mid / far soft masks -> 3 input channels.
DEPTH_BANDS_M = ((0.10, 0.45), (0.35, 0.80), (0.70, 1.50))
DEPTH_BAND_SOFT = 0.05   # meters of smooth falloff at each band edge

DEFAULT_AUGMENTATION = {
    "enable": True,
    "depth_noise_sigma0": 0.005,     # axial sigma = sigma0 * d^2  [m]
    "depth_px_dropout": 0.02,        # fraction of valid px zeroed per frame
    "depth_flying_px": 0.005,        # fraction of px given random depth per frame
    "depth_episode_dropout": 0.4,    # fraction of episodes with depth fully zeroed
    "prev_action_noise_mm": 0.5,     # gaussian on prev_action translation [mm]
    "prev_action_noise_deg": 0.05,   # gaussian on prev_action rotvec [deg]
    "latency_shift_prob": 0.5,       # per-episode prob a stream lags 1 control tick
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
                 depth_input="tiled",        # "tiled" | "layered" (H3DP-lite)
                 **kwargs):
        super().__init__()
        self.task_name = task_name
        self.depth_input = depth_input
        cprint(f'Loading LumiPlaceImageDataset (v6 anchored, depth={depth_input}) '
               f'from {zarr_path}', 'green')

        buffer_keys = ['head_camera', 'depth', 'tcp_pos_w', 'tcp_quat_w', 'cam_pos_w',
                       'cam_quat_cv', 'prev_action', 'gravity_cam', 'goal_pos_cam',
                       'goal_rot_cam', 'task']
        self.replay_buffer = ReplayBuffer.copy_from_path(zarr_path, keys=buffer_keys)

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
        """(T,1,S,S) float mm -> (T,3,S,S) float32 in [0,1].
        tiled: normalized depth repeated x3. layered: 3 soft depth-band masks."""
        m = depth_mm_float / 1000.0                      # meters, 0 = invalid
        valid = depth_mm_float > 0
        if self.depth_input == "layered":
            chans = []
            for lo, hi in DEPTH_BANDS_M:
                s = DEPTH_BAND_SOFT
                up = np.clip((m - (lo - s)) / s, 0.0, 1.0)
                down = np.clip(((hi + s) - m) / s, 0.0, 1.0)
                band = np.minimum(up, down) * valid
                chans.append(band)
            return np.concatenate(chans, axis=1).astype(np.float32)   # (T,3,S,S)
        norm = np.clip(m / (DEPTH_MM_MAX / 1000.0), 0.0, 1.0) * valid
        return np.repeat(norm, 3, axis=1).astype(np.float32)

    # ------------------------------------------------------------------ sample assembly

    def _sample_to_data(self, sample, ep_idx=None, sample_idx=0):
        task = sample['task'][:, ].astype(np.float32)
        prev_action = sample['prev_action'][:, ].astype(np.float32)
        depth_mm = sample['depth'][:, ].astype(np.float32)   # (T,1,S,S) mm (float for aug)

        anchor = min(self.pad_before, len(sample['tcp_pos_w']) - 1)
        action = build_anchored_chunk(
            sample['tcp_pos_w'], sample['tcp_quat_w'], sample['cam_quat_cv'], anchor)
        goal_cam = np.concatenate(
            [sample['goal_pos_cam'][anchor], sample['goal_rot_cam'][anchor]]).astype(np.float32)

        if self.augment and ep_idx is not None:
            ep_rng = np.random.default_rng([self.seed, 1000003, ep_idx])
            sample_rng = np.random.default_rng([self.seed, 7777777, ep_idx, sample_idx])
            depth_mm = self._augment_depth_mm(depth_mm, ep_rng, sample_rng)
            prev_action = self._augment_prev_action(prev_action, sample_rng)
            lat_rng = np.random.default_rng([self.seed, 424243, ep_idx])
            p = self.aug["latency_shift_prob"]
            if lat_rng.random() < p:
                depth_mm = self._lag_one_tick(depth_mm)
            if lat_rng.random() < p:
                prev_action = self._lag_one_tick(prev_action)

        head_cam = sample['head_camera'][:, ].astype(np.float32) / 255.0
        depth_cam = self._encode_depth(depth_mm)

        return {
            'obs': {
                'head_cam': head_cam,
                'depth_cam': depth_cam,
                'prev_action': prev_action,
                'task': task,
            },
            'action': action,
            'goal_cam': goal_cam,       # supervision only; NOT in obs / ONNX
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample, ep_idx=self._episode_of(idx), sample_idx=idx)
        return dict_apply(data, torch.from_numpy)
