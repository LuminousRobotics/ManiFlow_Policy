"""Lumi place-policy dataset for ManiFlow (pickplace_v4 -> train.zarr contract).

Clone of maniflow.dataset.robotwin_image_dataset.RoboTwinImageDataset extended with:
- buffer keys: head_camera, depth, state, imu, task, action
- obs dict keys: head_cam, depth_cam (1ch depth tiled to 3ch), agent_pos, imu, task
- spec 4.4 dataloader-side augmentation (train split only, zarr stays clean):
    depth : axial Gaussian sigma0*d^2, random pixel dropout, flying-pixel speckle,
            full-episode dropout
    imu   : per-episode mounting-offset + gravity-direction rotation on the quat,
            per-episode bias + slow random-walk on ang_vel / lin_acc,
            full-episode dropout
    latency: per-episode, per-stream 0/1-tick shift (edge-repeat at window start)
  All parameters come from the `augmentation` config dict (training config, versioned).
  Per-episode decisions are seeded from (seed, episode); per-sample noise from
  (seed, episode, sample_idx) — fully deterministic.

This file is COPY'd into the ManiFlow clone as
ManiFlow/maniflow/dataset/lumi_place_image_dataset.py at docker build.
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

DEPTH_NEAR_M = 0.1
DEPTH_FAR_M = 3.0

DEFAULT_AUGMENTATION = {
    "enable": True,
    "depth_noise_sigma0": 0.005,     # axial sigma = sigma0 * d^2  [m]
    "depth_px_dropout": 0.02,        # fraction of valid px zeroed per frame
    "depth_flying_px": 0.005,        # fraction of px given random depth per frame
    "depth_episode_dropout": 0.4,    # fraction of episodes with depth fully zeroed
    "imu_grav_noise_deg": [0.5, 2.0],  # per-episode gravity-direction error (uniform range)
    "imu_mount_offset_deg": 0.5,       # per-episode fixed mounting offset
    "imu_bias_angvel": 0.02,           # per-episode constant bias sigma [rad/s]
    "imu_bias_linacc": 0.1,            # per-episode constant bias sigma [m/s^2]
    "imu_walk_scale": 0.1,             # random-walk step sigma as fraction of bias sigma
    "imu_episode_dropout": 0.15,       # fraction of episodes with IMU fully zeroed
    "latency_shift_prob": 0.5,         # per-episode prob a stream lags 1 control tick
}


def _rotvec_to_rotmat(rv):
    angle = float(np.linalg.norm(rv))
    if angle < 1e-12:
        return np.eye(3)
    axis = rv / angle
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _quat_wxyz_mul(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)


def _rotmat_to_quat_wxyz(R):
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-8:
        return np.array([w,
                         (R[2, 1] - R[1, 2]) / (4 * w),
                         (R[0, 2] - R[2, 0]) / (4 * w),
                         (R[1, 0] - R[0, 1]) / (4 * w)])
    # fall back for 180-degree rotations (never hit for our sub-degree noise)
    d = np.diagonal(R)
    k = int(np.argmax(d))
    q = np.zeros(4)
    q[k + 1] = np.sqrt(max(0.0, 1 + 2 * d[k] - np.trace(R))) / 2.0
    return q


def _random_small_rotation_quat(rng, angle_deg):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    rv = axis * np.radians(angle_deg)
    return _rotmat_to_quat_wxyz(_rotvec_to_rotmat(rv))


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
                 **kwargs):
        super().__init__()
        self.task_name = task_name
        cprint(f'Loading LumiPlaceImageDataset from {zarr_path}', 'green')

        buffer_keys = ['head_camera', 'depth', 'state', 'imu', 'task', 'action']
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

    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'],
            'imu': self.replay_buffer['imu'],
            'task': self.replay_buffer['task'],
        }
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['head_cam'] = SingleFieldLinearNormalizer.create_identity()
        normalizer['depth_cam'] = SingleFieldLinearNormalizer.create_identity()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    # ------------------------------------------------------------------ augmentation

    def _episode_of(self, idx: int) -> int:
        buffer_start_idx = int(self.sampler.indices[idx][0])
        return int(np.searchsorted(self._episode_ends, buffer_start_idx, side='right'))

    def _augment_depth(self, depth_u8, ep_rng, sample_rng):
        a = self.aug
        if ep_rng.random() < a["depth_episode_dropout"]:
            return np.zeros_like(depth_u8)
        d = depth_u8.astype(np.float32)
        valid = d > 0
        meters = (d - 1.0) / 254.0 * (DEPTH_FAR_M - DEPTH_NEAR_M) + DEPTH_NEAR_M
        # axial noise sigma0 * d^2
        noise = sample_rng.normal(size=d.shape).astype(np.float32) \
            * (a["depth_noise_sigma0"] * meters * meters)
        meters = meters + noise
        # random pixel dropout
        drop = sample_rng.random(d.shape) < a["depth_px_dropout"]
        # flying pixels: random px get a random in-range depth
        fly = sample_rng.random(d.shape) < a["depth_flying_px"]
        meters[fly] = sample_rng.uniform(DEPTH_NEAR_M, DEPTH_FAR_M, size=int(fly.sum()))
        meters = np.clip(meters, DEPTH_NEAR_M, DEPTH_FAR_M)
        out = ((meters - DEPTH_NEAR_M) / (DEPTH_FAR_M - DEPTH_NEAR_M) * 254.0 + 1.0)
        out[~valid | drop] = 0.0
        return out.astype(np.uint8)

    def _augment_imu(self, imu, ep_rng, sample_rng):
        a = self.aug
        if ep_rng.random() < a["imu_episode_dropout"]:
            return np.zeros_like(imu)
        imu = imu.copy()
        T = imu.shape[0]
        # per-episode fixed rotations: mounting offset + gravity-direction error
        lo, hi = a["imu_grav_noise_deg"]
        q_err = _random_small_rotation_quat(ep_rng, ep_rng.uniform(lo, hi))
        q_mount = _random_small_rotation_quat(ep_rng, a["imu_mount_offset_deg"])
        q_off = _quat_wxyz_mul(q_mount[None, :], q_err[None, :])[0]
        imu[:, 0:4] = _quat_wxyz_mul(np.tile(q_off, (T, 1)), imu[:, 0:4])
        imu[:, 0:4] /= np.linalg.norm(imu[:, 0:4], axis=1, keepdims=True)
        # per-episode constant bias + slow random walk
        bias_w = ep_rng.normal(scale=a["imu_bias_angvel"], size=3)
        bias_a = ep_rng.normal(scale=a["imu_bias_linacc"], size=3)
        walk_w = np.cumsum(sample_rng.normal(
            scale=a["imu_bias_angvel"] * a["imu_walk_scale"], size=(T, 3)), axis=0)
        walk_a = np.cumsum(sample_rng.normal(
            scale=a["imu_bias_linacc"] * a["imu_walk_scale"], size=(T, 3)), axis=0)
        imu[:, 4:7] += bias_w[None, :] + walk_w
        imu[:, 7:10] += bias_a[None, :] + walk_a
        return imu

    @staticmethod
    def _lag_one_tick(arr):
        lagged = np.empty_like(arr)
        lagged[1:] = arr[:-1]
        lagged[0] = arr[0]  # edge-repeat at window start
        return lagged

    # ------------------------------------------------------------------ sample assembly

    def _sample_to_data(self, sample, ep_idx=None, sample_idx=0):
        state = sample['state'][:, ].astype(np.float32)
        imu = sample['imu'][:, ].astype(np.float32)
        task = sample['task'][:, ].astype(np.float32)
        depth_u8 = sample['depth'][:, ]  # (T,1,S,S) uint8

        if self.augment and ep_idx is not None:
            ep_rng = np.random.default_rng([self.seed, 1000003, ep_idx])
            sample_rng = np.random.default_rng([self.seed, 7777777, ep_idx, sample_idx])
            depth_u8 = self._augment_depth(depth_u8, ep_rng, sample_rng)
            imu = self._augment_imu(imu, ep_rng, sample_rng)
            # per-episode, per-stream latency: stream lags the rgb anchor by one tick
            lat_rng = np.random.default_rng([self.seed, 424243, ep_idx])
            p = self.aug["latency_shift_prob"]
            if lat_rng.random() < p:
                depth_u8 = self._lag_one_tick(depth_u8)
            if lat_rng.random() < p:
                imu = self._lag_one_tick(imu)
            if lat_rng.random() < p:
                state = self._lag_one_tick(state)

        head_cam = sample['head_camera'][:, ].astype(np.float32) / 255.0
        depth_cam = np.repeat(depth_u8.astype(np.float32) / 255.0, 3, axis=1)  # 1ch -> 3ch

        return {
            'obs': {
                'head_cam': head_cam,
                'depth_cam': depth_cam,
                'agent_pos': state,
                'imu': imu,
                'task': task,
            },
            'action': sample['action'].astype(np.float32),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample, ep_idx=self._episode_of(idx), sample_idx=idx)
        return dict_apply(data, torch.from_numpy)
