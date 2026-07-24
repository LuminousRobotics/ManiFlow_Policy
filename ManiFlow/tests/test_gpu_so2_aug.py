"""Validate that gpu_augment's SO(2) label co-rotation is numerically IDENTICAL to the
CPU dataset path (lumi_place_image_dataset._augment_so2), which is itself unit-tested for
sign correctness in test_so2_rotation_aug.py. Runs on CPU (no GPU/model needed)."""
import math
import numpy as np
import torch

from maniflow.model.vision_2d.gpu_augment import _rz_transpose, _corotate, SO2_SIGN


def _cpu_co(arr, phi_deg):
    """Exact replica of lumi_place_image_dataset._augment_so2's label co-rotation (numpy)."""
    a = np.radians(-phi_deg)  # CPU uses SO2_SIGN = -1 literally
    ca, sa = np.cos(a), np.sin(a)
    Rz_T = np.array([[ca, sa, 0.0], [-sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    out = arr.copy()
    out[..., :3] = arr[..., :3] @ Rz_T
    out[..., 3:6] = arr[..., 3:6] @ Rz_T
    return out


def test_sign_matches_cpu():
    assert SO2_SIGN == -1, "gpu_augment sign drifted from the pinned CPU convention"
    rs = np.random.RandomState(0)
    for phi in (0.0, 5.0, -5.0, 12.0, -12.0, 30.0, -30.0):
        arr = rs.randn(4, 6).astype(np.float64)          # 4 camera-frame [pos|rotvec] vectors
        want = _cpu_co(arr, phi)
        rz_t = _rz_transpose(phi, torch.device('cpu'), torch.float64)
        got = _corotate(torch.from_numpy(arr), rz_t).numpy()
        err = np.abs(got - want).max()
        assert err < 1e-9, f"phi={phi}: gpu vs cpu co-rotation mismatch {err}"
    print("GPU SO(2) label co-rotation == CPU path for all phi. sign OK.")


if __name__ == "__main__":
    test_sign_matches_cpu()
