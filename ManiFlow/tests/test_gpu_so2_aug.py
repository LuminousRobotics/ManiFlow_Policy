"""Validate that gpu_augment's SO(2) label co-rotation is numerically IDENTICAL to the
CPU dataset path (lumi_place_image_dataset._augment_so2), which is itself unit-tested for
sign correctness in test_so2_rotation_aug.py. Runs on CPU (no GPU/model needed)."""
import math
import numpy as np
import torch

from maniflow.model.vision_2d.gpu_augment import _rz_transpose, _corotate6, _project_norm, SO2_SIGN


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
        got = _corotate6(torch.from_numpy(arr), rz_t).numpy()
        err = np.abs(got - want).max()
        assert err < 1e-9, f"phi={phi}: gpu vs cpu co-rotation mismatch {err}"
    print("GPU SO(2) label co-rotation == CPU path for all phi. sign OK.")


def test_project_norm_matches_converter():
    """_project_norm must equal the converter's normalized projection ((fx*X/Z+cx)/W etc.), so
    re-projecting rotated keypoints is consistent with the stored arm_kpts_uv. Full-res K -> norm."""
    fx, fy, cx, cy = 616.67, 618.23, 639.15, 362.66; W, H = 1280.0, 720.0
    k = (fx / W, fy / H, cx / W, cy / H)
    rs = np.random.RandomState(1)
    for _ in range(20):
        P = np.array([rs.uniform(-0.3, 0.3), rs.uniform(-0.2, 0.2), rs.uniform(0.4, 1.5)])
        u = fx * P[0] / P[2] + cx; v = fy * P[1] / P[2] + cy       # converter's full-res projection
        want = np.array([u / W, v / H, 1.0 if (0 <= u < W and 0 <= v < H) else 0.0])
        got = _project_norm(torch.tensor(P).view(1, 3), k)[0].numpy()
        assert abs(got[2] - want[2]) < 1e-6, "visibility mismatch"
        if want[2] > 0:
            assert np.abs(got[:2] - want[:2]).max() < 1e-6, f"uv mismatch {got[:2]} vs {want[:2]}"
    # co-rotating the 3-D point then re-projecting inherits the validated sign (test above)
    rz_t = _rz_transpose(12.0, torch.device('cpu'), torch.float64)
    P = torch.tensor([0.1, 0.05, 0.7], dtype=torch.float64).view(1, 3)
    uv_rot = _project_norm(P @ rz_t, k)
    assert uv_rot[0, 2] == 1.0 and (0 <= uv_rot[0, 0] <= 1) and (0 <= uv_rot[0, 1] <= 1)
    print("_project_norm == converter projection; rotated-keypoint re-projection consistent.")


if __name__ == "__main__":
    test_sign_matches_cpu()
    test_project_norm_matches_converter()
