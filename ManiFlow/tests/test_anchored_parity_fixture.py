"""Regression guard (fork side): build_anchored_chunk must produce the pinned fixture
outputs. The SAME fixture lives in the pipeline repo (tests/test_anchored_parity.py) where
a bit-exact pipeline<->fork check runs; if either side's anchored math drifts, one of the
two tests fails. Loads only the numpy math (no torch/maniflow import).

Run: python tests/test_anchored_parity_fixture.py
"""
import os

import numpy as np

FIX_P = np.array([
    [0.000, 0.000, 0.000], [0.010, -0.005, -0.030], [0.018, -0.011, -0.061],
    [0.022, -0.014, -0.095], [0.024, -0.016, -0.128], [0.025, -0.017, -0.160],
], dtype=np.float64)
FIX_Q = np.array([
    [1.0, 0.0, 0.0, 0.0], [0.99998, 0.001, 0.002, 0.006], [0.99990, 0.002, 0.004, 0.013],
    [0.99979, 0.003, 0.005, 0.019], [0.99964, 0.004, 0.007, 0.025], [0.99946, 0.005, 0.008, 0.031],
], dtype=np.float64)
FIX_CQ = np.array([
    [0.707, 0.0, 0.0, 0.707], [0.7071, 0.002, 0.001, 0.7071], [0.7069, 0.004, 0.002, 0.7072],
    [0.7068, 0.006, 0.003, 0.7073], [0.7066, 0.008, 0.004, 0.7074], [0.7064, 0.010, 0.005, 0.7075],
], dtype=np.float64)
for _a in (FIX_Q, FIX_CQ):
    _a /= np.linalg.norm(_a, axis=1, keepdims=True)

# Pinned from the verified pipeline<->fork parity run (anchor=1).
EXPECTED = np.array([
    [0.005042, 0.010127, 0.029950, -0.004017, 0.001949, -0.012003],
    [0.000000, 0.000000, 0.000000, -0.000000, 0.000000, 0.000000],
    [-0.006044, -0.008131, -0.030957, 0.004018, -0.001945, 0.014004],
    [-0.009092, -0.012276, -0.064936, 0.006036, -0.003906, 0.026009],
    [-0.011139, -0.014416, -0.097924, 0.010054, -0.005856, 0.038015],
    [-0.012184, -0.015551, -0.129918, 0.012072, -0.007818, 0.050024],
], dtype=np.float64)


def _load_build():
    ds = os.path.join(os.path.dirname(__file__), "..", "maniflow", "dataset",
                      "lumi_place_image_dataset.py")
    src = open(ds).read()
    ns = {"np": np}
    exec(src[src.index("def _quat_wxyz_to_rotmat"):src.index("def _rv_to_R")], ns)
    return ns["build_anchored_chunk"]


def test_anchored_fixture():
    out = _load_build()(FIX_P, FIX_Q, FIX_CQ, 1)
    assert np.allclose(out[1], 0.0), "anchor row must be zero"
    assert np.allclose(out, EXPECTED, atol=1e-5), \
        f"anchored fixture drift:\n got {np.round(out, 6)}\n exp {EXPECTED}"


if __name__ == "__main__":
    test_anchored_fixture()
    print("fork anchored fixture OK")
