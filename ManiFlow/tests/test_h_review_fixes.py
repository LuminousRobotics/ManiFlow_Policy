"""Regression tests for the H-series post-review corrections (contract §9.2-§9.5, §9.4, §9.3).

Every test here FAILS on the pre-fix code — that is the point of the file. They are written
against the SAME synthetic v10 fixture as `test_h_dataset_policy.py` (imported, not duplicated),
so a fixture change cannot make one file pass while the other rots.

What each block pins, and the measured failure it exists to prevent:

  A  goal-coincidence integral   the in-reach mask was computed against the NOISED goal with a
                                 5 mm tolerance while scheduled sampling injects 15 mm -> mask
                                 empty on ~99.6% of samples; the logged loss read 0.0, i.e.
                                 "perfect", on every batch with nothing supervised behind it.
  B  terminal-zero               applied unmasked to EVERY anchor, fighting the BC labels on the
                                 ~47% of windows whose last row is full descent speed.
  C  perception labels           heads that see ONLY pixels were supervised with labels from the
                                 proprio/anchor frame -> ~40% of train samples off by one control
                                 frame under the live timing augmentation.
  D  augmentation RNG            no epoch in the seed -> every sample's "random" perturbation was
                                 a fixed constant for the whole run.
  E  normalizer coverage         goal-frame rows fit on CLEAN data only -> near-terminal rows are
                                 noise-dominated and left the fitted range (2.48x).
  F  dt honesty                  padded episode-start windows report the nominal gap although the
                                 two obs images are byte-identical duplicates.
  G  so3_torch near pi           acos clamp collapsed |rotvec| to 0.70 within 4.5e-4 rad of pi.
  H  n_action_steps              one config scalar for four modes with 15/15/15/16 rows.

Run:  python tests/test_h_review_fixes.py
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from maniflow.common import so3_torch as st                                     # noqa: E402
from maniflow.dataset import lumi_place_image_dataset as fk                     # noqa: E402
from maniflow.dataset.lumi_place_image_dataset import LumiPlaceImageDataset     # noqa: E402

import test_h_dataset_policy as base                                            # noqa: E402

POSE_H = base.POSE_H
CONTROL_HZ = base.CONTROL_HZ
SEED = base.SEED

# The live v4 config's augmentation (config/robotwin_task/lumi_place_v4.yaml). Tests that claim
# "under the live augmentation X happens" must actually use it, or they measure nothing.
LIVE_AUG = dict(
    enable=True, gpu_offload=True, depth_episode_dropout=0.1,
    obs_gap_probs=[0.1, 0.7, 0.2], image_lag_probs=[0.6, 0.4],
    proprio_noise_joint_deg=0.7, proprio_noise_twist_frac=0.05, proprio_noise_ego_mm=1.0,
    goal_frame_noise_mm=15.0, goal_frame_noise_deg=1.5)


def _ds(zarr_path, action_param="twist", action_frame="goal", val_ratio=0.0, **aug):
    a = dict(LIVE_AUG)
    a.update(aug)
    return LumiPlaceImageDataset(
        zarr_path=zarr_path, horizon=POSE_H, pad_before=1, pad_after=7, seed=SEED,
        val_ratio=val_ratio, use_depth=True, depth_input='xyz', n_obs_steps=2,
        control_hz=CONTROL_HZ, action_param=action_param, action_frame=action_frame,
        augmentation=a)


def _collate(ds, idxs):
    from torch.utils.data._utils.collate import default_collate
    return default_collate([ds[int(i)] for i in idxs])


def _buffer_index(ds, idx, w):
    """Window slot `w` of sample `idx` -> absolute replay-buffer row (padding-aware)."""
    b_start, _, s_start, _ = (int(v) for v in ds.sampler.indices[idx])
    return b_start + (max(int(w), s_start) - s_start)


# ============================================================== A. goal-coincidence integral
def test_A_inreach_mask_is_clean_goal(zarr_path):
    """The in-reach mask must be a property of the LABELS, so goal-frame noise cannot move it."""
    clean = _ds(zarr_path, goal_frame_noise_mm=0.0, goal_frame_noise_deg=0.0)
    noisy = _ds(zarr_path)                                    # 15 mm / 1.5 deg = the live config
    a = np.array([float(clean[i]['h_goal_inreach']) for i in range(len(clean))])
    b = np.array([float(noisy[i]['h_goal_inreach']) for i in range(len(noisy))])
    assert a.mean() > 0.05, (
        f"fixture has no in-reach anchors ({a.mean():.3f}) — this test would pass vacuously")
    assert np.array_equal(a, b), (
        f"goal-frame noise moved the in-reach mask: clean {a.mean():.3f} -> noised {b.mean():.3f}. "
        f"The mask asks whether the HORIZON REACHES THE GOAL (label geometry); a 15 mm output-frame "
        f"perturbation against a 5 mm tolerance turns it off (P(pass) ~ 0.4%) and silently deletes "
        f"the flagship arm's signature loss.")
    print(f"  A: in-reach fraction clean {a.mean():.3f} == noised {b.mean():.3f} "
          f"({int(a.sum())}/{len(a)} anchors supervised)")


def test_A_integral_loss_is_alive(zarr_path, policies):
    """twist+goal: the integral loss must actually be computed, and its mask fraction logged."""
    ds = _ds(zarr_path, "twist", "goal")
    policy = policies[("twist", "goal")]
    policy.set_normalizer(ds.get_normalizer())
    policy.set_epoch(0)                       # no self-frame: isolate the mask, not the ramp
    policy.train()
    reach = np.array([float(ds[i]['h_goal_inreach']) for i in range(len(ds))])
    idxs = list(np.nonzero(reach)[0][:2]) + list(np.nonzero(1 - reach)[0][:2])
    batch = _collate(ds, idxs)
    torch.manual_seed(0)
    _, log = policy.compute_loss(batch, ema_model=policy)
    assert 'goal_int_frac' in log and 'goal_reach_frac' in log, sorted(log)
    # the constraints are evaluated on the FLOW sub-batch (the endpoint reconstruction lives
    # there), so the reported fraction is over its first int(B*flow_batch_ratio) samples
    n_flow = int(len(idxs) * policy.flow_batch_ratio)
    want = float(batch['h_goal_inreach'][:n_flow].mean())
    assert 0.0 < want < 1.0, want                     # a mixed batch, or this proves nothing
    assert abs(log['goal_int_frac'] - want) < 1e-6, (log['goal_int_frac'], want)
    assert abs(log['goal_reach_frac'] - want) < 1e-6, (log['goal_reach_frac'], want)
    assert np.isfinite(log['loss_goal_int']) and log['loss_goal_int'] > 0.0, log['loss_goal_int']
    assert np.isfinite(log['goal_int_mm']) and log['goal_int_mm'] > 0.0, log['goal_int_mm']

    # ... and a batch with NOTHING supervised must NOT log as 0.0 ("perfect").
    empty = dict(batch)
    empty['h_goal_inreach'] = torch.zeros_like(batch['h_goal_inreach'])
    torch.manual_seed(0)
    _, log0 = policy.compute_loss(empty, ema_model=policy)
    assert log0['goal_int_frac'] == 0.0, log0['goal_int_frac']
    assert np.isnan(log0['loss_goal_int']), (
        f"a masked-EMPTY batch logged loss_goal_int={log0['loss_goal_int']} — a dashboard reads "
        f"that as a perfectly satisfied constraint. It must be NaN + a 0.00 mask fraction.")
    assert np.isnan(log0['goal_int_mm']), log0['goal_int_mm']
    print(f"  A: twist/goal loss_goal_int={log['loss_goal_int']:.4f} "
          f"({log['goal_int_mm']:.1f} mm) at mask {log['goal_int_frac']:.2f}; "
          f"empty mask -> NaN (not 0.0)")


def test_A_delta_goal_needs_no_mask(zarr_path, policies):
    """delta+goal: u_0 IS the remaining transform identically (noise included), so the integral
    constraint holds at EVERY anchor and masking it would throw away most of the supervision."""
    ds = _ds(zarr_path, "delta", "goal")
    worst = 0.0
    for i in range(0, len(ds), 3):
        s = ds[i]
        worst = max(worst, float((s['action'][0, :6] - s['h_goal_remaining']).abs().max()))
    assert worst < 1e-6, f"u_0 != remaining by {worst:.2e} — the identity §9.2 relies on is broken"
    policy = policies[("delta", "goal")]
    policy.set_normalizer(ds.get_normalizer())
    policy.set_epoch(0)
    policy.train()
    reach = np.array([float(ds[i]['h_goal_inreach']) for i in range(len(ds))])
    idxs = list(np.nonzero(1 - reach)[0][:4])          # ALL out of reach
    assert len(idxs) == 4
    torch.manual_seed(0)
    _, log = policy.compute_loss(_collate(ds, idxs), ema_model=policy)
    assert log['goal_reach_frac'] == 0.0, log['goal_reach_frac']
    assert log['goal_int_frac'] == 1.0, (
        f"delta+goal masked its integral to {log['goal_int_frac']} on out-of-reach anchors; "
        f"u_0 == remaining is an IDENTITY there, so the whole batch is valid supervision")
    assert np.isfinite(log['loss_goal_int'])
    print(f"  A: delta/goal u_0 == remaining to {worst:.1e} (noised); integral mask 1.00 "
          f"while reach 0.00 -> loss_goal_int={log['loss_goal_int']:.4f}")


# ====================================================================== B. terminal-zero loss
def test_B_terminal_zero_is_masked(zarr_path, policies):
    """The terminal row of an out-of-reach anchor is full descent speed, not zero. Pushing it to
    zero anyway fights the labels on ~half the dataset."""
    ds = _ds(zarr_path, "twist", "goal")
    reach = np.array([float(ds[i]['h_goal_inreach']) for i in range(len(ds))])
    out_idx = list(np.nonzero(1 - reach)[0][:4])
    last = np.stack([ds[int(i)]['action'][-1, :6].numpy() for i in out_idx])
    assert np.abs(last).max() > 0.01, (
        f"fixture's out-of-reach terminal rows are already ~0 ({np.abs(last).max():.4f}) — the "
        f"conflict this test measures is not present")
    policy = policies[("twist", "goal")]
    policy.set_normalizer(ds.get_normalizer())
    policy.set_epoch(0)
    policy.train()
    batch = _collate(ds, out_idx)
    torch.manual_seed(0)
    l_on, log = policy.compute_loss(batch, ema_model=policy)
    assert np.isnan(log['loss_term']), (
        f"loss_term={log['loss_term']} on a batch where EVERY GT terminal row is up to "
        f"{np.abs(last).max():.3f} away from zero — the loss is fighting the labels")
    # ... and it must contribute NOTHING to the objective there
    w = policy.terminal_zero_weight
    policy.terminal_zero_weight = 0.0
    torch.manual_seed(0)
    l_off, _ = policy.compute_loss(batch, ema_model=policy)
    policy.terminal_zero_weight = w
    l_on, l_off = float(l_on.detach()), float(l_off.detach())
    assert abs(l_on - l_off) < 1e-9, (l_on, l_off)
    print(f"  B: out-of-reach batch |GT row_last| up to {np.abs(last).max():.3f} -> loss_term NaN "
          f"and zero objective contribution (weight {w} vs 0: {l_on:.6f})")


# ============================================================ C. perception labels follow pixels
def test_C_labels_follow_pixels(zarr_path):
    """Every target of an image-only head is indexed at the IMAGE frames, not the anchor."""
    ds = _ds(zarr_path, "twist", "goal", image_lag_probs=[0.0, 1.0], obs_gap_probs=[0.0, 1.0, 0.0])
    rb = ds.replay_buffer
    anchor = ds.pad_before
    checked = 0
    gaps = []
    for idx in range(0, len(ds), 5):
        if int(ds.sampler.indices[idx][2]) != 0:
            continue                              # skip front-padded windows (labels alias)
        s = ds[idx]
        b_img = _buffer_index(ds, idx, anchor - 1)          # lag 1 => newest image is anchor-1
        b_anc = _buffer_index(ds, idx, anchor)
        for key, zkey in (('panel_goal', 'panel_goal_cam'), ('rail_a', 'rail_a_cam')):
            want = np.asarray(rb[zkey][b_img], np.float64)
            other = np.asarray(rb[zkey][b_anc], np.float64)
            got = s[key].numpy().astype(np.float64)
            assert np.abs(got - want).max() < 1e-6, (
                f"{key} came from the anchor, not the image frame: err vs image "
                f"{np.abs(got - want).max():.2e} vs err against anchor "
                f"{np.abs(got - other).max():.2e}")
            gaps.append(float(np.linalg.norm(want[:3] - other[:3])))
        for key, zkey in (('arm_kpts_uv', 'arm_kpts_uv'), ('arm_kpts_cam', 'arm_kpts_cam')):
            for j, w in enumerate((anchor - 2, anchor - 1)):
                want = np.asarray(rb[zkey][_buffer_index(ds, idx, w)], np.float64)
                assert np.abs(s[key][j].numpy().astype(np.float64) - want).max() < 1e-6, key
        checked += 1
    assert checked >= 3, checked
    assert max(gaps) > 1e-4, (
        f"the two frames' labels differ by only {max(gaps):.2e} m — the fixture's camera is not "
        f"moving, so this test cannot distinguish the frames")
    print(f"  C: {checked} lag-1 windows — panel_goal/rail_a/arm_kpts all taken at the IMAGE "
          f"frames (frame-to-frame label motion up to {max(gaps) * 1000:.1f} mm)")


def test_C_actions_stay_anchored(zarr_path):
    """ACTION labels must be bit-identical across every gap/lag draw (they describe motion from
    the anchor forward). This is the invariant the §9.3 re-indexing must NOT regress."""
    ref = None
    for gp, lp in (([0.0, 1.0, 0.0], [1.0, 0.0]), ([1.0, 0.0, 0.0], [1.0, 0.0]),
                   ([0.0, 0.0, 1.0], [1.0, 0.0]), ([0.0, 1.0, 0.0], [0.0, 1.0]),
                   ([0.1, 0.7, 0.2], [0.6, 0.4])):
        ds = _ds(zarr_path, "delta", "goal", obs_gap_probs=gp, image_lag_probs=lp,
                 goal_frame_noise_mm=0.0, goal_frame_noise_deg=0.0)
        cur = np.stack([ds[i]['action'].numpy() for i in range(0, len(ds), 4)])
        phase = np.stack([ds[i]['phase_id'].numpy() for i in range(0, len(ds), 4)])
        if ref is None:
            ref, ref_phase = cur, phase
        else:
            assert np.array_equal(cur, ref), (
                f"action rows changed with the timing draw {gp}/{lp} — the chunk must stay "
                f"anchored at the anchor")
            assert np.array_equal(phase, ref_phase), "phase_id must stay at the anchor too"
    print(f"  C: action rows + phase_id bit-identical across 5 gap/lag configurations "
          f"({ref.shape[0]} windows)")


def test_C_goal_maps_track_the_image_frame(zarr_path):
    """The policy composes the goal frame from the place head's output, which now lives in the
    IMAGE camera frame — so the dataset's composition maps must carry that frame change (`A`).
    Feeding the GT label through the policy's own formula must reproduce the GT goal frame."""
    ds = _ds(zarr_path, "delta", "goal", image_lag_probs=[0.0, 1.0],
             obs_gap_probs=[0.0, 1.0, 0.0], goal_frame_noise_mm=0.0, goal_frame_noise_deg=0.0)
    worst_rot, worst_pos, worst_naive = 0.0, 0.0, 0.0
    for idx in range(0, len(ds), 5):
        if int(ds.sampler.indices[idx][2]) != 0:
            continue
        s = ds[idx]
        Y = fk.rotvec_to_rotmat(s['panel_goal'].numpy()[3:6].astype(np.float64))
        A = s['h_gc_A'].numpy().astype(np.float64)
        # (a) goal FRAME: A @ Y @ R_frame == F  (what `_h_reframe` builds as F_hat)
        F_hat = A @ Y @ s['h_gc_Rframe'].numpy().astype(np.float64)
        F_gt = s['h_goal_frame'].numpy().astype(np.float64)
        worst_rot = max(worst_rot, float(np.abs(fk.rotmat_to_rotvec(F_hat @ F_gt.T)).max()))
        # what the pre-fix (anchor-frame) formula would have produced
        worst_naive = max(worst_naive, float(np.abs(fk.rotmat_to_rotvec(
            (Y @ s['h_gc_Rframe'].numpy().astype(np.float64)) @ F_gt.T)).max()))
        # (b) ee goal POSITION in C0 relative to p_ee0 (what `gc_target` builds)
        p_hat = A @ (s['panel_goal'].numpy()[:3].astype(np.float64)
                     + Y @ s['h_gc_tvec'].numpy().astype(np.float64)) \
            + s['h_gc_toff'].numpy().astype(np.float64)
        b_anc = _buffer_index(ds, idx, ds.pad_before)
        R_cv0 = fk.quat_wxyz_to_rotmat(ds.replay_buffer['cam_quat_cv'][b_anc])
        p_gt = R_cv0.T @ (np.asarray(ds.replay_buffer['ee_goal_pos_w'][b_anc], np.float64)
                          - np.asarray(ds.replay_buffer['ee_pos_w'][b_anc], np.float64))
        worst_pos = max(worst_pos, float(np.linalg.norm(p_hat - p_gt)))
    assert worst_rot < 1e-5, f"composed goal FRAME off by {np.degrees(worst_rot):.4f} deg"
    assert worst_pos < 1e-4, f"composed goal POSITION off by {worst_pos * 1000:.3f} mm"
    assert worst_naive > 10.0 * max(worst_rot, 1e-9), (
        f"the anchor-frame formula is indistinguishable here ({np.degrees(worst_naive):.5f} deg) "
        f"— the fixture's camera does not rotate between frames, so this test proves nothing")
    print(f"  C: goal maps under lag-1 — frame {np.degrees(worst_rot):.6f} deg / position "
          f"{worst_pos * 1000:.4f} mm (anchor-frame formula would be "
          f"{np.degrees(worst_naive):.4f} deg out)")


# ============================================================================ D. epoch RNG
def test_D_epoch_changes_the_draws(zarr_path):
    """Augmentation must differ across epochs and be deterministic within one."""
    ds = _ds(zarr_path, "twist", "goal")
    assert hasattr(ds, 'set_epoch'), "dataset has no set_epoch (contract §9.4)"
    idxs = list(range(0, len(ds), 3))

    def draw():
        return (np.stack([ds[i]['obs']['dt'].numpy() for i in idxs]),
                np.stack([ds[i]['action'].numpy() for i in idxs]),
                np.stack([ds[i]['obs']['agent_pos'].numpy() for i in idxs]))

    ds.set_epoch(0); e0 = draw()
    ds.set_epoch(0); e0b = draw()
    ds.set_epoch(1); e1 = draw()
    ds.set_epoch(7); e7 = draw()
    for a, b, name in zip(e0, e0b, ('dt', 'action', 'agent_pos')):
        assert np.array_equal(a, b), f"{name} is not deterministic within an epoch"
    changed = [name for a, b, name in zip(e0, e1, ('dt', 'action', 'agent_pos'))
               if not np.array_equal(a, b)]
    assert set(changed) == {'dt', 'action', 'agent_pos'}, (
        f"only {changed} changed between epoch 0 and 1 — the epoch is not reaching the timing "
        f"draw (dt), the goal-frame noise (action) or the proprio DR (agent_pos)")
    assert not np.array_equal(e1[0], e7[0])
    # the counter must live in SHARED memory or persistent DataLoader workers never see it
    assert ds._epoch.is_shared(), (
        "the epoch counter is not in shared memory; with persistent_workers=True the forked "
        "DataLoader workers would keep epoch 0 forever and set_epoch would be a no-op")
    gapfrac = float((e0[0] != e1[0]).mean())
    print(f"  D: epoch enters dt / goal noise / proprio DR (timing changed on "
          f"{gapfrac * 100:.0f}% of samples), deterministic within an epoch, counter shared")


def test_D_workspace_calls_set_epoch():
    """The dataset-side counter is useless if nothing advances it."""
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'maniflow', 'workspace',
                            'train_maniflow_robotwin_workspace.py')).read()
    assert "for _d in (dataset, val_dataset):" in src and "_d.set_epoch(self.epoch)" in src, \
        "the training workspace does not call dataset.set_epoch (contract §9.4)"
    print("  D: workspace advances dataset.set_epoch(self.epoch) each epoch (resume-safe)")


# =================================================================== E. normalizer coverage
def test_E_normalizer_covers_noised_rows(zarr_path):
    """Goal-frame rows contract to a few mm near the goal, so a 15 mm output-frame perturbation
    dominates them; a clean-only fit leaves the training tensors outside its own range."""
    out = {}
    for param in ("twist", "delta"):
        ds = _ds(zarr_path, param, "goal")
        got = {}
        for draws, tag in ((0, 'clean-fit'), (fk.H_NORM_NOISE_DRAWS, 'noise-fit')):
            old = fk.H_NORM_NOISE_DRAWS
            fk.H_NORM_NOISE_DRAWS = draws
            try:
                norm = ds.get_normalizer()
            finally:
                fk.H_NORM_NOISE_DRAWS = old
            acts = torch.stack([ds[i]['action'] for i in range(len(ds))])
            got[tag] = float(norm['action'].normalize(acts).abs().max())
        out[param] = got
        assert got['noise-fit'] < 1.5, (
            f"{param}+goal: normalized training actions reach {got['noise-fit']:.2f} even with "
            f"the noise-aware fit")
        assert got['noise-fit'] < got['clean-fit'], (
            f"{param}+goal: the noise-aware fit ({got['noise-fit']:.2f}) is no better than the "
            f"clean fit ({got['clean-fit']:.2f}) — the draws are not reaching the fit")
    assert max(v['clean-fit'] for v in out.values()) > 1.5, (
        "the clean-only fit already covers the noised rows on this fixture — the coverage gap "
        "§9.5 measured is not being exercised")
    for param, got in out.items():
        print(f"  E: {param}+goal max |normalized action| {got['clean-fit']:.2f} (clean fit) "
              f"-> {got['noise-fit']:.2f} (noise-aware fit)")


# ========================================================================= F. dt never lies
def test_F_dt_zero_on_padded_windows(zarr_path):
    """A front-padded window's two obs images are the SAME repeated frame; dt must say 0."""
    ds = _ds(zarr_path, "twist", "goal", image_lag_probs=[1.0, 0.0],
             obs_gap_probs=[0.0, 1.0, 0.0])
    n_pad = n_ok = 0
    for idx in range(len(ds)):
        s_start = int(ds.sampler.indices[idx][2])
        s = ds[idx]
        dt = float(s['obs']['dt'][0])
        img = s['obs']['head_cam'].numpy()
        dup = bool(np.array_equal(img[0], img[1]))
        if s_start >= ds.pad_before:
            assert dup, "expected duplicated obs images in a front-padded window"
            assert dt == 0.0, (
                f"window {idx} reports dt={dt} s but its two obs images are byte-identical "
                f"duplicates (sampler front-padding). dt never lies — and gap=0 is a MODELLED "
                f"failure mode, so a nominal 100 ms here trains parallax that is not in the pixels")
            n_pad += 1
        else:
            assert not dup and abs(dt - 0.1) < 1e-6, (idx, dt, dup)
            n_ok += 1
    assert n_pad > 0 and n_ok > 0, (n_pad, n_ok)
    print(f"  F: {n_pad} front-padded windows report dt=0 (duplicate pixels), "
          f"{n_ok} normal windows report 0.1 s")


# =================================================================== G. so3 rotvec near pi
def test_G_rotmat_to_rotvec_near_pi():
    """`rotmat_to_rotvec` must hold up to and AT pi. Reachable through the goal-consistency
    target, whose rotation is built from an early-training place head's garbage output."""
    axis = np.array([0.3, -0.5, 0.81]); axis /= np.linalg.norm(axis)
    worst = 0.0
    for delta in (1e-2, 1e-3, 1e-4, 1e-6, 0.0):
        rv = axis * (np.pi - delta)
        R = fk.rotvec_to_rotmat(rv)
        out = st.rotmat_to_rotvec(torch.from_numpy(R[None])).numpy()[0]
        assert abs(np.linalg.norm(out) - (np.pi - delta)) < 1e-6, (
            f"|rotvec| = {np.linalg.norm(out):.4f} at delta={delta:g}, want {np.pi - delta:.4f} "
            f"(the acos clamp collapsed the angle)")
        err = float(np.abs(fk.rotmat_to_rotvec(
            st.rotvec_to_rotmat(torch.from_numpy(out)).numpy() @ R.T)).max())
        assert err < 1e-8, f"geodesic error {err:.2e} at delta={delta:g}"
        worst = max(worst, err)
    # in-domain numerics must be untouched (this is what the parity test also pins)
    rng = np.random.default_rng(5)
    rv = rng.normal(scale=0.4, size=(64, 3))
    Rn = np.stack([fk.rotvec_to_rotmat(r) for r in rv])
    e = np.abs(st.rotmat_to_rotvec(torch.from_numpy(Rn)).numpy()
               - np.stack([fk.rotmat_to_rotvec(m) for m in Rn])).max()
    assert e < 1e-12, e
    assert torch.isfinite(st.rotmat_to_rotvec(torch.eye(3).expand(2, 3, 3))).all()
    print(f"  G: exact at pi and within 1e-6 of it (worst geodesic {worst:.2e}); "
          f"in-domain unchanged ({e:.1e} vs the numpy twin)")


# ================================================================== H. n_action_steps per mode
def test_H_n_action_steps(policies):
    for (param, frame), p in policies.items():
        want = fk.h_action_rows(param, frame, POSE_H)
        assert p.n_action_steps == want == p.action_rows, (
            f"{param}/{frame}: n_action_steps={p.n_action_steps} but the model ships {want} rows; "
            f"predict_action's [:, :n_action_steps] slice is then wrong by one mode")
    print("  H: n_action_steps == action_rows for all four modes (15/15/15/16)")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("test_h_review_fixes")
    tmpd = tempfile.mkdtemp()
    try:
        zp = base.build_fixture(os.path.join(tmpd, "train.zarr"))
        sm = base._shape_meta()
        pol = {(pp, ff): base._make_policy(pp, ff, sm)
               for pp in ("twist", "delta") for ff in ("cam0", "goal")}
        print(" A. goal-coincidence integral mask (§9.2):")
        test_A_inreach_mask_is_clean_goal(zp)
        test_A_integral_loss_is_alive(zp, pol)
        test_A_delta_goal_needs_no_mask(zp, pol)
        print(" B. terminal-zero mask (§9.2):")
        test_B_terminal_zero_is_masked(zp, pol)
        print(" C. perception labels follow the pixels (§9.3):")
        test_C_labels_follow_pixels(zp)
        test_C_actions_stay_anchored(zp)
        test_C_goal_maps_track_the_image_frame(zp)
        print(" D. augmentation RNG carries the epoch (§9.4):")
        test_D_epoch_changes_the_draws(zp)
        test_D_workspace_calls_set_epoch()
        print(" E. normalizer covers noised goal-frame rows (§9.5):")
        test_E_normalizer_covers_noised_rows(zp)
        print(" F. dt never lies (§9.4):")
        test_F_dt_zero_on_padded_windows(zp)
        print(" G. so3_torch near pi:")
        test_G_rotmat_to_rotvec_near_pi()
        print(" H. n_action_steps:")
        test_H_n_action_steps(pol)
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    print("PASS")
