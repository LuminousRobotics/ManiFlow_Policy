from typing import Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from termcolor import cprint

from maniflow.model.common.normalizer import LinearNormalizer
from maniflow.policy.base_policy import BasePolicy
from maniflow.common.pytorch_util import dict_apply
from maniflow.common.model_util import print_params
from maniflow.model.vision_2d.timm_obs_encoder import TimmObsEncoder
from maniflow.model.diffusion.ditx import DiTX
from maniflow.model.common.sample_util import *

class ManiFlowTransformerImagePolicy(BasePolicy):
    def __init__(self, 
             shape_meta: dict,
            horizon, 
            n_action_steps, 
            n_obs_steps,
            num_inference_steps=None,
            obs_as_global_cond=True,
            diffusion_timestep_embed_dim=256,
            diffusion_target_t_embed_dim=256,
            visual_cond_len=1024,
            n_layer=3,
            n_head=4,
            n_emb=256,
            qkv_bias=False,
            qk_norm=False,
            block_type="DiTX",
            obs_encoder: TimmObsEncoder = None,
            language_conditioned=False,
            # consistency flow training parameters
            flow_batch_ratio=0.75,
            consistency_batch_ratio=0.25,
            denoise_timesteps=10,
            sample_t_mode_flow="beta",
            sample_t_mode_consistency="discrete",
            sample_dt_mode_consistency="uniform",
            sample_target_t_mode="relative", # relative, absolute
            # --- Lumi v2 precision losses (all default OFF => stock ManiFlow behaviour) ---
            endpoint_loss_weight=0.0,          # lambda for the x1-reconstruction aux loss (flow branch only)
            endpoint_loss_type="mse",          # "mse" or "l1"
            endpoint_t_clip=0.98,              # clip t before dividing by (1-t) in x1_hat reconstruction
            action_step_weights=None,          # optional per-chunk-step weight list, len==horizon (up-weight contact steps)
            action_dim_weights=None,           # optional per-action-dim weight list, len==action_dim (up-weight rotvec/yaw)
            # --- Lumi v3 auxiliary goal head (default OFF => no goal head built) ---
            goal_loss_weight=0.0,              # lambda for the goal-in-camera-frame aux head (Huber); 0 => off
            goal_loss_type="huber",            # "huber" | "mse"
            # --- Lumi C2: low-dim -> AdaLN-Zero routing (default OFF => stock v3 behaviour) ---
            lowdim_to_adaln=False,             # route prev_action/task into DiTX AdaLN instead of cross-attn tokens
            proprio_mask_p=0.0,                # per-sample prob of masking prev_action during training (GAP/ManiFlow
                                               # proprio masking: keeps vision load-bearing for goal localization)
            # --- v7 Rewire-MVP: training-only vision-forcing aux (default OFF => prior behaviour) ---
            idm_loss_weight=0.0,               # lambda for the vision-only inverse-dynamics aux head (Huber)
            # --- F-series: soft-argmax keypoint head + self-predicted goal token ON the actor path ---
            kpt_loss_weight=0.0,               # lambda for rail/tube keypoint soft-argmax head; 0 => no F heads
            place_loss_weight=0.0,             # lambda for keypoint->place-pose regression (vs goal_cam)
            n_keypoints=7,                     # converter N_KEYPOINTS (rail samples + tube)
            kpt_head_hires=False,              # G1: 56x56 deconv heatmap head + log-z -> N 3-D point tokens
            pointnet_dim=0,                    # F1-depth: PointNet on the xyz point-map -> 3D token; 0 => off
            # --- H-series (v10): ee_link action space + goal-frame anchoring (contract §3/§4/§6).
            # action_param=None keeps every prior arm bit-identical; the whole H chassis is OFF.
            action_param=None,                 # None (legacy) | "twist" | "delta"
            action_frame="cam0",               # "cam0" | "goal"
            control_hz=10.0,                   # Delta t = 1/control_hz for the twist integral
            rot_integral_approx=False,         # ablation: small-angle SUM instead of composition
            goal_integral_weight=0.0,          # goal-coincidence integral constraint (goal frames)
            terminal_zero_weight=0.0,          # terminal settle ||row_last|| (goal frames)
            goal_consistency_weight=0.0,       # ||u_0 - ad(P_0, G_hat, R_g)|| (delta+goal)
            action_rate_weight=0.05,           # sum_k ||row_k - row_{k-1}||^2 (always)
            goal_frame_self_p_max=0.0,         # scheduled sampling: max prob of the SELF frame
            goal_frame_anneal_epochs=40,       # ... ramped 0 -> max over this many epochs
            phase_loss_weight=0.0,             # phase head CE (0=place_xy, 1=place_z)
            done_loss_weight=0.0,              # done head BCE
            rail_aux_weight=0.25,              # AUX rail_a_cam head, as a fraction of place_loss_weight
            rgb3d_pos_enc=False,               # 3D-aware RGB tokens (zero-init xyz patch add)
            pointnet_tokens=16,                # K PointNet tokens (4x4 masked-max-pool grid)
            **kwargs):
        super().__init__()

        # parse shape_meta
        action_shape = shape_meta['action']['shape']
        self.action_shape = action_shape
        if len(action_shape) == 1:
            action_dim = action_shape[0]
        elif len(action_shape) == 2: # use multiple hands
            action_dim = action_shape[0] * action_shape[1]
        else:
            raise NotImplementedError(f"Unsupported action shape {action_shape}")

        obs_shape_meta = shape_meta['obs']
        obs_dict = dict_apply(obs_shape_meta, lambda x: x['shape'])

        # ------------------------------------------------------------------ H-series switch
        # `horizon` in H means POSES (H=16). The row count is per-mode (contract §3, FINAL
        # 2026-07-31): 15 / 15 / 15 / 16 for twist+cam0 / twist+goal / delta+cam0 / delta+goal.
        # The rule that fixes it: A ROW IS SHIPPED IFF THE MODEL PREDICTS IT AND IT CARRIES
        # INFORMATION. `delta`+`cam0`'s k=0 row is identically zero, so it is neither predicted
        # nor shipped — a constant we insert ourselves is not a signal, and a runtime gate on it
        # cannot tell "mis-anchored" from "wants to move" (the G1 field run rejected 26% of
        # HEALTHY inferences that way). `delta`+`goal`'s k=0 row IS predicted: it is the model's
        # own goal estimate expressed in action space, so it ships (see `_h_goal_integral`).
        # `action_row_k_start` (1,1,1,0) is the authoritative index of the first shipped row and
        # travels in metadata_props next to `action_rows`.
        # Everything downstream that is shaped by the action tensor — DiTX pos_emb, cond_data,
        # the normalizer, eval's noise input — must use the ROW count, so self.horizon becomes
        # the row count and self.pose_horizon keeps H.
        self.action_param = action_param
        self.action_frame = str(action_frame)
        self.h_mode = action_param is not None
        self.pose_horizon = int(horizon)
        self.control_hz = float(control_hz)
        if self.h_mode:
            from maniflow.dataset.lumi_place_image_dataset import (
                H_ACTION_DIM, H_ACTION_FRAMES, H_ACTION_PARAMS,
                h_action_row_k_start, h_action_rows)
            assert action_param in H_ACTION_PARAMS, f"action_param={action_param!r}"
            assert self.action_frame in H_ACTION_FRAMES, f"action_frame={action_frame!r}"
            assert action_dim == H_ACTION_DIM, (
                f"H needs action_dim {H_ACTION_DIM} ([6-DoF ee_link | dJ7]); shape_meta says "
                f"{action_dim}")
            self.action_rows = h_action_rows(action_param, self.action_frame, self.pose_horizon)
            # index k of the FIRST shipped row: 0 for delta+goal, 1 for everything else
            self.action_row_k_start = h_action_row_k_start(action_param, self.action_frame)
            # ... and whether the (unshipped) k=0 row is structurally zero. Used ONLY by the
            # action-rate penalty (the row is real at EXECUTION time even though it is not an
            # output, so the first shipped row must not jump off the anchor) and by the
            # desk-gate/label tests. Never an output, never a runtime gate.
            self.k0_row_structurally_zero = (action_param == "delta"
                                             and self.action_frame == "cam0")
            horizon = self.action_rows
            cprint(f"[H] {action_param}/{self.action_frame}: {self.action_rows} rows "
                   f"(k={self.action_row_k_start}..{self.action_row_k_start + self.action_rows - 1})"
                   f" x {action_dim} dims (poses={self.pose_horizon})", "green")
        else:
            self.action_rows = int(horizon)
            self.action_row_k_start = 0
            self.k0_row_structurally_zero = False

        # --- Lumi C2: low-dim (prev_action/task) -> AdaLN-Zero conditioning ---
        # When enabled the encoder should be configured with lowdim_as_tokens=false so the
        # same signal is not double-fed via cross-attention. Flattened layout is
        # [prev_action(To*6) | task(To*1)] — prev_action FIRST (the masking slice relies on it).
        self.lowdim_to_adaln = bool(lowdim_to_adaln)
        self.proprio_mask_p = float(proprio_mask_p)
        self.lowdim_adaln_keys = [k for k in ('prev_action', 'task') if k in obs_shape_meta]
        lowdim_cond_dim = 0
        self._pa_flat_dim = 0
        if self.lowdim_to_adaln:
            assert self.lowdim_adaln_keys, "lowdim_to_adaln=True but no prev_action/task in shape_meta"
            lowdim_cond_dim = int(n_obs_steps) * sum(
                int(np.prod(obs_shape_meta[k]['shape'])) for k in self.lowdim_adaln_keys)
            if 'prev_action' in obs_shape_meta:
                self._pa_flat_dim = int(n_obs_steps) * int(np.prod(obs_shape_meta['prev_action']['shape']))
            cprint(f"[ManiFlow C2] low-dim -> AdaLN-Zero: keys={self.lowdim_adaln_keys} "
                   f"dim={lowdim_cond_dim} proprio_mask_p={self.proprio_mask_p}", "green")

        # create ManiFlow model
        obs_feature_dim = obs_encoder.output_shape()[-1]
        input_dim = action_dim + obs_feature_dim
        global_cond_dim = None
        if obs_as_global_cond:
            input_dim = action_dim
            global_cond_dim = obs_feature_dim

     
        cprint(f"[ManiFlowTransformerPointcloudPolicy] Using DiTX model", "red")
        model = DiTX(
            input_dim=input_dim,
            output_dim=action_dim,
            horizon=horizon,
            n_obs_steps=n_obs_steps,
            cond_dim=global_cond_dim,
            visual_cond_len=visual_cond_len,
            diffusion_timestep_embed_dim=diffusion_timestep_embed_dim,
            diffusion_target_t_embed_dim=diffusion_target_t_embed_dim,
            n_layer=n_layer,
            n_head=n_head,
            n_emb=n_emb,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            block_type=block_type,
            language_conditioned=language_conditioned,
            lowdim_cond_dim=lowdim_cond_dim,
        )
        
        self.obs_encoder = obs_encoder
        self.model = model

        # v3 dense tokens: DiTX slices the positional-embedding table to the actual token
        # count, so a token count ABOVE the table capacity would be silently truncated.
        # Fail loud instead. (obs_feature_dim carries the token width; the token COUNT
        # is output_shape()[1] when the encoder is in token mode.)
        enc_out = obs_encoder.output_shape()
        if len(enc_out) == 3:
            n_tokens = int(enc_out[1])
            cap = int(visual_cond_len) * int(n_obs_steps)
            assert n_tokens <= cap, (
                f"encoder emits {n_tokens} conditioning tokens but visual_cond_len*"
                f"n_obs_steps={cap}; raise visual_cond_len (pos-embed table @ ditx.py) "
                f"to >= {int(np.ceil(n_tokens / n_obs_steps))}")
            cprint(f"[ManiFlow] dense-token conditioning: {n_tokens} tokens (cap {cap})", "green")

        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        # H: `n_action_steps` is PER-MODE and equals the shipped row count (contract §3 FINAL:
        # 15/15/15/16). The shared config carries a single scalar (16), which made
        # `predict_action`'s `[:, :n_action_steps]` a wrong-by-one no-op on the three 15-row
        # modes — harmless today only because the slice happens to be a full slice, and a trap
        # the moment anyone lowers it or reads the field as "how many rows does this model
        # ship". Every H row IS executable (deploy re-plans at k=1..2 by policy, not by tensor
        # shape), so the honest per-mode value is `action_rows`.
        self.n_action_steps = self.action_rows if self.h_mode else n_action_steps
        if self.h_mode and int(n_action_steps) != int(self.action_rows):
            cprint(f"[H] n_action_steps {int(n_action_steps)} -> {self.action_rows} "
                   f"(= action_rows for {action_param}/{self.action_frame}; config carries one "
                   f"scalar for all four modes)", "yellow")
        self.n_obs_steps = n_obs_steps
        self.obs_as_global_cond = obs_as_global_cond
        self.language_conditioned = language_conditioned
        self.kwargs = kwargs

        self.num_inference_steps = num_inference_steps
        self.flow_batch_ratio = flow_batch_ratio
        self.consistency_batch_ratio = consistency_batch_ratio
        assert flow_batch_ratio + consistency_batch_ratio == 1.0, "Sum of batch ratios should be equal to 1.0"
        self.denoise_timesteps = denoise_timesteps
        self.sample_t_mode_flow = sample_t_mode_flow
        self.sample_t_mode_consistency = sample_t_mode_consistency
        self.sample_dt_mode_consistency = sample_dt_mode_consistency
        self.sample_target_t_mode = sample_target_t_mode
        assert self.sample_target_t_mode in ["absolute", "relative"], "sample_target_t_mode must be either 'absolute' or 'relative'"

        # --- Lumi v2 precision-loss config ---
        self.endpoint_loss_weight = float(endpoint_loss_weight)
        self.endpoint_loss_type = endpoint_loss_type
        self.endpoint_t_clip = float(endpoint_t_clip)
        # per-step / per-dim weight vectors as buffers so they move with .to(device); None => uniform
        if action_step_weights is not None:
            w = torch.as_tensor(list(action_step_weights), dtype=torch.float32).view(1, -1, 1)
            assert w.shape[1] == horizon, f"action_step_weights len {w.shape[1]} != horizon {horizon}"
            self.register_buffer("action_step_weights", w)
        else:
            self.action_step_weights = None
        if action_dim_weights is not None:
            w = torch.as_tensor(list(action_dim_weights), dtype=torch.float32).view(1, 1, -1)
            assert w.shape[2] == action_dim, f"action_dim_weights len {w.shape[2]} != action_dim {action_dim}"
            self.register_buffer("action_dim_weights", w)
        else:
            self.action_dim_weights = None

        # --- Lumi v3 auxiliary goal head (regularizes the encoder toward localizing the
        # goal; TRAINING ONLY — never used by predict_action / ONNX). Predicts the goal
        # pose in the obs-time camera frame (6-D: pos + rotvec) from mean-pooled tokens. ---
        self.goal_loss_weight = float(goal_loss_weight)
        self.goal_loss_type = goal_loss_type
        if self.goal_loss_weight > 0.0:
            # v7 Rewire: learned-query ATTENTION-POOL over visual tokens (replaces mean-pool)
            # -> a spatially-localized goal readout. TRAINING ONLY; it shapes the SHARED
            # encoder to localize the goal. Never conditions the actor, never in the ONNX
            # signature (a torch-only goal_pred is exported for the val_goalhead diagnostic).
            self.goal_query = nn.Parameter(torch.randn(1, 1, obs_feature_dim) * 0.02)
            self.goal_attn = nn.MultiheadAttention(
                obs_feature_dim, num_heads=4, batch_first=True)
            self.goal_head = nn.Sequential(
                nn.LayerNorm(obs_feature_dim),
                nn.Linear(obs_feature_dim, 256), nn.GELU(), nn.Linear(256, 6))
        else:
            self.goal_query = None
            self.goal_attn = None
            self.goal_head = None

        # v7 Rewire: learned proprio MASK TOKEN (replaces masking-to-zero, since 0 is an
        # in-distribution episode-start value the net can detect and ignore). Built only when
        # proprio is routed to AdaLN. Init zeros => starts like the old zero-masking, then drifts.
        if self.lowdim_to_adaln and self._pa_flat_dim > 0:
            self.proprio_mask_token = nn.Parameter(torch.zeros(1, self._pa_flat_dim))
        else:
            self.proprio_mask_token = None

        # v7 Rewire: vision-only INVERSE-DYNAMICS aux head. From the FIRST vs LAST obs-frame
        # pooled visual tokens (NO prev_action) regress the most-recent inter-frame motion
        # (the prev_action label) -> forces the eyes to encode motion, not just static goal.
        # TRAINING ONLY.
        self.idm_loss_weight = float(idm_loss_weight)
        if self.idm_loss_weight > 0.0:
            self.idm_head = nn.Sequential(
                nn.LayerNorm(2 * obs_feature_dim),
                nn.Linear(2 * obs_feature_dim, 256), nn.GELU(), nn.Linear(256, action_dim))
        else:
            self.idm_head = None

        # --- F-series: soft-argmax keypoint head + self-predicted goal TOKEN (ON the actor path) ---
        # Reads the dense visual feature map -> N rail/tube keypoints (soft-argmax, sub-pixel); a small
        # MLP derives the place pose (6-D camera frame); a projection makes it ONE extra cross-attn
        # token appended to vis_cond. The actor's delta is generated TOWARD the self-predicted goal, so
        # the action loss backprops THROUGH the keypoints into the shared encoder -> vision becomes
        # load-bearing by construction. Keypoints supervised by projected-GT arm_kpts_uv (visibility-
        # masked); place pose by goal_cam. Self-predicted only (never the GT goal as the token).
        self.kpt_loss_weight = float(kpt_loss_weight)
        self.place_loss_weight = float(place_loss_weight)
        self.n_keypoints = int(n_keypoints)
        self.kpt_head_hires = bool(kpt_head_hires)
        self.goal_token_proj = None
        if self.kpt_loss_weight > 0.0:
            ds = int(getattr(obs_encoder, 'downsample_ratio', 32))
            img_hw = int(obs_shape_meta['head_cam']['shape'][-1])
            self._kpt_grid = max(1, img_hw // ds)                       # 224 // 32 = 7
            if self.kpt_head_hires:
                # G1: SimpleBaseline deconv 7->56 heatmaps + windowed soft-argmax + metric
                # log-z -> camera-frame 3-D points -> N POINT TOKENS (not one pooled goal
                # token; fixes the 1-token collapse). Backprojection needs cam_k_norm
                # (set from the dataset by the workspace; baked into the ckpt/ONNX).
                from maniflow.model.vision_2d.heatmap_head import HeatmapKeypointHead
                self.kpt_head = HeatmapKeypointHead(obs_feature_dim, self.n_keypoints)
                self.register_buffer('cam_k_norm_buf', torch.zeros(4))
                self.kpt_token_proj = nn.Linear(4, obs_feature_dim)     # [X,Y,Z,conf] -> token
                self.kpt_id_emb = nn.Parameter(
                    torch.randn(self.n_keypoints, obs_feature_dim) * 0.02)
                self.goal_from_kpts = nn.Sequential(                    # [pts3d,conf]*N -> place (AUX)
                    nn.LayerNorm(self.n_keypoints * 4),
                    nn.Linear(self.n_keypoints * 4, 256), nn.GELU(), nn.Linear(256, 6))
                cprint(f"[G1] hires heatmap head: {self.n_keypoints} kpts @56x56 + log-z "
                       f"-> {self.n_keypoints} 3-D point tokens "
                       f"(kpt_w={self.kpt_loss_weight} place_w={self.place_loss_weight})", "green")
            else:
                from maniflow.model.vision_2d.soft_argmax import SoftArgmaxKeypointHead
                self.kpt_head = SoftArgmaxKeypointHead(obs_feature_dim, self.n_keypoints)
                self.goal_from_kpts = nn.Sequential(                    # [u,v,conf]*N -> place pose
                    nn.LayerNorm(self.n_keypoints * 3),
                    nn.Linear(self.n_keypoints * 3, 256), nn.GELU(), nn.Linear(256, 6))
                self.goal_token_proj = nn.Sequential(                  # place pose -> conditioning token
                    nn.Linear(6, obs_feature_dim), nn.GELU(),
                    nn.Linear(obs_feature_dim, obs_feature_dim))
                cprint(f"[F-series] soft-argmax keypoint head: {self.n_keypoints} kpts on a "
                       f"{self._kpt_grid}x{self._kpt_grid} grid -> place-pose goal token "
                       f"(kpt_w={self.kpt_loss_weight} place_w={self.place_loss_weight})", "green")
        else:
            self.kpt_head = None
            self.goal_from_kpts = None

        # --- F1-depth: PointNet on the xyz point-map -> a 3-D conditioning token (DP3-native;
        # NOT depth-as-2D-channels through the encoder, which our C2 run proved the model ignores) ---
        self.pointnet_dim = int(pointnet_dim)
        # H (contract §4.2): K tokens instead of ONE global vector. pointnet_tokens must be a
        # perfect square (it is a grid x grid spatial partition of the sub-sampled point map);
        # 1 reproduces the F1-depth single-token behaviour exactly.
        self.pointnet_tokens = int(pointnet_tokens)
        self.pointnet_grid = int(round(self.pointnet_tokens ** 0.5))
        if self.pointnet_dim > 0:
            from maniflow.model.vision_2d.pointnet import PointNetEncoder
            assert self.pointnet_grid ** 2 == self.pointnet_tokens, (
                f"pointnet_tokens must be a perfect square (4x4 grid => 16); got "
                f"{self.pointnet_tokens}")
            self.pointnet = PointNetEncoder(out_dim=obs_feature_dim)
            cprint(f"[F1-depth] PointNet on xyz point-map -> {self.pointnet_tokens} "
                   f"({self.pointnet_grid}x{self.pointnet_grid} grid) x {obs_feature_dim}-d "
                   f"3D tokens", "green")
        else:
            self.pointnet = None

        # ==================================================================================
        # H-series chassis. Everything here is inert unless action_param is set (or the
        # individual flag is on), so every prior arm reproduces bit-for-bit.
        # ==================================================================================
        self.rot_integral_approx = bool(rot_integral_approx)
        self.goal_integral_weight = float(goal_integral_weight)
        self.terminal_zero_weight = float(terminal_zero_weight)
        self.goal_consistency_weight = float(goal_consistency_weight)
        self.action_rate_weight = float(action_rate_weight) if self.h_mode else 0.0
        self.goal_frame_self_p_max = float(goal_frame_self_p_max)
        self.goal_frame_anneal_epochs = max(1, int(goal_frame_anneal_epochs))
        self.phase_loss_weight = float(phase_loss_weight)
        self.done_loss_weight = float(done_loss_weight)
        self.rgb3d_pos_enc = bool(rgb3d_pos_enc)
        # epoch counter driving the scheduled-sampling ramp; a buffer so it survives
        # checkpoint/resume (a resumed run must not restart the anneal at 0).
        self.register_buffer('h_epoch', torch.zeros((), dtype=torch.long))

        # --- 3D-aware RGB tokens (contract §4.1) -----------------------------------------
        # For each 32x32 patch of the xyz point-map: valid-masked mean (X,Y,Z) + valid_frac ->
        # Linear(4 -> D) -> ADDED to the corresponding RGB token. Zero new tokens, and the
        # projection is ZERO-INIT so the model starts EXACTLY at the RGB baseline and can only
        # improve on it (the same identity-at-init discipline as AdaLN-Zero). This gives every
        # RGB patch token its own metric 3-D position instead of a learned 2-D index.
        if self.rgb3d_pos_enc:
            self.rgb3d_proj = nn.Linear(4, obs_feature_dim)
            nn.init.zeros_(self.rgb3d_proj.weight)
            nn.init.zeros_(self.rgb3d_proj.bias)
            cprint(f"[H] 3D-aware RGB tokens: xyz patch-pool -> Linear(4->{obs_feature_dim}) "
                   f"(zero-init, additive)", "green")
        else:
            self.rgb3d_proj = None

        # --- typed low-dim tokens + per-key block masking (contract §4/§4.3) --------------
        # The encoder emits one (or n) projected token(s) per low-dim key; here each key's
        # block gets a learned TYPE-ID embedding (the kpt_id_emb pattern) so cross-attention
        # can tell "this is twist" from "this is egomotion" instead of having to disentangle
        # them from a shared projection. Influence in cross-attn is not proportional to token
        # COUNT — typing buys SEPARABILITY, which is what the copycat DR needs to bite.
        # Masking substitutes a LEARNED mask token (zero is an in-distribution value the net
        # can detect and route around — the v7 lesson), zero-init so it starts as the old
        # zero-masking and then drifts.
        self.lowdim_typed_tokens = bool(self.h_mode and getattr(obs_encoder, 'token_output', False)
                                        and getattr(obs_encoder, 'lowdim_as_tokens', False))
        self.h_proprio_mask_keys = ('agent_pos', 'ego', 'twist', 'tcpcam', 'twist_hist')
        if self.lowdim_typed_tokens:
            spec = list(obs_encoder.lowdim_token_spec)
            self.lowdim_spec_keys = [k for k, _ in spec]
            self.lowdim_type_emb = nn.Parameter(
                torch.randn(len(spec), obs_feature_dim) * 0.02)
            self.lowdim_mask_token = nn.Parameter(torch.zeros(len(spec), obs_feature_dim))
            cprint(f"[H] typed low-dim tokens: {self.lowdim_spec_keys} "
                   f"(mask_p={self.proprio_mask_p} on {self.h_proprio_mask_keys})", "green")
        else:
            self.lowdim_spec_keys = []
            self.lowdim_type_emb = None
            self.lowdim_mask_token = None

        # --- AUX rail_a head (contract §1.2b) ---------------------------------------------
        # The PRIMARY place target is now `panel_goal_cam` (the panel's target pose), because
        # `rail_a` + `grasp_offset` does NOT determine the goal: measured on real episodes, in the
        # rail frame the panel goal has x fixed at module_width/2 but y FREE OVER 0.19 m and z
        # jittering 24 mm. A panel is placed flush against the previously installed panel, so the
        # along-rail DoF comes from the neighbour edge, not the rail. `rail_a_cam` stays wired as
        # an AUX landmark-grounding target only — it is a real, learnable, geometrically clean
        # signal that anchors the rail cluster, it just cannot BE the goal.
        # NOTE THE CONVENTION DIFFERENCE: `panel_goal` is camera-frame-ABSOLUTE, `rail_a` is a
        # TCP-anchored delta. Separate heads, separate normalizer fields, never interchanged.
        self.rail_aux_weight = float(rail_aux_weight)
        if self.h_mode and self.kpt_loss_weight > 0.0 and self.rail_aux_weight > 0.0:
            self.rail_aux_head = nn.Sequential(
                nn.LayerNorm(self.n_keypoints * 4),
                nn.Linear(self.n_keypoints * 4, 256), nn.GELU(), nn.Linear(256, 6))
        else:
            self.rail_aux_head = None

        # --- observability heads (contract §3.2 / §6.2) -----------------------------------
        # Tiny MLPs off the mean-pooled conditioning context. Both are FREE supervision
        # (phase_id / done ship in the zarr) that forces the context to encode "where in the
        # place am I" and "am I done", and both are exported so the deploy node can gate.
        if self.h_mode:
            self.phase_head = nn.Sequential(
                nn.LayerNorm(obs_feature_dim),
                nn.Linear(obs_feature_dim, 128), nn.GELU(), nn.Linear(128, 2))
            self.done_head = nn.Sequential(
                nn.LayerNorm(obs_feature_dim),
                nn.Linear(obs_feature_dim, 128), nn.GELU(), nn.Linear(128, 1))
        else:
            self.phase_head = None
            self.done_head = None

        # --- H structural requirements: fail loud, never degrade silently -----------------
        # The H conditioning stack is not optional plumbing — the place head IS the goal source
        # for the goal-frame arms and for deploy, and the kpt grid defines the RGB token layout
        # the 3D positional add indexes into. A config that leaves them off would train a model
        # whose ONNX contract it cannot satisfy.
        if self.h_mode:
            assert self.kpt_head is not None and self.kpt_head_hires, (
                "H requires the hires heatmap keypoint head (kpt_loss_weight>0 and "
                "kpt_head_hires=true): place_pred/kpt_uv/kpt_cam are ONNX outputs and the goal "
                "frame is derived from the place head")
            assert self.lowdim_typed_tokens, (
                "H requires typed low-dim CROSS-ATTN tokens: set obs_encoder.token_output=true "
                "and obs_encoder.lowdim_as_tokens=true (and policy.lowdim_to_adaln=false)")
            assert not self.lowdim_to_adaln, (
                "H routes all low-dim keys as cross-attn tokens (contract §4); "
                "lowdim_to_adaln=true would double-feed them into AdaLN")
            # The RGB token block is assumed to be the LEADING To*g*g tokens (the kpt head, the
            # 3D positional add and the token accounting all index it). With a second rgb-typed
            # key the encoder emits key-major blocks and that assumption breaks — depth_cam is
            # deliberately kept OUT of shape_meta precisely so this holds.
            assert len(obs_encoder.rgb_keys) == 1, (
                f"H assumes exactly ONE rgb-typed shape_meta key (head_cam); got "
                f"{obs_encoder.rgb_keys}. depth_cam must stay out of shape_meta.")

        # --- token-budget guard (updated for the H budget) --------------------------------
        # DiTX SLICES its positional-embedding table to the actual token count, so exceeding
        # the table would silently truncate conditioning. The v3 guard only counted the
        # ENCODER's tokens; H appends kpt point tokens (both frames now) + K PointNet tokens,
        # so count the real total. Contract §4 budget: 98 RGB + 14 kpt + 16 PC + 19 low-dim
        # = 147 of 256 (visual_cond_len 128 x n_obs_steps 2).
        if len(enc_out) == 3:
            n_total = int(enc_out[1])
            n_kpt = 0
            if self.kpt_head is not None:
                per_frame = self.n_keypoints if self.kpt_head_hires else 1
                n_kpt = per_frame * (int(n_obs_steps) if self.h_mode else 1)
            # legacy `_pointnet_token` emits ONE global token; only H's `_h_pointnet_tokens`
            # emits K — do not over-count on a pre-H config sitting near the cap.
            n_pc = 0 if self.pointnet is None else (self.pointnet_tokens if self.h_mode else 1)
            n_total += n_kpt + n_pc
            cap = int(visual_cond_len) * int(n_obs_steps)
            assert n_total <= cap, (
                f"conditioning needs {n_total} tokens (encoder {int(enc_out[1])} + kpt {n_kpt} "
                f"+ pointnet {n_pc}) but visual_cond_len*n_obs_steps={cap}; raise "
                f"visual_cond_len (pos-embed table @ ditx.py) to "
                f">= {int(np.ceil(n_total / n_obs_steps))}")
            cprint(f"[ManiFlow] token budget: {n_total} of {cap} "
                   f"(enc {int(enc_out[1])} + kpt {n_kpt} + pc {n_pc})", "green")

    # ================================================================= H helper methods
    def set_epoch(self, epoch: int):
        """Drives the goal-frame scheduled-sampling ramp (contract §3.3). The workspace calls
        this once per epoch; a buffer (not a python int) so resume keeps the schedule."""
        self.h_epoch.fill_(int(epoch))

    def goal_frame_self_p(self) -> float:
        """Probability of expressing this sample's action rows in the model's OWN predicted
        goal frame instead of the (noisy) GT one. Ramps 0 -> goal_frame_self_p_max linearly
        over goal_frame_anneal_epochs, then holds. Deploy is 100% self-predicted, so the ramp
        is the train/deploy bridge; going straight to 1.0 would train against a random frame
        while the place head is still garbage."""
        if not (self.h_mode and self.action_frame == "goal" and self.goal_frame_self_p_max > 0):
            return 0.0
        frac = float(self.h_epoch.item()) / float(self.goal_frame_anneal_epochs)
        return float(min(1.0, max(0.0, frac)) * self.goal_frame_self_p_max)

    def _apply_lowdim_typing(self, vis_cond, To, train_mask=False):
        """Add per-key type-ID embeddings to the encoder's low-dim token block, and (training
        only) block-mask whole proprio keys with a learned mask token.

        The encoder emits [ ...rgb tokens... | ...low-dim tokens... ], low-dim in sorted-key
        order, so the low-dim block is the TRAILING slice."""
        if not self.lowdim_typed_tokens:
            return vis_cond
        spec = self.obs_encoder.lowdim_token_spec
        counts = [(n if n is not None else int(To)) for _, n in spec]
        n_low = int(sum(counts))
        if n_low == 0:
            return vis_cond
        B = vis_cond.shape[0]
        head = vis_cond[:, :vis_cond.shape[1] - n_low]
        low = vis_cond[:, vis_cond.shape[1] - n_low:]
        parts, off = [], 0
        for i, ((key, _), c) in enumerate(zip(spec, counts)):
            blk = low[:, off:off + c] + self.lowdim_type_emb[i].view(1, 1, -1)
            if (train_mask and self.proprio_mask_p > 0.0
                    and key in self.h_proprio_mask_keys):
                sel = torch.rand(B, 1, 1, device=blk.device) < self.proprio_mask_p
                mtok = (self.lowdim_mask_token[i] + self.lowdim_type_emb[i]).view(1, 1, -1)
                blk = torch.where(sel, mtok.to(blk.dtype).expand_as(blk), blk)
            parts.append(blk)
            off += c
        return torch.cat([head] + parts, dim=1)

    def _rgb3d_tokens(self, this_nobs, To):
        """3D-aware RGB positional add (contract §4.1) -> (B, To*g*g, D) to be ADDED to the
        RGB tokens. Replays the encoder's resize+crop so the pooled patches line up with the
        RGB token grid exactly (the encoder records its crop offsets in `_last_crop`)."""
        enc = self.obs_encoder
        dp = this_nobs['depth_cam'][:, :To]                        # (B,To,4,S,S)
        B = dp.shape[0]
        x = dp.reshape(B * To, *dp.shape[2:])
        rgb_key = enc.rgb_keys[0]
        # ALL sizes come from STATIC config, never from tensor shapes: during torch.onnx.export
        # (TS tracing) `tensor.shape[-1]` yields a traced Tensor, and feeding that to avg_pool2d
        # as a kernel size fails at export time. int(x.shape[-1]) below is the sole shape read
        # and is a genuine constant for the fixed-shape deployment graph.
        out_hw = int(enc.key_shape_map[rgb_key][-1])               # 224
        if int(x.shape[-1]) != out_hw:
            # NEAREST, not bilinear: interpolating the validity channel (and across the
            # invalid/glass boundary) would invent geometry. Sub-pixel accuracy is irrelevant
            # for a 32x32 patch mean.
            x = F.interpolate(x, size=(out_hw, out_hw), mode='nearest')
        final_hw = out_hw
        if getattr(enc, '_paired_crop', False):
            ci, cj = enc._last_crop
            cs = int(enc._crop_size)
            x = x[..., ci:ci + cs, cj:cj + cs]
            final_hw = int(enc._crop_out)
            x = F.interpolate(x, size=(final_hw, final_hw), mode='nearest')
        g = (self._kpt_grid if self.kpt_head is not None
             else max(1, final_hw // int(getattr(enc, 'downsample_ratio', 32))))
        ds = int(final_hw // g)                                    # 224 // 7 = 32
        v = x[:, 3:4]                                              # (BTo,1,H,W) validity
        num = F.avg_pool2d(x[:, :3] * v, ds)                       # mean of xyz*valid
        den = F.avg_pool2d(v, ds)                                  # valid_frac in [0,1]
        feat = torch.cat([num / den.clamp(min=1e-6), den], dim=1)  # (BTo,4,g,g)
        tok = self.rgb3d_proj(feat.flatten(2).transpose(1, 2))     # (BTo,g*g,D)
        return tok.reshape(B, To * g * g, -1)

    def _h_pointnet_tokens(self, this_nobs, To):
        """xyz point-map -> masked cloud -> K PointNet tokens (contract §4.2)."""
        from maniflow.model.vision_2d.pointnet import cloud_from_pointmap
        dp = this_nobs['depth_cam'][:, To - 1]                     # (B,C,S,S) most-recent frame
        pts, valid, hw = cloud_from_pointmap(dp, stride=4, return_hw=True)
        if self.pointnet_tokens <= 1:
            return self.pointnet(pts, valid).unsqueeze(1)          # (B,1,D) legacy
        return self.pointnet.forward_tokens(pts, valid, hw, grid=self.pointnet_grid)

    def _h_context(self, vis_cond):
        """Mean-pooled conditioning context -> (phase logits (B,2), done logit (B,1))."""
        pooled = vis_cond.mean(dim=1)
        return self.phase_head(pooled), self.done_head(pooled)

    def _h_goal_integral(self, rows):
        """The goal-coincidence quantity for the `goal` frames (contract §3.2), as a (B,6)
        `[pos(3), rotvec(3)]` to be matched against `ad(P_0, G, R_g)` — the remaining transform
        at the anchor.

        `twist`+`goal` (H1): the INTEGRAL of the predicted profile. Position integrates EXACTLY
        because every row lives in one fixed frame (Σ v_k Δt == p_last − p_0); rotation
        integrates by COMPOSITION (Π exp(ω_k Δt)), which is what `compose_rotvecs` does.
        `rot_integral_approx` selects the small-angle SUM instead, as an ablation — post-yaw-skip
        total rotation is ≲9°, so the approximation error is ~1e-2 deg (documented, not relied on).

        `delta`+`goal` (H1-delta): the quantity IS row k=0. `u_0 = ad(P_0, G, R_g)` is the
        remaining transform, so the constraint reduces to pinning the first shipped row — and
        that is the POINT, not a degeneracy. `u_0` is the model's own goal estimate re-expressed
        in action space: the same self-predicted goal serves simultaneously as the output
        coordinate system and as a predicted quantity, so a wrong goal is directly penalised in
        the units the actor emits. That dual use is the flagship's structural coupling made
        explicit and measurable (and is why contract §3 FINAL ships this row while dropping
        `delta`+`cam0`'s structurally-zero one: this row is predicted and load-bearing, that one
        is a constant). `goal_consistency_weight` closes the same loop from the other side by
        tying `u_0` to `ad(P_0, Ĝ, R_g)` built from the place head.
        """
        dt = 1.0 / self.control_hz
        if self.action_param == "twist":
            pos = rows[..., 0:3].sum(dim=1) * dt
            rv = rows[..., 3:6] * dt
            if self.rot_integral_approx:
                rot = rv.sum(dim=1)
            else:
                from maniflow.common.so3_torch import compose_rotvecs
                rot = compose_rotvecs(rv)
            return torch.cat([pos, rot], dim=-1)
        return rows[:, 0, 0:6]

    def _kpt_and_goal(self, vis_cond, To, depth_z=None, cam_k=None):
        """Slice the most-recent visual frame's tokens from vis_cond (visual tokens come FIRST,
        row-major), reshape to (B,D,g,g), run the keypoint head.

        Legacy (soft-argmax): -> place pose -> ONE goal token. Returns (uv, conf, place, tok, {}).
        G1 hires: 56x56 heatmaps + windowed soft-argmax + metric log-z -> camera-frame 3-D
        points -> N point tokens (+ aux place readout). depth_z (B,2,H,W) [z_m, valid] optionally
        fuses sensor depth into the z head; cam_k (B,4) per-sample effective intrinsics when the
        GPU aug shift-jitters the frame (falls back to the cam_k_norm_buf buffer).
        Returns (uv, conf, place, goal_tokens (B,N,D), extras dict)."""
        B = vis_cond.shape[0]; D = vis_cond.shape[-1]
        g = self._kpt_grid; Lg = g * g
        vis = vis_cond[:, (To - 1) * Lg: To * Lg, :]                    # (B, Lg, D) most-recent frame
        fmap = vis.transpose(1, 2).reshape(B, D, g, g)                  # (B, D, g, g)
        if self.kpt_head_hires:
            k = cam_k if cam_k is not None else self.cam_k_norm_buf
            out = self.kpt_head(fmap, cam_k_norm=k, depth_z=depth_z)
            conf_s = torch.sigmoid(out["conf"]).unsqueeze(-1)           # (B,N,1)
            feat = torch.cat([out["pts3d"], conf_s], dim=-1)            # (B,N,4)
            place = self.goal_from_kpts(feat.reshape(B, -1))            # (B,6) AUX readout
            tokens = self.kpt_token_proj(feat) + self.kpt_id_emb.unsqueeze(0)  # (B,N,D)
            return out["uv"], out["conf"], place, tokens, out
        uv, conf = self.kpt_head(fmap)                                 # (B,N,2), (B,N)
        kp_feat = torch.cat([uv, torch.sigmoid(conf).unsqueeze(-1)], dim=-1).reshape(B, -1)
        place = self.goal_from_kpts(kp_feat)                           # (B,6) normalized place pose
        goal_token = self.goal_token_proj(place).unsqueeze(1)          # (B,1,D)
        return uv, conf, place, goal_token, {}

    def _pointnet_token(self, this_nobs, To):
        """xyz point-map (depth_cam) -> masked cloud -> PointNet -> (B,1,D) 3-D token."""
        from maniflow.model.vision_2d.pointnet import cloud_from_pointmap
        dp = this_nobs['depth_cam'][:, To - 1]                         # (B,C,S,S) most-recent frame
        pts, valid = cloud_from_pointmap(dp, stride=4)
        return self.pointnet(pts, valid).unsqueeze(1)                  # (B,1,D)

    def _kpt_depth_z(self, this_nobs, To):
        """(B,2,S,S) [z_metres, valid] from the xyz point-map for the G1 z-fusion, or None.
        xyz map channels are [X,Y,Z,valid] scaled by XYZ_SCALE_M=2.5 (dataset _encode_depth)."""
        if not self.kpt_head_hires or 'depth_cam' not in this_nobs:
            return None
        dp = this_nobs['depth_cam'][:, To - 1]                         # (B,4,S,S) most-recent frame
        if dp.shape[1] < 4:
            return None
        return torch.stack([dp[:, 2] * 2.5, dp[:, 3]], dim=1)          # (B,2,S,S)

        cprint(f"[ManiFlowTransformerImagePolicy] Initialized with parameters:", "yellow")
        cprint(f"  - horizon: {self.horizon}", "yellow")
        cprint(f"  - n_action_steps: {self.n_action_steps}", "yellow")
        cprint(f"  - n_obs_steps: {self.n_obs_steps}", "yellow")
        cprint(f"  - num_inference_steps: {self.num_inference_steps}", "yellow")
        cprint(f"  - flow_batch_ratio: {self.flow_batch_ratio}", "yellow")
        cprint(f"  - consistency_batch_ratio: {self.consistency_batch_ratio}", "yellow")
        cprint(f"  - denoise_timesteps: {self.denoise_timesteps}", "yellow")
        cprint(f"  - sample_t_mode_flow: {self.sample_t_mode_flow}", "yellow")
        cprint(f"  - sample_t_mode_consistency: {self.sample_t_mode_consistency}", "yellow")
        cprint(f"  - sample_dt_mode_consistency: {self.sample_dt_mode_consistency}", "yellow")
        cprint(f"  - sample_target_t_mode: {self.sample_target_t_mode}", "yellow")
        cprint(f"  - endpoint_loss_weight: {self.endpoint_loss_weight} ({self.endpoint_loss_type})", "yellow")
        cprint(f"  - action_step_weights: {'set' if self.action_step_weights is not None else 'uniform'}", "yellow")
        cprint(f"  - action_dim_weights: {'set' if self.action_dim_weights is not None else 'uniform'}", "yellow")
        cprint(f"  - goal_loss_weight: {self.goal_loss_weight} ({self.goal_loss_type})", "yellow")

        print_params(self)

    # ------------------------------------------------------------------ H keypoint tokens
    def _kpt_depth_z_h(self, this_nobs, To):
        """(B*To,2,S,S) [z_metres, valid] for ALL obs frames (the H kpt head runs on both)."""
        if 'depth_cam' not in this_nobs:
            return None
        dp = this_nobs['depth_cam'][:, :To]                          # (B,To,4,S,S)
        if dp.shape[2] < 4:
            return None
        z = dp[:, :, 2] * 2.5                                        # XYZ_SCALE_M
        v = dp[:, :, 3]
        return torch.stack([z, v], dim=2).reshape(-1, 2, *dp.shape[-2:])

    def _kpt_and_goal_h(self, vis_cond, To, depth_z=None, cam_k=None):
        """H: run the heatmap keypoint head on BOTH obs frames -> 2*N point tokens.

        Why both frames (plan §2.2): two 3-D fixes of the same STATIC landmarks one step apart
        are implicit camera ego-motion expressed in landmark space — the cheapest "two-frame 3D"
        signal available, and the supervision already exists per frame (arm_kpts_* is per-frame).
        The last-frame-only version threw half the free supervision away.

        The PLACE readout is taken from the MOST RECENT frame only: it is the deploy servo
        target, and deploy always servos on the newest observation. `place` is the normalized
        `panel_goal_cam` (§1.2b — the panel's target pose, camera-frame-absolute); `rail_aux` is
        the normalized `rail_a_cam` (a TCP-anchored delta) and is TRAINING-ONLY.

        WHICH CAMERA FRAME (contract §9.3): this head sees IMAGE FEATURES ONLY — there is no
        proprio path into it — so its output can only be expressed in the frame of the pixels it
        was given, i.e. the camera pose when `head_cam[:, To-1]` was captured. The dataset indexes
        `panel_goal_cam`/`rail_a_cam`/`arm_kpts_*` at exactly those image frames for the same
        reason. With the §5 timing augmentation that frame is 0 or 1 control frames before the
        action anchor; at eval/export/deploy it is the anchor itself.

        KEYPOINTS 5-6 ARE LOAD-BEARING, not a minor cue: they are the neighbour-edge points and
        the ONLY visual source of the along-rail DoF (§1.2b — the panel goal's y is free by
        0.19 m relative to the rail). Do not down-weight them; when `has_neighbor` is false the
        converter sets their visibility to 0, so their presence logits double as the model's
        "is the along-rail DoF observable at all?" signal (and `task` carries the flag on input).
        Returns (uv, conf, place (B,6), tokens (B,To*N,D), extras); extras['rail_aux'] when built."""
        assert self.kpt_head_hires, "H requires the hires heatmap head (kpt_head_hires=true)"
        B, D = vis_cond.shape[0], vis_cond.shape[-1]
        g = self._kpt_grid
        Lg = g * g
        N = self.n_keypoints
        vis = vis_cond[:, :To * Lg, :]                                # visual tokens come FIRST
        fmap = vis.reshape(B * To, Lg, D).transpose(1, 2).reshape(B * To, D, g, g)
        k = cam_k if cam_k is not None else self.cam_k_norm_buf
        if k is not None and k.dim() == 2 and k.shape[0] == B:
            k = k.repeat_interleave(To, dim=0)                        # per-sample -> per-frame
        out = self.kpt_head(fmap, cam_k_norm=k, depth_z=depth_z)
        conf_s = torch.sigmoid(out["conf"]).unsqueeze(-1)             # (B*To,N,1)
        feat = torch.cat([out["pts3d"], conf_s], dim=-1)              # (B*To,N,4)
        tokens = self.kpt_token_proj(feat) + self.kpt_id_emb.unsqueeze(0)
        tokens = tokens.reshape(B, To * N, D)
        last = feat.reshape(B, To, N, 4)[:, To - 1].reshape(B, -1)
        place = self.goal_from_kpts(last)                             # panel_goal_cam (PRIMARY)
        if self.rail_aux_head is not None:
            out["rail_aux"] = self.rail_aux_head(last)                # rail_a_cam (AUX, no export)
        uv = out["uv"].reshape(B, To, N, 2)
        conf = out["conf"].reshape(B, To, N)
        return uv, conf, place, tokens, out

    def _h_encode(self, nobs, To, train_mask=False, cam_k=None):
        """The whole H conditioning stack, in ONE place so training / predict_action / the ONNX
        wrapper cannot drift apart (the F1-depth bug was exactly that drift).

        Returns (vis_cond, aux) where aux carries the perception outputs the losses and the
        exported graph need: place (`panel_goal_cam`, normalized — the PRIMARY target and the
        deploy goal source), kpt uv/conf, kpt 3-D, phase, done (+ the training-only rail aux)."""
        B = next(iter(nobs.values())).shape[0]
        this_nobs = {k: v[:, :To] for k, v in nobs.items()}
        vis_cond = self.obs_encoder(this_nobs).reshape(B, -1, self.obs_feature_dim)
        # (1) 3D-aware positional ADD on the RGB tokens (zero new tokens, identity at init)
        if self.rgb3d_proj is not None and 'depth_cam' in this_nobs:
            n_rgb = self._kpt_grid * self._kpt_grid * To
            add = self._rgb3d_tokens(this_nobs, To)
            vis_cond = torch.cat([vis_cond[:, :n_rgb] + add, vis_cond[:, n_rgb:]], dim=1)
        # (2) typed low-dim tokens + per-key block masking (train only)
        vis_cond = self._apply_lowdim_typing(vis_cond, To, train_mask=train_mask)
        # (3) keypoint point tokens from BOTH obs frames
        uv, conf, place, kpt_tok, extras = self._kpt_and_goal_h(
            vis_cond, To, depth_z=self._kpt_depth_z_h(this_nobs, To), cam_k=cam_k)
        vis_cond = torch.cat([vis_cond, kpt_tok], dim=1)
        # (4) K PointNet tokens
        if self.pointnet is not None and 'depth_cam' in this_nobs:
            vis_cond = torch.cat([vis_cond, self._h_pointnet_tokens(this_nobs, To)], dim=1)
        phase, done = self._h_context(vis_cond)
        aux = {'place': place, 'kpt_uv': uv, 'kpt_conf': conf,
               'kpt_extras': extras, 'phase': phase, 'done': done}
        return vis_cond, aux

    # ========= inference  ============
    def _build_lowdim_cond(self, nobs, batch_size, To, device, train_mask=False):
        """Lumi C2: flatten normalized low-dim obs (prev_action/task over the To obs frames)
        into the DiTX AdaLN conditioning vector. Layout [prev_action | task] (prev first —
        the masking slice depends on it). train_mask applies per-sample proprio dropout
        (zero the prev_action slab with prob proprio_mask_p) so the policy cannot lean on
        proprioception alone and vision stays load-bearing for goal localization."""
        if not self.lowdim_to_adaln:
            return None
        parts = [nobs[k][:, :To].reshape(batch_size, -1).to(device)
                 for k in self.lowdim_adaln_keys]
        cond = torch.cat(parts, dim=-1)
        if train_mask and self.proprio_mask_p > 0.0 and self._pa_flat_dim > 0:
            masked = (torch.rand(batch_size, 1, device=cond.device) < self.proprio_mask_p)
            pa = cond[:, :self._pa_flat_dim]
            if self.proprio_mask_token is not None:
                tok = self.proprio_mask_token.to(cond.dtype).expand(batch_size, -1)
                pa = torch.where(masked, tok, pa)                # learned mask token
            else:
                pa = pa * (~masked).to(cond.dtype)               # fallback: mask to zero
            cond = torch.cat([pa, cond[:, self._pa_flat_dim:]], dim=-1)
        return cond

    def _predict_goal(self, vis_cond):
        """Attention-pool over the visual tokens -> normalized 6-D goal in the camera frame.
        Used by the training-only goal aux loss AND the torch-only goal_pred diagnostic; it
        NEVER conditions the actor and is NEVER in the ONNX signature."""
        q = self.goal_query.expand(vis_cond.shape[0], -1, -1)    # (B,1,Do) learned query
        pooled, _ = self.goal_attn(q, vis_cond, vis_cond)        # (B,1,Do)
        return self.goal_head(pooled.squeeze(1))                 # (B,6)

    def conditional_sample(self,
            condition_data,
            vis_cond=None,
            lang_cond=None,
            **kwargs
            ):
        
        noise = torch.randn(
            size=condition_data.shape, 
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=None)
        
        ode_traj = self.sample_ode(
            x0 = noise, 
            N = self.num_inference_steps,
            vis_cond=vis_cond,
            lang_cond=lang_cond,
           **kwargs)
        
        return ode_traj[-1] # sample ode returns the whole traj, return the last one
    


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict: must include "obs" key
        result: must include "action" key
        """
        # normalize input
        nobs = self.normalizer.normalize(obs_dict)
        
        value = next(iter(nobs.values()))
        B, To = value.shape[:2]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = self.device
        dtype = self.dtype

        # handle different ways of passing observation
        vis_cond = None
        lang_cond = None

        if self.language_conditioned:
            # assume nobs has 'task_name' key for language condition
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        # H: ONE shared conditioning path (training / inference / ONNX all call _h_encode) so
        # the F1-depth class of bug — a token silently present at train and absent at deploy —
        # cannot recur.
        if self.h_mode:
            nobs = dict_apply(nobs, lambda x: x.to(device))
            vis_cond, aux = self._h_encode(nobs, To, train_mask=False)
            lowdim_cond = self._build_lowdim_cond(nobs, B, To, device)
            cond_data = torch.zeros(size=(B, self.action_rows, Da), device=device, dtype=dtype)
            nsample = self.conditional_sample(
                cond_data, vis_cond=vis_cond, lang_cond=None,
                lowdim_cond=lowdim_cond, **self.kwargs)
            # Every shipped row is PREDICTED (contract §3 FINAL): nothing is prepended. Row j of
            # the tensor is chunk step k = action_row_k_start + j, which the consumer reads from
            # metadata_props rather than assuming.
            action_pred = self.normalizer['action'].unnormalize(nsample[..., :Da])
            # H rows start AT the obs time, so the legacy `To-1` slice does not apply.
            return {
                'action': action_pred[:, :self.n_action_steps],
                'action_pred': action_pred,
                # `panel_goal_cam`: the panel's target pose in the camera frame. Deploy composes
                # T_tcp_goal = T_panel_goal @ inv(T_tcp_panel)  (contract §1.2a, invert=True).
                # FRAME (contract §9.3): the camera pose at the capture time of the MOST RECENT
                # INPUT IMAGE (`head_cam[:, To-1]`), not "now" and not the action anchor. At
                # deploy/eval/export the newest image IS the anchor (image_lag_frames = 0), so
                # this is the current camera frame; under training's timing augmentation it can
                # be one control frame back, which is precisely why the labels are indexed there
                # too. A consumer that lifts place_pred with a camera pose MUST use the pose at
                # that image's capture time.
                'place_pred': self.normalizer['panel_goal'].unnormalize(aux['place']),
                'kpt_uv': torch.cat([aux['kpt_uv'][:, To - 1],
                                     torch.sigmoid(aux['kpt_conf'][:, To - 1]).unsqueeze(-1)],
                                    dim=-1),
                'kpt_cam': aux['kpt_extras']['pts3d'].reshape(
                    B, To, self.n_keypoints, 3)[:, To - 1],
                'phase': aux['phase'],
                'done': aux['done'],
            }

        # condition through visual feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].to(device))
        nobs_features = self.obs_encoder(this_nobs).to(device)
        vis_cond = nobs_features.reshape(B, -1, Do) # B, self.n_obs_steps*L, Do
        if self.kpt_head is not None:                # F-series: append self-predicted goal token(s)
            _, _, _, _goal_tok, _ = self._kpt_and_goal(
                vis_cond, To, depth_z=self._kpt_depth_z(this_nobs, To))
            vis_cond = torch.cat([vis_cond, _goal_tok], dim=1)
        if self.pointnet is not None and 'depth_cam' in this_nobs:   # F1-depth 3-D token
            vis_cond = torch.cat([vis_cond, self._pointnet_token(this_nobs, To)], dim=1)
        # Lumi C2: compact control conditioning for AdaLN (no masking at inference)
        lowdim_cond = self._build_lowdim_cond(nobs, B, To, device)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)

        # run sampling
        nsample = self.conditional_sample(
            cond_data,
            vis_cond=vis_cond,
            lang_cond=lang_cond,
            lowdim_cond=lowdim_cond,
            **self.kwargs)
        
        # unnormalize prediction
        naction_pred = nsample[...,:Da]
        action_pred = self.normalizer['action'].unnormalize(naction_pred)

        # get action
        start = To - 1
        end = start + self.n_action_steps
        action = action_pred[:,start:end]
        
        # get prediction
        result = {
            'action': action,
            'action_pred': action_pred,
        }
        # Lumi v7: image-derived goal estimate, present iff the aux goal head was trained
        # (goal_loss_weight>0). Torch-only — the ONNX wrapper reimplements inference inline
        # and returns just (action, action_pred), so this never touches the export/gate. Used
        # for the val_goalhead_pos_mm metric now and a deploy-time servo target later.
        if getattr(self, 'goal_head', None) is not None and 'goal_cam' in self.normalizer.params_dict:
            result['goal_pred'] = self.normalizer['goal_cam'].unnormalize(
                self._predict_goal(vis_cond))

        return result

    # ========= training  ============
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def get_optimizer(
            self, 
            lr: float,
            weight_decay: float,
            obs_encoder_lr: float = None,
            obs_encoder_weight_decay: float = None,
            betas: Tuple[float, float] = (0.9, 0.95)
        ) -> torch.optim.Optimizer:
        optim_groups = self.model.get_optim_groups(
            weight_decay=weight_decay)
        
        backbone_params = list()
        other_obs_params = list()
        if obs_encoder_lr is not None:
            cprint(f"[ManiFlowTransformerImagePolicy] Use different lr for obs_encoder: {obs_encoder_lr}", "yellow")
            for key, value in self.obs_encoder.named_parameters():
                if key.startswith('key_model_map'):
                    backbone_params.append(value)
                else:
                    other_obs_params.append(value)
            optim_groups.append({
                "params": backbone_params,
                "weight_decay": obs_encoder_weight_decay,
                "lr": obs_encoder_lr # for fine tuning
            })
            optim_groups.append({
                "params": other_obs_params,
                "weight_decay": obs_encoder_weight_decay
            })
        optimizer = torch.optim.AdamW(
            optim_groups, lr=lr, betas=betas
        )
        return optimizer
    
    def sample_t(self, batch_size, mode="uniform"):
        """
        Sample t for flow matching or consistency training.
        """
        if mode == "uniform":
            t = torch.rand((batch_size,), device=self.device)
        elif mode == "lognorm":
            t = sample_logit_normal(batch_size, m=self.lognorm_m, s=self.lognorm_s, device=self.device)
        elif mode == "mode":
            t = sample_mode(batch_size, s=self.mode_s, device=self.device)
        elif mode == "cosmap":
            t = sample_cosmap(batch_size, device=self.device)
        elif mode == "beta":
            t = sample_beta(batch_size, device=self.device)
        elif mode == "discrete":
            t = torch.randint(low=0, high=self.denoise_timesteps, size=(batch_size,)).float()
            t = t / self.denoise_timesteps
        else:
            raise ValueError(f" Unsupported sample_t_mode {mode}. Choose from 'uniform', 'lognorm', 'mode', 'cosmap', 'beta', 'discrete'.")
        return t

    def sample_dt(self, batch_size, sample_dt_mode="uniform"):
        """
        Sample dt for consistency training.
        """
        if sample_dt_mode == "uniform":
            dt = torch.rand((batch_size,), device=self.device)
        else:
            raise ValueError(f"Unsupported sample_dt_mode {sample_dt_mode}")
        
        return dt
    
    def linear_interpolate(self, noise, target, timestep, epsilon=0.0):
        """
        Linear interpolation between noise and target data with optional noise preservation.
        
        Args:
            noise (Tensor): Initial noise at t=0
            target (Tensor): Target data point at t=1  
            timestep (float): Interpolation parameter in [0, 1]
                            t=0 returns pure noise, t=1 returns target + epsilon*noise
            epsilon (float): Noise preservation factor. Controls minimum noise retained.
                            Default 0.0 for standard linear interpolation.
                            
        Returns:
            Tensor: Interpolated data point at given timestep
            
        Examples:
            >>> # Standard linear interpolation (epsilon=0)
            >>> result = linear_interpolate(noise, data, 0.5)  # 50% noise + 50% data
            
            >>> # With noise preservation (epsilon=0.01) 
            >>> result = linear_interpolate(noise, data, 1.0, epsilon=0.01)  # data + 1% noise
        """
        # Calculate noise coefficient with epsilon adjustment
        noise_coeff = 1.0 - (1.0 - epsilon) * timestep
        
        # Linear combination: preserved_noise + scaled_target
        interpolated_data_point = noise_coeff * noise + timestep * target
        
        return interpolated_data_point

    
    def get_flow_velocity(self, actions, **model_kwargs):
        """
        Get flow velocity targets for training.
        Flow training is used to train the model to predict instantaneous velocity given a timestep.
        """
        target_dict = {}
        
        # get visual and language conditions
        vis_cond = model_kwargs.get('vis_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        flow_batchsize = actions.shape[0]
        device = actions.device
        
        # sample t and dt for flow
        # dt is zero for flow, as we aim to predict the instantaneous velocity at t
        t_flow = self.sample_t(flow_batchsize, mode=self.sample_t_mode_flow).to(device)
        t_flow = t_flow.view(-1, 1, 1)
        dt_flow = torch.zeros((flow_batchsize,), device=device)
        
        # get target timestep
        # target_t_flow is the target timestep for the flow step
        # it can be either absolute or relative to t_flow
        # if absolute, it is t_flow + dt_flow
        # if relative, it is just dt_flow
        if self.sample_target_t_mode == "absolute":
            target_t_flow = t_flow.squeeze() + dt_flow
        elif self.sample_target_t_mode == "relative":
            target_t_flow = dt_flow
        
        # compute interpolated data points at t and predict flow velocity
        x_0_flow = torch.randn_like(actions, device=device) 
        x_1_flow = actions.to(device) 
        x_t_flow = self.linear_interpolate(x_0_flow, x_1_flow, t_flow, epsilon=0.0)
        v_t_flow = x_1_flow - x_0_flow

        target_dict['x_t'] = x_t_flow
        target_dict['t'] = t_flow
        target_dict['target_t'] = target_t_flow
        target_dict['v_target'] = v_t_flow
        target_dict['vis_cond'] = vis_cond
        target_dict['lang_cond'] = lang_cond

        return target_dict
    
    def get_consistency_velocity(self, actions, **model_kwargs):
        """
        Get consistency velocity targets for training.
        Consistency training is used to train the model to be consistent across different timesteps.
        """
        target_dict = {}
        
        # get visual and language conditions
        vis_cond = model_kwargs.get('vis_cond', None)
        lang_cond = model_kwargs.get('lang_cond', None)
        lowdim_cond = model_kwargs.get('lowdim_cond', None)  # Lumi C2 AdaLN conditioning
        ema_model = model_kwargs.get('ema_model', None)
        consistency_batchsize = actions.shape[0]
        device = actions.device

        # sample t and dt for consistency training
        t_ct = self.sample_t(consistency_batchsize, mode=self.sample_t_mode_consistency).to(device)
        t_ct = t_ct.view(-1, 1, 1)
        delta_t1 = self.sample_dt(consistency_batchsize, sample_dt_mode=self.sample_dt_mode_consistency).to(device)
        # delta_t2 = self.sample_dt(consistency_batchsize, sample_dt_mode=self.sample_dt_mode_consistency).to(device)
        delta_t2 = delta_t1.clone() # use the same delta_t or resample a new one

        # compute next timestep
        t_next = t_ct.squeeze() + delta_t1
        t_next = torch.clamp(t_next, max=1.0) # clip t to ensure it does not exceed 1.0
        t_next = t_next.view(-1, 1, 1)
        
        # compute target timestep
        # target_t_next is the target timestep for the next step
        # it can be either absolute or relative to t_next
        # if absolute, it is t_next + delta_t2
        # if relative, it is just delta_t2
        if self.sample_target_t_mode == "absolute":
            target_t_next = t_next.squeeze() + delta_t2
        elif self.sample_target_t_mode == "relative":
            target_t_next = delta_t2

        # compute interpolated data points at timestep t and t_next
        x0_ct = torch.randn_like(actions, device=device) 
        x1_ct = actions.to(device) 
        x_t_ct = self.linear_interpolate(x0_ct, x1_ct, t_ct, epsilon=0.0)
        x_t_next = self.linear_interpolate(x0_ct, x1_ct, t_next, epsilon=0.0)

        # predict the average velocity from t_next toward next target (t_next + delta_t2)
        with torch.no_grad():
            v_avg_to_next_target = ema_model.model(
                sample=x_t_next,
                timestep=t_next.squeeze(),
                target_t=target_t_next.squeeze(),
                vis_cond=vis_cond[-consistency_batchsize:],
                lang_cond=lang_cond[-consistency_batchsize:] if lang_cond is not None else None,
                lowdim_cond=lowdim_cond[-consistency_batchsize:] if lowdim_cond is not None else None,
            )
        # predict the target data point using the average velocity
        pred_x1_ct = x_t_next + (1 - t_next) * v_avg_to_next_target
        # estimate the velocity at t by using the predicted endpoint
        v_ct = (pred_x1_ct - x_t_ct) / (1 - t_ct)

        # target_t_ct is the target timestep for the current timestep t
        target_t_ct = delta_t1 if self.sample_target_t_mode == "relative" else t_next.squeeze()
        
        target_dict['x_t'] = x_t_ct
        target_dict['t'] = t_ct
        target_dict['target_t'] = target_t_ct
        target_dict['v_target'] = v_ct

        return target_dict
    
    @torch.no_grad()
    def sample_ode(self, x0=None, N=None, **model_kwargs):
        ### NOTE: Use Euler method to sample from the learned flow
        if N is None:
            N = self.num_inference_steps
        dt = 1./N
        traj = [] # to store the trajectory
        x = x0.detach().clone()
        batchsize = x.shape[0]

        t = torch.arange(0, N, device=x0.device, dtype=x0.dtype) / N
        traj.append(x.detach().clone())

        for i in range(N):
            ti = torch.ones((batchsize,), device=self.device) * t[i]
            if self.sample_target_t_mode == "absolute":
                target_t = ti + dt
            elif self.sample_target_t_mode == "relative":
                target_t = dt
            pred = self.model(x, ti, target_t=target_t, **model_kwargs)
            x = x.detach().clone() + pred * dt
            traj.append(x.detach().clone())

        return traj

    def _weighted_step_mean(self, per_elem):
        """Reduce a (B, T, Da) per-element loss to (B,) applying optional per-chunk-step and
        per-action-dim weights. When no weights are set this equals reduce(.,'b ... -> b','mean')
        i.e. the stock ManiFlow behaviour."""
        w = None
        if self.action_step_weights is not None:
            w = self.action_step_weights.to(per_elem.dtype)                       # (1,T,1)
        if self.action_dim_weights is not None:
            dw = self.action_dim_weights.to(per_elem.dtype)                       # (1,1,Da)
            w = dw if w is None else (w * dw)
        if w is None:
            return reduce(per_elem, 'b ... -> b (...)', 'mean').mean(dim=1)
        # weighted mean over (T, Da): sum(w*loss) / sum(w)
        num = (per_elem * w).sum(dim=(1, 2))
        den = w.expand_as(per_elem).sum(dim=(1, 2))
        return num / den

    def _h_reframe(self, batch, h_aux):
        """H: put the action rows (and the goal-coincidence target) in the OUTPUT frame that
        this sample will actually be supervised in — contract §3.3 scheduled sampling.

        Changing ONLY the expression frame of goal-frame rows is EXACTLY a single fixed
        rotation of both triplets of every row:
            R_g_hat = dR @ R_g  =>  v' = M v,  w' = M w   with  M = R_g_hat.T @ R_g
        and in the anchor-camera frame ("C0", where R_g = R_cv0 @ F) the R_cv0 cancels:
            M = F_hat.T @ F.
        That is why the dataset ships F (and the constant map `panel_goal_cam -> F_hat`) instead
        of the raw world poses: the whole re-framing is three small matmuls, no pose math in torch.
        The map composes from the PANEL GOAL, not `rail_a` — `rail_a` + `grasp_offset` leaves the
        along-rail DoF free by 0.19 m (contract §1.2b), so a rail-derived frame would be wrong by
        that much whenever the model's y estimate moved.

        FRAME OF `place` (contract §9.3): the place head reads image features only, so its output
        — and its label — live in the camera frame of the MOST RECENT USED IMAGE, which the §5
        timing augmentation puts `image_lag_frames` control frames before the anchor. Everything
        here is expressed in the ANCHOR camera frame C0, so the per-sample constant
        `h_gc_A = R_cv0.T @ R_cimg` carries the head's output across that one-step camera
        egomotion. `A` is the identity whenever lag == 0, which is every eval/export/deploy
        sample — so the deploy-side composition is unchanged by this.

        The self-predicted frame is DETACHED (no second-order gradient through the frame
        construction); the goal-consistency target deliberately is NOT (that loss exists to push
        gradient into the place head)."""
        from maniflow.common.so3_torch import rotmat_to_rotvec, rotvec_to_rotmat
        dev = self.device
        action = batch['action'].to(dev).float()
        out = {'action': action,
               'remaining': batch['h_goal_remaining'].to(dev).float(),
               'inreach': batch['h_goal_inreach'].to(dev).float(),
               'gc_target': None, 'self_frame_frac': 0.0}
        if self.action_frame != "goal":
            return out
        B = action.shape[0]
        # `panel_goal_cam` in RAW units -> Y = exp(panel-goal rotvec) is the panel goal's
        # orientation in the IMAGE camera frame; A brings it into C0 (see the docstring), and
        # F_hat = A @ Y @ (the C0 right-multiplier realising "T_panel_goal @ inv(T_tcp_panel)",
        # precomputed by `panel_to_ee_goal_maps`).
        place_raw = self.normalizer['panel_goal'].unnormalize(h_aux['place'])
        X = rotvec_to_rotmat(place_raw[..., 3:6])
        A = batch['h_gc_A'].to(dev).float()
        F_gt = batch['h_goal_frame'].to(dev).float()
        F_hat = A @ X @ batch['h_gc_Rframe'].to(dev).float()
        F_used = F_gt
        p = self.goal_frame_self_p()
        if self.training and p > 0.0:
            sel = torch.rand(B, 1, 1, device=dev) < p
            eye = torch.eye(3, device=dev, dtype=action.dtype).expand(B, 3, 3)
            M = torch.where(sel, F_hat.detach().transpose(1, 2) @ F_gt, eye)
            out['action'] = torch.cat([
                torch.einsum('bij,bkj->bki', M, action[..., 0:3]),
                torch.einsum('bij,bkj->bki', M, action[..., 3:6]),
                action[..., 6:7]], dim=-1)
            rem = out['remaining']
            out['remaining'] = torch.cat([
                torch.einsum('bij,bj->bi', M, rem[:, 0:3]),
                torch.einsum('bij,bj->bi', M, rem[:, 3:6])], dim=-1)
            F_used = torch.where(sel, F_hat.detach(), F_gt)
            out['self_frame_frac'] = float(sel.float().mean().item())
        if self.action_param == "delta" and self.goal_consistency_weight > 0.0:
            # ad(P_0, G_hat, R_g): the remaining transform to the model's OWN goal. P_0 sits at
            # the anchor so its C0 position is 0 by construction (the dataset anchors p at p_ee0).
            # `h_gc_toff` re-anchors from the CAMERA (where panel_goal_cam lives) to p_ee0.
            p_G = torch.einsum(
                'bij,bj->bi', A,
                place_raw[..., 0:3] + torch.einsum(
                    'bij,bj->bi', X, batch['h_gc_tvec'].to(dev).float())) \
                + batch['h_gc_toff'].to(dev).float()
            R_G = A @ X @ batch['h_gc_Ree'].to(dev).float()
            R0 = batch['h_ee0_rot'].to(dev).float()
            Ft = F_used.transpose(1, 2)
            out['gc_target'] = torch.cat([
                torch.einsum('bij,bj->bi', Ft, p_G),
                rotmat_to_rotvec(Ft @ (R_G @ R0.transpose(1, 2)) @ F_used)], dim=-1)
        return out

    def compute_loss(self, batch, ema_model=None, **kwargs):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])

        # handle different ways of passing observation
        local_cond = None
        vis_cond = None
        lang_cond = None
        ema_model = ema_model
        f_uv = f_conf = f_place = None
        f_extras = {}
        h_aux = h_ref = None

        if self.language_conditioned:
            # we assume language condition is passed as 'task_name'
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        if self.h_mode:
            # H: the ENCODER must run BEFORE the action normalization, because the goal-frame
            # scheduled sampling re-expresses the action rows in the model's own predicted goal
            # frame — which needs the place head, which needs the encoder. Legacy ordering is
            # untouched below for every prior arm.
            nobs_dev = dict_apply(nobs, lambda x: x.to(self.device))
            vis_cond, h_aux = self._h_encode(
                nobs_dev, self.n_obs_steps, train_mask=self.training,
                cam_k=batch.get('cam_k_eff', None))
            h_ref = self._h_reframe(batch, h_aux)
            nactions = self.normalizer['action'].normalize(h_ref['action']).to(self.device)
            batch_size = nactions.shape[0]
        else:
            nactions = self.normalizer['action'].normalize(batch['action']).to(self.device)
            batch_size = nactions.shape[0]
            # reshape B, T, ... to B*T
            this_nobs = dict_apply(nobs,
                lambda x: x[:,:self.n_obs_steps,...].to(self.device))
            nobs_features = self.obs_encoder(this_nobs)
            vis_cond = nobs_features.reshape(batch_size, -1, self.obs_feature_dim)
            # F-series: self-predicted goal token appended to the visual conditioning (actor path).
            # Gradient from the action loss flows through the goal token -> keypoints -> encoder.
            if self.kpt_head is not None:
                f_uv, f_conf, f_place, _goal_tok, f_extras = self._kpt_and_goal(
                    vis_cond, self.n_obs_steps,
                    depth_z=self._kpt_depth_z(this_nobs, self.n_obs_steps),
                    cam_k=batch.get('cam_k_eff', None))
                vis_cond = torch.cat([vis_cond, _goal_tok], dim=1)
            if self.pointnet is not None and 'depth_cam' in this_nobs:   # F1-depth 3-D token
                vis_cond = torch.cat([vis_cond, self._pointnet_token(this_nobs, self.n_obs_steps)], dim=1)

        horizon = nactions.shape[1]
        assert horizon == self.action_rows, (
            f"action tensor has {horizon} rows but the policy predicts {self.action_rows} "
            f"(action_param={self.action_param} action_frame={self.action_frame}) — the dataset "
            f"and policy modes disagree")
        trajectory = nactions
        cond_data = trajectory
        # Lumi C2: AdaLN control conditioning; proprio masking active in training only
        lowdim_cond = self._build_lowdim_cond(
            nobs, batch_size, self.n_obs_steps, self.device, train_mask=self.training)

        """Get flow and consistency targets"""
        flow_batchsize = int(batch_size * self.flow_batch_ratio)
        consistency_batchsize = int(batch_size * self.consistency_batch_ratio)


        # Get flow targets
        flow_target_dict = self.get_flow_velocity(nactions[:flow_batchsize],
                                                    vis_cond=vis_cond[:flow_batchsize],
                                                    lang_cond=lang_cond[:flow_batchsize] if lang_cond is not None else None)
        v_flow_pred = self.model(
            sample=flow_target_dict['x_t'],
            timestep=flow_target_dict['t'].squeeze(),
            target_t=flow_target_dict['target_t'].squeeze(),
            vis_cond=vis_cond[:flow_batchsize],
            lang_cond=flow_target_dict['lang_cond'][:flow_batchsize] if lang_cond is not None else None,
            lowdim_cond=lowdim_cond[:flow_batchsize] if lowdim_cond is not None else None)
        v_flow_pred_magnitude = torch.sqrt(torch.mean(v_flow_pred ** 2)).item()

        # Get consistency targets
        consistency_target_dict = self.get_consistency_velocity(nactions[flow_batchsize:flow_batchsize+consistency_batchsize],
                                                                        vis_cond=vis_cond[flow_batchsize:flow_batchsize+consistency_batchsize],
                                                                        lang_cond=lang_cond[flow_batchsize:flow_batchsize+consistency_batchsize] if lang_cond is not None else None,
                                                                        lowdim_cond=lowdim_cond[flow_batchsize:flow_batchsize+consistency_batchsize] if lowdim_cond is not None else None,
                                                                        ema_model=ema_model
                                                                        )
        v_ct_pred = self.model(
            sample=consistency_target_dict['x_t'],
            timestep=consistency_target_dict['t'].squeeze(),
            target_t=consistency_target_dict['target_t'].squeeze(),
            vis_cond=vis_cond[flow_batchsize:flow_batchsize+consistency_batchsize],
            lang_cond=lang_cond[flow_batchsize:flow_batchsize+consistency_batchsize] if lang_cond is not None else None,
            lowdim_cond=lowdim_cond[flow_batchsize:flow_batchsize+consistency_batchsize] if lowdim_cond is not None else None,
            )
        v_ct_pred_magnitude = torch.sqrt(torch.mean(v_ct_pred ** 2)).item()

        """Compute losses"""
        loss = 0.

        # compute flow loss (per-element MSE, then optional step/dim-weighted reduction)
        v_flow_target = flow_target_dict['v_target']
        loss_flow_elem = F.mse_loss(v_flow_pred, v_flow_target, reduction='none')
        loss_flow_b = self._weighted_step_mean(loss_flow_elem)                    # (B,)
        loss += loss_flow_b.mean()
        loss_flow = loss_flow_b.mean().item()

        # compute consistency training loss (same weighting; CT branch otherwise untouched)
        v_ct_target = consistency_target_dict['v_target']
        loss_ct_elem = F.mse_loss(v_ct_pred, v_ct_target, reduction='none')
        loss_ct_b = self._weighted_step_mean(loss_ct_elem)                        # (B,)
        loss += loss_ct_b.mean()
        loss_ct = loss_ct_b.mean().item()

        # --- Lumi v2: endpoint (x1-reconstruction) aux loss, FLOW BRANCH ONLY ---
        # For the linear interpolant x_t = (1-t)*x0 + t*x1 with v = x1 - x0, the predicted
        # clean endpoint is x1_hat = x_t + (1-t)*v_pred. MSE-ing x1_hat vs the true action
        # chunk is an exact reparameterization of the velocity loss that reweights samples by
        # (1-t)^2 (up-weights high-noise/low-t samples, which few-step inference exercises).
        # CT branch is intentionally excluded (NaN history; it already reconstructs x1 via the
        # EMA teacher). Default weight 0 => stock behaviour.
        loss_endpoint = 0.0
        if self.endpoint_loss_weight > 0.0:
            t_flow = flow_target_dict['t']                                        # (B,1,1)
            x_t_flow = flow_target_dict['x_t']
            x1_true = nactions[:flow_batchsize]
            one_minus_t = (1.0 - t_flow).clamp(min=1.0 - self.endpoint_t_clip)    # avoid /0 at t->1
            x1_hat = x_t_flow + one_minus_t * v_flow_pred
            if self.endpoint_loss_type == "l1":
                ep_elem = F.l1_loss(x1_hat, x1_true, reduction='none')
            else:
                ep_elem = F.mse_loss(x1_hat, x1_true, reduction='none')
            ep_b = self._weighted_step_mean(ep_elem)                             # (B,)
            loss = loss + self.endpoint_loss_weight * ep_b.mean()
            loss_endpoint = ep_b.mean().item()

        # --- Lumi v3: auxiliary goal head (training-only encoder regularizer) ---
        # Predict the normalized goal-in-camera-frame (6-D) from mean-pooled conditioning
        # tokens and match batch['goal_cam']. Encourages the encoder to localize the goal.
        loss_goal = 0.0
        if self.goal_head is not None and 'goal_cam' in batch:
            goal_norm = self.normalizer['goal_cam'].normalize(batch['goal_cam']).to(self.device)
            goal_pred = self._predict_goal(vis_cond)                             # (B,6) attn-pool
            if self.goal_loss_type == "mse":
                gl = F.mse_loss(goal_pred, goal_norm)
            else:
                gl = F.smooth_l1_loss(goal_pred, goal_norm)
            loss = loss + self.goal_loss_weight * gl
            loss_goal = gl.item()

        # --- v7 Rewire: vision-only inverse-dynamics aux (training-only encoder regularizer) ---
        # Predict the most-recent inter-frame motion (prev_action label) from the FIRST vs LAST
        # obs-frame pooled visual tokens ONLY (no prev_action input) -> the motion signal that
        # proprio monopolizes is forced INTO the vision path.
        loss_idm = 0.0
        if self.idm_head is not None and 'prev_action' in nobs:
            To = self.n_obs_steps
            Lc = vis_cond.shape[1]
            if Lc % To == 0:
                Lg = Lc // To
                f_first = vis_cond[:, :Lg].mean(dim=1)                            # (B,Do) frame 0
                f_last = vis_cond[:, (To - 1) * Lg: To * Lg].mean(dim=1)          # (B,Do) frame To-1
            else:
                f_first = f_last = vis_cond.mean(dim=1)
            idm_pred = self.idm_head(torch.cat([f_first, f_last], dim=-1))        # (B,Da)
            idm_tgt = nobs['prev_action'][:, To - 1].to(self.device)             # (B,Da) recent motion
            li = F.smooth_l1_loss(idm_pred, idm_tgt)
            loss = loss + self.idm_loss_weight * li
            loss_idm = li.item()

        # --- F-series: soft-argmax keypoint loss (vis-masked) + place-pose regression loss ---
        loss_kpt = 0.0
        loss_place = 0.0
        kpt_px = 0.0
        place_mm = 0.0
        if self.kpt_head is not None and f_uv is not None and 'arm_kpts_uv' in batch:
            gt = batch['arm_kpts_uv'][:, self.n_obs_steps - 1].to(self.device)   # (B,N,3) recent obs frame
            if self.kpt_head_hires:
                # G1: Gaussian-target heatmap CE (vis-masked) + presence BCE (0.2) + metric log-z
                from maniflow.model.vision_2d.heatmap_head import heatmap_kpt_loss
                gt_z = (batch['arm_kpts_cam'][:, self.n_obs_steps - 1, :, 2].to(self.device)
                        if 'arm_kpts_cam' in batch else None)
                lk_heat, lk_pres, lk_z = heatmap_kpt_loss(f_extras, gt[..., :2], gt[..., 2], gt_z)
                lk_total = lk_heat + 0.2 * lk_pres + lk_z
                loss = loss + self.kpt_loss_weight * lk_total
                loss_kpt = float(lk_total.item())
            else:
                from maniflow.model.vision_2d.soft_argmax import keypoint_loss
                lk_pos, lk_pres = keypoint_loss(f_uv, f_conf, gt[..., :2], gt[..., 2])
                loss = loss + self.kpt_loss_weight * (lk_pos + lk_pres)
                loss_kpt = float((lk_pos + lk_pres).item())
            if self.place_loss_weight > 0.0 and 'goal_cam' in batch:
                goal_norm = self.normalizer['goal_cam'].normalize(batch['goal_cam']).to(self.device)
                lpl_el = F.smooth_l1_loss(f_place, goal_norm, reduction='none')
                lpl_b = lpl_el.reshape(lpl_el.shape[0], -1).mean(-1)             # (B,)
                # FOV-dropout aug blanks the frame: no keypoints => don't force the place
                # readout to hallucinate (the DiT should lean on the goal_prior latch there)
                keep = 1.0 - batch.get('fov_dropped', torch.zeros_like(lpl_b)).to(lpl_b.dtype)
                lpl = (lpl_b * keep).sum() / keep.sum().clamp(min=1e-6)
                loss = loss + self.place_loss_weight * lpl
                loss_place = float(lpl.item())
            with torch.no_grad():                    # honest diagnostics (full-res px / mm)
                vism = gt[..., 2]
                scale = torch.tensor([1280.0, 800.0], device=f_uv.device, dtype=f_uv.dtype)
                pxe = torch.linalg.norm((f_uv - gt[..., :2]) * scale, dim=-1)
                kpt_px = float(((pxe * vism).sum() / vism.sum().clamp(min=1)).item())
                if self.place_loss_weight > 0.0 and 'goal_cam' in batch:
                    pl_un = self.normalizer['goal_cam'].unnormalize(f_place)
                    gc = batch['goal_cam'].to(self.device).reshape(pl_un.shape[0], -1)
                    dist = torch.linalg.norm(pl_un[:, :3] - gc[:, :3], dim=-1)
                    # FOV-mask like the loss: blanked samples are hallucinated priors and
                    # would inflate G1's place_mm vs G0 (review 2026-07-26)
                    kd = 1.0 - batch.get('fov_dropped',
                                         torch.zeros_like(dist)).to(dist.dtype)
                    place_mm = float(((dist * kd).sum()
                                      / kd.sum().clamp(min=1) * 1000.0).item())

        # ================================================================= H-series losses
        # Everything here is gated on h_mode; each term has its own weight knob defaulting
        # to 0 (except action_rate_weight = 0.05, which is the executability guard).
        h_log = {}
        if self.h_mode:
            To = self.n_obs_steps
            N = self.n_keypoints
            dev = self.device
            # ---- keypoints, BOTH obs frames (2x the free supervision of the G1 head) -------
            # !! DO NOT DOWN-WEIGHT KEYPOINTS 5-6 (`edge_n`, `edge_f` — the NEIGHBOUR EDGE). !!
            # They look like a minor context cue and they are not: contract §1.2b measured that
            # the panel goal's position along the rail is FREE BY 0.19 m relative to `rail_a`,
            # because a panel is placed flush against the previously installed one. The edge
            # keypoints are the ONLY visual source of that DoF, so they carry a whole
            # translational axis of the goal. All 7 keypoints are therefore weighted uniformly
            # here (the per-keypoint visibility mask inside `heatmap_kpt_loss` is the only
            # modulation, and it exists to suppress genuinely absent points, not to rank them).
            # When `has_neighbor` is false those two are invisible and the DoF is genuinely
            # unobservable from the wrist view (~1% of installs): the presence logits learn to
            # say so, and `task` carries the flag on the input side. Deploy must supply y
            # externally in that case — see the `place_pred` note in the export report.
            from maniflow.model.vision_2d.heatmap_head import heatmap_kpt_loss
            gt_uv = batch['arm_kpts_uv'][:, :To].to(dev).reshape(-1, N, 3)
            gt_z = batch['arm_kpts_cam'][:, :To, :, 2].to(dev).reshape(-1, N)
            lk_heat, lk_pres, lk_z = heatmap_kpt_loss(
                h_aux['kpt_extras'], gt_uv[..., :2], gt_uv[..., 2], gt_z)
            lk_total = lk_heat + 0.2 * lk_pres + lk_z
            loss = loss + self.kpt_loss_weight * lk_total
            loss_kpt = float(lk_total.item())
            with torch.no_grad():
                uvp = h_aux['kpt_uv'].reshape(-1, N, 2)
                vism = gt_uv[..., 2]
                scale = torch.tensor([1280.0, 800.0], device=dev, dtype=uvp.dtype)
                pxe = torch.linalg.norm((uvp - gt_uv[..., :2]) * scale, dim=-1)
                kpt_px = float(((pxe * vism).sum() / vism.sum().clamp(min=1)).item())

            # ---- place head: PRIMARY target is panel_goal_cam (§1.2b) ----------------------
            # NOT rail_a: `rail_a` + `grasp_offset` under-determines the goal by one translational
            # DoF (y free over 0.19 m along the rail). The panel goal IS what deploy needs
            # (T_tcp_goal = T_panel_goal @ inv(T_tcp_panel)) and is still grasp-offset-free — a
            # scene property — which was the reason for moving off the TCP goal in the first place.
            # `rail_a` stays as a separate AUX head: a clean landmark-grounding signal, DIFFERENT
            # CONVENTION (TCP-anchored delta vs camera-absolute), never mixed with the primary.
            pg_norm = self.normalizer['panel_goal'].normalize(batch['panel_goal']).to(dev)
            lpl_el = F.smooth_l1_loss(h_aux['place'], pg_norm, reduction='none')
            lpl_b = lpl_el.reshape(lpl_el.shape[0], -1).mean(-1)
            keep = 1.0 - batch.get('fov_dropped', torch.zeros_like(lpl_b)).to(lpl_b.dtype)
            lpl = (lpl_b * keep).sum() / keep.sum().clamp(min=1e-6)
            loss = loss + self.place_loss_weight * lpl
            loss_place = float(lpl.item())
            if self.rail_aux_head is not None and 'rail_aux' in h_aux['kpt_extras']:
                rail_norm = self.normalizer['rail_a'].normalize(batch['rail_a']).to(dev)
                lra_b = F.smooth_l1_loss(h_aux['kpt_extras']['rail_aux'], rail_norm,
                                         reduction='none').reshape(lpl_b.shape[0], -1).mean(-1)
                lra = (lra_b * keep).sum() / keep.sum().clamp(min=1e-6)
                loss = loss + self.place_loss_weight * self.rail_aux_weight * lra
                h_log['loss_rail_aux'] = float(lra.item())
            with torch.no_grad():
                pl_un = self.normalizer['panel_goal'].unnormalize(h_aux['place'])
                pgt = batch['panel_goal'].to(dev).reshape(pl_un.shape[0], -1)
                d = torch.linalg.norm(pl_un[:, :3] - pgt[:, :3], dim=-1)
                place_mm = float(((d * keep).sum() / keep.sum().clamp(min=1) * 1000.0).item())
                # Per-axis error in the CAMERA frame is not the along-rail axis, but a big gap
                # between place_mm and the kpt error is the first hint that the identifiability
                # story (§1.2b point 4) is playing out. Logged, not gated.
                h_log['place_mm_z'] = float(
                    ((pl_un[:, 2] - pgt[:, 2]).abs() * keep).sum().item()
                    / max(float(keep.sum().item()), 1.0) * 1000.0)

            # ---- observability heads (free supervision, exported for deploy gating) -------
            if self.phase_loss_weight > 0.0:
                lph = F.cross_entropy(h_aux['phase'], batch['phase_id'].to(dev).reshape(-1))
                loss = loss + self.phase_loss_weight * lph
                h_log['loss_phase'] = float(lph.item())
                with torch.no_grad():
                    h_log['phase_acc'] = float((h_aux['phase'].argmax(-1)
                                                == batch['phase_id'].to(dev).reshape(-1)
                                                ).float().mean().item())
            if self.done_loss_weight > 0.0:
                ldn = F.binary_cross_entropy_with_logits(
                    h_aux['done'], batch['done'].to(dev).float().reshape(-1, 1))
                loss = loss + self.done_loss_weight * ldn
                h_log['loss_done'] = float(ldn.item())

            # ---- action-space constraints, in RAW PHYSICAL UNITS -------------------------
            # The predicted CHUNK is read off the flow branch's endpoint reconstruction
            # x1_hat = x_t + (1-t) v_pred (an exact reparameterization of the velocity target),
            # then unnormalized: metres, radians, m/s, rad/s. Applying these constraints in
            # normalized space would make their relative weighting depend on the data range.
            t_f = flow_target_dict['t']
            omt = (1.0 - t_f).clamp(min=1.0 - self.endpoint_t_clip)
            rows = self.normalizer['action'].unnormalize(
                flow_target_dict['x_t'] + omt * v_flow_pred)           # (Bf, rows, 7)
            Bf = rows.shape[0]
            if self.action_rate_weight > 0.0:
                # Executability: the converter MEASURES per-step rates but deliberately does not
                # clamp labels (clamping breaks the anchored/goal geometry). Feasibility is
                # enforced HERE (smoothness) and deploy-side by saturation + k=1 replan.
                seq = rows
                if self.k0_row_structurally_zero:
                    # delta+cam0: the k=0 row is not an OUTPUT but it is real at EXECUTION time
                    # (the chunk starts at the anchor), so penalise a jump off it.
                    seq = torch.cat([torch.zeros_like(rows[:, :1]), rows], dim=1)
                d = seq[:, 1:] - seq[:, :-1]
                lrate = (d * d).sum(-1).mean()
                loss = loss + self.action_rate_weight * lrate
                h_log['loss_rate'] = float(lrate.item())

            if self.action_frame == "goal":
                pred6 = self._h_goal_integral(rows)
                tgt6 = h_ref['remaining'][:Bf]
                # ---- the in-reach mask (contract §9.2) -------------------------------------
                # Valid ONLY where the horizon reaches the goal (see H_GOAL_INTEGRAL_TOL_M in
                # the dataset): otherwise sum(v dt) == p_last - p_0 != p_goal - p_0 and the
                # "constraint" would inject the un-covered distance as an error. The dataset
                # evaluates it against the CLEAN GT goal — against the noised goal a 15 mm
                # perturbation vs a 5 mm tolerance zeroed it on 100% of measured batches.
                m_reach = h_ref['inreach'][:Bf].reshape(-1)
                h_log['goal_reach_frac'] = float(m_reach.mean().item())
                # `delta`+`goal` needs NO mask for the INTEGRAL: there the "integral" IS row
                # k=0, and u_0 == ad(P_0, G, R_g) == `remaining` identically — for the noised
                # goal too, because both sides are built from the same perturbed G (verified to
                # 1.2e-7). The constraint is an identity, not an approximation, so it holds at
                # every anchor whether or not the horizon reaches the goal.
                m_int = (torch.ones_like(m_reach) if self.action_param == "delta" else m_reach)
                h_log['goal_int_frac'] = float(m_int.mean().item())
                # A masked-empty batch must NEVER log as 0.0. A zero here reads as "perfect" on
                # every dashboard and is how a dead flagship loss survived a full training run:
                # loss_goal_int = 0.0 with mask sum 0, every batch. NaN is the honest value and
                # `*_frac` says why.
                n_int = m_int.sum()
                ok_int = bool(n_int.item() > 0.0)
                if self.goal_integral_weight > 0.0:
                    if ok_int:
                        li_el = F.smooth_l1_loss(pred6, tgt6, reduction='none').mean(-1)
                        li = (li_el * m_int).sum() / n_int
                        loss = loss + self.goal_integral_weight * li
                        h_log['loss_goal_int'] = float(li.item())
                    else:
                        h_log['loss_goal_int'] = float('nan')
                with torch.no_grad():
                    e = torch.linalg.norm(pred6[:, :3] - tgt6[:, :3], dim=-1)
                    h_log['goal_int_mm'] = (
                        float(((e * m_int).sum() / n_int * 1000.0).item()) if ok_int
                        else float('nan'))
                if self.terminal_zero_weight > 0.0:
                    # Terminal settle: the last row must contract to zero (twist -> 0 / u -> 0)
                    # — but ONLY for anchors whose horizon actually reaches the goal. EVERY frame
                    # is a valid anchor, so a mid-episode window's true last row is full descent
                    # speed (measured: 47.5% of clean samples with |row_15| > 0.01, up to 0.097,
                    # and a mid-episode batch where 100% of samples had |row_15| = 0.097). Unmasked
                    # this term does not "encourage settling", it FIGHTS the BC labels on half the
                    # data. Same clean-goal reach mask as the integral (§9.2).
                    n_reach = m_reach.sum()
                    if n_reach.item() > 0.0:
                        lt_el = F.smooth_l1_loss(rows[:, -1], torch.zeros_like(rows[:, -1]),
                                                 reduction='none').mean(-1)
                        lt = (lt_el * m_reach).sum() / n_reach
                        loss = loss + self.terminal_zero_weight * lt
                        h_log['loss_term'] = float(lt.item())
                    else:
                        h_log['loss_term'] = float('nan')
                if h_ref['gc_target'] is not None and self.goal_consistency_weight > 0.0:
                    # ties row 0 to the model's OWN goal estimate -> the action loss becomes a
                    # supervisor of the place head (the anti-shortcut mechanism)
                    lgc = F.smooth_l1_loss(rows[:, 0, 0:6], h_ref['gc_target'][:Bf])
                    loss = loss + self.goal_consistency_weight * lgc
                    h_log['loss_goal_cons'] = float(lgc.item())
                h_log['self_frame_frac'] = h_ref['self_frame_frac']

        loss = loss.mean()
        loss_dict = {
                **h_log,
                'loss_flow': loss_flow,
                'loss_ct': loss_ct,
                'loss_endpoint': loss_endpoint,
                'loss_goal': loss_goal,
                'loss_idm': loss_idm,
                'loss_kpt': loss_kpt,
                'kpt_px': kpt_px,
                'place_mm': place_mm,
                'loss_place': loss_place,
                'v_flow_pred_magnitude': v_flow_pred_magnitude,
                'v_ct_pred_magnitude': v_ct_pred_magnitude,
                'bc_loss': loss.item(),
        }


        return loss, loss_dict
