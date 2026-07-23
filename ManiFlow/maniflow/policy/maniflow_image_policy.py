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
        self.n_action_steps = n_action_steps
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

        # condition through visual feature
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].to(device))
        nobs_features = self.obs_encoder(this_nobs).to(device)
        vis_cond = nobs_features.reshape(B, -1, Do) # B, self.n_obs_steps*L, Do
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

    def compute_loss(self, batch, ema_model=None, **kwargs):
        # normalize input
        nobs = self.normalizer.normalize(batch['obs'])
        nactions = self.normalizer['action'].normalize(batch['action']).to(self.device)

        batch_size = nactions.shape[0]
        horizon = nactions.shape[1]

        # handle different ways of passing observation
        local_cond = None
        vis_cond = None
        trajectory = nactions
        cond_data = trajectory
        lang_cond = None
        ema_model = ema_model

        if self.language_conditioned:
            # we assume language condition is passed as 'task_name'
            lang_cond = nobs.get('task_name', None)
            assert lang_cond is not None, "Language goal is required"

        # reshape B, T, ... to B*T
        this_nobs = dict_apply(nobs,
            lambda x: x[:,:self.n_obs_steps,...].to(self.device))
        nobs_features = self.obs_encoder(this_nobs)
        vis_cond = nobs_features.reshape(batch_size, -1, self.obs_feature_dim)
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

        loss = loss.mean()
        loss_dict = {
                'loss_flow': loss_flow,
                'loss_ct': loss_ct,
                'loss_endpoint': loss_endpoint,
                'loss_goal': loss_goal,
                'loss_idm': loss_idm,
                'v_flow_pred_magnitude': v_flow_pred_magnitude,
                'v_ct_pred_magnitude': v_ct_pred_magnitude,
                'bc_loss': loss.item(),
        }


        return loss, loss_dict
