import copy

import numpy as np
import timm
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
from termcolor import cprint

from maniflow.model.common.module_attr_mixin import ModuleAttrMixin

from maniflow.common.pytorch_util import replace_submodules

logger = logging.getLogger(__name__)


def _adapt_first_conv(model, in_channels):
    """Replace the FIRST nn.Conv2d in a (possibly truncated Sequential) trunk with one that
    takes `in_channels` inputs, Kaiming-initialized. Used for the v3 depth trunk whose band-
    mask channels are non-photometric, so pretrained RGB filters don't transfer. Preserves
    out_channels/kernel/stride/padding/bias. Walks modules in definition order and swaps the
    first Conv2d it finds (resnet 'conv1')."""
    first = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first = m
            break
    if first is None:
        raise RuntimeError("no Conv2d found in trunk to adapt for depth input")
    new_conv = nn.Conv2d(
        in_channels, first.out_channels, kernel_size=first.kernel_size,
        stride=first.stride, padding=first.padding, bias=(first.bias is not None))
    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
    if first.bias is not None:
        nn.init.zeros_(new_conv.bias)

    # swap by identity match (replace_submodules matches by predicate; the first conv is unique)
    replaced = {"done": False}

    def _pred(x):
        return isinstance(x, nn.Conv2d) and x is first and not replaced["done"]

    def _func(x):
        replaced["done"] = True
        return new_conv

    return replace_submodules(root_module=model, predicate=_pred, func=_func)

class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)
    

class TimmObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            model_name: str,
            pretrained: bool,
            frozen: bool,
            global_pool: str,
            transforms: list,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            # renormalize rgb input with imagenet normalization
            # assuming input in [0,1]
            imagenet_norm: bool=False,
            feature_aggregation: str='spatial_embedding',
            downsample_ratio: int=32,
            position_encording: str='learnable',
            # v3 dense-token mode: emit (B, L, feature_dim) conditioning TOKENS instead of
            # one flattened (B, K*feature_dim) vector. RGB trunks -> H*W spatial tokens/frame
            # (no pooling); each low-dim key -> 1 projected token/frame. This un-starves
            # DiT-X cross-attention (v1/v2 collapsed all obs into a single token).
            token_output: bool=False,

        ):
        """
        Assumes rgb input: B,T,C,H,W
        Assumes low_dim input: B,T,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_shape_map = dict()

        assert global_pool == ''
        self.token_output = token_output
        # in token mode force the dense (no-aggregation) resnet path
        if token_output and not model_name.startswith('vit'):
            feature_aggregation = None

        if model_name == "r3m":
            from r3m import load_r3m
            model = load_r3m("resnet18", pretrained=pretrained) # resnet18, resnet34
            model.eval()
            cprint(f"Loaded R3M model using {model_name}. pretrained={pretrained}", 'green')
        else:
            model = timm.create_model(
                model_name=model_name,
                pretrained=pretrained,
                global_pool=global_pool, # '' means no pooling
                num_classes=0            # remove classification layer
            )

        if frozen:
            assert pretrained
            for param in model.parameters():
                param.requires_grad = False
        
        feature_dim = None
        if model_name.startswith('resnet'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 512
            elif downsample_ratio == 16:
                modules = list(model.children())[:-3]
                model = torch.nn.Sequential(*modules)
                feature_dim = 256
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('convnext'):
            # the last layer is nn.Identity() because num_classes is 0
            # second last layer is AdaptivePool2d, which is also identity because global_pool is empty
            if downsample_ratio == 32:
                modules = list(model.children())[:-2]
                model = torch.nn.Sequential(*modules)
                feature_dim = 1024
            else:
                raise NotImplementedError(f"Unsupported downsample_ratio: {downsample_ratio}")
        elif model_name.startswith('r3m'):
            # feature_dim = 2048
            feature_dim = 512
        
        self.feature_dim = feature_dim

        if use_group_norm and not pretrained:
            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=(x.num_features // 16) if (x.num_features % 16 == 0) else (x.num_features // 8),
                    num_channels=x.num_features)
            )

        # v3 token mode with a PRETRAINED trunk: freeze BatchNorm (FrozenBatchNorm2d).
        # Two hazards otherwise: (a) EMAModel deepcopies BN running stats once and then
        # only averages parameters -> the EMA policy runs live-averaged weights against
        # stale BN stats; (b) small-batch, heavily-augmented two-trunk batches drift the
        # running stats away from the ImageNet statistics the pretrained weights expect.
        # FrozenBatchNorm2d is immune to .train() flips — the standard finetune recipe.
        if token_output and pretrained:
            from torchvision.ops.misc import FrozenBatchNorm2d

            def _to_frozen_bn(bn):
                fbn = FrozenBatchNorm2d(bn.num_features, eps=bn.eps)
                with torch.no_grad():
                    fbn.weight.copy_(bn.weight)
                    fbn.bias.copy_(bn.bias)
                    fbn.running_mean.copy_(bn.running_mean)
                    fbn.running_var.copy_(bn.running_var)
                return fbn

            model = replace_submodules(
                root_module=model,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=_to_frozen_bn)
            cprint("[TimmObsEncoder] token mode: BatchNorm frozen (FrozenBatchNorm2d)", "green")
        
        image_shape = None
        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            if type == 'rgb':
                assert image_shape is None or image_shape == shape[1:]
                image_shape = shape[1:]
        # v3 token mode: a "depth" rgb key (name contains 'depth') is a metric depth map
        # banded into N channels, NOT a photometric image. Random geometric aug (crop/rotation)
        # done per-key with independent RNG would DE-REGISTER it from the paired head_cam (same
        # physical camera), and ColorJitter / ImageNet norm are meaningless on depth. So depth
        # keys get ONLY a deterministic crop+resize matching head_cam's static crop size (depth's
        # own axial/dropout aug happens upstream in the dataset). depth_transform is built HERE,
        # inside the same block, BEFORE `transforms` is reassigned below (else the re-test would
        # fail once transforms[0] becomes an nn.Module and depth would wrongly fall back to the
        # RGB stack — which uses antialias=True and rejects >3 channels).
        # antialias=False: torchvision's AA path only accepts 1/3 channels ("permitted channel
        # values are [1,3], but found N"); AA is meaningless on discrete band-masks anyway.
        depth_transform = None
        if transforms is not None and not isinstance(transforms[0], torch.nn.Module):
            assert transforms[0].type == 'RandomCrop'
            ratio = transforms[0].ratio
            depth_transform = torch.nn.Sequential(
                torchvision.transforms.CenterCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=False))
            transforms = [
                torchvision.transforms.RandomCrop(size=int(image_shape[0] * ratio)),
                torchvision.transforms.Resize(size=image_shape[0], antialias=True)
            ] + transforms[1:]
        transform = nn.Identity() if transforms is None else torch.nn.Sequential(*transforms)

        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)

                is_depth = token_output and ('depth' in key)
                # depth trunk must be its OWN copy (never shared) — it has a different
                # first-conv width (N band channels) and no ImageNet stats.
                this_model = copy.deepcopy(model) if (is_depth or not share_rgb_model) else model
                depth_ch = int(shape[0]) if is_depth else None
                if is_depth and depth_ch != 3:
                    # adapt the trunk's first conv to depth_ch input channels. Band-mask
                    # channels are NOT photometric, so init the new conv from scratch
                    # (Kaiming) rather than averaging pretrained RGB filters. Also swap the
                    # depth trunk's norm -> GroupNorm (frozen ImageNet BN is meaningless on a
                    # fresh 10-ch conv); RGB trunk keeps its FrozenBN. NOTE: if token_output+
                    # pretrained already ran, the depth deepcopy carries FrozenBatchNorm2d
                    # (NOT an nn.BatchNorm2d subclass), so match BOTH here.
                    from torchvision.ops.misc import FrozenBatchNorm2d as _FBN

                    def _nfeat(x):
                        # nn.BatchNorm2d has .num_features; FrozenBatchNorm2d does not (only buffers)
                        return getattr(x, "num_features", None) or x.weight.shape[0]

                    def _to_gn(x):
                        nf = _nfeat(x)
                        return nn.GroupNorm(
                            num_groups=(nf // 16) if (nf % 16 == 0) else (nf // 8),
                            num_channels=nf)

                    this_model = _adapt_first_conv(this_model, depth_ch)
                    this_model = replace_submodules(
                        root_module=this_model,
                        predicate=lambda x: isinstance(x, (nn.BatchNorm2d, _FBN)),
                        func=_to_gn)
                    cprint(f"[TimmObsEncoder] depth trunk '{key}': first conv -> {depth_ch}ch "
                           f"(scratch) + GroupNorm", "green")
                key_model_map[key] = this_model

                if is_depth and depth_transform is not None:
                    key_transform_map[key] = depth_transform
                else:
                    key_transform_map[key] = transform
            elif type == 'low_dim':
                if not attr.get('ignore_by_policy', False):
                    low_dim_keys.append(key)
            else:
                cprint(f"Skipping obs key {key} with type {type}", 'red')
        
        feature_map_shape = [x // downsample_ratio for x in image_shape]
            
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)
        # depth rgb keys skip ImageNet normalization in token mode (metric depth, not RGB)
        self._depth_rgb_keys = {k for k in rgb_keys if token_output and ('depth' in k)}
        print('rgb keys:         ', rgb_keys)
        print('low_dim_keys keys:', low_dim_keys)

        self.model_name = model_name
        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map
        self.feature_aggregation = feature_aggregation

        # v3 token mode: (a) explicit ImageNet normalization (a timm-pretrained trunk
        # expects it; the base encoder's `imagenet_norm` flag is otherwise a no-op here),
        # (b) a per-low-dim-key Linear -> feature_dim so every modality yields tokens of
        # the same width for concatenation on the token axis.
        self.token_imagenet_norm = bool(token_output and imagenet_norm)
        if self.token_imagenet_norm:
            self.register_buffer('_in_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('_in_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        if token_output:
            self.lowdim_proj = nn.ModuleDict({
                key: nn.Linear(int(np.prod(key_shape_map[key])), feature_dim)
                for key in low_dim_keys})

        if model_name.startswith('vit'):
            # assert self.feature_aggregation is None # vit uses the CLS token
            if self.feature_aggregation == 'all_tokens':
                # Use all tokens from ViT
                pass
            elif self.feature_aggregation is not None:
                logger.warn(f'vit will use the CLS token. feature_aggregation ({self.feature_aggregation}) is ignored!')
                self.feature_aggregation = None
        
        if self.feature_aggregation == 'soft_attention':
            self.attention = nn.Sequential(
                nn.Linear(feature_dim, 1, bias=False),
                nn.Softmax(dim=1)
            )
        elif self.feature_aggregation == 'spatial_embedding':
            self.spatial_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1], feature_dim))
        elif self.feature_aggregation == 'transformer':
            if position_encording == 'learnable':
                self.position_embedding = torch.nn.Parameter(torch.randn(feature_map_shape[0] * feature_map_shape[1] + 1, feature_dim))
            elif position_encording == 'sinusoidal':
                num_features = feature_map_shape[0] * feature_map_shape[1] + 1
                self.position_embedding = torch.zeros(num_features, feature_dim)
                position = torch.arange(0, num_features, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, feature_dim, 2).float() * (-math.log(2 * num_features) / feature_dim))
                self.position_embedding[:, 0::2] = torch.sin(position * div_term)
                self.position_embedding[:, 1::2] = torch.cos(position * div_term)
            self.aggregation_transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(d_model=feature_dim, nhead=4),
                num_layers=4)
        elif self.feature_aggregation == 'attention_pool_2d':
            self.attention_pool_2d = AttentionPool2d(
                spacial_dim=feature_map_shape[0],
                embed_dim=feature_dim,
                num_heads=feature_dim // 64,
                output_dim=feature_dim
            )
        logger.info(
            "number of parameters: %e", sum(p.numel() for p in self.parameters())
        )

    def aggregate_feature(self, feature):
        if self.model_name == 'r3m':
            return feature
        if self.model_name.startswith('vit'):
            assert self.feature_aggregation is None # vit uses the CLS token
            return feature[:, 0, :]
        
        # resnet
        assert len(feature.shape) == 4
        if self.feature_aggregation == 'attention_pool_2d':
            return self.attention_pool_2d(feature)

        feature = torch.flatten(feature, start_dim=-2) # B, 512, 7*7
        feature = torch.transpose(feature, 1, 2) # B, 7*7, 512

        if self.feature_aggregation == 'avg':
            return torch.mean(feature, dim=[1])
        elif self.feature_aggregation == 'max':
            return torch.amax(feature, dim=[1])
        elif self.feature_aggregation == 'soft_attention':
            weight = self.attention(feature)
            return torch.sum(feature * weight, dim=1)
        elif self.feature_aggregation == 'spatial_embedding':
            return torch.mean(feature * self.spatial_embedding, dim=1)
        elif self.feature_aggregation == 'transformer':
            zero_feature = torch.zeros(feature.shape[0], 1, feature.shape[-1], device=feature.device)
            if self.position_embedding.device != feature.device:
                self.position_embedding = self.position_embedding.to(feature.device)
            feature_with_pos_embedding = torch.concat([zero_feature, feature], dim=1) + self.position_embedding
            feature_output = self.aggregation_transformer(feature_with_pos_embedding)
            return feature_output[:, 0]
        else:
            assert self.feature_aggregation is None
            return feature
        
    def forward(self, obs_dict):
        features = list()
        batch_size = next(iter(obs_dict.values())).shape[0]
        
        # process rgb input
        for key in self.rgb_keys:
            img = obs_dict[key]
            # normalize image by hand
            if img.max() > 1.0:
                # assume input in [0, 255]
                img = img / 255.0
            if img.shape[-1] == 3:
                if len(img.shape) == 5:
                    # B, T, H, W, C --> B, T, C, H, W
                    img = img.permute(0, 1, 4, 2, 3)
                elif len(img.shape) == 4:
                    # B, H, W, C --> B, C, H, W
                    img = img.permute(0, 3, 1, 2)

            B, T = img.shape[:2]
            assert B == batch_size
            img = img.reshape(B*T, *img.shape[2:])

            if img.shape[2:] != self.key_shape_map[key]:
                target_H, target_W = self.key_shape_map[key][1], self.key_shape_map[key][2]
                # do torchvision resize
                # img shape: Bx3xHxW
                # new size: Bx3xnHxnW
                img = F.interpolate(img, size=(target_H, target_W), mode='bilinear', align_corners=False)
            
            assert img.shape[1:] == self.key_shape_map[key]
            # Image augmentation (RandomCrop/Rotation/ColorJitter) has no learnable
            # params, so its intermediates never need to be retained for backward.
            # Running it under no_grad frees several GB of activations (ColorJitter's
            # hsv2rgb einsum alone is multi-GB across two trunks x B*T images) and
            # avoids OOM. The trainable encoder below still backprops normally from
            # the augmented pixels.
            with torch.no_grad():
                img = self.key_transform_map[key](img).to(self.device)
            # ImageNet norm only for photometric RGB keys (skip metric depth maps)
            if self.token_imagenet_norm and key not in self._depth_rgb_keys:
                img = (img - self._in_mean) / self._in_std
            raw_feature = self.key_model_map[key](img).to(self.device)
            feature = self.aggregate_feature(raw_feature)
            if self.token_output:
                # feature: (B*T, H*W, D) dense tokens -> (B, T*H*W, D)
                assert len(feature.shape) == 3 and feature.shape[0] == B * T
                features.append(feature.reshape(B, T * feature.shape[1], feature.shape[2]))
            else:
                assert len(feature.shape) == 2 and feature.shape[0] == B * T
                features.append(feature.reshape(B, -1))

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key].to(self.device)
            B, T = data.shape[:2]
            assert B == batch_size
            assert data.shape[2:] == self.key_shape_map[key]
            if self.token_output:
                # (B,T,*) -> one projected token per frame: (B, T, D)
                tok = self.lowdim_proj[key](data.reshape(B, T, -1))
                features.append(tok)
            else:
                features.append(data.reshape(B, -1))

        # token mode: concat on the TOKEN axis -> (B, L, D); the policy passes this
        # straight to DiT-X cross-attention (its reshape(B,-1,D) is then a no-op).
        # legacy mode: concat flattened features -> (B, K*D).
        result = torch.cat(features, dim=1 if self.token_output else -1)

        return result


    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (1, attr['horizon']) + shape,
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        if self.token_output:
            assert len(example_output.shape) == 3 and example_output.shape[0] == 1
        else:
            assert len(example_output.shape) == 2
            assert example_output.shape[0] == 1
        return example_output.shape


if __name__=='__main__':
    timm_obs_encoder = TimmObsEncoder(
        shape_meta=None,
        model_name='resnet18.a1_in1k',
        pretrained=False,
        global_pool='',
        transforms=None
    )