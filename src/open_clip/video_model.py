from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from dataclasses import dataclass

from .transformer import (
    LayerNormFp32,
    LayerNorm,
    QuickGELU,
    Transformer,
)
from .model import CLIPTextCfg, CLIPVisionCfg, _build_vision_tower, _build_text_tower


@dataclass
class TemporalCfg:
    context_length: int = 32 # number of input frames
    width: int = 512
    heads: int = 8
    layers: int = 12
    mlp_ratio: int = 4
    pooler_type: str = "cls_pooler"


# TODO: ViViT class makes this function a bit pointless
# still thinking about how to organize this better
def _build_video_tower(
        embed_dim,
        vision_cfg,
        temporal_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
    ):

        model = ViViT(
            embed_dim,
            vision_cfg,
            temporal_cfg,
            quick_gelu,
            cast_dtype,
        )

        return model

# TODO: implement
class ViViT(nn.Module):
    """ViViT model (https://arxiv.org/abs/2103.15691)"""
    def __init__(
        self,
        embed_dim,
        vision_cfg,
        temporal_cfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        temporal_cfg = TemporalCfg(**temporal_cfg) if isinstance(temporal_cfg, dict) else temporal_cfg

        act_layer = QuickGELU if quick_gelu else nn.GELU
        norm_layer = (
            LayerNormFp32 if cast_dtype in (torch.float16, torch.bfloat16) else LayerNorm
        )

        self.spatial = _build_vision_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )
        self.temporal = Transformer(
            width=temporal_cfg.width,
            layers=temporal_cfg.layers,
            heads=temporal_cfg.heads,
            mlp_ratio=temporal_cfg.mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

    def forward(self, video):
        return torch.randn((video.shape[0], 512))


# TODO: turn into VideoCoCa
# TODO: set_grad_checkpointing
class VideoCLIP(nn.Module):
    def __init__(
        self,
        embed_dim,
        vision_cfg: CLIPVisionCfg,
        text_cfg: CLIPTextCfg,
        temporal_cfg: TemporalCfg,
        quick_gelu: bool = False,
        cast_dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        vision_cfg = CLIPVisionCfg(**vision_cfg) if isinstance(vision_cfg, dict) else vision_cfg
        text_cfg = CLIPTextCfg(**text_cfg) if isinstance(text_cfg, dict) else text_cfg
        temporal_cfg = TemporalCfg(**temporal_cfg) if isinstance(temporal_cfg, dict) else temporal_cfg

        self.visual = _build_video_tower(
            embed_dim=embed_dim,
            vision_cfg=vision_cfg,
            temporal_cfg=temporal_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        self.text = _build_text_tower(
            embed_dim=embed_dim,
            text_cfg=text_cfg,
            quick_gelu=quick_gelu,
            cast_dtype=cast_dtype,
        )

        vocab_size = (
            text_cfg.vocab_size  # for hf models
            if hasattr(text_cfg, "hf_model_name") and text_cfg.hf_model_name is not None
            else text_cfg.vocab_size
        )

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def encode_video(self, video, normalize: bool = False):
        features = self.visual(video)
        return F.normalize(features, dim=-1) if normalize else features
    def encode_text(self, text, normalize: bool = False):
        features = self.text(text)
        return F.normalize(features, dim=-1) if normalize else features
    def forward(self, video, text):
        video_features = self.encode_video(video, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        # TODO: make loss funcitons generalize to all types of modality pairs
        # i.e. make keys more general, for now keeping as image_features
        return {
            "image_features": video_features,
            "text_features": text_features,
            "logit_scale": self.logit_scale.exp()
        }
