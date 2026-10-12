# qwen_content_aware_spatial_controller.py


import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.normalization import RMSNorm

from .transformer_qwenimage import (
    QwenImageTransformerBlock,
    QwenEmbedRope,
    QwenTimestepProjEmbeddings,
)


class QwenImageContentAwareSpatialControllerOutput:


    def __init__(self, block_samples):
        self.block_samples = block_samples


class TimestepOnlyEmbedding(nn.Module):

    def __init__(self, embedding_dim: int = 256, time_embed_dim: int = 1536):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=embedding_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=1000,
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=embedding_dim,
            time_embed_dim=time_embed_dim,
        )

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        t_proj = self.time_proj(timestep)
        t_emb = self.time_embedder(t_proj.to(dtype=dtype))
        return t_emb


class QwenImageContentAwareSpatialController(ModelMixin, ConfigMixin):


    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        num_layers: int = 60,
        attention_head_dim: int = 96,
        num_attention_heads: int = 16,
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 40, 40),
        content_aware_spatial_controller_in_channels: int = 64,
        num_content_aware_spatial_controller_layers: Optional[int] = None,
        output_inner_dim: int = 3072,
    ):
        super().__init__()

        self.inner_dim = num_attention_heads * attention_head_dim
        self.output_inner_dim = output_inner_dim
        self.num_layers = num_content_aware_spatial_controller_layers or num_layers

        self.pos_embed = QwenEmbedRope(
            theta=10000,
            axes_dim=list(axes_dims_rope),
            scale_rope=True,
        )

        # 时间步嵌入
        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim
        )
        self.time_only_embed = TimestepOnlyEmbedding(
            embedding_dim=256,
            time_embed_dim=self.inner_dim,
        )


        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)
        self.img_in = nn.Linear(in_channels, self.inner_dim)

        # Double Stream Transformer Blocks
        self.transformer_blocks = nn.ModuleList([
            QwenImageTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
            )
            for _ in range(self.num_layers)
        ])


        self.controller_blocks = nn.ModuleList([
            nn.Linear(self.inner_dim, self.output_inner_dim)
            for _ in range(self.num_layers * 3)
        ])

        self.controller_blocks_scale = nn.ModuleList([
            nn.Linear(self.inner_dim, 1)
            for _ in range(self.num_layers * 3)
        ])

        self._init_zero_convs()

        self.gradient_checkpointing = False

    def _init_zero_convs(self):

        for block in self.controller_blocks:
            nn.init.normal_(block.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(block.bias)
        for block in self.controller_blocks_scale:
            nn.init.normal_(block.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(block.bias)

    @classmethod
    def from_transformer(
        cls,
        transformer,
        content_aware_spatial_controller_in_channels: int = 64,
        num_content_aware_spatial_controller_layers: Optional[int] = None,
        load_weights: bool = True,
    ):

        config = transformer.config
        num_layers = num_content_aware_spatial_controller_layers or config.num_layers
        main_inner_dim = config.num_attention_heads * config.attention_head_dim

        controller = cls(
            in_channels=128,
            num_layers=config.num_layers,
            attention_head_dim=96,
            num_attention_heads=16,
            joint_attention_dim=config.joint_attention_dim,
            guidance_embeds=getattr(config, 'guidance_embeds', False),
            axes_dims_rope=(16, 40, 40),
            content_aware_spatial_controller_in_channels=content_aware_spatial_controller_in_channels,
            num_content_aware_spatial_controller_layers=num_layers,
            output_inner_dim=main_inner_dim,
        )

        return controller
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_mask: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_shapes: Optional[List[Tuple[int, int, int]]] = None,
        txt_seq_lens: Optional[List[int]] = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        conditioning_scale: float = 1.0,
        return_dict: bool = True,
        point_emb=None,
    ) -> Union[QwenImageContentAwareSpatialControllerOutput, Tuple]:


        point_emb_cat = torch.cat((point_emb[1], point_emb[0]), dim=1)
        hidden_states = torch.cat((hidden_states, point_emb_cat), dim=2)


        hidden_states = self.img_in(hidden_states)

        timestep = timestep.to(hidden_states.dtype)
        t_only = self.time_only_embed(timestep, hidden_states.dtype)
        temb = self.time_text_embed(timestep, hidden_states)


        encoder_hidden_states = self.txt_norm(encoder_hidden_states)
        encoder_hidden_states = self.txt_in(encoder_hidden_states)


        image_rotary_emb = self.pos_embed(
            img_shapes, txt_seq_lens, device=hidden_states.device
        )


        block_samples = []

        for i, block in enumerate(self.transformer_blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                encoder_hidden_states, hidden_states = self._gradient_checkpointing_func(
                    block,
                    hidden_states,
                    encoder_hidden_states,
                    encoder_hidden_states_mask,
                    temb,
                    image_rotary_emb,
                )
            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_hidden_states_mask=encoder_hidden_states_mask,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            for j in range(3):

                ctrl_out = self.controller_blocks[i * 3 + j](hidden_states)

                learned_scale = (1 + self.controller_blocks_scale[i * 3 + j](t_only)).unsqueeze(1)
                block_samples.append(ctrl_out * learned_scale * conditioning_scale)

        output_samples = (block_samples,)

        if not return_dict:
            return output_samples

        return QwenImageContentAwareSpatialControllerOutput(block_samples=output_samples)