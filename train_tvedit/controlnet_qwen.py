# qwen_controlnet.py
"""
QwenImage Edit ControlNet (轻量版)

架构概述:
- 基于 QwenImageTransformerBlock 双流 (image + text) 注意力架构
- ControlNet inner_dim = 16 × 96 = 1536 (主模型 24 × 128 = 3072)
- RoPE axes_dims = (16, 40, 40), sum = 96 = attention_head_dim
- 每个 block 输出: zero_linear(h) × (1 + scale_linear(t_emb)) × conditioning_scale
- 零初始化确保训练初期不干扰主模型
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Tuple, List, Union

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from diffusers.models.normalization import RMSNorm

# 从 QwenImage Edit Transformer 模块导入
from .transformer_qwenimage import (
    QwenImageTransformerBlock,
    QwenEmbedRope,
    QwenTimestepProjEmbeddings,
)
import os
import numpy as np

class ScaleRecorder:
    """推理期间记录每个 timestep × block 的 learned_scale"""

    def __init__(self):
        self.data = []  # list of (timestep_value, {block_idx: scale_value})

    def log(self, timestep_val: float, block_idx: int, scale: torch.Tensor):
        """在 forward 中每个 block 输出后调用"""
        s = scale.detach().float().mean().item()
        if not self.data or self.data[-1][0] != timestep_val:
            self.data.append((timestep_val, {}))
        self.data[-1][1][block_idx] = s

    def dump_heatmap(self, save_path: str = "learned_scale_heatmap.png"):
        """推理结束后调用，生成正方热力图"""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        if not self.data:
            print("[ScaleRecorder] No data recorded.")
            return

        n_steps = len(self.data)
        all_blocks = sorted(set(k for _, d in self.data for k in d))
        n_blocks = len(all_blocks)

        mat = np.zeros((n_steps, n_blocks))
        t_values = []
        for i, (t, d) in enumerate(self.data):
            t_values.append(t * 1000)  # 0-1 → 0-1000
            for j, b in enumerate(all_blocks):
                mat[i, j] = d.get(b, 0.0)

        # ---- 正方形绘图 ----
        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
        im = ax.imshow(
            mat, aspect="auto", cmap="RdYlBu_r",
            origin="upper", interpolation="nearest",
        )

        # X 轴: block index
        ax.set_xticks(range(n_blocks))
        ax.set_xticklabels([str(b) for b in all_blocks], fontsize=7)
        ax.set_xlabel("alpha", fontsize=9)

        # Y 轴: 每隔 3 个 timestep 采样标注
        step = 3
        ticks = list(range(0, n_steps, step))
        if (n_steps - 1) not in ticks:
            ticks.append(n_steps - 1)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t_values[i]:.1f}" for i in ticks], fontsize=7)
        ax.set_ylabel("Timestep (×1000)", fontsize=9)

        # Colorbar
        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cbar.ax.tick_params(labelsize=7)

        ax.set_title(f"Learned Scale ({n_steps} steps × {n_blocks} blocks)", fontsize=10)

        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"[ScaleRecorder] Saved → {save_path}  "
            f"| range [{mat.min():.3f}, {mat.max():.3f}]")

    def clear(self):
        self.data.clear()


# 全局实例
scale_recorder = ScaleRecorder()

class QwenImageControlNetOutput:
    """
    ControlNet 输出容器

    Attributes:
        controlnet_block_samples: (per_block_list,)
            per_block_list[i] 对应第 i 个 transformer block 的控制信号,
            形状为 [B, N_img, output_inner_dim]

    兼容 QwenImageTransformer2DModel 的消费方式:
        controlnet_block_samples[0][index_block]
    """

    def __init__(self, controlnet_block_samples):
        self.controlnet_block_samples = controlnet_block_samples


class TimestepOnlyEmbedding(nn.Module):
    """
    独立的时间步嵌入模块，用于预测 per-block 的 learned scale。
    与 QwenTimestepProjEmbeddings 保持一致，内置 scale=1000。
    """

    def __init__(self, embedding_dim: int = 256, time_embed_dim: int = 1536):
        super().__init__()
        self.time_proj = Timesteps(
            num_channels=embedding_dim,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
            scale=1000,  # 与 QwenTimestepProjEmbeddings 一致
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=embedding_dim,
            time_embed_dim=time_embed_dim,
        )

    def forward(self, timestep: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """
        Args:
            timestep: [B] 时间步
            dtype: 目标数据类型
        Returns:
            [B, time_embed_dim] 时间步嵌入
        """
        t_proj = self.time_proj(timestep)                    # [B] -> [B, 256]
        t_emb = self.time_embedder(t_proj.to(dtype=dtype))   # [B, 256] -> [B, time_embed_dim]
        return t_emb


class QwenImageControlNet(ModelMixin, ConfigMixin):
    """
    QwenImage Edit ControlNet (轻量版) - 仅包含 Double Stream 部分

    与主模型 QwenImageTransformer2DModel 的对比:
    ┌─────────────────────┬────────────────┬───────────────┐
    │                     │  主模型         │  ControlNet   │
    ├─────────────────────┼────────────────┼───────────────┤
    │ num_attention_heads  │  24            │  16           │
    │ attention_head_dim   │  128           │  96           │
    │ inner_dim            │  3072          │  1536         │
    │ axes_dims_rope       │  (16, 56, 56) │  (16, 40, 40) │
    │ num_layers           │  60            │  可配置       │
    └─────────────────────┴────────────────┴───────────────┘

    输出投影: inner_dim (1536) → output_inner_dim (3072) 以匹配主模型
    """

    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        in_channels: int = 128,
        num_layers: int = 60,
        attention_head_dim: int = 96,           # 原 128 → 96
        num_attention_heads: int = 16,          # 原 24  → 16
        joint_attention_dim: int = 3584,
        guidance_embeds: bool = False,
        axes_dims_rope: Tuple[int, int, int] = (16, 40, 40),  # sum = 96 = head_dim
        # ControlNet 特有参数
        controlnet_in_channels: int = 64,
        num_controlnet_layers: Optional[int] = None,
        output_inner_dim: int = 3072,           # 主模型的 inner_dim
    ):
        super().__init__()

        self.inner_dim = num_attention_heads * attention_head_dim  # 16 * 96 = 1536
        self.output_inner_dim = output_inner_dim                  # 3072
        self.num_layers = num_controlnet_layers or num_layers

        # ========================
        #  位置编码 (QwenEmbedRope)
        # ========================
        # axes_dims_rope 各分量之和 = attention_head_dim
        # (16, 40, 40) → 16 + 40 + 40 = 96 ✓
        # scale_rope=True 与主模型保持一致
        self.pos_embed = QwenEmbedRope(
            theta=10000,
            axes_dim=list(axes_dims_rope),
            scale_rope=True,
        )

        # ========================
        #  时间步嵌入
        # ========================
        # temb: 用于 Transformer Block 中的 AdaLN 调制 (SiLU + Linear)
        self.time_text_embed = QwenTimestepProjEmbeddings(
            embedding_dim=self.inner_dim
        )
        # t_only: 用于预测每个 block 的 learned scale
        self.time_only_embed = TimestepOnlyEmbedding(
            embedding_dim=256,
            time_embed_dim=self.inner_dim,
        )

        # ========================
        #  输入投影
        # ========================
        # 文本: RMSNorm → Linear (与主模型结构一致)
        self.txt_norm = RMSNorm(joint_attention_dim, eps=1e-6)
        self.txt_in = nn.Linear(joint_attention_dim, self.inner_dim)
        # 图像: Linear (in_channels 包含 point_emb 拼接后的通道数)
        self.img_in = nn.Linear(in_channels, self.inner_dim)

        # ========================
        #  Double Stream Transformer Blocks
        # ========================
        self.transformer_blocks = nn.ModuleList([
            QwenImageTransformerBlock(
                dim=self.inner_dim,
                num_attention_heads=num_attention_heads,
                attention_head_dim=attention_head_dim,
            )
            for _ in range(self.num_layers)
        ])

        # ========================
        #  ControlNet 输出层 (零初始化)
        # ========================
        # 零卷积投影: inner_dim (1536) → output_inner_dim (3072)
        self.controlnet_blocks = nn.ModuleList([
            nn.Linear(self.inner_dim, self.output_inner_dim)
            for _ in range(self.num_layers*3)
        ])
        # Scale 预测: inner_dim (1536) → 1
        self.controlnet_blocks_scale = nn.ModuleList([
            nn.Linear(self.inner_dim, 1)
            for _ in range(self.num_layers*3)
        ])

        # 零初始化确保训练初期 ControlNet 输出为零
        self._init_zero_convs()

        self.gradient_checkpointing = False

    def _init_zero_convs(self):
        """将 ControlNet 输出投影和 scale 预测层初始化为零"""
        for block in self.controlnet_blocks:
            nn.init.normal_(block.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(block.bias)
        for block in self.controlnet_blocks_scale:
            nn.init.normal_(block.weight, mean=0.0, std=1e-4)
            nn.init.zeros_(block.bias)

    @classmethod
    def from_transformer(
        cls,
        transformer,
        controlnet_in_channels: int = 64,
        num_controlnet_layers: Optional[int] = None,
        load_weights: bool = True,
    ):
        """
        从预训练的 QwenImageTransformer2DModel 创建轻量版 ControlNet。

        由于 ControlNet 的 inner_dim (1536) 与原始 Transformer (3072) 不同，
        大部分权重无法直接复制。仅加载维度兼容的模块 (如 txt_norm)。
        """
        config = transformer.config
        num_layers = num_controlnet_layers or config.num_layers
        main_inner_dim = config.num_attention_heads * config.attention_head_dim

        controlnet = cls(
            in_channels=128,
            num_layers=config.num_layers,
            attention_head_dim=96,
            num_attention_heads=16,
            joint_attention_dim=config.joint_attention_dim,
            guidance_embeds=getattr(config, 'guidance_embeds', False),
            axes_dims_rope=(16, 40, 40),
            controlnet_in_channels=controlnet_in_channels,
            num_controlnet_layers=num_layers,
            output_inner_dim=main_inner_dim,
        )

        if load_weights:
            controlnet._load_weights_from_transformer(transformer)

        return controlnet

    def _load_weights_from_transformer(self, transformer):
        """
        从 QwenImageTransformer2DModel 加载兼容的权重。
        """
        loaded, skipped = [], []

        # txt_norm: RMSNorm(joint_attention_dim=3584)，维度与主模型一致，可直接加载
        try:
            self.txt_norm.load_state_dict(transformer.txt_norm.state_dict())
            loaded.append("txt_norm")
        except Exception as e:
            skipped.append(f"txt_norm ({e})")

        # 以下模块由于维度不兼容，无法直接加载
        skipped.extend([
            "pos_embed (axes_dim 不匹配: ControlNet (16,40,40) vs 主模型 (16,56,56))",
            "time_text_embed (inner_dim 不匹配: 1536 vs 3072)",
            "txt_in (inner_dim 不匹配)",
            "img_in (inner_dim 不匹配)",
            f"transformer_blocks x {self.num_layers} (inner_dim 不匹配)",
        ])

        print(f"[QwenImage ControlNet] 权重加载:")
        print(f"  已加载: {loaded or '无'}")
        print(f"  已跳过: {skipped}")
        print(f"  ControlNet inner_dim={self.inner_dim}, 主模型 inner_dim={self.output_inner_dim}")
        print(f"  Transformer blocks 将从随机初始化开始训练。")

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
    ) -> Union[QwenImageControlNetOutput, Tuple]:
        """
        Args:
            hidden_states (`torch.Tensor`): 噪声 latent [B, N_img, C]
            encoder_hidden_states (`torch.Tensor`): 文本编码 [B, L_txt, D_text=3584]
            encoder_hidden_states_mask (`torch.Tensor`): 文本 mask [B, L_txt]
            timestep (`torch.LongTensor`): 时间步 [B]
            img_shapes (`List[Tuple[int,int,int]]`): 图像形状 [(F, H, W), ...]
            txt_seq_lens (`List[int]`): 文本序列长度
            guidance (`torch.Tensor`): 保留接口兼容，当前未使用
            joint_attention_kwargs (`Dict`): 传递给 attention 的额外参数
            conditioning_scale (`float`): ControlNet 输出的全局缩放系数
            point_emb: 点嵌入 (tuple of two tensors)

        Returns:
            QwenImageControlNetOutput:
                controlnet_block_samples = (per_block_list,)
                per_block_list[i]: [B, N_img, output_inner_dim=3072]
        """

        # ========== 拼接 point embedding 到 hidden_states (通道维度) ==========
        point_emb_cat = torch.cat((point_emb[1], point_emb[0]), dim=1)

        hidden_states = torch.cat((hidden_states, point_emb_cat), dim=2)

        # ========== 输入投影 ==========
        hidden_states = self.img_in(hidden_states)  # [B, N, in_channels] -> [B, N, inner_dim]

        # ========== 时间步处理 ==========
        timestep = timestep.to(hidden_states.dtype)

        # t_only: [B, inner_dim]，用于 per-block 的 learned scale
        t_only = self.time_only_embed(timestep, hidden_states.dtype)

        # temb: [B, inner_dim]，用于 AdaLN 调制 (SiLU + Linear → 6*dim)
        temb = self.time_text_embed(timestep, hidden_states)

        # ========== 文本嵌入 ==========
        encoder_hidden_states = self.txt_norm(encoder_hidden_states)     # RMSNorm
        encoder_hidden_states = self.txt_in(encoder_hidden_states)       # [B, L, 3584] -> [B, L, inner_dim]

        # ========== 位置编码 (QwenEmbedRope) ==========
        # 返回 (vid_freqs, txt_freqs)，complex tensor
        image_rotary_emb = self.pos_embed(
            img_shapes, txt_seq_lens, device=hidden_states.device
        )

        # ========== Transformer Blocks + ControlNet 输出收集 ==========
        controlnet_block_samples = []

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

            # --- 零卷积投影: inner_dim (1536) → output_inner_dim (3072) ---
            for j in range(3):
                ctrl_out = self.controlnet_blocks[i*3+j](hidden_states)

            # --- 时间步依赖的 learned scale ---
            # [B, inner_dim] → Linear → [B, 1] → unsqueeze → [B, 1, 1]
                learned_scale = (1 + self.controlnet_blocks_scale[i*3+j](t_only)).unsqueeze(1)
                scale_recorder.log(timestep[0].item(), i * 3 + j, learned_scale)
            # ctrl_out [B, N, 3072] * learned_scale [B, 1, 1] * scalar
                controlnet_block_samples.append(
                    ctrl_out * learned_scale * conditioning_scale
                )

        # ========== 输出 ==========
        # 包装为 (per_block_list,)
        # 兼容 QwenImageTransformer2DModel 的: controlnet_block_samples[0][index_block]
        output_samples = (controlnet_block_samples,)

        if not return_dict:
            return output_samples

        return QwenImageControlNetOutput(controlnet_block_samples=output_samples)