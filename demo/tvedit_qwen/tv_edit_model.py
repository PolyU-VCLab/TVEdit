import lightning as L
import torch
import torch.nn.functional as F
import math
from peft import LoraConfig, get_peft_model_state_dict
import os
from pathlib import Path
import prodigyopt
from .pipelines import QwenImageEditPipeline
from .transformer_qwenimage import QwenImageTransformer2DModel
from .sparse_point_encoder import SparsePointEncoder
import re
from safetensors.torch import save_file
from safetensors import safe_open
from .content_aware_spatial_controller import QwenImageContentAwareSpatialController
from torchvision.transforms.functional import to_pil_image
from diffusers import FlowMatchEulerDiscreteScheduler
import torch.distributions as dist





class TVEditModel(L.LightningModule):
    def __init__(
        self,
        model_id: str,
        weight_path: str = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.transformer = QwenImageTransformer2DModel.from_pretrained(
            model_id, subfolder="transformer", torch_dtype=torch.bfloat16
        )
        self.casc = create_controller(
            transformer=self.transformer,
            content_aware_spatial_controller_in_channels=128,
            num_content_aware_spatial_controller_layers=5,
            torch_dtype=torch.bfloat16,
        )
        self.casc = self.casc.to(dtype=dtype).to(device)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_id, subfolder="scheduler"
        )
        self.qwen_pipe = QwenImageEditPipeline.from_pretrained(
            model_id, transformer=self.transformer
        ).to(dtype=dtype)

        self.text_encoder = self.qwen_pipe.text_encoder
        self.transformer = self.qwen_pipe.transformer
        self.transformer = self.transformer.to(dtype=dtype).to(device)

        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.text_encoder.requires_grad_(False).eval()
        self.vae = self.qwen_pipe.vae
        self.qwen_pipe.vae.requires_grad_(False).eval()
        self.qwen_pipe.vae.to(self.device).to(dtype=dtype)
        self.point_encoder = SparsePointEncoder().to(dtype=dtype).to(device)

        if weight_path is not None:
            self.load_weight(weight_path)
        self.to(dtype=dtype)


    def load_weight(self, path):
        state = torch.load(path, map_location=self.device)
        self.point_encoder.load_state_dict(state["point_encoder"])
        self.casc.load_state_dict(state["casc"])


def create_controller(
    transformer=None,
    content_aware_spatial_controller_in_channels: int = 64,
    num_content_aware_spatial_controller_layers: int = 19,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    controller = QwenImageContentAwareSpatialController.from_transformer(
        transformer,
        content_aware_spatial_controller_in_channels=content_aware_spatial_controller_in_channels,
        num_content_aware_spatial_controller_layers=num_content_aware_spatial_controller_layers,
        load_weights=False,
    )
    return controller