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


# ===================== 需要根据实际情况调整的常量 =====================
VAE_IMAGE_SIZE = 512        # 目标图 / 条件图送 VAE 的参考边长
CONDITION_IMAGE_SIZE = 512   # 条件图送文本编码器的参考边长


def calculate_dimensions(target_side, aspect_ratio):
    target_area = target_side ** 2
    h = int((target_area / aspect_ratio) ** 0.5)
    w = int(h * aspect_ratio)
    return w, h

def calculate_shift(image_seq_len, base_image_seq_len, max_image_seq_len, base_shift, max_shift):
    m = (max_shift - base_shift) / (max_image_seq_len - base_image_seq_len)
    b = base_shift - m * base_image_seq_len
    return image_seq_len * m + b

def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


class TVEditModel(L.LightningModule):
    def __init__(
        self,
        model_id: str,
        encoder_path: str = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        use_offset_noise: bool = False,
    ):
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config
        self._checked = False
        self.max_prompt_length = int(self.model_config.get("max_prompt_length", 2048))
        self.logit_mean = float(self.model_config.get("logit_mean", 1.0))
        self.logit_std = float(self.model_config.get("logit_std", 0.5))
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

        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.enable_gradient_checkpointing()
        self.transformer.eval()
        self.transformer.requires_grad_(False)

        self.text_encoder.requires_grad_(False).eval()
        self.vae = self.qwen_pipe.vae
        self.qwen_pipe.vae.requires_grad_(False).eval()
        self.qwen_pipe.vae.to(self.device).to(dtype=dtype)
        self.use_offset_noise = use_offset_noise
        self.point_encoder = SparsePointEncoder().to(dtype=dtype).to(device)

        if use_offset_noise:
            print('[debug] use OFFSET NOISE.')

        self.to(dtype=dtype)

    def expand_x_embedder(self, model, in_channels):
        old_embedder = model.img_in
        old_in_features = old_embedder.in_features
        out_features = old_embedder.out_features
        new_in_features = 2 * old_in_features

        new_embedder = torch.nn.Linear(
            new_in_features, out_features, bias=old_embedder.bias is not None
        )
        with torch.no_grad():
            new_embedder.weight.zero_()
            new_embedder.weight[:, :old_in_features].copy_(old_embedder.weight)
            if old_embedder.bias is not None:
                new_embedder.bias.copy_(old_embedder.bias)

        model.img_in = new_embedder
        return model

    def save_encoder(self, path: str):
        Path(path).mkdir(parents=True, exist_ok=True)
        weight_path = os.path.join(path, "weight.pth")
        torch.save({
            "point_encoder": self.point_encoder.state_dict(),
            "casc": self.controlnet.state_dict(), 
        }, weight_path)

    def configure_optimizers(self):
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        self.trainable_params = []
        self.trainable_params += list(self.point_encoder.parameters())
        self.trainable_params += list(self.casc.parameters())
        
        for p in self.trainable_params:
            p.requires_grad_(True)

        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        scheduler_config = opt_config.get("scheduler", {})
        warmup_steps = scheduler_config.get("warmup_steps", 1000)

        def warmup_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_after_backward(self):
        if not self._checked:
            self._checked = True
            print("not used para:")
            for name, param in self.named_parameters():
                if param.requires_grad and param.grad is None:
                    print(f"  - {name}")

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        imgs = batch["image"]              # (B, C, H, W), [0,1]
        cond_images = batch["condition"]   # (B, C, H, W), [0,1]
        prompts = batch["description"]
        points = batch["points"]

        pipe = self.qwen_pipe
        vae_scale_factor = pipe.vae_scale_factor
        latent_channels = pipe.latent_channels
        dtype = self.dtype
        device = self.device

        if isinstance(prompts, str):
            prompts = [prompts]
        batch_size = len(prompts)

        imgs = imgs.to(device=device, dtype=dtype)
        cond_images = cond_images.to(device=device, dtype=dtype)

        with torch.no_grad():
            _, _, img_h, img_w = imgs.shape
            aspect_ratio = img_w / img_h

            target_w, target_h = calculate_dimensions(VAE_IMAGE_SIZE, aspect_ratio)
            mul = vae_scale_factor * 2
            target_w = target_w // mul * mul
            target_h = target_h // mul * mul

            cond_vae_w, cond_vae_h = calculate_dimensions(VAE_IMAGE_SIZE, aspect_ratio)
            cond_vae_w = cond_vae_w // mul * mul
            cond_vae_h = cond_vae_h // mul * mul


            cond_images_for_text = []
            for i in range(batch_size):
                pil_img = to_pil_image(cond_images[i].clamp(0, 1).float().cpu())
                pil_img = pil_img.resize((round(pil_img.width * 28 / 32), round(pil_img.height * 28 / 32)))
                cond_images_for_text.append(pil_img)


            prompt_images = [[img] for img in cond_images_for_text]  
            prompt_embeds, prompt_embeds_mask = pipe._get_qwen_prompt_embeds(
                prompts, prompt_images, device=device, dtype=dtype
            )
            prompt_embeds = prompt_embeds[:, :self.max_prompt_length]
            prompt_embeds_mask = prompt_embeds_mask[:, :self.max_prompt_length]


            target_resized = F.interpolate(
                imgs, size=(target_h, target_w), mode="bilinear", align_corners=False
            )
            target_tensors = (2.0 * target_resized - 1.0).unsqueeze(2)  # (B,C,1,H,W)

            x_0_5d = self.vae.encode(target_tensors).latent_dist.sample()
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, pipe.latent_channels, 1, 1, 1)
                .to(x_0_5d.device, x_0_5d.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, pipe.latent_channels, 1, 1, 1)
                .to(x_0_5d.device, x_0_5d.dtype)
            )
            x_0_5d = (x_0_5d - latents_mean) / latents_std
            lat_h, lat_w = x_0_5d.shape[3], x_0_5d.shape[4]


            cond_resized = F.interpolate(
                cond_images, size=(cond_vae_h, cond_vae_w), mode="bilinear", align_corners=False
            )
            cond_tensors = (2.0 * cond_resized - 1.0).unsqueeze(2)
            cond_5d = pipe._encode_vae_image(image=cond_tensors, generator=None)
            clat_h, clat_w = cond_5d.shape[3], cond_5d.shape[4]
            image_latents = pipe._pack_latents(cond_5d, batch_size, latent_channels, clat_h, clat_w)


            img_shapes = [
                [
                    (1, lat_h // 2, lat_w // 2),
                    (1, clat_h // 2, clat_w // 2),
                ]
                for _ in range(batch_size)
            ]
            current_step = self.global_step


            curriculum_steps = 10000.0 

            
            alpha_start, beta_start = 20.0, 2.0

            alpha_end, beta_end = 5.0, 2.0

           
            progress = min(current_step / curriculum_steps, 1.0)

          
            current_alpha = alpha_start + progress * (alpha_end - alpha_start)
            current_beta = beta_start + progress * (beta_end - beta_start)


            device = self.device
            dtype = self.dtype 

            alpha_tensor = torch.tensor([current_alpha], device=device)
            beta_tensor = torch.tensor([current_beta], device=device)
            beta_dist = dist.Beta(alpha_tensor, beta_tensor)


            sigmas = beta_dist.sample((batch_size,)).squeeze(-1).to(dtype)

            image_seq_len = math.prod((1, lat_h // 2, lat_w // 2))
            mu = calculate_shift(
                image_seq_len,
                self.scheduler.config.get("base_image_seq_len", 256),
                self.scheduler.config.get("max_image_seq_len", 8192),
                self.scheduler.config.get("base_shift", 0.5),
                self.scheduler.config.get("max_shift", 0.9),
            )

            if self.scheduler.config.time_shift_type == "exponential":
                shift = math.exp(mu)
            elif self.scheduler.config.time_shift_type == "linear":
                shift = mu
            else:
                raise NotImplementedError


            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

            timesteps = sigmas * self.scheduler.num_train_timesteps
            timestep_model = timesteps / 1000.0

            sigmas_5d = sigmas.view(batch_size, 1, 1, 1, 1)

            noises = torch.randn_like(x_0_5d)
            x_t_5d = (1.0 - sigmas_5d) * x_0_5d + sigmas_5d * noises
            target = noises - x_0_5d

            x_t = pipe._pack_latents(x_t_5d, batch_size, latent_channels, lat_h, lat_w)

            guidance = (
                torch.full_like(sigmas, 1.0)
                if self.transformer.config.guidance_embeds
                else None
            )
            txt_seq_lens = prompt_embeds_mask.sum(dim=1).tolist()

        disk1, disk2 = points[0], points[1]
        point_emb = self.point_encoder(
            disk1.to(device).to(dtype),
            disk2.to(device).to(dtype),
            cond_images,
        )

        latent_model_input = torch.cat([x_t, image_latents], dim=1)

        control_output = self.casc(
            hidden_states=latent_model_input,
            timestep=timestep_model,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
            point_emb=point_emb
        )

        transformer_out = self.transformer(
            hidden_states=latent_model_input,
            timestep=timestep_model,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs={},
            return_dict=False,
            controlnet_block_samples=control_output,
        )

        pred_packed = transformer_out[0][:, :x_t.size(1)]
        pred_5d = pipe._unpack_latents(pred_packed, target_h, target_w, pipe.vae_scale_factor)

        if self.model_config.get("loss_type", "v") == "v":
            loss = F.mse_loss(pred_5d.float(), target.float(), reduction="mean")
        else:
            x_pred = x_t_5d - sigmas_5d * pred_5d
            loss = F.mse_loss(x_pred.float(), x_0_5d.float(), reduction="mean")
        self.last_t = sigmas.mean().item()

        return loss


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