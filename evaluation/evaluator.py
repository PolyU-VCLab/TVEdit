"""
Module for evaluating image editing with DINOv3 features and perceptual metrics.
Includes functionality for feature extraction and similarity measurements.
"""

import warnings
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn.functional as F
import lpips
from transformers import AutoModel

# Suppress UserWarnings
warnings.filterwarnings(action='ignore', category=UserWarning)

# ImageNet normalization constants
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class DINOv3Featurizer:
    """
    Extracts DINOv3 ViT-B/16 dense patch features for semantic point matching.
    Uses high-resolution input to produce fine-grained spatial feature maps.
    
    DINOv3-base: patch_size=16, hidden_dim=768, 4 register tokens, RoPE.
    Supports variable input resolutions (must be multiples of 16).
    """
    
    def __init__(
        self, 
        model_name: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m',
        device: str = 'cuda', 
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize DINOv3 feature extractor.
        
        Args:
            model_name: HuggingFace model identifier or local path.
            device: Device for inference.
            dtype: Data type for model weights.
        """
        self.model = AutoModel.from_pretrained(
            model_name, torch_dtype=dtype
        ).to(device).eval()
        self.device = device
        self.dtype = dtype
        self.patch_size = self.model.config.patch_size  # 16
        self.num_register_tokens = getattr(self.model.config, 'num_register_tokens', 4)
    
    @torch.no_grad()
    def __call__(self, img_tensor: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Extract DINOv3 patch features and reshape into a spatial feature map.
        
        Args:
            img_tensor: Image tensor of shape (1, 3, H, W), normalized to [-1, 1].
                        H and W must be multiples of patch_size (16).
            **kwargs: Unused, kept for API compatibility (e.g., prompt).
            
        Returns:
            torch.Tensor: Feature map of shape (1, hidden_dim, h, w)
                          where h = H // patch_size, w = W // patch_size.
                          For 768x768 input, returns (1, 768, 48, 48).
        """
        # Convert from [-1, 1] to [0, 1]
        img_01 = (img_tensor + 1.0) / 2.0
        
        # Apply ImageNet normalization
        mean = IMAGENET_MEAN.to(device=img_01.device, dtype=img_01.dtype)
        std = IMAGENET_STD.to(device=img_01.device, dtype=img_01.dtype)
        pixel_values = (img_01 - mean) / std
        
        B, C, H, W = pixel_values.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            f"Image size ({H}, {W}) must be divisible by patch_size ({self.patch_size})"
        
        h = H // self.patch_size
        w = W // self.patch_size
        
        # Forward pass through DINOv3
        outputs = self.model(pixel_values=pixel_values)
        # last_hidden_state: (B, 1 + num_register_tokens + h*w, hidden_dim)
        tokens = outputs.last_hidden_state
        
        # Remove CLS token (index 0) and register tokens (indices 1..num_reg)
        # Keep only patch tokens
        patch_tokens = tokens[:, 1 + self.num_register_tokens:, :]  # (B, h*w, hidden_dim)
        
        # Reshape to spatial feature map: (B, hidden_dim, h, w)
        feat_map = patch_tokens.permute(0, 2, 1).reshape(B, -1, h, w)
        
        return feat_map


class DragEvaluator:
    """
    Evaluator for computing perceptual and distance metrics between images.
    
    Uses DINOv3 dense features for semantic point matching and 
    LPIPS for perceptual similarity measurements.
    """
    
    def __init__(self, dino_model: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m'):
        """Initialize the evaluator with required models and settings."""
        self.dino_model = dino_model
        self.dinov3_loaded = False
        self.lpips_loaded = False
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype = torch.float16

    def load_dinov3(self):
        """Load DINOv3 feature extractor if not already loaded."""
        if not self.dinov3_loaded:
            self.dinov3_feat = DINOv3Featurizer(
                model_name=self.dino_model,
                device=self.device,
                dtype=self.dtype
            )
            self.dinov3_loaded = True

    def load_lpips(self):
        """Load LPIPS model if not already loaded."""
        if not self.lpips_loaded:
            # 设置 spatial=True 以获取空间 LPIPS map
            self.loss_fn_alex = lpips.LPIPS(net='alex', spatial=True).to(self.device).float()
            self.lpips_loaded = True

    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        Convert image to tensor and normalize to [-1, 1].
        
        Args:
            image: Input image as numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        image = torch.from_numpy(np.array(image)).float() / 127.5 - 1
        image = image.unsqueeze(0).permute(0, 3, 1, 2)
        return image.to(self.device).to(self.dtype)
    
    @torch.no_grad()
    def compute_lpips(
        self, 
        original_image: np.ndarray, 
        edited_image: np.ndarray,
        plot_path: str = None
    ) -> float:
        self.load_lpips()
        
        # LPIPS 输入也用 float32，不走 self.dtype (float16)
        image1 = self.preprocess_image(original_image).float()
        image2 = self.preprocess_image(edited_image).float()
        image1 = F.interpolate(image1, (224, 224), mode='bilinear')
        image2 = F.interpolate(image2, (224, 224), mode='bilinear')

        lpips_map = self.loss_fn_alex(image1, image2)
        
        if plot_path:
            self.plot_lpips_map(
                original_image=original_image,
                edited_image=edited_image,
                lpips_map=lpips_map,
                output_path=plot_path
            )
        
        return lpips_map.mean().item()

    def plot_lpips_map(
        self,
        original_image: np.ndarray,
        edited_image: np.ndarray,
        lpips_map: torch.Tensor,
        output_path: str
    ):
        """
        Plot LPIPS spatial contribution map.
        
        Args:
            original_image: Original image array
            edited_image: Edited image array
            lpips_map: LPIPS spatial map tensor (B, 1, H, W)
            output_path: Path to save the visualization
        """
        # 转换 lpips_map 到 numpy
        lpips_np = lpips_map[0, 0].float().cpu().numpy()
        
        # 上采样到原图尺寸
        orig_h, orig_w = original_image.shape[:2]
        lpips_resized = cv2.resize(lpips_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建图像
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        # 原图
        axes[0].imshow(original_image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # 编辑后的图
        axes[1].imshow(edited_image)
        axes[1].set_title('Edited Image')
        axes[1].axis('off')
        
        # LPIPS 热力图
        im = axes[2].imshow(lpips_resized, cmap='hot', interpolation='nearest')
        axes[2].set_title(f'LPIPS Map (Mean: {lpips_resized.mean():.4f})')
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
        
        # LPIPS 叠加到编辑图上
        axes[3].imshow(edited_image)
        overlay = axes[3].imshow(lpips_resized, cmap='hot', alpha=0.5, interpolation='nearest')
        axes[3].set_title('LPIPS Overlay')
        axes[3].axis('off')
        plt.colorbar(overlay, ax=axes[3], fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        # 同时保存单独的 LPIPS map（方便后续分析）
        base_path = os.path.splitext(output_path)[0]
        np.save(f"{base_path}_lpips_map.npy", lpips_resized)

    @torch.no_grad()
    def compute_distance(self, original_image: np.ndarray, edited_image: np.ndarray, 
                        handle_pts: np.ndarray, target_pts: np.ndarray, 
                        prompt: str = None, plot_path: str = None) -> float:
        """
        Compute mean distance metric between handle and target points
        using DINOv3 dense features for semantic matching.
        
        Args:
            original_image: Original image array
            edited_image: Edited image array
            handle_pts: Handle point coordinates
            target_pts: Target point coordinates
            prompt: Unused, kept for API compatibility
            plot_path: Optional path to save visualization
            
        Returns:
            float: Mean distance metric
        """
        self.load_dinov3()
        
        handle_pts = torch.tensor(handle_pts, device=self.device, dtype=torch.long)
        target_pts = torch.tensor(target_pts, device=self.device, dtype=torch.long)
        
        # Handle image size mismatch
        if original_image.shape != edited_image.shape:
            orig_h, orig_w = original_image.shape[:2]
            edit_h, edit_w = edited_image.shape[:2]
            edited_image = cv2.resize(edited_image, (orig_w, orig_h))
            target_pts = target_pts * torch.tensor([orig_w, orig_h], device=self.device)
            target_pts = (target_pts / torch.tensor([edit_w, edit_h], device=self.device)).long()

        image_h, image_w = original_image.shape[:2]
        
        # Resize to 768x768 (must be multiple of patch_size=16, 768/16=48)
        orig_img = F.interpolate(self.preprocess_image(original_image), size=(768, 768), mode='bilinear')
        edit_img = F.interpolate(self.preprocess_image(edited_image), size=(768, 768), mode='bilinear')

        # Extract DINOv3 features and upsample to original image size
        # DINOv3 output: (1, 768, 48, 48) for 768x768 input
        orig_feat = F.interpolate(self.dinov3_feat(orig_img), size=(image_h, image_w), mode='bilinear')
        edit_feat = F.interpolate(self.dinov3_feat(edit_img), size=(image_h, image_w), mode='bilinear')

        mask = self._create_mask(handle_pts, target_pts, (image_h, image_w))
        matched_pts = self._nn_get_matches(orig_feat, edit_feat, handle_pts, mask=mask)
        
        # Calculate distance metric
        dist = target_pts - matched_pts
        dist = dist.float() / torch.tensor([image_w, image_h], device=self.device)
        mean_dist = dist.norm(dim=-1).mean().item()

        if plot_path:
            self.plot_drag_result(
                edited_image,
                matched_pts.cpu().numpy(),
                target_pts.cpu().numpy(),
                output_path=plot_path
            )

        return mean_dist

    @staticmethod
    def plot_drag_result(edited_image: np.ndarray, handle_pts: np.ndarray, 
                        target_pts: np.ndarray, output_path: str = None):
        """
        Plot drag editing results with arrows showing point movements.
        
        Args:
            edited_image: Edited image array
            handle_pts: Handle point coordinates
            target_pts: Target point coordinates
            output_path: Optional path to save visualization
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(edited_image)
        
        # Convert points to numpy if needed
        if torch.is_tensor(handle_pts):
            handle_pts = handle_pts.cpu().numpy()
        if torch.is_tensor(target_pts):
            target_pts = target_pts.cpu().numpy()
        
        # Plot points and arrows
        plt.scatter(target_pts[:, 0], target_pts[:, 1], c='blue', label='Target Points')
        plt.scatter(handle_pts[:, 0], handle_pts[:, 1], c='red', label='Matched Points')
        
        for i in range(len(handle_pts)):
            plt.arrow(handle_pts[i, 0], handle_pts[i, 1],
                     target_pts[i, 0] - handle_pts[i, 0],
                     target_pts[i, 1] - handle_pts[i, 1],
                     color='white', head_width=5, head_length=5)
        
        plt.legend()
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def _create_mask(handle_pts: torch.Tensor, target_pts: torch.Tensor, 
                     img_size: tuple) -> torch.Tensor:
        """
        Create masks based on pixel distances to point pairs.
        
        Args:
            handle_pts: Handle point coordinates
            target_pts: Target point coordinates
            img_size: Image dimensions (H,W)
            
        Returns:
            torch.Tensor: Binary mask
        """
        handle_pts, target_pts = handle_pts.float(), target_pts.float()
        h, w = img_size
        
        min_dist = ((handle_pts - target_pts).norm(dim=1) / 2**0.5).clamp(min=5)
        
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=handle_pts.device),
            torch.arange(w, device=handle_pts.device),
            indexing="ij"
        )
        
        y_grid = y_grid.expand(len(handle_pts), -1, -1)
        x_grid = x_grid.expand(len(handle_pts), -1, -1)
        
        handle_dist = ((x_grid - handle_pts[:, None, None, 0])**2 + (y_grid - handle_pts[:, None, None, 1])**2).sqrt()
        target_dist = ((x_grid - target_pts[:, None, None, 0])**2 + (y_grid - target_pts[:, None, None, 1])**2).sqrt()
        
        return (handle_dist < min_dist[:, None, None]) | (target_dist < min_dist[:, None, None])

    @staticmethod
    def _nn_get_matches(src_featmaps: torch.Tensor, trg_featmaps: torch.Tensor, query: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Find nearest neighbor matches between source and target feature maps.
        
        Args:
            src_featmaps: Source feature maps
            trg_featmaps: Target feature maps
            query: Query points
            mask: Optional mask for valid matches
            
        Returns:
            torch.Tensor: Matched point coordinates
        """
        _, c, h, w = src_featmaps.shape
        query = query.long()
        src_feat = src_featmaps[0, :, query[:, 1], query[:, 0]]
        src_feat = F.normalize(src_feat, p=2, dim=0)

        trg_featmaps = F.normalize(trg_featmaps, p=2, dim=1)
        trg_featmaps = trg_featmaps.view(c, -1)
        similarity = torch.mm(src_feat.t(), trg_featmaps)
        
        if mask is not None:
            similarity = torch.where(
                mask.view(-1, h * w), 
                similarity, 
                torch.full_like(similarity, -torch.inf)
            )
        
        best_idx = similarity.argmax(dim=-1)
        y_coords = best_idx // w
        x_coords = best_idx % w
        
        return torch.stack((x_coords, y_coords), dim=1).float()