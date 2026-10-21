"""
dino_evaluator.py

DINOv3-based evaluation metrics for drag editing:
  - Global DINO Score     : CLS-token cosine similarity (source ↔ edited)
  - Local DINO Src        : patch-level NN matching under src_mask  (source → edited)
  - Local DINO Tgt        : patch-level NN matching under tgt_mask  (target → edited)
  - Local match visualization (optional)

Requires: transformers >= 4.56.0 for HF backend
          DINOv3 model weights access (gated model, see HuggingFace)
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Optional, Dict, Tuple
from pathlib import Path


# ======================================================================
#  DINOv3 Evaluator
# ======================================================================

class DINOv3Evaluator:
    """
    Evaluate image editing quality using DINOv3 features.

    Global DINO     : cosine similarity of CLS tokens  (src ↔ edit).
    Local  DINO Src : avg cosine sim of NN-matched patch tokens under src_mask.
    Local  DINO Tgt : avg cosine sim of NN-matched patch tokens under tgt_mask.

    Key DINOv3 differences from DINOv2:
      - patch_size = 16  (was 14 in DINOv2)
      - 4 register tokens between CLS and patch tokens
      - RoPE positional embeddings (variable resolution support)
      - Gram anchoring for stable dense features

    input_resolution:
      - None / 0  → use processor default (typically 224×224)
      - int > 0   → resize to this resolution (must be divisible by patch_size)
    """

    # ------------------------------------------------------------------
    #  Init / model loading
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_name: str = 'dinov3_vitb16',
        backend: str = 'hf',
        device: str = 'cuda',
        dtype: torch.dtype = torch.float32,
        input_resolution: Optional[int] = None,
    ):
        self.model_name = model_name
        self.backend = backend
        self.device = torch.device(device)
        self.dtype = dtype
        self.patch_size = 16
        self.num_register_tokens = 4
        self.input_resolution = input_resolution

        self.model, self.transform = self._load_model(model_name, backend)
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.eval()

        self._verify_model()

    # ── loaders ──────────────────────────────────────────────────────

    def _load_model(self, model_name: str, backend: str):
        if backend == 'hf':
            return self._load_from_hf(model_name)
        elif backend == 'torch_hub':
            return self._load_from_torch_hub(model_name)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _make_highres_transform(self, resolution: int, processor=None):
        from torchvision import transforms as T

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        if processor is not None:
            if hasattr(processor, 'image_mean') and processor.image_mean:
                mean = list(processor.image_mean)
            if hasattr(processor, 'image_std') and processor.image_std:
                std = list(processor.image_std)

        if resolution % self.patch_size != 0:
            old_res = resolution
            resolution = (resolution // self.patch_size) * self.patch_size
            print(f"[WARN] input_resolution {old_res} not divisible by "
                  f"patch_size {self.patch_size}, rounding down to {resolution}")

        print(f"[INFO] High-res transform: resize to {resolution}×{resolution}, "
              f"mean={mean}, std={std}")

        transform_fn = T.Compose([
            T.Resize(
                (resolution, resolution),
                interpolation=T.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
        return transform_fn

    def _load_from_hf(self, model_name: str):
        from transformers import AutoModel, AutoImageProcessor

        HF_MODEL_MAP = {
            'dinov3_vits16':     'facebook/dinov3-vits16-pretrain-lvd1689m',
            'dinov3_vitb16':     'facebook/dinov3-vitb16-pretrain-lvd1689m',
            'dinov3_vitl16':     'facebook/dinov3-vitl16-pretrain-lvd1689m',
            'dinov3_vith16plus': 'facebook/dinov3-vith16plus-pretrain-lvd1689m',
            'dinov3_vit7b16':    'facebook/dinov3-vit7b16-pretrain-lvd1689m',
            'dinov2_vits14': 'facebook/dinov2-small',
            'dinov2_vitb14': 'facebook/dinov2-base',
            'dinov2_vitl14': 'facebook/dinov2-large',
            'dinov2_vitg14': 'facebook/dinov2-giant',
        }
        model_id = HF_MODEL_MAP.get(model_name, model_name)
        print(f"[INFO] Loading HF model: {model_id}")

        processor = AutoImageProcessor.from_pretrained(model_id)
        hf_model = AutoModel.from_pretrained(model_id)

        config = hf_model.config
        if hasattr(config, 'patch_size'):
            self.patch_size = config.patch_size
            print(f"[INFO] Auto-detected patch_size = {self.patch_size}")

        if hasattr(config, 'num_register_tokens'):
            self.num_register_tokens = config.num_register_tokens
            print(f"[INFO] Auto-detected num_register_tokens = "
                  f"{self.num_register_tokens}")
        else:
            self.num_register_tokens = 0

        # ── build transform ──
        if self.input_resolution and self.input_resolution > 0:
            transform_fn = self._make_highres_transform(
                self.input_resolution, processor=processor
            )
        else:
            test_img = Image.new('RGB', (512, 512), (128, 128, 128))
            test_inputs = processor(images=test_img, return_tensors="pt")
            proc_shape = test_inputs['pixel_values'].shape
            print(f"[INFO] Using default processor, output shape: {proc_shape}")

            def transform_fn(img: Image.Image):
                inputs = processor(images=img, return_tensors="pt")
                return inputs['pixel_values'].squeeze(0)

        n_reg = self.num_register_tokens

        class _HFWrapper(torch.nn.Module):
            def __init__(self, m, num_register_tokens: int):
                super().__init__()
                self.m = m
                self.num_register_tokens = num_register_tokens

            def forward(self, x):
                out = self.m(pixel_values=x)
                hidden = out.last_hidden_state
                cls_tok = hidden[:, 0]
                pat_tok = hidden[:, 1 + self.num_register_tokens:]
                return cls_tok, pat_tok

        return _HFWrapper(hf_model, n_reg), transform_fn

    def _load_from_torch_hub(self, model_name: str):
        model = torch.hub.load('facebookresearch/dinov3', model_name)

        if hasattr(model, 'patch_size'):
            self.patch_size = model.patch_size
        if hasattr(model, 'num_register_tokens'):
            self.num_register_tokens = model.num_register_tokens

        if self.input_resolution and self.input_resolution > 0:
            transform_fn = self._make_highres_transform(self.input_resolution)
        else:
            from torchvision import transforms as T
            transform_fn = T.Compose([
                T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        class _HubWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                out = self.m.forward_features(x)
                cls_tok = out['x_norm_clstoken']
                pat_tok = out['x_norm_patchtokens']
                return cls_tok, pat_tok

        return _HubWrapper(model), transform_fn

    # ── verification ─────────────────────────────────────────────────

    def _verify_model(self):
        dummy = Image.new('RGB', (512, 512), (128, 128, 128))
        try:
            with torch.no_grad():
                pv = self.transform(dummy).unsqueeze(0).to(
                    self.device, dtype=self.dtype
                )
                cls_tok, pat_tok = self.model(pv)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            n_patches = pat_tok.shape[1]
            _, _, H, W = pv.shape

            grid_from_ps = (H // self.patch_size) * (W // self.patch_size)
            if grid_from_ps == n_patches:
                grid_h = H // self.patch_size
                grid_w = W // self.patch_size
            else:
                grid_side = int(round(n_patches ** 0.5))
                if grid_side * grid_side == n_patches:
                    grid_h = grid_w = grid_side
                    corrected_ps = H // grid_h
                    print(
                        f"[WARN] patch_size mismatch! correcting "
                        f"{self.patch_size} → {corrected_ps}"
                    )
                    self.patch_size = corrected_ps
                else:
                    grid_h = n_patches // (W // self.patch_size)
                    grid_w = W // self.patch_size

            self._proc_h = H
            self._proc_w = W
            self._verified_grid_h = grid_h
            self._verified_grid_w = grid_w
            self._verified_n_patches = n_patches

            print(
                f"[INFO] DINOv3 verified: input={H}×{W}, "
                f"patch_size={self.patch_size}, "
                f"num_register_tokens={self.num_register_tokens}, "
                f"grid={grid_h}×{grid_w}={n_patches} patches, "
                f"cls={cls_tok.shape}, patch={pat_tok.shape}"
            )

        except Exception as e:
            print(f"[ERROR] DINOv3 model verification failed: {e}")
            if torch.cuda.is_available():
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
            raise

    # ------------------------------------------------------------------
    #  Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def extract_features(
        self, image: Image.Image,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pv = self.transform(image).unsqueeze(0).to(
            self.device, dtype=self.dtype
        )
        return self.model(pv)

    def _get_patch_grid_size(
        self, image: Image.Image,
    ) -> Tuple[int, int]:
        if hasattr(self, '_verified_grid_h'):
            return self._verified_grid_h, self._verified_grid_w
        pv = self.transform(image).unsqueeze(0)
        _, _, H, W = pv.shape
        return H // self.patch_size, W // self.patch_size

    # ------------------------------------------------------------------
    #  Mask → patch mask
    # ------------------------------------------------------------------

    def _mask_to_patch_mask(
        self,
        mask: Image.Image,
        grid_h: int,
        grid_w: int,
        threshold: float = 0.3,
    ) -> torch.Tensor:
        proc_h = grid_h * self.patch_size
        proc_w = grid_w * self.patch_size
        mask_resized = mask.resize((proc_w, proc_h), Image.NEAREST)
        mask_np = np.array(mask_resized).astype(np.float32) / 255.0

        t = torch.from_numpy(mask_np).unsqueeze(0).unsqueeze(0)
        pooled = F.avg_pool2d(
            t, kernel_size=self.patch_size, stride=self.patch_size
        )
        return (pooled.squeeze() > threshold).flatten()

    # ------------------------------------------------------------------
    #  Metric computation
    # ------------------------------------------------------------------

    def compute_global_dino(
        self,
        src_cls: torch.Tensor,
        edit_cls: torch.Tensor,
    ) -> float:
        return float(
            F.cosine_similarity(src_cls, edit_cls, dim=-1).mean().cpu()
        )

    def compute_local_dino(
        self,
        ref_patches: torch.Tensor,
        edit_patches: torch.Tensor,
        patch_mask: torch.Tensor,
    ) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
        """
        For each *masked* reference patch, find its NN in the edited image.
        ref_patches can be from source or target image.
        """
        ref_p = F.normalize(ref_patches.squeeze(0), dim=-1)
        edit_p = F.normalize(edit_patches.squeeze(0), dim=-1)

        mask_idx = torch.where(patch_mask)[0]

        n_ref = ref_p.shape[0]
        if len(mask_idx) > 0:
            max_idx = int(mask_idx.max().item())
            if max_idx >= n_ref:
                print(f"  [WARN] mask index {max_idx} >= ref patches "
                      f"{n_ref}, clamping")
                mask_idx = mask_idx[mask_idx < n_ref]

        if len(mask_idx) == 0:
            empty = np.array([], dtype=np.float64)
            return 0.0, empty, empty, empty

        ref_masked = ref_p[mask_idx]
        sim_mat = torch.mm(ref_masked, edit_p.t())
        max_sims, max_ids = sim_mat.max(dim=1)

        return (
            float(max_sims.mean().cpu()),
            mask_idx.cpu().numpy(),
            max_ids.cpu().numpy(),
            max_sims.cpu().numpy(),
        )

    # ------------------------------------------------------------------
    #  Resolve grid for a given patch tensor
    # ------------------------------------------------------------------

    def _resolve_grid(
        self, patches: torch.Tensor, label: str = '',
    ) -> Optional[Tuple[int, int]]:
        """
        Given patch tokens (1, N, D), determine (grid_h, grid_w).
        Returns None if it cannot be resolved.
        """
        grid_h, grid_w = self._get_patch_grid_size(None)  # from verification
        expected = grid_h * grid_w
        actual = patches.shape[1]
        if expected == actual:
            return grid_h, grid_w
        # try square
        grid_side = int(round(actual ** 0.5))
        if grid_side * grid_side == actual:
            return grid_side, grid_side
        print(f"  [WARN] Cannot determine grid for {label} "
              f"(expected {expected}, got {actual}), skipping")
        return None

    def _get_patch_grid_size(
        self, image: Optional[Image.Image],
    ) -> Tuple[int, int]:
        if hasattr(self, '_verified_grid_h'):
            return self._verified_grid_h, self._verified_grid_w
        if image is None:
            raise RuntimeError("No verified grid and no image provided")
        pv = self.transform(image).unsqueeze(0)
        _, _, H, W = pv.shape
        return H // self.patch_size, W // self.patch_size

    # ------------------------------------------------------------------
    #  Public evaluate()
    # ------------------------------------------------------------------

    def evaluate(
        self,
        src_image: Image.Image,
        edit_image: Image.Image,
        src_mask: Optional[Image.Image] = None,
        tgt_image: Optional[Image.Image] = None,
        tgt_mask: Optional[Image.Image] = None,
        visualize_local: bool = False,
        vis_save_path: Optional[str] = None,
    ) -> Dict:
        """
        Evaluate edited image against source (and optionally target).

        Returns dict with keys:
          - 'global_dino'     : CLS cosine similarity (tgt ↔ edit)
          - 'local_dino_src'  : patch NN similarity under src_mask  (src → edit)
          - 'local_dino_tgt'  : patch NN similarity under tgt_mask  (tgt → edit)
        """
        # ── extract features ──
        src_cls, src_patches = self.extract_features(src_image)
        edit_cls, edit_patches = self.extract_features(edit_image)

        results: Dict = {}

        # # ── Global DINO (src ↔ edit) ──
        # results['global_dino'] = self.compute_global_dino(src_cls, edit_cls)

        grid_h, grid_w = self._get_patch_grid_size(src_image)

        # ── Local DINO Src (source patches under src_mask → edit) ──
        if src_mask is not None:
            grid_info = self._resolve_grid(src_patches, label='src')
            if grid_info is not None:
                gh, gw = grid_info
                patch_mask = self._mask_to_patch_mask(src_mask, gh, gw)
                score, src_idx, matched_idx, sims = self.compute_local_dino(
                    src_patches, edit_patches, patch_mask,
                )
                results['local_dino_src'] = score

                if visualize_local and len(src_idx) > 0:
                    save_p = None
                    if vis_save_path:
                        base = Path(vis_save_path)
                        save_p = str(base.parent / (base.stem + '_src' + base.suffix))
                    self.visualize_local_matches(
                        ref_image=src_image,
                        edit_image=edit_image,
                        ref_mask=src_mask,
                        ref_indices=src_idx,
                        matched_indices=matched_idx,
                        similarities=sims,
                        grid_h=gh, grid_w=gw,
                        title_prefix='Source→Edit',
                        save_path=save_p,
                    )

        # ── Local DINO Tgt (target patches under tgt_mask → edit) ──
        if tgt_image is not None and tgt_mask is not None:
            tgt_cls, tgt_patches = self.extract_features(tgt_image)
            # ── Global DINO (src ↔ edit) ──
            results['global_dino'] = self.compute_global_dino(tgt_cls, edit_cls)
            grid_info = self._resolve_grid(tgt_patches, label='tgt')
            if grid_info is not None:
                gh, gw = grid_info
                patch_mask = self._mask_to_patch_mask(tgt_mask, gh, gw)
                score, tgt_idx, matched_idx, sims = self.compute_local_dino(
                    tgt_patches, edit_patches, patch_mask,
                )
                results['local_dino_tgt'] = score

                if visualize_local and len(tgt_idx) > 0:
                    save_p = None
                    if vis_save_path:
                        base = Path(vis_save_path)
                        save_p = str(base.parent / (base.stem + '_tgt' + base.suffix))
                    self.visualize_local_matches(
                        ref_image=tgt_image,
                        edit_image=edit_image,
                        ref_mask=tgt_mask,
                        ref_indices=tgt_idx,
                        matched_indices=matched_idx,
                        similarities=sims,
                        grid_h=gh, grid_w=gw,
                        title_prefix='Target→Edit',
                        save_path=save_p,
                    )

        return results

    # ------------------------------------------------------------------
    #  ★  Local-DINO Visualization  (unified for src / tgt)
    # ------------------------------------------------------------------

    def visualize_local_matches(
        self,
        ref_image: Image.Image,
        edit_image: Image.Image,
        ref_mask: Image.Image,
        ref_indices: np.ndarray,
        matched_indices: np.ndarray,
        similarities: np.ndarray,
        grid_h: int,
        grid_w: int,
        title_prefix: str = 'Ref→Edit',
        save_path: Optional[str] = None,
    ):
        ps = self.patch_size
        proc_h, proc_w = grid_h * ps, grid_w * ps

        ref_disp = np.array(
            ref_image.resize((proc_w, proc_h), Image.LANCZOS)
        )
        edit_disp = np.array(
            edit_image.resize((proc_w, proc_h), Image.LANCZOS)
        )
        mask_disp = np.array(
            ref_mask.resize((proc_w, proc_h), Image.NEAREST)
        ).astype(np.float32) / 255.0

        ref_rows   = ref_indices // grid_w
        ref_cols   = ref_indices % grid_w
        match_rows = matched_indices // grid_w
        match_cols = matched_indices % grid_w

        ref_cx   = (ref_cols + 0.5) * ps
        ref_cy   = (ref_rows + 0.5) * ps
        match_cx = (match_cols + 0.5) * ps
        match_cy = (match_rows + 0.5) * ps

        sim_min = float(similarities.min())
        sim_max = float(similarities.max())
        eps = 1e-8
        if sim_max - sim_min < eps:
            sim_norm = np.ones_like(similarities)
        else:
            sim_norm = (similarities - sim_min) / (sim_max - sim_min)

        cmap = cm.get_cmap('RdYlBu_r')
        rgba = cmap(sim_norm)

        heat = np.zeros((grid_h, grid_w), dtype=np.float32)
        cnt  = np.zeros((grid_h, grid_w), dtype=np.float32)
        for i in range(len(matched_indices)):
            r, c = int(match_rows[i]), int(match_cols[i])
            heat[r, c] += similarities[i]
            cnt[r, c]  += 1.0
        valid = cnt > 0
        heat[valid] /= cnt[valid]

        heat_full = np.repeat(
            np.repeat(heat, ps, axis=0), ps, axis=1
        )
        valid_full = np.repeat(
            np.repeat(valid.astype(np.float32), ps, axis=0), ps, axis=1
        )

        n_matches = len(ref_indices)
        avg_sim = float(similarities.mean())

        fig, axes = plt.subplots(2, 2, figsize=(22, 18))

        # ── Panel 1: Reference with mask overlay ──
        ax1 = axes[0, 0]
        overlay = ref_disp.astype(np.float32).copy()
        mask_3d = np.stack([mask_disp] * 3, axis=-1)
        tint = np.zeros_like(overlay)
        tint[:, :, 0] = 255.0
        overlay = overlay * (1.0 - 0.3 * mask_3d) + tint * (0.3 * mask_3d)
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        ax1.imshow(overlay)
        ax1.scatter(
            ref_cx, ref_cy, c=rgba[:, :3], s=28,
            edgecolors='white', linewidths=0.5, zorder=5,
        )
        ax1.set_title(
            f'{title_prefix}  (mask + masked patch centres)',
            fontsize=13, fontweight='bold',
        )
        ax1.axis('off')

        # ── Panel 2: Edited with matched centres ──
        ax2 = axes[0, 1]
        ax2.imshow(edit_disp)
        sc = ax2.scatter(
            match_cx, match_cy,
            c=similarities, cmap='RdYlBu_r',
            vmin=sim_min, vmax=sim_max,
            s=28, edgecolors='white', linewidths=0.5, zorder=5,
        )
        plt.colorbar(
            sc, ax=ax2, fraction=0.046, pad=0.04,
            label='Cosine Similarity',
        )
        ax2.set_title(
            'Edited  (matched patch centres)',
            fontsize=13, fontweight='bold',
        )
        ax2.axis('off')

        # ── Panel 3: Heatmap on edited ──
        ax3 = axes[1, 0]
        edit_dark = (edit_disp.astype(np.float32) * 0.35).astype(np.uint8)
        ax3.imshow(edit_dark)

        heat_rgba = cmap(
            (heat_full - sim_min) / max(sim_max - sim_min, eps)
        )
        heat_rgba[..., 3] = valid_full * 0.75
        ax3.imshow(heat_rgba)

        sm = plt.cm.ScalarMappable(
            cmap='RdYlBu_r',
            norm=plt.Normalize(vmin=sim_min, vmax=sim_max),
        )
        plt.colorbar(
            sm, ax=ax3, fraction=0.046, pad=0.04,
            label='Cosine Similarity',
        )
        ax3.set_title(
            'Similarity Heatmap on Edited',
            fontsize=13, fontweight='bold',
        )
        ax3.axis('off')

        # ── Panel 4: Correspondence lines ──
        ax4 = axes[1, 1]
        gap = 16
        canvas = np.ones(
            (proc_h, proc_w * 2 + gap, 3), dtype=np.uint8
        ) * 180
        canvas[:, :proc_w] = ref_disp
        canvas[:, proc_w + gap:] = edit_disp
        ax4.imshow(canvas)

        max_lines = 120
        if n_matches > max_lines:
            order = np.argsort(-similarities)
            step = max(1, n_matches // max_lines)
            sel = order[::step][:max_lines]
        else:
            sel = np.arange(n_matches)

        for idx in sel:
            x1, y1 = float(ref_cx[idx]), float(ref_cy[idx])
            x2 = float(match_cx[idx]) + proc_w + gap
            y2 = float(match_cy[idx])
            ax4.plot(
                [x1, x2], [y1, y2],
                color=rgba[idx, :3], alpha=0.55, linewidth=0.8,
            )

        ax4.scatter(
            ref_cx[sel], ref_cy[sel],
            c=rgba[sel, :3], s=14,
            edgecolors='white', linewidths=0.3, zorder=5,
        )
        ax4.scatter(
            match_cx[sel] + proc_w + gap, match_cy[sel],
            c=rgba[sel, :3], s=14,
            edgecolors='white', linewidths=0.3, zorder=5,
        )
        ax4.set_title(
            f'Patch Correspondences  (showing {len(sel)}/{n_matches})',
            fontsize=13, fontweight='bold',
        )
        ax4.axis('off')

        fig.suptitle(
            f'{title_prefix}  Local DINOv3  │  score = {avg_sim:.4f}  │  '
            f'{n_matches} masked patches  │  '
            f'sim ∈ [{sim_min:.3f}, {sim_max:.3f}]',
            fontsize=15, fontweight='bold', y=0.98,
        )
        plt.tight_layout(rect=[0, 0, 1, 0.96])

        if save_path is not None:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"  [VIS] Local DINOv3 saved → {save_path}")
        else:
            plt.show()
        plt.close(fig)