"""
eval_drag.py

Evaluation pipeline for drag editing results.

Dataset layout (--data_dir), as released in tvedit-bench.zip:
    <data_dir>/
        tvedit-bench.jsonl
        tvedit-bench/
            <filename>/
                image1.jpg
                image2.jpg
                union_mask.png
            ...

Passing the inner sample folder also works when --jsonl_path is set.
Points come from the jsonl (sparse_point1/2, dense_point1/2), not from a pickle.

Result layout (--result_dir):
    <result_dir>/
        <filename>.jpg (or .png)
        ...

Metrics:
  - LPIPS (vs source / vs target)
  - MD (Mean Distance) – sparse & dense
  - Global DINO Score      (CLS-token cosine similarity, tgt ↔ edit)
  - Local DINO Src         (patch-level NN under src_mask, src → edit)
  - Local DINO Tgt         (patch-level NN under tgt_mask, tgt → edit)
"""

from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import json
import torch
import lpips as lpips_lib
import matplotlib.pyplot as plt
from evaluator import DragEvaluator
import traceback


# ======================================================================
#  CLI
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate drag editing results')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Unzipped benchmark root, or the folder of sample directories')
    parser.add_argument('--jsonl_path', type=str, default=None,
                        help='Annotation jsonl. Defaults to <data_dir>/tvedit-bench.jsonl')
    parser.add_argument('--result_dir', type=str, required=True,
                        help='Directory containing edited images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--plot', action='store_true',
                        help='Generate distance visualization plots')
    parser.add_argument('--plot_lpips', action='store_true',
                        help='Generate LPIPS spatial visualization plots')
    parser.add_argument('--plot_local_dino', action='store_true',
                        help='Generate local DINO match visualization for each sample')

    # ── DINO ──
    parser.add_argument('--no_dino', action='store_true',
                        help='Skip DINO metric computation entirely')
    parser.add_argument('--dino_model', type=str,
                        default='facebook/dinov3-vitb16-pretrain-lvd1689m',
                        help='DINOv3 model id or local path')
    parser.add_argument('--dino_backend', type=str, default='hf',
                        choices=['hf', 'torch_hub'],
                        help='DINO loading backend')
    parser.add_argument('--mask_dilate_radius', type=int, default=15,
                        help='Dilation radius when mask is generated from points as fallback')
    parser.add_argument('--dino_resolution', type=int, default=512,
                        help='Input resolution for DINO model. '
                             '0 = use default (224). '
                             '512 = high-res 32×32 grid for patch_size=16')

    return parser.parse_args()


# ======================================================================
#  Utilities
# ======================================================================

def load_annotations(path: Path) -> list[dict]:
    records = []
    with path.open(encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSON on line {line_num} of {path}: {exc}") from exc
    if not records:
        raise SystemExit(f"No records in {path}")
    return records


def find_jsonl(data_dir: Path, jsonl_path: str | None) -> Path:
    if jsonl_path:
        path = Path(jsonl_path)
        if not path.is_file():
            raise SystemExit(f"JSONL not found: {path}")
        return path
    preferred = data_dir / 'tvedit-bench.jsonl'
    if preferred.is_file():
        return preferred
    candidates = sorted(data_dir.glob('*.jsonl'))
    if len(candidates) == 1:
        return candidates[0]
    raise SystemExit(
        f"No annotation jsonl under {data_dir}. Pass --jsonl_path."
    )


def find_sample_root(data_dir: Path, filenames: list[str]) -> Path:
    def hits(root: Path) -> int:
        if not root.is_dir():
            return -1
        return sum((root / name).is_dir() for name in filenames)

    candidates = [data_dir]
    if data_dir.is_dir():
        candidates.extend(path for path in data_dir.iterdir() if path.is_dir())
    best = max(candidates, key=hits)
    if hits(best) == 0:
        raise SystemExit(f"No sample folders matching the jsonl under {data_dir}")
    return best


def as_points(raw) -> np.ndarray:
    if not raw:
        return np.zeros((0, 2), dtype=np.float64)
    return np.asarray(raw, dtype=np.float64).reshape(-1, 2)


def find_image(directory: Path, stem: str) -> Path | None:
    for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
        p = directory / f'{stem}{ext}'
        if p.exists():
            return p
    return None


def locate_named_image(sample_dir: Path, filename: str | None, fallback_stem: str) -> Path | None:
    if filename:
        direct = sample_dir / filename
        if direct.is_file():
            return direct
        found = find_image(sample_dir, Path(filename).stem)
        if found is not None:
            return found
    return find_image(sample_dir, fallback_stem)


def locate_union_mask(sample_dir: Path) -> Path | None:
    for name in ('union_mask.png', 'union_mask.jpg', 'union_mask.jpeg'):
        path = sample_dir / name
        if path.is_file():
            return path
    return None


def resize_image_to_match(img: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target_h and w == target_w:
        return img
    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((target_w, target_h), Image.LANCZOS)
    return np.array(pil_img)


def resize_mask_to_match(mask_pil: Image.Image, target_h: int, target_w: int) -> Image.Image:
    w, h = mask_pil.size
    if h == target_h and w == target_w:
        return mask_pil
    mask_pil = mask_pil.resize((target_w, target_h), Image.NEAREST)
    mask_np = np.array(mask_pil)
    mask_np = ((mask_np > 127).astype(np.uint8)) * 255
    return Image.fromarray(mask_np)


def rescale_points(points: np.ndarray,
                   src_h: int, src_w: int,
                   dst_h: int, dst_w: int) -> np.ndarray:
    if points.size == 0:
        return points
    if src_h == dst_h and src_w == dst_w:
        return points
    scale = np.array([dst_w / src_w, dst_h / src_h], dtype=np.float64)
    return points * scale


def generate_mask_from_points(
    points: np.ndarray,
    height: int,
    width: int,
    dilate_radius: int = 15,
) -> Image.Image:
    from PIL import ImageDraw
    mask_img = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(-1, 2)
    n = pts.shape[0]
    if n == 0:
        return mask_img

    hull_ok = False
    if n >= 3:
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
            polygon = [(float(p[0]), float(p[1])) for p in hull_pts]
            draw.polygon(polygon, fill=255)
            hull_ok = True
        except Exception:
            hull_ok = False

    if not hull_ok:
        r = max(dilate_radius, 8)
        for pt in pts:
            x, y = float(pt[0]), float(pt[1])
            draw.ellipse([x - r, y - r, x + r, y + r], fill=255)

    if dilate_radius > 0 and hull_ok:
        mask_np = np.array(mask_img)
        try:
            from scipy.ndimage import binary_dilation
            k = dilate_radius
            Y, X = np.ogrid[-k:k + 1, -k:k + 1]
            struct = (X ** 2 + Y ** 2 <= k ** 2)
            mask_np = binary_dilation(mask_np > 0, structure=struct).astype(np.uint8) * 255
        except ImportError:
            pass
        mask_img = Image.fromarray(mask_np)

    return mask_img


# ======================================================================
#  Evaluator Wrapper
# ======================================================================

class DragEvaluatorWrapper:
    """Handles evaluation of drag editing results"""

    def __init__(self, use_dino: bool = True,
                 dino_model: str = 'dinov3_vitb16',
                 dino_backend: str = 'hf',
                 dino_resolution: int = 0):
        # ── existing evaluators ──
        self.evaluator = DragEvaluator(dino_model=dino_model)
        self.lpips_spatial_fn = lpips_lib.LPIPS(net='alex', spatial=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lpips_spatial_fn = self.lpips_spatial_fn.to(self.device)

        # ── DINO evaluator (optional) ──
        self.dino_evaluator = None
        if use_dino:
            from dino_evaluator import DINOv3Evaluator
            print(f"[INFO] Loading DINO model: {dino_model} "
                  f"(backend={dino_backend}, resolution={dino_resolution}) ...")
            self.dino_evaluator = DINOv3Evaluator(
                model_name=dino_model,
                backend=dino_backend,
                device=str(self.device),
                dtype=torch.float32,
                input_resolution=dino_resolution if dino_resolution > 0 else None,
            )

    # ── helpers ───────────────────────────────────────────────────────

    def _img_to_tensor(self, img_np):
        t = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
        return t.to(self.device)

    def _compute_lpips_map(self, img1_np, img2_np):
        t1 = self._img_to_tensor(img1_np)
        t2 = self._img_to_tensor(img2_np)
        if t1.shape != t2.shape:
            t2 = torch.nn.functional.interpolate(
                t2, size=(t1.shape[2], t1.shape[3]),
                mode='bilinear', align_corners=False
            )
        with torch.no_grad():
            spatial_map = self.lpips_spatial_fn(t1, t2)
        return spatial_map.squeeze().cpu().numpy()

    def _plot_lpips_pair(self, map_source, map_target,
                         lpips_source_val, lpips_target_val,
                         source_img, target_img, edited_img,
                         output_path):
        vmin = min(float(map_source.min()), float(map_target.min()))
        vmax = max(float(map_source.max()), float(map_target.max()))
        if vmax - vmin < 1e-8:
            vmax = vmin + 1e-8
        cmap = 'hot'

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(map_source, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'LPIPS vs Source = {lpips_source_val:.4f}', fontsize=14)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(output_path / 'lpips_source_viz.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(map_target, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'LPIPS vs Target = {lpips_target_val:.4f}', fontsize=14)
        ax.axis('off')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(output_path / 'lpips_target_viz.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

        fig, axes = plt.subplots(2, 3, figsize=(21, 12))
        axes[0, 0].imshow(source_img);  axes[0, 0].set_title('Source Image', fontsize=13);  axes[0, 0].axis('off')
        axes[0, 1].imshow(edited_img);  axes[0, 1].set_title('Edited Image', fontsize=13);  axes[0, 1].axis('off')
        axes[0, 2].imshow(target_img);  axes[0, 2].set_title('Target Image', fontsize=13);  axes[0, 2].axis('off')
        im0 = axes[1, 0].imshow(map_source, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, 0].set_title(f'LPIPS vs Source = {lpips_source_val:.4f}', fontsize=13);  axes[1, 0].axis('off')
        axes[1, 1].axis('off')
        im1 = axes[1, 2].imshow(map_target, cmap=cmap, vmin=vmin, vmax=vmax)
        axes[1, 2].set_title(f'LPIPS vs Target = {lpips_target_val:.4f}', fontsize=13);  axes[1, 2].axis('off')
        cbar_ax = fig.add_axes([0.38, 0.08, 0.24, 0.02])
        fig.colorbar(im0, cax=cbar_ax, orientation='horizontal', label='LPIPS')
        plt.tight_layout(rect=[0, 0.12, 1, 1])
        plt.savefig(output_path / 'lpips_comparison.png', dpi=150, bbox_inches='tight')
        plt.close(fig)

    # ── core evaluation ───────────────────────────────────────────────

    def evaluate_sample(self, source_img_path, target_img_path, edited_img_path,
                        handle_points, target_points,
                        dense_handle_points, dense_target_points,
                        src_mask_pil=None,
                        tgt_mask_pil=None,
                        output_path=None, plot=False, plot_lpips=False,
                        plot_local_dino=False):
        """
        Evaluate a single edited image against source and target.

        所有图像统一 resize 到 target (GT) 尺寸后再计算指标；
        sparse / dense points 也做相应的坐标缩放。
        mask 由调用方提前 resize 到 target 分辨率后传入。
        """
        source_img = np.array(Image.open(source_img_path).convert('RGB'))
        target_img = np.array(Image.open(target_img_path).convert('RGB'))
        edited_img = np.array(Image.open(edited_img_path).convert('RGB'))

        # ---------- 以 target (GT) 尺寸为基准 ----------
        target_h, target_w = target_img.shape[:2]
        source_h, source_w = source_img.shape[:2]

        source_img = resize_image_to_match(source_img, target_h, target_w)
        edited_img = resize_image_to_match(edited_img, target_h, target_w)

        # ---------- 缩放 sparse points ----------
        handle_points = np.array(handle_points, dtype=np.float64).reshape(-1, 2)
        target_points = np.array(target_points, dtype=np.float64).reshape(-1, 2)
        handle_points = rescale_points(handle_points, source_h, source_w, target_h, target_w)

        # ---------- 缩放 dense points ----------
        dense_handle_points = np.array(dense_handle_points, dtype=np.float64).reshape(-1, 2)
        dense_target_points = np.array(dense_target_points, dtype=np.float64).reshape(-1, 2)
        dense_handle_points = rescale_points(dense_handle_points, source_h, source_w, target_h, target_w)

        # ---- LPIPS 标量 ----
        lpips_source = self.evaluator.compute_lpips(source_img, edited_img)
        lpips_target = self.evaluator.compute_lpips(target_img, edited_img)

        # ---- LPIPS 可视化 ----
        if plot_lpips and output_path:
            map_source = self._compute_lpips_map(source_img, edited_img)
            map_target = self._compute_lpips_map(target_img, edited_img)
            self._plot_lpips_pair(
                map_source, map_target,
                lpips_source, lpips_target,
                source_img, target_img, edited_img,
                output_path
            )

        # ---- MD sparse ----
        dist_plot_path = output_path / 'distance_viz.png' if (plot and output_path) else None
        distance = self.evaluator.compute_distance(
            source_img, edited_img,
            handle_points, target_points,
            plot_path=dist_plot_path
        )

        # ---- MD dense ----
        dense_dist_plot_path = output_path / 'distance_dense_viz.png' if (plot and output_path) else None
        distance_dense = self.evaluator.compute_distance(
            source_img, edited_img,
            dense_handle_points, dense_target_points,
            plot_path=dense_dist_plot_path
        )

        results = {
            'lpips_source': float(lpips_source),
            'lpips_target': float(lpips_target),
            'distance': float(distance),
            'distance_dense': float(distance_dense),
        }

        # ---- DINO metrics ----
        if self.dino_evaluator is not None:
            src_pil  = Image.fromarray(source_img)
            tgt_pil  = Image.fromarray(target_img)
            edit_pil = Image.fromarray(edited_img)

            vis_path = None
            if plot_local_dino and output_path:
                vis_path = str(output_path / 'local_dino_viz.png')

            dino_scores = self.dino_evaluator.evaluate(
                src_image=src_pil,
                edit_image=edit_pil,
                src_mask=src_mask_pil,
                tgt_image=tgt_pil,
                tgt_mask=tgt_mask_pil,
                visualize_local=plot_local_dino,
                vis_save_path=vis_path,
            )

            results['global_dino'] = dino_scores['global_dino']
            if 'local_dino_src' in dino_scores:
                results['local_dino_src'] = dino_scores['local_dino_src']
            if 'local_dino_tgt' in dino_scores:
                results['local_dino_tgt'] = dino_scores['local_dino_tgt']

        return results


# ======================================================================
#  Main
# ======================================================================

def main():
    args = parse_args()
    data_dir   = Path(args.data_dir)
    result_dir = Path(args.result_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── discover samples ──
    jsonl_path = find_jsonl(data_dir, args.jsonl_path)
    records = load_annotations(jsonl_path)
    filenames = [rec.get('filename', '') for rec in records if rec.get('filename')]
    sample_root = find_sample_root(data_dir, filenames)
    print(f"JSONL: {jsonl_path} ({len(records)} records)")
    print(f"Sample root: {sample_root}")

    if len(records) == 0:
        print("[ERROR] No valid samples found. Exiting.")
        return

    # ── init evaluator ──
    use_dino = not args.no_dino
    evaluator = DragEvaluatorWrapper(
        use_dino=use_dino,
        dino_model=args.dino_model,
        dino_backend=args.dino_backend,
        dino_resolution=args.dino_resolution,
    )
    need_plot = args.plot or args.plot_lpips or args.plot_local_dino

    # ── evaluate ──
    metrics = []
    skipped = []
    errors  = []
    mask_stats = {
        'src_from_file': 0, 'src_from_points': 0, 'src_missing': 0,
        'tgt_from_file': 0, 'tgt_from_points': 0, 'tgt_missing': 0,
    }

    for rec in tqdm(records, desc="Evaluating"):
        sample_name = rec.get('filename', '')
        if not sample_name:
            print("[WARN] jsonl record missing filename")
            continue
        sample_dir = sample_root / sample_name
        if not sample_dir.is_dir():
            print(f"[WARN] {sample_name}: sample folder not found")
            skipped.append(sample_name)
            continue

        source_path = locate_named_image(sample_dir, rec.get('image1'), 'image1')
        target_path = locate_named_image(sample_dir, rec.get('image2'), 'image2')

        if source_path is None or target_path is None:
            print(f"[WARN] {sample_name}: cannot find image1/image2")
            skipped.append(sample_name)
            continue

        edited_path = find_image(result_dir, sample_name)
        if edited_path is None:
            print(f"[WARN] {sample_name}: edited result not found in {result_dir}")
            skipped.append(sample_name)
            continue

        sparse_handle = as_points(rec.get('sparse_point1'))
        sparse_target = as_points(rec.get('sparse_point2'))
        dense_handle = as_points(rec.get('dense_point1'))
        dense_target = as_points(rec.get('dense_point2'))
        if sparse_handle.size == 0 and dense_handle.size == 0:
            print(f"[WARN] {sample_name}: no sparse or dense points in jsonl")

        # ── output subdir for plots ──
        output_subdir = output_dir / sample_name if need_plot else None
        if output_subdir:
            output_subdir.mkdir(parents=True, exist_ok=True)

        # ── resolve masks ──
        src_mask_pil = None
        tgt_mask_pil = None

        if use_dino:
            with Image.open(target_path) as _tmp:
                _target_w, _target_h = _tmp.size
            with Image.open(source_path) as _tmp:
                _source_w, _source_h = _tmp.size

            union_path = locate_union_mask(sample_dir)
            if union_path is not None:
                union_mask = Image.open(union_path).convert('L')
                union_mask = resize_mask_to_match(union_mask, _target_h, _target_w)
                src_mask_pil = union_mask
                tgt_mask_pil = union_mask
                mask_stats['src_from_file'] += 1
                mask_stats['tgt_from_file'] += 1
            else:
                if dense_handle.size > 0:
                    pts_rescaled = rescale_points(
                        dense_handle, _source_h, _source_w, _target_h, _target_w
                    )
                    src_mask_pil = generate_mask_from_points(
                        pts_rescaled, _target_h, _target_w,
                        dilate_radius=args.mask_dilate_radius,
                    )
                    mask_stats['src_from_points'] += 1
                else:
                    mask_stats['src_missing'] += 1
                if dense_target.size > 0:
                    pts_rescaled = rescale_points(
                        dense_target, _source_h, _source_w, _target_h, _target_w
                    )
                    tgt_mask_pil = generate_mask_from_points(
                        pts_rescaled, _target_h, _target_w,
                        dilate_radius=args.mask_dilate_radius,
                    )
                    mask_stats['tgt_from_points'] += 1
                else:
                    mask_stats['tgt_missing'] += 1

        # ── evaluate ──
        try:
            result = evaluator.evaluate_sample(
                source_path, target_path, edited_path,
                handle_points=sparse_handle,
                target_points=sparse_target,
                dense_handle_points=dense_handle,
                dense_target_points=dense_target,
                src_mask_pil=src_mask_pil,
                tgt_mask_pil=tgt_mask_pil,
                output_path=output_subdir,
                plot=args.plot,
                plot_lpips=args.plot_lpips,
                plot_local_dino=args.plot_local_dino,
            )
            result['sample_name'] = sample_name
            result['n_sparse_pairs'] = sparse_handle.shape[0]
            result['n_dense_pairs']  = dense_handle.shape[0]
            metrics.append(result)
        except Exception as e:
            print(f"[ERROR] {sample_name}: {e}")
            traceback.print_exc()
            errors.append({'sample': sample_name, 'error': str(e)})

    # ── compute averages ──
    if len(metrics) > 0:
        avg_metrics = {
            'avg_lpips_source':   float(np.mean([m['lpips_source']   for m in metrics])),
            'avg_lpips_target':   float(np.mean([m['lpips_target']   for m in metrics])),
            'avg_distance':       float(np.mean([m['distance']       for m in metrics])),
            'avg_distance_dense': float(np.mean([m['distance_dense'] for m in metrics])),
        }
        global_dino_vals     = [m['global_dino']     for m in metrics if 'global_dino'     in m]
        local_dino_src_vals  = [m['local_dino_src']  for m in metrics if 'local_dino_src'  in m]
        local_dino_tgt_vals  = [m['local_dino_tgt']  for m in metrics if 'local_dino_tgt'  in m]

        if global_dino_vals:
            avg_metrics['avg_global_dino']     = float(np.mean(global_dino_vals))
            avg_metrics['num_global_dino']     = len(global_dino_vals)
        if local_dino_src_vals:
            avg_metrics['avg_local_dino_src']  = float(np.mean(local_dino_src_vals))
            avg_metrics['num_local_dino_src']  = len(local_dino_src_vals)
        if local_dino_tgt_vals:
            avg_metrics['avg_local_dino_tgt']  = float(np.mean(local_dino_tgt_vals))
            avg_metrics['num_local_dino_tgt']  = len(local_dino_tgt_vals)
    else:
        avg_metrics = {
            'avg_lpips_source': 0.0,
            'avg_lpips_target': 0.0,
            'avg_distance': 0.0,
            'avg_distance_dense': 0.0,
        }

    all_results = {
        'individual_metrics': metrics,
        'average_metrics': avg_metrics,
        'info': {
            'total_samples': len(records),
            'evaluated_samples': len(metrics),
            'skipped_samples': skipped,
            'error_samples': errors,
            'data_dir': str(data_dir),
            'sample_root': str(sample_root),
            'jsonl_path': str(jsonl_path),
            'result_dir': str(result_dir),
            'dino_enabled': use_dino,
            'dino_model': args.dino_model if use_dino else None,
            'dino_resolution': args.dino_resolution if use_dino else None,
            'mask_stats': mask_stats if use_dino else None,
        }
    }

    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

    # ── print summary ──
    print(f"\n{'=' * 60}")
    print(f"  Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  LPIPS (vs source):   {avg_metrics['avg_lpips_source']:.4f}")
    print(f"  LPIPS (vs target):   {avg_metrics['avg_lpips_target']:.4f}")
    print(f"  MD (sparse):         {avg_metrics['avg_distance']:.4f}")
    print(f"  MD (dense):          {avg_metrics['avg_distance_dense']:.4f}")

    if 'avg_global_dino' in avg_metrics:
        print(f"  Global DINO:         {avg_metrics['avg_global_dino']:.4f}  "
              f"({avg_metrics['num_global_dino']} samples)")
    if 'avg_local_dino_src' in avg_metrics:
        print(f"  Local DINO (src):    {avg_metrics['avg_local_dino_src']:.4f}  "
              f"({avg_metrics['num_local_dino_src']} samples)")
    if 'avg_local_dino_tgt' in avg_metrics:
        print(f"  Local DINO (tgt):    {avg_metrics['avg_local_dino_tgt']:.4f}  "
              f"({avg_metrics['num_local_dino_tgt']} samples)")

    print(f"{'─' * 60}")
    print(f"  Evaluated: {len(metrics)} / {len(records)} samples")
    if skipped:
        print(f"  Skipped (missing files): {len(skipped)}")
    if errors:
        print(f"  Errors: {len(errors)}")
    if use_dino:
        print(f"  Src masks: {mask_stats['src_from_file']} file, "
              f"{mask_stats['src_from_points']} points, "
              f"{mask_stats['src_missing']} missing")
        print(f"  Tgt masks: {mask_stats['tgt_from_file']} file, "
              f"{mask_stats['tgt_from_points']} points, "
              f"{mask_stats['tgt_missing']} missing")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()