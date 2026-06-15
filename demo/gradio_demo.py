#!/usr/bin/env python3
"""
Gradio Interactive Demo for Drag-based Image Editing
with optional Lightning LoRA acceleration.

Compatible with Gradio 5.34.2
"""

import sys
import os
import math
import argparse
import traceback
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

import gradio as gr
import torch
import numpy as np
from PIL import Image, ImageDraw
import yaml

from diffusers import FlowMatchEulerDiscreteScheduler
from tvedit_qwen.tv_edit_model import TVEditModel


# ═══════════════════════════ Utility Functions ═══════════════════════════

def remap_coordinates(coordinates, original_size):
    coords = coordinates.astype(np.float64).copy()
    coords[:, 0] = coords[:, 0] / original_size[0] * 1024
    coords[:, 1] = coords[:, 1] / original_size[1] * 1024
    return coords


def point2disk(points, H, W, device='cpu'):
    points = torch.round(points).to(device)
    disk_map = torch.zeros((H, W)).long().to(device)
    if len(points) == 0:
        return disk_map.unsqueeze(0)
    idx = torch.arange(len(points)).to(device) + 1
    disk_map[points[:, 1].long(), points[:, 0].long()] = idx
    return disk_map.unsqueeze(0)


def draw_arrow(draw, start, end, color, width):
    sx, sy = start
    tx, ty = end
    draw.line([(sx, sy), (tx, ty)], fill=color, width=width)
    dx, dy = tx - sx, ty - sy
    length = math.hypot(dx, dy)
    if length < 1:
        return
    ux, uy = dx / length, dy / length
    arrow_size = min(15, length * 0.3)
    angle = math.pi / 6
    ax1 = tx - arrow_size * (ux * math.cos(angle) + uy * math.sin(angle))
    ay1 = ty - arrow_size * (uy * math.cos(angle) - ux * math.sin(angle))
    ax2 = tx - arrow_size * (ux * math.cos(angle) - uy * math.sin(angle))
    ay2 = ty - arrow_size * (uy * math.cos(angle) + ux * math.sin(angle))
    draw.polygon([(tx, ty), (ax1, ay1), (ax2, ay2)], fill=color)


def draw_annotations(image, src_points, tgt_points, mode="source"):
    """在图像上绘制 src(红)、tgt(蓝)、箭头(黄)"""
    if image is None:
        return None
    img = image.copy()
    draw = ImageDraw.Draw(img)
    w, h = img.size
    r = max(6, min(w, h) // 80)
    lw = max(2, r // 3)

    n_pairs = min(len(src_points), len(tgt_points))

    # 已配对: source → target 箭头
    for i in range(n_pairs):
        s, t = src_points[i], tgt_points[i]
        draw_arrow(draw, s, t, (255, 230, 0), lw)
        draw.ellipse([s[0]-r, s[1]-r, s[0]+r, s[1]+r],
                     fill=(235, 64, 52), outline='white', width=2)
        draw.ellipse([t[0]-r, t[1]-r, t[0]+r, t[1]+r],
                     fill=(52, 100, 235), outline='white', width=2)
        draw.text((s[0]+r+3, s[1]-r), str(i+1), fill='white')
        draw.text((t[0]+r+3, t[1]-r), str(i+1), fill='white')

    # 未配对的 source 点
    for i in range(n_pairs, len(src_points)):
        s = src_points[i]
        draw.ellipse([s[0]-r-3, s[1]-r-3, s[0]+r+3, s[1]+r+3],
                     outline=(255, 230, 0), width=3)
        draw.ellipse([s[0]-r, s[1]-r, s[0]+r, s[1]+r],
                     fill=(235, 64, 52), outline='white', width=2)
        draw.text((s[0]+r+3, s[1]-r), str(i+1), fill='yellow')

    # 在画布上方显示当前状态提示
    mode_text = "🔴 Click: SOURCE point" if mode == "source" else "🔵 Click: TARGET point"
    try:
        draw.text((10, 10), mode_text, fill='yellow')
    except Exception:
        pass

    return img


# ═══════════════════════════ Model Holder ═══════════════════════════

class ModelHolder:
    model = None
    pipe = None
    device = "cuda:0"
    loaded = False
    lightning_enabled = False
    original_scheduler = None

MH = ModelHolder()


# ═══════════════════════════ 蒸馏 Scheduler 配置 ═══════════════════════════

DISTILL_SCHEDULER_CONFIG = {
    "base_image_seq_len": 256,
    "base_shift": math.log(3),
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": math.log(3),
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}


def load_model_fn(model_id, model_path, gpu_id,
                  enable_lightning, lightning_lora_path):
    """加载模型，可选启用 Lightning LoRA 加速"""
    try:
        if not model_id or not model_id.strip():
            return "❌ Please provide a Model ID (base model)."
        if not model_path or not model_path.strip():
            return "❌ Please provide a Model Path (trained weights)."

        device = f"cuda:{int(gpu_id)}"

        # ========== 1. 加载基础 TVEditModel ==========
        MH.model = TVEditModel(
            model_id=model_id.strip(),
            device=device,
            weight_path=model_path.strip(),
        )
        MH.pipe = MH.model.qwen_pipe
        MH.model.eval()
        MH.pipe = MH.pipe.to(device)
        MH.device = device
        MH.loaded = True

        MH.original_scheduler = MH.pipe.scheduler

        # ========== 2. 可选：加载 Lightning LoRA 加速 ==========
        MH.lightning_enabled = False

        if enable_lightning and lightning_lora_path and lightning_lora_path.strip():
            lora_path = lightning_lora_path.strip()
            if not os.path.exists(lora_path):
                return (f"✅ Base model loaded on {device}\n"
                        f"⚠️ Lightning LoRA file not found: {lora_path}")

            distill_scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                DISTILL_SCHEDULER_CONFIG
            )
            MH.pipe.scheduler = distill_scheduler

            MH.pipe.load_lora_weights(lora_path)

            MH.lightning_enabled = True
            return (f"✅ Model loaded on {device}\n"
                    f"⚡ Lightning LoRA loaded: {os.path.basename(lora_path)}\n"
                    f"   推荐参数: steps=4, CFG=1.0")

        return f"✅ Model loaded on {device} (标准模式，无加速 LoRA)"

    except Exception as e:
        traceback.print_exc()
        return f"❌ Failed: {e}"


# ═══════════════════════════ Event Handlers ═══════════════════════════

def on_upload_image(image):
    """
    用户上传图片到画布 → 初始化状态
    返回顺序对应 outputs：canvas, st_orig, st_src, st_tgt, st_mode, info, pts_text
    """
    if image is None:
        return None, None, [], [], "source", "Upload an image to begin.", "No points yet."
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    annotated = draw_annotations(image, [], [], "source")
    info = (f"✅ Image loaded ({image.size[0]}×{image.size[1]}). "
            f"Now 🔴 click on the image to place source point #1.")
    return annotated, image, [], [], "source", info, "No points yet."


def on_canvas_select(original_img, src, tgt, mode, evt: gr.SelectData):
    """用户点击画布 → 交替添加 source / target 点"""
    if original_img is None:
        return gr.update(), src, tgt, mode, "⚠️ Please upload an image first.", ""

    x, y = int(evt.index[0]), int(evt.index[1])

    if mode == "source":
        src = src + [[x, y]]
        mode = "target"
        info = (f"🔴 Source #{len(src)} placed at ({x}, {y}). "
                f"Now click the 🔵 target position for this point.")
    else:
        tgt = tgt + [[x, y]]
        mode = "source"
        n = len(tgt)
        info = (f"🔵 Target #{n} placed at ({x}, {y}). "
                f"Pair #{n} complete ✓  Add more pairs or press ▶️ Run.")

    annotated = draw_annotations(original_img, src, tgt, mode)
    pts_text = format_points_text(src, tgt)
    return annotated, src, tgt, mode, info, pts_text


def on_undo(original_img, src, tgt, mode):
    """撤销上一个点"""
    if mode == "target" and len(src) > len(tgt):
        src = src[:-1]
        mode = "source"
    elif mode == "source" and len(tgt) > 0 and len(src) == len(tgt):
        tgt = tgt[:-1]
        mode = "target"
    else:
        return (
            draw_annotations(original_img, src, tgt, mode) if original_img else None,
            src, tgt, mode, "Nothing to undo.",
            format_points_text(src, tgt)
        )

    img = draw_annotations(original_img, src, tgt, mode) if original_img else None
    info = f"↩️ Undo done. Pairs: {min(len(src), len(tgt))}, mode: {mode}"
    return img, src, tgt, mode, info, format_points_text(src, tgt)


def on_clear_points(original_img):
    """清空所有点"""
    img = draw_annotations(original_img, [], [], "source") if original_img else None
    return img, [], [], "source", "🗑️ All points cleared. 🔴 Click to add source #1.", "No points yet."


def format_points_text(src, tgt):
    """生成当前点的文本摘要"""
    if len(src) == 0:
        return "No points yet."
    lines = []
    for i in range(len(src)):
        s = src[i]
        if i < len(tgt):
            t = tgt[i]
            lines.append(f"  Pair {i+1}: ({s[0]},{s[1]}) → ({t[0]},{t[1]})")
        else:
            lines.append(f"  Pair {i+1}: ({s[0]},{s[1]}) → ???")
    return "\n".join(lines)


def on_add_manual(original_img, src, tgt, mode, sx, sy, tx, ty):
    """手动输入坐标添加一对点（备用方式）"""
    if original_img is None:
        return None, src, tgt, mode, "⚠️ Upload image first.", ""

    src = src + [[int(sx), int(sy)]]
    tgt = tgt + [[int(tx), int(ty)]]
    mode = "source"

    annotated = draw_annotations(original_img, src, tgt, mode)
    n = len(tgt)
    info = f"✏️ Manually added pair #{n}: ({int(sx)},{int(sy)}) → ({int(tx)},{int(ty)})"
    return annotated, src, tgt, mode, info, format_points_text(src, tgt)


def on_toggle_lightning(enabled):
    """勾选/取消 Lightning 时自动切换推荐参数"""
    if enabled:
        return gr.update(value=4), gr.update(value=1.0)
    else:
        return gr.update(value=50), gr.update(value=3.0)


@torch.inference_mode()
def on_run(original_img, src, tgt, instruction, seed, cfg_scale, num_steps):
    """执行推理"""
    if not MH.loaded:
        return None, "❌ Model not loaded. Open ⚙️ Model Configuration and click Load."
    if original_img is None:
        return None, "❌ No image uploaded."
    if len(src) == 0:
        return None, "❌ Add at least one source-target point pair."
    if len(src) != len(tgt):
        return None, f"❌ Incomplete pair: {len(src)} source vs {len(tgt)} target."

    try:
        device = MH.device
        orig_size = original_img.size

        src_np = remap_coordinates(np.array(src, dtype=np.float64), orig_size)
        tgt_np = remap_coordinates(np.array(tgt, dtype=np.float64), orig_size)

        img_1024 = original_img.resize((1024, 1024))

        src_disk = point2disk(torch.tensor(src_np).float(), 1024, 1024, device)
        tgt_disk = point2disk(torch.tensor(tgt_np).float(), 1024, 1024, device)

        seed_val = int(seed)
        gen = (torch.Generator(device).manual_seed(seed_val)
               if seed_val >= 0 else None)

        pipe_kwargs = dict(
            prompt=instruction,
            image=img_1024,
            true_cfg_scale=cfg_scale,
            num_inference_steps=int(num_steps),
            generator=gen,
            point_encoder=MH.model.point_encoder,
            src_disk=src_disk.unsqueeze(0),
            tgt_disk=tgt_disk.unsqueeze(0),
            controlnet=MH.model.casc,
        )

        if MH.lightning_enabled:
            pipe_kwargs["negative_prompt"] = " "

        result = MH.pipe(**pipe_kwargs).images[0]

        mode_str = "⚡Lightning" if MH.lightning_enabled else "Standard"
        msg = (f"✅ Done! [{mode_str}] {len(src)} pairs, "
               f"{int(num_steps)} steps, CFG {cfg_scale}, seed {seed_val}")
        return result, msg

    except Exception as e:
        traceback.print_exc()
        return None, f"❌ Inference error: {e}"


def on_save_results(original_img, result_img,
                    src, tgt, instruction, seed, cfg_scale, num_steps, save_dir):
    """保存结果到指定目录"""
    if result_img is None:
        return "❌ No result to save. Run editing first."

    try:
        save_dir = save_dir.strip() if save_dir else "./outputs"
        os.makedirs(save_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        prefix = os.path.join(save_dir, timestamp)

        saved_files = []

        if original_img is not None:
            path = f"{prefix}_input.png"
            original_img.save(path)
            saved_files.append(path)

        if result_img is not None:
            path = f"{prefix}_result.png"
            result_img.save(path)
            saved_files.append(path)

        meta_path = f"{prefix}_meta.txt"
        with open(meta_path, "w") as f:
            f.write(f"timestamp: {timestamp}\n")
            f.write(f"instruction: {instruction}\n")
            f.write(f"seed: {seed}\n")
            f.write(f"cfg_scale: {cfg_scale}\n")
            f.write(f"num_steps: {num_steps}\n")
            f.write(f"lightning: {MH.lightning_enabled}\n")
            if original_img is not None:
                f.write(f"original_size: {original_img.size}\n")
            f.write(f"num_pairs: {min(len(src), len(tgt))}\n")
            for i in range(min(len(src), len(tgt))):
                s, t = src[i], tgt[i]
                f.write(f"pair_{i+1}: ({s[0]},{s[1]}) -> ({t[0]},{t[1]})\n")
        saved_files.append(meta_path)

        file_list = "\n".join([f"  📄 {os.path.basename(f)}" for f in saved_files])
        return f"✅ Saved {len(saved_files)} files to {save_dir}/\n{file_list}"

    except Exception as e:
        traceback.print_exc()
        return f"❌ Save error: {e}"


# ═══════════════════════════ Build UI ═══════════════════════════

def build_demo(default_model_id, default_model_path):

    with gr.Blocks(title="Drag Image Edit", theme=gr.themes.Soft()) as demo:

        # ---- Hidden States ----
        st_orig = gr.State(None)
        st_src  = gr.State([])
        st_tgt  = gr.State([])
        st_mode = gr.State("source")

        # ---- Header ----
        gr.Markdown("# 🎨  Text Vision Co-Instructed Image Editing")
        gr.Markdown(
            "**Steps:** ① Upload image to canvas  →  ② 🔴 Click source  →  "
            "③ 🔵 Click target  →  ④ Repeat  →  "
            "⑤ Enter instruction  →  ⑥ ▶️ Run"
        )

        # ---- Model Config ----
        with gr.Accordion("⚙️  Model Configuration", open=True):
            with gr.Row():
                ui_model_id = gr.Textbox(
                    label="Model ID (base model)",
                    value=default_model_id,
                    placeholder="e.g. Qwen/Qwen-Image-Edit or /path/to/base_model",
                    scale=3,
                )
            with gr.Row():
                ui_model_path = gr.Textbox(
                    label="Model Path (trained weights)",
                    value=default_model_path,
                    placeholder="e.g. /path/to/train/runs/xxxx",
                    scale=3,
                )
                ui_gpu = gr.Number(label="GPU ID", value=0,
                                   precision=0, scale=1)

            gr.Markdown("#### ⚡ Lightning LoRA Acceleration (Optional)")
            gr.Markdown(
                "启用后可将推理步数从 50 降低到 **4 步**，大幅加速生成。"
                "需提供 Lightning LoRA 的 `.safetensors` 文件路径。"
            )
            with gr.Row():
                ui_enable_lightning = gr.Checkbox(
                    label="Enable Lightning LoRA",
                    value=False,
                    scale=1,
                )
                ui_lightning_path = gr.Textbox(
                    label="Lightning LoRA Path (.safetensors)",
                    value="",
                    placeholder="e.g. Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors",
                    scale=4,
                )

            ui_load = gr.Button("🚀 Load Model", variant="primary")
            ui_mstatus = gr.Textbox(label="Model Status", interactive=False, lines=3)

            ui_load.click(
                load_model_fn,
                inputs=[ui_model_id, ui_model_path, ui_gpu,
                        ui_enable_lightning, ui_lightning_path],
                outputs=[ui_mstatus],
            )

        # ---- Main Layout ----
        with gr.Row(equal_height=True):

            # ======== 左列：输入画布 ========
            with gr.Column():
                gr.Markdown("### 📷  Input Canvas")
                gr.Markdown(
                    "👇 **Drag & drop or click below to upload, "
                    "then click on the image to add points** "
                    "(Red = Source, Blue = Target)"
                )

                # ✅ Gradio 5.x: 单一 Image 组件，既可上传又可点击
                ui_canvas = gr.Image(
                    type="pil",
                    label="Upload & Click Canvas",
                    sources=["upload", "clipboard"],
                    interactive=True,                  # ✅ 必须 True 才能 select
                    height=512,
                    show_download_button=False,
                    show_share_button=False,
                    show_fullscreen_button=False,      # 关闭工具栏避免干扰
                )

                ui_info = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value="Upload an image to begin.",
                    lines=2,
                )

                ui_pts_text = gr.Textbox(
                    label="Current Points",
                    interactive=False,
                    value="No points yet.",
                    lines=3,
                )

                with gr.Row():
                    ui_undo    = gr.Button("↩️ Undo Last")
                    ui_clear_p = gr.Button("🗑️ Clear Points")

                # 手动输入坐标（备用）
                with gr.Accordion("✏️  Manual Coordinate Input (fallback)", open=False):
                    gr.Markdown("If clicking doesn't work, manually enter coordinates here:")
                    with gr.Row():
                        ui_sx = gr.Number(label="Src X", value=0, precision=0)
                        ui_sy = gr.Number(label="Src Y", value=0, precision=0)
                        ui_tx = gr.Number(label="Tgt X", value=0, precision=0)
                        ui_ty = gr.Number(label="Tgt Y", value=0, precision=0)
                    ui_add_manual = gr.Button("➕ Add This Pair")

            # ======== 右列：输出 ========
            with gr.Column():
                gr.Markdown("### 🖼️  Result")
                ui_out = gr.Image(
                    type="pil", height=512,
                    interactive=False, label="Edited Output",
                    show_download_button=True,
                )
                ui_rstatus = gr.Textbox(label="Result", interactive=False)

                with gr.Row():
                    ui_save_dir = gr.Textbox(
                        label="Save Directory",
                        value="./outputs",
                        placeholder="e.g. ./outputs or /path/to/save",
                        scale=3,
                    )
                    ui_save_btn = gr.Button("💾 Save Results", variant="secondary", scale=1)
                ui_save_status = gr.Textbox(label="Save Status", interactive=False, lines=4)

        # ---- Parameters ----
        with gr.Row():
            ui_instr = gr.Textbox(
                label="Instruction",
                placeholder="e.g. 'move the cat to the right'",
                lines=2, scale=4,
            )
            ui_seed  = gr.Number(label="Seed (-1=random)",
                                 value=42, precision=0, scale=1)
            ui_cfg   = gr.Slider(minimum=1, maximum=10, value=3.0,
                                 step=0.5, label="CFG Scale (⚡Lightning → 1.0)",
                                 scale=1)
            ui_nstep = gr.Slider(minimum=1, maximum=100, value=50,
                                 step=1, label="Steps (⚡Lightning → 4)",
                                 scale=1)

        ui_run = gr.Button("▶️  Run Editing", variant="primary", size="lg")

        # ═══════════════ Event Wiring ═══════════════

        # 用户上传新图（拖拽 / 点击上传 / 粘贴）→ 初始化状态并把带标注的图回写画布
        ui_canvas.upload(
            fn=on_upload_image,
            inputs=[ui_canvas],
            outputs=[ui_canvas, st_orig, st_src, st_tgt, st_mode, ui_info, ui_pts_text],
        )

        # 用户清空画布（点 X 按钮）→ 重置 state
        ui_canvas.clear(
            fn=lambda: (None, [], [], "source", "Upload an image to begin.", "No points yet."),
            inputs=None,
            outputs=[st_orig, st_src, st_tgt, st_mode, ui_info, ui_pts_text],
        )

        # 用户点击画布 → 添加点
        ui_canvas.select(
            fn=on_canvas_select,
            inputs=[st_orig, st_src, st_tgt, st_mode],
            outputs=[ui_canvas, st_src, st_tgt, st_mode, ui_info, ui_pts_text],
        )

        # 撤销
        ui_undo.click(
            fn=on_undo,
            inputs=[st_orig, st_src, st_tgt, st_mode],
            outputs=[ui_canvas, st_src, st_tgt, st_mode, ui_info, ui_pts_text],
        )

        # 清空所有点（保留原图）
        ui_clear_p.click(
            fn=on_clear_points,
            inputs=[st_orig],
            outputs=[ui_canvas, st_src, st_tgt, st_mode, ui_info, ui_pts_text],
        )

        # 手动添加坐标
        ui_add_manual.click(
            fn=on_add_manual,
            inputs=[st_orig, st_src, st_tgt, st_mode, ui_sx, ui_sy, ui_tx, ui_ty],
            outputs=[ui_canvas, st_src, st_tgt, st_mode, ui_info, ui_pts_text],
        )

        # Lightning 开关联动：自动切换推荐参数
        ui_enable_lightning.change(
            fn=on_toggle_lightning,
            inputs=[ui_enable_lightning],
            outputs=[ui_nstep, ui_cfg],
        )

        # 运行编辑 → 输出结果
        ui_run.click(
            fn=on_run,
            inputs=[st_orig, st_src, st_tgt, ui_instr, ui_seed, ui_cfg, ui_nstep],
            outputs=[ui_out, ui_rstatus],
        )

        # 保存结果
        ui_save_btn.click(
            fn=on_save_results,
            inputs=[st_orig, ui_out,
                    st_src, st_tgt, ui_instr, ui_seed, ui_cfg, ui_nstep, ui_save_dir],
            outputs=[ui_save_status],
        )

    return demo


# ═══════════════════════════ Entry ═══════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gradio Demo")
    parser.add_argument("--model-id", type=str, default="",
        help="Base model id or path (passed to TVEditModel model_id)")
    parser.add_argument("--model-path", type=str,
        default="/home/notebook/data/group/xcx/ICEdit-main_PPU/"
                "ICEdit-main_control_5inject15_lite_qwen/train/runs/"
                "20260309-231343_ns_102end",
        help="Trained weights path (passed to TVEditModel weight_path)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--server-name", type=str, default="0.0.0.0")
    args = parser.parse_args()

    demo = build_demo(args.model_id, args.model_path)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.port,
        share=args.share,
    )