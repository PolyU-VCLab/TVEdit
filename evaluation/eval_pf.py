#!/usr/bin/env python3
"""
Image Editing Prompt Following (PF) Scorer
Based on DreamBench++ Three-Stage Evaluation Pipeline (ICLR 2025)

Pipeline:
  Stage 1: Meta-Prompt (Human-written task description, criteria, range, format)
  Stage 2: Self-Aligned Reasoning (Model summarizes task & plans evaluation)
  Stage 3: Score Generation (Images + All Prompts → CoT + Score)

Reads the released TV-Edit bench:
  <bench>/tvedit-bench.jsonl
  <bench>/tvedit-bench/<filename>/image1.jpg, image2.jpg
  Results: result_root/<filename>.jpg
  Instruction field: instruction1
"""

import os
import re
import io
import json
import glob
import base64
import time
import argparse
import logging
from pathlib import Path
from typing import Optional
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from PIL import Image

# ============================================================
# Logging
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ============================================================
# Stage 1: Meta-Prompt (Human-Written) — Prompt Following
# ============================================================

META_PROMPT = """#### Task Description
As an experienced image editing evaluator, your task is to assess the "Prompt Following (PF)" between three given images and an editing instruction. Prompt Following measures whether the AI-edited image correctly and accurately executes the editing operation described in the text instruction.

CRUCIAL RULE: You must focus on whether the INTENDED EDIT was correctly performed. This is fundamentally different from Concept Preservation — here you are judging the ACCURACY OF THE EDIT itself, NOT the preservation of unedited regions. Specifically, you should evaluate:
- Was the correct object/region targeted?
- Was the correct type of operation performed (e.g., dragging, moving, resizing, reshaping)?
- Was the edit applied in the correct direction, to the correct position, and with the correct magnitude?
- Does the final result match the Reference GT in terms of what was changed?

#### PF Scoring Criteria
Compare the Result Image against the Source Image, Reference GT, and the Instruction across the following four dimensions:
1. Operation Type: Was the correct type of editing operation performed? For example, if the instruction says "drag the cat to the right", was a spatial translation performed (rather than deletion, color change, or no edit at all)?
2. Target Object: Was the correct object or region identified and edited? If the instruction specifies editing a particular element (e.g., "move the red ball"), did the model operate on exactly that element, and not a different one?
3. Spatial Accuracy: Was the edit applied in the correct direction and to the correct location? Compare the edited object's position, pose, or shape in the Result Image against the Reference GT. Is the spatial transformation consistent with what the instruction describes?
4. Magnitude & Completeness: Is the degree/extent of the edit appropriate? Was the operation completed fully, or was it only partially applied? Compare the magnitude of change in the Result Image to that in the Reference GT.

#### Scoring Range
Assign an integer score from 0 to 4 based on the following scale:
0 - Very Poor: The instruction was completely ignored. No visible editing was performed, or the edit is entirely wrong (e.g., wrong object, wrong operation type, or the image is unchanged/corrupted).
1 - Poor: Some attempt at editing is visible, but it fundamentally fails to follow the instruction. For example: the wrong object was edited, the edit goes in the completely wrong direction, or the type of operation is incorrect.
2 - Fair: The general type of edit is correct and the right object was targeted, but the specifics are significantly off. For example: the object was moved but to a noticeably wrong position, or the magnitude of change is far from the Reference GT.
3 - Good: The instruction is largely followed correctly. The right object was edited with the right operation in roughly the right direction and magnitude, but there are minor deviations in precision compared to the Reference GT (e.g., slightly off-target position, slightly under/over-applied).
4 - Excellent: The instruction is perfectly followed. The edit matches the Reference GT closely in terms of which object was changed, what operation was applied, the direction, position, and magnitude of the edit. The result is virtually indistinguishable from the Reference GT in terms of the intended editing operation.

#### Input Format
Each time, you will receive 3 images and an Instruction:
- Image 1 (Source Image): The original image before editing.
- Image 2 (Reference GT): The ground truth image, serving as the perfect reference for what the correct edit should look like.
- Image 3 (Result Image): The AI-edited image to be evaluated.
- Instruction: The text prompt describing the intended operation on the Source Image.
(Note: The evaluation of Prompt Following for Image 3 must be strictly based on how well it executes the Instruction, using Image 2 as the reference for the correct outcome.)

#### Output Format
[internal thinking] (Briefly describe how you evaluated the edit step-by-step across the 4 dimensions: Operation Type, Target Object, Spatial Accuracy, and Magnitude & Completeness.)
[Score]Prompt Following (PF): <an integer from 0 to 4>
"""


# ============================================================
# Stage 2: Self-Aligned Reasoning Prompt — Prompt Following
# ============================================================

SELF_ALIGN_PROMPT = """You are about to perform an image editing evaluation task focused on Prompt Following (PF). Before you begin scoring, please carefully read the evaluation guidelines above and then:

1. **Summarize** the core evaluation objective in your own words (what exactly is "Prompt Following" measuring? How is it different from "Concept Preservation"?).
2. **Plan** your step-by-step evaluation procedure: how will you compare the three images and the instruction across the 4 dimensions (Operation Type, Target Object, Spatial Accuracy, Magnitude & Completeness)? What will you look at first, second, third, and fourth?
3. **Restate** the scoring criteria for each level (0-4) in brief, emphasizing the key differentiators between adjacent levels.

Please output your summary and plan now. This will help you perform a more accurate evaluation afterwards."""


# ============================================================
# Stage 3: Score Generation Prompt Template — Prompt Following
# ============================================================

SCORE_GENERATION_PROMPT = """Now, based on your understanding of the Prompt Following task and your evaluation plan above, please evaluate the following case.

**Instruction:** {instruction}

The three images are provided in order:
- Image 1: Source Image (original, before editing)
- Image 2: Reference GT (ground truth, the ideal editing result)
- Image 3: Result Image (the AI-edited image to be evaluated)

Remember: You must evaluate whether the EDITING INSTRUCTION was correctly executed. Focus on the accuracy and completeness of the edit itself (operation type, target object, spatial accuracy, magnitude), NOT on the preservation of unedited regions.

Please follow the output format strictly:
[internal thinking] (Your step-by-step evaluation across the 4 dimensions: Operation Type, Target Object, Spatial Accuracy, Magnitude & Completeness)
[Score]Prompt Following (PF): <an integer from 0 to 4>
"""


# ============================================================
# Utility Functions
# ============================================================

def encode_image_to_base64(image_path: str, resize_to: int = 512) -> str:
    """
    读取图片，resize 到固定大小 (resize_to x resize_to)，编码为 base64 JPEG。
    统一尺寸可以减少 token 消耗并保证输入一致性。
    """
    img = Image.open(image_path).convert("RGB")
    if resize_to and (img.width != resize_to or img.height != resize_to):
        img = img.resize((resize_to, resize_to), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return b64_str


def load_annotations(jsonl_path: str) -> dict:
    """Return {filename: record}. Each record includes instruction1 and image names."""
    records = {}
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"  JSONL line {line_num} parse error: {e}, skipping.")
                continue

            filename = item.get('filename', '')
            instruction = item.get('instruction1', '')

            if not filename:
                logger.warning(f"  JSONL line {line_num}: missing 'filename', skipping.")
                continue
            if not instruction:
                logger.warning(f"  JSONL line {line_num} ({filename}): missing 'instruction1', skipping.")
                continue

            records[filename] = item

    logger.info(f"Loaded {len(records)} annotations from {jsonl_path}")
    return records


def resolve_sample_root(gt_data_root: str, filenames: list) -> str:
    """Accept either the unzipped bench root or the inner folder of sample directories."""
    root = Path(gt_data_root)

    def hits(path: Path) -> int:
        if not path.is_dir():
            return -1
        return sum((path / name).is_dir() for name in filenames)

    candidates = [root]
    if root.is_dir():
        candidates.extend(path for path in root.iterdir() if path.is_dir())
    best = max(candidates, key=hits)
    logger.info(f"Sample root: {best} ({hits(best)} / {len(filenames)} folders)")
    return str(best)


def resolve_image(sample_dir: str, filename: str, fallback_stem: str) -> Optional[str]:
    direct = os.path.join(sample_dir, filename) if filename else ""
    if direct and os.path.isfile(direct):
        return direct
    stem = Path(filename).stem if filename else fallback_stem
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        path = os.path.join(sample_dir, stem + ext)
        if os.path.isfile(path):
            return path
    return None


def find_result_image_flat(result_root: str, sample_name: str) -> Optional[str]:
    """
    在扁平的结果目录中，根据样本名查找对应的结果图片。
    """
    for ext in [".jpg", ".jpeg", ".png", ".webp", ".bmp"]:
        path = os.path.join(result_root, sample_name + ext)
        if os.path.exists(path):
            return path

    pattern = os.path.join(result_root, sample_name + ".*")
    matches = glob.glob(pattern)
    img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    for m in sorted(matches):
        if Path(m).suffix.lower() in img_exts:
            return m

    pattern2 = os.path.join(result_root, sample_name + "*")
    matches2 = glob.glob(pattern2)
    for m in sorted(matches2):
        if Path(m).suffix.lower() in img_exts:
            return m

    return None


def load_samples(gt_data_root: str, result_root: str, annotations: dict) -> list:
    """Pair each jsonl record with its sample folder and a flat result image."""
    samples = []
    skipped = 0

    result_files = {}
    if os.path.isdir(result_root):
        img_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
        for fname in os.listdir(result_root):
            fpath = os.path.join(result_root, fname)
            if os.path.isfile(fpath) and Path(fname).suffix.lower() in img_exts:
                stem = Path(fname).stem
                result_files[stem] = fpath
        logger.info(f"Found {len(result_files)} result images in {result_root}")
    else:
        logger.error(f"Result root does not exist: {result_root}")
        return []

    sample_root = resolve_sample_root(gt_data_root, list(annotations))

    for filename, item in annotations.items():
        gt_subdir = os.path.join(sample_root, filename)
        if not os.path.isdir(gt_subdir):
            logger.warning(f"  [{filename}] Sample folder not found, skipping.")
            skipped += 1
            continue

        source_path = resolve_image(gt_subdir, item.get("image1", "image1.jpg"), "image1")
        if source_path is None:
            logger.warning(f"  [{filename}] Missing image1, skipping.")
            skipped += 1
            continue

        gt_path = resolve_image(gt_subdir, item.get("image2", "image2.jpg"), "image2")
        if gt_path is None:
            logger.warning(f"  [{filename}] Missing image2, skipping.")
            skipped += 1
            continue

        result_path = result_files.get(filename)
        if result_path is None:
            result_path = find_result_image_flat(result_root, filename)
        if result_path is None:
            logger.warning(f"  [{filename}] No matching result image found, skipping.")
            skipped += 1
            continue

        samples.append({
            "id": filename,
            "source": source_path,
            "gt": gt_path,
            "result": result_path,
            "instruction": item["instruction1"],
        })

    logger.info(f"Loaded {len(samples)} valid samples, skipped {skipped}.")

    for s in samples[:3]:
        logger.info(f"  Sample: {s['id']}")
        logger.info(f"    source:      {s['source']}")
        logger.info(f"    gt:          {s['gt']}")
        logger.info(f"    result:      {s['result']}")
        logger.info(f"    instruction: {s['instruction'][:120]}...")

    return samples


def parse_score(response_text: str) -> Optional[int]:
    """Parse the PF score from the model's response."""
    if not response_text:
        return None

    # 严格匹配: [Score]Prompt Following (PF): <digit>
    pattern = r'\[Score\]\s*Prompt\s*Following\s*\(PF\)\s*:\s*(\d)'
    match = re.search(pattern, response_text)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 4:
            return score

    # 宽松: PF): <digit> 或 PF: <digit>
    fallback = r'(?:PF\)?\s*:\s*)(\d)'
    match = re.search(fallback, response_text, re.IGNORECASE)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 4:
            return score

    # [Score] 后跟数字
    fallback2 = r'\[Score\].*?(\d)'
    match = re.search(fallback2, response_text, re.DOTALL)
    if match:
        score = int(match.group(1))
        if 0 <= score <= 4:
            return score

    # 兜底：最后一个 0-4
    digits = re.findall(r'\b([0-4])\b', response_text)
    if digits:
        return int(digits[-1])

    return None


# ============================================================
# Core Scorer: DreamBench++ Three-Stage Pipeline (PF)
# ============================================================

class PFScorer:
    """
    Prompt Following Scorer using DreamBench++ three-stage pipeline.
    """

    def __init__(self, api_base: str, api_key: str, model_name: str,
                 temperature: float = 0.1, max_tokens: int = 2048,
                 max_retries: int = 3, retry_delay: float = 2.0,
                 num_repeats: int = 1, image_size: int = 512,
                 output_dir: str = "./eval_output_pf"):
        self.api_url = f"{api_base.rstrip('/')}/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.num_repeats = num_repeats
        self.image_size = image_size
        self.output_dir = output_dir

        # ======== Stage 2: Self-Aligned Reasoning (仅执行一次) ========
        logger.info("=" * 60)
        logger.info("Stage 2: Self-Aligned Reasoning (Prompt Following)")
        logger.info("  Model internalizes PF evaluation criteria...")
        logger.info("=" * 60)
        self.self_aligned_output = self._run_self_alignment()

    def _run_self_alignment(self) -> str:
        """
        DreamBench++ Stage 2:
        发送 meta-prompt 给模型，让它总结 PF 任务、规划评估步骤、复述评分标准。
        """
        messages = [
            {
                "role": "system",
                "content": "You are a professional image editing evaluation assistant specializing in Prompt Following assessment."
            },
            {
                "role": "user",
                "content": META_PROMPT + "\n\n" + SELF_ALIGN_PROMPT
            }
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        response = self._call_api(payload)
        if response:
            logger.info("  Self-alignment completed.")
            return response
        else:
            logger.warning("  Self-alignment failed. Proceeding without it.")
            return ""

    def score_sample(self, sample: dict, repeat_idx: int = 0) -> dict:
        """
        DreamBench++ Stage 3: Score Generation for PF.

        多轮对话结构:
          [system]    角色设定
          [user]      Meta-Prompt + Self-Align 请求     ← Stage 1+2 input
          [assistant] Self-Aligned 输出 (缓存)           ← Stage 2 output
          [user]      三张图 + Instruction + 打分请求    ← Stage 3
        """
        sample_id = sample["id"]
        instruction = sample["instruction"]

        # 所有图片统一 resize 到 image_size x image_size 后编码
        source_b64 = encode_image_to_base64(sample["source"], resize_to=self.image_size)
        gt_b64 = encode_image_to_base64(sample["gt"], resize_to=self.image_size)
        result_b64 = encode_image_to_base64(sample["result"], resize_to=self.image_size)

        mime = "image/jpeg"

        scoring_text = SCORE_GENERATION_PROMPT.format(instruction=instruction)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a professional image editing evaluation assistant "
                    "specializing in Prompt Following assessment. "
                    "You must follow the evaluation guidelines precisely and "
                    "provide scores strictly according to the criteria."
                )
            },
            {
                "role": "user",
                "content": META_PROMPT + "\n\n" + SELF_ALIGN_PROMPT
            },
            {
                "role": "assistant",
                "content": self.self_aligned_output
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": scoring_text},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{source_b64}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{gt_b64}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{result_b64}"
                        }
                    },
                ]
            }
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        raw_response = self._call_api(payload)
        score = parse_score(raw_response) if raw_response else None

        if score is not None:
            logger.info(f"  [{sample_id}] repeat={repeat_idx} PF Score: {score}")
        else:
            logger.warning(f"  [{sample_id}] repeat={repeat_idx} Failed to parse PF score.")

        return {
            "id": sample_id,
            "instruction": instruction,
            "source": sample["source"],
            "gt": sample["gt"],
            "result": sample["result"],
            "score": score,
        }

    def score_sample_with_repeats(self, sample: dict) -> dict:
        """多次评估取多数投票（DreamBench++ 稳定性策略）。"""
        scores = []

        for i in range(self.num_repeats):
            result = self.score_sample(sample, repeat_idx=i)
            if result["score"] is not None:
                scores.append(result["score"])

        if scores:
            vote = Counter(scores).most_common(1)[0][0]
            avg = round(sum(scores) / len(scores), 4)
        else:
            vote = None
            avg = None

        return {
            "id": sample["id"],
            "instruction": sample["instruction"],
            "source": sample["source"],
            "gt": sample["gt"],
            "result": sample["result"],
            "score": vote,
            "score_avg": avg,
            "all_scores": scores,
        }

    def _call_api(self, payload: dict) -> Optional[str]:
        """调用本地模型 API，带重试。"""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = requests.post(
                    self.api_url,
                    headers=self.headers,
                    json=payload,
                    timeout=600,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except requests.exceptions.Timeout:
                logger.warning(f"  Timeout (attempt {attempt}/{self.max_retries})")
            except requests.exceptions.RequestException as e:
                logger.warning(f"  API error (attempt {attempt}/{self.max_retries}): {e}")
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                logger.warning(f"  Parse error (attempt {attempt}/{self.max_retries}): {e}")

            if attempt < self.max_retries:
                time.sleep(self.retry_delay * attempt)

        logger.error("  All retries exhausted.")
        return None

    def evaluate_all(self, samples: list, max_workers: int = 1) -> list:
        """批量评估所有样本。"""
        results = []
        total = len(samples)

        if max_workers <= 1:
            for i, sample in enumerate(samples):
                logger.info(f"Progress: {i + 1}/{total}")
                result = self.score_sample_with_repeats(sample)
                results.append(result)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(self.score_sample_with_repeats, s): s
                    for s in samples
                }
                done_count = 0
                for future in as_completed(future_map):
                    done_count += 1
                    try:
                        result = future.result()
                        results.append(result)
                        logger.info(
                            f"Progress: {done_count}/{total} | "
                            f"{result['id']} -> PF Score: {result['score']}"
                        )
                    except Exception as e:
                        s = future_map[future]
                        logger.error(f"Error on {s['id']}: {e}")
                        results.append({
                            "id": s["id"], "score": None, "error": str(e)
                        })

        results.sort(key=lambda x: x.get("id", ""))
        return results

    @staticmethod
    def compute_statistics(results: list) -> dict:
        valid_scores = [r["score"] for r in results if r.get("score") is not None]
        total = len(results)
        valid = len(valid_scores)

        if not valid_scores:
            return {"total_samples": total, "valid_samples": 0,
                    "mean_score": None, "normalized_score": None}

        dist = Counter(valid_scores)
        mean = sum(valid_scores) / valid

        return {
            "total_samples": total,
            "valid_samples": valid,
            "failed_samples": total - valid,
            "mean_score": round(mean, 4),
            "normalized_score": round(mean / 4, 4),
            "score_distribution": {str(k): dist.get(k, 0) for k in range(5)},
            "score_distribution_pct": {
                str(k): f"{dist.get(k, 0) / valid * 100:.1f}%" for k in range(5)
            },
        }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DreamBench++ Style PF (Prompt Following) Scorer for Image Editing"
    )

    # ---- Paths ----
    parser.add_argument(
        "--gt_data_root", type=str, required=True,
        help="GT data root. Each sample is a subdirectory with image1.jpg and image2.jpg."
    )
    parser.add_argument(
        "--result_root", type=str, required=True,
        help="结果图片目录 (扁平结构, 图片文件名 = GT子文件夹名 + 扩展名)"
    )
    parser.add_argument(
        "--jsonl_path", type=str,
        required=True,
        help="JSONL with filename and instruction1 fields"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./eval_output_pf",
        help="输出目录"
    )

    # ---- Model ----
    parser.add_argument(
        "--api_base", type=str, default="http://localhost:8000/v1",
        help="本地模型 API 地址 (OpenAI 兼容)"
    )
    parser.add_argument(
        "--api_key", type=str, default="EMPTY",
        help="API key"
    )
    parser.add_argument(
        "--model", type=str,
        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
        help="模型名称"
    )

    # ---- Evaluation ----
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_repeats", type=int, default=3,
                        help="每样本重复评估次数 (>1 则多数投票)")
    parser.add_argument("--max_workers", type=int, default=1,
                        help="并行线程数")
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--image_size", type=int, default=512,
                        help="所有输入图片统一 resize 到此尺寸 (image_size x image_size)")

    args = parser.parse_args()

    # ---- Load instructions from JSONL ----
    logger.info("=" * 60)
    logger.info("Loading instructions from JSONL...")
    logger.info(f"  JSONL path:  {args.jsonl_path}")
    logger.info("=" * 60)

    instruction_records = load_annotations(args.jsonl_path)

    if not instruction_records:
        logger.error("No valid records found in JSONL file. Check the file format.")
        return

    # ---- Load samples ----
    logger.info("=" * 60)
    logger.info("Loading samples...")
    logger.info(f"  GT root:            {args.gt_data_root}")
    logger.info(f"  Result root:        {args.result_root}")
    logger.info(f"  Image size:         {args.image_size}x{args.image_size}")
    logger.info("=" * 60)

    samples = load_samples(
        gt_data_root=args.gt_data_root,
        result_root=args.result_root,
        instruction_records=instruction_records,
    )

    if not samples:
        logger.error("No valid samples found. Check your directories and JSONL file.")
        return

    # ---- Initialize scorer (triggers Stage 2 self-alignment) ----
    scorer = PFScorer(
        api_base=args.api_base,
        api_key=args.api_key,
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        num_repeats=args.num_repeats,
        image_size=args.image_size,
        output_dir=args.output_dir,
    )

    # ---- Run Stage 3 ----
    logger.info("=" * 60)
    logger.info("Stage 3: Scoring all samples (Prompt Following)...")
    logger.info("=" * 60)

    results = scorer.evaluate_all(
        samples=samples,
        max_workers=args.max_workers,
    )

    # ---- Statistics ----
    stats = scorer.compute_statistics(results)

    logger.info("=" * 60)
    logger.info("PF Evaluation Complete!")
    logger.info(f"  Total:        {stats['total_samples']}")
    logger.info(f"  Valid:        {stats['valid_samples']}")
    logger.info(f"  Failed:       {stats.get('failed_samples', 'N/A')}")
    if stats.get("mean_score") is not None:
        logger.info(f"  Mean PF:      {stats['mean_score']}")
        logger.info(f"  Normalized:   {stats['normalized_score']}")
        logger.info(f"  Distribution: {stats['score_distribution']}")
        logger.info(f"  Dist (%):     {stats['score_distribution_pct']}")
    logger.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "pf_results_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "metric": "Prompt Following (PF)",
                "gt_data_root": args.gt_data_root,
                "result_root": args.result_root,
                "jsonl_path": args.jsonl_path,
                "model": args.model,
                "temperature": args.temperature,
                "num_repeats": args.num_repeats,
                "image_size": args.image_size,
            },
            "statistics": stats,
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()