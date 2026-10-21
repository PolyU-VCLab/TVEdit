# Evaluation

Two evaluation tracks share the same benchmark layout.

| Script | What it measures |
| --- | --- |
| `evaluation_pure.py` | LPIPS, mean distance (MD), and DINOv3 global / local scores |
| `eval_cp.py` | Concept Preservation (CP), 0–4, via a vision-language judge |
| `eval_pf.py` | Prompt Following (PF), 0–4, via a vision-language judge |

`evaluator.py` and `dino_evaluator.py` are libraries used by `evaluation_pure.py`. `batch.sh` runs `evaluation_pure.py` on several result folders.

## Weights and packages

Quantitative metrics:

- **DINOv3 ViT-B/16**: [facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m). This checkpoint is gated. Accept the license on Hugging Face, then `hf auth login`, and pass either the repo id or a local directory to `--dino_model`. The default input resolution is 512, which yields a 32×32 patch grid (`patch_size=16`).
- **LPIPS (AlexNet)**: loaded by `lpips` with `net='alex'`. The first run downloads the AlexNet LPIPS weights.

VLM scores (CP / PF):

- **Qwen3-VL-30B-A3B-Instruct**: [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct). Serve it with an OpenAI-compatible endpoint (for example vLLM) and pass `--api_base`.

```shell
pip install torch torchvision lpips transformers pillow numpy matplotlib opencv-python tqdm requests scipy
```

`scipy` is only used when `union_mask.png` is missing and a mask is built from dense points.

## Benchmark directory

Unzip `tvedit-bench.zip`. `--data_dir` and `--gt_data_root` both accept this root. The scripts find `tvedit-bench.jsonl` and the inner folder of samples.

```text
tvedit-bench/
  tvedit-bench.jsonl
  tvedit-bench/
    nb_0000_00/
      image1.jpg          # source
      image2.jpg          # ground-truth edit
      union_mask.png      # object mask used for both local DINO scores
    nb_0000_01/
      ...
```

`tvedit-bench.jsonl` has one JSON object per line. `filename` is the sample folder name. Points are separate source and target lists of `[x, y]`, in source-image pixels. Images and points are rescaled to the ground-truth resolution before scoring. CP and PF use `instruction1`.

```json
{
  "filename": "nb_0000_00",
  "image1": "image1.jpg",
  "image2": "image2.jpg",
  "sparse_point1": [[368.5, 117.0], [240.1, 306.7]],
  "sparse_point2": [[365.7, 289.3], [222.2, 486.5]],
  "dense_point1": [[368.5, 117.0], [368.3, 86.1]],
  "dense_point2": [[365.7, 289.3], [366.0, 252.7]],
  "instruction1": "Make this building lower.",
  "instruction2": "Make this building taller."
}
```

`sparse_point1` / `dense_point1` are handle points. `sparse_point2` / `dense_point2` are the corresponding target points. If `union_mask.png` is absent, local DINO builds a mask from the dense points.

## Edited-image directory

Results are a flat folder. The file stem must equal the sample folder name.

```text
results/
  nb_0000_00.jpg
  nb_0000_01.png
  ...
```

Accepted extensions: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`.

## Quantitative evaluation

```shell
cd evaluation
python evaluation_pure.py \
  --data_dir /path/to/tvedit-bench \
  --result_dir /path/to/results \
  --output_dir /path/to/eval_out \
  --dino_model facebook/dinov3-vitb16-pretrain-lvd1689m \
  --dino_backend hf \
  --dino_resolution 512
```

Optional plots: `--plot` (MD arrows), `--plot_lpips`, `--plot_local_dino`. `--no_dino` skips DINOv3 and still reports LPIPS and MD.

Several result folders:

```shell
DATA_DIR=/path/to/tvedit-bench DINO_MODEL=/path/to/dinov3 \
  bash batch.sh /path/to/results_a /path/to/results_b
```

Each run writes `<result_dir>_eval/`.

### Output

```text
eval_out/
  evaluation_results.json
  sample_001/                  # only when a --plot* flag is set
    distance_viz.png           # sparse MD
    distance_dense_viz.png     # dense MD
    lpips_source_viz.png
    lpips_target_viz.png
    lpips_comparison.png
    local_dino_viz_src.png
    local_dino_viz_tgt.png
```

`evaluation_results.json`:

- `average_metrics`: `avg_lpips_source`, `avg_lpips_target`, `avg_distance` (sparse MD), `avg_distance_dense`, and, when DINO is enabled, `avg_global_dino`, `avg_local_dino_src`, `avg_local_dino_tgt`.
- `individual_metrics`: the same scores per sample, plus `sample_name`, `n_sparse_pairs`, `n_dense_pairs`.
- `info`: sample counts, skipped names, errors, and mask source counts.

Metric definitions:

- **LPIPS vs source / target**: AlexNet LPIPS between the edit and the source, and between the edit and the ground truth. Lower is closer.
- **MD**: mean normalized distance between DINOv3 nearest-neighbor matches of the handle points and the target points. Lower means the dragged points landed closer to the targets.
- **Global DINO**: cosine similarity of DINOv3 CLS tokens between the ground truth and the edit.
- **Local DINO (src / tgt)**: average cosine similarity of nearest-neighbor patch tokens inside `union_mask.png`. The source score matches source patches to the edit. The target score matches ground-truth patches to the edit.

## Concept Preservation and Prompt Following

Start an OpenAI-compatible server for Qwen3-VL, then:

```shell
python eval_cp.py \
  --gt_data_root /path/to/tvedit-bench \
  --result_root /path/to/results \
  --jsonl_path /path/to/tvedit-bench/tvedit-bench.jsonl \
  --output_dir /path/to/cp_out \
  --api_base http://localhost:8000/v1 \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --num_repeats 1

python eval_pf.py \
  --gt_data_root /path/to/tvedit-bench \
  --result_root /path/to/results \
  --jsonl_path /path/to/tvedit-bench/tvedit-bench.jsonl \
  --output_dir /path/to/pf_out \
  --api_base http://localhost:8000/v1 \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct \
  --num_repeats 3
```

CP judges whether the whole image keeps identity, semantics, color, and texture. PF judges whether the instruction was carried out, using the ground-truth image as the reference edit. Both follow a DreamBench++-style three-stage prompt: criteria, a one-time self-alignment pass, then per-sample scoring. `--num_repeats` greater than 1 keeps the majority vote (`score`) and the mean (`score_avg`).

### Output

Each run writes one file: `cp_results_summary.json` or `pf_results_summary.json`. It contains the run config, `statistics`, and one record per sample (`score`, `score_avg`, `all_scores`). `mean_score` is on the 0–4 scale, and `normalized_score` is `mean_score / 4`.
