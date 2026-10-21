#!/bin/bash
# Evaluate one or more flat result directories with evaluation_pure.py.
#
# Usage:
#   DATA_DIR=/path/to/bench DINO_MODEL=/path/to/dinov3 \
#     bash batch.sh /path/to/results_a /path/to/results_b
#
# Each result directory is written to <result_dir>_eval/.

set -euo pipefail

DATA_DIR="${DATA_DIR:?Set DATA_DIR to the benchmark root}"
DINO_MODEL="${DINO_MODEL:-facebook/dinov3-vitb16-pretrain-lvd1689m}"
DINO_BACKEND="${DINO_BACKEND:-hf}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

if [ "$#" -eq 0 ]; then
    echo "Usage: DATA_DIR=... bash batch.sh RESULT_DIR [RESULT_DIR ...]"
    exit 1
fi

for RDIR in "$@"; do
    RDIR="${RDIR%/}"
    OUTPUT_DIR="${RDIR}_eval"
    echo "Evaluating ${RDIR} -> ${OUTPUT_DIR}"
    python evaluation_pure.py \
        --data_dir "${DATA_DIR}" \
        --result_dir "${RDIR}" \
        --output_dir "${OUTPUT_DIR}" \
        --dino_model "${DINO_MODEL}" \
        --dino_backend "${DINO_BACKEND}" \
        ${EXTRA_FLAGS}
done
