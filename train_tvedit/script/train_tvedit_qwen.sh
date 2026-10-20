#!/bin/bash

# usage: ./train.sh [CONFIG_FILE] [PORT]
# example: ./train.sh normal_lora.yaml 41353


CONFIG_FILE=${1:-"tvedit_qwen_ppu.yaml"}
PORT=${2:-41353}

export WANDB_API_KEY=5e91b7848b07e4b6f6a1bb37196ec254e027358d
export XFL_CONFIG=./train/config/${CONFIG_FILE}
echo "Using config: $XFL_CONFIG"
export TOKENIZERS_PARALLELISM=true
export PYTHONPATH=.
accelerate launch --main_process_port ${PORT} -m train_tvedit.train