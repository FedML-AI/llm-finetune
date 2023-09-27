#!/bin/bash
set -e

BASE_DIR="$(dirname "$0")"
BASE_DIR="$(realpath "${BASE_DIR}/../")"
cd "${BASE_DIR}"

bash scripts/setup.sh

MASTER_ADDR="${1:-"localhost"}"
MASTER_PORT="${2:-12355}"
NUM_NODES="${3:-1}"
NUM_GPU="$(python3 -c "import torch; print(torch.cuda.device_count())")"

DATASET_PATHS=(
  ".data/databricks-dolly-15k.jsonl"
  # add your datasets here
)

deepspeed \
  --num_nodes="${NUM_NODES}" \
  --num_gpus="${NUM_GPU}" \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  run_train.py \
  --task "instruction" \
  --deepspeed "configs/deepspeed/ds_z3_bf16_config.json" \
  --model_name "EleutherAI/pythia-70m" \
  --dataset_path "${DATASET_PATHS[@]}" \
  --test_dataset_size 200 \
  --seed 1234 \
  --fp16 "False" \
  --bf16 "False" \
  --use_lora \
  --gradient_checkpointing "True" \
  --per_device_train_batch_size 6 \
  --per_device_eval_batch_size 6 \
  --learning_rate "5e-6" \
  --warmup_steps 50 \
  --num_train_epochs 5 \
  --output_dir ".logs/dolly_pythia-70m" \
  --logging_steps 50 \
  --eval_steps 200 \
  --save_steps 200 \
  --save_total_limit 10 \
  --logging_strategy "steps" \
  --evaluation_strategy "steps" \
  --save_strategy "steps" \
  --eval_accumulation_steps 4 \
  --do_train "True" \
  --do_eval "True" \
  --remove_long_seq "True" \
  "${@:4}" # skip first 3 arguments
