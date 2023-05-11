#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../

base_dir="/mnt/cephfs/hjh/train_record/nlp/lora_stanford_alpaca/conversation_mask_instruct"

data_path="${base_dir}/train.json"
output_dir="${base_dir}/ft_out"

your_random_port=11235
#-------------------
#多gpu训练
#-------------------
torchrun --nproc_per_node=8 --master_port=${your_random_port} test_models/mask_instruct/finetune_mask_instruct.py \
  --base_model 'decapoda-research/llama-7b-hf' \
  --data_path ${data_path} \
  --output_dir ${output_dir} \
  --batch_size 64 \
  --save_steps 500 \
  --save_total_limit 10 \
  --num_epochs 1
