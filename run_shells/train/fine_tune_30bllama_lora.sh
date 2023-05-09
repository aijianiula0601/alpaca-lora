#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../

base_model_dir="/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/pretrain_models/llama/converted_30B"
data_path="/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/ft_52k/alpaca_data_cleaned.json"

your_random_port=11236

#-------------------
#多gpu训练
#-------------------
#output_dir="/mnt/cephfs/hjh/train_record/nlp/lora_stanford_alpaca/ft_52k/llama-30b-hf_fine_tune_out"
#CUDA_VISIBLE_DEVICES=6,7 \
#  torchrun --nproc_per_node=2 --master_port=${your_random_port} finetune.py \
#  --base_model ${base_model_dir} \
#  --data_path ${data_path} \
#  --output_dir ${output_dir} \
#  --batch_size 4

#-------------------
#自定义自己的参数
#-------------------
output_dir="/mnt/cephfs/hjh/train_record/nlp/lora_stanford_alpaca/ft_52k/llama-30b-hf_fine_tune_out_v1"
CUDA_VISIBLE_DEVICES=4,5,6,7 \
  torchrun --nproc_per_node=4 --master_port=${your_random_port} finetune.py \
  --base_model ${base_model_dir} \
  --data_path ${data_path} \
  --output_dir ${output_dir} \
  --batch_size 128 \
  --micro_batch_size 4 \
  --num_epochs 3 \
  --learning_rate 1e-4 \
  --cutoff_len 512 \
  --val_set_size 2000 \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target_modules '[q_proj,v_proj]' \
  --train_on_inputs \
  --group_by_length
