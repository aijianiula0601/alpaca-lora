#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../



data_path="/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/alpaca_cleand_and_gpt4.json"
output_dir="/mnt/cephfs/hjh/train_record/nlp/lora_stanford_alpaca/llama-7b-hf_alpaca_cleand_and_gpt4_fine_out"
your_random_port=11235


#-------------------
#多gpu训练
#-------------------
CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --nproc_per_node=4 --master_port=${your_random_port} finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
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