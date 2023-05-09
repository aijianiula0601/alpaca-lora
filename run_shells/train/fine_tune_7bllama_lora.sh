#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../



base_model_dir="/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/pretrain_models/llama/new_llama_7b"
data_path="/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/ft_52k/alpaca_data_cleaned.json"

output_dir="/mnt/cephfs/hjh/train_record/nlp/lora_stanford_alpaca/ft_52k/llama-7b-hf_fine_tune_out_v1"
your_random_port=11235



#-------------------
#多gpu训练
#-------------------
CUDA_VISIBLE_DEVICES=4,5,6,7 \
torchrun --nproc_per_node=4 --master_port=${your_random_port} finetune.py \
    --base_model ${base_model_dir} \
    --data_path ${data_path} \
    --output_dir ${output_dir}



#CUDA_VISIBLE_DEVICES=6,7 \
#torchrun --nproc_per_node=2 --master_port=${your_random_port} finetune.py \
#    --base_model ${base_model_dir} \
#    --data_path ${data_path} \
#    --output_dir ${output_dir} \
#    --batch_size 128 \
#    --micro_batch_size 4 \
#    --num_epochs 3 \
#    --learning_rate 1e-4 \
#    --cutoff_len 512 \
#    --val_set_size 2000 \
#    --lora_r 8 \
#    --lora_alpha 16 \
#    --lora_dropout 0.05 \
#    --lora_target_modules '[q_proj,v_proj]' \
#    --train_on_inputs \
#    --group_by_length
#


#CUDA_VISIBLE_DEVICES=6 \
#python finetune.py \
#    --base_model ${base_model_dir} \
#    --data_path ${data_path} \
#    --output_dir ${output_dir}
