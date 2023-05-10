#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../



base_dir="/mnt/cephfs/hjh/train_record/nlp/lora_stanford_alpaca"
data_path="${base_dir}/debug.json"
output_dir="${base_dir}/debug"
your_random_port=11235


rm -rf ${output_dir}
mkdir -p ${output_dir}


#-------------------
#多gpu训练
#-------------------
CUDA_VISIBLE_DEVICES=4 \
torchrun --nproc_per_node=1 --master_port=${your_random_port} finetune.py \
    --base_model 'decapoda-research/llama-7b-hf' \
    --data_path ${data_path} \
    --output_dir ${output_dir}
