#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../../

base_model_dir="/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/pretrain_models/llama/new_llama_7b"

lora_model_dir="/mnt/cephfs/hjh/train_record/nlp/lora_stanford_alpaca/ft_52k/llama-7b-hf_fine_tune_out_v1/checkpoint-1000"

#CUDA_VISIBLE_DEVICES=7 \
#python generate.py \
#  --load_8bit \
#  --base_model ${base_model_dir} \
#  --lora_weights ${lora_model_dir}


CUDA_VISIBLE_DEVICES=7 \
python generate.py \
    --load_8bit \
    --base_model 'decapoda-research/llama-7b-hf' \
    --lora_weights 'tloen/alpaca-lora-7b'