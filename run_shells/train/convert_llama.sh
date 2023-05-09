#!/bin/bash

set -ex

curdir=$(pwd)
echo "curdir:$curdir"
cd "$curdir" || exit

cd ../


cd /home/huangjiahong.dracu/hjh/pycharm_projects/nlp/TencentPretrain




base_dir="/mnt/cephfs/hjh/train_record/nlp/stanford_alpaca/pretrain_models/llama"
save_outputs="${base_dir}/converted_30B"
model_size="30B"

python transformers/models/llama/convert_llama_weights_to_hf.py --input_dir ${base_dir} --model_size ${model_size} --output_dir ${save_outputs}

