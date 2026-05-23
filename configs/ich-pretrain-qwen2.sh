#!/bin/bash
#
# ICH Beacon 微调脚本 — 适配 PyTorch 2.8 + transformers 4.46
# 单卡运行，不用 deepspeed
#

shopt -s expand_aliases
source ~/.bashrc 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=0

# 直接用 python 跑，不用 torchrun（单卡不需要）
python -m main.train \
    --output_dir data/outputs/ich-pretrain-qwen2 \
    --model_name_or_path /root/models/memorag-qwen2-7b-inst \
    --train_data data/ich/train.json \
    --enable_beacon \
    --only_train_beacon \
    --beacon_window 512 \
    --beacon_stride 512 \
    --beacon_attn full-coverage \
    --beacon_attend_prev True \
    --beacon_sink_size 0 \
    --beacon_ratio 2 4 8 16 32 \
    --beacon_ratio_mix step-random \
    --beacon_param q k v \
    --beacon_pos interleave \
    --attn_impl sdpa \
    --max_length 16384 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-4 \
    --warmup_ratio 0.05 \
    --bf16 \
    --logging_steps 1 \
    --save_strategy epoch \
    --save_only_model \
    --remove_unused_columns False
