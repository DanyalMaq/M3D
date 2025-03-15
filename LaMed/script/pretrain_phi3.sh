#!/bin/bash

# run "accelerate config" first!

accelerate launch LaMed/src/train/train.py \
    --version v0 \
    --model_name_or_path GoodBaiBai88/M3D-LaMed-Phi-3-4B \
    --model_type phi3 \
    --vision_tower vit3d \
    --pretrain_vision_model /mym3d/LaMed/pretrained_ViT.bin \
    --pretrain_mm_mlp_adapter /mym3d/LaMed/mm_projector.bin \
    --tune_mm_mlp_adapter True \
    --bf16 True \
    --output_dir ./LaMed/output/LaMed-Phi3-4B-pretrain-alpha-0000 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 0.04 \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 0.001 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 8 \
    --report_to all

# sh LaMed/script/pretrain_phi3.sh
