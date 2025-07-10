#!/bin/bash

# run "accelerate config" first!
# sh LaMed/script/finetune_lora_full_phi3.sh

# 0001 is normal
# 0002 is with the structured output and CT + mask
# 0003 is baseline with structured output

# accelerate launch LaMed/src/train/train.py \
#     --version v0 \
#     --model_name_or_path GoodBaiBai88/M3D-LaMed-Phi-3-4B \
#     --model_type phi3 \
#     --vision_tower vit3d \
#     --pretrain_vision_model /mym3d/LaMed/pretrained_ViT.bin \
#     --tune_mm_mlp_adapter True \
#     --bf16 True \
#     --output_dir /mym3d/LaMed/output/LaMed-Phi3-4B/totalseg/ct-mask-adapter-0000 \
#     --num_train_epochs 3 \
#     --per_device_train_batch_size 8 \
#     --per_device_eval_batch_size 2 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "steps" \
#     --eval_accumulation_steps 1 \
#     --eval_steps 0.2 \
#     --save_strategy "steps" \
#     --save_steps 2000 \
#     --save_total_limit 1 \
#     --learning_rate 1e-4 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 0.001 \
#     --gradient_checkpointing False \
#     --dataloader_pin_memory True\
#     --dataloader_num_workers 8 \
#     --report_to all \
#     --model_max_length 512 \
#     --use_mask True \
#     --use_ct True

accelerate launch /mym3d/LaMed/src/train/train.py \
    --version v0 \
    --model_name_or_path GoodBaiBai88/M3D-LaMed-Phi-3-4B \
    --model_type phi3 \
    --lora_enable True \
    --vision_tower vit3d \
    --pretrain_vision_model /mym3d/LaMed/pretrained_ViT.bin \
    --bf16 True \
    --output_dir /mym3d/LaMed/output/LaMed-Phi3-4B/totalseg/ct-mask-finetune-0000 \
    --pretrain_mm_mlp_adapter /mym3d/LaMed/output/LaMed-Phi3-4B/totalseg/ct-mask-adapter-0000/mm_projector.bin \
    --num_train_epochs 3 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "steps" \
    --eval_accumulation_steps 1 \
    --eval_steps 9112 \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 5 \
    --learning_rate 5e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 46 \
    --gradient_checkpointing False \
    --dataloader_pin_memory True\
    --dataloader_num_workers 8 \
    --report_to all \
    --model_max_length 512 \
    --use_mask True \
    --use_ct True