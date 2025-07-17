#!/bin/bash

# sh LaMed/script/eval_phi3.sh "focal/ct-mask-finetune-0001" True True

EXPERIMENT_NAME=$1  # e.g., cvpr or mytest/run1
MASK=$2
CT=$3

# Check if argument is provided
if [ -z "$EXPERIMENT_NAME" ]; then
  echo "Usage: $0 <experiment_name>"
  exit 1
fi

OUTPUT_BIN="/mym3d/LaMed/output/LaMed-Phi3-4B/$EXPERIMENT_NAME/model_with_lora.bin"
CHECKPOINT_DIR="/mym3d/models/LaMed-Phi3-4B/$EXPERIMENT_NAME"
OUTPUT_DIR="/mym3d/LaMed/output/LaMed-Phi3-4B/$EXPERIMENT_NAME"

# Merge LoRA weights
CUDA_VISIBLE_DEVICES="" python3 LaMed/src/utils/merge_lora_weights_and_save_hf_model.py \
    --version="" \
    --model_type "phi3" \
    --vision_tower "vit3d" \
    --model_with_lora "$OUTPUT_BIN" \
    --output_dir "$CHECKPOINT_DIR" \
    --use_mask $MASK \
    --use_ct $CT

# Run captioning evaluation
python3 Bench/eval/eval_caption.py \
    --model_name_or_path "$CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_DIR"
