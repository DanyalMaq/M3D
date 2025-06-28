import sys
sys.path.append("~/M3D/")

import os
import logging
from typing import Optional, List, Dict
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM
from dataclasses import dataclass, field
from LaMed.src.dataset.multi_dataset import UniDatasets, CapDataset, TextDatasets, VQADataset, MyCapDataset
from LaMed.src.model.language_model import LamedLlamaForCausalLM, LamedPhi3ForCausalLM
from LaMed.src.train.lamed_trainer import LaMedTrainer
import warnings

class DataArguments:
    data_root: str = field(default="/mym3d/Data/data/", metadata={"help": "Root directory for all data."})

    # caption data
    cap_data_path: str = field(default="/mym3d/Data/output.json", metadata={"help": "Path to caption data."})

tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )


# ret = {
#         'pet': pet,
#         'contour': contour,
#         'ct': ct,
# }
dataset = MyCapDataset(data_args, tokenizer, mode='validation')