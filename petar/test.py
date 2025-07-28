from petar.src.model import *
from petar.src.model.language_model import *
import torch 
from transformers import BitsAndBytesConfig


customized_kwargs = dict()
bnb_model_from_pretrained_args = {}

bnb_model_from_pretrained_args.update(
    dict(
        device_map={"": "cuda"},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',  # {'fp4', 'nf4'}
        ),
    )
)
customized_kwargs.update(bnb_model_from_pretrained_args)

model = PetarQwenForCausalLM.from_pretrained(
    "Qwen/Qwen2-0.5B-Instruct",
    cache_dir=None,
    attn_implementation="sdpa",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=False,
    **customized_kwargs,
)