import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig


def get_lora_config():
    """Returns the LoRA configuration for MedLLM fine-tuning.

    r=4 (reduced from 8) — halves trainable parameters while keeping quality.
    All attention + MLP projections are targeted for maximum coverage.
    """
    return LoraConfig(
        r=4,                          # ✅ reduced from 8 — 2x fewer params → faster per step
        lora_alpha=8,                 # keep alpha/r = 2 ratio
        target_modules=[              # all attention + MLP projections
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )


def get_bnb_config():
    """Returns the BitsAndBytes configuration for 4-bit NF4 quantization.

    Automatically selects bf16 on A100/H100, falls back to fp16 on T4.
    T4 does NOT have native bf16 tensor cores — using fp16 gives native speed.
    """
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4 — best for LLMs
        bnb_4bit_compute_dtype=compute_dtype, # ✅ auto: fp16 on T4, bf16 on A100
        bnb_4bit_use_double_quant=True,       # quantize the quantization constants too
    )
