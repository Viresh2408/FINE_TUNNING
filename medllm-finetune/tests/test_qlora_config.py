import pytest
import torch
from peft import LoraConfig
from transformers import BitsAndBytesConfig
from train.qlora_config import get_lora_config, get_bnb_config

def test_get_lora_config():
    """Verify LoRA configuration targets optimal modules with correct rank/alpha ratio."""
    config = get_lora_config()
    
    assert isinstance(config, LoraConfig)
    assert config.r == 4
    assert config.lora_alpha == 8
    assert config.lora_dropout == 0.05
    assert config.bias == "none"
    assert config.task_type == "CAUSAL_LM"
    
    # Target all attention and MLP projections
    expected_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    assert set(config.target_modules) == expected_targets

def test_get_bnb_config():
    """Verify quantization parameter targets NF4 with double quantization."""
    config = get_bnb_config()
    
    assert isinstance(config, BitsAndBytesConfig)
    assert config.load_in_4bit is True
    assert config.bnb_4bit_quant_type == "nf4"
    assert config.bnb_4bit_use_double_quant is True
    
    # Check that compute_dtype is a torch.dtype
    assert isinstance(config.bnb_4bit_compute_dtype, torch.dtype)
    # On CPU/local CPU testing it should fall back to float16, on BF16-supported GPUs to bfloat16
    assert config.bnb_4bit_compute_dtype in (torch.float16, torch.bfloat16)
