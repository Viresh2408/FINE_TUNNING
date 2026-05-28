import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def main():
    print("--- Merging LoRA Adapters into Base Model ---")
    
    base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
    adapter_path = "./medllm-lora"
    save_path = "./medllm-merged"
    
    if not os.path.exists(adapter_path):
        print(f"Adapter not found at {adapter_path}. Please run finetune.py first.")
        return
        
    print(f"Loading base model: {base_model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    print(f"Loading tokenizer: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    print(f"Loading PEFT adapter from {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Merging adapters into base model...")
    merged_model = model.merge_and_unload()
    
    print(f"Saving merged model to {save_path}")
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print("Merge complete!")

if __name__ == "__main__":
    main()
