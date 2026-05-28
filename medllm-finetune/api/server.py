import torch
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn
import json
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from threading import Thread

app = FastAPI(title="MedLLM API", description="Inference API for MedLLM fine-tuned model")

# Global variables for model and tokenizer
model = None
tokenizer = None
model_id = "Viresh24/medllm-lora"  # Assume pulling merged model from Hub or local
# To use local merged model, use model_id = "../train/medllm-merged"

class GenerateRequest(BaseModel):
    question: str
    max_tokens: int = 256
    temperature: float = 0.1
    top_p: float = 0.9

@app.on_event("startup")
async def load_model():
    global model, tokenizer
    print(f"Loading model {model_id}...")
    try:
        # Load base model meta-llama/Llama-3.2-3B-Instruct tokenizer and weights
        base_model_id = "meta-llama/Llama-3.2-3B-Instruct"
        print(f"Loading base model tokenizer: {base_model_id}")
        hf_token = os.environ.get("HF_TOKEN")
        tokenizer = AutoTokenizer.from_pretrained(base_model_id, token=hf_token)
        
        print(f"Loading base model weights: {base_model_id}")
        print(f"Loading base model: {base_model_id}")
        
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        dtype = torch.bfloat16 if use_bf16 else torch.float32
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            torch_dtype=dtype,
            low_cpu_mem_usage=True
        )
        
        from peft import PeftModel, LoraConfig
        import inspect
        from huggingface_hub import hf_hub_download
        
        print(f"Sanitizing and downloading PEFT adapters locally for {model_id}...")
        local_adapter_dir = "./local_api_adapter"
        os.makedirs(local_adapter_dir, exist_ok=True)
        
        # 1. Download adapter weights
        try:
            print("Downloading adapter_model.safetensors...")
            hf_hub_download(repo_id=model_id, filename="adapter_model.safetensors", token=hf_token, local_dir=local_adapter_dir)
        except Exception:
            print("safetensors fallback: Downloading adapter_model.bin...")
            hf_hub_download(repo_id=model_id, filename="adapter_model.bin", token=hf_token, local_dir=local_adapter_dir)
            
        # 2. Download and sanitize adapter_config.json to strip custom/incompatible keys
        config_cache_path = hf_hub_download(repo_id=model_id, filename="adapter_config.json", token=hf_token)
        with open(config_cache_path, "r") as f:
            config = json.load(f)
            
        # Dynamically filter config keys based on the installed PEFT version's LoraConfig signature
        lora_config_params = set(inspect.signature(LoraConfig.__init__).parameters.keys())
        sanitized_config = {k: v for k, v in config.items() if k in lora_config_params}
        
        # Save the sanitized config locally
        with open(os.path.join(local_adapter_dir, "adapter_config.json"), "w") as f:
            json.dump(sanitized_config, f, indent=4)
        print("SUCCESS: Adapters successfully sanitized locally!")
        
        print("Merging PEFT LoRA adapters...")
        model = PeftModel.from_pretrained(base_model, local_adapter_dir)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Not exiting here so the API still starts, but endpoints will fail

@app.post("/generate")
async def generate_endpoint(req: GenerateRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")
        
    prompt = f"[INST] {req.question.strip()} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # We use a custom generator for streaming
    async def stream_generator():
        # TextIteratorStreamer from transformers is ideal here
        from transformers import TextIteratorStreamer
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        
        generation_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=req.max_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            pad_token_id=tokenizer.eos_token_id
        )
        
        # Run generation in a background thread
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()
        
        for text in streamer:
            # Yield SSE format
            yield f"data: {json.dumps({'text': text})}\n\n"
            await asyncio.sleep(0.01)
            
        yield "data: [DONE]\n\n"
        
    return StreamingResponse(stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
