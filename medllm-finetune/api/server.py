import torch
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
model_id = "Viresh2408/medllm-lora"  # Assume pulling merged model from Hub or local
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
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
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
