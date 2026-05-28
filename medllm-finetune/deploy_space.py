import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

def main():
    print("--- Starting Hugging Face Space Deployment ---")
    
    # 1. Load env vars
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
        
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("ERROR: HF_TOKEN not found in .env. Please configure your .env file.")
        return
        
    # Standard repository identifiers
    username = "Viresh24"
    repo_name = "medllm-demo"
    repo_id = f"{username}/{repo_name}"
    
    # Initialize Hugging Face API Client
    api = HfApi(token=hf_token)
    
    # 2. Create the Space repo
    print(f"Provisioning Hugging Face Space: {repo_id} ...")
    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            private=False,
            exist_ok=True
        )
        print("SUCCESS: Repository created or verified successfully!")
    except Exception as e:
        print(f"Warning during repo creation: {e}")
        
    # 3. Add HF_TOKEN securely as a Space Secret
    print("Configuring HF_TOKEN as a secure Space Secret...")
    try:
        # Standard huggingface_hub API to add variables/secrets
        api.add_space_secret(
            repo_id=repo_id,
            key="HF_TOKEN",
            value=hf_token
        )
        print("SUCCESS: Secret HF_TOKEN configured successfully!")
    except Exception as e:
        print(f"Note: Could not programmatically set Space secret. You can set it manually in Settings: {e}")

    # 4. Prepare local deployment assets
    print("Creating temporary build assets locally...")
    
    # Readme metadata header
    readme_content = """---
title: MedLLM Assistant
emoji: 🩺
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.37.2
app_file: app.py
pinned: false
license: apache-2.0
python_version: "3.10"
---

# 🩺 MedLLM Interactive Demo
Interactive Web UI for **Llama 3.2 3B Instruct** fine-tuned on medical Q&A using QLoRA.
"""

    # Add transformers, peft, and accelerate dependencies for local CPU inference
    requirements_content = """transformers
peft
accelerate
huggingface_hub<0.25.0
gradio>=4.0.0
torch
"""

    # Gradio Web UI app.py loading model locally on CPU using TextIteratorStreamer
    app_py_content = """import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from peft import PeftModel
from threading import Thread
import gradio as gr

# Retrieve token securely from Hugging Face Space Settings
HF_TOKEN = os.environ.get("HF_TOKEN", "")

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
ADAPTER_ID = "Viresh24/medllm-lora"

print("--- Initializing Local CPU Inference Pipeline ---")
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID, token=HF_TOKEN)

print("Loading base Llama 3.2 3B Instruct model in bfloat16 on CPU...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    token=HF_TOKEN
)

print("Merging PEFT LoRA adapters locally...")
model = PeftModel.from_pretrained(base_model, ADAPTER_ID, token=HF_TOKEN)
model.eval()
print("SUCCESS: Local CPU Pipeline loaded successfully!")

def generate_medical_response(question):
    # Wrap in instruction prompt template
    prompt = f"[INST] {question.strip()} [/INST]"
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # TextIteratorStreamer is ideal for asynchronous streaming output
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Run text generation in a separate background thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Yield streamed text token by token
    partial_response = ""
    for chunk in streamer:
        partial_response += chunk
        yield partial_response

# Premium HSL Custom CSS Styling
custom_css = \"\"\"
footer {visibility: hidden}
.gradio-container {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
    font-family: 'Outfit', 'Inter', sans-serif !important;
}
.title-box {
    text-align: center;
    padding: 20px;
    margin-bottom: 20px;
    background: white;
    border-radius: 12px;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}
.title-box h1 {
    color: #1e3a8a !important;
    font-size: 2.5rem !important;
    font-weight: 800 !important;
    margin-bottom: 10px !important;
}
.title-box p {
    color: #475569 !important;
    font-size: 1.1rem !important;
}
.warning-box {
    background-color: #fef3c7 !important;
    border-left: 4px solid #d97706 !important;
    padding: 15px !important;
    border-radius: 6px !important;
    margin-top: 20px !important;
}
.warning-box p {
    color: #78350f !important;
    font-size: 0.95rem !important;
    margin: 0 !important;
}
\"\"\"

# Build Gradio Blocks Layout
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.HTML(\"\"\"
        <div class="title-box">
            <h1>%EF%B8%8F MedLLM Interactive Assistant</h1>
            <p>Meta's Llama 3.2 3B Instruct fine-tuned on Medical Q&A (QLoRA)</p>
        </div>
    \"\"\")
        
    with gr.Row():
        with gr.Column(scale=2):
            input_box = gr.Textbox(
                lines=4,
                placeholder="Type your medical query here... (e.g. What are the common symptoms of hypothyroidism?)",
                label="Ask a medical question"
            )
            submit_btn = gr.Button("Generate Answer %F0%9F%9A%80", variant="primary")
            
        with gr.Column(scale=3):
            output_box = gr.Textbox(
                lines=12,
                label="MedLLM Response (Streamed)",
                interactive=False
            )
            
    gr.Examples(
        examples=[
            ["What are the common symptoms of hypothyroidism?"],
            ["Is it safe to take ibuprofen with lisinopril?"],
            ["What is the standard dosage for amoxicillin in adults?"]
        ],
        inputs=input_box
    )
    
    gr.HTML(\"\"\"
        <div class="warning-box">
            <p>%E2%9A%A0%EF%B8%8F <strong>Disclaimer:</strong> This model is an AI research proof-of-concept. It is not a certified medical tool or clinical decision support system. Information generated is purely educational and should not replace professional medical diagnosis or consultation.</p>
        </div>
    \"\"\")
        
    # Wire events
    submit_btn.click(
        fn=generate_medical_response,
        inputs=input_box,
        outputs=output_box
    )
    input_box.submit(
        fn=generate_medical_response,
        inputs=input_box,
        outputs=output_box
    )

if __name__ == "__main__":
    demo.launch()
"""

# 5. Push files to the Hub
    print("Uploading deployment files directly to Hugging Face...")
    try:
        api.upload_file(
            path_or_fileobj=readme_content.encode("utf-8"),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="space"
        )
        api.upload_file(
            path_or_fileobj=requirements_content.encode("utf-8"),
            path_in_repo="requirements.txt",
            repo_id=repo_id,
            repo_type="space"
        )
        api.upload_file(
            path_or_fileobj=app_py_content.encode("utf-8"),
            path_in_repo="app.py",
            repo_id=repo_id,
            repo_type="space"
        )
        print("\nSUCCESS: Deployment successful! All files pushed to the Space Hub.")
        print(f"   Visit your live website here: https://huggingface.co/spaces/{repo_id}")
    except Exception as e:
        print(f"Error uploading assets: {e}")

if __name__ == "__main__":
    main()
