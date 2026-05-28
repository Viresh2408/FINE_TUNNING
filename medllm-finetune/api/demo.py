import gradio as gr
import requests
import json
import os

# Default to local FastAPI server if HF space env var not set
API_URL = os.environ.get("API_URL", "http://localhost:8000/generate")

def generate_answer(question):
    """Calls the FastAPI backend to generate a medical answer."""
    try:
        response = requests.post(
            API_URL,
            json={"question": question, "max_tokens": 256},
            stream=True
        )
        response.raise_for_status()
        
        # Handle SSE stream
        partial_answer = ""
        for line in response.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith("data: "):
                    data_str = decoded_line[6:]
                    if data_str == "[DONE]":
                        break
                    
                    data = json.loads(data_str)
                    partial_answer += data.get("text", "")
                    yield partial_answer
                    
    except requests.exceptions.RequestException as e:
        yield f"Error connecting to backend API: {str(e)}"

# Define Gradio UI
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    gr.Markdown(
        """
        # 🩺 MedLLM Assistant
        **Llama 3.2 3B Instruct** fine-tuned on medical Q&A using QLoRA.
        
        *Disclaimer: This is an AI research project and not meant for actual medical diagnosis. Always consult a certified healthcare professional.*
        """
    )
    
    with gr.Row():
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                lines=3, 
                placeholder="E.g. What are the common symptoms of hypothyroidism?", 
                label="Ask a medical question"
            )
            submit_btn = gr.Button("Generate Answer", variant="primary")
            
        with gr.Column(scale=3):
            answer_output = gr.Textbox(
                lines=10, 
                label="MedLLM Answer",
                interactive=False
            )
            
    gr.Examples(
        examples=[
            ["What are the common symptoms of hypothyroidism?"],
            ["Is it safe to take ibuprofen with lisinopril?"],
            ["What is the standard dosage for amoxicillin in adults?"]
        ],
        inputs=question_input
    )
    
    submit_btn.click(
        fn=generate_answer,
        inputs=question_input,
        outputs=answer_output
    )
    question_input.submit(
        fn=generate_answer,
        inputs=question_input,
        outputs=answer_output
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
