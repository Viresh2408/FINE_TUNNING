# 🩺 MedLLM Fine-Tuning Workspace

Welcome to the MedLLM Fine-Tuning workspace! This repository contains the complete codebase and pipeline for fine-tuning Meta's **Llama 3.2 3B Instruct** model for clinical medical question answering using **QLoRA**.

## 📁 Workspace Layout

*   **[`medllm-finetune/`](file:///c:/Project/FINE_Tunning/medllm-finetune/)**: The primary project directory containing all pipeline code, evaluation setups, and deployment configs.
    *   📘 **[Detailed README.md](file:///c:/Project/FINE_Tunning/medllm-finetune/README.md)**: Comprehensive deep-dive documentation detailing installation, architecture, metrics, and serving.
    *   📂 **[`data/`](file:///c:/Project/FINE_Tunning/medllm-finetune/data/)**: Raw/processed data curation scripts and Hugging Face uploading helpers.
    *   📂 **[`train/`](file:///c:/Project/FINE_Tunning/medllm-finetune/train/)**: Quantization configurations, fine-tuning scripts, adapter-merging code, and Google Colab Jupyter Notebooks.
    *   📂 **[`eval/`](file:///c:/Project/FINE_Tunning/medllm-finetune/eval/)**: Scripts for calculating ROUGE-L / BERTScore, plus Gemini-powered LLM Clinician Judge checks.
    *   📂 **[`api/`](file:///c:/Project/FINE_Tunning/medllm-finetune/api/)**: High-performance FastAPI server with streaming Server-Sent Events (SSE) and Gradio Web UI.
*   **[`load_data.py`](file:///c:/Project/FINE_Tunning/load_data.py)**: Root level standalone helper script for loading the `lavita/MedQuAD` dataset from Hugging Face for quick sandboxed inspection.

## 🚀 Getting Started

To dive straight into setup, installation, metric comparisons, or local docker-compose deployment, please head over to the main project README:

👉 **[Go to medllm-finetune/README.md](file:///c:/Project/FINE_Tunning/medllm-finetune/README.md)**
