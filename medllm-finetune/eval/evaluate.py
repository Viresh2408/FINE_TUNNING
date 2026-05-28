import os
import json
import torch
import numpy as np
import wandb
from tqdm import tqdm
from unsloth import FastLanguageModel


def evaluate_model(
    adapter_path   = "./medllm-lora",
    gt_path        = "./eval/ground_truth.jsonl",
    run_name       = "finetuned-eval",
    max_samples    = 200,
    max_new_tokens = 200,
):
    """
    Evaluate the fine-tuned MedLLM adapter using ROUGE-L and BERTScore.

    Args:
        adapter_path   : Path to the saved LoRA adapter (or HF Hub repo id)
        gt_path        : Path to ground_truth.jsonl produced during dataset prep
        run_name       : W&B run name
        max_samples    : Number of test samples to evaluate (keep low for speed)
        max_new_tokens : Max tokens to generate per answer
    """
    print(f"--- Starting Evaluation: {run_name} ---")

    # ── Load model with Unsloth (required — model was trained with Unsloth patches) ──
    print(f"Loading adapter from {adapter_path} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name     = adapter_path,
        max_seq_length = 256,
        dtype          = None,
        load_in_4bit   = True,
        token          = os.environ.get("HF_TOKEN"),
    )
    FastLanguageModel.for_inference(model)   # 2x faster inference
    tokenizer.pad_token = tokenizer.eos_token

    # ── Load ground truth ─────────────────────────────────────────────────────
    if not os.path.exists(gt_path):
        raise FileNotFoundError(
            f"Ground truth not found at {gt_path}. "
            "Run data/prepare_dataset.py or Step 3 in the Colab notebook first."
        )

    questions, references = [], []
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            questions.append(data["question"])
            references.append(data["answer"])
            if len(questions) >= max_samples:
                break

    print(f"Loaded {len(questions)} samples for evaluation.")

    # ── Generate predictions ──────────────────────────────────────────────────
    predictions = []
    print("Generating predictions...")

    for q in tqdm(questions, desc="Generating"):
        prompt = f"[INST] {q.strip()} [/INST]"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens  = max_new_tokens,
                do_sample       = False,          # greedy for reproducibility
                pad_token_id    = tokenizer.eos_token_id,
            )

        response = tokenizer.decode(
            output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True
        )
        predictions.append(response.strip())

    # ── Compute metrics ───────────────────────────────────────────────────────
    print("Computing ROUGE-L ...")
    import evaluate as hf_evaluate

    rouge        = hf_evaluate.load("rouge")
    rouge_result = rouge.compute(predictions=predictions, references=references)
    rougeL       = rouge_result["rougeL"]
    rouge1       = rouge_result["rouge1"]

    print("Computing BERTScore (this takes ~1–2 min)...")
    bertscore    = hf_evaluate.load("bertscore")
    bert_result  = bertscore.compute(
        predictions=predictions, references=references, lang="en"
    )
    avg_bert_f1  = float(np.mean(bert_result["f1"]))

    # ── Print results ─────────────────────────────────────────────────────────
    print("\n" + "="*45)
    print(f"  ROUGE-1   : {rouge1:.4f}")
    print(f"  ROUGE-L   : {rougeL:.4f}")
    print(f"  BERTScore : {avg_bert_f1:.4f}")
    print("="*45)

    # ── Log to W&B ────────────────────────────────────────────────────────────
    wandb.init(project="medllm-finetune", name=run_name, resume="allow")
    wandb.log({
        "rouge1"       : rouge1,
        "rougeL"       : rougeL,
        "bertscore_f1" : avg_bert_f1,
        "num_samples"  : len(questions),
    })

    # Side-by-side qualitative table (first 20 samples)
    table = wandb.Table(columns=["Question", "Ground Truth", "Prediction"])
    for i in range(min(20, len(questions))):
        table.add_data(questions[i], references[i], predictions[i])
    wandb.log({f"{run_name}_examples": table})
    wandb.finish()

    print("✅ Evaluation complete!")
    return {"rouge1": rouge1, "rougeL": rougeL, "bertscore_f1": avg_bert_f1}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate fine-tuned MedLLM adapter")
    parser.add_argument("--adapter_path",   type=str, default="./medllm-lora",
                        help="Local adapter path or HF Hub repo id (e.g. Viresh24/medllm-lora)")
    parser.add_argument("--gt_path",        type=str, default="./eval/ground_truth.jsonl")
    parser.add_argument("--run_name",       type=str, default="finetuned-eval")
    parser.add_argument("--max_samples",    type=int, default=200)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    args = parser.parse_args()

    evaluate_model(
        adapter_path   = args.adapter_path,
        gt_path        = args.gt_path,
        run_name       = args.run_name,
        max_samples    = args.max_samples,
        max_new_tokens = args.max_new_tokens,
    )
