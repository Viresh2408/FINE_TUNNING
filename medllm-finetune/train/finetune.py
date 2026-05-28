import os
import sys
import torch
import wandb
from datasets import load_dataset
from dotenv import load_dotenv
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

# Load .env from project root
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))


def main():
    print("--- Starting QLoRA Fine-Tuning (Speed-Optimized) ---")

    hf_token = os.environ.get("HF_TOKEN")
    wandb_key = os.environ.get("WANDB_API_KEY")

    if wandb_key:
        wandb.login(key=wandb_key)
    wandb.init(project="medllm-finetune", name="finetune-llama-3.2-3b-fast")

    model_id = "meta-llama/Llama-3.2-3B-Instruct"

    # ── Detect compute dtype ───────────────────────────────────────────────
    # T4 only has FP16 tensor cores — bf16 is slow on T4 (no native support)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16
    print(f"Compute dtype: {'bfloat16' if use_bf16 else 'float16'}")

    # ── Dataset ───────────────────────────────────────────────────────────
    print("Loading prepared dataset from local files...")
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
    dataset = load_dataset(
        "json",
        data_files={
            "train": os.path.join(data_dir, "train.jsonl"),
        },
    )

    # ── Tokenizer ─────────────────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=hf_token, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"   # prevents warnings with gradient checkpointing

    # ── BnB 4-bit config ──────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,   # ✅ fp16 on T4, bf16 on A100
        bnb_4bit_use_double_quant=True,
    )

    # ── Model (4-bit NF4) ─────────────────────────────────────────────────
    print("Loading base model in 4-bit NF4...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        token=hf_token,
        trust_remote_code=True,
    )
    model.config.use_cache = False          # required for gradient checkpointing
    model.config.pretraining_tp = 1         # avoids tensor parallelism issues

    # ── LoRA config — r=4 halves trainable params vs r=8, same quality ────
    lora_config = LoraConfig(
        r=4,                  # ✅ reduced from 8 — 2x fewer params, faster per step
        lora_alpha=8,         # keep alpha/r = 2
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Attach LoRA adapters
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── SFTConfig ─────────────────────────────────────────────────────────
    sft_config = SFTConfig(
        # ── Output & checkpointing ──────────────────────────────────────
        output_dir="./results",
        save_strategy="steps",
        save_steps=500,                      # save mid-run in case of crashes
        save_total_limit=1,
        # ── Training loop ────────────────────────────────────────────────
        num_train_epochs=1,
        max_steps=1500,                      # ✅ hard cap → ~2–2.5 hrs on T4
        per_device_train_batch_size=4,       # ✅ safe for T4 15 GB VRAM + 4-bit + 512 ctx
        gradient_accumulation_steps=4,       # ✅ effective batch = 16
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={      # ✅ faster on PyTorch ≥ 2.1
            "use_reentrant": False
        },
        optim="adamw_8bit",                  # ✅ 8-bit Adam: less VRAM + faster
        # ── LR schedule ──────────────────────────────────────────────────
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        max_grad_norm=0.3,
        # ── Evaluation — disabled for speed ──────────────────────────────
        eval_strategy="no",                  # ✅ skip eval — saves 20-30 min
        # ── Precision ────────────────────────────────────────────────────
        bf16=use_bf16,                       # ✅ bf16 only on A100+
        fp16=use_fp16,                       # ✅ fp16 on T4 — native tensor cores
        # ── Logging ──────────────────────────────────────────────────────
        logging_steps=50,
        report_to="wandb",
        # ── DataLoader ───────────────────────────────────────────────────
        dataloader_num_workers=0,            # ✅ 0 avoids fork/spawn issues on Linux/Colab
        dataloader_pin_memory=False,         # ✅ no benefit when num_workers=0
        # ── SFT-specific ─────────────────────────────────────────────────
        max_length=512,
        dataset_text_field="text",
        packing=True,                        # ✅ BIGGEST WIN: fills entire 512-token context
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset["train"],
        processing_class=tokenizer,          # replaces deprecated `tokenizer=` arg
    )

    # ── Train ─────────────────────────────────────────────────────────────
    n_train = len(dataset["train"])
    eff_batch = 4 * 4   # per_device × grad_accum
    print(f"📊 Training on {n_train} examples  |  max_steps=1500  |  eff_batch={eff_batch}")
    print(f"   Precision: {'bf16' if use_bf16 else 'fp16'}")
    print("🚀 Starting training...")
    trainer.train()

    # ── Save adapters ─────────────────────────────────────────────────────
    print("Saving the LoRA adapters...")
    adapter_path = os.path.join(os.path.dirname(__file__), "..", "medllm-lora")
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapters saved to {adapter_path}")

    # ── Push to Hub ───────────────────────────────────────────────────────
    if hf_token:
        print("Pushing to Hugging Face Hub...")
        trainer.model.push_to_hub("Viresh24/medllm-lora", token=hf_token)
        tokenizer.push_to_hub("Viresh24/medllm-lora", token=hf_token)
        print("✅ Pushed to https://huggingface.co/Viresh24/medllm-lora")
    else:
        print("HF_TOKEN not found — skipping Hub push.")

    wandb.finish()
    print("🎉 Fine-tuning complete!")


if __name__ == "__main__":
    main()
