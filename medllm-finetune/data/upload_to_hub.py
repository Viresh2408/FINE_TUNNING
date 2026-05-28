import os
from datasets import load_dataset
from huggingface_hub import login

def main():
    # You can set HF_TOKEN environment variable
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("Please set the HF_TOKEN environment variable to push to the Hub.")
        print("Example: export HF_TOKEN='your_hf_token'")
        return

    login(token=hf_token)
    
    out_dir = os.path.join(os.path.dirname(__file__), "processed")
    
    if not os.path.exists(out_dir):
        print(f"Processed dataset directory not found at {out_dir}. Please run prepare_dataset.py first.")
        return
        
    print("Loading prepared dataset from disk...")
    dataset = load_dataset(
        "json", 
        data_files={
            "train": os.path.join(out_dir, "train.jsonl"),
            "validation": os.path.join(out_dir, "val.jsonl"),
            "test": os.path.join(out_dir, "test.jsonl")
        }
    )
    
    repo_id = "Viresh2408/medqa-instruct"
    print(f"Pushing dataset to Hugging Face Hub: {repo_id}")
    dataset.push_to_hub(repo_id, private=False)
    print("Dataset successfully pushed to the Hub!")

if __name__ == "__main__":
    main()
