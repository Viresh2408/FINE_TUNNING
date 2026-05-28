import os
import pandas as pd
from datasets import load_dataset, Dataset, DatasetDict

def format_instruction(question, answer):
    """Formats the data into Llama 3 instruction format."""
    return f"[INST] {question.strip()} [/INST] {answer.strip()}"

def main():
    print("--- Preparing MedQuAD Dataset ---")
    
    # Load dataset
    print("Loading lavita/MedQuAD...")
    dataset = load_dataset("lavita/MedQuAD")
    df = dataset['train'].to_pandas()
    
    print(f"Original dataset size: {len(df)}")
    
    # Filter short answers (noise)
    # Estimate tokens by splitting by space (rough heuristic, sufficient for filtering)
    df['answer_len'] = df['answer'].astype(str).apply(lambda x: len(x.split()))
    df = df[df['answer_len'] >= 20].copy()
    print(f"Dataset size after filtering short answers (< 20 words): {len(df)}")
    
    # Format into instruction format
    print("Formatting into instruction format...")
    df['text'] = df.apply(lambda row: format_instruction(row['question'], row['answer']), axis=1)
    
    # Keep only the formatted text
    df_clean = df[['text']]
    
    # Create HuggingFace dataset from pandas
    hf_dataset = Dataset.from_pandas(df_clean, preserve_index=False)
    
    # Train / Val / Test Split (90 / 5 / 5)
    print("Splitting into 90/5/5 train/val/test...")
    train_testval = hf_dataset.train_test_split(test_size=0.1, seed=42)
    test_val = train_testval['test'].train_test_split(test_size=0.5, seed=42)
    
    final_dataset = DatasetDict({
        'train': train_testval['train'],
        'validation': test_val['train'],
        'test': test_val['test']
    })
    
    print("Final Split Sizes:")
    print(f"Train: {len(final_dataset['train'])}")
    print(f"Validation: {len(final_dataset['validation'])}")
    print(f"Test: {len(final_dataset['test'])}")
    
    # Save as JSONL
    out_dir = os.path.join(os.path.dirname(__file__), "processed")
    os.makedirs(out_dir, exist_ok=True)
    
    train_path = os.path.join(out_dir, "train.jsonl")
    val_path = os.path.join(out_dir, "val.jsonl")
    test_path = os.path.join(out_dir, "test.jsonl")
    
    print(f"Saving to {out_dir}...")
    final_dataset['train'].to_json(train_path)
    final_dataset['validation'].to_json(val_path)
    final_dataset['test'].to_json(test_path)
    
    # Save test set ground truth separately for evaluation
    df_test = df.loc[df.index.isin(test_val['test'].to_pandas().index)] if not test_val['test'].to_pandas().empty else test_val['test'].to_pandas()
    # Actually, we can just save the test dataset with question/answer pairs for ground truth
    df_test_raw = df.sample(n=len(final_dataset['test']), random_state=42)[['question', 'answer']]
    
    eval_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "eval")
    os.makedirs(eval_dir, exist_ok=True)
    gt_path = os.path.join(eval_dir, "ground_truth.jsonl")
    df_test_raw.to_json(gt_path, orient='records', lines=True)
    
    print(f"Saved ground truth for eval to {gt_path}")
    print("Done!")

if __name__ == "__main__":
    main()
