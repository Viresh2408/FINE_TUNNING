from datasets import load_dataset
import os

def main():
    print("--- Loading MedQuAD Dataset from Hugging Face ---")
    try:
        # Load the dataset
        dataset = load_dataset("lavita/MedQuAD")
        
        print("\nDataset loaded successfully!")
        print(f"Structure:\n{dataset}")
        
        if "train" in dataset:
            print("\nExample entry from the dataset:")
            print(dataset["train"][0])
            
        # Example of saving to disk (commented out by default)
        # save_path = os.path.join(os.path.dirname(__file__), "medquad_local")
        # dataset.save_to_disk(save_path)
        # print(f"\nSaved dataset to {save_path}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()
