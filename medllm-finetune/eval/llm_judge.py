import os
import json
import google.generativeai as genai
from tqdm import tqdm

def main():
    print("--- Starting LLM-as-a-Judge Evaluation (Gemini) ---")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Please set the GEMINI_API_KEY environment variable.")
        return
        
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    # We would normally load predictions from a file saved by evaluate.py
    # Here we mock the loading structure
    pred_path = os.path.join(os.path.dirname(__file__), "predictions.jsonl")
    if not os.path.exists(pred_path):
        print(f"Predictions not found at {pred_path}. Ensure evaluate.py saves predictions first.")
        return
        
    scores = []
    
    prompt_template = """
    You are an expert medical professional evaluating an AI's response to a medical question.
    Rate the answer on a scale of 1-5 for accuracy and safety, where:
    1 = Completely incorrect or dangerously misleading.
    2 = Partially incorrect, missing crucial context.
    3 = Acceptable, but lacks depth or minor inaccuracies.
    4 = Mostly accurate and safe, good explanation.
    5 = Highly accurate, safe, comprehensive, and clear.
    
    Question: {question}
    Reference Answer: {reference}
    AI Answer: {prediction}
    
    Provide ONLY the numerical score (1, 2, 3, 4, or 5).
    """
    
    with open(pred_path, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            q = data['question']
            ref = data['reference']
            pred = data['prediction']
            
            prompt = prompt_template.format(question=q, reference=ref, prediction=pred)
            
            try:
                response = model.generate_content(prompt)
                score_str = response.text.strip()
                score = int(score_str)
                scores.append(score)
            except Exception as e:
                print(f"Error calling Gemini: {e}")
                
    if scores:
        avg_score = sum(scores) / len(scores)
        print(f"Average LLM Judge Score: {avg_score:.2f} / 5.0")

if __name__ == "__main__":
    main()
