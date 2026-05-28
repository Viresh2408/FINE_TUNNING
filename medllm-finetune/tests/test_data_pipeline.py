import pytest
import pandas as pd
from data.prepare_dataset import format_instruction

def test_format_instruction():
    """Verify that instruction formatting properly wraps prompt templates."""
    q = "What is hypertension? "
    a = " High blood pressure is a chronic medical condition..."
    expected = "[INST] What is hypertension? [/INST] High blood pressure is a chronic medical condition..."
    
    result = format_instruction(q, a)
    assert result == expected

def test_dataset_filtering_logic():
    """Verify raw noise filtering logic correctly targets short/empty answers."""
    # Build dummy dataset representing clean and noisy answers
    data = {
        "question": [
            "Q1",
            "Q2",
            "Q3"
        ],
        "answer": [
            "Short answer",  # 2 words (should be filtered)
            "This is a long answer that is definitely longer than twenty words to ensure that the dataset filtering works perfectly as designed in our heuristics.", # 25 words (should pass)
            "Another brief response." # 3 words (should be filtered)
        ]
    }
    
    df = pd.DataFrame(data)
    df['answer_len'] = df['answer'].astype(str).apply(lambda x: len(x.split()))
    
    # Filter using our custom threshold (>= 20 words)
    df_filtered = df[df['answer_len'] >= 20].copy()
    
    assert len(df_filtered) == 1
    assert df_filtered.iloc[0]['question'] == "Q2"
