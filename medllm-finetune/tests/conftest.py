import os
import sys
import pytest
from unittest.mock import MagicMock
import torch

# Ensure the project directory is on the path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class MockBatchEncoding(dict):
    """Simulates Hugging Face BatchEncoding with support for device migration."""
    def to(self, device):
        # Tensors are simple CPU mocks, so no-op device conversion
        return self

class MockTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.eos_token_id = 2
        self.pad_token_id = 0
        self.padding_side = "right"

    def __call__(self, text, *args, **kwargs):
        # Return custom BatchEncoding wrapper supporting .to()
        return MockBatchEncoding({
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]])
        })

    def decode(self, token_ids, skip_special_tokens=True, *args, **kwargs):
        # Simple token-to-text mapping
        token_map = {
            0: "",
            1: "This",
            2: " is",
            3: " a",
            4: " mock",
            5: " answer."
        }
        # If token_ids is a tensor or list, handle it
        if hasattr(token_ids, "tolist"):
            ids = token_ids.tolist()
        elif isinstance(token_ids, int):
            ids = [token_ids]
        else:
            ids = token_ids
            
        # Standard TextIteratorStreamer passes 1D lists of token IDs
        decoded_words = []
        for i in ids:
            if isinstance(i, list):
                decoded_words.append(" ".join(token_map.get(x, "") for x in i))
            else:
                decoded_words.append(token_map.get(i, ""))
        return "".join(decoded_words)

    def save_pretrained(self, *args, **kwargs):
        pass

class MockModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.config = MagicMock()
        self.config.use_cache = True
        self.config.pretraining_tp = 1

    def eval(self):
        """Mock the eval mode switch used by standard inference pipelines."""
        pass

    def generate(self, *args, **kwargs):
        streamer = kwargs.get("streamer", None)
        if streamer is not None:
            # Emulate streamer behavior by putting mock tensors
            # TextIteratorStreamer.put expects a tensor
            # We prefix a dummy token (0) so it's skipped by skip_prompt logic
            for token_id in [0, 1, 2, 3, 4, 5]:
                streamer.put(torch.tensor([token_id]))
            streamer.end()
        return torch.tensor([[1, 2, 3, 4, 5]])

    def save_pretrained(self, *args, **kwargs):
        pass

@pytest.fixture
def mock_tokenizer():
    return MockTokenizer()

@pytest.fixture
def mock_model():
    return MockModel()

@pytest.fixture(autouse=True)
def patch_transformers(monkeypatch, mock_tokenizer, mock_model):
    """Automatically intercepts Hugging Face model and tokenizer loads in tests."""
    import transformers
    import peft
    
    # Patch AutoTokenizer from_pretrained
    monkeypatch.setattr(
        transformers.AutoTokenizer, 
        "from_pretrained", 
        lambda *args, **kwargs: mock_tokenizer
    )
    
    # Patch AutoModelForCausalLM from_pretrained
    monkeypatch.setattr(
        transformers.AutoModelForCausalLM, 
        "from_pretrained", 
        lambda *args, **kwargs: mock_model
    )
    
    # Patch PeftModel from_pretrained to prevent PyTorch module attribute errors
    monkeypatch.setattr(
        peft.PeftModel,
        "from_pretrained",
        lambda *args, **kwargs: mock_model
    )
