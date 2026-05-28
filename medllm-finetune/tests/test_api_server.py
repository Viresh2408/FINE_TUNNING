import pytest
import json
import asyncio
from pydantic import ValidationError
from api.server import load_model, generate_endpoint, GenerateRequest

# Use a synchronous autouse fixture to initialize the model cleanly
@pytest.fixture(autouse=True)
def init_model():
    asyncio.run(load_model())

@pytest.mark.asyncio
async def test_api_startup():
    """Verify server startup logic binds mock model and tokenizer correctly."""
    from api.server import model, tokenizer
    assert model is not None
    assert tokenizer is not None

@pytest.mark.asyncio
async def test_api_generate_streaming_success():
    """Verify endpoint generates token stream using SSE format and ends with [DONE]."""
    req = GenerateRequest(
        question="What is diabetes?",
        max_tokens=50,
        temperature=0.2
    )
    
    # Call the endpoint directly as a regular async function
    response = await generate_endpoint(req)
    assert response is not None
    
    # Read the streaming response from the body iterator
    chunks = []
    async for chunk in response.body_iterator:
        # Standardize bytes vs string types
        chunk_str = chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk
        chunks.append(chunk_str)
        
    # We expect 5 word chunks + 1 '[DONE]' chunk
    assert len(chunks) >= 6
    
    # Parse mock words returned by MockTokenizer and MockModel
    decoded_text = ""
    for chunk in chunks:
        if chunk.startswith("data: "):
            data_str = chunk[6:].strip()
            if data_str == "[DONE]":
                break
            data = json.loads(data_str)
            decoded_text += data.get("text", "")
            
    assert decoded_text == "This is a mock answer."

def test_generate_request_validation():
    """Verify that pydantic model enforces question schema requirements."""
    with pytest.raises(ValidationError):
        # Missing required 'question' field
        GenerateRequest(max_tokens=100)
