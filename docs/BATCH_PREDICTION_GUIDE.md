# Batch Prediction Guide

## Overview

This document provides guidelines for batch prediction using the Movie Sentiment Analysis API, including recommended limits and best practices.

## Current Implementation Status

**Status**: ⏳ **Pending Phase 4-5 Implementation**

The API endpoints are planned but not yet implemented. This guide provides recommendations for when implementing batch prediction functionality.

## Model Loading Verification

### ✅ DistilBERT Model Compatibility

The DistilBERT model is saved using HuggingFace's standard format and is fully compatible with the Transformers library.

**Verification:**
- Model saved using `model.save_pretrained()` and `tokenizer.save_pretrained()`
- Model can be loaded using `DistilBertForSequenceClassification.from_pretrained()`
- Tokenizer can be loaded using `DistilBertTokenizer.from_pretrained()`
- All required files present: `config.json`, `model.safetensors`, tokenizer files

**To verify model loading:**
```bash
python scripts/verify_distilbert_loading.py
```

## Batch Prediction Limits

### Recommended Limits

| Resource | Recommended Limit | Hard Limit | Notes |
|----------|------------------|------------|-------|
| **Texts per Request** | 100 texts | 1,000 texts | Based on memory and timeout constraints |
| **Total Characters** | 500,000 chars | 5,000,000 chars | ~10,000 chars per text average |
| **Request Timeout** | 30 seconds | 60 seconds | For batch processing |
| **Batch Size (Internal)** | 16 texts | 32 texts | Model inference batch size |

### Memory Considerations

**Per Text Memory Usage:**
- Tokenization: ~1-2 KB per text
- Model Input: ~2-4 KB per text (512 tokens max)
- Model Output: ~0.5 KB per text
- **Total per text**: ~4-7 KB

**Batch Memory Usage:**
- 100 texts: ~400-700 KB
- 1,000 texts: ~4-7 MB
- 10,000 texts: ~40-70 MB

**GPU Memory (if available):**
- Model weights: ~250 MB
- Batch processing: ~50-100 MB per batch
- **Total GPU memory needed**: ~300-350 MB

### Performance Estimates

**Single Text Prediction:**
- CPU: ~50-100 ms per text
- GPU: ~10-20 ms per text

**Batch Prediction (100 texts):**
- CPU: ~5-10 seconds
- GPU: ~1-2 seconds

**Batch Prediction (1,000 texts):**
- CPU: ~50-100 seconds (may timeout)
- GPU: ~10-20 seconds

## API Design Recommendations

### Endpoint Structure

```python
# Single prediction
POST /predict
{
    "text": "This movie is great!",
    "model_type": "transformer"  # or "traditional"
}

# Batch prediction
POST /predict/batch
{
    "texts": ["text1", "text2", ...],
    "model_type": "transformer",
    "batch_size": 16  # Optional, default from config
}
```

### Request Validation

```python
# Recommended validation rules
MAX_TEXTS_PER_REQUEST = 1000
MAX_CHARS_PER_TEXT = 5000
MAX_TOTAL_CHARS = 5000000
MAX_REQUEST_TIMEOUT = 60  # seconds
```

### Response Format

```python
# Single prediction response
{
    "sentiment": "positive",
    "confidence": 0.9234,
    "model_used": "transformer",
    "processing_time_ms": 45.2
}

# Batch prediction response
{
    "results": [
        {
            "text": "text1",
            "sentiment": "positive",
            "confidence": 0.9234
        },
        ...
    ],
    "total_processed": 100,
    "processing_time_ms": 5234.5,
    "model_used": "transformer"
}
```

## Implementation Example

### FastAPI Batch Prediction Endpoint

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List
import time

app = FastAPI()

# Limits
MAX_TEXTS_PER_REQUEST = 1000
MAX_CHARS_PER_TEXT = 5000
MAX_TOTAL_CHARS = 5000000

class BatchPredictionRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, max_items=MAX_TEXTS_PER_REQUEST)
    model_type: str = "transformer"
    batch_size: int = Field(default=16, ge=1, le=32)

    @validator('texts')
    def validate_texts(cls, v):
        total_chars = sum(len(text) for text in v)
        if total_chars > MAX_TOTAL_CHARS:
            raise ValueError(f"Total characters ({total_chars}) exceeds limit ({MAX_TOTAL_CHARS})")

        for text in v:
            if len(text) > MAX_CHARS_PER_TEXT:
                raise ValueError(f"Text length ({len(text)}) exceeds limit ({MAX_CHARS_PER_TEXT})")

        return v

@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    start_time = time.time()

    try:
        # Load model (should be cached on startup)
        model = app.state.transformer_model if request.model_type == "transformer" else app.state.traditional_model

        # Process in batches
        results = []
        for i in range(0, len(request.texts), request.batch_size):
            batch_texts = request.texts[i:i + request.batch_size]
            batch_results = model.predict_batch(batch_texts)
            results.extend(batch_results)

        processing_time = (time.time() - start_time) * 1000

        return {
            "results": results,
            "total_processed": len(request.texts),
            "processing_time_ms": processing_time,
            "model_used": request.model_type
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Best Practices

### 1. **Chunking Large Batches**

For requests with >100 texts, consider chunking:
- Process in smaller batches (e.g., 50 texts at a time)
- Return partial results as they complete
- Use async processing for very large batches

### 2. **Timeout Handling**

- Set reasonable timeouts (30-60 seconds)
- Return partial results if timeout occurs
- Log timeout events for monitoring

### 3. **Error Handling**

- Validate input before processing
- Handle individual text errors gracefully
- Return error details for failed texts
- Continue processing remaining texts

### 4. **Performance Optimization**

- Use GPU if available
- Batch processing for efficiency
- Cache model in memory
- Use async processing for large batches

### 5. **Monitoring**

- Track processing time per batch size
- Monitor memory usage
- Log timeout events
- Track error rates

## Testing Recommendations

### Unit Tests

```python
def test_batch_prediction_small():
    """Test with small batch (10 texts)"""
    texts = ["text"] * 10
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 10

def test_batch_prediction_large():
    """Test with large batch (100 texts)"""
    texts = ["text"] * 100
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 100

def test_batch_prediction_limit():
    """Test with maximum allowed texts (1000)"""
    texts = ["text"] * 1000
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200

def test_batch_prediction_exceeds_limit():
    """Test with texts exceeding limit"""
    texts = ["text"] * 1001
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 422  # Validation error
```

## Summary

- **Model Loading**: ✅ Verified compatible with Transformers library
- **Recommended Batch Size**: 100 texts per request
- **Maximum Batch Size**: 1,000 texts per request
- **Processing Time**: ~1-2 seconds per 100 texts (GPU), ~5-10 seconds (CPU)
- **Memory Usage**: ~4-7 KB per text
- **Implementation**: Pending Phase 4-5

For questions or issues, refer to the main README.md or create an issue in the repository.
