"""
Pydantic schemas for API request/response validation.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, validator


class PredictionRequest(BaseModel):
    """Single text prediction request."""

    text: str = Field(
        ..., min_length=1, max_length=5000, description="Text to analyze for sentiment"
    )

    include_probabilities: bool = Field(
        default=False, description="Include prediction probabilities in response"
    )

    @validator("text")
    def text_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty or only whitespace")
        return v


class PredictionResponse(BaseModel):
    """Single text prediction response."""

    text: str = Field(description="Original input text")
    sentiment: str = Field(description="Predicted sentiment (positive/negative)")
    confidence: float = Field(ge=0.0, le=1.0, description="Prediction confidence score")
    model: str = Field(default="distilbert", description="Model used for prediction")
    inference_time: Optional[float] = Field(
        default=None, description="Inference time in seconds"
    )
    prediction_probabilities: Optional[Dict[str, float]] = Field(
        default=None, description="Probability scores for each class"
    )


class BatchPredictionRequest(BaseModel):
    """Batch text prediction request."""

    texts: List[str] = Field(
        ..., min_items=1, max_items=1000, description="List of texts to analyze"
    )

    batch_size: int = Field(
        default=16, ge=1, le=32, description="Batch size for processing"
    )

    include_probabilities: bool = Field(
        default=False, description="Include prediction probabilities in response"
    )

    @validator("texts")
    def texts_must_not_be_empty(cls, v):
        for i, text in enumerate(v):
            if not text.strip():
                raise ValueError(
                    f"Text at index {i} cannot be empty or only whitespace"
                )
            if len(text) > 5000:
                raise ValueError(
                    f"Text at index {i} exceeds maximum length of 5000 characters"
                )
        return v


class BatchPredictionResponse(BaseModel):
    """Batch text prediction response."""

    predictions: List[PredictionResponse] = Field(description="List of predictions")
    total_texts: int = Field(description="Total number of texts processed")
    total_time: float = Field(description="Total processing time in seconds")
    average_time_per_text: float = Field(description="Average time per text in seconds")


class HealthResponse(BaseModel):
    """Health check response."""

    healthy: bool = Field(description="Overall health status")
    model_loaded: bool = Field(description="Whether model is loaded")
    device: str = Field(description="Computation device")
    timestamp: str = Field(description="Health check timestamp")
    error: Optional[str] = Field(description="Error message if unhealthy")


class ModelInfoResponse(BaseModel):
    """Model information response."""

    model_name: str = Field(description="Model name")
    model_path: str = Field(description="Path to model files")
    device: str = Field(description="Computation device")
    is_loaded: bool = Field(description="Whether model is loaded")
    parameters: int = Field(description="Number of model parameters")
    tokenizer_vocab_size: int = Field(description="Tokenizer vocabulary size")
    max_sequence_length: int = Field(description="Maximum sequence length")
    metadata: Dict = Field(description="Model training metadata")
    performance: Optional[Dict] = Field(description="Performance statistics")


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(description="Error message")
    detail: Optional[str] = Field(description="Detailed error information")
    timestamp: str = Field(description="Error timestamp")
