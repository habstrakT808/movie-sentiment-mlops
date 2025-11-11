"""
FastAPI application for movie sentiment analysis.
Provides REST API endpoints for sentiment prediction using DistilBERT.
"""

import time
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.deployment.middleware import LoggingMiddleware, PrometheusMiddleware
from src.deployment.model_loader import get_model_loader, initialize_model
from src.deployment.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Movie Sentiment Analysis API",
    description="REST API for movie review sentiment analysis using DistilBERT",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(PrometheusMiddleware)
app.add_middleware(LoggingMiddleware)

# Global model loader
model_loader = None


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup."""
    global model_loader
    logger.info("Starting up Movie Sentiment Analysis API...")

    try:
        # Initialize model
        logger.info("Initializing DistilBERT model...")
        success = initialize_model()

        if success:
            model_loader = get_model_loader()
            logger.info("[SUCCESS] API startup complete - Model loaded and ready!")
        else:
            logger.error("[ERROR] Failed to initialize model during startup")
            raise RuntimeError("Model initialization failed")

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Movie Sentiment Analysis API...")


# Error handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    errors = exc.errors()
    error_messages = [f"{err['loc']}: {err['msg']}" for err in errors]
    error_detail = "; ".join(error_messages)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Validation error",
            detail=error_detail,
            timestamp=datetime.now().isoformat(),
        ).dict(),
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=str(exc.detail) if exc.detail else "HTTP error occurred",
            detail=str(exc.detail) if exc.detail else None,
            timestamp=datetime.now().isoformat(),
        ).dict(),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}")
    import traceback

    logger.error(traceback.format_exc())
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=datetime.now().isoformat(),
        ).dict(),
    )


# API Endpoints


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Movie Sentiment Analysis API",
        "version": "1.0.0",
        "model": "DistilBERT",
        "status": "running",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "health": "/health",
            "model_info": "/model/info",
            "metrics": "/metrics",
            "docs": "/docs",
        },
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_sentiment(request: PredictionRequest):
    """
    Predict sentiment for a single text.

    Args:
        request: Prediction request with text to analyze

    Returns:
        Prediction response with sentiment and confidence
    """
    try:
        if model_loader is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        logger.info(
            f"Processing single prediction request (length: {len(request.text)})"
        )

        # Get prediction
        result = model_loader.predict_single(request.text)

        # Create response
        response = PredictionResponse(
            text=result["text"],
            sentiment=result["sentiment"],
            confidence=result["confidence"],
            model=result["model"],
            inference_time=result.get("inference_time"),
            prediction_probabilities=result.get("prediction_probabilities")
            if request.include_probabilities
            else None,
        )

        logger.info(
            f"Prediction complete: {result['sentiment']} ({result['confidence']:.4f})"
        )

        return response

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}",
        )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch_sentiment(request: BatchPredictionRequest):
    """
    Predict sentiment for multiple texts.

    Args:
        request: Batch prediction request with list of texts

    Returns:
        Batch prediction response with all results
    """
    try:
        if model_loader is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        logger.info(f"Processing batch prediction request ({len(request.texts)} texts)")

        start_time = time.time()

        # Get batch predictions
        results = model_loader.predict_batch(request.texts, request.batch_size)

        total_time = time.time() - start_time

        # Create response objects
        predictions = []
        for result in results:
            pred_response = PredictionResponse(
                text=result["text"],
                sentiment=result["sentiment"],
                confidence=result["confidence"],
                model=result["model"],
                inference_time=result.get("inference_time"),
                prediction_probabilities=result.get("prediction_probabilities")
                if request.include_probabilities
                else None,
            )

            predictions.append(pred_response)

        # Create batch response
        response = BatchPredictionResponse(
            predictions=predictions,
            total_texts=len(request.texts),
            total_time=total_time,
            average_time_per_text=total_time / len(request.texts),
        )

        logger.info(
            f"Batch prediction complete: {len(predictions)} results in {total_time:.2f}s"
        )

        return response

    except ValueError as e:
        # Handle validation errors (e.g., batch too large)
        logger.warning(f"Batch prediction validation error: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}",
        )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        Health status of the API and model
    """
    try:
        if model_loader is None:
            return HealthResponse(
                healthy=False,
                model_loaded=False,
                device="unknown",
                timestamp=datetime.now().isoformat(),
                error="Model not initialized",
            )

        # Perform model health check
        health_result = model_loader.health_check()

        return HealthResponse(
            healthy=health_result["healthy"],
            model_loaded=health_result.get("model_loaded", False),
            device=health_result.get("device", "unknown"),
            timestamp=datetime.now().isoformat(),
            error=health_result.get("error"),
        )

    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            healthy=False,
            model_loaded=False,
            device="unknown",
            timestamp=datetime.now().isoformat(),
            error=str(e),
        )


@app.get("/model/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get model information and metadata.

    Returns:
        Model information including metadata and performance stats
    """
    try:
        if model_loader is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded",
            )

        # Get model info
        info = model_loader.get_model_info()

        return ModelInfoResponse(**info)

    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get model info: {str(e)}",
        )


@app.get("/metrics")
async def get_metrics():
    """
    Get Prometheus metrics.

    Returns:
        Prometheus metrics in text format
    """
    from src.deployment.utils import create_metrics_response

    return create_metrics_response()


# For development/testing
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting development server...")
    uvicorn.run(
        "src.deployment.api:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=True,
        log_level="info",
    )
