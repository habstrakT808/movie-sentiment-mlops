"""
FastAPI application for movie sentiment analysis.
Provides REST API endpoints for sentiment prediction using DistilBERT.
"""

import time
from datetime import datetime
from typing import Dict

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.deployment.middleware import (
    LoggingMiddleware,
    PrometheusMiddleware,
    update_drift_metrics,
)
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
from src.monitoring.drift_detector import DataDriftDetector
from src.training.continuous_learning import ContinuousLearner
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

# Initialize scheduler for background tasks
scheduler = AsyncIOScheduler()

# Global drift detector
drift_detector = None

# Global continuous learner
continuous_learner = None

# Retraining lock to prevent concurrent retraining
is_retraining = False


async def run_drift_detection():
    """
    Background task to run drift detection periodically.
    Runs every 15 minutes.
    """
    global drift_detector

    try:
        logger.info("Running scheduled drift detection...")

        # Initialize detector if not already done
        if drift_detector is None:
            drift_detector = DataDriftDetector()

        # Check if sufficient data available
        if not drift_detector.has_sufficient_data(min_samples=50):
            logger.info(
                "Insufficient data for drift detection (< 50 predictions), skipping..."
            )
            return

        # Run drift detection
        results = drift_detector.detect_drift(window_size=100)

        # Update Prometheus metrics
        update_drift_metrics(results)

        # Log summary
        alert_level = results.get("alert_level", "ok")
        overall_score = results["drift_scores"]["overall"]

        if alert_level == "critical":
            logger.warning(
                f"🚨 CRITICAL DRIFT DETECTED! Score: {overall_score:.4f} | "
                f"Text Length: {results['drift_scores']['text_length']:.4f}, "
                f"Word Count: {results['drift_scores']['word_count']:.4f}, "
                f"Sentiment: {results['drift_scores']['sentiment']:.4f}"
            )
        elif alert_level == "warning":
            logger.info(f"⚠️ Moderate drift detected. Score: {overall_score:.4f}")
        else:
            logger.info(f"✅ No significant drift. Score: {overall_score:.4f}")

    except Exception as e:
        logger.error(f"Drift detection task failed: {e}")
        import traceback

        logger.error(traceback.format_exc())


async def run_continuous_learning():
    """
    Background task untuk continuous learning.
    Run setiap 1 jam (configurable).

    Checks if sufficient feedback is available and triggers retraining.
    """
    global continuous_learner, is_retraining

    # Check if already retraining
    if is_retraining:
        logger.info("⏸️ Retraining already in progress, skipping...")
        return

    try:
        logger.info("🔄 Running continuous learning check...")

        # Initialize learner if not already done
        if continuous_learner is None:
            continuous_learner = ContinuousLearner()

        # Check if sufficient feedback available
        if not continuous_learner.has_sufficient_feedback():
            feedback_count = len(continuous_learner.collect_feedback_from_db())
            logger.info(
                f"⏳ Insufficient feedback for retraining "
                f"({feedback_count} samples, need {continuous_learner.retrain_threshold}), "
                f"skipping..."
            )
            return

        # Set retraining flag
        is_retraining = True
        logger.info("🚀 Starting model retraining...")

        # Trigger retraining
        results = continuous_learner.trigger_retraining()

        # Check results
        if results.get("status") == "success":
            if results.get("should_deploy"):
                logger.info(
                    f"✅ NEW MODEL DEPLOYED! "
                    f"F1: {results['new_model_metrics']['test_f1']:.4f} "
                    f"(improvement: {results['improvement']['f1']:.2%})"
                )
                logger.warning(
                    "⚠️ Please restart API to load new model: "
                    "docker-compose restart sentiment-api"
                )
            else:
                logger.info(
                    f"⚠️ New model not deployed. "
                    f"F1: {results['new_model_metrics']['test_f1']:.4f} "
                    f"(production: {results['production_metrics']['test_f1']:.4f}, "
                    f"improvement: {results['improvement']['f1']:.2%})"
                )
        else:
            logger.error(
                f"❌ Retraining failed: {results.get('message', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"❌ Continuous learning task failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        # Always reset retraining flag
        is_retraining = False


async def run_periodic_data_collection():
    """
    Background task untuk collect data baru secara berkala.
    Run setiap N jam (configurable via env var).

    Flow:
    1. Collect data baru dari Reddit dan Kaggle
    2. Preprocess data baru
    3. Merge dengan training data existing
    4. Check jika ada cukup data baru untuk retraining
    5. Trigger retraining jika threshold tercapai
    """
    global continuous_learner, is_retraining

    import os
    from pathlib import Path

    # Check if already retraining
    if is_retraining:
        logger.info("⏸️ Retraining already in progress, skipping data collection...")
        return

    try:
        logger.info("=" * 80)
        logger.info("📥 STARTING PERIODIC DATA COLLECTION")
        logger.info("=" * 80)

        # 1. Initialize collectors
        from src.data_collection.periodic_collector import PeriodicDataCollector
        from src.preprocessing.incremental_preprocessor import IncrementalPreprocessor

        collector = PeriodicDataCollector()
        preprocessor = IncrementalPreprocessor()

        # 2. Collect new data
        logger.info("🔄 Collecting new data from Reddit and Kaggle...")
        collection_result = collector.collect_incremental_data(
            min_samples_per_source=100,  # Configurable
            time_filter="week",  # Collect from last week
            max_reddit_samples=1000,
            max_kaggle_samples=500,
        )

        if collection_result["status"] != "success":
            logger.warning(
                f"⚠️ Data collection failed or no new data: "
                f"{collection_result.get('message', 'Unknown error')}"
            )
            return

        new_samples = collection_result["total_new_samples"]
        logger.info(f"✅ Collected {new_samples} new samples")

        # 3. Check threshold for retraining
        DATA_COLLECTION_RETRAIN_THRESHOLD = int(
            os.getenv("DATA_COLLECTION_RETRAIN_THRESHOLD", "3000")
        )

        if new_samples < DATA_COLLECTION_RETRAIN_THRESHOLD:
            logger.info(
                f"⏳ Insufficient new data for retraining "
                f"({new_samples} samples, need {DATA_COLLECTION_RETRAIN_THRESHOLD}), "
                f"skipping retraining..."
            )
            return

        # 4. Preprocess and merge
        logger.info("🔄 Preprocessing and merging new data...")
        preprocess_result = preprocessor.preprocess_incremental_data(
            incremental_data_path=Path(collection_result["output_path"]),
            merge_with_existing=True,
        )

        if preprocess_result["status"] != "success":
            logger.error(
                f"❌ Preprocessing failed: {preprocess_result.get('message', 'Unknown error')}"
            )
            return

        logger.info(
            f"✅ Preprocessing complete. "
            f"Total samples after merge: {preprocess_result['total_samples_after_merge']}"
        )

        # 5. Trigger retraining
        is_retraining = True
        logger.info("🚀 Starting model retraining with new collected data...")

        if continuous_learner is None:
            continuous_learner = ContinuousLearner()

        results = continuous_learner.trigger_data_collection_retraining(
            new_data_path=Path(preprocess_result["output_paths"]["train"])
        )

        # 6. Check results
        if results.get("status") == "success":
            if results.get("should_deploy"):
                logger.info(
                    f"✅ NEW MODEL DEPLOYED! "
                    f"F1: {results['new_model_metrics']['test_f1']:.4f} "
                    f"(improvement: {results['improvement']['f1']:.2%})"
                )
                logger.warning(
                    "⚠️ Please restart API to load new model: "
                    "docker-compose restart sentiment-api"
                )
            else:
                logger.info(
                    f"⚠️ New model not deployed. "
                    f"F1: {results['new_model_metrics']['test_f1']:.4f} "
                    f"(production: {results['production_metrics']['test_f1']:.4f}, "
                    f"improvement: {results['improvement']['f1']:.2%})"
                )
        else:
            logger.error(
                f"❌ Retraining failed: {results.get('message', 'Unknown error')}"
            )

    except Exception as e:
        logger.error(f"❌ Periodic data collection task failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
    finally:
        is_retraining = False
        logger.info("=" * 80)
        logger.info("✅ PERIODIC DATA COLLECTION COMPLETED")
        logger.info("=" * 80)


@app.on_event("startup")
async def startup_event():
    """Initialize model and start background tasks on startup."""
    global model_loader, drift_detector, continuous_learner
    logger.info("Starting up Movie Sentiment Analysis API...")

    try:
        # Initialize model
        logger.info("Initializing DistilBERT model...")
        success = initialize_model()

        if success:
            model_loader = get_model_loader()
            logger.info("[SUCCESS] Model loaded and ready!")
        else:
            logger.error("[ERROR] Failed to initialize model during startup")
            raise RuntimeError("Model initialization failed")

        # Initialize drift detector
        logger.info("Initializing drift detector...")
        drift_detector = DataDriftDetector()
        logger.info("[SUCCESS] Drift detector initialized!")

        # Initialize continuous learner
        logger.info("Initializing continuous learner...")
        continuous_learner = ContinuousLearner()
        logger.info("[SUCCESS] Continuous learner initialized!")

        # Start background scheduler for drift detection
        logger.info("Starting background scheduler...")
        scheduler.add_job(
            run_drift_detection,
            "interval",
            minutes=15,  # Run every 15 minutes
            id="drift_detection",
            replace_existing=True,
        )
        logger.info("[SUCCESS] Drift detection scheduled (every 15 minutes)")

        # Add scheduler job for continuous learning (feedback-based)
        from src.training.continuous_learning import RETRAIN_INTERVAL_HOURS

        scheduler.add_job(
            run_continuous_learning,
            "interval",
            hours=RETRAIN_INTERVAL_HOURS,  # Run every N hours (default: 1)
            id="continuous_learning",
            replace_existing=True,
        )
        logger.info(
            f"[SUCCESS] Continuous learning (feedback-based) scheduled (every {RETRAIN_INTERVAL_HOURS} hour(s))"
        )

        # Add scheduler job for periodic data collection
        import os

        DATA_COLLECTION_INTERVAL_HOURS = int(
            os.getenv("DATA_COLLECTION_INTERVAL_HOURS", "24")
        )  # Default: daily

        scheduler.add_job(
            run_periodic_data_collection,
            "interval",
            hours=DATA_COLLECTION_INTERVAL_HOURS,
            id="periodic_data_collection",
            replace_existing=True,
        )
        logger.info(
            f"[SUCCESS] Periodic data collection scheduled (every {DATA_COLLECTION_INTERVAL_HOURS} hour(s))"
        )

        scheduler.start()
        logger.info("[SUCCESS] Background scheduler started")

        logger.info("=" * 80)
        logger.info("🚀 API STARTUP COMPLETE")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down Movie Sentiment Analysis API...")

    # Shutdown scheduler
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Background scheduler stopped")


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
            "drift_status": "/monitoring/drift",
            "manual_retrain": "/training/retrain",
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
            prediction_probabilities=(
                result.get("prediction_probabilities")
                if request.include_probabilities
                else None
            ),
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
                prediction_probabilities=(
                    result.get("prediction_probabilities")
                    if request.include_probabilities
                    else None
                ),
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


@app.get("/monitoring/drift")
async def get_drift_status():
    """
    Get current drift detection status (manual trigger).

    Returns:
        Drift detection results
    """
    try:
        global drift_detector

        if drift_detector is None:
            drift_detector = DataDriftDetector()

        if not drift_detector.has_sufficient_data(min_samples=10):
            return {
                "status": "insufficient_data",
                "message": "Not enough predictions for drift detection (minimum: 10)",
                "drift_scores": {
                    "overall": 0.0,
                    "text_length": 0.0,
                    "word_count": 0.0,
                    "sentiment": 0.0,
                },
                "alert_level": "ok",
            }

        # Run drift detection
        results = drift_detector.detect_drift(window_size=100)

        # Update metrics
        update_drift_metrics(results)

        return {
            "status": "success",
            "timestamp": results["timestamp"],
            "drift_scores": results["drift_scores"],
            "alert_level": results["alert_level"],
            "production_samples": results["production_samples"],
        }

    except Exception as e:
        logger.error(f"Manual drift detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Drift detection failed: {str(e)}",
        )


@app.post("/training/retrain")
async def manual_retrain():
    """
    Manually trigger retraining (for testing/admin use).

    Returns:
        Retraining results including metrics and deployment decision
    """
    global continuous_learner, is_retraining

    try:
        # Check if already retraining
        if is_retraining:
            return {
                "status": "in_progress",
                "message": "Retraining already in progress, please wait...",
            }

        # Initialize learner if needed
        if continuous_learner is None:
            continuous_learner = ContinuousLearner()

        # Check sufficient feedback
        if not continuous_learner.has_sufficient_feedback():
            feedback_count = len(continuous_learner.collect_feedback_from_db())
            return {
                "status": "insufficient_feedback",
                "message": f"Need at least {continuous_learner.retrain_threshold} feedback samples",
                "current_feedback": feedback_count,
                "threshold": continuous_learner.retrain_threshold,
            }

        # Set retraining flag
        is_retraining = True

        logger.info("🚀 Manual retraining triggered...")

        # Trigger retraining
        results = continuous_learner.trigger_retraining()

        # Reset flag
        is_retraining = False

        # Return results
        if results.get("status") == "success":
            return {
                "status": "success",
                "message": "Retraining completed successfully",
                "new_model_metrics": results["new_model_metrics"],
                "production_metrics": results["production_metrics"],
                "improvement": results["improvement"],
                "deployed": results["should_deploy"],
                "model_version": results["model_version"],
                "feedback_samples": results["feedback_samples"],
                "training_time": results["training_time"],
                "note": (
                    "Please restart API to load new model if deployed"
                    if results["should_deploy"]
                    else None
                ),
            }
        else:
            return {
                "status": "error",
                "message": results.get("message", "Retraining failed"),
                "error": results.get("traceback"),
            }

    except Exception as e:
        is_retraining = False  # Reset flag on error
        logger.error(f"Manual retraining failed: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retraining failed: {str(e)}",
        )


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
