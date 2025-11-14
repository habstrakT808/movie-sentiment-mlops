"""
Custom middleware for FastAPI application.
Includes logging and Prometheus metrics collection.
"""

import time
from typing import Callable

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "endpoint", "status"]
)

REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "endpoint"],
)

PREDICTION_COUNT = Counter(
    "predictions_total", "Total predictions made", ["type"]  # single or batch
)

PREDICTION_DURATION = Histogram(
    "prediction_duration_seconds", "Prediction duration in seconds", ["type"]
)

MODEL_ERRORS = Counter("model_errors_total", "Total model errors", ["error_type"])

# Data Drift metrics
DATA_DRIFT_SCORE = Gauge(
    "data_drift_score",
    "Data drift detection score (0-1)",
    ["metric_type"],  # overall, text_length, word_count, sentiment
)

DATA_DRIFT_ALERT = Gauge(
    "data_drift_alert",
    "Data drift alert level (0=ok, 1=warning, 2=critical)",
)


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and log details."""
        start_time = time.time()

        # Log request
        logger.info(f"Request: {request.method} {request.url.path}")

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Log response
        logger.info(
            f"Response: {response.status_code} | "
            f"Duration: {duration:.3f}s | "
            f"Path: {request.url.path}"
        )

        return response


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware for Prometheus metrics collection."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and collect metrics."""
        start_time = time.time()

        # Get endpoint path (remove query parameters)
        endpoint = request.url.path
        method = request.method

        # Process request
        response = await call_next(request)

        # Calculate duration
        duration = time.time() - start_time

        # Record metrics
        REQUEST_COUNT.labels(
            method=method, endpoint=endpoint, status=response.status_code
        ).inc()

        REQUEST_DURATION.labels(method=method, endpoint=endpoint).observe(duration)

        # Record prediction metrics
        if endpoint == "/predict":
            PREDICTION_COUNT.labels(type="single").inc()
            PREDICTION_DURATION.labels(type="single").observe(duration)
        elif endpoint == "/predict/batch":
            PREDICTION_COUNT.labels(type="batch").inc()
            PREDICTION_DURATION.labels(type="batch").observe(duration)

        # Record error metrics
        if response.status_code >= 400:
            if response.status_code >= 500:
                MODEL_ERRORS.labels(error_type="server_error").inc()
            else:
                MODEL_ERRORS.labels(error_type="client_error").inc()

        return response


def get_metrics():
    """Get Prometheus metrics."""
    return generate_latest()


def get_metrics_content_type():
    """Get Prometheus metrics content type."""
    return CONTENT_TYPE_LATEST


def update_drift_metrics(drift_results: dict):
    """
    Update Prometheus drift metrics.

    Args:
        drift_results: Dict from DataDriftDetector.detect_drift()
    """
    try:
        # Update drift scores
        drift_scores = drift_results.get("drift_scores", {})

        DATA_DRIFT_SCORE.labels(metric_type="overall").set(
            drift_scores.get("overall", 0.0)
        )
        DATA_DRIFT_SCORE.labels(metric_type="text_length").set(
            drift_scores.get("text_length", 0.0)
        )
        DATA_DRIFT_SCORE.labels(metric_type="word_count").set(
            drift_scores.get("word_count", 0.0)
        )
        DATA_DRIFT_SCORE.labels(metric_type="sentiment").set(
            drift_scores.get("sentiment", 0.0)
        )

        # Update alert level
        alert_level = drift_results.get("alert_level", "ok")
        alert_value = {"ok": 0, "warning": 1, "critical": 2}.get(alert_level, 0)
        DATA_DRIFT_ALERT.set(alert_value)

        logger.info(
            f"Drift metrics updated: overall={drift_scores.get('overall', 0.0):.4f}, alert={alert_level}"
        )

    except Exception as e:
        logger.error(f"Failed to update drift metrics: {e}")
