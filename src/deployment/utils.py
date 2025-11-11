"""
Utility functions for the deployment module.
"""

from datetime import datetime
from typing import Any, Dict

from fastapi import Response

from src.deployment.middleware import get_metrics, get_metrics_content_type
from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_metrics_response() -> Response:
    """Create Prometheus metrics response."""
    try:
        metrics_data = get_metrics()
        return Response(content=metrics_data, media_type=get_metrics_content_type())
    except Exception as e:
        logger.error(f"Failed to generate metrics: {str(e)}")
        return Response(
            content="# Failed to generate metrics",
            media_type="text/plain",
            status_code=500,
        )


def create_error_response(error_message: str, status_code: int = 500) -> Dict[str, Any]:
    """Create standardized error response."""
    return {
        "error": error_message,
        "timestamp": datetime.now().isoformat(),
        "status_code": status_code,
    }
