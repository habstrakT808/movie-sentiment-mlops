"""
Unit tests for API endpoints.
"""

import pytest
from fastapi.testclient import TestClient

from src.deployment.api import app

client = TestClient(app)


class TestRootEndpoint:
    """Test root endpoint."""

    def test_root_endpoint(self):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert data["message"] == "Movie Sentiment Analysis API"
        assert data["version"] == "1.0.0"
        assert data["model"] == "DistilBERT"
        assert "endpoints" in data


class TestHealthEndpoint:
    """Test health check endpoint."""

    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert "healthy" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "timestamp" in data


class TestModelInfoEndpoint:
    """Test model info endpoint."""

    def test_model_info(self):
        """Test model info endpoint."""
        response = client.get("/model/info")
        assert response.status_code == 200

        data = response.json()
        assert data["model_name"] == "distilbert"
        assert "parameters" in data
        assert "device" in data
        assert data["is_loaded"] is True


class TestPredictionEndpoint:
    """Test prediction endpoints."""

    def test_single_prediction_positive(self):
        """Test single prediction with positive text."""
        response = client.post(
            "/predict",
            json={"text": "This movie was absolutely amazing and fantastic!"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["sentiment"] == "positive"
        assert 0 <= data["confidence"] <= 1
        assert data["model"] == "distilbert"
        assert "inference_time" in data

    def test_single_prediction_negative(self):
        """Test single prediction with negative text."""
        response = client.post(
            "/predict",
            json={"text": "This movie was terrible, awful, and completely boring!"},
        )
        assert response.status_code == 200

        data = response.json()
        assert data["sentiment"] == "negative"
        assert 0 <= data["confidence"] <= 1
        assert data["model"] == "distilbert"

    def test_single_prediction_with_probabilities(self):
        """Test single prediction with probabilities."""
        response = client.post(
            "/predict", json={"text": "Great movie!", "include_probabilities": True}
        )
        assert response.status_code == 200

        data = response.json()
        assert "prediction_probabilities" in data
        assert "negative" in data["prediction_probabilities"]
        assert "positive" in data["prediction_probabilities"]

    def test_batch_prediction(self):
        """Test batch prediction."""
        response = client.post(
            "/predict/batch",
            json={
                "texts": [
                    "Amazing movie with great acting!",
                    "Terrible film, waste of time.",
                    "It was okay, nothing special.",
                ]
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["total_texts"] == 3
        assert len(data["predictions"]) == 3
        assert "total_time" in data
        assert "average_time_per_text" in data

        # Check individual predictions
        for pred in data["predictions"]:
            assert pred["sentiment"] in ["positive", "negative"]
            assert 0 <= pred["confidence"] <= 1
            assert pred["model"] == "distilbert"

    def test_batch_prediction_with_probabilities(self):
        """Test batch prediction with probabilities."""
        response = client.post(
            "/predict/batch",
            json={"texts": ["Good film", "Bad movie"], "include_probabilities": True},
        )
        assert response.status_code == 200

        data = response.json()
        for pred in data["predictions"]:
            assert "prediction_probabilities" in pred


class TestValidation:
    """Test input validation."""

    def test_empty_text_validation(self):
        """Test empty text validation."""
        response = client.post("/predict", json={"text": ""})
        assert response.status_code == 422  # Validation error

    def test_whitespace_only_text_validation(self):
        """Test whitespace-only text validation."""
        response = client.post("/predict", json={"text": "   "})
        assert response.status_code == 422

    def test_very_long_text(self):
        """Test very long text (should work but be truncated)."""
        long_text = "This movie is great! " * 1000  # Very long text
        response = client.post("/predict", json={"text": long_text})
        assert response.status_code == 200  # Should still work

    def test_empty_batch_validation(self):
        """Test empty batch validation."""
        response = client.post("/predict/batch", json={"texts": []})
        assert response.status_code == 422

    def test_large_batch_validation(self):
        """Test large batch validation."""
        large_batch = ["Test text"] * 1001  # Over limit
        response = client.post("/predict/batch", json={"texts": large_batch})
        assert response.status_code == 422

    def test_invalid_batch_size(self):
        """Test invalid batch size."""
        response = client.post(
            "/predict/batch", json={"texts": ["Test"], "batch_size": 0}  # Invalid
        )
        assert response.status_code == 422


class TestMetricsEndpoint:
    """Test metrics endpoint."""

    def test_metrics_endpoint(self):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200

        content = response.text
        assert "http_requests_total" in content
        assert "predictions_total" in content
        assert "prediction_duration_seconds" in content


class TestErrorHandling:
    """Test error handling."""

    def test_invalid_json(self):
        """Test invalid JSON handling."""
        response = client.post(
            "/predict",
            data="invalid json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 422

    def test_missing_required_field(self):
        """Test missing required field."""
        response = client.post("/predict", json={})  # Missing 'text' field
        assert response.status_code == 422

    def test_wrong_content_type(self):
        """Test wrong content type."""
        response = client.post(
            "/predict", data='{"text": "test"}', headers={"Content-Type": "text/plain"}
        )
        assert response.status_code == 422


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
