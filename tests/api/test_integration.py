"""
Integration tests for the complete API workflow.
"""

import time

import pytest
from fastapi.testclient import TestClient

from src.deployment.api import app

client = TestClient(app)


class TestAPIIntegration:
    """Integration tests for API workflow."""

    def test_complete_workflow(self):
        """Test complete API workflow."""
        # 1. Check health
        health_response = client.get("/health")
        assert health_response.status_code == 200
        assert health_response.json()["healthy"] is True

        # 2. Get model info
        info_response = client.get("/model/info")
        assert info_response.status_code == 200
        assert info_response.json()["is_loaded"] is True

        # 3. Make predictions
        pred_response = client.post(
            "/predict", json={"text": "This movie is fantastic!"}
        )
        assert pred_response.status_code == 200
        assert pred_response.json()["sentiment"] == "positive"

        # 4. Check metrics
        metrics_response = client.get("/metrics")
        assert metrics_response.status_code == 200
        assert "predictions_total" in metrics_response.text

    def test_performance_benchmark(self):
        """Test API performance."""
        test_texts = [
            "Amazing movie with incredible acting!",
            "Terrible film, complete waste of time.",
            "It was okay, nothing too special.",
            "Brilliant cinematography and storytelling!",
            "Boring and predictable plot throughout.",
        ]

        # Single predictions
        single_times = []
        for text in test_texts:
            start_time = time.time()
            response = client.post("/predict", json={"text": text})
            end_time = time.time()

            assert response.status_code == 200
            single_times.append(end_time - start_time)

        avg_single_time = sum(single_times) / len(single_times)
        print(f"Average single prediction time: {avg_single_time:.3f}s")
        assert avg_single_time < 1.0  # Should be under 1 second

        # Batch prediction
        start_time = time.time()
        batch_response = client.post("/predict/batch", json={"texts": test_texts})
        end_time = time.time()

        assert batch_response.status_code == 200
        batch_time = end_time - start_time
        batch_avg = batch_time / len(test_texts)

        print(f"Batch prediction time: {batch_time:.3f}s")
        print(f"Batch average per text: {batch_avg:.3f}s")

        # Batch should be faster per text than individual requests
        assert batch_avg < avg_single_time

    def test_concurrent_requests(self):
        """Test handling concurrent requests."""
        import concurrent.futures

        def make_request(text):
            return client.post("/predict", json={"text": text})

        test_texts = [
            "Great movie!",
            "Terrible film!",
            "It was okay.",
            "Amazing story!",
            "Boring movie.",
        ]

        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request, text) for text in test_texts]
            responses = [future.result() for future in futures]

        # All requests should succeed
        for response in responses:
            assert response.status_code == 200
            data = response.json()
            assert data["sentiment"] in ["positive", "negative"]
            assert 0 <= data["confidence"] <= 1

    def test_model_consistency(self):
        """Test model prediction consistency."""
        # Same input should give same output
        test_text = "This movie was absolutely fantastic!"

        responses = []
        for _ in range(5):
            response = client.post("/predict", json={"text": test_text})
            assert response.status_code == 200
            responses.append(response.json())

        # All responses should be identical
        first_response = responses[0]
        for response in responses[1:]:
            assert response["sentiment"] == first_response["sentiment"]
            # Confidence might have tiny floating point differences
            assert abs(response["confidence"] - first_response["confidence"]) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
