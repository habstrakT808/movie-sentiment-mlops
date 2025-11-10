"""Unit tests for model modules."""

from pathlib import Path

import pandas as pd
import pytest

from src.models.utils import (
    load_label_encoder,
    load_traditional_ml_data,
    prepare_features_and_labels,
)


class TestModelUtils:
    """Test model utility functions."""

    def test_load_traditional_ml_data(self):
        """Test loading traditional ML data."""
        train_df, val_df, test_df = load_traditional_ml_data()

        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

        assert "sentiment" in train_df.columns
        assert train_df.shape[1] > 100  # Should have many features

    def test_prepare_features_and_labels(self):
        """Test feature and label preparation."""
        df = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "sentiment": ["negative", "positive", "negative"],
            }
        )

        X, y = prepare_features_and_labels(df)

        assert len(X) == 3
        assert len(y) == 3
        assert "sentiment" not in X.columns
        assert set(y.unique()) == {0, 1}  # Binary labels

    def test_load_label_encoder(self):
        """Test loading label encoder."""
        encoder = load_label_encoder()

        assert hasattr(encoder, "classes_")
        assert len(encoder.classes_) == 2
        assert "negative" in encoder.classes_
        assert "positive" in encoder.classes_


class TestModelExistence:
    """Test that all models exist."""

    def test_logistic_regression_exists(self):
        """Test logistic regression model exists."""
        model_path = Path("models/logistic_regression/model.pkl")
        assert model_path.exists()

    def test_random_forest_exists(self):
        """Test random forest model exists."""
        model_path = Path("models/random_forest/model.pkl")
        assert model_path.exists()

    def test_svm_exists(self):
        """Test SVM model exists."""
        model_path = Path("models/svm/model.pkl")
        assert model_path.exists()

    def test_distilbert_exists(self):
        """Test DistilBERT model exists."""
        model_dir = Path("models/distilbert")
        assert model_dir.exists()
        assert (model_dir / "config.json").exists()


class TestModelMetrics:
    """Test model metrics."""

    def test_all_models_have_metrics(self):
        """Test all models have metadata with metrics."""
        models = ["logistic_regression", "random_forest", "svm", "distilbert"]

        for model_name in models:
            metadata_path = Path(f"models/{model_name}/metadata.json")
            assert metadata_path.exists(), f"{model_name} metadata missing"

            import json

            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            assert "metrics" in metadata
            assert "test_f1" in metadata["metrics"]
            assert "test_accuracy" in metadata["metrics"]

    def test_best_model_exceeds_threshold(self):
        """Test best model exceeds performance threshold."""
        import json

        metadata_path = Path("models/distilbert/metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        # DistilBERT should exceed 90% F1
        assert metadata["metrics"]["test_f1"] > 0.90


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
