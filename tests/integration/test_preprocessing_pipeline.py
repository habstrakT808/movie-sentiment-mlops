"""
Integration test for preprocessing pipeline.
"""

import json

import pandas as pd
import pytest

from src.utils.config import Config


class TestPreprocessingPipeline:
    """Integration tests for preprocessing pipeline."""

    def test_processed_files_exist(self):
        """Test that all processed files are created."""
        processed_dir = Config.PROCESSED_DATA_DIR

        required_files = [
            "train.csv",
            "validation.csv",
            "test.csv",
            "train_features.csv",
            "validation_features.csv",
            "test_features.csv",
            "tfidf_vectorizer.pkl",
            "label_encoder.pkl",
            "preprocessing_stats.json",
        ]

        for filename in required_files:
            filepath = processed_dir / filename
            assert filepath.exists(), f"Missing file: {filename}"

    def test_data_splits_valid(self):
        """Test that data splits are valid."""
        processed_dir = Config.PROCESSED_DATA_DIR

        train_df = pd.read_csv(processed_dir / "train.csv")
        val_df = pd.read_csv(processed_dir / "validation.csv")
        test_df = pd.read_csv(processed_dir / "test.csv")

        # Check non-empty
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

        # Check columns
        required_cols = ["text_cleaned", "sentiment"]
        for col in required_cols:
            assert col in train_df.columns
            assert col in val_df.columns
            assert col in test_df.columns

        # Check no neutral class (binary classification)
        assert "neutral" not in train_df["sentiment"].values
        assert "neutral" not in val_df["sentiment"].values
        assert "neutral" not in test_df["sentiment"].values

    def test_feature_files_valid(self):
        """Test that feature files are valid."""
        processed_dir = Config.PROCESSED_DATA_DIR

        train_feat = pd.read_csv(processed_dir / "train_features.csv")
        val_feat = pd.read_csv(processed_dir / "validation_features.csv")
        test_feat = pd.read_csv(processed_dir / "test_features.csv")

        # Check shapes match
        assert train_feat.shape[1] == val_feat.shape[1] == test_feat.shape[1]

        # Check has many features (statistical + TF-IDF)
        assert train_feat.shape[1] > 100  # Should have 5000+ features

        # Check has sentiment column
        assert "sentiment" in train_feat.columns

    def test_preprocessing_stats_valid(self):
        """Test that preprocessing stats are valid."""
        stats_path = Config.PROCESSED_DATA_DIR / "preprocessing_stats.json"

        with open(stats_path, "r") as f:
            stats = json.load(f)

        # Check required keys
        required_keys = [
            "pipeline_steps",
            "data_shapes",
            "class_distributions",
            "feature_counts",
        ]

        for key in required_keys:
            assert key in stats, f"Missing key: {key}"

        # Check pipeline completed all steps
        assert len(stats["pipeline_steps"]) >= 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
