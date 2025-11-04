"""
Unit tests for feature_engineer module.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.feature_engineer import FeatureEngineer, extract_features


class TestFeatureEngineer:
    """Test cases for FeatureEngineer class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.engineer = FeatureEngineer({"max_features_tfidf": 50})
        self.test_texts = pd.Series(
            [
                "This movie was amazing! Great acting and stunning visuals.",
                "Terrible film. Complete waste of time.",
                "It was okay, nothing special.",
                "LOVED IT!!! Must watch!!!",
                "Boring and predictable movie.",
            ]
        )

    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        assert self.engineer.max_features == 50
        assert self.engineer.tfidf_vectorizer is None

    def test_extract_statistical_features(self):
        """Test statistical feature extraction."""
        features = self.engineer.extract_statistical_features(self.test_texts)

        # Check shape
        assert features.shape[0] == len(self.test_texts)
        assert features.shape[1] > 10  # Should have multiple features

        # Check required columns
        required_cols = [
            "char_count",
            "word_count",
            "avg_word_length",
            "exclamation_count",
            "question_count",
            "punctuation_count",
            "capital_count",
            "textblob_polarity",
            "textblob_subjectivity",
        ]
        for col in required_cols:
            assert col in features.columns

    def test_statistical_features_values(self):
        """Test statistical feature values are reasonable."""
        features = self.engineer.extract_statistical_features(self.test_texts)

        # Char count should be positive
        assert (features["char_count"] > 0).all()

        # Word count should be positive
        assert (features["word_count"] > 0).all()

        # Ratios should be between 0 and 1
        assert (features["punctuation_ratio"] >= 0).all()
        assert (features["punctuation_ratio"] <= 1).all()

        # TextBlob polarity should be between -1 and 1
        assert (features["textblob_polarity"] >= -1).all()
        assert (features["textblob_polarity"] <= 1).all()

    def test_fit_tfidf(self):
        """Test TF-IDF vectorizer fitting."""
        self.engineer.fit_tfidf(self.test_texts)

        assert self.engineer.tfidf_vectorizer is not None
        assert len(self.engineer.feature_names) > 0
        assert len(self.engineer.feature_names) <= 50  # max_features

    def test_transform_tfidf(self):
        """Test TF-IDF transformation."""
        self.engineer.fit_tfidf(self.test_texts)
        tfidf_features = self.engineer.transform_tfidf(self.test_texts)

        # Check shape
        assert tfidf_features.shape[0] == len(self.test_texts)
        assert tfidf_features.shape[1] > 0

        # Check column names have tfidf_ prefix
        assert all(col.startswith("tfidf_") for col in tfidf_features.columns)

    def test_transform_tfidf_without_fit(self):
        """Test that transform raises error without fit."""
        with pytest.raises(ValueError, match="not fitted"):
            self.engineer.transform_tfidf(self.test_texts)

    def test_extract_all_features(self):
        """Test extraction of all features."""
        all_features = self.engineer.extract_all_features(
            self.test_texts, fit_tfidf=True
        )

        # Check shape
        assert all_features.shape[0] == len(self.test_texts)

        # Should have both statistical and TF-IDF features
        stat_cols = [
            col for col in all_features.columns if not col.startswith("tfidf_")
        ]
        tfidf_cols = [col for col in all_features.columns if col.startswith("tfidf_")]

        assert len(stat_cols) > 10  # Statistical features
        assert len(tfidf_cols) > 0  # TF-IDF features

    def test_save_and_load_vectorizer(self):
        """Test saving and loading TF-IDF vectorizer."""
        # Fit vectorizer
        self.engineer.fit_tfidf(self.test_texts)
        original_vocab_size = len(self.engineer.feature_names)

        # Save to temp file
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_vectorizer.pkl"
            self.engineer.save_vectorizer(filepath)

            # Create new engineer and load
            new_engineer = FeatureEngineer()
            new_engineer.load_vectorizer(filepath)

            # Check loaded vectorizer
            assert len(new_engineer.feature_names) == original_vocab_size
            assert new_engineer.tfidf_vectorizer is not None

    def test_save_without_fit(self):
        """Test that save raises error without fit."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_vectorizer.pkl"
            with pytest.raises(ValueError, match="No vectorizer to save"):
                self.engineer.save_vectorizer(filepath)

    def test_no_nan_in_features(self):
        """Test that features don't contain NaN values."""
        features = self.engineer.extract_statistical_features(self.test_texts)
        assert not features.isnull().any().any()

    def test_no_inf_in_features(self):
        """Test that features don't contain inf values."""
        features = self.engineer.extract_statistical_features(self.test_texts)
        assert not np.isinf(features).any().any()

    def test_empty_text_handling(self):
        """Test handling of empty texts."""
        texts = pd.Series(["", "Normal text", ""])
        features = self.engineer.extract_statistical_features(texts)

        # Should not crash and should have valid values
        assert features.shape[0] == 3
        assert not features.isnull().any().any()


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_extract_features_function(self):
        """Test extract_features convenience function."""
        texts = pd.Series(["Text 1", "Text 2", "Text 3"])
        config = {"max_features_tfidf": 20}

        features, engineer = extract_features(texts, config, fit_tfidf=True)

        assert features.shape[0] == 3
        assert engineer.tfidf_vectorizer is not None


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
