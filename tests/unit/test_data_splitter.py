"""
Unit tests for data_splitter module.
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessing.data_splitter import DataSplitter, split_data


class TestDataSplitter:
    """Test cases for DataSplitter class."""

    def setup_method(self):
        """Setup test fixtures."""
        # Create test dataset
        np.random.seed(42)
        n_samples = 1000

        self.test_df = pd.DataFrame(
            {
                "text": [f"Review {i}" for i in range(n_samples)],
                "sentiment": (
                    ["positive"] * 450 + ["negative"] * 450 + ["neutral"] * 100
                ),
                "score": np.random.randint(1, 6, n_samples),
            }
        )

        self.config = {
            "train_size": 0.7,
            "val_size": 0.15,
            "test_size": 0.15,
            "random_state": 42,
            "stratify": True,
            "classification_type": "binary",
        }

    def test_initialization(self):
        """Test DataSplitter initialization."""
        splitter = DataSplitter(self.config)

        assert splitter.train_size == 0.7
        assert splitter.val_size == 0.15
        assert splitter.test_size == 0.15
        assert splitter.random_state == 42
        assert splitter.stratify is True

    def test_invalid_split_ratios(self):
        """Test that invalid split ratios raise error."""
        invalid_config = {
            "train_size": 0.5,
            "val_size": 0.3,
            "test_size": 0.3,  # Sum = 1.1, invalid
        }

        with pytest.raises(ValueError, match="must sum to 1.0"):
            DataSplitter(invalid_config)

    def test_binary_classification(self):
        """Test binary classification (neutral removal)."""
        splitter = DataSplitter(self.config)
        df_binary = splitter.prepare_binary_classification(self.test_df)

        # Should only have positive and negative
        assert set(df_binary["sentiment"].unique()) == {"positive", "negative"}

        # Should have dropped neutral samples
        assert len(df_binary) == 900  # 450 + 450

    def test_split_data_binary(self):
        """Test data splitting for binary classification."""
        splitter = DataSplitter(self.config)
        train_df, val_df, test_df = splitter.split_data(self.test_df)

        # Check that splits exist
        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0

        # Check that neutral class is removed
        assert "neutral" not in train_df["sentiment"].values
        assert "neutral" not in val_df["sentiment"].values
        assert "neutral" not in test_df["sentiment"].values

        # Check total samples (900 after removing 100 neutral)
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == 900

    def test_split_data_multiclass(self):
        """Test data splitting for multi-class classification."""
        config_multi = self.config.copy()
        config_multi["classification_type"] = "multiclass"

        splitter = DataSplitter(config_multi)
        train_df, val_df, test_df = splitter.split_data(self.test_df)

        # Check that all classes are present
        all_sentiments = set(
            list(train_df["sentiment"].unique())
            + list(val_df["sentiment"].unique())
            + list(test_df["sentiment"].unique())
        )
        assert "neutral" in all_sentiments

        # Check total samples (all 1000)
        total = len(train_df) + len(val_df) + len(test_df)
        assert total == 1000

    def test_split_ratios(self):
        """Test that split ratios are approximately correct."""
        splitter = DataSplitter(self.config)
        train_df, val_df, test_df = splitter.split_data(self.test_df)

        total = len(train_df) + len(val_df) + len(test_df)

        train_ratio = len(train_df) / total
        val_ratio = len(val_df) / total
        test_ratio = len(test_df) / total

        # Check ratios are within 5% tolerance
        assert abs(train_ratio - 0.7) < 0.05
        assert abs(val_ratio - 0.15) < 0.05
        assert abs(test_ratio - 0.15) < 0.05

    def test_stratification(self):
        """Test that stratification maintains class distribution."""
        splitter = DataSplitter(self.config)
        train_df, val_df, test_df = splitter.split_data(self.test_df)

        # Calculate class distributions
        train_dist = train_df["sentiment"].value_counts(normalize=True)
        val_dist = val_df["sentiment"].value_counts(normalize=True)
        test_dist = test_df["sentiment"].value_counts(normalize=True)

        # All splits should have similar distributions (within 10%)
        for sentiment in ["positive", "negative"]:
            assert abs(train_dist[sentiment] - 0.5) < 0.1
            assert abs(val_dist[sentiment] - 0.5) < 0.1
            assert abs(test_dist[sentiment] - 0.5) < 0.1

    def test_no_data_leakage(self):
        """Test that there's no overlap between splits."""
        splitter = DataSplitter(self.config)
        train_df, val_df, test_df = splitter.split_data(self.test_df)

        # Get indices
        train_indices = set(train_df.index)
        val_indices = set(val_df.index)
        test_indices = set(test_df.index)

        # Check no overlap
        assert len(train_indices & val_indices) == 0
        assert len(train_indices & test_indices) == 0
        assert len(val_indices & test_indices) == 0

    def test_reproducibility(self):
        """Test that splits are reproducible with same random_state."""
        splitter1 = DataSplitter(self.config)
        train1, val1, test1 = splitter1.split_data(self.test_df.copy())

        splitter2 = DataSplitter(self.config)
        train2, val2, test2 = splitter2.split_data(self.test_df.copy())

        # Should have same indices
        assert train1.index.tolist() == train2.index.tolist()
        assert val1.index.tolist() == val2.index.tolist()
        assert test1.index.tolist() == test2.index.tolist()

    def test_get_split_statistics(self):
        """Test split statistics calculation."""
        splitter = DataSplitter(self.config)
        train_df, val_df, test_df = splitter.split_data(self.test_df)

        stats = splitter.get_split_statistics(train_df, val_df, test_df)

        # Check required keys
        required_keys = [
            "total_samples",
            "train_samples",
            "val_samples",
            "test_samples",
            "train_ratio",
            "val_ratio",
            "test_ratio",
            "train_distribution",
            "val_distribution",
            "test_distribution",
        ]

        for key in required_keys:
            assert key in stats

        # Check values make sense
        assert stats["total_samples"] == 900  # Binary classification
        assert (
            stats["train_samples"] + stats["val_samples"] + stats["test_samples"] == 900
        )


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_split_data_function(self):
        """Test split_data convenience function."""
        np.random.seed(42)
        test_df = pd.DataFrame(
            {
                "text": [f"Review {i}" for i in range(100)],
                "sentiment": ["positive"] * 50 + ["negative"] * 50,
            }
        )

        config = {
            "train_size": 0.7,
            "val_size": 0.15,
            "test_size": 0.15,
            "random_state": 42,
            "classification_type": "binary",
        }

        train_df, val_df, test_df, stats = split_data(test_df, config)

        assert len(train_df) > 0
        assert len(val_df) > 0
        assert len(test_df) > 0
        assert isinstance(stats, dict)
        assert "total_samples" in stats


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
