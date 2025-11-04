"""
Data splitting utilities for train/validation/test sets.
Handles stratified splitting and class balancing.
"""

from typing import Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DataSplitter:
    """
    Data splitter with stratification support.

    Features:
    - Stratified splitting (maintains class distribution)
    - Binary classification (drops neutral class)
    - Configurable split ratios
    - Reproducible with random_state
    """

    def __init__(self, config: Dict = None):
        """
        Initialize data splitter.

        Args:
            config: Configuration dictionary with split parameters
        """
        self.config = config or {}

        # Split ratios
        self.train_size = self.config.get("train_size", 0.7)
        self.val_size = self.config.get("val_size", 0.15)
        self.test_size = self.config.get("test_size", 0.15)

        # Validation
        total = self.train_size + self.val_size + self.test_size
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Split ratios must sum to 1.0, got {total} "
                f"(train={self.train_size}, val={self.val_size}, test={self.test_size})"
            )

        # Other parameters
        self.random_state = self.config.get("random_state", 42)
        self.stratify = self.config.get("stratify", True)
        self.classification_type = self.config.get("classification_type", "binary")

        logger.info(
            f"DataSplitter initialized: train={self.train_size}, "
            f"val={self.val_size}, test={self.test_size}, "
            f"stratify={self.stratify}, classification={self.classification_type}"
        )

    def prepare_binary_classification(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare data for binary classification (positive/negative only).

        Args:
            df: DataFrame with 'sentiment' column

        Returns:
            DataFrame with neutral class removed
        """
        logger.info("Preparing binary classification (dropping neutral class)...")

        original_count = len(df)

        # Filter to only positive and negative
        df_binary = df[df["sentiment"].isin(["positive", "negative"])].copy()

        dropped_count = original_count - len(df_binary)

        logger.info(
            f"Dropped {dropped_count} neutral samples. "
            f"Remaining: {len(df_binary)} ({len(df_binary)/original_count*100:.1f}%)"
        )

        # Check class distribution
        class_dist = df_binary["sentiment"].value_counts()
        logger.info(f"Class distribution after filtering:\n{class_dist}")

        return df_binary

    def split_data(
        self, df: pd.DataFrame, target_column: str = "sentiment"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Input DataFrame
            target_column: Name of target column for stratification

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting data: {len(df)} samples...")

        # Prepare for binary classification if specified
        if self.classification_type == "binary":
            df = self.prepare_binary_classification(df)

        # First split: separate test set
        # Calculate test_size relative to full dataset
        test_size_ratio = self.test_size

        stratify_col = df[target_column] if self.stratify else None

        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size_ratio,
            random_state=self.random_state,
            stratify=stratify_col,
            shuffle=True,
        )

        logger.info(f"Test set: {len(test_df)} samples ({test_size_ratio*100:.1f}%)")

        # Second split: separate validation from training
        # Calculate val_size relative to remaining data
        val_size_ratio = self.val_size / (self.train_size + self.val_size)

        stratify_col = train_val_df[target_column] if self.stratify else None

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_ratio,
            random_state=self.random_state,
            stratify=stratify_col,
            shuffle=True,
        )

        logger.info(
            f"Train set: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)"
        )
        logger.info(
            f"Validation set: {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)"
        )

        # Verify split ratios
        self._verify_splits(df, train_df, val_df, test_df)

        # Verify stratification
        if self.stratify:
            self._verify_stratification(df, train_df, val_df, test_df, target_column)

        return train_df, val_df, test_df

    def _verify_splits(
        self,
        original_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        """Verify split ratios are correct."""
        total = len(original_df)
        train_ratio = len(train_df) / total
        val_ratio = len(val_df) / total
        test_ratio = len(test_df) / total

        logger.info(
            f"Actual split ratios: train={train_ratio:.3f}, "
            f"val={val_ratio:.3f}, test={test_ratio:.3f}"
        )

        # Check if ratios are within tolerance (±2%)
        tolerance = 0.02

        if abs(train_ratio - self.train_size) > tolerance:
            logger.warning(
                f"Train ratio {train_ratio:.3f} differs from target {self.train_size}"
            )

        if abs(val_ratio - self.val_size) > tolerance:
            logger.warning(
                f"Val ratio {val_ratio:.3f} differs from target {self.val_size}"
            )

        if abs(test_ratio - self.test_size) > tolerance:
            logger.warning(
                f"Test ratio {test_ratio:.3f} differs from target {self.test_size}"
            )

    def _verify_stratification(
        self,
        original_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str,
    ):
        """Verify class distributions are maintained."""
        logger.info("Verifying stratification...")

        # Calculate class distributions
        original_dist = original_df[target_column].value_counts(normalize=True)
        train_dist = train_df[target_column].value_counts(normalize=True)
        val_dist = val_df[target_column].value_counts(normalize=True)
        test_dist = test_df[target_column].value_counts(normalize=True)

        # Log distributions
        logger.info(f"Original distribution:\n{original_dist}")
        logger.info(f"Train distribution:\n{train_dist}")
        logger.info(f"Val distribution:\n{val_dist}")
        logger.info(f"Test distribution:\n{test_dist}")

        # Check if distributions are similar (within 5%)
        tolerance = 0.05

        for class_label in original_dist.index:
            orig_pct = original_dist[class_label]
            train_pct = train_dist.get(class_label, 0)
            val_pct = val_dist.get(class_label, 0)
            test_pct = test_dist.get(class_label, 0)

            if abs(train_pct - orig_pct) > tolerance:
                logger.warning(
                    f"Train set {class_label} distribution ({train_pct:.3f}) "
                    f"differs from original ({orig_pct:.3f})"
                )

            if abs(val_pct - orig_pct) > tolerance:
                logger.warning(
                    f"Val set {class_label} distribution ({val_pct:.3f}) "
                    f"differs from original ({orig_pct:.3f})"
                )

            if abs(test_pct - orig_pct) > tolerance:
                logger.warning(
                    f"Test set {class_label} distribution ({test_pct:.3f}) "
                    f"differs from original ({orig_pct:.3f})"
                )

    def get_split_statistics(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_column: str = "sentiment",
    ) -> Dict:
        """
        Get statistics about the splits.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            target_column: Name of target column

        Returns:
            Dictionary with split statistics
        """
        total = len(train_df) + len(val_df) + len(test_df)

        stats = {
            "total_samples": total,
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "train_ratio": len(train_df) / total,
            "val_ratio": len(val_df) / total,
            "test_ratio": len(test_df) / total,
            "train_distribution": train_df[target_column].value_counts().to_dict(),
            "val_distribution": val_df[target_column].value_counts().to_dict(),
            "test_distribution": test_df[target_column].value_counts().to_dict(),
        }

        return stats


# Convenience function
def split_data(
    df: pd.DataFrame, config: Dict = None, target_column: str = "sentiment"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    Convenience function to split data.

    Args:
        df: Input DataFrame
        config: Configuration dictionary
        target_column: Name of target column

    Returns:
        Tuple of (train_df, val_df, test_df, statistics)
    """
    splitter = DataSplitter(config)
    train_df, val_df, test_df = splitter.split_data(df, target_column)
    stats = splitter.get_split_statistics(train_df, val_df, test_df, target_column)

    return train_df, val_df, test_df, stats


# Example usage and testing
if __name__ == "__main__":
    # Create test data
    import numpy as np

    np.random.seed(42)

    # Simulate imbalanced dataset with neutral class
    n_samples = 1000
    sentiments = ["positive"] * 450 + ["negative"] * 450 + ["neutral"] * 100

    test_df = pd.DataFrame(
        {
            "text": [f"Review {i}" for i in range(n_samples)],
            "sentiment": sentiments,
            "score": np.random.randint(1, 6, n_samples),
        }
    )

    print("=" * 80)
    print("DATA SPLITTER TEST")
    print("=" * 80)

    # Test 1: Binary classification with stratification
    print("\n1. BINARY CLASSIFICATION (DROP NEUTRAL)")
    print("-" * 80)

    config = {
        "train_size": 0.7,
        "val_size": 0.15,
        "test_size": 0.15,
        "random_state": 42,
        "stratify": True,
        "classification_type": "binary",
    }

    splitter = DataSplitter(config)
    train_df, val_df, test_df = splitter.split_data(test_df)

    print(f"\nOriginal dataset: {len(test_df)} samples")
    print(f"Train set: {len(train_df)} samples")
    print(f"Val set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")

    # Test 2: Get statistics
    print("\n2. SPLIT STATISTICS")
    print("-" * 80)

    stats = splitter.get_split_statistics(train_df, val_df, test_df)

    print(f"Total samples: {stats['total_samples']}")
    print(f"Train: {stats['train_samples']} ({stats['train_ratio']:.1%})")
    print(f"Val: {stats['val_samples']} ({stats['val_ratio']:.1%})")
    print(f"Test: {stats['test_samples']} ({stats['test_ratio']:.1%})")

    print("\nClass distributions:")
    print(f"Train: {stats['train_distribution']}")
    print(f"Val: {stats['val_distribution']}")
    print(f"Test: {stats['test_distribution']}")

    # Test 3: Multi-class (keep neutral)
    print("\n3. MULTI-CLASS CLASSIFICATION (KEEP NEUTRAL)")
    print("-" * 80)

    config_multiclass = config.copy()
    config_multiclass["classification_type"] = "multiclass"

    splitter_multi = DataSplitter(config_multiclass)
    train_multi, val_multi, test_multi = splitter_multi.split_data(test_df)

    print(f"\nOriginal dataset: {len(test_df)} samples")
    print(f"Train set: {len(train_multi)} samples")
    print(f"Val set: {len(val_multi)} samples")
    print(f"Test set: {len(test_multi)} samples")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
