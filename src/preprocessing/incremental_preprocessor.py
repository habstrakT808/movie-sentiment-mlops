"""
Incremental Preprocessor for Continuous Learning.
Preprocesses new collected data and merges with existing training data.
"""

import pickle
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml

from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.text_cleaner import TextCleaner
from src.utils.config import Config
from src.utils.helpers import timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class IncrementalPreprocessor:
    """
    Preprocessor for incremental data collection.

    Features:
    - Preprocesses new collected data
    - Loads existing TF-IDF vectorizer and label encoder
    - Merges new data with existing training data
    - Re-splits data with same proportions (70/15/15)
    - Saves merged and processed data
    """

    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize incremental preprocessor.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.preprocessing_config = self.config.get("preprocessing", {})

        # Initialize components
        self.text_cleaner = TextCleaner(self.preprocessing_config)
        self.feature_engineer = FeatureEngineer(self.preprocessing_config)
        self.data_splitter = DataSplitter(self.preprocessing_config)

        # Paths
        self.processed_dir = Config.PROCESSED_DATA_DIR
        self.existing_train_path = self.processed_dir / "train.csv"
        self.existing_val_path = self.processed_dir / "validation.csv"
        self.existing_test_path = self.processed_dir / "test.csv"

        # Artifacts paths
        self.tfidf_path = self.processed_dir / "tfidf_vectorizer.pkl"
        self.label_encoder_path = self.processed_dir / "label_encoder.pkl"

        # Load existing artifacts
        self.tfidf_vectorizer = None
        self.label_encoder = None
        self._load_existing_artifacts()

        logger.info("IncrementalPreprocessor initialized")

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}

    def _load_existing_artifacts(self):
        """Load existing TF-IDF vectorizer and label encoder."""
        try:
            # Load TF-IDF vectorizer
            if self.tfidf_path.exists():
                with open(self.tfidf_path, "rb") as f:
                    self.tfidf_vectorizer = pickle.load(f)
                logger.info(
                    f"✅ Loaded existing TF-IDF vectorizer from {self.tfidf_path}"
                )
            else:
                logger.warning(f"⚠️ TF-IDF vectorizer not found at {self.tfidf_path}")

            # Load label encoder (if exists)
            if self.label_encoder_path.exists():
                with open(self.label_encoder_path, "rb") as f:
                    self.label_encoder = pickle.load(f)
                logger.info(
                    f"✅ Loaded existing label encoder from {self.label_encoder_path}"
                )
            else:
                logger.warning(
                    f"⚠️ Label encoder not found at {self.label_encoder_path}"
                )

        except Exception as e:
            logger.error(f"❌ Failed to load existing artifacts: {e}")
            import traceback

            logger.error(traceback.format_exc())

    @timer
    def preprocess_incremental_data(
        self,
        incremental_data_path: Path,
        merge_with_existing: bool = True,
    ) -> Dict:
        """
        Preprocess new data and optionally merge with existing training data.

        Args:
            incremental_data_path: Path to new collected data CSV
            merge_with_existing: Whether to merge with existing training data

        Returns:
            Dict with preprocessing results
        """
        logger.info("=" * 80)
        logger.info("🔄 STARTING INCREMENTAL PREPROCESSING")
        logger.info("=" * 80)

        try:
            # Step 1: Load new data
            logger.info(f"\n📥 Loading new data from: {incremental_data_path}")
            new_df = pd.read_csv(incremental_data_path)
            logger.info(f"Loaded {len(new_df)} new samples")

            # Check required columns
            required_columns = ["text", "sentiment"]
            missing_columns = [
                col for col in required_columns if col not in new_df.columns
            ]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            # Step 2: Clean texts
            logger.info("\n🧹 Cleaning texts...")
            new_df["text_cleaned"] = self.text_cleaner.clean_batch(
                new_df["text"].tolist(), preserve_case=True
            )

            # Remove empty texts
            empty_count = (new_df["text_cleaned"].str.len() == 0).sum()
            if empty_count > 0:
                logger.warning(f"Removing {empty_count} empty texts after cleaning")
                new_df = new_df[new_df["text_cleaned"].str.len() > 0].copy()

            logger.info(f"✅ Cleaned {len(new_df)} texts")

            # Step 3: Filter for binary classification (if needed)
            if self.preprocessing_config.get("classification_type") == "binary":
                original_count = len(new_df)
                new_df = new_df[
                    new_df["sentiment"].isin(["positive", "negative"])
                ].copy()
                dropped = original_count - len(new_df)
                if dropped > 0:
                    logger.info(
                        f"Dropped {dropped} neutral samples for binary classification"
                    )

            # Step 4: Merge with existing training data (if requested)
            if merge_with_existing:
                logger.info("\n🔗 Merging with existing training data...")
                merged_df = self._merge_with_existing(new_df)
                logger.info(f"✅ Merged data: {len(merged_df)} total samples")
            else:
                merged_df = new_df
                logger.info("⏭️ Skipping merge (merge_with_existing=False)")

            # Step 5: Re-split data
            logger.info("\n✂️ Re-splitting data with stratification...")
            train_df, val_df, test_df = self.data_splitter.split_data(merged_df)

            logger.info(f"Train: {len(train_df)} samples")
            logger.info(f"Validation: {len(val_df)} samples")
            logger.info(f"Test: {len(test_df)} samples")

            # Step 6: Engineer features (only for train set to fit TF-IDF)
            logger.info("\n🔧 Engineering features...")
            train_features, val_features, test_features = self._engineer_features(
                train_df, val_df, test_df
            )

            # Step 7: Save processed data
            logger.info("\n💾 Saving processed data...")
            output_paths = self._save_processed_data(
                train_df, val_df, test_df, train_features, val_features, test_features
            )

            # Step 8: Generate statistics
            stats = self._generate_statistics(
                new_df, merged_df, train_df, val_df, test_df
            )

            logger.info("=" * 80)
            logger.info("✅ INCREMENTAL PREPROCESSING COMPLETED")
            logger.info("=" * 80)

            return {
                "status": "success",
                "new_samples": len(new_df),
                "total_samples_after_merge": len(merged_df),
                "train_samples": len(train_df),
                "val_samples": len(val_df),
                "test_samples": len(test_df),
                "output_paths": output_paths,
                "statistics": stats,
            }

        except Exception as e:
            logger.error(f"❌ Incremental preprocessing failed: {e}")
            import traceback

            logger.error(traceback.format_exc())
            return {
                "status": "error",
                "message": str(e),
                "traceback": traceback.format_exc(),
            }

    def _merge_with_existing(self, new_df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge new data with existing training data.

        Args:
            new_df: New collected data

        Returns:
            Merged DataFrame
        """
        # Load existing training data
        if not self.existing_train_path.exists():
            logger.warning(
                f"⚠️ Existing training data not found at {self.existing_train_path}. "
                f"Using only new data."
            )
            return new_df

        try:
            existing_train = pd.read_csv(self.existing_train_path)
            logger.info(f"Loaded {len(existing_train)} existing training samples")

            # Check if existing data has text_cleaned column
            if "text_cleaned" not in existing_train.columns:
                logger.warning(
                    "Existing data doesn't have 'text_cleaned'. Cleaning now..."
                )
                existing_train["text_cleaned"] = self.text_cleaner.clean_batch(
                    existing_train["text"].tolist(), preserve_case=True
                )

            # Ensure both have same columns for merge
            common_columns = ["text_cleaned", "sentiment"]
            if all(col in existing_train.columns for col in common_columns):
                # Select only common columns
                existing_subset = existing_train[common_columns].copy()
                new_subset = new_df[common_columns].copy()

                # Combine
                merged_df = pd.concat([existing_subset, new_subset], ignore_index=True)

                # Remove duplicates based on text_cleaned
                initial_count = len(merged_df)
                merged_df = merged_df.drop_duplicates(subset=["text_cleaned"])
                duplicates_removed = initial_count - len(merged_df)
                if duplicates_removed > 0:
                    logger.info(f"Removed {duplicates_removed} duplicate texts")

                # Shuffle
                merged_df = merged_df.sample(frac=1, random_state=42).reset_index(
                    drop=True
                )

                logger.info(
                    f"✅ Merged: {len(existing_train)} existing + {len(new_df)} new = "
                    f"{len(merged_df)} total (after dedup)"
                )

                return merged_df
            else:
                logger.warning("Columns mismatch. Using only new data.")
                return new_df

        except Exception as e:
            logger.error(f"Failed to merge with existing data: {e}")
            logger.warning("Using only new data")
            return new_df

    def _engineer_features(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Engineer features for train, validation, and test sets.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame

        Returns:
            Tuple of (train_features, val_features, test_features)
        """
        # Extract statistical features
        train_stat_features = self.feature_engineer.extract_statistical_features(
            train_df["text_cleaned"]
        )
        val_stat_features = self.feature_engineer.extract_statistical_features(
            val_df["text_cleaned"]
        )
        test_stat_features = self.feature_engineer.extract_statistical_features(
            test_df["text_cleaned"]
        )

        # Extract TF-IDF features
        if self.tfidf_vectorizer is not None:
            logger.info("Using existing TF-IDF vectorizer...")
            # Fit on train, transform on all sets
            train_tfidf = self.tfidf_vectorizer.transform(
                train_df["text_cleaned"].tolist()
            )
            val_tfidf = self.tfidf_vectorizer.transform(val_df["text_cleaned"].tolist())
            test_tfidf = self.tfidf_vectorizer.transform(
                test_df["text_cleaned"].tolist()
            )

            # Convert to DataFrame
            train_tfidf_df = pd.DataFrame(
                train_tfidf.toarray(),
                columns=[f"tfidf_{i}" for i in range(train_tfidf.shape[1])],
            )
            val_tfidf_df = pd.DataFrame(
                val_tfidf.toarray(),
                columns=[f"tfidf_{i}" for i in range(val_tfidf.shape[1])],
            )
            test_tfidf_df = pd.DataFrame(
                test_tfidf.toarray(),
                columns=[f"tfidf_{i}" for i in range(test_tfidf.shape[1])],
            )

            # Combine statistical and TF-IDF features
            train_features = pd.concat([train_stat_features, train_tfidf_df], axis=1)
            val_features = pd.concat([val_stat_features, val_tfidf_df], axis=1)
            test_features = pd.concat([test_stat_features, test_tfidf_df], axis=1)

        else:
            logger.warning("No TF-IDF vectorizer found. Fitting new one...")
            # Fit new TF-IDF on train data
            train_features = self.feature_engineer.extract_all_features(
                train_df["text_cleaned"], fit_tfidf=True
            )
            val_features = self.feature_engineer.extract_all_features(
                val_df["text_cleaned"], fit_tfidf=False
            )
            test_features = self.feature_engineer.extract_all_features(
                test_df["text_cleaned"], fit_tfidf=False
            )

        logger.info(f"✅ Engineered features: {train_features.shape[1]} features")

        return train_features, val_features, test_features

    def _save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_features: pd.DataFrame,
        val_features: pd.DataFrame,
        test_features: pd.DataFrame,
    ) -> Dict[str, str]:
        """
        Save processed data and features.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame
            train_features: Training features
            val_features: Validation features
            test_features: Test features

        Returns:
            Dict with output paths
        """
        # Save data splits (with text_cleaned and sentiment)
        train_output = self.processed_dir / "train.csv"
        val_output = self.processed_dir / "validation.csv"
        test_output = self.processed_dir / "test.csv"

        # Save only necessary columns
        train_df[["text_cleaned", "sentiment"]].to_csv(train_output, index=False)
        val_df[["text_cleaned", "sentiment"]].to_csv(val_output, index=False)
        test_df[["text_cleaned", "sentiment"]].to_csv(test_output, index=False)

        logger.info(f"💾 Saved data splits to {self.processed_dir}")

        # Save features (if needed for traditional ML)
        # Note: Transformer models don't need these, but we save for compatibility
        train_features_output = self.processed_dir / "train_features.csv"
        val_features_output = self.processed_dir / "validation_features.csv"
        test_features_output = self.processed_dir / "test_features.csv"

        train_features.to_csv(train_features_output, index=False)
        val_features.to_csv(val_features_output, index=False)
        test_features.to_csv(test_features_output, index=False)

        logger.info(f"💾 Saved features to {self.processed_dir}")

        return {
            "train": str(train_output),
            "validation": str(val_output),
            "test": str(test_output),
            "train_features": str(train_features_output),
            "val_features": str(val_features_output),
            "test_features": str(test_features_output),
        }

    def _generate_statistics(
        self,
        new_df: pd.DataFrame,
        merged_df: pd.DataFrame,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Dict:
        """Generate preprocessing statistics."""
        stats = {
            "new_samples": len(new_df),
            "total_samples_after_merge": len(merged_df),
            "train_samples": len(train_df),
            "val_samples": len(val_df),
            "test_samples": len(test_df),
            "new_sentiment_distribution": new_df["sentiment"].value_counts().to_dict(),
            "merged_sentiment_distribution": merged_df["sentiment"]
            .value_counts()
            .to_dict(),
            "train_sentiment_distribution": train_df["sentiment"]
            .value_counts()
            .to_dict(),
            "val_sentiment_distribution": val_df["sentiment"].value_counts().to_dict(),
            "test_sentiment_distribution": test_df["sentiment"]
            .value_counts()
            .to_dict(),
        }

        return stats
