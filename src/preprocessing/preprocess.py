# Placeholder - will be replaced
"""
Main preprocessing pipeline for movie sentiment analysis.
Orchestrates text cleaning, feature engineering, and data splitting.
"""

import pickle
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from src.preprocessing.data_splitter import DataSplitter
from src.preprocessing.feature_engineer import FeatureEngineer
from src.preprocessing.text_cleaner import TextCleaner
from src.utils.config import Config
from src.utils.helpers import save_dataframe, save_json, timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PreprocessingPipeline:
    """
    Complete preprocessing pipeline.

    Steps:
    1. Load validated data
    2. Clean text (remove noise, preserve sentiment signals)
    3. Split data (train/val/test with stratification)
    4. Engineer features (statistical + TF-IDF)
    5. Save processed data and artifacts
    6. Generate preprocessing report
    """

    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize preprocessing pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()

        # Initialize components
        self.text_cleaner = TextCleaner(self.config["preprocessing"])
        self.feature_engineer = FeatureEngineer(self.config["preprocessing"])
        self.data_splitter = DataSplitter(self.config["preprocessing"])

        # Paths
        self.raw_data_path = Config.RAW_DATA_DIR / "validated_data.csv"
        self.processed_dir = Config.PROCESSED_DATA_DIR

        # Statistics
        self.stats = {
            "pipeline_steps": [],
            "data_shapes": {},
            "class_distributions": {},
            "feature_counts": {},
            "cleaning_stats": {},
        }

        logger.info(f"PreprocessingPipeline initialized with config: {config_path}")

    def _load_config(self) -> Dict:
        """Load configuration from YAML file."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded configuration from {self.config_path}")
        return config

    @timer
    def load_data(self) -> pd.DataFrame:
        """
        Load validated data.

        Returns:
            DataFrame with validated data
        """
        logger.info(f"Loading data from {self.raw_data_path}...")

        df = pd.read_csv(self.raw_data_path)

        logger.info(f"Loaded {len(df)} samples with columns: {df.columns.tolist()}")

        # Log class distribution
        class_dist = df["sentiment"].value_counts()
        logger.info(f"Class distribution:\n{class_dist}")

        self.stats["pipeline_steps"].append("load_data")
        self.stats["data_shapes"]["original"] = df.shape
        self.stats["class_distributions"]["original"] = class_dist.to_dict()

        return df

    @timer
    def clean_texts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean text data.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with cleaned text
        """
        logger.info(f"Cleaning {len(df)} texts...")

        # Clean text column
        df["text_cleaned"] = self.text_cleaner.clean_batch(
            df["text"].tolist(), preserve_case=True
        )

        # Get cleaning statistics
        cleaning_stats = self.text_cleaner.get_stats()
        logger.info(f"Cleaning statistics: {cleaning_stats}")

        # Verify no empty texts after cleaning
        empty_count = (df["text_cleaned"].str.len() == 0).sum()
        if empty_count > 0:
            logger.warning(f"Found {empty_count} empty texts after cleaning")
            df = df[df["text_cleaned"].str.len() > 0].copy()
            logger.info(f"Removed empty texts. Remaining: {len(df)}")

        self.stats["pipeline_steps"].append("clean_texts")
        self.stats["data_shapes"]["after_cleaning"] = df.shape
        self.stats["cleaning_stats"] = cleaning_stats

        return df

    @timer
    def split_data(self, df: pd.DataFrame) -> tuple:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Input DataFrame

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info("Splitting data into train/val/test sets...")

        train_df, val_df, test_df = self.data_splitter.split_data(df)

        # Get split statistics
        split_stats = self.data_splitter.get_split_statistics(train_df, val_df, test_df)

        logger.info(
            f"Split complete: train={len(train_df)}, "
            f"val={len(val_df)}, test={len(test_df)}"
        )

        self.stats["pipeline_steps"].append("split_data")
        self.stats["data_shapes"]["train"] = train_df.shape
        self.stats["data_shapes"]["val"] = val_df.shape
        self.stats["data_shapes"]["test"] = test_df.shape
        self.stats["class_distributions"]["train"] = split_stats["train_distribution"]
        self.stats["class_distributions"]["val"] = split_stats["val_distribution"]
        self.stats["class_distributions"]["test"] = split_stats["test_distribution"]

        return train_df, val_df, test_df

    @timer
    def engineer_features(
        self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> tuple:
        """
        Engineer features for all splits.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame

        Returns:
            Tuple of (train_features, val_features, test_features)
        """
        logger.info("Engineering features...")

        # Extract features from training data (fit TF-IDF)
        train_features = self.feature_engineer.extract_all_features(
            train_df["text_cleaned"], fit_tfidf=True
        )

        # Extract features from validation data (transform only)
        val_features = self.feature_engineer.extract_all_features(
            val_df["text_cleaned"], fit_tfidf=False
        )

        # Extract features from test data (transform only)
        test_features = self.feature_engineer.extract_all_features(
            test_df["text_cleaned"], fit_tfidf=False
        )

        logger.info(
            f"Feature engineering complete. "
            f"Features per sample: {train_features.shape[1]}"
        )

        self.stats["pipeline_steps"].append("engineer_features")
        self.stats["feature_counts"]["statistical"] = 16  # Known from FeatureEngineer
        self.stats["feature_counts"]["tfidf"] = len(self.feature_engineer.feature_names)
        self.stats["feature_counts"]["total"] = train_features.shape[1]

        return train_features, val_features, test_features

    @timer
    def save_processed_data(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        train_features: pd.DataFrame,
        val_features: pd.DataFrame,
        test_features: pd.DataFrame,
    ):
        """
        Save processed data and features.

        Args:
            train_df: Training DataFrame (with cleaned text)
            val_df: Validation DataFrame
            test_df: Test DataFrame
            train_features: Training features
            val_features: Validation features
            test_features: Test features
        """
        logger.info("Saving processed data...")

        # Ensure processed directory exists
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Save raw text data (for transformer models)
        save_dataframe(
            train_df[["text_cleaned", "sentiment"]], self.processed_dir / "train.csv"
        )
        save_dataframe(
            val_df[["text_cleaned", "sentiment"]], self.processed_dir / "validation.csv"
        )
        save_dataframe(
            test_df[["text_cleaned", "sentiment"]], self.processed_dir / "test.csv"
        )

        # Save feature data (for traditional ML models)
        # Combine features with target
        train_with_features = train_features.copy()
        train_with_features["sentiment"] = train_df["sentiment"].values

        val_with_features = val_features.copy()
        val_with_features["sentiment"] = val_df["sentiment"].values

        test_with_features = test_features.copy()
        test_with_features["sentiment"] = test_df["sentiment"].values

        save_dataframe(train_with_features, self.processed_dir / "train_features.csv")
        save_dataframe(
            val_with_features, self.processed_dir / "validation_features.csv"
        )
        save_dataframe(test_with_features, self.processed_dir / "test_features.csv")

        logger.info("Processed data saved successfully")

        self.stats["pipeline_steps"].append("save_processed_data")

    @timer
    def save_artifacts(self):
        """Save preprocessing artifacts (vectorizers, encoders, etc.)."""
        logger.info("Saving preprocessing artifacts...")

        # Save TF-IDF vectorizer
        vectorizer_path = self.processed_dir / "tfidf_vectorizer.pkl"
        self.feature_engineer.save_vectorizer(vectorizer_path)

        # Save label encoder (for sentiment labels)
        from sklearn.preprocessing import LabelEncoder

        label_encoder = LabelEncoder()

        # Fit on all possible labels (positive, negative)
        label_encoder.fit(["negative", "positive"])

        encoder_path = self.processed_dir / "label_encoder.pkl"
        with open(encoder_path, "wb") as f:
            pickle.dump(label_encoder, f)

        logger.info(f"Label encoder saved to {encoder_path}")
        logger.info(
            f"Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}"
        )

        self.stats["pipeline_steps"].append("save_artifacts")

    @timer
    def generate_report(self):
        """Generate preprocessing report."""
        logger.info("Generating preprocessing report...")

        # Add metadata
        self.stats["config"] = self.config["preprocessing"]
        self.stats["completed_steps"] = len(self.stats["pipeline_steps"])

        # Save report
        report_path = self.processed_dir / "preprocessing_stats.json"
        save_json(self.stats, report_path)

        logger.info(f"Preprocessing report saved to {report_path}")

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print preprocessing summary."""
        print("\n" + "=" * 80)
        print("PREPROCESSING PIPELINE SUMMARY")
        print("=" * 80)

        print(f"\n✅ Completed Steps: {len(self.stats['pipeline_steps'])}")
        for i, step in enumerate(self.stats["pipeline_steps"], 1):
            print(f"   {i}. {step}")

        print("\n📊 Data Shapes:")
        for split, shape in self.stats["data_shapes"].items():
            print(f"   {split}: {shape}")

        print("\n🎯 Class Distributions:")
        for split, dist in self.stats["class_distributions"].items():
            print(f"   {split}: {dist}")

        print("\n🔧 Feature Counts:")
        for feat_type, count in self.stats["feature_counts"].items():
            print(f"   {feat_type}: {count}")

        print("\n🧹 Cleaning Statistics:")
        for stat, value in self.stats["cleaning_stats"].items():
            print(f"   {stat}: {value}")

        print("\n" + "=" * 80)

    @timer
    def run(self):
        """Run the complete preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline...")

        try:
            # Step 1: Load data
            df = self.load_data()

            # Step 2: Clean texts
            df = self.clean_texts(df)

            # Step 3: Split data
            train_df, val_df, test_df = self.split_data(df)

            # Step 4: Engineer features
            train_features, val_features, test_features = self.engineer_features(
                train_df, val_df, test_df
            )

            # Step 5: Save processed data
            self.save_processed_data(
                train_df, val_df, test_df, train_features, val_features, test_features
            )

            # Step 6: Save artifacts
            self.save_artifacts()

            # Step 7: Generate report
            self.generate_report()

            logger.info("✅ Preprocessing pipeline completed successfully!")

            return True

        except Exception as e:
            logger.error(f"❌ Preprocessing pipeline failed: {str(e)}")
            raise


def main():
    """Main entry point."""
    pipeline = PreprocessingPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
