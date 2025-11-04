"""
Feature engineering for sentiment analysis.
Extracts statistical features and TF-IDF features for traditional ML models.
"""

import pickle
import string
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FeatureEngineer:
    """
    Feature engineer for sentiment analysis.

    Extracts:
    1. Statistical features (text length, word count, etc.)
    2. TF-IDF features
    3. Sentiment lexicon features (TextBlob polarity/subjectivity)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize feature engineer.

        Args:
            config: Configuration dictionary with parameters
        """
        self.config = config or {}
        self.tfidf_vectorizer = None
        self.feature_names = []

        # TF-IDF parameters
        self.max_features = self.config.get("max_features_tfidf", 5000)
        # Use a permissive default for small corpora to avoid pruning all terms
        self.min_df = self.config.get("min_df", 1)
        self.max_df = self.config.get("max_df", 0.95)
        self.ngram_range = self.config.get("ngram_range", (1, 2))

        logger.info(
            f"FeatureEngineer initialized with max_features={self.max_features}, "
            f"ngram_range={self.ngram_range}"
        )

    def extract_statistical_features(self, texts: pd.Series) -> pd.DataFrame:
        """
        Extract statistical features from texts.

        Args:
            texts: Series of text strings

        Returns:
            DataFrame with statistical features
        """
        logger.info(f"Extracting statistical features from {len(texts)} texts...")

        features = pd.DataFrame()

        # Basic length features
        features["char_count"] = texts.apply(len)
        features["word_count"] = texts.apply(lambda x: len(str(x).split()))
        features["avg_word_length"] = texts.apply(
            lambda x: np.mean([len(word) for word in str(x).split()])
            if len(str(x).split()) > 0
            else 0
        )

        # Sentence features
        features["sentence_count"] = texts.apply(lambda x: len(str(x).split(".")))
        features["avg_sentence_length"] = (
            features["word_count"] / features["sentence_count"]
        )

        # Punctuation features
        features["exclamation_count"] = texts.apply(lambda x: str(x).count("!"))
        features["question_count"] = texts.apply(lambda x: str(x).count("?"))
        features["punctuation_count"] = texts.apply(
            lambda x: sum(1 for char in str(x) if char in string.punctuation)
        )
        features["punctuation_ratio"] = (
            features["punctuation_count"] / features["char_count"]
        )

        # Capital letters (emphasis indicator)
        features["capital_count"] = texts.apply(
            lambda x: sum(1 for char in str(x) if char.isupper())
        )
        features["capital_ratio"] = features["capital_count"] / features["char_count"]

        # Special characters
        features["digit_count"] = texts.apply(
            lambda x: sum(1 for char in str(x) if char.isdigit())
        )
        features["digit_ratio"] = features["digit_count"] / features["char_count"]

        # Sentiment lexicon features (TextBlob)
        logger.info("Computing sentiment lexicon features (TextBlob)...")
        sentiment_scores = texts.apply(self._get_textblob_sentiment)
        features["textblob_polarity"] = sentiment_scores.apply(lambda x: x[0])
        features["textblob_subjectivity"] = sentiment_scores.apply(lambda x: x[1])

        # Word diversity (unique words / total words)
        features["unique_word_ratio"] = texts.apply(
            lambda x: len(set(str(x).lower().split())) / len(str(x).split())
            if len(str(x).split()) > 0
            else 0
        )

        # Handle any NaN or inf values
        features = features.replace([np.inf, -np.inf], 0)
        features = features.fillna(0)

        logger.info(f"Extracted {features.shape[1]} statistical features")

        return features

    def _get_textblob_sentiment(self, text: str) -> Tuple[float, float]:
        """
        Get TextBlob sentiment scores.

        Args:
            text: Input text

        Returns:
            Tuple of (polarity, subjectivity)
        """
        try:
            blob = TextBlob(str(text))
            return blob.sentiment.polarity, blob.sentiment.subjectivity
        except Exception as e:
            logger.warning(f"TextBlob error: {e}")
            return 0.0, 0.0

    def fit_tfidf(self, texts: pd.Series) -> "FeatureEngineer":
        """
        Fit TF-IDF vectorizer on training texts.

        Args:
            texts: Series of text strings (training data)

        Returns:
            Self for method chaining
        """
        logger.info(
            f"Fitting TF-IDF vectorizer on {len(texts)} texts "
            f"(max_features={self.max_features})..."
        )

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=self.min_df,
            max_df=self.max_df,
            ngram_range=self.ngram_range,
            strip_accents="unicode",
            lowercase=True,
            analyzer="word",
            token_pattern=r"\w{1,}",
            stop_words="english",
        )

        self.tfidf_vectorizer.fit(texts)
        self.feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist()

        logger.info(
            f"TF-IDF vectorizer fitted. Vocabulary size: {len(self.feature_names)}"
        )

        return self

    def transform_tfidf(self, texts: pd.Series) -> pd.DataFrame:
        """
        Transform texts to TF-IDF features.

        Args:
            texts: Series of text strings

        Returns:
            DataFrame with TF-IDF features
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf() first.")

        logger.info(f"Transforming {len(texts)} texts to TF-IDF features...")

        tfidf_matrix = self.tfidf_vectorizer.transform(texts)
        tfidf_df = pd.DataFrame(
            tfidf_matrix.toarray(),
            columns=[f"tfidf_{name}" for name in self.feature_names],
            index=texts.index,  # preserve original index to align with statistical features
        )

        logger.info(f"TF-IDF transformation complete. Shape: {tfidf_df.shape}")

        return tfidf_df

    def extract_all_features(
        self, texts: pd.Series, fit_tfidf: bool = False
    ) -> pd.DataFrame:
        """
        Extract all features (statistical + TF-IDF).

        Args:
            texts: Series of text strings
            fit_tfidf: Whether to fit TF-IDF vectorizer (True for training data)

        Returns:
            DataFrame with all features
        """
        logger.info(f"Extracting all features from {len(texts)} texts...")

        # Statistical features
        stat_features = self.extract_statistical_features(texts)

        # TF-IDF features
        if fit_tfidf:
            self.fit_tfidf(texts)

        tfidf_features = self.transform_tfidf(texts)

        # Combine all features
        all_features = pd.concat([stat_features, tfidf_features], axis=1)

        logger.info(
            f"Feature extraction complete. Total features: {all_features.shape[1]} "
            f"(statistical: {stat_features.shape[1]}, TF-IDF: {tfidf_features.shape[1]})"
        )

        return all_features

    def save_vectorizer(self, filepath: Path):
        """
        Save TF-IDF vectorizer to disk.

        Args:
            filepath: Path to save the vectorizer
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("No vectorizer to save. Fit the vectorizer first.")

        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(self.tfidf_vectorizer, f)

        logger.info(f"TF-IDF vectorizer saved to {filepath}")

    def load_vectorizer(self, filepath: Path):
        """
        Load TF-IDF vectorizer from disk.

        Args:
            filepath: Path to load the vectorizer from
        """
        with open(filepath, "rb") as f:
            self.tfidf_vectorizer = pickle.load(f)

        self.feature_names = self.tfidf_vectorizer.get_feature_names_out().tolist()

        logger.info(
            f"TF-IDF vectorizer loaded from {filepath}. "
            f"Vocabulary size: {len(self.feature_names)}"
        )

    def get_feature_importance(self, n_top: int = 20) -> pd.DataFrame:
        """
        Get top TF-IDF features by average score.

        Args:
            n_top: Number of top features to return

        Returns:
            DataFrame with top features and their average scores
        """
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted.")

        # This is a placeholder - actual importance would come from model coefficients
        logger.info(f"Getting top {n_top} TF-IDF features...")

        return pd.DataFrame(
            {"feature": self.feature_names[:n_top], "vocabulary_index": range(n_top)}
        )


# Convenience functions
def extract_features(
    texts: pd.Series, config: Optional[Dict] = None, fit_tfidf: bool = False
) -> Tuple[pd.DataFrame, FeatureEngineer]:
    """
    Convenience function to extract features.

    Args:
        texts: Series of text strings
        config: Configuration dictionary
        fit_tfidf: Whether to fit TF-IDF vectorizer

    Returns:
        Tuple of (features DataFrame, FeatureEngineer instance)
    """
    engineer = FeatureEngineer(config)
    features = engineer.extract_all_features(texts, fit_tfidf=fit_tfidf)
    return features, engineer


# Example usage and testing
if __name__ == "__main__":
    # Test data
    test_texts = pd.Series(
        [
            "This movie was absolutely AMAZING!!! Best film I've ever seen! 😍",
            "Terrible movie. Waste of time and money. Very disappointed.",
            "It was okay, nothing special. Some good parts, some bad parts.",
            "LOVED IT! Must watch! Incredible acting and stunning visuals!!!",
            "Boring and predictable. Would not recommend to anyone.",
        ]
    )

    print("=" * 80)
    print("FEATURE ENGINEER TEST")
    print("=" * 80)

    # Initialize feature engineer
    config = {"max_features_tfidf": 50, "ngram_range": (1, 2)}
    engineer = FeatureEngineer(config)

    # Extract statistical features
    print("\n1. STATISTICAL FEATURES")
    print("-" * 80)
    stat_features = engineer.extract_statistical_features(test_texts)
    print(stat_features.head())
    print(f"\nStatistical features shape: {stat_features.shape}")

    # Extract all features
    print("\n2. ALL FEATURES (Statistical + TF-IDF)")
    print("-" * 80)
    all_features = engineer.extract_all_features(test_texts, fit_tfidf=True)
    print(f"All features shape: {all_features.shape}")
    print("\nFirst 5 statistical features:")
    print(all_features.iloc[:, :5])

    # Show feature names
    print("\n3. FEATURE NAMES")
    print("-" * 80)
    print(f"Total features: {len(all_features.columns)}")
    print(f"Statistical features: {len(stat_features.columns)}")
    print(f"TF-IDF features: {len(all_features.columns) - len(stat_features.columns)}")
    print("\nStatistical feature names:")
    print(stat_features.columns.tolist())

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
