"""
Kaggle dataset downloader for IMDb movie reviews.
Downloads and processes pre-labeled movie review datasets.
"""

from pathlib import Path
from typing import Optional

import pandas as pd

from src.utils.config import Config
from src.utils.helpers import compute_hash, timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class KaggleDownloader:
    """Downloader for Kaggle datasets."""

    def __init__(self):
        """Initialize Kaggle API."""
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            self.api = KaggleApi()
            self.api.authenticate()
            logger.info("Successfully authenticated with Kaggle API")
        except Exception as e:
            logger.error(f"Failed to authenticate with Kaggle API: {e}")
            logger.error(
                "Make sure kaggle.json is in ~/.kaggle/ or KAGGLE_USERNAME and KAGGLE_KEY are set"
            )
            raise

    @timer
    def download_imdb_dataset(
        self,
        output_dir: Path,
        dataset_name: str = "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
    ) -> Path:
        """
        Download IMDb dataset from Kaggle.

        Args:
            output_dir: Directory to save the dataset
            dataset_name: Kaggle dataset identifier

        Returns:
            Path to the downloaded CSV file
        """
        logger.info(f"Downloading dataset: {dataset_name}")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Download dataset
            self.api.dataset_download_files(
                dataset_name, path=str(output_dir), unzip=True
            )

            # Find the CSV file
            csv_files = list(output_dir.glob("*.csv"))

            if not csv_files:
                raise FileNotFoundError("No CSV file found in downloaded dataset")

            csv_path = csv_files[0]
            logger.info(f"Dataset downloaded to: {csv_path}")

            return csv_path

        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise

    @timer
    def process_imdb_dataset(
        self, csv_path: Path, sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Process IMDb dataset to match our schema.

        Args:
            csv_path: Path to the CSV file
            sample_size: Number of samples to use (None for all)

        Returns:
            Processed DataFrame
        """
        logger.info(f"Processing IMDb dataset from: {csv_path}")

        # Read CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} samples from Kaggle")

        # Sample if requested
        if sample_size and sample_size < len(df):
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} samples")

        # Process to match our schema
        processed_df = pd.DataFrame()

        processed_df["id"] = df.index.astype(str)
        processed_df["title"] = "IMDb Review"  # Generic title
        processed_df["text"] = df["review"]
        processed_df["score"] = None  # Not available
        processed_df["upvote_ratio"] = None
        processed_df["num_comments"] = None
        processed_df["created_utc"] = pd.NaT
        processed_df["author"] = "anonymous"
        processed_df["subreddit"] = "kaggle_imdb"
        processed_df["url"] = None
        processed_df["movie_title"] = "various"
        processed_df["source_type"] = "kaggle"

        # Map sentiment
        sentiment_map = {"positive": "positive", "negative": "negative"}
        processed_df["sentiment"] = df["sentiment"].map(sentiment_map)

        # Add hash
        processed_df["hash"] = processed_df["text"].apply(compute_hash)

        # Add metadata
        from datetime import datetime

        processed_df["collection_date"] = datetime.now()
        processed_df["text_length"] = processed_df["text"].str.len()
        processed_df["word_count"] = processed_df["text"].str.split().str.len()

        # Remove any rows with missing sentiment
        processed_df = processed_df.dropna(subset=["sentiment"])

        logger.info(f"Processed {len(processed_df)} samples")
        logger.info(
            f"Sentiment distribution:\n{processed_df['sentiment'].value_counts()}"
        )

        return processed_df

    @timer
    def get_balanced_kaggle_data(
        self, output_dir: Path, target_per_sentiment: int = 10000
    ) -> pd.DataFrame:
        """
        Download and process Kaggle dataset with balanced sentiments.

        Args:
            output_dir: Directory to save the dataset
            target_per_sentiment: Target number of samples per sentiment

        Returns:
            Balanced DataFrame
        """
        # Download dataset
        csv_path = self.download_imdb_dataset(output_dir)

        # Process dataset
        df = self.process_imdb_dataset(csv_path)

        # Balance dataset
        balanced_dfs = []
        for sentiment in ["positive", "negative"]:
            sentiment_df = df[df["sentiment"] == sentiment]
            if len(sentiment_df) > target_per_sentiment:
                sentiment_df = sentiment_df.sample(
                    n=target_per_sentiment, random_state=42
                )
            balanced_dfs.append(sentiment_df)

        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

        logger.info(f"Final balanced Kaggle dataset: {len(balanced_df)} samples")
        logger.info(
            f"Sentiment distribution:\n{balanced_df['sentiment'].value_counts()}"
        )

        return balanced_df


def main():
    """Main function for testing."""
    downloader = KaggleDownloader()

    # Test download
    df = downloader.get_balanced_kaggle_data(
        output_dir=Config.EXTERNAL_DATA_DIR, target_per_sentiment=1000
    )

    print(f"\nDownloaded {len(df)} samples")
    print(f"\nSentiment distribution:\n{df['sentiment'].value_counts()}")
    print(f"\nSample data:\n{df.head()}")


if __name__ == "__main__":
    main()
