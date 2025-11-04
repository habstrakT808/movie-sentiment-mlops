"""
Main script to collect data from all sources.
Orchestrates Reddit and Kaggle data collection.
"""

from datetime import datetime

import pandas as pd
import yaml

from src.data_collection.kaggle_downloader import KaggleDownloader
from src.data_collection.reddit_collector import RedditCollector
from src.utils.config import Config
from src.utils.helpers import save_dataframe, save_json, timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


@timer
def collect_all_data():
    """Main function to collect data from all sources."""
    logger.info("=" * 80)
    logger.info("STARTING DATA COLLECTION")
    logger.info("=" * 80)

    # Load parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    data_params = params["data_collection"]

    # Initialize collectors
    reddit_collector = RedditCollector()
    kaggle_downloader = KaggleDownloader()

    # ========== COLLECT FROM KAGGLE ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1: Collecting from Kaggle")
    logger.info("=" * 80)

    kaggle_df = kaggle_downloader.get_balanced_kaggle_data(
        output_dir=Config.EXTERNAL_DATA_DIR,
        target_per_sentiment=data_params["kaggle"]["sample_size"]
        // 2,  # Divide by 2 for pos/neg
    )

    # Save Kaggle data
    kaggle_output_path = Config.RAW_DATA_DIR / "kaggle_imdb.csv"
    save_dataframe(kaggle_df, kaggle_output_path)

    # ========== COLLECT FROM REDDIT ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 2: Collecting from Reddit")
    logger.info("=" * 80)

    # Calculate target per sentiment for Reddit
    # Total target - Kaggle samples = Reddit samples needed
    total_target = data_params["target_total_samples"]
    kaggle_samples = len(kaggle_df)
    reddit_samples_needed = total_target - kaggle_samples
    reddit_per_sentiment = reddit_samples_needed // 3  # Divide by 3 sentiments

    logger.info(f"Target Reddit samples per sentiment: {reddit_per_sentiment}")

    reddit_df = reddit_collector.collect_balanced_dataset(
        positive_movies=data_params["positive_movies"],
        negative_movies=data_params["negative_movies"],
        neutral_movies=data_params["neutral_movies"],
        subreddits=data_params["reddit"]["subreddits"],
        target_per_sentiment=reddit_per_sentiment,
    )

    # Save Reddit data
    reddit_output_path = Config.RAW_DATA_DIR / "reddit_reviews.csv"
    save_dataframe(reddit_df, reddit_output_path)

    # ========== COMBINE AND FINALIZE ==========
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 3: Combining and Finalizing")
    logger.info("=" * 80)

    # Combine datasets
    combined_df = pd.concat([kaggle_df, reddit_df], ignore_index=True)

    # Remove duplicates
    initial_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=["hash"])
    duplicates_removed = initial_count - len(combined_df)
    logger.info(f"Removed {duplicates_removed} duplicates")

    # Filter by length
    combined_df = combined_df[
        (combined_df["text_length"] >= data_params["min_review_length"])
        & (combined_df["text_length"] <= data_params["max_review_length"])
    ]

    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"\nFinal dataset: {len(combined_df)} samples")
    logger.info(f"Sentiment distribution:\n{combined_df['sentiment'].value_counts()}")
    logger.info(f"Source distribution:\n{combined_df['source_type'].value_counts()}")

    # ========== SAVE STATISTICS ==========
    collection_stats = {
        "total_samples": len(combined_df),
        "kaggle_samples": len(kaggle_df),
        "reddit_samples": len(reddit_df),
        "duplicates_removed": duplicates_removed,
        "sentiment_distribution": combined_df["sentiment"].value_counts().to_dict(),
        "source_distribution": combined_df["source_type"].value_counts().to_dict(),
        "text_length_stats": {
            "mean": float(combined_df["text_length"].mean()),
            "median": float(combined_df["text_length"].median()),
            "min": int(combined_df["text_length"].min()),
            "max": int(combined_df["text_length"].max()),
        },
        "word_count_stats": {
            "mean": float(combined_df["word_count"].mean()),
            "median": float(combined_df["word_count"].median()),
            "min": int(combined_df["word_count"].min()),
            "max": int(combined_df["word_count"].max()),
        },
        "collection_date": datetime.now().isoformat(),
    }

    stats_path = Config.RAW_DATA_DIR / "collection_stats.json"
    save_json(collection_stats, stats_path)

    # Save metadata
    metadata = {
        "data_sources": ["kaggle_imdb", "reddit"],
        "subreddits": data_params["reddit"]["subreddits"],
        "movies_collected": {
            "positive": data_params["positive_movies"],
            "negative": data_params["negative_movies"],
            "neutral": data_params["neutral_movies"],
        },
        "collection_parameters": data_params,
        "collection_date": datetime.now().isoformat(),
    }

    metadata_path = Config.RAW_DATA_DIR / "collection_metadata.json"
    save_json(metadata, metadata_path)

    logger.info("\n" + "=" * 80)
    logger.info("DATA COLLECTION COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"Total samples collected: {len(combined_df)}")
    logger.info(f"Data saved to: {Config.RAW_DATA_DIR}")
    logger.info(f"Statistics saved to: {stats_path}")
    logger.info(f"Metadata saved to: {metadata_path}")

    return combined_df


def main():
    """Entry point."""
    try:
        collect_all_data()
        logger.info("\n✓ Data collection completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"\n✗ Data collection failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
