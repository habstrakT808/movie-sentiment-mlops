"""
Periodic Data Collector for Continuous Learning.
Collects new data from Reddit and Kaggle on a scheduled basis.
"""

from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml

from src.data_collection.kaggle_downloader import KaggleDownloader
from src.data_collection.reddit_collector import RedditCollector
from src.utils.config import Config
from src.utils.helpers import save_dataframe, save_json, timer
from src.utils.logger import get_logger

logger = get_logger(__name__)


class PeriodicDataCollector:
    """
    Collector for periodic data collection from Reddit and Kaggle.

    Features:
    - Collects new data from Reddit (recent posts/comments)
    - Collects new data from Kaggle (if dataset updated)
    - Avoids duplicates by checking against existing data
    - Saves incremental data to separate folder
    - Returns collection statistics
    """

    def __init__(self, config_path: str = "params.yaml"):
        """
        Initialize periodic data collector.

        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.data_params = self.config.get("data_collection", {})

        # Initialize collectors
        try:
            self.reddit_collector = RedditCollector()
            logger.info("✅ Reddit collector initialized")
        except Exception as e:
            logger.warning(f"⚠️ Reddit collector initialization failed: {e}")
            self.reddit_collector = None

        try:
            self.kaggle_downloader = KaggleDownloader()
            logger.info("✅ Kaggle downloader initialized")
        except Exception as e:
            logger.warning(f"⚠️ Kaggle downloader initialization failed: {e}")
            self.kaggle_downloader = None

        # Setup incremental data directory
        self.incremental_dir = Config.RAW_DATA_DIR / "incremental"
        self.incremental_dir.mkdir(parents=True, exist_ok=True)

        # Load existing data hashes to avoid duplicates
        self.existing_hashes = self._load_existing_hashes()

        logger.info(
            f"PeriodicDataCollector initialized. "
            f"Found {len(self.existing_hashes)} existing data hashes to avoid duplicates."
        )

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

    def _load_existing_hashes(self) -> set:
        """
        Load hashes from existing data to avoid duplicates.

        Returns:
            Set of existing data hashes
        """
        existing_hashes = set()

        # Check main raw data files
        raw_files = [
            Config.RAW_DATA_DIR / "reddit_reviews.csv",
            Config.RAW_DATA_DIR / "kaggle_imdb.csv",
            Config.RAW_DATA_DIR / "validated_data.csv",
        ]

        # Check incremental data files
        if self.incremental_dir.exists():
            incremental_files = list(self.incremental_dir.glob("*.csv"))
            raw_files.extend(incremental_files)

        for file_path in raw_files:
            if file_path.exists():
                try:
                    df = pd.read_csv(file_path)
                    if "hash" in df.columns:
                        existing_hashes.update(df["hash"].tolist())
                        logger.debug(f"Loaded {len(df)} hashes from {file_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to load hashes from {file_path}: {e}")

        return existing_hashes

    def _collect_movie_data_batch(
        self, movies: list, subreddits: list, reddit_data: list
    ) -> None:
        """Collect data for a batch of movies."""
        for movie in movies[:3]:
            try:
                movie_data = self.reddit_collector.collect_movie_data(
                    movie_title=movie,
                    subreddits=subreddits[:2],
                    posts_per_subreddit=20,
                    comments_per_post=10,
                )
                if len(movie_data) > 0:
                    reddit_data.append(movie_data)
                    logger.info(f"Collected {len(movie_data)} samples for '{movie}'")
            except Exception as e:
                logger.warning(f"Failed to collect data for '{movie}': {e}")

    def _process_reddit_data(
        self, reddit_data: list, max_reddit_samples: int, duplicates_skipped: int
    ) -> tuple:
        """Process and deduplicate Reddit data."""
        if not reddit_data:
            return 0, None, duplicates_skipped

        reddit_df = pd.concat(reddit_data, ignore_index=True)
        initial_count = len(reddit_df)
        reddit_df = reddit_df[~reddit_df["hash"].isin(self.existing_hashes)]
        duplicates_skipped += initial_count - len(reddit_df)

        if len(reddit_df) > max_reddit_samples:
            reddit_df = reddit_df.sample(n=max_reddit_samples, random_state=42)
            logger.info(f"Limited Reddit samples to {max_reddit_samples}")

        reddit_samples = len(reddit_df)
        logger.info(f"✅ Collected {reddit_samples} new samples from Reddit")
        return reddit_samples, reddit_df, duplicates_skipped

    def _collect_from_reddit(
        self, max_reddit_samples: int, duplicates_skipped: int
    ) -> tuple:
        """Collect data from Reddit."""
        reddit_samples = 0
        reddit_df = None

        if self.reddit_collector is None:
            return reddit_samples, reddit_df, duplicates_skipped

        try:
            logger.info("\n" + "-" * 80)
            logger.info("PHASE 1: Collecting from Reddit")
            logger.info("-" * 80)

            positive_movies = self.data_params.get("positive_movies", [])
            negative_movies = self.data_params.get("negative_movies", [])
            subreddits = self.data_params.get("reddit", {}).get(
                "subreddits", ["movies"]
            )

            reddit_data = []

            # Collect from positive movies
            self._collect_movie_data_batch(positive_movies, subreddits, reddit_data)

            # Collect from negative movies
            self._collect_movie_data_batch(negative_movies, subreddits, reddit_data)

            # Process and deduplicate
            if reddit_data:
                (
                    reddit_samples,
                    reddit_df,
                    duplicates_skipped,
                ) = self._process_reddit_data(
                    reddit_data, max_reddit_samples, duplicates_skipped
                )
            else:
                logger.warning("⚠️ No Reddit data collected")

        except Exception as e:
            logger.error(f"❌ Reddit collection failed: {e}")
            import traceback

            logger.error(traceback.format_exc())

        return reddit_samples, reddit_df, duplicates_skipped

    def _collect_from_kaggle(
        self, max_kaggle_samples: int, duplicates_skipped: int
    ) -> tuple:
        """Collect data from Kaggle."""
        kaggle_samples = 0
        kaggle_df = None

        if self.kaggle_downloader is None:
            return kaggle_samples, kaggle_df, duplicates_skipped

        try:
            logger.info("\n" + "-" * 80)
            logger.info("PHASE 2: Collecting from Kaggle")
            logger.info("-" * 80)

            kaggle_df = self.kaggle_downloader.get_balanced_kaggle_data(
                output_dir=Config.EXTERNAL_DATA_DIR,
                target_per_sentiment=max_kaggle_samples // 2,
            )

            if len(kaggle_df) > 0:
                initial_count = len(kaggle_df)
                kaggle_df = kaggle_df[~kaggle_df["hash"].isin(self.existing_hashes)]
                duplicates_skipped += initial_count - len(kaggle_df)

                if len(kaggle_df) > max_kaggle_samples:
                    kaggle_df = kaggle_df.sample(n=max_kaggle_samples, random_state=42)
                    logger.info(f"Limited Kaggle samples to {max_kaggle_samples}")

                kaggle_samples = len(kaggle_df)
                logger.info(f"✅ Collected {kaggle_samples} new samples from Kaggle")
            else:
                logger.warning("⚠️ No Kaggle data collected")

        except Exception as e:
            logger.error(f"❌ Kaggle collection failed: {e}")
            import traceback

            logger.error(traceback.format_exc())

        return kaggle_samples, kaggle_df, duplicates_skipped

    @timer
    def collect_incremental_data(
        self,
        min_samples_per_source: int = 100,
        time_filter: str = "week",
        max_reddit_samples: int = 1000,
        max_kaggle_samples: int = 500,
    ) -> Dict:
        """
        Collect new data from Reddit and Kaggle.

        Args:
            min_samples_per_source: Minimum samples to collect from each source
            time_filter: Time filter for Reddit ("day", "week", "month")
            max_reddit_samples: Maximum samples to collect from Reddit
            max_kaggle_samples: Maximum samples to collect from Kaggle

        Returns:
            Dict with collection results
        """
        logger.info("=" * 80)
        logger.info("📥 STARTING PERIODIC DATA COLLECTION")
        logger.info("=" * 80)

        all_new_data = []
        duplicates_skipped = 0

        # Collect from Reddit
        reddit_samples, reddit_df, duplicates_skipped = self._collect_from_reddit(
            max_reddit_samples, duplicates_skipped
        )
        if reddit_df is not None:
            all_new_data.append(reddit_df)

        # Collect from Kaggle
        kaggle_samples, kaggle_df, duplicates_skipped = self._collect_from_kaggle(
            max_kaggle_samples, duplicates_skipped
        )
        if kaggle_df is not None:
            all_new_data.append(kaggle_df)

        # ========== COMBINE AND FINALIZE ==========
        if not all_new_data:
            logger.warning("⚠️ No new data collected from any source")
            return {
                "status": "no_data",
                "message": "No new data collected",
                "reddit_samples": 0,
                "kaggle_samples": 0,
                "total_new_samples": 0,
                "duplicates_skipped": duplicates_skipped,
            }

        # Combine all data
        combined_df = pd.concat(all_new_data, ignore_index=True)

        # Remove duplicates within new data
        initial_combined = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=["hash"])
        duplicates_skipped += initial_combined - len(combined_df)

        # Filter by length (from config)
        min_length = self.data_params.get("min_review_length", 20)
        max_length = self.data_params.get("max_review_length", 5000)

        combined_df = combined_df[
            (combined_df["text_length"] >= min_length)
            & (combined_df["text_length"] <= max_length)
        ]

        # Shuffle
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

        total_new_samples = len(combined_df)

        logger.info("\n" + "-" * 80)
        logger.info("COLLECTION SUMMARY")
        logger.info("-" * 80)
        logger.info(f"Reddit samples: {reddit_samples}")
        logger.info(f"Kaggle samples: {kaggle_samples}")
        logger.info(f"Total new samples: {total_new_samples}")
        logger.info(f"Duplicates skipped: {duplicates_skipped}")
        logger.info(
            f"Sentiment distribution:\n{combined_df['sentiment'].value_counts()}"
        )

        # ========== SAVE INCREMENTAL DATA ==========
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"incremental_data_{timestamp}.csv"
        output_path = self.incremental_dir / output_filename

        save_dataframe(combined_df, output_path)
        logger.info(f"💾 Saved incremental data to: {output_path}")

        # Update existing hashes
        self.existing_hashes.update(combined_df["hash"].tolist())

        # Save metadata
        metadata = {
            "collection_timestamp": timestamp,
            "collection_date": datetime.now().isoformat(),
            "reddit_samples": int(reddit_samples),
            "kaggle_samples": int(kaggle_samples),
            "total_new_samples": int(total_new_samples),
            "duplicates_skipped": int(duplicates_skipped),
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
            "output_path": str(output_path),
        }

        metadata_path = self.incremental_dir / f"metadata_{timestamp}.json"
        save_json(metadata, metadata_path)

        logger.info("=" * 80)
        logger.info("✅ PERIODIC DATA COLLECTION COMPLETED")
        logger.info("=" * 80)

        return {
            "status": "success",
            "reddit_samples": int(reddit_samples),
            "kaggle_samples": int(kaggle_samples),
            "total_new_samples": int(total_new_samples),
            "duplicates_skipped": int(duplicates_skipped),
            "output_path": str(output_path),
            "metadata": metadata,
        }
