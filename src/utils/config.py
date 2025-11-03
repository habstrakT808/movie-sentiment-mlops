"""
Configuration management for the project.
Loads environment variables and provides centralized config access.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Central configuration class."""

    # Project paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"
    MODELS_DIR = PROJECT_ROOT / "models"
    LOGS_DIR = PROJECT_ROOT / "logs"
    CONFIGS_DIR = PROJECT_ROOT / "configs"

    # Reddit API
    REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
    REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
    REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "MovieSentimentBot/1.0")

    # Kaggle API
    KAGGLE_USERNAME = os.getenv("KAGGLE_USERNAME")
    KAGGLE_KEY = os.getenv("KAGGLE_KEY")

    # YouTube API
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")

    # MLflow
    MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    MLFLOW_EXPERIMENT_NAME = os.getenv(
        "MLFLOW_EXPERIMENT_NAME", "movie_sentiment_analysis"
    )

    # DVC
    DVC_REMOTE_URL = os.getenv("DVC_REMOTE_URL", "/tmp/dvc-storage")

    # API
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", 8000))

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = os.getenv("LOG_FILE", "logs/app.log")

    # Model
    DEFAULT_MODEL_TYPE = os.getenv("DEFAULT_MODEL_TYPE", "transformer")
    MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "./models")

    # Data Collection
    DATA_COLLECTION_BATCH_SIZE = int(os.getenv("DATA_COLLECTION_BATCH_SIZE", 1000))
    MAX_RETRIES = int(os.getenv("MAX_RETRIES", 3))
    RETRY_DELAY = int(os.getenv("RETRY_DELAY", 5))

    @classmethod
    def validate(cls) -> bool:
        """Validate that all required configurations are set."""
        required_configs = [
            ("REDDIT_CLIENT_ID", cls.REDDIT_CLIENT_ID),
            ("REDDIT_CLIENT_SECRET", cls.REDDIT_CLIENT_SECRET),
            ("KAGGLE_USERNAME", cls.KAGGLE_USERNAME),
            ("KAGGLE_KEY", cls.KAGGLE_KEY),
        ]

        missing = [name for name, value in required_configs if not value]

        if missing:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing)}. "
                f"Please set these in your .env file."
            )

        return True

    @classmethod
    def create_directories(cls):
        """Create all necessary directories."""
        directories = [
            cls.DATA_DIR,
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.EXTERNAL_DATA_DIR,
            cls.MODELS_DIR,
            cls.LOGS_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Create directories on import
Config.create_directories()
