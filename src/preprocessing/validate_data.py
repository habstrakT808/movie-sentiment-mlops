"""Data validation script."""
import pandas as pd

from src.utils.config import Config
from src.utils.helpers import save_json
from src.utils.logger import get_logger

logger = get_logger(__name__)


def validate_data():
    """Validate collected data."""
    logger.info("Starting data validation...")

    # Load data
    reddit_df = pd.read_csv(Config.RAW_DATA_DIR / "reddit_reviews.csv")
    kaggle_df = pd.read_csv(Config.RAW_DATA_DIR / "kaggle_imdb.csv")

    # Combine
    df = pd.concat([reddit_df, kaggle_df], ignore_index=True)

    # Validation checks
    checks = {
        "total_samples": int(len(df)),
        "no_missing_text": bool(df["text"].notna().all()),
        "no_missing_sentiment": bool(df["sentiment"].notna().all()),
        "valid_sentiments": bool(
            df["sentiment"].isin(["positive", "negative", "neutral"]).all()
        ),
        "min_length_ok": bool((df["text"].str.len() >= 20).all()),
        "max_length_ok": bool((df["text"].str.len() <= 5000).all()),
        "no_duplicates": bool(len(df) == len(df.drop_duplicates(subset=["hash"]))),
        "sentiment_balance": {
            str(k): int(v) for k, v in df["sentiment"].value_counts().to_dict().items()
        },
    }

    # Save validated data
    output_path = Config.RAW_DATA_DIR / "validated_data.csv"
    df.to_csv(output_path, index=False)

    # Save validation report
    report_path = Config.RAW_DATA_DIR / "validation_report.json"
    save_json(checks, report_path)

    logger.info(f"Validation complete. Report saved to {report_path}")

    return df, checks


if __name__ == "__main__":
    validate_data()
