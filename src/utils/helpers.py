"""
Helper utility functions used across the project.
"""

import hashlib
import json
import time
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def timer(func: Callable) -> Callable:
    """Decorator to measure function execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        logger.info(f"{func.__name__} executed in {execution_time:.2f} seconds")
        return result

    return wrapper


def retry(max_attempts: int = 3, delay: int = 1, backoff: int = 2) -> Callable:
    """
    Decorator to retry a function on failure with exponential backoff.

    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between retries in seconds
        backoff: Multiplier for delay after each retry
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_delay = delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__name__} failed after {max_attempts} attempts: {str(e)}"
                        )
                        raise
                    logger.warning(
                        f"{func.__name__} attempt {attempt} failed: {str(e)}. "
                        f"Retrying in {current_delay} seconds..."
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff

        return wrapper

    return decorator


def compute_hash(text: str) -> str:
    """Compute SHA256 hash of text."""
    return hashlib.sha256(text.encode()).hexdigest()


def save_json(data: Dict[str, Any], filepath: Path):
    """Save dictionary to JSON file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved JSON to {filepath}")


def load_json(filepath: Path) -> Dict[str, Any]:
    """Load dictionary from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    logger.info(f"Loaded JSON from {filepath}")
    return data


def save_dataframe(df: pd.DataFrame, filepath: Path, index: bool = False):
    """Save DataFrame to CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filepath, index=index)
    logger.info(f"Saved DataFrame ({len(df)} rows) to {filepath}")


def load_dataframe(filepath: Path) -> pd.DataFrame:
    """Load DataFrame from CSV file."""
    df = pd.read_csv(filepath)
    logger.info(f"Loaded DataFrame ({len(df)} rows) from {filepath}")
    return df


def get_file_size(filepath: Path) -> str:
    """Get human-readable file size."""
    size_bytes = filepath.stat().st_size
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"


def ensure_directory(directory: Path):
    """Ensure directory exists, create if it doesn't."""
    directory.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {directory}")
