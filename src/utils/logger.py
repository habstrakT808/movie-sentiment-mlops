"""
Logging configuration for the project.
Provides centralized logging with file and console output.
"""

import sys
from pathlib import Path

from loguru import logger

from src.utils.config import Config

# Remove default logger
logger.remove()

# Add console logger
logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level=Config.LOG_LEVEL,
    colorize=True,
)

# Add file logger
log_file = Path(Config.LOG_FILE)
log_file.parent.mkdir(parents=True, exist_ok=True)

logger.add(
    Config.LOG_FILE,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level=Config.LOG_LEVEL,
    rotation="10 MB",
    retention="30 days",
    compression="zip",
)


def get_logger(name: str):
    """Get a logger with the specified name."""
    return logger.bind(name=name)
