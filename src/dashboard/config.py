"""
Dashboard configuration.
"""

from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"
DATABASE_DIR = DATA_DIR / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_PATH = DATABASE_DIR / "predictions.db"

# Model paths
DISTILBERT_PATH = MODELS_DIR / "distilbert"
LOGISTIC_REGRESSION_PATH = MODELS_DIR / "logistic_regression"

# App settings
APP_TITLE = "🎬 Movie Sentiment Analysis"
APP_ICON = "🎬"
PAGE_TITLE = "Movie Sentiment Analysis"
LAYOUT = "wide"
INITIAL_SIDEBAR_STATE = "expanded"

# Theme colors
PRIMARY_COLOR = "#FF4B4B"
BACKGROUND_COLOR = "#0E1117"
SECONDARY_BG_COLOR = "#262730"
TEXT_COLOR = "#FAFAFA"

# Prediction settings
MAX_TEXT_LENGTH = 5000
MIN_TEXT_LENGTH = 10
DEFAULT_MODEL = "DistilBERT"

# Available models
AVAILABLE_MODELS = {
    "DistilBERT": {
        "path": DISTILBERT_PATH,
        "type": "transformer",
        "accuracy": 0.9250,
        "description": "State-of-the-art transformer model",
    },
    "Logistic Regression": {
        "path": LOGISTIC_REGRESSION_PATH,
        "type": "traditional",
        "accuracy": 0.8740,
        "description": "Fast and efficient baseline model",
    },
}

# Sentiment emojis
SENTIMENT_EMOJIS = {"positive": "😊", "negative": "😞", "neutral": "😐"}

# Sentiment colors
SENTIMENT_COLORS = {
    "positive": "#00D26A",  # Green
    "negative": "#FF4B4B",  # Red
    "neutral": "#FFA500",  # Orange
}
