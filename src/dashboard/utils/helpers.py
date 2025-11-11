"""
Helper functions for dashboard.
"""

import re
from datetime import datetime
from typing import Optional

import pandas as pd
import streamlit as st


def clean_text_for_display(text: str, max_length: int = 100) -> str:
    """
    Clean and truncate text for display.

    Args:
        text: Input text
        max_length: Maximum length

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length] + "..."

    return text


def format_confidence(confidence: float) -> str:
    """
    Format confidence score as percentage.

    Args:
        confidence: Confidence score (0-1)

    Returns:
        Formatted string
    """
    return f"{confidence * 100:.1f}%"


def format_timestamp(timestamp: datetime) -> str:
    """
    Format timestamp for display.

    Args:
        timestamp: Datetime object

    Returns:
        Formatted string
    """
    now = datetime.now()
    diff = now - timestamp

    if diff.seconds < 60:
        return "Just now"
    elif diff.seconds < 3600:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    elif diff.seconds < 86400:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    else:
        return timestamp.strftime("%Y-%m-%d %H:%M")


def get_sentiment_badge(sentiment: str, confidence: float) -> str:
    """
    Get HTML badge for sentiment.

    Args:
        sentiment: Sentiment label
        confidence: Confidence score

    Returns:
        HTML string
    """
    from src.dashboard.config import SENTIMENT_COLORS, SENTIMENT_EMOJIS

    color = SENTIMENT_COLORS.get(sentiment.lower(), "#808080")
    emoji = SENTIMENT_EMOJIS.get(sentiment.lower(), "")

    return f"""
    <div style="
        display: inline-block;
        padding: 8px 16px;
        background-color: {color};
        color: white;
        border-radius: 20px;
        font-weight: bold;
        font-size: 18px;
        margin: 10px 0;
    ">
        {emoji} {sentiment.upper()} ({format_confidence(confidence)})
    </div>
    """


def create_metric_card(title: str, value: str, delta: Optional[str] = None) -> str:
    """
    Create metric card HTML.

    Args:
        title: Metric title
        value: Metric value
        delta: Change indicator

    Returns:
        HTML string
    """
    delta_html = ""
    if delta:
        delta_color = "#00D26A" if delta.startswith("+") else "#FF4B4B"
        delta_html = (
            f'<div style="color: {delta_color}; font-size: 14px;">{delta}</div>'
        )

    return f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    ">
        <div style="color: rgba(255, 255, 255, 0.8); font-size: 14px; margin-bottom: 5px;">
            {title}
        </div>
        <div style="color: white; font-size: 32px; font-weight: bold; margin-bottom: 5px;">
            {value}
        </div>
        {delta_html}
    </div>
    """


def validate_text_input(
    text: str, min_length: int = 10, max_length: int = 5000
) -> tuple[bool, str]:
    """
    Validate text input.

    Args:
        text: Input text
        min_length: Minimum length
        max_length: Maximum length

    Returns:
        (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "⚠️ Please enter some text"

    if len(text) < min_length:
        return False, f"⚠️ Text too short (minimum {min_length} characters)"

    if len(text) > max_length:
        return False, f"⚠️ Text too long (maximum {max_length} characters)"

    return True, ""


def create_progress_bar(value: float, label: str, color: str = "#00D26A") -> str:
    """
    Create progress bar HTML.

    Args:
        value: Progress value (0-1)
        label: Label text
        color: Bar color

    Returns:
        HTML string
    """
    percentage = value * 100

    return f"""
    <div style="margin: 10px 0;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
            <span style="font-size: 14px;">{label}</span>
            <span style="font-size: 14px; font-weight: bold;">{percentage:.1f}%</span>
        </div>
        <div style="
            width: 100%;
            height: 20px;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 10px;
            overflow: hidden;
        ">
            <div style="
                width: {percentage}%;
                height: 100%;
                background-color: {color};
                transition: width 0.3s ease;
            "></div>
        </div>
    </div>
    """


def export_to_csv(df: pd.DataFrame, filename: str = "predictions.csv") -> bytes:
    """
    Export dataframe to CSV.

    Args:
        df: Dataframe to export
        filename: Output filename

    Returns:
        CSV bytes
    """
    return df.to_csv(index=False).encode("utf-8")


def display_error(message: str):
    """Display error message."""
    st.error(f"❌ {message}")


def display_success(message: str):
    """Display success message."""
    st.success(f"✅ {message}")


def display_info(message: str):
    """Display info message."""
    st.info(f"ℹ️ {message}")


def display_warning(message: str):
    """Display warning message."""
    st.warning(f"⚠️ {message}")
