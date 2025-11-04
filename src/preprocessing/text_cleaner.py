"""
Text cleaning utilities for sentiment analysis preprocessing.
Preserves sentiment signals while removing noise.
"""

import re
import unicodedata
from typing import Dict, List, Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TextCleaner:
    """
    Text cleaner that preserves sentiment signals.

    Strategy:
    - Remove: URLs, HTML tags, emails, mentions
    - Normalize: Whitespace, unicode characters
    - Preserve: Punctuation (!?.), emojis, capitalization (for sentiment)
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize text cleaner with configuration.

        Args:
            config: Dictionary with cleaning parameters
        """
        self.config = config or {}
        self.stats = {
            "total_processed": 0,
            "urls_removed": 0,
            "html_removed": 0,
            "emails_removed": 0,
            "mentions_removed": 0,
            "excessive_spaces_normalized": 0,
        }

        # Compile regex patterns once for efficiency
        self._compile_patterns()

        logger.info("TextCleaner initialized")

    def _compile_patterns(self):
        """Compile regex patterns for text cleaning."""
        # URL pattern (http, https, www)
        self.url_pattern = re.compile(
            r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
            r"|www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
        )

        # HTML tags
        self.html_pattern = re.compile(r"<[^>]+>")

        # Email addresses
        self.email_pattern = re.compile(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        )

        # Reddit/Twitter mentions (@username)
        self.mention_pattern = re.compile(r"@\w+")

        # Multiple spaces/newlines
        self.space_pattern = re.compile(r"\s+")

        # Multiple punctuation (reduce !!! -> !!, ??? -> ??, but keep ellipsis ...)
        self.punct_pattern = re.compile(r"([!?]){3,}")

    def clean(self, text: str, preserve_case: bool = True) -> str:
        """
        Clean text while preserving sentiment signals.

        Args:
            text: Input text to clean
            preserve_case: Whether to preserve original case (True for sentiment)

        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            logger.warning(f"Non-string input received: {type(text)}")
            return ""

        original_text = text

        # Remove URLs
        text = self.url_pattern.sub(" ", text)
        if text != original_text:
            self.stats["urls_removed"] += 1

        # Remove HTML tags
        text = self.html_pattern.sub(" ", text)
        if "<" not in text:
            self.stats["html_removed"] += 1

        # Remove email addresses
        text = self.email_pattern.sub(" ", text)

        # Remove mentions
        text = self.mention_pattern.sub(" ", text)

        # Normalize unicode characters (e.g., é -> e, but keep emojis)
        text = self._normalize_unicode(text)

        # Normalize excessive punctuation (!!!! -> !!, ???? -> ??) and keep ellipsis
        # But keep some emphasis for sentiment
        text = self.punct_pattern.sub(r"\1\1", text)

        # Normalize whitespace (multiple spaces -> single space)
        text = self.space_pattern.sub(" ", text)
        self.stats["excessive_spaces_normalized"] += 1

        # Strip leading/trailing whitespace
        text = text.strip()

        # Optional: Convert to lowercase (NOT recommended for sentiment)
        if not preserve_case and self.config.get("lowercase", False):
            text = text.lower()

        self.stats["total_processed"] += 1

        return text

    def _normalize_unicode(self, text: str) -> str:
        """
        Normalize unicode characters while preserving emojis.

        Args:
            text: Input text

        Returns:
            Text with normalized unicode
        """
        # Normalize unicode (NFC form)
        text = unicodedata.normalize("NFC", text)

        # Replace common unicode variants
        replacements = {
            "\u2018": "'",  # Left single quote
            "\u2019": "'",  # Right single quote
            "\u201c": '"',  # Left double quote
            "\u201d": '"',  # Right double quote
            "\u2013": "-",  # En dash
            "\u2014": "-",  # Em dash
            "\u2026": "...",  # Ellipsis
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text

    def clean_batch(self, texts: List[str], preserve_case: bool = True) -> List[str]:
        """
        Clean a batch of texts.

        Args:
            texts: List of texts to clean
            preserve_case: Whether to preserve original case

        Returns:
            List of cleaned texts
        """
        logger.info(f"Cleaning batch of {len(texts)} texts...")
        cleaned = [self.clean(text, preserve_case) for text in texts]
        logger.info(f"Batch cleaning complete. Stats: {self.get_stats()}")
        return cleaned

    def get_stats(self) -> Dict:
        """Get cleaning statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset cleaning statistics."""
        for key in self.stats:
            self.stats[key] = 0
        logger.info("Cleaning statistics reset")


def clean_text(text: str, config: Optional[Dict] = None) -> str:
    """
    Convenience function to clean a single text.

    Args:
        text: Text to clean
        config: Optional configuration dictionary

    Returns:
        Cleaned text
    """
    cleaner = TextCleaner(config)
    return cleaner.clean(text)


def clean_texts(texts: List[str], config: Optional[Dict] = None) -> List[str]:
    """
    Convenience function to clean multiple texts.

    Args:
        texts: List of texts to clean
        config: Optional configuration dictionary

    Returns:
        List of cleaned texts
    """
    cleaner = TextCleaner(config)
    return cleaner.clean_batch(texts)


# Example usage and testing
if __name__ == "__main__":
    # Test cases
    test_texts = [
        "Check out this movie review at https://example.com/review !!!",
        "This movie was AMAZING!!! Best film ever 😍😍😍",
        "<p>HTML tags should be removed</p> but punctuation stays!",
        "Contact me at test@email.com for more reviews",
        "@username mentioned this movie... it was okay.",
        "Multiple     spaces    and\n\nnewlines\n\nshould be normalized",
    ]

    cleaner = TextCleaner()

    print("=" * 80)
    print("TEXT CLEANER TEST")
    print("=" * 80)

    for i, text in enumerate(test_texts, 1):
        cleaned = cleaner.clean(text)
        print(f"\n{i}. ORIGINAL: {text}")
        print(f"   CLEANED:  {cleaned}")

    print("\n" + "=" * 80)
    print("CLEANING STATISTICS")
    print("=" * 80)
    stats = cleaner.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
