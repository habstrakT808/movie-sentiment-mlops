"""
Unit tests for text_cleaner module.
"""

import pytest

from src.preprocessing.text_cleaner import TextCleaner, clean_text, clean_texts


class TestTextCleaner:
    """Test cases for TextCleaner class."""

    def setup_method(self):
        """Setup test fixtures."""
        self.cleaner = TextCleaner()

    def test_remove_urls(self):
        """Test URL removal."""
        text = "Check this out: https://example.com/page"
        cleaned = self.cleaner.clean(text)
        assert "https://" not in cleaned
        assert "example.com" not in cleaned

    def test_remove_html(self):
        """Test HTML tag removal."""
        text = "<p>This is a paragraph</p> with <b>bold</b> text"
        cleaned = self.cleaner.clean(text)
        assert "<p>" not in cleaned
        assert "<b>" not in cleaned
        assert "This is a paragraph" in cleaned

    def test_remove_emails(self):
        """Test email removal."""
        text = "Contact me at test@example.com for info"
        cleaned = self.cleaner.clean(text)
        assert "test@example.com" not in cleaned
        assert "Contact me at" in cleaned

    def test_remove_mentions(self):
        """Test mention removal."""
        text = "@username said this movie was great"
        cleaned = self.cleaner.clean(text)
        assert "@username" not in cleaned
        assert "said this movie" in cleaned

    def test_preserve_punctuation(self):
        """Test that punctuation is preserved."""
        text = "This movie was amazing! Really? Yes!!!"
        cleaned = self.cleaner.clean(text)
        assert "!" in cleaned
        assert "?" in cleaned

    def test_preserve_emojis(self):
        """Test that emojis are preserved."""
        text = "Great movie 😍😊👍"
        cleaned = self.cleaner.clean(text)
        assert "😍" in cleaned or "😊" in cleaned  # At least some emojis preserved

    def test_preserve_case(self):
        """Test case preservation."""
        text = "This Movie Was AMAZING"
        cleaned = self.cleaner.clean(text, preserve_case=True)
        assert "AMAZING" in cleaned
        assert cleaned != cleaned.lower()

    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        text = "Multiple    spaces   and\n\nnewlines"
        cleaned = self.cleaner.clean(text)
        assert "    " not in cleaned
        assert "\n\n" not in cleaned
        assert "Multiple spaces and newlines" == cleaned

    def test_normalize_excessive_punctuation(self):
        """Test excessive punctuation normalization."""
        text = "Amazing!!!!! Really?????"
        cleaned = self.cleaner.clean(text)
        assert "!!!!!" not in cleaned
        assert "?????" not in cleaned
        assert "!!" in cleaned
        assert "??" in cleaned

    def test_empty_string(self):
        """Test empty string handling."""
        cleaned = self.cleaner.clean("")
        assert cleaned == ""

    def test_non_string_input(self):
        """Test non-string input handling."""
        cleaned = self.cleaner.clean(None)
        assert cleaned == ""

        cleaned = self.cleaner.clean(123)
        assert cleaned == ""

    def test_batch_cleaning(self):
        """Test batch cleaning."""
        texts = ["Text with https://url.com", "<p>HTML text</p>", "Normal text"]
        cleaned = self.cleaner.clean_batch(texts)
        assert len(cleaned) == 3
        assert "https://" not in cleaned[0]
        assert "<p>" not in cleaned[1]

    def test_statistics_tracking(self):
        """Test that statistics are tracked."""
        self.cleaner.clean("https://example.com")
        self.cleaner.clean("<p>HTML</p>")
        stats = self.cleaner.get_stats()

        assert stats["total_processed"] == 2
        assert stats["urls_removed"] >= 1

    def test_unicode_normalization(self):
        """Test unicode normalization."""
        text = 'Café\'s "special" review – great…'
        cleaned = self.cleaner.clean(text)
        # Should normalize quotes and dashes
        assert '"' in cleaned or "'" in cleaned
        assert "..." in cleaned or "…" in cleaned


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_clean_text_function(self):
        """Test clean_text convenience function."""
        text = "Text with https://url.com"
        cleaned = clean_text(text)
        assert "https://" not in cleaned

    def test_clean_texts_function(self):
        """Test clean_texts convenience function."""
        texts = ["Text 1 https://url.com", "Text 2 <p>HTML</p>"]
        cleaned = clean_texts(texts)
        assert len(cleaned) == 2
        assert "https://" not in cleaned[0]
        assert "<p>" not in cleaned[1]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
