"""
Text cleaning and preprocessing for extracted PDF text.
Removes noise, normalizes whitespace, and strips artifacts.
"""

import re
import unicodedata


class TextCleaner:
    """Clean and preprocess extracted text from research papers."""

    def clean(self, text: str) -> str:
        """
        Apply all cleaning steps to text.

        Steps:
        1. Fix unicode / encoding artifacts
        2. Remove excessive whitespace
        3. Remove page numbers and headers/footers
        4. Normalize line breaks
        5. Remove reference markers
        """
        text = self._fix_unicode(text)
        text = self._remove_headers_footers(text)
        text = self._normalize_whitespace(text)
        text = self._remove_page_numbers(text)
        text = self._clean_references_markers(text)
        text = self._remove_urls(text)
        return text.strip()

    def _fix_unicode(self, text: str) -> str:
        """Normalize unicode characters and fix common encoding issues."""
        text = unicodedata.normalize("NFKC", text)
        # Fix common ligature issues
        replacements = {
            "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
            "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
            "\u2013": "-", "\u2014": "--", "\u2026": "...",
            "\u00a0": " ",
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _remove_headers_footers(self, text: str) -> str:
        """Remove common header/footer patterns from academic papers."""
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Skip very short lines that look like headers/footers
            if len(stripped) < 3:
                cleaned.append(line)
                continue
            # Skip common footer patterns
            if re.match(r"^\d+\s*$", stripped):  # Just a page number
                continue
            if re.match(r"^(Page|page)\s+\d+", stripped):
                continue
            if re.match(r"^(Preprint|Draft|Manuscript|arXiv:)", stripped, re.IGNORECASE):
                continue
            cleaned.append(line)
        return "\n".join(cleaned)

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace while preserving paragraph breaks."""
        # Replace multiple spaces with single space
        text = re.sub(r"[ \t]+", " ", text)
        # Replace 3+ newlines with double newline (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Fix hyphenated line breaks (e.g., "com-\npute" -> "compute")
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        return text

    def _remove_page_numbers(self, text: str) -> str:
        """Remove standalone page numbers."""
        text = re.sub(r"\n\s*\d{1,3}\s*\n", "\n", text)
        return text

    def _clean_references_markers(self, text: str) -> str:
        """Clean inline reference markers like [1], [2,3], etc."""
        # Keep the text but normalize reference markers
        text = re.sub(r"\[(\d+(?:,\s*\d+)*)\]", r"[\1]", text)
        return text

    def _remove_urls(self, text: str) -> str:
        """Remove long URLs that clutter the text."""
        text = re.sub(
            r"https?://\S{50,}",  # Only remove very long URLs
            "[URL]",
            text,
        )
        return text
