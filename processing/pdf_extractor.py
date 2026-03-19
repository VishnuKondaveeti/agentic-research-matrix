"""
PDF text extraction using PyMuPDF (fitz).
Extracts text page-by-page with metadata tracking.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import Optional


class PDFExtractor:
    """Extract text content from PDF files."""

    def extract(self, pdf_path: str | Path) -> dict:
        """
        Extract text from a PDF file.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            Dict with keys: 'text', 'pages', 'metadata'
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            raise RuntimeError(f"Failed to open PDF {pdf_path}: {e}")

        pages = []
        full_text_parts = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            pages.append({
                "page_number": page_num + 1,
                "text": text,
                "char_count": len(text),
            })
            full_text_parts.append(text)

        metadata = {
            "filename": pdf_path.name,
            "filepath": str(pdf_path),
            "page_count": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
        }

        doc.close()

        return {
            "text": "\n\n".join(full_text_parts),
            "pages": pages,
            "metadata": metadata,
        }

    def extract_pages(self, pdf_path: str | Path) -> list[dict]:
        """Extract text as a list of page dicts."""
        result = self.extract(pdf_path)
        return result["pages"]

    def extract_text(self, pdf_path: str | Path) -> str:
        """Extract full text as a single string."""
        result = self.extract(pdf_path)
        return result["text"]
