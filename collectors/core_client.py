"""
CORE API v3 client for fetching research papers.
Requires CORE API key.
"""

import re
import time
import requests
from pathlib import Path
from typing import Optional

from config.settings import settings

CORE_API_URL = "https://api.core.ac.uk/v3"


class CoreClient:
    """Fetch research papers from CORE."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ResearchAutomation/1.0"})
        if settings.core_api_key:
            self.session.headers["Authorization"] = f"Bearer {settings.core_api_key}"

    @property
    def available(self) -> bool:
        return bool(settings.core_api_key)

    def search(self, query: str, max_results: int = 10) -> list[dict]:
        """
        Search CORE for papers.

        Returns list of paper metadata dicts.
        """
        if not self.available:
            print("[CORE] No API key configured - skipping CORE search.")
            return []

        params = {
            "q": query,
            "limit": min(max_results, 100),
        }

        try:
            resp = self.session.get(
                f"{CORE_API_URL}/search/works",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"[CORE] Request failed: {e}")
            return []
        except ValueError:
            print("[CORE] Invalid JSON response")
            return []

        papers = []
        for item in data.get("results", []):
            paper = self._normalize(item)
            if paper:
                papers.append(paper)

        return papers

    def _normalize(self, item: dict) -> Optional[dict]:
        """Normalize CORE response to standard format."""
        title = (item.get("title") or "").strip()
        if not title:
            return None

        authors = []
        for a in item.get("authors", []) or []:
            name = a if isinstance(a, str) else a.get("name", "")
            if name:
                authors.append(name.strip())

        pdf_url = item.get("downloadUrl", "") or item.get("sourceFulltextUrls", [""])[0] if item.get("sourceFulltextUrls") else ""

        return {
            "source": "core",
            "core_id": str(item.get("id", "")),
            "title": title,
            "authors": authors,
            "abstract": (item.get("abstract") or "").strip(),
            "published": (item.get("publishedDate") or item.get("yearPublished") or ""),
            "year": item.get("yearPublished"),
            "pdf_url": pdf_url if isinstance(pdf_url, str) else "",
            "doi": item.get("doi", "") or "",
            "language": item.get("language", {}).get("code", "") if isinstance(item.get("language"), dict) else "",
        }

    def download_pdf(self, pdf_url: str, paper_id: str) -> Optional[Path]:
        """Download PDF from CORE."""
        if not pdf_url:
            return None

        safe_id = re.sub(r"[^\w\-.]", "_", paper_id)
        filepath = settings.papers_dir / f"core_{safe_id}.pdf"

        if filepath.exists():
            return filepath

        try:
            resp = self.session.get(pdf_url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            time.sleep(0.3)
            return filepath
        except requests.RequestException as e:
            print(f"[CORE] PDF download failed for {paper_id}: {e}")
            return None
