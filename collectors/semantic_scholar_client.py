"""
Semantic Scholar API client for fetching research papers.
Uses the Semantic Scholar Graph API v1.
Requires API key for higher rate limits (optional).
"""

import re
import time
import requests
from pathlib import Path
from typing import Optional

from config.settings import settings

S2_API_URL = "https://api.semanticscholar.org/graph/v1"


class SemanticScholarClient:
    """Fetch research papers from Semantic Scholar."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ResearchAutomation/1.0"})
        if settings.semantic_scholar_api_key:
            self.session.headers["x-api-key"] = settings.semantic_scholar_api_key

    def search(
        self,
        query: str,
        max_results: int = 10,
        fields: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Search Semantic Scholar for papers.

        Returns list of paper metadata dicts.
        """
        if fields is None:
            fields = [
                "paperId", "title", "abstract", "authors",
                "year", "citationCount", "openAccessPdf",
                "externalIds", "fieldsOfStudy", "publicationDate",
            ]

        params = {
            "query": query,
            "limit": min(max_results, 100),
            "fields": ",".join(fields),
        }

        try:
            resp = self.session.get(
                f"{S2_API_URL}/paper/search",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"[SemanticScholar] Request failed: {e}")
            return []
        except ValueError:
            print("[SemanticScholar] Invalid JSON response")
            return []

        papers = []
        for item in data.get("data", []):
            paper = self._normalize(item)
            if paper:
                papers.append(paper)

        return papers

    def _normalize(self, item: dict) -> Optional[dict]:
        """Normalize Semantic Scholar response to standard format."""
        title = (item.get("title") or "").strip()
        if not title:
            return None

        authors = [
            a.get("name", "") for a in (item.get("authors") or []) if a.get("name")
        ]

        pdf_url = ""
        oap = item.get("openAccessPdf")
        if oap and isinstance(oap, dict):
            pdf_url = oap.get("url", "")

        ext_ids = item.get("externalIds") or {}
        doi = ext_ids.get("DOI", "")
        arxiv_id = ext_ids.get("ArXiv", "")

        return {
            "source": "semantic_scholar",
            "s2_id": item.get("paperId", ""),
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": (item.get("abstract") or "").strip(),
            "published": item.get("publicationDate", "") or "",
            "year": item.get("year"),
            "citation_count": item.get("citationCount", 0),
            "fields_of_study": item.get("fieldsOfStudy") or [],
            "pdf_url": pdf_url,
            "doi": doi,
        }

    def download_pdf(self, pdf_url: str, paper_id: str) -> Optional[Path]:
        """Download PDF from open-access URL."""
        if not pdf_url:
            return None

        safe_id = re.sub(r"[^\w\-.]", "_", paper_id)
        filepath = settings.papers_dir / f"s2_{safe_id}.pdf"

        if filepath.exists():
            return filepath

        try:
            resp = self.session.get(pdf_url, timeout=60, stream=True)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")
            if "pdf" not in content_type and "octet-stream" not in content_type:
                return None
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            time.sleep(0.3)
            return filepath
        except requests.RequestException as e:
            print(f"[SemanticScholar] PDF download failed for {paper_id}: {e}")
            return None
