"""
OpenAlex API client for fetching research papers.
Free to use, no API key required. 250M+ works, generous rate limits.
Docs: https://docs.openalex.org/
"""

import re
import time
import requests
from pathlib import Path
from typing import Optional

from config.settings import settings

OPENALEX_API_URL = "https://api.openalex.org"


class OpenAlexClient:
    """Fetch research papers from OpenAlex."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "ResearchAutomation/1.0 (mailto:research@example.com)",
        })

    def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> list[dict]:
        """
        Search OpenAlex for papers matching query.

        Returns list of paper metadata dicts with standardized keys.
        """
        params = {
            "search": query,
            "per_page": min(max_results, 50),
            "sort": "relevance_score:desc",
            "filter": "type:article",
        }

        try:
            resp = self.session.get(
                f"{OPENALEX_API_URL}/works",
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            print(f"[OpenAlex] Request failed: {e}")
            return []
        except ValueError:
            print("[OpenAlex] Invalid JSON response")
            return []

        papers = []
        for item in data.get("results", []):
            paper = self._normalize(item)
            if paper:
                papers.append(paper)

        return papers

    def _normalize(self, item: dict) -> Optional[dict]:
        """Normalize OpenAlex response to standard format."""
        title = (item.get("title") or "").strip()
        if not title:
            return None

        # Extract author names
        authors = []
        for authorship in item.get("authorships", []):
            author = authorship.get("author", {})
            name = author.get("display_name", "")
            if name:
                authors.append(name)

        # Publication date
        pub_date = item.get("publication_date", "") or ""

        # Year
        year = item.get("publication_year")

        # DOI
        doi_raw = item.get("doi") or ""
        doi = doi_raw.replace("https://doi.org/", "") if doi_raw else ""

        # Citation count
        citation_count = item.get("cited_by_count", 0)

        # PDF URL - check open access
        pdf_url = ""
        oa = item.get("open_access", {})
        if oa.get("is_oa"):
            pdf_url = oa.get("oa_url", "") or ""
        # Also check best_oa_location
        best_oa = item.get("best_oa_location") or {}
        if not pdf_url and best_oa:
            pdf_url = best_oa.get("pdf_url", "") or best_oa.get("landing_page_url", "") or ""

        # Abstract - OpenAlex returns inverted index, reconstruct it
        abstract = self._reconstruct_abstract(item.get("abstract_inverted_index"))

        # OpenAlex ID
        openalex_id = (item.get("id") or "").replace("https://openalex.org/", "")

        # Concepts/topics
        concepts = []
        for concept in item.get("concepts", []):
            if concept.get("score", 0) > 0.3:
                concepts.append(concept.get("display_name", ""))

        return {
            "source": "openalex",
            "openalex_id": openalex_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "published": pub_date,
            "year": year,
            "citation_count": citation_count,
            "concepts": concepts,
            "pdf_url": pdf_url,
            "doi": doi,
        }

    @staticmethod
    def _reconstruct_abstract(inverted_index: Optional[dict]) -> str:
        """
        Reconstruct abstract from OpenAlex inverted index format.
        The inverted index maps words to their positions in the text.
        """
        if not inverted_index:
            return ""

        try:
            # Build position -> word mapping
            word_positions = []
            for word, positions in inverted_index.items():
                for pos in positions:
                    word_positions.append((pos, word))

            # Sort by position and join
            word_positions.sort(key=lambda x: x[0])
            return " ".join(word for _, word in word_positions)
        except Exception:
            return ""

    def download_pdf(self, pdf_url: str, paper_id: str) -> Optional[Path]:
        """Download PDF from open-access URL."""
        if not pdf_url:
            return None

        safe_id = re.sub(r"[^\w\-.]", "_", paper_id)
        filepath = settings.papers_dir / f"oalex_{safe_id}.pdf"

        if filepath.exists():
            return filepath

        try:
            resp = self.session.get(
                pdf_url,
                timeout=60,
                stream=True,
                allow_redirects=True,
            )
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
            print(f"[OpenAlex] PDF download failed for {paper_id}: {e}")
            return None
