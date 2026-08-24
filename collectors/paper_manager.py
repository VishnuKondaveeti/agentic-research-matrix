"""
Unified paper manager that aggregates results from all collection sources,
handles deduplication, metadata storage, and PDF downloads.
"""

import json
import hashlib
from pathlib import Path
from typing import Optional
from difflib import SequenceMatcher

from config.settings import settings
from collectors.arxiv_client import ArxivClient
from collectors.semantic_scholar_client import SemanticScholarClient
from collectors.core_client import CoreClient
from collectors.openalex_client import OpenAlexClient


class PaperManager:
    """Unified interface for collecting research papers from multiple sources."""

    def __init__(self):
        self.arxiv = ArxivClient()
        self.semantic_scholar = SemanticScholarClient()
        self.core = CoreClient()
        self.openalex = OpenAlexClient()

    def search_all(
        self,
        query: str,
        max_per_source: int = 10,
        sources: Optional[list[str]] = None,
        min_year: Optional[int] = None,
        max_year: Optional[int] = None,
    ) -> list[dict]:
        """
        Search all configured sources and return deduplicated results.
        """
        if sources is None:
            sources = ["arxiv", "semantic_scholar", "core"]

        all_papers = []

        if "arxiv" in sources:
            try:
                papers = self.arxiv.search(query, max_results=max_per_source)
                all_papers.extend(papers)
                print(f"[PaperManager] arXiv returned {len(papers)} papers")
            except Exception as e:
                print(f"[PaperManager] arXiv search failed: {e}")

        if "semantic_scholar" in sources:
            try:
                papers = self.semantic_scholar.search(query, max_results=max_per_source)
                all_papers.extend(papers)
                print(f"[PaperManager] Semantic Scholar returned {len(papers)} papers")
            except Exception as e:
                print(f"[PaperManager] Semantic Scholar search failed: {e}")

        if "core" in sources:
            try:
                papers = self.core.search(query, max_results=max_per_source)
                all_papers.extend(papers)
                print(f"[PaperManager] CORE returned {len(papers)} papers")
            except Exception as e:
                print(f"[PaperManager] CORE search failed: {e}")

        # Post-fetch filtering for year if not handled by client
        if min_year or max_year:
            filtered = []
            for p in all_papers:
                pub_date = p.get("published", "")
                if not pub_date:
                    filtered.append(p) # Keep if no date and we don't have a hard requirement
                    continue
                try:
                    year = int(pub_date[:4])
                    if min_year is not None and year < min_year:
                        continue
                    if max_year is not None and year > max_year:
                        continue
                    filtered.append(p)
                except (ValueError, TypeError):
                    filtered.append(p)
            all_papers = filtered

        deduplicated = self._deduplicate(all_papers)
        print(f"[PaperManager] Total: {len(all_papers)} -> Deduplicated: {len(deduplicated)}")

        return deduplicated

    def _deduplicate(self, papers: list[dict]) -> list[dict]:
        """Remove duplicate papers based on title similarity and DOI matching."""
        unique = []
        seen_titles: list[str] = []
        seen_dois: set[str] = set()

        for paper in papers:
            title = paper.get("title", "").lower().strip()
            doi = paper.get("doi", "").strip()

            # Skip if DOI already seen
            if doi and doi in seen_dois:
                continue

            # Skip if title is too similar to one already seen
            is_dup = False
            for seen in seen_titles:
                if SequenceMatcher(None, title, seen).ratio() > 0.85:
                    is_dup = True
                    break

            if not is_dup:
                unique.append(paper)
                seen_titles.append(title)
                if doi:
                    seen_dois.add(doi)

        return unique

    def download_papers(self, papers: list[dict], max_downloads: int = 10) -> list[dict]:
        """
        Download PDFs for papers and update metadata with local file paths.

        Returns list of papers with 'local_pdf' field added.
        """
        downloaded = []
        count = 0

        for paper in papers:
            if count >= max_downloads:
                break

            pdf_url = paper.get("pdf_url", "")
            if not pdf_url:
                paper["local_pdf"] = ""
                downloaded.append(paper)
                continue

            source = paper.get("source", "unknown")
            paper_id = (
                paper.get("arxiv_id")
                or paper.get("s2_id")
                or paper.get("core_id")
                or hashlib.md5(paper["title"].encode()).hexdigest()[:12]
            )

            filepath = None
            if source == "arxiv":
                filepath = self.arxiv.download_pdf(pdf_url, paper_id)
            elif source == "core":
                filepath = self.core.download_pdf(pdf_url, paper_id)
            elif source == "openalex":
                filepath = self.openalex.download_pdf(pdf_url, paper_id)

            paper["local_pdf"] = str(filepath) if filepath else ""
            downloaded.append(paper)

            if filepath:
                count += 1

        return downloaded

    def save_metadata(self, papers: list[dict], query: str) -> Path:
        """Save paper metadata to JSON file in data/metadata/."""
        safe_query = "".join(c if c.isalnum() or c in " -_" else "" for c in query)
        safe_query = safe_query.replace(" ", "_")[:50]
        filepath = settings.metadata_dir / f"{safe_query}_papers.json"

        existing = []
        if filepath.exists():
            try:
                existing = json.loads(filepath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, IOError):
                existing = []

        # Merge with existing metadata
        all_papers = existing + papers
        all_papers = self._deduplicate(all_papers)

        filepath.write_text(
            json.dumps(all_papers, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[PaperManager] Saved {len(all_papers)} paper metadata to {filepath}")
        return filepath

    def load_metadata(self, query: str) -> list[dict]:
        """Load previously saved metadata for a query."""
        safe_query = "".join(c if c.isalnum() or c in " -_" else "" for c in query)
        safe_query = safe_query.replace(" ", "_")[:50]
        filepath = settings.metadata_dir / f"{safe_query}_papers.json"

        if not filepath.exists():
            return []

        try:
            return json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return []

    def list_all_metadata(self) -> list[dict]:
        """List all saved paper metadata files."""
        results = []
        for f in settings.metadata_dir.glob("*_papers.json"):
            try:
                papers = json.loads(f.read_text(encoding="utf-8"))
                results.append({
                    "file": f.name,
                    "query": f.stem.replace("_papers", "").replace("_", " "),
                    "count": len(papers),
                })
            except (json.JSONDecodeError, IOError):
                continue
        return results
