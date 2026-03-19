"""
arXiv API client for fetching research papers.
Uses the arXiv Atom feed API (no API key required).
"""

import re
import time
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from config.settings import settings

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


class ArxivClient:
    """Fetch research papers from arXiv."""

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "ResearchAutomation/1.0"})

    def search(
        self,
        query: str,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> list[dict]:
        """
        Search arXiv for papers matching query.

        Returns list of paper metadata dicts with keys:
        source, arxiv_id, title, authors, abstract, published, updated,
        categories, pdf_url, doi
        """
        params = {
            "search_query": f"all:{query}",
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        try:
            resp = self.session.get(ARXIV_API_URL, params=params, timeout=30)
            resp.raise_for_status()
        except requests.RequestException as e:
            print(f"[ArxivClient] Request failed: {e}")
            return []

        return self._parse_response(resp.text)

    def _parse_response(self, xml_text: str) -> list[dict]:
        """Parse Atom XML response into paper metadata list."""
        papers = []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []

        for entry in root.findall(f"{ATOM_NS}entry"):
            paper = self._parse_entry(entry)
            if paper:
                papers.append(paper)

        return papers

    def _parse_entry(self, entry: ET.Element) -> Optional[dict]:
        """Parse a single Atom entry into a paper dict."""
        title_el = entry.find(f"{ATOM_NS}title")
        if title_el is None or not title_el.text:
            return None

        title = re.sub(r"\s+", " ", title_el.text.strip())

        # Extract arXiv ID from the entry id URL
        id_el = entry.find(f"{ATOM_NS}id")
        arxiv_id = ""
        if id_el is not None and id_el.text:
            arxiv_id = id_el.text.split("/abs/")[-1]

        # Authors
        authors = []
        for author_el in entry.findall(f"{ATOM_NS}author"):
            name_el = author_el.find(f"{ATOM_NS}name")
            if name_el is not None and name_el.text:
                authors.append(name_el.text.strip())

        # Abstract
        summary_el = entry.find(f"{ATOM_NS}summary")
        abstract = ""
        if summary_el is not None and summary_el.text:
            abstract = re.sub(r"\s+", " ", summary_el.text.strip())

        # Dates
        published = self._get_text(entry, f"{ATOM_NS}published", "")
        updated = self._get_text(entry, f"{ATOM_NS}updated", "")

        # Categories
        categories = []
        for cat_el in entry.findall(f"{ARXIV_NS}primary_category"):
            term = cat_el.get("term", "")
            if term:
                categories.append(term)
        for cat_el in entry.findall(f"{ATOM_NS}category"):
            term = cat_el.get("term", "")
            if term and term not in categories:
                categories.append(term)

        # PDF link
        pdf_url = ""
        for link_el in entry.findall(f"{ATOM_NS}link"):
            if link_el.get("title") == "pdf":
                pdf_url = link_el.get("href", "")
                break

        # DOI
        doi_el = entry.find(f"{ARXIV_NS}doi")
        doi = doi_el.text.strip() if doi_el is not None and doi_el.text else ""

        return {
            "source": "arxiv",
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "published": published[:10] if published else "",
            "updated": updated[:10] if updated else "",
            "categories": categories,
            "pdf_url": pdf_url,
            "doi": doi,
        }

    def download_pdf(self, pdf_url: str, paper_id: str) -> Optional[Path]:
        """Download PDF and save to data/papers/."""
        if not pdf_url:
            return None

        safe_id = re.sub(r"[^\w\-.]", "_", paper_id)
        filepath = settings.papers_dir / f"{safe_id}.pdf"

        if filepath.exists():
            return filepath

        try:
            resp = self.session.get(pdf_url, timeout=60, stream=True)
            resp.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            time.sleep(0.5)  # Be polite to arXiv servers
            return filepath
        except requests.RequestException as e:
            print(f"[ArxivClient] PDF download failed for {paper_id}: {e}")
            return None

    @staticmethod
    def _get_text(element: ET.Element, tag: str, default: str = "") -> str:
        el = element.find(tag)
        return el.text.strip() if el is not None and el.text else default
