"""
arXiv API client for fetching research papers.

Uses the arXiv Atom feed API (no API key required).

The search implementation converts broad natural-language
research questions into focused arXiv title/abstract queries
to improve retrieval relevance.
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
        self.session.headers.update({
            "User-Agent": "ResearchAutomation/1.0"
        })

    def search(
        self,
        query: str,
        max_results: int = 10,
        start: int = 0,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> list[dict]:
        """
        Search arXiv using focused title/abstract queries.

        For known RAG/LLM research topics, important concepts are
        extracted from the natural-language query instead of sending
        the entire sentence to arXiv's `all:` search field.

        Returns a list of paper metadata dictionaries.
        """

        q = query.lower().strip()

        # ---------------------------------------------------------
        # RAG / Retrieval concepts
        # ---------------------------------------------------------
        rag_terms = []

        if (
            "retrieval-augmented generation" in q
            or re.search(r"\brag\b", q)
        ):
            rag_terms.extend([
                'ti:"retrieval-augmented generation"',
                'abs:"retrieval-augmented generation"',
            ])

        if "graphrag" in q or "graph rag" in q:
            rag_terms.extend([
                'ti:"GraphRAG"',
                'abs:"GraphRAG"',
            ])

        if "agentic rag" in q or "agentic retrieval" in q:
            rag_terms.extend([
                'ti:"Agentic RAG"',
                'abs:"Agentic RAG"',
            ])

        # ---------------------------------------------------------
        # Question-answering concepts
        # ---------------------------------------------------------
        qa_terms = []

        if "question answering" in q or "question-answering" in q:
            qa_terms.extend([
                'ti:"question answering"',
                'abs:"question answering"',
            ])

        if "multi-hop" in q or "multihop" in q:
            qa_terms.extend([
                'ti:"multi-hop"',
                'abs:"multi-hop"',
            ])

        # ---------------------------------------------------------
        # General research concepts
        # ---------------------------------------------------------
        evaluation_terms = []

        if "factuality" in q:
            evaluation_terms.append(
                'abs:"factuality"'
            )

        if "retrieval accuracy" in q:
            evaluation_terms.append(
                'abs:"retrieval accuracy"'
            )

        if "scalability" in q:
            evaluation_terms.append(
                'abs:"scalability"'
            )

        if "latency" in q:
            evaluation_terms.append(
                'abs:"latency"'
            )

        # ---------------------------------------------------------
        # Construct focused query
        # ---------------------------------------------------------

        if rag_terms and qa_terms:

            # Core requirement:
            # The paper should discuss BOTH retrieval/RAG
            # AND question answering/multi-hop reasoning.

            search_query = (
                "("
                + " OR ".join(rag_terms)
                + ")"
                + " AND "
                + "("
                + " OR ".join(qa_terms)
                + ")"
            )

        elif rag_terms:

            search_query = (
                "("
                + " OR ".join(rag_terms)
                + ")"
            )

        elif qa_terms:

            search_query = (
                "("
                + " OR ".join(qa_terms)
                + ")"
            )

        else:
            # -----------------------------------------------------
            # Generic fallback
            # -----------------------------------------------------
            #
            # For topics that aren't specifically RAG/QA topics,
            # use the original natural-language query.
            #
            safe_query = query.strip().replace('"', '\\"')

            search_query = f'all:"{safe_query}"'

        # ---------------------------------------------------------
        # Optional evaluation terms
        # ---------------------------------------------------------
        #
        # We deliberately don't add these to the main AND query.
        # Terms such as "latency" and "scalability" are too broad
        # and can cause irrelevant results.
        #
        # They are printed for debugging and can later be used by
        # a semantic reranking stage.
        # ---------------------------------------------------------

        if evaluation_terms:
            print(
                "[ArxivClient] Secondary evaluation concepts: "
                + ", ".join(evaluation_terms)
            )

        params = {
            "search_query": search_query,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }

        print(f"[ArxivClient] Query: {search_query}")

        try:
            resp = self.session.get(
                ARXIV_API_URL,
                params=params,
                timeout=30,
            )

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
            print("[ArxivClient] Failed to parse arXiv XML response.")
            return []

        for entry in root.findall(f"{ATOM_NS}entry"):

            paper = self._parse_entry(entry)

            if paper:
                papers.append(paper)

        return papers

    def _parse_entry(self, entry: ET.Element) -> Optional[dict]:
        """Parse a single Atom entry into a paper dictionary."""

        # ---------------------------------------------------------
        # Title
        # ---------------------------------------------------------

        title_el = entry.find(f"{ATOM_NS}title")

        if title_el is None or not title_el.text:
            return None

        title = re.sub(
            r"\s+",
            " ",
            title_el.text.strip()
        )

        # ---------------------------------------------------------
        # arXiv ID
        # ---------------------------------------------------------

        id_el = entry.find(f"{ATOM_NS}id")

        arxiv_id = ""

        if id_el is not None and id_el.text:

            arxiv_id = id_el.text.split("/abs/")[-1]

        # ---------------------------------------------------------
        # Authors
        # ---------------------------------------------------------

        authors = []

        for author_el in entry.findall(
            f"{ATOM_NS}author"
        ):

            name_el = author_el.find(
                f"{ATOM_NS}name"
            )

            if name_el is not None and name_el.text:

                authors.append(
                    name_el.text.strip()
                )

        # ---------------------------------------------------------
        # Abstract
        # ---------------------------------------------------------

        summary_el = entry.find(
            f"{ATOM_NS}summary"
        )

        abstract = ""

        if summary_el is not None and summary_el.text:

            abstract = re.sub(
                r"\s+",
                " ",
                summary_el.text.strip()
            )

        # ---------------------------------------------------------
        # Dates
        # ---------------------------------------------------------

        published = self._get_text(
            entry,
            f"{ATOM_NS}published",
            ""
        )

        updated = self._get_text(
            entry,
            f"{ATOM_NS}updated",
            ""
        )

        # ---------------------------------------------------------
        # Categories
        # ---------------------------------------------------------

        categories = []

        for cat_el in entry.findall(
            f"{ARXIV_NS}primary_category"
        ):

            term = cat_el.get(
                "term",
                ""
            )

            if term:
                categories.append(term)

        for cat_el in entry.findall(
            f"{ATOM_NS}category"
        ):

            term = cat_el.get(
                "term",
                ""
            )

            if (
                term
                and term not in categories
            ):
                categories.append(term)

        # ---------------------------------------------------------
        # PDF link
        # ---------------------------------------------------------

        pdf_url = ""

        for link_el in entry.findall(
            f"{ATOM_NS}link"
        ):

            if link_el.get("title") == "pdf":

                pdf_url = link_el.get(
                    "href",
                    ""
                )

                break

        # ---------------------------------------------------------
        # DOI
        # ---------------------------------------------------------

        doi_el = entry.find(
            f"{ARXIV_NS}doi"
        )

        doi = (
            doi_el.text.strip()
            if doi_el is not None
            and doi_el.text
            else ""
        )

        # ---------------------------------------------------------
        # Final paper object
        # ---------------------------------------------------------

        return {
            "source": "arxiv",
            "arxiv_id": arxiv_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "published": (
                published[:10]
                if published
                else ""
            ),
            "updated": (
                updated[:10]
                if updated
                else ""
            ),
            "categories": categories,
            "pdf_url": pdf_url,
            "doi": doi,
        }

    def download_pdf(
        self,
        pdf_url: str,
        paper_id: str
    ) -> Optional[Path]:
        """Download PDF and save to data/papers/."""

        if not pdf_url:
            return None

        safe_id = re.sub(
            r"[^\w\-.]",
            "_",
            paper_id
        )

        filepath = (
            settings.papers_dir
            / f"{safe_id}.pdf"
        )

        if filepath.exists():
            return filepath

        try:

            resp = self.session.get(
                pdf_url,
                timeout=60,
                stream=True
            )

            resp.raise_for_status()

            with open(
                filepath,
                "wb"
            ) as f:

                for chunk in resp.iter_content(
                    chunk_size=8192
                ):
                    f.write(chunk)

            time.sleep(0.5)

            return filepath

        except requests.RequestException as e:

            print(
                f"[ArxivClient] "
                f"PDF download failed for "
                f"{paper_id}: {e}"
            )

            return None

    @staticmethod
    def _get_text(
        element: ET.Element,
        tag: str,
        default: str = ""
    ) -> str:

        el = element.find(tag)

        return (
            el.text.strip()
            if el is not None
            and el.text
            else default
        )