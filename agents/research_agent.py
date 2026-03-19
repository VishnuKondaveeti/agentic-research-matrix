"""
Research Agent - Searches databases and collects papers.
"""

from agents.base_agent import BaseAgent
from collectors.paper_manager import PaperManager
from processing.pipeline import ProcessingPipeline


class ResearchAgent(BaseAgent):
    """Finds new research papers and updates the dataset."""

    def __init__(self):
        super().__init__("ResearchAgent")
        self.paper_manager = PaperManager()
        self.pipeline = ProcessingPipeline()

    def execute(self, task: dict) -> dict:
        """
        Execute research task.

        Task keys:
            - query (str): Research topic to search for
            - max_papers (int): Max papers per source (default 5)
            - sources (list): Sources to use (default all)
            - download (bool): Whether to download PDFs (default True)
            - process (bool): Whether to process downloaded PDFs (default True)
            - paper_ids (list): Optional list of titles/ids to filter by

        """
        query = task.get("query", "")
        if not query:
            return {"status": "error", "message": "No query provided"}

        max_papers = task.get("max_papers", 5)
        sources = task.get("sources", None)
        should_download = task.get("download", True)
        should_process = task.get("process", True)

        self.log(f"Searching for: '{query}' (max {max_papers} per source)")

        # Step 1: Search (Skip if we already have paper_ids and just want to process)
        paper_ids = task.get("paper_ids", [])
        filters = task.get("filters", {})
        
        papers = self.paper_manager.search_all(
            query=query,
            max_per_source=max_papers,
            sources=sources or filters.get("sources"),
            min_year=filters.get("min_year"),
            max_year=filters.get("max_year"),
        )

        if paper_ids:
            # Filter papers by the provided IDs or titles
            papers = [p for p in papers if (p.get("title") in paper_ids or p.get("id") in paper_ids)]
            self.log(f"Filtered to {len(papers)} selected papers")


        if not papers:
            return {
                "status": "success",
                "message": "No papers found",
                "papers_found": 0,
                "papers_downloaded": 0,
                "papers_processed": 0,
            }

        # Step 2: Save metadata
        self.paper_manager.save_metadata(papers, query)

        # Step 3: Download PDFs
        papers_downloaded = 0
        if should_download:
            papers = self.paper_manager.download_papers(papers, max_downloads=max_papers)
            papers_downloaded = sum(1 for p in papers if p.get("local_pdf"))
            self.log(f"Downloaded {papers_downloaded} PDFs")

        # Step 4: Process into vector DB
        papers_processed = 0
        processing_results = []
        if should_process and papers_downloaded > 0:
            processing_results = self.pipeline.process_batch(papers)
            papers_processed = sum(
                1 for r in processing_results if r.get("status") == "success"
            )
            self.log(f"Processed {papers_processed} papers into vector DB")

        # Step 5: Calculate influence scores & sort
        from analytics.trend_detector import TrendDetector
        td = TrendDetector()
        
        scored_papers = []
        for p in papers:
            p["influence_score"] = td.calculate_influence_score(p)
            scored_papers.append(p)
            
        # Sort by score descending
        scored_papers.sort(key=lambda x: x.get("influence_score", 0), reverse=True)

        return {
            "status": "success",
            "query": query,
            "papers_found": len(papers),
            "papers_downloaded": papers_downloaded,
            "papers_processed": papers_processed,
            "papers": [
                {
                    "title": p.get("title", ""),
                    "authors": p.get("authors", []),
                    "source": p.get("source", ""),
                    "published": p.get("published", ""),
                    "has_pdf": bool(p.get("local_pdf")),
                    "url": p.get("pdf_url") or (f"https://doi.org/{p.get('doi')}" if p.get("doi") else ""),
                    "influence_score": p.get("influence_score", 60)
                }
                for p in scored_papers
            ],
            "processing_results": processing_results,
        }
