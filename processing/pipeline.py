"""
End-to-end document processing pipeline.
Orchestrates: extract → clean → chunk → store in vector DB.
"""

from pathlib import Path

from processing.pdf_extractor import PDFExtractor
from processing.text_cleaner import TextCleaner
from processing.chunker import DocumentChunker
from rag.vector_store import VectorStore


class ProcessingPipeline:
    """End-to-end document processing: PDF → chunks → vector store."""

    def __init__(self):
        self.extractor = PDFExtractor()
        self.cleaner = TextCleaner()
        self.chunker = DocumentChunker()
        self.vector_store = VectorStore()

    def process_paper(
        self,
        pdf_path: str | Path,
        paper_metadata: dict | None = None,
    ) -> dict:
        """
        Process a single paper: extract text, clean, chunk, and store.

        Args:
            pdf_path: Path to PDF file.
            paper_metadata: Optional metadata (title, authors, etc.)

        Returns:
            Dict with processing results.
        """
        pdf_path = Path(pdf_path)
        paper_metadata = paper_metadata or {}

        # Step 1: Extract text
        extraction = self.extractor.extract(pdf_path)
        raw_text = extraction["text"]

        if not raw_text.strip():
            return {
                "status": "error",
                "message": f"No text extracted from {pdf_path.name}",
                "chunks_created": 0,
            }

        # Step 2: Clean text
        cleaned_text = self.cleaner.clean(raw_text)

        # Step 3: Chunk
        chunk_metadata = {
            "source_file": pdf_path.name,
            "title": paper_metadata.get("title", extraction["metadata"].get("title", "")),
            "authors": ", ".join(paper_metadata.get("authors", [])) if isinstance(paper_metadata.get("authors"), list) else paper_metadata.get("authors", ""),
            "source": paper_metadata.get("source", ""),
            "published": paper_metadata.get("published", ""),
            "paper_id": paper_metadata.get("arxiv_id", "") or paper_metadata.get("s2_id", "") or paper_metadata.get("core_id", ""),
        }

        chunks = self.chunker.chunk(cleaned_text, metadata=chunk_metadata)

        if not chunks:
            return {
                "status": "error",
                "message": "No chunks produced",
                "chunks_created": 0,
            }

        # Step 4: Store in vector DB
        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [
            f"{chunk_metadata.get('paper_id', pdf_path.stem)}_{c['chunk_index']}"
            for c in chunks
        ]

        self.vector_store.add_documents(texts=texts, metadatas=metadatas, ids=ids)

        return {
            "status": "success",
            "source_file": pdf_path.name,
            "pages_extracted": extraction["metadata"]["page_count"],
            "raw_chars": len(raw_text),
            "cleaned_chars": len(cleaned_text),
            "chunks_created": len(chunks),
        }

    def process_batch(
        self,
        papers: list[dict],
    ) -> list[dict]:
        """
        Process a batch of papers. Each paper dict should have:
        - 'local_pdf': path to downloaded PDF
        - other metadata fields

        Returns list of processing results.
        """
        results = []
        for paper in papers:
            pdf_path = paper.get("local_pdf", "")
            if not pdf_path or not Path(pdf_path).exists():
                results.append({
                    "status": "skipped",
                    "title": paper.get("title", "Unknown"),
                    "reason": "No local PDF available",
                })
                continue

            try:
                result = self.process_paper(pdf_path, paper_metadata=paper)
                result["title"] = paper.get("title", "Unknown")
                results.append(result)
            except Exception as e:
                results.append({
                    "status": "error",
                    "title": paper.get("title", "Unknown"),
                    "message": str(e),
                })

        return results
