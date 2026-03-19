"""
Retrieval Agent - Queries the vector database.
"""

from agents.base_agent import BaseAgent
from rag.retriever import Retriever
from rag.vector_store import VectorStore


class RetrievalAgent(BaseAgent):
    """Queries the vector database for relevant research content."""

    def __init__(self, llm_provider: str = None):
        super().__init__("RetrievalAgent", llm_provider=llm_provider)
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.vector_store)

    def execute(self, task: dict) -> dict:
        """
        Execute retrieval task.

        Task keys:
            - query (str): Search query
            - top_k (int): Number of results (default 5)
            - source_filter (str): Optional source filter
            - format (str): 'chunks' or 'context' (default 'chunks')
        """
        query = task.get("query", "")
        if not query:
            return {"status": "error", "message": "No query provided"}

        top_k = task.get("top_k", 5)
        source_filter = task.get("source_filter")
        output_format = task.get("format", "chunks")

        self.log(f"Retrieving for: '{query}' (top {top_k})")

        if output_format == "context":
            context = self.retriever.retrieve_with_context(query, top_k=top_k)
            return {
                "status": "success",
                "query": query,
                "context": context,
            }

        chunks = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            source_filter=source_filter,
        )

        return {
            "status": "success",
            "query": query,
            "results_count": len(chunks),
            "results": chunks,
            "db_stats": self.vector_store.get_collection_stats(),
        }
