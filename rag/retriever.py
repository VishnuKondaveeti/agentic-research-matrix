"""
Retriever for semantic search over the vector store.
Retrieves relevant document chunks based on user queries.
"""

from rag.vector_store import VectorStore


class Retriever:
    """Retrieves relevant document chunks from the vector store."""

    def __init__(self, vector_store: VectorStore | None = None):
        self.vector_store = vector_store or VectorStore()

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: str | None = None,
        min_relevance: float = 0.0,
    ) -> list[dict]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: User query string.
            top_k: Number of chunks to retrieve.
            source_filter: Optional filter by source (e.g., 'arxiv').
            min_relevance: Minimum relevance score (0-1, where 1 is exact match).

        Returns:
            List of chunk dicts with 'text', 'metadata', 'score', 'id'.
        """
        where = None
        if source_filter:
            where = {"source": source_filter}

        results = self.vector_store.search(
            query=query,
            n_results=top_k,
            where=where,
        )

        # Convert distance to similarity score (cosine distance)
        enriched = []
        for r in results:
            score = max(0.0, 1.0 - r["distance"])
            if score >= min_relevance:
                enriched.append({
                    "text": r["text"],
                    "metadata": r["metadata"],
                    "score": round(score, 4),
                    "id": r["id"],
                })

        return enriched

    def retrieve_with_context(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """
        Retrieve chunks and format them as a context string for LLM.

        Returns formatted context string.
        """
        chunks = self.retrieve(query, top_k=top_k)

        if not chunks:
            return "No relevant research documents found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk["metadata"].get("title", "Unknown")
            authors = chunk["metadata"].get("authors", "Unknown")
            source = chunk["metadata"].get("source", "")
            score = chunk["score"]

            context_parts.append(
                f"--- Source {i} (Relevance: {score:.2f}) ---\n"
                f"Title: {title}\n"
                f"Authors: {authors}\n"
                f"Source: {source}\n\n"
                f"{chunk['text']}\n"
            )

        return "\n\n".join(context_parts)
