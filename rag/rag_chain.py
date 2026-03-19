"""
End-to-end RAG chain.
Query → Retrieve → Format Context → Generate Answer.
"""

from rag.vector_store import VectorStore
from rag.retriever import Retriever
from rag.generator import Generator


class RAGChain:
    """Full RAG pipeline: retrieval + generation."""

    def __init__(self, llm_provider: str = None):
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.vector_store)
        self.llm_provider = llm_provider
        self._generator = None  # Lazy init (needs API key)

    @staticmethod
    def _format_error(e: Exception) -> str:
        err_msg = str(e)
        if "429" in err_msg or "RESOURCE_EXHAUSTED" in err_msg:
            return "Gemini API Quota Exceeded. You have hit the free tier rate limit. Please wait a minute or try again later."
        return err_msg

    @property
    def generator(self) -> Generator:
        if self._generator is None:
            self._generator = Generator(llm_provider=self.llm_provider)
        return self._generator

    def query(self, question: str, top_k: int = 5, complexity: str = "standard") -> dict:
        """
        Full RAG query: retrieve context and generate answer.

        Args:
            question: User question.
            top_k: Number of context chunks to retrieve.
            complexity: Complexity level ('beginner', 'expert', 'standard').

        Returns:
            Dict with 'answer', 'sources', 'context_used'.
        """
        # Retrieve relevant chunks
        chunks = self.retriever.retrieve(question, top_k=top_k)
        context = self.retriever.retrieve_with_context(question, top_k=top_k)

        if not chunks:
            return {
                "answer": "No relevant research documents found in the knowledge base. Please ensure papers have been collected and processed.",
                "sources": [],
                "context_used": 0,
            }

        # Generate answer
        try:
            answer = self.generator.generate_answer(context, question, complexity=complexity)
        except Exception as e:
            return {
                "answer": f"Error generating response: {self._format_error(e)}",
                "sources": [c["metadata"] for c in chunks],
                "context_used": len(chunks),
            }

        # Format sources
        sources = []
        seen_titles = set()
        for c in chunks:
            title = c["metadata"].get("title", "Unknown")
            if title not in seen_titles:
                sources.append({
                    "title": title,
                    "authors": c["metadata"].get("authors", ""),
                    "source": c["metadata"].get("source", ""),
                    "score": c["score"],
                })
                seen_titles.add(title)

        return {
            "answer": answer,
            "sources": sources,
            "context_used": len(chunks),
        }

    def generate_report(self, topic: str, top_k: int = 10) -> dict:
        """
        Generate a structured research report on a topic.

        Returns dict with 'report', 'sources', 'context_used'.
        """
        context = self.retriever.retrieve_with_context(topic, top_k=top_k)
        chunks = self.retriever.retrieve(topic, top_k=top_k)

        if not chunks:
            return {
                "report": f"No research documents found for topic: {topic}",
                "sources": [],
                "context_used": 0,
            }

        try:
            report = self.generator.generate_report(topic, context)
        except Exception as e:
            return {
                "report": f"Error generating report: {self._format_error(e)}",
                "sources": [],
                "context_used": len(chunks),
            }

        sources = []
        seen = set()
        for c in chunks:
            t = c["metadata"].get("title", "Unknown")
            if t not in seen:
                sources.append({
                    "title": t,
                    "authors": c["metadata"].get("authors", ""),
                    "source": c["metadata"].get("source", ""),
                })
                seen.add(t)

        return {
            "report": report,
            "sources": sources,
            "context_used": len(chunks),
        }

    def generate_summary(self, query: str, top_k: int = 5, complexity: str = "standard") -> dict:
        """Generate a summary of papers related to query."""
        context = self.retriever.retrieve_with_context(query, top_k=top_k)
        chunks = self.retriever.retrieve(query, top_k=top_k)

        if not chunks:
            return {"summary": "No relevant documents found.", "sources": []}

        try:
            summary = self.generator.generate_summary(context, complexity=complexity)
        except Exception as e:
            return {"summary": f"Error: {self._format_error(e)}", "sources": []}

        return {
            "summary": summary,
            "sources": [c["metadata"].get("title", "") for c in chunks],
        }

    def get_advice(self, topic: str, top_k: int = 10) -> dict:
        """Get research advice and gap analysis."""
        context = self.retriever.retrieve_with_context(topic, top_k=top_k)
        chunks = self.retriever.retrieve(topic, top_k=top_k)

        if not chunks:
            return {"advice": "Need more research data to provide advice.", "sources": []}

        try:
            advice = self.generator.generate_advice(topic, context)
        except Exception as e:
            return {"advice": f"Error: {self._format_error(e)}", "sources": []}

        return {
            "advice": advice,
            "sources": [c["metadata"].get("title", "") for c in chunks],
        }

    def analyze_papers(self, query: str, top_k: int = 5, complexity: str = "standard") -> dict:
        """Analyze papers related to a query."""
        context = self.retriever.retrieve_with_context(query, top_k=top_k)
        chunks = self.retriever.retrieve(query, top_k=top_k)

        if not chunks:
            return {"analysis": "No papers found to analyze.", "sources": []}

        try:
            analysis = self.generator.generate_analysis(context, complexity=complexity)
        except Exception as e:
            return {"analysis": f"Error: {self._format_error(e)}", "sources": []}

        return {
            "analysis": analysis,
            "sources": [c["metadata"].get("title", "") for c in chunks],
        }
