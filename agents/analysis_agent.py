"""
Analysis Agent - Analyzes research papers and extracts insights.
"""

from agents.base_agent import BaseAgent
from rag.rag_chain import RAGChain


class AnalysisAgent(BaseAgent):
    """Summarizes research papers and extracts insights using LLM."""

    def __init__(self, llm_provider: str = None):
        super().__init__("AnalysisAgent", llm_provider=llm_provider)
        self.rag_chain = RAGChain(llm_provider=llm_provider)

    def execute(self, task: dict) -> dict:
        """
        Execute analysis task.

        Task keys:
            - query (str): Topic or paper to analyze
            - analysis_type (str): 'analysis', 'summary', or 'both' (default 'analysis')
            - complexity (str): 'beginner', 'expert', or 'standard' (default 'standard')
            - top_k (int): Number of context chunks (default 5)
        """
        query = task.get("query", "")
        if not query:
            return {"status": "error", "message": "No query provided"}

        analysis_type = task.get("analysis_type", "analysis")
        complexity = task.get("complexity", "standard")
        top_k = task.get("top_k", 5)

        self.log(f"Analyzing: '{query}' (type: {analysis_type}, complexity: {complexity})")

        results = {"status": "success", "query": query, "complexity": complexity}

        if analysis_type in ("analysis", "both"):
            analysis_result = self.rag_chain.analyze_papers(query, top_k=top_k, complexity=complexity)
            results["analysis"] = analysis_result.get("analysis", "")
            results["analysis_sources"] = analysis_result.get("sources", [])

        if analysis_type in ("summary", "both"):
            summary_result = self.rag_chain.generate_summary(query, top_k=top_k, complexity=complexity)
            results["summary"] = summary_result.get("summary", "")
            results["summary_sources"] = summary_result.get("sources", [])

        return results
