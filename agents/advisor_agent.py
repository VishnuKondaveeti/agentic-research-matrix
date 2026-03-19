"""
Research Advisor Agent - Identifies gaps and suggests ideas.
"""

from agents.base_agent import BaseAgent
from rag.rag_chain import RAGChain


class AdvisorAgent(BaseAgent):
    """Identifies research gaps and suggests future directions."""

    def __init__(self, llm_provider: str = None):
        super().__init__("AdvisorAgent", llm_provider=llm_provider)
        self.rag_chain = RAGChain(llm_provider=llm_provider)

    def execute(self, task: dict) -> dict:
        """
        Execute research advising task.

        Task keys:
            - topic (str): Research topic to advise on
            - top_k (int): Number of context chunks (default 10)
        """
        topic = task.get("topic", "") or task.get("query", "")
        if not topic:
            return {"status": "error", "message": "No topic provided"}

        top_k = task.get("top_k", 10)

        self.log(f"Advising on: '{topic}'")

        result = self.rag_chain.get_advice(topic, top_k=top_k)

        return {
            "status": "success",
            "topic": topic,
            "advice": result.get("advice", ""),
            "sources": result.get("sources", []),
        }
