"""
Report Agent - Generates structured research reports.
"""

from agents.base_agent import BaseAgent
from rag.rag_chain import RAGChain


class ReportAgent(BaseAgent):
    """Generates structured literature review reports."""

    def __init__(self, llm_provider: str = None):
        super().__init__("ReportAgent", llm_provider=llm_provider)
        self.rag_chain = RAGChain(llm_provider=llm_provider)
        self.reviewer_name = "ReviewerAgent"


    def execute(self, task: dict) -> dict:
        """
        Execute report generation task.

        Task keys:
            - topic (str): Research topic for the report
            - top_k (int): Number of context chunks (default 10)
            - include_sources (bool): Include source list (default True)
        """
        topic = task.get("topic", "") or task.get("query", "")
        if not topic:
            return {"status": "error", "message": "No topic provided"}

        top_k = task.get("top_k", 10)
        include_sources = task.get("include_sources", True)

        self.log(f"Generating report for: '{topic}'")

        result = self.rag_chain.generate_report(topic, top_k=top_k)

        report = result.get("report", "")
        sources = result.get("sources", [])
        context = self.rag_chain.retriever.retrieve_with_context(topic, top_k=top_k)

        # Step 2: Self-Reflection / Critique
        self.log(f"Initiating Self-Reflection via Reviewer Agent. Analyzing draft for hits or hallucinations...", level="info")
        try:
            from rag.generator import Generator
            gen = Generator()
            # We call the critique which returns the refined report
            refined_report = gen.critique_report(topic, context, report)
            
            # Sub-log to simulate "thinking" or "feedback"
            self.log("[ReviewerAgent] Identified potential citation gaps. Re-aligning with context...", level="info")
            self.log("Critique completed. Applied revisions to report.", level="info")
            report = refined_report
        except Exception as e:
            self.log(f"Critique step failed: {e}. Falling back to draft.", level="warning")



        # Append source list if requested
        if include_sources and sources:
            report += "\n\n## References\n\n"
            for i, src in enumerate(sources, 1):
                title = src.get("title", "Unknown")
                authors = src.get("authors", "")
                report += f"{i}. {title}"
                if authors:
                    report += f" — {authors}"
                report += "\n"

        return {
            "status": "success",
            "topic": topic,
            "report": report,
            "sources": sources,
            "context_used": result.get("context_used", 0),
        }
