"""
Report Agent - Generates structured research reports.
"""

from agents.base_agent import BaseAgent
from rag.rag_chain import RAGChain


class ReportAgent(BaseAgent):
    """Generates structured literature review reports."""

    def __init__(self, llm_provider: str = None):
        super().__init__("ReportAgent", llm_provider=llm_provider)
        from config.settings import settings
        # In demo mode, skip RAGChain init (avoids slow ChromaDB ONNX embedding).
        if getattr(settings, "demo_gemini_only", False):
            self.rag_chain = None
        else:
            self.rag_chain = RAGChain(llm_provider=llm_provider)
        self.reviewer_name = "ReviewerAgent"


    def execute(self, task: dict) -> dict:
        """
        Execute report generation task.

        Task keys:
            - topic (str): Research topic for the report
            - analysis (str): Analysis text from AnalysisAgent (passed via orchestrator state)
            - debate (str): Debate text from DebateAgent (passed via orchestrator state)
            - revision_notes (str): Feedback from EvaluationAgent for reflection loops
            - top_k (int): Number of context chunks (default 10)
            - include_sources (bool): Include source list (default True)
        """
        topic = task.get("topic", "") or task.get("query", "")
        if not topic:
            return {"status": "error", "message": "No topic provided"}

        analysis = task.get("analysis", "")
        debate = task.get("debate", "")
        revision_notes = task.get("revision_notes", "")
        top_k = task.get("top_k", 10)
        include_sources = task.get("include_sources", True)

        self.log(f"Generating report for: '{topic}'")

        # Build enriched context from upstream agents
        enriched_context_parts = []

        if analysis:
            self.log("Incorporating analysis from AnalysisAgent.")
            enriched_context_parts.append(f"## Analysis Context\n{analysis}")

        if debate:
            self.log("Incorporating debate perspectives from DebateAgent.")
            enriched_context_parts.append(f"## Debate Context\n{debate}")

        if revision_notes:
            self.log(f"Applying revision notes from EvaluationAgent: {revision_notes[:100]}...")
            enriched_context_parts.append(
                f"## Revision Instructions\nThe previous draft was rejected. You MUST address these issues:\n{revision_notes}"
            )

        # If we have upstream context, inject it alongside RAG context
        if enriched_context_parts:
            rag_context = ""
            from config.settings import settings
            if not getattr(settings, "demo_gemini_only", False):
                rag_context = self.rag_chain.retriever.retrieve_with_context(topic, top_k=top_k)
            full_context = (rag_context + "\n\n" if rag_context else "") + "\n\n".join(enriched_context_parts)

            from rag.generator import Generator
            gen = Generator(llm_provider=self.llm_provider)
            report = gen.generate_report(topic, full_context)
            sources = task.get("evidence") or task.get("sources") or []
            if not sources:
                chunks = self.rag_chain.retriever.retrieve(topic, top_k=top_k)
                seen = set()
                for c in chunks:
                    t = c.get("metadata", {}).get("title", "Unknown")
                    if t not in seen:
                        sources.append({
                            "title": t,
                            "authors": c.get("metadata", {}).get("authors", ""),
                            "source": c.get("metadata", {}).get("source", ""),
                        })
                        seen.add(t)
        else:
            # Legacy fallback: no upstream data
            from config.settings import settings
            if getattr(settings, "demo_gemini_only", False):
                # In demo mode, generate report directly without RAGChain
                self.log("No upstream context in demo mode. Generating report with minimal context.")
                from rag.generator import Generator
                gen = Generator(llm_provider=self.llm_provider)
                report = gen.generate_report(topic, f"Generate a comprehensive literature review on: {topic}")
                sources = []
            else:
                self.log("No upstream context. Falling back to pure RAG report generation.")
                result = self.rag_chain.generate_report(topic, top_k=top_k)
                report = result.get("report", "")
                sources = result.get("sources", [])

        # Self-Reflection / Critique (only on first draft, skip during revision loops or in demo mode)
        from config.settings import settings
        if not revision_notes and not getattr(settings, "demo_gemini_only", False):
            self.log("Initiating Self-Reflection via Reviewer Agent.", level="info")
            try:
                from rag.generator import Generator
                gen = Generator(llm_provider=self.llm_provider)
                context = self.rag_chain.retriever.retrieve_with_context(topic, top_k=top_k)
                refined_report = gen.critique_report(topic, context, report)
                self.log("[ReviewerAgent] Critique completed. Applied revisions.", level="info")
                report = refined_report
            except Exception as e:
                self.log(f"Critique step failed: {e}. Falling back to draft.", level="warning")
        elif getattr(settings, "demo_gemini_only", False):
            self.log("Reviewer evaluation unified with downstream EvaluationAgent for demo call budget optimization.", level="info")

        # Append source list if requested
        if include_sources and sources:
            report += "\n\n## References\n\n"
            for i, src in enumerate(sources, 1):
                title = src.get("title", "Unknown")
                authors = src.get("authors", "")
                report += f"{i}. {title}"
                if authors:
                    report += f" - {authors}"
                report += "\n"

        return {
            "status": "success",
            "topic": topic,
            "report": report,
            "sources": sources,
        }
