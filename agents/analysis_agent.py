"""
Analysis Agent - Analyzes research papers and extracts insights.
"""

from agents.base_agent import BaseAgent
from rag.rag_chain import RAGChain


class AnalysisAgent(BaseAgent):
    """Summarizes research papers and extracts insights using LLM."""

    def __init__(self, llm_provider: str = None):
        super().__init__("AnalysisAgent", llm_provider=llm_provider)
        from config.settings import settings
        # In demo mode, skip RAGChain init (avoids slow ChromaDB ONNX embedding).
        # Context comes from orchestrator state, not vector retrieval.
        if getattr(settings, "demo_gemini_only", False):
            self.rag_chain = None
        else:
            self.rag_chain = RAGChain(llm_provider=llm_provider)

    def execute(self, task: dict) -> dict:
        """
        Execute analysis task.

        Task keys:
            - query (str): Topic or paper to analyze
            - context_data (list[dict]): Papers passed from ResearchAgent via orchestrator state
            - hypotheses (str): Optional hypotheses from HypothesisAgent to test against
            - analysis_type (str): 'analysis', 'summary', or 'both' (default 'analysis')
            - complexity (str): 'beginner', 'expert', or 'standard' (default 'standard')
            - top_k (int): Number of context chunks (default 5)
        """
        query = task.get("query", "")
        if not query:
            return {"status": "error", "message": "No query provided"}

        context_data = task.get("context_data", [])
        hypotheses = task.get("hypotheses", "")
        analysis_type = task.get("analysis_type", "analysis")
        complexity = task.get("complexity", "standard")
        top_k = task.get("top_k", 5)

        self.log(f"Analyzing: '{query}' (type: {analysis_type}, complexity: {complexity})")

        results = {"status": "success", "query": query, "complexity": complexity}

        from config.settings import settings
        is_demo = getattr(settings, "demo_gemini_only", False)

        # In demo mode, if context_data is missing or missing abstracts, load from cached paper metadata
        if is_demo and (not context_data or not any(p.get("abstract") for p in context_data)):
            from collectors.paper_manager import PaperManager
            cached = PaperManager().load_metadata(query)
            if cached:
                self.log(f"Loaded {len(cached)} cached papers with abstracts for demo analysis.")
                context_data = cached[:max(5, top_k)]

        # Build context from orchestrator state or cached papers
        if context_data:
            self.log(f"Using {len(context_data)} papers for analysis.")
            context_parts = []
            for paper in context_data:
                title = paper.get("title", "Unknown")
                authors = ", ".join(paper.get("authors", [])) if isinstance(paper.get("authors"), list) else paper.get("authors", "")
                abstract = paper.get("abstract", paper.get("snippet", ""))
                source = paper.get("source", "arxiv")
                published = paper.get("published", "")
                context_parts.append(
                    f"Title: {title}\nAuthors: {authors}\nPublished: {published}\nSource: {source}\nAbstract: {abstract}"
                )
            injected_context = "\n\n---\n\n".join(context_parts)

            # Append hypotheses if provided by HypothesisAgent
            if hypotheses:
                self.log("Incorporating hypotheses from HypothesisAgent.")
                injected_context += f"\n\n---\n\nHypotheses to Evaluate:\n{hypotheses}"

            # Use the Generator directly with the compact injected context (1 real Gemini API call)
            from rag.generator import Generator
            gen = Generator(llm_provider=self.llm_provider)

            if is_demo or analysis_type == "analysis":
                analysis_text = gen.generate_analysis(injected_context, complexity=complexity)
                results["analysis"] = analysis_text
                results["summary"] = analysis_text[:500] + "..." if analysis_text else ""
                results["analysis_sources"] = [p.get("title", "") for p in context_data]
            elif analysis_type == "summary":
                summary_text = gen.generate_summary(injected_context, complexity=complexity)
                results["summary"] = summary_text
                results["analysis"] = summary_text
                results["summary_sources"] = [p.get("title", "") for p in context_data]
            else:
                analysis_text = gen.generate_analysis(injected_context, complexity=complexity)
                summary_text = gen.generate_summary(injected_context, complexity=complexity)
                results["analysis"] = analysis_text
                results["summary"] = summary_text
                results["analysis_sources"] = [p.get("title", "") for p in context_data]
                results["summary_sources"] = [p.get("title", "") for p in context_data]
        else:
            # Legacy fallback: no state data, query RAG directly (non-demo mode only)
            if self.rag_chain is None:
                self.log("No context data and RAGChain disabled in demo mode. Using topic query directly.", level="warning")
                from rag.generator import Generator
                gen = Generator(llm_provider=self.llm_provider)
                analysis_text = gen.generate_analysis(f"Research Topic: {query}", complexity=complexity)
                results["analysis"] = analysis_text
                results["summary"] = analysis_text[:500] + "..." if analysis_text else ""
                results["analysis_sources"] = []
            else:
                self.log("No context_data from orchestrator. Falling back to RAG retrieval.")
                if analysis_type == "analysis":
                    analysis_result = self.rag_chain.analyze_papers(query, top_k=top_k, complexity=complexity)
                    results["analysis"] = analysis_result.get("analysis", "")
                    results["summary"] = results["analysis"][:500] + "..." if results["analysis"] else ""
                    results["analysis_sources"] = analysis_result.get("sources", [])
                elif analysis_type == "summary":
                    summary_result = self.rag_chain.generate_summary(query, top_k=top_k, complexity=complexity)
                    results["summary"] = summary_result.get("summary", "")
                    results["analysis"] = results["summary"]
                    results["summary_sources"] = summary_result.get("sources", [])
                else:
                    analysis_result = self.rag_chain.analyze_papers(query, top_k=top_k, complexity=complexity)
                    summary_result = self.rag_chain.generate_summary(query, top_k=top_k, complexity=complexity)
                    results["analysis"] = analysis_result.get("analysis", "")
                    results["summary"] = summary_result.get("summary", "")
                    results["analysis_sources"] = analysis_result.get("sources", [])
                    results["summary_sources"] = summary_result.get("sources", [])

        return results
