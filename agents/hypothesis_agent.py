"""
HypothesisAgent: Generates novel research hypotheses by finding cross-domain connections.
"""

from agents.base_agent import BaseAgent
from rag.generator import Generator

HYPOTHESIS_PROMPT = """
You are an ELITE Research Scientist and Visionary. Your job is to analyze the provided research context and generate 3 NOVEL, high-impact research hypotheses.

### Context:
Topic: {query}
Key Research Insights:
{context}

### Instructions:
1. **Cross-Domain Synthesis**: Look for patterns where a technique from one paper could solve a problem mentioned in another.
2. **The "What If" Factor**: Propose "What if we applied X to Y?" scenarios.
3. **Feasibility & Impact**: Briefly explain why each hypothesis is worth pursuing and what its potential impact could be.
4. **Research Gaps**: Explicitly name the gap being filled.

Format:
# Novel Research Hypotheses: {query}

## 🧪 Hypothesis 1: [Short Descriptive Title]
**Concept**: ...
**Technical Gap**: ...
**Potential Impact**: ...

## 🧪 Hypothesis 2: [Short Descriptive Title]
...

## 🧪 Hypothesis 3: [Short Descriptive Title]
...
"""

class HypothesisAgent(BaseAgent):
    """Generates speculative and innovative research hypotheses."""

    def __init__(self, llm_provider: str = None):
        super().__init__("HypothesisAgent", llm_provider=llm_provider)
        self.generator = Generator(llm_provider=llm_provider)

    def execute(self, task: dict) -> dict:
        """
        Generate hypotheses for a given topic/context.
        """
        query = task.get("query", "")
        if not query:
            return {"status": "error", "message": "No query provided"}

        from rag.retriever import Retriever
        retriever = Retriever()
        
        self.log(f"Retrieving diverse context for hypothesis generation: {query}")
        # Retrieve more chunks for broader synthesis
        context = retriever.retrieve_with_context(f"innovations, future work, limitations of {query}", top_k=8)
        
        self.log(f"Generating hypotheses for: {query}")
        prompt = HYPOTHESIS_PROMPT.format(
            query=query,
            context=context
        )
        
        try:
            hypotheses = self.generator.llm.invoke(prompt).content
            return {
                "status": "success",
                "hypotheses": hypotheses
            }
        except Exception as e:
            self.log(f"Hypothesis generation failed: {e}", level="error")
            return {
                "status": "error",
                "message": str(e)
            }
