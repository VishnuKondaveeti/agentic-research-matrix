"""
DebateAgent: Performs a dialectic review of research findings (Skeptic vs Optimist).
"""

from agents.base_agent import BaseAgent
from rag.generator import Generator

DEBATE_PROMPT = """
You are an ELITE Peer Reviewer and Scientific Debater. Your job is to analyze the provided research context and present a DIALECTIC DEBATE between a "Skeptic" and an "Optimist".

### Context:
Topic: {query}
Research Chunks:
{context}

### Instructions:
1. **The Optimist**: Focuses on the breakthroughs, potential impact, and validity of the results. Highlights why this approach is superior to previous ones.
2. **The Skeptic**: Focuses on the limitations, potential biases, dataset constraints, and reproducibility concerns. Questions if the results are generalized or overfitted.
3. **Synthesis**: Provide a final balanced view on where the community should go next based on this debate.

Format:
# Scientific Debate: {query}

## 🌟 The Optimist's Perspective
...

## 🔍 The Skeptic's Critique
...

## ⚖️ Balanced Synthesis & Future Directions
...
"""

class DebateAgent(BaseAgent):
    """Generates contrasting perspectives on research findings."""

    def __init__(self, llm_provider: str = None):
        super().__init__("DebateAgent", llm_provider=llm_provider)
        self.generator = Generator(llm_provider=llm_provider)

    def execute(self, task: dict) -> dict:
        """
        Perform a debate evaluation for a given topic/paper.
        """
        query = task.get("query", "")
        if not query:
            return {"status": "error", "message": "No query provided"}

        from rag.retriever import Retriever
        retriever = Retriever()
        
        self.log(f"Retrieving multi-perspective context for: {query}")
        # Search for results/limitations specifically
        context = retriever.retrieve_with_context(f"results, limitations, methodology of {query}", top_k=6)
        
        self.log(f"Generating debate for: {query}")
        prompt = DEBATE_PROMPT.format(
            query=query,
            context=context
        )
        
        try:
            debate = self.generator.llm.invoke(prompt).content
            return {
                "status": "success",
                "debate": debate
            }
        except Exception as e:
            self.log(f"Debate generation failed: {e}", level="error")
            return {
                "status": "error",
                "message": str(e)
            }
