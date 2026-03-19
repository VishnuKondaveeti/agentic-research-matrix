"""
BlueprintAgent: Translates research methodologies into implementation skeletons.
"""

from agents.base_agent import BaseAgent
from rag.generator import Generator

BLUEPRINT_PROMPT = """
You are an ELITE Software Architect and Research Engineer. Your job is to translate the research methodology from a paper into a high-level Python implementation skeleton.

### Context:
Topic: {query}
Methodology Chunks:
{context}

### Instructions:
1. Identify the core algorithm, model architecture, or processing pipeline described.
2. Generate a clean, modular Python skeleton (classes, functions, docstrings) that represents the methodology.
3. Use placeholders for complex logic but define the API clearly.
4. Include comments explaining how each part maps to the paper's methodology.
5. Provide a 'Requirements' section for necessary libraries.

Format:
# Implementation Blueprint: {title}

```python
# ... code ...
```

## Methodology Mapping
- [Component X]: Maps to section Y in the paper...
"""

class BlueprintAgent(BaseAgent):
    """Generates implementation blueprints from methodologies."""

    def __init__(self, llm_provider: str = None):
        super().__init__("BlueprintAgent", llm_provider=llm_provider)
        self.generator = Generator(llm_provider=llm_provider)

    def execute(self, task: dict) -> dict:
        """
        Generate a blueprint for a given topic/paper.
        """
        query = task.get("query", "")
        if not query:
            return {"status": "error", "message": "No query provided"}

        from rag.retriever import Retriever
        retriever = Retriever()
        
        # Retrieve methodology-specific chunks if possible, or just top chunks
        self.log(f"Retrieving methodology context for: {query}")
        context = retriever.retrieve_with_context(f"methodology, algorithms, architecture of {query}", top_k=5)
        
        self.log(f"Generating blueprint for: {query}")
        prompt = BLUEPRINT_PROMPT.format(
            query=query,
            context=context,
            title=query
        )
        
        try:
            blueprint = self.generator.llm.invoke(prompt).content
            return {
                "status": "success",
                "blueprint": blueprint
            }
        except Exception as e:
            self.log(f"Blueprint generation failed: {e}", level="error")
            return {
                "status": "error",
                "message": str(e)
            }
