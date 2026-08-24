"""
IntentAgent: Parses natural language queries into structured search filters.
"""

import json
from typing import Any
from agents.base_agent import BaseAgent
from rag.generator import Generator

INTENT_PROMPT = """
You are an ELITE Research Intent Parser. Your job is to take a natural language research query and extract structured search filters.

### Query:
"{query}"

### Instructions:
Extract the following fields in JSON format:
1. `topic`: The core research topic (e.g. "CNN for medical imaging").
2. `min_year`: Minimum publication year (integer or null).
3. `max_year`: Maximum publication year (integer or null).
4. `sources`: List of preferred sources (arxiv, semantic_scholar, core). Default to null if not specified.
5. `focus`: Specific focus or methodology (e.g. "attention mechanisms", "reproducibility").
6. `intent_type`: One of: "search" (generic search), "deep_dive" (intensive investigation), "code_lookup" (looking for implementations), "survey" (broad overview).

### Example:
Query: "papers using CNN for medical imaging after 2020 on arxiv"
Result:
{{
  "topic": "CNN for medical imaging",
  "min_year": 2021,
  "max_year": null,
  "sources": ["arxiv"],
  "focus": "CNN",
  "intent_type": "search"
}}

Return ONLY the raw JSON.
"""

class IntentAgent(BaseAgent):
    """Parses research intent from natural language."""

    def __init__(self, llm_provider: str = None):
        super().__init__("IntentAgent", llm_provider=llm_provider)
        self.generator = Generator(llm_provider=llm_provider)

    def execute(self, task: dict) -> dict:
        """
        Extract intent from a query string.
        """
        query = task.get("query", "")
        if not query:
            return {"status": "error", "message": "No query provided"}

        from config.settings import settings, CallBudgetTracker

        # Deterministic extraction for demo mode to conserve API calls
        if getattr(settings, "demo_gemini_only", False):
            self.log(f"Deterministic intent extracted for demo: '{query}'", level="info")
            return {
                "status": "success",
                "filters": {
                    "topic": query,
                    "min_year": None,
                    "max_year": None,
                    "sources": ["arxiv"],
                    "focus": "methodology & architecture",
                    "intent_type": "deep_dive",
                }
            }

        prompt = INTENT_PROMPT.format(query=query)
        
        try:
            CallBudgetTracker.record_call("IntentAgent")
            raw_res = self.generator.llm.invoke(prompt).content
            if isinstance(raw_res, list):
                response = "".join([c.get("text", str(c)) if isinstance(c, dict) else str(c) for c in raw_res])
            else:
                response = str(raw_res)
            
            # Clean response if LLM adds markdown blocks
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                response = response.split("```")[1].strip()
            
            structured_data = json.loads(response.strip())
            
            self.log(f"Parsed intent for: {query}")
            return {
                "status": "success",
                "filters": structured_data
            }
        except Exception as e:
            self.log(f"Failed to parse intent: {e}", level="error")
            return {
                "status": "error",
                "message": f"Intent parsing failed: {str(e)}",
                "filters": {
                    "topic": query,
                    "min_year": None,
                    "max_year": None,
                    "sources": None,
                    "focus": None,
                    "intent_type": "search"
                }
            }
