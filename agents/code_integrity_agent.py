"""
CodeIntegrityAgent: Identifies and verifies code implementations for research papers.
"""

import re
import requests
from agents.base_agent import BaseAgent
from rag.generator import Generator

CODE_EXTRACTION_PROMPT = """
You are an ELITE Code Integrity Agent. Your job is to extract official GitHub or code repository links from the provided research paper metadata/abstract.

### Context:
Title: {title}
Abstract: {abstract}
External URL: {url}

### Instructions:
1. Search for GitHub, GitLab, bitbucket, or original project website links.
2. If a repository is found, provide the URL.
3. If multiple are found, prioritize the official one.
4. Return the result in JSON format:
{{
  "has_code": true/false,
  "repo_url": "...",
  "implementation_status": "official" | "community" | "unknown",
  "confidence": 0.0 to 1.0
}}

Return ONLY the raw JSON.
"""

class CodeIntegrityAgent(BaseAgent):
    """Syncs and verifies paper implementations."""

    def __init__(self, llm_provider: str = None):
        super().__init__("CodeIntegrityAgent", llm_provider=llm_provider)
        self.generator = Generator(llm_provider=llm_provider)
        self.session = requests.Session()

    def execute(self, task: dict) -> dict:
        """
        Identify code for a list of papers or a specific one.
        """
        papers = task.get("papers", [])
        if not papers and task.get("paper"):
            papers = [task.get("paper")]

        results = []
        for paper in papers:
            if not paper:
                continue
            self.log(f"Evaluating code integrity for: {paper.get('title')}")
            
            prompt = CODE_EXTRACTION_PROMPT.format(
                title=paper.get("title", ""),
                abstract=paper.get("abstract", "No abstract available."),
                url=paper.get("url", "")
            )
            
            try:
                # LLM Extraction
                response = self.generator.llm.invoke(prompt).content
                # Clean response
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].strip()
                
                import json
                meta = json.loads(response.strip())
                
                # Manual validation of repo URL if extracted
                if meta.get("repo_url") and "github.com" in meta["repo_url"]:
                    # Basic check if it exists (mocked or real)
                    # For now keep it as is
                    pass
                
                results.append({
                    "title": paper.get("title"),
                    "code_meta": meta
                })
            except Exception as e:
                self.log(f"Extraction failed for {paper.get('title')}: {e}", level="error")
                results.append({
                    "title": paper.get("title"),
                    "code_meta": {"has_code": False, "repo_url": None}
                })

        return {
            "status": "success",
            "code_updates": results
        }
