"""
PodcastAgent: Transforms research reports into an engaging 2-persona dialogue script.
"""

from agents.base_agent import BaseAgent
from rag.generator import Generator

PODCAST_PROMPT = """
You are an ELITE Podcast Scriptwriter. Your goal is to transform the following technical research report into a high-energy, engaging dialogue between two hosts: **Host A (The Expert)** and **Host B (The Curious Layman)**.

### Technical Report:
{report}

### Script Format:
- **Host A**: (Technical, provides the "meat", uses analogies but keeps it scientific)
- **Host B**: (Asks the "stupid" questions, reacts with wonder, summarizes "why it matters" for the listener)

### Tone:
Brainy but accessible. Like "Deep Dive" or "Radiolab".

Write a script that covers the main findings, the methodology (simplified), and the "So what?" of the research.
"""

class PodcastAgent(BaseAgent):
    """Generates engaging podcast scripts from research data."""

    def __init__(self, llm_provider: str = None):
        super().__init__("PodcastAgent", llm_provider=llm_provider)
        self.generator = Generator(llm_provider=llm_provider)

    def execute(self, task: dict) -> dict:
        """
        Convert a report into a podcast script.
        """
        report = task.get("report", "")
        if not report:
            return {"status": "error", "message": "No report provided for scriptwriting"}

        self.log("Drafting podcast script...")
        prompt = PODCAST_PROMPT.format(report=report)
        
        try:
            script = self.generator.llm.invoke(prompt).content
            return {
                "status": "success",
                "script": script,
                "agent": self.name
            }
        except Exception as e:
            self.log(f"Podcast generation failed: {e}", level="error")
            return {"status": "error", "message": str(e)}
