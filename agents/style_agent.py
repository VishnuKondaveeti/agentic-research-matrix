"""
StyleAgent: Adapts research reports into different communication styles (Beginner, Expert, Blog Post, etc.).
"""

from agents.base_agent import BaseAgent
from rag.generator import Generator

STYLE_PROMPT = """
You are an ELITE Communications Specialist for Scientific Research. Your job is to rewrite the provided research report into the requested style.

### Requested Style: {style}

### Original Report:
{report}

### Style Guide:
- **Beginner**: Avoid jargon, use analogies, explain core concepts, focus on the "why it matters".
- **Expert**: Keep technical details, use precise terminology, focus on methodology and statistical significance.
- **Blog Post**: Use a hook, catchy headings, bullet points, and an engaging, conversational tone.
- **Executive Summary**: Focus on high-level outcomes, business/field impact, and strategic recommendations. Concise and professional.

Rewrite the report strictly adhering to the style instructions. Maintain the scientific integrity while changing the tone and structure.
"""

class StyleAgent(BaseAgent):
    """Adapts report styles for different audiences."""

    def __init__(self, llm_provider: str = None):
        super().__init__("StyleAgent", llm_provider=llm_provider)
        self.generator = Generator(llm_provider=llm_provider)

    def execute(self, task: dict) -> dict:
        """
        Rewrite a report in a specific style.
        """
        report = task.get("report", "")
        style = task.get("style", "Standard")
        
        if not report:
            return {"status": "error", "message": "No report provided"}

        self.log(f"Adapting report to style: {style}")

        from config.settings import settings, CallBudgetTracker

        if getattr(settings, "demo_gemini_only", False) or str(style).lower() in ("standard", "literature review", "academic", "comprehensive literature review"):
            self.log(f"Report format already aligned with style '{style}'. Passing through without extra LLM call.", level="info")
            return {
                "status": "success",
                "styled_report": report,
                "style": style,
            }

        prompt = STYLE_PROMPT.format(
            style=style,
            report=report
        )
        
        try:
            CallBudgetTracker.record_call("StyleAgent")
            styled_report = self.generator.llm.invoke(prompt).content
            return {
                "status": "success",
                "styled_report": styled_report,
                "style": style
            }
        except Exception as e:
            self.log(f"Style adaptation failed: {e}", level="error")
            return {
                "status": "error",
                "message": str(e)
            }
