"""
TrendAgent: Analysts citation velocity and keyword bursts to forecast research trends.
"""

from agents.base_agent import BaseAgent
from rag.generator import Generator

TREND_PROMPT = """
You are an ELITE Research Strategist. Your task is to analyze the research context provided and forecast the 3-year impact trajectory of the current topic.

### Research Context:
{context}

### Analysis Goal:
1. **Velocity**: How fast is this topic growing based on the paper abstracts?
2. **Impact Score**: On a scale of 1-100, what is the probability of this becoming a foundational concept?
3. **Forecast**: What is the most likely "Next Breakthrough" in this specific subspace?

### Output Format (Markdown):
- **Impact Probability**: [SCORE]%
- **Growth Velocity**: [Low/Medium/High/Exponential]
- **Market/Scientific Forecast**: (3-4 sentences on the future direction)
- **Emerging Tags**: (List 3 keywords that are starting to pop up)
"""

class TrendAgent(BaseAgent):
    """Forecasts research trends and impact scores."""

    def __init__(self, llm_provider: str = None):
        super().__init__("TrendAgent", llm_provider=llm_provider)
        self.generator = Generator(llm_provider=llm_provider)

    def execute(self, task: dict) -> dict:
        """
        Perform trend forecasting on research data.
        """
        analysis = task.get("analysis", "")
        if not analysis:
            return {"status": "error", "message": "No analysis data provided for forecasting"}

        self.log("Analyzing topic velocity and impact probability...")
        prompt = TREND_PROMPT.format(context=analysis)
        
        try:
            forecast = self.generator.llm.invoke(prompt).content
            return {
                "status": "success",
                "forecast": forecast,
                "agent": self.name
            }
        except Exception as e:
            self.log(f"Trend forecasting failed: {e}", level="error")
            return {"status": "error", "message": str(e)}
