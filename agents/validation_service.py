"""
ValidationService: Generates reproducible environments (Dockerfiles) for research blueprints.
"""

from agents.base_agent import BaseAgent
from rag.generator import Generator

DOCKER_PROMPT = """
You are a Cloud Infrastructure Engineer. Given the following Python Blueprint (which is a methodology from a research paper), generate a minimal Dockerfile and a `requirements.txt` to run this logic in a reproducible environment.

### Python Blueprint:
{blueprint}

### Output Requirements:
1. **Dockerfile**: Use a lean base image (e.g., python:3.9-slim).
2. **requirements.txt**: List likely dependencies (numpy, torch, etc.) based on the code structure.
3. **Setup Script**: A short bash command to build and run.

Format the output as a clear markdown block with separate sections for Dockerfile and requirements.
"""

class ValidationService(BaseAgent):
    """Generates containerized validation environments for paper methodologies."""

    def __init__(self, llm_provider: str = None):
        super().__init__("ValidationService", llm_provider=llm_provider)
        self.generator = Generator(llm_provider=llm_provider)

    def execute(self, task: dict) -> dict:
        """
        Generate a validation environment for the given blueprint.
        """
        blueprint = task.get("blueprint", "")
        if not blueprint:
            return {"status": "error", "message": "No blueprint provided for validation environment"}

        self.log("Generating Dockerized validation environment...")
        prompt = DOCKER_PROMPT.format(blueprint=blueprint)
        
        try:
            environment = self.generator.llm.invoke(prompt).content
            return {
                "status": "success",
                "environment_spec": environment,
                "agent": self.name
            }
        except Exception as e:
            self.log(f"Environment generation failed: {e}", level="error")
            return {"status": "error", "message": str(e)}
