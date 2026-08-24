"""
Dynamic LangGraph orchestrator for the Agentic Research Matrix.

The orchestrator keeps the specialist agents intact, but no longer forces every
request through a fixed research -> hypothesis -> analysis -> debate chain.

Intent is parsed first, a planner selects a minimal agent plan, selected agents
collaborate through shared state, and EvaluationAgent can trigger replanning.
"""

import json
from typing import Any, Literal, TypedDict

from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agents.advisor_agent import AdvisorAgent
from agents.analysis_agent import AnalysisAgent
from agents.base_agent import BaseAgent
from agents.blueprint_agent import BlueprintAgent
from agents.code_integrity_agent import CodeIntegrityAgent
from agents.debate_agent import DebateAgent
from agents.evaluation_agent import EvaluationAgent
from agents.hypothesis_agent import HypothesisAgent
from agents.intent_agent import IntentAgent
from agents.podcast_agent import PodcastAgent
from agents.report_agent import ReportAgent
from agents.research_agent import ResearchAgent
from agents.retrieval_agent import RetrievalAgent
from agents.style_agent import StyleAgent
from agents.trend_agent import TrendAgent
from agents.validation_service import ValidationService
from config.settings import settings


# ============================================================
# Types
# ============================================================

AgentName = Literal[
    "research",
    "retrieval",
    "analysis",
    "advisor",
    "hypothesis",
    "debate",
    "code_integrity",
    "blueprint",
    "trend",
    "validation",
    "report",
    "style",
    "podcast",
]


class AgentPlanStep(BaseModel):
    agent: AgentName = Field(
        description="Specialist agent to execute."
    )
    reason: str = Field(
        description="Why this agent is needed for the current request."
    )
    required: bool = Field(
        default=True,
        description="Whether failure should be considered significant."
    )


class PlannerOutput(BaseModel):
    rationale: str = Field(
        description="Brief explanation of the selected workflow."
    )
    steps: list[AgentPlanStep] = Field(
        description="Ordered specialist agents to execute."
    )


class ResearchState(TypedDict, total=False):
    query: str
    action: str
    task: dict

    intent_filters: dict

    plan: list[dict]
    plan_rationale: str
    current_step: int

    agent_outputs: dict
    agent_errors: dict
    shared_memory: list[dict]

    research_results: list[dict]
    retrieval_context: str
    hypotheses: str
    analysis_results: str
    advice_output: str
    debate_output: str
    code_integrity_results: list[dict]
    blueprint_output: str
    trend_output: str
    validation_output: str

    draft_report: str
    styled_report: str
    podcast_script: str
    final_output: str

    sources: list[dict]

    evaluation_feedback: str
    evaluation_score: int
    evaluation_passed: bool
    iterations: int

    next_agent: str
    workflow_error: bool


# ============================================================
# Orchestrator
# ============================================================

class OrchestratorAgent(BaseAgent):
    """
    Planner-directed, evaluation-driven multi-agent orchestrator.

    The orchestrator itself uses the configured LLM provider for planning.
    Specialist agents receive the same provider so that the complete pipeline
    uses one consistent LLM backend unless explicitly overridden.
    """

    MAX_REFLECTIONS = 3

    AGENT_CLASSES = {
        "research": ResearchAgent,
        "retrieval": RetrievalAgent,
        "analysis": AnalysisAgent,
        "advisor": AdvisorAgent,
        "hypothesis": HypothesisAgent,
        "debate": DebateAgent,
        "code_integrity": CodeIntegrityAgent,
        "blueprint": BlueprintAgent,
        "trend": TrendAgent,
        "validation": ValidationService,
        "report": ReportAgent,
        "style": StyleAgent,
        "podcast": PodcastAgent,
        "intent": IntentAgent,
        "evaluation": EvaluationAgent,
    }

    PLAN_AGENT_NAMES = {
        "research",
        "retrieval",
        "analysis",
        "advisor",
        "hypothesis",
        "debate",
        "code_integrity",
        "blueprint",
        "trend",
        "validation",
        "report",
        "style",
        "podcast",
    }

    # ========================================================
    # Initialization
    # ========================================================

    def __init__(self, llm_provider: str = None):
        """
        Initialize the orchestrator.

        If llm_provider is not explicitly supplied, use the provider from
        config.settings.

        Example:
            OrchestratorAgent()
            -> uses settings.llm_provider

            OrchestratorAgent(llm_provider="ollama")
            -> explicitly uses Ollama
        """

        effective_provider = (
            llm_provider or settings.llm_provider
        ).strip().lower()

        if getattr(settings, "demo_gemini_only", False):
            effective_provider = "gemini"

        super().__init__(
            "OrchestratorAgent",
            llm_provider=effective_provider,
        )

        self.llm_provider = effective_provider

        # Lazy-loaded specialist agents.
        self._agents: dict[str, BaseAgent] = {}

        # Planner LLM is also lazy-loaded.
        self.llm = None

        # LangGraph is lazy-built.
        self.graph = None

        if getattr(settings, "demo_gemini_only", False):
            self.log("[DEMO] Gemini-only mode ENABLED", level="info")
            self.log("[DEMO] Provider = gemini", level="info")
            self.log("[DEMO] Ollama fallback = DISABLED", level="info")
        else:
            self.log(
                f"Initialized with LLM provider: {self.llm_provider}"
            )

    # ========================================================
    # LLM initialization
    # ========================================================

    def _init_llm(self, provider: str):
        """
        Initialize the LLM according to the selected provider.

        Supported:
            - gemini
            - google
            - openai
            - ollama
        """

        provider = (
            provider or settings.llm_provider
        ).strip().lower()

        # ----------------------------------------------------
        # Gemini
        # ----------------------------------------------------

        if provider in ("gemini", "google"):
            if not settings.google_api_key:
                raise ValueError(
                    "GOOGLE_API_KEY is required for Gemini provider."
                )

            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=settings.llm_model,
                google_api_key=settings.google_api_key,
                temperature=0.2,
                timeout=120,
            )

        # ----------------------------------------------------
        # OpenAI
        # ----------------------------------------------------

        if provider == "openai":
            if not settings.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required for OpenAI provider."
                )

            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=settings.openai_model,
                openai_api_key=settings.openai_api_key,
                temperature=0.2,
            )

        # ----------------------------------------------------
        # Ollama
        # ----------------------------------------------------

        if provider == "ollama":
            from langchain_ollama import ChatOllama

            return ChatOllama(
                model=settings.ollama_model,
                base_url=settings.ollama_host,
                temperature=0.2,
            )

        raise ValueError(
            f"Unsupported LLM provider: {provider}. "
            f"Expected gemini, openai, or ollama."
        )

    def _ensure_planner_llm(self):
        """Create the planner LLM only when it is first needed."""

        if self.llm is None:
            provider = (
                self.llm_provider or settings.llm_provider
            ).strip().lower()

            self.llm = self._init_llm(provider)

            self.log(
                f"Planner LLM initialized: "
                f"provider={provider}, "
                f"model={getattr(self.llm, 'model', 'unknown')}"
            )

        return self.llm

    # ========================================================
    # LLM Fallback Management
    # ========================================================

    def _is_quota_error(self, error_text: str) -> bool:
        """Check if an error string indicates an LLM quota or rate-limit error."""
        if not error_text:
            return False
        return any(
            marker in error_text.upper()
            for marker in [
                "RESOURCE_EXHAUSTED",
                "429",
                "QUOTA",
                "RATE LIMIT",
                "RATE_LIMIT",
            ]
        )

    def _is_ollama_available(self) -> bool:
        """Check if the configured Ollama instance is reachable."""
        host = getattr(settings, "ollama_host", "") or "http://localhost:11434"
        try:
            import urllib.request
            url = f"{host.rstrip('/')}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as response:
                return response.status == 200
        except Exception:
            return False

    def _switch_to_ollama_fallback(self) -> bool:
        """
        Switch active LLM provider from Gemini to Ollama fallback.

        Guarantees:
        - When DEMO_GEMINI_ONLY is True, fallback is completely disabled.
        - Maximum fallback chain: Gemini -> Ollama (no infinite loop).
        - Resets planner LLM and clears specialist agent cache so subsequent
          agents in the same workflow use Ollama.
        """
        if getattr(settings, "demo_gemini_only", False):
            self.log(
                "[DEMO] Gemini quota exceeded. Ollama fallback is DISABLED for demo.",
                level="warning",
            )
            return False

        if self.llm_provider == "ollama":
            return False

        if not self._is_ollama_available():
            self.log(
                "[LLM] Ollama host is not available for fallback.",
                level="warning",
            )
            return False

        self.log(
            "[LLM] Gemini quota exhausted. Falling back to Ollama.",
            level="warning",
        )

        self.llm_provider = "ollama"
        self.llm = None
        self._agents.clear()

        self.log(
            "[LLM] Ollama fallback active.",
            level="info",
        )
        return True

    # ========================================================
    # Graph
    # ========================================================

    def _ensure_graph(self):
        """Build the LangGraph workflow lazily."""

        if self.graph is None:
            self.graph = self._build_graph()

        return self.graph

    # ========================================================
    # Agent management
    # ========================================================

    def _get_agent(self, name: str) -> BaseAgent:
        """
        Lazy-initialize a specialist agent.

        IMPORTANT:
        The orchestrator's provider is explicitly propagated to every
        specialist agent.
        """

        if name not in self.AGENT_CLASSES:
            raise ValueError(
                f"Unknown agent: {name}"
            )

        if getattr(settings, "demo_gemini_only", False) and self.llm_provider != "gemini":
            raise ValueError(
                f"DEMO_GEMINI_ONLY is active: cannot initialize agent '{name}' with provider '{self.llm_provider}'"
            )

        if name not in self._agents:
            self._agents[name] = self.AGENT_CLASSES[name](
                llm_provider=self.llm_provider
            )

            self.log(
                f"Initialized specialist agent '{name}' "
                f"with provider={self.llm_provider}"
            )

        return self._agents[name]

    # ========================================================
    # Shared state helpers
    # ========================================================

    def _remember(
        self,
        state: ResearchState,
        event: str,
        payload: dict,
    ) -> list[dict]:
        memory = list(
            state.get("shared_memory", [])
        )

        memory.append(
            {
                "event": event,
                "payload": payload,
            }
        )

        return memory[-50:]

    def _paper_context(
        self,
        state: ResearchState,
    ) -> str:

        papers = state.get(
            "research_results",
            [],
        )

        parts = []

        for paper in papers[:8]:
            title = paper.get(
                "title",
                "Unknown",
            )

            authors = paper.get(
                "authors",
                "",
            )

            abstract = paper.get(
                "abstract",
                paper.get("snippet", ""),
            )

            source = paper.get(
                "source",
                "",
            )

            parts.append(
                f"Title: {title}\n"
                f"Authors: {authors}\n"
                f"Source: {source}\n"
                f"Abstract: {abstract}"
            )

        return "\n\n---\n\n".join(parts)

    def _specialist_outputs(
        self,
        state: ResearchState,
    ) -> dict[str, Any]:

        return {
            "retrieval_context": state.get(
                "retrieval_context",
                "",
            ),
            "hypotheses": state.get(
                "hypotheses",
                "",
            ),
            "advice": state.get(
                "advice_output",
                "",
            ),
            "code_integrity": state.get(
                "code_integrity_results",
                [],
            ),
            "blueprint": state.get(
                "blueprint_output",
                "",
            ),
            "trend": state.get(
                "trend_output",
                "",
            ),
            "validation": state.get(
                "validation_output",
                "",
            ),
        }

    def _compact_papers(
        self,
        papers: list[dict],
    ) -> list[dict]:

        return [
            {
                key: value
                for key, value in paper.items()
                if key != "full_text"
            }
            for paper in papers
        ]

    # ========================================================
    # Intent
    # ========================================================

    def _intent_node(
        self,
        state: ResearchState,
    ) -> ResearchState:

        self.log(
            "Parsing research intent..."
        )

        try:
            result = self._safe_agent_execute(
                "intent",
                {
                    "query": state["query"]
                },
            )

            filters = result.get(
                "filters",
                {},
            )

        except Exception as e:

            self.log(
                f"IntentAgent unavailable, "
                f"using fallback intent: {e}",
                level="warning",
            )

            filters = {
                "topic": state["query"],
                "min_year": None,
                "max_year": None,
                "sources": None,
                "focus": None,
                "intent_type": "search",
            }

        return {
            "intent_filters": filters,
            "shared_memory": self._remember(
                state,
                "intent",
                {
                    "filters": filters
                },
            ),
        }

    # ========================================================
    # Planner
    # ========================================================

    def _planner_node(
        self,
        state: ResearchState,
    ) -> ResearchState:

        self.log(
            "Planner: selecting adaptive specialist workflow..."
        )

        try:

            planner_output = self._llm_plan(
                state
            )

            rationale = (
                planner_output.rationale
            )

            steps = [
                step.model_dump()
                for step in planner_output.steps
            ]

        except Exception as e:

            error_text = str(e)

            if self._is_quota_error(error_text) and self.llm_provider in ("gemini", "google"):
                if self._switch_to_ollama_fallback():
                    try:
                        planner_output = self._llm_plan(state)
                        rationale = planner_output.rationale
                        steps = [
                            step.model_dump()
                            for step in planner_output.steps
                        ]
                    except Exception as fallback_e:
                        self.log(
                            f"Planner Ollama fallback failed: {fallback_e}, "
                            "using heuristic plan",
                            level="warning",
                        )
                        rationale, steps = self._heuristic_plan(state)
                else:
                    rationale, steps = self._heuristic_plan(state)
            else:
                self.log(
                    f"Planner LLM failed, "
                    f"using heuristic plan: {e}",
                    level="warning",
                )
                rationale, steps = (
                    self._heuristic_plan(state)
                )

        steps = self._normalize_plan(
            steps,
            state,
        )

        self.log(
            "Planner selected: "
            + " -> ".join(
                step["agent"]
                for step in steps
            )
        )

        return {
            "plan": steps,
            "plan_rationale": rationale,
            "current_step": 0,
            "shared_memory": self._remember(
                state,
                "plan",
                {
                    "rationale": rationale,
                    "steps": steps,
                },
            ),
        }

    def _llm_plan(
        self,
        state: ResearchState,
    ) -> PlannerOutput:

        prompt = PromptTemplate.from_template(
            """
You are the Planner for a multi-agent academic research framework.

User query:
{query}

Intent filters:
{intent_filters}

Original task options:
{task}

Current shared memory:
{memory}

Evaluation feedback, if replanning:
{feedback}

Available specialist agents:

- research: search and ingest fresh academic papers.
- retrieval: retrieve context from the existing vector knowledge base.
- analysis: analyze evidence and extract scholarly insights.
- advisor: identify research gaps and roadmap advice.
- hypothesis: generate novel hypotheses/future-work ideas.
- debate: adversarial peer-review style critique and balanced synthesis.
- code_integrity: identify official code repositories for papers.
- blueprint: translate methodology into an implementation blueprint.
- trend: forecast topic velocity and likely future impact.
- validation: create reproducible environment specs from a blueprint.
- report: synthesize final scholarly output from shared state.
- style: adapt final report tone/style.
- podcast: transform final report into a two-host podcast script.

Rules:

1. Select only agents that add measurable value.
2. Do not include intent or evaluation.
3. The report agent is normally required for report/full_pipeline requests.
4. On replanning, prefer the smallest corrective plan that addresses feedback.
5. Keep the workflow ordered logically.
6. Avoid unnecessary agents.
"""
        )

        llm = (
            self._ensure_planner_llm()
            .with_structured_output(
                PlannerOutput
            )
        )

        chain = prompt | llm

        from config.settings import CallBudgetTracker
        CallBudgetTracker.record_call("PlannerAgent")

        return chain.invoke(
            {
                "query": state.get(
                    "query",
                    "",
                ),
                "intent_filters": json.dumps(
                    state.get(
                        "intent_filters",
                        {},
                    ),
                    default=str,
                ),
                "task": json.dumps(
                    state.get(
                        "task",
                        {},
                    ),
                    default=str,
                ),
                "memory": json.dumps(
                    state.get(
                        "shared_memory",
                        [],
                    )[-12:],
                    default=str,
                ),
                "feedback": state.get(
                    "evaluation_feedback",
                    "None",
                ),
            }
        )

    def _heuristic_plan(
        self,
        state: ResearchState,
    ) -> tuple[str, list[dict]]:

        query = state.get(
            "query",
            "",
        )

        task = state.get(
            "task",
            {},
        )

        text = (
            f"{query} "
            f"{task.get('style', '')} "
            f"{task.get('output_format', '')}"
        ).lower()

        feedback = state.get(
            "evaluation_feedback",
            "",
        )

        steps: list[dict] = []

        def add(
            agent: str,
            reason: str,
            required: bool = True,
        ):
            if agent not in [
                step["agent"]
                for step in steps
            ]:
                steps.append(
                    {
                        "agent": agent,
                        "reason": reason,
                        "required": required,
                    }
                )

        # ----------------------------------------------------
        # Replanning
        # ----------------------------------------------------

        if feedback and state.get(
            "draft_report"
        ):

            if any(
                word in feedback.lower()
                for word in (
                    "source",
                    "citation",
                    "evidence",
                    "paper",
                )
            ):
                add(
                    "retrieval",
                    "Refresh evidence to address evaluation feedback.",
                )

            add(
                "report",
                "Revise the report using evaluation feedback.",
            )

        # ----------------------------------------------------
        # Initial plan
        # ----------------------------------------------------

        else:

            add(
                "research",
                "Collect current papers and metadata for the topic.",
            )

            if any(
                word in text
                for word in (
                    "implementation",
                    "implement",
                    "methodology",
                    "architecture",
                    "reproduce",
                    "reproducible",
                    "docker",
                    "validation",
                )
            ):
                add(
                    "blueprint",
                    "Translate methodology into an implementation blueprint.",
                )

            if any(
                word in text
                for word in (
                    "validate",
                    "validation",
                    "docker",
                    "environment",
                    "reproducible",
                )
            ):
                add(
                    "validation",
                    "Create a reproducible validation environment from the blueprint.",
                )

            if any(
                word in text
                for word in (
                    "code",
                    "github",
                    "repository",
                    "repo",
                    "implementation",
                )
            ):
                add(
                    "code_integrity",
                    "Find official or high-confidence implementation repositories.",
                )

            if any(
                word in text
                for word in (
                    "trend",
                    "forecast",
                    "emerging",
                    "velocity",
                    "future impact",
                )
            ):
                add(
                    "trend",
                    "Forecast research trajectory and emerging themes.",
                )

            if any(
                word in text
                for word in (
                    "hypothesis",
                    "hypotheses",
                    "novel",
                    "future work",
                )
            ):
                add(
                    "hypothesis",
                    "Generate novel research hypotheses.",
                )

            if any(
                word in text
                for word in (
                    "critique",
                    "debate",
                    "skeptic",
                    "limitations",
                    "peer review",
                )
            ):
                add(
                    "debate",
                    "Stress-test findings through adversarial review.",
                )

            if any(
                word in text
                for word in (
                    "advise",
                    "advisor",
                    "roadmap",
                    "gap",
                    "gaps",
                )
            ):
                add(
                    "advisor",
                    "Identify gaps and actionable research advice.",
                )

            if not any(
                step["agent"]
                in {
                    "trend",
                    "blueprint",
                    "code_integrity",
                    "hypothesis",
                    "debate",
                    "advisor",
                }
                for step in steps
            ):
                add(
                    "analysis",
                    "Analyze evidence before synthesis.",
                )

            add(
                "report",
                "Synthesize selected agent outputs into a final report.",
            )

        style = (
            task.get("style") or ""
        ).lower()

        if style and style not in {
            "professional",
            "standard",
            "podcast script",
        }:
            add(
                "style",
                f"Adapt final output to requested style: "
                f"{task.get('style')}.",
                required=False,
            )

        if (
            "podcast" in text
            or style == "podcast script"
        ):
            add(
                "podcast",
                "Convert the final report into a podcast script.",
            )

        return (
            "Heuristic plan selected from query intent and evaluation feedback.",
            steps,
        )

    def _normalize_plan(
        self,
        steps: list[dict],
        state: ResearchState,
    ) -> list[dict]:

        normalized: list[dict] = []
        seen = set()

        for raw in steps:

            agent = raw.get(
                "agent"
            )

            if (
                agent not in self.PLAN_AGENT_NAMES
                or agent in seen
            ):
                continue

            normalized.append(
                {
                    "agent": agent,
                    "reason": raw.get(
                        "reason",
                        "Selected by planner.",
                    ),
                    "required": bool(
                        raw.get(
                            "required",
                            True,
                        )
                    ),
                }
            )

            seen.add(agent)

        action = state.get(
            "action"
        )

        if (
            action in {
                "report",
                "full_pipeline",
            }
            and "report" not in seen
        ):

            normalized.append(
                {
                    "agent": "report",
                    "reason": (
                        "A synthesized final report "
                        "is required for this action."
                    ),
                    "required": True,
                }
            )

            seen.add("report")

        # Validation requires blueprint.
        if (
            "validation" in seen
            and "blueprint" not in seen
            and not state.get(
                "blueprint_output"
            )
        ):

            validation = [
                step
                for step in normalized
                if step["agent"] == "validation"
            ][0]

            normalized = [
                step
                for step in normalized
                if step["agent"] != "validation"
            ]

            insert_at = max(
                0,
                len(normalized)
                - (
                    1
                    if "report" in seen
                    else 0
                ),
            )

            normalized.insert(
                insert_at,
                {
                    "agent": "blueprint",
                    "reason": (
                        "Validation requires an implementation "
                        "blueprint first."
                    ),
                    "required": True,
                },
            )

            normalized.insert(
                insert_at + 1,
                validation,
            )

        return normalized

    # ========================================================
    # Dynamic agent nodes
    # ========================================================

    def _research_node(
        self,
        state: ResearchState,
    ) -> ResearchState:
        return self._run_planned_agent(
            state,
            "research",
        )

    def _retrieval_node(
        self,
        state: ResearchState,
    ) -> ResearchState:
        # In demo mode, skip slow ChromaDB vector retrieval.
        # Analysis agent already processes the cached research papers.
        if getattr(settings, "demo_gemini_only", False):
            self.logger.info(
                "[OrchestratorAgent] [DEMO] Skipping retrieval node "
                "(analysis uses cached research results directly)"
            )
            return state
        return self._run_planned_agent(
            state,
            "retrieval",
        )

    def _analysis_node(
        self,
        state: ResearchState,
    ) -> ResearchState:
        return self._run_planned_agent(
            state,
            "analysis",
        )

    def _advisor_node(
        self,
        state: ResearchState,
    ) -> ResearchState:
        return self._run_planned_agent(
            state,
            "advisor",
        )

    def _hypothesis_node(
        self,
        state: ResearchState,
    ) -> ResearchState:
        return self._run_planned_agent(
            state,
            "hypothesis",
        )

    def _debate_node(
        self,
        state: ResearchState,
    ) -> ResearchState:
        return self._run_planned_agent(
            state,
            "debate",
        )

    def _code_integrity_node(
        self,
        state: ResearchState,
    ) -> ResearchState:
        if getattr(settings, "demo_gemini_only", False):
            self.logger.info("[OrchestratorAgent] [DEMO] Skipping code_integrity node")
            return state
        return self._run_planned_agent(
            state,
            "code_integrity",
        )

    def _blueprint_node(
        self,
        state: ResearchState,
    ) -> ResearchState:
        if getattr(settings, "demo_gemini_only", False):
            self.logger.info("[OrchestratorAgent] [DEMO] Skipping blueprint node")
            return state
        return self._run_planned_agent(
            state,
            "blueprint",
        )

    def _trend_node(
        self,
        state: ResearchState,
    ) -> ResearchState:
        return self._run_planned_agent(
            state,
            "trend",
        )

    def _validation_node(
        self,
        state: ResearchState,
    ) -> ResearchState:
        return self._run_planned_agent(
            state,
            "validation",
        )

    def _report_node(
        self,
        state: ResearchState,
    ) -> ResearchState:
        return self._run_planned_agent(
            state,
            "report",
        )

    def _style_node(
        self,
        state: ResearchState,
    ) -> ResearchState:
        return self._run_planned_agent(
            state,
            "style",
        )

    def _podcast_node(
        self,
        state: ResearchState,
    ) -> ResearchState:
        return self._run_planned_agent(
            state,
            "podcast",
        )

    # ========================================================
    # Agent execution
    # ========================================================

    def _run_planned_agent(
        self,
        state: ResearchState,
        agent_name: str,
    ) -> ResearchState:

        step = self._current_plan_step(
            state,
            agent_name,
        )

        self.log(
            f"Executing {agent_name}: "
            f"{step.get('reason', 'selected by planner')}"
        )

        task = self._build_agent_task(
            agent_name,
            state,
        )

        result = self._safe_agent_execute(
            agent_name,
            task,
        )

        return self._apply_agent_result(
            agent_name,
            result,
            state,
            step,
        )

    def _current_plan_step(
        self,
        state: ResearchState,
        agent_name: str,
    ) -> dict:

        plan = state.get(
            "plan",
            [],
        )

        index = state.get(
            "current_step",
            0,
        )

        if (
            0 <= index < len(plan)
        ):
            return plan[index]

        return {
            "agent": agent_name,
            "reason": "Routed dynamically.",
            "required": True,
        }

    def _safe_agent_execute(
        self,
        agent_name: str,
        task: dict,
    ) -> dict:
        """
        Execute a specialist agent safely.

        Agent-level failures are returned as structured errors so
        the orchestrator can decide whether the workflow should stop.
        If Gemini quota is exhausted, automatically falls back to Ollama
        if available.
        """

        try:
            result = self._get_agent(
                agent_name
            )._safe_execute(task)

            if not isinstance(result, dict):
                return {
                    "status": "error",
                    "agent": agent_name,
                    "message": (
                        f"{agent_name} returned an invalid result type: "
                        f"{type(result).__name__}"
                    ),
                }

            # ----------------------------------------------------
            # Detect model/API failures returned by BaseAgent
            # ----------------------------------------------------

            if result.get("status") == "error":

                error_text = str(
                    result.get("error")
                    or result.get("message")
                    or "Unknown agent error"
                )

                self.log(
                    f"{agent_name} failed: {error_text}",
                    level="error",
                )

                # Gemini quota/rate-limit errors: attempt fallback to Ollama
                if self._is_quota_error(error_text):
                    if self.llm_provider in ("gemini", "google") and self._switch_to_ollama_fallback():
                        self.log(
                            f"Retrying [{agent_name}] with Ollama fallback...",
                            level="info",
                        )
                        retry_agent = self._get_agent(agent_name)
                        retry_result = retry_agent._safe_execute(task)

                        if isinstance(retry_result, dict) and retry_result.get("status") != "error":
                            return retry_result

                        retry_error = str(
                            retry_result.get("error")
                            or retry_result.get("message")
                            or "Ollama retry failed"
                        )
                        self.log(
                            f"[{agent_name}] Ollama fallback also failed: {retry_error}",
                            level="error",
                        )
                        retry_result["fatal_error"] = True
                        retry_result["stop_workflow"] = True
                        return retry_result

                    # If already on Ollama or Ollama is unavailable:
                    self.log(
                        f"[{agent_name}] LLM quota/rate limit detected. Stopping workflow.",
                        level="error",
                    )
                    result["fatal_error"] = True
                    result["stop_workflow"] = True

                return result

            return result

        except Exception as e:

            error_text = str(e)

            self.log(
                f"{agent_name} failed before execution: {error_text}",
                level="error",
            )

            if self._is_quota_error(error_text):
                if self.llm_provider in ("gemini", "google") and self._switch_to_ollama_fallback():
                    self.log(
                        f"Retrying [{agent_name}] with Ollama fallback after exception...",
                        level="info",
                    )
                    try:
                        retry_agent = self._get_agent(agent_name)
                        retry_result = retry_agent._safe_execute(task)
                        if isinstance(retry_result, dict) and retry_result.get("status") != "error":
                            return retry_result
                        retry_result["fatal_error"] = True
                        retry_result["stop_workflow"] = True
                        return retry_result
                    except Exception as retry_e:
                        return {
                            "status": "error",
                            "message": str(retry_e),
                            "agent": agent_name,
                            "fatal_error": True,
                            "stop_workflow": True,
                        }

            return {
                "status": "error",
                "message": error_text,
                "agent": agent_name,
                "fatal_error": True,
                "stop_workflow": True,
            }

    # ========================================================
    # Agent task construction
    # ========================================================

    def _build_agent_task(
        self,
        agent_name: str,
        state: ResearchState,
    ) -> dict:

        query = state.get(
            "query",
            "",
        )

        task = state.get(
            "task",
            {},
        )

        filters = state.get(
            "intent_filters",
            {},
        )

        top_k = task.get(
            "top_k",
            10,
        )

        max_papers = task.get(
            "max_papers",
            5,
        )

        complexity = task.get(
            "complexity",
            "standard",
        )

        # ----------------------------------------------------
        # Research
        # ----------------------------------------------------

        if agent_name == "research":

            return {
                "action": "search",
                "query": (
                    filters.get(
                        "topic",
                        query,
                    )
                    if filters
                    else query
                ),
                "filters": filters,
                "sources": (
                    task.get("sources")
                    or filters.get("sources")
                ),
                "max_papers": max_papers,
                "download": task.get(
                    "download",
                    True,
                ),
                "process": task.get(
                    "process",
                    True,
                ),
            }

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        if agent_name == "retrieval":

            return {
                "query": query,
                "top_k": top_k,
                "format": "context",
            }

        # ----------------------------------------------------
        # Analysis
        # ----------------------------------------------------

        if agent_name == "analysis":

            return {
                "action": "analyze",
                "query": query,
                "context_data": state.get(
                    "research_results",
                    [],
                ),
                "hypotheses": state.get(
                    "hypotheses",
                    "",
                ),
                "analysis_type": "both",
                "complexity": complexity,
                "top_k": top_k,
            }

        # ----------------------------------------------------
        # Advisor
        # ----------------------------------------------------

        if agent_name == "advisor":

            return {
                "action": "advise",
                "topic": query,
                "top_k": top_k,
            }

        # ----------------------------------------------------
        # Hypothesis
        # ----------------------------------------------------

        if agent_name == "hypothesis":

            return {
                "query": query,
                "context": (
                    state.get(
                        "analysis_results",
                        "",
                    )
                    or self._paper_context(
                        state
                    )
                ),
            }

        # ----------------------------------------------------
        # Debate
        # ----------------------------------------------------

        if agent_name == "debate":

            return {
                "query": query,
                "context": (
                    state.get(
                        "analysis_results",
                        "",
                    )
                    or self._paper_context(
                        state
                    )
                ),
            }

        # ----------------------------------------------------
        # Code Integrity
        # ----------------------------------------------------

        if agent_name == "code_integrity":

            return {
                "papers": state.get(
                    "research_results",
                    [],
                )
            }

        # ----------------------------------------------------
        # Blueprint
        # ----------------------------------------------------

        if agent_name == "blueprint":

            return {
                "query": query,
                "analysis": state.get(
                    "analysis_results",
                    "",
                ),
                "papers": state.get(
                    "research_results",
                    [],
                ),
            }

        # ----------------------------------------------------
        # Trend
        # ----------------------------------------------------

        if agent_name == "trend":

            return {
                "analysis": (
                    state.get(
                        "analysis_results",
                        "",
                    )
                    or state.get(
                        "retrieval_context",
                        "",
                    )
                    or self._paper_context(
                        state
                    )
                    or query
                )
            }

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        if agent_name == "validation":

            return {
                "blueprint": state.get(
                    "blueprint_output",
                    "",
                )
            }

        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------

        if agent_name == "report":

            payload = {
                "action": "report",
                "topic": query,
                "analysis": state.get(
                    "analysis_results",
                    "",
                ),
                "debate": state.get(
                    "debate_output",
                    "",
                ),
                "hypotheses": state.get(
                    "hypotheses",
                    "",
                ),
                "advice": state.get(
                    "advice_output",
                    "",
                ),
                "retrieval_context": state.get(
                    "retrieval_context",
                    "",
                ),
                "evidence": state.get(
                    "research_results",
                    [],
                ),
                "specialist_outputs": (
                    self._specialist_outputs(
                        state
                    )
                ),
                "top_k": top_k,
                "include_sources": task.get(
                    "include_sources",
                    True,
                ),
            }

            if state.get(
                "evaluation_feedback"
            ):
                payload[
                    "revision_notes"
                ] = state.get(
                    "evaluation_feedback"
                )

            return payload

        # ----------------------------------------------------
        # Style
        # ----------------------------------------------------

        if agent_name == "style":

            return {
                "report": (
                    state.get(
                        "final_output",
                        "",
                    )
                    or state.get(
                        "draft_report",
                        "",
                    )
                ),
                "style": task.get(
                    "style",
                    "Professional",
                ),
            }

        # ----------------------------------------------------
        # Podcast
        # ----------------------------------------------------

        if agent_name == "podcast":

            return {
                "report": (
                    state.get(
                        "final_output",
                        "",
                    )
                    or state.get(
                        "draft_report",
                        "",
                    )
                )
            }

        return {
            "query": query
        }

    # ========================================================
    # Result application
    # ========================================================

    def _apply_agent_result(
        self,
        agent_name: str,
        result: dict,
        state: ResearchState,
        step: dict,
    ) -> ResearchState:

        updates: ResearchState = {}

        agent_outputs = dict(
            state.get(
                "agent_outputs",
                {},
            )
        )

        agent_outputs[
            agent_name
        ] = result

        updates[
            "agent_outputs"
        ] = agent_outputs

        # ----------------------------------------------------
        # Errors
        # ----------------------------------------------------

        if result.get(
            "status"
        ) == "error":

            errors = dict(
                state.get(
                    "agent_errors",
                    {},
                )
            )

            errors[
                agent_name
            ] = (
                result.get("message")
                or result.get(
                    "error",
                    "Unknown error",
                )
            )

            updates[
                "agent_errors"
            ] = errors

            if result.get("stop_workflow"):
                updates["workflow_error"] = True
                updates["next_agent"] = "end"

                self.log(
                    f"[{agent_name}] Fatal error detected. "
                    "Stopping workflow immediately.",
                    level="error",
                )

                return updates

        # ----------------------------------------------------
        # Research
        # ----------------------------------------------------

        if agent_name == "research":

            papers = self._compact_papers(
                result.get(
                    "papers",
                    [],
                )
            )

            updates[
                "research_results"
            ] = papers

            updates[
                "sources"
            ] = papers

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        elif agent_name == "retrieval":

            updates[
                "retrieval_context"
            ] = result.get(
                "context",
                "",
            )

        # ----------------------------------------------------
        # Analysis
        # ----------------------------------------------------

        elif agent_name == "analysis":

            analysis = result.get(
                "analysis",
                "",
            )

            summary = result.get(
                "summary",
                "",
            )

            updates[
                "analysis_results"
            ] = (
                analysis
                or summary
            )

        # ----------------------------------------------------
        # Advisor
        # ----------------------------------------------------

        elif agent_name == "advisor":

            updates[
                "advice_output"
            ] = result.get(
                "advice",
                "",
            )

        # ----------------------------------------------------
        # Hypothesis
        # ----------------------------------------------------

        elif agent_name == "hypothesis":

            updates[
                "hypotheses"
            ] = result.get(
                "hypotheses",
                "",
            )

        # ----------------------------------------------------
        # Debate
        # ----------------------------------------------------

        elif agent_name == "debate":

            updates[
                "debate_output"
            ] = result.get(
                "debate",
                "",
            )

        # ----------------------------------------------------
        # Code integrity
        # ----------------------------------------------------

        elif agent_name == "code_integrity":

            updates[
                "code_integrity_results"
            ] = result.get(
                "code_updates",
                [],
            )

        # ----------------------------------------------------
        # Blueprint
        # ----------------------------------------------------

        elif agent_name == "blueprint":

            updates[
                "blueprint_output"
            ] = result.get(
                "blueprint",
                "",
            )

        # ----------------------------------------------------
        # Trend
        # ----------------------------------------------------

        elif agent_name == "trend":

            updates[
                "trend_output"
            ] = result.get(
                "forecast",
                "",
            )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        elif agent_name == "validation":

            updates[
                "validation_output"
            ] = result.get(
                "environment_spec",
                "",
            )

        # ----------------------------------------------------
        # Report
        # ----------------------------------------------------

        elif agent_name == "report":

            report = result.get(
                "report",
                "",
            )

            updates[
                "draft_report"
            ] = report

            updates[
                "final_output"
            ] = report

            if result.get(
                "sources"
            ):
                updates[
                    "sources"
                ] = result.get(
                    "sources",
                    [],
                )

        # ----------------------------------------------------
        # Style
        # ----------------------------------------------------

        elif agent_name == "style":

            styled = result.get(
                "styled_report",
                state.get(
                    "final_output",
                    "",
                ),
            )

            updates[
                "styled_report"
            ] = styled

            updates[
                "final_output"
            ] = styled

        # ----------------------------------------------------
        # Podcast
        # ----------------------------------------------------

        elif agent_name == "podcast":

            script = result.get(
                "script",
                "",
            )

            updates[
                "podcast_script"
            ] = script

            updates[
                "final_output"
            ] = (
                script
                or state.get(
                    "final_output",
                    "",
                )
            )

        # ----------------------------------------------------
        # Advance plan
        # ----------------------------------------------------

        updates[
            "current_step"
        ] = (
            state.get(
                "current_step",
                0,
            )
            + 1
        )

        updates[
            "shared_memory"
        ] = self._remember(
            {
                **state,
                **updates,
            },
            "agent_result",
            {
                "agent": agent_name,
                "required": step.get(
                    "required",
                    True,
                ),
                "status": result.get(
                    "status",
                    "unknown",
                ),
                "keys": sorted(
                    result.keys()
                ),
            },
        )

        return updates

    # ========================================================
    # Evaluation
    # ========================================================

    def _evaluate_node(
        self,
        state: ResearchState,
    ) -> ResearchState:

        self.log(
            "Routing to EvaluationAgent for output critique..."
        )

        output = (
            state.get(
                "final_output"
            )
            or state.get(
                "styled_report"
            )
            or state.get(
                "draft_report",
                "",
            )
        )

        result = self._safe_agent_execute(
            "evaluation",
            {
                "output_to_evaluate": output,
                "original_query": state.get(
                    "query",
                    "",
                ),
                "task_type": state.get(
                    "task",
                    {},
                ).get(
                    "style",
                    "report",
                ),
                "plan": state.get(
                    "plan",
                    [],
                ),
                "agent_outputs": state.get(
                    "agent_outputs",
                    {},
                ),
            },
        )

        if result.get("stop_workflow"):
            self.log(
                "[evaluation] Fatal error detected. "
                "Stopping workflow immediately.",
                level="error",
            )
            return {
                "workflow_error": True,
                "evaluation_feedback": (
                    result.get("feedback")
                    or result.get("message")
                    or "Evaluation stopped due to fatal error."
                ),
                "evaluation_score": 0,
                "evaluation_passed": False,
                "iterations": (
                    state.get(
                        "iterations",
                        0,
                    )
                    + 1
                ),
                "final_output": output,
                "next_agent": "end",
            }

        passed = result.get(
            "pass",
            False,
        )

        score = result.get(
            "score",
            0,
        )

        iterations = (
            state.get(
                "iterations",
                0,
            )
            + 1
        )

        reached_limit = (
            iterations
            >= self.MAX_REFLECTIONS
        )

        feedback = result.get(
            "feedback",
            "",
        )

        from config.settings import settings
        is_demo = getattr(settings, "demo_gemini_only", False)

        if passed or reached_limit or is_demo:

            if (
                reached_limit
                and not passed
                and not is_demo
            ):
                self.log(
                    "Evaluation still failing after "
                    f"{iterations} passes. "
                    "Finalizing best available output.",
                    level="warning",
                )
            else:
                self.log(
                    f"Evaluation passed with score "
                    f"{score}/10."
                )

            return {
                "evaluation_feedback": feedback,
                "evaluation_score": score,
                "evaluation_passed": bool(
                    passed
                ),
                "iterations": iterations,
                "final_output": output,
                "next_agent": "end",
                "shared_memory": self._remember(
                    state,
                    "evaluation",
                    {
                        "passed": passed,
                        "score": score,
                        "feedback": feedback,
                    },
                ),
            }

        self.log(
            f"Evaluation failed with score "
            f"{score}/10. Replanning with feedback."
        )

        return {
            "evaluation_feedback": feedback,
            "evaluation_score": score,
            "evaluation_passed": False,
            "iterations": iterations,
            "next_agent": "planner",
            "shared_memory": self._remember(
                state,
                "evaluation",
                {
                    "passed": passed,
                    "score": score,
                    "feedback": feedback,
                },
            ),
        }

    # ========================================================
    # Routing
    # ========================================================

    def _route_plan(
        self,
        state: ResearchState,
    ) -> str:

        # ----------------------------------------------------
        # Fatal workflow error
        # ----------------------------------------------------
        # If an agent hit an unrecoverable API/LLM error
        # (for example Gemini 429 quota exhaustion), stop
        # the LangGraph workflow immediately.
        if state.get("workflow_error"):
            self.log(
                "Fatal workflow error detected. "
                "Stopping workflow.",
                level="error",
            )
            return END

        plan = state.get(
            "plan",
            [],
        )

        index = state.get(
            "current_step",
            0,
        )

        if index >= len(plan):
            return "evaluate"

        agent = plan[index].get(
            "agent"
        )

        if agent in self.PLAN_AGENT_NAMES:
            return agent

        self.log(
            f"Planner selected invalid agent "
            f"'{agent}', skipping.",
            level="warning",
        )

        return "evaluate"

    def _route_after_evaluation(
        self,
        state: ResearchState,
    ) -> str:

        if state.get("workflow_error"):
            return END

        if (
            state.get(
                "next_agent"
            )
            == "planner"
        ):
            return "planner"

        return END

    # ========================================================
    # LangGraph construction
    # ========================================================

    def _build_graph(self):

        workflow = StateGraph(
            ResearchState
        )

        workflow.add_node(
            "intent",
            self._intent_node,
        )

        workflow.add_node(
            "planner",
            self._planner_node,
        )

        workflow.add_node(
            "research",
            self._research_node,
        )

        workflow.add_node(
            "retrieval",
            self._retrieval_node,
        )

        workflow.add_node(
            "analysis",
            self._analysis_node,
        )

        workflow.add_node(
            "advisor",
            self._advisor_node,
        )

        workflow.add_node(
            "hypothesis",
            self._hypothesis_node,
        )

        workflow.add_node(
            "debate",
            self._debate_node,
        )

        workflow.add_node(
            "code_integrity",
            self._code_integrity_node,
        )

        workflow.add_node(
            "blueprint",
            self._blueprint_node,
        )

        workflow.add_node(
            "trend",
            self._trend_node,
        )

        workflow.add_node(
            "validation",
            self._validation_node,
        )

        workflow.add_node(
            "report",
            self._report_node,
        )

        workflow.add_node(
            "style",
            self._style_node,
        )

        workflow.add_node(
            "podcast",
            self._podcast_node,
        )

        workflow.add_node(
            "evaluate",
            self._evaluate_node,
        )

        workflow.set_entry_point(
            "intent"
        )

        workflow.add_edge(
            "intent",
            "planner",
        )

        workflow.add_conditional_edges(
            "planner",
            self._route_plan,
        )

        for agent_name in self.PLAN_AGENT_NAMES:

            workflow.add_conditional_edges(
                agent_name,
                self._route_plan,
            )

        workflow.add_conditional_edges(
            "evaluate",
            self._route_after_evaluation,
        )

        return workflow.compile()

    # ========================================================
    # Public API
    # ========================================================

    def execute(
        self,
        task: dict,
    ) -> dict:

        action = task.get(
            "action",
            "",
        )

        if not action:
            return {
                "status": "error",
                "message": "No action specified",
            }

        self.log(
            f"Received action: {action}"
        )

        # ----------------------------------------------------
        # Legacy actions
        # ----------------------------------------------------

        if action in [
            "search",
            "query",
            "analyze",
            "advise",
            "ingest",
        ]:
            return self._fallback_legacy_routing(
                task
            )

        # ----------------------------------------------------
        # Dynamic workflow
        # ----------------------------------------------------

        if action in (
            "full_pipeline",
            "report",
        ):

            query = (
                task.get(
                    "query",
                    "",
                )
                or task.get(
                    "topic",
                    "",
                )
            )

            if not query:

                return {
                    "status": "error",
                    "message": "No query provided",
                }

            self._ensure_graph()

            from config.settings import CallBudgetTracker
            CallBudgetTracker.reset()
            if getattr(settings, "demo_gemini_only", False):
                self.log("[DEMO] Gemini call budget optimization ENABLED")

            initial_state: ResearchState = {

                "query": query,

                "action": action,

                "task": task,

                "intent_filters": {},

                "plan": [],

                "plan_rationale": "",

                "current_step": 0,

                "agent_outputs": {},

                "agent_errors": {},

                "shared_memory": [],

                "research_results": [],

                "retrieval_context": "",

                "hypotheses": "",

                "analysis_results": "",

                "advice_output": "",

                "debate_output": "",

                "code_integrity_results": [],

                "blueprint_output": "",

                "trend_output": "",

                "validation_output": "",

                "draft_report": "",

                "styled_report": "",

                "podcast_script": "",

                "final_output": "",

                "sources": [],

                "evaluation_feedback": "",

                "evaluation_score": 0,

                "evaluation_passed": False,

                "iterations": 0,

                "next_agent": "planner",

                "workflow_error": False,
            }

            try:

                final_state = (
                    self.graph.invoke(
                        initial_state
                    )
                )

                self.log(
                    "Dynamic LangGraph workflow completed."
                )

                if getattr(settings, "demo_gemini_only", False):
                    self.log(
                        f"[DEMO] Total Gemini generation calls: {CallBudgetTracker.get_count()}"
                    )

                return self._format_response(
                    final_state
                )

            except Exception as e:

                self.log(
                    "Dynamic LangGraph execution failed: "
                    f"{e}",
                    level="error",
                )

                return {
                    "status": "error",
                    "message": str(e),
                }

        return {
            "status": "error",
            "message": (
                f"Unknown action: {action}"
            ),
        }

    # ========================================================
    # Response formatting
    # ========================================================

    def _format_response(
        self,
        state: ResearchState,
    ) -> dict:

        agent_outputs = state.get(
            "agent_outputs",
            {},
        )

        final_report = (
            state.get("final_output", "")
            or state.get("styled_report", "")
            or state.get("draft_report", "")
            or state.get("analysis_results", "")
        )

        is_error = state.get("workflow_error", False)
        status_str = "error" if (is_error and not final_report) else "success"

        return {
            "status": status_str,
            "message": "Workflow stopped due to rate limit/quota." if is_error else "",

            "query": state.get(
                "query",
                "",
            ),

            "report": final_report,

            "sources": state.get(
                "sources",
                state.get(
                    "research_results",
                    [],
                ),
            ),

            "iterations": state.get(
                "iterations",
                1,
            ),

            "evaluation_score": state.get(
                "evaluation_score",
                0,
            ),

            "evaluation_passed": state.get(
                "evaluation_passed",
                False,
            ),

            "plan": state.get(
                "plan",
                [],
            ),

            "shared_memory": state.get(
                "shared_memory",
                [],
            ),

            "stages": {

                "plan": {
                    "rationale": state.get(
                        "plan_rationale",
                        "",
                    ),
                    "steps": state.get(
                        "plan",
                        [],
                    ),
                },

                "intent": state.get(
                    "intent_filters",
                    {},
                ),

                "research": {
                    "papers": state.get(
                        "research_results",
                        [],
                    )
                },

                "retrieval": {
                    "context": state.get(
                        "retrieval_context",
                        "",
                    )
                },

                "hypotheses": {
                    "hypotheses": state.get(
                        "hypotheses",
                        "",
                    )
                },

                "analysis": {
                    "analysis": state.get(
                        "analysis_results",
                        "",
                    )
                },

                "advisor": {
                    "advice": state.get(
                        "advice_output",
                        "",
                    )
                },

                "debate": {
                    "debate": state.get(
                        "debate_output",
                        "",
                    )
                },

                "code_integrity": {
                    "code_updates": state.get(
                        "code_integrity_results",
                        [],
                    )
                },

                "blueprint": {
                    "blueprint": state.get(
                        "blueprint_output",
                        "",
                    )
                },

                "trends": {
                    "forecast": state.get(
                        "trend_output",
                        "",
                    )
                },

                "validation": {
                    "environment_spec": state.get(
                        "validation_output",
                        "",
                    )
                },

                "podcast": {
                    "script": state.get(
                        "podcast_script",
                        "",
                    )
                },

                "feedback": state.get(
                    "evaluation_feedback",
                    "",
                ),

                "agent_outputs": agent_outputs,

                "agent_errors": state.get(
                    "agent_errors",
                    {},
                ),
            },
        }

    # ========================================================
    # Legacy routing
    # ========================================================

    def _fallback_legacy_routing(
        self,
        task: dict,
    ) -> dict:

        action = task.get(
            "action"
        )

        if action in (
            "search",
            "ingest",
        ):

            return self._get_agent(
                "research"
            )._safe_execute(
                task
            )

        if action == "query":

            return self._get_agent(
                "retrieval"
            )._safe_execute(
                task
            )

        if action == "analyze":

            return self._get_agent(
                "analysis"
            )._safe_execute(
                task
            )

        if action == "advise":

            return self._get_agent(
                "advisor"
            )._safe_execute(
                task
            )

        return {
            "status": "error",
            "message": (
                f"Unknown legacy action: {action}"
            ),
        }