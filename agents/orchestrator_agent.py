"""
Orchestrator Agent - Coordinates all agents and task flow.
"""

from agents.base_agent import BaseAgent
from agents.research_agent import ResearchAgent
from agents.retrieval_agent import RetrievalAgent
from agents.analysis_agent import AnalysisAgent
from agents.report_agent import ReportAgent
from agents.advisor_agent import AdvisorAgent
from agents.intent_agent import IntentAgent
from agents.code_integrity_agent import CodeIntegrityAgent
from agents.blueprint_agent import BlueprintAgent
from agents.debate_agent import DebateAgent
from agents.hypothesis_agent import HypothesisAgent
from agents.style_agent import StyleAgent
from agents.podcast_agent import PodcastAgent
from agents.trend_agent import TrendAgent
from agents.validation_service import ValidationService


class OrchestratorAgent(BaseAgent):
    """
    Master orchestrator that routes tasks to appropriate agents
    and coordinates multi-step workflows.
    """

    def __init__(self, llm_provider: str = None):
        super().__init__("OrchestratorAgent")
        self.llm_provider = llm_provider
        self._agents = {}

    def _get_agent(self, name: str) -> BaseAgent:
        """Lazy-initialize agents on demand."""
        if name not in self._agents:
            agent_map = {
                "research": ResearchAgent,
                "retrieval": RetrievalAgent,
                "analysis": AnalysisAgent,
                "report": ReportAgent,
                "advisor": AdvisorAgent,
                "intent": IntentAgent,
                "code_integrity": CodeIntegrityAgent,
                "blueprint": BlueprintAgent,
                "debate": DebateAgent,
                "hypothesis": HypothesisAgent,
                "style": StyleAgent,
                "podcast": PodcastAgent,
                "trend": TrendAgent,
                "validation": ValidationService,
            }
            if name in agent_map:
                self._agents[name] = agent_map[name](llm_provider=self.llm_provider)
            else:
                raise ValueError(f"Unknown agent: {name}")
        return self._agents[name]

    def execute(self, task: dict) -> dict:
        """
        Route and execute tasks.

        Task keys:
            - action (str): What to do:
                'search' → ResearchAgent
                'query' → RetrievalAgent + RAG
                'analyze' → AnalysisAgent
                'report' → ReportAgent
                'advise' → AdvisorAgent
                'ingest' → ResearchAgent (selective)
                'full_pipeline' → Full research pipeline

            - Additional keys passed to the appropriate agent.
        """
        action = task.get("action", "")
        if not action:
            return {"status": "error", "message": "No action specified"}

        self.log(f"Routing action: {action}")

        if action == "search":
            return self._handle_search(task)
        elif action == "query":
            return self._handle_query(task)
        elif action == "analyze":
            return self._handle_analysis(task)
        elif action == "report":
            return self._handle_report(task)
        elif action == "advise":
            return self._handle_advise(task)
        elif action == "ingest":
            return self._handle_ingest(task)
        elif action == "full_pipeline":
            return self._handle_full_pipeline(task)

        else:
            return {"status": "error", "message": f"Unknown action: {action}"}

    def _handle_search(self, task: dict) -> dict:
        """Handle paper search task."""
        agent = self._get_agent("research")
        return agent._safe_execute(task)

    def _handle_query(self, task: dict) -> dict:
        """Handle RAG query task."""
        agent = self._get_agent("retrieval")
        result = agent._safe_execute(task)

        # If query format is 'context', also generate LLM answer
        if task.get("generate_answer", True):
            from rag.rag_chain import RAGChain
            rag = RAGChain()
            rag_result = rag.query(
                task.get("query", ""), 
                top_k=task.get("top_k", 5),
                complexity=task.get("complexity", "standard")
            )
            result["answer"] = rag_result.get("answer", "")
            result["sources"] = rag_result.get("sources", [])

        return result

    def _handle_analysis(self, task: dict) -> dict:
        """Handle paper analysis task."""
        agent = self._get_agent("analysis")
        return agent._safe_execute(task)

    def _handle_report(self, task: dict) -> dict:
        """
        Handle report generation task.
        In Phase 6, this is now a full Deep Synthesis pipeline.
        """
        # Map 'topic' to 'query' for full pipeline consistency
        if "topic" in task and "query" not in task:
            task["query"] = task["topic"]
        return self._handle_full_pipeline(task)

    def _handle_advise(self, task: dict) -> dict:
        """Handle research advising task."""
        agent = self._get_agent("advisor")
        return agent._safe_execute(task)

    def _handle_ingest(self, task: dict) -> dict:
        """Handle selective paper ingestion."""
        agent = self._get_agent("research")
        return agent._safe_execute(task)


    def _handle_full_pipeline(self, task: dict) -> dict:
        """
        Execute a full research pipeline.
        """
        query = task.get("query", "")
        if not query:
            return {"status": "error", "message": "No query provided"}

        self.log(f"Running full pipeline for: '{query}'")
        
        # Initialize result containers
        pipeline_stages = {}
        analysis_result = {}
        blueprint_result = {}
        base_report = ""
        
        # Stage 1: Research
        self.log("Stage 1: Intent Parsing & Research")
        
        # Step 1a: Extract intent
        intent_agent = self._get_agent("intent")
        intent_result = intent_agent._safe_execute({"query": query})
        filters = intent_result.get("filters", {})
        refined_query = filters.get("topic", query)
        
        research_result = self._handle_search({
            "action": "search",
            "query": refined_query,
            "filters": filters,
            "max_papers": task.get("max_papers", 5),
            "download": True,
            "process": True,
        })
        pipeline_stages["research"] = {
            "papers_found": research_result.get("papers_found", 0),
            "papers_processed": research_result.get("papers_processed", 0),
        }

        # Step 1b: Code Integrity
        self.log("Stage 1b: Identifying official code repositories")
        papers = research_result.get("papers", [])
        if papers:
            code_result = self._get_agent("code_integrity")._safe_execute({"papers": papers})
            code_lookup = {r["title"]: r["code_meta"] for r in code_result.get("code_updates", [])}
            for p in research_result.get("papers", []):
                if p["title"] in code_lookup:
                    p["code_meta"] = code_lookup[p["title"]]

        # Step 2: Analysis & Blueprinting
        self.log("Stage 2: Analyzing papers and generating blueprint")
        try:
            analysis_result = self._handle_analysis({
                "action": "analyze",
                "query": query,
                "analysis_type": "both",
                "complexity": task.get("complexity", "standard"),
            })
            pipeline_stages["analysis"] = {
                "analysis": analysis_result.get("analysis", ""),
                "summary": analysis_result.get("summary", ""),
            }
            
            # Sub-stage: Blueprint
            self.log("Stage 2b: Generating Implementation Blueprint")
            blueprint_result = self._get_agent("blueprint")._safe_execute({"query": query})
            pipeline_stages["blueprint"] = {
                "blueprint": blueprint_result.get("blueprint", ""),
            }

            # Phase 6: Sub-stage: Validation
            self.log("Stage 2c: Generating Autonomous Validation Environment")
            val_result = self._get_agent("validation")._safe_execute({"blueprint": blueprint_result.get("blueprint", "")})
            pipeline_stages["validation"] = val_result
        except Exception as e:
            pipeline_stages["analysis"] = {"error": str(e)}

        # Stage 3: Report & Debate
        self.log("Stage 3: Generating report and critical debate")
        try:
            # Stage 3a: Report Generation (Direct Agent Call)
            report_agent = self._get_agent("report")
            report_result = report_agent._safe_execute({
                "action": "report",
                "topic": query,
            })
            base_report = report_result.get("report", "")
            
            # Apply Personalization Style if requested
            requested_style = task.get("style", "Professional")
            if requested_style != "Professional":
                self.log(f"Stage 3-Style: Adapting to {requested_style}")
                style_result = self._get_agent("style")._safe_execute({
                    "report": base_report,
                    "style": requested_style
                })
                base_report = style_result.get("styled_report", base_report)

            pipeline_stages["report"] = {
                "report": base_report,
            }
            
            # Sub-stage: Debate
            self.log("Stage 3b: Performing Scientific Debate")
            debate_result = self._get_agent("debate")._safe_execute({"query": query})
            pipeline_stages["debate"] = {
                "debate": debate_result.get("debate", ""),
            }

            # Sub-stage: Hypotheses
            self.log("Stage 3c: Generating Novel Hypotheses")
            hyp_result = self._get_agent("hypothesis")._safe_execute({"query": query})
            pipeline_stages["hypotheses"] = {
                "hypotheses": hyp_result.get("hypotheses", ""),
            }

            # Phase 6: Sub-stage: Trend Prediction
            self.log("Stage 3d: Forecasting Research Impact")
            trend_result = self._get_agent("trend")._safe_execute({"analysis": analysis_result.get("analysis", "")})
            pipeline_stages["trends"] = trend_result

            # Phase 6: Sub-stage: Podcast
            if requested_style == "Podcast Script":
                self.log("Stage 3e: Drafting Podcast Script")
                podcast_result = self._get_agent("podcast")._safe_execute({"report": base_report})
                pipeline_stages["podcast"] = podcast_result
        except Exception as e:
            pipeline_stages["report"] = {"error": str(e)}

        # Stage 4: Advice
        self.log("Stage 4: Generating research advice")
        try:
            advice_result = self._handle_advise({
                "action": "advise",
                "topic": query,
            })
            pipeline_stages["advice"] = {
                "advice": advice_result.get("advice", ""),
            }
        except Exception as e:
            pipeline_stages["advice"] = {"error": str(e)}

        self.log("Full pipeline completed")
        return {
            "status": "success",
            "query": query,
            "report": base_report,
            "sources": research_result.get("papers", []),
            "stages": pipeline_stages
        }
