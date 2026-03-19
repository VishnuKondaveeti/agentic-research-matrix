"""
Webhook endpoints for n8n automation integration.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks

from api.models import WebhookTopicRequest
from agents.orchestrator_agent import OrchestratorAgent

webhook_router = APIRouter(prefix="/webhook")


def _run_research_pipeline(topic: str, max_papers: int):
    """Background task for research pipeline."""
    try:
        orchestrator = OrchestratorAgent()
        orchestrator.execute({
            "action": "full_pipeline",
            "query": topic,
            "max_papers": max_papers,
        })
    except Exception as e:
        print(f"[Webhook] Pipeline error: {e}")


def _run_report_generation(topic: str):
    """Background task for report generation."""
    try:
        orchestrator = OrchestratorAgent()
        orchestrator.execute({
            "action": "report",
            "topic": topic,
        })
    except Exception as e:
        print(f"[Webhook] Report error: {e}")


@webhook_router.post("/new-topic")
async def webhook_new_topic(
    request: WebhookTopicRequest,
    background_tasks: BackgroundTasks,
):
    """
    Webhook for n8n: triggered when a new research topic is submitted.
    Runs the full research pipeline in the background.
    """
    background_tasks.add_task(
        _run_research_pipeline,
        request.topic,
        request.max_papers,
    )
    return {
        "status": "accepted",
        "message": f"Research pipeline started for: {request.topic}",
    }


@webhook_router.post("/ingest")
async def webhook_ingest(
    request: WebhookTopicRequest,
    background_tasks: BackgroundTasks,
):
    """Webhook for n8n: trigger paper ingestion."""
    background_tasks.add_task(
        _run_research_pipeline,
        request.topic,
        request.max_papers,
    )
    return {
        "status": "accepted",
        "message": f"Ingestion started for: {request.topic}",
    }


@webhook_router.post("/report")
async def webhook_report(
    request: WebhookTopicRequest,
    background_tasks: BackgroundTasks,
):
    """Webhook for n8n: trigger report generation."""
    background_tasks.add_task(_run_report_generation, request.topic)
    return {
        "status": "accepted",
        "message": f"Report generation started for: {request.topic}",
    }
