"""
API route definitions for the FastAPI backend.
"""

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
from api.websocket_manager import manager as ws_manager
from agents.base_agent import set_ws_manager

# Set the websocket manager for agents
set_ws_manager(ws_manager)


from api.models import (
    SearchRequest, SearchResponse,
    QueryRequest, QueryResponse,
    ReportRequest, ReportResponse,
    PapersListResponse, ErrorResponse,
    IngestRequest, HealthResponse,
)
from agents.orchestrator_agent import OrchestratorAgent
from rag.rag_chain import RAGChain
from rag.vector_store import VectorStore
from collectors.paper_manager import PaperManager
from analytics.trend_detector import TrendDetector
from graph.knowledge_graph import KnowledgeGraph

router = APIRouter(prefix="/api")

# Lazy singletons
_orchestrator = None
_rag_chain = None
_paper_manager = None
_trend_detector = None


def get_orchestrator():
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = OrchestratorAgent()
    return _orchestrator


def get_rag_chain():
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


def get_paper_manager():
    global _paper_manager
    if _paper_manager is None:
        _paper_manager = PaperManager()
    return _paper_manager


def get_trend_detector():
    global _trend_detector
    if _trend_detector is None:
        _trend_detector = TrendDetector()
    return _trend_detector


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        vs = VectorStore()
        stats = vs.get_collection_stats()
        doc_count = stats.get("document_count", 0)
    except Exception:
        doc_count = 0

    return HealthResponse(
        status="healthy",
        vector_db_documents=doc_count,
    )


@router.post("/search", response_model=SearchResponse)
async def search_papers(request: SearchRequest):
    """Search for research papers across all sources."""
    try:
        orchestrator = OrchestratorAgent(llm_provider=request.llm_provider)
        result = await run_in_threadpool(orchestrator.execute, {
            "action": "search",
            "query": request.query,
            "max_papers": request.max_papers,
            "sources": request.sources,
            "download": request.download,
            "process": request.process,
        })

        return SearchResponse(
            status=result.get("status", "error"),
            query=request.query,
            papers_found=result.get("papers_found", 0),
            papers_downloaded=result.get("papers_downloaded", 0),
            papers_processed=result.get("papers_processed", 0),
            papers=result.get("papers", []),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG knowledge base."""
    try:
        rag = RAGChain(llm_provider=request.llm_provider)
        result = await run_in_threadpool(rag.query, request.query, top_k=request.top_k)

        return QueryResponse(
            answer=result.get("answer", ""),
            sources=result.get("sources", []),
            context_used=result.get("context_used", 0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/report", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """Generate a structured research report."""
    try:
        orchestrator = OrchestratorAgent(llm_provider=request.llm_provider)
        result = await run_in_threadpool(orchestrator.execute, {
            "action": "report",
            "topic": request.topic,
            "top_k": request.top_k,
            "style": request.style,
            "include_sources": request.include_sources,
        })

        return ReportResponse(
            status=result.get("status", "error"),
            topic=request.topic,
            report=result.get("report", ""),
            sources=result.get("sources", []),
            results=result.get("stages", {})
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/papers", response_model=PapersListResponse)
async def list_papers():
    """List all collected papers and database stats."""
    try:
        pm = get_paper_manager()
        metadata_files = pm.list_all_metadata()

        vs = VectorStore()
        stats = vs.get_collection_stats()

        return PapersListResponse(
            metadata_files=metadata_files,
            vector_db_stats=stats,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest")
async def ingest_papers(request: IngestRequest):
    """Trigger paper ingestion pipeline."""
    try:
        orchestrator = get_orchestrator()
        result = await run_in_threadpool(orchestrator.execute, {
            "action": "ingest",
            "query": request.query,
            "paper_ids": request.paper_ids,
            "download": True,
            "process": True,
        })
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/analyze")
async def analyze_papers(request: QueryRequest):
    """Analyze papers on a topic."""
    try:
        orchestrator = get_orchestrator()
        result = await run_in_threadpool(orchestrator.execute, {
            "action": "analyze",
            "query": request.query,
            "analysis_type": "both",
            "top_k": request.top_k,
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/advise")
async def get_research_advice(request: QueryRequest):
    """Get research advice and gap analysis."""
    try:
        orchestrator = get_orchestrator()
        result = await run_in_threadpool(orchestrator.execute, {
            "action": "advise",
            "topic": request.query,
            "top_k": request.top_k,
        })
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends")
async def get_trends():
    """Get research trend analysis."""
    try:
        td = get_trend_detector()
        trends = await run_in_threadpool(td.detect_trending_topics)
        timeline = await run_in_threadpool(td.get_publication_timeline)
        top_papers = await run_in_threadpool(td.get_most_referenced_papers)

        return {
            "trends": trends,
            "timeline": timeline,
            "top_papers": top_papers,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/trends/{topic}")
async def get_topic_trends(topic: str):
    """Get research trends for a specific topic."""
    try:
        td = get_trend_detector()
        return td.detect_topic_trends(topic)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/leaderboard")
async def get_impact_leaderboard():
    """Get globally impactful papers across all subspaces."""
    try:
        td = get_trend_detector()
        leaderboard = td.get_global_leaderboard()
        return {"status": "success", "leaderboard": leaderboard}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/analytics/graph")
async def get_knowledge_graph_data():
    """Get nodes and links for Knowledge Graph visualization."""
    try:
        kg = KnowledgeGraph()
        if not kg.available:
            # Fallback to a small informative graph if Neo4j is down
            return {
                "nodes": [{"id": "System", "group": 1, "label": "Neo4j Offline"}],
                "links": [],
                "error": "Knowledge Graph database is currently unavailable."
            }
        
        graph_data = await run_in_threadpool(kg.get_full_graph, limit=150)
        return graph_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.websocket("/ws/logs")

async def websocket_logs(websocket: WebSocket):
    """WebSocket for real-time agent logs."""
    await ws_manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)


@router.get("/analytics/embeddings")
async def get_embeddings_3d_view():
    """Get 3D dimensionality reduction vectors for ChromaDB documents."""
    try:
        vs = VectorStore()
        result = await run_in_threadpool(vs.get_embeddings_3d, limit=200)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def upload_pdf(file: list[UploadFile] = File(...)):
    """Upload and process local PDFs."""
    from processing.pipeline import ProcessingPipeline
    from config.settings import settings
    
    pipeline = ProcessingPipeline()
    results = []
    
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    for pdf in file:
        file_path = upload_dir / pdf.filename
        try:
            with open(file_path, "wb") as buffer:
                content = await pdf.read()
                buffer.write(content)
            
            result = pipeline.process_paper(
                file_path, 
                paper_metadata={
                    "title": pdf.filename.replace(".pdf", ""),
                    "source": "local_upload",
                    "published": "Local File"
                }
            )
            results.append(result)
        except Exception as e:
            results.append({"status": "error", "filename": pdf.filename, "message": str(e)})
            
    return {"status": "success", "results": results}
