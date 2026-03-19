"""
Pydantic request/response models for the API.
"""

from pydantic import BaseModel, Field
from typing import Optional


# ── Request Models ──

class SearchRequest(BaseModel):
    query: str = Field(..., description="Research topic to search for")
    max_papers: int = Field(5, ge=1, le=50, description="Max papers per source")
    complexity: str = Field("standard", description="Complexity level ('beginner', 'expert', 'standard')")
    sources: Optional[list[str]] = Field(None, description="Sources to search: arxiv, semantic_scholar, core")
    download: bool = Field(True, description="Whether to download PDFs")
    process: bool = Field(True, description="Whether to process into vector DB")
    llm_provider: Optional[str] = Field(None, description="LLM provider override ('gemini', 'ollama', 'openai')")


class QueryRequest(BaseModel):
    query: str = Field(..., description="Question to ask the research knowledge base")
    top_k: int = Field(5, ge=1, le=20, description="Number of context chunks to retrieve")
    complexity: str = Field("standard", description="Complexity level ('beginner', 'expert', 'standard')")
    llm_provider: Optional[str] = Field(None, description="LLM provider override")


class ReportRequest(BaseModel):
    topic: str = Field(..., description="Research topic for the report")
    top_k: int = Field(10, ge=1, le=30, description="Number of context chunks")
    complexity: str = Field("standard", description="Complexity level ('beginner', 'expert', 'standard')")
    style: Optional[str] = Field("Professional", description="Personalization style ('Expert', 'Beginner', 'Podcast', etc.)")
    include_sources: bool = Field(True, description="Include references in report")
    llm_provider: Optional[str] = Field(None, description="LLM provider override")


class IngestRequest(BaseModel):
    query: str = Field(..., description="Query to identify papers for ingestion")
    paper_ids: Optional[list[str]] = Field(None, description="List of paper IDs or titles to ingest")



class WebhookTopicRequest(BaseModel):
    topic: str = Field(..., description="Research topic")
    max_papers: int = Field(5, ge=1, le=20)


# ── Response Models ──

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
    vector_db_documents: int = 0


class SearchResponse(BaseModel):
    status: str
    query: str
    papers_found: int
    papers_downloaded: int
    papers_processed: int
    papers: list[dict] = []


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict] = []
    context_used: int = 0


class ReportResponse(BaseModel):
    status: str
    topic: str
    report: str
    sources: list[dict] = []
    results: Optional[dict] = Field(None, description="Phase 6: Advanced agent outputs (trends, validation, podcast)")


class PapersListResponse(BaseModel):
    metadata_files: list[dict] = []
    vector_db_stats: dict = {}


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
