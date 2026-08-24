"""
Central configuration using Pydantic Settings.
Loads values from .env file in the project root.
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolve project root (parent of config/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── LLM ──
    llm_provider: str = "gemini"  # "gemini", "ollama", "openai"
    google_api_key: str = ""
    llm_model: str = "gemini-2.0-flash"
    embedding_model: str = "models/embedding-001"
    demo_gemini_only: bool = True  # Enforces Gemini-only execution and disables Ollama fallback for demo
    
    # Ollama settings
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:7b"
    
    # OpenAI settings
    openai_api_key: str = ""
    openai_model: str = "gpt-4o"

    # ── Paper source APIs ──
    semantic_scholar_api_key: str = ""
    core_api_key: str = ""

    # ── Neo4j ──
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # ── ChromaDB ──
    chroma_persist_dir: str = "C:/AIData/research_chroma"

    # ── Processing ──
    chunk_size: int = 600
    chunk_overlap: int = 50

    # ── API ──
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    cors_origins: list[str] = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    ]

    # ── Logging ──
    log_level: str = "INFO"

    # ── Derived paths ──
    @property
    def papers_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "papers"

    @property
    def metadata_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "metadata"

    @property
    def chroma_path(self) -> Path:
        path = Path(self.chroma_persist_dir)

        if path.is_absolute():
            return path

        return PROJECT_ROOT / path

    @property
    def logs_dir(self) -> Path:
        return PROJECT_ROOT / "data" / "logs"


settings = Settings()

# Ensure runtime directories exist
for d in [settings.papers_dir, settings.metadata_dir, settings.chroma_path, settings.logs_dir]:
    d.mkdir(parents=True, exist_ok=True)


class CallBudgetTracker:
    """Tracks LLM calls during execution for telemetry and quota protection."""
    _count: int = 0

    @classmethod
    def record_call(cls, caller_name: str = "LLM") -> int:
        cls._count += 1
        print(f"[LLM] Gemini generation call #{cls._count} ({caller_name})")
        return cls._count

    @classmethod
    def get_count(cls) -> int:
        return cls._count

    @classmethod
    def reset(cls) -> None:
        cls._count = 0
