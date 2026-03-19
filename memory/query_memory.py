"""
JSON-backed conversation memory with session tracking.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path

from config.settings import settings


class QueryMemory:
    """Stores conversation history and query memory per session."""

    def __init__(self, memory_dir: Path | None = None):
        self.memory_dir = memory_dir or (settings.metadata_dir / "memory")
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.current_session = str(uuid.uuid4())[:8]

    def add_interaction(
        self,
        query: str,
        response: str,
        session_id: str | None = None,
        metadata: dict | None = None,
    ):
        """Add a query-response interaction to memory."""
        session_id = session_id or self.current_session
        filepath = self.memory_dir / f"session_{session_id}.json"

        history = self._load_session(filepath)
        history.append({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "response": response[:2000],  # Truncate long responses
            "metadata": metadata or {},
        })

        filepath.write_text(
            json.dumps(history, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def get_history(
        self,
        session_id: str | None = None,
        last_n: int = 10,
    ) -> list[dict]:
        """Get conversation history for a session."""
        session_id = session_id or self.current_session
        filepath = self.memory_dir / f"session_{session_id}.json"
        history = self._load_session(filepath)
        return history[-last_n:]

    def get_context_string(
        self,
        session_id: str | None = None,
        last_n: int = 3,
    ) -> str:
        """Get recent history formatted as context for the LLM."""
        history = self.get_history(session_id, last_n)
        if not history:
            return ""

        parts = []
        for entry in history:
            parts.append(f"User: {entry['query']}")
            parts.append(f"Assistant: {entry['response'][:500]}")
        return "\n".join(parts)

    def list_sessions(self) -> list[dict]:
        """List all stored sessions."""
        sessions = []
        for f in self.memory_dir.glob("session_*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": f.stem.replace("session_", ""),
                    "interactions": len(data),
                    "last_query": data[-1]["query"] if data else "",
                })
            except (json.JSONDecodeError, IOError, IndexError):
                continue
        return sessions

    def _load_session(self, filepath: Path) -> list:
        """Load session data from file."""
        if not filepath.exists():
            return []
        try:
            return json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            return []
