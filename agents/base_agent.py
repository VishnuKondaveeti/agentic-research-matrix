"""
Abstract base agent class.
All agents inherit from this and implement execute().
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from logs.system_logger import get_logger


# ============================================================
# GLOBAL WEBSOCKET MANAGER
# ============================================================

_ws_manager = None


def set_ws_manager(manager):
    """
    Register the application's WebSocket connection manager.

    This is called during FastAPI application startup.
    """
    global _ws_manager
    _ws_manager = manager


# ============================================================
# BASE AGENT
# ============================================================

class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(
        self,
        name: str,
        llm_provider: str = None,
    ):
        self.name = name
        self.llm_provider = llm_provider
        self.logger = get_logger(f"agent.{name}")

    def set_llm_provider(self, provider: str):
        """Update the LLM provider for this agent."""
        self.llm_provider = provider

    # ========================================================
    # ABSTRACT EXECUTION
    # ========================================================

    @abstractmethod
    def execute(self, task: dict) -> dict:
        """
        Execute the agent's primary task.

        Args:
            task: Dict describing what the agent should do.

        Returns:
            Dict with execution results.
        """
        pass

    # ========================================================
    # WEBSOCKET TELEMETRY
    # ========================================================

    def _broadcast_event(
        self,
        event_type: str,
        message: str,
        level: str = "info",
        channel: str = "research",
        progress: int | None = None,
    ):
        """
        Send a structured event to the frontend.

        IMPORTANT:
        Agent execution can happen inside a worker thread.
        Therefore we DO NOT use asyncio.get_running_loop()
        here.

        ConnectionManager.broadcast_from_thread() handles
        forwarding the message safely to FastAPI's main
        event loop.
        """

        if _ws_manager is None:
            return

        try:
            data = {
                "type": event_type,
                "channel": channel,
                "agent": self.name,
                "message": message,
                "level": level,
            }

            if progress is not None:
                data["progress"] = progress

            # Thread-safe handoff to FastAPI event loop.
            _ws_manager.broadcast_from_thread(data)

        except Exception as e:
            # WebSocket telemetry must NEVER break an agent.
            try:
                self.logger.debug(
                    f"[{self.name}] WebSocket telemetry failed: {e}"
                )
            except Exception:
                pass

    # ========================================================
    # NORMAL LOGGING
    # ========================================================

    def log(
        self,
        message: str,
        level: str = "info",
        channel: str = "research",
    ):
        """
        Log a message locally and send it to the UI.

        This method is safe to call from:
        - normal synchronous code
        - FastAPI worker threads
        - agent execution
        - LangGraph nodes
        """

        # ----------------------------------------------------
        # 1. Normal backend logging
        # ----------------------------------------------------

        log_func = getattr(
            self.logger,
            level,
            self.logger.info,
        )

        log_msg = f"[{self.name}] {message}"

        log_func(log_msg)

        # ----------------------------------------------------
        # 2. Real-time UI telemetry
        # ----------------------------------------------------

        self._broadcast_event(
            event_type="agent_log",
            message=message,
            level=level,
            channel=channel,
        )

    # ========================================================
    # STRUCTURED AGENT EVENTS
    # ========================================================

    def emit_event(
        self,
        event: str,
        message: str = "",
        level: str = "info",
        progress: int | None = None,
        channel: str = "research",
    ):
        """
        Emit a structured agent lifecycle event.

        Examples:

            self.emit_event(
                "started",
                "Starting research...",
                progress=10
            )

            self.emit_event(
                "running",
                "Searching arXiv...",
                progress=40
            )

            self.emit_event(
                "completed",
                "Research completed.",
                level="success",
                progress=100
            )

            self.emit_event(
                "failed",
                "Gemini request failed.",
                level="error"
            )
        """

        self._broadcast_event(
            event_type="agent_event",
            message=message,
            level=level,
            channel=channel,
            progress=progress,
        )

        # Also write lifecycle events to backend logs.
        if message:
            log_func = getattr(
                self.logger,
                level,
                self.logger.info,
            )

            log_func(
                f"[{self.name}] "
                f"{event.upper()}: "
                f"{message}"
            )

    # ========================================================
    # SAFE EXECUTION WRAPPER
    # ========================================================

    def _safe_execute(self, task: dict) -> dict:
        """
        Wrapper around execute() with error handling
        and real-time lifecycle telemetry.
        """

        action = task.get("action", "unknown")

        # ----------------------------------------------------
        # STARTED
        # ----------------------------------------------------

        self.emit_event(
            event="started",
            message=f"Starting task: {action}",
            level="info",
            progress=5,
        )

        try:

            # ------------------------------------------------
            # RUN AGENT
            # ------------------------------------------------

            result = self.execute(task)

            # ------------------------------------------------
            # COMPLETED
            # ------------------------------------------------

            self.emit_event(
                event="completed",
                message=f"Task completed: {action}",
                level="success",
                progress=100,
            )

            return result

        except Exception as e:

            # ------------------------------------------------
            # FAILED
            # ------------------------------------------------

            self.emit_event(
                event="failed",
                message=f"Task failed: {e}",
                level="error",
            )

            return {
                "status": "error",
                "agent": self.name,
                "error": str(e),
            }