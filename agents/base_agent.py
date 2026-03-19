"""
Abstract base agent class.
All agents inherit from this and implement execute().
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

from logs.system_logger import get_logger
import asyncio

# Global flag to enable/disable websocket broadcasting
_ws_manager = None

def set_ws_manager(manager):
    global _ws_manager
    _ws_manager = manager



class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(self, name: str, llm_provider: str = None):
        self.name = name
        self.llm_provider = llm_provider
        self.logger = get_logger(f"agent.{name}")

    @abstractmethod
    def execute(self, task: dict) -> dict:
        """
        Execute the agent's primary task.

        Args:
            task: Dict describing what the agent should do.
                  Expected keys vary by agent type.

        Returns:
            Dict with execution results.
        """
        pass

    def log(self, message: str, level: str = "info"):
        """Log a message with the agent's name and broadcast it."""
        log_func = getattr(self.logger, level, self.logger.info)
        log_msg = f"[{self.name}] {message}"
        log_func(log_msg)
        
        # Broadcast to websocket
        if _ws_manager:
            try:
                # Since log might be called from sync code, we need to handle the event loop
                data = {
                    "type": "log",
                    "agent": self.name,
                    "message": message,
                    "level": level
                }
                # Check if there is a running loop
                try:
                    loop = asyncio.get_running_loop()
                    if loop.is_running():
                        asyncio.create_task(_ws_manager.broadcast(data))
                except RuntimeError:
                    # No running loop, might be in a thread or startup
                    pass
            except Exception:
                pass


    def _safe_execute(self, task: dict) -> dict:
        """Wrapper with error handling."""
        try:
            self.log(f"Starting task: {task.get('action', 'unknown')}")
            result = self.execute(task)
            self.log(f"Task completed: {task.get('action', 'unknown')}")
            return result
        except Exception as e:
            self.log(f"Task failed: {e}", level="error")
            return {
                "status": "error",
                "agent": self.name,
                "error": str(e),
            }
