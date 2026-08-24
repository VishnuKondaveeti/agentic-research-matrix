import asyncio
from typing import List
from fastapi import WebSocket


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.loop: asyncio.AbstractEventLoop | None = None

    async def connect(self, websocket: WebSocket):
        await websocket.accept()

        # Store the main FastAPI/Uvicorn event loop.
        self.loop = asyncio.get_running_loop()

        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """
        Send a message to every connected WebSocket client.
        """

        dead_connections = []

        for connection in self.active_connections:
            try:
                await connection.send_json(message)

            except Exception as e:
                print(f"[WebSocket] Connection failed: {e}")
                dead_connections.append(connection)

        # Remove dead connections
        for connection in dead_connections:
            self.disconnect(connection)

    def broadcast_from_thread(self, message: dict):
        """
        Thread-safe WebSocket broadcast.

        Agent execution happens inside FastAPI's threadpool,
        so synchronous agent code cannot directly await broadcast().
        """

        if self.loop is None:
            return

        if not self.loop.is_running():
            return

        try:
            asyncio.run_coroutine_threadsafe(
                self.broadcast(message),
                self.loop,
            )

        except Exception as e:
            print(f"[WebSocket] Thread broadcast failed: {e}")


manager = ConnectionManager()