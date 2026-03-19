"""
FastAPI application entry point.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import router
from api.webhooks import webhook_router

from config.settings import settings

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import time
from collections import defaultdict

app = FastAPI(
    title="Agentic Research Matrix",
    description="Autonomous Multi-Agent Research Paper Collection & Analysis API",
    version="1.0.0",
)

# Simple Rate Limiting (Production Safeguard)
RATE_LIMIT = 60  # requests per minute
request_counts = defaultdict(list)

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    client_ip = request.client.host
    now = time.time()
    
    # Filter out requests older than 1 minute
    request_counts[client_ip] = [t for t in request_counts[client_ip] if now - t < 60]
    
    if len(request_counts[client_ip]) >= RATE_LIMIT:
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Too many requests. Please try again later."}
        )
    
    request_counts[client_ip].append(now)
    return await call_next(request)

# Global Exception Handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "status": "error",
            "message": "An internal server error occurred.",
            "detail": str(exc) if not settings.log_level == "INFO" else "Check server logs for details."
        },
    )

# CORS middleware - Restrict to allowed origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(router)
app.include_router(webhook_router)

from fastapi.staticfiles import StaticFiles

@app.get("/api/info")
async def info():
    return {
        "name": "Agentic Research Matrix",
        "version": "1.0.0",

        "docs": "/docs",
        "health": "/api/health",
    }

# Mount the 'ui' directory at the root to serve index.html, styles.css, script.js
app.mount("/", StaticFiles(directory="ui", html=True), name="ui")


if __name__ == "__main__":
    import uvicorn
    from config.settings import settings

    uvicorn.run(
        "api.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True,
    )
