"""OpenAI-compatible HTTP server for OpenDrop.

Wraps one or more llama-server instances behind a single FastAPI app that
speaks the OpenAI REST API.  Endpoints:
  GET  /v1/models                  — list loaded models
  POST /v1/chat/completions        — chat (streaming + non-streaming)
  POST /v1/completions             — legacy completions
  GET  /health                     — liveness check

The server proxies requests to the appropriate llama-server instance based on
the `model` field in the request body.  If the model is not currently running,
it is auto-started.
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from opendrop.config import get_config
from opendrop.core.registry import AsyncRegistry
from opendrop.inference.llamacpp import ServerManager, get_manager

# ---------------------------------------------------------------------------
# Pydantic models (subset of OpenAI API)
# ---------------------------------------------------------------------------


class ChatMessage(BaseModel):
    role: str
    content: str | list[dict]


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False
    stop: Optional[str | list[str]] = None


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    prompt: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    stream: bool = False


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(
    manager: Optional[ServerManager] = None,
    registry: Optional[AsyncRegistry] = None,
    allow_cors: bool = True,
) -> FastAPI:
    cfg = get_config()
    _manager = manager or get_manager()
    _registry: AsyncRegistry = registry or AsyncRegistry(cfg.registry_db())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        await _registry.init()
        yield
        _manager.stop_all()

    app = FastAPI(title="OpenDrop", version="0.1.0b1", lifespan=lifespan)

    if allow_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    async def _resolve_server(model_id: str) -> str:
        """Return base_url for the requested model, starting it if needed."""
        # Check running servers first
        running = _manager.running_models()
        for rec_id, srv in running.items():
            if rec_id == model_id or rec_id.startswith(model_id):
                return srv.base_url

        # Try to auto-start from registry
        rec = await _registry.get_model(model_id)
        if not rec:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_id}' not found. "
                       "Run `opendrop pull <url>` first.",
            )
        if not Path(rec.path).exists():
            raise HTTPException(
                status_code=503,
                detail=f"Model file missing: {rec.path}",
            )

        inf = cfg.inference
        srv = _manager.start_model(
            model_record_id=rec.id,
            gguf_path=Path(rec.path),
            ctx_size=inf.context_size,
            gpu_layers=inf.gpu_layers,
            parallel=inf.parallel,
            flash_attn=inf.flash_attn,
        )
        await _registry.touch_model(rec.id)
        return srv.base_url

    async def _proxy_stream(url: str, payload: dict) -> AsyncIterator[bytes]:
        async with httpx.AsyncClient(timeout=300) as client:
            async with client.stream("POST", url, json=payload) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

    async def _proxy_json(url: str, payload: dict) -> dict:
        async with httpx.AsyncClient(timeout=300) as client:
            r = await client.post(url, json=payload)
            r.raise_for_status()
            return r.json()

    # -----------------------------------------------------------------------
    # Routes
    # -----------------------------------------------------------------------

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "running_models": len(_manager.running_models())}

    @app.get("/v1/models")
    async def list_models() -> dict:
        records = await _registry.list_models()
        running = _manager.running_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": r.display_name,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "opendrop",
                    "status": "running" if r.id in running else "idle",
                }
                for r in records
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest) -> Any:
        base = await _resolve_server(req.model)
        payload = req.model_dump(exclude_none=True)
        payload["messages"] = [m.model_dump() for m in req.messages]

        if req.stream:
            return StreamingResponse(
                _proxy_stream(f"{base}/v1/chat/completions", payload),
                media_type="text/event-stream",
            )
        data = await _proxy_json(f"{base}/v1/chat/completions", payload)
        return JSONResponse(data)

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest) -> Any:
        base = await _resolve_server(req.model)
        payload = req.model_dump(exclude_none=True)

        if req.stream:
            return StreamingResponse(
                _proxy_stream(f"{base}/v1/completions", payload),
                media_type="text/event-stream",
            )
        data = await _proxy_json(f"{base}/v1/completions", payload)
        return JSONResponse(data)

    @app.post("/v1/embeddings")
    async def embeddings(request: Request) -> Any:
        body = await request.json()
        model_id = body.get("model", "")
        base = await _resolve_server(model_id)
        data = await _proxy_json(f"{base}/v1/embeddings", body)
        return JSONResponse(data)

    return app


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------

def run_server(
    host: str = "127.0.0.1",
    port: int = 11400,
    reload: bool = False,
) -> None:
    """Start the OpenDrop API server (blocking)."""
    import uvicorn

    uvicorn.run(
        "opendrop.inference.server:create_app",
        host=host,
        port=port,
        reload=reload,
        factory=True,
        log_level="info",
    )
