from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..core.master_agent import MasterAgent
from ..llm.config import FrameworkLLMConfig, LLMConfig
from ..tools.mcp_provider import MCPServerConfig


STATIC_DIR = Path(__file__).parent / "static"
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
logger = logging.getLogger(__name__)

_mcp_health: dict[str, Any] = {
    "status": "starting",
    "error": "",
}


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", "disabled"}


def _web_search_mcp_config() -> MCPServerConfig:
    forwarded_names = (
        "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "no_proxy", "all_proxy",
        "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE",
    )
    forwarded = {
        name: os.environ[name] for name in forwarded_names if name in os.environ
    }
    script = os.environ.get("SYNAPSE_WEB_SEARCH_MCP_SCRIPT")
    args = (script,) if script else (
        "-m", "agent_framework.tools.web_search_server"
    )
    return MCPServerConfig(
        name="web-search",
        command=os.environ.get("SYNAPSE_WEB_SEARCH_MCP_COMMAND", sys.executable),
        args=args,
        cwd=REPOSITORY_ROOT,
        env=forwarded,
        tool_allowlist=frozenset({"tool_search_web", "tool_fetch_webpage"}),
        category="web_search",
        call_timeout_seconds=float(
            os.environ.get("SYNAPSE_WEB_SEARCH_TIMEOUT", "30")
        ),
    )


class StartRunRequest(BaseModel):
    request: str = Field(min_length=1, max_length=100_000)


class SteerRunRequest(BaseModel):
    requirement: str = Field(min_length=1, max_length=50_000)


def create_master_from_env() -> MasterAgent:
    base_url = os.environ.get("SYNAPSE_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("SYNAPSE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "local"
    planner_model = os.environ.get("SYNAPSE_PLANNER_MODEL", "gpt-4o")
    executor_model = os.environ.get("SYNAPSE_EXECUTOR_MODEL", planner_model)
    lightweight_model = os.environ.get("SYNAPSE_LIGHTWEIGHT_MODEL", planner_model)
    timeout = float(os.environ.get("SYNAPSE_LLM_TIMEOUT", "120"))
    config = FrameworkLLMConfig(
        planner=LLMConfig(
            base_url=base_url,
            api_key=api_key,
            model=planner_model,
            timeout=timeout,
        ),
        executor=LLMConfig(
            base_url=base_url,
            api_key=api_key,
            model=executor_model,
            timeout=timeout,
        ),
        lightweight=LLMConfig(
            base_url=base_url,
            api_key=api_key,
            model=lightweight_model,
            timeout=timeout,
        ),
    )
    return MasterAgent(
        llm_config=config,
        memory_root=os.environ.get("SYNAPSE_MEMORY_ROOT"),
        cache_dir=os.environ.get("SYNAPSE_CACHE_DIR"),
        run_root=os.environ.get("SYNAPSE_RUN_ROOT"),
    )


master = create_master_from_env()


@asynccontextmanager
async def lifespan(_: FastAPI):
    if _env_enabled("SYNAPSE_WEB_SEARCH_MCP"):
        try:
            tools = await master.connect_mcp_server(_web_search_mcp_config())
            _mcp_health.update({"status": "connected", "tools": tools, "error": ""})
        except Exception as error:
            if "MCP SDK is missing" in str(error):
                logger.error("Web Search MCP unavailable: %s", error)
            else:
                logger.exception("Web Search MCP failed to start")
            _mcp_health.update({"status": "degraded", "tools": [], "error": str(error)})
    else:
        _mcp_health.update({"status": "disabled", "tools": [], "error": ""})
    try:
        yield
    finally:
        await master.close_mcp_servers()


app = FastAPI(title="Synapse Control Room", version="0.2.0", lifespan=lifespan)


def _not_found(error: KeyError) -> HTTPException:
    return HTTPException(status_code=404, detail=str(error).strip("'"))


@app.get("/api/health")
async def health() -> dict[str, Any]:
    runtime = master.get_runtime_info()
    return {
        "status": "ok",
        "version": "0.2.0",
        **runtime,
        "mcp": {
            **_mcp_health,
            "servers": master.get_mcp_status(),
        },
        "configured": bool(
            os.environ.get("SYNAPSE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or runtime["base_url"].startswith(("http://localhost", "http://127.0.0.1"))
        ),
    }


@app.get("/api/runs")
async def list_runs() -> dict[str, Any]:
    return {"runs": await master.list_run_states()}


@app.post("/api/runs", status_code=202)
async def start_run(body: StartRunRequest) -> dict[str, Any]:
    request = body.request.strip()
    if not request:
        raise HTTPException(status_code=422, detail="Request cannot be blank")
    try:
        run_id = await master.start_request(request)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"run_id": run_id, "status": "running"}


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    try:
        return await master.get_run_state(run_id)
    except KeyError as error:
        raise _not_found(error) from error


@app.get("/api/runs/{run_id}/events")
async def get_events(
    run_id: str,
    after: int = Query(default=0, ge=0),
    limit: int = Query(default=500, ge=1, le=2_000),
) -> dict[str, Any]:
    try:
        events = await master.get_run_events(run_id, after, limit)
    except KeyError as error:
        raise _not_found(error) from error
    return {"events": events}


@app.get("/api/runs/{run_id}/agents/{agent_id}")
async def get_agent(run_id: str, agent_id: str) -> dict[str, Any]:
    try:
        snapshot = await master.get_agent_snapshot(run_id, agent_id)
    except KeyError as error:
        raise _not_found(error) from error
    if snapshot is None:
        raise HTTPException(status_code=404, detail="Agent snapshot is not available")
    return snapshot


@app.post("/api/runs/{run_id}/steer", status_code=202)
async def steer_run(run_id: str, body: SteerRunRequest) -> dict[str, Any]:
    requirement = body.requirement.strip()
    if not requirement:
        raise HTTPException(status_code=422, detail="Requirement cannot be blank")
    try:
        revision = await master.steer(run_id, requirement)
    except KeyError as error:
        raise _not_found(error) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"run_id": run_id, "revision": revision, "status": "accepted"}


@app.post("/api/runs/{run_id}/cancel", status_code=202)
async def cancel_run(run_id: str) -> dict[str, Any]:
    try:
        await master.cancel_run(run_id)
    except KeyError as error:
        raise _not_found(error) from error
    return {"run_id": run_id, "status": "cancelled_retained"}


@app.post("/api/runs/{run_id}/archive")
async def archive_run(run_id: str) -> dict[str, Any]:
    try:
        await master.archive_run(run_id)
    except KeyError as error:
        raise _not_found(error) from error
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"run_id": run_id, "status": "archived"}


async def _event_stream(
    request: Request,
    run_id: str,
    after: int,
) -> AsyncIterator[str]:
    cursor = after
    while not await request.is_disconnected():
        try:
            events = await master.get_run_events(run_id, cursor, 500)
        except KeyError:
            yield "event: error\ndata: {\"detail\":\"Run not found\"}\n\n"
            return
        if events:
            for event in events:
                cursor = max(cursor, int(event.get("sequence", cursor)))
                payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                yield f"id: {cursor}\nevent: run_event\ndata: {payload}\n\n"
        else:
            yield ": keep-alive\n\n"
        await asyncio.sleep(0.6)


@app.get("/api/runs/{run_id}/stream")
async def stream_events(
    request: Request,
    run_id: str,
    after: int = Query(default=0, ge=0),
) -> StreamingResponse:
    try:
        await master.get_run_events(run_id, after, 1)
    except KeyError as error:
        raise _not_found(error) from error
    return StreamingResponse(
        _event_stream(request, run_id, after),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    uvicorn.run(
        "agent_framework.web.app:app",
        host=os.environ.get("SYNAPSE_WEB_HOST", "127.0.0.1"),
        port=int(os.environ.get("SYNAPSE_WEB_PORT", "8008")),
        reload=False,
    )
