from __future__ import annotations

import asyncio
import json
import os
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


class StartRunRequest(BaseModel):
    request: str = Field(min_length=1, max_length=100_000)


class SteerRunRequest(BaseModel):
    requirement: str = Field(min_length=1, max_length=50_000)


class PermissionModeRequest(BaseModel):
    mode: str


class ApprovalDecisionRequest(BaseModel):
    allow: bool


class MCPServerRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    command: str = Field(min_length=1, max_length=10_000)
    args: list[str] = Field(default_factory=list, max_length=100)
    cwd: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    tool_allowlist: list[str] | None = None
    connect_timeout_seconds: float = Field(default=20.0, gt=0, le=300)
    call_timeout_seconds: float = Field(default=30.0, gt=0, le=600)


def create_master_from_env() -> MasterAgent:
    base_url = os.environ.get("SYNAPSE_BASE_URL", "https://api.openai.com/v1")
    api_key = os.environ.get("SYNAPSE_API_KEY") or os.environ.get("OPENAI_API_KEY") or "local"
    planner_model = os.environ.get("SYNAPSE_PLANNER_MODEL", "gpt-4o")
    executor_model = os.environ.get("SYNAPSE_EXECUTOR_MODEL", planner_model)
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
    )
    return MasterAgent(
        llm_config=config,
        memory_root=os.environ.get("SYNAPSE_MEMORY_ROOT"),
        run_root=os.environ.get("SYNAPSE_RUN_ROOT"),
        workspace_root=os.environ.get("SYNAPSE_WORKSPACE", str(REPOSITORY_ROOT)),
        permission_mode=os.environ.get("SYNAPSE_PERMISSION_MODE", "auto"),
    )


master = create_master_from_env()


@asynccontextmanager
async def lifespan(_: FastAPI):
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
    tools = master.get_tool_info()
    search_tools = [
        name for name in ("WebSearch", "WebFetch") if name in tools["builtin"]
    ]
    return {
        "status": "ok",
        "version": "0.2.0",
        **runtime,
        "tools": tools,
        "search": {
            "status": "ready" if len(search_tools) == 2 else "degraded",
            "tools": search_tools,
        },
        "mcp": {"status": "user_configured", "servers": master.get_mcp_status()},
        "permissions": master.get_permission_state(),
        "configured": bool(
            os.environ.get("SYNAPSE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or runtime["base_url"].startswith(("http://localhost", "http://127.0.0.1"))
        ),
    }


@app.get("/api/permissions")
async def get_permissions() -> dict[str, Any]:
    return {
        **master.get_permission_state(),
        "pending": master.list_pending_approvals(),
    }


@app.put("/api/permissions")
async def set_permissions(body: PermissionModeRequest) -> dict[str, Any]:
    try:
        return master.set_permission_mode(body.mode)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/api/approvals")
async def list_approvals(run_id: str | None = None) -> dict[str, Any]:
    return {"approvals": master.list_pending_approvals(run_id)}


@app.post("/api/approvals/{approval_id}")
async def resolve_approval(
    approval_id: str,
    body: ApprovalDecisionRequest,
) -> dict[str, Any]:
    try:
        return await master.resolve_approval(approval_id, body.allow)
    except KeyError as error:
        raise _not_found(error) from error


@app.get("/api/mcp/servers")
async def list_mcp_servers() -> dict[str, Any]:
    return {"servers": master.get_mcp_status()}


@app.post("/api/mcp/servers", status_code=201)
async def connect_mcp_server(body: MCPServerRequest) -> dict[str, Any]:
    config = MCPServerConfig(
        name=body.name,
        command=body.command,
        args=tuple(body.args),
        cwd=body.cwd or REPOSITORY_ROOT,
        env=body.env,
        tool_allowlist=frozenset(body.tool_allowlist) if body.tool_allowlist else None,
        category=f"mcp:{body.name}",
        connect_timeout_seconds=body.connect_timeout_seconds,
        call_timeout_seconds=body.call_timeout_seconds,
    )
    try:
        tools = await master.connect_mcp_server(config)
    except (RuntimeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return {"name": body.name, "tools": tools, "status": "connected"}


@app.delete("/api/mcp/servers/{name}")
async def disconnect_mcp_server(name: str) -> dict[str, Any]:
    try:
        await master.disconnect_mcp_server(name)
    except KeyError as error:
        raise _not_found(error) from error
    return {"name": name, "status": "disconnected"}


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
