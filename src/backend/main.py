import asyncio
import json
import time
import uuid

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.server.contribution_gate import ContributionAdmission, get_gate
from backend.server.openai_compat import openai_error_response, openai_models_payload
from backend.server.request_handler import RequestHandler
from backend.server.route_capability_api import (
    configure_request_agent_authority,
    router as request_agent_authority_router,
)
from backend.server.server_args import parse_args
from parallax_utils.ascii_anime import display_parallax_run
from parallax_utils.file_util import get_project_root
from parallax_utils.logging_config import get_logger, set_log_level
from parallax_utils.version_check import check_latest_release

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(request_agent_authority_router)

logger = get_logger(__name__)

scheduler_manage = None
request_handler = RequestHandler()


def bearer_credential(raw_request: Request):
    authorization = raw_request.headers.get("authorization", "")
    scheme, separator, credential = authorization.partition(" ")
    if separator and scheme.lower() == "bearer":
        return credential.strip() or None
    return None


def release_after_stream(response: StreamingResponse, admission: ContributionAdmission):
    """Hold contribution concurrency until the response body actually finishes."""

    original = response.body_iterator

    async def guarded_iterator():
        try:
            async for chunk in original:
                yield chunk
        finally:
            get_gate().release(admission)

    response.body_iterator = guarded_iterator()
    return response


@app.post("/weight/refit")
async def weight_refit(raw_request: Request):
    request_data = await raw_request.json()
    status = scheduler_manage.weight_refit(request_data)
    if status:
        return JSONResponse(
            content={
                "type": "weight_refit",
                "data": None,
            },
            status_code=200,
        )
    else:
        return JSONResponse(
            content={
                "type": "weight_refit",
                "data": "Sever not ready",
            },
            status_code=500,
        )


@app.get("/weight/refit/timestamp")
async def weight_refit_timstamp():
    last_refit_time = scheduler_manage.get_last_refit_time()

    return JSONResponse(
        content={
            "latest_timestamp": last_refit_time,
        },
        status_code=200,
    )


@app.get("/model/list")
async def model_list():
    # Keep importing the ASGI control surface cheap. Model/runtime dependencies
    # (torch, NumPy, MLX) belong to scheduler startup, not to route discovery or
    # OpenAI compatibility tests that only import ``backend.main:app``.
    from backend.server.static_config import get_model_list

    return JSONResponse(
        content={
            "type": "model_list",
            "data": get_model_list(),
        },
        status_code=200,
    )


@app.get("/v1/models")
async def openai_v1_models():
    model_name = None
    if scheduler_manage is not None:
        try:
            model_name = scheduler_manage.get_model_name()
        except Exception as e:
            logger.debug(f"Unable to get scheduler model name: {e}")

    return JSONResponse(content=openai_models_payload(model_name), status_code=200)


@app.post("/scheduler/init")
async def scheduler_init(raw_request: Request):
    request_data = await raw_request.json()
    model_name = request_data.get("model_name")
    init_nodes_num = request_data.get("init_nodes_num")
    is_local_network = request_data.get("is_local_network")

    # Validate required parameters
    if model_name is None:
        return JSONResponse(
            content={
                "type": "scheduler_init",
                "error": "model_name is required",
            },
            status_code=400,
        )
    if init_nodes_num is None:
        return JSONResponse(
            content={
                "type": "scheduler_init",
                "error": "init_nodes_num is required",
            },
            status_code=400,
        )

    try:
        # If scheduler is already running, stop it first
        if scheduler_manage.is_running():
            logger.info(f"Stopping existing scheduler to switch to model: {model_name}")
            scheduler_manage.stop()

        # Start scheduler with new model
        logger.info(
            f"Initializing scheduler with model: {model_name}, init_nodes_num: {init_nodes_num}"
        )
        scheduler_manage.run(model_name, init_nodes_num, is_local_network)
        configure_request_agent_authority(scheduler_manage, get_gate())

        return JSONResponse(
            content={
                "type": "scheduler_init",
                "data": {
                    "model_name": model_name,
                    "init_nodes_num": init_nodes_num,
                    "is_local_network": is_local_network,
                },
            },
            status_code=200,
        )
    except Exception as e:
        logger.exception(f"Error initializing scheduler: {e}")
        return JSONResponse(
            content={
                "type": "scheduler_init",
                "error": str(e),
            },
            status_code=500,
        )


@app.get("/node/join/command")
async def node_join_command():
    from backend.server.static_config import get_node_join_command

    peer_id = scheduler_manage.get_peer_id()
    is_local_network = scheduler_manage.get_is_local_network()

    return JSONResponse(
        content={
            "type": "node_join_command",
            "data": get_node_join_command(peer_id, is_local_network),
        },
        status_code=200,
    )


@app.get("/cluster/status")
async def cluster_status():
    async def stream_cluster_status():
        while True:
            yield json.dumps(scheduler_manage.get_cluster_status(), ensure_ascii=False) + "\n"
            await asyncio.sleep(1)

    return StreamingResponse(
        stream_cluster_status(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/cluster/status_json")
async def cluster_status_json() -> JSONResponse:
    if scheduler_manage is None:
        return JSONResponse(content={"error": "Scheduler is not initialized"}, status_code=503)
    return JSONResponse(content=scheduler_manage.get_cluster_status(), status_code=200)


@app.get("/v1/contribution/status")
async def contribution_status(raw_request: Request) -> JSONResponse:
    """Account-scoped admission state used by the IDE to reveal its prompt."""

    gate = get_gate()
    scheduler = scheduler_manage.scheduler if scheduler_manage is not None else None
    status = gate.status(bearer_credential(raw_request), scheduler)
    return JSONResponse(content=status.public_payload(enabled=gate.enabled), status_code=200)


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(raw_request: Request):
    gate = get_gate()
    scheduler = scheduler_manage.scheduler if scheduler_manage is not None else None
    admission = gate.admit(bearer_credential(raw_request), scheduler)
    if not admission.allowed:
        if admission.status.reason == "capacity_reached":
            status_code = 429
        elif admission.status.reason == "swarm_not_ready":
            status_code = 503
        else:
            status_code = 403
        return JSONResponse(
            content=gate.denial_payload(admission.status),
            status_code=status_code,
            headers={"Retry-After": "1"} if status_code in {429, 503} else None,
        )
    try:
        request_data = await raw_request.json()
    except Exception:
        gate.release(admission)
        return openai_error_response(
            "Invalid request body",
            status_code=400,
            err_type="invalid_request_error",
            code="invalid_request_error",
        )
    if not isinstance(request_data, dict):
        gate.release(admission)
        return openai_error_response(
            "Request body must be a JSON object",
            status_code=400,
            err_type="invalid_request_error",
            code="invalid_request_error",
        )

    request_id = uuid.uuid4()
    received_ts = time.time()
    try:
        response = await request_handler.v1_chat_completions(
            request_data,
            request_id,
            received_ts,
            raw_request.is_disconnected,
        )
    except BaseException:
        gate.release(admission)
        raise
    if isinstance(response, StreamingResponse):
        return release_after_stream(response, admission)
    gate.release(admission)
    return response


# Disable caching for index.html
@app.get("/")
async def serve_index():
    response = FileResponse(str(get_project_root()) + "/src/frontend/dist/index.html")
    # Disable cache
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# mount the frontend
app.mount(
    "/",
    StaticFiles(directory=str(get_project_root() / "src" / "frontend" / "dist"), html=True),
    name="static",
)

if __name__ == "__main__":
    from backend.server.scheduler_manage import SchedulerManage
    from backend.server.static_config import init_model_info_dict_cache

    args = parse_args()
    set_log_level(args.log_level)
    logger.info(f"args: {args}")

    if args.model_name is None:
        init_model_info_dict_cache(args.use_hfcache)

    if args.log_level != "DEBUG":
        display_parallax_run()

    check_latest_release()

    scheduler_manage = SchedulerManage(
        initial_peers=args.initial_peers,
        relay_servers=args.relay_servers,
        dht_prefix=args.dht_prefix,
        host_maddrs=[
            f"/ip4/0.0.0.0/tcp/{args.tcp_port}",
            f"/ip4/0.0.0.0/udp/{args.udp_port}/quic-v1",
        ],
        announce_maddrs=args.announce_maddrs,
        http_port=args.port,
        use_hfcache=args.use_hfcache,
        enable_weight_refit=args.enable_weight_refit,
        weight_refit_mode=args.weight_refit_mode,
        allocation_strategy=args.allocation_strategy,
        routing_strategy=args.routing_strategy,
        heartbeat_timeout=args.heartbeat_timeout,
    )

    request_handler.set_scheduler_manage(scheduler_manage)

    model_name = args.model_name
    init_nodes_num = args.init_nodes_num
    is_local_network = args.is_local_network
    if model_name is not None and init_nodes_num is not None:
        scheduler_manage.run(model_name, init_nodes_num, is_local_network)
        configure_request_agent_authority(scheduler_manage, get_gate())

    host = args.host
    port = args.port

    uvicorn.run(app, host=host, port=port, log_level="info", loop="auto")
