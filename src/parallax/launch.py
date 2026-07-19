"""
Launch the Parallax server.

This script is used to launch the Parallax server.
It will start the following services:
    1.Executor each tp_rank as a subprocess.
    2.vLLM Rust frontend as the HTTP server subprocess.
    3.P2P server as a subprocess.

Example command:
python src/parallax/launch.py \
    --model-path Qwen/Qwen3-0.6B \
    --max-num-tokens-per-batch 16384 \
    --max-batch-size 128 \
    --start-layer 0 \
    --end-layer 28
"""

import argparse
import multiprocessing
import os
import time

from parallax.p2p.server import ServerState, launch_p2p_server_process, stop_p2p_server
from parallax.server.executor.factory import run_executor_process, stop_executor_process
from parallax.server.server_args import parse_args
from parallax.server.vllm_rust_frontend import (
    launch_vllm_rust_frontend,
    stop_vllm_rust_frontend,
)
from parallax.utils.shared_state import SharedState
from parallax.utils.utils import (
    cleanup_local_zmq_endpoints,
    create_local_zmq_endpoints,
    initialize_nccl_port,
    load_config_only,
)
from parallax_utils.ascii_anime import display_parallax_join
from parallax_utils.logging_config import get_logger, set_log_level
from parallax_utils.version_check import check_latest_release

logger = get_logger("parallax.launch")


def _update_args_from_shared_state(args, shared_state: SharedState, force_update: bool):
    """Update args with layer allocation from shared state"""
    model_info = shared_state.get_model_info()
    args.start_layer = model_info["block_start_index"]
    args.end_layer = model_info["block_end_index"]
    # Keep the scheduler's public model name even when this worker loads weights
    # from a local path. The Rust frontend uses it as the OpenAI API alias.
    if not getattr(args, "served_model_name", None) or force_update:
        args.served_model_name = model_info["model_name"]

    # A worker may load an optimized local artifact (for example an MLX model on
    # macOS) while serving the scheduler's public model name. Preserve that
    # explicit CLI choice across every DP layer reallocation. Without a stable
    # override, a join/leave event replaces the local path with the public Hub
    # name and can either download the wrong artifact or fail during recovery.
    if not hasattr(args, "_worker_model_path_override"):
        args._worker_model_path_override = args.model_path

    if args._worker_model_path_override is not None:
        args.model_path = args._worker_model_path_override
    elif model_info["model_name"]:
        args.model_path = model_info["model_name"]
        logger.debug(f"Updated model_path to: {args.model_path}")
    else:
        assert False, "Neither scheduler nor worker provides a valid model path!"
    # Update tp_size if provided, otherwise keep current value
    args.tp_size = model_info["tp_size"] or args.tp_size
    # Update weight refit switch
    args.enable_weight_refit = model_info["enable_weight_refit"] or args.enable_weight_refit
    args.weight_refit_mode = model_info["weight_refit_mode"] or args.weight_refit_mode
    negotiated_chunk_size = model_info.get("chunked_prefill_size")
    if negotiated_chunk_size is not None:
        args.chunked_prefill_size = (
            None if int(negotiated_chunk_size) == 0 else int(negotiated_chunk_size)
        )


def _stop_executor_processes(executor_subprocs):
    """Stop all executor processes"""
    for executor_process in executor_subprocs:
        if executor_process.is_alive():
            logger.debug(f"Terminating executor process {executor_process.pid}")
            stop_executor_process(executor_process)


def _prepare_engine_core_generation(args, shared_state: SharedState, frontend_required: bool):
    """Create private engine-core endpoints and publish frontend readiness.

    The Rust frontend binds both engine-core IPC endpoints. A frontend that has
    to be killed during a layer reload can leave those filesystem endpoints
    behind on POSIX, so an endpoint pair belongs to exactly one executor
    generation and must never be reused.
    """
    args.executor_input_ipc, args.executor_output_ipc = create_local_zmq_endpoints(2)
    logger.debug(f"executor_input_addr: {args.executor_input_ipc}")
    logger.debug(f"executor_output_addr: {args.executor_output_ipc}")
    shared_state.update(
        frontend_required=bool(frontend_required),
        frontend_alive=False,
    )


def _set_frontend_alive(shared_state: SharedState, frontend_process) -> None:
    shared_state.set(
        "frontend_alive",
        frontend_process is not None and frontend_process.is_alive(),
    )


def _cleanup_engine_core_generation(args) -> None:
    cleanup_local_zmq_endpoints(
        [
            endpoint
            for endpoint in (
                getattr(args, "executor_input_ipc", None),
                getattr(args, "executor_output_ipc", None),
            )
            if endpoint is not None
        ]
    )


def _wait_executors_check_layer_change(
    shared_state: SharedState,
    executor_subprocs,
    frontend_process=None,
):
    """Wait for executor processes and check if layer allocation changed.

    Returns:
        True if layer allocation changed (need to reload executors),
        False if all executors exited normally.
    """
    while any(proc.is_alive() for proc in executor_subprocs):
        if frontend_process is not None and not frontend_process.is_alive():
            shared_state.update(
                frontend_alive=False,
                status=ServerState.INITIALIZING.value,
            )
            raise RuntimeError("vLLM Rust frontend exited while its executor was running")

        poll_timeout = 1.0 / max(len(executor_subprocs), 1)
        for proc in executor_subprocs:
            if proc.is_alive():
                proc.join(timeout=poll_timeout)
            if frontend_process is not None and not frontend_process.is_alive():
                shared_state.update(
                    frontend_alive=False,
                    status=ServerState.INITIALIZING.value,
                )
                raise RuntimeError("vLLM Rust frontend exited while its executor was running")

        if shared_state.get_layer_allocation_changed():
            return True

    # Check race condition: layer allocation changed after all processes exited
    return shared_state.get_layer_allocation_changed()


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    p2p_server_process = None
    frontend_process = None
    executor_subprocs = []
    args = None
    # Shared state for layer allocation info (used when P2P server is in subprocess)
    shared_state = SharedState.create()
    shared_state.set_status(ServerState.JOINING.value)

    try:
        args = parse_args()
        set_log_level(args.log_level)
        logger.debug(f"args: {args}")
        (
            args.recv_from_peer_addr,
            args.send_to_peer_addr,
        ) = create_local_zmq_endpoints(2)
        if args.nccl_port is None:
            args.nccl_port = initialize_nccl_port()

        # Silence tokenizer warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        logger.debug(f"nccl_port: {args.nccl_port}")

        # Pipe for subprocess communication
        conn_main, conn_refit = multiprocessing.Pipe()

        if args.scheduler_addr is None:
            if args.log_level != "DEBUG":
                display_parallax_join(args.model_path)
            check_latest_release()

            config = load_config_only(args.model_path, local_files_only=args.use_hfcache)
            if args.start_layer is None:
                args.start_layer = 0
            if args.end_layer is None:
                args.end_layer = config.get("num_hidden_layers")

            # Only launch the Rust HTTP frontend on head node.
            _prepare_engine_core_generation(args, shared_state, args.start_layer == 0)
            if args.start_layer == 0:
                frontend_process = launch_vllm_rust_frontend(args)
                _set_frontend_alive(shared_state, frontend_process)
            # Launch P2P server as subprocess
            if not (args.start_layer == 0 and args.end_layer == config.get("num_hidden_layers")):
                p2p_server_process = launch_p2p_server_process(
                    initial_peers=args.initial_peers,
                    scheduler_addr=args.scheduler_addr,
                    relay_servers=args.relay_servers,
                    pp_start_layer=args.start_layer,
                    pp_end_layer=args.end_layer,
                    hidden_layers=config.get("num_hidden_layers"),
                    tp_size=args.tp_size,
                    dp_size=args.dp_size,
                    tcp_port=args.tcp_port,
                    udp_port=args.udp_port,
                    dht_prefix=args.dht_prefix,
                    announce_maddrs=args.announce_maddrs,
                    http_port=args.port,
                    notify_url=args.notify_url,
                    recv_from_peer_addr=args.recv_from_peer_addr,
                    send_to_peer_addr=args.send_to_peer_addr,
                    model_name=args.model_path,
                    max_batch_size=args.max_batch_size,
                    max_sequence_length=args.max_sequence_length,
                    param_mem_ratio=args.param_mem_ratio,
                    kvcache_mem_ratio=args.kvcache_mem_ratio,
                    gpu_backend=args.gpu_backend,
                    chunked_prefill_size=args.chunked_prefill_size,
                    shared_state=shared_state.dict,
                    log_level=args.log_level,
                    conn=conn_main,
                )

            # Build connectors for tp communication
            conn_tp_0 = [conn_refit]
            conn_tp_i = []
            for i in range(1, args.tp_size):
                conn1, conn2 = multiprocessing.Pipe()
                conn_tp_0.append(conn1)
                conn_tp_i.append(conn2)
            # Launch all executor processes (including tp_rank=0)
            for tp_rank in range(args.tp_size):
                args_copy = argparse.Namespace(**vars(args))
                args_copy.tp_rank = tp_rank
                proc = multiprocessing.Process(
                    target=run_executor_process,
                    args=(
                        args_copy,
                        shared_state.dict,  # Pass dict to subprocess
                        conn_tp_0 if tp_rank == 0 else [conn_tp_i[tp_rank - 1]],
                    ),
                )
                proc.start()
                executor_subprocs.append(proc)

            time.sleep(2)  # Give executors time to start
            if frontend_process is not None and not frontend_process.is_alive():
                raise RuntimeError("vLLM Rust frontend exited during executor startup")
            shared_state.set_status(ServerState.READY.value)

            # Wait for all executor processes while supervising ingress.
            _wait_executors_check_layer_change(
                shared_state,
                executor_subprocs,
                frontend_process,
            )
        else:
            # Launch P2P server as subprocess (with scheduler)
            # Pass dict to subprocess (multiprocessing requires serializable objects)
            p2p_server_process = launch_p2p_server_process(
                initial_peers=args.initial_peers,
                scheduler_addr=args.scheduler_addr,
                relay_servers=args.relay_servers,
                pp_start_layer=args.start_layer,
                pp_end_layer=args.end_layer,
                hidden_layers=None,
                tp_size=args.tp_size,
                dp_size=args.dp_size,
                tcp_port=args.tcp_port,
                udp_port=args.udp_port,
                dht_prefix=args.dht_prefix,
                announce_maddrs=args.announce_maddrs,
                http_port=args.port,
                notify_url=args.notify_url,
                recv_from_peer_addr=args.recv_from_peer_addr,
                send_to_peer_addr=args.send_to_peer_addr,
                model_name=args.model_path,
                max_batch_size=args.max_batch_size,
                max_sequence_length=args.max_sequence_length,
                param_mem_ratio=args.param_mem_ratio,
                kvcache_mem_ratio=args.kvcache_mem_ratio,
                gpu_backend=args.gpu_backend,
                chunked_prefill_size=args.chunked_prefill_size,
                shared_state=shared_state.dict,  # Pass dict to subprocess
                log_level=args.log_level,
                conn=conn_main,
            )

            # Wait for layer allocation from scheduler (via shared state)
            logger.debug("Waiting for layer allocation from scheduler...")
            max_wait_time = 300  # 5 minutes
            wait_start = time.time()
            while True:
                model_info = shared_state.get_model_info()
                if (
                    model_info["block_start_index"] is not None
                    and model_info["block_end_index"] is not None
                    and model_info["model_name"] is not None
                ):
                    break
                if time.time() - wait_start > max_wait_time:
                    logger.error("Timeout waiting for layer allocation from scheduler")
                    raise RuntimeError("Failed to get layer allocation from scheduler")
                time.sleep(1)

            # Get layer allocation from shared state
            _update_args_from_shared_state(args, shared_state, force_update=False)

            logger.debug(
                f"Start Executor with start_layer: {args.start_layer}, end_layer: {args.end_layer}, "
                f"model: {args.model_path}"
            )

            if args.log_level != "DEBUG":
                display_parallax_join(args.model_path)
            check_latest_release()

            # Main execution loop with layer reallocation support
            while True:
                try:
                    # Only launch the Rust HTTP frontend on head node.
                    _prepare_engine_core_generation(args, shared_state, args.start_layer == 0)
                    if args.start_layer == 0:
                        frontend_process = launch_vllm_rust_frontend(args)
                        _set_frontend_alive(shared_state, frontend_process)

                    # Build connectors for tp communication
                    conn_tp_0 = [conn_refit]
                    conn_tp_i = []
                    for i in range(1, args.tp_size):
                        conn1, conn2 = multiprocessing.Pipe()
                        conn_tp_0.append(conn1)
                        conn_tp_i.append(conn2)
                    # Launch all executor processes (including tp_rank=0)
                    executor_subprocs = []
                    for tp_rank in range(args.tp_size):
                        args_copy = argparse.Namespace(**vars(args))
                        args_copy.tp_rank = tp_rank
                        proc = multiprocessing.Process(
                            target=run_executor_process,
                            args=(
                                args_copy,
                                shared_state.dict,  # Pass dict to subprocess
                                conn_tp_0 if tp_rank == 0 else [conn_tp_i[tp_rank - 1]],
                            ),
                        )
                        proc.start()
                        executor_subprocs.append(proc)

                    # Wait for executors and restart if layer allocation changes
                    if _wait_executors_check_layer_change(
                        shared_state,
                        executor_subprocs,
                        frontend_process,
                    ):
                        logger.warning("Layer allocation changed! Stopping executors to reload...")
                        # Reset flag and set status to INITIALIZING
                        shared_state.update(
                            _layer_allocation_changed=False,
                            status=ServerState.INITIALIZING.value,
                            frontend_alive=False,
                        )
                        # Stop ingress before the engine so the frontend cannot
                        # accept work while its generation is being dismantled.
                        if frontend_process is not None:
                            stop_vllm_rust_frontend(frontend_process)
                            frontend_process = None
                        _stop_executor_processes(executor_subprocs)
                        _cleanup_engine_core_generation(args)
                        _update_args_from_shared_state(args, shared_state, force_update=True)
                        logger.info(
                            f"Reloading executor with layers [{args.start_layer}, {args.end_layer})"
                        )
                        continue

                    # All processes exited normally
                    break
                except KeyboardInterrupt:
                    logger.debug("Received interrupt signal, shutting down...")
                    break
                except Exception as e:
                    logger.exception(f"Executor error: {e}")
                    # Shutdown all executor processes on error
                    for proc in executor_subprocs:
                        if proc.is_alive():
                            stop_executor_process(proc)
                    raise
    except KeyboardInterrupt:
        logger.debug("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.exception(e)
    finally:
        # Shutdown all processes
        logger.debug("Shutting down all processes...")

        try:
            shared_state.update(
                status=ServerState.OFFLINE.value,
                frontend_alive=False,
            )
        except (BrokenPipeError, EOFError, OSError):
            logger.debug("Shared launch state was already unavailable during shutdown")

        # Stop ingress before executors for the same reason as a reload.
        if frontend_process is not None:
            stop_vllm_rust_frontend(frontend_process)
            frontend_process = None

        # Shutdown executor subprocesses
        for executor_process in executor_subprocs:
            if executor_process.is_alive():
                stop_executor_process(executor_process)
        _cleanup_engine_core_generation(args)

        # Shutdown P2P server subprocess
        if p2p_server_process is not None:
            stop_p2p_server(p2p_server_process)

        logger.debug("All processes shut down.")
