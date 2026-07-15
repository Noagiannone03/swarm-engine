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
from parallax.utils.utils import create_local_zmq_endpoints, initialize_nccl_port, load_config_only
from parallax_utils.ascii_anime import display_parallax_join
from parallax_utils.cuda_memory import configure_torch_cuda_memory_limit
from parallax_utils.logging_config import get_logger, set_log_level
from parallax_utils.version_check import check_latest_release

logger = get_logger("parallax.launch")


def _update_args_from_shared_state(args, shared_state: SharedState, force_update: bool):
    """Update args with layer allocation from shared state"""
    model_info = shared_state.get_model_info()
    args.start_layer = model_info["block_start_index"]
    args.end_layer = model_info["block_end_index"]
    if args.model_path is not None and not force_update:
        # Use local model path first
        pass
    elif model_info["model_name"]:
        # Update model_path if provided
        args.model_path = model_info["model_name"]
        logger.debug(f"Updated model_path to: {args.model_path}")
    else:
        assert False, "Neither scheduler nor worker provides a valid model path!"
    # Update tp_size if provided, otherwise keep current value
    args.tp_size = model_info["tp_size"] or args.tp_size
    # Update weight refit switch
    args.enable_weight_refit = model_info["enable_weight_refit"] or args.enable_weight_refit
    args.weight_refit_mode = model_info["weight_refit_mode"] or args.weight_refit_mode


def _stop_executor_processes(executor_subprocs):
    """Stop all executor processes"""
    for executor_process in executor_subprocs:
        if executor_process.is_alive():
            logger.debug(f"Terminating executor process {executor_process.pid}")
            stop_executor_process(executor_process)


def _wait_executors_check_layer_change(shared_state: SharedState, executor_subprocs):
    """Wait for executor processes and check if layer allocation changed.

    Returns:
        True if layer allocation changed (need to reload executors),
        False if all executors exited normally.
    """
    while any(proc.is_alive() for proc in executor_subprocs):
        for proc in executor_subprocs:
            if proc.is_alive():
                proc.join(timeout=1.0)  # Check every second

        if shared_state.get_layer_allocation_changed():
            return True

    # Check race condition: layer allocation changed after all processes exited
    return shared_state.get_layer_allocation_changed()


def _executors_crashed(executor_subprocs) -> bool:
    """True if any executor process exited abnormally (non-zero/​signal).

    exitcode 0 = clean exit; None = still running (shouldn't happen post-join);
    anything else (including negative = killed by signal, e.g. OOM-killer) is a
    crash. Used to retry a transient crash instead of leaving the swarm.
    """
    return any(proc.exitcode not in (0, None) for proc in executor_subprocs)


def _wait_for_initial_layer_allocation(
    shared_state: SharedState,
    p2p_server_process: multiprocessing.Process,
    *,
    max_wait_time: float = 300,
) -> None:
    """Wait for initial layers while allowing a valid standby join.

    In scheduler mode, a worker can be accepted as STANDBY when an existing
    pipeline already covers the model. That is not a launch failure: the P2P
    process must remain alive and send node_update heartbeats until the scheduler
    later assigns layers. It is still an error if the P2P process exits, because
    then no future allocation can arrive.
    """
    logger.debug("Waiting for layer allocation from scheduler...")
    wait_start = time.time()
    standby_reported = False
    while True:
        model_info = shared_state.get_model_info()
        if (
            model_info["block_start_index"] is not None
            and model_info["block_end_index"] is not None
            and model_info["model_name"] is not None
        ):
            return
        if p2p_server_process is not None and not p2p_server_process.is_alive():
            raise RuntimeError(
                "P2P server exited before layer allocation "
                f"(exitcode={p2p_server_process.exitcode})"
            )
        if time.time() - wait_start > max_wait_time:
            if not standby_reported:
                logger.info(
                    "No layer allocation after %ss; staying online as a standby contributor",
                    max_wait_time,
                )
                standby_reported = True
            wait_start = time.time()
        time.sleep(1)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    p2p_server_process = None
    frontend_process = None
    executor_subprocs = []
    # Shared state for layer allocation info (used when P2P server is in subprocess)
    shared_state = SharedState.create()
    shared_state.set_status(ServerState.JOINING.value)

    try:
        args = parse_args()
        set_log_level(args.log_level)
        configure_torch_cuda_memory_limit()
        logger.debug(f"args: {args}")
        (
            args.recv_from_peer_addr,
            args.send_to_peer_addr,
            args.executor_input_ipc,
            args.executor_output_ipc,
        ) = create_local_zmq_endpoints(4)
        if args.nccl_port is None:
            args.nccl_port = initialize_nccl_port()

        # Silence tokenizer warnings
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        logger.debug(f"executor_input_addr: {args.executor_input_ipc}")
        logger.debug(f"executor_output_addr: {args.executor_output_ipc}")
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
            if args.start_layer == 0:
                frontend_process = launch_vllm_rust_frontend(args)
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
            shared_state.set_status(ServerState.READY.value)

            # Wait for all executor processes
            for proc in executor_subprocs:
                proc.join()
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
                shared_state=shared_state.dict,  # Pass dict to subprocess
                log_level=args.log_level,
                conn=conn_main,
            )

            # Wait for layer allocation from scheduler (via shared state). If the
            # scheduler accepted us as standby, keep this process alive until a
            # later node_update carries real layers.
            _wait_for_initial_layer_allocation(shared_state, p2p_server_process)

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
            executor_crash_count = 0
            try:
                max_executor_retries = max(0, int(os.environ.get("PARALLAX_EXECUTOR_MAX_RETRIES", "2")))
            except ValueError:
                max_executor_retries = 2
            while True:
                try:
                    # Only launch the Rust HTTP frontend on head node.
                    if args.start_layer == 0:
                        frontend_process = launch_vllm_rust_frontend(args)

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
                    if _wait_executors_check_layer_change(shared_state, executor_subprocs):
                        logger.warning("Layer allocation changed! Stopping executors to reload...")
                        # Reset flag and set status to INITIALIZING
                        shared_state.update(
                            _layer_allocation_changed=False,
                            status=ServerState.INITIALIZING.value,
                        )
                        _stop_executor_processes(executor_subprocs)
                        if frontend_process is not None:
                            stop_vllm_rust_frontend(frontend_process)
                            frontend_process = None
                        _update_args_from_shared_state(args, shared_state, force_update=True)
                        logger.info(
                            f"Reloading executor with layers [{args.start_layer}, {args.end_layer})"
                        )
                        continue

                    # Executors exited WITHOUT a layer-allocation change. Tell a
                    # genuine crash (OOM, transient CUDA error -> non-zero exit)
                    # from a clean shutdown. A single node hitting a transient
                    # crash must NOT tear itself out of the swarm: leaving forces
                    # a global re-bootstrap that, on a small no-surplus cluster,
                    # collapses the whole pipeline. Instead report ERROR (the
                    # scheduler stops routing to us but keeps the node) and retry
                    # a bounded number of times before finally giving up.
                    if _executors_crashed(executor_subprocs):
                        if p2p_server_process is not None and not p2p_server_process.is_alive():
                            logger.error("P2P server is gone; not retrying executor.")
                            break
                        executor_crash_count += 1
                        codes = [p.exitcode for p in executor_subprocs]
                        _stop_executor_processes(executor_subprocs)
                        if http_server_process is not None:
                            stop_http_server(http_server_process)
                            http_server_process = None
                        if executor_crash_count > max_executor_retries:
                            logger.error(
                                "Executor crashed %d times (exitcodes=%s); giving up and leaving.",
                                executor_crash_count,
                                codes,
                            )
                            break
                        shared_state.set_status(ServerState.ERROR.value)
                        backoff = min(30.0, 3.0 * executor_crash_count)
                        logger.error(
                            "Executor(s) crashed (exitcodes=%s), retry %d/%d in %.0fs; "
                            "staying in the swarm as ERROR so we don't trigger a global re-bootstrap.",
                            codes,
                            executor_crash_count,
                            max_executor_retries,
                            backoff,
                        )
                        time.sleep(backoff)
                        _update_args_from_shared_state(args, shared_state, force_update=True)
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

        # Shutdown executor subprocesses
        for executor_process in executor_subprocs:
            if executor_process.is_alive():
                stop_executor_process(executor_process)

        # Shutdown P2P server subprocess
        if p2p_server_process is not None:
            stop_p2p_server(p2p_server_process)

        # Shutdown Rust frontend
        if frontend_process is not None:
            stop_vllm_rust_frontend(frontend_process)

        logger.debug("All processes shut down.")
