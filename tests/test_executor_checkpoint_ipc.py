from types import MethodType
from uuid import uuid4

import msgpack
import zmq

from parallax.server.executor.base_executor import BaseExecutor


def _executor_and_client():
    context = zmq.Context()
    endpoint = f"inproc://fabi-checkpoint-{uuid4().hex}"
    server = context.socket(zmq.REP)
    server.bind(endpoint)
    client = context.socket(zmq.REQ)
    client.connect(endpoint)
    executor = BaseExecutor.__new__(BaseExecutor)
    executor.tp_rank = 0
    executor.executor_control_socket = server
    return executor, client, context


def test_executor_checkpoint_ipc_uses_msgpack_and_separate_binary_frame():
    executor, client, context = _executor_and_client()

    def handle(_self, request, binary_request):
        assert request == {"command": "read", "offset": 0}
        assert binary_request is None
        return {"next_offset": 3, "done": True}, b"kv!"

    executor.handle_executor_control = MethodType(handle, executor)
    try:
        client.send(msgpack.packb({"command": "read", "offset": 0}, use_bin_type=True))
        executor.process_executor_control_requests()
        frames = client.recv_multipart()

        assert len(frames) == 2
        assert msgpack.unpackb(frames[0], raw=False) == {
            "ok": True,
            "result": {"next_offset": 3, "done": True},
        }
        assert frames[1] == b"kv!"
    finally:
        client.close()
        executor.executor_control_socket.close()
        context.term()


def test_executor_checkpoint_ipc_returns_bounded_error_metadata():
    executor, client, context = _executor_and_client()

    def handle(_self, request, binary_request):
        del request, binary_request
        raise ValueError("bad checkpoint")

    executor.handle_executor_control = MethodType(handle, executor)
    try:
        client.send(msgpack.packb({"command": "fail"}, use_bin_type=True))
        executor.process_executor_control_requests()
        frames = client.recv_multipart()
        response = msgpack.unpackb(frames[0], raw=False)

        assert len(frames) == 1
        assert response == {
            "ok": False,
            "error_type": "ValueError",
            "error": "bad checkpoint",
        }
    finally:
        client.close()
        executor.executor_control_socket.close()
        context.term()
