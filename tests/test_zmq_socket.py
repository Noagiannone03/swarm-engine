from __future__ import annotations

import uuid

import pytest
import zmq

from parallax.utils.utils import get_zmq_socket


@pytest.mark.parametrize("server_type,client_type", [(zmq.REP, zmq.REQ), (zmq.PAIR, zmq.PAIR)])
def test_get_zmq_socket_supports_bidirectional_control_pairs(
    server_type: zmq.SocketType,
    client_type: zmq.SocketType,
) -> None:
    context = zmq.Context()
    endpoint = f"inproc://fabi-control-{uuid.uuid4()}"
    server = get_zmq_socket(context, server_type, endpoint, bind=True)
    client = get_zmq_socket(context, client_type, endpoint, bind=False)
    try:
        client.send(b"checkpoint")
        assert server.recv() == b"checkpoint"
        server.send(b"committed")
        assert client.recv() == b"committed"
    finally:
        client.close(linger=0)
        server.close(linger=0)
        context.term()


def test_get_zmq_socket_rejects_unknown_patterns() -> None:
    context = zmq.Context()
    try:
        with pytest.raises(ValueError, match="Unsupported socket type"):
            get_zmq_socket(context, zmq.SUB, "inproc://unsupported", bind=True)
    finally:
        context.term()
