import os
from urllib.parse import urlparse

import pytest
import zmq

from parallax.utils.utils import create_local_zmq_endpoints
from parallax.server.capacity_profile import detect_http_frontend_capability


def test_rejects_empty_endpoint_batch():
    with pytest.raises(ValueError, match="at least 1"):
        create_local_zmq_endpoints(0)


def test_posix_endpoints_are_unique_ipc_paths():
    endpoints = create_local_zmq_endpoints(8, platform="posix")

    assert len(endpoints) == len(set(endpoints)) == 8
    assert all(endpoint.startswith("ipc://") for endpoint in endpoints)
    assert all("parallax-zmq-" in endpoint for endpoint in endpoints)


def test_windows_endpoints_are_distinct_private_tcp_ports():
    endpoints = create_local_zmq_endpoints(8, platform="nt")

    assert len(endpoints) == len(set(endpoints)) == 8
    for endpoint in endpoints:
        parsed = urlparse(endpoint)
        assert parsed.scheme == "tcp"
        assert parsed.hostname == "127.0.0.1"
        assert 0 < (parsed.port or 0) <= 65535


def test_native_windows_worker_does_not_claim_unix_rust_frontend():
    capability = detect_http_frontend_capability(platform="nt")

    assert capability["available"] is False
    assert capability["protocol"] == "vllm-engine-core-v1"


@pytest.mark.skipif(os.name == "nt", reason="the native Windows run covers the platform branch")
def test_windows_style_tcp_endpoints_bind_with_zmq_on_unix_too():
    endpoints = create_local_zmq_endpoints(4, platform="nt")
    context = zmq.Context()
    sockets = []
    try:
        for endpoint in endpoints:
            sock = context.socket(zmq.PULL)
            sock.bind(endpoint)
            sockets.append(sock)
    finally:
        for sock in sockets:
            sock.close(linger=0)
        context.term()
