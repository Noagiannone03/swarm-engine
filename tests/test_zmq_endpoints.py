from urllib.parse import urlparse

import pytest
import zmq

from parallax.utils.utils import cleanup_local_zmq_endpoints, create_local_zmq_endpoints


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


def test_windows_style_tcp_endpoints_bind_with_zmq():
    context = zmq.Context()
    sockets = []
    try:
        for endpoint in create_local_zmq_endpoints(4, platform="nt"):
            sock = context.socket(zmq.PULL)
            sock.bind(endpoint)
            sockets.append(sock)
    finally:
        for sock in sockets:
            sock.close(linger=0)
        context.term()


def test_cleanup_removes_only_owned_posix_ipc_files(tmp_path):
    owned = tmp_path / "parallax-zmq-owned"
    foreign = tmp_path / "foreign-service"
    owned.touch()
    foreign.touch()

    cleanup_local_zmq_endpoints(
        [
            f"ipc://{owned}",
            f"ipc://{foreign}",
            "tcp://127.0.0.1:9999",
        ]
    )

    assert not owned.exists()
    assert foreign.exists()
