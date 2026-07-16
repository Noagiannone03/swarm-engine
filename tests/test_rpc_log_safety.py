from backend.server.rpc_connection_handler import node_log_summary


def test_node_log_summary_never_includes_credentials_or_capacity_contracts():
    summary = node_log_summary(
        {
            "node_id": "peer-a",
            "status": "ready",
            "account_token": "secret-account-token",
            "worker_session_id": "private-session",
            "capacity_profile": {"layer_weight_bytes": [1, 2, 3]},
            "hardware": {
                "gpu_name": "Example GPU",
                "device": "cuda",
                "memory_gb": 16,
                "usable_memory_bytes": 12 * 1024**3,
                "node_id": "peer-a",
            },
        }
    )

    assert summary == {
        "node_id": "peer-a",
        "status": "ready",
        "hardware": {
            "gpu_name": "Example GPU",
            "device": "cuda",
            "memory_gb": 16,
            "usable_memory_bytes": 12 * 1024**3,
        },
    }
    assert "secret-account-token" not in repr(summary)
