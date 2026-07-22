from types import SimpleNamespace
from unittest.mock import patch

import mlx.core as mx
import pytest

from parallax.server.cache_manager import CacheManager
from parallax.server.memory_contract import MemoryContractError


def _build_manager(additional_bytes: int) -> CacheManager:
    with patch(
        "parallax.server.cache_manager.current_mlx_memory_budget",
        return_value=SimpleNamespace(
            additional_bytes=additional_bytes,
            process_limit_bytes=int(mx.get_active_memory()) + additional_bytes,
        ),
    ):
        return CacheManager(
            num_layers=1,
            num_kv_heads=1,
            head_dim=4,
            dtype=mx.float32,
            block_size=16,
            cache_memory_fraction=0.10,
            minimum_kv_tokens=128,
        )


def test_mlx_cache_expands_legacy_fraction_to_meet_planned_context():
    manager = _build_manager(8_192)

    assert manager.num_gpu_blocks * manager.block_size >= 128


def test_mlx_cache_fails_before_activation_when_measured_remainder_is_too_small():
    with pytest.raises(MemoryContractError, match="128-token KV contract") as failure:
        _build_manager(4_000)

    assert failure.value.backend == "mlx"
    assert failure.value.requested_tokens == 128
    assert failure.value.supported_tokens < 128
    assert failure.value.as_report(allocation_epoch=7) == {
        "kind": "kv_materialization",
        "backend": "mlx",
        "allocation_epoch": 7,
        "requested_tokens": 128,
        "supported_tokens": failure.value.supported_tokens,
    }
