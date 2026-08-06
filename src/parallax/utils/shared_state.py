"""
Inter-process communication utilities using multiprocessing.Manager().

Provides a clean abstraction for sharing state between processes (executor, P2P server, etc.)
with dict-like interface and get/set methods.
"""

from __future__ import annotations

import multiprocessing
import time
from typing import Any, Dict, Optional, Union


class SharedState:
    """Wrapper for multiprocessing.Manager().dict() with dict-like interface.

    Supports both dict-like access (shared_state['key']) and method access (shared_state.get('key')).
    Automatically handles conversion from dict to SharedState.
    """

    def __init__(self, manager_dict: Optional[Union[Dict[str, Any], "SharedState"]] = None):
        """Initialize SharedState with a Manager().dict(), dict, SharedState, or None.

        Args:
            manager_dict: A Manager().dict(), regular dict, SharedState instance, or None.
                         If None, creates a new Manager().dict().
                         If dict, wraps it (assumes it's a Manager().dict()).
                         If SharedState, uses its underlying dict.
        """
        if manager_dict is None:
            manager = multiprocessing.Manager()
            self._dict = manager.dict()
        elif isinstance(manager_dict, SharedState):
            self._dict = manager_dict._dict
        else:
            self._dict = manager_dict

    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from shared state."""
        return self._dict.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set a value in shared state."""
        self._dict[key] = value

    def update(self, **kwargs) -> None:
        """Batch update multiple values in shared state.

        Args:
            **kwargs: Key-value pairs to update.
        """
        for key, value in kwargs.items():
            self._dict[key] = value

    def __getitem__(self, key: str) -> Any:
        """Dict-like access: shared_state['key']"""
        return self._dict[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Dict-like access: shared_state['key'] = value"""
        self._dict[key] = value

    def __contains__(self, key: str) -> bool:
        """Check if key exists: 'key' in shared_state"""
        return key in self._dict

    @property
    def dict(self) -> Dict[str, Any]:
        """Get the underlying Manager().dict() for multiprocessing serialization."""
        return self._dict

    def get_metrics(self) -> Dict[str, Any]:
        """Get a shallow copy of current metrics suitable for JSON serialization."""
        metrics_dict = self._dict.get("metrics")
        if not metrics_dict:
            return {}
        # For Manager().dict(), create a copy by accessing each key explicitly
        return {k: metrics_dict[k] for k in metrics_dict.keys()}

    def update_metrics(
        self,
        *,
        current_requests: Optional[int] = None,
        layer_latency_ms_sample: Optional[float] = None,
        prefill_tokens_per_second_sample: Optional[float] = None,
        decode_tokens_per_second_sample: Optional[float] = None,
        ewma_alpha: float = 0.2,
    ) -> None:
        """Update metrics with optional fields and EWMA smoothing for latency.

        Args:
            current_requests: Number of in-flight requests on this node.
            layer_latency_ms_sample: A new sample of per-layer latency in ms.
            prefill_tokens_per_second_sample: Executor-stage prefill throughput sample.
            decode_tokens_per_second_sample: Executor-stage decode throughput sample.
            ewma_alpha: Smoothing factor in [0, 1] for latency EWMA.
        """
        metrics_dict = self._dict.get("metrics")
        if not metrics_dict:
            raise RuntimeError("metrics not initialized in shared_state")

        # Update metrics
        if current_requests is not None:
            metrics_dict["current_requests"] = int(current_requests)
        if layer_latency_ms_sample is not None:
            prev = metrics_dict.get("layer_latency_ms")
            if prev is None:
                metrics_dict["layer_latency_ms"] = float(layer_latency_ms_sample)
            else:
                metrics_dict["layer_latency_ms"] = float(
                    (1.0 - ewma_alpha) * float(prev) + ewma_alpha * float(layer_latency_ms_sample)
                )
        for key, sample in (
            ("prefill_tokens_per_second", prefill_tokens_per_second_sample),
            ("decode_tokens_per_second", decode_tokens_per_second_sample),
        ):
            if sample is None or float(sample) <= 0:
                continue
            previous = metrics_dict.get(key)
            metrics_dict[key] = (
                float(sample)
                if previous is None
                else (1.0 - ewma_alpha) * float(previous) + ewma_alpha * float(sample)
            )
        metrics_dict["_last_update_ts"] = time.time()

    def get_model_info(self) -> Dict[str, Any]:
        """Get model and layer allocation information."""
        return {
            "model_name": self._dict.get("model_name"),
            "model_revision": self._dict.get("model_revision"),
            "block_start_index": self._dict.get("block_start_index"),
            "block_end_index": self._dict.get("block_end_index"),
            "tp_size": self._dict.get("tp_size"),
            "enable_weight_refit": self._dict.get("enable_weight_refit"),
            "weight_refit_mode": self._dict.get("weight_refit_mode"),
            "model_max_sequence_length": self._dict.get("model_max_sequence_length"),
            "planned_context_tokens": self._dict.get("planned_context_tokens"),
            "allocation_epoch": self._dict.get("allocation_epoch"),
            "chunked_prefill_size": self._dict.get("chunked_prefill_size"),
            "_layer_allocation_changed": self._dict.get("_layer_allocation_changed", False),
        }

    def get_layer_allocation_changed(self) -> bool:
        """Check if layer allocation has changed."""
        return self._dict.get("_layer_allocation_changed", False)

    def get_status(self) -> Optional[str]:
        """Get current status."""
        return self._dict.get("status")

    def set_status(self, status: str) -> None:
        """Set current status."""
        self._dict["status"] = status

    @classmethod
    def create(cls) -> "SharedState":
        """Create a new SharedState with default initialization.

        Returns:
            A new SharedState instance with initialized default values.
        """
        manager = multiprocessing.Manager()
        shared_dict = manager.dict()

        # Initialize default values
        shared_dict["block_start_index"] = None
        shared_dict["block_end_index"] = None
        shared_dict["model_name"] = None
        shared_dict["model_revision"] = None
        shared_dict["tp_size"] = None
        shared_dict["enable_weight_refit"] = None
        shared_dict["weight_refit_mode"] = None
        shared_dict["model_max_sequence_length"] = None
        shared_dict["planned_context_tokens"] = None
        shared_dict["allocation_epoch"] = None
        shared_dict["swarm_v3_placement_generation"] = 0
        shared_dict["swarm_v3_placement_phase"] = "legacy"
        shared_dict["swarm_v3_placement_error"] = None
        shared_dict["swarm_v3_previous_start_layer"] = None
        shared_dict["swarm_v3_previous_end_layer"] = None
        shared_dict["swarm_v3_placement_decision"] = None
        shared_dict["capacity_probe_state"] = "disabled"
        shared_dict["capacity_probe_error"] = None
        shared_dict["capacity_probe_device"] = None
        shared_dict["capacity_probe_sequence"] = 0
        shared_dict["capacity_hardware"] = None
        shared_dict["capacity_contract_frozen"] = False
        shared_dict["capacity_probe_stop"] = False
        shared_dict["chunked_prefill_size"] = None
        shared_dict["kv_cache_token_capacity"] = None
        shared_dict["kv_cache_block_size"] = None
        shared_dict["max_concurrent_requests"] = None
        shared_dict["frontend_required"] = False
        shared_dict["frontend_alive"] = False
        shared_dict["_layer_allocation_changed"] = False
        shared_dict["_memory_shutdown_requested"] = False
        shared_dict["memory_pressure"] = "normal"
        shared_dict["memory_pressure_resources"] = {}
        shared_dict["memory_contract_failure"] = None
        shared_dict["storage_contract_failure"] = None
        shared_dict["swarm_v3_storage_status"] = None
        shared_dict["swarm_v3_context_failure"] = None
        shared_dict["status"] = None

        # Create nested shared dict for metrics
        shared_dict["metrics"] = manager.dict()
        shared_dict["metrics"]["current_requests"] = 0
        shared_dict["metrics"]["layer_latency_ms"] = None
        shared_dict["metrics"]["prefill_tokens_per_second"] = None
        shared_dict["metrics"]["decode_tokens_per_second"] = None
        shared_dict["metrics"]["_last_update_ts"] = 0.0

        return cls(shared_dict)
