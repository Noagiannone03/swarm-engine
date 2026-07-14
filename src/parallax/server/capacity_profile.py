"""Worker-owned, model-specific capacity contracts.

The scheduler must not guess how many layers a remote device can host.  A
worker builds this contract from the exact safetensors metadata of the model it
will load and from its locally enforced memory budget.  The scheduler consumes
only the resulting admissible layer ranges.

The contract is deliberately marked ``provisional`` until the executor has
loaded the shard and reported its real KV block capacity.  Runtime measurements
remain the final authority; metadata is used only to choose the first placement.
"""

from __future__ import annotations

import os
import re
import time
from typing import Dict, Mapping, Optional

from huggingface_hub import get_safetensors_metadata

from parallax.utils.weight_filter_utils import normalize_language_model_weight_key
from parallax_utils.logging_config import get_logger
from scheduling.model_info import ModelInfo

logger = get_logger(__name__)

CAPACITY_PROTOCOL_VERSION = 1
_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


def _tensor_size_bytes(tensor: object) -> int:
    offsets = getattr(tensor, "data_offsets", None)
    if not offsets or len(offsets) != 2:
        return 0
    return max(0, int(offsets[1]) - int(offsets[0]))


def _layer_index(name: str) -> Optional[int]:
    match = _LAYER_RE.search(normalize_language_model_weight_key(name))
    return int(match.group(1)) if match else None


def _classify_tensor_sizes(
    tensor_sizes: Mapping[str, int], num_layers: int
) -> tuple[list[int], int, int, list[str]]:
    """Return exact per-layer, first-stage and last-stage weight bytes.

    Unknown non-layer tensors make a profile unsafe.  They are returned to the
    caller instead of being silently ignored, so unsupported architectures do
    not get an optimistic capacity contract.
    """

    layers = [0] * num_layers
    first_stage = 0
    last_stage = 0
    unknown: list[str] = []

    for raw_name, raw_size in tensor_sizes.items():
        name = normalize_language_model_weight_key(raw_name)
        size = max(0, int(raw_size))
        idx = _layer_index(name)
        if idx is not None:
            if 0 <= idx < num_layers:
                layers[idx] += size
            else:
                unknown.append(raw_name)
            continue

        if "embed_tokens" in name:
            first_stage += size
        elif "lm_head" in name or "model.norm" in name or name.endswith(".norm.weight"):
            last_stage += size
        elif name == "__metadata__":
            continue
        else:
            # Biases/scales outside a decoder block are architecture-specific.
            # Refuse to pretend they consume zero bytes.
            unknown.append(raw_name)

    return layers, first_stage, last_stage, unknown


def _compute_max_ends(
    *,
    layer_weights: list[int],
    first_stage_weight_bytes: int,
    last_stage_weight_bytes: int,
    tied_embedding: bool,
    kv_bytes_per_token_by_layer: list[int],
    target_context_tokens: int,
    budget_bytes: int,
) -> list[int]:
    num_layers = len(layer_weights)
    max_ends: list[int] = []
    for start in range(num_layers):
        weight_bytes = first_stage_weight_bytes if start == 0 else 0
        best_end = start
        for end in range(start + 1, num_layers + 1):
            weight_bytes += layer_weights[end - 1]
            endpoint_bytes = last_stage_weight_bytes if end == num_layers else 0
            if tied_embedding and start == 0 and end == num_layers:
                endpoint_bytes -= first_stage_weight_bytes
            kv_bytes = target_context_tokens * sum(kv_bytes_per_token_by_layer[start:end])
            if weight_bytes + endpoint_bytes + kv_bytes <= budget_bytes:
                best_end = end
            else:
                break
        max_ends.append(best_end)
    return max_ends


def build_capacity_profile_from_tensor_sizes(
    *,
    model_name: str,
    tensor_sizes: Mapping[str, int],
    model_info: ModelInfo,
    usable_memory_bytes: int,
    target_context_tokens: int,
    runtime_reserve_bytes: int,
    backend: str,
    generated_at: Optional[float] = None,
) -> dict:
    """Build the worker's admissible contiguous ranges.

    ``max_end_by_start[s]`` is the largest exclusive end layer the worker says
    it can host when its shard starts at ``s``.  Endpoint weights and one full
    request worth of KV storage at ``target_context_tokens`` are included.
    Non-uniform MoE/quantized layer sizes are preserved rather than averaged.
    """

    num_layers = int(model_info.num_layers)
    if num_layers <= 0:
        raise ValueError("model has no decoder layers")
    if usable_memory_bytes <= 0:
        raise ValueError("worker usable memory budget is not positive")
    if target_context_tokens <= 0:
        raise ValueError("target context must be positive")

    layer_weights, first_stage, last_stage, unknown = _classify_tensor_sizes(
        tensor_sizes, num_layers
    )
    if unknown:
        sample = ", ".join(unknown[:5])
        raise ValueError(
            f"unsupported non-layer tensors in {model_name}: {sample}"
            + (f" (+{len(unknown) - 5} more)" if len(unknown) > 5 else "")
        )
    if any(size <= 0 for size in layer_weights):
        missing = [str(i) for i, size in enumerate(layer_weights) if size <= 0]
        raise ValueError(f"missing safetensors metadata for layers {', '.join(missing[:10])}")

    tied_embedding = bool(getattr(model_info, "tie_embedding", False))
    if tied_embedding:
        last_stage += first_stage

    unsupported_reason = getattr(model_info, "capacity_profile_unsupported_reason", None)
    if unsupported_reason:
        raise ValueError(f"unsupported cache geometry: {unsupported_reason}")
    exact_kv = getattr(model_info, "kv_bytes_per_token_by_layer", None)
    if exact_kv is None:
        kv_by_layer = [int(model_info.per_token_per_layer_kv_size)] * num_layers
    else:
        kv_by_layer = [int(value) for value in exact_kv]
    if len(kv_by_layer) != num_layers or any(value <= 0 for value in kv_by_layer):
        raise ValueError("exact per-layer KV geometry is unavailable")

    budget = int(usable_memory_bytes) - max(0, int(runtime_reserve_bytes))
    if budget <= 0:
        raise ValueError("runtime reserve consumes the worker memory budget")

    max_end_by_start = _compute_max_ends(
        layer_weights=layer_weights,
        first_stage_weight_bytes=first_stage,
        last_stage_weight_bytes=last_stage,
        tied_embedding=tied_embedding,
        kv_bytes_per_token_by_layer=kv_by_layer,
        target_context_tokens=target_context_tokens,
        budget_bytes=budget,
    )

    return {
        "protocol_version": CAPACITY_PROTOCOL_VERSION,
        "model_name": model_name,
        "backend": backend,
        "state": "provisional",
        "source": "safetensors-metadata+worker-budget",
        "generated_at": float(generated_at if generated_at is not None else time.time()),
        "num_layers": num_layers,
        "target_context_tokens": int(target_context_tokens),
        "usable_memory_bytes": int(usable_memory_bytes),
        "runtime_reserve_bytes": max(0, int(runtime_reserve_bytes)),
        "kv_bytes_per_token_by_layer": kv_by_layer,
        "first_stage_weight_bytes": int(first_stage),
        "last_stage_weight_bytes": int(last_stage),
        "tie_embedding": tied_embedding,
        "layer_weight_bytes": [int(size) for size in layer_weights],
        "max_end_by_start": max_end_by_start,
    }


def calibrate_profile_from_runtime(
    profile: Mapping[str, object],
    *,
    start_layer: int,
    end_layer: int,
    kv_capacity_tokens: int,
) -> dict:
    """Conservatively recalibrate a provisional contract from a loaded shard.

    The executor's actual KV block pool is the final memory authority.  If it is
    smaller than the target assumed by metadata planning, derive the effective
    weight+KV budget observed for this concrete shard and recompute every
    admissible range.  This never increases a worker's advertised budget.
    """

    result = dict(profile)
    target = int(profile.get("target_context_tokens", 0))
    actual = max(0, int(kv_capacity_tokens))
    if actual >= target or target <= 0:
        return result

    layers = [int(value) for value in profile.get("layer_weight_bytes", [])]
    num_layers = int(profile.get("num_layers", 0))
    start, end = int(start_layer), int(end_layer)
    if len(layers) != num_layers or not (0 <= start < end <= num_layers):
        raise ValueError("runtime calibration range does not match capacity profile")

    first = int(profile.get("first_stage_weight_bytes", 0))
    last = int(profile.get("last_stage_weight_bytes", 0))
    tied = bool(profile.get("tie_embedding", False))
    reserve = int(profile.get("runtime_reserve_bytes", 0))
    kv_by_layer = [int(value) for value in profile.get("kv_bytes_per_token_by_layer", [])]
    if len(kv_by_layer) != num_layers or any(value <= 0 for value in kv_by_layer):
        raise ValueError("runtime calibration is missing KV geometry")

    loaded_weights = sum(layers[start:end])
    if start == 0:
        loaded_weights += first
    if end == num_layers:
        loaded_weights += last
    if tied and start == 0 and end == num_layers:
        loaded_weights -= first
    observed_budget = reserve + loaded_weights + actual * sum(kv_by_layer[start:end])
    old_budget = int(profile.get("usable_memory_bytes", observed_budget))
    budget = min(old_budget, observed_budget) - reserve

    max_ends = _compute_max_ends(
        layer_weights=layers,
        first_stage_weight_bytes=first,
        last_stage_weight_bytes=last,
        tied_embedding=tied,
        kv_bytes_per_token_by_layer=kv_by_layer,
        target_context_tokens=target,
        budget_bytes=budget,
    )

    result.update(
        {
            "state": "runtime_calibrated",
            "source": "runtime-kv-block-measurement",
            "generated_at": time.time(),
            "usable_memory_bytes": max(0, budget + reserve),
            "max_end_by_start": max_ends,
            "calibrated_from": {
                "start_layer": start,
                "end_layer": end,
                "kv_capacity_tokens": actual,
            },
        }
    )
    return result


def constrain_profile_to_memory(profile: Mapping[str, object], usable_memory_bytes: int) -> dict:
    """Shrink a cached contract when the worker's governed budget drops."""

    result = dict(profile)
    old_usable = int(profile.get("usable_memory_bytes", 0))
    new_usable = max(0, min(old_usable, int(usable_memory_bytes)))
    if new_usable >= old_usable:
        return result
    layers = [int(value) for value in profile.get("layer_weight_bytes", [])]
    kv_by_layer = [int(value) for value in profile.get("kv_bytes_per_token_by_layer", [])]
    if not layers or len(layers) != len(kv_by_layer):
        raise ValueError("capacity profile cannot be constrained safely")
    reserve = int(profile.get("runtime_reserve_bytes", 0))
    result.update(
        {
            "state": "runtime_calibrated",
            "source": "worker-memory-governor",
            "generated_at": time.time(),
            "usable_memory_bytes": new_usable,
            "max_end_by_start": _compute_max_ends(
                layer_weights=layers,
                first_stage_weight_bytes=int(profile.get("first_stage_weight_bytes", 0)),
                last_stage_weight_bytes=int(profile.get("last_stage_weight_bytes", 0)),
                tied_embedding=bool(profile.get("tie_embedding", False)),
                kv_bytes_per_token_by_layer=kv_by_layer,
                target_context_tokens=int(profile.get("target_context_tokens", 0)),
                budget_bytes=max(0, new_usable - reserve),
            ),
        }
    )
    return result


def _remote_tensor_sizes(model_name: str) -> Dict[str, int]:
    metadata = get_safetensors_metadata(model_name)
    tensors: Dict[str, int] = {}
    for file_metadata in metadata.files_metadata.values():
        for name, tensor in file_metadata.tensors.items():
            tensors[name] = _tensor_size_bytes(tensor)
    if not tensors:
        raise ValueError(f"no safetensors metadata found for {model_name}")
    return tensors


def build_worker_capacity_profile(
    *,
    model_name: str,
    model_info: ModelInfo,
    hardware: Mapping[str, object],
    target_context_tokens: int,
) -> dict:
    """Network-backed worker entry point used during capacity negotiation."""

    usable = int(float(hardware.get("usable_memory_bytes") or 0))
    backend = str(hardware.get("device") or "unknown")
    reserve_gb = 0.25 if backend == "mlx" else 0.75
    raw_reserve = os.environ.get("PARALLAX_GPU_RUNTIME_WORKSPACE_GB", "")
    if raw_reserve.strip():
        try:
            reserve_gb = max(0.0, float(raw_reserve))
        except ValueError:
            logger.warning("Ignoring PARALLAX_GPU_RUNTIME_WORKSPACE_GB=%r", raw_reserve)

    return build_capacity_profile_from_tensor_sizes(
        model_name=model_name,
        tensor_sizes=_remote_tensor_sizes(model_name),
        model_info=model_info,
        usable_memory_bytes=usable,
        target_context_tokens=target_context_tokens,
        runtime_reserve_bytes=int(reserve_gb * 1024**3),
        backend=backend,
    )


def profile_allows_range(profile: Mapping[str, object], start: int, end: int) -> bool:
    """Validate a concrete range against an untrusted JSON contract."""

    try:
        if int(profile.get("protocol_version", 0)) != CAPACITY_PROTOCOL_VERSION:
            return False
        max_ends = profile.get("max_end_by_start")
        if not isinstance(max_ends, list) or start < 0 or start >= len(max_ends):
            return False
        return start < end <= int(max_ends[start])
    except (TypeError, ValueError):
        return False
