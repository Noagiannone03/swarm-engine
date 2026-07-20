"""Compatibility helpers for the vLLM request API.

The native Windows vLLM fork carries the post-0.16 EOS API while official
vLLM 0.16 still expects ``eos_token_id`` on ``Request``.  Keep this difference
at the adapter boundary instead of branching throughout the batch scheduler.
"""

from __future__ import annotations

import inspect
from typing import Any


def create_vllm_request(
    request_type: type,
    *,
    sampling_params: Any,
    eos_token_id: int | None,
    **request_kwargs: Any,
) -> Any:
    """Construct a vLLM request across the old and new EOS API contracts.

    Official vLLM 0.16 receives EOS on ``Request``.  Newer vLLM code, including
    the SystemPanic Windows 0.16 wheel, stores it on ``SamplingParams`` instead.
    Signature inspection is intentional: catching ``TypeError`` from the
    constructor would also hide genuine bugs raised inside vLLM.
    """

    parameters = inspect.signature(request_type).parameters
    if "eos_token_id" in parameters:
        request_kwargs["eos_token_id"] = eos_token_id
    elif eos_token_id is not None:
        update_generation_config = getattr(sampling_params, "update_from_generation_config", None)
        if not callable(update_generation_config) or not hasattr(sampling_params, "eos_token_id"):
            raise RuntimeError(
                "Unsupported vLLM request API: Request has no eos_token_id parameter "
                "and SamplingParams cannot store the model EOS token"
            )
        update_generation_config({}, eos_token_id)

    return request_type(sampling_params=sampling_params, **request_kwargs)
