from types import SimpleNamespace

from parallax.server.executor.base_executor import BaseExecutor
from parallax.utils.shared_state import SharedState


class _TensorLike:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


def test_backend_adapters_expose_exact_scheduled_token_counts():
    assert (
        BaseExecutor._processed_token_count(
            "prefill_batch",
            {"requests": [object()], "actual_processed_lengths": _TensorLike([100, 20])},
        )
        == 120
    )
    assert (
        BaseExecutor._processed_token_count(
            "prefill_batch",
            {
                "requests": [object()],
                "scheduler_output": SimpleNamespace(total_num_scheduled_tokens=96),
            },
        )
        == 96
    )
    assert (
        BaseExecutor._processed_token_count(
            "prefill_batch",
            {
                "requests": [object()],
                "forward_batch": object(),
                "context_lengths": _TensorLike([32, 64]),
            },
        )
        == 96
    )
    assert (
        BaseExecutor._processed_token_count("decode_batch", {"requests": [object(), object()]}) == 2
    )


def test_shared_metrics_keep_separate_prefill_and_decode_ewmas():
    state = SharedState(
        {
            "metrics": {
                "current_requests": 0,
                "layer_latency_ms": None,
                "prefill_tokens_per_second": None,
                "decode_tokens_per_second": None,
                "_last_update_ts": 0.0,
            }
        }
    )

    state.update_metrics(
        prefill_tokens_per_second_sample=100,
        decode_tokens_per_second_sample=20,
    )
    state.update_metrics(
        prefill_tokens_per_second_sample=200,
        decode_tokens_per_second_sample=40,
        ewma_alpha=0.25,
    )

    metrics = state.get_metrics()
    assert metrics["prefill_tokens_per_second"] == 125
    assert metrics["decode_tokens_per_second"] == 25
