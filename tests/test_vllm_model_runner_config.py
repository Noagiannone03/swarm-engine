from types import SimpleNamespace

import pytest

model_runner = pytest.importorskip("parallax.vllm.model_runner", exc_type=ImportError)


def test_vllm_runner_disables_async_scheduling():
    config = model_runner._build_scheduler_config(
        max_num_batched_tokens=2048,
        max_num_seqs=1,
        max_model_len=2048,
    )

    assert config.async_scheduling is False


def test_vllm_runner_bounds_stale_worker_batch_budget_to_allocation_context():
    config = model_runner._build_scheduler_config(
        max_num_batched_tokens=65536,
        max_num_seqs=1,
        max_model_len=32768,
    )

    assert config.max_num_batched_tokens == 32768


def test_vllm_runner_keeps_useful_multi_request_batch_budget():
    config = model_runner._build_scheduler_config(
        max_num_batched_tokens=65536,
        max_num_seqs=4,
        max_model_len=32768,
    )

    assert config.max_num_batched_tokens == 65536


def test_extracts_synchronous_sampled_token_ids():
    token_ids = [[17]]

    device_ids, cpu_ids = model_runner._extract_sampled_token_ids(
        SimpleNamespace(sampled_token_ids=token_ids)
    )

    assert device_ids is token_ids
    assert cpu_ids.dtype is model_runner.torch.int64
    assert cpu_ids.tolist() == token_ids


def test_extracts_asynchronous_sampled_token_ids():
    device_ids = object()
    cpu_ids = object()

    actual_device_ids, actual_cpu_ids = model_runner._extract_sampled_token_ids(
        SimpleNamespace(
            _sampled_token_ids=device_ids,
            sampled_token_ids_cpu=cpu_ids,
        )
    )

    assert actual_device_ids is device_ids
    assert actual_cpu_ids is cpu_ids
