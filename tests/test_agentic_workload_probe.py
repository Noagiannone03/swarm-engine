from __future__ import annotations

import runpy


PROBE = runpy.run_path("tools/agentic_workload_sim/probe_llmd_profile.py")


def test_completion_payload_preserves_exact_prompt_and_output_contract() -> None:
    payload = PROBE["completion_payload"](
        prompt_tokens=12_220,
        max_output_tokens=7,
        stream=True,
        force_exact_output=True,
    )

    assert payload.count(b",") >= 12_220
    assert b'"max_tokens":7' in payload
    assert b'"ignore_eos":true' in payload
    assert b'"stream":true' in payload


def test_prometheus_gauge_reads_the_qualified_metric_surface() -> None:
    metrics = (
        '# TYPE vllm:num_requests_waiting gauge\n'
        'vllm:num_requests_waiting{model_name="fabi-qwen3-4b-v3"} 4\n'
    )

    assert PROBE["prometheus_gauge"](
        metrics, "vllm:num_requests_waiting"
    ) == 4
