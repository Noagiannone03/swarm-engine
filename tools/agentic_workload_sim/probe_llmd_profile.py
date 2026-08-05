"""Probe the pinned llm-d Fabi workload profile using only the Python stdlib."""

from __future__ import annotations

import argparse
import http.client
import json
import re
import statistics
import threading
import time
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CompletionProbe:
    status: int
    elapsed_seconds: float
    ttft_seconds: float | None
    chunks: int
    content_chunks: int
    body_prefix: str = ""


def completion_payload(
    *,
    prompt_tokens: int,
    max_output_tokens: int,
    stream: bool,
    force_exact_output: bool,
) -> bytes:
    if prompt_tokens <= 0 or max_output_tokens <= 0:
        raise ValueError("prompt and output token counts must be positive")
    payload: dict[str, Any] = {
        "model": "fabi-qwen3-4b-v3",
        "prompt": list(range(prompt_tokens)),
        "max_tokens": max_output_tokens,
        "stream": stream,
    }
    if force_exact_output:
        payload["ignore_eos"] = True
    return json.dumps(payload, separators=(",", ":")).encode()


def prometheus_gauge(metrics: str, name: str) -> int:
    match = re.search(rf"^{re.escape(name)}\{{[^}}]+\}} ([0-9.]+)$", metrics, re.MULTILINE)
    if match is None:
        raise ValueError(f"Prometheus gauge is missing: {name}")
    return int(float(match.group(1)))


def request_completion(
    host: str,
    port: int,
    *,
    prompt_tokens: int,
    max_output_tokens: int,
    stream: bool,
    force_exact_output: bool,
    timeout_seconds: float,
) -> CompletionProbe:
    body = completion_payload(
        prompt_tokens=prompt_tokens,
        max_output_tokens=max_output_tokens,
        stream=stream,
        force_exact_output=force_exact_output,
    )
    connection = http.client.HTTPConnection(host, port, timeout=timeout_seconds)
    started = time.monotonic()
    connection.request(
        "POST",
        "/v1/completions",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if not stream:
        response_body = response.read().decode("utf-8", errors="replace")
        return CompletionProbe(
            status=response.status,
            elapsed_seconds=time.monotonic() - started,
            ttft_seconds=None,
            chunks=0,
            content_chunks=0,
            body_prefix=response_body[:256],
        )

    first_content: float | None = None
    chunks = 0
    content_chunks = 0
    for raw_line in response:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        event = json.loads(line[6:])
        chunks += 1
        if event.get("choices", [{}])[0].get("text", ""):
            content_chunks += 1
            if first_content is None:
                first_content = time.monotonic() - started
    return CompletionProbe(
        status=response.status,
        elapsed_seconds=time.monotonic() - started,
        ttft_seconds=first_content,
        chunks=chunks,
        content_chunks=content_chunks,
    )


def probe_single(host: str, port: int) -> dict[str, Any]:
    result = request_completion(
        host,
        port,
        prompt_tokens=12_220,
        max_output_tokens=7,
        stream=True,
        force_exact_output=True,
        timeout_seconds=60,
    )
    return {
        "scenario": "open_code_prefill",
        "prompt_tokens": 12_220,
        # Fabi reserves this separately. The latency simulator generates only
        # seven tokens so a calibration run does not sleep for nine minutes.
        "reserved_output_tokens": 4_096,
        "simulated_output_tokens": 7,
        **asdict(result),
    }


def probe_overflow(host: str, port: int) -> dict[str, Any]:
    result = request_completion(
        host,
        port,
        prompt_tokens=62_000,
        max_output_tokens=4_096,
        stream=False,
        force_exact_output=False,
        timeout_seconds=10,
    )
    return {
        "scenario": "context_overflow",
        "required_context_tokens": 66_096,
        **asdict(result),
    }


def _metrics(host: str, port: int) -> str:
    with urllib.request.urlopen(f"http://{host}:{port}/metrics", timeout=2) as response:
        return response.read().decode("utf-8")


def probe_saturation(host: str, port: int) -> dict[str, Any]:
    barrier = threading.Barrier(9)

    def run_one() -> CompletionProbe:
        barrier.wait()
        return request_completion(
            host,
            port,
            prompt_tokens=1_000,
            max_output_tokens=7,
            stream=False,
            force_exact_output=True,
            timeout_seconds=30,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures: list[Future[CompletionProbe]] = [
            executor.submit(run_one) for _ in range(8)
        ]
        barrier.wait()
        max_running = 0
        max_waiting = 0
        while not all(future.done() for future in futures):
            metrics = _metrics(host, port)
            max_running = max(
                max_running,
                prometheus_gauge(metrics, "vllm:num_requests_running"),
            )
            max_waiting = max(
                max_waiting,
                prometheus_gauge(metrics, "vllm:num_requests_waiting"),
            )
            time.sleep(0.05)
        results = [future.result() for future in futures]

    durations = sorted(result.elapsed_seconds for result in results)
    return {
        "scenario": "eight_requests_four_slots",
        "statuses": [result.status for result in results],
        "max_running": max_running,
        "max_waiting": max_waiting,
        "durations_seconds": [round(value, 3) for value in durations],
        "median_seconds": round(statistics.median(durations), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "scenario",
        choices=("single", "saturation", "overflow"),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18_080)
    args = parser.parse_args()
    if not 1 <= args.port <= 65_535:
        parser.error("port must be between 1 and 65535")
    probe = {
        "single": probe_single,
        "saturation": probe_saturation,
        "overflow": probe_overflow,
    }[args.scenario]
    print(json.dumps(probe(args.host, args.port), sort_keys=True))


if __name__ == "__main__":
    main()
