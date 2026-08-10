#!/usr/bin/env python3
"""Open one signed Skippy span through the production executor path.

This is intentionally a real file (rather than a stdin snippet): macOS uses the
``spawn`` multiprocessing strategy and native runtime initialization must be
safe when the interpreter imports ``__main__`` in a child process.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--start-layer", required=True, type=int)
    parser.add_argument("--end-layer", required=True, type=int)
    parser.add_argument("--device", required=True)
    parser.add_argument("--context-tokens", required=True, type=int)
    parser.add_argument("--max-num-tokens-per-batch", type=int, default=4096)
    return parser


def main() -> int:
    args = _parser().parse_args()

    from parallax.server.executor.skippy_executor import SkippyExecutor

    executor: Any | None = None
    try:
        executor = SkippyExecutor(
            model_repo=args.model,
            model_revision=args.revision,
            start_layer=args.start_layer,
            end_layer=args.end_layer,
            device=args.device,
            max_batch_size=1,
            max_sequence_length=args.context_tokens,
            max_num_tokens_per_batch=args.max_num_tokens_per_batch,
            planned_context_tokens=args.context_tokens,
        )
        print(
            json.dumps(
                {
                    "event": "skippy_span_ready",
                    "backend_device": executor.runner.backend_device,
                    "context_tokens": args.context_tokens,
                    "end_layer": args.end_layer,
                    "execution_plan_id": executor.execution_plan.plan_id,
                    "start_layer": args.start_layer,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        return 0
    finally:
        if executor is not None:
            executor.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
