#!/usr/bin/env python3
"""Qualify a distributed OpenAI chat endpoint with an exact context budget.

The generated request resembles an IDE coding-agent turn: system instructions,
tool schemas, a tool call/result pair, history, and inert TypeScript context.
The prompt is calibrated with the model's canonical chat template so admission
tests are expressed in tokens rather than approximate source bytes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from transformers import AutoTokenizer

DEFAULT_SENTINEL = "FABIOPENCODE-62219"


@dataclass(frozen=True)
class CalibratedRequest:
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    prompt_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:3001")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--target-prompt-tokens", type=int, default=12_220)
    parser.add_argument("--max-completion-tokens", type=int, default=4_096)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--sentinel", default=DEFAULT_SENTINEL)
    parser.add_argument("--expect-http", type=int, default=200)
    parser.add_argument(
        "--bearer-token-file",
        help=(
            "Read the HTTP bearer credential from this file. Use '-' to read one line "
            "from stdin. The credential is never printed or placed in the process arguments."
        ),
    )
    parser.add_argument(
        "--allow-tokenizer-download",
        action="store_true",
        help="Allow AutoTokenizer to contact Hugging Face instead of requiring its local cache.",
    )
    args = parser.parse_args()
    for name in ("target_prompt_tokens", "max_completion_tokens", "repeat"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 100 <= args.expect_http <= 599:
        parser.error("--expect-http must be a valid HTTP status")
    return args


def read_bearer_token(path: str | None) -> str | None:
    """Read an optional bearer credential without accepting it on the command line."""

    if path is None:
        return None
    raw = sys.stdin.readline() if path == "-" else Path(path).read_text(encoding="utf-8")
    token = raw.strip()
    if not token:
        raise ValueError("bearer token file is empty")
    if "\r" in token or "\n" in token:
        raise ValueError("bearer token must be a single line")
    return token


def tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 source file from the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_code",
                "description": "Search source code with a regular expression.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["pattern", "path"],
                    "additionalProperties": False,
                },
            },
        },
    ]


def generated_code(function_count: int) -> str:
    return "\n\n".join(
        f'''export async function normalize_{index:04d}(
  input: UserRecord,
): Promise<Result<UserRecord>> {{
  const values = input.items.map((item) => item.value?.trim() ?? "");
  const valid = values.filter((value) => value.length > 0);
  return {{ ok: true, value: {{ ...input, values: valid }}, source: "module_{index:04d}" }};
}}'''
        for index in range(function_count)
    )


def build_messages(function_count: int, padding: str, sentinel: str) -> list[dict[str, Any]]:
    code = generated_code(function_count)
    final_content = (
        "Review the following generated TypeScript modules for consistency. "
        "This is inert source context; do not execute it.\n\n"
        f"```ts\n{code}{padding}\n```\n"
        "All checks are already complete. Do not call a tool. "
        f"Reply with exactly {sentinel} and nothing else."
    )
    return [
        {
            "role": "system",
            "content": (
                "You are Fabi, a precise coding agent embedded in an IDE. Inspect evidence, "
                "preserve user code, use tools only when needed, and follow the final response "
                "constraint exactly."
            ),
        },
        {
            "role": "user",
            "content": "Inspect src/runtime/router.ts and explain how the active model is selected.",
        },
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_read_router",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"src/runtime/router.ts"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_read_router",
            "name": "read_file",
            "content": (
                "export function selectModel(registry: Registry, id: string) "
                "{ return registry.models.get(id); }"
            ),
        },
        {
            "role": "assistant",
            "content": (
                "The router resolves the requested model ID from the runtime registry and "
                "returns the registered model instance."
            ),
        },
        {"role": "user", "content": final_content},
    ]


def token_count(tokenizer: Any, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> int:
    tokenized = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(tokenized, "keys"):
        tokenized = tokenized["input_ids"]
    if hasattr(tokenized, "tolist"):
        tokenized = tokenized.tolist()
    if tokenized and isinstance(tokenized[0], list):
        tokenized = tokenized[0]
    if not isinstance(tokenized, list):
        raise TypeError("chat template did not return a token ID list")
    return len(tokenized)


def calibrate(tokenizer: Any, target: int, sentinel: str) -> CalibratedRequest:
    tools = tool_schemas()
    low, high = 0, 512
    while low < high:
        middle = (low + high + 1) // 2
        messages = build_messages(middle, "", sentinel)
        if token_count(tokenizer, messages, tools) <= target:
            low = middle
        else:
            high = middle - 1

    function_count = low
    messages = build_messages(function_count, "", sentinel)
    base = token_count(tokenizer, messages, tools)
    if base > target:
        raise ValueError(f"request metadata alone exceeds target prompt size: {base} > {target}")

    padding = " x" * (target - base)
    for _ in range(16):
        messages = build_messages(function_count, padding, sentinel)
        measured = token_count(tokenizer, messages, tools)
        if measured == target:
            return CalibratedRequest(messages, tools, measured)
        if measured < target:
            padding += " x" * (target - measured)
            continue
        removable = min(measured - target, len(padding) // 2)
        padding = padding[: len(padding) - 2 * removable]
    raise RuntimeError(f"unable to calibrate exact prompt size; last measurement was {measured}")


def post_json(
    url: str,
    payload: dict[str, Any],
    timeout: float,
    *,
    bearer_token: str | None = None,
) -> tuple[int, Any]:
    headers = {"Content-Type": "application/json"}
    if bearer_token is not None:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        raw_body = exc.read().decode(errors="replace")
        try:
            body = json.loads(raw_body)
        except json.JSONDecodeError:
            body = raw_body
        return exc.code, body


def main() -> int:
    args = parse_args()
    try:
        bearer_token = read_bearer_token(args.bearer_token_file)
    except (OSError, ValueError) as exc:
        print(f"unable to read bearer credential: {exc}", file=sys.stderr)
        return 2
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=not args.allow_tokenizer_download,
    )
    calibrated = calibrate(tokenizer, args.target_prompt_tokens, args.sentinel)
    required_tokens = calibrated.prompt_tokens + args.max_completion_tokens
    print(
        json.dumps(
            {
                "event": "calibrated",
                "prompt_tokens": calibrated.prompt_tokens,
                "max_completion_tokens": args.max_completion_tokens,
                "required_tokens": required_tokens,
            }
        ),
        flush=True,
    )

    payload = {
        "model": args.model,
        "messages": calibrated.messages,
        "tools": calibrated.tools,
        "chat_template_kwargs": {"enable_thinking": False},
        "max_completion_tokens": args.max_completion_tokens,
        "temperature": 0,
        "stream": False,
    }
    exit_code = 0
    for attempt in range(1, args.repeat + 1):
        started = time.perf_counter()
        status, body = post_json(
            f"{args.url.rstrip('/')}/v1/chat/completions",
            payload,
            args.timeout,
            bearer_token=bearer_token,
        )
        elapsed = time.perf_counter() - started
        content = ""
        if status == 200 and isinstance(body, dict):
            try:
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError):
                pass
        result = {
            "event": "response",
            "attempt": attempt,
            "http_status": status,
            "elapsed_seconds": round(elapsed, 3),
            "content": content,
            "sentinel_exact": content == args.sentinel,
            "sentinel_whitespace_normalized": content.strip() == args.sentinel,
            "usage": body.get("usage") if isinstance(body, dict) else None,
        }
        if status != 200:
            result["error_body"] = body
        print(json.dumps(result, ensure_ascii=False), flush=True)
        if status != args.expect_http:
            exit_code = 1
        if args.expect_http == 200 and content.strip() != args.sentinel:
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
