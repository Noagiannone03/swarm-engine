"""Bounded OpenAI SSE decoding for exact generation recovery.

vLLM's maintained Python and Rust frontends expose ``prompt_token_ids`` in the
first chat-completion chunk and delta ``token_ids`` on each generated update
when ``return_token_ids`` is enabled.  This adapter consumes those official
fields, removes scheduler-private metadata, and returns complete events so the
caller can commit every token before emitting the corresponding bytes.

It intentionally does not reconstruct tokens from text.  Upstream byte chunks
may split or combine SSE events at arbitrary positions.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any

_EVENT_BOUNDARY = re.compile(rb"\r?\n\r?\n")
_MAX_TOKEN_ID = 2**32 - 1


class RecoveryStreamProtocolError(RuntimeError):
    """The upstream stream cannot satisfy the exact-token recovery contract."""


@dataclass(frozen=True)
class RecoveryStreamEvent:
    """One complete client event plus its scheduler-private token metadata."""

    client_bytes: bytes
    prompt_token_ids: tuple[int, ...] | None = None
    output_token_ids: tuple[int, ...] = ()
    done: bool = False
    finish_reason: str | None = None


class OpenAIRecoveryStream:
    """Incrementally decode and sanitize one vLLM chat-completion SSE stream."""

    def __init__(
        self,
        *,
        expose_token_ids: bool = False,
        expose_reasoning: bool = True,
        max_event_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if max_event_bytes <= 0:
            raise ValueError("max_event_bytes must be positive")
        self.expose_token_ids = bool(expose_token_ids)
        self.expose_reasoning = bool(expose_reasoning)
        self.max_event_bytes = int(max_event_bytes)
        self._buffer = bytearray()
        self._prompt_token_ids: tuple[int, ...] | None = None
        self._done = False

    @property
    def prompt_token_ids(self) -> tuple[int, ...] | None:
        return self._prompt_token_ids

    @property
    def done(self) -> bool:
        return self._done

    def feed(self, chunk: bytes) -> tuple[RecoveryStreamEvent, ...]:
        """Consume arbitrary upstream bytes and return all complete SSE events."""

        if not isinstance(chunk, bytes):
            raise TypeError("SSE chunks must be bytes")
        if self._done and chunk:
            raise RecoveryStreamProtocolError("upstream emitted bytes after [DONE]")
        self._buffer.extend(chunk)
        events: list[RecoveryStreamEvent] = []
        while match := _EVENT_BOUNDARY.search(self._buffer):
            if self._done:
                raise RecoveryStreamProtocolError("upstream emitted an event after [DONE]")
            raw_event = bytes(self._buffer[: match.start()])
            del self._buffer[: match.end()]
            if len(raw_event) > self.max_event_bytes:
                raise RecoveryStreamProtocolError("upstream SSE event exceeds the size limit")
            events.append(self._decode_event(raw_event))
        if len(self._buffer) > self.max_event_bytes:
            raise RecoveryStreamProtocolError("unterminated upstream SSE event exceeds the limit")
        return tuple(events)

    def finalize(self) -> None:
        """Validate that transport EOF did not cut an event or omit `[DONE]`."""

        if self._buffer:
            raise RecoveryStreamProtocolError("upstream ended inside an SSE event")
        if not self._done:
            raise RecoveryStreamProtocolError("upstream ended without [DONE]")

    def _decode_event(self, raw_event: bytes) -> RecoveryStreamEvent:
        lines = raw_event.replace(b"\r\n", b"\n").split(b"\n")
        data_lines = []
        passthrough_lines = []
        for line in lines:
            if line.startswith(b"data:"):
                value = line[5:]
                if value.startswith(b" "):
                    value = value[1:]
                data_lines.append(value)
            else:
                passthrough_lines.append(line)

        if not data_lines:
            return RecoveryStreamEvent(client_bytes=raw_event + b"\n\n")
        data = b"\n".join(data_lines)
        if data.strip() == b"[DONE]":
            self._done = True
            return RecoveryStreamEvent(client_bytes=b"data: [DONE]\n\n", done=True)

        try:
            payload = json.loads(data)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RecoveryStreamProtocolError("upstream SSE data is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise RecoveryStreamProtocolError("upstream SSE JSON must be an object")

        prompt_token_ids = None
        if "prompt_token_ids" in payload and payload["prompt_token_ids"] is not None:
            prompt_token_ids = _validated_token_ids(
                payload["prompt_token_ids"],
                field="prompt_token_ids",
            )
            if not prompt_token_ids:
                raise RecoveryStreamProtocolError("prompt_token_ids must not be empty")
            if self._prompt_token_ids is None:
                self._prompt_token_ids = prompt_token_ids
            elif self._prompt_token_ids != prompt_token_ids:
                raise RecoveryStreamProtocolError("prompt_token_ids changed during the stream")

        output_token_ids: list[int] = []
        finish_reason = None
        choices = payload.get("choices", ())
        if not isinstance(choices, list):
            raise RecoveryStreamProtocolError("chat completion choices must be an array")
        for choice in choices:
            if not isinstance(choice, dict):
                raise RecoveryStreamProtocolError("chat completion choice must be an object")
            index = choice.get("index", 0)
            raw_token_ids = choice.get("token_ids")
            if raw_token_ids is not None:
                if index != 0:
                    raise RecoveryStreamProtocolError(
                        "exact recovery currently supports only choice index zero"
                    )
                output_token_ids.extend(
                    _validated_token_ids(raw_token_ids, field="choices[].token_ids")
                )
            reason = choice.get("finish_reason")
            if reason is not None:
                if not isinstance(reason, str):
                    raise RecoveryStreamProtocolError("finish_reason must be a string or null")
                finish_reason = reason
            if not self.expose_reasoning:
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    delta.pop("reasoning", None)
                    delta.pop("reasoning_content", None)
            if not self.expose_token_ids:
                choice.pop("token_ids", None)

        if not self.expose_token_ids:
            payload.pop("prompt_token_ids", None)
        client_data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        prefix = b""
        if passthrough_lines:
            prefix = b"\n".join(passthrough_lines) + b"\n"
        return RecoveryStreamEvent(
            client_bytes=prefix + b"data: " + client_data + b"\n\n",
            prompt_token_ids=prompt_token_ids,
            output_token_ids=tuple(output_token_ids),
            finish_reason=finish_reason,
        )


def _validated_token_ids(value: Any, *, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise RecoveryStreamProtocolError(f"{field} must be an array")
    token_ids = []
    for token_id in value:
        if (
            isinstance(token_id, bool)
            or not isinstance(token_id, int)
            or token_id < 0
            or token_id > _MAX_TOKEN_ID
        ):
            raise RecoveryStreamProtocolError(f"{field} contains an invalid token id")
        token_ids.append(token_id)
    return tuple(token_ids)
