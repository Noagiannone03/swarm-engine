import json

import pytest

from backend.server.recovery_stream import (
    OpenAIChatReplayStream,
    OpenAIRecoveryStream,
    RecoveryStreamProtocolError,
)


def sse(payload):
    if payload == "[DONE]":
        return b"data: [DONE]\n\n"
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


def test_fragmented_stream_extracts_exact_tokens_and_removes_private_metadata():
    wire = b"".join(
        [
            sse(
                {
                    "id": "chatcmpl-1",
                    "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                    "prompt_token_ids": [10, 20, 30],
                }
            ),
            sse(
                {
                    "id": "chatcmpl-1",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "hello"},
                            "token_ids": [40, 41],
                            "finish_reason": None,
                        }
                    ],
                }
            ),
            sse(
                {
                    "id": "chatcmpl-1",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "token_ids": [42],
                            "finish_reason": "stop",
                        }
                    ],
                }
            ),
            sse("[DONE]"),
        ]
    )
    decoder = OpenAIRecoveryStream()
    events = []
    for byte in wire:
        events.extend(decoder.feed(bytes([byte])))
    decoder.finalize()

    assert events[0].prompt_token_ids == (10, 20, 30)
    assert events[1].output_token_ids == (40, 41)
    assert events[2].output_token_ids == (42,)
    assert events[2].finish_reason == "stop"
    assert events[3].done
    client_wire = b"".join(event.client_bytes for event in events)
    assert b"prompt_token_ids" not in client_wire
    assert b"token_ids" not in client_wire
    assert client_wire.endswith(b"data: [DONE]\n\n")


def test_reasoning_and_tool_tokens_are_committed_even_without_visible_text():
    decoder = OpenAIRecoveryStream(expose_reasoning=False)
    events = decoder.feed(
        sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning": "private"},
                        "token_ids": [7],
                        "finish_reason": None,
                    }
                ]
            }
        )
        + sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"path":"README.md"}'},
                                }
                            ]
                        },
                        "token_ids": [8, 9],
                        "finish_reason": None,
                    }
                ]
            }
        )
    )

    assert [event.output_token_ids for event in events] == [(7,), (8, 9)]
    assert b"private" not in events[0].client_bytes
    assert b"tool_calls" in events[1].client_bytes


def test_explicit_token_id_client_keeps_official_vllm_fields():
    decoder = OpenAIRecoveryStream(expose_token_ids=True)
    event = decoder.feed(
        sse(
            {
                "choices": [{"index": 0, "delta": {}, "token_ids": [3]}],
                "prompt_token_ids": [1, 2],
            }
        )
    )[0]

    assert b'"prompt_token_ids":[1,2]' in event.client_bytes
    assert b'"token_ids":[3]' in event.client_bytes


@pytest.mark.parametrize(
    "wire,match",
    [
        (b"data: not-json\n\n", "not valid JSON"),
        (sse({"choices": [{"index": 1, "delta": {}, "token_ids": [1]}]}), "choice index"),
        (sse({"choices": [{"index": 0, "delta": {}, "token_ids": [True]}]}), "invalid"),
        (sse({"choices": [{"index": 0, "delta": {}, "token_ids": [-1]}]}), "invalid"),
        (sse({"error": {"message": "failed"}}), "SSE error"),
    ],
)
def test_invalid_exact_token_stream_fails_closed(wire, match):
    decoder = OpenAIRecoveryStream()
    with pytest.raises(RecoveryStreamProtocolError, match=match):
        decoder.feed(wire)


def test_changed_prompt_and_bytes_after_done_fail_closed():
    decoder = OpenAIRecoveryStream()
    decoder.feed(sse({"choices": [], "prompt_token_ids": [1]}))
    with pytest.raises(RecoveryStreamProtocolError, match="changed"):
        decoder.feed(sse({"choices": [], "prompt_token_ids": [2]}))

    terminal = OpenAIRecoveryStream()
    terminal.feed(sse("[DONE]"))
    with pytest.raises(RecoveryStreamProtocolError, match="after"):
        terminal.feed(b"x")


def test_truncated_and_oversized_events_fail_closed():
    decoder = OpenAIRecoveryStream()
    decoder.feed(b'data: {"choices":')
    with pytest.raises(RecoveryStreamProtocolError, match="inside"):
        decoder.finalize()

    bounded = OpenAIRecoveryStream(max_event_bytes=8)
    with pytest.raises(RecoveryStreamProtocolError, match="exceeds"):
        bounded.feed(b"data: " + b"x" * 9)


def test_fragmented_chat_replay_suppresses_exact_prefix_and_returns_future_events():
    wire = (
        sse(
            {
                "choices": [{"index": 0, "delta": {"role": "assistant"}}],
                "prompt_token_ids": [10, 20, 30],
            }
        )
        + sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "function": {"arguments": '{"path":"README'},
                                }
                            ]
                        },
                        "token_ids": [40, 41],
                        "finish_reason": None,
                    }
                ],
            }
        )
        + sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": "future"},
                        "token_ids": [42],
                        "finish_reason": None,
                    }
                ],
            }
        )
        + sse(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "token_ids": [],
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        + sse(
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                    "total_tokens": 6,
                    "prompt_tokens_details": {"cached_tokens": 5},
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            }
        )
        + sse("[DONE]")
    )
    decoder = OpenAIChatReplayStream(
        expected_prompt_token_ids=(10, 20, 30),
        committed_output_token_ids=(40, 41),
    )
    events = []
    for byte in wire:
        events.extend(decoder.feed(bytes([byte])))
    decoder.finalize()

    assert decoder.replay_complete
    assert [event.output_token_ids for event in events] == [(42,), (), (), ()]
    assert b"README" not in b"".join(event.client_bytes for event in events)
    assert b"future" in events[0].client_bytes
    assert events[1].finish_reason == "stop"
    usage = json.loads(events[2].client_bytes.removeprefix(b"data: "))
    assert usage["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 3,
        "total_tokens": 6,
    }
    assert events[-1].done


@pytest.mark.parametrize(
    "wire,match",
    [
        (
            sse({"choices": [], "prompt_token_ids": [99]}),
            "prompt token IDs",
        ),
        (
            sse({"choices": [], "prompt_token_ids": [10, 20, 30]})
            + sse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "wrong"},
                            "token_ids": [99],
                        }
                    ]
                }
            ),
            "output token IDs",
        ),
        (
            sse({"choices": [], "prompt_token_ids": [10, 20, 30]})
            + sse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "token_ids": [40, 41, 42],
                        }
                    ]
                }
            ),
            "beyond",
        ),
        (
            sse({"choices": [], "prompt_token_ids": [10, 20, 30]})
            + sse(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "token_ids": [40],
                            "finish_reason": "stop",
                        }
                    ]
                }
            ),
            "terminated before",
        ),
    ],
)
def test_invalid_chat_replay_stream_fails_closed(wire, match):
    decoder = OpenAIChatReplayStream(
        expected_prompt_token_ids=(10, 20, 30),
        committed_output_token_ids=(40, 41),
    )
    with pytest.raises(RecoveryStreamProtocolError, match=match):
        decoder.feed(wire)


def test_chat_replay_done_requires_exact_boundary():
    decoder = OpenAIChatReplayStream(
        expected_prompt_token_ids=(10, 20, 30),
        committed_output_token_ids=(40, 41),
    )
    with pytest.raises(RecoveryStreamProtocolError, match="committed token boundary"):
        decoder.feed(sse({"choices": [], "prompt_token_ids": [10, 20, 30]}) + sse("[DONE]"))
