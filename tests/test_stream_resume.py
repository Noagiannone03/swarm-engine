"""Unit tests for the mid-generation SSE resume logic (pure, no swarm needed)."""

import json

from parallax_utils.stream_resume import (
    ResumableSSEStream,
    delta_content,
    finish_reason,
    has_tool_calls,
    iter_sse_events,
    terminal_chunk_from,
)


def chunk(content=None, *, role=None, fr=None, tool_calls=None, done=False) -> bytes:
    if done:
        return b"data: [DONE]\n\n"
    delta = {"role": role, "content": content}
    if tool_calls is not None:
        delta["tool_calls"] = tool_calls
    obj = {
        "id": "rid-1",
        "object": "chat.completion.chunk",
        "model": "Qwen3-Coder",
        "created": 1700000000,
        "choices": [{"index": 0, "logprobs": None, "finish_reason": fr, "delta": delta}],
        "usage": {"prompt_tokens": 10, "total_tokens": 12, "completion_tokens": 2},
    }
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode()


def _text_of(chunks):
    """Concatenate the assistant content from a list of emitted SSE byte-chunks."""
    out = ""
    for c in chunks:
        for kind, obj in iter_sse_events(c):
            if kind == "data":
                out += delta_content(obj)
    return out


# --------------------------------------------------------------------------- #
# Parsing helpers                                                             #
# --------------------------------------------------------------------------- #
def test_iter_sse_events_parses_data_done_and_garbage():
    raw = chunk("hi") + b"data: not-json\n\n" + b"data: [DONE]\n\n"
    kinds = [k for k, _ in iter_sse_events(raw)]
    assert kinds == ["data", "other", "done"]


def test_delta_and_finish_and_tool_helpers():
    obj = json.loads(chunk("x", fr="stop").split(b"data: ")[1])
    assert delta_content(obj) == "x"
    assert finish_reason(obj) == "stop"
    assert has_tool_calls(obj) is False
    tobj = json.loads(chunk(None, tool_calls=[{"id": "t"}]).split(b"data: ")[1])
    assert has_tool_calls(tobj) is True


def test_terminal_chunk_from_last_data_chunk():
    terminal = terminal_chunk_from(chunk("partial", role="assistant"), finish_reason="length")
    assert terminal is not None
    events = [(k, o) for k, o in iter_sse_events(terminal)]
    assert len(events) == 1
    kind, obj = events[0]
    assert kind == "data"
    assert delta_content(obj) == ""
    assert finish_reason(obj) == "length"


# --------------------------------------------------------------------------- #
# Nominal stream bookkeeping                                                  #
# --------------------------------------------------------------------------- #
def test_feed_original_passes_through_and_tracks_text():
    s = ResumableSSEStream()
    assert s.feed_original(chunk("Hello ", role="assistant")) == [chunk("Hello ", role="assistant")]
    s.feed_original(chunk("world"))
    assert s.emitted_text == "Hello world"
    assert s.should_resume() is True  # not completed yet


def test_completed_stream_does_not_resume():
    s = ResumableSSEStream()
    s.feed_original(chunk("done answer"))
    s.feed_original(chunk(None, fr="stop"))
    s.feed_original(chunk(done=True))
    assert s.completed is True
    assert s.should_resume() is False


def test_tool_calls_disable_resume():
    s = ResumableSSEStream()
    s.feed_original(chunk("let me call ", role="assistant"))
    s.feed_original(chunk(None, tool_calls=[{"id": "t", "function": {"name": "f"}}]))
    assert s.resume_safe is False
    assert s.should_resume() is False


# --------------------------------------------------------------------------- #
# Resume: prefix suppression + tail emission                                  #
# --------------------------------------------------------------------------- #
def test_resume_suppresses_seen_prefix_and_emits_tail():
    s = ResumableSSEStream()
    # Client already saw "The capital of France"
    s.feed_original(chunk("The capital ", role="assistant"))
    s.feed_original(chunk("of France"))
    assert s.emitted_text == "The capital of France"
    assert s.should_resume() is True

    # Replacement regenerates the whole answer from the prompt, in different chunking.
    s.begin_resume()
    emitted = []
    emitted += s.feed_resumed(chunk("The capital of France is Paris.", role="assistant"))
    emitted += s.feed_resumed(chunk(None, fr="stop"))
    emitted += s.feed_resumed(chunk(done=True))

    # Only the NEW tail must reach the client — no duplication of the prefix.
    assert _text_of(emitted) == " is Paris."
    assert s.emitted_text == "The capital of France is Paris."
    assert s.completed is True
    # The closing chunk preserves finish_reason, and [DONE] is forwarded.
    finishes = [o for c in emitted for k, o in iter_sse_events(c) if k == "data"]
    assert any(finish_reason(o) == "stop" for o in finishes)
    assert emitted[-1] == b"data: [DONE]\n\n"


def test_resume_when_prefix_split_across_chunks():
    s = ResumableSSEStream()
    s.feed_original(chunk("abcdef", role="assistant"))  # emitted = "abcdef"

    s.begin_resume()
    emitted = []
    # Replacement streams "ab","cde","fGH","IJ" -> tail beyond "abcdef" is "GHIJ"
    for piece in ["ab", "cde", "fGH", "IJ"]:
        emitted += s.feed_resumed(chunk(piece, role="assistant"))
    assert _text_of(emitted) == "GHIJ"
    assert s.emitted_text == "abcdefGHIJ"


def test_resume_emits_full_replacement_when_nothing_seen_yet():
    s = ResumableSSEStream()
    s.feed_original(chunk(None, role="assistant"))  # role only, no content emitted
    assert s.emitted_text == ""
    s.begin_resume()
    emitted = s.feed_resumed(chunk("Full answer.", role="assistant"))
    assert _text_of(emitted) == "Full answer."


def test_resume_bails_if_tool_call_appears_on_replacement():
    s = ResumableSSEStream()
    s.feed_original(chunk("partial ", role="assistant"))
    s.begin_resume()
    out = s.feed_resumed(chunk(None, tool_calls=[{"id": "t"}]))
    assert out == []
    assert s.resume_safe is False
