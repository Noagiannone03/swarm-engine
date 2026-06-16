"""Mid-generation failover for OpenAI-compatible SSE chat streams.

Port of Petals' in-flight recovery (``petals/client/inference_session.py``):
when the serving pipeline drops *after* tokens have already been streamed, we
recover the request transparently instead of truncating the user's answer.

Mapping from Petals' client-orchestrated, per-block model to our scheduler +
pipeline-push model:

    Petals                                  Fabi
    ------                                  ----
    on_request_failure(peer) + Blacklist    peer-reliability ban + fresh routing table
    make_sequence() replacement span        scheduler re-routes to another pipeline
    copy `history` to the new session       keep the assistant text already streamed
    replay the prefix to rebuild KV cache   re-send the original prompt to the new pipeline
    continue from `position`                suppress the regenerated prefix, stream only the tail

Petals can replay the exact KV history to the replacement server; the OpenAI
HTTP surface can't express that, so we re-generate the prefix on the new
pipeline and *suppress* the part the client has already seen. The client never
receives a duplicated or a malformed chunk: in the nominal stream we forward the
worker's bytes untouched, and only on the replacement stream do we synthesize
chunks for the genuinely-new tail.

This module is intentionally dependency-free (stdlib only) so the logic is unit
tested in isolation; the async orchestration lives in
``backend/server/request_handler.py``.
"""

from __future__ import annotations

import json
from typing import Iterator, List, Optional, Tuple

DONE_PAYLOAD = "[DONE]"
DONE_BYTES = b"data: [DONE]\n\n"


def iter_sse_events(chunk: bytes) -> Iterator[Tuple[str, Optional[dict]]]:
    """Yield ``(kind, obj)`` for each SSE ``data:`` line in a raw chunk.

    ``kind`` is ``"done"`` for the ``[DONE]`` sentinel, ``"data"`` for a parsed
    JSON object, or ``"other"`` for anything we can't parse (forwarded as-is by
    callers, never dropped). JSON ``data:`` payloads never contain raw newlines
    (``json.dumps`` escapes them), so a line-based split is safe.
    """
    for raw_line in chunk.split(b"\n"):
        line = raw_line.strip()
        if not line or not line.startswith(b"data:"):
            continue
        payload = line[len(b"data:"):].strip()
        if payload == DONE_PAYLOAD.encode():
            yield "done", None
            continue
        try:
            yield "data", json.loads(payload.decode("utf-8"))
        except Exception:
            yield "other", None


def delta_content(obj: dict) -> str:
    """Assistant text carried by a chat.completion.chunk delta (``""`` if none)."""
    try:
        delta = obj["choices"][0].get("delta") or {}
    except (KeyError, IndexError, TypeError):
        return ""
    content = delta.get("content")
    return content if isinstance(content, str) else ""


def has_tool_calls(obj: dict) -> bool:
    """True if this chunk carries a tool-call delta (not plain text)."""
    try:
        delta = obj["choices"][0].get("delta") or {}
    except (KeyError, IndexError, TypeError):
        return False
    return bool(delta.get("tool_calls"))


def finish_reason(obj: dict) -> Optional[str]:
    """The chunk's ``finish_reason`` (``None`` while generation continues)."""
    try:
        return obj["choices"][0].get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return None


def _synth_chunk(
    template: dict,
    *,
    content: Optional[str],
    keep_finish: bool,
    finish_reason_override: Optional[str] = None,
) -> bytes:
    """Re-serialize a chunk from ``template`` carrying only ``content`` as the tail.

    Preserves id/model/created/usage so downstream metrics keep working. ``role``
    is forced to ``None`` (the client already received the opening ``role`` chunk),
    and ``finish_reason`` is cleared unless ``keep_finish`` (the closing chunk).
    """
    obj = dict(template)
    choices = template.get("choices") or [{}]
    choice = dict(choices[0])
    choice["delta"] = {"role": None, "content": content}
    choice.pop("probs", None)
    choice.pop("token_ids", None)
    if finish_reason_override is not None:
        choice["finish_reason"] = finish_reason_override
    elif not keep_finish:
        choice["finish_reason"] = None
    obj["choices"] = [choice]
    return f"data: {json.dumps(obj, separators=(',', ':'))}\n\n".encode()


def terminal_chunk_from(chunk: bytes, *, finish_reason: str = "length") -> Optional[bytes]:
    """Build a final OpenAI SSE chunk from the latest data event in ``chunk``.

    ``data: [DONE]`` alone is not enough for strict OpenAI clients. They expect a
    final ``chat.completion.chunk`` whose first choice has a non-null
    ``finish_reason`` before the sentinel. When a worker dies mid-stream and all
    resume attempts fail, this helper closes the already-partial answer with a
    protocol-valid terminal chunk.
    """
    template: Optional[dict] = None
    for kind, obj in iter_sse_events(chunk):
        if kind == "data" and obj is not None:
            template = obj
    if template is None:
        return None
    return _synth_chunk(
        template,
        content=None,
        keep_finish=True,
        finish_reason_override=finish_reason,
    )


class ResumableSSEStream:
    """Tracks streamed assistant text so a broken stream can be resumed.

    Usage: feed every nominal chunk through :meth:`feed_original` (raw passthrough
    + bookkeeping). If the upstream breaks before completion and
    :meth:`should_resume` is True, open a replacement upstream and feed its chunks
    through :meth:`feed_resumed`, which suppresses the regenerated prefix and emits
    only the new tail.
    """

    def __init__(self) -> None:
        self.emitted_text = ""  # assistant content the client has already received
        self.completed = False  # saw [DONE] or a finish_reason
        self.resume_safe = True  # cleared if tool-calls appear (text suppression unsafe)
        self._resume_text = ""  # assistant content regenerated on the replacement stream

    # -- nominal stream ----------------------------------------------------- #
    def feed_original(self, chunk: bytes) -> List[bytes]:
        """Forward ``chunk`` unchanged; record state needed for a later resume."""
        for kind, obj in iter_sse_events(chunk):
            if kind == "done":
                self.completed = True
            elif kind == "data" and obj is not None:
                if has_tool_calls(obj):
                    self.resume_safe = False
                self.emitted_text += delta_content(obj)
                if finish_reason(obj) is not None:
                    self.completed = True
        return [chunk]

    def should_resume(self) -> bool:
        """True if the stream broke mid-answer and a text-only resume is safe."""
        return not self.completed and self.resume_safe

    def begin_resume(self) -> None:
        """Reset the per-attempt counter before consuming a replacement stream."""
        self._resume_text = ""

    # -- replacement stream ------------------------------------------------- #
    def feed_resumed(self, chunk: bytes) -> List[bytes]:
        """Consume a replacement-stream chunk, emitting only the genuinely-new tail.

        The replacement re-generates from the original prompt, so its first
        ``len(emitted_text)`` characters repeat what the client already saw and are
        suppressed; everything beyond that is the continuation.
        """
        out: List[bytes] = []
        for kind, obj in iter_sse_events(chunk):
            if kind == "done":
                self.completed = True
                out.append(DONE_BYTES)
                continue
            if kind != "data" or obj is None:
                continue
            if has_tool_calls(obj):
                # A tool-call appeared on the replacement; we can't suppress-match
                # it against plain text. Stop emitting to avoid corrupting output.
                self.resume_safe = False
                continue
            text = delta_content(obj)
            fr = finish_reason(obj)
            if text:
                prev_len = len(self._resume_text)
                self._resume_text += text
                # Characters of THIS chunk that fall beyond the already-seen prefix.
                start = max(0, len(self.emitted_text) - prev_len)
                tail = text[start:] if start < len(text) else ""
                if tail:
                    self.emitted_text += tail
                    out.append(_synth_chunk(obj, content=tail, keep_finish=False))
            if fr is not None:
                self.completed = True
                out.append(_synth_chunk(obj, content=None, keep_finish=True))
        return out
