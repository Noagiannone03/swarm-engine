"""Stable adapters for authenticated Hugging Face Xet range streams.

Fabi uses the public ``hf_xet.XetSession`` byte-stream API directly.  Keeping
session ownership here avoids depending on private ``huggingface_hub`` helpers,
whose availability is unrelated to the Xet wire protocol and differs between
the Hub releases supported by Transformers 4 and 5.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import httpx
import requests
from huggingface_hub import get_hf_file_metadata

_HUB_METADATA_RETRIES = 5
_HUB_NETWORK_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
)

logger = logging.getLogger(__name__)


class _XetSessionHolder:
    """Own one thread-safe Xet runtime per process.

    A Rust async runtime inherited through ``fork()`` cannot safely be reused.
    Comparing the creator PID makes the lazy singleton safe for both worker
    threads and forked model processes without relying on Hub internals.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._session: Any | None = None
        self._session_pid: int | None = None

    def get(self) -> Any:
        with self._lock:
            current_pid = os.getpid()
            if self._session is not None and self._session_pid != current_pid:
                self._session = None
                self._session_pid = None
            if self._session is None:
                from hf_xet import XetSession

                self._session = XetSession()
                self._session_pid = current_pid
            return self._session


_XET_SESSION_HOLDER = _XetSessionHolder()


def get_xet_session() -> Any:
    """Return the process-local session from the maintained ``hf-xet`` API."""

    return _XET_SESSION_HOLDER.get()


def xet_headers_without_auth(headers: dict[str, str]) -> dict[str, str]:
    """Do not leak a Hub bearer token to the separate Xet CAS endpoint."""

    return {key: value for key, value in headers.items() if key.lower() != "authorization"}


def get_hf_file_metadata_with_backoff(
    url: str,
    *,
    token: bool | str | None,
    headers: dict[str, str],
) -> Any:
    """Fetch immutable Hub metadata across the supported HTTP client generations.

    ``huggingface_hub`` 0.36 uses Requests and has no public retry flag here;
    the 1.x line uses HTTPX.  Retry transport failures only, while authentication,
    repository and integrity errors remain immediate failures.
    """

    for attempt in range(_HUB_METADATA_RETRIES + 1):
        try:
            return get_hf_file_metadata(url, token=token, headers=headers)
        except _HUB_NETWORK_ERRORS as exc:
            if attempt == _HUB_METADATA_RETRIES:
                raise
            delay = min(2**attempt, 8)
            logger.warning(
                "Hub file metadata fetch failed (%s); retrying in %ds",
                type(exc).__name__,
                delay,
            )
            time.sleep(delay)
    raise AssertionError("unreachable Hub metadata retry state")
