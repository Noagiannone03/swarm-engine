"""Automatic, endpoint-owned enrollment for the authenticated Iroh relay."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_HEX_32 = re.compile(r"^[0-9a-f]{64}$")
_HEX_SIGNATURE = re.compile(r"^[0-9a-f]{128}$")
_MAX_RESPONSE_BYTES = 16 * 1024
_DEFAULT_HTTP_TIMEOUT_SECONDS = 10.0
_MIN_REFRESH_DELAY_SECONDS = 30.0
_MAX_RETRY_DELAY_SECONDS = 30 * 60.0


@dataclass(frozen=True)
class RelayEnrollmentLease:
    endpoint_id: str
    enrolled_at_ms: int
    expires_at_ms: int
    refresh_at_ms: int


def _account_id(credential: str) -> str:
    normalized = credential.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized):
        raise ValueError("FABI_ACCOUNT_TOKEN must be a 32-byte hexadecimal credential")
    return hashlib.sha256(normalized.encode("ascii")).hexdigest()


def _create_proof(
    identity_path: Path,
    account_id: str,
    issued_at_ms: int,
    nonce: str,
) -> tuple[str, str]:
    try:
        import fabi_network_native
    except ImportError as error:  # pragma: no cover - release packaging boundary
        raise RuntimeError(
            "fabi-network-native is required for automatic relay enrollment"
        ) from error
    endpoint_id, signature = fabi_network_native.create_relay_enrollment_proof(
        identity_path,
        account_id,
        issued_at_ms,
        nonce,
    )
    return str(endpoint_id), str(signature)


class RelayEnrollmentClient:
    """Create and refresh a short-lived relay authorization lease.

    The refresh thread is deliberately independent from model generation and
    worker heartbeats. A long prefill/decode can therefore never starve network
    authorization renewal.
    """

    def __init__(
        self,
        enrollment_url: str,
        account_credential: str,
        identity_path: Path,
        *,
        timeout_seconds: float = _DEFAULT_HTTP_TIMEOUT_SECONDS,
    ) -> None:
        self.enrollment_url = _validated_enrollment_url(enrollment_url)
        self._credential = account_credential.strip().lower()
        self._account_id = _account_id(self._credential)
        self._identity_path = identity_path
        if not timeout_seconds > 0:
            raise ValueError("relay enrollment timeout must be greater than zero")
        self._timeout_seconds = timeout_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lease_lock = threading.Lock()
        self._lease: RelayEnrollmentLease | None = None

    @classmethod
    def from_environment(cls, identity_path: Path) -> RelayEnrollmentClient | None:
        url = os.environ.get("FABI_RELAY_ENROLLMENT_URL", "").strip()
        if not url:
            return None
        credential = os.environ.get("FABI_ACCOUNT_TOKEN", "").strip()
        if not credential:
            raise ValueError("FABI_ACCOUNT_TOKEN is required for automatic relay enrollment")
        raw_timeout = os.environ.get(
            "FABI_RELAY_ENROLLMENT_TIMEOUT_SECONDS",
            str(_DEFAULT_HTTP_TIMEOUT_SECONDS),
        ).strip()
        try:
            timeout = float(raw_timeout)
        except ValueError as error:
            raise ValueError(
                "FABI_RELAY_ENROLLMENT_TIMEOUT_SECONDS must be a positive number"
            ) from error
        return cls(url, credential, identity_path, timeout_seconds=timeout)

    @property
    def lease(self) -> RelayEnrollmentLease | None:
        with self._lease_lock:
            return self._lease

    def enroll(self) -> RelayEnrollmentLease:
        issued_at_ms = int(time.time() * 1_000)
        nonce = secrets.token_hex(32)
        endpoint_id, signature = _create_proof(
            self._identity_path,
            self._account_id,
            issued_at_ms,
            nonce,
        )
        if not _HEX_32.fullmatch(endpoint_id) or not _HEX_SIGNATURE.fullmatch(signature):
            raise RuntimeError("native relay enrollment proof has an invalid encoding")
        request_body = json.dumps(
            {
                "endpoint_id": endpoint_id,
                "issued_at_ms": issued_at_ms,
                "nonce": nonce,
                "signature": signature,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        request = urllib.request.Request(
            self.enrollment_url,
            data=request_body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self._credential}",
                "Content-Type": "application/json",
                "User-Agent": "fabi-worker/relay-enrollment-v1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                payload = _read_bounded_json(response)
                status = response.status
        except urllib.error.HTTPError as error:
            code = _error_code(error)
            raise RuntimeError(f"relay enrollment was rejected ({error.code}: {code})") from error
        except urllib.error.URLError as error:
            raise RuntimeError(f"relay enrollment endpoint is unreachable: {error.reason}") from error
        if status != 200:
            raise RuntimeError(f"relay enrollment returned unexpected HTTP status {status}")
        lease = _parse_lease(payload, endpoint_id, issued_at_ms)
        with self._lease_lock:
            self._lease = lease
        return lease

    def start_refresh(self, initial_lease: RelayEnrollmentLease) -> None:
        if self._thread is not None:
            raise RuntimeError("relay enrollment refresh is already running")
        with self._lease_lock:
            self._lease = initial_lease
        self._thread = threading.Thread(
            target=self._refresh_loop,
            name="fabi-relay-enrollment",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _refresh_loop(self) -> None:
        retry_delay = _MIN_REFRESH_DELAY_SECONDS
        while not self._stop.is_set():
            current = self.lease
            now_ms = int(time.time() * 1_000)
            refresh_at_ms = current.refresh_at_ms if current is not None else now_ms
            delay = max(_MIN_REFRESH_DELAY_SECONDS, (refresh_at_ms - now_ms) / 1_000)
            if self._stop.wait(delay):
                return
            try:
                self.enroll()
                retry_delay = _MIN_REFRESH_DELAY_SECONDS
            except Exception as error:
                # The current lease remains usable until its explicit expiry.
                # Retry is bounded and independent from worker inference load.
                logger.warning("relay enrollment refresh failed; will retry: %s", error)
                if self._stop.wait(retry_delay):
                    return
                retry_delay = min(_MAX_RETRY_DELAY_SECONDS, retry_delay * 2)


def _validated_enrollment_url(value: str) -> str:
    parsed = urllib.parse.urlparse(value.strip())
    local_http = parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1", "localhost"}
    if parsed.scheme != "https" and not local_http:
        raise ValueError("FABI_RELAY_ENROLLMENT_URL must use HTTPS")
    if not parsed.netloc or parsed.username or parsed.password or parsed.fragment:
        raise ValueError("FABI_RELAY_ENROLLMENT_URL is malformed")
    return value.strip()


def _read_bounded_json(response: Any) -> object:
    encoded = response.read(_MAX_RESPONSE_BYTES + 1)
    if len(encoded) > _MAX_RESPONSE_BYTES:
        raise RuntimeError("relay enrollment response exceeds the size limit")
    try:
        return json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("relay enrollment returned invalid JSON") from error


def _error_code(error: urllib.error.HTTPError) -> str:
    try:
        payload = _read_bounded_json(error)
    except Exception:
        return "request_rejected"
    if isinstance(payload, dict) and isinstance(payload.get("error"), str):
        return payload["error"]
    return "request_rejected"


def _parse_lease(payload: object, endpoint_id: str, issued_at_ms: int) -> RelayEnrollmentLease:
    if not isinstance(payload, dict) or payload.get("apiVersion") != "v1":
        raise RuntimeError("relay enrollment response has an unsupported contract")
    raw = payload.get("lease")
    if not isinstance(raw, dict) or raw.get("endpoint_id") != endpoint_id:
        raise RuntimeError("relay enrollment lease is not bound to this endpoint")
    fields = [raw.get(name) for name in ("enrolled_at_ms", "expires_at_ms", "refresh_at_ms")]
    if any(not isinstance(value, int) or isinstance(value, bool) for value in fields):
        raise RuntimeError("relay enrollment lease timestamps are invalid")
    enrolled_at_ms, expires_at_ms, refresh_at_ms = fields
    if not issued_at_ms - 5 * 60_000 <= enrolled_at_ms <= issued_at_ms + 5 * 60_000:
        raise RuntimeError("relay enrollment lease has an invalid server timestamp")
    if not enrolled_at_ms < refresh_at_ms < expires_at_ms:
        raise RuntimeError("relay enrollment lease lifetime is invalid")
    return RelayEnrollmentLease(endpoint_id, enrolled_at_ms, expires_at_ms, refresh_at_ms)
