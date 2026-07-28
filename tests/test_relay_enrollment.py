from __future__ import annotations

import json
from pathlib import Path

import pytest

from fabi_network.relay_enrollment import RelayEnrollmentClient


class FakeResponse:
    status = 200

    def __init__(self, payload: object) -> None:
        self._body = json.dumps(payload).encode()

    def read(self, limit: int) -> bytes:
        return self._body[:limit]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args


def test_enrollment_sends_endpoint_owned_proof_and_validates_lease(monkeypatch, tmp_path):
    credential = "ab" * 32
    endpoint_id = "cd" * 32
    signature = "ef" * 64
    captured = {}
    monkeypatch.setattr("fabi_network.relay_enrollment.time.time", lambda: 1_800_000_000.0)
    monkeypatch.setattr("fabi_network.relay_enrollment.secrets.token_hex", lambda _: "12" * 32)

    def proof(identity_path, account_id, issued_at_ms, nonce):
        captured.update(
            identity_path=identity_path,
            account_id=account_id,
            issued_at_ms=issued_at_ms,
            nonce=nonce,
        )
        return endpoint_id, signature

    def urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse(
            {
                "apiVersion": "v1",
                "lease": {
                    "endpoint_id": endpoint_id,
                    "enrolled_at_ms": 1_800_000_000_000,
                    "expires_at_ms": 1_800_086_400_000,
                    "refresh_at_ms": 1_800_021_600_000,
                },
            }
        )

    monkeypatch.setattr("fabi_network.relay_enrollment._create_proof", proof)
    monkeypatch.setattr("fabi_network.relay_enrollment.urllib.request.urlopen", urlopen)
    identity_path = tmp_path / "worker.key"
    client = RelayEnrollmentClient(
        "https://registry.example/v1/network/enroll",
        credential,
        identity_path,
    )
    lease = client.enroll()

    assert lease.endpoint_id == endpoint_id
    assert captured["identity_path"] == identity_path
    assert captured["nonce"] == "12" * 32
    request = captured["request"]
    assert request.get_header("Authorization") == f"Bearer {credential}"
    assert json.loads(request.data)["signature"] == signature
    # The account credential is hashed before entering the native signed proof.
    assert captured["account_id"] != credential


@pytest.mark.parametrize(
    "url",
    [
        "http://registry.example/v1/network/enroll",
        "ftp://registry.example/enroll",
        "https://user:password@registry.example/enroll",
    ],
)
def test_enrollment_rejects_insecure_or_credentialed_urls(url, tmp_path):
    with pytest.raises(ValueError):
        RelayEnrollmentClient(url, "ab" * 32, tmp_path / "worker.key")


def test_environment_requires_account_credential(monkeypatch, tmp_path):
    del tmp_path
    monkeypatch.setenv("FABI_RELAY_ENROLLMENT_URL", "https://registry.example/enroll")
    monkeypatch.delenv("FABI_ACCOUNT_TOKEN", raising=False)
    with pytest.raises(ValueError, match="FABI_ACCOUNT_TOKEN"):
        RelayEnrollmentClient.from_environment(Path("worker.key"))
