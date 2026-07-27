from __future__ import annotations

import io
import json
import sys
import urllib.error

import pytest

from scripts import qualify_distributed_context


def test_read_bearer_token_is_optional() -> None:
    assert qualify_distributed_context.read_bearer_token(None) is None


def test_read_bearer_token_from_file(tmp_path) -> None:
    token_file = tmp_path / "account-token"
    token_file.write_text("credential-value\n", encoding="utf-8")

    assert (
        qualify_distributed_context.read_bearer_token(str(token_file))
        == "credential-value"
    )


def test_read_bearer_token_from_stdin(monkeypatch) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("credential-value\n"))

    assert qualify_distributed_context.read_bearer_token("-") == "credential-value"


@pytest.mark.parametrize("contents", ["", "\n", "   \n"])
def test_read_bearer_token_rejects_empty_value(tmp_path, contents: str) -> None:
    token_file = tmp_path / "account-token"
    token_file.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        qualify_distributed_context.read_bearer_token(str(token_file))


def test_post_json_adds_bearer_header(monkeypatch) -> None:
    captured = {}

    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b'{"ok":true}'

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    status, body = qualify_distributed_context.post_json(
        "https://scheduler.example/v1/chat/completions",
        {"stream": False},
        12.5,
        bearer_token="credential-value",
    )

    request = captured["request"]
    assert status == 200
    assert body == {"ok": True}
    assert captured["timeout"] == 12.5
    assert request.get_header("Authorization") == "Bearer credential-value"
    assert json.loads(request.data) == {"stream": False}


def test_post_json_omits_authorization_without_token(monkeypatch) -> None:
    class Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self) -> bytes:
            return b"{}"

    captured = {}

    def fake_urlopen(request, *, timeout):
        captured["request"] = request
        return Response()

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    qualify_distributed_context.post_json(
        "https://scheduler.example/v1/chat/completions",
        {},
        1,
    )

    assert captured["request"].get_header("Authorization") is None
