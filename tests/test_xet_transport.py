from types import SimpleNamespace

import httpx

from swarm_protocol import xet_transport


def test_hub_metadata_retries_transient_transport_errors(monkeypatch):
    calls = 0
    sleeps = []

    def fetch(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise httpx.ConnectError("temporary disconnect")
        return "metadata"

    monkeypatch.setattr(xet_transport, "get_hf_file_metadata", fetch)
    monkeypatch.setattr(xet_transport.time, "sleep", sleeps.append)

    assert (
        xet_transport.get_hf_file_metadata_with_backoff(
            "https://huggingface.example/model",
            token=None,
            headers={"user-agent": "fabi-test"},
        )
        == "metadata"
    )
    assert calls == 3
    assert sleeps == [1, 2]


def test_xet_cas_headers_never_receive_hub_authorization():
    headers = {
        "Authorization": "Bearer private-hub-token",
        "user-agent": "fabi-test",
        "X-Custom": "value",
    }

    assert xet_transport.xet_headers_without_auth(headers) == {
        "user-agent": "fabi-test",
        "X-Custom": "value",
    }


def test_xet_session_is_recreated_after_fork_pid_change(monkeypatch):
    created = []

    class Session:
        pass

    fake_module = SimpleNamespace(XetSession=lambda: created.append(Session()) or created[-1])
    monkeypatch.setitem(__import__("sys").modules, "hf_xet", fake_module)
    holder = xet_transport._XetSessionHolder()
    monkeypatch.setattr(xet_transport.os, "getpid", lambda: 10)
    first = holder.get()
    assert holder.get() is first

    monkeypatch.setattr(xet_transport.os, "getpid", lambda: 11)
    second = holder.get()

    assert second is not first
    assert len(created) == 2
