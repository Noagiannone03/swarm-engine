from pathlib import Path

from parallax.p2p.server import _resolve_worker_key_path


def test_worker_key_path_is_persistent_and_private(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("PARALLAX_KEY_PATH", raising=False)

    first = Path(_resolve_worker_key_path())
    second = Path(_resolve_worker_key_path())

    assert first == second == tmp_path / ".parallax"
    assert first.is_dir()
    assert first.stat().st_mode & 0o777 == 0o700
