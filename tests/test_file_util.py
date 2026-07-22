from pathlib import Path

import pytest

from parallax_utils.file_util import get_project_root


def test_get_project_root_prefers_valid_explicit_source(monkeypatch, tmp_path):
    source = tmp_path / "parallax-source"
    source.mkdir()
    (source / "pyproject.toml").write_text("[project]\nname = 'parallax'\n")
    monkeypatch.setenv("PARALLAX_SRC", str(source))

    assert get_project_root() == source.resolve()


def test_get_project_root_rejects_invalid_explicit_source(monkeypatch, tmp_path):
    missing_source = tmp_path / "missing-source"
    monkeypatch.setenv("PARALLAX_SRC", str(missing_source))

    with pytest.raises(FileNotFoundError, match="PARALLAX_SRC"):
        get_project_root()


def test_get_project_root_discovers_repository_without_override(monkeypatch):
    monkeypatch.delenv("PARALLAX_SRC", raising=False)

    root = get_project_root()

    assert root == Path(__file__).resolve().parents[1]
    assert (root / "pyproject.toml").is_file()
