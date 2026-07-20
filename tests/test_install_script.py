import os
import subprocess
from pathlib import Path


INSTALL_SCRIPT = Path(__file__).parents[1] / "install.sh"


def test_install_help_documents_frontend_only_mode():
    result = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--frontend-only" in result.stdout
    assert "PARALLAX_VENV_DIR" in result.stdout


def test_frontend_only_requires_existing_posix_virtualenv(tmp_path):
    env = os.environ.copy()
    env["PARALLAX_VENV_DIR"] = str(tmp_path / "missing-venv")
    result = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), "--frontend-only"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Existing POSIX virtualenv is required" in result.stderr
