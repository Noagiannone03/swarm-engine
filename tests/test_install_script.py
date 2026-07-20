import os
import subprocess
from pathlib import Path


INSTALL_SCRIPT = Path(__file__).parents[1] / "install.sh"
PORTABILITY_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "check-vllm-rs-portability.sh"
)


def test_install_help_documents_frontend_only_mode():
    result = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "--frontend-only" in result.stdout
    assert "PARALLAX_VENV_DIR" in result.stdout
    assert "PCRE2_SYS_STATIC=1" in INSTALL_SCRIPT.read_text()


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


def test_macos_frontend_dependency_audit_accepts_only_system_libraries(tmp_path):
    binary = tmp_path / "vllm-rs"
    binary.write_text("placeholder")
    tools = tmp_path / "tools"
    tools.mkdir()
    otool = tools / "otool"
    otool.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$2:\"\n"
        "printf '\\t%s\\n' '/usr/lib/libc++.1.dylib (compatibility version 1.0.0)'\n"
        "printf '\\t%s\\n' '/System/Library/Frameworks/Security.framework/Versions/A/Security "
        "(compatibility version 1.0.0)'\n"
    )
    otool.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{tools}:{env['PATH']}"
    env["FABI_PORTABILITY_PLATFORM"] = "Darwin"

    result = subprocess.run(
        ["bash", str(PORTABILITY_SCRIPT), str(binary)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_macos_frontend_dependency_audit_rejects_homebrew(tmp_path):
    binary = tmp_path / "vllm-rs"
    binary.write_text("placeholder")
    tools = tmp_path / "tools"
    tools.mkdir()
    otool = tools / "otool"
    otool.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$2:\"\n"
        "printf '\\t%s\\n' '/opt/homebrew/opt/pcre2/lib/libpcre2-8.0.dylib "
        "(compatibility version 16.0.0)'\n"
    )
    otool.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{tools}:{env['PATH']}"
    env["FABI_PORTABILITY_PLATFORM"] = "Darwin"

    result = subprocess.run(
        ["bash", str(PORTABILITY_SCRIPT), str(binary)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "/opt/homebrew/opt/pcre2" in result.stderr
