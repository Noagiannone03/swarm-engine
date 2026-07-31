import os
import subprocess
from pathlib import Path


INSTALL_SCRIPT = Path(__file__).parents[1] / "install.sh"
VLLM_REPLAY_PATCH = (
    Path(__file__).parents[1]
    / "patches"
    / "vllm-v0.24.0-fabi-chat-replay.patch"
)
VLLM_PORTABLE_FRONTEND_PATCH = (
    Path(__file__).parents[1]
    / "patches"
    / "vllm-v0.24.0-portable-frontend.patch"
)
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


def test_frontend_build_pins_and_hashes_all_fabi_patches():
    install_script = INSTALL_SCRIPT.read_text()

    assert VLLM_REPLAY_PATCH.is_file()
    assert VLLM_PORTABLE_FRONTEND_PATCH.is_file()
    assert "ee0da84ab9e04ac7610e28580af62c365e898389" in install_script
    assert 'git -C "$clone_root" apply --unidiff-zero --check' in install_script
    assert 'hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest()' in install_script
    assert "/inference/v1/chat-replay" in VLLM_REPLAY_PATCH.read_text()
    portable_patch = VLLM_PORTABLE_FRONTEND_PATCH.read_text()
    assert "--listen-address" in portable_patch
    assert "cfg(not(unix))" in portable_patch
    assert "cfg(windows)" in portable_patch
    assert 'StdCommand::new("taskkill")' in portable_patch


def test_frontend_only_requires_existing_virtualenv(tmp_path):
    env = os.environ.copy()
    env["PARALLAX_VENV_DIR"] = str(tmp_path / "missing-venv")
    result = subprocess.run(
        ["bash", str(INSTALL_SCRIPT), "--frontend-only"],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 1
    assert "Existing virtualenv is required" in result.stderr


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
