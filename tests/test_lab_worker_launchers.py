from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CURRENT_DHT_BOOTSTRAPS = (
    "12D3KooWMQrc1rWXwaeQcshtANiw9FyyGWmfqAnVsStGRqsJ54Yi",
    "12D3KooWG8jJaC1upci3eDZ7XSobTPC5hT6bdqFGzzptH8q7b1eG",
)


def _launcher(relative_path: str) -> str:
    return (REPOSITORY_ROOT / relative_path).read_text(encoding="utf-8")


def test_macos_iroh_launcher_uses_the_installed_skippy_contract() -> None:
    launcher = _launcher("deploy/worker/lab/macos-iroh.sh")

    assert 'manifest_path="$state/MANIFEST"' in launcher
    assert 'execution_engine="$(manifest_value execution_engine)"' in launcher
    assert 'execution_device="$(manifest_value execution_device)"' in launcher
    assert '[[ "$execution_engine" == "skippy" ]]' in launcher
    assert '[[ "$execution_device" == "metal" ]]' in launcher
    assert "--gpu-backend skippy" in launcher
    assert '--execution-device "$execution_device"' in launcher
    assert "--gpu-backend sglang" not in launcher
    assert "--gpu-backend vllm" not in launcher
    assert "--max-sequence-length 32768" in launcher
    assert "--max-num-tokens-per-batch 4096" in launcher
    for peer_id in CURRENT_DHT_BOOTSTRAPS:
        assert peer_id in launcher


def test_windows_iroh_launcher_uses_the_installed_skippy_contract() -> None:
    launcher = _launcher("deploy/worker/lab/windows-iroh.ps1")

    assert '$manifestPath = Join-Path $state "MANIFEST"' in launcher
    assert '$manifest = Read-RuntimeManifest -Path $manifestPath' in launcher
    assert '$manifest.execution_engine -ne "skippy"' in launcher
    assert '$executionDevice = $manifest.execution_device' in launcher
    assert "--gpu-backend skippy" in launcher
    assert "--execution-device $executionDevice" in launcher
    assert "--gpu-backend sglang" not in launcher
    assert "--gpu-backend vllm" not in launcher
    assert "--max-sequence-length 32768" in launcher
    assert "--max-num-tokens-per-batch 8192" in launcher
    for peer_id in CURRENT_DHT_BOOTSTRAPS:
        assert peer_id in launcher


def test_skippy_span_qualification_is_spawn_safe() -> None:
    source = _launcher("tools/qualify_skippy_span.py")

    assert 'if __name__ == "__main__"' in source
    assert "SkippyExecutor(" in source
    assert "executor.shutdown()" in source
    assert '"event": "skippy_span_ready"' in source


def test_macos_request_agent_launcher_keeps_credentials_out_of_argv() -> None:
    launcher = _launcher("deploy/worker/lab/macos-request-agent.sh")

    assert 'export FABI_ACCOUNT_TOKEN="$(<"$account_token_file")"' in launcher
    assert "screen env" not in launcher
    assert "FABI_REQUEST_AGENT_MODEL_SWARM_ID" in launcher
    assert "FABI_REQUEST_AGENT_AUTHORITY_URL" in launcher
    assert "--ready-file" in launcher
    for peer_id in CURRENT_DHT_BOOTSTRAPS:
        assert peer_id in launcher
