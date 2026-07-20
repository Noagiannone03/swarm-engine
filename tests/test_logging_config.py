import json
import os
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from parallax.server.executor import factory


def test_process_diagnostic_log_is_scoped_by_session_and_pid(tmp_path: Path):
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    env["PARALLAX_PROCESS_LOG_DIR"] = str(tmp_path)
    env["FABI_WORKER_SESSION_ID"] = "session/with unsafe characters"
    script = """
import json
import os
from pathlib import Path
from parallax_utils.logging_config import get_logger

get_logger("parallax.test").error("diagnostic marker")
files = [path.name for path in Path(os.environ["PARALLAX_PROCESS_LOG_DIR"]).glob("*.log")]
print(json.dumps({"pid": os.getpid(), "files": files}))
"""

    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])

    assert payload["files"] == [f"parallax-session-with-unsafe-characters-{payload['pid']}.log"]
    log_text = (tmp_path / payload["files"][0]).read_text(encoding="utf-8")
    assert f"pid={payload['pid']}" in log_text
    assert "process=MainProcess" in log_text
    assert "[parallax.test] [ERROR]" in log_text
    assert "diagnostic marker" in log_text


def test_vllm_backend_keeps_parallax_logging_by_default(monkeypatch):
    class FakeExecutor:
        def __init__(self, **config):
            self.config = config

    fake_module = types.ModuleType("parallax.server.executor.vllm_executor")
    fake_module.VLLMExecutor = FakeExecutor
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setattr(factory, "create_executor_config", lambda *_args: {})
    monkeypatch.delenv("VLLM_CONFIGURE_LOGGING", raising=False)

    executor = factory.create_from_args(
        SimpleNamespace(gpu_backend="vllm"),
        device="cuda",
    )

    assert isinstance(executor, FakeExecutor)
    assert os.environ["VLLM_CONFIGURE_LOGGING"] == "0"


def test_vllm_backend_preserves_explicit_logging_override(monkeypatch):
    class FakeExecutor:
        def __init__(self, **config):
            self.config = config

    fake_module = types.ModuleType("parallax.server.executor.vllm_executor")
    fake_module.VLLMExecutor = FakeExecutor
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    monkeypatch.setattr(factory, "create_executor_config", lambda *_args: {})
    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "1")

    factory.create_from_args(
        SimpleNamespace(gpu_backend="vllm"),
        device="cuda",
    )

    assert os.environ["VLLM_CONFIGURE_LOGGING"] == "1"
