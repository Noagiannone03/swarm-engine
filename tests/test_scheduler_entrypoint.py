from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_scheduler_entrypoint_executes_exactly_one_initialized_process(tmp_path: Path) -> None:
    calls = tmp_path / "calls"
    fake_parallax = tmp_path / "parallax"
    fake_parallax.write_text(
        "#!/bin/sh\n"
        "printf '%s\\n' \"$*\" >> \"$FABI_TEST_CALLS\"\n",
        encoding="utf-8",
    )
    fake_parallax.chmod(0o755)

    repository = Path(__file__).resolve().parents[1]
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{tmp_path}{os.pathsep}{environment['PATH']}",
            "FABI_TEST_CALLS": str(calls),
            "PARALLAX_MODEL": "Qwen/Qwen3-4B",
            "PARALLAX_WORKERS": "2",
            # This legacy variable must no longer start a second init path.
            "PARALLAX_AUTO_INIT": "1",
        }
    )
    subprocess.run(
        ["/bin/sh", str(repository / "deploy/scheduler/entrypoint.sh")],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )

    assert calls.read_text(encoding="utf-8").splitlines() == [
        "run --host 0.0.0.0 --port 3001 --tcp-port 18080 "
        "--udp-port 18080 -m Qwen/Qwen3-4B -n 2"
    ]
