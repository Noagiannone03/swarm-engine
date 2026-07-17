import os
import subprocess
import sys
from pathlib import Path


def test_cuda_modules_import_without_mlx():
    source_root = Path(__file__).parents[2] / "src"
    script = """
import sys
sys.modules["mlx"] = None
sys.modules["mlx_lm"] = None
sys.modules["uvloop"] = None

from parallax.p2p import message_util
from parallax.p2p import utils as p2p_utils
from parallax.server import scheduler
from parallax.utils import tokenizer_utils
from parallax.utils import utils as parallax_utils
from parallax_utils import prepare_adapter

assert message_util.mx is None
assert parallax_utils.mx is None
assert not parallax_utils.is_metal_available()
assert scheduler.Scheduler is not None
assert tokenizer_utils._mlx_load_tokenizer is None
assert prepare_adapter.download_adapter_config is not None
try:
    message_util.bytes_to_tensor(b"invalid", device="mlx")
except RuntimeError as error:
    assert "requires the MLX runtime" in str(error)
else:
    raise AssertionError("MLX serialization did not reject a missing runtime")

try:
    parallax_utils.get_device_dtype("float16", "mlx")
except RuntimeError as error:
    assert "requires the MLX runtime" in str(error)
else:
    raise AssertionError("MLX dtype lookup did not reject a missing runtime")

loop = p2p_utils.switch_to_uvloop()
assert loop is not None
loop.close()
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(source_root)

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0, result.stderr
