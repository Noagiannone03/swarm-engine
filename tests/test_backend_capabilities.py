from types import SimpleNamespace

import numpy as np
import pytest

from parallax.server.backend_capabilities import (
    DeviceKind,
    TensorRuntime,
    canonical_device_for_rank,
    capability_for_device,
    detect_best_device,
    device_kind,
    require_executor_backend,
    normalize_ort_provider,
    tensor_runtime_for_device,
)
from parallax.utils import utils, weight_refit_utils


def fake_torch(*, cuda: bool = False, xpu: bool = False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        xpu=SimpleNamespace(is_available=lambda: xpu),
    )


def test_device_registry_separates_tensor_runtime_from_executor_engine():
    assert tensor_runtime_for_device("cuda:1") is TensorRuntime.TORCH
    assert tensor_runtime_for_device("xpu:0") is TensorRuntime.TORCH
    assert tensor_runtime_for_device("cpu") is TensorRuntime.TORCH
    assert tensor_runtime_for_device("mlx") is TensorRuntime.MLX
    assert tensor_runtime_for_device("directml:0") is TensorRuntime.NUMPY
    assert tensor_runtime_for_device("openvino") is TensorRuntime.NUMPY
    assert capability_for_device("xpu").executor_backends == frozenset({"sglang"})
    assert capability_for_device("cpu").executor_backends == frozenset({"skippy"})


def test_device_parser_and_rank_canonicalization_fail_closed():
    assert device_kind("XPU:2") is DeviceKind.XPU
    assert canonical_device_for_rank("cuda", 3) == "cuda:3"
    assert canonical_device_for_rank("xpu:1", 3) == "xpu:1"
    assert canonical_device_for_rank("mlx", 0) == "mlx"
    assert device_kind("directml") is DeviceKind.DIRECTML
    assert canonical_device_for_rank("directml", 2) == "directml:2"
    with pytest.raises(ValueError, match="Invalid execution device"):
        device_kind("cpu:0")


def test_executor_registry_accepts_only_real_device_engine_pairs():
    assert require_executor_backend("mlx", "sglang") == "mlx"
    assert require_executor_backend("cuda", "vllm") == "vllm"
    assert require_executor_backend("xpu", "sglang") == "sglang"
    with pytest.raises(ValueError, match="none qualified"):
        require_executor_backend("winml", "onnxruntime")
    with pytest.raises(ValueError, match="not supported on xpu"):
        require_executor_backend("xpu", "vllm")
    assert require_executor_backend("cpu", "skippy") == "skippy"
    with pytest.raises(ValueError, match="not supported on cpu"):
        require_executor_backend("cpu", "sglang")


def test_best_device_preserves_mlx_priority_then_cuda_xpu_cpu():
    assert (
        detect_best_device(torch_module=fake_torch(cuda=True, xpu=True), metal_available=True)
        == "mlx"
    )
    assert (
        detect_best_device(torch_module=fake_torch(cuda=True, xpu=True), metal_available=False)
        == "cuda"
    )
    assert detect_best_device(torch_module=fake_torch(xpu=True), metal_available=False) == "xpu"
    assert (
        detect_best_device(
            torch_module=fake_torch(),
            metal_available=False,
            ort_providers=("OpenVINOExecutionProvider", "CPUExecutionProvider"),
        )
        == "openvino"
    )
    assert (
        detect_best_device(
            torch_module=fake_torch(),
            metal_available=False,
            ort_providers=("DmlExecutionProvider",),
        )
        == "directml"
    )
    assert detect_best_device(torch_module=fake_torch(), metal_available=False) == "cpu"


def test_ort_provider_names_are_normalized_without_guessing_unknown_plugins():
    assert normalize_ort_provider("WindowsML") is DeviceKind.WINML
    assert normalize_ort_provider("QNNExecutionProvider") is DeviceKind.QNN
    assert normalize_ort_provider("SomeExperimentalExecutionProvider") is None


def test_portable_device_detection_does_not_require_torch(monkeypatch):
    monkeypatch.setattr(utils, "torch", None)
    monkeypatch.setattr(
        utils,
        "available_ort_execution_providers",
        lambda: ("DmlExecutionProvider", "CPUExecutionProvider"),
    )
    monkeypatch.setattr(utils, "is_metal_available", lambda: False)

    assert utils.is_cuda_available() is False
    assert utils.is_mps_available() is False
    assert utils.get_current_device() == "directml"
    assert utils.get_device_dtype("float16", "directml") is np.float16
    with pytest.raises(RuntimeError, match="PyTorch runtime"):
        utils.get_device_dtype("float16", "cpu")


def test_portable_server_keeps_torch_weight_refit_optional(monkeypatch):
    monkeypatch.setattr(weight_refit_utils, "torch", None)

    assert weight_refit_utils.calculate_cid_manual(b"fabi").startswith("b")
    with pytest.raises(RuntimeError, match="Weight refit requires"):
        weight_refit_utils.concat_weight_partition({})
    with pytest.raises(RuntimeError, match="Weight refit requires"):
        weight_refit_utils.parse_safetensors_from_memory(b"")
