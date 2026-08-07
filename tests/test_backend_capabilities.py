from types import SimpleNamespace

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
    with pytest.raises(ValueError, match="none qualified"):
        require_executor_backend("cpu", "sglang")


def test_best_device_preserves_mlx_priority_then_cuda_xpu_cpu():
    assert detect_best_device(torch_module=fake_torch(cuda=True, xpu=True), metal_available=True) == "mlx"
    assert detect_best_device(torch_module=fake_torch(cuda=True, xpu=True), metal_available=False) == "cuda"
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
