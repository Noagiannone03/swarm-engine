from types import SimpleNamespace
from unittest.mock import patch

from parallax.utils import utils
from parallax.utils.layer_types import (
    DSA_ATTENTION,
    LINEAR,
    MLA_ATTENTION,
    MSA_ATTENTION,
)


def test_is_metal_available_uses_mlx_metal_is_available(monkeypatch):
    fake_mx = SimpleNamespace(metal=SimpleNamespace(is_available=lambda: True))

    monkeypatch.setattr(utils, "mx", fake_mx)

    assert utils.is_metal_available() is True


def test_is_metal_available_returns_false_when_metal_api_missing(monkeypatch):
    fake_mx = SimpleNamespace()

    monkeypatch.setattr(utils, "mx", fake_mx)

    assert utils.is_metal_available() is False


def test_get_current_device_prefers_mlx_when_metal_available(monkeypatch):
    monkeypatch.setattr(utils, "is_cuda_available", lambda: False)
    monkeypatch.setattr(utils, "is_metal_available", lambda: True)
    monkeypatch.setattr(utils, "available_ort_execution_providers", lambda: ())

    assert utils.get_current_device() == "mlx"


def test_get_current_device_prefers_mlx_when_both_backends_report_available(monkeypatch):
    monkeypatch.setattr(utils, "is_cuda_available", lambda: True)
    monkeypatch.setattr(utils, "is_metal_available", lambda: True)
    monkeypatch.setattr(utils, "available_ort_execution_providers", lambda: ())

    assert utils.get_current_device() == "mlx"


def test_get_current_device_uses_xpu_when_cuda_and_mlx_are_unavailable(monkeypatch):
    monkeypatch.setattr(utils, "is_metal_available", lambda: False)
    monkeypatch.setattr(utils, "available_ort_execution_providers", lambda: ())
    monkeypatch.setattr(utils.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        utils.torch,
        "xpu",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )

    assert utils.get_current_device() == "xpu"


def test_get_device_dtype_uses_torch_for_xpu_and_cpu():
    assert utils.get_device_dtype("bfloat16", "xpu:0") is utils.torch.bfloat16
    assert utils.get_device_dtype("float32", "cpu") is utils.torch.float32


def test_get_device_dtype_uses_numpy_for_onnx_execution_providers():
    assert utils.get_device_dtype("float16", "directml") is utils.np.float16
    assert utils.get_device_dtype("bfloat16", "openvino") is utils.np.float32


def test_load_config_only_reads_local_config(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"num_hidden_layers": 2}')

    config = utils.load_config_only(str(model_path))
    assert config == {"num_hidden_layers": 2}


def test_load_config_only_downloads_config_json(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"num_hidden_layers": 78}')

    with patch("parallax.utils.utils.download_model_file", return_value=config_path) as download:
        config = utils.load_config_only("mlx-community/GLM-5.1", local_files_only=True)

    assert config == {"num_hidden_layers": 78}
    download.assert_called_once_with(
        repo_id="mlx-community/GLM-5.1",
        filename="config.json",
        local_files_only=True,
    )


def test_load_local_model_config_applies_generation_eos_token(tmp_path):
    (tmp_path / "config.json").write_text('{"eos_token_id": 1}')
    (tmp_path / "generation_config.json").write_text('{"eos_token_id": [2, 3]}')

    config = utils.load_local_model_config(tmp_path)

    assert config["eos_token_id"] == [2, 3]


def test_get_layer_types_marks_dense_mla_attention():
    config = {
        "model_type": "deepseek_v3",
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
    }

    assert utils.get_layer_types(config, 0, 2) == [MLA_ATTENTION, MLA_ATTENTION]


def test_get_layer_types_marks_dsa_attention():
    config = {
        "model_type": "glm_moe_dsa",
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "index_head_dim": 128,
        "index_n_heads": 4,
    }

    assert utils.get_layer_types(config, 0, 2) == [DSA_ATTENTION, DSA_ATTENTION]


def test_get_layer_types_preserves_hybrid_linear_layers_with_mla_cache():
    config = {
        "model_type": "kimi_k2",
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "linear_attn_config": {"full_attn_layers": [1, 3]},
    }

    assert utils.get_layer_types(config, 0, 4) == [
        LINEAR,
        MLA_ATTENTION,
        LINEAR,
        MLA_ATTENTION,
    ]


def test_get_layer_types_marks_msa_attention_cache():
    config = {"model_type": "minimax_m3"}

    assert utils.get_layer_types(config, 0, 2) == [
        MSA_ATTENTION,
        MSA_ATTENTION,
    ]
