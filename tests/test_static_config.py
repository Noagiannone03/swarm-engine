from backend.server import static_config
from backend.server import model_weight_metadata
from backend.server.static_config import (
    MODELS,
    get_model_context_limit,
    get_model_info,
)
from parallax.utils.utils import clamp_model_sequence_length, normalize_model_config
from scheduling.model_info import ModelWeightProfile


def test_glm_5_1_uses_mlx_community_model():
    assert MODELS["zai-org/GLM-5.1"] == "mlx-community/GLM-5.1"


def test_exact_metadata_pins_config_and_weights_to_the_same_revision(monkeypatch):
    config_calls = []
    profile_calls = []

    monkeypatch.setitem(MODELS, "test/model", "test/model")
    monkeypatch.setattr(
        model_weight_metadata,
        "resolve_hub_revision",
        lambda repo_id: "immutable-sha",
    )

    def fake_load_config(model_name, local_files_only=False, revision=None):
        config_calls.append((model_name, revision))
        return {
            "head_dim": 1,
            "hidden_size": 1,
            "intermediate_size": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "vocab_size": 1,
            "num_hidden_layers": 2,
            "tie_word_embeddings": True,
        }

    def fake_profile(repo_id, **kwargs):
        profile_calls.append((repo_id, kwargs["revision"]))
        return ModelWeightProfile(
            (100, 100),
            10,
            10,
            shared_endpoint_bytes=10,
            source_revision=kwargs["revision"],
        )

    monkeypatch.setattr(static_config, "load_config_only", fake_load_config)
    monkeypatch.setattr(model_weight_metadata, "load_hub_weight_profile", fake_profile)

    info = get_model_info("test/model", load_weight_metadata=True)

    assert config_calls == [("test/model", "immutable-sha")]
    assert profile_calls == [("test/model", "immutable-sha")]
    assert info.model_revision == "immutable-sha"
    assert info.mlx_model_revision == "immutable-sha"


def test_glm_5_2_uses_mlx_community_mxfp4_model():
    assert MODELS["zai-org/GLM-5.2"] == "mlx-community/GLM-5.2-mxfp4"


def test_qwen3_6_mxfp4_is_scheduler_supported():
    assert MODELS["Qwen/Qwen3.6-27B"] == "mlx-community/Qwen3.6-27B-mxfp4"
    assert "mlx-community/Qwen3.6-27B-mxfp4" not in MODELS


def test_model_context_limit_reads_nested_text_config_and_ignores_sentinel():
    assert (
        get_model_context_limit(
            {
                "model_max_length": 10**30,
                "text_config": {"max_position_embeddings": 32768},
            }
        )
        == 32768
    )


def test_model_sequence_limit_is_a_safe_cap_not_a_permanent_worker_override():
    config = {"max_position_embeddings": 40960}

    assert clamp_model_sequence_length(65536, config) == 40960
    assert clamp_model_sequence_length(32768, config) == 32768
    assert clamp_model_sequence_length(None, config) == 40960


def test_model_info_uses_common_context_limit_across_runtime_variants(monkeypatch):
    def fake_load_config_only(model_name, local_files_only=False, revision=None):
        max_context = 40960 if model_name == "Qwen/Qwen3-1.7B" else 65536
        return {
            "head_dim": 128,
            "hidden_size": 2048,
            "intermediate_size": 6144,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "vocab_size": 151936,
            "num_hidden_layers": 28,
            "max_position_embeddings": max_context,
        }

    monkeypatch.setattr(static_config, "load_config_only", fake_load_config_only)
    monkeypatch.setitem(
        MODELS,
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-1.7B-MLX-4bit",
    )

    model_info = get_model_info("Qwen/Qwen3-1.7B")

    assert model_info.max_context_length == 40960


def test_minimax_m3_uses_mlx_community_4bit_model():
    assert MODELS["MiniMaxAI/MiniMax-M3"] == "mlx-community/MiniMax-M3-4bit"


def test_qwen3_6_mxfp4_model_info_uses_text_config(monkeypatch):
    def fake_load_config_only(model_name, local_files_only=False, revision=None):
        assert model_name in {
            "Qwen/Qwen3.6-27B",
            "mlx-community/Qwen3.6-27B-mxfp4",
        }
        return normalize_model_config(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "quantization_config": {"bits": 4, "mode": "mxfp4"},
                "text_config": {
                    "num_hidden_layers": 64,
                    "head_dim": 256,
                    "hidden_size": 5120,
                    "intermediate_size": 17408,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 4,
                    "vocab_size": 248320,
                },
            }
        )

    monkeypatch.setattr(static_config, "load_config_only", fake_load_config_only)

    model_info = get_model_info("Qwen/Qwen3.6-27B")

    assert model_info.num_layers == 64
    assert model_info.mlx_model_name == "mlx-community/Qwen3.6-27B-mxfp4"
    assert model_info.head_size == 256
    assert model_info.hidden_dim == 5120
    assert model_info.num_attention_heads == 24
    assert model_info.num_kv_heads == 4
    assert model_info.param_bytes_per_element == 0.5
    assert model_info.mlx_param_bytes_per_element == 0.5


def test_minimax_m3_model_info_uses_text_config(monkeypatch):
    def fake_load_config_only(model_name, local_files_only=False, revision=None):
        assert model_name in {
            "MiniMaxAI/MiniMax-M3",
            "mlx-community/MiniMax-M3-4bit",
        }
        return normalize_model_config(
            {
                "model_type": "minimax_m3_vl",
                "architectures": ["MiniMaxM3SparseForConditionalGeneration"],
                "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
                "text_config": {
                    "model_type": "minimax_m3",
                    "architectures": ["MiniMaxM3SparseForCausalLM"],
                    "num_hidden_layers": 60,
                    "head_dim": 128,
                    "hidden_size": 6144,
                    "intermediate_size": 3072,
                    "dense_intermediate_size": 12288,
                    "num_attention_heads": 64,
                    "num_key_value_heads": 4,
                    "num_local_experts": 128,
                    "num_experts_per_tok": 4,
                    "vocab_size": 200064,
                    "sparse_attention_config": {
                        "sparse_index_dim": 128,
                        "sparse_num_index_heads": 4,
                        "sparse_topk_blocks": 16,
                        "sparse_block_size": 128,
                    },
                },
            }
        )

    monkeypatch.setattr(static_config, "load_config_only", fake_load_config_only)

    model_info = get_model_info("MiniMaxAI/MiniMax-M3")

    assert model_info.num_layers == 60
    assert model_info.mlx_model_name == "mlx-community/MiniMax-M3-4bit"
    assert model_info.hidden_dim == 6144
    assert model_info.num_attention_heads == 64
    assert model_info.num_kv_heads == 4
    assert model_info.num_local_experts == 128
    assert model_info.num_experts_per_tok == 4
    assert model_info.moe_intermediate_dim == 3072
    assert model_info.param_bytes_per_element == 0.5
    assert model_info.mlx_param_bytes_per_element == 0.5


def test_qwen3_5_moe_4bit_model_info_uses_text_config(monkeypatch):
    def fake_load_config_only(model_name, local_files_only=False, revision=None):
        assert model_name in {
            "Qwen/Qwen3.5-35B-A3B",
            "mlx-community/Qwen3.5-35B-A3B-4bit",
        }
        return normalize_model_config(
            {
                "model_type": "qwen3_5_moe",
                "architectures": ["Qwen3_5MoeForConditionalGeneration"],
                "quantization": {"bits": 4, "group_size": 64, "mode": "affine"},
                "text_config": {
                    "num_hidden_layers": 40,
                    "full_attention_interval": 4,
                    "head_dim": 256,
                    "hidden_size": 2048,
                    "moe_intermediate_size": 512,
                    "num_attention_heads": 16,
                    "num_experts": 256,
                    "num_experts_per_tok": 8,
                    "num_key_value_heads": 2,
                    "vocab_size": 248320,
                },
            }
        )

    monkeypatch.setattr(static_config, "load_config_only", fake_load_config_only)

    model_info = get_model_info("Qwen/Qwen3.5-35B-A3B")

    assert model_info.num_layers == 40
    assert model_info.mlx_model_name == "mlx-community/Qwen3.5-35B-A3B-4bit"
    assert model_info.head_size == 256
    assert model_info.hidden_dim == 2048
    assert model_info.num_attention_heads == 16
    assert model_info.num_kv_heads == 2
    assert model_info.num_local_experts == 256
    assert model_info.num_experts_per_tok == 8
    assert model_info.moe_intermediate_dim == 512
    assert model_info.param_bytes_per_element == 0.5
    assert model_info.mlx_param_bytes_per_element == 0.5
