from scheduling.model_info import ModelInfo, ModelWeightProfile


def test_model_info_uses_distinct_value_head_dim():
    model_info = ModelInfo(
        model_name="zai-org/GLM-5.1",
        mlx_model_name="mlx-community/GLM-5.1",
        head_size=64,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        hidden_dim=6144,
        intermediate_dim=12288,
        num_attention_heads=64,
        num_kv_heads=64,
        vocab_size=154880,
        num_layers=78,
        cache_bytes_per_element=2,
    )

    assert model_info.head_size_k == 256
    assert model_info.head_size_v == 256
    assert model_info.per_token_per_layer_kv_size == 2 * 64 * (256 + 256)


def test_model_info_uses_runtime_specific_exact_weight_profiles():
    model_info = ModelInfo(
        model_name="main",
        mlx_model_name="mlx",
        head_size=8,
        hidden_dim=8,
        intermediate_dim=16,
        num_attention_heads=1,
        num_kv_heads=1,
        vocab_size=32,
        num_layers=2,
        weight_profile=ModelWeightProfile((100, 200), 10, 20),
        mlx_weight_profile=ModelWeightProfile((50, 60), 5, 6),
    )

    assert model_info.stage_weight_bytes(0, 2) == 330
    assert model_info.stage_weight_bytes(0, 2, using_mlx=True) == 121
