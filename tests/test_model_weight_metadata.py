from dataclasses import dataclass

import pytest

from backend.server.model_weight_metadata import build_weight_profile_from_tensors


@dataclass(frozen=True)
class Tensor:
    data_offsets: tuple[int, int]


def _tensor(size: int) -> Tensor:
    return Tensor((100, 100 + size))


def test_exact_profile_counts_quantization_tensors_and_tied_endpoints_once_per_stage():
    profile = build_weight_profile_from_tensors(
        {
            "model.embed_tokens.weight": _tensor(1_000),
            "model.layers.0.self_attn.q_proj.weight": _tensor(200),
            "model.layers.0.self_attn.q_proj.scales": _tensor(20),
            "model.layers.1.mlp.down_proj.weight": _tensor(300),
            "model.norm.weight": _tensor(10),
        },
        num_layers=2,
        tie_word_embeddings=True,
        source_revision="abc123",
    )

    assert profile.layer_bytes == (220, 300)
    assert profile.input_endpoint_bytes == 1_000
    assert profile.output_endpoint_bytes == 1_010
    assert profile.shared_endpoint_bytes == 1_000
    assert profile.stage_bytes(0, 1) == 1_220
    assert profile.stage_bytes(1, 2) == 1_310
    assert profile.stage_bytes(0, 2) == 1_530


def test_exact_profile_rejects_incomplete_layer_metadata():
    with pytest.raises(ValueError, match=r"decoder layers \[1\]"):
        build_weight_profile_from_tensors(
            {
                "model.embed_tokens.weight": _tensor(1_000),
                "model.layers.0.self_attn.q_proj.weight": _tensor(200),
                "lm_head.weight": _tensor(1_000),
            },
            num_layers=2,
            tie_word_embeddings=False,
        )
