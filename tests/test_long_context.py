import pytest

from parallax.utils.long_context import configure_long_context, long_context_overrides


def test_qwen3_64k_uses_documented_yarn_factor_two():
    original = {
        "model_type": "qwen3",
        "max_position_embeddings": 40960,
        "rope_scaling": None,
    }

    configured = configure_long_context(original, 65536)

    assert original["max_position_embeddings"] == 40960
    assert configured["max_position_embeddings"] == 65536
    assert configured["rope_scaling"] == {
        "rope_type": "yarn",
        "factor": 2.0,
        "original_max_position_embeddings": 32768,
    }
    assert long_context_overrides(original, configured) == {
        "max_position_embeddings": 65536,
        "rope_scaling": configured["rope_scaling"],
    }


def test_context_within_model_limit_needs_no_override():
    original = {"model_type": "qwen3", "max_position_embeddings": 40960}
    configured = configure_long_context(original, 32768)

    assert configured == original
    assert long_context_overrides(original, configured) == {}


def test_unknown_model_cannot_claim_an_unsupported_context():
    with pytest.raises(ValueError, match="no validated long-context policy"):
        configure_long_context(
            {"model_type": "unknown", "max_position_embeddings": 8192},
            65536,
        )
