from pathlib import Path

import pytest

onnx = pytest.importorskip("onnx")
from onnx import TensorProto, helper

from swarm_protocol.onnx_stage_builder import (
    _attention_symbolic_dimensions,
    _validate_graph_structure,
    normalize_pipeline_boundaries,
)


def value(name, dtype=TensorProto.FLOAT, shape=("batch_size", "sequence_length", 8)):
    return helper.make_tensor_value_info(name, dtype, list(shape))


def gqa(layer: int, hidden: str):
    return helper.make_node(
        "GroupQueryAttention",
        [
            hidden,
            f"k.{layer}",
            f"v.{layer}",
            f"past_key_values.{layer}.key",
            f"past_key_values.{layer}.value",
            "seqlens_k",
            "total_seq_len",
        "cos_cache",
        "sin_cache",
        ],
        [f"attn.{layer}", f"present.{layer}.key", f"present.{layer}.value"],
        name=f"/model/layers.{layer}/attn/GroupQueryAttention",
        domain="com.microsoft",
        num_heads=1,
        kv_num_heads=1,
        do_rotary=1,
    )


def synthetic_export():
    inputs = [
        value("input_ids", TensorProto.INT64, ("batch_size", "sequence_length")),
        value("attention_mask", TensorProto.INT64, ("batch_size", "total_sequence_length")),
    ]
    for layer in range(2):
        inputs.extend(
            [
                value(
                    f"past_key_values.{layer}.key",
                    shape=("batch_size", 1, "past_sequence_length", "kv_cache_dim"),
                ),
                value(
                    f"past_key_values.{layer}.value",
                    shape=("batch_size", 1, "past_sequence_length", "kv_cache_dim"),
                ),
            ]
        )
    initializers = [
        helper.make_tensor("cos_cache", TensorProto.FLOAT, [16, 4], [0.0] * 64),
        helper.make_tensor("sin_cache", TensorProto.FLOAT, [16, 4], [0.0] * 64),
        helper.make_tensor("norm.0", TensorProto.FLOAT, [8], [1.0] * 8),
        helper.make_tensor("norm.1", TensorProto.FLOAT, [8], [1.0] * 8),
        helper.make_tensor("post.0", TensorProto.FLOAT, [8], [1.0] * 8),
        helper.make_tensor("post.1", TensorProto.FLOAT, [8], [1.0] * 8),
        helper.make_tensor("final", TensorProto.FLOAT, [8], [1.0] * 8),
    ]
    nodes = [
        helper.make_node("Cast", ["input_ids"], ["embed"], name="embed", to=TensorProto.FLOAT),
        helper.make_node("Identity", ["attention_mask"], ["seqlens_k"], name="mask-left"),
        helper.make_node("Identity", ["attention_mask"], ["total_seq_len"], name="mask-right"),
        helper.make_node(
            "SimplifiedLayerNormalization",
            ["embed", "norm.0"],
            ["normed.0"],
            name="/model/layers.0/input_layernorm/LayerNorm",
            epsilon=1e-6,
        ),
        helper.make_node("Identity", ["normed.0"], ["k.0"], name="k0"),
        helper.make_node("Identity", ["normed.0"], ["v.0"], name="v0"),
        gqa(0, "normed.0"),
        helper.make_node(
            "SkipSimplifiedLayerNormalization",
            ["embed", "attn.0", "post.0"],
            ["postnorm.0", "", "", "residual.0"],
            name="/model/layers.0/post_attention_layernorm/SkipLayerNorm",
            domain="com.microsoft",
            epsilon=1e-6,
        ),
        helper.make_node("Identity", ["postnorm.0"], ["mlp.0"], name="mlp0"),
        helper.make_node(
            "SkipSimplifiedLayerNormalization",
            ["residual.0", "mlp.0", "norm.1"],
            ["normed.1", "", "", "residual.1"],
            name="/model/layers.1/input_layernorm/SkipLayerNorm",
            domain="com.microsoft",
            epsilon=1e-6,
        ),
        helper.make_node("Identity", ["normed.1"], ["k.1"], name="k1"),
        helper.make_node("Identity", ["normed.1"], ["v.1"], name="v1"),
        gqa(1, "normed.1"),
        helper.make_node(
            "SkipSimplifiedLayerNormalization",
            ["residual.1", "attn.1", "post.1"],
            ["postnorm.1", "", "", "residual.2"],
            name="/model/layers.1/post_attention_layernorm/SkipLayerNorm",
            domain="com.microsoft",
            epsilon=1e-6,
        ),
        helper.make_node("Identity", ["postnorm.1"], ["mlp.1"], name="mlp1"),
        helper.make_node(
            "SkipSimplifiedLayerNormalization",
            ["residual.2", "mlp.1", "final"],
            ["hidden_states"],
            name="/model/layers.2/final_norm_layernorm/SkipLayerNorm",
            domain="com.microsoft",
            epsilon=1e-6,
        ),
        helper.make_node("Identity", ["hidden_states"], ["logits"], name="lm-head"),
    ]
    value_names = {
        "embed",
        "normed.0",
        "normed.1",
        "k.0",
        "k.1",
        "v.0",
        "v.1",
        "attn.0",
        "attn.1",
        "postnorm.0",
        "postnorm.1",
        "residual.0",
        "residual.1",
        "residual.2",
        "mlp.0",
        "mlp.1",
        "hidden_states",
    }
    outputs = [value("logits")]
    for layer in range(2):
        outputs.extend(
            [
                value(
                    f"present.{layer}.key",
                    shape=("batch_size", 1, "total_sequence_length", "kv_cache_dim"),
                ),
                value(
                    f"present.{layer}.value",
                    shape=("batch_size", 1, "total_sequence_length", "kv_cache_dim"),
                ),
            ]
        )
    graph = helper.make_graph(
        nodes,
        "synthetic-ort-export",
        inputs,
        outputs,
        initializer=initializers,
        value_info=[value(name) for name in sorted(value_names)],
    )
    return helper.make_model(
        graph,
        ir_version=10,
        opset_imports=[helper.make_opsetid("", 21), helper.make_opsetid("com.microsoft", 1)],
    )


def test_normalization_exposes_exact_residual_boundaries():
    transformed, layers, boundaries, shared = normalize_pipeline_boundaries(synthetic_export())

    assert layers == 2
    assert boundaries == (
        "fabi.hidden_states.0",
        "fabi.hidden_states.1",
        "fabi.hidden_states.2",
    )
    assert shared == ("cos_cache", "sin_cache")
    nodes = {node.name: node for node in transformed.graph.node}
    assert nodes["/fabi/boundaries/1/Add"].input == ["residual.0", "mlp.0"]
    assert nodes["/fabi/boundaries/1/LayerNorm"].input[0] == "fabi.hidden_states.1"
    assert nodes["/model/layers.1/post_attention_layernorm/SkipLayerNorm"].input[0] == (
        "fabi.hidden_states.1"
    )
    assert nodes["/fabi/boundaries/2/Add"].input == ["residual.2", "mlp.1"]


def test_attention_cache_dimension_is_derived_from_signed_graph(tmp_path):
    path = tmp_path / "model.onnx"
    onnx.save(synthetic_export(), path)

    assert _attention_symbolic_dimensions(path) == {"kv_cache_dim": 8}


def test_structure_validator_rejects_unresolved_input(tmp_path):
    model = synthetic_export()
    model.graph.node[0].input[0] = "missing"
    path = Path(tmp_path) / "broken.onnx"

    with pytest.raises(ValueError, match="unresolved inputs"):
        _validate_graph_structure(model, path)
