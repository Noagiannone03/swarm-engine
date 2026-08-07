"""Build independently executable ONNX decoder stages from an ORT GenAI graph.

The Microsoft ONNX Runtime GenAI exporter emits a complete decoder graph.  It
also fuses each residual addition into the next layer normalization.  That is
efficient for a monolithic process, but it hides the raw hidden-state boundary
required by a pipeline.  This module performs a validated, semantics-preserving
normalization pass, extracts endpoint and decoder graphs, and writes only the
external tensor bytes each graph needs.

``onnx`` and ``onnxruntime`` remain publisher-only dependencies. Importing the
runtime protocol on a worker does not import either package.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_BUILDER_FORMAT_VERSION = 1

# Measured against the official ORT GenAI DML Qwen export.  DirectML owns all
# transformer compute; ORT deliberately retains two attention-mask shape nodes
# and the quantized embedding lookup on CPU.  These exact names are copied into
# the signed execution plan and re-checked from a live provider profile before
# a worker may publish READY.
_DML_ALLOWED_CPU_NODES = (
    "/model/attn_mask_reformat/attn_mask_subgraph/Gather",
    "/model/attn_mask_reformat/attn_mask_subgraph/Gather/Cast",
    "/model/embed_tokens/GatherBlockQuantized",
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_LAYER_GQA_RE = re.compile(r"^/model/layers\.(\d+)/attn/GroupQueryAttention$")
_COPY_CHUNK_BYTES = 8 * 1024 * 1024
_ALIGNMENT = 64


@dataclass(frozen=True)
class StageBuild:
    stage_id: str
    kind: str
    start_layer: int
    end_layer: int
    graph_path: str
    external_data_paths: tuple[str, ...]
    io_contract_hash: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]


def _onnx_modules():
    try:
        import onnx
        from onnx import helper
        from onnx.utils import Extractor
    except ImportError as exc:  # pragma: no cover - depends on publisher environment
        raise RuntimeError(
            "portable graph publishing requires the pinned 'onnx' builder dependency"
        ) from exc
    return onnx, helper, Extractor


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_COPY_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(domain: str, payload: object) -> str:
    encoded = json.dumps(
        {"domain": domain, "payload": payload},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def portable_build_inventory_hash(inventory: Mapping[str, object]) -> str:
    """Return the canonical identity of one portable builder inventory.

    The stored digest is excluded from its own preimage. Keeping this helper in
    the builder module gives registry tooling one implementation of the exact
    domain-separated identity emitted at export time.
    """

    payload = dict(inventory)
    payload.pop("inventory_hash", None)
    return _canonical_hash("fabi/portable-onnx-build-inventory/v1", payload)


def _external_metadata(tensor) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


def _set_external_metadata(tensor, *, location: str, offset: int, length: int) -> None:
    del tensor.external_data[:]
    for key, value in (
        ("location", location),
        ("offset", str(offset)),
        ("length", str(length)),
    ):
        entry = tensor.external_data.add()
        entry.key = key
        entry.value = value


def _safe_external_source(model_path: Path, location: str) -> Path:
    if not location or Path(location).is_absolute() or "\\" in location:
        raise ValueError(f"unsafe ONNX external-data location: {location!r}")
    root = model_path.parent.resolve()
    candidate = (root / location).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"external data escapes source model directory: {location!r}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"missing ONNX external data: {candidate}")
    return candidate


def _copy_external_tensor(source_model: Path, tensor, destination, *, offset: int) -> int:
    metadata = _external_metadata(tensor)
    try:
        source_offset = int(metadata.get("offset", "0"))
        length = int(metadata["length"])
        source_path = _safe_external_source(source_model, metadata["location"])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid external tensor descriptor for {tensor.name!r}") from exc
    if source_offset < 0 or length <= 0 or source_offset + length > source_path.stat().st_size:
        raise ValueError(f"external tensor range is invalid for {tensor.name!r}")
    destination.seek(offset)
    with source_path.open("rb") as source:
        source.seek(source_offset)
        remaining = length
        while remaining:
            chunk = source.read(min(remaining, _COPY_CHUNK_BYTES))
            if not chunk:
                raise OSError(f"short read while copying external tensor {tensor.name!r}")
            destination.write(chunk)
            remaining -= len(chunk)
    return length


def _aligned(offset: int) -> int:
    return (offset + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT


def _value_info_shape(value_info) -> list[int | str | None]:
    tensor_type = value_info.type.tensor_type
    shape: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            shape.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            shape.append(dimension.dim_param)
        else:
            shape.append(None)
    return shape


def _io_contract(model, *, kind: str, start_layer: int, end_layer: int) -> str:
    def describe(value_info) -> dict[str, object]:
        tensor_type = value_info.type.tensor_type
        return {
            "name": value_info.name,
            "element_type": int(tensor_type.elem_type),
            "shape": _value_info_shape(value_info),
        }

    return _canonical_hash(
        "fabi/portable-onnx-stage-io/v1",
        {
            "kind": kind,
            "start_layer": start_layer,
            "end_layer": end_layer,
            "inputs": [describe(value) for value in model.graph.input],
            "outputs": [describe(value) for value in model.graph.output],
        },
    )


def _copy_value_info(value_info, name: str):
    result = copy.deepcopy(value_info)
    result.name = name
    return result


def _model_values(graph) -> dict[str, Any]:
    return {
        value.name: value
        for value in (*graph.input, *graph.output, *graph.value_info)
        if value.name
    }


def _validate_graph_structure(model, model_path: Path) -> None:
    """Validate graph topology and every external byte range.

    ORT GenAI v0.15 emits ``SimplifiedLayerNormalization`` in the standard
    domain at opset 21. ONNX Runtime supports that graph, while ONNX 1.22's
    schema checker rejects the *official export itself*. We therefore validate
    all format-independent invariants here and use ORT session construction plus
    numerical parity as the executable compatibility proof.
    """

    onnx, _, _ = _onnx_modules()
    graph = model.graph
    if model.ir_version <= 0 or not model.opset_import:
        raise ValueError("ONNX model has no valid IR/opset declaration")
    node_names: set[str] = set()
    available = {value.name for value in graph.input}
    initializer_names: set[str] = set()
    for tensor in graph.initializer:
        if not tensor.name or tensor.name in initializer_names:
            raise ValueError(f"duplicate or empty initializer name: {tensor.name!r}")
        initializer_names.add(tensor.name)
        available.add(tensor.name)
        if tensor.data_location == onnx.TensorProto.EXTERNAL:
            metadata = _external_metadata(tensor)
            try:
                offset = int(metadata.get("offset", "0"))
                length = int(metadata["length"])
                external = _safe_external_source(model_path, metadata["location"])
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"invalid external initializer descriptor for {tensor.name!r}"
                ) from exc
            if offset < 0 or length <= 0 or offset + length > external.stat().st_size:
                raise ValueError(f"external initializer range is invalid for {tensor.name!r}")
    produced: set[str] = set()
    for node in graph.node:
        if not node.name or node.name in node_names:
            raise ValueError(f"duplicate or empty node name: {node.name!r}")
        node_names.add(node.name)
        unresolved = [name for name in node.input if name and name not in available]
        if unresolved:
            raise ValueError(f"node {node.name!r} has unresolved inputs: {unresolved}")
        for name in node.output:
            if not name:
                continue
            if name in produced or name in initializer_names:
                raise ValueError(f"tensor {name!r} has multiple producers")
            produced.add(name)
            available.add(name)
    missing_outputs = [value.name for value in graph.output if value.name not in available]
    if missing_outputs:
        raise ValueError(f"graph outputs have no producer: {missing_outputs}")


def _single_node(nodes_by_name: dict[str, list[Any]], name: str):
    matches = nodes_by_name.get(name, [])
    if len(matches) != 1:
        raise ValueError(f"expected exactly one exporter node {name!r}, found {len(matches)}")
    return matches[0]


def _export_target_and_rotary_initializers(model, gqa_nodes: Sequence[Any]):
    """Infer and validate the two official ORT GenAI graph families we publish."""

    _, helper, _ = _onnx_modules()
    graph_inputs = {value.name for value in model.graph.input}
    initializers = {tensor.name for tensor in model.graph.initializer}
    producers = {output: node for node in model.graph.node for output in node.output if output}
    modes: set[int] = set()
    rotary_names: set[str] = set()
    for node in gqa_nodes:
        attributes = {
            attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute
        }
        mode = int(attributes.get("do_rotary", 0))
        modes.add(mode)
        if mode == 1:
            rotary_names.update(name for name in node.input[7:9] if name)
            continue
        if mode != 0:
            raise ValueError(f"unsupported GQA rotary mode {mode} at {node.name!r}")
        for input_name in node.input[:2]:
            rotary = producers.get(input_name)
            if (
                rotary is None
                or rotary.domain != "com.microsoft"
                or rotary.op_type != "RotaryEmbedding"
                or len(rotary.input) < 4
            ):
                raise ValueError(
                    f"DML GQA input is not produced by RotaryEmbedding: {input_name!r}"
                )
            rotary_names.update(rotary.input[2:4])
    if len(modes) != 1:
        raise ValueError("decoder layers mix incompatible GQA rotary modes")
    if any(name not in initializers for name in rotary_names) or len(rotary_names) != 2:
        raise ValueError("decoder layers do not share exactly two rotary initializers")
    mode = next(iter(modes))
    if mode == 1 and "position_ids" not in graph_inputs:
        return "cpu", tuple(sorted(rotary_names))
    if mode == 0 and "position_ids" in graph_inputs:
        return "dml", tuple(sorted(rotary_names))
    raise ValueError("exporter graph has an inconsistent rotary/input contract")


def normalize_pipeline_boundaries(model):
    """Expose raw residual boundaries without changing decoder mathematics.

    Returns a deep-copied model, its layer count, boundary tensor names and the
    rotary initializers shared by every decoder stage. Unsupported exporter
    graph shapes fail closed.
    """

    onnx, helper, _ = _onnx_modules()
    transformed = copy.deepcopy(model)
    graph = transformed.graph
    nodes_by_name: dict[str, list[Any]] = {}
    layer_indices: list[int] = []
    gqa_nodes: list[Any] = []
    for node in graph.node:
        nodes_by_name.setdefault(node.name, []).append(node)
        match = _LAYER_GQA_RE.fullmatch(node.name)
        if match:
            if node.domain != "com.microsoft" or node.op_type != "GroupQueryAttention":
                raise ValueError(f"unexpected attention operator at {node.name!r}")
            layer_indices.append(int(match.group(1)))
            gqa_nodes.append(node)
    if not layer_indices or sorted(layer_indices) != list(range(len(layer_indices))):
        raise ValueError("GroupQueryAttention layers must be unique and contiguous from zero")
    num_layers = len(layer_indices)

    layer_zero_norm = _single_node(nodes_by_name, "/model/layers.0/input_layernorm/LayerNorm")
    if layer_zero_norm.op_type != "SimplifiedLayerNormalization" or len(layer_zero_norm.input) < 2:
        raise ValueError("layer zero does not expose the expected RMS normalization")
    embedding_output = layer_zero_norm.input[0]
    values = _model_values(graph)
    embedding_value = values.get(embedding_output)
    if embedding_value is None:
        raise ValueError("exporter omitted embedding hidden-state value information")

    _, shared_rotary = _export_target_and_rotary_initializers(transformed, gqa_nodes)

    boundary_names = tuple(f"fabi.hidden_states.{index}" for index in range(num_layers + 1))
    replacement_inputs: dict[str, str] = {embedding_output: boundary_names[0]}
    replacement_nodes: dict[str, tuple[Any, ...]] = {}

    for layer in range(1, num_layers):
        name = f"/model/layers.{layer}/input_layernorm/SkipLayerNorm"
        node = _single_node(nodes_by_name, name)
        if node.domain != "com.microsoft" or node.op_type != "SkipSimplifiedLayerNormalization":
            raise ValueError(f"unsupported fused input normalization at layer {layer}")
        if len(node.input) != 3 or len(node.output) < 4 or not node.output[0] or not node.output[3]:
            raise ValueError(f"invalid fused input normalization contract at layer {layer}")
        epsilon = next(
            (helper.get_attribute_value(attr) for attr in node.attribute if attr.name == "epsilon"),
            None,
        )
        if epsilon is None:
            raise ValueError(f"fused input normalization at layer {layer} has no epsilon")
        boundary = boundary_names[layer]
        add = helper.make_node(
            "Add",
            list(node.input[:2]),
            [boundary],
            name=f"/fabi/boundaries/{layer}/Add",
        )
        norm = helper.make_node(
            "SimplifiedLayerNormalization",
            [boundary, node.input[2]],
            [node.output[0]],
            name=f"/fabi/boundaries/{layer}/LayerNorm",
            axis=-1,
            epsilon=float(epsilon),
            stash_type=onnx.TensorProto.FLOAT,
        )
        replacement_nodes[name] = (add, norm)
        replacement_inputs[node.output[3]] = boundary

    final_name = f"/model/layers.{num_layers}/final_norm_layernorm/SkipLayerNorm"
    final_node = _single_node(nodes_by_name, final_name)
    if (
        final_node.domain != "com.microsoft"
        or final_node.op_type != "SkipSimplifiedLayerNormalization"
        or len(final_node.input) != 3
        or len(final_node.output) < 1
        or not final_node.output[0]
    ):
        raise ValueError("unsupported fused final normalization contract")
    epsilon = next(
        (
            helper.get_attribute_value(attr)
            for attr in final_node.attribute
            if attr.name == "epsilon"
        ),
        None,
    )
    if epsilon is None:
        raise ValueError("fused final normalization has no epsilon")
    final_boundary = boundary_names[num_layers]
    replacement_nodes[final_name] = (
        helper.make_node(
            "Add",
            list(final_node.input[:2]),
            [final_boundary],
            name=f"/fabi/boundaries/{num_layers}/Add",
        ),
        helper.make_node(
            "SimplifiedLayerNormalization",
            [final_boundary, final_node.input[2]],
            [final_node.output[0]],
            name=f"/fabi/boundaries/{num_layers}/LayerNorm",
            axis=-1,
            epsilon=float(epsilon),
            stash_type=onnx.TensorProto.FLOAT,
        ),
    )

    identity = helper.make_node(
        "Identity",
        [embedding_output],
        [boundary_names[0]],
        name="/fabi/boundaries/0/Identity",
    )
    rewritten: list[Any] = []
    identity_inserted = False
    for node in graph.node:
        replacements = replacement_nodes.get(node.name)
        if replacements is not None:
            rewritten.extend(replacements)
            continue
        rewritten_node = copy.deepcopy(node)
        if node.name == layer_zero_norm.name:
            if identity_inserted:
                raise ValueError("duplicate layer-zero boundary insertion")
            rewritten.append(identity)
            identity_inserted = True
        for index, input_name in enumerate(rewritten_node.input):
            if input_name in replacement_inputs:
                rewritten_node.input[index] = replacement_inputs[input_name]
        rewritten.append(rewritten_node)
    if not identity_inserted:
        raise ValueError("layer-zero boundary was not inserted")
    del graph.node[:]
    graph.node.extend(rewritten)

    # Every boundary has the same activation contract as the embedding output.
    existing_values = {value.name for value in graph.value_info}
    for boundary in boundary_names:
        if boundary not in existing_values:
            graph.value_info.append(_copy_value_info(embedding_value, boundary))

    return transformed, num_layers, boundary_names, tuple(shared_rotary)


def _write_external_data(
    *,
    source_model: Path,
    stage_model,
    stage_id: str,
    output_dir: Path,
    shared_names: set[str],
    shared_offsets: dict[str, tuple[int, int]],
) -> tuple[str, ...]:
    onnx, _, _ = _onnx_modules()
    external_paths: set[str] = set()
    local_tensors = [
        tensor
        for tensor in stage_model.graph.initializer
        if tensor.data_location == onnx.TensorProto.EXTERNAL and tensor.name not in shared_names
    ]
    local_name = f"{stage_id}.data"
    if local_tensors:
        local_path = output_dir / local_name
        offset = 0
        with local_path.open("wb") as destination:
            for tensor in sorted(local_tensors, key=lambda value: value.name):
                aligned = _aligned(offset)
                if aligned > offset:
                    destination.write(b"\0" * (aligned - offset))
                length = _copy_external_tensor(source_model, tensor, destination, offset=aligned)
                _set_external_metadata(tensor, location=local_name, offset=aligned, length=length)
                offset = aligned + length
        external_paths.add(local_name)

    for tensor in stage_model.graph.initializer:
        if tensor.name not in shared_names:
            continue
        try:
            offset, length = shared_offsets[tensor.name]
        except KeyError as exc:
            raise ValueError(f"missing shared tensor bytes for {tensor.name!r}") from exc
        _set_external_metadata(tensor, location="shared.data", offset=offset, length=length)
        external_paths.add("shared.data")
    return tuple(sorted(external_paths))


def _write_shared_data(
    *,
    source_model: Path,
    model,
    shared_names: Sequence[str],
    output_dir: Path,
) -> dict[str, tuple[int, int]]:
    initializers = {tensor.name: tensor for tensor in model.graph.initializer}
    output = output_dir / "shared.data"
    offsets: dict[str, tuple[int, int]] = {}
    offset = 0
    with output.open("wb") as destination:
        for name in sorted(shared_names):
            tensor = initializers[name]
            aligned = _aligned(offset)
            if aligned > offset:
                destination.write(b"\0" * (aligned - offset))
            length = _copy_external_tensor(source_model, tensor, destination, offset=aligned)
            offsets[name] = (aligned, length)
            offset = aligned + length
    return offsets


def _artifact(path: Path, root: Path, *, media_type: str, role: str) -> dict[str, object]:
    return {
        "path": path.relative_to(root).as_posix(),
        "size": path.stat().st_size,
        "sha256": _sha256_file(path),
        "media_type": media_type,
        "role": role,
    }


def _prepare_output_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"output directory must be empty: {path}")
    path.mkdir(parents=True, exist_ok=True)


def build_portable_onnx_stages(
    source_model: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    source_model_id: str,
    source_model_revision: str,
    exporter_revision: str,
    target_execution_provider: str,
    expected_num_layers: int | None = None,
) -> dict[str, object]:
    """Build portable endpoint/layer graphs and return their signed inventory."""

    if not source_model_id:
        raise ValueError("source model id must be non-empty")
    target_execution_provider = target_execution_provider.strip().lower()
    if target_execution_provider not in {"cpu", "dml"}:
        raise ValueError("portable builder currently supports official CPU and DML exports")
    for label, revision in (
        ("source model", source_model_revision),
        ("exporter", exporter_revision),
    ):
        if not _COMMIT_RE.fullmatch(revision):
            raise ValueError(f"{label} revision must be a full lowercase commit hash")

    onnx, _, Extractor = _onnx_modules()
    source_path = Path(source_model).resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    root = Path(output_root).resolve()
    _prepare_output_directory(root)
    execution_dir = root / "execution"
    execution_dir.mkdir()

    source = onnx.load(source_path, load_external_data=False)
    _validate_graph_structure(source, source_path)
    gqa_nodes = [
        node
        for node in source.graph.node
        if node.domain == "com.microsoft" and node.op_type == "GroupQueryAttention"
    ]
    actual_target, _ = _export_target_and_rotary_initializers(source, gqa_nodes)
    if actual_target != target_execution_provider:
        raise ValueError(
            f"graph targets {actual_target!r}, requested {target_execution_provider!r}"
        )
    normalized, num_layers, boundaries, shared_names = normalize_pipeline_boundaries(source)
    execution_geometry = _execution_geometry(source)
    if expected_num_layers is not None and num_layers != expected_num_layers:
        raise ValueError(f"export contains {num_layers} layers, expected {expected_num_layers}")
    shared_offsets = _write_shared_data(
        source_model=source_path,
        model=normalized,
        shared_names=shared_names,
        output_dir=execution_dir,
    )
    extractor = Extractor(normalized)
    graph_inputs = {value.name for value in normalized.graph.input}
    decoder_auxiliary_inputs = ["attention_mask"]
    if "position_ids" in graph_inputs:
        decoder_auxiliary_inputs.append("position_ids")
    specifications: list[tuple[str, str, int, int, list[str], list[str]]] = [
        ("input", "input", 0, 0, ["input_ids"], [boundaries[0]])
    ]
    for layer in range(num_layers):
        specifications.append(
            (
                f"decoder-{layer:03d}",
                "decoder",
                layer,
                layer + 1,
                [
                    boundaries[layer],
                    *decoder_auxiliary_inputs,
                    f"past_key_values.{layer}.key",
                    f"past_key_values.{layer}.value",
                ],
                [
                    boundaries[layer + 1],
                    f"present.{layer}.key",
                    f"present.{layer}.value",
                ],
            )
        )
    specifications.append(
        ("output", "output", num_layers, num_layers, [boundaries[-1]], ["logits"])
    )

    stages: list[StageBuild] = []
    for stage_id, kind, start_layer, end_layer, inputs, outputs in specifications:
        stage_model = extractor.extract_model(inputs, outputs)
        external_names = _write_external_data(
            source_model=source_path,
            stage_model=stage_model,
            stage_id=stage_id,
            output_dir=execution_dir,
            shared_names=set(shared_names),
            shared_offsets=shared_offsets,
        )
        graph_path = execution_dir / f"{stage_id}.onnx"
        onnx.save_model(stage_model, graph_path, save_as_external_data=False)
        _validate_graph_structure(onnx.load(graph_path, load_external_data=False), graph_path)
        stages.append(
            StageBuild(
                stage_id=stage_id,
                kind=kind,
                start_layer=start_layer,
                end_layer=end_layer,
                graph_path=graph_path.relative_to(root).as_posix(),
                external_data_paths=tuple(
                    (execution_dir / name).relative_to(root).as_posix() for name in external_names
                ),
                io_contract_hash=_io_contract(
                    stage_model,
                    kind=kind,
                    start_layer=start_layer,
                    end_layer=end_layer,
                ),
                inputs=tuple(inputs),
                outputs=tuple(outputs),
            )
        )

    files = sorted(path for path in execution_dir.iterdir() if path.is_file())
    artifacts = [
        _artifact(
            path,
            root,
            media_type="application/onnx" if path.suffix == ".onnx" else "application/octet-stream",
            role="execution_graph" if path.suffix == ".onnx" else "execution_data",
        )
        for path in files
    ]
    source_external_locations = sorted(
        {
            _external_metadata(tensor).get("location", "")
            for tensor in source.graph.initializer
            if tensor.data_location == onnx.TensorProto.EXTERNAL
        }
    )
    source_artifacts = [
        {
            "path": source_path.name,
            "size": source_path.stat().st_size,
            "sha256": _sha256_file(source_path),
        }
    ]
    for location in source_external_locations:
        source_data = _safe_external_source(source_path, location)
        source_artifacts.append(
            {
                "path": source_data.relative_to(source_path.parent).as_posix(),
                "size": source_data.stat().st_size,
                "sha256": _sha256_file(source_data),
            }
        )
    inventory: dict[str, object] = {
        "format_version": _BUILDER_FORMAT_VERSION,
        "builder": "fabi/swarm-engine/portable-onnx-stage-builder",
        "source_model_id": source_model_id,
        "source_model_revision": source_model_revision,
        "exporter": "microsoft/onnxruntime-genai",
        "exporter_revision": exporter_revision,
        "target_execution_provider": target_execution_provider,
        "execution_geometry": execution_geometry,
        "provider_assignment_policy": (
            {
                "allowed_cpu_fallback_nodes": list(_DML_ALLOWED_CPU_NODES),
                "allowed_cpu_only_stages": ["input"],
                "require_profiled_assignment": True,
            }
            if target_execution_provider == "dml"
            else {
                "allowed_cpu_fallback_nodes": [],
                "allowed_cpu_only_stages": [],
                "require_profiled_assignment": False,
            }
        ),
        "num_layers": num_layers,
        "shared_initializers": list(sorted(shared_names)),
        "source_artifacts": source_artifacts,
        "artifacts": artifacts,
        "stages": [
            {
                "stage_id": stage.stage_id,
                "kind": stage.kind,
                "start_layer": stage.start_layer,
                "end_layer": stage.end_layer,
                "graph_path": stage.graph_path,
                "external_data_paths": list(stage.external_data_paths),
                "io_contract_hash": stage.io_contract_hash,
                "inputs": list(stage.inputs),
                "outputs": list(stage.outputs),
            }
            for stage in stages
        ],
    }
    inventory["inventory_hash"] = portable_build_inventory_hash(inventory)
    inventory_path = root / "portable-build.json"
    inventory_path.write_text(
        json.dumps(inventory, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return inventory


def _numpy_dtype(type_name: str):
    import numpy as np

    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
    }
    try:
        return mapping[type_name]
    except KeyError as exc:
        raise ValueError(f"parity verifier does not support ONNX input type {type_name!r}") from exc


def _attention_symbolic_dimensions(model_path: Path) -> dict[str, int]:
    """Derive symbolic KV geometry from GQA tensor shapes and signed attrs."""

    onnx, helper, _ = _onnx_modules()
    model = onnx.load(model_path, load_external_data=False)
    values = _model_values(model.graph)
    head_sizes: set[int] = set()
    for node in model.graph.node:
        if node.op_type != "GroupQueryAttention" or node.domain != "com.microsoft":
            continue
        attributes = {
            attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute
        }
        num_heads = int(attributes.get("num_heads", 0))
        kv_heads = int(attributes.get("kv_num_heads", 0))
        if num_heads <= 0 or kv_heads <= 0 or len(node.input) < 3:
            raise ValueError(f"invalid GroupQueryAttention geometry at {node.name!r}")
        projected: list[int] = []
        for input_name, heads in (
            (node.input[0], num_heads),
            (node.input[1], kv_heads),
            (node.input[2], kv_heads),
        ):
            value = values.get(input_name)
            if value is None:
                raise ValueError(f"missing GQA value information for {input_name!r}")
            shape = _value_info_shape(value)
            if not shape or not isinstance(shape[-1], int) or shape[-1] % heads:
                raise ValueError(f"cannot derive GQA head size from {input_name!r}")
            projected.append(shape[-1] // heads)
        if len(set(projected)) != 1:
            raise ValueError(f"GQA key/value cache dimensions differ at {node.name!r}")
        head_sizes.add(projected[0])
    if len(head_sizes) != 1:
        raise ValueError("portable parity requires one cache head dimension across decoder layers")
    return {"kv_cache_dim": next(iter(head_sizes))}


def _execution_geometry(model) -> dict[str, object]:
    """Return the concrete host-boundary and KV tensor geometry."""

    _, helper, _ = _onnx_modules()
    values = _model_values(model.graph)
    geometries: set[tuple[int, int]] = set()
    for node in model.graph.node:
        if node.domain != "com.microsoft" or node.op_type != "GroupQueryAttention":
            continue
        attributes = {
            attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute
        }
        kv_heads = int(attributes.get("kv_num_heads", 0))
        key = values.get(node.input[1])
        if key is None or kv_heads <= 0:
            raise ValueError(f"cannot derive KV geometry at {node.name!r}")
        shape = _value_info_shape(key)
        if not shape or not isinstance(shape[-1], int) or shape[-1] % kv_heads:
            raise ValueError(f"cannot derive KV head dimension at {node.name!r}")
        geometries.add((kv_heads, shape[-1] // kv_heads))
    if len(geometries) != 1:
        raise ValueError("decoder does not expose one uniform KV geometry")
    kv_heads, kv_head_dim = next(iter(geometries))
    layer_zero = values.get("/model/layers.0/input_layernorm/output_0")
    if layer_zero is None:
        raise ValueError("decoder has no layer-zero activation value information")
    hidden_shape = _value_info_shape(layer_zero)
    if not hidden_shape or not isinstance(hidden_shape[-1], int):
        raise ValueError("decoder hidden size is not concrete")
    element_type = int(layer_zero.type.tensor_type.elem_type)
    dtype_by_type = {1: "float32", 10: "float16"}
    try:
        activation_dtype = dtype_by_type[element_type]
    except KeyError as exc:
        raise ValueError(f"unsupported portable activation element type {element_type}") from exc
    return {
        "activation_dtype": activation_dtype,
        "activation_hidden_size": hidden_shape[-1],
        "kv_num_heads": kv_heads,
        "kv_head_dim": kv_head_dim,
    }


def _empty_past(session, name: str, symbolic_dimensions: dict[str, int]):
    import numpy as np

    meta = next(value for value in session.get_inputs() if value.name == name)
    shape: list[int] = []
    for dimension in meta.shape:
        if isinstance(dimension, int):
            shape.append(dimension)
        elif dimension == "batch_size":
            shape.append(1)
        elif dimension == "past_sequence_length":
            shape.append(0)
        elif dimension in symbolic_dimensions:
            shape.append(symbolic_dimensions[dimension])
        else:
            raise ValueError(f"cannot synthesize dimension {dimension!r} for {name!r}")
    return np.empty(shape, dtype=_numpy_dtype(meta.type))


def _assert_close(label: str, expected, actual, *, atol: float, rtol: float) -> float:
    import numpy as np

    if expected.shape != actual.shape:
        raise AssertionError(f"{label} shape mismatch: {expected.shape} != {actual.shape}")
    difference = float(np.max(np.abs(expected.astype(np.float64) - actual.astype(np.float64))))
    if not np.allclose(expected, actual, atol=atol, rtol=rtol):
        raise AssertionError(f"{label} is not numerically equivalent; max_abs_error={difference}")
    return difference


def verify_stage_parity(
    source_model: str | os.PathLike[str],
    output_root: str | os.PathLike[str],
    *,
    token_ids: Sequence[int] = (1, 2, 3, 4),
    atol: float | None = None,
    rtol: float | None = None,
) -> dict[str, float]:
    """Compare monolithic and composed prefill/decode logits and KV caches."""

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - depends on publisher environment
        raise RuntimeError("parity verification requires numpy and onnxruntime") from exc
    root = Path(output_root).resolve()
    inventory = json.loads((root / "portable-build.json").read_text(encoding="utf-8"))
    target = inventory.get("target_execution_provider", "cpu")
    if atol is None:
        # Pinned ORT GenAI tests use atol=0.5/rtol=0.1 as their standard FP16
        # comparison profile. CPU/FP32 stages remain strict.
        atol = 0.5 if target == "dml" else 1e-4
    if rtol is None:
        rtol = 0.1 if target == "dml" else 1e-5
    stages = inventory["stages"]
    decoder_stages = [stage for stage in stages if stage["kind"] == "decoder"]
    options = ort.SessionOptions()
    options.log_severity_level = 3

    input_ids = np.asarray([list(token_ids)], dtype=np.int64)
    attention_mask = np.ones(input_ids.shape, dtype=np.int64)
    position_ids = np.arange(input_ids.shape[1], dtype=np.int64).reshape(1, -1)
    source_session = ort.InferenceSession(
        str(Path(source_model).resolve()), options, providers=["CPUExecutionProvider"]
    )
    symbolic_dimensions = _attention_symbolic_dimensions(Path(source_model).resolve())
    source_inputs: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }
    if any(meta.name == "position_ids" for meta in source_session.get_inputs()):
        source_inputs["position_ids"] = position_ids
    for meta in source_session.get_inputs():
        if meta.name.startswith("past_key_values."):
            source_inputs[meta.name] = _empty_past(source_session, meta.name, symbolic_dimensions)
    source_output_names = ["logits"] + [
        output.name for output in source_session.get_outputs() if output.name.startswith("present.")
    ]
    source_prefill_values = source_session.run(source_output_names, source_inputs)
    source_prefill = dict(zip(source_output_names, source_prefill_values, strict=True))

    def session_for(stage: dict[str, Any]):
        return ort.InferenceSession(
            str(root / stage["graph_path"]), options, providers=["CPUExecutionProvider"]
        )

    input_stage = next(stage for stage in stages if stage["kind"] == "input")
    output_stage = next(stage for stage in stages if stage["kind"] == "output")
    hidden = session_for(input_stage).run(None, {"input_ids": input_ids})[0]
    composed_prefill_cache: dict[str, Any] = {}
    for stage in decoder_stages:
        layer = int(stage["start_layer"])
        session = session_for(stage)
        stage_inputs = {
            stage["inputs"][0]: hidden,
            "attention_mask": attention_mask,
            f"past_key_values.{layer}.key": _empty_past(
                session, f"past_key_values.{layer}.key", symbolic_dimensions
            ),
            f"past_key_values.{layer}.value": _empty_past(
                session, f"past_key_values.{layer}.value", symbolic_dimensions
            ),
        }
        if "position_ids" in stage["inputs"]:
            stage_inputs["position_ids"] = position_ids
        outputs = session.run(
            stage["outputs"],
            stage_inputs,
        )
        hidden = outputs[0]
        composed_prefill_cache[f"present.{layer}.key"] = outputs[1]
        composed_prefill_cache[f"present.{layer}.value"] = outputs[2]
    composed_prefill_logits = session_for(output_stage).run(
        ["logits"], {output_stage["inputs"][0]: hidden}
    )[0]

    metrics: dict[str, float] = {
        "prefill_logits_max_abs_error": _assert_close(
            "prefill logits",
            source_prefill["logits"],
            composed_prefill_logits,
            atol=atol,
            rtol=rtol,
        )
    }
    if int(np.argmax(source_prefill["logits"][0, -1])) != int(
        np.argmax(composed_prefill_logits[0, -1])
    ):
        raise AssertionError("prefill top-1 token differs from monolithic graph")
    metrics["prefill_top1_match"] = 1.0
    cache_error = 0.0
    for name, actual in composed_prefill_cache.items():
        cache_error = max(
            cache_error,
            _assert_close(f"prefill {name}", source_prefill[name], actual, atol=atol, rtol=rtol),
        )
    metrics["prefill_kv_max_abs_error"] = cache_error

    next_token = np.asarray([[int(np.argmax(source_prefill["logits"][0, -1]))]], dtype=np.int64)
    decode_mask = np.ones((1, input_ids.shape[1] + 1), dtype=np.int64)
    source_decode_inputs: dict[str, Any] = {
        "input_ids": next_token,
        "attention_mask": decode_mask,
    }
    decode_position_ids = np.asarray([[input_ids.shape[1]]], dtype=np.int64)
    if any(meta.name == "position_ids" for meta in source_session.get_inputs()):
        source_decode_inputs["position_ids"] = decode_position_ids
    for layer in range(len(decoder_stages)):
        source_decode_inputs[f"past_key_values.{layer}.key"] = source_prefill[
            f"present.{layer}.key"
        ]
        source_decode_inputs[f"past_key_values.{layer}.value"] = source_prefill[
            f"present.{layer}.value"
        ]
    source_decode_values = source_session.run(source_output_names, source_decode_inputs)
    source_decode = dict(zip(source_output_names, source_decode_values, strict=True))

    hidden = session_for(input_stage).run(None, {"input_ids": next_token})[0]
    composed_decode_cache: dict[str, Any] = {}
    for stage in decoder_stages:
        layer = int(stage["start_layer"])
        session = session_for(stage)
        stage_inputs = {
            stage["inputs"][0]: hidden,
            "attention_mask": decode_mask,
            f"past_key_values.{layer}.key": composed_prefill_cache[f"present.{layer}.key"],
            f"past_key_values.{layer}.value": composed_prefill_cache[f"present.{layer}.value"],
        }
        if "position_ids" in stage["inputs"]:
            stage_inputs["position_ids"] = decode_position_ids
        outputs = session.run(
            stage["outputs"],
            stage_inputs,
        )
        hidden = outputs[0]
        composed_decode_cache[f"present.{layer}.key"] = outputs[1]
        composed_decode_cache[f"present.{layer}.value"] = outputs[2]
    composed_decode_logits = session_for(output_stage).run(
        ["logits"], {output_stage["inputs"][0]: hidden}
    )[0]
    metrics["decode_logits_max_abs_error"] = _assert_close(
        "decode logits", source_decode["logits"], composed_decode_logits, atol=atol, rtol=rtol
    )
    if int(np.argmax(source_decode["logits"][0, -1])) != int(
        np.argmax(composed_decode_logits[0, -1])
    ):
        raise AssertionError("decode top-1 token differs from monolithic graph")
    metrics["decode_top1_match"] = 1.0
    cache_error = 0.0
    for name, actual in composed_decode_cache.items():
        cache_error = max(
            cache_error,
            _assert_close(f"decode {name}", source_decode[name], actual, atol=atol, rtol=rtol),
        )
    metrics["decode_kv_max_abs_error"] = cache_error
    return metrics


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build independently executable, content-addressed ONNX layer stages"
    )
    parser.add_argument("--source-model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-model-id", required=True)
    parser.add_argument("--source-model-revision", required=True)
    parser.add_argument("--exporter-revision", required=True)
    parser.add_argument("--target-provider", required=True, choices=["cpu", "dml"])
    parser.add_argument("--expected-num-layers", type=int)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    inventory = build_portable_onnx_stages(
        args.source_model,
        args.output,
        source_model_id=args.source_model_id,
        source_model_revision=args.source_model_revision,
        exporter_revision=args.exporter_revision,
        target_execution_provider=args.target_provider,
        expected_num_layers=args.expected_num_layers,
    )
    result: dict[str, object] = {
        "inventory_hash": inventory["inventory_hash"],
        "num_layers": inventory["num_layers"],
    }
    if args.verify:
        result["parity"] = verify_stage_parity(args.source_model, args.output)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
