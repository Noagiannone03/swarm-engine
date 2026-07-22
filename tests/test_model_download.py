import json
from unittest.mock import patch

from parallax.utils import model_download


def test_snapshot_download_forwards_immutable_revision(tmp_path):
    snapshot_path = tmp_path / "snapshot"

    with patch(
        "parallax.utils.model_download._snapshot_download",
        return_value=str(snapshot_path),
    ) as snapshot_download:
        result = model_download.download_model_snapshot(
            "remote-org/remote-model",
            revision="immutable-sha",
        )

    assert result == snapshot_path
    snapshot_download.assert_called_once_with(
        repo_id="remote-org/remote-model",
        allow_patterns=None,
        ignore_patterns=None,
        local_dir=None,
        local_files_only=False,
        revision="immutable-sha",
        max_workers=1,
    )


def test_selective_download_fetches_needed_weights_serially(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps({"num_hidden_layers": 4}))
    (model_path / "tokenizer.json").write_text("{}")
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": "layer-0.safetensors",
                    "model.layers.1.self_attn.q_proj.weight": "layer-1-attn.safetensors",
                    "model.layers.1.mlp.up_proj.weight": "layer-1-mlp.safetensors",
                    "model.layers.2.self_attn.q_proj.weight": "layer-2.safetensors",
                }
            }
        )
    )

    with (
        patch(
            "parallax.utils.model_download.download_model_snapshot",
            return_value=model_path,
        ) as download_snapshot,
        patch("parallax.utils.model_download.download_model_file") as download_file,
    ):
        result = model_download.selective_model_download(
            "remote-org/remote-model",
            start_layer=1,
            end_layer=2,
            revision="immutable-sha",
        )

    assert result == model_path
    download_snapshot.assert_called_once()
    assert download_snapshot.call_args.kwargs["revision"] == "immutable-sha"
    assert download_snapshot.call_args.kwargs["local_files_only"] is False
    assert download_file.call_count == 2
    assert [call.kwargs for call in download_file.call_args_list] == [
        {
            "repo_id": "remote-org/remote-model",
            "filename": "layer-1-attn.safetensors",
            "local_files_only": False,
            "revision": "immutable-sha",
        },
        {
            "repo_id": "remote-org/remote-model",
            "filename": "layer-1-mlp.safetensors",
            "local_files_only": False,
            "revision": "immutable-sha",
        },
    ]


def test_selective_download_prefers_complete_immutable_metadata_cache(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps({"num_hidden_layers": 1}))
    (model_path / "tokenizer.json").write_text("{}")
    (model_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": "layer-0.safetensors",
                }
            }
        )
    )

    with (
        patch(
            "parallax.utils.model_download.download_model_snapshot",
            return_value=model_path,
        ) as download_snapshot,
        patch("parallax.utils.model_download.download_model_file") as download_file,
    ):
        result = model_download.selective_model_download(
            "remote-org/remote-model",
            start_layer=0,
            end_layer=1,
            revision="0123456789abcdef0123456789abcdef01234567",
        )

    assert result == model_path
    download_snapshot.assert_called_once_with(
        repo_id="remote-org/remote-model",
        ignore_patterns=model_download._EXCLUDE_WEIGHT_PATTERNS,
        local_files_only=True,
        revision="0123456789abcdef0123456789abcdef01234567",
    )
    download_file.assert_called_once_with(
        repo_id="remote-org/remote-model",
        filename="layer-0.safetensors",
        local_files_only=False,
        revision="0123456789abcdef0123456789abcdef01234567",
    )


def test_selective_download_refreshes_incomplete_immutable_metadata_cache(tmp_path):
    incomplete_path = tmp_path / "incomplete"
    incomplete_path.mkdir()
    (incomplete_path / "config.json").write_text("{}")

    complete_path = tmp_path / "complete"
    complete_path.mkdir()
    (complete_path / "config.json").write_text(json.dumps({"num_hidden_layers": 1}))
    (complete_path / "tokenizer.json").write_text("{}")
    (complete_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.layers.0.self_attn.q_proj.weight": "layer-0.safetensors",
                }
            }
        )
    )

    with (
        patch(
            "parallax.utils.model_download.download_model_snapshot",
            side_effect=[incomplete_path, complete_path],
        ) as download_snapshot,
        patch("parallax.utils.model_download.download_model_file"),
    ):
        result = model_download.selective_model_download(
            "remote-org/remote-model",
            start_layer=0,
            end_layer=1,
            revision="0123456789abcdef0123456789abcdef01234567",
        )

    assert result == complete_path
    assert download_snapshot.call_count == 2
    assert download_snapshot.call_args_list[0].kwargs["local_files_only"] is True
    assert download_snapshot.call_args_list[1].kwargs["local_files_only"] is False
