import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from huggingface_hub import hf_hub_download as _hf_hub_download
from huggingface_hub import snapshot_download as _snapshot_download
from modelscope import snapshot_download as _ms_snapshot_download
from modelscope.hub.file_download import model_file_download as _ms_model_file_download

from parallax.utils.weight_filter_utils import (
    normalize_language_model_weight_key,
    should_include_weight_key,
)

logger = logging.getLogger(__name__)
_USE_MODELSCOPE_ENV = "USE_MODELSCOPE"
_MAX_CACHED_BUNDLE_BYTES = 32 * 1024 * 1024

if TYPE_CHECKING:
    from swarm_protocol.contracts import ModelArtifactIndex

__all__ = [
    "download_model_file",
    "download_model_snapshot",
    "selective_model_download",
]


def download_model_snapshot(
    repo_id: str,
    allow_patterns: Optional[list[str] | str] = None,
    ignore_patterns: Optional[list[str] | str] = None,
    local_dir: Optional[str | Path] = None,
    local_files_only: bool = False,
    revision: Optional[str] = None,
    max_workers: int = 1,
) -> Path:
    if _use_modelscope():
        return Path(
            _ms_snapshot_download(
                model_id=repo_id,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                local_dir=str(local_dir) if local_dir is not None else None,
                local_files_only=local_files_only,
            )
        )

    return Path(
        _snapshot_download(
            repo_id=repo_id,
            allow_patterns=allow_patterns,
            ignore_patterns=ignore_patterns,
            local_dir=local_dir,
            local_files_only=local_files_only,
            revision=revision,
            max_workers=max_workers,
        )
    )


def download_model_file(
    repo_id: str,
    filename: str,
    local_files_only: bool = False,
    revision: Optional[str] = None,
) -> Path:
    if _use_modelscope():
        return Path(
            _ms_model_file_download(
                model_id=repo_id,
                file_path=filename,
                local_files_only=local_files_only,
            )
        )

    return Path(
        _hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_files_only=local_files_only,
            revision=revision,
        )
    )


def selective_model_download(
    repo_id: str,
    start_layer: Optional[int] = None,
    end_layer: Optional[int] = None,
    local_files_only: bool = False,
    revision: Optional[str] = None,
    artifact_index: "ModelArtifactIndex | None" = None,
    token: bool | str | None = None,
) -> Path:
    local_path = Path(repo_id)
    if local_path.exists():
        logger.debug(f"Using local model path: {local_path}")
        return local_path

    logger.debug(f"Downloading model metadata for {repo_id}")
    model_path: Optional[Path] = None

    # A commit SHA names immutable Hub content.  Prefer a complete cached
    # metadata snapshot before contacting the network again: workers are often
    # restarted after an interrupted multi-gigabyte checkpoint download, and a
    # fresh remote validation can itself stall on constrained Windows/home
    # connections.  Mutable revisions still go online so they cannot silently
    # reuse stale metadata.
    if not local_files_only and _is_immutable_commit_revision(revision):
        try:
            cached_path = download_model_snapshot(
                repo_id=repo_id,
                ignore_patterns=_EXCLUDE_WEIGHT_PATTERNS,
                local_files_only=True,
                revision=revision,
            )
            if _cached_metadata_snapshot_is_usable(cached_path):
                model_path = cached_path
                logger.info("Using cached immutable model metadata at %s", model_path)
            else:
                logger.debug("Cached metadata snapshot is incomplete: %s", cached_path)
        except Exception as cache_error:
            # This is a best-effort cache probe.  The normal online path below
            # remains authoritative and will surface a useful error if it also
            # fails.
            logger.debug("Immutable metadata cache miss for %s: %s", repo_id, cache_error)

    if model_path is None:
        model_path = download_model_snapshot(
            repo_id=repo_id,
            ignore_patterns=_EXCLUDE_WEIGHT_PATTERNS,
            local_files_only=local_files_only,
            revision=revision,
        )
    logger.debug(f"Downloaded model metadata to {model_path}")

    trusted_index = artifact_index or _cached_v3_artifact_index(repo_id, revision)
    if (
        trusted_index is not None
        and trusted_index.tensors
        and start_layer is not None
        and end_layer is not None
        and not _use_modelscope()
    ):
        if revision is None:
            raise ValueError("signed selective downloads require an immutable model revision")
        from parallax.utils.selective_safetensors import materialize_tensor_span

        logger.info(
            "Materializing signed tensor ranges for layers [%d, %d)",
            start_layer,
            end_layer,
        )
        return materialize_tensor_span(
            repo_id=repo_id,
            immutable_revision=str(revision),
            metadata_root=model_path,
            artifact_index=trusted_index,
            start_layer=start_layer,
            end_layer=end_layer,
            local_files_only=local_files_only,
            token=token,
        )

    if start_layer is not None and end_layer is not None:
        logger.debug(f"Determining required weight files for layers [{start_layer}, {end_layer})")

        needed_weight_files = _determine_needed_weight_files_for_download(
            model_path=model_path,
            start_layer=start_layer,
            end_layer=end_layer,
        )

        if not needed_weight_files:
            logger.debug("Could not determine specific weight files, downloading all")
            download_model_snapshot(
                repo_id=repo_id,
                local_files_only=local_files_only,
                revision=revision,
            )
        else:
            missing_weight_files = [
                weight_file
                for weight_file in needed_weight_files
                if not (model_path / weight_file).exists()
            ]

            if missing_weight_files:
                logger.info(f"Downloading {len(missing_weight_files)} weight files")
                logger.debug(f"Downloading weight files: {missing_weight_files}")
                try:
                    # ``snapshot_download`` downloads files concurrently (eight
                    # workers by default).  That is useful for many small
                    # artifacts, but large checkpoint shards can leave several
                    # long-lived HTTP transfers competing for the same home
                    # connection.  Download the exact shard set serially with
                    # the Hub's supported single-file API instead.  We retain
                    # its cache locking, integrity checks and resumable
                    # ``.incomplete`` files without implementing transport code
                    # of our own.
                    for index, weight_file in enumerate(missing_weight_files, start=1):
                        logger.info(
                            "Downloading weight shard %d/%d: %s",
                            index,
                            len(missing_weight_files),
                            weight_file,
                        )
                        download_model_file(
                            repo_id=repo_id,
                            filename=weight_file,
                            local_files_only=local_files_only,
                            revision=revision,
                        )
                except Exception as e:
                    logger.error(
                        f"Failed to download weight files {missing_weight_files} "
                        f"for {repo_id}: {e}"
                    )
                    logger.error(
                        "This node cannot reach the model hub to download weight files. "
                        "Please check network connectivity or pre-download the model."
                    )
                    raise

            logger.debug(f"Downloaded weight files for layers [{start_layer}, {end_layer})")
    else:
        logger.debug("No layer range specified, downloading all model files")
        download_model_snapshot(
            repo_id=repo_id,
            local_files_only=local_files_only,
            revision=revision,
        )

    return model_path


_EXCLUDE_WEIGHT_PATTERNS = [
    "*.safetensors",
    "*.bin",
    "*.pt",
    "*.pth",
    "pytorch_model*.bin",
    "model*.safetensors",
    "weight*.safetensors",
]

_TOKENIZER_METADATA_CANDIDATES = (
    "tokenizer.json",
    "tokenizer.model",
    "spiece.model",
    "vocab.json",
)


def _is_immutable_commit_revision(revision: Optional[str]) -> bool:
    return bool(
        revision
        and len(revision) == 40
        and all(character in "0123456789abcdefABCDEF" for character in revision)
    )


def _cached_metadata_snapshot_is_usable(model_path: Path) -> bool:
    if not (model_path / "config.json").is_file():
        return False
    if not any((model_path / filename).is_file() for filename in _TOKENIZER_METADATA_CANDIDATES):
        return False

    # Sharded checkpoints need their index to map layers to files.  An
    # unsharded checkpoint is usable only when its single weight file is
    # already complete in the cache.
    return any(
        (model_path / filename).is_file()
        for filename in (
            "model.safetensors.index.json",
            "pytorch_model.bin.index.json",
            "model.safetensors",
            "pytorch_model.bin",
            "model.bin",
        )
    )


def _determine_needed_weight_files_for_download(
    model_path: Path,
    start_layer: int,
    end_layer: int,
    config: Optional[Dict] = None,
) -> List[str]:
    is_first_shard = start_layer == 0

    is_last_shard = False
    if config:
        num_hidden_layers = config.get("num_hidden_layers", 0)
        is_last_shard = end_layer >= num_hidden_layers
    else:
        config_file = model_path / "config.json"
        if config_file.exists():
            from parallax.utils.utils import normalize_model_config

            with open(config_file, "r") as f:
                cfg = normalize_model_config(json.load(f))
                num_hidden_layers = cfg.get("num_hidden_layers", 0)
                is_last_shard = end_layer >= num_hidden_layers

    index_file = model_path / "model.safetensors.index.json"

    if not index_file.exists():
        logger.debug(f"Index file not found at {index_file}, checking for single weight file")
        # For non-sharded models, look for single weight file
        single_weight_files = [
            "model.safetensors",
            "pytorch_model.bin",
            "model.bin",
        ]
        for weight_file in single_weight_files:
            if (model_path / weight_file).exists():
                logger.debug(f"Found single weight file: {weight_file}")
                return [weight_file]

        logger.debug("No weight files found (neither index nor single file)")
        return []

    with open(index_file, "r") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    if not weight_map:
        logger.debug("weight_map is empty in index file")
        return []

    tie_word_embeddings = False
    if config:
        tie_word_embeddings = config.get("tie_word_embeddings", False)

    needed_files: Set[str] = set()

    for key, filename in weight_map.items():
        if filename in needed_files:
            continue
        key = normalize_language_model_weight_key(key)
        if should_include_weight_key(
            key=key,
            start_layer=start_layer,
            end_layer=end_layer,
            is_first_shard=is_first_shard,
            is_last_shard=is_last_shard,
            tie_word_embeddings=tie_word_embeddings,
        ):
            needed_files.add(filename)

    result = sorted(list(needed_files))
    logger.debug(
        f"Determined {len(result)} weight files needed for layers [{start_layer}, {end_layer})"
    )
    return result


def _use_modelscope() -> bool:
    return _USE_MODELSCOPE_ENV in os.environ


def _cached_v3_artifact_index(
    repo_id: str,
    revision: Optional[str],
) -> "ModelArtifactIndex | None":
    """Read only a bundle already authenticated into the worker's TUF cache.

    The P2P process resolves the bundle before autonomous materialization.  The
    executor process shares the same registry state directory and can consume
    that verified target without racing a second python-tuf updater against it.
    """

    if not revision or os.environ.get("FABI_SWARM_V3_MODE", "off").strip().lower() in {
        "",
        "off",
        "disabled",
    }:
        return None
    state_dir = Path(
        os.environ.get(
            "FABI_SWARM_V3_STATE_DIR",
            str(Path.home() / ".fabi" / "swarm-v3" / "registry"),
        )
    )
    target_dir = state_dir / "targets"
    if not target_dir.is_dir():
        return None

    from swarm_protocol.registry import ModelRegistryBundle, ModelRegistryCatalog

    catalog_path = target_dir / "catalog.json"
    if not catalog_path.is_file():
        placement_mode = os.environ.get("FABI_SWARM_V3_PLACEMENT")
        if placement_mode is None:
            placement_mode = (
                "autonomous"
                if os.environ.get("FABI_SWARM_V3_MODE", "off").strip().lower() == "active"
                else "legacy"
            )
        if placement_mode.strip().lower() == "autonomous":
            raise RuntimeError("autonomous executor has no authenticated registry catalog cache")
        return None
    try:
        catalog = ModelRegistryCatalog.model_validate_json(catalog_path.read_bytes())
    except (OSError, ValueError) as exc:
        raise RuntimeError("authenticated registry catalog cache is invalid") from exc
    requested_swarm_id = os.environ.get("FABI_MODEL_SWARM_ID")
    entries = [
        entry
        for entry in catalog.models
        if entry.model_id == repo_id
        and entry.immutable_revision == revision
        and (requested_swarm_id is None or entry.model_swarm_id == requested_swarm_id)
    ]
    if not entries:
        return None
    if len(entries) != 1:
        raise RuntimeError(
            "current registry catalog has ambiguous execution variants; "
            "set FABI_MODEL_SWARM_ID to the trusted variant"
        )
    current_swarm_id = entries[0].model_swarm_id

    matches = []
    for candidate in sorted(path for path in target_dir.rglob("*") if path.is_file()):
        try:
            if candidate.stat().st_size > _MAX_CACHED_BUNDLE_BYTES:
                continue
            bundle = ModelRegistryBundle.model_validate_json(candidate.read_bytes())
        except (OSError, ValueError):
            continue
        if bundle.model_swarm_id == current_swarm_id:
            matches.append(bundle.artifact_index)
    if len(matches) > 1:
        identities = {index.model_dump_json() for index in matches}
        if len(identities) != 1:
            raise RuntimeError("trusted registry cache contains conflicting current model targets")
    if not matches:
        raise RuntimeError("current authenticated model target is missing from the registry cache")
    return matches[0]
