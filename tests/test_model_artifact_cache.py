import json
import os
from collections import namedtuple
from pathlib import Path

import pytest

from parallax.utils.model_artifact_cache import (
    CacheObjectRequirement,
    ModelArtifactCache,
    ModelArtifactStorageError,
)

DiskUsage = namedtuple("DiskUsage", "total used free")
MIB = 1024 * 1024


class SyntheticVolume:
    """Exact test volume whose free bytes follow Fabi projection contents."""

    def __init__(self, root: Path, *, total: int = 100 * MIB, other_used: int = 0):
        self.root = root
        self.total = total
        self.other_used = other_used

    def __call__(self, _path: Path) -> DiskUsage:
        cache_bytes = sum(
            path.stat().st_size
            for identity in self.root.iterdir()
            if identity.is_dir() and len(identity.name) == 64
            for path in identity.rglob("*")
            if path.is_file()
        )
        used = min(self.other_used + cache_bytes, self.total)
        return DiskUsage(self.total, used, self.total - used)


def _identity(character: str) -> str:
    return character * 64


def _requirement(size: int = 2 * MIB) -> CacheObjectRequirement:
    return CacheObjectRequirement(
        relative_path="model-fabi-layer-00000.safetensors",
        size_bytes=size,
        is_weight_pack=True,
    )


def _projection(root: Path, identity: str, *, size: int = 2 * MIB) -> None:
    projection = root / identity
    projection.mkdir(parents=True)
    pack = projection / "model-fabi-layer-00000.safetensors"
    pack.write_bytes(b"x" * size)
    (projection / ".fabi-selective-artifacts.json").write_text(
        json.dumps(
            {
                "version": 1,
                "artifact_index_sha256": identity,
                "model_id": f"test/{identity[0]}",
                "immutable_revision": identity,
                "packs": {
                    pack.name: {
                        "size": size,
                        "sha256": "0" * 64,
                        "spec_sha256": "1" * 64,
                    }
                },
            }
        )
    )


def _reserve_existing(
    cache: ModelArtifactCache,
    identity: str,
) -> None:
    reservation, snapshot = cache.reserve(
        artifact_identity=identity,
        model_id=f"test/{identity[0]}",
        immutable_revision=identity,
        required_objects=(_requirement(),),
    )
    assert snapshot.required_growth_bytes == MIB
    cache.commit(reservation)


def _drop_process_leases(root: Path) -> None:
    for path in (root / ".leases-v1").glob("*.json"):
        path.unlink()


def test_exact_growth_reuses_existing_pack_and_tracks_successful_access(tmp_path):
    identity = _identity("a")
    _projection(tmp_path, identity)
    volume = SyntheticVolume(tmp_path)
    cache = ModelArtifactCache(
        tmp_path,
        minimum_free_bytes=0,
        cleanup_hysteresis_bytes=0,
        disk_usage=volume,
    )

    _reserve_existing(cache, identity)
    _drop_process_leases(tmp_path)
    _reserve_existing(cache, identity)

    with cache._database() as database:
        frequency = database.execute(
            "SELECT frequency FROM objects WHERE identity = ?", (identity,)
        ).fetchone()[0]
    assert frequency == 2


def test_active_process_lease_prevents_eviction(tmp_path):
    active = _identity("a")
    target = _identity("b")
    _projection(tmp_path, active)
    volume = SyntheticVolume(tmp_path, other_used=94 * MIB)
    cache = ModelArtifactCache(
        tmp_path,
        minimum_free_bytes=2 * MIB,
        cleanup_hysteresis_bytes=0,
        disk_usage=volume,
    )
    _reserve_existing(cache, active)

    with pytest.raises(ModelArtifactStorageError):
        cache.reserve(
            artifact_identity=target,
            model_id="test/b",
            immutable_revision=target,
            required_objects=(_requirement(),),
        )
    assert (tmp_path / active / _requirement().relative_path).is_file()


def test_pressure_evicts_less_frequently_used_pack_first(tmp_path):
    frequent = _identity("a")
    cold = _identity("b")
    target = _identity("c")
    _projection(tmp_path, frequent)
    _projection(tmp_path, cold)
    volume = SyntheticVolume(tmp_path)
    cache = ModelArtifactCache(
        tmp_path,
        minimum_free_bytes=0,
        cleanup_hysteresis_bytes=0,
        disk_usage=volume,
    )
    _reserve_existing(cache, frequent)
    _drop_process_leases(tmp_path)
    _reserve_existing(cache, frequent)
    _drop_process_leases(tmp_path)
    _reserve_existing(cache, cold)
    _drop_process_leases(tmp_path)

    # Leave enough room that deleting one complete cold projection reaches
    # the low watermark, including its receipt bytes.
    volume.other_used = 93 * MIB - 1024
    pressure_cache = ModelArtifactCache(
        tmp_path,
        minimum_free_bytes=2 * MIB,
        cleanup_hysteresis_bytes=0,
        disk_usage=volume,
    )
    reservation, snapshot = pressure_cache.reserve(
        artifact_identity=target,
        model_id="test/c",
        immutable_revision=target,
        required_objects=(_requirement(),),
    )
    try:
        assert snapshot.reclaimed_bytes >= 2 * MIB
        assert (tmp_path / frequent / _requirement().relative_path).is_file()
        assert not (tmp_path / cold).exists()
    finally:
        pressure_cache.abort(reservation)


def test_stale_download_reservation_is_reaped_by_process_identity(tmp_path):
    identity = _identity("a")
    volume = SyntheticVolume(tmp_path)
    cache = ModelArtifactCache(
        tmp_path,
        minimum_free_bytes=0,
        cleanup_hysteresis_bytes=0,
        disk_usage=volume,
    )
    reservation, _ = cache.reserve(
        artifact_identity=identity,
        model_id="test/a",
        immutable_revision=identity,
        required_objects=(_requirement(),),
    )
    lease_path = tmp_path / ".leases-v1" / f"{reservation.lease_id}.json"
    payload = json.loads(lease_path.read_text())
    payload["pid"] = max(os.getpid() + 1_000_000, 9_999_999)
    lease_path.write_text(json.dumps(payload))

    replacement, snapshot = cache.reserve(
        artifact_identity=_identity("b"),
        model_id="test/b",
        immutable_revision=_identity("b"),
        required_objects=(_requirement(),),
    )
    try:
        assert snapshot.reserved_bytes == 0
        assert not lease_path.exists()
    finally:
        cache.abort(replacement)


def test_selected_cached_pack_is_not_sacrificed_to_make_its_own_reservation(tmp_path):
    identity = _identity("a")
    _projection(tmp_path, identity)
    volume = SyntheticVolume(tmp_path, other_used=97 * MIB)
    cache = ModelArtifactCache(
        tmp_path,
        minimum_free_bytes=MIB,
        cleanup_hysteresis_bytes=0,
        disk_usage=volume,
    )

    with pytest.raises(ModelArtifactStorageError):
        cache.reserve(
            artifact_identity=identity,
            model_id="test/a",
            immutable_revision=identity,
            required_objects=(_requirement(),),
        )
    assert (tmp_path / identity / _requirement().relative_path).is_file()


def test_cache_quota_counts_persistent_content_not_atomic_workspace(tmp_path):
    first = _identity("a")
    second = _identity("b")
    target = _identity("c")
    _projection(tmp_path, first)
    _projection(tmp_path, second)
    volume = SyntheticVolume(tmp_path)
    cache = ModelArtifactCache(
        tmp_path,
        minimum_free_bytes=0,
        cleanup_hysteresis_bytes=0,
        maximum_cache_bytes=5 * MIB,
        disk_usage=volume,
    )

    reservation, snapshot = cache.reserve(
        artifact_identity=target,
        model_id="test/c",
        immutable_revision=target,
        required_objects=(_requirement(),),
    )
    try:
        assert snapshot.required_content_growth_bytes == 2 * MIB
        assert snapshot.required_growth_bytes == 3 * MIB
        assert sum((tmp_path / identity).exists() for identity in (first, second)) == 1
    finally:
        cache.abort(reservation)


def test_concurrent_downloads_cannot_overcommit_the_same_free_bytes(tmp_path):
    volume = SyntheticVolume(tmp_path, other_used=95 * MIB)
    cache = ModelArtifactCache(
        tmp_path,
        minimum_free_bytes=MIB,
        cleanup_hysteresis_bytes=0,
        disk_usage=volume,
    )
    first, _ = cache.reserve(
        artifact_identity=_identity("a"),
        model_id="test/a",
        immutable_revision=_identity("a"),
        required_objects=(_requirement(),),
    )
    try:
        with pytest.raises(ModelArtifactStorageError):
            cache.reserve(
                artifact_identity=_identity("b"),
                model_id="test/b",
                immutable_revision=_identity("b"),
                required_objects=(_requirement(),),
            )
    finally:
        cache.abort(first)
