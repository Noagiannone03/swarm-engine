"""Non-blocking bridge from qualified Parallax workers to protocol-v3 advertisements."""

from __future__ import annotations

import logging
import os
import platform
import threading
import time
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Protocol

from swarm_protocol.artifact_verification import (
    VerifiedSpanArtifacts,
    verify_worker_span,
)
from swarm_protocol.contracts import (
    BackendKind,
    EffectiveSpanMode,
    KvGeometry,
    LayerSpan,
    LinkMetric,
    ModelArtifactIndex,
    ModelMemberAdvertisement,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
)
from swarm_protocol.registry import ModelRegistryBundle, TrustedModelRegistry

_REPORT_TTL_MS = 45_000
_VERIFICATION_RETRY_SECONDS = 30.0
logger = logging.getLogger(__name__)


class AdvertisementPublisher(Protocol):
    def publish_advertisement(
        self,
        advertisement: ModelMemberAdvertisement,
    ) -> None: ...


@dataclass(frozen=True)
class WorkerServingSnapshot:
    """Live worker values needed to produce one v3 offer and hosted-span lease."""

    worker_id: str
    endpoint_id: str
    model_id: str
    immutable_revision: str
    span: LayerSpan
    backend: BackendKind
    stable_memory_envelope_bytes: int
    kv_cache_token_capacity: int
    kv_cache_block_size: int
    max_sessions: int
    is_ready: bool
    current_requests: int = 0
    supports_frontend: bool = False
    outgoing_links: tuple[LinkMetric, ...] = ()
    measured_prefill_tokens_per_second: float | None = None
    measured_decode_tokens_per_second: float | None = None

    @property
    def verification_key(self) -> tuple[str, str, int, int]:
        return (
            self.model_id,
            self.immutable_revision,
            self.span.start,
            self.span.end,
        )


@dataclass(frozen=True)
class _VerifiedServingContract:
    key: tuple[str, str, int, int]
    bundle: ModelRegistryBundle
    artifacts: VerifiedSpanArtifacts


def _runtime_version() -> str:
    try:
        return version("parallax")
    except PackageNotFoundError:
        return "0.1.2+source"


def _local_model_root(
    model_id: str,
    immutable_revision: str,
    span: LayerSpan,
    artifact_index: ModelArtifactIndex | None = None,
) -> Path:
    # Keep the protocol contracts importable in registry/control-plane-only
    # environments that intentionally do not install model executors.
    from parallax.utils.model_download import selective_model_download

    return selective_model_download(
        repo_id=model_id,
        start_layer=span.start,
        end_layer=span.end,
        local_files_only=True,
        revision=immutable_revision,
        artifact_index=artifact_index,
    )


class WorkerProtocolV3Reporter:
    """Verify large checkpoints off the heartbeat path and expose fresh live reports."""

    def __init__(self, registry: TrustedModelRegistry, *, mode: str = "shadow") -> None:
        if mode not in {"shadow", "active"}:
            raise ValueError("worker protocol-v3 mode must be 'shadow' or 'active'")
        self.registry = registry
        self.mode = mode
        self._lock = threading.RLock()
        self._pending_key: tuple[str, str, int, int] | None = None
        self._verification_active = False
        self._verified: _VerifiedServingContract | None = None
        self._error: dict[str, str] | None = None
        self._error_key: tuple[str, str, int, int] | None = None
        self._retry_after = 0.0
        self._offer_seq = 0
        self._lease_seq = 0
        self._catalog: AdvertisementPublisher | None = None
        self._catalog_pending: ModelMemberAdvertisement | None = None
        self._catalog_active = False
        self._catalog_status: dict[str, object] = {"state": "off"}

    def attach_catalog(self, catalog: AdvertisementPublisher) -> None:
        """Attach the production DHT publisher after transport startup."""

        with self._lock:
            self._catalog = catalog
            self._catalog_status = {"state": "waiting_advertisement"}

    def trusted_manifest(self, model_swarm_id: str):
        """Return only metadata already verified through the pinned registry."""

        with self._lock:
            verified = self._verified
            if verified is None or verified.bundle.model_swarm_id != model_swarm_id:
                return None
            return verified.bundle.manifest

    def resolve_trusted_bundle(
        self,
        model_id: str,
        *,
        immutable_revision: str,
    ) -> ModelRegistryBundle:
        """Resolve a cold worker target through the pinned TUF registry."""

        return self.registry.resolve(
            model_id,
            immutable_revision=immutable_revision,
        )

    def bootstrap_offer(
        self,
        *,
        worker_id: str,
        endpoint_id: str,
        backend: BackendKind,
        stable_memory_envelope_bytes: int,
        supports_frontend: bool,
    ) -> WorkerOffer:
        """Build a fresh signed-endpoint offer before any executor is loaded."""

        if stable_memory_envelope_bytes <= 0:
            raise ValueError("worker has no positive stable memory envelope")
        now_ms = time.time_ns() // 1_000_000
        with self._lock:
            self._offer_seq += 1
            offer_seq = self._offer_seq
        roles = {WorkerRole.EXECUTOR}
        if supports_frontend:
            roles.add(WorkerRole.FRONTEND)
        return WorkerOffer(
            worker_id=worker_id,
            endpoint_id=endpoint_id,
            runtime_version=_runtime_version(),
            platform=f"{platform.system().lower()}-{platform.machine().lower()}",
            backend=backend,
            stable_memory_envelope_bytes=stable_memory_envelope_bytes,
            supported_roles=frozenset(roles),
            offer_seq=offer_seq,
            issued_at_ms=now_ms,
            expires_at_ms=now_ms + _REPORT_TTL_MS,
        )

    def publish_span_state(
        self,
        advertisement: ModelMemberAdvertisement,
        state: SpanState,
    ) -> ModelMemberAdvertisement:
        """Publish a sequenced non-READY transition for the current generation."""

        if state is SpanState.READY:
            raise ValueError("READY must come from a freshly verified serving snapshot")
        now_ms = time.time_ns() // 1_000_000
        with self._lock:
            self._offer_seq = max(self._offer_seq, advertisement.offer.offer_seq) + 1
            self._lease_seq = max(self._lease_seq, advertisement.lease.lease_seq) + 1
            transitioned = advertisement.model_copy(
                update={
                    "offer": advertisement.offer.model_copy(
                        update={
                            "offer_seq": self._offer_seq,
                            "issued_at_ms": now_ms,
                            "expires_at_ms": now_ms + _REPORT_TTL_MS,
                        }
                    ),
                    "lease": advertisement.lease.model_copy(
                        update={
                            "state": state,
                            "available_kv_bytes_snapshot": 0,
                            "lease_seq": self._lease_seq,
                            "issued_at_ms": now_ms,
                            "expires_at_ms": now_ms + _REPORT_TTL_MS,
                        }
                    ),
                }
            )
        self._queue_catalog_publish(transitioned)
        return transitioned

    def publish_bootstrap_state(
        self,
        advertisement: ModelMemberAdvertisement,
    ) -> None:
        """Publish a non-routable cold-join intent derived from signed metadata."""

        if advertisement.lease.state is not SpanState.BUILDING:
            raise ValueError("cold bootstrap publication must use BUILDING state")
        self._queue_catalog_publish(advertisement)

    @classmethod
    def from_environment(cls) -> "WorkerProtocolV3Reporter | None":
        """Build the opt-in shadow reporter without trust-on-first-use."""

        mode = os.environ.get("FABI_SWARM_V3_MODE", "off").strip().lower()
        if mode in {"", "off", "disabled"}:
            return None
        if mode not in {"shadow", "active"}:
            raise ValueError("FABI_SWARM_V3_MODE supports only 'off', 'shadow', or 'active'")

        required = {
            "metadata URL": os.environ.get("FABI_MODEL_REGISTRY_METADATA_URL"),
            "targets URL": os.environ.get("FABI_MODEL_REGISTRY_TARGETS_URL"),
            "pinned root path": os.environ.get("FABI_MODEL_REGISTRY_ROOT"),
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"protocol-v3 shadow mode is missing {', '.join(missing)}")
        root_path = Path(str(required["pinned root path"]))
        bootstrap_root = root_path.read_bytes()
        state_dir = Path(
            os.environ.get(
                "FABI_SWARM_V3_STATE_DIR",
                str(Path.home() / ".fabi" / "swarm-v3" / "registry"),
            )
        )
        return cls(
            TrustedModelRegistry(
                state_dir,
                metadata_base_url=str(required["metadata URL"]),
                target_base_url=str(required["targets URL"]),
                bootstrap_root=bootstrap_root,
            ),
            mode=mode,
        )

    def snapshot(self, serving: WorkerServingSnapshot) -> dict[str, object]:
        """Return current shadow state immediately; verification runs in a daemon thread."""

        key = serving.verification_key
        with self._lock:
            verified = self._verified if self._verified and self._verified.key == key else None
            retry_blocked = (
                self._error_key == key
                and self._error is not None
                and time.monotonic() < self._retry_after
            )
            if verified is None and not self._verification_active and not retry_blocked:
                self._pending_key = key
                self._verification_active = True
                self._error = None
                self._error_key = None
                thread = threading.Thread(
                    target=self._verify_contract,
                    args=(serving,),
                    name="SwarmV3ArtifactVerifier",
                    daemon=True,
                )
                thread.start()
            if verified is None:
                return {
                    "mode": self.mode,
                    "state": "rejected" if retry_blocked else "verifying",
                    "error": self._error,
                }
            try:
                advertisement = self._advertisement(serving, verified)
            except ValueError as exc:
                logger.warning(
                    "Swarm v3 advertisement rejected for %s: %s: %s",
                    key,
                    type(exc).__name__,
                    str(exc)[:256],
                )
                return {
                    "mode": self.mode,
                    "state": "rejected",
                    "error": {"code": type(exc).__name__, "detail": str(exc)[:256]},
                }
            self._queue_catalog_publish(advertisement)
            return {
                "mode": self.mode,
                "state": "ready" if serving.is_ready else "warming",
                "model_swarm_id": verified.bundle.model_swarm_id,
                "advertisement": advertisement.model_dump(mode="json"),
                "catalog": dict(self._catalog_status),
            }

    def _queue_catalog_publish(self, advertisement: ModelMemberAdvertisement) -> None:
        with self._lock:
            if self._catalog is None:
                return
            self._catalog_pending = advertisement
            if self._catalog_active:
                return
            self._catalog_active = True
            threading.Thread(
                target=self._catalog_publish_loop,
                name="SwarmV3CatalogPublisher",
                daemon=True,
            ).start()

    def _catalog_publish_loop(self) -> None:
        while True:
            with self._lock:
                advertisement = self._catalog_pending
                self._catalog_pending = None
                catalog = self._catalog
                if advertisement is None or catalog is None:
                    self._catalog_active = False
                    return
            try:
                catalog.publish_advertisement(advertisement)
            except Exception as exc:  # noqa: BLE001 - asynchronous network status boundary
                status: dict[str, object] = {
                    "state": "error",
                    "error": {
                        "code": type(exc).__name__,
                        "detail": str(exc)[:256],
                    },
                }
            else:
                status = {
                    "state": "published",
                    "offer_seq": advertisement.offer.offer_seq,
                    "lease_seq": advertisement.lease.lease_seq,
                }
            with self._lock:
                self._catalog_status = status

    def _verify_contract(self, serving: WorkerServingSnapshot) -> None:
        key = serving.verification_key
        try:
            bundle = self.registry.resolve(
                serving.model_id,
                immutable_revision=serving.immutable_revision,
            )
            model_root = _local_model_root(
                serving.model_id,
                serving.immutable_revision,
                serving.span,
                bundle.artifact_index,
            )
            artifacts = verify_worker_span(
                model_root,
                bundle.artifact_index,
                bundle.manifest,
                serving.span,
                include_tokenizer=serving.supports_frontend,
            )
            result = _VerifiedServingContract(key=key, bundle=bundle, artifacts=artifacts)
        except Exception as exc:  # noqa: BLE001 - converted to a fail-closed status boundary
            logger.warning(
                "Swarm v3 serving contract verification failed for %s: %s: %s",
                key,
                type(exc).__name__,
                str(exc)[:256],
            )
            with self._lock:
                if self._pending_key == key:
                    self._error = {
                        "code": type(exc).__name__,
                        "detail": str(exc)[:256],
                    }
                    self._error_key = key
                    self._retry_after = time.monotonic() + _VERIFICATION_RETRY_SECONDS
                    self._pending_key = None
                    self._verification_active = False
            return

        with self._lock:
            if self._pending_key == key:
                self._verified = result
                self._error = None
                self._error_key = None
                self._pending_key = None
                self._verification_active = False

    def _advertisement(
        self,
        serving: WorkerServingSnapshot,
        verified: _VerifiedServingContract,
    ) -> ModelMemberAdvertisement:
        if serving.stable_memory_envelope_bytes <= 0:
            raise ValueError("worker has no positive stable memory envelope")
        if serving.kv_cache_token_capacity <= 0 or serving.kv_cache_block_size <= 0:
            raise ValueError("worker has no measured positive KV cache geometry")
        if serving.max_sessions <= 0:
            raise ValueError("worker has no measured positive session capacity")

        now_ms = time.time_ns() // 1_000_000
        self._offer_seq += 1
        self._lease_seq += 1
        roles = {WorkerRole.EXECUTOR}
        if serving.supports_frontend:
            roles.add(WorkerRole.FRONTEND)
        offer = WorkerOffer(
            worker_id=serving.worker_id,
            endpoint_id=serving.endpoint_id,
            runtime_version=_runtime_version(),
            platform=f"{platform.system().lower()}-{platform.machine().lower()}",
            backend=serving.backend,
            stable_memory_envelope_bytes=serving.stable_memory_envelope_bytes,
            supported_roles=frozenset(roles),
            offer_seq=self._offer_seq,
            issued_at_ms=now_ms,
            expires_at_ms=now_ms + _REPORT_TTL_MS,
        )
        per_layer = verified.bundle.manifest.kv_bytes_per_token_by_layer
        bytes_per_token_for_span = sum(per_layer[serving.span.start : serving.span.end])
        allocatable_bytes = serving.kv_cache_token_capacity * bytes_per_token_for_span
        # The legacy executor does not yet publish exact live token occupancy. While requests are
        # active, advertise zero free bytes instead of manufacturing an estimate. Scheduler-side
        # reservations remain the old serving authority throughout shadow mode.
        available_bytes = allocatable_bytes if serving.current_requests == 0 else 0
        lease = SpanLease(
            model_swarm_id=verified.bundle.model_swarm_id,
            worker_id=serving.worker_id,
            hosted_span=serving.span,
            effective_span_mode=EffectiveSpanMode.FIXED,
            state=SpanState.READY if serving.is_ready else SpanState.WARMING,
            weight_hashes=verified.artifacts.weight_hashes,
            # Current metrics are valid for the qualified lab's one-session workers. Multi-session
            # routing needs load-conditioned distributions before these scalar fields can be used.
            measured_prefill_tokens_per_second=(
                serving.measured_prefill_tokens_per_second if serving.max_sessions == 1 else None
            ),
            measured_decode_tokens_per_second=(
                serving.measured_decode_tokens_per_second if serving.max_sessions == 1 else None
            ),
            kv_geometry=KvGeometry(
                block_size_tokens=serving.kv_cache_block_size,
                bytes_per_token_by_layer=per_layer,
                allocatable_bytes=allocatable_bytes,
            ),
            available_kv_bytes_snapshot=available_bytes,
            max_sessions=serving.max_sessions,
            lease_seq=self._lease_seq,
            issued_at_ms=now_ms,
            expires_at_ms=now_ms + _REPORT_TTL_MS,
        )
        return ModelMemberAdvertisement(
            offer=offer,
            lease=lease,
            outgoing_links=serving.outgoing_links,
        )
