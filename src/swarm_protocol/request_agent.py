"""Local V3 Request Agent planning and route coordination."""

from __future__ import annotations

import logging
import os
import stat
import threading
import time
import hashlib
from collections import deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import requests
from pydantic import BaseModel, ConfigDict, Field

from fabi_network.capability import RouteRecoveryPolicy
from swarm_protocol.contracts import RequestContract, RoutePlan
from swarm_protocol.control import SignedControlMessage
from swarm_protocol.coordinator import (
    CommittedRoute,
    ControlTransport,
    RouteReservationCoordinator,
)
from swarm_protocol.discovery import DiscoverySnapshot, DiscoveryStore
from swarm_protocol.epochs import EpochAllocator
from swarm_protocol.registry import ModelRegistryBundle, TrustedModelRegistry
from swarm_protocol.route_authority import RouteAdmissionEnvelope
from swarm_protocol.routing import ExactRoutePlanner, PlannedRoute, RoutePlanningError
from swarm_protocol.speculative import SpeculativeWindowFence

_MAX_AUTHORITY_RESPONSE_BYTES = 1024 * 1024
_DISCOVERY_SNAPSHOT_CACHE_MS = 60_000

logger = logging.getLogger(__name__)


def _permit_keepalive_key(*parts: object) -> str:
    """Build one bounded, non-secret idempotency key from request state."""

    digest = hashlib.sha256()
    digest.update(b"fabi-request-agent-permit-keepalive\0")
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return f"permit-keepalive:{digest.hexdigest()}"


def _permit_issue_key(
    *,
    request_id: str,
    coordinator_endpoint_id: str,
    model_swarm_id: str,
    max_context_tokens: int,
    recovery_policies: tuple[RouteRecoveryPolicy, ...],
    ttl_ms: int,
) -> str:
    """Fingerprint one exact permit contract for transport-safe retries.

    The logical request ID remains stable across an exact-token replan.  The
    idempotency key does not: changing any authority-visible parameter creates
    a new operation, while retrying the same HTTP operation reuses the same
    bounded key.
    """

    digest = hashlib.sha256()
    digest.update(b"fabi-request-agent-permit-issue-v1\0")
    parts: tuple[object, ...] = (
        request_id,
        coordinator_endpoint_id,
        model_swarm_id,
        max_context_tokens,
        *(policy.value for policy in sorted(recovery_policies, key=lambda item: item.value)),
        ttl_ms,
    )
    for part in parts:
        encoded = str(part).encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "big"))
        digest.update(encoded)
    return f"permit-issue:{digest.hexdigest()}"


def _system_clock_ms() -> int:
    return time.time_ns() // 1_000_000


def _steady_clock_ms() -> int:
    return time.monotonic_ns() // 1_000_000


def _account_credential_from_environment() -> str:
    credential = os.environ.get("FABI_ACCOUNT_TOKEN", "").strip()
    credential_file = os.environ.get("FABI_ACCOUNT_TOKEN_FILE", "").strip()
    if credential and credential_file:
        raise ValueError("set only one of FABI_ACCOUNT_TOKEN and FABI_ACCOUNT_TOKEN_FILE")
    if credential_file:
        path = Path(credential_file).expanduser()
        if os.name != "nt" and stat.S_IMODE(path.stat().st_mode) & 0o077:
            raise PermissionError(f"account credential file must be owner-only: {path}")
        credential = path.read_text(encoding="utf-8").strip()
    if not credential:
        raise ValueError("FABI_ACCOUNT_TOKEN or FABI_ACCOUNT_TOKEN_FILE is required")
    return credential


class RequestAgentAuthorityError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        code: str = "route_authority_error",
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.retry_after_seconds = retry_after_seconds


class RoutePermitGrant(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    permit_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_id: str = Field(min_length=1, max_length=512)
    coordinator_endpoint_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_swarm_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    max_context_tokens: int = Field(gt=0)
    recovery_policies: tuple[RouteRecoveryPolicy, ...]
    issued_at_ms: int = Field(ge=0)
    expires_at_ms: int = Field(gt=0)
    authorization_generation: int = Field(default=0, ge=0)


class IssuedAdmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    admission: RouteAdmissionEnvelope
    expires_at_ms: int = Field(gt=0)
    root_revocation_id: str = Field(pattern=r"^[0-9a-f]{128}$")


class AuthorityHttpSession(Protocol):
    def request(self, method: str, url: str, **kwargs): ...


class RequestAgentAuthorityClient:
    """Bounded HTTPS client that never exposes the account credential."""

    def __init__(
        self,
        base_url: str,
        account_credential: str,
        *,
        session: AuthorityHttpSession | None = None,
        connect_timeout_seconds: float = 3,
        read_timeout_seconds: float = 15,
    ) -> None:
        parsed = urlparse(base_url)
        loopback = parsed.hostname in {"127.0.0.1", "::1", "localhost"}
        if parsed.scheme != "https" and not (parsed.scheme == "http" and loopback):
            raise ValueError("route authority URL must use HTTPS except on loopback")
        if not parsed.netloc or parsed.query or parsed.fragment:
            raise ValueError("route authority URL must be an absolute base URL")
        if len(account_credential) != 64 or any(
            character not in "0123456789abcdefABCDEF" for character in account_credential
        ):
            raise ValueError("account credential must be a 32-byte hexadecimal value")
        if connect_timeout_seconds <= 0 or read_timeout_seconds <= 0:
            raise ValueError("authority HTTP timeouts must be positive")
        self._base_url = base_url.rstrip("/")
        self._credential = account_credential.lower()
        self._session = session or requests.Session()
        self._timeout = (connect_timeout_seconds, read_timeout_seconds)

    @staticmethod
    def _retry_after(response) -> int | None:
        raw = response.headers.get("Retry-After")
        if raw is None:
            return None
        try:
            return max(0, int(raw))
        except ValueError:
            return None

    def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        not_found_is_false: bool = False,
    ) -> dict[str, object] | bool:
        request_headers = {
            "Authorization": f"Bearer {self._credential}",
            "Accept": "application/json",
        }
        if headers:
            request_headers.update(headers)
        try:
            response = self._session.request(
                method,
                f"{self._base_url}{path}",
                json=json,
                headers=request_headers,
                timeout=self._timeout,
            )
        except requests.RequestException as error:
            raise RequestAgentAuthorityError(
                "route authority request failed",
                code="route_authority_unreachable",
            ) from error
        if len(response.content) > _MAX_AUTHORITY_RESPONSE_BYTES:
            raise RequestAgentAuthorityError(
                "route authority response exceeds 1 MiB",
                status_code=response.status_code,
                code="route_authority_response_too_large",
            )
        if not_found_is_false and response.status_code == 404:
            return False
        try:
            payload = response.json()
        except ValueError as error:
            raise RequestAgentAuthorityError(
                "route authority returned invalid JSON",
                status_code=response.status_code,
                code="invalid_route_authority_response",
            ) from error
        if not isinstance(payload, dict):
            raise RequestAgentAuthorityError(
                "route authority returned a non-object response",
                status_code=response.status_code,
                code="invalid_route_authority_response",
            )
        if response.status_code < 200 or response.status_code >= 300:
            error_payload = payload.get("error")
            code = "route_authority_error"
            message = f"route authority returned HTTP {response.status_code}"
            if isinstance(error_payload, dict):
                if isinstance(error_payload.get("code"), str):
                    code = str(error_payload["code"])
                if isinstance(error_payload.get("message"), str):
                    message = str(error_payload["message"])
            raise RequestAgentAuthorityError(
                message,
                status_code=response.status_code,
                code=code,
                retry_after_seconds=self._retry_after(response),
            )
        return payload

    def issue_permit(
        self,
        *,
        request_id: str,
        coordinator_endpoint_id: str,
        model_swarm_id: str,
        max_context_tokens: int,
        recovery_policies: tuple[RouteRecoveryPolicy, ...],
        ttl_ms: int = 120_000,
    ) -> RoutePermitGrant:
        idempotency_key = _permit_issue_key(
            request_id=request_id,
            coordinator_endpoint_id=coordinator_endpoint_id,
            model_swarm_id=model_swarm_id,
            max_context_tokens=max_context_tokens,
            recovery_policies=recovery_policies,
            ttl_ms=ttl_ms,
        )
        payload = self._request(
            "POST",
            "/v1/swarm/route-permits",
            headers={"Idempotency-Key": idempotency_key},
            json={
                "request_id": request_id,
                "coordinator_endpoint_id": coordinator_endpoint_id,
                "model_swarm_id": model_swarm_id,
                "max_context_tokens": max_context_tokens,
                "recovery_policies": [policy.value for policy in recovery_policies],
                "ttl_ms": ttl_ms,
            },
        )
        assert isinstance(payload, dict)
        try:
            return RoutePermitGrant.model_validate(payload)
        except ValueError as error:
            raise RequestAgentAuthorityError(
                "route authority returned an invalid permit",
                code="invalid_route_authority_response",
            ) from error

    def observe_unmet_context_demand(
        self,
        *,
        request_id: str,
        model_swarm_id: str,
        required_context_tokens: int,
    ) -> bool:
        payload = self._request(
            "POST",
            "/v1/swarm/context-demand",
            json={
                "request_id": request_id,
                "model_swarm_id": model_swarm_id,
                "required_context_tokens": required_context_tokens,
            },
        )
        assert isinstance(payload, dict)
        observed = payload.get("observed")
        if not isinstance(observed, bool):
            raise RequestAgentAuthorityError(
                "route authority returned an invalid context demand acknowledgement",
                code="invalid_route_authority_response",
            )
        return observed

    def issue_capability(
        self,
        *,
        permit_id: str,
        signed_plan: SignedControlMessage,
        recovery_policy: RouteRecoveryPolicy,
    ) -> IssuedAdmission:
        payload = self._request(
            "POST",
            "/v1/swarm/route-capabilities",
            json={
                "permit_id": permit_id,
                "signed_plan": signed_plan.model_dump(mode="json"),
                "recovery_policy": recovery_policy.value,
            },
        )
        assert isinstance(payload, dict)
        try:
            return IssuedAdmission.model_validate(payload)
        except ValueError as error:
            raise RequestAgentAuthorityError(
                "route authority returned an invalid capability",
                code="invalid_route_authority_response",
            ) from error

    def keepalive_permit(
        self,
        permit_id: str,
        *,
        ttl_ms: int,
        idempotency_key: str,
    ) -> RoutePermitGrant:
        payload = self._request(
            "POST",
            f"/v1/swarm/route-permits/{permit_id}/keepalive",
            headers={"Idempotency-Key": idempotency_key},
            json={"ttl_ms": ttl_ms},
        )
        assert isinstance(payload, dict)
        try:
            return RoutePermitGrant.model_validate(payload)
        except ValueError as error:
            raise RequestAgentAuthorityError(
                "route authority returned an invalid permit keepalive",
                code="invalid_route_authority_response",
            ) from error

    def release_permit(self, permit_id: str) -> bool:
        payload = self._request(
            "DELETE",
            f"/v1/swarm/route-permits/{permit_id}",
            not_found_is_false=True,
        )
        if payload is False:
            return False
        return isinstance(payload, dict) and payload.get("released") is True


class TrustedBundleSource(Protocol):
    def fetch(self, model_swarm_id: str) -> ModelRegistryBundle: ...


class ReservationCoordinator(Protocol):
    def reserve(self, plan: RoutePlan) -> CommittedRoute: ...

    def renew(self, route: CommittedRoute, *, ttl_ms: int) -> CommittedRoute: ...

    def release(self, route: CommittedRoute) -> None: ...


CoordinatorFactory = Callable[
    [
        ControlTransport,
        Callable[[SignedControlMessage], RouteAdmissionEnvelope],
        Callable[[SignedControlMessage], RouteAdmissionEnvelope],
    ],
    ReservationCoordinator,
]


def _default_coordinator_factory(
    transport: ControlTransport,
    authorizer: Callable[[SignedControlMessage], RouteAdmissionEnvelope],
    renewal_authorizer: Callable[[SignedControlMessage], RouteAdmissionEnvelope],
) -> RouteReservationCoordinator:
    return RouteReservationCoordinator(
        transport,
        admission_authorizer=authorizer,
        renewal_authorizer=renewal_authorizer,
    )


class _RouteRenewalAuthorizer:
    """Refresh account authority before any worker KV lease is extended."""

    def __init__(
        self,
        *,
        authority: RequestAgentAuthorityClient,
        permit: RoutePermitGrant,
        recovery_policy: RouteRecoveryPolicy,
        ttl_ms: int,
    ) -> None:
        self.authority = authority
        self.permit = permit
        self.recovery_policy = recovery_policy
        self.ttl_ms = ttl_ms

    def __call__(self, signed_plan: SignedControlMessage) -> RouteAdmissionEnvelope:
        current = self.permit
        renewed = self.authority.keepalive_permit(
            current.permit_id,
            ttl_ms=self.ttl_ms,
            idempotency_key=_permit_keepalive_key(
                current.request_id,
                "renew",
                current.authorization_generation,
            ),
        )
        issued = self.authority.issue_capability(
            permit_id=renewed.permit_id,
            signed_plan=signed_plan,
            recovery_policy=self.recovery_policy,
        )
        self.permit = renewed
        return issued.admission


@dataclass(frozen=True)
class RequestAgentReservation:
    permit: RoutePermitGrant
    planned: PlannedRoute
    committed: CommittedRoute
    coordinator: ReservationCoordinator
    recovery_policy: RouteRecoveryPolicy
    excluded_worker_ids: frozenset[str] = frozenset()


@dataclass
class _ManagedReservation:
    request: RequestContract
    reservation: RequestAgentReservation
    next_renew_at_ms: int
    lease_deadline_ms: int
    consecutive_renewal_failures: int = 0
    last_renewal_error: str | None = None


@dataclass
class _RecoverableRequest:
    request: RequestContract
    permit: RoutePermitGrant
    recovery_policy: RouteRecoveryPolicy
    failed_epoch: int
    failed_route_id: str
    excluded_worker_ids: frozenset[str]
    failure: str


@dataclass
class _RequestLockEntry:
    lock: threading.Lock = field(default_factory=threading.Lock)
    users: int = 0


@dataclass(frozen=True)
class _PlanningSnapshotCacheEntry:
    bundle: ModelRegistryBundle
    snapshot: DiscoverySnapshot
    refreshed_at_ms: int


class RequestAgentRouteRuntime:
    """Plan and maintain exact routes from the local Request Agent.

    Reservation keepalives run independently from token generation and SSE
    publication. A long prefill, a slow tool or a quiet stream therefore
    cannot starve worker leases.
    """

    def __init__(
        self,
        *,
        transport: ControlTransport,
        discovery: DiscoveryStore,
        registry: TrustedBundleSource,
        authority: RequestAgentAuthorityClient,
        epoch_allocator: EpochAllocator,
        planner: ExactRoutePlanner | None = None,
        coordinator_factory: CoordinatorFactory = _default_coordinator_factory,
        clock_ms: Callable[[], int] = _system_clock_ms,
        steady_clock_ms: Callable[[], int] = _steady_clock_ms,
        prepare_ttl_ms: int = 5_000,
        plan_ttl_ms: int = 10_000,
        permit_ttl_ms: int = 120_000,
        session_ttl_ms: int = 60_000,
        renew_interval_ms: int = 20_000,
        renew_retry_interval_ms: int = 2_000,
        renew_attempt_budget_ms: int = 5_000,
        lease_expiry_guard_ms: int = 1_000,
        start_maintenance_thread: bool = True,
        close_transport: bool = False,
        state_dir: Path | None = None,
        speculative_fence: SpeculativeWindowFence | None = None,
    ) -> None:
        if prepare_ttl_ms <= 0 or plan_ttl_ms <= prepare_ttl_ms:
            raise ValueError("plan TTL must exceed the positive prepare TTL")
        if permit_ttl_ms < plan_ttl_ms or permit_ttl_ms > 300_000:
            raise ValueError("permit TTL must cover the plan and remain below five minutes")
        if renew_interval_ms <= 0 or session_ttl_ms <= renew_interval_ms * 2:
            raise ValueError("session TTL must exceed two renewal intervals")
        if renew_retry_interval_ms <= 0 or renew_retry_interval_ms >= renew_interval_ms:
            raise ValueError("renew retry interval must be positive and below renew interval")
        if renew_attempt_budget_ms <= 0:
            raise ValueError("renew attempt budget must be positive")
        if lease_expiry_guard_ms <= 0:
            raise ValueError("lease expiry guard must be positive")
        if session_ttl_ms <= renew_attempt_budget_ms + lease_expiry_guard_ms:
            raise ValueError("session TTL leaves no safe renewal retry window")
        self.transport = transport
        self.discovery = discovery
        self.registry = registry
        self.authority = authority
        self.epoch_allocator = epoch_allocator
        self.planner = planner or ExactRoutePlanner()
        self._clock_ms = clock_ms
        self._steady_clock_ms = steady_clock_ms
        self.prepare_ttl_ms = prepare_ttl_ms
        self.plan_ttl_ms = plan_ttl_ms
        self.permit_ttl_ms = permit_ttl_ms
        self.session_ttl_ms = session_ttl_ms
        self.renew_interval_ms = renew_interval_ms
        self.renew_retry_interval_ms = renew_retry_interval_ms
        self.renew_attempt_budget_ms = renew_attempt_budget_ms
        self.lease_expiry_guard_ms = lease_expiry_guard_ms
        if coordinator_factory is _default_coordinator_factory:
            self._coordinator_factory = lambda transport, authorizer, renewal_authorizer: (
                RouteReservationCoordinator(
                    transport,
                    admission_authorizer=authorizer,
                    renewal_authorizer=renewal_authorizer,
                    session_ttl_ms=self.session_ttl_ms,
                )
            )
        else:
            self._coordinator_factory = coordinator_factory
        self._close_transport = close_transport
        self.state_dir = None if state_dir is None else Path(state_dir)
        self.speculative_fence = speculative_fence or SpeculativeWindowFence()
        self._request_locks: dict[str, _RequestLockEntry] = {}
        self._reservations: dict[str, _ManagedReservation] = {}
        self._recoverable_requests: dict[str, _RecoverableRequest] = {}
        self._failures: deque[dict[str, object]] = deque(maxlen=64)
        self._planning_snapshots: dict[str, _PlanningSnapshotCacheEntry] = {}
        self._discovery_refresh_lock = threading.Lock()
        self._lock = threading.RLock()
        self._phase_observer: Callable[[str, str], None] | None = None
        self._stop_event = threading.Event()
        self._maintenance_thread: threading.Thread | None = None
        if start_maintenance_thread:
            self._maintenance_thread = threading.Thread(
                target=self._maintenance_loop,
                name="FabiRequestAgentRouteLeases",
                daemon=True,
            )
            self._maintenance_thread.start()

    def set_phase_observer(self, observer: Callable[[str, str], None] | None) -> None:
        """Attach the local UI observer without making it part of route correctness."""

        with self._lock:
            self._phase_observer = observer

    def _emit_phase(self, request_id: str, phase: str) -> None:
        with self._lock:
            observer = self._phase_observer
        if observer is None:
            return
        try:
            observer(str(request_id), phase)
        except Exception:
            logger.warning("Request Agent phase observer failed", exc_info=True)

    @classmethod
    def from_environment(cls) -> "RequestAgentRouteRuntime":
        """Build the packaged Request Agent from the same Iroh/TUF configuration as workers."""

        from fabi_network.transport import IrohTransport
        from swarm_protocol.epochs import SqliteEpochAllocator

        authority_url = os.environ.get("FABI_REQUEST_AGENT_AUTHORITY_URL", "").strip()
        metadata_url = os.environ.get("FABI_MODEL_REGISTRY_METADATA_URL", "").strip()
        targets_url = os.environ.get("FABI_MODEL_REGISTRY_TARGETS_URL", "").strip()
        root_path = os.environ.get("FABI_MODEL_REGISTRY_ROOT", "").strip()
        if not authority_url:
            raise ValueError("FABI_REQUEST_AGENT_AUTHORITY_URL is required")
        if not metadata_url or not targets_url or not root_path:
            raise ValueError("Request Agent requires registry URLs and a pinned TUF root")
        state_dir = Path(
            os.environ.get(
                "FABI_REQUEST_AGENT_STATE_DIR",
                str(Path.home() / ".fabi" / "request-agent"),
            )
        ).expanduser()
        state_dir.mkdir(parents=True, exist_ok=True)
        transport = IrohTransport.from_environment("request-agent")
        try:
            if transport.catalog_discovery is None:
                raise RuntimeError(
                    "Request Agent requires FABI_CATALOG_DHT_MODE=client and bootstraps"
                )
            registry = TrustedModelRegistry(
                state_dir / "registry",
                metadata_base_url=metadata_url,
                target_base_url=targets_url,
                bootstrap_root=Path(root_path).expanduser().read_bytes(),
            )
            return cls(
                transport=transport,
                discovery=transport.catalog_discovery,
                registry=registry,
                authority=RequestAgentAuthorityClient(
                    authority_url,
                    _account_credential_from_environment(),
                ),
                epoch_allocator=SqliteEpochAllocator(
                    state_dir / "epochs.sqlite3",
                    namespace=f"request-agent:{transport.peer_id()}",
                ),
                close_transport=True,
                state_dir=state_dir,
            )
        except BaseException:
            transport.close()
            raise

    def _now_ms(self) -> int:
        now_ms = int(self._clock_ms())
        if now_ms < 0:
            raise RuntimeError("Request Agent clock returned a negative timestamp")
        return now_ms

    def _steady_now_ms(self) -> int:
        now_ms = int(self._steady_clock_ms())
        if now_ms < 0:
            raise RuntimeError("Request Agent steady clock returned a negative timestamp")
        return now_ms

    @contextmanager
    def _request_operation(
        self,
        request_id: str,
        *,
        create: bool,
    ) -> Iterator[threading.Lock | None]:
        """Serialize one request without deleting a lock underneath waiters."""

        with self._lock:
            entry = self._request_locks.get(request_id)
            if entry is None and create:
                entry = _RequestLockEntry()
                self._request_locks[request_id] = entry
            if entry is not None:
                entry.users += 1
        if entry is None:
            yield None
            return
        entry.lock.acquire()
        try:
            yield entry.lock
        finally:
            entry.lock.release()
            with self._lock:
                entry.users -= 1
                if (
                    entry.users == 0
                    and request_id not in self._reservations
                    and request_id not in self._recoverable_requests
                    and self._request_locks.get(request_id) is entry
                ):
                    self._request_locks.pop(request_id, None)

    def reserve(
        self,
        request: RequestContract,
        *,
        recovery_policy: RouteRecoveryPolicy = RouteRecoveryPolicy.REPLAN_COLD,
    ) -> RequestAgentReservation:
        if recovery_policy not in {
            RouteRecoveryPolicy.BEST_EFFORT,
            RouteRecoveryPolicy.REPLAN_COLD,
        }:
            raise ValueError("Request Agent recovery policy is not implemented")
        with self._request_operation(request.request_id, create=True):
            with self._lock:
                existing = self._reservations.get(request.request_id)
            if existing is not None:
                if (
                    existing.request != request
                    or existing.reservation.recovery_policy != recovery_policy
                ):
                    raise ValueError(
                        "request id is already reserved with a different route contract"
                    )
                if self._steady_now_ms() >= existing.lease_deadline_ms:
                    raise RuntimeError("request route lease is no longer active")
                return existing.reservation
            self._emit_phase(request.request_id, "planning")
            bundle, snapshot = self._trusted_planning_snapshot(request.model_swarm_id)
            manifest = bundle.manifest
            epoch = self.epoch_allocator.next_epoch()
            self._emit_phase(request.request_id, "authorizing")
            permit = self.authority.issue_permit(
                request_id=request.request_id,
                coordinator_endpoint_id=self.transport.peer_id(),
                model_swarm_id=request.model_swarm_id,
                max_context_tokens=request.required_context_tokens,
                recovery_policies=(recovery_policy,),
                ttl_ms=self.permit_ttl_ms,
            )
            now_ms = self._now_ms()
            try:
                planned = self.planner.plan(
                    manifest=manifest,
                    request=request,
                    offers=snapshot.offers,
                    leases=snapshot.leases,
                    links=snapshot.links,
                    snapshot_time_ms=snapshot.captured_at_ms,
                    coordinator_id=self.transport.peer_id(),
                    reservation_deadline_ms=now_ms + self.prepare_ttl_ms,
                    plan_expires_at_ms=now_ms + self.plan_ttl_ms,
                    epoch=epoch,
                )

                def authorize(signed_plan: SignedControlMessage) -> RouteAdmissionEnvelope:
                    return self.authority.issue_capability(
                        permit_id=permit.permit_id,
                        signed_plan=signed_plan,
                        recovery_policy=recovery_policy,
                    ).admission

                renewal_authorizer = _RouteRenewalAuthorizer(
                    authority=self.authority,
                    permit=permit,
                    recovery_policy=recovery_policy,
                    ttl_ms=self.permit_ttl_ms,
                )
                coordinator = self._coordinator_factory(
                    self.transport,
                    authorize,
                    renewal_authorizer,
                )
                self._emit_phase(request.request_id, "reserving")
                lease_started_at_ms = self._steady_now_ms()
                committed = coordinator.reserve(planned.plan)
                acknowledged_at_ms = self._steady_now_ms()
                lease_deadline_ms = lease_started_at_ms + self.session_ttl_ms
                if acknowledged_at_ms >= lease_deadline_ms - self.lease_expiry_guard_ms:
                    try:
                        coordinator.release(committed)
                    finally:
                        raise RuntimeError(
                            "route session lease was acknowledged too close to expiry"
                        )
                reservation = RequestAgentReservation(
                    permit=renewal_authorizer.permit,
                    planned=planned,
                    committed=committed,
                    coordinator=coordinator,
                    recovery_policy=recovery_policy,
                )
                with self._lock:
                    self._reservations[request.request_id] = _ManagedReservation(
                        request=request,
                        reservation=reservation,
                        next_renew_at_ms=acknowledged_at_ms + self.renew_interval_ms,
                        lease_deadline_ms=lease_deadline_ms,
                    )
                return reservation
            except BaseException:
                self.authority.release_permit(permit.permit_id)
                raise

    def replan_cold(
        self,
        request_id: str,
        *,
        failed_epoch: int,
    ) -> RequestAgentReservation:
        """Replace a failed route from a fresh DHT snapshot without a hot spare.

        Failed workers are excluded from the replacement decision, following
        Petals' client-side peer banning.  The logical request keeps its
        contribution permit while a strictly newer epoch fences the old data
        plane.  The caller rebuilds KV from its durable token journal after
        this method acknowledges the new worker leases.
        """

        request_id = str(request_id)
        self._emit_phase(request_id, "recovering")
        with self._request_operation(request_id, create=False) as request_lock:
            if request_lock is None:
                raise RuntimeError("request has no route authority to replan")
            with self._lock:
                managed = self._reservations.get(request_id)
                recoverable = self._recoverable_requests.get(request_id)

            if managed is not None:
                active = managed.reservation
                active_plan = active.committed.plan
                if active.recovery_policy != RouteRecoveryPolicy.REPLAN_COLD:
                    raise RuntimeError("active route was not admitted for cold replanning")
                if active_plan.epoch != failed_epoch:
                    raise RuntimeError(
                        f"failed epoch {failed_epoch} differs from active epoch {active_plan.epoch}"
                    )
                self.speculative_fence.fence_at_least(
                    request_id,
                    newer_epoch=failed_epoch + 1,
                )
                excluded_worker_ids = active.excluded_worker_ids | frozenset(
                    stage.worker_id for stage in active_plan.stages
                )
                current_permit = getattr(
                    getattr(active.coordinator, "renewal_authorizer", None),
                    "permit",
                    active.permit,
                )
                recoverable = _RecoverableRequest(
                    request=managed.request,
                    permit=current_permit,
                    recovery_policy=active.recovery_policy,
                    failed_epoch=failed_epoch,
                    failed_route_id=active_plan.route_id,
                    excluded_worker_ids=excluded_worker_ids,
                    failure="data-plane route failed",
                )
                with self._lock:
                    if self._reservations.get(request_id) is not managed:
                        raise RuntimeError("request route changed while replanning")
                    self._reservations.pop(request_id, None)
                    self._recoverable_requests[request_id] = recoverable
                try:
                    active.coordinator.release(active.committed)
                except Exception:
                    logger.warning(
                        "Failed route %s could not acknowledge fencing release",
                        active_plan.route_id,
                        exc_info=True,
                    )
            elif recoverable is None:
                raise RuntimeError("request has no failed route to replan")

            assert recoverable is not None
            if recoverable.failed_epoch != failed_epoch:
                raise RuntimeError(
                    f"failed epoch {failed_epoch} differs from recoverable epoch "
                    f"{recoverable.failed_epoch}"
                )
            if recoverable.recovery_policy != RouteRecoveryPolicy.REPLAN_COLD:
                raise RuntimeError("failed route was not admitted for cold replanning")

            self.speculative_fence.fence_at_least(
                request_id,
                newer_epoch=failed_epoch + 1,
            )

            renewed_permit = recoverable.permit
            try:
                self._emit_phase(request_id, "planning")
                bundle, snapshot = self._trusted_planning_snapshot(
                    recoverable.request.model_swarm_id,
                    force_refresh=True,
                )
                epoch = self.epoch_allocator.next_epoch()
                self.speculative_fence.fence_at_least(request_id, newer_epoch=epoch)
                self._emit_phase(request_id, "authorizing")
                renewed_permit = self.authority.keepalive_permit(
                    recoverable.permit.permit_id,
                    ttl_ms=self.permit_ttl_ms,
                    idempotency_key=_permit_keepalive_key(
                        request_id,
                        "cold-replan",
                        failed_epoch,
                        epoch,
                        recoverable.permit.authorization_generation,
                    ),
                )
                recoverable.permit = renewed_permit
                now_ms = self._now_ms()
                planned = self.planner.plan(
                    manifest=bundle.manifest,
                    request=recoverable.request,
                    offers=snapshot.offers,
                    leases=snapshot.leases,
                    links=snapshot.links,
                    snapshot_time_ms=snapshot.captured_at_ms,
                    coordinator_id=self.transport.peer_id(),
                    reservation_deadline_ms=now_ms + self.prepare_ttl_ms,
                    plan_expires_at_ms=now_ms + self.plan_ttl_ms,
                    epoch=epoch,
                    excluded_worker_ids=recoverable.excluded_worker_ids,
                )

                def authorize(signed_plan: SignedControlMessage) -> RouteAdmissionEnvelope:
                    return self.authority.issue_capability(
                        permit_id=renewed_permit.permit_id,
                        signed_plan=signed_plan,
                        recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
                    ).admission

                renewal_authorizer = _RouteRenewalAuthorizer(
                    authority=self.authority,
                    permit=renewed_permit,
                    recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
                    ttl_ms=self.permit_ttl_ms,
                )
                coordinator = self._coordinator_factory(
                    self.transport,
                    authorize,
                    renewal_authorizer,
                )
                self._emit_phase(request_id, "reserving")
                lease_started_at_ms = self._steady_now_ms()
                committed = coordinator.reserve(planned.plan)
                acknowledged_at_ms = self._steady_now_ms()
                lease_deadline_ms = lease_started_at_ms + self.session_ttl_ms
                if acknowledged_at_ms >= lease_deadline_ms - self.lease_expiry_guard_ms:
                    try:
                        coordinator.release(committed)
                    finally:
                        raise RuntimeError(
                            "replacement route lease was acknowledged too close to expiry"
                        )
                reservation = RequestAgentReservation(
                    permit=renewal_authorizer.permit,
                    planned=planned,
                    committed=committed,
                    coordinator=coordinator,
                    recovery_policy=RouteRecoveryPolicy.REPLAN_COLD,
                    excluded_worker_ids=recoverable.excluded_worker_ids,
                )
                with self._lock:
                    cancelled = self._recoverable_requests.get(request_id) is not recoverable
                    if not cancelled:
                        self._recoverable_requests.pop(request_id, None)
                        self._reservations[request_id] = _ManagedReservation(
                            request=recoverable.request,
                            reservation=reservation,
                            next_renew_at_ms=acknowledged_at_ms + self.renew_interval_ms,
                            lease_deadline_ms=lease_deadline_ms,
                        )
                if cancelled:
                    try:
                        coordinator.release(committed)
                    finally:
                        raise RuntimeError("request recovery was cancelled while replanning")
                return reservation
            except BaseException as error:
                with self._lock:
                    if self._recoverable_requests.get(request_id) is recoverable:
                        recoverable.permit = renewed_permit
                        recoverable.failure = f"{type(error).__name__}: {error}"[:256]
                        self._failures.append(
                            {
                                "request_id": request_id,
                                "route_id": recoverable.failed_route_id,
                                "epoch": failed_epoch,
                                "error": recoverable.failure,
                                "failed_at_ms": self._now_ms(),
                            }
                        )
                raise

    def _trusted_planning_snapshot(
        self,
        model_swarm_id: str,
        *,
        force_refresh: bool = False,
    ) -> tuple[ModelRegistryBundle, DiscoverySnapshot]:
        """Return a short-lived, TUF-matched and expiry-filtered DHT snapshot.

        Petals refreshes routing state independently and builds requests from
        the resulting local view. Fabi keeps the same separation while
        retaining worker-local PREPARE/COMMIT as the admission authority: a
        cached discovery record can only propose a route, never authorize one.
        Cold recovery bypasses this cache to discover newly joined replicas.
        """

        model_swarm_id = str(model_swarm_id)
        if not force_refresh:
            cached = self._cached_planning_snapshot(model_swarm_id)
            if cached is not None:
                return cached
        with self._discovery_refresh_lock:
            if not force_refresh:
                cached = self._cached_planning_snapshot(model_swarm_id)
                if cached is not None:
                    return cached
            bundle = self.registry.fetch(model_swarm_id)
            snapshot = self.discovery.snapshot(model_swarm_id=model_swarm_id)
            manifest = snapshot.manifest(model_swarm_id)
            if manifest is None or manifest != bundle.manifest:
                raise PermissionError("DHT manifest does not match the TUF-authenticated bundle")
            refreshed = self._filter_live_snapshot(snapshot, now_ms=self._now_ms())
            entry = _PlanningSnapshotCacheEntry(
                bundle=bundle,
                snapshot=refreshed,
                refreshed_at_ms=self._steady_now_ms(),
            )
            with self._lock:
                self._planning_snapshots[model_swarm_id] = entry
            return entry.bundle, entry.snapshot

    def _cached_planning_snapshot(
        self,
        model_swarm_id: str,
    ) -> tuple[ModelRegistryBundle, DiscoverySnapshot] | None:
        steady_now_ms = self._steady_now_ms()
        with self._lock:
            entry = self._planning_snapshots.get(model_swarm_id)
        if (
            entry is None
            or steady_now_ms - entry.refreshed_at_ms > _DISCOVERY_SNAPSHOT_CACHE_MS
        ):
            return None
        refreshed = self._filter_live_snapshot(entry.snapshot, now_ms=self._now_ms())
        return entry.bundle, refreshed

    @staticmethod
    def _filter_live_snapshot(
        snapshot: DiscoverySnapshot,
        *,
        now_ms: int,
    ) -> DiscoverySnapshot:
        """Advance a cached view without extending any signed soft-state TTL."""

        offers = tuple(offer for offer in snapshot.offers if offer.expires_at_ms > now_ms)
        live_worker_ids = {offer.worker_id for offer in offers}
        leases = tuple(
            lease
            for lease in snapshot.leases
            if lease.expires_at_ms > now_ms and lease.worker_id in live_worker_ids
        )
        links = tuple(
            link
            for link in snapshot.links
            if link.expires_at_ms > now_ms
            and link.from_worker_id in live_worker_ids
            and link.to_worker_id in live_worker_ids
        )
        return DiscoverySnapshot(
            captured_at_ms=now_ms,
            manifests=snapshot.manifests,
            offers=offers,
            leases=leases,
            links=links,
        )

    def max_supported_context_tokens(self, model_swarm_id: str, upper_bound: int) -> int:
        """Probe the largest live context with the exact planner, without reserving.

        The probe uses one immutable DHT snapshot and never allocates an epoch,
        a permit or worker KV. Feasibility is monotone for that snapshot, so a
        binary search returns the exact live boundary without configured tiers
        or memory estimates.
        """

        maximum = int(upper_bound)
        if maximum < 2:
            return 0
        bundle, snapshot = self._trusted_planning_snapshot(model_swarm_id)
        epoch = max(1, int(self.epoch_allocator.current()))
        now_ms = self._now_ms()

        def feasible(required_tokens: int) -> bool:
            request = RequestContract(
                request_id=f"readiness-{model_swarm_id[:12]}-{required_tokens}",
                model_swarm_id=model_swarm_id,
                prompt_tokens=required_tokens - 1,
                reserved_output_tokens=1,
            )
            try:
                self.planner.plan(
                    manifest=bundle.manifest,
                    request=request,
                    offers=snapshot.offers,
                    leases=snapshot.leases,
                    links=snapshot.links,
                    snapshot_time_ms=snapshot.captured_at_ms,
                    coordinator_id=self.transport.peer_id(),
                    reservation_deadline_ms=now_ms + self.prepare_ttl_ms,
                    plan_expires_at_ms=now_ms + self.plan_ttl_ms,
                    epoch=epoch,
                    route_id=f"readiness-{model_swarm_id[:12]}-{required_tokens}",
                )
            except RoutePlanningError:
                return False
            return True

        if not feasible(2):
            return 0
        if feasible(maximum):
            return maximum
        supported = 2
        rejected = maximum
        while supported + 1 < rejected:
            candidate = (supported + rejected) // 2
            if feasible(candidate):
                supported = candidate
            else:
                rejected = candidate
        return supported

    def observe_unmet_context_demand(
        self,
        request_id: str,
        model_swarm_id: str,
        required_context_tokens: int,
    ) -> bool:
        """Report aggregate placement pressure through the account authority."""

        if str(model_swarm_id) != str(self.registry.fetch(model_swarm_id).manifest.model_swarm_id):
            raise PermissionError("context demand model does not match the trusted bundle")
        return self.authority.observe_unmet_context_demand(
            request_id=str(request_id),
            model_swarm_id=str(model_swarm_id),
            required_context_tokens=int(required_context_tokens),
        )

    def renew(
        self,
        reservation: RequestAgentReservation,
        *,
        ttl_ms: int,
    ) -> RequestAgentReservation:
        request_id = reservation.committed.plan.request_id
        with self._request_operation(request_id, create=False) as request_lock:
            if request_lock is None:
                raise RuntimeError("cannot renew an inactive Request Agent reservation")
            with self._lock:
                managed = self._reservations.get(request_id)
                if (
                    managed is None
                    or managed.reservation.permit.permit_id != reservation.permit.permit_id
                ):
                    raise RuntimeError("cannot renew an inactive Request Agent reservation")
            return self._renew_locked(request_id, managed, ttl_ms=ttl_ms)

    def release(self, reservation: RequestAgentReservation) -> None:
        request_id = reservation.committed.plan.request_id
        with self._lock:
            current = self._reservations.get(request_id)
            if (
                current is None
                or current.reservation.committed.plan.route_id
                != reservation.committed.plan.route_id
            ):
                return
        self.release_request(request_id)

    def release_request(self, request_id: str) -> bool:
        """Release either an active route or a failed request awaiting replan."""

        request_id = str(request_id)
        with self._request_operation(request_id, create=False) as request_lock:
            if request_lock is None:
                return False
            with self._lock:
                current = self._reservations.pop(request_id, None)
                recoverable = self._recoverable_requests.pop(request_id, None)
            if current is not None or recoverable is not None:
                self.speculative_fence.retire(request_id)
            if current is not None:
                active = current.reservation
                try:
                    active.coordinator.release(active.committed)
                finally:
                    self.authority.release_permit(active.permit.permit_id)
                return True
            if recoverable is not None:
                self.authority.release_permit(recoverable.permit.permit_id)
                return True
            return False

    def active_reservation(self, request_id: str) -> RequestAgentReservation | None:
        """Return the latest acknowledged reservation while its lease is safe."""

        with self._lock:
            managed = self._reservations.get(str(request_id))
            if managed is None or self._steady_now_ms() >= managed.lease_deadline_ms:
                return None
            return managed.reservation

    def status(self) -> dict[str, object]:
        """Expose route health without leaking permits or capabilities."""

        steady_now_ms = self._steady_now_ms()
        with self._lock:
            routes = [
                {
                    "request_id": request_id,
                    "route_id": managed.reservation.committed.plan.route_id,
                    "epoch": managed.reservation.committed.plan.epoch,
                    "model_swarm_id": managed.request.model_swarm_id,
                    "required_context_tokens": managed.request.required_context_tokens,
                    "recovery_policy": managed.reservation.recovery_policy.value,
                    "lease_expires_in_ms": max(0, managed.lease_deadline_ms - steady_now_ms),
                    "renewal_failures": managed.consecutive_renewal_failures,
                    "last_renewal_error": managed.last_renewal_error,
                }
                for request_id, managed in sorted(self._reservations.items())
            ]
            replans = [
                {
                    "request_id": request_id,
                    "failed_route_id": recoverable.failed_route_id,
                    "failed_epoch": recoverable.failed_epoch,
                    "model_swarm_id": recoverable.request.model_swarm_id,
                    "required_context_tokens": (recoverable.request.required_context_tokens),
                    "excluded_worker_count": len(recoverable.excluded_worker_ids),
                    "last_error": recoverable.failure,
                }
                for request_id, recoverable in sorted(self._recoverable_requests.items())
            ]
            return {
                "mode": "request_agent",
                "active_routes": routes,
                "cold_replans": replans,
                "recent_failures": list(self._failures),
                "session_ttl_ms": self.session_ttl_ms,
                "renew_interval_ms": self.renew_interval_ms,
                "renew_retry_interval_ms": self.renew_retry_interval_ms,
                "lease_expiry_guard_ms": self.lease_expiry_guard_ms,
            }

    def close(self) -> None:
        self._stop_event.set()
        if self._maintenance_thread is not None:
            self._maintenance_thread.join(timeout=2.0)
            self._maintenance_thread = None
        with self._lock:
            request_ids = tuple(set(self._reservations) | set(self._recoverable_requests))
        first_error: BaseException | None = None
        for request_id in request_ids:
            try:
                self.release_request(request_id)
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if self._close_transport:
            close = getattr(self.transport, "close", None)
            if callable(close):
                close()
            self._close_transport = False
        if first_error is not None:
            raise first_error

    def _maintenance_loop(self) -> None:
        interval_seconds = min(2.0, self.renew_interval_ms / 1000)
        while not self._stop_event.wait(interval_seconds):
            try:
                self.maintain_once()
            except Exception:
                logger.exception("Unexpected Request Agent lease maintenance failure")

    def maintain_once(self) -> None:
        """Renew every due route once; public to support deterministic tests."""

        now_ms = self._steady_now_ms()
        with self._lock:
            due = [
                (request_id, managed)
                for request_id, managed in self._reservations.items()
                if managed.next_renew_at_ms <= now_ms
            ]
        for request_id, managed in due:
            self._renew_managed(request_id, managed)

    def _renew_managed(self, request_id: str, managed: _ManagedReservation) -> None:
        with self._request_operation(request_id, create=False) as request_lock:
            if request_lock is None:
                return
            with self._lock:
                if self._reservations.get(request_id) is not managed:
                    return
                attempt_started_at_ms = self._steady_now_ms()
                safe_retry_deadline_ms = (
                    managed.lease_deadline_ms
                    - self.renew_attempt_budget_ms
                    - self.lease_expiry_guard_ms
                )
                unsafe_to_attempt = attempt_started_at_ms >= safe_retry_deadline_ms
            if unsafe_to_attempt:
                self._invalidate_under_request_lock(
                    request_id,
                    managed,
                    TimeoutError("no safe lease window remains for another renewal attempt"),
                )
                return
            try:
                self._renew_locked(
                    request_id,
                    managed,
                    ttl_ms=self.session_ttl_ms,
                    attempt_started_at_ms=attempt_started_at_ms,
                )
            except Exception as error:
                now_ms = self._steady_now_ms()
                with self._lock:
                    if self._reservations.get(request_id) is not managed:
                        return
                    managed.consecutive_renewal_failures += 1
                    managed.last_renewal_error = f"{type(error).__name__}: {error}"[:256]
                    safe_retry_deadline_ms = (
                        managed.lease_deadline_ms
                        - self.renew_attempt_budget_ms
                        - self.lease_expiry_guard_ms
                    )
                    can_retry = now_ms + self.renew_retry_interval_ms < safe_retry_deadline_ms
                    if can_retry:
                        managed.next_renew_at_ms = now_ms + self.renew_retry_interval_ms
                if can_retry:
                    logger.warning(
                        "Request Agent route %s lease renewal failed "
                        "(attempt %d); retrying before acknowledged expiry: %s",
                        request_id,
                        managed.consecutive_renewal_failures,
                        error,
                    )
                    return
                self._invalidate_under_request_lock(request_id, managed, error)

    def _renew_locked(
        self,
        request_id: str,
        managed: _ManagedReservation,
        *,
        ttl_ms: int,
        attempt_started_at_ms: int | None = None,
    ) -> RequestAgentReservation:
        """Renew while the caller serializes this request against release."""

        if ttl_ms <= self.lease_expiry_guard_ms:
            raise ValueError("renew TTL must exceed the lease expiry guard")
        if attempt_started_at_ms is None:
            attempt_started_at_ms = self._steady_now_ms()
        current = managed.reservation
        committed = current.coordinator.renew(current.committed, ttl_ms=ttl_ms)
        acknowledged_at_ms = self._steady_now_ms()
        deadline_ms = attempt_started_at_ms + ttl_ms
        if acknowledged_at_ms >= deadline_ms - self.lease_expiry_guard_ms:
            raise TimeoutError("lease renewal acknowledgement arrived after the safe deadline")
        renewed = RequestAgentReservation(
            permit=getattr(
                getattr(current.coordinator, "renewal_authorizer", None),
                "permit",
                current.permit,
            ),
            planned=current.planned,
            committed=committed,
            coordinator=current.coordinator,
            recovery_policy=current.recovery_policy,
        )
        with self._lock:
            if self._reservations.get(request_id) is not managed:
                raise RuntimeError("cannot renew an inactive Request Agent reservation")
            managed.reservation = renewed
            managed.lease_deadline_ms = deadline_ms
            managed.next_renew_at_ms = acknowledged_at_ms + min(
                self.renew_interval_ms,
                max(1, (ttl_ms - self.lease_expiry_guard_ms) // 3),
            )
            managed.consecutive_renewal_failures = 0
            managed.last_renewal_error = None
        return renewed

    def _invalidate_under_request_lock(
        self,
        request_id: str,
        managed: _ManagedReservation,
        error: Exception,
    ) -> None:
        """Fence locally first, then clean up the remote route best-effort."""

        self.speculative_fence.fence_at_least(
            request_id,
            newer_epoch=managed.reservation.committed.plan.epoch + 1,
        )
        with self._lock:
            if self._reservations.get(request_id) is not managed:
                return
            self._reservations.pop(request_id, None)
            active = managed.reservation
            current_permit = getattr(
                getattr(active.coordinator, "renewal_authorizer", None),
                "permit",
                active.permit,
            )
            retain_for_replan = active.recovery_policy == RouteRecoveryPolicy.REPLAN_COLD
            if retain_for_replan:
                self._recoverable_requests[request_id] = _RecoverableRequest(
                    request=managed.request,
                    permit=current_permit,
                    recovery_policy=active.recovery_policy,
                    failed_epoch=active.committed.plan.epoch,
                    failed_route_id=active.committed.plan.route_id,
                    excluded_worker_ids=active.excluded_worker_ids
                    | frozenset(stage.worker_id for stage in active.committed.plan.stages),
                    failure=f"{type(error).__name__}: {error}"[:256],
                )
            self._failures.append(
                {
                    "request_id": request_id,
                    "route_id": managed.reservation.committed.plan.route_id,
                    "epoch": managed.reservation.committed.plan.epoch,
                    "error": f"{type(error).__name__}: {error}"[:256],
                    "failed_at_ms": self._now_ms(),
                }
            )
        try:
            managed.reservation.coordinator.release(managed.reservation.committed)
        except Exception:
            logger.warning(
                "Failed to release expired Request Agent route %s",
                request_id,
                exc_info=True,
            )
        finally:
            if not retain_for_replan:
                try:
                    self.authority.release_permit(current_permit.permit_id)
                except Exception:
                    logger.warning(
                        "Failed to release expired Request Agent permit %s",
                        request_id,
                        exc_info=True,
                    )
