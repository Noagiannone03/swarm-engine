"""Local V3 Request Agent planning and route coordination."""

from __future__ import annotations

import os
import stat
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
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
from swarm_protocol.discovery import DiscoveryStore
from swarm_protocol.epochs import EpochAllocator
from swarm_protocol.registry import ModelRegistryBundle, TrustedModelRegistry
from swarm_protocol.route_authority import RouteAdmissionEnvelope
from swarm_protocol.routing import ExactRoutePlanner, PlannedRoute

_MAX_AUTHORITY_RESPONSE_BYTES = 1024 * 1024


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
        payload = self._request(
            "POST",
            "/v1/swarm/route-permits",
            headers={"Idempotency-Key": request_id},
            json={
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
    [ControlTransport, Callable[[SignedControlMessage], RouteAdmissionEnvelope]],
    ReservationCoordinator,
]


def _default_coordinator_factory(
    transport: ControlTransport,
    authorizer: Callable[[SignedControlMessage], RouteAdmissionEnvelope],
) -> RouteReservationCoordinator:
    return RouteReservationCoordinator(
        transport,
        admission_authorizer=authorizer,
    )


@dataclass(frozen=True)
class RequestAgentReservation:
    permit: RoutePermitGrant
    planned: PlannedRoute
    committed: CommittedRoute
    coordinator: ReservationCoordinator
    recovery_policy: RouteRecoveryPolicy


class RequestAgentRouteRuntime:
    """Plan from local DHT state and coordinate one exact route from the client."""

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
        clock_ms: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
        prepare_ttl_ms: int = 5_000,
        plan_ttl_ms: int = 10_000,
        permit_ttl_ms: int = 120_000,
        close_transport: bool = False,
    ) -> None:
        if prepare_ttl_ms <= 0 or plan_ttl_ms <= prepare_ttl_ms:
            raise ValueError("plan TTL must exceed the positive prepare TTL")
        if permit_ttl_ms < plan_ttl_ms or permit_ttl_ms > 300_000:
            raise ValueError("permit TTL must cover the plan and remain below five minutes")
        self.transport = transport
        self.discovery = discovery
        self.registry = registry
        self.authority = authority
        self.epoch_allocator = epoch_allocator
        self.planner = planner or ExactRoutePlanner()
        self._coordinator_factory = coordinator_factory
        self._clock_ms = clock_ms
        self.prepare_ttl_ms = prepare_ttl_ms
        self.plan_ttl_ms = plan_ttl_ms
        self.permit_ttl_ms = permit_ttl_ms
        self._close_transport = close_transport
        self._request_locks: dict[str, threading.Lock] = {}
        self._reservations: dict[
            str,
            tuple[RequestContract, RequestAgentReservation],
        ] = {}
        self._lock = threading.RLock()

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
            )
        except BaseException:
            transport.close()
            raise

    def _now_ms(self) -> int:
        now_ms = int(self._clock_ms())
        if now_ms < 0:
            raise RuntimeError("Request Agent clock returned a negative timestamp")
        return now_ms

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
        with self._lock:
            request_lock = self._request_locks.setdefault(request.request_id, threading.Lock())
        with request_lock:
            with self._lock:
                existing = self._reservations.get(request.request_id)
            if existing is not None:
                existing_request, reservation = existing
                if existing_request != request or reservation.recovery_policy != recovery_policy:
                    raise ValueError(
                        "request id is already reserved with a different route contract"
                    )
                return reservation
            bundle = self.registry.fetch(request.model_swarm_id)
            snapshot = self.discovery.snapshot(model_swarm_id=request.model_swarm_id)
            manifest = snapshot.manifest(request.model_swarm_id)
            if manifest is None or manifest != bundle.manifest:
                raise PermissionError("DHT manifest does not match the TUF-authenticated bundle")
            epoch = self.epoch_allocator.next_epoch()
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

                coordinator = self._coordinator_factory(self.transport, authorize)
                committed = coordinator.reserve(planned.plan)
                reservation = RequestAgentReservation(
                    permit=permit,
                    planned=planned,
                    committed=committed,
                    coordinator=coordinator,
                    recovery_policy=recovery_policy,
                )
                with self._lock:
                    self._reservations[request.request_id] = (request, reservation)
                return reservation
            except BaseException:
                self.authority.release_permit(permit.permit_id)
                raise

    def renew(
        self,
        reservation: RequestAgentReservation,
        *,
        ttl_ms: int,
    ) -> RequestAgentReservation:
        with self._lock:
            request_lock = self._request_locks.get(reservation.committed.plan.request_id)
        if request_lock is None:
            raise RuntimeError("cannot renew an inactive Request Agent reservation")
        with request_lock:
            committed = reservation.coordinator.renew(
                reservation.committed,
                ttl_ms=ttl_ms,
            )
            renewed = RequestAgentReservation(
                permit=reservation.permit,
                planned=reservation.planned,
                committed=committed,
                coordinator=reservation.coordinator,
                recovery_policy=reservation.recovery_policy,
            )
            request_id = committed.plan.request_id
            with self._lock:
                current = self._reservations.get(request_id)
                if current is None or current[1] != reservation:
                    raise RuntimeError("cannot renew an inactive Request Agent reservation")
                self._reservations[request_id] = (current[0], renewed)
            return renewed

    def release(self, reservation: RequestAgentReservation) -> None:
        request_id = reservation.committed.plan.request_id
        with self._lock:
            request_lock = self._request_locks.get(request_id)
        if request_lock is None:
            return
        with request_lock:
            with self._lock:
                current = self._reservations.pop(request_id, None)
                self._request_locks.pop(request_id, None)
            if current is None:
                return
            active = current[1]
            try:
                active.coordinator.release(active.committed)
            finally:
                self.authority.release_permit(active.permit.permit_id)

    def close(self) -> None:
        with self._lock:
            reservations = tuple(item[1] for item in self._reservations.values())
        first_error: BaseException | None = None
        for reservation in reservations:
            try:
                self.release(reservation)
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
