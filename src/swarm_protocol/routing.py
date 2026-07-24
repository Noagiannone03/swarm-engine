"""Exact request-time route planning over ready Fabi v3 span leases.

The algorithm borrows Petals' ability to compose effective subspans and Parallax's dynamic path
optimization.  Discovery snapshots are hints only: the resulting plan must still pass worker-local
PREPARE/COMMIT admission before execution.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from swarm_protocol.contracts import (
    EffectiveSpanMode,
    LayerSpan,
    LinkMetric,
    ModelManifest,
    PathKind,
    RecoveryLevel,
    RequestContract,
    RoutePlan,
    RouteStage,
    SpanLease,
    SpanState,
    WorkerOffer,
    WorkerRole,
)


class RoutePlanningError(RuntimeError):
    """Base class for deterministic route planning failures."""


class NoFeasibleRoute(RoutePlanningError):
    """No exact, connected and context-capable pipeline exists in the snapshot."""


@dataclass(frozen=True)
class RouteCandidate:
    offer: WorkerOffer
    lease: SpanLease

    def __post_init__(self) -> None:
        if self.offer.worker_id != self.lease.worker_id:
            raise ValueError("worker offer and span lease identities do not match")


@dataclass(frozen=True)
class RouteEstimate:
    ttft_ms: float
    inter_token_ms: float
    complete: bool = True

    def projected_total_ms(self, reserved_output_tokens: int) -> float:
        return self.ttft_ms + self.inter_token_ms * reserved_output_tokens


@dataclass(frozen=True)
class PlannedRoute:
    plan: RoutePlan
    estimate: RouteEstimate


@dataclass(frozen=True)
class _Segment:
    candidate: RouteCandidate
    effective_span: LayerSpan


@dataclass(frozen=True)
class _PartialPath:
    segments: tuple[_Segment, ...]
    ttft_ms: float
    inter_token_ms: float
    unknown_cost_components: int

    def score(self, reserved_output_tokens: int) -> tuple[int, float, int, tuple[str, ...]]:
        return (
            self.unknown_cost_components,
            self.ttft_ms + self.inter_token_ms * reserved_output_tokens,
            len(self.segments),
            tuple(segment.candidate.offer.worker_id for segment in self.segments),
        )


class ExactRoutePlanner:
    """Build a complete route from immutable discovery snapshots."""

    def __init__(self, *, relay_penalty: float = 1.15) -> None:
        if relay_penalty < 1:
            raise ValueError("relay_penalty must be at least 1")
        self.relay_penalty = relay_penalty

    @staticmethod
    def _stage_compute_cost(
        candidate: RouteCandidate, span: LayerSpan, request: RequestContract
    ) -> RouteEstimate:
        lease = candidate.lease
        if (
            lease.measured_prefill_tokens_per_second is None
            or lease.measured_decode_tokens_per_second is None
        ):
            # Missing cold-start telemetry must not masquerade as a capacity
            # failure. Keep the estimate explicitly incomplete; exact memory
            # admission remains worker-local during PREPARE.
            return RouteEstimate(0, 0, complete=False)
        fraction = span.length / lease.hosted_span.length
        return RouteEstimate(
            ttft_ms=(
                request.prompt_tokens / lease.measured_prefill_tokens_per_second * 1000 * fraction
            ),
            inter_token_ms=(1000 / lease.measured_decode_tokens_per_second * fraction),
        )

    def _link_cost(
        self, metric: LinkMetric, manifest: ModelManifest, request: RequestContract
    ) -> RouteEstimate:
        multiplier = self.relay_penalty if metric.path_kind == PathKind.RELAY else 1.0
        # Measured goodput already captures ordinary loss.  The explicit loss factor is a
        # conservative tail-risk penalty, not a second bandwidth correction.
        reliability_penalty = 1.0 + metric.loss_rate
        if metric.throughput_bytes_per_second is None:
            # The authenticated health probe proves the edge exists.  Until a
            # real transfer or bounded calibration supplies goodput, preserve
            # the route without inventing bandwidth and mark its estimate
            # incomplete so measured alternatives always rank first.
            return RouteEstimate(
                ttft_ms=metric.rtt_ms * multiplier * reliability_penalty,
                inter_token_ms=metric.rtt_ms * multiplier * reliability_penalty,
                complete=False,
            )
        prefill_transfer_ms = (
            manifest.activation_bytes_per_token
            * request.prompt_tokens
            / metric.throughput_bytes_per_second
            * 1000
        )
        decode_transfer_ms = (
            manifest.activation_bytes_per_token / metric.throughput_bytes_per_second * 1000
        )
        return RouteEstimate(
            ttft_ms=(metric.rtt_ms + prefill_transfer_ms) * multiplier * reliability_penalty,
            inter_token_ms=(metric.rtt_ms + decode_transfer_ms) * multiplier * reliability_penalty,
        )

    def _link_map(
        self,
        links: tuple[LinkMetric, ...],
        snapshot_time_ms: int,
        manifest: ModelManifest,
        request: RequestContract,
    ) -> dict[tuple[str, str], LinkMetric]:
        result: dict[tuple[str, str], LinkMetric] = {}
        for metric in links:
            if metric.expires_at_ms <= snapshot_time_ms:
                continue
            key = (metric.from_worker_id, metric.to_worker_id)
            previous = result.get(key)
            metric_cost = self._link_cost(metric, manifest, request)
            previous_cost = self._link_cost(previous, manifest, request) if previous else None
            if previous_cost is None or (
                int(not metric_cost.complete),
                metric_cost.projected_total_ms(request.reserved_output_tokens),
                metric.path_kind.value,
            ) < (
                int(not previous_cost.complete),
                previous_cost.projected_total_ms(request.reserved_output_tokens),
                previous.path_kind.value,
            ):
                result[key] = metric
        return result

    @staticmethod
    def _eligible_candidates(
        *,
        request: RequestContract,
        offers: tuple[WorkerOffer, ...],
        leases: tuple[SpanLease, ...],
        snapshot_time_ms: int,
    ) -> tuple[RouteCandidate, ...]:
        offers_by_worker = {
            offer.worker_id: offer
            for offer in offers
            if offer.expires_at_ms > snapshot_time_ms
            and WorkerRole.EXECUTOR in offer.supported_roles
        }
        candidates = []
        for lease in leases:
            offer = offers_by_worker.get(lease.worker_id)
            if (
                offer is None
                or lease.model_swarm_id != request.model_swarm_id
                or lease.state != SpanState.READY
                or lease.expires_at_ms <= snapshot_time_ms
            ):
                continue
            candidates.append(RouteCandidate(offer=offer, lease=lease))
        return tuple(sorted(candidates, key=lambda item: item.offer.worker_id))

    @staticmethod
    def _stage_options(
        candidate: RouteCandidate,
        *,
        start_layer: int,
        request: RequestContract,
        model_num_layers: int,
    ) -> tuple[LayerSpan, ...]:
        lease = candidate.lease
        hosted = lease.hosted_span
        if not (hosted.start <= start_layer < hosted.end):
            return ()

        if lease.effective_span_mode == EffectiveSpanMode.FIXED:
            if start_layer != hosted.start or hosted.end > model_num_layers:
                return ()
            required_bytes = lease.kv_geometry.required_bytes(
                hosted, request.required_context_tokens
            )
            return (hosted,) if required_bytes <= lease.available_kv_bytes_snapshot else ()

        return tuple(
            span
            for end in range(start_layer + 1, min(hosted.end, model_num_layers) + 1)
            if lease.kv_geometry.required_bytes(
                (span := LayerSpan(start=start_layer, end=end)),
                request.required_context_tokens,
            )
            <= lease.available_kv_bytes_snapshot
        )

    def plan(
        self,
        *,
        manifest: ModelManifest,
        request: RequestContract,
        offers: tuple[WorkerOffer, ...],
        leases: tuple[SpanLease, ...],
        links: tuple[LinkMetric, ...],
        snapshot_time_ms: int,
        coordinator_id: str,
        reservation_deadline_ms: int,
        plan_expires_at_ms: int,
        epoch: int = 0,
        route_id: str | None = None,
    ) -> PlannedRoute:
        if request.model_swarm_id != manifest.model_swarm_id:
            raise NoFeasibleRoute("request and model manifest identify different swarms")
        if request.recovery_level == RecoveryLevel.RECOVERABLE:
            raise NoFeasibleRoute("recoverable routing requires an alternate coverage plan")

        candidates = self._eligible_candidates(
            request=request,
            offers=offers,
            leases=leases,
            snapshot_time_ms=snapshot_time_ms,
        )
        link_map = self._link_map(links, snapshot_time_ms, manifest, request)
        best_complete: _PartialPath | None = None

        heads = [
            candidate
            for candidate in candidates
            if candidate.lease.hosted_span.start == 0
            and WorkerRole.FRONTEND in candidate.offer.supported_roles
        ]
        for head in heads:
            for head_span in self._stage_options(
                head,
                start_layer=0,
                request=request,
                model_num_layers=manifest.num_layers,
            ):
                head_cost = self._stage_compute_cost(head, head_span, request)
                initial = _PartialPath(
                    (_Segment(head, head_span),),
                    head_cost.ttft_ms,
                    head_cost.inter_token_ms,
                    int(not head_cost.complete),
                )
                states: dict[tuple[int, str], _PartialPath] = {
                    (head_span.end, head.offer.worker_id): initial
                }

                for position in range(head_span.end, manifest.num_layers + 1):
                    current_states = [
                        (key, path) for key, path in tuple(states.items()) if key[0] == position
                    ]
                    for (_, previous_worker), partial in current_states:
                        if position == manifest.num_layers:
                            tail = partial.segments[-1].candidate
                            if tail.offer.worker_id == head.offer.worker_id:
                                closure_cost = RouteEstimate(0, 0)
                            else:
                                closure = link_map.get((tail.offer.worker_id, head.offer.worker_id))
                                if closure is None:
                                    continue
                                closure_cost = self._link_cost(closure, manifest, request)
                            complete = _PartialPath(
                                partial.segments,
                                partial.ttft_ms,
                                partial.inter_token_ms + closure_cost.inter_token_ms,
                                partial.unknown_cost_components + int(not closure_cost.complete),
                            )
                            if best_complete is None or complete.score(
                                request.reserved_output_tokens
                            ) < best_complete.score(request.reserved_output_tokens):
                                best_complete = complete
                            continue

                        for candidate in candidates:
                            if candidate.offer.worker_id == previous_worker:
                                continue
                            link = link_map.get((previous_worker, candidate.offer.worker_id))
                            if link is None:
                                continue
                            for span in self._stage_options(
                                candidate,
                                start_layer=position,
                                request=request,
                                model_num_layers=manifest.num_layers,
                            ):
                                compute = self._stage_compute_cost(candidate, span, request)
                                network = self._link_cost(link, manifest, request)
                                proposed = _PartialPath(
                                    partial.segments + (_Segment(candidate, span),),
                                    partial.ttft_ms + network.ttft_ms + compute.ttft_ms,
                                    partial.inter_token_ms
                                    + network.inter_token_ms
                                    + compute.inter_token_ms,
                                    partial.unknown_cost_components
                                    + int(not compute.complete)
                                    + int(not network.complete),
                                )
                                key = (span.end, candidate.offer.worker_id)
                                previous = states.get(key)
                                if previous is None or proposed.score(
                                    request.reserved_output_tokens
                                ) < previous.score(request.reserved_output_tokens):
                                    states[key] = proposed

        if best_complete is None:
            raise NoFeasibleRoute(
                "no complete route satisfies model, context, endpoint and link constraints"
            )

        segments = best_complete.segments
        route_stages = []
        for index, segment in enumerate(segments):
            next_segment = segments[(index + 1) % len(segments)]
            if segment.candidate.offer.worker_id == next_segment.candidate.offer.worker_id:
                path_kind = PathKind.DIRECT
            else:
                metric = link_map[
                    (segment.candidate.offer.worker_id, next_segment.candidate.offer.worker_id)
                ]
                path_kind = metric.path_kind
            geometry = segment.candidate.lease.kv_geometry
            route_stages.append(
                RouteStage(
                    worker_id=segment.candidate.offer.worker_id,
                    endpoint_id=segment.candidate.offer.endpoint_id,
                    hosted_span=segment.candidate.lease.hosted_span,
                    effective_span=segment.effective_span,
                    path_to_next=path_kind,
                    rounded_context_tokens=geometry.rounded_tokens(request.required_context_tokens),
                    exact_kv_bytes=geometry.required_bytes(
                        segment.effective_span, request.required_context_tokens
                    ),
                )
            )

        plan = RoutePlan(
            request_id=request.request_id,
            route_id=route_id or uuid.uuid4().hex,
            epoch=epoch,
            model_swarm_id=request.model_swarm_id,
            model_num_layers=manifest.num_layers,
            prompt_tokens=request.prompt_tokens,
            reserved_output_tokens=request.reserved_output_tokens,
            stages=tuple(route_stages),
            recovery_level=request.recovery_level,
            coordinator_id=coordinator_id,
            reservation_deadline_ms=reservation_deadline_ms,
            plan_expires_at_ms=plan_expires_at_ms,
        )
        return PlannedRoute(
            plan=plan,
            estimate=RouteEstimate(
                best_complete.ttft_ms,
                best_complete.inter_token_ms,
                complete=best_complete.unknown_cost_components == 0,
            ),
        )
