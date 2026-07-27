from __future__ import annotations

from parallax.p2p.liveness import (
    LocalHealthAwareness,
    PhiAccrualFailureDetector,
)


def test_phi_detector_learns_intervals_and_recovers_without_poisoning_history() -> None:
    now = [0.0]
    detector = PhiAccrualFailureDetector(clock=lambda: now[0])

    detector.heartbeat()
    for timestamp in (10.0, 20.0, 30.0):
        now[0] = timestamp
        detector.heartbeat()

    now[0] = 70.0
    healthy = detector.snapshot()
    assert healthy.state == "healthy"

    now[0] = 90.0
    suspect = detector.snapshot()
    assert suspect.state == "suspect"
    samples_before_recovery = suspect.samples

    detector.heartbeat()
    recovered = detector.snapshot()
    assert recovered.state == "healthy"
    assert recovered.samples == samples_before_recovery

    now[0] = 100.0
    detector.heartbeat()
    assert detector.snapshot().samples == samples_before_recovery + 1


def test_local_health_scales_suspicion_but_never_the_hard_lease() -> None:
    now = [0.0]
    detector = PhiAccrualFailureDetector(clock=lambda: now[0])
    detector.heartbeat()

    now[0] = 60.0
    assert detector.snapshot(local_health_multiplier=1).state == "suspect"
    assert detector.snapshot(local_health_multiplier=2).state == "healthy"

    # Local observer degradation may reduce confidence, but it cannot extend
    # the fencing/expiry boundary.
    now[0] = 120.0
    assert (
        detector.snapshot(
            local_health_multiplier=8,
            hard_timeout_seconds=120.0,
        ).state
        == "expired"
    )


def test_local_health_awareness_is_bounded_and_heals_gradually() -> None:
    awareness = LocalHealthAwareness(max_multiplier=4)

    assert awareness.observe_loop_delay(0.51, 0.5) == 1
    assert awareness.observe_loop_delay(1.6, 0.5) == 3
    assert awareness.observe_loop_delay(10.0, 0.5) == 4
    assert awareness.observe_loop_delay(0.5, 0.5) == 3
    assert awareness.observe_loop_delay(0.5, 0.5) == 2
    assert awareness.observe_loop_delay(0.5, 0.5) == 1


def test_phi_uses_monotonic_injected_clock() -> None:
    monotonic_now = [1_000.0]
    detector = PhiAccrualFailureDetector(clock=lambda: monotonic_now[0])
    detector.heartbeat()

    monotonic_now[0] += 10.0
    detector.heartbeat()
    monotonic_now[0] += 10.0

    snapshot = detector.snapshot()
    assert snapshot.state == "healthy"
    assert snapshot.heartbeat_age_seconds == 10.0
