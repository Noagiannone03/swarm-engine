"""Liveness policy for the worker-to-scheduler control channel.

There are deliberately two different decisions:

* an adaptive phi-accrual detector marks a worker *suspect* and prevents new
  work from being assigned while observations are abnormal;
* the scheduler's hard heartbeat lease eventually expires the worker and
  triggers fencing/reallocation.

Suspicion is reversible and never destroys an allocation.  Expiry is a safety
boundary.  Keeping those decisions separate prevents a congested network or a
locally overloaded scheduler from turning one late heartbeat into cluster-wide
model reloads.

The phi calculation follows Hayashibara et al. and Akka's production
implementation.  ``LocalHealthAwareness`` follows the timeout-scaling invariant
from HashiCorp memberlist/Lifeguard: an observer that is itself missing
deadlines becomes less confident when accusing peers.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import statistics
import threading
import time
from typing import Callable, Deque

WORKER_HEARTBEAT_INTERVAL_SECONDS = 10.0
WORKER_HEARTBEAT_RPC_TIMEOUT_SECONDS = 30.0
MIN_SCHEDULER_HEARTBEAT_TIMEOUT_SECONDS = (
    WORKER_HEARTBEAT_RPC_TIMEOUT_SECONDS + 3 * WORKER_HEARTBEAT_INTERVAL_SECONDS
)
DEFAULT_SCHEDULER_HEARTBEAT_TIMEOUT_SECONDS = 120.0
DEFAULT_PHI_THRESHOLD = 8.0
DEFAULT_PHI_MAX_SAMPLE_SIZE = 1_000
DEFAULT_PHI_MIN_STD_DEVIATION_SECONDS = 1.0
DEFAULT_PHI_ACCEPTABLE_PAUSE_SECONDS = WORKER_HEARTBEAT_RPC_TIMEOUT_SECONDS
DEFAULT_LOCAL_HEALTH_MAX_MULTIPLIER = 8


@dataclass(frozen=True)
class LivenessSnapshot:
    """JSON-safe diagnostic view of one adaptive failure detector."""

    state: str
    phi: float
    heartbeat_age_seconds: float | None
    mean_interval_seconds: float
    std_deviation_seconds: float
    samples: int
    local_health_multiplier: int


class LocalHealthAwareness:
    """Bounded Lifeguard-style estimate of the observer's own responsiveness.

    A late scheduler loop increases the score, while an on-time loop heals it
    one step at a time.  The score scales only the reversible suspicion margin;
    it never extends the hard worker lease.
    """

    def __init__(self, max_multiplier: int = DEFAULT_LOCAL_HEALTH_MAX_MULTIPLIER) -> None:
        if max_multiplier < 1:
            raise ValueError("max_multiplier must be at least 1")
        self._max_score = max_multiplier - 1
        self._score = 0
        self._lock = threading.Lock()

    def observe_loop_delay(self, actual_seconds: float, expected_seconds: float) -> int:
        """Record scheduler-loop responsiveness and return the new multiplier."""

        actual = max(0.0, float(actual_seconds))
        expected = float(expected_seconds)
        if not math.isfinite(actual) or not math.isfinite(expected) or expected <= 0:
            raise ValueError("loop delays must be finite and expected_seconds must be positive")

        # Missing one or more complete expected periods means the local
        # observer was degraded.  On-time observations heal gradually, exactly
        # like memberlist's bounded awareness score.
        missed_periods = max(0, math.floor(actual / expected) - 1)
        delta = min(self._max_score, missed_periods) if missed_periods else -1
        with self._lock:
            self._score = min(self._max_score, max(0, self._score + delta))
            return self._score + 1

    @property
    def multiplier(self) -> int:
        with self._lock:
            return self._score + 1


class PhiAccrualFailureDetector:
    """Adaptive, monotonic-clock failure suspicion for one worker.

    ``heartbeat`` updates a bounded sample window.  A heartbeat received after
    the detector already considered the worker suspect resets liveness but is
    not added to the history, preventing a long outage from teaching the
    detector that very large intervals are normal.
    """

    def __init__(
        self,
        *,
        threshold: float = DEFAULT_PHI_THRESHOLD,
        max_sample_size: int = DEFAULT_PHI_MAX_SAMPLE_SIZE,
        min_std_deviation_seconds: float = DEFAULT_PHI_MIN_STD_DEVIATION_SECONDS,
        acceptable_pause_seconds: float = DEFAULT_PHI_ACCEPTABLE_PAUSE_SECONDS,
        first_heartbeat_estimate_seconds: float = WORKER_HEARTBEAT_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        if max_sample_size < 2:
            raise ValueError("max_sample_size must be at least 2")
        for name, value, allow_zero in (
            ("min_std_deviation_seconds", min_std_deviation_seconds, False),
            ("acceptable_pause_seconds", acceptable_pause_seconds, True),
            ("first_heartbeat_estimate_seconds", first_heartbeat_estimate_seconds, False),
        ):
            numeric = float(value)
            if not math.isfinite(numeric) or numeric < 0 or (not allow_zero and numeric == 0):
                qualifier = "non-negative" if allow_zero else "positive"
                raise ValueError(f"{name} must be finite and {qualifier}")

        estimate = float(first_heartbeat_estimate_seconds)
        bootstrap_deviation = estimate / 4
        self.threshold = float(threshold)
        self.max_sample_size = int(max_sample_size)
        self.min_std_deviation_seconds = float(min_std_deviation_seconds)
        self.acceptable_pause_seconds = float(acceptable_pause_seconds)
        self._clock = clock
        self._intervals: Deque[float] = deque(
            (estimate - bootstrap_deviation, estimate + bootstrap_deviation),
            maxlen=self.max_sample_size,
        )
        self._last_heartbeat: float | None = None
        self._lock = threading.RLock()

    def heartbeat(
        self,
        *,
        now: float | None = None,
        local_health_multiplier: int = 1,
        sample_interval: bool = True,
    ) -> None:
        """Record a successful worker heartbeat."""

        observed_at = self._resolve_now(now)
        with self._lock:
            previous = self._last_heartbeat
            if previous is not None and sample_interval:
                interval = max(0.0, observed_at - previous)
                if self._phi_locked(observed_at, local_health_multiplier) < self.threshold:
                    self._intervals.append(interval)
            self._last_heartbeat = observed_at

    def phi(self, *, now: float | None = None, local_health_multiplier: int = 1) -> float:
        """Return the current suspicion level; larger values are less healthy."""

        observed_at = self._resolve_now(now)
        with self._lock:
            return self._phi_locked(observed_at, local_health_multiplier)

    def is_available(self, *, now: float | None = None, local_health_multiplier: int = 1) -> bool:
        return self.phi(now=now, local_health_multiplier=local_health_multiplier) < self.threshold

    def heartbeat_age(self, *, now: float | None = None) -> float | None:
        observed_at = self._resolve_now(now)
        with self._lock:
            if self._last_heartbeat is None:
                return None
            return max(0.0, observed_at - self._last_heartbeat)

    def snapshot(
        self,
        *,
        now: float | None = None,
        local_health_multiplier: int = 1,
        hard_timeout_seconds: float | None = None,
    ) -> LivenessSnapshot:
        observed_at = self._resolve_now(now)
        multiplier = self._validate_multiplier(local_health_multiplier)
        with self._lock:
            age = (
                None
                if self._last_heartbeat is None
                else max(0.0, observed_at - self._last_heartbeat)
            )
            mean, std_deviation = self._distribution_locked()
            phi = self._phi_locked(observed_at, multiplier)
            if (
                hard_timeout_seconds is not None
                and age is not None
                and age >= float(hard_timeout_seconds)
            ):
                state = "expired"
            elif phi >= self.threshold:
                state = "suspect"
            else:
                state = "healthy"
            return LivenessSnapshot(
                state=state,
                phi=round(phi, 6),
                heartbeat_age_seconds=None if age is None else round(age, 6),
                mean_interval_seconds=round(mean, 6),
                std_deviation_seconds=round(std_deviation, 6),
                samples=len(self._intervals),
                local_health_multiplier=multiplier,
            )

    def _phi_locked(self, now: float, local_health_multiplier: int) -> float:
        if self._last_heartbeat is None:
            return 0.0
        multiplier = self._validate_multiplier(local_health_multiplier)
        elapsed = max(0.0, now - self._last_heartbeat)
        mean, std_deviation = self._distribution_locked()
        adjusted_mean = mean + self.acceptable_pause_seconds * multiplier
        y = (elapsed - adjusted_mean) / std_deviation
        exponent = -y * (1.5976 + 0.070566 * y * y)

        # The logistic approximation is the same stable approximation used by
        # Akka. Algebra reduces both CDF branches to softplus(-exponent);
        # handle its linear tail explicitly so extreme pauses cannot overflow.
        if exponent < -700:
            return -exponent / math.log(10)
        return math.log1p(math.exp(-exponent)) / math.log(10)

    def _distribution_locked(self) -> tuple[float, float]:
        mean = statistics.fmean(self._intervals)
        variance = statistics.fmean((sample - mean) ** 2 for sample in self._intervals)
        return mean, max(math.sqrt(max(0.0, variance)), self.min_std_deviation_seconds)

    @staticmethod
    def _validate_multiplier(value: int) -> int:
        multiplier = int(value)
        if multiplier < 1:
            raise ValueError("local_health_multiplier must be at least 1")
        return multiplier

    def _resolve_now(self, now: float | None) -> float:
        observed_at = self._clock() if now is None else float(now)
        if not math.isfinite(observed_at):
            raise ValueError("clock value must be finite")
        return observed_at


def validate_scheduler_heartbeat_timeout(value: float) -> float:
    """Validate the product lease without constraining low-level test schedulers."""

    timeout = float(value)
    if not math.isfinite(timeout):
        raise ValueError("heartbeat timeout must be finite")
    if timeout < MIN_SCHEDULER_HEARTBEAT_TIMEOUT_SECONDS:
        raise ValueError(
            "heartbeat timeout must be at least "
            f"{MIN_SCHEDULER_HEARTBEAT_TIMEOUT_SECONDS:.0f}s: one worker update may wait "
            f"{WORKER_HEARTBEAT_RPC_TIMEOUT_SECONDS:.0f}s and retries every "
            f"{WORKER_HEARTBEAT_INTERVAL_SECONDS:.0f}s"
        )
    return timeout
