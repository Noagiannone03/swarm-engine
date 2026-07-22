"""Structured failures for scheduler-negotiated runtime memory contracts."""

from __future__ import annotations


class MemoryContractError(RuntimeError):
    """The initialized backend cannot materialize the negotiated KV tier.

    This is deliberately distinct from an OOM or an arbitrary executor crash.
    The worker may report its measured limit to the scheduler, which can perform
    one fenced downward replan without retrying an unchanged contract forever.
    """

    def __init__(
        self,
        *,
        backend: str,
        requested_tokens: int,
        supported_tokens: int,
        detail: str,
    ) -> None:
        self.backend = str(backend)
        self.requested_tokens = max(0, int(requested_tokens))
        self.supported_tokens = max(0, int(supported_tokens))
        self.detail = str(detail)
        super().__init__(
            f"{self.backend} cannot satisfy the {self.requested_tokens}-token "
            f"KV contract (measured limit: {self.supported_tokens} tokens): {self.detail}"
        )

    def as_report(self, *, allocation_epoch: int | None) -> dict:
        """Return the bounded, JSON-safe report sent in worker heartbeats."""

        return {
            "kind": "kv_materialization",
            "backend": self.backend,
            "allocation_epoch": (
                None if allocation_epoch is None else int(allocation_epoch)
            ),
            "requested_tokens": self.requested_tokens,
            "supported_tokens": self.supported_tokens,
        }
