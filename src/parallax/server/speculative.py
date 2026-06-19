"""Speculative-decoding core for the swarm pipeline (WAN-adapted).

Autoregressive decode generates one token per step, and in our system EVERY
token traverses the whole pipeline over a P2P/WAN relay — so wall-clock is
dominated by round-trips, not GPU compute (≈1 token / pipeline-RTT). Speculative
decoding amortises that: a cheap *drafter* proposes K candidate tokens, the real
(split) target VERIFIES all K in ONE pipeline traversal, and we keep the longest
prefix the target agrees with. Net: ~(accepted+1) tokens per round-trip instead
of 1 — the lever that takes a WAN pipeline from ~5 to ~30 tok/s
(cf. Shard, arXiv 2602.16760; Leviathan et al. 2022, arXiv 2211.17192).

This module is the **backend-agnostic, pure-logic core** — no torch, no GPU, no
network — so it is fully unit-testable on its own:

  * ``NgramProposer``    — zero-model drafting: copy the continuation of the
                           longest recent suffix that already occurred in the
                           context (a.k.a. prompt-lookup / n-gram). Ideal for
                           code, which is highly repetitive (identifiers,
                           brackets, boilerplate) and needs no draft model.
  * ``greedy_accept``    — Shard's acceptance rule: longest prefix where the
                           draft equals the target's greedy token, plus the one
                           target correction. With greedy verification the output
                           is token-for-token identical to plain greedy decode —
                           that identity is the correctness oracle.
  * ``AdaptiveK``        — tune K live from the running acceptance rate (EMA),
                           so a high-latency link drafts deeper to amortise more.

The GPU/pipeline parts (running K+1 tokens through the split target in one pass,
and rolling back paged-KV for rejected positions) live in the executor and are
gated behind ``PARALLAX_SPECULATIVE`` — see speculative_config().
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


# --- Configuration -----------------------------------------------------------


@dataclass(frozen=True)
class SpeculativeConfig:
    """Resolved speculative-decoding settings (see speculative_config())."""

    enabled: bool
    method: str  # "ngram" (no draft model) | "draft" (small same-family model)
    k: int  # base number of tokens to propose per round
    k_min: int
    k_max: int
    adaptive: bool
    ngram_min: int  # shortest suffix to match for n-gram lookup
    draft_model: Optional[str]  # HF id of the draft model when method == "draft"


def _env_int(key: str, default: int, *, minimum: int, maximum: int) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return max(minimum, min(maximum, int(raw)))
    except ValueError:
        return default


def speculative_config() -> SpeculativeConfig:
    """Read speculative settings from the environment. OFF by default — the
    normal one-token-per-traversal decode path is completely unaffected unless
    ``PARALLAX_SPECULATIVE`` is explicitly enabled.

    Env:
      PARALLAX_SPECULATIVE          1|true|yes to enable (default off)
      PARALLAX_SPEC_METHOD          "ngram" (default) | "draft"
      PARALLAX_SPEC_K               base proposal length (default 5)
      PARALLAX_SPEC_K_MIN/_K_MAX    adaptive bounds (default 1 / 12)
      PARALLAX_SPEC_ADAPTIVE        1 to tune K live (default on)
      PARALLAX_SPEC_NGRAM_MIN       min suffix length for n-gram match (default 2)
      PARALLAX_SPEC_DRAFT_MODEL     HF id of the draft model (method "draft")
    """
    enabled = os.environ.get("PARALLAX_SPECULATIVE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    method = os.environ.get("PARALLAX_SPEC_METHOD", "ngram").strip().lower()
    if method not in ("ngram", "draft"):
        method = "ngram"
    k = _env_int("PARALLAX_SPEC_K", 5, minimum=1, maximum=64)
    k_min = _env_int("PARALLAX_SPEC_K_MIN", 1, minimum=1, maximum=64)
    k_max = _env_int("PARALLAX_SPEC_K_MAX", 12, minimum=k_min, maximum=64)
    adaptive = os.environ.get("PARALLAX_SPEC_ADAPTIVE", "1").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    ngram_min = _env_int("PARALLAX_SPEC_NGRAM_MIN", 2, minimum=1, maximum=16)
    draft_model = os.environ.get("PARALLAX_SPEC_DRAFT_MODEL", "").strip() or None
    return SpeculativeConfig(
        enabled=enabled,
        method=method,
        k=max(k_min, min(k_max, k)),
        k_min=k_min,
        k_max=k_max,
        adaptive=adaptive,
        ngram_min=ngram_min,
        draft_model=draft_model,
    )


# --- Acceptance (Shard's greedy rule) ----------------------------------------


def greedy_accept(
    drafts: Sequence[int], target: Sequence[int]
) -> Tuple[List[int], int]:
    """Accept the longest prefix where the draft matches the target's greedy
    token, then append the target's correction at the first divergence.

    ``drafts``  : the K proposed tokens (positions p+1 .. p+K).
    ``target``  : the target's greedy argmax for positions p+1 .. p+K+1 (length
                  K+1) — i.e. one prediction per draft slot plus the bonus slot
                  that follows the last draft.

    Returns ``(committed, n_accepted)`` where ``committed`` is
    ``drafts[:n] + [target[n]]`` (n accepted + 1 correction/bonus). Because every
    committed token is exactly the target's greedy choice, the resulting stream
    is identical to plain greedy decode — the correctness oracle.

    With all K accepted, the correction is the bonus token ``target[K]``, so a
    fully-accepted round commits K+1 tokens in a single traversal.
    """
    k = len(drafts)
    if len(target) < k + 1:
        raise ValueError(
            f"target must have len(drafts)+1 = {k + 1} entries, got {len(target)}"
        )
    n = 0
    while n < k and drafts[n] == target[n]:
        n += 1
    committed = list(drafts[:n]) + [target[n]]
    return committed, n


# --- Adaptive K controller ---------------------------------------------------


class AdaptiveK:
    """Tune the proposal length K from the running acceptance rate.

    Aim a couple tokens beyond the EMA of accepted-per-round: if the draft is
    landing ~4 tokens/round, propose ~6 so there's headroom to accept more;
    if it's landing ~1, pull back so a rejected round wastes less work. This is
    Shard's heuristic (specdec.py): ``K = clip(round(ema)+2, k_min, k_max)``.
    """

    def __init__(self, cfg: SpeculativeConfig) -> None:
        self._cfg = cfg
        self._ema = float(cfg.k)
        self.k = cfg.k

    def update(self, n_accepted: int, alpha: float = 0.3) -> int:
        """Feed the number accepted this round; returns the next round's K."""
        self._ema = (1.0 - alpha) * self._ema + alpha * float(n_accepted)
        if self._cfg.adaptive:
            self.k = max(self._cfg.k_min, min(self._cfg.k_max, round(self._ema) + 2))
        return self.k


# --- N-gram / prompt-lookup proposer (zero draft model) ----------------------


class NgramProposer:
    """Propose the next tokens by copying the continuation of the longest recent
    suffix that already appeared earlier in the sequence (prompt-lookup / n-gram
    speculative decoding).

    Mechanism: take the last ``L`` tokens (the suffix), find the most recent
    earlier occurrence of that same L-gram in the history, and propose the tokens
    that followed it. Try the longest suffix first (down to ``ngram_min``) for the
    most specific match. Needs NO model and NO matching tokenizer — it operates on
    the target's own token ids — which is why it's the safe, free first method,
    and unusually effective on code (repeated identifiers, closing brackets,
    duplicated lines).

    Returns up to ``k`` proposed token ids, or ``[]`` when nothing matches (the
    caller then falls back to a single normal decode step that round).
    """

    def __init__(self, ngram_min: int = 2, max_suffix: int = 16) -> None:
        self.ngram_min = max(1, ngram_min)
        self.max_suffix = max(self.ngram_min, max_suffix)

    def propose(self, tokens: Sequence[int], k: int) -> List[int]:
        n = len(tokens)
        if k <= 0 or n < self.ngram_min + 1:
            return []
        # Longest suffix first → most specific continuation.
        max_l = min(self.max_suffix, n - 1)
        for l in range(max_l, self.ngram_min - 1, -1):
            suffix = tokens[n - l :]
            # Search for the most RECENT earlier occurrence of `suffix` (excluding
            # the trailing suffix itself), scanning right-to-left.
            for start in range(n - l - 1, -1, -1):
                if self._matches(tokens, start, suffix):
                    cont_start = start + l
                    cont = tokens[cont_start : cont_start + k]
                    if cont:
                        return list(cont)
        return []

    @staticmethod
    def _matches(tokens: Sequence[int], start: int, suffix: Sequence[int]) -> bool:
        for i, t in enumerate(suffix):
            if tokens[start + i] != t:
                return False
        return True
