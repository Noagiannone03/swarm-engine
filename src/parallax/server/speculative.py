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


class DraftModelProposer:
    """Propose K tokens with a small SAME-FAMILY draft model held locally.

    This is the higher-acceptance option (vs n-gram) for non-repetitive text: a
    tiny model (e.g. Qwen3-0.6B for a Qwen3 target) drafts the next K tokens
    autoregressively from its own KV cache. It MUST share the target's tokenizer
    so its token ids are directly comparable (the cross-tokenizer case needs UAG
    and is out of scope here). torch/transformers are imported lazily so this
    module stays importable (and the n-gram path usable) on hosts without them.

    The proposer self-manages its KV cache against the committed sequence the
    coordinator passes each round: it (1) crops its cache back to the committed
    length — dropping the speculative tokens rejected last round — then (2) feeds
    any newly-committed tokens, then (3) drafts K new tokens. This is exactly
    Shard's draft bookkeeping (phase0/specdec.py), kept on the entry node so it
    pays no WAN cost. GPU-runtime; the propose/rollback logic is unit-tested via
    the coordinator with mock proposers, the real model path is validated on GPU.
    """

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device
        self._model = None
        self._cache = None
        self._committed = 0  # tokens already folded into the draft cache

    def _ensure_loaded(self):
        if self._model is None:
            import torch  # noqa: F401
            from transformers import AutoModelForCausalLM

            self._model = (
                AutoModelForCausalLM.from_pretrained(self.model_id)
                .to(self.device)
                .eval()
            )

    def reset(self, prompt_ids: Sequence[int]) -> None:
        """Prefill the draft cache on the prompt (call once before generation)."""
        import torch
        from transformers import DynamicCache

        self._ensure_loaded()
        self._cache = DynamicCache()
        with torch.no_grad():
            self._model(
                input_ids=torch.tensor([list(prompt_ids)], device=self.device),
                past_key_values=self._cache,
                use_cache=True,
            )
        self._committed = len(prompt_ids)

    def propose(self, tokens: Sequence[int], k: int) -> List[int]:
        import torch

        if k <= 0:
            return []
        self._ensure_loaded()
        if self._cache is None:
            self.reset(tokens[:-1] if len(tokens) > 1 else tokens)
        with torch.no_grad():
            # 1. roll the cache back to the confirmed length (drop last round's
            #    rejected speculation), then 2. feed newly-committed tokens.
            target_len = max(0, len(tokens) - 1)  # cache holds [0 .. cur-1]
            if target_len < self._committed:
                self._cache.crop(target_len)
                self._committed = target_len
            if self._committed < len(tokens) - 0:
                new = list(tokens[self._committed :])
                if new:
                    self._model(
                        input_ids=torch.tensor([new], device=self.device),
                        past_key_values=self._cache,
                        use_cache=True,
                    )
                    self._committed = len(tokens)
            # 3. draft K tokens autoregressively from the last committed token.
            drafts: List[int] = []
            dtok = tokens[-1]
            for _ in range(k):
                logits = self._model(
                    input_ids=torch.tensor([[dtok]], device=self.device),
                    past_key_values=self._cache,
                    use_cache=True,
                ).logits
                dtok = int(logits[0, -1].argmax())
                drafts.append(dtok)
            # the K drafted tokens are speculatively in the cache; they'll be
            # cropped next round down to whatever the target actually accepted.
            self._committed = len(tokens) + k
        return drafts


# --- The coordinator: Shard's speculative loop, backend-agnostic -------------


@dataclass
class SpecResult:
    """Outcome of a speculative generation run (for stats/telemetry)."""

    tokens: List[int]
    rounds: int
    accepted_total: int

    @property
    def mean_accept(self) -> float:
        return self.accepted_total / max(self.rounds, 1)

    @property
    def tokens_per_round(self) -> float:
        # accepted + the 1 correction/bonus committed each round
        return (self.accepted_total + self.rounds) / max(self.rounds, 1)


# A verifier runs ``[cur] + drafts`` through the (local or distributed) TARGET in
# ONE traversal and returns the target's greedy token for each of the
# ``len(drafts)+1`` positions. drafts == [] ⇒ a plain single-token decode.
VerifyFn = "Callable[[int, List[int]], List[int]]"

# A proposer maps the committed token sequence + a budget K to up to K guesses
# (NgramProposer / DraftModelProposer both satisfy ``propose(tokens, k) -> list``).


class SpeculativeCoordinator:
    """Drives speculative decoding on the entry side — a faithful port of Shard's
    ``phase0/specdec.py::generate`` loop, with the target VERIFY abstracted behind
    a callback so the same brain works whether the target runs locally or is split
    across the pipeline:

      per round (cache invariant: ``cur`` is the committed token at the current
      position, not yet folded into the target cache):
        1. proposer proposes K tokens d_1..d_K from the committed context;
        2. verify_fn runs [cur, d_1..d_K] through the target in ONE traversal →
           the target's greedy token r_1..r_{K+1} for each position;
        3. greedy_accept keeps the longest prefix with d_j == r_j, then commits
           the correction r_{n+1} — so the stream is exactly the target's greedy
           decode (the correctness oracle);
        4. adaptive K aims a couple beyond the running acceptance.

    A no-proposal round (K==0, e.g. an n-gram miss) degenerates to a single normal
    decode step (verify_fn(cur, []) → one token), so the loop is always correct
    and never stalls. Stateful proposers (a draft model with its own KV cache)
    self-manage rollback by keying off the committed length each call.
    """

    def __init__(self, cfg: SpeculativeConfig, proposer) -> None:
        self._cfg = cfg
        self._proposer = proposer
        self._adaptive = AdaptiveK(cfg)

    def generate(
        self,
        first_token: int,
        verify_fn,
        *,
        eos_token_id: Optional[int],
        max_new_tokens: int,
        on_commit=None,
    ) -> SpecResult:
        """Generate from ``first_token`` (the target's token for the position
        right after the prompt). ``verify_fn(cur, drafts) -> List[int]`` returns
        ``len(drafts)+1`` greedy tokens. ``on_commit(list)`` is called with each
        round's committed tokens (for streaming)."""
        out: List[int] = [first_token]
        if on_commit is not None:
            on_commit([first_token])
        cur = first_token
        rounds = accepted_total = 0
        k = self._adaptive.k
        while len(out) < max_new_tokens and cur != eos_token_id:
            drafts = list(self._proposer.propose(out, k))
            target = verify_fn(cur, drafts)
            committed, n = greedy_accept(drafts, target)
            # Trim to the generation budget and stop cleanly at EOS.
            stop = False
            if eos_token_id is not None and eos_token_id in committed:
                committed = committed[: committed.index(eos_token_id) + 1]
                stop = True
            if len(out) + len(committed) > max_new_tokens:
                committed = committed[: max_new_tokens - len(out)]
                stop = True
            out.extend(committed)
            rounds += 1
            accepted_total += n
            if committed:
                cur = committed[-1]
                if on_commit is not None:
                    on_commit(committed)
            k = self._adaptive.update(n)
            if stop:
                break
        return SpecResult(tokens=out, rounds=rounds, accepted_total=accepted_total)
