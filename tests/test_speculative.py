"""Unit tests for the backend-agnostic speculative-decoding core.

These cover the pure logic (acceptance rule, n-gram drafting, adaptive K,
config gating) with no torch/GPU/network — the GPU/pipeline verify + paged-KV
rollback are integration-tested separately on hardware.
"""

import os

import pytest

from parallax.server.speculative import (
    AdaptiveK,
    NgramProposer,
    SpeculativeConfig,
    greedy_accept,
    speculative_config,
)


def _cfg(**kw):
    base = dict(
        enabled=True,
        method="ngram",
        k=5,
        k_min=1,
        k_max=12,
        adaptive=True,
        ngram_min=2,
        draft_model=None,
    )
    base.update(kw)
    return SpeculativeConfig(**base)


def test_greedy_accept_all():
    # every draft matches the target's greedy token -> commit K + bonus
    committed, n = greedy_accept([10, 11, 12], [10, 11, 12, 99])
    assert committed == [10, 11, 12, 99]
    assert n == 3


def test_greedy_accept_partial():
    # first divergence at index 1 -> keep 1 + the target's correction
    committed, n = greedy_accept([10, 11, 12], [10, 77, 12, 99])
    assert committed == [10, 77]
    assert n == 1


def test_greedy_accept_none():
    # nothing accepted -> a single correction token, identical to plain decode
    committed, n = greedy_accept([5, 6], [7, 6, 8])
    assert committed == [7]
    assert n == 0


def test_greedy_accept_validates_target_length():
    with pytest.raises(ValueError):
        greedy_accept([1, 2, 3], [1, 2, 3])  # needs len drafts + 1


def test_ngram_proposes_repeated_continuation():
    ng = NgramProposer(ngram_min=2)
    # the suffix [1,2,3] recurs; earlier it was followed by 4,5,6 -> propose those
    toks = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3]
    assert ng.propose(toks, k=3) == [4, 5, 6]


def test_ngram_no_match_returns_empty():
    ng = NgramProposer(ngram_min=2)
    assert ng.propose([1, 2, 3, 4, 5], k=3) == []


def test_ngram_respects_k():
    ng = NgramProposer(ngram_min=2)
    toks = [1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3]
    assert ng.propose(toks, k=1) == [4]


def test_ngram_too_short():
    ng = NgramProposer(ngram_min=2)
    assert ng.propose([1], k=3) == []


def test_adaptive_k_climbs_and_falls():
    ak = AdaptiveK(_cfg(k=5))
    for _ in range(30):
        ak.update(8)
    assert ak.k >= 9  # converges toward ema(8)+2, clipped at k_max
    for _ in range(30):
        ak.update(0)
    assert ak.k <= 3  # pulls back toward k_min when acceptance collapses


def test_adaptive_k_disabled_is_constant():
    ak = AdaptiveK(_cfg(k=5, adaptive=False))
    for _ in range(10):
        ak.update(0)
    assert ak.k == 5


def test_config_default_off(monkeypatch):
    for v in list(os.environ):
        if v.startswith("PARALLAX_SPEC"):
            monkeypatch.delenv(v, raising=False)
    monkeypatch.delenv("PARALLAX_SPECULATIVE", raising=False)
    assert speculative_config().enabled is False


def test_config_opt_in(monkeypatch):
    monkeypatch.setenv("PARALLAX_SPECULATIVE", "1")
    monkeypatch.setenv("PARALLAX_SPEC_K", "6")
    cfg = speculative_config()
    assert cfg.enabled is True
    assert cfg.method == "ngram"
    assert cfg.k == 6
