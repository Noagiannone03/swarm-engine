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
    SpeculativeCoordinator,
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


# --- SpeculativeCoordinator: correctness oracle against a deterministic target


class _ListProposer:
    """Proposes the next tokens from a fixed script (for testing acceptance)."""

    def __init__(self, script):
        self._script = list(script)

    def propose(self, tokens, k):
        # propose what *would* come next per the script, given how far we are
        i = len(tokens)
        return self._script[i : i + k]


def _make_verify(true_seq):
    """A deterministic greedy target: after a committed prefix of length L, its
    greedy continuation is true_seq[L], true_seq[L+1], ... — regardless of the
    drafts (greedy is a function of the correct prefix). Returns len(drafts)+1
    tokens. The coordinator must reconstruct true_seq exactly."""
    state = {"out_len": 1}  # first_token already committed

    def verify(cur, drafts):
        L = state["out_len"]
        r = [true_seq[L + j] if (L + j) < len(true_seq) else -1 for j in range(len(drafts) + 1)]
        return r

    return verify, state


def _run(proposer, true_seq, k=6, max_new=64, eos=None):
    cfg = SpeculativeConfig(enabled=True, method="x", k=k, k_min=1, k_max=12,
                            adaptive=True, ngram_min=2, draft_model=None)
    coord = SpeculativeCoordinator(cfg, proposer)
    verify, state = _make_verify(true_seq)
    # keep the mock target's notion of committed length in sync via on_commit
    def on_commit(committed):
        state["out_len"] += len(committed)
    # first_token is true_seq[0]; on_commit fires for it too, so start out_len at 0
    state["out_len"] = 0
    res = coord.generate(true_seq[0], verify, eos_token_id=eos,
                         max_new_tokens=max_new, on_commit=on_commit)
    return res


def test_coordinator_perfect_draft_reconstructs_target():
    true_seq = list(range(100, 140))  # arbitrary "greedy" output
    # perfect proposer = proposes exactly the true continuation
    res = _run(_ListProposer(true_seq), true_seq, k=6, max_new=len(true_seq))
    assert res.tokens[: len(true_seq)] == true_seq
    assert res.mean_accept > 4  # near-full acceptance with a perfect draft


def test_coordinator_bad_draft_still_correct():
    true_seq = list(range(200, 240))
    # always-wrong proposer -> n=0 every round, but output still == target greedy
    class Bad:
        def propose(self, tokens, k):
            return [999999] * k
    res = _run(Bad(), true_seq, k=6, max_new=len(true_seq))
    assert res.tokens[: len(true_seq)] == true_seq  # correctness independent of draft
    assert res.mean_accept == 0  # nothing accepted, one correction per round


def test_coordinator_ngram_proposer_correct_and_accelerates_on_repeats():
    # a repetitive "code-like" sequence the n-gram proposer can exploit
    true_seq = ([1, 2, 3, 4, 5] * 8)
    res = _run(NgramProposer(ngram_min=2), true_seq, k=6, max_new=len(true_seq))
    assert res.tokens[: len(true_seq)] == true_seq
    assert res.mean_accept > 1  # repeats -> the n-gram lands multiple tokens/round


def test_coordinator_stops_at_eos():
    true_seq = [10, 11, 12, 13, 99, 14, 15]  # 99 = eos
    res = _run(_ListProposer(true_seq), true_seq, k=6, max_new=64, eos=99)
    assert res.tokens[-1] == 99
    assert 14 not in res.tokens  # nothing past eos
