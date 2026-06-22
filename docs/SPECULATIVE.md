# Speculative decoding over the swarm pipeline (WAN-adapted)

## Why
Autoregressive decode emits one token per step, and in this engine **every token
traverses the whole pipeline over a P2P/WAN relay**. So throughput is bounded by
round-trips, not GPU compute: a multi-hop home-internet pipeline does ~1–5 tok/s
no matter how fast the GPUs are. Speculative decoding is the lever that fixes
this (Shard reports ~5 → ~30 tok/s): a cheap **drafter** proposes K tokens, the
real split **target verifies all K in ONE traversal**, we keep the longest prefix
it agrees with — so we commit ~(accepted+1) tokens per round-trip instead of 1.

With **greedy** verification the output is token-for-token identical to plain
greedy decode, so the drafter can never hurt quality — only the *speedup* varies
with how often it guesses right. Code is unusually predictable (brackets,
indentation, keywords, repeated identifiers) → high acceptance → big win, and the
**n-gram** drafter needs no model and no matching tokenizer at all.

Refs: Shard (arXiv 2602.16760, repo `leyten/shard` `phase0/specdec.py`),
Leviathan et al. 2022 (arXiv 2211.17192).

## Design: where the pieces run
```
client/IDE ── prompt ──▶ entry node (layers 0..a) ──▶ … ──▶ last node (… ..L) ──▶ sample
                         │ drafter proposes K                         │ verify K+1 in one pass,
                         │ (n-gram from context, or a small           │ argmax + greedy-accept,
                         │  same-family draft model)                  │ return committed tokens
```
The drafter sits at the **entry** (or on the client machine — the IDE node is
already a worker+consumer, so a tiny local draft model is natural and removes the
client→entry hop too). Verification is a normal forward of **K+1 tokens** through
every stage in **one** traversal. The last stage samples greedily for each of the
K+1 positions and applies the acceptance rule.

## Critical architecture fact (measured)

**Speculative decoding only helps the MULTI-NODE / WAN case.** A single-GPU
benchmark (`research/spec_bench.py`, Qwen3-8B target + Qwen3-0.6B draft, code
prompts) measured:

```
mean accepted / round = 2.68   → ~3.7 tokens per verify traversal
local speedup          = 0.66x  (spec 17 vs greedy 26 tok/s)   ← SLOWER on 1 GPU
```

On one fast GPU spec-decode is *slower*: verifying K+1 tokens + drafting costs
more compute than one greedy step, and there is no network latency to amortise.
The win appears only when each token otherwise costs a full pipeline round-trip
over the WAN — then ~3.7 accepted tokens per round-trip ≈ a 3.7× throughput win.
So this feature must be wired into the **distributed** path, and is pointless to
enable on a single-node pipeline. (The 2.68 acceptance also confirms the premise
holds on real Qwen3 + code; a better draft or n-gram on repetitive code lifts it.)

Prerequisite: it needs a true **multi-node split pipeline** (layers spread across
nodes, activations crossing the relay). On NAT-isolated pods that don't share a
direct route, the inter-stage transport must go through the lattica relay — that
path must work before spec-decode matters.

## Status — staged so the working path is never at risk

**Phase A — core brain (DONE, shipped, tested, gated OFF).**
`src/parallax/server/speculative.py` + `tests/test_speculative.py`:
- `NgramProposer` — prompt-lookup drafting (copy the continuation of the longest
  recurring suffix). Zero model, zero tokenizer constraint.
- `DraftModelProposer` — small same-family draft model held on the entry node,
  self-managing its KV cache against the committed sequence (Shard's draft
  bookkeeping). Lazy torch import (module usable without a GPU).
- `greedy_accept(drafts, target)` — Shard's rule: longest matching prefix + one
  correction; output is the target's greedy decode (the correctness oracle).
- `AdaptiveK` — tune K live from the running acceptance EMA (`round(ema)+2`).
- `SpeculativeCoordinator` — the full Shard `generate()` loop, **backend-agnostic
  via a `verify_fn(cur, drafts) -> List[int]` callback**. Validated against a
  deterministic mock target: output reconstructs the target exactly with a
  perfect draft (accept≈K), with a bad draft (accept=0), and with n-gram on
  repeats (accept≈2) — correctness is INDEPENDENT of draft quality. Stops at EOS.
- `speculative_config()` — all settings, **OFF by default** (`PARALLAX_SPECULATIVE`).

The brain is complete and the seam is `verify_fn`: it runs `[cur] + drafts`
through the target in one traversal and returns the greedy token per position.
Single-node verify is a local K+1 forward; distributed verify is one pipeline
traversal. `drafts == []` (an n-gram miss) degenerates to a normal 1-token decode.

**Phase B — wire/protocol (GPU-validated).** Extend the `forward` message so a
verify carries K+1 candidate tokens and the last stage returns K+1 argmax results
+ the accepted length. Strictly additive; only used when the flag is on.

**Phase C — distributed `verify_fn` in the executor (GPU + multi-node-validated).**
Run K+1 tokens for a sequence through the pipeline in one traversal (right causal
mask), last stage produces per-position logits, `greedy_accept`, commit. Needs a
working multi-node split pipeline first (see the architecture fact above).

**Phase D — paged-KV rollback (GPU-validated).** Free the KV of rejected
speculative positions on each node. Use Shard's **lazy crop**: local caches crop
immediately; downstream stages' crop is piggybacked on the next verify message
(no extra WAN round-trip). The subtle, hardware-specific part (vLLM paged KV);
validate on real GPUs + multi-node before enabling the flag anywhere.

## Validation harness
`research/spec_bench.py` — single-GPU spec-decode bench (draft + whole target,
`greedy_accept`), measures acceptance / speedup / greedy-identity on code prompts.
This is how the acceptance numbers above were obtained on an A40.

## Tunables (env, all no-ops unless `PARALLAX_SPECULATIVE=1`)
| var | default | meaning |
|---|---|---|
| `PARALLAX_SPECULATIVE` | off | master switch |
| `PARALLAX_SPEC_METHOD` | `ngram` | `ngram` (no model) or `draft` (small same-family model) |
| `PARALLAX_SPEC_K` | 5 | base proposal length |
| `PARALLAX_SPEC_K_MIN` / `_K_MAX` | 1 / 12 | adaptive bounds |
| `PARALLAX_SPEC_ADAPTIVE` | on | tune K from acceptance rate |
| `PARALLAX_SPEC_NGRAM_MIN` | 2 | shortest suffix to match for n-gram |
| `PARALLAX_SPEC_DRAFT_MODEL` | — | HF id of the draft model (method `draft`) |
