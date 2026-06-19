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

## Status — staged so the working path is never at risk

**Phase A — core logic (DONE, shipped, unit-tested, gated OFF).**
`src/parallax/server/speculative.py` + `tests/test_speculative.py`:
- `NgramProposer` — prompt-lookup drafting (copy the continuation of the longest
  recurring suffix). Zero model, zero tokenizer constraint.
- `greedy_accept(drafts, target)` — Shard's rule: longest matching prefix + one
  correction; output identical to greedy decode.
- `AdaptiveK` — tune K live from the running acceptance EMA (`round(ema)+2`).
- `speculative_config()` — all settings, **OFF by default** (`PARALLAX_SPECULATIVE`).

**Phase B — wire/protocol (next, low-risk).** Extend the `forward` message so a
verify carries K+1 candidate tokens and the last stage returns K+1 argmax results
+ the accepted length. Strictly additive; only used when the flag is on.

**Phase C — K-in-one-pass verify in the executor (GPU-validated).** A spec-verify
forward that runs K+1 tokens for a sequence with the right causal mask, produces
per-position logits, runs `greedy_accept`, commits. Parallel to the normal decode
path; selected only when the flag is on.

**Phase D — paged-KV rollback (GPU-validated).** Free the KV blocks of rejected
speculative positions on each node. Use Shard's **lazy crop**: the head/local
caches crop immediately; the downstream stages' crop is piggybacked on the next
verify message (no extra WAN round-trip). This is the subtle, hardware-specific
part (vLLM paged KV) and must be validated on real GPUs + multi-node before the
flag is enabled in any deployment.

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
