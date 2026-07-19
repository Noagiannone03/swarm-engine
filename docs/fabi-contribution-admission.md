# Fabi contribution admission

## Product contract

Fabi has no points, wallet, balance, or delayed credit. Consumption is a live
capability:

- the account has a worker **ready now**, allocated at least one real model
  layer, publishing measured KV capacity, and heartbeating within the scheduler
  timeout: it may start inference;
- otherwise it may not start inference;
- a request already admitted is allowed to finish;
- one eligible worker opens one concurrent request by default. This avoids an
  unbounded consumer fan-out from one minimal contribution without inventing a
  currency.

The IDE does not expose the credential. It asks the account-scoped
`GET /v1/contribution/status` endpoint and only reveals the prompt when both the
pipeline and the account are admitted.

## Why this design

BitTorrent's deployed choking algorithm reciprocates observed service, caps the
number of upload slots, and changes decisions slowly enough to avoid
"fibrillation". It also gives newcomers a bounded optimistic slot. Fabi keeps
the relevant properties: current contribution, bounded concurrency, and
heartbeat hysteresis. It does not copy piece-level accounting because pipeline
inference has no independently verifiable file pieces.

Petals discussed centralized "bloom points" for future priority, and AI Horde
uses non-tradable kudos. HyperSpace uses presence/work points and signed work
receipts. Those systems support delayed consumption or a public economy. Fabi's
rule is intentionally narrower: contribute to the selected model while using
it, so a ledger adds state and failure modes without improving the contract.

Primary references:

- BitTorrent protocol, choking and optimistic unchoking:
  <https://www.bittorrent.org/beps/bep_0003.html>
- Petals paper and public implementation: <https://github.com/bigscience-workshop/petals>
- AI Horde's non-tradable kudos model: <https://github.com/Haidra-Org/AI-Horde>
- HyperSpace node pulse/work-receipt model:
  <https://github.com/hyperspaceai/hyperspace-node>

## Scheduler-authoritative eligibility

The worker sends its 32-byte hexadecimal account credential through the
encrypted Lattica RPC. The RPC handler validates it, immediately hashes it with
SHA-256, and stores only that identifier on the scheduler's `Node`. Logs are
constructed from an allowlist of operational fields and cannot contain the
credential.

Admission derives eligibility from the current `NodeManager` snapshot. A node
must satisfy every condition:

1. lifecycle state `ACTIVE` (standby does not count);
2. runtime state `READY`;
3. a non-empty scheduler-owned layer allocation;
4. measured executor KV-token capacity;
5. heartbeat age within `Scheduler.heartbeat_timeout`;
6. the full model pipeline passes `Scheduler.serving_ready()`.

There is no contribution TTL store and no Redis dependency. Graceful leave
removes the node immediately; a crash loses eligibility at heartbeat expiry.
The gate is evaluated at admission, so a transient failure never truncates a
response merely to enforce policy.

## HTTP behavior

- `GET /v1/contribution/status`: always returns the account-scoped state and
  counters; it never returns an account id or credential.
- `POST /v1/chat/completions` without an eligible worker: HTTP 403,
  `code=contribution_required`.
- eligible contribution but globally incomplete pipeline: HTTP 503,
  `code=swarm_not_ready`.
- account concurrency already occupied: HTTP 429,
  `code=contribution_capacity_reached`.
- `FABI_GATE=off` keeps upstream/private deployments open. Fabi public
  schedulers must explicitly set `FABI_GATE=on`.

## Threat model and remaining identity work

The gate prevents a normal client, a direct scheduler caller, or a shared API
URL from consuming without the matching live worker. TLS and encrypted Lattica
protect the bearer credential in transit; the local file is created with mode
`0600` and its directory with `0700` where supported.

A malicious custom worker can lie in a heartbeat, but it cannot produce a
successful routed generation unless its allocated shard actually runs. A later
hardening phase may bind admission to successful activation checks or signed
work receipts. That is deliberately separate from this MVP and must not become
a fake proof-of-work patch.

Multiple user devices must share one Fabi account credential. Today this is an
internal file/env contract. The finished consumer onboarding should use an
account login or device-pairing flow to provision it without displaying or
copying the secret; changing the scheduler gate does not solve device identity.
