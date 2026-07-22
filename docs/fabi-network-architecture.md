# Fabi network transport architecture

## Decision

Fabi will replace Lattica on the centrally scheduled runtime path with a small,
owned transport built on Iroh 1.x. This is a transport replacement, not a
second competing RPC layer.

The decision is evidence-gated: the reusable native transport is first tested
between independent NATs in forced-relay and automatic modes. Migration of the
runtime begins only after integrity, throughput, cancellation and path
telemetry are qualified on macOS, Windows and Linux.

## Why

Lattica currently combines four concerns: RPC, DHT discovery, peer telemetry
and Bitswap block exchange. Its public RPC path rejects relay-only connections,
which makes a healthy worker unusable behind NATs where DCUtR cannot establish
a direct path. Making this one check permissive would still leave an unsuitable
Relay V2 data plane whose default circuit limits are designed for control
traffic, not multi-megabyte activation transfers.

Iroh provides the required product semantics in one maintained stack:

- mutually authenticated QUIC endpoints identified by Ed25519 keys;
- direct UDP hole punching with live relay-to-direct path upgrades;
- a reliable encrypted relay fallback when direct connectivity is impossible;
- self-hosted stateless relays with authentication and rate limiting;
- explicit selected-path and RTT telemetry.

Petals/Hivemind validates the policy of keeping relay-backed peers eligible and
penalising them by measured throughput instead of declaring them offline.
Tailscale validates the direct-first, relay-always-available operational model.

## Boundaries

The scheduler remains the source of truth for accounts, contribution,
membership, model allocation, epochs and routing. In scheduler mode it already
returns every worker's outbound peers, so a global DHT is not needed for the
critical path.

The native transport owns:

- persistent endpoint identity;
- relay and direct connectivity;
- authenticated, versioned and size-bounded byte streams;
- connection/path health and network measurements;
- deadlines, cancellation and backpressure.

Python owns protobuf schemas and RPC dispatch initially. The native boundary
now exposes byte-oriented unary and streaming operations through a stable-ABI
PyO3 extension. The service adapter uses explicit MessagePack, protobuf and raw
bytes codecs; it never accepts Python pickle from the network.

Bitswap-based weight refitting is optional in today's product runtime and will
not be silently mixed into the new RPC plane. It will get a separate
content-addressed transfer design if retained.

## Admission and routing policy

A peer is `reachable` when either a direct or relay path is healthy. Workers
publish separate reachable, direct and relayed peer sets plus the selected
path RTT. Direct peers are naturally preferred when their measured cost is
lower; relay peers remain eligible. Only loss of all usable paths makes a
worker unavailable. Continuous per-peer throughput measurement and the
Petals-style bandwidth penalty are still required before final route scoring;
the scheduler does not pretend that RTT alone is bandwidth telemetry.

The relay is a product dependency, not an emergency debug service. Production
deployment therefore requires authentication, bandwidth/connection quotas,
metrics, multiple regions and capacity alerts.

## Qualification gates

1. Persist identities without silent regeneration and reject corrupt keys.
2. Reject unsupported protocol versions and oversized or corrupted frames.
3. Transfer representative prefill activation sizes through a forced relay
   between two independent NATs.
4. In automatic mode, record whether the connection stays relayed or upgrades
   to direct without interrupting streams.
5. Exercise small decode messages, large prefill messages, timeouts, aborts,
   peer restart, relay restart and invalid relay credentials.
6. Build and run the same crate on macOS, Windows CUDA host and Linux relay.
7. Compare end-to-end generation TTFT and throughput with the qualified Lattica
   baseline before making Iroh the default runtime transport.

No result is considered qualified unless the observed path and command output
are captured in the IDE handoff.

## Qualified on 22 July 2026

- Official Iroh `v1.0.3` relay deployed on the lab VPS with TLS, bearer
  authentication, per-client limits, QUIC address discovery and local metrics.
- macOS to remote Mac mini across independent NATs: forced-relay integrity for
  three 64 MiB payloads; automatic mode remained relayed after Tailscale was
  disabled, proving usable fallback when hole punching cannot win.
- macOS to Windows RTX host across independent NATs: three 16 MiB payloads and
  100 small decode-like messages passed through the forced relay.
- Cancelling a 64 MiB stream reset only that QUIC stream; the same cached peer
  connection immediately carried a second successful 64 MiB request.
- Under saturated local stream backpressure, QUIC `STOP_SENDING` released the
  remote Python generator in about 22 ms; a unary RPC then reused the same
  relayed connection successfully.
- The Python extension passed unary and streaming RPC qualification through the
  relay: 4 MiB integrity, typed remote failures, deadlines, multi-chunk SSE,
  mid-stream failure, cancellation and post-cancel connection reuse.
- With only endpoint ID plus relay URL at dial time, two automatic-mode Python
  endpoints discovered a local direct candidate and upgraded to a selected
  direct path (`10.0.1.54`, about 0.22 ms RTT) while retaining relay fallback.
- The real scheduler `node_update` handler passed through Iroh with reachable
  and relayed topology telemetry. The full Python suite is green at 417 passed
  and 7 skipped (the additional skip is the opt-in live-relay regression);
  Rust unit tests and strict Clippy are green.

## Staged runtime activation

Lattica remains the default rollback path until a full model generation is
qualified. The central scheduler/worker path can be selected explicitly:

```shell
export FABI_NETWORK_TRANSPORT=iroh
export FABI_RELAY_URL=https://relay.example.com
export FABI_RELAY_TOKEN_FILE=/path/to/protected/token  # bootstrap resolves this
```

The current process interface reads either `FABI_RELAY_TOKEN` or a protected
`FABI_RELAY_TOKEN_FILE`; release bootstrap must resolve that secret from the
account credential without placing it on the command line. `FABI_FORCE_RELAY=1`
exists only for qualification. Scheduler and worker identities default to
`~/.fabi/network/{role}.key`, or can be set with
`FABI_NETWORK_IDENTITY_PATH`.

Before making Iroh the default, the remaining gates are a complete two-worker
model load and generation (prefill, decode, SSE and abort), Windows/macOS wheel
packaging, path-aware bandwidth measurement, relay failover and production
credential bootstrap. Weight refit remains intentionally disabled on Iroh
until a separate content-addressed plane is qualified.
