# Fabi local Request Agent frontend

`fabi-request-agent` is the loopback OpenAI/OpenCode endpoint for protocol v3.
It does not allocate model layers and it does not proxy inference through the
VPS. The local process reads the signed model catalogue, plans a complete route
from the DHT, obtains an account-scoped authority capability, reserves worker
KV, then calls the authenticated frontend worker directly over Iroh.

## Product flow

1. The IDE selects one immutable `model_swarm_id`.
2. The Request Agent verifies that model bundle with the pinned TUF root.
3. It downloads only architecture and tokenizer artifacts, never model
   weights, and verifies every file against the TUF-authenticated artifact
   index.
4. The local tokenizer produces a provisional exact context budget.
5. The same V3 planner used for admission computes the current largest route
   context from live DHT leases. It uses binary search over exact KV geometry;
   no 16k/32k tier or RAM estimate is configured.
6. The route head tokenizes the final chat again with the actual maintained
   vLLM frontend. Any disagreement releases the route and replans before
   inference.
7. The Request Agent sends `/v1/chat/completions` directly to that authenticated
   route head. Existing route fences protect tokenize, chat, abort and every
   pipeline hop.
8. Route and authority keepalives run on their independent control thread until
   the HTTP response ends or is aborted.

Long prefills deliberately have no wall-clock read deadline once the local
frontend request and Iroh RPC stream are established. Connection setup remains
bounded, while termination comes from an explicit engine result, client abort,
QUIC failure, route fencing or reservation-lease expiry. The OpenAI stream emits
WHATWG SSE comments while no model event is available; these comments keep HTTP
intermediaries alive but are never journalled as tokens or interpreted as model
progress. If the worker RPC itself fails, the Request Agent either performs the
qualified cold replan or emits one typed SSE error followed by `[DONE]`.

Worker lease expiry is active cleanup, not only accounting: each worker consumes
newly expired V3 reservations and injects a local fenced abort into its executor.
The inherited Parallax ten-minute request timer therefore remains a legacy-only
safety net and cannot race a valid renewable V3 lease during a slow relay
transfer.

The OpenAI endpoint is loopback-only. `/v1/models`,
`/v1/request-agent/status`, `/v1/request-agent/events` and
`/v1/chat/completions` require the same account Bearer credential that the
Request Agent uses for contribution admission. `/health` exposes only `ready`
or `waiting`.

`/v1/request-agent/events` is a bounded WHATWG SSE control stream for the IDE.
Its phases come from actual planner, authority, reservation and durable-journal
transitions: `planning`, `authorizing`, `reserving`, `prefilling`, `decoding`,
`recovering`, `replaying` and terminal states. Every event has a monotonic
`id`; reconnecting clients send `Last-Event-ID`, and receive either retained
events or a current-state `reset` snapshot after compaction. Quiet-stream
comments are transport keepalives only and never classify a worker as failed.

## Environment

The process consumes the normal native network variables plus:

- `FABI_REQUEST_AGENT_MODEL_SWARM_ID`: lowercase immutable SHA-256 selected by
  the IDE;
- `FABI_REQUEST_AGENT_AUTHORITY_URL`: HTTPS authority for permits and
  capabilities;
- `FABI_ACCOUNT_TOKEN` or owner-only `FABI_ACCOUNT_TOKEN_FILE`;
- `FABI_MODEL_REGISTRY_METADATA_URL`, `FABI_MODEL_REGISTRY_TARGETS_URL` and
  pinned `FABI_MODEL_REGISTRY_ROOT`;
- `FABI_CATALOG_DHT_MODE=client` and the signed catalogue bootstrap/relay
  configuration;
- optional `FABI_REQUEST_AGENT_STATE_DIR`;
- optional `FABI_USE_HFCACHE=true` for an offline, already verified cache.

Example:

```console
fabi-request-agent --host 127.0.0.1 --port 7778
```

The CLI rejects a non-loopback bind.

## Exact recovery without an immobilized backup

Deterministic streaming requests use `replan_cold`; they never reserve a second
complete pipeline:

1. before prefill, the authenticated route-head tokenization is committed to a
   local SQLite journal;
2. every output-token ID is committed in a `BEGIN IMMEDIATE` transaction
   before the corresponding SSE event is published;
3. the journal uses SQLite WAL with `synchronous=FULL`, owner-only file
   permissions, bounded prompt/output counts and checksum verification;
4. when a route disappears, its workers are excluded from the next decision,
   its worker leases are released best-effort and its old epoch is fenced;
5. the Request Agent refreshes the same contribution permit, reads one fresh
   TUF-matched DHT snapshot and reserves a new complete route at a higher
   epoch;
6. `/inference/v1/chat-replay` receives the exact original prompt IDs and
   committed output prefix, rebuilds KV and parser/tool state, and proves the
   replay boundary before any new event is published;
7. replayed SSE events are suppressed, so OpenCode observes every committed
   token exactly once.

The same mechanism works when the primary dies before its first generation
chunk: the qualified tokenize RPC is journalled before prefill. A replacement
that dies while replaying can itself be replaced at a still newer epoch.
Without another feasible DHT route, the stream ends with a typed OpenAI error;
Fabi never invents a continuation.

The deterministic integration harness covers both failure boundaries.  In the
decode case it commits one token on epoch 1, loses epoch 2 with a transport
error after that replacement has replayed the committed prefix, and completes
on epoch 3.  It asserts that the SQLite journal contains the exact two-token
sequence, the client sees the first token once, failed workers accumulate in
the exclusion fence, the original contribution permit is reused, and each
replacement route has a strictly newer epoch.  This is protocol evidence, not
a substitute for the pending native Mac/RTX/RunPod process-kill qualification.

Exact replay is currently admitted only for greedy sampling and the bounded
parameter subset implemented in `swarm_protocol.recovery`. Portable RNG state
between MLX, vLLM and SGLang is not yet a wire contract, so sampled requests
remain restartable.

This follows Petals' proven failure rule: ban the failed peer, reconstruct a
path from current discovery state and replay the session history into the
replacement. Petals keeps per-span input activations; Fabi's portable baseline
keeps token IDs and recomputes the complete route. A future negotiated fast
path may keep bounded boundary activations or use vLLM's official external KV
connector interface when model revision, layer span, dtype, KV layout and
backend match exactly. Cross-backend RTX/MLX recovery always retains token
replay as the correctness fallback.

Primary references:

- [Petals inference session recovery](https://github.com/bigscience-workshop/petals/blob/main/src/petals/client/inference_session.py)
- [HTTPX timeout semantics](https://www.python-httpx.org/advanced/timeouts/)
- [WHATWG server-sent event keepalive comments](https://html.spec.whatwg.org/dev/server-sent-events.html#authoring-notes)
- [gRPC keepalive versus health checking](https://grpc.io/docs/guides/keepalive/)
- [Iroh QUIC stream cancellation](https://docs.iroh.computer/protocols/using-quic#aborting-streams)
- [SQLite write-ahead logging](https://www.sqlite.org/wal.html)
- [SQLite transaction semantics](https://www.sqlite.org/lang_transaction.html)
- [vLLM KV connector base](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/base.py)
