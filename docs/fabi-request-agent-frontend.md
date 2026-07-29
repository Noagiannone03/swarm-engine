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

The OpenAI endpoint is loopback-only. `/v1/models`,
`/v1/request-agent/status` and `/v1/chat/completions` require the same account
Bearer credential that the Request Agent uses for contribution admission.
`/health` exposes only `ready` or `waiting`.

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

## Current recovery boundary

The local frontend already inherits explicit engine abort and exact route
fencing. Its first committed version intentionally leaves token capture
disabled until the durable local journal and `replan_cold` transaction are
connected. Until that next milestone, loss of an executing route returns an
OpenAI upstream error and never silently duplicates output.

The next milestone reuses `OpenAIRecoveryStream`, the exact replay sampling
contract and the worker `/inference/v1/chat-replay` endpoint already qualified
in the gateway runtime. It will replace the old pre-reserved backup model with
a fresh DHT plan and a new fencing epoch.
