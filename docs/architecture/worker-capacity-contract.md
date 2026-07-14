# Worker-authoritative capacity negotiation

## Decision

The worker owns every statement about what it can load and serve. The central
scheduler never derives a remote worker's layer capacity from GPU names, raw
VRAM, average model size, or a cluster-wide ratio.

The scheduler has only two responsibilities:

1. validate signed-by-session worker observations as untrusted protocol data;
2. solve a placement problem using the admissible ranges supplied by workers.

Legacy workers retain the former estimate path for protocol compatibility. A
worker that advertises capacity protocol v1 never falls back to that path. Its
capacity is zero until it publishes a valid model-specific contract.

## Lifecycle

1. **Discovery** — the worker joins with `capacity_protocol_version=1` and no
   profile because it does not yet know which backend-specific model the swarm
   selected. The scheduler acknowledges it as standby and returns that model.
2. **Provisional measurement** — the worker applies its local memory governor,
   reads exact Safetensors tensor offsets, classifies endpoint and decoder-layer
   weights, and reserves KV space for the configured context target.
3. **Placement contract** — the worker publishes `max_end_by_start`. Entry `s`
   is the largest exclusive layer boundary it admits for a shard starting at
   `s`. This preserves non-uniform quantized and MoE layer sizes.
4. **Placement** — the DP allocator uses those start-dependent edges directly.
   Every concrete allocation is validated again at the mutation boundary.
5. **Runtime authority** — after weights and caches are resident, the executor
   publishes total KV token capacity and live free KV tokens from the backend's
   real allocator. A worker is not routing-ready until total capacity exists.
6. **Calibration** — when runtime capacity is below the provisional context
   target, the worker derives a lower effective budget and republishes a
   `runtime_calibrated` contract. It can only shrink admissible ranges.
7. **Safe replan** — the scheduler waits for in-flight work to drain, moves the
   affected topology to standby, and replans from the calibrated contracts.

## Contract v1

The placement-relevant fields are:

```json
{
  "protocol_version": 1,
  "model_name": "backend-specific/model",
  "state": "provisional",
  "num_layers": 36,
  "target_context_tokens": 65536,
  "usable_memory_bytes": 123,
  "runtime_reserve_bytes": 123,
  "kv_bytes_per_token_by_layer": [123],
  "first_stage_weight_bytes": 123,
  "last_stage_weight_bytes": 123,
  "layer_weight_bytes": [123],
  "max_end_by_start": [4, 5]
}
```

The scheduler rejects the profile when the protocol, model identity, layer
count, state, or range bounds do not match the swarm. Unknown global tensors
also make profile construction fail rather than being counted as zero.

## Context semantics

Three values intentionally remain distinct:

- `target_context_tokens`: the context guaranteed by a provisional placement;
- `kv_capacity_tokens`: the idle, load-independent ceiling measured after the
  actual shard is loaded;
- `kv_free_tokens`: live headroom used by per-request routing.

The registry exposes the minimum load-independent capacity across a complete
ready path. Request routing additionally checks the live value on every stage.
This prevents model metadata from fluctuating with traffic while still refusing
a long coding request when any stage lacks current cache space.

## Trust and failure behavior

- Capacity data is scoped to the stable peer id plus the worker process session
  id; stale sessions cannot mutate it.
- A missing, malformed, mismatched, or unsupported v1 profile means standby,
  never an optimistic estimate.
- A zero runtime KV pool is a hard zero.
- Runtime calibration never increases the provisional budget.
- Replanning does not interrupt in-flight requests.
- Unsupported cache geometries must be measured by their backend allocator; a
  standard-attention formula is not treated as runtime truth.

## Future protocol evolution

Protocol v2 can add benchmark-derived latency curves, thermal envelopes, and
multi-GPU topology without changing the authority boundary. Version negotiation
must remain explicit, and the scheduler must never reinterpret an unknown
contract as a legacy worker.
