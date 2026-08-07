# Scheduler image

The scheduler image builds the native Iroh extension and the Python runtime
from one required, immutable `PARALLAX_COMMIT`.

Build from the repository root:

```shell
docker build \
  --file deploy/scheduler/Dockerfile \
  --build-arg PARALLAX_COMMIT=<full-commit-sha> \
  --tag local/parallax-scheduler:<short-commit-sha> \
  .
```

The build fails if the checked-out source does not match the requested commit
or if the native module cannot be imported in the final image. Relay
credentials belong in the deployment secret store, never in this image.

The versioned lab override switches only the qualified Qwen3-1.7B scheduler
service to Iroh while leaving the other model swarms on their current runtime:

```shell
docker compose \
  -f docker-compose.yml \
  -f docker-compose.lab-iroh.yml \
  up -d --no-build parallax-scheduler
```

`docker-compose.lab-iroh.yml` intentionally contains no credential. The host
must provide `/etc/fabi-iroh-relay/relay.env` as a root-owned, mode `0600`
secret file. Promote a commit-addressed image to the local
`local/parallax-scheduler:iroh-qualified` tag only after its smoke tests pass;
the immutable commit remains available in the image revision label. The
scheduler identity persists in its existing state volume.

To qualify the existing Qwen3-8B service on Iroh, apply its dedicated override:

```shell
docker compose \
  -f docker-compose.yml \
  -f docker-compose.lab-iroh-qwen3-8b.yml \
  up -d --no-deps --force-recreate parallax-scheduler-qwen3-8b
```

The 8B service keeps a separate persistent scheduler identity in the existing
`parallax-state-qwen3-8b` volume. Workers must use the endpoint reported by
that service, not the Qwen3-1.7B endpoint.

The legacy fixed parameter/KV split could not fit the 8B model endpoints and
decoder layers inside the 16 GB Mac/RTX live envelopes. The exact adaptive
planner can form a context-qualified split when both live envelopes allow it.
Use the dedicated 4B service first for a deterministic two-worker network and
runtime qualification:

```shell
docker compose \
  -f docker-compose.lab-iroh-qwen3-4b.yml \
  up -d --no-deps parallax-scheduler-qwen3-4b
```

This service is deliberately isolated on HTTP `3025`, keeps its own scheduler
identity and uses the same commit-addressed qualified image. The override is
the authoritative V3 deployment contract: it includes the registry discovery
labels, TUF root, route-authority key, catalogue identity and persistent state
mounts instead of relying on variables inherited from an unrelated compose
service.

An existing laboratory must name its already-provisioned state volume and
identity path explicitly during promotion. This preserves the Iroh EndpointId
and fencing ledger while still selecting the candidate image by immutable tag:

```shell
FABI_QWEN3_4B_SCHEDULER_IMAGE=local/parallax-scheduler:<short-commit-sha> \
FABI_QWEN3_4B_STATE_VOLUME=parallax-state-qwen3-4b-v3-eb3d4ff \
FABI_QWEN3_4B_NETWORK_IDENTITY_PATH=/opt/parallax-runtime/network/scheduler-2fabfbf.key \
docker compose \
  -f docker-compose.lab-iroh-qwen3-4b.yml \
  up -d --no-deps --force-recreate parallax-scheduler-qwen3-4b
```

Fresh laboratories may omit those two state overrides and receive a new named
volume and scheduler identity. Private relay and route-authority bytes remain
root-owned bind mounts; they are never placed in Compose environment values.

The direct-GGUF/Skippy product qualification uses a separate Qwen3-0.6B swarm
so it cannot mutate the qualified Qwen3-4B control state:

```shell
FABI_QWEN3_0_6B_SCHEDULER_IMAGE=local/parallax-scheduler:<short-commit-sha> \
docker compose \
  -f docker-compose.lab-iroh-qwen3-0.6b.yml \
  up -d --no-deps parallax-scheduler-qwen3-0-6b
```

It owns HTTP `3026`, transport `18161`, catalogue `19193`, a dedicated state
volume and a distinct persistent Iroh identity. The public TUF catalogue must
already contain the exact Qwen3-0.6B execution plan before this service is
started; Compose never injects or weakens model trust metadata.

Every scheduler identity is an infrastructure endpoint. After the state volume
has created `network/scheduler.key`, derive its public EndpointId with the
bundled native network module and add that public ID to the registry service's
`FABI_RELAY_INFRA_ENDPOINTS` allow-list before restarting the scheduler. Do not
reuse another scheduler's authorization: relay access is intentionally bound to
the persistent identity. User workers follow the separate account-owned
`/v1/network/enroll` flow automatically and never need this operator step.

Product scheduler context tiers can be configured without changing worker
ratios:

```shell
PARALLAX_PLANNING_CONTEXT_TOKENS=16384
PARALLAX_PREFERRED_CONTEXT_TOKENS=32768
```

The scheduler publishes its selected tier, and a worker is not READY until its
runtime-measured KV pages satisfy that contract.
