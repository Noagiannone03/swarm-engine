# Fabi Request Agent authority

## Role

The public V3 authority turns one live contribution slot into one short-lived
route permit, then turns one exact client-signed `RoutePlan` into a Biscuit
capability. It does not choose layers or plan routes. Workers still choose
their spans from the signed DHT catalogue, and the local Request Agent chooses
one complete request route.

The authority is intentionally fail-closed and disabled unless all of these
conditions are true:

- `FABI_GATE=on`;
- `FABI_SWARM_V3_MODE=active`;
- the scheduler has an authenticated Iroh endpoint and an active V3 planner;
- `FABI_ROUTE_CAPABILITY_PRIVATE_KEY` points to an owner-only Ed25519 seed;
- `FABI_ROUTE_PERMIT_DB` points to persistent storage;
- the key is currently active in the TUF-authenticated
  `route-authorities.json` target.

The private key file contains exactly one lowercase 32-byte hexadecimal seed
and must be mode `0600` on Unix. The SQLite ledger is also created as `0600`
because it contains short-lived bearer capabilities. An operator must publish
the matching public key through the normal TUF threshold-signing workflow
before starting the authority. Fabi never silently generates a new public
authority key at runtime.

## HTTP contract

All endpoints use the same account bearer credential as contribution
admission. Account identifiers and credentials are never returned.

### `POST /v1/swarm/route-permits`

The Request Agent sends:

- `Authorization: Bearer <account credential>`;
- `Idempotency-Key: <request id>`;
- its Iroh `coordinator_endpoint_id`;
- the selected `model_swarm_id`;
- the maximum context needed by this request;
- one or more allowed recovery policies;
- a TTL no longer than five minutes.

The idempotency scope is `(account, Request Agent EndpointId, request id)`.
An exact retry returns the first permit even if its contribution slot is now
occupied. Reusing the key with a different model, context, policy or TTL
returns HTTP 422. New permits are accepted only while the same account has a
verified READY worker on this model and the live route supports the requested
context.

Public permits currently allow only `best_effort` and `replan_cold`.
`activation_replay`, `reserved_route` and `hot_replica` stay unavailable until
their runtime guarantees are implemented and qualified.

### `POST /v1/swarm/route-capabilities`

The body contains the permit ID, the exact signed `RoutePlan` and one recovery
policy granted by that permit. The service verifies:

- HTTP account ownership;
- signer and coordinator Iroh EndpointId;
- control signature over the exact plan bytes;
- request, model, context, epoch, digest, expiry and recovery policy;
- the issuer key against the current TUF trust snapshot.

The `(permit, epoch)` emission is durable and idempotent. A retry returns the
same persisted Biscuit. A different plan on the same epoch is rejected.

### `DELETE /v1/swarm/route-permits/{permit_id}`

Release is account-scoped. An unknown permit and another account's permit both
return 404, avoiding an object-existence oracle. Normal release frees the
contribution slot; emergency Biscuit revocation remains the TUF revocation
path.

## Capacity and scaling

Gateway generations and local Request Agent permits share the same local
contribution counter. One eligible worker therefore opens one request slot by
default regardless of which API path consumes it.

SQLite WAL with `BEGIN IMMEDIATE` is the supported single-authority backend.
It provides atomic quotas and epoch compare-and-swap across threads and
processes sharing one database file. Multiple horizontally scaled authority
instances require a PostgreSQL adapter implementing the same
`RoutePermitLedger` contract with transactions and row locks; copying the
SQLite file or replacing it with a simple Redis `SETNX` lock is unsupported.

The mutating POST contract follows the IETF HTTPAPI idempotency-key design:
missing key is HTTP 400, payload reuse is HTTP 422, capacity is HTTP 429, and
temporarily unavailable authority or swarm state is HTTP 503 with
`Retry-After`.
