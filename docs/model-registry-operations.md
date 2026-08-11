# Fabi model registry operations

The protocol-v3 model registry is a public TUF repository. Its targets bind a human model name,
an immutable source revision, runtime artifacts, execution contracts, and one deterministic model
swarm ID. The DHT can make those records available, but it is never the trust root.

## Staging authority

`fabi-swarm-registry` supports the 1-of-1 staging authority used by the laboratory. It generates a
different Ed25519 key for each TUF top-level role and stores every private key as encrypted PKCS#8
PEM. On Unix, the key directory is `0700`, private keys and passphrase files are `0600`, while the
published TUF repository is `0644` and contains no secret.

This is not the production root ceremony. Production root keys must be offline and thresholded;
online targets/snapshot/timestamp keys should use separate operator identities or a KMS/HSM signer
behind `RegistryRoleSigners`.

Create a random staging passphrase without writing it to stdout:

```console
fabi-swarm-registry generate-passphrase \
  --output /private/operator/secrets/registry.passphrase
```

Build a bundle from an exact Hub commit. In addition to the small runtime files and official LFS
metadata, the publishing authority streams every immutable SafeTensors data section once through
Hugging Face Xet. It does not persist a second full checkpoint: it computes and signs each tensor
SHA-256, absolute byte range, dtype and shape, together with the Xet Merkle file identity. This is
an operator-side, once-per-model-revision cost; it is what lets every worker later verify a partial
checkpoint without first downloading the complete source files:

```console
fabi-swarm-registry build-hub-bundle \
  --model-id Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --quantization unquantized \
  --dtype bfloat16 \
  --output /private/operator/bundles/qwen3-4b.json
```

The command retries only transient Hub transport failures with bounded exponential backoff. A
metadata mismatch, invalid SafeTensors layout, missing Xet identity, short reconstruction, or
digest mismatch fails closed and no bundle is published.

### Attach a portable execution plan

Publisher-built ONNX stages live in a separate immutable Hub repository. Upload the complete
`execution/` directory, resolve that repository to a 40-character commit, then bind the local
builder inventory to the source-model bundle:

```console
fabi-swarm-registry attach-portable-execution \
  --bundle /private/operator/bundles/qwen3-600m.json \
  --inventory /private/operator/portable/qwen3-600m/portable-build.json \
  --artifact-root /private/operator/portable/qwen3-600m \
  --source-root /private/operator/exports/qwen3-600m \
  --artifact-repository-id fabi-ai/Qwen3-0.6B-onnx-stages \
  --artifact-revision 0123456789abcdef0123456789abcdef01234567 \
  --plan-id onnx-dml-int4-v1 \
  --precision int4 \
  --quantization rtn-block-32 \
  --provider directml \
  --output /private/operator/bundles/qwen3-600m-portable.json
```

This operation never edits either input and refuses to replace an existing output. It recomputes
the domain-separated inventory identity; verifies the source export and every graph/data file by
size and SHA-256 from stable, root-confined regular files; rejects unreferenced bytes and layer
gaps; pins the artifact repository and revision; and recomputes the manifest execution-plan hash.
The resulting bundle is still only an unsigned operator artifact until `init-staging` or `publish`
places it under TUF targets metadata. A worker downloads only the stages for its assigned span and
rechecks the same signed descriptors before announcing READY. Omit `--provider` only to select the
builder target's exact default; list an additional provider only after the same graph set has been
qualified on that provider.

## Selective worker materialization

For a bundle carrying the signed tensor index, a worker materializes deterministic
`model-fabi-layer-NNNNN.safetensors` packs for its exact span. Input embeddings and output
norm/head tensors use separate endpoint packs. The maintained `hf-xet` client reconstructs only
the signed byte ranges; an existing complete Hub shard is reused only after its legacy size and
SHA-256 pass verification. A generated SafeTensors weight map makes those packs consumable by the
existing MLX, vLLM and SGLang loaders.

Packs live in Fabi-owned content-specific caches. `FABI_MODEL_ARTIFACT_CACHE` selects the primary
cache (default `~/.cache/fabi/models`). `FABI_MODEL_ARTIFACT_CACHE_ROOTS` may add explicitly
authorized writable volumes as a JSON string array; JSON is required on Windows because drive
letters contain colons. Missing extra roots are treated as unmounted and are never recreated, which
prevents an unplugged `/Volumes/...` path from silently becoming a directory on the system disk.
The product UI must obtain extra roots through a native directory grant; it must not write to every
enumerated removable or network volume.

The worker performs a non-destructive exact plan on every authorized volume after layer placement.
It first minimizes missing content bytes (reuse), then required eviction, then chooses the largest
remaining safe headroom. An infeasible volume is not cleaned merely because it was inspected. One
projection remains wholly on one volume; Fabi does not assemble model directories with privileged
Windows links or cross-volume symlink farms. Reallocation is incremental on the selected volume:
already verified packs are reused and only new layers are fetched. Every READY admission rechecks
the deterministic weight map, pack digest, tensor metadata and each signed tensor SHA-256. A locally
altered pack and receipt is therefore rejected even while offline. Bundles published before the
tensor index remain compatible and use the older whole-shard path.

Before the first range request, the worker reserves the exact net growth of all signed packs and
runtime metadata plus a bounded 1 MiB atomic-transaction workspace. The default free-space floor is
2% of the containing volume, bounded to 1–10 GiB; cleanup then targets an additional 0.5%, bounded
to 256 MiB–2 GiB, so repeated reallocations do not oscillate at the threshold. Operators may set
`FABI_MODEL_CACHE_MIN_FREE_BYTES`, `FABI_MODEL_CACHE_HYSTERESIS_BYTES`, or
`FABI_MODEL_CACHE_MAX_BYTES` to non-negative byte counts when a managed volume has an explicit
quota.

Reservations and active packs are protected by durable process leases that bind both PID and
process creation time, so PID reuse cannot preserve stale content. Concurrent downloads subtract
their outstanding reservations before a new one is admitted. A SQLite journal records only
successfully verified uses; eviction uses GreedyDual-Size-Frequency dynamic ageing and therefore
retains spans that repeatedly avoid a download without feeding cache popularity back into placement.
An abrupt process death may prevent Python's `mkstemp()` cleanup from running. Before a new
admission plan, Fabi reaps only exact hidden pack-temporary names from projections with no live
lease, while holding the cache lock and acquiring the projection writer lock without waiting. A
live downloader is therefore never touched, and an antivirus-held Windows orphan remains charged
to measured free space instead of turning cleanup contention into a worker crash.
The selected span is always protected during its own cleanup. A worker either materializes that
placement, reclaims unleased cold packs, or raises a typed storage error before network transfer.
That typed failure carries the exact missing bytes and fenced placement generation. The P2P
controller excludes only the proven-impossible span, keeps the worker heartbeat alive, and applies
the unchanged placement score to the remaining spans. Only after every locally feasible span has
failed does contribution report `insufficient_storage`; the IDE must show that terminal condition
instead of an indefinite model-loading animation.

This garbage collector owns only Fabi's `model-fabi-*.safetensors` projections. Never delete files
inside Hugging Face's shared blob cache manually: use the maintained `scan_cache_dir()` and
`delete_revisions()` APIs so blobs referenced by another snapshot remain intact.

Initialize a new staging repository. Existing keys, bootstrap roots, and non-empty repositories
are never overwritten:

```console
fabi-swarm-registry init-staging \
  --repository-dir /srv/fabi-registry-v3 \
  --key-dir /private/operator/keys \
  --bundle /private/operator/bundles/qwen3-4b.json \
  --bootstrap-root-output /private/operator/bootstrap-root.json \
  --passphrase-file /private/operator/secrets/registry.passphrase
```

Publish a complete subsequent snapshot with the existing authority:

```console
fabi-swarm-registry publish \
  --repository-dir /srv/fabi-registry-v3 \
  --key-dir /private/operator/keys \
  --bundle /private/operator/bundles/qwen3-4b.json \
  --passphrase-file /private/operator/secrets/registry.passphrase
```

If the online timestamp refresher has run since the previous offline publish,
synchronize its public state first. The command accepts only credential-free
HTTPS, verifies the timestamp against the retained root history, rejects
rollback/equivocation/expiry, and requires it to authenticate the exact
versioned snapshot already present in the offline repository:

```console
fabi-swarm-registry sync-online-timestamp \
  --repository-dir /srv/fabi-registry-v3 \
  --timestamp-url https://example.invalid/fabi-swarm-registry-v3/metadata/timestamp.json
```

This imports only `timestamp.json`; it never downloads a root, snapshot,
targets file, model target or private key. A subsequent `publish` therefore
increments from the real online timestamp version instead of creating two
different metadata documents with the same rollback-sensitive version.

The timestamp role is deliberately short-lived. Refresh it from an automated
operator before half of its 24-hour validity has elapsed:

```console
fabi-swarm-registry refresh-timestamp \
  --repository-dir /srv/fabi-registry-v3 \
  --key-dir /private/operator/timestamp-key \
  --passphrase-file /private/operator/secrets/timestamp.passphrase
```

The refresh host needs only `timestamp.pem`, its owner-only passphrase and the
public repository. It must not receive root, targets or snapshot private keys.
A separate full `publish` must run before the seven-day snapshot expires.
`refresh-timestamp` refuses to extend an already expired snapshot.

For a Linux registry mirror, maintained hardened systemd units are provided in
`deploy/model-registry/`. The timer runs every six hours, catches up after
downtime and executes an exact operator image without network or Linux
capabilities. Provisioning performs one immediate refresh and remote
verification; a timer that has merely been enabled is not evidence that its
credentials or image are valid.

Generate the short-lived Request Agent capability authority without printing
its private seed:

```console
fabi-swarm-registry generate-route-authority \
  --private-key-output /private/operator/secrets/route-capability.key \
  --keyset-output /private/operator/route-authorities.json \
  --generation 1 \
  --valid-for-days 90
```

Publish the public keyset with `publish --route-authorities ...`; mount only
the owner-only seed into the request authority. Never copy that seed into the
public repository.

Only the public repository and `bootstrap-root.json` are distributed. Private keys and their
passphrase never go to the scheduler, workers, web root, container image, shell arguments, logs,
or source control.

## HTTPS publication

Serve `metadata/` and `targets/` byte-for-byte over HTTPS. Do not rewrite JSON and do not expose a
directory containing operator keys. `timestamp.json` must be fetched fresh: configure it with
`Cache-Control: no-store` (or an equivalently bounded policy). Versioned root/targets/snapshot
metadata and hash-prefixed targets are immutable and may be cached.

Verify the deployed repository through the same TUF client used by workers and the scheduler:

```console
fabi-swarm-registry verify-remote \
  --bootstrap-root /private/operator/bootstrap-root.json \
  --metadata-url https://example.invalid/fabi-swarm-registry-v3/metadata/ \
  --targets-url https://example.invalid/fabi-swarm-registry-v3/targets/ \
  --state-dir /tmp/fabi-registry-verification \
  --model-id Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --quantization unquantized \
  --dtype bfloat16
```

Workers receive the bootstrap root out of band. They must never download an unpinned root from the
same server they are trying to authenticate. Rotate root with the dual-signature continuity rule
implemented by `TufRegistryPublisher.rotate_root` before revoking an old key.

## Shadow deployment invariant

Set `FABI_SWARM_V3_MODE=shadow` only after the registry is reachable and the root has been pinned.
In shadow mode:

- v2 remains the only serving and reservation path;
- model verification runs outside the heartbeat thread;
- v3 offers and leases are emitted only after local artifact verification;
- the scheduler compares v2 and v3 decisions but cannot route a real prompt through v3;
- any registry, verification, or planner failure is observable and fail-closed for v3 without
  stopping v2.

Do not enable v3 `PREPARE/COMMIT` or real traffic until both worker join orders, long-context
generation, abort, memory/KV telemetry, and rollback have been qualified on the laboratory.
