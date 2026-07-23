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

Build a bundle from an exact Hub commit. The command downloads small runtime metadata files, but
uses the Hub's official LFS size and SHA-256 metadata instead of downloading weight blobs:

```console
fabi-swarm-registry build-hub-bundle \
  --model-id Qwen/Qwen3-4B \
  --revision 1cfa9a7208912126459214e8b04321603b3df60c \
  --quantization unquantized \
  --dtype bfloat16 \
  --output /private/operator/bundles/qwen3-4b.json
```

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
