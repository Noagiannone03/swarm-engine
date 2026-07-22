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
