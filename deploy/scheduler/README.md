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
