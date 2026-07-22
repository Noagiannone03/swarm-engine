# Fabi Iroh relay deployment

The qualification relay uses Iroh's official `iroh-relay` binary, a shared
bearer token, per-client traffic limits, QUIC address discovery and the existing
Let's Encrypt certificate for `server.undefinedstudio.fr`.

It intentionally listens beside the production Caddy service:

- HTTPS relay: TCP 4443
- HTTP captive portal: TCP 4442
- QUIC address discovery: UDP 7842
- local Prometheus metrics: TCP 127.0.0.1:9091

The systemd unit reads the secret from `/etc/fabi-iroh-relay/relay.env` as
`IROH_RELAY_ACCESS_TOKEN`. That file must be root-owned with mode `0600` and is
never committed. Clients receive the same value through the account/bootstrap
control plane; the laboratory harness reads it from `FABI_RELAY_TOKEN` or a
protected file.

The relay binary must come from the pinned Iroh `v1.0.3` release. GitHub did not
publish a Sigstore attestation for the Linux asset as of 22 July 2026, so its
SHA-256 must be captured during deployment and the binary must later be built
by Fabi's own reproducible release workflow.
