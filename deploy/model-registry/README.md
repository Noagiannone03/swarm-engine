# Fabi TUF timestamp refresher

TUF clients must reject expired metadata. The registry therefore refreshes its
short-lived timestamp every six hours, well before half of the 24-hour validity
window has elapsed. This does not extend the snapshot: a complete, separately
authorized publication is still required before the seven-day snapshot expiry.

The service follows the TUF role separation:

- it receives only `timestamp.pem` and its passphrase;
- root, targets and snapshot private keys remain offline;
- the public repository is the only writable bind mount;
- the operator container has no network, no Linux capabilities and a read-only
  root filesystem;
- `flock` prevents overlapping refreshes;
- the timestamp is published atomically by `TufTimestampRefresher`.

`Persistent=true` makes systemd catch up after downtime. `RandomizedDelaySec`
avoids synchronized refresh traffic when the same units are deployed on
multiple registry mirrors.

The templated `fabi-tuf-timestamp-refresh@.service` and `.timer` variants let
an operator stage a new independent TUF root alongside the currently served
authority. Each instance has a separate credential directory, repository,
container name and lock. This is required for a fail-safe root migration: the
old timestamp continues to refresh until every qualified client has moved.

## Provisioning

Build an operator image from the exact reviewed engine commit. The image must
provide the `fabi-swarm-registry` console script and must not contain any
private key.

Install the public repository at `/srv/fabi-swarm-registry-v3`, owned by
`root:root` with `0755` directories and `0644` public metadata/targets. The
operator container runs as root with every capability dropped; it needs write
access only to atomically replace `metadata/timestamp.json`. Then provision the
online timestamp role:

```console
sudo install -d -m 0700 /etc/fabi-tuf-timestamp
sudo install -m 0600 timestamp.pem /etc/fabi-tuf-timestamp/timestamp.pem
sudo install -m 0600 timestamp.passphrase \
  /etc/fabi-tuf-timestamp/timestamp.passphrase
sudo install -m 0644 refresh.env \
  /etc/fabi-tuf-timestamp/refresh.env
sudo install -m 0644 fabi-tuf-timestamp-refresh.service \
  /etc/systemd/system/fabi-tuf-timestamp-refresh.service
sudo install -m 0644 fabi-tuf-timestamp-refresh.timer \
  /etc/systemd/system/fabi-tuf-timestamp-refresh.timer
sudo systemctl daemon-reload
sudo systemctl enable --now fabi-tuf-timestamp-refresh.timer
sudo systemctl start fabi-tuf-timestamp-refresh.service
```

The explicit first start is a fail-closed provisioning check. Validate both
the service and the public repository before considering the registry ready:

```console
systemctl status fabi-tuf-timestamp-refresh.service
systemctl list-timers fabi-tuf-timestamp-refresh.timer
fabi-swarm-registry verify-remote \
  --bootstrap-root /private/operator/bootstrap-root.json \
  --metadata-url https://example.invalid/fabi-swarm-registry-v3/metadata/ \
  --targets-url https://example.invalid/fabi-swarm-registry-v3/targets/ \
  --state-dir /tmp/fabi-registry-verification \
  --model-swarm-id MODEL_SWARM_ID
```

The separate metadata checker authenticates the complete chain from an
out-of-band bootstrap root and exits nonzero when the timestamp has less than
eight hours left or any offline role has less than 72 hours left. This makes an
approaching snapshot ceremony visible as a failed systemd unit without placing
the snapshot key on the mirror. Alert on any failed refresh or metadata-check
execution. The timers are not a replacement for the full snapshot publication
ceremony.

For a parallel authority named `root3`, install the templated units and put its
three instance files under `/etc/fabi-tuf-timestamp/root3/`:

```console
sudo install -d -m 0700 /etc/fabi-tuf-timestamp/root3
sudo install -m 0600 timestamp.pem /etc/fabi-tuf-timestamp/root3/timestamp.pem
sudo install -m 0600 timestamp.passphrase \
  /etc/fabi-tuf-timestamp/root3/timestamp.passphrase
sudo install -m 0644 refresh.env /etc/fabi-tuf-timestamp/root3/refresh.env
sudo install -m 0644 fabi-tuf-timestamp-refresh@.service /etc/systemd/system/
sudo install -m 0644 fabi-tuf-timestamp-refresh@.timer /etc/systemd/system/
sudo install -m 0644 fabi-tuf-metadata-check@.service /etc/systemd/system/
sudo install -m 0644 fabi-tuf-metadata-check@.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now fabi-tuf-timestamp-refresh@root3.timer
sudo systemctl enable --now fabi-tuf-metadata-check@root3.timer
sudo systemctl start fabi-tuf-timestamp-refresh@root3.service
sudo systemctl start fabi-tuf-metadata-check@root3.service
```

Do not stop the previous authority's timer during this staging phase. As with
the single-authority unit, only the timestamp key and its passphrase are copied
to the mirror; the offline root, targets and snapshot keys never leave the
operator host.

Before the 72-hour warning becomes active, synchronize the current online
timestamp into the offline operator repository, perform the reviewed targets
and snapshot publication, deploy versioned metadata first and `timestamp.json`
last, then verify the public HTTPS repository from the pinned bootstrap root.
