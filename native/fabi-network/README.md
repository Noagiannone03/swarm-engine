# Fabi network transport

This crate is the product transport boundary for Fabi's centrally scheduled
worker swarms. It uses Iroh's authenticated QUIC endpoints, direct UDP hole
punching and relay fallback. It does not own scheduling, model routing or
contribution policy.

The current binary is a qualification harness built from the same bounded
framing, persistent identity and path telemetry intended for the runtime. It
must pass forced-relay and automatic direct-upgrade tests between independent
NATs before the Python RPC layer is migrated.

The optional PyO3 extension exposes the same endpoint as `NetworkNode`.
`src/fabi_network/rpc.py` builds bounded concurrent services on top of it:

- unary calls return standard Python futures;
- server-streaming calls use one cancellable QUIC stream per request;
- cancellation resets only that stream, never the shared peer connection;
- handlers are dispatched concurrently so long generations cannot block
  scheduler heartbeats;
- application values use an explicit codec byte followed by MessagePack,
  protobuf or raw bytes. Python pickle is intentionally forbidden on the wire.

Relay credentials are read from `FABI_RELAY_TOKEN` or a protected token file;
they are never emitted in JSON or logs. `--force-relay` disables IP transports
to prove that fallback works instead of accidentally succeeding over a LAN or
VPN route.

```shell
cargo test --manifest-path native/fabi-network/Cargo.toml
cargo test --features python --manifest-path native/fabi-network/Cargo.toml
cargo run --release --manifest-path native/fabi-network/Cargo.toml -- \
  --identity ./worker.key \
  --relay-url https://relay.example.com \
  serve
```

Build the stable-ABI Python extension with Maturin's dedicated extension
feature (kept separate from native Rust tests so macOS links correctly):

```shell
maturin develop --release --manifest-path native/fabi-network/Cargo.toml
```
