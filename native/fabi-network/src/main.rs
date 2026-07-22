use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, ensure};
use clap::{Args, Parser, Subcommand};
use fabi_network::{
    ALPN,
    endpoint::{EndpointConfig, bind},
    identity,
    protocol::{
        DEFAULT_MAX_PAYLOAD, Header, MessageKind, digest_for_payload, read_header,
        receive_and_verify_payload, send_payload, write_header,
    },
    telemetry::{PathSnapshot, snapshot},
};
use iroh::{Endpoint, EndpointAddr, EndpointId, RelayUrl, TransportAddr, endpoint::Connection};
use serde::Serialize;
use tokio::task::JoinSet;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

#[derive(Debug, Parser)]
#[command(version, about)]
struct Cli {
    #[command(flatten)]
    network: NetworkArgs,
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Args)]
struct NetworkArgs {
    /// Stable Ed25519 endpoint identity.
    #[arg(long)]
    identity: PathBuf,
    /// Self-hosted Iroh relay URL, for example <https://relay.example.com>.
    #[arg(long)]
    relay_url: RelayUrl,
    /// Relay bearer token. Prefer `FABI_RELAY_TOKEN` over the command line.
    #[arg(long, env = "FABI_RELAY_TOKEN", hide_env_values = true)]
    relay_token: Option<String>,
    /// Read the relay bearer token from a file.
    #[arg(long, conflicts_with = "relay_token")]
    relay_token_file: Option<PathBuf>,
    /// Disable all direct UDP paths to prove relay fallback independently.
    #[arg(long)]
    force_relay: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Accept bounded benchmark streams until interrupted.
    Serve(ServeArgs),
    /// Measure connection, upload and path selection to a worker endpoint.
    Benchmark(BenchmarkArgs),
}

#[derive(Debug, Args)]
struct ServeArgs {
    #[arg(long, default_value_t = DEFAULT_MAX_PAYLOAD)]
    max_payload_bytes: u64,
    /// Exit after handling this many connections; zero serves indefinitely.
    #[arg(long, default_value_t = 0)]
    max_connections: u64,
}

#[derive(Debug, Args)]
struct BenchmarkArgs {
    #[arg(long)]
    endpoint_id: EndpointId,
    /// Optional direct addresses advertised by the remote endpoint.
    #[arg(long = "remote-addr")]
    remote_addrs: Vec<SocketAddr>,
    #[arg(long, default_value_t = 64 * 1024 * 1024)]
    payload_bytes: u64,
    #[arg(long, default_value_t = 3)]
    iterations: u32,
    /// Wait for relay-to-direct path upgrades before the measured upload.
    #[arg(long, default_value_t = 3)]
    settle_seconds: u64,
    #[arg(long, default_value_t = 120)]
    timeout_seconds: u64,
    /// Reset one upload after this delay, then prove the connection remains usable.
    #[arg(long)]
    cancel_first_after_ms: Option<u64>,
}

#[derive(Debug, Serialize)]
struct ReadyEvent {
    event: &'static str,
    endpoint_id: String,
    relay_url: String,
    direct_addresses: Vec<String>,
    force_relay: bool,
}

#[derive(Debug, Serialize)]
struct IterationResult {
    request_id: u64,
    payload_bytes: u64,
    duration_ms: f64,
    throughput_mib_s: f64,
}

#[derive(Debug, Serialize)]
struct BenchmarkResult {
    event: &'static str,
    local_endpoint_id: String,
    remote_endpoint_id: String,
    connect_ms: f64,
    force_relay: bool,
    paths_before_settle: Vec<PathSnapshot>,
    paths_after_settle: Vec<PathSnapshot>,
    paths_after_benchmark: Vec<PathSnapshot>,
    cancellation: Option<CancellationResult>,
    iterations: Vec<IterationResult>,
}

#[derive(Debug, Serialize)]
struct CancellationResult {
    requested_after_ms: u64,
    observed_after_ms: f64,
    payload_bytes: u64,
}

fn emit_json(value: &impl Serialize) -> Result<()> {
    println!("{}", serde_json::to_string(value)?);
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();
    let secret_key = identity::load_or_create(&cli.network.identity)?;
    let relay_token = match cli.network.relay_token {
        Some(token) => Some(token),
        None => EndpointConfig::token_from_file(cli.network.relay_token_file.as_deref())?,
    };
    let config = EndpointConfig {
        relay_url: cli.network.relay_url.clone(),
        relay_token,
        force_relay: cli.network.force_relay,
    };
    let endpoint = bind(secret_key, &config).await?;

    match cli.command {
        Command::Serve(args) => serve(endpoint, &config, args).await,
        Command::Benchmark(args) => benchmark(endpoint, &config, args).await,
    }
}

async fn serve(endpoint: Endpoint, config: &EndpointConfig, args: ServeArgs) -> Result<()> {
    let addr = endpoint.addr();
    emit_json(&ReadyEvent {
        event: "ready",
        endpoint_id: endpoint.id().to_string(),
        relay_url: config.relay_url.to_string(),
        direct_addresses: addr.ip_addrs().map(ToString::to_string).collect(),
        force_relay: config.force_relay,
    })?;

    let max_payload = args.max_payload_bytes;
    let mut connection_count = 0_u64;
    let mut tasks = JoinSet::new();
    loop {
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {
                info!("received shutdown signal");
                break;
            }
            incoming = endpoint.accept() => {
                let Some(incoming) = incoming else { break; };
                let mut accepting = match incoming.accept() {
                    Ok(accepting) => accepting,
                    Err(error) => {
                        error!(%error, "rejected incoming transport connection");
                        continue;
                    }
                };
                let alpn = accepting.alpn().await.context("failed to negotiate ALPN")?;
                ensure!(alpn == ALPN, "unexpected ALPN {}", String::from_utf8_lossy(&alpn));
                let connection = accepting.await.context("failed to authenticate peer")?;
                connection_count += 1;
                tasks.spawn(handle_connection(connection, max_payload));
                if args.max_connections > 0 && connection_count >= args.max_connections {
                    break;
                }
            }
            Some(result) = tasks.join_next(), if !tasks.is_empty() => {
                match result {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => warn!(%error, "peer connection ended with an error"),
                    Err(error) => error!(%error, "peer connection task panicked"),
                }
            }
        }
    }

    while let Some(result) = tasks.join_next().await {
        match result {
            Ok(Ok(())) => {}
            Ok(Err(error)) => warn!(%error, "peer connection ended with an error"),
            Err(error) => error!(%error, "peer connection task panicked"),
        }
    }
    endpoint.close().await;
    Ok(())
}

async fn handle_connection(connection: Connection, max_payload: u64) -> Result<()> {
    let remote = connection.remote_id();
    info!(%remote, "accepted authenticated peer");
    let connection = Arc::new(connection);
    let mut streams = JoinSet::new();
    loop {
        tokio::select! {
            stream = connection.accept_bi() => {
                match stream {
                    Ok((send, recv)) => {
                        streams.spawn(handle_stream(send, recv, max_payload));
                    }
                    Err(error) => {
                        if streams.is_empty() {
                            info!(%remote, %error, "peer connection closed");
                            break;
                        }
                        error!(%remote, %error, "failed to accept peer stream");
                        break;
                    }
                }
            }
            Some(result) = streams.join_next(), if !streams.is_empty() => {
                report_stream_result(result);
            }
        }
    }
    while let Some(result) = streams.join_next().await {
        report_stream_result(result);
    }
    Ok(())
}

fn report_stream_result(result: Result<Result<()>, tokio::task::JoinError>) {
    match result {
        Ok(Ok(())) => {}
        Ok(Err(error)) => warn!(%error, "isolated invalid or cancelled stream"),
        Err(error) => error!(%error, "stream task panicked"),
    }
}

async fn handle_stream(
    mut send: iroh::endpoint::SendStream,
    mut recv: iroh::endpoint::RecvStream,
    max_payload: u64,
) -> Result<()> {
    let header = read_header(&mut recv, max_payload).await?;
    ensure!(
        matches!(header.kind, MessageKind::Upload | MessageKind::Ping),
        "server received invalid request kind {:?}",
        header.kind
    );
    receive_and_verify_payload(&mut recv, &header).await?;
    let response_kind = if header.kind == MessageKind::Ping {
        MessageKind::Pong
    } else {
        MessageKind::Acknowledgement
    };
    let response = Header {
        kind: response_kind,
        request_id: header.request_id,
        payload_len: header.payload_len,
        digest: header.digest,
    };
    write_header(&mut send, &response).await?;
    send.finish().context("failed to finish response stream")?;
    Ok(())
}

async fn benchmark(endpoint: Endpoint, config: &EndpointConfig, args: BenchmarkArgs) -> Result<()> {
    ensure!(args.iterations > 0, "iterations must be greater than zero");
    ensure!(
        args.payload_bytes <= DEFAULT_MAX_PAYLOAD,
        "payload exceeds client safety limit {DEFAULT_MAX_PAYLOAD}"
    );
    let remote = EndpointAddr::from_parts(
        args.endpoint_id,
        args.remote_addrs
            .iter()
            .copied()
            .map(TransportAddr::Ip)
            .chain(std::iter::once(TransportAddr::Relay(
                config.relay_url.clone(),
            ))),
    );

    let connect_started = Instant::now();
    let connection = tokio::time::timeout(
        Duration::from_secs(args.timeout_seconds),
        endpoint.connect(remote, ALPN),
    )
    .await
    .context("connection timed out")?
    .context("failed to connect to remote endpoint")?;
    let connect_duration = connect_started.elapsed();
    let paths_before_settle = snapshot(&connection);
    tokio::time::sleep(Duration::from_secs(args.settle_seconds)).await;
    let paths_after_settle = snapshot(&connection);

    let cancellation = match args.cancel_first_after_ms {
        Some(delay_ms) => Some(
            cancel_upload(
                &connection,
                args.payload_bytes,
                delay_ms,
                args.timeout_seconds,
            )
            .await?,
        ),
        None => None,
    };

    let mut results = Vec::with_capacity(args.iterations as usize);
    for iteration in 0..args.iterations {
        let request_id = u64::from(iteration) + 1;
        let digest = digest_for_payload(request_id, args.payload_bytes);
        let header = Header::new(MessageKind::Upload, request_id, args.payload_bytes, digest);
        let started = Instant::now();
        tokio::time::timeout(Duration::from_secs(args.timeout_seconds), async {
            let (mut send, mut recv) = connection
                .open_bi()
                .await
                .context("failed to open benchmark stream")?;
            write_header(&mut send, &header).await?;
            send_payload(&mut send, request_id, args.payload_bytes).await?;
            send.finish().context("failed to finish request stream")?;
            let acknowledgement = read_header(&mut recv, DEFAULT_MAX_PAYLOAD).await?;
            ensure!(
                acknowledgement.kind == MessageKind::Acknowledgement,
                "unexpected benchmark response kind {:?}",
                acknowledgement.kind
            );
            ensure!(
                acknowledgement.request_id == request_id
                    && acknowledgement.payload_len == args.payload_bytes
                    && acknowledgement.digest == header.digest,
                "benchmark acknowledgement does not match request {request_id}"
            );
            Result::<()>::Ok(())
        })
        .await
        .with_context(|| format!("benchmark request {request_id} timed out"))??;
        let duration = started.elapsed();
        let payload_bytes_f64 = f64::from(
            u32::try_from(args.payload_bytes)
                .context("payload exceeds the benchmark's numeric reporting range")?,
        );
        results.push(IterationResult {
            request_id,
            payload_bytes: args.payload_bytes,
            duration_ms: duration.as_secs_f64() * 1_000.0,
            throughput_mib_s: (payload_bytes_f64 / (1024.0 * 1024.0)) / duration.as_secs_f64(),
        });
    }

    emit_json(&BenchmarkResult {
        event: "benchmark_complete",
        local_endpoint_id: endpoint.id().to_string(),
        remote_endpoint_id: args.endpoint_id.to_string(),
        connect_ms: connect_duration.as_secs_f64() * 1_000.0,
        force_relay: config.force_relay,
        paths_before_settle,
        paths_after_settle,
        paths_after_benchmark: snapshot(&connection),
        cancellation,
        iterations: results,
    })?;
    connection.close(0_u32.into(), b"benchmark complete");
    endpoint.close().await;
    Ok(())
}

async fn cancel_upload(
    connection: &Connection,
    payload_bytes: u64,
    delay_ms: u64,
    timeout_seconds: u64,
) -> Result<CancellationResult> {
    ensure!(delay_ms > 0, "cancel delay must be greater than zero");
    let request_id = u64::MAX;
    let header = Header::new(
        MessageKind::Upload,
        request_id,
        payload_bytes,
        digest_for_payload(request_id, payload_bytes),
    );
    let (mut send, mut recv) =
        tokio::time::timeout(Duration::from_secs(timeout_seconds), connection.open_bi())
            .await
            .context("opening cancellation stream timed out")?
            .context("failed to open cancellation stream")?;
    write_header(&mut send, &header).await?;

    let started = Instant::now();
    let upload = tokio::time::timeout(
        Duration::from_millis(delay_ms),
        send_payload(&mut send, request_id, payload_bytes),
    )
    .await;
    ensure!(
        upload.is_err(),
        "cancellation payload completed before the requested delay"
    );
    let _ = send.reset(0xFAB1_u32.into());
    let _ = recv.stop(0xFAB1_u32.into());

    Ok(CancellationResult {
        requested_after_ms: delay_ms,
        observed_after_ms: started.elapsed().as_secs_f64() * 1_000.0,
        payload_bytes,
    })
}
