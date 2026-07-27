//! Minimal rust-libp2p Kademlia adapter for signed swarm catalogue records.
//!
//! Stable public nodes run in server mode and store validated soft state. Intermittent workers
//! behind NAT run in client mode: they query and publish without polluting server routing tables.
//! This is discovery only; all application RPC and model data continue to use Iroh.

use std::{
    collections::HashMap,
    fs::{self, File, OpenOptions},
    io::{ErrorKind, Read, Write},
    net::Ipv4Addr,
    num::NonZeroUsize,
    path::Path,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, anyhow, ensure};
use futures::StreamExt;
use libp2p::{
    Multiaddr, PeerId, SwarmBuilder, identify,
    identity::Keypair,
    kad::{
        self, GetRecordError, GetRecordOk, InboundRequest, Mode, QueryId, QueryResult, Quorum,
        Record, RecordKey,
        store::{MemoryStore, MemoryStoreConfig, RecordStore},
    },
    noise,
    swarm::{NetworkBehaviour, StreamProtocol, SwarmEvent},
    tcp, yamux,
};
use tokio::sync::{mpsc, oneshot};

use crate::catalog::{
    CatalogRecordKind, MAX_CATALOG_SET_BYTES, ValidatedCatalogRecord, keys,
    merge_catalog_membership_values, verify_catalog_membership_value, verify_catalog_record,
};

const DHT_PROTOCOL: StreamProtocol = StreamProtocol::new("/fabi/swarm/kad/3");
const IDENTIFY_PROTOCOL: &str = "/fabi/swarm/identify/3";
const COMMAND_BUFFER: usize = 256;
const DEFAULT_QUERY_TIMEOUT: Duration = Duration::from_secs(15);
const DEFAULT_BOOTSTRAP_INTERVAL: Duration = Duration::from_secs(30);
const DEFAULT_RECORD_TTL: Duration = Duration::from_mins(5);
const DEFAULT_REPLICATION_INTERVAL: Duration = Duration::from_secs(30);
const DEFAULT_MAX_RECORDS: usize = 25_000;
const DEFAULT_MAX_CLOCK_SKEW: Duration = Duration::from_secs(30);

#[derive(NetworkBehaviour)]
struct CatalogueBehaviour {
    identify: identify::Behaviour,
    kad: kad::Behaviour<MemoryStore>,
}

/// Runtime mode described by the libp2p Kademlia specification.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CatalogDhtMode {
    Client,
    Server,
}

/// One bootstrap routing node, with a transport address that excludes `/p2p/<peer-id>`.
#[derive(Clone, Debug)]
pub struct BootstrapPeer {
    pub peer_id: PeerId,
    pub address: Multiaddr,
}

/// Configuration for one embedded catalogue DHT participant.
pub struct CatalogDhtConfig {
    pub keypair: Keypair,
    pub mode: CatalogDhtMode,
    pub listen_address: Multiaddr,
    pub bootstrap_peers: Vec<BootstrapPeer>,
    pub query_timeout: Duration,
    pub bootstrap_interval: Duration,
    pub max_clock_skew: Duration,
    pub max_records: usize,
}

/// Load a stable, separate libp2p discovery identity, creating it if absent.
///
/// The Iroh endpoint key signs application records; this key authenticates only the Kademlia
/// transport. Keeping separate key material prevents accidental cross-protocol key reuse.
///
/// # Errors
///
/// Returns an error when the parent directory or key file cannot be read/written, or when an
/// existing protobuf key is malformed. Malformed identities are never silently replaced.
pub fn load_or_create_dht_keypair(path: &Path) -> Result<Keypair> {
    if path.exists() {
        return load_dht_keypair(path);
    }
    let parent = path
        .parent()
        .context("DHT identity path must have a parent directory")?;
    fs::create_dir_all(parent).with_context(|| {
        format!(
            "failed to create DHT identity directory {}",
            parent.display()
        )
    })?;

    let keypair = Keypair::generate_ed25519();
    let encoded = keypair
        .to_protobuf_encoding()
        .context("failed to encode DHT identity")?;
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    match options.open(path) {
        Ok(mut file) => {
            file.write_all(&encoded)
                .with_context(|| format!("failed to write DHT identity {}", path.display()))?;
            file.sync_all()
                .with_context(|| format!("failed to sync DHT identity {}", path.display()))?;
            Ok(keypair)
        }
        Err(error) if error.kind() == ErrorKind::AlreadyExists => load_dht_keypair(path),
        Err(error) => {
            Err(error).with_context(|| format!("failed to create DHT identity {}", path.display()))
        }
    }
}

fn load_dht_keypair(path: &Path) -> Result<Keypair> {
    let mut encoded = Vec::new();
    File::open(path)
        .with_context(|| format!("failed to open DHT identity {}", path.display()))?
        .read_to_end(&mut encoded)
        .with_context(|| format!("failed to read DHT identity {}", path.display()))?;
    ensure!(
        !encoded.is_empty(),
        "DHT identity {} is empty",
        path.display()
    );
    Keypair::from_protobuf_encoding(&encoded)
        .with_context(|| format!("DHT identity {} is malformed", path.display()))
}

impl CatalogDhtConfig {
    #[must_use]
    pub fn client(keypair: Keypair, bootstrap_peers: Vec<BootstrapPeer>) -> Self {
        let mut listen_address = Multiaddr::empty();
        listen_address.push(libp2p::multiaddr::Protocol::Ip4(Ipv4Addr::LOCALHOST));
        listen_address.push(libp2p::multiaddr::Protocol::Tcp(0));
        Self {
            keypair,
            mode: CatalogDhtMode::Client,
            listen_address,
            bootstrap_peers,
            query_timeout: DEFAULT_QUERY_TIMEOUT,
            bootstrap_interval: DEFAULT_BOOTSTRAP_INTERVAL,
            max_clock_skew: DEFAULT_MAX_CLOCK_SKEW,
            max_records: DEFAULT_MAX_RECORDS,
        }
    }

    #[must_use]
    pub fn server(keypair: Keypair, listen_address: Multiaddr) -> Self {
        Self {
            keypair,
            mode: CatalogDhtMode::Server,
            listen_address,
            bootstrap_peers: Vec::new(),
            query_timeout: DEFAULT_QUERY_TIMEOUT,
            bootstrap_interval: DEFAULT_BOOTSTRAP_INTERVAL,
            max_clock_skew: DEFAULT_MAX_CLOCK_SKEW,
            max_records: DEFAULT_MAX_RECORDS,
        }
    }
}

enum Command {
    Bootstrap {
        reply: oneshot::Sender<Result<()>>,
    },
    Put {
        logical_key: String,
        encoded: Vec<u8>,
        quorum: NonZeroUsize,
        reply: oneshot::Sender<Result<()>>,
    },
    Get {
        logical_key: String,
        reply: oneshot::Sender<Result<ValidatedCatalogRecord>>,
    },
    GetMembers {
        logical_key: String,
        reply: oneshot::Sender<Result<Vec<ValidatedCatalogRecord>>>,
    },
    Shutdown {
        reply: oneshot::Sender<()>,
    },
}

enum PendingQuery {
    Bootstrap(oneshot::Sender<Result<()>>),
    Put(oneshot::Sender<Result<()>>),
    Get {
        logical_key: String,
        best: Option<ValidatedCatalogRecord>,
        reply: oneshot::Sender<Result<ValidatedCatalogRecord>>,
    },
    GetMembers {
        logical_key: String,
        values: Vec<Vec<u8>>,
        reply: oneshot::Sender<Result<Vec<ValidatedCatalogRecord>>>,
    },
}

/// Async command handle for one running DHT event loop.
#[derive(Clone)]
pub struct CatalogDhtHandle {
    peer_id: PeerId,
    listen_address: Multiaddr,
    commands: mpsc::Sender<Command>,
}

impl CatalogDhtHandle {
    #[must_use]
    pub fn peer_id(&self) -> PeerId {
        self.peer_id
    }

    #[must_use]
    pub fn listen_address(&self) -> &Multiaddr {
        &self.listen_address
    }

    /// Populate the local routing table from the configured bootstrap peers.
    ///
    /// # Errors
    ///
    /// Returns an error when there are no routing peers, the DHT query fails, or the event loop
    /// has stopped.
    pub async fn bootstrap(&self) -> Result<()> {
        let (reply, receive) = oneshot::channel();
        self.commands
            .send(Command::Bootstrap { reply })
            .await
            .context("catalogue DHT event loop stopped")?;
        receive.await.context("catalogue bootstrap reply dropped")?
    }

    /// Replicate one already-signed soft-state record to the closest DHT servers.
    ///
    /// # Errors
    ///
    /// Returns an error when local validation, key binding, quorum, or network publication fails.
    pub async fn put(
        &self,
        logical_key: impl Into<String>,
        encoded: Vec<u8>,
        quorum: NonZeroUsize,
    ) -> Result<()> {
        let (reply, receive) = oneshot::channel();
        self.commands
            .send(Command::Put {
                logical_key: logical_key.into(),
                encoded,
                quorum,
                reply,
            })
            .await
            .context("catalogue DHT event loop stopped")?;
        receive.await.context("catalogue put reply dropped")?
    }

    /// Fetch all responses and return the highest valid sequence for one logical key.
    ///
    /// # Errors
    ///
    /// Returns an error when no valid live record is found or the event loop has stopped.
    pub async fn get(&self, logical_key: impl Into<String>) -> Result<ValidatedCatalogRecord> {
        let (reply, receive) = oneshot::channel();
        self.commands
            .send(Command::Get {
                logical_key: logical_key.into(),
                reply,
            })
            .await
            .context("catalogue DHT event loop stopped")?;
        receive.await.context("catalogue get reply dropped")?
    }

    /// Fetch and merge all independently signed live members of one model shard.
    ///
    /// # Errors
    ///
    /// Returns an error when the shard has no valid live members or the event loop has stopped.
    pub async fn get_members(
        &self,
        logical_key: impl Into<String>,
    ) -> Result<Vec<ValidatedCatalogRecord>> {
        let (reply, receive) = oneshot::channel();
        self.commands
            .send(Command::GetMembers {
                logical_key: logical_key.into(),
                reply,
            })
            .await
            .context("catalogue DHT event loop stopped")?;
        receive
            .await
            .context("catalogue membership lookup reply dropped")?
    }

    /// Fetch all deterministic membership shards for one model with bounded concurrency.
    ///
    /// # Errors
    ///
    /// Returns an error if any shard lookup fails, preventing a partial network view from being
    /// mistaken for a complete discovery snapshot.
    pub async fn get_model_members(
        &self,
        model_swarm_id: &str,
    ) -> Result<Vec<ValidatedCatalogRecord>> {
        let lookups = futures::stream::iter(keys::all_model_membership_shards(model_swarm_id))
            .map(|logical_key| {
                let handle = self.clone();
                async move { handle.get_members(logical_key).await }
            })
            .buffer_unordered(16)
            .collect::<Vec<_>>()
            .await;
        let mut members = Vec::new();
        for lookup in lookups {
            members.extend(lookup?);
        }
        members.sort_by_key(|record| record.publisher_endpoint_id.to_string());
        Ok(members)
    }

    /// Stop the DHT event loop and close its transports.
    ///
    /// # Errors
    ///
    /// Returns an error when the event loop has already stopped.
    pub async fn shutdown(&self) -> Result<()> {
        let (reply, receive) = oneshot::channel();
        self.commands
            .send(Command::Shutdown { reply })
            .await
            .context("catalogue DHT event loop stopped")?;
        receive.await.context("catalogue shutdown reply dropped")
    }
}

fn now_ms() -> Result<u64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .context("system clock is before the Unix epoch")?
        .as_millis();
    millis
        .try_into()
        .context("system clock milliseconds exceed u64")
}

fn validate_value(
    expected_key: &str,
    encoded: &[u8],
    max_clock_skew: Duration,
) -> Result<ValidatedCatalogRecord> {
    let skew_ms = max_clock_skew
        .as_millis()
        .try_into()
        .context("clock skew exceeds u64 milliseconds")?;
    let validated = verify_catalog_record(encoded, now_ms()?, skew_ms)?;
    ensure!(
        validated.logical_key == expected_key,
        "Kademlia key does not match signed catalogue key"
    );
    Ok(validated)
}

fn instant_expiry(expires_at_ms: u64) -> Result<Instant> {
    let remaining = expires_at_ms
        .checked_sub(now_ms()?)
        .context("catalogue record expired before DHT publication")?;
    ensure!(
        remaining > 0,
        "catalogue record expired before DHT publication"
    );
    Ok(Instant::now() + Duration::from_millis(remaining))
}

fn membership_expiry(
    expected_key: &str,
    encoded: &[u8],
    max_clock_skew: Duration,
) -> Result<Instant> {
    let skew_ms = max_clock_skew
        .as_millis()
        .try_into()
        .context("clock skew exceeds u64 milliseconds")?;
    let latest_expiry = verify_catalog_membership_value(expected_key, encoded, now_ms()?, skew_ms)?
        .into_iter()
        .map(|record| record.expires_at_ms)
        .max()
        .context("catalogue membership set is empty")?;
    instant_expiry(latest_expiry)
}

fn merge_membership_inbound(
    behaviour: &mut CatalogueBehaviour,
    record: Record,
    expected_key: &str,
    max_clock_skew: Duration,
) {
    let current = behaviour
        .kad
        .store_mut()
        .get(&record.key)
        .map(|stored| stored.into_owned().value);
    let Ok(skew_ms) = max_clock_skew.as_millis().try_into() else {
        return;
    };
    let Ok(timestamp) = now_ms() else {
        return;
    };
    let merged = match current.as_deref() {
        Some(existing) => merge_catalog_membership_values(
            expected_key,
            [existing, record.value.as_slice()],
            timestamp,
            skew_ms,
        ),
        None => merge_catalog_membership_values(
            expected_key,
            [record.value.as_slice()],
            timestamp,
            skew_ms,
        ),
    };
    let Ok(value) = merged else {
        return;
    };
    let Ok(expires) = membership_expiry(expected_key, &value, max_clock_skew) else {
        return;
    };
    let merged_record = Record {
        key: record.key,
        value,
        publisher: None,
        expires: Some(expires),
    };
    if let Err(error) = behaviour.kad.store_mut().put(merged_record) {
        tracing::warn!(%error, "validated catalogue membership set could not be stored");
    }
}

fn maybe_store_inbound(
    behaviour: &mut CatalogueBehaviour,
    record: Record,
    max_clock_skew: Duration,
) {
    let Ok(expected_key) = std::str::from_utf8(record.key.as_ref()).map(str::to_owned) else {
        return;
    };
    if let Ok(incoming) = validate_value(&expected_key, &record.value, max_clock_skew) {
        if incoming.kind == CatalogRecordKind::ModelMember {
            merge_membership_inbound(behaviour, record, &expected_key, max_clock_skew);
            return;
        }

        if let Some(current_record) = behaviour.kad.store_mut().get(&record.key) {
            let current_record = current_record.into_owned();
            if let Ok(current) =
                validate_value(&expected_key, &current_record.value, max_clock_skew)
                && current.sequence >= incoming.sequence
            {
                return;
            }
        }
        if let Err(error) = behaviour.kad.store_mut().put(record) {
            tracing::warn!(%error, "validated catalogue record could not be stored");
        }
        return;
    }

    let (Ok(timestamp), Ok(skew_ms)) = (now_ms(), max_clock_skew.as_millis().try_into()) else {
        return;
    };
    if verify_catalog_membership_value(&expected_key, &record.value, timestamp, skew_ms).is_ok() {
        merge_membership_inbound(behaviour, record, &expected_key, max_clock_skew);
    }
}

fn choose_best(
    current: Option<ValidatedCatalogRecord>,
    candidate: ValidatedCatalogRecord,
) -> ValidatedCatalogRecord {
    match current {
        Some(previous) if previous.sequence >= candidate.sequence => previous,
        _ => candidate,
    }
}

/// Start one encrypted Kademlia participant and return after its TCP listener is ready.
///
/// # Errors
///
/// Returns an error when transport construction, behaviour construction, or listening fails.
pub async fn spawn_catalog_dht(config: CatalogDhtConfig) -> Result<CatalogDhtHandle> {
    ensure!(config.max_records > 0, "max_records must be positive");
    ensure!(
        !config.bootstrap_interval.is_zero(),
        "bootstrap_interval must be positive"
    );
    let local_peer_id = config.keypair.public().to_peer_id();
    let mode = config.mode;
    let query_timeout = config.query_timeout;
    let max_clock_skew = config.max_clock_skew;
    let max_records = config.max_records;

    let mut swarm = SwarmBuilder::with_existing_identity(config.keypair)
        .with_tokio()
        .with_tcp(
            tcp::Config::default(),
            noise::Config::new,
            yamux::Config::default,
        )?
        .with_dns()?
        .with_behaviour(move |key| {
            let peer_id = key.public().to_peer_id();
            let mut kad_config = kad::Config::new(DHT_PROTOCOL);
            kad_config
                .set_query_timeout(query_timeout)
                .set_periodic_bootstrap_interval(Some(config.bootstrap_interval))
                .set_record_ttl(Some(DEFAULT_RECORD_TTL))
                .set_replication_interval(Some(DEFAULT_REPLICATION_INTERVAL))
                .set_publication_interval(None)
                .set_record_filtering(kad::StoreInserts::FilterBoth)
                .set_max_packet_size(MAX_CATALOG_SET_BYTES + 1024);
            let store = MemoryStore::with_config(
                peer_id,
                MemoryStoreConfig {
                    max_records,
                    max_value_bytes: MAX_CATALOG_SET_BYTES + 1,
                    ..MemoryStoreConfig::default()
                },
            );
            CatalogueBehaviour {
                identify: identify::Behaviour::new(identify::Config::new(
                    IDENTIFY_PROTOCOL.to_owned(),
                    key.public(),
                )),
                kad: kad::Behaviour::with_config(peer_id, store, kad_config),
            }
        })?
        .build();

    swarm.behaviour_mut().kad.set_mode(Some(match mode {
        CatalogDhtMode::Client => Mode::Client,
        CatalogDhtMode::Server => Mode::Server,
    }));
    for bootstrap in config.bootstrap_peers {
        swarm
            .behaviour_mut()
            .kad
            .add_address(&bootstrap.peer_id, bootstrap.address);
    }
    swarm
        .listen_on(config.listen_address)
        .context("failed to listen for catalogue DHT")?;

    let listen_address = loop {
        match swarm.select_next_some().await {
            SwarmEvent::NewListenAddr { address, .. } => break address,
            SwarmEvent::ListenerError { error, .. } => {
                return Err(error).context("catalogue DHT listener failed");
            }
            _ => {}
        }
    };
    let advertised_address = listen_address
        .clone()
        .with_p2p(local_peer_id)
        .map_err(|address| {
            anyhow!("catalogue listen address already contains a peer id: {address}")
        })?;
    let (commands, receiver) = mpsc::channel(COMMAND_BUFFER);
    tokio::spawn(run_event_loop(swarm, receiver, max_clock_skew));

    Ok(CatalogDhtHandle {
        peer_id: local_peer_id,
        listen_address: advertised_address,
        commands,
    })
}

async fn run_event_loop(
    mut swarm: libp2p::Swarm<CatalogueBehaviour>,
    mut commands: mpsc::Receiver<Command>,
    max_clock_skew: Duration,
) {
    let mut pending: HashMap<QueryId, PendingQuery> = HashMap::new();
    loop {
        tokio::select! {
            command = commands.recv() => {
                let Some(command) = command else { break };
                if handle_command(command, &mut swarm, &mut pending, max_clock_skew) {
                    break;
                }
            }
            event = swarm.select_next_some() => {
                handle_swarm_event(event, &mut swarm, &mut pending, max_clock_skew);
            }
        }
    }
    for (_, query) in pending {
        match query {
            PendingQuery::Bootstrap(reply) | PendingQuery::Put(reply) => {
                let _ = reply.send(Err(anyhow!(
                    "catalogue DHT stopped before query completion"
                )));
            }
            PendingQuery::Get { reply, .. } => {
                let _ = reply.send(Err(anyhow!(
                    "catalogue DHT stopped before query completion"
                )));
            }
            PendingQuery::GetMembers { reply, .. } => {
                let _ = reply.send(Err(anyhow!(
                    "catalogue DHT stopped before membership lookup completion"
                )));
            }
        }
    }
}

fn current_record_for_publisher(
    logical_key: &str,
    encoded: &[u8],
    publisher: iroh::EndpointId,
    max_clock_skew: Duration,
) -> Option<ValidatedCatalogRecord> {
    validate_value(logical_key, encoded, max_clock_skew)
        .ok()
        .filter(|record| record.publisher_endpoint_id == publisher)
        .or_else(|| {
            let skew_ms = max_clock_skew.as_millis().try_into().ok()?;
            verify_catalog_membership_value(logical_key, encoded, now_ms().ok()?, skew_ms)
                .ok()?
                .into_iter()
                .find(|record| record.publisher_endpoint_id == publisher)
        })
}

fn start_put(
    swarm: &mut libp2p::Swarm<CatalogueBehaviour>,
    logical_key: &str,
    encoded: Vec<u8>,
    quorum: NonZeroUsize,
    max_clock_skew: Duration,
) -> Result<Option<QueryId>> {
    let validated = validate_value(logical_key, &encoded, max_clock_skew)?;
    let record_key = RecordKey::new(&logical_key);
    if let Some(current_record) = swarm.behaviour_mut().kad.store_mut().get(&record_key) {
        let current_record = current_record.into_owned();
        if let Some(current) = current_record_for_publisher(
            logical_key,
            &current_record.value,
            validated.publisher_endpoint_id,
            max_clock_skew,
        ) {
            ensure!(
                validated.sequence >= current.sequence,
                "catalogue publication sequence would roll back local soft state"
            );
            if validated.sequence == current.sequence {
                ensure!(
                    validated == current,
                    "catalogue publication reuses a sequence with different contents"
                );
                return Ok(None);
            }
        }
    }
    let record = Record {
        key: record_key,
        value: encoded,
        publisher: Some(*swarm.local_peer_id()),
        expires: Some(instant_expiry(validated.expires_at_ms)?),
    };
    swarm
        .behaviour_mut()
        .kad
        .put_record(record, Quorum::N(quorum))
        .map(Some)
        .map_err(anyhow::Error::from)
}

fn handle_command(
    command: Command,
    swarm: &mut libp2p::Swarm<CatalogueBehaviour>,
    pending: &mut HashMap<QueryId, PendingQuery>,
    max_clock_skew: Duration,
) -> bool {
    match command {
        Command::Bootstrap { reply } => match swarm.behaviour_mut().kad.bootstrap() {
            Ok(query_id) => {
                pending.insert(query_id, PendingQuery::Bootstrap(reply));
            }
            Err(error) => {
                let _ = reply.send(Err(anyhow!(error).context("catalogue bootstrap failed")));
            }
        },
        Command::Put {
            logical_key,
            encoded,
            quorum,
            reply,
        } => {
            let result = start_put(swarm, &logical_key, encoded, quorum, max_clock_skew);
            match result {
                Ok(Some(query_id)) => {
                    pending.insert(query_id, PendingQuery::Put(reply));
                }
                Ok(None) => {
                    let _ = reply.send(Ok(()));
                }
                Err(error) => {
                    let _ = reply.send(Err(error.context("catalogue publication failed")));
                }
            }
        }
        Command::Get { logical_key, reply } => {
            let query_id = swarm
                .behaviour_mut()
                .kad
                .get_record(RecordKey::new(&logical_key));
            pending.insert(
                query_id,
                PendingQuery::Get {
                    logical_key,
                    best: None,
                    reply,
                },
            );
        }
        Command::GetMembers { logical_key, reply } => {
            let query_id = swarm
                .behaviour_mut()
                .kad
                .get_record(RecordKey::new(&logical_key));
            pending.insert(
                query_id,
                PendingQuery::GetMembers {
                    logical_key,
                    values: Vec::new(),
                    reply,
                },
            );
        }
        Command::Shutdown { reply } => {
            let _ = reply.send(());
            return true;
        }
    }
    false
}

fn handle_swarm_event(
    event: SwarmEvent<CatalogueBehaviourEvent>,
    swarm: &mut libp2p::Swarm<CatalogueBehaviour>,
    pending: &mut HashMap<QueryId, PendingQuery>,
    max_clock_skew: Duration,
) {
    match event {
        SwarmEvent::Behaviour(CatalogueBehaviourEvent::Identify(identify::Event::Received {
            peer_id,
            info,
            ..
        })) => {
            for address in info.listen_addrs {
                swarm.behaviour_mut().kad.add_address(&peer_id, address);
            }
        }
        SwarmEvent::Behaviour(CatalogueBehaviourEvent::Kad(kad::Event::InboundRequest {
            request:
                InboundRequest::PutRecord {
                    record: Some(record),
                    ..
                },
        })) => maybe_store_inbound(swarm.behaviour_mut(), record, max_clock_skew),
        SwarmEvent::Behaviour(CatalogueBehaviourEvent::Kad(
            kad::Event::OutboundQueryProgressed {
                id, result, step, ..
            },
        )) => handle_query_progress(id, result, step.last, pending, max_clock_skew),
        _ => {}
    }
}

fn finish_membership_lookup(
    logical_key: &str,
    values: &[Vec<u8>],
    result: QueryResult,
    max_clock_skew: Duration,
) -> Result<Vec<ValidatedCatalogRecord>> {
    if values.is_empty() {
        return match result {
            QueryResult::GetRecord(Ok(_) | Err(GetRecordError::NotFound { .. })) => Ok(Vec::new()),
            QueryResult::GetRecord(Err(network_error)) => {
                Err(anyhow!(network_error).context("catalogue membership lookup failed"))
            }
            _ => Err(anyhow!("unexpected catalogue membership query result")),
        };
    }
    let timestamp = now_ms()?;
    let skew_ms = max_clock_skew
        .as_millis()
        .try_into()
        .context("clock skew exceeds u64 milliseconds")?;
    let merged = merge_catalog_membership_values(
        logical_key,
        values.iter().map(Vec::as_slice),
        timestamp,
        skew_ms,
    )?;
    verify_catalog_membership_value(logical_key, &merged, timestamp, skew_ms)
}

fn handle_query_progress(
    query_id: QueryId,
    result: QueryResult,
    last: bool,
    pending: &mut HashMap<QueryId, PendingQuery>,
    max_clock_skew: Duration,
) {
    let Some(query) = pending.get_mut(&query_id) else {
        return;
    };
    match (query, result) {
        (PendingQuery::Bootstrap(_), QueryResult::Bootstrap(result)) if last => {
            let PendingQuery::Bootstrap(reply) = pending.remove(&query_id).expect("query exists")
            else {
                unreachable!()
            };
            let _ = reply.send(result.map(|_| ()).map_err(anyhow::Error::from));
        }
        (PendingQuery::Put(_), QueryResult::PutRecord(result)) if last => {
            let PendingQuery::Put(reply) = pending.remove(&query_id).expect("query exists") else {
                unreachable!()
            };
            let _ = reply.send(result.map(|_| ()).map_err(anyhow::Error::from));
        }
        (
            PendingQuery::Get {
                logical_key, best, ..
            },
            QueryResult::GetRecord(Ok(GetRecordOk::FoundRecord(peer_record))),
        ) => {
            if let Ok(candidate) =
                validate_value(logical_key, &peer_record.record.value, max_clock_skew)
            {
                *best = Some(choose_best(best.take(), candidate));
            }
        }
        (PendingQuery::Get { .. }, QueryResult::GetRecord(result)) if last => {
            let PendingQuery::Get {
                logical_key,
                best,
                reply,
            } = pending.remove(&query_id).expect("query exists")
            else {
                unreachable!()
            };
            let answer = best.ok_or_else(|| match result {
                Err(error) => anyhow!(error).context("catalogue lookup failed"),
                Ok(_) => anyhow!("no valid live catalogue record found for {logical_key}"),
            });
            let _ = reply.send(answer);
        }
        (
            PendingQuery::GetMembers {
                logical_key,
                values,
                ..
            },
            QueryResult::GetRecord(Ok(GetRecordOk::FoundRecord(peer_record))),
        ) => {
            let skew_ms = max_clock_skew.as_millis().try_into().ok();
            if let (Ok(timestamp), Some(skew_ms)) = (now_ms(), skew_ms)
                && verify_catalog_membership_value(
                    logical_key,
                    &peer_record.record.value,
                    timestamp,
                    skew_ms,
                )
                .is_ok()
            {
                values.push(peer_record.record.value);
            }
        }
        (PendingQuery::GetMembers { .. }, QueryResult::GetRecord(result)) if last => {
            let PendingQuery::GetMembers {
                logical_key,
                values,
                reply,
            } = pending.remove(&query_id).expect("query exists")
            else {
                unreachable!()
            };
            let answer = finish_membership_lookup(
                &logical_key,
                &values,
                QueryResult::GetRecord(result),
                max_clock_skew,
            )
            .map_err(|error| error.context(format!("membership shard {logical_key}")));
            let _ = reply.send(answer);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{CatalogRecordKind, CatalogRecordParams, keys, sign_catalog_record};
    use iroh::SecretKey;

    #[test]
    fn dht_transport_identity_is_stable_and_malformed_files_fail_closed() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("discovery.key");
        let first = load_or_create_dht_keypair(&path)?;
        let second = load_or_create_dht_keypair(&path)?;
        assert_eq!(first.public().to_peer_id(), second.public().to_peer_id());

        let malformed = directory.path().join("malformed.key");
        fs::write(&malformed, b"not a protobuf private key")?;
        let error = load_or_create_dht_keypair(&malformed)
            .expect_err("malformed identity must not be replaced");
        assert!(format!("{error:#}").contains("malformed"));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn client_publishes_through_server_and_another_client_reads() -> Result<()> {
        let server = spawn_catalog_dht(CatalogDhtConfig::server(
            Keypair::generate_ed25519(),
            "/ip4/127.0.0.1/tcp/0".parse()?,
        ))
        .await?;
        let mut server_address = server.listen_address().clone();
        let peer_suffix = server_address.pop();
        ensure!(
            matches!(peer_suffix, Some(libp2p::multiaddr::Protocol::P2p(_))),
            "server address has no /p2p component"
        );
        let bootstrap = BootstrapPeer {
            peer_id: server.peer_id(),
            address: server_address,
        };
        let writer = spawn_catalog_dht(CatalogDhtConfig::client(
            Keypair::generate_ed25519(),
            vec![bootstrap.clone()],
        ))
        .await?;
        writer.bootstrap().await?;

        let publisher = SecretKey::generate();
        let logical_key = keys::worker_offer(&publisher.public());
        let issued_at_ms = now_ms()?;
        let signed = sign_catalog_record(
            &publisher,
            CatalogRecordParams {
                kind: CatalogRecordKind::WorkerOffer,
                logical_key: &logical_key,
                discovery_peer_id: &writer.peer_id().to_string(),
                sequence: 9,
                issued_at_ms,
                expires_at_ms: issued_at_ms + 60_000,
                payload: br#"{"worker":"ready"}"#,
            },
        )?;
        writer
            .put(
                &logical_key,
                signed.clone(),
                NonZeroUsize::new(1).expect("one"),
            )
            .await?;
        writer
            .put(&logical_key, signed, NonZeroUsize::new(1).expect("one"))
            .await?;

        let stale = sign_catalog_record(
            &publisher,
            CatalogRecordParams {
                kind: CatalogRecordKind::WorkerOffer,
                logical_key: &logical_key,
                discovery_peer_id: &writer.peer_id().to_string(),
                sequence: 8,
                issued_at_ms,
                expires_at_ms: issued_at_ms + 60_000,
                payload: b"stale",
            },
        )?;
        let stale_error = writer
            .put(&logical_key, stale, NonZeroUsize::new(1).expect("one"))
            .await
            .expect_err("publisher must reject its own rollback");
        assert!(format!("{stale_error:#}").contains("roll back"));

        let reader = spawn_catalog_dht(CatalogDhtConfig::client(
            Keypair::generate_ed25519(),
            vec![bootstrap],
        ))
        .await?;
        reader.bootstrap().await?;
        let found = reader.get(&logical_key).await?;
        assert_eq!(found.sequence, 9);
        assert_eq!(found.payload, br#"{"worker":"ready"}"#);

        reader.shutdown().await?;
        writer.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn client_reconnects_after_bootstrap_server_restart() -> Result<()> {
        let server_key = Keypair::generate_ed25519();
        let server = spawn_catalog_dht(CatalogDhtConfig::server(
            server_key.clone(),
            "/ip4/127.0.0.1/tcp/0".parse()?,
        ))
        .await?;
        let mut server_address = server.listen_address().clone();
        let peer_suffix = server_address.pop();
        ensure!(
            matches!(peer_suffix, Some(libp2p::multiaddr::Protocol::P2p(_))),
            "server address has no /p2p component"
        );
        let bootstrap = BootstrapPeer {
            peer_id: server.peer_id(),
            address: server_address.clone(),
        };
        let mut writer_config =
            CatalogDhtConfig::client(Keypair::generate_ed25519(), vec![bootstrap]);
        writer_config.bootstrap_interval = Duration::from_millis(250);
        writer_config.query_timeout = Duration::from_secs(1);
        let writer = spawn_catalog_dht(writer_config).await?;
        writer.bootstrap().await?;

        server.shutdown().await?;
        let publisher = SecretKey::generate();
        let logical_key = keys::worker_offer(&publisher.public());
        let disconnected_at_ms = now_ms()?;
        let disconnected_record = sign_catalog_record(
            &publisher,
            CatalogRecordParams {
                kind: CatalogRecordKind::WorkerOffer,
                logical_key: &logical_key,
                discovery_peer_id: &writer.peer_id().to_string(),
                sequence: 1,
                issued_at_ms: disconnected_at_ms,
                expires_at_ms: disconnected_at_ms + 60_000,
                payload: br#"{"worker":"disconnected"}"#,
            },
        )?;
        writer
            .put(
                &logical_key,
                disconnected_record,
                NonZeroUsize::new(1).expect("one"),
            )
            .await
            .expect_err("publication must fail while the only routing server is down");

        let replacement =
            spawn_catalog_dht(CatalogDhtConfig::server(server_key, server_address)).await?;

        let deadline = tokio::time::Instant::now() + Duration::from_secs(8);
        let mut sequence = 2;
        loop {
            let issued_at_ms = now_ms()?;
            let signed = sign_catalog_record(
                &publisher,
                CatalogRecordParams {
                    kind: CatalogRecordKind::WorkerOffer,
                    logical_key: &logical_key,
                    discovery_peer_id: &writer.peer_id().to_string(),
                    sequence,
                    issued_at_ms,
                    expires_at_ms: issued_at_ms + 60_000,
                    payload: br#"{"worker":"reconnected"}"#,
                },
            )?;
            if writer
                .put(&logical_key, signed, NonZeroUsize::new(1).expect("one"))
                .await
                .is_ok()
            {
                break;
            }
            ensure!(
                tokio::time::Instant::now() < deadline,
                "client did not reconnect through periodic Kademlia bootstrap"
            );
            sequence += 1;
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let found = replacement.get(&logical_key).await?;
        assert_eq!(found.sequence, sequence);
        assert_eq!(found.payload, br#"{"worker":"reconnected"}"#);

        writer.shutdown().await?;
        replacement.shutdown().await?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn routing_server_merges_independent_model_members_for_reader() -> Result<()> {
        let server = spawn_catalog_dht(CatalogDhtConfig::server(
            Keypair::generate_ed25519(),
            "/ip4/127.0.0.1/tcp/0".parse()?,
        ))
        .await?;
        let mut server_address = server.listen_address().clone();
        ensure!(
            matches!(
                server_address.pop(),
                Some(libp2p::multiaddr::Protocol::P2p(_))
            ),
            "server address has no /p2p component"
        );
        let bootstrap = BootstrapPeer {
            peer_id: server.peer_id(),
            address: server_address,
        };
        let first_writer = spawn_catalog_dht(CatalogDhtConfig::client(
            Keypair::generate_ed25519(),
            vec![bootstrap.clone()],
        ))
        .await?;
        let second_writer = spawn_catalog_dht(CatalogDhtConfig::client(
            Keypair::generate_ed25519(),
            vec![bootstrap.clone()],
        ))
        .await?;
        first_writer.bootstrap().await?;
        second_writer.bootstrap().await?;

        let model = "c".repeat(64);
        let first_publisher = SecretKey::generate();
        let shard = crate::catalog::membership_shard(&first_publisher.public());
        let second_publisher = (0..10_000)
            .map(|_| SecretKey::generate())
            .find(|key| crate::catalog::membership_shard(&key.public()) == shard)
            .context("failed to find two endpoint ids in one membership shard")?;
        let logical_key = keys::model_member(&model, &first_publisher.public());
        let timestamp = now_ms()?;
        let sign_member = |publisher: &SecretKey, peer_id: PeerId, sequence| {
            sign_catalog_record(
                publisher,
                CatalogRecordParams {
                    kind: CatalogRecordKind::ModelMember,
                    logical_key: &logical_key,
                    discovery_peer_id: &peer_id.to_string(),
                    sequence,
                    issued_at_ms: timestamp,
                    expires_at_ms: timestamp + 60_000,
                    payload: b"member",
                },
            )
        };
        first_writer
            .put(
                &logical_key,
                sign_member(&first_publisher, first_writer.peer_id(), 2)?,
                NonZeroUsize::new(1).expect("one"),
            )
            .await?;
        second_writer
            .put(
                &logical_key,
                sign_member(&second_publisher, second_writer.peer_id(), 7)?,
                NonZeroUsize::new(1).expect("one"),
            )
            .await?;

        let reader = spawn_catalog_dht(CatalogDhtConfig::client(
            Keypair::generate_ed25519(),
            vec![bootstrap],
        ))
        .await?;
        reader.bootstrap().await?;
        let members = reader.get_members(&logical_key).await?;
        assert_eq!(members.len(), 2);
        let mut sequences = members
            .iter()
            .map(|member| member.sequence)
            .collect::<Vec<_>>();
        sequences.sort_unstable();
        assert_eq!(sequences, vec![2, 7]);
        let all_model_members = reader.get_model_members(&model).await?;
        assert_eq!(all_model_members.len(), 2);

        reader.shutdown().await?;
        second_writer.shutdown().await?;
        first_writer.shutdown().await?;
        server.shutdown().await?;
        Ok(())
    }

    #[test]
    fn highest_sequence_wins_independent_of_arrival_order() {
        let endpoint = SecretKey::generate().public();
        let record = |sequence| ValidatedCatalogRecord {
            kind: crate::catalog::CatalogRecordKind::WorkerOffer,
            logical_key: keys::worker_offer(&endpoint),
            publisher_endpoint_id: endpoint,
            discovery_peer_id: "peer".to_owned(),
            sequence,
            issued_at_ms: 1,
            expires_at_ms: 2,
            payload: sequence.to_le_bytes().to_vec(),
        };
        let best = choose_best(None, record(10));
        let best = choose_best(Some(best), record(3));
        assert_eq!(best.sequence, 10);
    }
}
