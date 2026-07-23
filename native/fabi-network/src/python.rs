use std::{
    collections::HashMap,
    path::PathBuf,
    str::FromStr,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, AtomicU64, Ordering},
        mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel},
    },
    time::Duration,
};

use anyhow::{Context, Result, bail, ensure};
use iroh::{
    Endpoint, EndpointAddr, EndpointId, RelayUrl, SecretKey, TransportAddr, endpoint::Connection,
};
use libp2p::{Multiaddr, multiaddr::Protocol};
use pyo3::{
    exceptions::{PyRuntimeError, PyTimeoutError},
    prelude::*,
    types::{PyBytes, PyModule},
};
use tokio::{
    runtime::Runtime,
    sync::{mpsc as tokio_mpsc, oneshot},
};
use tracing::{debug, warn};

use crate::{
    ALPN,
    catalog::{
        self, CatalogRecordKind, CatalogRecordParams, ValidatedCatalogRecord, sign_catalog_record,
        verify_catalog_record,
    },
    catalog_dht::{
        BootstrapPeer, CatalogDhtConfig, CatalogDhtHandle, load_or_create_dht_keypair,
        spawn_catalog_dht,
    },
    endpoint::{EndpointConfig, bind},
    identity,
    protocol::{
        DEFAULT_MAX_PAYLOAD, Header, MessageKind, decode_rpc_request, encode_rpc_request,
        read_and_verify_payload, read_header, write_header,
    },
    telemetry,
};

const INBOUND_QUEUE_CAPACITY: usize = 1024;
const STREAM_QUEUE_CAPACITY: usize = 16;
const DEFAULT_RPC_TIMEOUT_SECONDS: u64 = 600;
const DEFAULT_RPC_TIMEOUT_SECONDS_F64: f64 = 600.0;
const STREAM_CANCEL_CODE: u32 = 0xFAB1;

type RpcResponse = std::result::Result<Vec<u8>, String>;
type ConnectionCache = Arc<tokio::sync::Mutex<HashMap<EndpointId, Connection>>>;

#[pyclass(name = "CatalogRecord", frozen)]
struct PyCatalogRecord {
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    logical_key: String,
    #[pyo3(get)]
    publisher_endpoint_id: String,
    #[pyo3(get)]
    discovery_peer_id: String,
    #[pyo3(get)]
    sequence: u64,
    #[pyo3(get)]
    issued_at_ms: u64,
    #[pyo3(get)]
    expires_at_ms: u64,
    payload: Vec<u8>,
}

#[pymethods]
impl PyCatalogRecord {
    #[getter]
    fn payload<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.payload)
    }
}

impl From<ValidatedCatalogRecord> for PyCatalogRecord {
    fn from(record: ValidatedCatalogRecord) -> Self {
        Self {
            kind: catalog_kind_name(record.kind).to_owned(),
            logical_key: record.logical_key,
            publisher_endpoint_id: record.publisher_endpoint_id.to_string(),
            discovery_peer_id: record.discovery_peer_id,
            sequence: record.sequence,
            issued_at_ms: record.issued_at_ms,
            expires_at_ms: record.expires_at_ms,
            payload: record.payload,
        }
    }
}

fn catalog_kind_name(kind: CatalogRecordKind) -> &'static str {
    match kind {
        CatalogRecordKind::ModelManifest => "model_manifest",
        CatalogRecordKind::WorkerOffer => "worker_offer",
        CatalogRecordKind::SpanLease => "span_lease",
        CatalogRecordKind::LinkMetric => "link_metric",
        CatalogRecordKind::ModelMember => "model_member",
        CatalogRecordKind::Unspecified => "unspecified",
    }
}

fn parse_catalog_kind(kind: &str) -> PyResult<CatalogRecordKind> {
    match kind {
        "model_manifest" => Ok(CatalogRecordKind::ModelManifest),
        "worker_offer" => Ok(CatalogRecordKind::WorkerOffer),
        "span_lease" => Ok(CatalogRecordKind::SpanLease),
        "link_metric" => Ok(CatalogRecordKind::LinkMetric),
        "model_member" => Ok(CatalogRecordKind::ModelMember),
        _ => Err(PyRuntimeError::new_err(format!(
            "unsupported catalogue record kind {kind:?}"
        ))),
    }
}

#[derive(Clone)]
struct InboundDispatcher {
    sender: SyncSender<InboundRequest>,
    max_payload: u64,
    response_timeout: Duration,
}

impl InboundDispatcher {
    fn spawn(&self, connection: Connection) {
        tokio::spawn(handle_rpc_connection(
            connection,
            self.sender.clone(),
            self.max_payload,
            self.response_timeout,
        ));
    }
}

enum InboundResponse {
    Unary(oneshot::Sender<RpcResponse>),
    Stream(tokio_mpsc::Sender<RpcResponse>),
}

impl InboundResponse {
    fn is_stream(&self) -> bool {
        matches!(self, Self::Stream(_))
    }

    fn reject(self, message: String) {
        match self {
            Self::Unary(sender) => {
                let _ = sender.send(Err(message));
            }
            Self::Stream(sender) => {
                let _ = sender.try_send(Err(message));
            }
        }
    }
}

struct InboundRequest {
    peer_id: String,
    method: String,
    body: Vec<u8>,
    response: InboundResponse,
}

#[pyclass(name = "RpcRequest")]
struct PyRpcRequest {
    #[pyo3(get)]
    peer_id: String,
    #[pyo3(get)]
    method: String,
    body: Vec<u8>,
    response: Mutex<Option<InboundResponse>>,
}

#[pymethods]
impl PyRpcRequest {
    #[getter]
    fn body<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.body)
    }

    fn respond(&self, body: &[u8]) -> PyResult<()> {
        let response = self.take_response()?;
        match response {
            InboundResponse::Unary(sender) => sender
                .send(Ok(body.to_vec()))
                .map_err(|_| PyRuntimeError::new_err("remote RPC stream is no longer waiting")),
            InboundResponse::Stream(sender) => {
                sender.blocking_send(Ok(body.to_vec())).map_err(|_| {
                    PyRuntimeError::new_err("remote RPC stream is no longer waiting")
                })?;
                Ok(())
            }
        }
    }

    fn fail(&self, message: &str) -> PyResult<()> {
        if message.is_empty() {
            return Err(PyRuntimeError::new_err(
                "RPC error message must not be empty",
            ));
        }
        let response = self.take_response()?;
        match response {
            InboundResponse::Unary(sender) => sender
                .send(Err(message.to_owned()))
                .map_err(|_| PyRuntimeError::new_err("remote RPC stream is no longer waiting")),
            InboundResponse::Stream(sender) => sender
                .blocking_send(Err(message.to_owned()))
                .map_err(|_| PyRuntimeError::new_err("remote RPC stream is no longer waiting")),
        }
    }

    #[getter]
    fn is_stream(&self) -> PyResult<bool> {
        let response = self
            .response
            .lock()
            .map_err(|_| PyRuntimeError::new_err("RPC response lock is poisoned"))?;
        Ok(response.as_ref().is_some_and(InboundResponse::is_stream))
    }

    fn send_chunk(&self, py: Python<'_>, body: &[u8]) -> PyResult<()> {
        let sender = {
            let response = self
                .response
                .lock()
                .map_err(|_| PyRuntimeError::new_err("RPC response lock is poisoned"))?;
            match response.as_ref() {
                Some(InboundResponse::Stream(sender)) => sender.clone(),
                Some(InboundResponse::Unary(_)) => {
                    return Err(PyRuntimeError::new_err(
                        "send_chunk is only valid for streaming RPCs",
                    ));
                }
                None => return Err(PyRuntimeError::new_err("RPC request already completed")),
            }
        };
        let body = body.to_vec();
        py.detach(move || sender.blocking_send(Ok(body)))
            .map_err(|_| PyRuntimeError::new_err("remote RPC stream is no longer waiting"))
    }

    fn finish(&self) -> PyResult<()> {
        match self.take_response()? {
            InboundResponse::Stream(_) => Ok(()),
            InboundResponse::Unary(sender) => {
                let _ = sender.send(Err("unary RPC finished without a response".to_owned()));
                Err(PyRuntimeError::new_err(
                    "finish is only valid for streaming RPCs",
                ))
            }
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "RpcRequest(peer_id='{}', method='{}', body_bytes={})",
            self.peer_id,
            self.method,
            self.body.len()
        )
    }
}

impl PyRpcRequest {
    fn take_response(&self) -> PyResult<InboundResponse> {
        self.response
            .lock()
            .map_err(|_| PyRuntimeError::new_err("RPC response lock is poisoned"))?
            .take()
            .ok_or_else(|| PyRuntimeError::new_err("RPC request already completed"))
    }
}

impl Drop for PyRpcRequest {
    fn drop(&mut self) {
        if let Ok(slot) = self.response.get_mut()
            && let Some(response) = slot.take()
        {
            response.reject("RPC request dropped without a response".to_owned());
        }
    }
}

enum StreamEvent {
    Chunk(Vec<u8>),
    End,
    Error(String),
}

#[pyclass(name = "RpcStream")]
struct PyRpcStream {
    events: Arc<Mutex<Receiver<StreamEvent>>>,
    cancel: Mutex<Option<oneshot::Sender<()>>>,
    timeout: Duration,
    finished: AtomicBool,
}

#[pymethods]
impl PyRpcStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<Py<PyBytes>>> {
        if self.finished.load(Ordering::SeqCst) {
            return Ok(None);
        }
        let events = Arc::clone(&self.events);
        let timeout = self.timeout;
        let event = py.detach(move || {
            events
                .lock()
                .map_err(|_| RecvError::Disconnected)?
                .recv_timeout(timeout)
                .map_err(RecvError::from)
        });
        match event {
            Ok(StreamEvent::Chunk(chunk)) => Ok(Some(PyBytes::new(py, &chunk).unbind())),
            Ok(StreamEvent::End) | Err(RecvError::Disconnected) => {
                self.finished.store(true, Ordering::SeqCst);
                Ok(None)
            }
            Ok(StreamEvent::Error(message)) => {
                self.finished.store(true, Ordering::SeqCst);
                Err(PyRuntimeError::new_err(message))
            }
            Err(RecvError::Timeout) => {
                self.cancel_inner();
                self.finished.store(true, Ordering::SeqCst);
                Err(PyTimeoutError::new_err("RPC stream receive timed out"))
            }
        }
    }

    fn cancel(&self) {
        self.cancel_inner();
        self.finished.store(true, Ordering::SeqCst);
    }

    #[getter]
    fn closed(&self) -> bool {
        self.finished.load(Ordering::SeqCst)
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        _exception_type: &Bound<'_, PyAny>,
        _exception: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        self.cancel();
    }
}

impl PyRpcStream {
    fn cancel_inner(&self) {
        if let Ok(mut cancel) = self.cancel.lock()
            && let Some(sender) = cancel.take()
        {
            let _ = sender.send(());
        }
    }
}

impl Drop for PyRpcStream {
    fn drop(&mut self) {
        self.cancel_inner();
    }
}

#[pyclass(name = "NetworkNode")]
struct PyNetworkNode {
    runtime: Arc<Runtime>,
    endpoint: Endpoint,
    secret_key: SecretKey,
    catalog_dht: Mutex<Option<CatalogDhtHandle>>,
    relay_url: RelayUrl,
    incoming: Arc<Mutex<Receiver<InboundRequest>>>,
    dispatcher: InboundDispatcher,
    connections: ConnectionCache,
    request_id: AtomicU64,
    max_payload: u64,
    closed: Arc<AtomicBool>,
}

#[pymethods]
impl PyNetworkNode {
    #[new]
    #[allow(clippy::needless_pass_by_value)]
    #[pyo3(signature = (
        identity_path,
        relay_url,
        relay_token=None,
        force_relay=false,
        max_payload_bytes=DEFAULT_MAX_PAYLOAD,
        response_timeout_seconds=DEFAULT_RPC_TIMEOUT_SECONDS,
    ))]
    fn new(
        py: Python<'_>,
        identity_path: PathBuf,
        relay_url: &str,
        relay_token: Option<String>,
        force_relay: bool,
        max_payload_bytes: u64,
        response_timeout_seconds: u64,
    ) -> PyResult<Self> {
        if max_payload_bytes == 0 {
            return Err(PyRuntimeError::new_err(
                "max_payload_bytes must be greater than zero",
            ));
        }
        if response_timeout_seconds == 0 {
            return Err(PyRuntimeError::new_err(
                "response_timeout_seconds must be greater than zero",
            ));
        }
        let relay_url = RelayUrl::from_str(relay_url).map_err(py_error)?;
        let secret_key = identity::load_or_create(&identity_path).map_err(py_error)?;
        let runtime = Arc::new(Runtime::new().map_err(py_error)?);
        let config = EndpointConfig {
            relay_url: relay_url.clone(),
            relay_token,
            force_relay,
        };
        let runtime_for_bind = Arc::clone(&runtime);
        let endpoint_secret_key = secret_key.clone();
        let endpoint = py
            .detach(move || runtime_for_bind.block_on(bind(endpoint_secret_key, &config)))
            .map_err(py_error)?;
        let (incoming_tx, incoming_rx) = sync_channel(INBOUND_QUEUE_CAPACITY);
        let dispatcher = InboundDispatcher {
            sender: incoming_tx,
            max_payload: max_payload_bytes,
            response_timeout: Duration::from_secs(response_timeout_seconds),
        };
        let connections = Arc::new(tokio::sync::Mutex::new(HashMap::new()));
        let closed = Arc::new(AtomicBool::new(false));
        runtime.spawn(accept_loop(
            endpoint.clone(),
            dispatcher.clone(),
            Arc::clone(&connections),
            Arc::clone(&closed),
        ));

        Ok(Self {
            runtime,
            endpoint,
            secret_key,
            catalog_dht: Mutex::new(None),
            relay_url,
            incoming: Arc::new(Mutex::new(incoming_rx)),
            dispatcher,
            connections,
            request_id: AtomicU64::new(1),
            max_payload: max_payload_bytes,
            closed,
        })
    }

    #[getter]
    fn endpoint_id(&self) -> String {
        self.endpoint.id().to_string()
    }

    #[pyo3(signature = (
        kind,
        logical_key,
        discovery_peer_id,
        sequence,
        issued_at_ms,
        expires_at_ms,
        payload,
    ))]
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn sign_catalog_record<'py>(
        &self,
        py: Python<'py>,
        kind: &str,
        logical_key: &str,
        discovery_peer_id: &str,
        sequence: u64,
        issued_at_ms: u64,
        expires_at_ms: u64,
        payload: &[u8],
    ) -> PyResult<Bound<'py, PyBytes>> {
        self.ensure_open()?;
        let encoded = sign_catalog_record(
            &self.secret_key,
            CatalogRecordParams {
                kind: parse_catalog_kind(kind)?,
                logical_key,
                discovery_peer_id,
                sequence,
                issued_at_ms,
                expires_at_ms,
                payload,
            },
        )
        .map_err(py_error)?;
        Ok(PyBytes::new(py, &encoded))
    }

    #[staticmethod]
    fn verify_catalog_record(
        encoded: &[u8],
        now_ms: u64,
        max_clock_skew_ms: u64,
    ) -> PyResult<PyCatalogRecord> {
        verify_catalog_record(encoded, now_ms, max_clock_skew_ms)
            .map(PyCatalogRecord::from)
            .map_err(py_error)
    }

    #[pyo3(signature = (kind, model_swarm_id=None, target_endpoint_id=None))]
    fn catalog_key(
        &self,
        kind: &str,
        model_swarm_id: Option<&str>,
        target_endpoint_id: Option<&str>,
    ) -> PyResult<String> {
        let kind = parse_catalog_kind(kind)?;
        let source = self.endpoint.id();
        match kind {
            CatalogRecordKind::ModelManifest => model_swarm_id
                .map(catalog::keys::manifest)
                .ok_or_else(|| PyRuntimeError::new_err("model_swarm_id is required")),
            CatalogRecordKind::WorkerOffer => Ok(catalog::keys::worker_offer(&source)),
            CatalogRecordKind::SpanLease => model_swarm_id
                .map(|model| catalog::keys::span_lease(model, &source))
                .ok_or_else(|| PyRuntimeError::new_err("model_swarm_id is required")),
            CatalogRecordKind::LinkMetric => {
                let target = target_endpoint_id
                    .ok_or_else(|| PyRuntimeError::new_err("target_endpoint_id is required"))?;
                let target = EndpointId::from_str(target).map_err(py_error)?;
                Ok(catalog::keys::link_metric(&source, &target))
            }
            CatalogRecordKind::ModelMember => model_swarm_id
                .map(|model| catalog::keys::model_member(model, &source))
                .ok_or_else(|| PyRuntimeError::new_err("model_swarm_id is required")),
            CatalogRecordKind::Unspecified => Err(PyRuntimeError::new_err(
                "catalogue record kind is unspecified",
            )),
        }
    }

    #[pyo3(signature = (
        identity_path,
        server_mode=false,
        listen_address="/ip4/127.0.0.1/tcp/0",
        bootstrap_addresses=Vec::new(),
        query_timeout_seconds=15,
        max_records=25_000,
    ))]
    #[allow(clippy::needless_pass_by_value, clippy::too_many_arguments)]
    fn start_catalog_dht(
        &self,
        py: Python<'_>,
        identity_path: PathBuf,
        server_mode: bool,
        listen_address: &str,
        bootstrap_addresses: Vec<String>,
        query_timeout_seconds: u64,
        max_records: usize,
    ) -> PyResult<(String, String)> {
        self.ensure_open()?;
        if query_timeout_seconds == 0 || max_records == 0 {
            return Err(PyRuntimeError::new_err(
                "query_timeout_seconds and max_records must be positive",
            ));
        }
        if self
            .catalog_dht
            .lock()
            .map_err(|_| PyRuntimeError::new_err("catalogue DHT lock is poisoned"))?
            .is_some()
        {
            return Err(PyRuntimeError::new_err("catalogue DHT is already running"));
        }

        let keypair = load_or_create_dht_keypair(&identity_path).map_err(py_error)?;
        let listen_address: Multiaddr = listen_address.parse().map_err(py_error)?;
        let bootstrap_peers = bootstrap_addresses
            .iter()
            .map(|address| parse_bootstrap_peer(address))
            .collect::<PyResult<Vec<_>>>()?;
        let mut config = if server_mode {
            CatalogDhtConfig::server(keypair, listen_address)
        } else {
            let mut config = CatalogDhtConfig::client(keypair, bootstrap_peers.clone());
            config.listen_address = listen_address;
            config
        };
        if server_mode {
            config.bootstrap_peers = bootstrap_peers;
        }
        config.query_timeout = Duration::from_secs(query_timeout_seconds);
        config.max_records = max_records;

        let runtime = Arc::clone(&self.runtime);
        let handle = py
            .detach(move || runtime.block_on(spawn_catalog_dht(config)))
            .map_err(py_error)?;
        let identity = (
            handle.peer_id().to_string(),
            handle.listen_address().to_string(),
        );
        let mut slot = self
            .catalog_dht
            .lock()
            .map_err(|_| PyRuntimeError::new_err("catalogue DHT lock is poisoned"))?;
        if slot.is_some() {
            let runtime = Arc::clone(&self.runtime);
            py.detach(move || runtime.block_on(handle.shutdown()))
                .map_err(py_error)?;
            return Err(PyRuntimeError::new_err(
                "catalogue DHT was started concurrently",
            ));
        }
        *slot = Some(handle);
        Ok(identity)
    }

    fn catalog_bootstrap(&self, py: Python<'_>) -> PyResult<()> {
        let handle = self.catalog_dht_handle()?;
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || runtime.block_on(handle.bootstrap()))
            .map_err(py_error)
    }

    #[pyo3(signature = (logical_key, encoded, quorum=1))]
    fn catalog_put(
        &self,
        py: Python<'_>,
        logical_key: &str,
        encoded: &[u8],
        quorum: usize,
    ) -> PyResult<()> {
        let quorum = std::num::NonZeroUsize::new(quorum)
            .ok_or_else(|| PyRuntimeError::new_err("quorum must be positive"))?;
        let handle = self.catalog_dht_handle()?;
        let logical_key = logical_key.to_owned();
        let encoded = encoded.to_vec();
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || runtime.block_on(handle.put(logical_key, encoded, quorum)))
            .map_err(py_error)
    }

    fn catalog_get(&self, py: Python<'_>, logical_key: &str) -> PyResult<PyCatalogRecord> {
        let handle = self.catalog_dht_handle()?;
        let logical_key = logical_key.to_owned();
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || runtime.block_on(handle.get(logical_key)))
            .map(PyCatalogRecord::from)
            .map_err(py_error)
    }

    fn catalog_get_members(
        &self,
        py: Python<'_>,
        logical_key: &str,
    ) -> PyResult<Vec<PyCatalogRecord>> {
        let handle = self.catalog_dht_handle()?;
        let logical_key = logical_key.to_owned();
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || runtime.block_on(handle.get_members(logical_key)))
            .map(|records| records.into_iter().map(PyCatalogRecord::from).collect())
            .map_err(py_error)
    }

    fn catalog_get_model_members(
        &self,
        py: Python<'_>,
        model_swarm_id: &str,
    ) -> PyResult<Vec<PyCatalogRecord>> {
        let handle = self.catalog_dht_handle()?;
        let model_swarm_id = model_swarm_id.to_owned();
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || runtime.block_on(handle.get_model_members(&model_swarm_id)))
            .map(|records| records.into_iter().map(PyCatalogRecord::from).collect())
            .map_err(py_error)
    }

    fn stop_catalog_dht(&self, py: Python<'_>) -> PyResult<()> {
        let handle = self
            .catalog_dht
            .lock()
            .map_err(|_| PyRuntimeError::new_err("catalogue DHT lock is poisoned"))?
            .take();
        let Some(handle) = handle else {
            return Ok(());
        };
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || runtime.block_on(handle.shutdown()))
            .map_err(py_error)
    }

    fn call(
        &self,
        py: Python<'_>,
        peer_id: &str,
        method: &str,
        body: &[u8],
        timeout_seconds: Option<f64>,
    ) -> PyResult<Py<PyBytes>> {
        self.ensure_open()?;
        let timeout = positive_timeout(timeout_seconds)?;
        let peer_id = EndpointId::from_str(peer_id).map_err(py_error)?;
        let endpoint = self.endpoint.clone();
        let relay_url = self.relay_url.clone();
        let dispatcher = self.dispatcher.clone();
        let connections = Arc::clone(&self.connections);
        let request_id = self.request_id.fetch_add(1, Ordering::Relaxed);
        let method = method.to_owned();
        let body = body.to_vec();
        let max_payload = self.max_payload;
        let runtime = Arc::clone(&self.runtime);
        let response = py
            .detach(move || {
                runtime.block_on(call_rpc(
                    endpoint,
                    relay_url,
                    dispatcher,
                    connections,
                    peer_id,
                    request_id,
                    method,
                    body,
                    max_payload,
                    timeout,
                ))
            })
            .map_err(py_error)?;
        Ok(PyBytes::new(py, &response).unbind())
    }

    fn call_stream(
        &self,
        py: Python<'_>,
        peer_id: &str,
        method: &str,
        body: &[u8],
        timeout_seconds: Option<f64>,
    ) -> PyResult<Py<PyRpcStream>> {
        self.ensure_open()?;
        let timeout = positive_timeout(timeout_seconds)?;
        let peer_id = EndpointId::from_str(peer_id).map_err(py_error)?;
        let endpoint = self.endpoint.clone();
        let relay_url = self.relay_url.clone();
        let dispatcher = self.dispatcher.clone();
        let connections = Arc::clone(&self.connections);
        let request_id = self.request_id.fetch_add(1, Ordering::Relaxed);
        let method = method.to_owned();
        let body = body.to_vec();
        let max_payload = self.max_payload;
        let (python_event_tx, event_rx) = sync_channel(STREAM_QUEUE_CAPACITY);
        let (event_tx, mut event_rx_async) = tokio_mpsc::channel(STREAM_QUEUE_CAPACITY);
        let (cancel_tx, cancel_rx) = oneshot::channel();
        self.runtime.spawn_blocking(move || {
            while let Some(event) = event_rx_async.blocking_recv() {
                if python_event_tx.send(event).is_err() {
                    break;
                }
            }
        });
        self.runtime.spawn(async move {
            let errors = event_tx.clone();
            if let Err(error) = call_stream_rpc(
                endpoint,
                relay_url,
                dispatcher,
                connections,
                peer_id,
                request_id,
                method,
                body,
                max_payload,
                event_tx,
                cancel_rx,
            )
            .await
            {
                let _ = errors.send(StreamEvent::Error(error.to_string())).await;
            }
        });
        Py::new(
            py,
            PyRpcStream {
                events: Arc::new(Mutex::new(event_rx)),
                cancel: Mutex::new(Some(cancel_tx)),
                timeout,
                finished: AtomicBool::new(false),
            },
        )
    }

    fn recv(&self, py: Python<'_>, timeout_seconds: Option<f64>) -> PyResult<Py<PyRpcRequest>> {
        self.ensure_open()?;
        let timeout = positive_timeout(timeout_seconds)?;
        let incoming = Arc::clone(&self.incoming);
        let request = py.detach(move || {
            incoming
                .lock()
                .map_err(|_| RecvError::Disconnected)?
                .recv_timeout(timeout)
                .map_err(RecvError::from)
        });
        let request = match request {
            Ok(request) => request,
            Err(RecvError::Timeout) => {
                return Err(PyTimeoutError::new_err("RPC receive timed out"));
            }
            Err(RecvError::Disconnected) => {
                return Err(PyRuntimeError::new_err("RPC receive queue is closed"));
            }
        };
        Py::new(
            py,
            PyRpcRequest {
                peer_id: request.peer_id,
                method: request.method,
                body: request.body,
                response: Mutex::new(Some(request.response)),
            },
        )
    }

    fn paths(&self, py: Python<'_>, peer_id: &str) -> PyResult<String> {
        let peer_id = EndpointId::from_str(peer_id).map_err(py_error)?;
        let connections = Arc::clone(&self.connections);
        let runtime = Arc::clone(&self.runtime);
        let snapshots = py.detach(move || {
            runtime.block_on(async move {
                let cache = connections.lock().await;
                cache.get(&peer_id).map(telemetry::snapshot)
            })
        });
        serde_json::to_string(&snapshots.unwrap_or_default()).map_err(py_error)
    }

    fn peers(&self, py: Python<'_>) -> Vec<String> {
        let connections = Arc::clone(&self.connections);
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || {
            runtime.block_on(async move {
                connections
                    .lock()
                    .await
                    .iter()
                    .filter(|(_, connection)| connection.close_reason().is_none())
                    .map(|(peer_id, _)| peer_id.to_string())
                    .collect()
            })
        })
    }

    fn close(&self, py: Python<'_>) {
        if self.closed.swap(true, Ordering::SeqCst) {
            return;
        }
        let endpoint = self.endpoint.clone();
        let catalog_dht = self
            .catalog_dht
            .lock()
            .ok()
            .and_then(|mut slot| slot.take());
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || {
            runtime.block_on(async move {
                if let Some(handle) = catalog_dht {
                    let _ = handle.shutdown().await;
                }
                endpoint.close().await;
            });
        });
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _exception_type: &Bound<'_, PyAny>,
        _exception: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) {
        self.close(py);
    }
}

impl PyNetworkNode {
    fn ensure_open(&self) -> PyResult<()> {
        if self.closed.load(Ordering::SeqCst) {
            return Err(PyRuntimeError::new_err("network node is closed"));
        }
        Ok(())
    }

    fn catalog_dht_handle(&self) -> PyResult<CatalogDhtHandle> {
        self.ensure_open()?;
        self.catalog_dht
            .lock()
            .map_err(|_| PyRuntimeError::new_err("catalogue DHT lock is poisoned"))?
            .clone()
            .ok_or_else(|| PyRuntimeError::new_err("catalogue DHT is not running"))
    }
}

impl Drop for PyNetworkNode {
    fn drop(&mut self) {
        if !self.closed.swap(true, Ordering::SeqCst) {
            let endpoint = self.endpoint.clone();
            let catalog_dht = self
                .catalog_dht
                .lock()
                .ok()
                .and_then(|mut slot| slot.take());
            self.runtime.spawn(async move {
                if let Some(handle) = catalog_dht {
                    let _ = handle.shutdown().await;
                }
                endpoint.close().await;
            });
        }
    }
}

fn parse_bootstrap_peer(value: &str) -> PyResult<BootstrapPeer> {
    let mut address: Multiaddr = value.parse().map_err(py_error)?;
    let Some(Protocol::P2p(peer_id)) = address.pop() else {
        return Err(PyRuntimeError::new_err(
            "bootstrap address must end with /p2p/<peer-id>",
        ));
    };
    Ok(BootstrapPeer { peer_id, address })
}

#[derive(Debug)]
enum RecvError {
    Timeout,
    Disconnected,
}

impl From<RecvTimeoutError> for RecvError {
    fn from(value: RecvTimeoutError) -> Self {
        match value {
            RecvTimeoutError::Timeout => Self::Timeout,
            RecvTimeoutError::Disconnected => Self::Disconnected,
        }
    }
}

fn positive_timeout(seconds: Option<f64>) -> PyResult<Duration> {
    let seconds = seconds.unwrap_or(DEFAULT_RPC_TIMEOUT_SECONDS_F64);
    if !seconds.is_finite() || seconds <= 0.0 {
        return Err(PyRuntimeError::new_err(
            "timeout_seconds must be a positive finite number",
        ));
    }
    Ok(Duration::from_secs_f64(seconds))
}

fn py_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

async fn accept_loop(
    endpoint: Endpoint,
    dispatcher: InboundDispatcher,
    connections: ConnectionCache,
    closed: Arc<AtomicBool>,
) {
    while let Some(incoming) = endpoint.accept().await {
        if closed.load(Ordering::SeqCst) {
            break;
        }
        let mut accepting = match incoming.accept() {
            Ok(accepting) => accepting,
            Err(error) => {
                debug!(%error, "discarded invalid incoming connection");
                continue;
            }
        };
        let alpn = match accepting.alpn().await {
            Ok(alpn) if alpn == ALPN => alpn,
            Ok(alpn) => {
                warn!(alpn = %String::from_utf8_lossy(&alpn), "discarded unexpected ALPN");
                continue;
            }
            Err(error) => {
                debug!(%error, "incoming ALPN negotiation failed");
                continue;
            }
        };
        drop(alpn);
        let connection = match accepting.await {
            Ok(connection) => connection,
            Err(error) => {
                debug!(%error, "incoming peer authentication failed");
                continue;
            }
        };
        // A simultaneous cross-dial can produce two valid QUIC connections.
        // Keep the already cached one as the preferred outbound connection,
        // but serve streams on this incoming connection as well: it may
        // already carry the RPC which caused Endpoint::accept to wake up.
        connections
            .lock()
            .await
            .entry(connection.remote_id())
            .or_insert_with(|| connection.clone());
        dispatcher.spawn(connection);
    }
}

async fn handle_rpc_connection(
    connection: Connection,
    incoming_tx: SyncSender<InboundRequest>,
    max_payload: u64,
    response_timeout: Duration,
) {
    loop {
        let Ok((send, recv)) = connection.accept_bi().await else {
            break;
        };
        let sender = incoming_tx.clone();
        let peer_id = connection.remote_id().to_string();
        tokio::spawn(async move {
            if let Err(error) =
                handle_rpc_stream(send, recv, peer_id, sender, max_payload, response_timeout).await
            {
                debug!(%error, "isolated failed RPC stream");
            }
        });
    }
}

async fn handle_rpc_stream(
    mut send: iroh::endpoint::SendStream,
    mut recv: iroh::endpoint::RecvStream,
    peer_id: String,
    incoming_tx: SyncSender<InboundRequest>,
    max_payload: u64,
    response_timeout: Duration,
) -> Result<()> {
    let header = read_header(&mut recv, max_payload).await?;
    ensure!(
        matches!(
            header.kind,
            MessageKind::RpcRequest | MessageKind::RpcStreamRequest
        ),
        "incoming stream is not an RPC request"
    );
    let payload = read_and_verify_payload(&mut recv, &header).await?;
    let (method, body) = decode_rpc_request(&payload)?;
    if header.kind == MessageKind::RpcRequest {
        let (response_tx, response_rx) = oneshot::channel();
        enqueue_inbound(
            &incoming_tx,
            InboundRequest {
                peer_id,
                method: method.to_owned(),
                body: body.to_vec(),
                response: InboundResponse::Unary(response_tx),
            },
        );
        let response = tokio::time::timeout(response_timeout, response_rx)
            .await
            .context("Python RPC handler timed out")?
            .context("Python RPC request was dropped")?;
        let (kind, payload) = match response {
            Ok(payload) => (MessageKind::RpcResponse, payload),
            Err(message) => (MessageKind::RpcError, message.into_bytes()),
        };
        write_rpc_frame(&mut send, kind, header.request_id, &payload, max_payload).await?;
        send.finish().context("failed to finish RPC response")?;
        return Ok(());
    }

    let (response_tx, mut response_rx) = tokio_mpsc::channel(STREAM_QUEUE_CAPACITY);
    enqueue_inbound(
        &incoming_tx,
        InboundRequest {
            peer_id,
            method: method.to_owned(),
            body: body.to_vec(),
            response: InboundResponse::Stream(response_tx),
        },
    );
    loop {
        match tokio::time::timeout(response_timeout, response_rx.recv()).await {
            Ok(Some(Ok(payload))) => {
                write_stream_frame_or_stopped(
                    &mut send,
                    MessageKind::RpcStreamChunk,
                    header.request_id,
                    &payload,
                    max_payload,
                )
                .await?;
            }
            Ok(Some(Err(message))) => {
                write_stream_frame_or_stopped(
                    &mut send,
                    MessageKind::RpcStreamError,
                    header.request_id,
                    message.as_bytes(),
                    max_payload,
                )
                .await?;
                send.finish().context("failed to finish RPC stream error")?;
                return Ok(());
            }
            Ok(None) => {
                write_stream_frame_or_stopped(
                    &mut send,
                    MessageKind::RpcStreamEnd,
                    header.request_id,
                    &[],
                    max_payload,
                )
                .await?;
                send.finish().context("failed to finish RPC stream")?;
                return Ok(());
            }
            Err(_) => {
                write_stream_frame_or_stopped(
                    &mut send,
                    MessageKind::RpcStreamError,
                    header.request_id,
                    b"Python streaming RPC handler timed out",
                    max_payload,
                )
                .await?;
                send.finish()
                    .context("failed to finish timed-out RPC stream")?;
                return Ok(());
            }
        }
    }
}

fn enqueue_inbound(incoming_tx: &SyncSender<InboundRequest>, request: InboundRequest) {
    match incoming_tx.try_send(request) {
        Ok(()) => {}
        Err(TrySendError::Full(request)) => {
            request
                .response
                .reject("inbound RPC queue is full".to_owned());
        }
        Err(TrySendError::Disconnected(request)) => {
            request
                .response
                .reject("RPC dispatcher is unavailable".to_owned());
        }
    }
}

async fn write_rpc_frame(
    send: &mut iroh::endpoint::SendStream,
    kind: MessageKind,
    request_id: u64,
    payload: &[u8],
    max_payload: u64,
) -> Result<()> {
    ensure!(
        u64::try_from(payload.len()).unwrap_or(u64::MAX) <= max_payload,
        "RPC response exceeds configured payload limit"
    );
    let header = Header::new(
        kind,
        request_id,
        u64::try_from(payload.len()).context("RPC response size overflow")?,
        blake3::hash(payload),
    );
    write_header(send, &header).await?;
    send.write_all(payload)
        .await
        .context("failed to write RPC response payload")
}

async fn write_stream_frame_or_stopped(
    send: &mut iroh::endpoint::SendStream,
    kind: MessageKind,
    request_id: u64,
    payload: &[u8],
    max_payload: u64,
) -> Result<()> {
    // `write_all` may wait behind QUIC flow control. Poll STOP_SENDING at the
    // same time so remote cancellation promptly releases the producer.
    tokio::select! {
        result = send.stopped() => {
            let code = result.context("failed while waiting for remote stream cancellation")?;
            bail!("remote stopped RPC stream with code {code:?}");
        }
        result = write_rpc_frame(send, kind, request_id, payload, max_payload) => result,
    }
}

#[allow(clippy::too_many_arguments)]
async fn call_rpc(
    endpoint: Endpoint,
    relay_url: RelayUrl,
    dispatcher: InboundDispatcher,
    connections: ConnectionCache,
    peer_id: EndpointId,
    request_id: u64,
    method: String,
    body: Vec<u8>,
    max_payload: u64,
    timeout: Duration,
) -> Result<Vec<u8>> {
    tokio::time::timeout(timeout, async move {
        let payload = encode_rpc_request(&method, &body)?;
        ensure!(
            u64::try_from(payload.len()).unwrap_or(u64::MAX) <= max_payload,
            "RPC request exceeds configured payload limit"
        );
        let connection =
            get_connection(&endpoint, relay_url, &dispatcher, &connections, peer_id).await?;
        let (mut send, mut recv) = connection
            .open_bi()
            .await
            .context("failed to open RPC stream")?;
        let header = Header::new(
            MessageKind::RpcRequest,
            request_id,
            u64::try_from(payload.len()).context("RPC request size overflow")?,
            blake3::hash(&payload),
        );
        write_header(&mut send, &header).await?;
        send.write_all(&payload)
            .await
            .context("failed to write RPC request")?;
        send.finish().context("failed to finish RPC request")?;
        let response_header = read_header(&mut recv, max_payload).await?;
        ensure!(
            response_header.request_id == request_id,
            "RPC response request ID mismatch"
        );
        ensure!(
            matches!(
                response_header.kind,
                MessageKind::RpcResponse | MessageKind::RpcError
            ),
            "unexpected RPC response kind {:?}",
            response_header.kind
        );
        let response = read_and_verify_payload(&mut recv, &response_header).await?;
        if response_header.kind == MessageKind::RpcError {
            bail!("remote RPC failed: {}", String::from_utf8_lossy(&response));
        }
        Ok(response)
    })
    .await
    .context("RPC deadline exceeded")?
}

#[allow(clippy::too_many_arguments)]
async fn call_stream_rpc(
    endpoint: Endpoint,
    relay_url: RelayUrl,
    dispatcher: InboundDispatcher,
    connections: ConnectionCache,
    peer_id: EndpointId,
    request_id: u64,
    method: String,
    body: Vec<u8>,
    max_payload: u64,
    events: tokio_mpsc::Sender<StreamEvent>,
    mut cancel: oneshot::Receiver<()>,
) -> Result<()> {
    let payload = encode_rpc_request(&method, &body)?;
    ensure!(
        u64::try_from(payload.len()).unwrap_or(u64::MAX) <= max_payload,
        "RPC stream request exceeds configured payload limit"
    );
    let connection = tokio::select! {
        _ = &mut cancel => return Ok(()),
        result = get_connection(&endpoint, relay_url, &dispatcher, &connections, peer_id) => result?,
    };
    let (mut send, mut recv) = tokio::select! {
        _ = &mut cancel => return Ok(()),
        result = connection.open_bi() => result.context("failed to open RPC stream")?,
    };
    write_rpc_frame(
        &mut send,
        MessageKind::RpcStreamRequest,
        request_id,
        &payload,
        max_payload,
    )
    .await?;
    send.finish()
        .context("failed to finish RPC stream request")?;

    loop {
        let response_header = tokio::select! {
            _ = &mut cancel => {
                cancel_quic_stream(&mut send, &mut recv);
                return Ok(());
            }
            result = read_header(&mut recv, max_payload) => result?,
        };
        ensure!(
            response_header.request_id == request_id,
            "RPC stream response request ID mismatch"
        );
        ensure!(
            matches!(
                response_header.kind,
                MessageKind::RpcStreamChunk
                    | MessageKind::RpcStreamEnd
                    | MessageKind::RpcStreamError
            ),
            "unexpected RPC stream response kind {:?}",
            response_header.kind
        );
        let response = tokio::select! {
            _ = &mut cancel => {
                cancel_quic_stream(&mut send, &mut recv);
                return Ok(());
            }
            result = read_and_verify_payload(&mut recv, &response_header) => result?,
        };
        match response_header.kind {
            MessageKind::RpcStreamChunk => {
                let delivered = tokio::select! {
                    _ = &mut cancel => {
                        cancel_quic_stream(&mut send, &mut recv);
                        return Ok(());
                    }
                    result = events.send(StreamEvent::Chunk(response)) => result,
                };
                if delivered.is_err() {
                    cancel_quic_stream(&mut send, &mut recv);
                    return Ok(());
                }
            }
            MessageKind::RpcStreamEnd => {
                ensure!(response.is_empty(), "RPC stream end frame must be empty");
                tokio::select! {
                    _ = &mut cancel => {
                        cancel_quic_stream(&mut send, &mut recv);
                    }
                    _ = events.send(StreamEvent::End) => {}
                }
                return Ok(());
            }
            MessageKind::RpcStreamError => {
                let message = String::from_utf8_lossy(&response);
                tokio::select! {
                    _ = &mut cancel => {
                        cancel_quic_stream(&mut send, &mut recv);
                    }
                    _ = events.send(StreamEvent::Error(format!(
                        "remote streaming RPC failed: {message}"
                    ))) => {}
                }
                return Ok(());
            }
            _ => unreachable!("validated streaming response kind"),
        }
    }
}

fn cancel_quic_stream(
    send: &mut iroh::endpoint::SendStream,
    recv: &mut iroh::endpoint::RecvStream,
) {
    let _ = send.reset(STREAM_CANCEL_CODE.into());
    let _ = recv.stop(STREAM_CANCEL_CODE.into());
}

async fn get_connection(
    endpoint: &Endpoint,
    relay_url: RelayUrl,
    dispatcher: &InboundDispatcher,
    connections: &ConnectionCache,
    peer_id: EndpointId,
) -> Result<Connection> {
    if let Some(connection) = connections.lock().await.get(&peer_id).cloned()
        && connection.close_reason().is_none()
    {
        return Ok(connection);
    }
    let remote =
        EndpointAddr::from_parts(peer_id, std::iter::once(TransportAddr::Relay(relay_url)));
    let connection = endpoint
        .connect(remote, ALPN)
        .await
        .context("failed to connect RPC peer")?;
    let selected = {
        let mut cache = connections.lock().await;
        if let Some(existing) = cache
            .get(&peer_id)
            .filter(|existing| existing.close_reason().is_none())
            .cloned()
        {
            existing
        } else {
            cache.insert(peer_id, connection.clone());
            dispatcher.spawn(connection.clone());
            return Ok(connection);
        }
    };

    // Another task or a simultaneous inbound connection won the race. Avoid
    // leaking an unused parallel connection and reuse the registered one.
    connection.close(0u8.into(), b"superseded connection");
    Ok(selected)
}

#[pymodule(gil_used = false)]
fn fabi_network_native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNetworkNode>()?;
    module.add_class::<PyCatalogRecord>()?;
    module.add_class::<PyRpcRequest>()?;
    module.add_class::<PyRpcStream>()?;
    module.add("PROTOCOL_VERSION", 1)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc::sync_channel;

    use anyhow::{Context, Result, anyhow, bail, ensure};
    use iroh::{Endpoint, RelayMode, endpoint::presets};

    use super::*;

    async fn direct_endpoint() -> Result<Endpoint> {
        Endpoint::builder(presets::N0)
            .alpns(vec![ALPN.to_vec()])
            .relay_mode(RelayMode::Disabled)
            .bind()
            .await
            .context("failed to bind direct test endpoint")
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn dialed_connection_serves_reverse_rpc() -> Result<()> {
        let listener = direct_endpoint().await?;
        let dialer = direct_endpoint().await?;

        let accepting_endpoint = listener.clone();
        let accept_task = tokio::spawn(async move {
            let incoming = accepting_endpoint
                .accept()
                .await
                .context("listener closed before accepting")?;
            incoming
                .accept()
                .context("failed to accept direct connection")?
                .await
                .context("direct connection handshake failed")
        });
        let outbound_connection = tokio::time::timeout(
            Duration::from_secs(5),
            dialer.connect(listener.addr(), ALPN),
        )
        .await
        .context("direct dial timed out")?
        .context("failed to dial direct endpoint")?;
        let accepted = tokio::time::timeout(Duration::from_secs(5), accept_task)
            .await
            .context("direct accept timed out")?
            .context("accept task panicked")??;

        // The dialer initiated the QUIC connection. It must still accept a new
        // bidirectional stream opened later by the listening peer.
        let (dialer_tx, dialer_rx) = sync_channel(1);
        let dialer_dispatcher = InboundDispatcher {
            sender: dialer_tx,
            max_payload: DEFAULT_MAX_PAYLOAD,
            response_timeout: Duration::from_secs(5),
        };
        dialer_dispatcher.spawn(outbound_connection);
        let responder = tokio::task::spawn_blocking(move || -> Result<()> {
            let request = dialer_rx
                .recv_timeout(Duration::from_secs(5))
                .context("dialer never accepted the reverse RPC stream")?;
            ensure!(request.method == "test.reverse", "unexpected RPC method");
            ensure!(request.body == b"ping", "unexpected RPC body");
            match request.response {
                InboundResponse::Unary(sender) => sender
                    .send(Ok(b"pong".to_vec()))
                    .map_err(|_| anyhow!("reverse RPC caller stopped waiting")),
                InboundResponse::Stream(_) => bail!("expected a unary reverse RPC"),
            }
        });

        let connections = Arc::new(tokio::sync::Mutex::new(HashMap::from([(
            dialer.id(),
            accepted,
        )])));
        let (unused_tx, _unused_rx) = sync_channel(1);
        let listener_dispatcher = InboundDispatcher {
            sender: unused_tx,
            max_payload: DEFAULT_MAX_PAYLOAD,
            response_timeout: Duration::from_secs(5),
        };
        let reply = call_rpc(
            listener.clone(),
            RelayUrl::from_str("https://unused.invalid")?,
            listener_dispatcher,
            connections,
            dialer.id(),
            1,
            "test.reverse".to_owned(),
            b"ping".to_vec(),
            DEFAULT_MAX_PAYLOAD,
            Duration::from_secs(5),
        )
        .await?;

        ensure!(reply == b"pong", "unexpected reverse RPC response");
        responder.await.context("responder task panicked")??;
        listener.close().await;
        dialer.close().await;
        Ok(())
    }
}
