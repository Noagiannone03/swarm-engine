//! Narrow, verified bridge between Fabi and Skippy's staged llama.cpp runtime.
//!
//! Fabi authenticates the product archive before this crate is reached. This
//! bridge additionally validates Skippy's per-file runtime manifest, pins the
//! exact Mesh/ABI/backend contract, and is the only Fabi crate allowed to cross
//! the dynamic-library `unsafe` boundary. Model package bytes are independently
//! authenticated by Fabi's TUF model registry before [`SkippyStage::open`].

use std::{
    collections::HashMap,
    fs::File,
    io::Read,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, Result, bail, ensure};
use mesh_llm_native_runtime::{NativeRuntimeBackendKind, NativeRuntimeManifest};
use model_artifact::gguf::{GgufKvCacheQuant, scan_gguf_compact_meta};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use skippy_ffi::TensorRole;
use skippy_runtime::{
    ActivationFrame, BackendDeviceType, FlashAttentionType, RuntimeConfig, RuntimeLoadMode,
    SamplingConfig, StageModel, StageSession, backend_devices, parse_cache_type,
};

/// Stable Mesh release audited for Fabi's first Skippy product integration.
pub const SKIPPY_MESH_RELEASE: &str = "0.74.0";
/// Exact native ABI exported by the pinned release.
pub const SKIPPY_RUNTIME_ABI: &str = "0.1.32";
const FABI_INTEGRITY_MANIFEST: &str = "fabi-integrity.json";

#[derive(Debug, Deserialize)]
struct FabiIntegrityFile {
    path: String,
    sha256: String,
}

#[derive(Debug, Deserialize)]
struct FabiRuntimeIntegrity {
    schema_version: u32,
    mesh_release: String,
    skippy_abi: String,
    runtime_id: String,
    backend: String,
    files: Vec<FabiIntegrityFile>,
}

fn checked_runtime_file(root: &Path, relative: &str) -> Result<PathBuf> {
    let relative_path = Path::new(relative);
    ensure!(
        !relative_path.as_os_str().is_empty()
            && relative_path
                .components()
                .all(|component| matches!(component, Component::Normal(_))),
        "unsafe Skippy runtime path {relative}"
    );
    let path = root.join(relative_path);
    let canonical = path
        .canonicalize()
        .with_context(|| format!("canonicalize Skippy runtime file {}", path.display()))?;
    ensure!(
        canonical.starts_with(root) && canonical.is_file(),
        "Skippy runtime file escapes its product root: {relative}"
    );
    Ok(canonical)
}

fn file_sha256(path: &Path) -> Result<String> {
    let mut file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hash {}", path.display()))?;
        if read == 0 {
            break;
        }
        digest.update(&buffer[..read]);
    }
    Ok(digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn verify_fabi_runtime_integrity(
    root: &Path,
    runtime_id: &str,
    declared_libraries: &[String],
    expected_mesh_release: &str,
    expected_abi: &str,
    expected_backend: &str,
) -> Result<Vec<PathBuf>> {
    let integrity_path = root.join(FABI_INTEGRITY_MANIFEST);
    let integrity: FabiRuntimeIntegrity = serde_json::from_reader(
        File::open(&integrity_path)
            .with_context(|| format!("open {}", integrity_path.display()))?,
    )
    .with_context(|| format!("parse {}", integrity_path.display()))?;
    ensure!(
        integrity.schema_version == 1,
        "unsupported Fabi Skippy integrity schema {}",
        integrity.schema_version
    );
    ensure!(
        integrity.mesh_release == expected_mesh_release
            && integrity.skippy_abi == expected_abi
            && integrity.runtime_id == runtime_id
            && integrity.backend == expected_backend,
        "Fabi Skippy integrity contract does not match the selected runtime"
    );
    let expected_files = std::iter::once("manifest.json")
        .chain(declared_libraries.iter().map(String::as_str))
        .collect::<std::collections::HashSet<_>>();
    let integrity_files = integrity
        .files
        .iter()
        .map(|entry| entry.path.as_str())
        .collect::<std::collections::HashSet<_>>();
    ensure!(
        integrity_files == expected_files && integrity.files.len() == expected_files.len(),
        "Fabi Skippy integrity manifest does not cover exactly the declared runtime files"
    );
    for entry in &integrity.files {
        ensure!(
            entry.sha256.len() == 64 && entry.sha256.bytes().all(|byte| byte.is_ascii_hexdigit()),
            "invalid SHA-256 for Skippy runtime file {}",
            entry.path
        );
        let path = checked_runtime_file(root, &entry.path)?;
        ensure!(
            file_sha256(&path)? == entry.sha256.to_ascii_lowercase(),
            "Skippy runtime file hash mismatch: {}",
            entry.path
        );
    }
    declared_libraries
        .iter()
        .map(|relative| checked_runtime_file(root, relative))
        .collect()
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeDeviceInfo {
    pub name: String,
    pub description: Option<String>,
    pub device_id: Option<String>,
    pub memory_free: u64,
    pub memory_total: u64,
    pub kind: String,
    pub caps: u64,
}

/// Structural and KV geometry read by Mesh's audited GGUF parser.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SkippyPackageGeometry {
    pub architecture: String,
    pub context_length: u32,
    pub activation_width: u32,
    pub layer_count: u32,
    pub kv_bytes_per_token: u64,
    pub static_bytes_by_layer: Vec<u64>,
}

fn direct_static_bytes_by_layer(paths: &[PathBuf], layer_count: u32) -> Result<Vec<u64>> {
    let layer_count = usize::try_from(layer_count).context("GGUF layer count exceeds usize")?;
    let mut bytes_by_layer = vec![0_u64; layer_count];
    let mut shared_bytes = 0_u64;
    let mut seen = std::collections::HashSet::new();
    for path in paths {
        let model = skippy_runtime::ModelInfo::open(path)
            .with_context(|| format!("open GGUF tensor index {}", path.display()))?;
        let tensors = model
            .tensors()
            .with_context(|| format!("read GGUF tensor index {}", path.display()))?;
        for tensor in tensors {
            if !seen.insert(tensor.name) {
                continue;
            }
            match tensor
                .layer_index
                .and_then(|index| usize::try_from(index).ok())
            {
                Some(index) if index < layer_count => {
                    bytes_by_layer[index] = bytes_by_layer[index].saturating_add(tensor.byte_size);
                }
                Some(_) => {
                    let last = bytes_by_layer
                        .last_mut()
                        .context("GGUF has tensors but no transformer layers")?;
                    *last = last.saturating_add(tensor.byte_size);
                }
                None => match tensor.role {
                    TensorRole::Embedding => {
                        bytes_by_layer[0] = bytes_by_layer[0].saturating_add(tensor.byte_size);
                    }
                    TensorRole::FinalNorm | TensorRole::Output => {
                        let last = bytes_by_layer
                            .last_mut()
                            .context("GGUF has endpoint tensors but no transformer layers")?;
                        *last = last.saturating_add(tensor.byte_size);
                    }
                    TensorRole::Unknown
                    | TensorRole::Metadata
                    | TensorRole::Tokenizer
                    | TensorRole::Layer => {
                        shared_bytes = shared_bytes.saturating_add(tensor.byte_size);
                    }
                },
            }
        }
    }
    ensure!(
        bytes_by_layer.iter().all(|bytes| *bytes > 0),
        "GGUF tensor index does not account for every transformer layer"
    );
    bytes_by_layer[0] = bytes_by_layer[0].saturating_add(shared_bytes.div_ceil(2));
    let last = bytes_by_layer
        .last_mut()
        .context("GGUF has no transformer layers")?;
    *last = last.saturating_add(shared_bytes / 2);
    Ok(bytes_by_layer)
}

/// Inspect the already-authenticated metadata-only GGUF without loading model tensors.
pub fn inspect_package_geometry(
    metadata_path: &Path,
    cache_type_k: &str,
    cache_type_v: &str,
) -> Result<SkippyPackageGeometry> {
    inspect_compact_geometry(metadata_path, cache_type_k, cache_type_v, Vec::new())
}

/// Inspect one direct GGUF or an ordered split-GGUF set without loading tensors.
pub fn inspect_source_geometry(
    source_paths: &[PathBuf],
    cache_type_k: &str,
    cache_type_v: &str,
) -> Result<SkippyPackageGeometry> {
    ensure!(
        skippy_runtime::native_runtime_loaded(),
        "load a verified Skippy native runtime before inspecting direct GGUF tensors"
    );
    let metadata_path = source_paths
        .first()
        .context("Skippy source has no GGUF files")?;
    let metadata = scan_gguf_compact_meta(metadata_path)
        .with_context(|| format!("scan Skippy metadata {}", metadata_path.display()))?;
    let static_bytes_by_layer = direct_static_bytes_by_layer(source_paths, metadata.layer_count)?;
    inspect_compact_geometry(
        metadata_path,
        cache_type_k,
        cache_type_v,
        static_bytes_by_layer,
    )
}

fn inspect_compact_geometry(
    metadata_path: &Path,
    cache_type_k: &str,
    cache_type_v: &str,
    static_bytes_by_layer: Vec<u64>,
) -> Result<SkippyPackageGeometry> {
    let metadata = scan_gguf_compact_meta(metadata_path)
        .with_context(|| format!("scan Skippy metadata {}", metadata_path.display()))?;
    ensure!(
        !metadata.architecture.is_empty()
            && metadata.context_length > 0
            && metadata.embedding_size > 0
            && metadata.layer_count > 0,
        "Skippy package has incomplete GGUF architecture metadata"
    );
    let quant = GgufKvCacheQuant::from_llama_args(cache_type_k, cache_type_v)
        .context("unsupported signed Skippy KV cache type")?;
    let kv_bytes_per_token = quant
        .kv_cache_bytes_per_token(&metadata)
        .context("Skippy package has incomplete GGUF KV geometry")?;
    Ok(SkippyPackageGeometry {
        architecture: metadata.architecture,
        context_length: metadata.context_length,
        activation_width: metadata.embedding_size,
        layer_count: metadata.layer_count,
        kv_bytes_per_token,
        static_bytes_by_layer,
    })
}

#[derive(Clone, Debug)]
pub struct StageActivationFrame {
    inner: ActivationFrame,
}

impl StageActivationFrame {
    pub fn from_parts(
        version: u32,
        dtype: &str,
        layout: &str,
        producer_stage_index: i32,
        layer_start: i32,
        layer_end: i32,
        token_count: u32,
        sequence_count: u32,
        flags: u64,
        payload: Vec<u8>,
    ) -> Result<Self> {
        let dtype = match dtype.trim().to_ascii_lowercase().as_str() {
            "f32" => skippy_runtime::RuntimeActivationDType::F32,
            "f16" => skippy_runtime::RuntimeActivationDType::F16,
            "bf16" => skippy_runtime::RuntimeActivationDType::Bf16,
            value => bail!("unsupported Skippy activation dtype {value}"),
        };
        let layout = match layout.trim().to_ascii_lowercase().as_str() {
            "opaque" => skippy_runtime::RuntimeActivationLayout::Opaque,
            "token_major" | "token-major" => skippy_runtime::RuntimeActivationLayout::TokenMajor,
            value => bail!("unsupported Skippy activation layout {value}"),
        };
        ensure!(layer_end >= layer_start, "invalid activation layer range");
        ensure!(token_count > 0, "activation token count must be positive");
        ensure!(
            sequence_count > 0,
            "activation sequence count must be positive"
        );
        let payload_bytes = u64::try_from(payload.len()).context("activation exceeds u64")?;
        ensure!(payload_bytes > 0, "activation payload is empty");
        Ok(Self {
            inner: ActivationFrame {
                desc: skippy_runtime::ActivationDesc {
                    version,
                    dtype,
                    layout,
                    producer_stage_index,
                    layer_start,
                    layer_end,
                    token_count,
                    sequence_count,
                    payload_bytes,
                    flags,
                },
                payload,
            },
        })
    }

    #[must_use]
    pub fn version(&self) -> u32 {
        self.inner.desc.version
    }

    #[must_use]
    pub fn dtype(&self) -> &'static str {
        match self.inner.desc.dtype {
            skippy_runtime::RuntimeActivationDType::Unknown => "unknown",
            skippy_runtime::RuntimeActivationDType::F32 => "f32",
            skippy_runtime::RuntimeActivationDType::F16 => "f16",
            skippy_runtime::RuntimeActivationDType::Bf16 => "bf16",
        }
    }

    #[must_use]
    pub fn layout(&self) -> &'static str {
        match self.inner.desc.layout {
            skippy_runtime::RuntimeActivationLayout::Opaque => "opaque",
            skippy_runtime::RuntimeActivationLayout::TokenMajor => "token_major",
        }
    }

    #[must_use]
    pub fn producer_stage_index(&self) -> i32 {
        self.inner.desc.producer_stage_index
    }

    #[must_use]
    pub fn layer_start(&self) -> i32 {
        self.inner.desc.layer_start
    }

    #[must_use]
    pub fn layer_end(&self) -> i32 {
        self.inner.desc.layer_end
    }

    #[must_use]
    pub fn token_count(&self) -> u32 {
        self.inner.desc.token_count
    }

    #[must_use]
    pub fn sequence_count(&self) -> u32 {
        self.inner.desc.sequence_count
    }

    #[must_use]
    pub fn flags(&self) -> u64 {
        self.inner.desc.flags
    }

    #[must_use]
    pub fn payload(&self) -> &[u8] {
        &self.inner.payload
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct StageSamplingConfig {
    pub seed: u32,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: i32,
    pub min_p: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub repeat_penalty: f32,
    pub penalty_last_n: i32,
}

impl StageSamplingConfig {
    fn to_native(&self) -> SamplingConfig {
        SamplingConfig {
            enabled: true,
            seed: self.seed,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
            min_p: self.min_p,
            presence_penalty: self.presence_penalty,
            frequency_penalty: self.frequency_penalty,
            repeat_penalty: self.repeat_penalty,
            penalty_last_n: self.penalty_last_n,
            logit_bias: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct StageForwardOutput {
    pub predicted_token: Option<i32>,
    pub activation: StageActivationFrame,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageLoadMode {
    LayerPackage,
    RuntimeSlice,
}

impl StageLoadMode {
    pub fn parse(value: &str) -> Result<Self> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "layer_package" => Ok(Self::LayerPackage),
            "runtime_slice" => Ok(Self::RuntimeSlice),
            value => bail!("unsupported Skippy stage load mode {value}"),
        }
    }

    const fn native(self) -> RuntimeLoadMode {
        match self {
            Self::LayerPackage => RuntimeLoadMode::LayerPackage,
            Self::RuntimeSlice => RuntimeLoadMode::RuntimeSlice,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StageOpenOptions {
    pub stage_index: u32,
    pub layer_start: u32,
    pub layer_end: u32,
    pub model_layer_count: u32,
    pub context_tokens: u32,
    pub lane_count: u32,
    pub n_batch: Option<u32>,
    pub n_ubatch: Option<u32>,
    pub n_threads: Option<u32>,
    pub n_threads_batch: Option<u32>,
    pub selected_backend_device: Option<String>,
    pub cache_type_k: String,
    pub cache_type_v: String,
    pub use_mmap: Option<bool>,
    pub use_mlock: bool,
    pub load_mode: StageLoadMode,
}

/// Load an already-installed native bundle after Mesh's manifest contract and
/// Fabi's product-generated per-file integrity manifest have both passed.
///
/// The Fabi release archive containing `root` is authenticated before launch;
/// this second verification prevents partial/corrupt installations and binds
/// the runtime selected for this worker to its signed execution plan.
pub fn load_verified_native_runtime(
    root: &Path,
    expected_mesh_release: &str,
    expected_abi: &str,
    expected_backend: &str,
) -> Result<Vec<NativeDeviceInfo>> {
    ensure!(
        !skippy_runtime::native_runtime_loaded(),
        "a Skippy native runtime is already loaded in this process"
    );
    let root = root
        .canonicalize()
        .with_context(|| format!("canonicalize native runtime root {}", root.display()))?;
    let manifest = NativeRuntimeManifest::read_from_dir(&root)?;
    let runtime = &manifest.runtime;
    ensure!(
        runtime.mesh_version.as_deref() == Some(expected_mesh_release),
        "native runtime release mismatch: expected {expected_mesh_release}, got {}",
        runtime.mesh_version.as_deref().unwrap_or("missing")
    );
    ensure!(
        runtime.skippy_abi == expected_abi,
        "native runtime ABI mismatch: expected {expected_abi}, got {}",
        runtime.skippy_abi
    );
    let expected_backend = expected_backend.trim().to_ascii_lowercase();
    ensure!(
        runtime.backend.kind.as_str() == expected_backend,
        "native runtime backend mismatch: expected {expected_backend}, got {}",
        runtime.backend.kind
    );
    ensure!(
        !matches!(runtime.backend.kind, NativeRuntimeBackendKind::Other(_)),
        "unsupported Skippy native runtime backend {}",
        runtime.backend.kind
    );
    let libraries = verify_fabi_runtime_integrity(
        &root,
        &runtime.id,
        &runtime.libraries,
        expected_mesh_release,
        expected_abi,
        &expected_backend,
    )?;

    // Safety: `verify_fabi_runtime_integrity` rejects path traversal/symlink
    // escape, requires regular files inside `root`, and checks their SHA-256
    // against the integrity manifest authenticated by the Fabi product archive.
    // The exact Mesh release, ABI, runtime ID, and backend are bound above. No
    // other Fabi crate is permitted to load a Skippy dynamic library.
    unsafe { skippy_runtime::load_native_runtime_libraries(&libraries) }
        .context("load verified Skippy native runtime libraries")?;
    ensure!(
        skippy_runtime::native_runtime_loaded(),
        "Skippy runtime loader returned without activating the ABI"
    );
    let devices = backend_devices().context("enumerate Skippy backend devices")?;
    ensure!(
        !devices.is_empty(),
        "verified Skippy runtime exposed no backend device"
    );
    Ok(devices
        .into_iter()
        .map(|device| NativeDeviceInfo {
            name: device.name,
            description: device.description,
            device_id: device.device_id,
            memory_free: device.memory_free,
            memory_total: device.memory_total,
            kind: match device.device_type {
                BackendDeviceType::Cpu => "cpu",
                BackendDeviceType::Gpu => "gpu",
                BackendDeviceType::IntegratedGpu => "integrated_gpu",
                BackendDeviceType::Accelerator => "accelerator",
                BackendDeviceType::Meta => "meta",
            }
            .to_string(),
            caps: device.caps,
        })
        .collect())
}

/// One contiguous Skippy model span. Sessions are generation-scoped and own
/// their native KV state; removing a session releases that state immediately.
pub struct SkippyStage {
    model: StageModel,
    sessions: HashMap<String, StageSession>,
    layer_start: i32,
    layer_end: i32,
    lane_count: usize,
}

impl SkippyStage {
    pub fn open(parts: &[PathBuf], options: &StageOpenOptions) -> Result<Self> {
        ensure!(
            skippy_runtime::native_runtime_loaded(),
            "load a verified Skippy native runtime before opening a stage"
        );
        ensure!(
            !parts.is_empty(),
            "Skippy stage has no verified package parts"
        );
        ensure!(
            options.layer_start < options.layer_end
                && options.layer_end <= options.model_layer_count,
            "invalid Skippy stage layer range"
        );
        ensure!(
            options.context_tokens > 0,
            "Skippy context must be positive"
        );
        ensure!(options.lane_count > 0, "Skippy lane count must be positive");
        for part in parts {
            ensure!(
                part.is_file(),
                "Skippy package part is missing: {}",
                part.display()
            );
        }
        let config = RuntimeConfig {
            stage_index: options.stage_index,
            layer_start: options.layer_start,
            layer_end: options.layer_end,
            ctx_size: options.context_tokens,
            lane_count: options.lane_count,
            n_batch: options.n_batch,
            n_ubatch: options.n_ubatch,
            n_threads: options.n_threads,
            n_threads_batch: options.n_threads_batch,
            n_gpu_layers: -1,
            mmap: options.use_mmap,
            mlock: options.use_mlock,
            selected_backend_device: options.selected_backend_device.clone(),
            cache_type_k: parse_cache_type(&options.cache_type_k)?,
            cache_type_v: parse_cache_type(&options.cache_type_v)?,
            flash_attn_type: FlashAttentionType::Auto,
            load_mode: options.load_mode.native(),
            projector_path: None,
            include_embeddings: options.layer_start == 0,
            include_output: options.layer_end == options.model_layer_count,
            filter_tensors_on_load: true,
        };
        let model = StageModel::open_from_parts(parts, &config)
            .context("open verified Skippy stage source")?;
        Ok(Self {
            model,
            sessions: HashMap::new(),
            layer_start: i32::try_from(options.layer_start).context("layer_start exceeds i32")?,
            layer_end: i32::try_from(options.layer_end).context("layer_end exceeds i32")?,
            lane_count: usize::try_from(options.lane_count).context("lane_count exceeds usize")?,
        })
    }

    fn session(&mut self, session_id: &str) -> Result<&mut StageSession> {
        ensure!(!session_id.trim().is_empty(), "Skippy session ID is empty");
        if !self.sessions.contains_key(session_id) {
            ensure!(
                self.sessions.len() < self.lane_count,
                "all Skippy execution lanes are busy"
            );
            self.sessions
                .insert(session_id.to_string(), self.model.create_session()?);
        }
        self.sessions
            .get_mut(session_id)
            .context("Skippy session admission failed")
    }

    pub fn prefill(
        &mut self,
        session_id: &str,
        token_ids: &[i32],
        input: Option<&StageActivationFrame>,
        sampling: Option<&StageSamplingConfig>,
    ) -> Result<StageForwardOutput> {
        ensure!(!token_ids.is_empty(), "Skippy prefill chunk is empty");
        let session = self.session(session_id)?;
        match sampling {
            Some(sampling) => {
                let sampling = sampling.to_native();
                let (predicted_token, activation) = session.prefill_chunk_frame_sampled(
                    token_ids,
                    Some(&sampling),
                    input.map(|frame| &frame.inner),
                    0,
                )?;
                Ok(StageForwardOutput {
                    predicted_token: Some(predicted_token),
                    activation: StageActivationFrame { inner: activation },
                })
            }
            None => Ok(StageForwardOutput {
                predicted_token: None,
                activation: StageActivationFrame {
                    inner: session.prefill_chunk_frame(
                        token_ids,
                        input.map(|frame| &frame.inner),
                        0,
                    )?,
                },
            }),
        }
    }

    pub fn decode(
        &mut self,
        session_id: &str,
        token_id: i32,
        input: Option<&StageActivationFrame>,
        sampling: Option<&StageSamplingConfig>,
    ) -> Result<StageForwardOutput> {
        let native_sampling = sampling.map(StageSamplingConfig::to_native);
        let (predicted_token, activation) = self.session(session_id)?.decode_step_frame_sampled(
            token_id,
            native_sampling.as_ref(),
            input.map(|frame| &frame.inner),
            0,
        )?;
        Ok(StageForwardOutput {
            predicted_token: sampling.map(|_| predicted_token),
            activation: StageActivationFrame { inner: activation },
        })
    }

    pub fn reset_session(&mut self, session_id: &str) -> Result<()> {
        self.sessions
            .get_mut(session_id)
            .with_context(|| format!("unknown Skippy session {session_id}"))?
            .reset()
    }

    pub fn drop_session(&mut self, session_id: &str) -> bool {
        self.sessions.remove(session_id).is_some()
    }

    pub fn session_token_count(&self, session_id: &str) -> Result<u64> {
        Ok(self
            .sessions
            .get(session_id)
            .with_context(|| format!("unknown Skippy session {session_id}"))?
            .token_count())
    }

    pub fn export_full_state(&mut self, session_id: &str) -> Result<Vec<u8>> {
        let layer_start = self.layer_start;
        let layer_end = self.layer_end;
        self.session(session_id)?
            .export_full_state(layer_start, layer_end)
    }

    pub fn import_full_state(
        &mut self,
        session_id: &str,
        payload: &[u8],
        token_count: u64,
    ) -> Result<()> {
        ensure!(!payload.is_empty(), "Skippy full-state payload is empty");
        let layer_start = self.layer_start;
        let layer_end = self.layer_end;
        self.session(session_id)?.import_full_state_for_token_count(
            layer_start,
            layer_end,
            payload,
            token_count,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    fn push_string(bytes: &mut Vec<u8>, value: &str) {
        bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
        bytes.extend_from_slice(value.as_bytes());
    }

    fn push_string_kv(bytes: &mut Vec<u8>, key: &str, value: &str) {
        push_string(bytes, key);
        bytes.extend_from_slice(&8_u32.to_le_bytes());
        push_string(bytes, value);
    }

    fn push_u32_kv(bytes: &mut Vec<u8>, key: &str, value: u32) {
        push_string(bytes, key);
        bytes.extend_from_slice(&4_u32.to_le_bytes());
        bytes.extend_from_slice(&value.to_le_bytes());
    }

    fn write_qwen_metadata(path: &Path) {
        let mut bytes = b"GGUF".to_vec();
        bytes.extend_from_slice(&3_u32.to_le_bytes());
        bytes.extend_from_slice(&0_u64.to_le_bytes());
        bytes.extend_from_slice(&8_u64.to_le_bytes());
        push_string_kv(&mut bytes, "general.architecture", "qwen3");
        push_u32_kv(&mut bytes, "qwen3.context_length", 40_960);
        push_u32_kv(&mut bytes, "qwen3.embedding_length", 1_024);
        push_u32_kv(&mut bytes, "qwen3.attention.head_count", 16);
        push_u32_kv(&mut bytes, "qwen3.attention.head_count_kv", 8);
        push_u32_kv(&mut bytes, "qwen3.block_count", 28);
        push_u32_kv(&mut bytes, "qwen3.attention.key_length", 128);
        push_u32_kv(&mut bytes, "qwen3.attention.value_length", 128);
        fs::write(path, bytes).unwrap();
    }

    fn write_integrity(root: &Path) {
        let manifest_hash = file_sha256(&root.join("manifest.json")).unwrap();
        let library_hash = file_sha256(&root.join("lib/runtime.bin")).unwrap();
        fs::write(
            root.join(FABI_INTEGRITY_MANIFEST),
            format!(
                r#"{{
  "schema_version": 1,
  "mesh_release": "0.74.0",
  "skippy_abi": "0.1.32",
  "runtime_id": "runtime-id",
  "backend": "metal",
  "files": [
    {{"path": "manifest.json", "sha256": "{manifest_hash}"}},
    {{"path": "lib/runtime.bin", "sha256": "{library_hash}"}}
  ]
}}"#
            ),
        )
        .unwrap();
    }

    #[test]
    fn product_integrity_covers_exact_runtime_files_and_rejects_mutation() {
        let temp = tempfile::tempdir().unwrap();
        fs::create_dir(temp.path().join("lib")).unwrap();
        fs::write(temp.path().join("manifest.json"), b"mesh-manifest").unwrap();
        fs::write(temp.path().join("lib/runtime.bin"), b"native-library").unwrap();
        write_integrity(temp.path());
        let root = temp.path().canonicalize().unwrap();
        let declared = vec!["lib/runtime.bin".to_string()];

        let paths = verify_fabi_runtime_integrity(
            &root,
            "runtime-id",
            &declared,
            "0.74.0",
            "0.1.32",
            "metal",
        )
        .unwrap();
        assert_eq!(paths, vec![root.join("lib/runtime.bin")]);

        fs::write(root.join("lib/runtime.bin"), b"tampered").unwrap();
        let error = verify_fabi_runtime_integrity(
            &root,
            "runtime-id",
            &declared,
            "0.74.0",
            "0.1.32",
            "metal",
        )
        .unwrap_err();
        assert!(error.to_string().contains("file hash mismatch"));
    }

    #[test]
    fn runtime_path_rejects_parent_and_absolute_components() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().canonicalize().unwrap();

        assert!(checked_runtime_file(&root, "../escape").is_err());
        assert!(checked_runtime_file(&root, "/absolute").is_err());
    }

    #[test]
    fn package_geometry_uses_mesh_gguf_parser_and_exact_f16_kv_bytes() {
        let temp = tempfile::tempdir().unwrap();
        let metadata = temp.path().join("metadata.gguf");
        write_qwen_metadata(&metadata);

        let geometry = inspect_package_geometry(&metadata, "f16", "f16").unwrap();

        assert_eq!(geometry.architecture, "qwen3");
        assert_eq!(geometry.context_length, 40_960);
        assert_eq!(geometry.activation_width, 1_024);
        assert_eq!(geometry.layer_count, 28);
        assert_eq!(geometry.kv_bytes_per_token, 28 * 4_096);
    }

    #[test]
    fn stage_load_mode_accepts_only_official_skippy_modes() {
        assert_eq!(
            StageLoadMode::parse("runtime-slice").unwrap(),
            StageLoadMode::RuntimeSlice
        );
        assert_eq!(
            StageLoadMode::parse("layer_package").unwrap(),
            StageLoadMode::LayerPackage
        );
        assert!(StageLoadMode::parse("automatic").is_err());
    }
}
