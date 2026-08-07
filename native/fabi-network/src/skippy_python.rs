use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
};

use fabi_skippy_runtime::{
    NativeDeviceInfo, SkippyPackageGeometry, SkippyStage, StageActivationFrame, StageForwardOutput,
    StageLoadMode, StageOpenOptions, StageSamplingConfig, inspect_package_geometry,
    inspect_source_geometry, load_verified_native_runtime,
};
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyBytes, PyModule},
};

fn py_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

#[pyclass(name = "SkippyNativeDevice", frozen)]
#[derive(Clone)]
pub(crate) struct PySkippyNativeDevice {
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    description: Option<String>,
    #[pyo3(get)]
    device_id: Option<String>,
    #[pyo3(get)]
    memory_free: u64,
    #[pyo3(get)]
    memory_total: u64,
    #[pyo3(get)]
    kind: String,
    #[pyo3(get)]
    caps: u64,
}

impl From<NativeDeviceInfo> for PySkippyNativeDevice {
    fn from(value: NativeDeviceInfo) -> Self {
        Self {
            name: value.name,
            description: value.description,
            device_id: value.device_id,
            memory_free: value.memory_free,
            memory_total: value.memory_total,
            kind: value.kind,
            caps: value.caps,
        }
    }
}

#[pyclass(name = "SkippyPackageGeometry", frozen)]
#[derive(Clone)]
pub(crate) struct PySkippyPackageGeometry {
    #[pyo3(get)]
    architecture: String,
    #[pyo3(get)]
    context_length: u32,
    #[pyo3(get)]
    activation_width: u32,
    #[pyo3(get)]
    layer_count: u32,
    #[pyo3(get)]
    kv_bytes_per_token: u64,
    #[pyo3(get)]
    static_bytes_by_layer: Vec<u64>,
}

impl From<SkippyPackageGeometry> for PySkippyPackageGeometry {
    fn from(value: SkippyPackageGeometry) -> Self {
        Self {
            architecture: value.architecture,
            context_length: value.context_length,
            activation_width: value.activation_width,
            layer_count: value.layer_count,
            kv_bytes_per_token: value.kv_bytes_per_token,
            static_bytes_by_layer: value.static_bytes_by_layer,
        }
    }
}

#[pyclass(name = "SkippyActivationFrame", frozen)]
#[derive(Clone)]
pub(crate) struct PySkippyActivationFrame {
    inner: StageActivationFrame,
}

#[pymethods]
impl PySkippyActivationFrame {
    #[new]
    #[pyo3(signature = (
        payload,
        *,
        version,
        dtype,
        layout,
        producer_stage_index,
        layer_start,
        layer_end,
        token_count,
        sequence_count,
        flags=0,
    ))]
    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn new(
        payload: Vec<u8>,
        version: u32,
        dtype: &str,
        layout: &str,
        producer_stage_index: i32,
        layer_start: i32,
        layer_end: i32,
        token_count: u32,
        sequence_count: u32,
        flags: u64,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: StageActivationFrame::from_parts(
                version,
                dtype,
                layout,
                producer_stage_index,
                layer_start,
                layer_end,
                token_count,
                sequence_count,
                flags,
                payload,
            )
            .map_err(py_error)?,
        })
    }

    #[getter]
    fn version(&self) -> u32 {
        self.inner.version()
    }

    #[getter]
    fn dtype(&self) -> &'static str {
        self.inner.dtype()
    }

    #[getter]
    fn layout(&self) -> &'static str {
        self.inner.layout()
    }

    #[getter]
    fn producer_stage_index(&self) -> i32 {
        self.inner.producer_stage_index()
    }

    #[getter]
    fn layer_start(&self) -> i32 {
        self.inner.layer_start()
    }

    #[getter]
    fn layer_end(&self) -> i32 {
        self.inner.layer_end()
    }

    #[getter]
    fn token_count(&self) -> u32 {
        self.inner.token_count()
    }

    #[getter]
    fn sequence_count(&self) -> u32 {
        self.inner.sequence_count()
    }

    #[getter]
    fn flags(&self) -> u64 {
        self.inner.flags()
    }

    fn payload<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.inner.payload())
    }
}

#[pyclass(name = "SkippyForwardOutput", frozen)]
pub(crate) struct PySkippyForwardOutput {
    #[pyo3(get)]
    predicted_token: Option<i32>,
    activation: PySkippyActivationFrame,
}

impl From<StageForwardOutput> for PySkippyForwardOutput {
    fn from(value: StageForwardOutput) -> Self {
        Self {
            predicted_token: value.predicted_token,
            activation: PySkippyActivationFrame {
                inner: value.activation,
            },
        }
    }
}

#[pymethods]
impl PySkippyForwardOutput {
    #[getter]
    fn activation(&self) -> PySkippyActivationFrame {
        self.activation.clone()
    }
}

#[pyclass(name = "SkippyStage")]
pub(crate) struct PySkippyStage {
    inner: Arc<Mutex<SkippyStage>>,
}

#[pymethods]
impl PySkippyStage {
    #[new]
    #[pyo3(signature = (
        part_paths,
        *,
        stage_index,
        layer_start,
        layer_end,
        model_layer_count,
        context_tokens,
        lane_count=1,
        n_batch=None,
        n_ubatch=None,
        n_threads=None,
        n_threads_batch=None,
        selected_backend_device=None,
        cache_type_k="f16",
        cache_type_v="f16",
        use_mmap=None,
        use_mlock=false,
        load_mode="layer_package",
    ))]
    #[allow(clippy::similar_names, clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        part_paths: Vec<PathBuf>,
        stage_index: u32,
        layer_start: u32,
        layer_end: u32,
        model_layer_count: u32,
        context_tokens: u32,
        lane_count: u32,
        n_batch: Option<u32>,
        n_ubatch: Option<u32>,
        n_threads: Option<u32>,
        n_threads_batch: Option<u32>,
        selected_backend_device: Option<String>,
        cache_type_k: &str,
        cache_type_v: &str,
        use_mmap: Option<bool>,
        use_mlock: bool,
        load_mode: &str,
    ) -> PyResult<Self> {
        let options = StageOpenOptions {
            stage_index,
            layer_start,
            layer_end,
            model_layer_count,
            context_tokens,
            lane_count,
            n_batch,
            n_ubatch,
            n_threads,
            n_threads_batch,
            selected_backend_device,
            cache_type_k: cache_type_k.to_string(),
            cache_type_v: cache_type_v.to_string(),
            use_mmap,
            use_mlock,
            load_mode: StageLoadMode::parse(load_mode).map_err(py_error)?,
        };
        let stage = py
            .detach(move || SkippyStage::open(&part_paths, &options))
            .map_err(py_error)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(stage)),
        })
    }

    #[pyo3(signature = (
        session_id,
        token_ids,
        input=None,
        *,
        sample=false,
        seed=0,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repeat_penalty=1.0,
        penalty_last_n=-1,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn prefill(
        &self,
        py: Python<'_>,
        session_id: String,
        token_ids: Vec<i32>,
        input: Option<PyRef<'_, PySkippyActivationFrame>>,
        sample: bool,
        seed: u32,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        min_p: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
        repeat_penalty: f32,
        penalty_last_n: i32,
    ) -> PyResult<PySkippyForwardOutput> {
        let inner = Arc::clone(&self.inner);
        let input = input.map(|frame| frame.inner.clone());
        let sampling = sample.then_some(StageSamplingConfig {
            seed,
            temperature,
            top_p,
            top_k,
            min_p,
            presence_penalty,
            frequency_penalty,
            repeat_penalty,
            penalty_last_n,
        });
        py.detach(move || {
            inner
                .lock()
                .map_err(|_| anyhow::anyhow!("Skippy stage lock poisoned"))?
                .prefill(&session_id, &token_ids, input.as_ref(), sampling.as_ref())
        })
        .map(Into::into)
        .map_err(py_error)
    }

    #[pyo3(signature = (
        session_id,
        token_id,
        input=None,
        *,
        sample=false,
        seed=0,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        min_p=0.0,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repeat_penalty=1.0,
        penalty_last_n=-1,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        py: Python<'_>,
        session_id: String,
        token_id: i32,
        input: Option<PyRef<'_, PySkippyActivationFrame>>,
        sample: bool,
        seed: u32,
        temperature: f32,
        top_p: f32,
        top_k: i32,
        min_p: f32,
        presence_penalty: f32,
        frequency_penalty: f32,
        repeat_penalty: f32,
        penalty_last_n: i32,
    ) -> PyResult<PySkippyForwardOutput> {
        let inner = Arc::clone(&self.inner);
        let input = input.map(|frame| frame.inner.clone());
        let sampling = sample.then_some(StageSamplingConfig {
            seed,
            temperature,
            top_p,
            top_k,
            min_p,
            presence_penalty,
            frequency_penalty,
            repeat_penalty,
            penalty_last_n,
        });
        py.detach(move || {
            inner
                .lock()
                .map_err(|_| anyhow::anyhow!("Skippy stage lock poisoned"))?
                .decode(&session_id, token_id, input.as_ref(), sampling.as_ref())
        })
        .map(Into::into)
        .map_err(py_error)
    }

    fn reset_session(&self, py: Python<'_>, session_id: String) -> PyResult<()> {
        let inner = Arc::clone(&self.inner);
        py.detach(move || {
            inner
                .lock()
                .map_err(|_| anyhow::anyhow!("Skippy stage lock poisoned"))?
                .reset_session(&session_id)
        })
        .map_err(py_error)
    }

    fn drop_session(&self, session_id: &str) -> PyResult<bool> {
        Ok(self
            .inner
            .lock()
            .map_err(|_| py_error("Skippy stage lock poisoned"))?
            .drop_session(session_id))
    }

    fn session_token_count(&self, session_id: &str) -> PyResult<u64> {
        self.inner
            .lock()
            .map_err(|_| py_error("Skippy stage lock poisoned"))?
            .session_token_count(session_id)
            .map_err(py_error)
    }

    fn export_full_state<'py>(
        &self,
        py: Python<'py>,
        session_id: String,
    ) -> PyResult<Bound<'py, PyBytes>> {
        let inner = Arc::clone(&self.inner);
        let payload = py
            .detach(move || {
                inner
                    .lock()
                    .map_err(|_| anyhow::anyhow!("Skippy stage lock poisoned"))?
                    .export_full_state(&session_id)
            })
            .map_err(py_error)?;
        Ok(PyBytes::new(py, &payload))
    }

    fn import_full_state(
        &self,
        py: Python<'_>,
        session_id: String,
        payload: Vec<u8>,
        token_count: u64,
    ) -> PyResult<()> {
        let inner = Arc::clone(&self.inner);
        py.detach(move || {
            inner
                .lock()
                .map_err(|_| anyhow::anyhow!("Skippy stage lock poisoned"))?
                .import_full_state(&session_id, &payload, token_count)
        })
        .map_err(py_error)
    }
}

#[pyfunction(name = "load_skippy_native_runtime")]
fn py_load_skippy_native_runtime(
    py: Python<'_>,
    root: PathBuf,
    expected_mesh_release: String,
    expected_abi: String,
    expected_backend: String,
) -> PyResult<Vec<PySkippyNativeDevice>> {
    py.detach(move || {
        load_verified_native_runtime(
            &root,
            &expected_mesh_release,
            &expected_abi,
            &expected_backend,
        )
    })
    .map(|devices| devices.into_iter().map(Into::into).collect())
    .map_err(py_error)
}

#[pyfunction(name = "inspect_skippy_package_geometry")]
#[allow(clippy::needless_pass_by_value)] // PyO3 extracts Python path objects into owned PathBufs.
fn py_inspect_skippy_package_geometry(
    metadata_path: PathBuf,
    cache_type_k: &str,
    cache_type_v: &str,
) -> PyResult<PySkippyPackageGeometry> {
    inspect_package_geometry(&metadata_path, cache_type_k, cache_type_v)
        .map(Into::into)
        .map_err(py_error)
}

#[pyfunction(name = "inspect_skippy_source_geometry")]
fn py_inspect_skippy_source_geometry(
    source_paths: Vec<PathBuf>,
    cache_type_k: &str,
    cache_type_v: &str,
) -> PyResult<PySkippyPackageGeometry> {
    inspect_source_geometry(&source_paths, cache_type_k, cache_type_v)
        .map(Into::into)
        .map_err(py_error)
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(py_load_skippy_native_runtime, module)?)?;
    module.add_function(wrap_pyfunction!(
        py_inspect_skippy_package_geometry,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(py_inspect_skippy_source_geometry, module)?)?;
    module.add_class::<PySkippyNativeDevice>()?;
    module.add_class::<PySkippyPackageGeometry>()?;
    module.add_class::<PySkippyActivationFrame>()?;
    module.add_class::<PySkippyForwardOutput>()?;
    module.add_class::<PySkippyStage>()?;
    Ok(())
}
