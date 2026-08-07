//! Safe access to the Windows video-memory budget used by `DirectML`.
//!
//! DXGI owns the authoritative per-process budget.  The raw Windows methods
//! are marked `unsafe` because they fill caller-provided output structures;
//! this crate contains that boundary so the network/Python extension can keep
//! forbidding unsafe code entirely.

use anyhow::Result;

#[cfg(not(windows))]
use anyhow::bail;

/// One DXGI memory-segment observation, in bytes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct VideoMemorySegment {
    pub budget: u64,
    pub current_usage: u64,
    pub available_for_reservation: u64,
    pub current_reservation: u64,
}

impl VideoMemorySegment {
    /// Additional bytes the current process can consume without exceeding the
    /// operating-system budget at the instant of the observation.
    #[must_use]
    pub const fn headroom(self) -> u64 {
        self.budget.saturating_sub(self.current_usage)
    }
}

/// Adapter identity and both DXGI memory-segment budgets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VideoMemoryInfo {
    pub adapter_index: u32,
    pub description: String,
    pub vendor_id: u32,
    pub device_id: u32,
    pub dedicated_video_memory: u64,
    pub dedicated_system_memory: u64,
    pub shared_system_memory: u64,
    pub local: VideoMemorySegment,
    pub non_local: VideoMemorySegment,
}

#[cfg(windows)]
#[allow(unsafe_code)]
mod platform {
    use anyhow::{Context, Result};
    use windows::{
        Win32::Graphics::Dxgi::{
            CreateDXGIFactory1, DXGI_MEMORY_SEGMENT_GROUP_LOCAL,
            DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, DXGI_QUERY_VIDEO_MEMORY_INFO, IDXGIAdapter3,
            IDXGIFactory1,
        },
        core::Interface,
    };

    use super::{VideoMemoryInfo, VideoMemorySegment};

    fn query_segment(
        adapter: &IDXGIAdapter3,
        group: windows::Win32::Graphics::Dxgi::DXGI_MEMORY_SEGMENT_GROUP,
    ) -> Result<VideoMemorySegment> {
        let mut raw = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: `raw` is a valid, uniquely borrowed output structure for the
        // duration of the COM call.  Node 0 is the documented single-adapter
        // node queried by DirectML/ONNX Runtime for the selected DXGI adapter.
        unsafe { adapter.QueryVideoMemoryInfo(0, group, &raw mut raw) }
            .context("IDXGIAdapter3::QueryVideoMemoryInfo failed")?;
        Ok(VideoMemorySegment {
            budget: raw.Budget,
            current_usage: raw.CurrentUsage,
            available_for_reservation: raw.AvailableForReservation,
            current_reservation: raw.CurrentReservation,
        })
    }

    pub fn query(adapter_index: u32) -> Result<VideoMemoryInfo> {
        // SAFETY: `CreateDXGIFactory1`, adapter enumeration and descriptor
        // retrieval are Windows COM calls whose generated wrappers enforce
        // the returned interface types.  No raw pointer escapes this module.
        let factory: IDXGIFactory1 =
            unsafe { CreateDXGIFactory1() }.context("CreateDXGIFactory1 failed")?;
        let adapter = unsafe { factory.EnumAdapters1(adapter_index) }
            .with_context(|| format!("DXGI adapter {adapter_index} is unavailable"))?;
        let description =
            unsafe { adapter.GetDesc1() }.context("IDXGIAdapter1::GetDesc1 failed")?;
        let adapter3: IDXGIAdapter3 = adapter
            .cast()
            .context("DXGI adapter does not implement IDXGIAdapter3")?;
        let nul = description
            .Description
            .iter()
            .position(|code_unit| *code_unit == 0)
            .unwrap_or(description.Description.len());

        Ok(VideoMemoryInfo {
            adapter_index,
            description: String::from_utf16_lossy(&description.Description[..nul]),
            vendor_id: description.VendorId,
            device_id: description.DeviceId,
            dedicated_video_memory: u64::try_from(description.DedicatedVideoMemory)
                .context("dedicated video-memory size does not fit u64")?,
            dedicated_system_memory: u64::try_from(description.DedicatedSystemMemory)
                .context("dedicated system-memory size does not fit u64")?,
            shared_system_memory: u64::try_from(description.SharedSystemMemory)
                .context("shared system-memory size does not fit u64")?,
            local: query_segment(&adapter3, DXGI_MEMORY_SEGMENT_GROUP_LOCAL)
                .context("failed to query local video-memory segment")?,
            non_local: query_segment(&adapter3, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL)
                .context("failed to query non-local video-memory segment")?,
        })
    }
}

/// Query current Windows video-memory budgets for a DXGI adapter number.
///
/// The adapter number is the same identifier surfaced by ONNX Runtime's
/// `DirectML` execution-provider metadata.  Non-Windows callers fail closed.
///
/// # Errors
///
/// Returns an error when the adapter is unavailable, does not implement
/// `IDXGIAdapter3`, or Windows cannot provide both memory-segment budgets.
pub fn query_video_memory(adapter_index: u32) -> Result<VideoMemoryInfo> {
    #[cfg(windows)]
    {
        platform::query(adapter_index)
    }
    #[cfg(not(windows))]
    {
        let _ = adapter_index;
        bail!("DXGI video-memory telemetry is only available on Windows")
    }
}

#[cfg(test)]
mod tests {
    use super::VideoMemorySegment;
    #[cfg(windows)]
    use super::query_video_memory;

    #[test]
    fn headroom_saturates_when_usage_exceeds_budget() {
        let segment = VideoMemorySegment {
            budget: 100,
            current_usage: 125,
            available_for_reservation: 20,
            current_reservation: 0,
        };
        assert_eq!(segment.headroom(), 0);
    }

    #[cfg(windows)]
    #[test]
    fn live_adapter_budget_is_internally_consistent_when_available() {
        let Ok(info) = query_video_memory(0) else {
            // Headless Windows CI images are allowed to have no DXGI adapter.
            return;
        };
        assert!(!info.description.is_empty());
        assert_eq!(
            info.local.headroom(),
            info.local.budget.saturating_sub(info.local.current_usage)
        );
        assert_eq!(
            info.non_local.headroom(),
            info.non_local
                .budget
                .saturating_sub(info.non_local.current_usage)
        );
        eprintln!("{info:#?}");
    }
}
