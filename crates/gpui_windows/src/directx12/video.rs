//! Media Foundation's D3D11 facade is created on demand. UI rendering never
//! creates a D3D11 device or context. The facade shares the physical D3D12
//! device but uses a video producer queue to avoid translation-layer self-waits.

use anyhow::{Context, Result};
use windows::{
    Win32::Graphics::{
        Direct3D::D3D_FEATURE_LEVEL_11_0,
        Direct3D11::{
            D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_VIDEO_SUPPORT, ID3D11Device,
            ID3D11DeviceContext, ID3D11Multithread,
        },
        Direct3D11on12::{D3D11On12CreateDevice, ID3D11On12Device2},
        Direct3D12::{
            D3D12_COMMAND_LIST_TYPE_DIRECT, D3D12_COMMAND_QUEUE_DESC, ID3D12CommandQueue,
        },
    },
    core::Interface,
};

use super::device::Device;

/// Owns the renderer generation while Media Foundation borrows its D3D11 ABI.
pub struct RendererD3D11Handle {
    /// ID3D11Device owned by this lease.
    pub device: *mut std::ffi::c_void,
    /// Changes when GPUI replaces its physical renderer device.
    pub generation: u64,
    /// Retains the native renderer and its lazily created video facade.
    pub keep_alive: std::sync::Arc<dyn std::any::Any + Send + Sync>,
}

fn current() -> Option<(std::sync::Arc<Device>, u64)> {
    let current = super::CURRENT
        .lock()
        .unwrap_or_else(|error| error.into_inner());
    Some((current.device.upgrade()?, current.generation))
}

/// Returns only an existing video bridge. Calling this during the frame pump
/// does not initialize D3D11, its immediate context, or the video driver.
pub fn renderer_d3d11_device() -> (*mut std::ffi::c_void, u64) {
    let Some((native, generation)) = current() else {
        return (std::ptr::null_mut(), 0);
    };
    let pointer = native
        .existing_video_bridge()
        .map(|bridge| bridge.device.as_raw())
        .unwrap_or(std::ptr::null_mut());
    (pointer, generation)
}

/// Called by the video backend when it actually needs a decoder device.
pub fn ensure_renderer_d3d11_device() -> Result<Option<RendererD3D11Handle>> {
    let Some((native, generation)) = current() else {
        return Ok(None);
    };
    let bridge = native.video_bridge()?;
    Ok(Some(RendererD3D11Handle {
        device: bridge.device.as_raw(),
        generation,
        keep_alive: std::sync::Arc::new((native, bridge)),
    }))
}

pub(super) struct VideoBridge {
    pub(super) device: ID3D11Device,
    _context: ID3D11DeviceContext,
    _on12: ID3D11On12Device2,
    _queue: ID3D12CommandQueue,
}

impl VideoBridge {
    pub(super) fn new(native: &Device) -> Result<Self> {
        let mut device = None;
        let mut context = None;
        // Older D3D11On12 runtimes insert a wait before their own signal when
        // UnwrapUnderlyingResource targets the producer queue (Microsoft #51).
        // A separate queue on the same physical device avoids that self-wait.
        let queue: ID3D12CommandQueue = unsafe {
            native
                .device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })?
        };
        let queues = [Some(queue.cast()?)];
        unsafe {
            D3D11On12CreateDevice(
                &native.device,
                (D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_VIDEO_SUPPORT).0,
                Some(&[D3D_FEATURE_LEVEL_11_0]),
                Some(&queues),
                0,
                Some(&mut device),
                Some(&mut context),
                None,
            )?;
        }
        let device: ID3D11Device = device.context("D3D11On12 returned no video device")?;
        let context = context.context("D3D11On12 returned no immediate context")?;
        let protection: ID3D11Multithread = device
            .cast()
            .context("D3D11On12 video multithread protection")?;
        let was_protected = unsafe { protection.SetMultithreadProtected(true) };
        log::debug!("D3D11On12 video context previously multithread protected: {was_protected:?}");
        let on12 = device
            .cast()
            .context("D3D11On12 resource handoff requires Windows 10 2004 or newer")?;
        log::info!(
            "Created lazy D3D11On12 Media Foundation bridge on the native D3D12 renderer device"
        );
        Ok(Self {
            device,
            _context: context,
            _on12: on12,
            _queue: queue,
        })
    }
}

pub(super) struct ContextGuard(ID3D11Multithread);

impl ContextGuard {
    pub(super) fn lock(device: &ID3D11Device) -> Result<Self> {
        let protection: ID3D11Multithread = device.cast()?;
        unsafe {
            protection.Enter();
        }
        Ok(Self(protection))
    }
}

impl Drop for ContextGuard {
    fn drop(&mut self) {
        unsafe {
            self.0.Leave();
        }
    }
}
