use anyhow::Result;
use std::sync::{Arc, LazyLock, Mutex, OnceLock, Weak};
use windows::{Win32::Graphics::Direct3D12::ID3D12Device, core::Interface};

mod atlas;
mod device;
pub(crate) mod glyphs;
mod pipelines;
mod renderer;
mod video;
pub use video::{RendererD3D11Handle, ensure_renderer_d3d11_device, renderer_d3d11_device};

pub(crate) use renderer::DirectXRenderer;

struct Published {
    generation: u64,
    device: Weak<device::Device>,
}
static CURRENT: LazyLock<Mutex<Published>> = LazyLock::new(|| {
    Mutex::new(Published {
        generation: 0,
        device: Weak::new(),
    })
});

static GENERATION_OBSERVER: OnceLock<fn(u64)> = OnceLock::new();

/// Registers an embedder's generation invalidation hook before platform startup.
/// The callback runs after publication, without renderer locks held.
pub fn install_renderer_d3d12_generation_observer(observer: fn(u64)) -> Result<()> {
    GENERATION_OBSERVER
        .set(observer)
        .map_err(|_| anyhow::anyhow!("D3D12 generation observer already installed"))
}

/// Owned lease on the current native renderer device. Raw COM pointers allow
/// an embedder using a different windows-rs version to borrow the same device.
/// Keep this lease alive while borrowing either pointer.
pub struct RendererD3D12Handle {
    /// ID3D12Device owned by `keep_alive`.
    pub device: *mut std::ffi::c_void,
    /// The renderer's ID3D12CommandQueue.
    pub queue: *mut std::ffi::c_void,
    /// Changes when the renderer recovers from device loss.
    pub generation: u64,
    /// Retains the native device and renderer queue.
    pub keep_alive: Arc<dyn std::any::Any + Send + Sync>,
}

/// Returns a retained renderer generation without initializing WebGPU.
pub fn renderer_d3d12_device() -> Option<RendererD3D12Handle> {
    let current = CURRENT.lock().unwrap_or_else(|error| error.into_inner());
    let device = current.device.upgrade()?;
    Some(RendererD3D12Handle {
        device: device.device.as_raw(),
        queue: device.queue.as_raw(),
        generation: current.generation,
        keep_alive: device,
    })
}

#[derive(Clone)]
pub(crate) struct DirectXDevices {
    pub(super) gpu: Arc<device::Device>,
    pub(crate) device: ID3D12Device,
}
impl DirectXDevices {
    pub(crate) fn new() -> Result<Self> {
        let gpu = device::Device::new()?;
        let mut current = CURRENT.lock().unwrap_or_else(|error| error.into_inner());
        current.generation += 1;
        current.device = Arc::downgrade(&gpu);
        let generation = current.generation;
        drop(current);
        if let Some(observer) = GENERATION_OBSERVER.get() {
            observer(generation);
        }
        Ok(Self {
            device: gpu.device.clone(),
            gpu,
        })
    }
}
