use super::{
    DirectXDevices,
    atlas::{Atlas, AtlasTexture},
    device::{CommandStats, Device, KeepAlive, TextureLocation, transition},
    pipelines::{
        HISTORY_FORMAT, Pipelines, ROOT_BATCH, ROOT_CLIP_SEGMENTS, ROOT_EFFECT_TEXTURES,
        ROOT_GLOBALS, ROOT_INSTANCES, ROOT_TEXT_BACKGROUNDS, ROOT_TEXTURES, TABLE_TEXTURES,
    },
};
use crate::directx_renderer::{FontInfo, shader_resources::ShaderModule};
use anyhow::{Context, Result, ensure};
use gpui::*;
use std::{
    any::Any,
    collections::{HashMap, HashSet},
    mem::size_of_val,
    ptr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use windows::{
    Win32::{
        Foundation::{CloseHandle, HANDLE, HWND, RECT, WAIT_OBJECT_0},
        Graphics::{
            Direct3D::*,
            Direct3D12::*,
            DirectComposition::*,
            Dxgi::{Common::*, *},
        },
        System::Threading::WaitForSingleObject,
    },
    core::Interface,
};

const FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM;
const SWAP_CHAIN_FLAGS: DXGI_SWAP_CHAIN_FLAG = DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT;
/// Presents that may queue ahead of the compositor. DXGI's default is three,
/// which lets input latency grow by two frames whenever the GPU falls behind.
const FRAME_LATENCY: u32 = 2;
/// A frame waits this long for queue room before presenting anyway.
const FRAME_LATENCY_TIMEOUT_MS: u32 = 100;
const BUFFERS: usize = 3;
const SHADER_READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0,
);
const MAX_UPLOAD_BYTES: usize = 256 * 1024 * 1024;
const MAX_UPLOAD_POOL_BYTES: u64 = 256 * 1024 * 1024;
const MIN_UPLOAD_CAPACITY: u64 = 256 * 1024;
/// Drawn frames a slot's upload buffer may stay four times too large.
const UPLOAD_SHRINK_FRAMES: u32 = 240;
const MAX_LAYER_DEPTH: usize = 8;
const DESCRIPTORS_PER_PAGE: u32 = 256;
const MAX_CACHED_DESCRIPTOR_PAGES: usize = 16;
const _: () = assert!(std::mem::size_of::<GlobalParams>() == 48);
const _: () = assert!(std::mem::offset_of!(GlobalParams, is_bgr) == 32);
const _: () = assert!(std::mem::size_of::<PathRasterizationSprite>() == 292);
const _: () = assert!(std::mem::offset_of!(PathRasterizationSprite, color) == 16);
const _: () = assert!(std::mem::size_of::<PathSprite>() == 16);
type Retained = Arc<Mutex<Vec<KeepAlive>>>;

/// One persistently mapped upload-heap buffer that a frame slot sub-allocates
/// linearly: one committed resource per slot instead of one per array, no
/// Map/Unmap per frame, and no reallocation when the scene's mix of primitive
/// kinds changes.
struct UploadBuffer {
    resource: ID3D12Resource,
    address: u64,
    mapped: *mut u8,
    capacity: u64,
}
// The mapping belongs to `resource`, which is only written by the renderer
// thread that owns the pool, and only after the slot's fence completed.
unsafe impl Send for UploadBuffer {}
impl UploadBuffer {
    fn new(gpu: &Device, capacity: u64) -> Result<Self> {
        let resource = gpu.create_buffer(
            capacity,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut mapped = ptr::null_mut();
        unsafe {
            resource.Map(
                0,
                Some(&D3D12_RANGE { Begin: 0, End: 0 }),
                Some(&mut mapped),
            )?;
        }
        ensure!(!mapped.is_null(), "D3D12 upload buffer mapped to null");
        Ok(Self {
            address: unsafe { resource.GetGPUVirtualAddress() },
            resource,
            mapped: mapped.cast(),
            capacity,
        })
    }
}

#[derive(Default)]
struct UploadPool {
    buffer: Option<UploadBuffer>,
    cursor: u64,
    /// Whether this frame's `Retained` already holds `buffer`.
    retained: bool,
    /// Consecutive frames that used under a quarter of `buffer`, and the most
    /// any of them used: the size a shrunk buffer is recreated at.
    sparse_frames: u32,
    sparse_peak: u64,
    capacity_hint: u64,
    // Descriptor writes share the frame slot's upload-buffer fence guard.
    descriptor_pages: Vec<ID3D12DescriptorHeap>,
}
impl UploadPool {
    fn begin(&mut self, completed: u64, reusable_after: u64) -> Result<()> {
        ensure!(
            completed >= reusable_after,
            "D3D12 uploads cannot be reused before fence {reusable_after} completes"
        );
        self.cursor = 0;
        self.retained = false;
        Ok(())
    }

    /// Copies `data` into the slot's buffer and returns its GPU address. An
    /// empty array gets 256 zero bytes, a valid binding for any root view.
    fn write(&mut self, gpu: &Device, retained: &Retained, data: &[u8]) -> Result<u64> {
        ensure!(
            data.len() <= MAX_UPLOAD_BYTES,
            "D3D12 scene upload exceeds {MAX_UPLOAD_BYTES} bytes"
        );
        // 256 bytes: constant-buffer placement, and more than any root view needs.
        let size = (data.len().max(1) as u64 + 255) & !255;
        let fits = self
            .buffer
            .as_ref()
            .is_some_and(|buffer| self.cursor + size <= buffer.capacity);
        if !fits {
            // Whatever this frame already wrote stays alive in `retained`
            // until its fence; later writes continue in a larger buffer.
            let capacity = (self.cursor + size)
                .max(self.buffer.as_ref().map_or(0, |buffer| buffer.capacity * 2))
                .max(self.capacity_hint)
                .max(MIN_UPLOAD_CAPACITY)
                .checked_next_power_of_two()
                .context("D3D12 upload capacity overflow")?;
            ensure!(
                capacity <= MAX_UPLOAD_POOL_BYTES,
                "D3D12 frame upload pool exceeds {MAX_UPLOAD_POOL_BYTES} committed bytes"
            );
            self.buffer = Some(UploadBuffer::new(gpu, capacity)?);
            self.cursor = 0;
            self.retained = false;
            self.sparse_frames = 0;
        }
        let buffer = self
            .buffer
            .as_ref()
            .context("D3D12 upload buffer missing")?;
        if !self.retained {
            retain(retained, buffer.resource.clone());
            self.retained = true;
        }
        let offset = self.cursor;
        unsafe {
            let destination = buffer.mapped.add(offset as usize);
            if data.is_empty() {
                ptr::write_bytes(destination, 0, 256);
            } else {
                ptr::copy_nonoverlapping(data.as_ptr(), destination, data.len());
            }
        }
        self.cursor += size;
        Ok(buffer.address + offset)
    }

    /// Releases a buffer that stayed four times larger than its frames need
    /// for a while; the next frame recreates it at twice the recent peak.
    fn finish(&mut self) {
        let Some(buffer) = &self.buffer else { return };
        if buffer.capacity <= MIN_UPLOAD_CAPACITY || self.cursor * 4 >= buffer.capacity {
            self.sparse_frames = 0;
            self.sparse_peak = 0;
            return;
        }
        self.sparse_peak = self.sparse_peak.max(self.cursor);
        self.sparse_frames += 1;
        if self.sparse_frames >= UPLOAD_SHRINK_FRAMES {
            self.capacity_hint = (self.sparse_peak * 2).max(MIN_UPLOAD_CAPACITY);
            self.buffer = None;
            self.sparse_frames = 0;
            self.sparse_peak = 0;
        }
    }
}

#[derive(Clone, Copy, Default)]
struct RenderWork {
    descriptor_pages_created: u64,
    descriptor_pages_reused: u64,
}
impl RenderWork {
    fn add(&mut self, other: Self) {
        self.descriptor_pages_created += other.descriptor_pages_created;
        self.descriptor_pages_reused += other.descriptor_pages_reused;
    }
}

#[derive(Clone, Copy, Default)]
struct CpuStage {
    total: Duration,
    maximum: Duration,
}
impl CpuStage {
    fn add(&mut self, duration: Duration) {
        self.total = self.total.saturating_add(duration);
        self.maximum = self.maximum.max(duration);
    }
}

struct RendererTimings {
    started: Instant,
    frames: u64,
    fence_waits: u64,
    skipped_gpu_busy: u64,
    skipped_present_busy: u64,
    stages: [CpuStage; 9],
    last_render: [Duration; 5],
    work: RenderWork,
    commands: CommandStats,
}
impl RendererTimings {
    fn from_environment() -> Option<Self> {
        std::env::var_os("SK_DX12_TIMINGS")
            .filter(|value| value.to_str() == Some("1"))
            .map(|_| Self {
                started: Instant::now(),
                frames: 0,
                fence_waits: 0,
                skipped_gpu_busy: 0,
                skipped_present_busy: 0,
                stages: [CpuStage::default(); 9],
                last_render: [Duration::ZERO; 5],
                work: RenderWork::default(),
                commands: CommandStats::default(),
            })
    }

    fn frame(
        &mut self,
        fence: Duration,
        render: Duration,
        present: Duration,
        total: Duration,
        waited: bool,
        present_busy: bool,
    ) {
        self.frames += 1;
        self.fence_waits += u64::from(waited);
        self.skipped_present_busy += u64::from(present_busy);
        for (stage, duration) in self.stages.iter_mut().zip([
            fence,
            render,
            present,
            total,
            self.last_render[0],
            self.last_render[1],
            self.last_render[2],
            self.last_render[3],
            self.last_render[4],
        ]) {
            stage.add(duration);
        }
        self.emit_if_due();
    }

    fn emit_if_due(&mut self) {
        let elapsed = self.started.elapsed();
        if elapsed < Duration::from_secs(2) {
            return;
        }
        let names = [
            "fence",
            "render",
            "present",
            "draw",
            "prepare",
            "upload",
            "atlas",
            "record",
            "submit_other",
        ];
        let mut stages = String::new();
        for (name, stage) in names.into_iter().zip(self.stages) {
            use std::fmt::Write as _;
            // Writing to a String has no I/O failure; retain diagnostics if the
            // formatting contract ever changes to a fallible destination.
            if let Err(error) = write!(
                stages,
                " {name}_avg_ms={:.3} {name}_max_ms={:.3}",
                stage.total.as_secs_f64() * 1000. / self.frames.max(1) as f64,
                stage.maximum.as_secs_f64() * 1000.
            ) {
                log::error!("Formatting native D3D12 timings failed: {error}");
                return;
            }
        }
        log::info!(
            "[DX12 timings] frames={} window_ms={:.1} fps={:.1} fence_waits={} skipped_gpu_busy={} skipped_present_busy={}{}",
            self.frames,
            elapsed.as_secs_f64() * 1000.,
            self.frames as f64 / elapsed.as_secs_f64(),
            self.fence_waits,
            self.skipped_gpu_busy,
            self.skipped_present_busy,
            stages,
        );
        log::info!(
            "[DX12 work] descriptor_pages_created={} descriptor_pages_reused={} command_pairs_created_total={} command_pairs_reused_total={}",
            self.work.descriptor_pages_created,
            self.work.descriptor_pages_reused,
            self.commands.created,
            self.commands.reused,
        );
        self.work = RenderWork::default();
        self.started = Instant::now();
        self.frames = 0;
        self.fence_waits = 0;
        self.skipped_gpu_busy = 0;
        self.skipped_present_busy = 0;
        self.stages = [CpuStage::default(); 9];
    }
}

#[repr(C)]
struct GlobalParams {
    gamma_ratios: [f32; 4],
    viewport_size: [f32; 2],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    is_bgr: u32,
    padding: [u32; 3],
}
#[repr(C)]
#[derive(Clone, Copy)]
struct PathRasterizationSprite {
    xy_position: Point<ScaledPixels>,
    st_position: Point<f32>,
    color: Background,
    bounds: Bounds<ScaledPixels>,
    content_mask: ContentMask<ScaledPixels>,
}
#[repr(C)]
#[derive(Clone, Copy)]
struct PathSprite {
    bounds: Bounds<ScaledPixels>,
}

struct Target {
    resource: ID3D12Resource,
    rtv_heap: ID3D12DescriptorHeap,
    format: DXGI_FORMAT,
    width: u32,
    height: u32,
    state: Mutex<D3D12_RESOURCE_STATES>,
}
impl Target {
    fn from_resource(
        device: &Device,
        resource: ID3D12Resource,
        width: u32,
        height: u32,
        samples: u32,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<Arc<Self>> {
        Self::from_resource_in(device, resource, FORMAT, width, height, samples, state)
    }
    fn from_resource_in(
        device: &Device,
        resource: ID3D12Resource,
        format: DXGI_FORMAT,
        width: u32,
        height: u32,
        samples: u32,
        state: D3D12_RESOURCE_STATES,
    ) -> Result<Arc<Self>> {
        let rtv_heap: ID3D12DescriptorHeap = unsafe {
            device
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                    NodeMask: 0,
                })?
        };
        let descriptor = D3D12_RENDER_TARGET_VIEW_DESC {
            Format: format,
            ViewDimension: if samples == 1 {
                D3D12_RTV_DIMENSION_TEXTURE2D
            } else {
                D3D12_RTV_DIMENSION_TEXTURE2DMS
            },
            Anonymous: D3D12_RENDER_TARGET_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_RTV {
                    MipSlice: 0,
                    PlaneSlice: 0,
                },
            },
        };
        unsafe {
            device.device.CreateRenderTargetView(
                &resource,
                Some(&descriptor),
                rtv_heap.GetCPUDescriptorHandleForHeapStart(),
            );
        }
        Ok(Arc::new(Self {
            resource,
            rtv_heap,
            format,
            width,
            height,
            state: Mutex::new(state),
        }))
    }
    fn new(device: &Device, width: u32, height: u32, samples: u32) -> Result<Arc<Self>> {
        Self::new_in(device, FORMAT, width, height, samples)
    }
    fn new_in(
        device: &Device,
        format: DXGI_FORMAT,
        width: u32,
        height: u32,
        samples: u32,
    ) -> Result<Arc<Self>> {
        let resource = device.create_texture_multisampled(
            width,
            height,
            format,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COMMON,
            samples,
        )?;
        Self::from_resource_in(
            device,
            resource,
            format,
            width,
            height,
            samples,
            D3D12_RESOURCE_STATE_COMMON,
        )
    }
    fn rtv(&self) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        unsafe { self.rtv_heap.GetCPUDescriptorHandleForHeapStart() }
    }
    fn to(
        self: &Arc<Self>,
        list: &ID3D12GraphicsCommandList,
        state: D3D12_RESOURCE_STATES,
        changes: &mut StateChanges,
    ) {
        let mut old = self.state.lock().unwrap_or_else(|error| error.into_inner());
        changes
            .original
            .entry(Arc::as_ptr(self) as usize)
            .or_insert_with(|| (self.clone(), *old));
        transition(list, &self.resource, *old, state);
        *old = state;
    }
    fn sampled(&self) -> AtlasTexture {
        AtlasTexture {
            resource: self.resource.clone(),
            format: self.format,
            width: self.width,
            height: self.height,
        }
    }
}
struct StateChanges {
    original: HashMap<usize, (Arc<Target>, D3D12_RESOURCE_STATES)>,
    committed: bool,
}
impl StateChanges {
    fn new() -> Self {
        Self {
            original: HashMap::new(),
            committed: false,
        }
    }
    fn commit(&mut self) {
        self.committed = true;
    }
}
impl Drop for StateChanges {
    fn drop(&mut self) {
        if !self.committed {
            for (target, state) in self.original.values() {
                *target
                    .state
                    .lock()
                    .unwrap_or_else(|error| error.into_inner()) = *state;
            }
        }
    }
}
/// The window's composition tree:
///
/// ```text
/// root
/// ├─ underlay   (vue-native: empty; a host adds visuals drawn UNDER the app)
/// └─ visual     (the swap chain, premultiplied alpha)
/// ```
///
/// vue-native: `underlay` is where an `<iframe>`'s webview is composited
/// (crates/vn-webview). The app paints transparent over the iframe's box, so
/// the webview shows through there and everything the app paints later in
/// paint order — a menu, a tooltip — stays above it.
struct Composition {
    device: IDCompositionDevice,
    _target: IDCompositionTarget,
    _root: IDCompositionVisual,
    underlay: IDCompositionVisual,
    _visual: IDCompositionVisual,
}
impl Composition {
    fn new(hwnd: HWND, swap_chain: &IDXGISwapChain3) -> Result<Self> {
        let device: IDCompositionDevice =
            unsafe { DCompositionCreateDevice(None::<&IDXGIDevice>)? };
        let target = unsafe { device.CreateTargetForHwnd(hwnd, false)? };
        let root = unsafe { device.CreateVisual()? };
        let underlay = unsafe { device.CreateVisual()? };
        let visual = unsafe { device.CreateVisual()? };
        unsafe {
            visual.SetContent(swap_chain)?;
            root.AddVisual(&underlay, false, None::<&IDCompositionVisual>)?;
            root.AddVisual(&visual, true, &underlay)?;
            target.SetRoot(&root)?;
            device.Commit()?;
        }
        Ok(Self {
            device,
            _target: target,
            _root: root,
            underlay,
            _visual: visual,
        })
    }
}

/// The swap chain's frame-latency semaphore: signalled once per retired
/// present, so each present must consume exactly one wait.
struct FrameLatency(HANDLE);
impl FrameLatency {
    fn new(swap_chain: &IDXGISwapChain3) -> Result<Self> {
        unsafe {
            swap_chain.SetMaximumFrameLatency(FRAME_LATENCY)?;
            let handle = swap_chain.GetFrameLatencyWaitableObject();
            ensure!(!handle.is_invalid(), "D3D12 swap chain has no latency waitable");
            Ok(Self(handle))
        }
    }
    fn wait(&self, timeout_ms: u32) -> bool {
        unsafe { WaitForSingleObject(self.0, timeout_ms) == WAIT_OBJECT_0 }
    }
}
impl Drop for FrameLatency {
    fn drop(&mut self) {
        if let Err(error) = unsafe { CloseHandle(self.0) } {
            log::error!("D3D12 frame latency handle close failed: {error}");
        }
    }
}

pub(crate) struct DirectXRenderer {
    hwnd: HWND,
    gpu: Arc<Device>,
    atlas: Arc<Atlas>,
    pipelines: Pipelines,
    swap_chain: Option<IDXGISwapChain3>,
    latency: Option<FrameLatency>,
    /// A latency wait succeeded and no present has consumed it yet.
    latency_ready: bool,
    disable_direct_composition: bool,
    composition: Option<Composition>,
    back_buffers: Vec<Arc<Target>>,
    fences: [u64; BUFFERS],
    width: u32,
    height: u32,
    skip_draws: bool,
    present_no_wait: bool,
    layers: Vec<Arc<Target>>,
    path_msaa: Option<Arc<Target>>,
    path_resolved: Option<Arc<Target>>,
    snapshot: Option<Arc<Target>>,
    blur: Vec<Arc<Target>>,
    idle_frames: ScratchIdle,
    timings: Option<RendererTimings>,
    uploads: [UploadPool; BUFFERS],
    uploads_failed: bool,
    /// vue-native: the `[SkHistory]` states.
    histories: VnEffectHistories<Arc<Target>>,
    /// The images of `--shading` effects, bound at t3..t6.
    effect_textures: EffectTextureCache<AtlasTexture>,
    /// vue-native: the snapshots captures keep (`Scene::vn_captures`), by id.
    /// They outlive a resize; a device loss drops them with the renderer.
    vn_snapshots: HashMap<u64, VnSnapshot>,
}

/// vue-native: a capture's pixels, kept across frames until no frame has
/// captured or painted it for a few renders.
struct VnSnapshot {
    target: Arc<Target>,
    idle: u32,
}

/// vue-native: one frame's effect quads as uploaded: the scene's own, then one
/// per `Scene::vn_effect_draws` entry, then the history steps. Empty when the
/// scene has no effect layer.
#[derive(Default)]
struct VnEffectFrame {
    quads: Vec<EffectQuad>,
    histories: Vec<Option<DrawHistory>>,
}

struct DrawHistory {
    read: Arc<Target>,
    /// The texture this frame's step writes and the step's quad.
    step: Option<(Arc<Target>, u32)>,
    fresh: bool,
}

impl DirectXRenderer {
    /// vue-native: the composition device and the container drawn under the
    /// app's surface, each as an owned COM pointer (the caller releases them).
    /// `None` without DirectComposition. A lost device rebuilds the tree, so
    /// the pair changes: a host compares the pointers and re-attaches.
    pub(crate) fn vn_composition_underlay(&self) -> Option<(usize, usize)> {
        let composition = self.composition.as_ref()?;
        Some((
            composition.device.clone().into_raw() as usize,
            composition.underlay.clone().into_raw() as usize,
        ))
    }

    pub(crate) fn new(
        hwnd: HWND,
        devices: &DirectXDevices,
        disable_direct_composition: bool,
    ) -> Result<Self> {
        let gpu = devices.gpu.clone();
        let descriptor = DXGI_SWAP_CHAIN_DESC1 {
            Width: 1,
            Height: 1,
            Format: FORMAT,
            Stereo: false.into(),
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
            BufferCount: BUFFERS as u32,
            Scaling: DXGI_SCALING_STRETCH,
            SwapEffect: DXGI_SWAP_EFFECT_FLIP_SEQUENTIAL,
            AlphaMode: if disable_direct_composition {
                DXGI_ALPHA_MODE_IGNORE
            } else {
                DXGI_ALPHA_MODE_PREMULTIPLIED
            },
            Flags: SWAP_CHAIN_FLAGS.0 as u32,
        };
        let swap_chain: IDXGISwapChain3 = unsafe {
            if disable_direct_composition {
                gpu.dxgi_factory
                    .CreateSwapChainForHwnd(&gpu.queue, hwnd, &descriptor, None, None)?
                    .cast()?
            } else {
                gpu.dxgi_factory
                    .CreateSwapChainForComposition(&gpu.queue, &descriptor, None)?
                    .cast()?
            }
        };
        unsafe {
            gpu.dxgi_factory
                .MakeWindowAssociation(hwnd, DXGI_MWA_NO_ALT_ENTER)?;
        }
        let latency = FrameLatency::new(&swap_chain)?;
        let composition = if disable_direct_composition {
            None
        } else {
            Some(Composition::new(hwnd, &swap_chain)?)
        };
        let pipelines = Pipelines::new(&gpu.device)?;
        pipelines.prewarm(FORMAT);
        let atlas = Arc::new(Atlas::new(gpu.clone()));
        let back_buffers = Self::buffers(&gpu, &swap_chain, 1, 1)?;
        Ok(Self {
            hwnd,
            gpu,
            atlas,
            pipelines,
            swap_chain: Some(swap_chain),
            latency: Some(latency),
            latency_ready: false,
            disable_direct_composition,
            composition,
            back_buffers,
            fences: [0; BUFFERS],
            width: 1,
            height: 1,
            skip_draws: false,
            present_no_wait: false,
            layers: Vec::new(),
            path_msaa: None,
            path_resolved: None,
            snapshot: None,
            blur: Vec::new(),
            idle_frames: ScratchIdle::default(),
            timings: RendererTimings::from_environment(),
            uploads: std::array::from_fn(|_| UploadPool::default()),
            uploads_failed: false,
            histories: VnEffectHistories::default(),
            effect_textures: EffectTextureCache::default(),
            vn_snapshots: HashMap::new(),
        })
    }
    fn buffers(
        gpu: &Device,
        swap_chain: &IDXGISwapChain3,
        width: u32,
        height: u32,
    ) -> Result<Vec<Arc<Target>>> {
        (0..BUFFERS)
            .map(|index| {
                let resource: ID3D12Resource = unsafe { swap_chain.GetBuffer(index as u32)? };
                Target::from_resource(
                    gpu,
                    resource,
                    width,
                    height,
                    1,
                    D3D12_RESOURCE_STATE_PRESENT,
                )
            })
            .collect()
    }
    pub(crate) fn sprite_atlas(&self) -> Arc<dyn PlatformAtlas> {
        self.atlas.clone()
    }
    pub(crate) fn set_present_no_wait(&mut self, enabled: bool) {
        self.present_no_wait = enabled;
    }
    pub(crate) fn mark_drawable(&mut self) {
        self.skip_draws = false;
    }
    pub(crate) fn get_font_info() -> &'static FontInfo {
        crate::directx_renderer::DirectXRenderer::get_font_info()
    }
    pub(crate) fn handle_device_lost(&mut self, devices: &DirectXDevices) -> Result<()> {
        self.skip_draws = true;
        if unsafe { self.gpu.device.GetDeviceRemovedReason() }.is_ok() {
            self.gpu.wait_idle()?;
        }
        // An HWND can own only one flip-model swap chain. Release the old
        // visual and chain before creating its replacement, including retries.
        self.composition.take();
        self.back_buffers.clear();
        self.latency.take();
        self.swap_chain.take();
        self.layers.clear();
        self.path_msaa = None;
        self.path_resolved = None;
        self.snapshot = None;
        self.blur.clear();
        self.effect_textures.clear();
        let mut replacement = crate::try_to_recover_from_device_lost(|| {
            let mut replacement = Self::new(self.hwnd, devices, self.disable_direct_composition)?;
            replacement.resize(size(
                DevicePixels(self.width as i32),
                DevicePixels(self.height as i32),
            ))?;
            Ok(replacement)
        })?;
        self.atlas.handle_device_lost(devices.gpu.clone());
        replacement.atlas = self.atlas.clone();
        replacement.skip_draws = true;
        replacement.present_no_wait = self.present_no_wait;
        *self = replacement;
        Ok(())
    }
    pub(crate) fn resize(&mut self, new_size: Size<DevicePixels>) -> Result<()> {
        let width = new_size.width.0.max(1) as u32;
        let height = new_size.height.0.max(1) as u32;
        if [width, height] == [self.width, self.height] {
            return Ok(());
        }
        self.gpu.wait_idle()?;
        self.back_buffers.clear();
        unsafe {
            self.swap_chain
                .as_ref()
                .context("D3D12 swap chain unavailable during recovery")?
                .ResizeBuffers(
                    BUFFERS as u32,
                    width,
                    height,
                    FORMAT,
                    SWAP_CHAIN_FLAGS,
                )?;
        }
        self.back_buffers = Self::buffers(
            &self.gpu,
            self.swap_chain
                .as_ref()
                .context("D3D12 swap chain missing")?,
            width,
            height,
        )?;
        self.width = width;
        self.height = height;
        // The GPU is idle, so every slot's uploads are complete. Their buffers
        // do not depend on the window size; keep them so that live resize does
        // not recreate committed upload buffers on every step.
        self.fences = [0; BUFFERS];
        self.layers.clear();
        self.path_msaa = None;
        self.path_resolved = None;
        self.snapshot = None;
        self.blur.clear();
        Ok(())
    }
    pub(crate) fn draw(
        &mut self,
        scene: &Scene,
        appearance: WindowBackgroundAppearance,
    ) -> Result<()> {
        if self.skip_draws || self.uploads_failed {
            return Ok(());
        }
        let _profile = crate::profiling::span(self.hwnd.0 as usize, "native", "D3D12 draw");
        let started = self.timings.as_ref().map(|_| Instant::now());
        if !self.latency_ready {
            if let Some(latency) = &self.latency {
                let _wait =
                    crate::profiling::span(self.hwnd.0 as usize, "wait", "D3D12 frame latency");
                let timeout = if self.present_no_wait {
                    0
                } else {
                    FRAME_LATENCY_TIMEOUT_MS
                };
                self.latency_ready = latency.wait(timeout);
                if !self.latency_ready && self.present_no_wait {
                    if let Some(timings) = self.timings.as_mut() {
                        timings.skipped_gpu_busy += 1;
                        timings.emit_if_due();
                    }
                    return Ok(());
                }
                // On timeout (an occluded or stalled compositor) present anyway,
                // as before; the queue then holds one frame more until it drains.
            }
        }
        let index = unsafe {
            self.swap_chain
                .as_ref()
                .context("D3D12 swap chain missing")?
                .GetCurrentBackBufferIndex()
        } as usize;
        let fence = *self
            .fences
            .get(index)
            .context("D3D12 swap-chain index is out of range")?;
        let waited = self.gpu.completed_value() < fence;
        let fence_started = started.map(|_| Instant::now());
        if waited {
            if self.present_no_wait {
                if let Some(timings) = self.timings.as_mut() {
                    timings.skipped_gpu_busy += 1;
                    timings.emit_if_due();
                }
                return Ok(());
            }
            let _wait =
                crate::profiling::span(self.hwnd.0 as usize, "wait", "D3D12 backbuffer fence");
            self.gpu.wait_for(fence)?;
        }
        let fence_time = fence_started
            .map(|started| started.elapsed())
            .unwrap_or_default();
        let target = self
            .back_buffers
            .get(index)
            .context("D3D12 back buffer missing")?
            .clone();
        let render_started = started.map(|_| Instant::now());
        let mut uploads = std::mem::take(
            self.uploads
                .get_mut(index)
                .context("D3D12 back buffer upload pool missing")?,
        );
        uploads.begin(self.gpu.completed_value(), fence)?;
        let rendered = self.render(scene, appearance, target, true, &mut uploads);
        uploads.finish();
        *self
            .uploads
            .get_mut(index)
            .context("D3D12 back buffer upload pool missing")? = uploads;
        let completed = match rendered {
            Ok(completed) => completed,
            Err(error) => {
                // Execute may have succeeded even if its completion Signal failed.
                // Never map that generation again using the preceding frame's fence.
                //
                // An error raised before Execute (an effect that does not
                // compile, layers nested too deep, a missing atlas page) left
                // nothing for the GPU to read: the slot's uploads are reusable
                // under its previous fence, and the next scene may well draw.
                if self.gpu.submission_failed() {
                    self.skip_draws = true;
                    self.uploads_failed = true;
                }
                return Err(error);
            }
        };
        let render_time = render_started
            .map(|started| started.elapsed())
            .unwrap_or_default();
        self.fences[index] = completed;
        let flags = if self.present_no_wait {
            DXGI_PRESENT_DO_NOT_WAIT
        } else {
            DXGI_PRESENT(0)
        };
        let present_started = started.map(|_| Instant::now());
        let result = {
            let _present = crate::profiling::span(self.hwnd.0 as usize, "wait", "D3D12 Present");
            unsafe {
                self.swap_chain
                    .as_ref()
                    .context("D3D12 swap chain missing")?
                    .Present(0, flags)
            }
        };
        let present_busy = self.present_no_wait && result == DXGI_ERROR_WAS_STILL_DRAWING;
        if !present_busy {
            self.latency_ready = false;
        }
        if let (Some(started), Some(present_started), Some(timings)) =
            (started, present_started, self.timings.as_mut())
        {
            timings.commands = self.gpu.command_stats();
            timings.frame(
                fence_time,
                render_time,
                present_started.elapsed(),
                started.elapsed(),
                waited,
                present_busy,
            );
        }
        if present_busy {
            return Ok(());
        }
        result.ok().context("Presenting native D3D12 swap chain")
    }
    fn render(
        &mut self,
        scene: &Scene,
        appearance: WindowBackgroundAppearance,
        target: Arc<Target>,
        present: bool,
        uploads: &mut UploadPool,
    ) -> Result<u64> {
        let profile_target = self.hwnd.0 as usize;
        let prepare_profile =
            crate::profiling::span(profile_target, "native", "D3D12 prepare pipelines");
        let prepare_started = self.timings.as_ref().map(|_| Instant::now());
        // Scratch targets are window-sized; recreating one is a committed
        // allocation plus page zeroing. Keep them through short gaps (a
        // chevron scrolling out, a tooltip closing) and release them once
        // a run of drawn frames no longer needs them.
        let mut depth = layer_depth(scene, 0, 0)?;
        for capture in &scene.vn_captures {
            depth = depth.max(layer_depth(scene, capture.group as usize, 1)?);
        }
        trim_scratch(&mut self.layers, depth, &mut self.idle_frames.layers);
        self.vn_age_snapshots(scene);
        let path_targets = if scene.paths.is_empty() { 0 } else { 1 };
        let mut held = usize::from(self.path_msaa.is_some());
        trim_scratch_count(&mut held, path_targets, &mut self.idle_frames.paths);
        if held == 0 {
            self.path_msaa = None;
            self.path_resolved = None;
        }
        let needs_backdrop = self.prepare_pipelines(scene)?;
        let backdrop_levels = if needs_backdrop {
            scene
                .backdrop_blur_rects
                .iter()
                .map(BackdropBlurRect::effective_kernel_levels)
                .chain(
                    scene
                        .effect_quads
                        .iter()
                        .chain(scene.vn_effect_draws.iter().map(|draw| &draw.quad))
                        .map(EffectQuad::effective_kernel_levels),
                )
                .max()
                .unwrap_or(0) as usize
        } else {
            0
        };
        let effects = self.prepare_vn_effects(scene)?;
        // The snapshot counts as one more level below the pyramid.
        let mut held = usize::from(self.snapshot.is_some()) + self.blur.len();
        let needed = if needs_backdrop { backdrop_levels + 1 } else { 0 };
        trim_scratch_count(&mut held, needed, &mut self.idle_frames.backdrop);
        if held == 0 {
            self.snapshot = None;
        }
        self.blur.truncate(held.saturating_sub(1));
        let prepare_time = prepare_started
            .map(|started| started.elapsed())
            .unwrap_or_default();
        drop(prepare_profile);
        let upload_profile = crate::profiling::span(profile_target, "native", "D3D12 upload scene");
        let upload_started = prepare_started.map(|_| Instant::now());
        let retained: Retained = Arc::new(Mutex::new(Vec::new()));
        let gpu = self.gpu.clone();
        let font = Self::get_font_info();
        let globals = upload(
            &gpu,
            &retained,
            uploads,
            &[GlobalParams {
                gamma_ratios: font.gamma_ratios,
                viewport_size: [self.width as f32, self.height as f32],
                grayscale_enhanced_contrast: font.grayscale_enhanced_contrast,
                subpixel_enhanced_contrast: font.subpixel_enhanced_contrast,
                is_bgr: font.is_bgr as u32,
                padding: [0; 3],
            }],
        )?;
        let effect_quads = if effects.quads.is_empty() {
            &scene.effect_quads
        } else {
            &effects.quads
        };
        let data = SceneBuffers::new(&gpu, &retained, uploads, scene, effect_quads)?;
        let upload_time = upload_started
            .map(|started| started.elapsed())
            .unwrap_or_default();
        drop(upload_profile);
        let atlas_profile =
            crate::profiling::span(profile_target, "native", "D3D12 snapshot atlas");
        let atlas_started = prepare_started.map(|_| Instant::now());
        let descriptors = Descriptors::new(gpu.clone(), retained.clone());
        self.effect_textures.begin_frame();
        let cache = &mut self.effect_textures;
        let effect_textures: Vec<[Option<AtlasTexture>; EFFECT_TEXTURE_SLOTS]> = scene
            .effect_texture_sets
            .iter()
            .map(|set| {
                std::array::from_fn(|slot| {
                    let texture = set[slot].as_ref()?;
                    cache
                        .get(texture, |width, height, texels| {
                            upload_effect_texture(&gpu, width, height, texels)
                        })
                        .cloned()
                })
            })
            .collect();
        // Atlas insertion takes the atlas lock before submitting uploads. Snapshot it
        // before submit takes the queue lock so rendering cannot reverse that order.
        let mut textures = HashMap::new();
        // Sprites arrive in runs that share a page; look up each run once.
        let mut previous = None;
        for texture_id in scene
            .monochrome_sprites
            .iter()
            .map(|sprite| sprite.tile.texture_id)
            .chain(
                scene
                    .subpixel_sprites
                    .iter()
                    .map(|sprite| sprite.tile.texture_id),
            )
            .chain(
                scene
                    .polychrome_sprites
                    .iter()
                    .map(|sprite| sprite.tile.texture_id),
            )
        {
            if previous.replace(texture_id) == Some(texture_id) {
                continue;
            }
            if let std::collections::hash_map::Entry::Vacant(entry) = textures.entry(texture_id) {
                let texture = self.atlas.get_texture(texture_id)?.with_context(|| {
                    format!("D3D12 scene references missing atlas texture {texture_id:?}")
                })?;
                entry.insert(texture);
            }
        }
        let atlas_time = atlas_started
            .map(|started| started.elapsed())
            .unwrap_or_default();
        drop(atlas_profile);
        let _submit_profile =
            crate::profiling::span(profile_target, "native", "D3D12 submit commands");
        let mut changes = StateChanges::new();
        let submit_started = prepare_started.map(|_| Instant::now());
        let mut record_time = Duration::ZERO;
        let mut work = RenderWork::default();
        let result = gpu.submit(vec![retained.clone() as KeepAlive], |list| {
            let _record_profile =
                crate::profiling::span(profile_target, "native", "D3D12 record commands");
            let record_started = submit_started.map(|_| Instant::now());
            let mut encoding = Encoding {
                gpu: &gpu,
                list,
                pipelines: &mut self.pipelines,
                textures: &textures,
                effect_textures: &effect_textures,
                changes: &mut changes,
                scene,
                data,
                globals,
                retained,
                descriptors,
                work: &mut work,
                uploads,
                target: target.clone(),
                width: self.width,
                height: self.height,
                layers: &mut self.layers,
                path_msaa: &mut self.path_msaa,
                path_resolved: &mut self.path_resolved,
                snapshot: &mut self.snapshot,
                blur: &mut self.blur,
                bound: BoundState::default(),
                clip: full_rect(self.width, self.height),
                reads_backdrop: needs_backdrop,
                effects: &effects,
                vn_snapshots: &mut self.vn_snapshots,
            };
            encoding.captures()?;
            encoding.bind_target(&target)?;
            let clear = if appearance == WindowBackgroundAppearance::Opaque {
                [1.0; 4]
            } else {
                [0.0; 4]
            };
            unsafe {
                list.ClearRenderTargetView(target.rtv(), &clear, None);
            }
            encoding.group(0, 0)?;
            if present {
                target.to(list, D3D12_RESOURCE_STATE_PRESENT, encoding.changes);
            }
            record_time = record_started
                .map(|started| started.elapsed())
                .unwrap_or_default();
            Ok(())
        });
        let submit_time = submit_started
            .map(|started| started.elapsed())
            .unwrap_or_default();
        self.histories.finish_frame();
        if result.is_ok() {
            changes.commit();
            if let Some(timings) = self.timings.as_mut() {
                timings.work.add(work);
                timings.last_render = [
                    prepare_time,
                    upload_time,
                    atlas_time,
                    record_time,
                    submit_time.saturating_sub(record_time),
                ];
            }
        }
        result
    }
    fn prepare_pipelines(&mut self, scene: &Scene) -> Result<bool> {
        ensure!(
            scene.surfaces.is_empty(),
            "Native PaintSurface carries no Windows pixel source; use atlas GPU images"
        );
        // Compile before acquiring the submission mutex: an uncached shader
        // must not block concurrent atlas uploads behind the queue lock.
        for (needed, module, samples) in [
            (!scene.shadows.is_empty(), ShaderModule::Shadow, 1),
            (!scene.quads.is_empty(), ShaderModule::Quad, 1),
            (!scene.underlines.is_empty(), ShaderModule::Underline, 1),
            (
                !scene.monochrome_sprites.is_empty(),
                ShaderModule::MonochromeSprite,
                1,
            ),
            (
                !scene.subpixel_sprites.is_empty(),
                ShaderModule::SubpixelSprite,
                1,
            ),
            (
                !scene.polychrome_sprites.is_empty(),
                ShaderModule::PolychromeSprite,
                1,
            ),
            (!scene.paths.is_empty(), ShaderModule::PathRasterization, 4),
            (!scene.paths.is_empty(), ShaderModule::PathSprite, 1),
            (!scene.vn_layers.is_empty(), ShaderModule::VnLayer, 1),
            (
                !scene.backdrop_blur_rects.is_empty(),
                ShaderModule::BlurRect,
                1,
            ),
        ] {
            if needed {
                self.pipelines.pipeline(module, FORMAT, samples)?;
            }
        }
        if scene
            .vn_layers
            .iter()
            .any(|layer| layer.flags & vn_layer_flags::PLUS_LIGHTER != 0)
        {
            self.pipelines.vn_plus_lighter(FORMAT)?;
        }
        let mut needs_backdrop = !scene.backdrop_blur_rects.is_empty();
        let mut effects = HashSet::new();
        let layer_quads = scene.vn_effect_draws.iter().map(|draw| &draw.quad);
        for effect in scene.effect_quads.iter().chain(layer_quads) {
            if effects.insert(effect.effect_id)
                && let Some((_, backdrop)) = self.pipelines.effect(effect.effect_id, FORMAT, 1)?
            {
                needs_backdrop |= backdrop;
            }
        }
        if !scene.vn_effect_draws.is_empty() {
            self.pipelines.effect(effect_id::CONTENT, FORMAT, 1)?;
        }
        if needs_backdrop {
            self.pipelines
                .pipeline(ShaderModule::BlurDownsample, FORMAT, 1)?;
            self.pipelines
                .pipeline(ShaderModule::BlurUpsample, FORMAT, 1)?;
        }
        Ok(needs_backdrop)
    }
    /// The frame's effect layer quads and history steps ([`VnEffectFrame`]),
    /// after [`Self::prepare_pipelines`].
    fn prepare_vn_effects(&mut self, scene: &Scene) -> Result<VnEffectFrame> {
        let mut frame = VnEffectFrame::default();
        if scene.vn_effect_draws.is_empty() {
            return Ok(frame);
        }
        let first_step = scene.effect_quads.len() + scene.vn_effect_draws.len();
        frame.quads.reserve(first_step);
        frame.quads.extend_from_slice(&scene.effect_quads);
        let mut steps = Vec::new();
        for draw in &scene.vn_effect_draws {
            let mut quad = draw.quad;
            let history = if quad.has_history() && self.pipelines.history(quad.effect_id)?.is_some()
            {
                let gpu = &self.gpu;
                self.histories.prepare(
                    draw.history,
                    quad.history_size(),
                    quad.time,
                    |width, height| {
                        Target::new_in(gpu, HISTORY_FORMAT, width, height, 1)
                            .inspect_err(|error| {
                                log::error!("Creating a D3D12 effect history failed: {error:#}")
                            })
                            .ok()
                    },
                )
            } else {
                None
            };
            if history.is_none() {
                quad.flags &= !EFFECT_FLAG_HISTORY;
            }
            frame.quads.push(quad);
            frame.histories.push(history.map(|history| DrawHistory {
                read: history.read,
                step: history.write.map(|write| {
                    steps.push(quad.history_pass());
                    (write, (first_step + steps.len() - 1) as u32)
                }),
                fresh: history.fresh,
            }));
        }
        frame.quads.extend(steps);
        Ok(frame)
    }
    /// vue-native: a snapshot no render captured or painted for a few renders
    /// is freed.
    fn vn_age_snapshots(&mut self, scene: &Scene) {
        const IDLE_RENDERS: u32 = 4;
        if self.vn_snapshots.is_empty() {
            return;
        }
        for layer in &scene.vn_layers {
            if let Some(snapshot) = layer
                .snapshot_id()
                .and_then(|id| self.vn_snapshots.get_mut(&id))
            {
                snapshot.idle = 0;
            }
        }
        for capture in &scene.vn_captures {
            if let Some(snapshot) = self.vn_snapshots.get_mut(&capture.info.id) {
                snapshot.idle = 0;
            }
        }
        for snapshot in self.vn_snapshots.values_mut() {
            snapshot.idle += 1;
        }
        self.vn_snapshots
            .retain(|_, snapshot| snapshot.idle <= IDLE_RENDERS);
    }
    pub(crate) fn render_to_image(
        &mut self,
        scene: &Scene,
        appearance: WindowBackgroundAppearance,
    ) -> Result<image::RgbaImage> {
        ensure!(
            !self.skip_draws && !self.uploads_failed,
            "D3D12 capture unavailable until its renderer device generation recovers"
        );
        let _profile =
            crate::profiling::span(self.hwnd.0 as usize, "native", "D3D12 frame capture");
        let target = Target::new(&self.gpu, self.width, self.height, 1)?;
        let mut uploads = UploadPool::default();
        uploads.begin(self.gpu.completed_value(), 0)?;
        self.render(scene, appearance, target.clone(), false, &mut uploads)?;
        uploads.finish();
        self.capture_target(target)
    }
    fn capture_target(&self, target: Arc<Target>) -> Result<image::RgbaImage> {
        let row_pitch = (u64::from(self.width) * 4 + 255) & !255;
        let length = row_pitch
            .checked_mul(u64::from(self.height))
            .context("D3D12 capture size overflow")?;
        let readback = self.gpu.create_buffer(
            length,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let resource = readback.clone();
        let mut changes = StateChanges::new();
        let fence = self.gpu.submit(
            vec![
                target.clone() as KeepAlive,
                Arc::new(readback.clone()) as KeepAlive,
            ],
            |list| {
                target.to(list, D3D12_RESOURCE_STATE_COPY_SOURCE, &mut changes);
                let source = TextureLocation::subresource(&target.resource, 0);
                let destination = TextureLocation::placed(
                    &resource,
                    D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: FORMAT,
                            Width: self.width,
                            Height: self.height,
                            Depth: 1,
                            RowPitch: row_pitch as u32,
                        },
                    },
                );
                unsafe {
                    list.CopyTextureRegion(destination.raw(), 0, 0, 0, source.raw(), None);
                }
                Ok(())
            },
        )?;
        changes.commit();
        self.gpu.wait_for(fence)?;
        let mut mapped = ptr::null_mut();
        unsafe {
            readback.Map(
                0,
                Some(&D3D12_RANGE {
                    Begin: 0,
                    End: length as usize,
                }),
                Some(&mut mapped),
            )?;
        }
        let row_bytes = self.width as usize * 4;
        let mut pixels = vec![0u8; row_bytes * self.height as usize];
        for row in 0..self.height as usize {
            unsafe {
                ptr::copy_nonoverlapping(
                    (mapped as *const u8).add(row * row_pitch as usize),
                    pixels.as_mut_ptr().add(row * row_bytes),
                    row_bytes,
                );
            }
        }
        unsafe {
            readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
        }
        for pixel in pixels.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
        image::RgbaImage::from_raw(self.width, self.height, pixels)
            .context("Invalid D3D12 capture dimensions")
    }
    pub(crate) fn gpu_specs(&self) -> Result<GpuSpecs> {
        let description = unsafe { self.gpu.adapter.GetDesc1()? };
        let driver = match description.VendorId {
            0x10de => "NVIDIA Corporation",
            0x1002 => "AMD Corporation",
            0x8086 => "Intel Corporation",
            _ => "Unknown Vendor",
        };
        let version = unsafe { self.gpu.adapter.CheckInterfaceSupport(&IDXGIDevice::IID) }
            .ok()
            .map(|version| {
                format!(
                    "{}.{}.{}.{}",
                    (version >> 48) & 65535,
                    (version >> 32) & 65535,
                    (version >> 16) & 65535,
                    version & 65535
                )
            })
            .unwrap_or_else(|| "Unknown Driver".into());
        Ok(GpuSpecs {
            is_software_emulated: description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
            device_name: String::from_utf16_lossy(&description.Description)
                .trim_end_matches('\0')
                .into(),
            driver_name: driver.into(),
            driver_info: version,
        })
    }
}

/// Drawn frames a full-window scratch target may sit unused before release.
const SCRATCH_RELEASE_FRAMES: u32 = 90;

#[derive(Default)]
struct ScratchIdle {
    layers: u32,
    paths: u32,
    backdrop: u32,
}

fn trim_scratch<T>(scratch: &mut Vec<T>, needed: usize, idle: &mut u32) {
    let mut held = scratch.len();
    trim_scratch_count(&mut held, needed, idle);
    scratch.truncate(held);
}
/// Lowers `held` to `needed` once more is held than needed for
/// [`SCRATCH_RELEASE_FRAMES`] consecutive drawn frames.
fn trim_scratch_count(held: &mut usize, needed: usize, idle: &mut u32) {
    if needed >= *held {
        *idle = 0;
        return;
    }
    *idle += 1;
    if *idle >= SCRATCH_RELEASE_FRAMES {
        *held = needed;
        *idle = 0;
    }
}

fn full_rect(width: u32, height: u32) -> RECT {
    RECT {
        left: 0,
        top: 0,
        right: width as i32,
        bottom: height as i32,
    }
}
fn intersect(a: RECT, b: RECT) -> RECT {
    RECT {
        left: a.left.max(b.left),
        top: a.top.max(b.top),
        right: a.right.min(b.right),
        bottom: a.bottom.min(b.bottom),
    }
}
fn is_empty(rect: &RECT) -> bool {
    rect.left >= rect.right || rect.top >= rect.bottom
}
/// Whole device pixels covering `bounds`, grown by `margin` on every side.
fn round_out(bounds: &Bounds<ScaledPixels>, margin: i32) -> RECT {
    let clamp = |value: f32| value.clamp(i32::MIN as f32 / 2., i32::MAX as f32 / 2.) as i32;
    RECT {
        left: clamp(bounds.origin.x.0.floor()).saturating_sub(margin),
        top: clamp(bounds.origin.y.0.floor()).saturating_sub(margin),
        right: clamp((bounds.origin.x.0 + bounds.size.width.0).ceil()).saturating_add(margin),
        bottom: clamp((bounds.origin.y.0 + bounds.size.height.0).ceil()).saturating_add(margin),
    }
}
/// `rect` in a pyramid level `2^level` times smaller, rounded out by a texel.
fn scale_down(rect: RECT, level: usize) -> RECT {
    let divisor = 1i32 << level;
    RECT {
        left: rect.left.div_euclid(divisor) - 1,
        top: rect.top.div_euclid(divisor) - 1,
        right: (rect.right + divisor - 1).div_euclid(divisor) + 1,
        bottom: (rect.bottom + divisor - 1).div_euclid(divisor) + 1,
    }
}
/// Window pixels a blur of `levels` pyramid levels can carry into a pixel.
/// Each down- and up-sample pass reads two source texels to either side
/// (one-texel offsets, bilinear), i.e. `2 * 2^level` window pixels at level
/// `level`; both chains together stay below `4 * 2^levels`, and the final
/// bilinear read of the half-resolution level adds a few pixels more.
fn blur_reach(levels: usize) -> i32 {
    (4 << levels) + 8
}

fn layer_depth(scene: &Scene, group: usize, depth: usize) -> Result<usize> {
    ensure!(
        depth <= MAX_LAYER_DEPTH,
        "D3D12 isolated layer nesting exceeds {MAX_LAYER_DEPTH}"
    );
    ensure!(
        group < scene.vn_group_count(),
        "D3D12 isolated layer group is out of range"
    );
    let mut required = depth;
    for batch in scene.vn_batches(group) {
        if let VnBatch::Layers(range) = batch {
            for index in range {
                let layer = scene
                    .vn_layers
                    .get(index)
                    .context("D3D12 layer index is out of range")?;
                // An effect layer that does not read its content draws no group.
                if layer.group == 0 || layer.snapshot_id().is_some() {
                    continue;
                }
                required = required.max(layer_depth(scene, layer.group as usize, depth + 1)?);
                if layer.fill_rule >> 1 != 0 {
                    required =
                        required.max(layer_depth(scene, layer.group as usize + 1, depth + 2)?);
                }
            }
        }
    }
    Ok(required)
}

fn retain<T: Any + Send + Sync>(retained: &Retained, value: T) {
    retained
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .push(Arc::new(value));
}
/// A GPU virtual address inside a frame slot's upload buffer.
type GpuAddress = u64;

fn upload<T>(
    gpu: &Device,
    retained: &Retained,
    uploads: &mut UploadPool,
    data: &[T],
) -> Result<GpuAddress> {
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), size_of_val(data)) };
    uploads.write(gpu, retained, bytes)
}
fn upload_or_dummy<T>(
    gpu: &Device,
    retained: &Retained,
    dummy: GpuAddress,
    uploads: &mut UploadPool,
    data: &[T],
) -> Result<GpuAddress> {
    if data.is_empty() {
        Ok(dummy)
    } else {
        upload(gpu, retained, uploads, data)
    }
}

struct SceneBuffers {
    dummy: GpuAddress,
    shadows: GpuAddress,
    quads: GpuAddress,
    blur_rects: GpuAddress,
    effects: GpuAddress,
    underlines: GpuAddress,
    mono: GpuAddress,
    subpixel: GpuAddress,
    poly: GpuAddress,
    layers: GpuAddress,
    clips: GpuAddress,
    text_backgrounds: GpuAddress,
}
impl SceneBuffers {
    fn new(
        gpu: &Device,
        retained: &Retained,
        uploads: &mut UploadPool,
        scene: &Scene,
        effect_quads: &[EffectQuad],
    ) -> Result<Self> {
        let dummy = upload(gpu, retained, uploads, &[0u32; 64])?;
        Ok(Self {
            dummy,
            shadows: upload_or_dummy(gpu, retained, dummy, uploads, &scene.shadows)?,
            quads: upload_or_dummy(gpu, retained, dummy, uploads, &scene.quads)?,
            blur_rects: upload_or_dummy(
                gpu,
                retained,
                dummy,
                uploads,
                &scene.backdrop_blur_rects,
            )?,
            effects: upload_or_dummy(gpu, retained, dummy, uploads, effect_quads)?,
            underlines: upload_or_dummy(gpu, retained, dummy, uploads, &scene.underlines)?,
            mono: upload_or_dummy(gpu, retained, dummy, uploads, &scene.monochrome_sprites)?,
            subpixel: upload_or_dummy(gpu, retained, dummy, uploads, &scene.subpixel_sprites)?,
            poly: upload_or_dummy(gpu, retained, dummy, uploads, &scene.polychrome_sprites)?,
            layers: upload_or_dummy(gpu, retained, dummy, uploads, &scene.vn_layers)?,
            clips: upload_or_dummy(gpu, retained, dummy, uploads, &scene.vn_clip_segments)?,
            text_backgrounds: upload_or_dummy(
                gpu,
                retained,
                dummy,
                uploads,
                &scene.vn_text_backgrounds,
            )?,
        })
    }
}
/// The most views one descriptor table holds: [`TABLE_TEXTURES`] or an
/// effect's images.
const MAX_TABLE: usize = if TABLE_TEXTURES > EFFECT_TEXTURE_SLOTS {
    TABLE_TEXTURES
} else {
    EFFECT_TEXTURE_SLOTS
};
struct Descriptors {
    gpu: Arc<Device>,
    retained: Retained,
    heap: Option<ID3D12DescriptorHeap>,
    used: u32,
    next_page: usize,
    stride: u32,
    // Tables already written to the current page, keyed by the raw resource
    // pointers (0 for a null view). A view only names a resource; its pixels
    // are read when the GPU executes, so one table serves every batch that
    // samples the same textures. Cleared with the page that holds them.
    tables: HashMap<(usize, [usize; MAX_TABLE]), D3D12_GPU_DESCRIPTOR_HANDLE>,
}
impl Descriptors {
    fn new(gpu: Arc<Device>, retained: Retained) -> Self {
        Self {
            stride: gpu.srv_descriptor_stride,
            gpu,
            retained,
            heap: None,
            used: DESCRIPTORS_PER_PAGE,
            next_page: 0,
            tables: HashMap::new(),
        }
    }
    /// Returns the table for these textures (t0, t2, t7, t8) and whether the
    /// bound heap changed, which invalidates the descriptor table bound before it.
    fn textures(
        &mut self,
        list: &ID3D12GraphicsCommandList,
        textures: [Option<&AtlasTexture>; TABLE_TEXTURES],
        uploads: &mut UploadPool,
        work: &mut RenderWork,
    ) -> Result<(D3D12_GPU_DESCRIPTOR_HANDLE, bool)> {
        self.table(list, &textures, uploads, work)
    }

    /// Like [`Self::textures`], for up to [`MAX_TABLE`] views.
    fn table(
        &mut self,
        list: &ID3D12GraphicsCommandList,
        textures: &[Option<&AtlasTexture>],
        uploads: &mut UploadPool,
        work: &mut RenderWork,
    ) -> Result<(D3D12_GPU_DESCRIPTOR_HANDLE, bool)> {
        let mut resources = [0; MAX_TABLE];
        for (resource, texture) in resources.iter_mut().zip(textures) {
            *resource = texture.map_or(0, |texture| texture.resource.as_raw() as usize);
        }
        let key = (textures.len(), resources);
        if let Some(table) = self.tables.get(&key) {
            return Ok((*table, false));
        }
        let count = textures.len() as u32;
        let heap_changed = self.reserve(list, count, uploads, work)?;
        self.write_table(key, textures, heap_changed)
    }

    /// Starts a new page unless `count` more descriptors fit in this one, and
    /// says whether it did.
    fn reserve(
        &mut self,
        list: &ID3D12GraphicsCommandList,
        count: u32,
        uploads: &mut UploadPool,
        work: &mut RenderWork,
    ) -> Result<bool> {
        let mut heap_changed = false;
        if self.used + count > DESCRIPTORS_PER_PAGE {
            let heap = if let Some(heap) = uploads.descriptor_pages.get(self.next_page) {
                work.descriptor_pages_reused += 1;
                heap.clone()
            } else {
                let heap: ID3D12DescriptorHeap = unsafe {
                    self.gpu
                        .device
                        .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                            Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                            NumDescriptors: DESCRIPTORS_PER_PAGE,
                            Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                            NodeMask: 0,
                        })?
                };
                if self.next_page < MAX_CACHED_DESCRIPTOR_PAGES {
                    uploads.descriptor_pages.push(heap.clone());
                }
                work.descriptor_pages_created += 1;
                heap
            };
            self.next_page += 1;
            retain(&self.retained, heap.clone());
            unsafe { list.SetDescriptorHeaps(&[Some(heap.clone())]) };
            self.heap = Some(heap);
            self.used = 0;
            self.tables.clear();
            heap_changed = true;
        }
        Ok(heap_changed)
    }

    fn write_table(
        &mut self,
        key: (usize, [usize; MAX_TABLE]),
        textures: &[Option<&AtlasTexture>],
        heap_changed: bool,
    ) -> Result<(D3D12_GPU_DESCRIPTOR_HANDLE, bool)> {
        let heap = self
            .heap
            .as_ref()
            .context("D3D12 descriptor page missing")?;
        let stride = self.stride;
        let cpu = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart() };
        for (offset, texture) in textures.iter().copied().enumerate() {
            let descriptor = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: texture.map_or(FORMAT, |texture| texture.format),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: 0,
                        MipLevels: 1,
                        PlaneSlice: 0,
                        ResourceMinLODClamp: 0.0,
                    },
                },
            };
            unsafe {
                self.gpu.device.CreateShaderResourceView(
                    texture.map(|texture| &texture.resource),
                    Some(&descriptor),
                    D3D12_CPU_DESCRIPTOR_HANDLE {
                        ptr: cpu.ptr + ((self.used as usize + offset) * stride as usize),
                    },
                );
            }
            if let Some(texture) = texture {
                retain(&self.retained, texture.resource.clone());
            }
        }
        let table = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu.ptr + u64::from(self.used * stride),
        };
        self.used += textures.len() as u32;
        self.tables.insert(key, table);
        Ok((table, heap_changed))
    }
}

/// Command-list state already recorded, so batches only emit what changes.
/// The root signature and its frame-constant arguments are bound once per
/// command list; a heap switch invalidates only the descriptor table.
#[derive(Default)]
struct BoundState {
    root: bool,
    pipeline: usize,
    instances: u64,
    batch_start: Option<u32>,
    table: u64,
    effect_table: u64,
    topology: Option<D3D_PRIMITIVE_TOPOLOGY>,
    // Pipelines kept alive for this frame's fence; a cache entry outlives the
    // frame only while the renderer does.
    retained_pipelines: Vec<usize>,
}

struct Encoding<'a> {
    gpu: &'a Arc<Device>,
    list: &'a ID3D12GraphicsCommandList,
    pipelines: &'a mut Pipelines,
    textures: &'a HashMap<AtlasTextureId, AtlasTexture>,
    /// `Scene::effect_texture_sets`, uploaded.
    effect_textures: &'a [[Option<AtlasTexture>; EFFECT_TEXTURE_SLOTS]],
    changes: &'a mut StateChanges,
    scene: &'a Scene,
    data: SceneBuffers,
    globals: GpuAddress,
    retained: Retained,
    descriptors: Descriptors,
    work: &'a mut RenderWork,
    uploads: &'a mut UploadPool,
    target: Arc<Target>,
    width: u32,
    height: u32,
    layers: &'a mut Vec<Arc<Target>>,
    path_msaa: &'a mut Option<Arc<Target>>,
    path_resolved: &'a mut Option<Arc<Target>>,
    snapshot: &'a mut Option<Arc<Target>>,
    blur: &'a mut Vec<Arc<Target>>,
    bound: BoundState,
    /// Device pixels of the current window-sized target that can still reach
    /// the screen: the whole window, narrowed by each enclosing layer.
    clip: RECT,
    reads_backdrop: bool,
    effects: &'a VnEffectFrame,
    vn_snapshots: &'a mut HashMap<u64, VnSnapshot>,
}
impl Encoding<'_> {
    /// Binds a window-sized target, scissored to the current clip.
    fn bind_target(&mut self, target: &Arc<Target>) -> Result<()> {
        let clip = self.clip;
        self.bind_target_rect(target, clip)
    }
    fn bind_target_rect(&mut self, target: &Arc<Target>, rect: RECT) -> Result<()> {
        target.to(self.list, D3D12_RESOURCE_STATE_RENDER_TARGET, self.changes);
        retain(&self.retained, target.clone());
        let rect = intersect(rect, full_rect(target.width, target.height));
        unsafe {
            self.list
                .OMSetRenderTargets(1, Some(&target.rtv()), false, None);
            self.list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: target.width as f32,
                Height: target.height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            self.list.RSSetScissorRects(&[rect]);
        }
        Ok(())
    }
    fn bind_root(&mut self) {
        if self.bound.root {
            return;
        }
        retain(&self.retained, self.pipelines.root_signature.clone());
        unsafe {
            self.list
                .SetGraphicsRootSignature(&self.pipelines.root_signature);
            self.list.SetGraphicsRootConstantBufferView(
                ROOT_GLOBALS,
                self.globals,
            );
            self.list.SetGraphicsRootShaderResourceView(
                ROOT_CLIP_SEGMENTS,
                self.data.clips,
            );
            self.list.SetGraphicsRootShaderResourceView(
                ROOT_TEXT_BACKGROUNDS,
                self.data.text_backgrounds,
            );
        }
        self.bound.root = true;
    }
    fn draw(
        &mut self,
        module: ShaderModule,
        buffer: GpuAddress,
        start: u32,
        count: u32,
        vertices: u32,
        first: Option<&AtlasTexture>,
        second: Option<&AtlasTexture>,
        samples: u32,
    ) -> Result<()> {
        let pipeline = self.pipelines.pipeline(module, FORMAT, samples)?;
        self.draw_pipeline(&pipeline, buffer, start, count, vertices, first, second)
    }
    fn draw_pipeline(
        &mut self,
        pipeline: &ID3D12PipelineState,
        buffer: GpuAddress,
        start: u32,
        count: u32,
        vertices: u32,
        first: Option<&AtlasTexture>,
        second: Option<&AtlasTexture>,
    ) -> Result<()> {
        self.draw_pipeline_textures(
            pipeline,
            buffer,
            start,
            count,
            vertices,
            [first, second, None, None],
        )
    }
    fn draw_pipeline_textures(
        &mut self,
        pipeline: &ID3D12PipelineState,
        buffer: GpuAddress,
        start: u32,
        count: u32,
        vertices: u32,
        textures: [Option<&AtlasTexture>; TABLE_TEXTURES],
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let (table, heap_changed) = self
            .descriptors
            .textures(self.list, textures, self.uploads, self.work)?;
        if heap_changed {
            self.bound.table = 0;
            self.bound.effect_table = 0;
        }
        self.bind_root();
        let pipeline_key = pipeline.as_raw() as usize;
        if self.bound.pipeline != pipeline_key {
            if !self.bound.retained_pipelines.contains(&pipeline_key) {
                retain(&self.retained, pipeline.clone());
                self.bound.retained_pipelines.push(pipeline_key);
            }
            unsafe { self.list.SetPipelineState(pipeline) };
            self.bound.pipeline = pipeline_key;
        }
        if self.bound.batch_start != Some(start) {
            let batch = [start, 0, 0, 0];
            unsafe {
                self.list
                    .SetGraphicsRoot32BitConstants(ROOT_BATCH, 4, batch.as_ptr().cast(), 0);
            }
            self.bound.batch_start = Some(start);
        }
        let instances = buffer;
        if self.bound.instances != instances {
            unsafe {
                self.list
                    .SetGraphicsRootShaderResourceView(ROOT_INSTANCES, instances);
            }
            self.bound.instances = instances;
        }
        if self.bound.table != table.ptr {
            unsafe {
                self.list
                    .SetGraphicsRootDescriptorTable(ROOT_TEXTURES, table);
            }
            self.bound.table = table.ptr;
        }
        let topology = if vertices == 4 {
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP
        } else {
            D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST
        };
        if self.bound.topology != Some(topology) {
            unsafe { self.list.IASetPrimitiveTopology(topology) };
            self.bound.topology = Some(topology);
        }
        unsafe { self.list.DrawInstanced(vertices, count, 0, 0) };
        Ok(())
    }
    fn group(&mut self, group: usize, depth: usize) -> Result<()> {
        let scene = self.scene;
        for batch in scene.vn_batches(group) {
            match batch {
                VnBatch::Layers(range) => {
                    for index in range {
                        self.layer(index, depth)?;
                    }
                }
                VnBatch::Primitives(batch) => self.batch(batch)?,
            }
        }
        Ok(())
    }
    fn layer(&mut self, index: usize, depth: usize) -> Result<()> {
        let layer = *self
            .scene
            .vn_layers
            .get(index)
            .context("D3D12 layer index is out of range")?;
        if let Some(effect) = self.scene.vn_effect_layer(index) {
            return self.effect_layer(layer.group as usize, effect, depth);
        }
        if let Some(id) = layer.snapshot_id() {
            let Some(snapshot) = self.vn_snapshots.get(&id).map(|s| s.target.clone()) else {
                return Ok(());
            };
            snapshot.to(self.list, SHADER_READ, self.changes);
            retain(&self.retained, snapshot.clone());
            return self.composite(index, layer.flags, &snapshot.sampled(), None);
        }
        ensure!(
            depth < MAX_LAYER_DEPTH,
            "D3D12 isolated layer nesting exceeds {MAX_LAYER_DEPTH}"
        );
        // The composite covers only `layer.bounds`, inside the enclosing clip;
        // isolated pixels anywhere else can never reach the screen. A backdrop
        // read inside the group samples around its own bounds, so scenes with
        // one keep clearing and drawing the whole layer as before.
        let parent_clip = self.clip;
        let clip = if self.reads_backdrop {
            parent_clip
        } else {
            intersect(round_out(&layer.bounds, 0), parent_clip)
        };
        if is_empty(&clip) {
            return Ok(());
        }
        let target = self.draw_group_offscreen(layer.group as usize, depth, clip)?;
        let mask = if layer.fill_rule >> 1 != 0 {
            Some(
                self.draw_group_offscreen(layer.group as usize + 1, depth + 1, clip)?
                    .sampled(),
            )
        } else {
            None
        };
        self.composite(index, layer.flags, &target.sampled(), mask.as_ref())
    }
    /// Composites layer `index` from `source` over the current target,
    /// through `mask` (at t2) when it has one.
    fn composite(
        &mut self,
        index: usize,
        flags: u32,
        source: &AtlasTexture,
        mask: Option<&AtlasTexture>,
    ) -> Result<()> {
        let pipeline = if flags & vn_layer_flags::PLUS_LIGHTER != 0 {
            self.pipelines.vn_plus_lighter(FORMAT)?
        } else {
            self.pipelines.pipeline(ShaderModule::VnLayer, FORMAT, 1)?
        };
        let buffer = self.data.layers;
        self.draw_pipeline(&pipeline, buffer, index as u32, 1, 4, Some(source), mask)
    }
    /// Draws each capture's group into a cleared layer target and copies its
    /// ink into the capture's snapshot, before anything samples one.
    fn captures(&mut self) -> Result<()> {
        let scene = self.scene;
        for capture in &scene.vn_captures {
            let ink = capture.ink;
            let rect = intersect(round_out(&ink, 0), full_rect(self.width, self.height));
            let id = capture.info.id;
            if ink.is_empty() || is_empty(&rect) {
                self.vn_snapshots.remove(&id);
                continue;
            }
            let clip = if self.reads_backdrop {
                full_rect(self.width, self.height)
            } else {
                rect
            };
            let target = self.draw_group_offscreen(capture.group as usize, 0, clip)?;
            let (width, height) = ((rect.right - rect.left) as u32, (rect.bottom - rect.top) as u32);
            let reuse = self
                .vn_snapshots
                .get(&id)
                .is_some_and(|s| s.target.width == width && s.target.height == height);
            if !reuse {
                let snapshot = Target::new(self.gpu, width, height, 1)?;
                self.vn_snapshots
                    .insert(id, VnSnapshot { target: snapshot, idle: 0 });
            }
            let snapshot = self
                .vn_snapshots
                .get_mut(&id)
                .context("D3D12 snapshot missing")?;
            snapshot.idle = 0;
            let snapshot = snapshot.target.clone();
            target.to(self.list, D3D12_RESOURCE_STATE_COPY_SOURCE, self.changes);
            snapshot.to(self.list, D3D12_RESOURCE_STATE_COPY_DEST, self.changes);
            retain(&self.retained, snapshot.clone());
            let source = TextureLocation::subresource(&target.resource, 0);
            let destination = TextureLocation::subresource(&snapshot.resource, 0);
            let region = D3D12_BOX {
                left: rect.left as u32,
                top: rect.top as u32,
                front: 0,
                right: rect.right as u32,
                bottom: rect.bottom as u32,
                back: 1,
            };
            unsafe {
                self.list
                    .CopyTextureRegion(destination.raw(), 0, 0, 0, source.raw(), Some(&region));
            }
            snapshot.to(self.list, SHADER_READ, self.changes);
        }
        Ok(())
    }
    /// Renders `group` (an element's subtree) into the window-sized target
    /// of `depth`, cleared to transparent within `clip` (window pixels) and
    /// drawn only there, and returns it ready to sample; what draws inside
    /// sees that target as its Backdrop Root. The current target is bound
    /// again afterwards.
    fn draw_group_offscreen(
        &mut self,
        group: usize,
        depth: usize,
        clip: RECT,
    ) -> Result<Arc<Target>> {
        ensure!(
            depth < MAX_LAYER_DEPTH,
            "D3D12 isolated layer nesting exceeds {MAX_LAYER_DEPTH}"
        );
        while self.layers.len() <= depth {
            self.layers
                .push(Target::new(self.gpu, self.width, self.height, 1)?);
        }
        let target = self
            .layers
            .get(depth)
            .context("D3D12 isolated layer missing")?
            .clone();
        let parent = self.target.clone();
        let parent_clip = self.clip;
        self.target = target.clone();
        self.clip = clip;
        self.bind_target(&target)?;
        unsafe {
            self.list
                .ClearRenderTargetView(target.rtv(), &[0.0; 4], Some(&[clip]));
        }
        let drawn = self.group(group, depth + 1);
        self.clip = parent_clip;
        self.target = parent.clone();
        drawn?;
        self.bind_target(&parent)?;
        target.to(self.list, SHADER_READ, self.changes);
        Ok(target)
    }
    /// An effect layer: the group (when its effects read it) offscreen, each
    /// history's step, then each effect over the current target with the
    /// group at t7 and its history at t8.
    fn effect_layer(&mut self, group: usize, effect: VnEffectLayer, depth: usize) -> Result<()> {
        let effects = self.effects;
        let content = match group {
            0 => None,
            // An effect may sample its content anywhere in the window.
            group => Some(
                self.draw_group_offscreen(group, depth, self.clip)?
                    .sampled(),
            ),
        };
        let parent = self.target.clone();
        let buffer = self.data.effects;
        let base = self.scene.effect_quads.len();
        for draw in effect.draws() {
            let quad = *effects
                .quads
                .get(base + draw)
                .context("D3D12 effect layer quad is out of range")?;
            let images = quad
                .texture_set()
                .and_then(|set| self.effect_textures.get(set))
                .cloned()
                .unwrap_or_default();
            let history = effects.histories.get(draw).and_then(Option::as_ref);
            let history_texture = history.map(|history| history.read.sampled());
            if let Some(history) = history {
                if history.fresh {
                    history
                        .read
                        .to(self.list, D3D12_RESOURCE_STATE_RENDER_TARGET, self.changes);
                    retain(&self.retained, history.read.clone());
                    unsafe {
                        self.list
                            .ClearRenderTargetView(history.read.rtv(), &[0.0; 4], None);
                    }
                }
                history.read.to(self.list, SHADER_READ, self.changes);
                if let Some((write, step)) = &history.step
                    && let Some(pipeline) = self.pipelines.history(quad.effect_id)?
                {
                    self.bind_target_rect(write, full_rect(write.width, write.height))?;
                    self.bind_effect_textures(&images)?;
                    self.draw_pipeline_textures(
                        &pipeline,
                        buffer,
                        *step,
                        1,
                        4,
                        [None, None, content.as_ref(), history_texture.as_ref()],
                    )?;
                    write.to(self.list, SHADER_READ, self.changes);
                    self.bind_target(&parent)?;
                }
            }
            // An effect whose pipeline did not build leaves its share to the
            // content as it is.
            let found = match self.pipelines.effect(quad.effect_id, FORMAT, 1)? {
                Some(found) => Some(found),
                None if quad.needs_content() => {
                    self.pipelines.effect(effect_id::CONTENT, FORMAT, 1)?
                }
                None => None,
            };
            let Some((pipeline, backdrop)) = found else {
                continue;
            };
            let (blurred, original) = if backdrop {
                let levels = quad.effective_kernel_levels() as usize;
                let (blurred, original) = self.backdrop(levels, None)?;
                (Some(blurred), Some(original))
            } else {
                (None, None)
            };
            self.bind_effect_textures(&images)?;
            self.draw_pipeline_textures(
                &pipeline,
                buffer,
                (base + draw) as u32,
                1,
                4,
                [
                    blurred.as_ref(),
                    original.as_ref(),
                    content.as_ref(),
                    history_texture.as_ref(),
                ],
            )?;
        }
        Ok(())
    }
    fn batch(&mut self, batch: PrimitiveBatch) -> Result<()> {
        match batch {
            PrimitiveBatch::Quads(range) => {
                let buffer = self.data.quads;
                self.draw(
                    ShaderModule::Quad,
                    buffer,
                    range.start as u32,
                    range.len() as u32,
                    4,
                    None,
                    None,
                    1,
                )
            }
            PrimitiveBatch::Shadows(range) => {
                let buffer = self.data.shadows;
                self.draw(
                    ShaderModule::Shadow,
                    buffer,
                    range.start as u32,
                    range.len() as u32,
                    4,
                    None,
                    None,
                    1,
                )
            }
            PrimitiveBatch::Underlines(range) => {
                let buffer = self.data.underlines;
                self.draw(
                    ShaderModule::Underline,
                    buffer,
                    range.start as u32,
                    range.len() as u32,
                    4,
                    None,
                    None,
                    1,
                )
            }
            PrimitiveBatch::MonochromeSprites { texture_id, range } => {
                let texture = self
                    .textures
                    .get(&texture_id)
                    .context("D3D12 atlas snapshot is missing a scene texture")?
                    .clone();
                let buffer = self.data.mono;
                self.draw(
                    ShaderModule::MonochromeSprite,
                    buffer,
                    range.start as u32,
                    range.len() as u32,
                    4,
                    Some(&texture),
                    None,
                    1,
                )
            }
            PrimitiveBatch::SubpixelSprites { texture_id, range } => {
                let texture = self
                    .textures
                    .get(&texture_id)
                    .context("D3D12 atlas snapshot is missing a scene texture")?
                    .clone();
                let buffer = self.data.subpixel;
                self.draw(
                    ShaderModule::SubpixelSprite,
                    buffer,
                    range.start as u32,
                    range.len() as u32,
                    4,
                    Some(&texture),
                    None,
                    1,
                )
            }
            PrimitiveBatch::PolychromeSprites { texture_id, range } => {
                let texture = self
                    .textures
                    .get(&texture_id)
                    .context("D3D12 atlas snapshot is missing a scene texture")?
                    .clone();
                let buffer = self.data.poly;
                self.draw(
                    ShaderModule::PolychromeSprite,
                    buffer,
                    range.start as u32,
                    range.len() as u32,
                    4,
                    Some(&texture),
                    None,
                    1,
                )
            }
            PrimitiveBatch::Paths(range) => {
                let scene = self.scene;
                self.paths(&scene.paths[range])
            }
            PrimitiveBatch::BackdropBlurRects(range) => self.backdrops(range.start, range.end),
            PrimitiveBatch::EffectQuads { effect_id, range } => {
                self.effects(effect_id, range.start, range.end)
            }
            PrimitiveBatch::Surfaces(range) => {
                ensure!(
                    range.is_empty(),
                    "Native PaintSurface carries no Windows pixel source; use atlas GPU images"
                );
                Ok(())
            }
        }
    }
    fn paths(&mut self, paths: &[Path<ScaledPixels>]) -> Result<()> {
        let Some(first_path) = paths.first() else {
            return Ok(());
        };
        let mut union = first_path.clipped_bounds();
        for path in paths.iter().skip(1) {
            union = union.union(&path.clipped_bounds());
        }
        // Path sprites sample the resolved texture only inside their clipped
        // bounds, texel for pixel. Rasterize, clear and resolve just that
        // region (with a pixel of slack) instead of the whole window.
        let region = intersect(round_out(&union, 1), self.clip);
        if is_empty(&region) {
            return Ok(());
        }
        if self.path_msaa.is_none() {
            *self.path_msaa = Some(Target::new(self.gpu, self.width, self.height, 4)?);
        }
        if self.path_resolved.is_none() {
            *self.path_resolved = Some(Target::new(self.gpu, self.width, self.height, 1)?);
        }
        let msaa = self
            .path_msaa
            .as_ref()
            .context("D3D12 path MSAA target missing")?
            .clone();
        let resolved = self
            .path_resolved
            .as_ref()
            .context("D3D12 path resolve target missing")?
            .clone();
        let mut vertices = Vec::new();
        for path in paths {
            vertices.extend(path.vertices.iter().map(|vertex| PathRasterizationSprite {
                xy_position: vertex.xy_position,
                st_position: vertex.st_position,
                color: path.color,
                bounds: path.clipped_bounds(),
                content_mask: path.content_mask,
            }));
        }
        let buffer = upload(self.gpu, &self.retained, self.uploads, &vertices)?;
        self.bind_target_rect(&msaa, region)?;
        unsafe {
            self.list
                .ClearRenderTargetView(msaa.rtv(), &[0.0; 4], Some(&[region]));
        }
        self.draw(
            ShaderModule::PathRasterization,
            buffer,
            0,
            1,
            vertices.len() as u32,
            None,
            None,
            4,
        )?;
        msaa.to(self.list, D3D12_RESOURCE_STATE_RESOLVE_SOURCE, self.changes);
        resolved.to(self.list, D3D12_RESOURCE_STATE_RESOLVE_DEST, self.changes);
        retain(&self.retained, resolved.clone());
        match self.list.cast::<ID3D12GraphicsCommandList1>() {
            Ok(list) => unsafe {
                list.ResolveSubresourceRegion(
                    &resolved.resource,
                    0,
                    region.left as u32,
                    region.top as u32,
                    &msaa.resource,
                    0,
                    Some(&region),
                    FORMAT,
                    D3D12_RESOLVE_MODE_AVERAGE,
                );
            },
            Err(_) => unsafe {
                self.list
                    .ResolveSubresource(&resolved.resource, 0, &msaa.resource, 0, FORMAT);
            },
        }
        resolved.to(self.list, SHADER_READ, self.changes);
        let sprites = if paths
            .last()
            .is_some_and(|last| last.order == first_path.order)
        {
            paths
                .iter()
                .map(|path| PathSprite {
                    bounds: path.clipped_bounds(),
                })
                .collect::<Vec<_>>()
        } else {
            vec![PathSprite { bounds: union }]
        };
        let buffer = upload(self.gpu, &self.retained, self.uploads, &sprites)?;
        let target = self.target.clone();
        self.bind_target(&target)?;
        self.draw(
            ShaderModule::PathSprite,
            buffer,
            0,
            sprites.len() as u32,
            4,
            Some(&resolved.sampled()),
            None,
            1,
        )
    }
    fn ensure_blur(&mut self, levels: usize) -> Result<()> {
        if self.snapshot.is_none() {
            *self.snapshot = Some(Target::new(self.gpu, self.width, self.height, 1)?);
        }
        while self.blur.len() < levels {
            let divisor = 2u32.saturating_pow((self.blur.len() + 1) as u32);
            self.blur.push(Target::new(
                self.gpu,
                (self.width / divisor).max(1),
                (self.height / divisor).max(1),
                1,
            )?);
        }
        Ok(())
    }
    /// Snapshots the current target and blurs it. `region` limits the work to
    /// the window pixels the consumer reads, already including the kernel's
    /// reach; `None` keeps the whole window for effects that sample freely.
    fn backdrop(
        &mut self,
        levels: usize,
        region: Option<RECT>,
    ) -> Result<(AtlasTexture, AtlasTexture)> {
        let levels = levels.min(MAX_BACKDROP_BLUR_KERNEL_LEVELS as usize);
        self.ensure_blur(levels)?;
        let snapshot = self
            .snapshot
            .as_ref()
            .context("D3D12 backdrop snapshot missing")?
            .clone();
        let full = full_rect(self.width, self.height);
        let region = region.map_or(full, |region| intersect(region, full));
        self.target
            .to(self.list, D3D12_RESOURCE_STATE_COPY_SOURCE, self.changes);
        snapshot.to(self.list, D3D12_RESOURCE_STATE_COPY_DEST, self.changes);
        retain(&self.retained, snapshot.clone());
        if region == full {
            unsafe {
                self.list
                    .CopyResource(&snapshot.resource, &self.target.resource);
            }
        } else {
            let source = TextureLocation::subresource(&self.target.resource, 0);
            let destination = TextureLocation::subresource(&snapshot.resource, 0);
            let region_box = D3D12_BOX {
                left: region.left as u32,
                top: region.top as u32,
                front: 0,
                right: region.right as u32,
                bottom: region.bottom as u32,
                back: 1,
            };
            unsafe {
                self.list.CopyTextureRegion(
                    destination.raw(),
                    region.left as u32,
                    region.top as u32,
                    0,
                    source.raw(),
                    Some(&region_box),
                );
            }
        }
        snapshot.to(self.list, SHADER_READ, self.changes);
        let target = self.target.clone();
        self.bind_target(&target)?;
        let levels = levels.min(self.blur.len());
        if levels == 0 {
            return Ok((snapshot.sampled(), snapshot.sampled()));
        }
        let mut source = snapshot.sampled();
        for level in 0..levels {
            let target = self
                .blur
                .get(level)
                .context("D3D12 blur pyramid level missing")?
                .clone();
            let rect = scale_down(region, level + 1);
            self.blur_pass(ShaderModule::BlurDownsample, &source, &target, rect)?;
            source = target.sampled();
        }
        for level in (1..levels).rev() {
            let target = self
                .blur
                .get(level - 1)
                .context("D3D12 blur pyramid target missing")?
                .clone();
            let rect = scale_down(region, level);
            self.blur_pass(ShaderModule::BlurUpsample, &source, &target, rect)?;
            source = target.sampled();
        }
        let target = self.target.clone();
        self.bind_target(&target)?;
        Ok((source, snapshot.sampled()))
    }
    fn blur_pass(
        &mut self,
        module: ShaderModule,
        source: &AtlasTexture,
        target: &Arc<Target>,
        rect: RECT,
    ) -> Result<()> {
        self.bind_target_rect(target, rect)?;
        let buffer = self.data.dummy;
        self.draw(module, buffer, 0, 1, 4, Some(source), None, 1)?;
        target.to(self.list, SHADER_READ, self.changes);
        Ok(())
    }
    fn backdrops(&mut self, start: usize, end: usize) -> Result<()> {
        let mut first = start;
        while first < end {
            let levels = self.scene.backdrop_blur_rects[first].effective_kernel_levels() as usize;
            let mut next = first + 1;
            while next < end {
                let candidate = &self.scene.backdrop_blur_rects[next];
                if candidate.effective_kernel_levels() as usize != levels
                    || self.scene.backdrop_blur_rects[first..next]
                        .iter()
                        .any(|previous| previous.bounds.intersects(&candidate.bounds))
                {
                    break;
                }
                next += 1;
            }
            let mut union = self.scene.backdrop_blur_rects[first].bounds;
            for rect in &self.scene.backdrop_blur_rects[first + 1..next] {
                union = union.union(&rect.bounds);
            }
            let region = intersect(round_out(&union, blur_reach(levels)), self.clip);
            if is_empty(&intersect(round_out(&union, 0), self.clip)) {
                first = next;
                continue;
            }
            #[cfg(test)]
            let region = (!tests::FULL_WINDOW_BACKDROP.with(|full| full.get())).then_some(region);
            #[cfg(not(test))]
            let region = Some(region);
            let (blurred, original) = self.backdrop(levels, region)?;
            let buffer = self.data.blur_rects;
            self.draw(
                ShaderModule::BlurRect,
                buffer,
                first as u32,
                (next - first) as u32,
                4,
                Some(&blurred),
                Some(&original),
                1,
            )?;
            first = next;
        }
        Ok(())
    }
    fn effects(&mut self, effect: u32, start: usize, end: usize) -> Result<()> {
        let Some((pipeline, needs_backdrop)) = self.pipelines.effect(effect, FORMAT, 1)? else {
            return Ok(());
        };
        let buffer = self.data.effects;
        let textures = self.scene.effect_quads[start]
            .texture_set()
            .and_then(|set| self.effect_textures.get(set))
            .cloned()
            .unwrap_or_default();
        if !needs_backdrop {
            self.bind_effect_textures(&textures)?;
            return self.draw_pipeline(
                &pipeline,
                buffer,
                start as u32,
                (end - start) as u32,
                4,
                None,
                None,
            );
        }
        let mut first = start;
        while first < end {
            let levels = self.scene.effect_quads[first].effective_kernel_levels() as usize;
            let mut next = first + 1;
            while next < end {
                let candidate = &self.scene.effect_quads[next];
                if candidate.effective_kernel_levels() as usize != levels
                    || self.scene.effect_quads[first..next]
                        .iter()
                        .any(|previous| previous.bounds.intersects(&candidate.bounds))
                {
                    break;
                }
                next += 1;
            }
            let (blurred, original) = self.backdrop(levels, None)?;
            self.bind_effect_textures(&textures)?;
            self.draw_pipeline(
                &pipeline,
                buffer,
                first as u32,
                (next - first) as u32,
                4,
                Some(&blurred),
                Some(&original),
            )?;
            first = next;
        }
        Ok(())
    }

    /// Binds t3..t6 for the effect draw that follows. Every effect draw binds
    /// a table, of null views when its quads sample no image, and room is
    /// kept for the draw's own table so that a new page cannot orphan this one.
    fn bind_effect_textures(
        &mut self,
        textures: &[Option<AtlasTexture>; EFFECT_TEXTURE_SLOTS],
    ) -> Result<()> {
        let room = (TABLE_TEXTURES + EFFECT_TEXTURE_SLOTS) as u32;
        if self
            .descriptors
            .reserve(self.list, room, self.uploads, self.work)?
        {
            self.bound.table = 0;
            self.bound.effect_table = 0;
        }
        let views: [Option<&AtlasTexture>; EFFECT_TEXTURE_SLOTS] =
            std::array::from_fn(|slot| textures[slot].as_ref());
        let (table, _) = self
            .descriptors
            .table(self.list, &views, self.uploads, self.work)?;
        self.bind_root();
        if self.bound.effect_table != table.ptr {
            unsafe {
                self.list
                    .SetGraphicsRootDescriptorTable(ROOT_EFFECT_TEXTURES, table);
            }
            self.bound.effect_table = table.ptr;
        }
        Ok(())
    }
}

fn upload_effect_texture(
    gpu: &Device,
    width: u32,
    height: u32,
    texels: &[u8],
) -> Result<AtlasTexture> {
    let resource =
        gpu.create_texture(width, height, FORMAT, D3D12_RESOURCE_FLAG_NONE, SHADER_READ)?;
    gpu.upload_texture(&resource, (0, 0), (width, height), 4, texels, SHADER_READ)?;
    Ok(AtlasTexture {
        resource,
        format: FORMAT,
        width,
        height,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    thread_local! {
        /// Forces whole-window backdrops to produce a region-test reference.
        pub(super) static FULL_WINDOW_BACKDROP: std::cell::Cell<bool> =
            const { std::cell::Cell::new(false) };
    }

    #[::core::prelude::v1::test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly on Windows"]
    fn native_descriptor_pages_reuse_after_the_frame_slot_fence() -> Result<()> {
        struct ReleaseGate(ID3D12Fence);
        impl Drop for ReleaseGate {
            fn drop(&mut self) {
                if let Err(error) = unsafe { self.0.Signal(1) } {
                    log::error!("Releasing descriptor test gate: {error}");
                }
            }
        }
        let gpu = Device::new()?;
        let gate: ID3D12Fence = unsafe { gpu.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)? };
        let _release_gate = ReleaseGate(gate.clone());
        let mut uploads = UploadPool::default();
        uploads.begin(gpu.completed_value(), 0)?;
        // Distinct textures need distinct tables; one more table than fits a page.
        let textures = (0..(DESCRIPTORS_PER_PAGE / TABLE_TEXTURES as u32 + 1))
            .map(|_| {
                Ok(AtlasTexture {
                    resource: gpu.create_texture_multisampled(
                        1,
                        1,
                        FORMAT,
                        D3D12_RESOURCE_FLAG_NONE,
                        D3D12_RESOURCE_STATE_COMMON,
                        1,
                    )?,
                    format: FORMAT,
                    width: 1,
                    height: 1,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let retained: Retained = Arc::new(Mutex::new(Vec::new()));
        let mut first = RenderWork::default();
        let fence = gpu.submit_after(
            vec![retained.clone() as KeepAlive],
            Some((&gate, 1)),
            |list| {
                let mut descriptors = Descriptors::new(gpu.clone(), retained.clone());
                let (repeated, _) =
                    descriptors.textures(list, [None; TABLE_TEXTURES], &mut uploads, &mut first)?;
                let (again, heap_changed) =
                    descriptors.textures(list, [None; TABLE_TEXTURES], &mut uploads, &mut first)?;
                assert_eq!(
                    repeated.ptr, again.ptr,
                    "repeated textures must reuse their table"
                );
                assert!(!heap_changed);
                for texture in &textures {
                    descriptors.textures(
                        list,
                        [Some(texture), None, None, None],
                        &mut uploads,
                        &mut first,
                    )?;
                }
                Ok(())
            },
        )?;
        assert_eq!(first.descriptor_pages_created, 2);
        assert_eq!(first.descriptor_pages_reused, 0);
        assert!(uploads.begin(gpu.completed_value(), fence).is_err());
        let page = uploads
            .descriptor_pages
            .first()
            .context("descriptor page missing")?
            .as_raw();
        unsafe { gate.Signal(1)? };
        gpu.wait_for(fence)?;
        uploads.begin(gpu.completed_value(), fence)?;
        let retained: Retained = Arc::new(Mutex::new(Vec::new()));
        let mut second = RenderWork::default();
        let fence = gpu.submit(vec![retained.clone() as KeepAlive], |list| {
            let mut descriptors = Descriptors::new(gpu.clone(), retained.clone());
            descriptors.textures(list, [None; TABLE_TEXTURES], &mut uploads, &mut second)?;
            for texture in &textures {
                descriptors.textures(
                    list,
                    [Some(texture), None, None, None],
                    &mut uploads,
                    &mut second,
                )?;
            }
            Ok(())
        })?;
        gpu.wait_for(fence)?;
        assert_eq!(second.descriptor_pages_created, 0);
        assert_eq!(second.descriptor_pages_reused, 2);
        assert_eq!(
            uploads
                .descriptor_pages
                .first()
                .context("descriptor page missing")?
                .as_raw(),
            page
        );
        Ok(())
    }

    fn offscreen(width: u32, height: u32) -> Result<DirectXRenderer> {
        let gpu = Device::new()?;
        Ok(DirectXRenderer {
            hwnd: HWND(ptr::null_mut()),
            atlas: Arc::new(Atlas::new(gpu.clone())),
            pipelines: Pipelines::new(&gpu.device)?,
            gpu,
            swap_chain: None,
            latency: None,
            latency_ready: false,
            disable_direct_composition: true,
            composition: None,
            back_buffers: Vec::new(),
            fences: [0; BUFFERS],
            width,
            height,
            skip_draws: false,
            present_no_wait: false,
            layers: Vec::new(),
            path_msaa: None,
            path_resolved: None,
            snapshot: None,
            blur: Vec::new(),
            idle_frames: ScratchIdle::default(),
            timings: None,
            uploads: std::array::from_fn(|_| UploadPool::default()),
            uploads_failed: false,
            histories: VnEffectHistories::default(),
            effect_textures: EffectTextureCache::default(),
            vn_snapshots: HashMap::new(),
        })
    }

    fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(width), ScaledPixels(height)),
        )
    }

    fn mask(width: f32, height: f32) -> ContentMask<ScaledPixels> {
        ContentMask {
            bounds: bounds(0., 0., width, height),
            ..Default::default()
        }
    }

    fn quad(
        rect: Bounds<ScaledPixels>,
        color: Rgba,
        content_mask: ContentMask<ScaledPixels>,
    ) -> Quad {
        Quad {
            bounds: rect,
            background: color.into(),
            content_mask,
            ..Default::default()
        }
    }

    fn close_channel(actual: u8, expected: u8) {
        assert!(
            actual.abs_diff(expected) <= 2,
            "channel {actual}, expected {expected} within two quantization steps"
        );
    }

    /// gpui_wgpu's `a_blurred_text_shadow_composites_its_coverage_unaltered`.
    #[::core::prelude::v1::test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly for renderer qualification"]
    fn native_blurred_text_shadow_composites_its_coverage_unaltered() -> Result<()> {
        let mut renderer = offscreen(32, 32)?;
        let content_mask = mask(32., 32.);
        let glyph = size(DevicePixels(6), DevicePixels(6));
        let (tile_size, coverage) = gpui::blur_glyph_coverage(glyph, &[255; 36], 2.5);
        let key = AtlasKey::Svg(RenderSvgParams {
            path: "native-text-shadow-coverage".into(),
            size: tile_size,
        });
        let tile = renderer
            .atlas
            .write_bytes(&key, tile_size, &coverage)?
            .context("shadow tile was not allocated")?;
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., 32., 32.), rgb(0xffffff), content_mask));
        scene.insert_primitive(MonochromeSprite {
            order: 0,
            raw_coverage: true.into(),
            tile,
            bounds: bounds(5., 5., tile_size.width.0 as f32, tile_size.height.0 as f32),
            content_mask,
            color: rgb(0x000000).into(),
            transformation: TransformationMatrix::unit(),
        });
        scene.finish();
        let image = renderer.render_to_image(&scene, WindowBackgroundAppearance::Opaque)?;
        let width = tile_size.width.0 as u32;
        for y in 0..tile_size.height.0 as u32 {
            for x in 0..width {
                let expected = 255 - coverage[(y * width + x) as usize];
                for channel in &image.get_pixel(x + 5, y + 5).0[..3] {
                    close_channel(*channel, expected);
                }
            }
        }
        Ok(())
    }

    #[::core::prelude::v1::test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly for renderer qualification"]
    fn native_quads_preserve_premultiplied_alpha_and_paths_resolve_antialiasing() -> Result<()> {
        let mut renderer = offscreen(128, 64)?;
        let content_mask = mask(128., 64.);
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., 32., 32.), rgb(0xff0000), content_mask));
        scene.insert_primitive(quad(
            bounds(16., 0., 32., 32.),
            rgba(0x0000ff80),
            content_mask,
        ));
        let mut path = Path::new(point(px(70.25), px(4.25)));
        path.line_to(point(px(110.25), px(4.25)));
        path.line_to(point(px(70.25), px(44.25)));
        let mut path = path.scale(1.);
        path.content_mask = content_mask;
        path.color = rgb(0xffffff).into();
        scene.insert_primitive(path);
        scene.finish();
        let image = renderer.render_to_image(&scene, WindowBackgroundAppearance::Transparent)?;
        assert_eq!(image.get_pixel(8, 8).0, [255, 0, 0, 255]);
        let overlap = image.get_pixel(24, 8).0;
        close_channel(overlap[0], 127);
        close_channel(overlap[2], 128);
        close_channel(overlap[3], 255);
        let transparent = image.get_pixel(40, 8).0;
        close_channel(transparent[2], 128);
        close_channel(transparent[3], 128);
        assert_eq!(image.get_pixel(80, 10).0, [255; 4]);
        assert_eq!(image.get_pixel(110, 44).0, [0; 4]);
        assert!(
            (70..111).any(|x| (4..45).any(|y| {
                let alpha = image.get_pixel(x, y).0[3];
                alpha > 0 && alpha < 255
            })),
            "the fractional triangle edge must retain MSAA coverage"
        );
        renderer.render_to_image(&Scene::default(), WindowBackgroundAppearance::Transparent)?;
        assert!(
            renderer.path_msaa.is_some(),
            "a short gap must keep the MSAA scratch for the next path frame"
        );
        draw_repeatedly(&mut renderer, &Scene::default(), SCRATCH_RELEASE_FRAMES)?;
        assert!(
            renderer.path_msaa.is_none() && renderer.path_resolved.is_none(),
            "a scene without paths must release the full-window MSAA scratch textures"
        );
        Ok(())
    }

    #[::core::prelude::v1::test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly for renderer qualification"]
    fn native_nested_layers_apply_text_gradients_clips_and_color_matrix() -> Result<()> {
        let mut renderer = offscreen(64, 64)?;
        let content_mask = mask(64., 64.);
        let tile_size = size(DevicePixels(8), DevicePixels(8));
        let key = AtlasKey::Svg(RenderSvgParams {
            path: "native-renderer-gradient-mask".into(),
            size: tile_size,
        });
        let mut coverage = vec![255u8; 64];
        for y in 2..6 {
            for x in 2..6 {
                coverage[y * 8 + x] = 0;
            }
        }
        let tile = renderer
            .atlas
            .write_bytes(&key, tile_size, &coverage)?
            .context("Native glyph mask tile was not allocated")?;
        let triangle = Arc::new(VnClipGeometry {
            fill_rule: VnFillRule::NonZero,
            segments: vec![[4., 4., 60., 4.], [60., 4., 4., 60.], [4., 60., 4., 4.]],
            bounds: bounds(4., 4., 56., 56.),
        });
        let rectangle = Arc::new(VnClipGeometry {
            fill_rule: VnFillRule::NonZero,
            segments: vec![
                [0., 0., 64., 0.],
                [64., 0., 64., 64.],
                [64., 64., 0., 64.],
                [0., 64., 0., 0.],
            ],
            bounds: bounds(0., 0., 64., 64.),
        });
        let text_paint = Arc::new(VnTextPaint {
            fill: VnTextFill {
                group_id: 1,
                backgrounds: vec![linear_gradient(
                    90.,
                    linear_color_stop(rgb(0xff0000), 0.),
                    linear_color_stop(rgb(0x0000ff), 1.),
                )],
            },
            bounds: [0., 0., 64., 64.],
            inverse: [1., 0., 0., 1., 0., 0.],
            scale_factor: 1.,
            geometry: triangle.clone(),
            content_mask,
        });
        let mut values = VnColorMatrix::IDENTITY;
        values[18] = 0.5;
        let mut scene = Scene::default();
        scene.vn_push_isolation(VnIsolation {
            geometry: triangle,
            color_matrix: None,
            text_paint: Some(text_paint),
            content_mask,
            capture: None,
        });
        scene.vn_push_isolation(VnIsolation {
            geometry: rectangle,
            color_matrix: Some(VnColorMatrix {
                group_id: 2,
                values,
                linear_rgb: false,
            }),
            text_paint: None,
            content_mask,
            capture: None,
        });
        scene.insert_primitive(MonochromeSprite {
            order: 0,
            raw_coverage: false.into(),
            tile,
            bounds: bounds(0., 0., 64., 64.),
            content_mask,
            color: rgb(0xffffff).into(),
            transformation: TransformationMatrix::unit(),
        });
        scene.vn_pop_isolation();
        scene.vn_pop_isolation();
        scene.finish();
        let image = renderer.render_to_image(&scene, WindowBackgroundAppearance::Transparent)?;
        let left = image.get_pixel(8, 8).0;
        let right = image.get_pixel(46, 8).0;
        close_channel(left[3], 128);
        close_channel(right[3], 128);
        assert!(
            left[0] > left[2] + 30,
            "left side of the masked gradient must be red: {left:?}"
        );
        assert!(
            right[2] > right[0] + 20,
            "right side of the masked gradient must be blue: {right:?}"
        );
        assert_eq!(
            image.get_pixel(24, 24).0,
            [0; 4],
            "the atlas mask's hole must remain transparent"
        );
        assert_eq!(
            image.get_pixel(55, 55).0,
            [0; 4],
            "pixels outside the triangle clip must remain transparent"
        );
        assert_eq!(renderer.layers.len(), 2);
        draw_repeatedly(&mut renderer, &Scene::default(), SCRATCH_RELEASE_FRAMES)?;
        assert!(
            renderer.layers.is_empty(),
            "closed isolated groups must release their scratch targets"
        );
        Ok(())
    }

    /// The scene of gpui_wgpu's and gpui_apple's `masks_multiply_by_alpha_or_luminance`.
    #[::core::prelude::v1::test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly for renderer qualification"]
    fn native_masks_multiply_by_alpha_or_luminance() -> Result<()> {
        let mut renderer = offscreen(64, 64)?;
        let content_mask = mask(64., 64.);
        let rectangle = |x: f32, y: f32| {
            let (right, bottom) = (x + 32., y + 32.);
            VnIsolation {
                geometry: Arc::new(VnClipGeometry {
                    fill_rule: VnFillRule::NonZero,
                    segments: vec![
                        [x, y, right, y],
                        [right, y, right, bottom],
                        [right, bottom, x, bottom],
                        [x, bottom, x, y],
                    ],
                    bounds: bounds(x, y, 32., 32.),
                }),
                color_matrix: None,
                text_paint: None,
                content_mask,
                capture: None,
            }
        };
        let red = |x: f32, y: f32| quad(bounds(x, y, 32., 32.), rgb(0xff0000), content_mask);
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., 64., 64.), rgb(0x0000ff), content_mask));

        scene.vn_push_isolation(rectangle(0., 0.));
        scene.insert_primitive(red(0., 0.));
        scene.vn_begin_mask(false);
        scene.insert_primitive(quad(bounds(0., 0., 16., 32.), rgba(0xffffff80), content_mask));
        scene.vn_pop_isolation();

        scene.vn_push_isolation(rectangle(32., 0.));
        scene.insert_primitive(red(32., 0.));
        scene.vn_begin_mask(true);
        scene.insert_primitive(quad(bounds(32., 0., 32., 32.), rgb(0x00ff00), content_mask));
        scene.vn_pop_isolation();

        scene.vn_push_isolation(rectangle(0., 32.));
        scene.insert_primitive(red(0., 32.));
        scene.vn_begin_mask(false);
        scene.vn_pop_isolation();

        scene.vn_push_isolation(rectangle(32., 32.));
        scene.vn_push_isolation(rectangle(32., 32.));
        scene.insert_primitive(red(32., 32.));
        scene.vn_begin_mask(false);
        scene.insert_primitive(quad(bounds(32., 32., 16., 32.), rgb(0xffffff), content_mask));
        scene.vn_pop_isolation();
        scene.vn_pop_isolation();
        scene.finish();
        let image = renderer.render_to_image(&scene, WindowBackgroundAppearance::Opaque)?;
        for (x, y, expected) in [
            (8, 16, [128, 0, 127, 255]),
            (24, 16, [0, 0, 255, 255]),
            (40, 16, [182, 0, 73, 255]),
            (8, 48, [0, 0, 255, 255]),
            (40, 40, [255, 0, 0, 255]),
            (56, 40, [0, 0, 255, 255]),
        ] {
            let actual = image.get_pixel(x, y).0;
            for (actual, expected) in actual.into_iter().zip(expected) {
                close_channel(actual, expected);
            }
        }
        assert_eq!(renderer.layers.len(), 3, "a nested mask draws two levels down");
        Ok(())
    }

    #[::core::prelude::v1::test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly for renderer qualification"]
    fn native_captures_keep_snapshots_that_later_frames_composite() -> Result<()> {
        let mut renderer = offscreen(192, 100)?;
        let content_mask = mask(192., 100.);
        let whole = Arc::new(VnClipGeometry::rect(bounds(0., 0., 192., 100.)));
        let isolation = |capture| VnIsolation {
            geometry: whole.clone(),
            color_matrix: None,
            text_paint: None,
            content_mask,
            capture,
        };
        let info = VnCaptureInfo {
            id: 7,
            border_box: bounds(20., 20., 40., 40.),
            transform: TransformationMatrix::unit(),
        };
        let mut captured = Scene::default();
        captured.vn_push_isolation(isolation(Some(info)));
        captured.insert_primitive(quad(bounds(20., 20., 40., 40.), rgb(0xff0000), content_mask));
        captured.vn_pop_isolation();
        captured.finish();
        let ink = captured.vn_captures[0].ink;
        let image = renderer.render_to_image(&captured, WindowBackgroundAppearance::Transparent)?;
        assert_eq!(image.get_pixel(40, 40).0, [0; 4], "a capture paints nowhere");

        let snapshot = |scene: &mut Scene, x: f32, y: f32, side: f32, opacity: f32, plus: bool| {
            scene.vn_insert_snapshot(VnSnapshotPaint {
                id: 7,
                to_unit: [1. / side, 0., -x / side, 0., 1. / side, -y / side],
                bounds: bounds(x, y, side, side),
                content_mask,
                opacity,
                plus_lighter: plus,
                in_place: false,
            })
        };
        let mut moved = Scene::default();
        snapshot(&mut moved, 100., 10., 80., 1., false);
        moved.finish();
        moved.vn_resolve_snapshots(|_| Some((info, ink)));
        let image = renderer.render_to_image(&moved, WindowBackgroundAppearance::Transparent)?;
        assert_eq!(image.get_pixel(140, 50).0, [255, 0, 0, 255]);
        assert_eq!(image.get_pixel(40, 40).0, [0; 4]);
        assert_eq!(image.get_pixel(98, 50).0, [0; 4]);

        let mut faded = Scene::default();
        for (x, plus) in [(20., true), (100., false)] {
            faded.vn_push_isolation(isolation(None));
            snapshot(&mut faded, x, 20., 40., 0.5, plus);
            snapshot(&mut faded, x, 20., 40., 0.5, plus);
            faded.vn_pop_isolation();
        }
        faded.finish();
        faded.vn_resolve_snapshots(|_| Some((info, ink)));
        let image = renderer.render_to_image(&faded, WindowBackgroundAppearance::Transparent)?;
        let summed = image.get_pixel(40, 40).0;
        close_channel(summed[0], 255);
        close_channel(summed[3], 255);
        close_channel(image.get_pixel(120, 40).0[3], 191);
        Ok(())
    }

    #[::core::prelude::v1::test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly for renderer qualification"]
    fn native_backdrop_blur_and_effects_observe_scene_order() -> Result<()> {
        let mut renderer = offscreen(64, 64)?;
        let content_mask = mask(64., 64.);
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., 64., 64.), rgb(0x000000), content_mask));
        scene.insert_primitive(quad(bounds(0., 0., 32., 64.), rgb(0xffffff), content_mask));
        scene.insert_primitive(BackdropBlurRect {
            bounds: bounds(8., 8., 48., 48.),
            content_mask,
            blur_radius: ScaledPixels(10.),
            ..Default::default()
        });
        scene.finish();
        let image = renderer.render_to_image(&scene, WindowBackgroundAppearance::Transparent)?;
        let before_edge = image.get_pixel(30, 32).0[0];
        let after_edge = image.get_pixel(33, 32).0[0];
        assert!(
            before_edge > 16 && before_edge < 240,
            "blur must soften the white side: {before_edge}"
        );
        assert!(
            after_edge > 16 && after_edge < 240,
            "blur must soften the black side: {after_edge}"
        );
        assert_eq!(image.get_pixel(2, 32).0, [255; 4]);
        assert_eq!(image.get_pixel(61, 32).0, [0, 0, 0, 255]);
        assert_eq!(
            renderer.blur.len(),
            2,
            "unused blur pyramid levels must not allocate textures"
        );

        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., 64., 64.), rgb(0xff0000), content_mask));
        scene.insert_primitive(EffectQuad {
            effect_id: effect_id::FROST,
            bounds: bounds(8., 8., 48., 48.),
            content_mask,
            params: EffectQuad::params_from(&[1., 0., 0., 0., 0., 0., 1., 1.]),
            opacity: 1.,
            scale: 1.,
            flags: EFFECT_FLAG_NEEDS_BACKDROP,
            ..Default::default()
        });
        scene.insert_primitive(quad(bounds(24., 24., 8., 8.), rgb(0x00ff00), content_mask));
        scene.finish();
        let image = renderer.render_to_image(&scene, WindowBackgroundAppearance::Transparent)?;
        assert_eq!(image.get_pixel(2, 32).0, [255, 0, 0, 255]);
        assert_eq!(image.get_pixel(16, 16).0, [0, 0, 255, 255]);
        assert_eq!(
            image.get_pixel(28, 28).0,
            [0, 255, 0, 255],
            "a later primitive must paint above the effect"
        );
        assert_eq!(
            renderer.blur.len(),
            2,
            "a short gap must not release blur levels a recent frame used"
        );
        draw_repeatedly(&mut renderer, &scene, SCRATCH_RELEASE_FRAMES)?;
        assert!(
            renderer.blur.is_empty() && renderer.snapshot.is_some(),
            "zero-radius effects must not retain blur levels"
        );
        draw_repeatedly(&mut renderer, &Scene::default(), SCRATCH_RELEASE_FRAMES)?;
        assert!(
            renderer.snapshot.is_none() && renderer.blur.is_empty(),
            "closing the backdrop scene must release its scratch textures"
        );
        Ok(())
    }

    /// Linux draws backdrop blur and the built-in effects through gpui_wgpu;
    /// both renderers must agree on the same scene.
    #[::core::prelude::v1::test]
    #[ignore = "requires a native D3D12 adapter and a Vulkan or GL adapter; run explicitly for renderer qualification"]
    fn native_backdrop_and_effects_match_the_wgpu_renderer() -> Result<()> {
        let content_mask = mask(64., 64.);
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., 64., 64.), rgb(0x000000), content_mask));
        scene.insert_primitive(quad(bounds(0., 0., 32., 64.), rgb(0xffffff), content_mask));
        scene.insert_primitive(quad(bounds(4., 40., 56., 6.), rgb(0x3060d0), content_mask));
        scene.insert_primitive(BackdropBlurRect {
            bounds: bounds(4., 4., 40., 30.),
            content_mask: mask(36., 64.),
            corner_radii: Corners::all(ScaledPixels(10.)),
            blur_radius: ScaledPixels(10.),
            opacity: 0.8,
            tint: Hsla { a: 0.3, ..red() },
            ..Default::default()
        });
        scene.insert_primitive(EffectQuad {
            effect_id: effect_id::FROST,
            bounds: bounds(36., 4., 24., 30.),
            content_mask,
            corner_radii: Corners::all(ScaledPixels(6.)),
            params: EffectQuad::params_from(&[0.7, 6., 0., 0., 1., 1., 1., 0.1]),
            opacity: 1.,
            backdrop_blur_radius: ScaledPixels(6.),
            scale: 1.,
            flags: EFFECT_FLAG_NEEDS_BACKDROP,
            ..Default::default()
        });
        scene.insert_primitive(EffectQuad {
            effect_id: effect_id::GLOW,
            bounds: bounds(6., 34., 28., 28.),
            content_mask,
            params: EffectQuad::params_from(&[6., 0.9, 0., 0., 0.2, 1., 0.4, 1.]),
            opacity: 1.,
            bleed: ScaledPixels(6.),
            scale: 1.,
            ..Default::default()
        });
        scene.insert_primitive(EffectQuad {
            effect_id: effect_id::NOISE,
            bounds: bounds(40., 40., 20., 20.),
            content_mask,
            params: EffectQuad::params_from(&[0.6, 2., 0., 0.]),
            opacity: 1.,
            scale: 1.,
            ..Default::default()
        });
        scene.insert_primitive(EffectQuad {
            effect_id: effect_id::PUNCH,
            bounds: bounds(44., 44., 18., 14.),
            content_mask: mask(64., 54.),
            corner_radii: Corners::all(ScaledPixels(5.)),
            opacity: 0.75,
            scale: 1.,
            ..Default::default()
        });
        scene.finish();

        let native =
            offscreen(64, 64)?.render_to_image(&scene, WindowBackgroundAppearance::Transparent)?;
        let wgpu = gpui_wgpu::WgpuHeadlessRenderer::new()?
            .render_scene_to_image(&scene, size(DevicePixels(64), DevicePixels(64)))?;
        let differences: Vec<u8> = native
            .as_raw()
            .iter()
            .zip(wgpu.as_raw())
            .map(|(native, wgpu)| native.abs_diff(*wgpu))
            .collect();
        let largest = differences.iter().copied().max().unwrap_or_default();
        let mean = differences
            .iter()
            .map(|&difference| f64::from(difference))
            .sum::<f64>()
            / differences.len() as f64;
        let outliers = differences
            .iter()
            .filter(|&&difference| difference > 4)
            .count();
        assert!(
            mean < 0.5 && outliers * 200 < differences.len() && largest <= 16,
            "wgpu must match D3D12: mean {mean:.3}, {outliers} channels off by more than 4, \
             largest {largest}"
        );
        Ok(())
    }

    #[::core::prelude::v1::test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly for renderer qualification"]
    fn native_reloaded_app_effect_draws_on_the_next_frame_and_a_broken_one_keeps_the_last()
    -> Result<()> {
        let install = |hlsl: &str| -> Result<()> {
            reload_app_effects(
                AppEffects::new(
                    "test",
                    r#"{"schema": 10, "effects": [
                        {"name": "tint", "backdrop": false, "hlsl": "tint.hlsl"}
                    ]}"#,
                    [("tint.hlsl".to_string(), hlsl.to_string())],
                )
                .map_err(anyhow::Error::msg)?,
            );
            Ok(())
        };
        let solid = |rgb: &str| {
            format!("float4 effect_fragment() : SV_Target {{ return float4({rgb}, 1.0); }}")
        };
        let content_mask = mask(64., 64.);
        let scene = || {
            let mut scene = Scene::default();
            scene.insert_primitive(quad(bounds(0., 0., 64., 64.), rgb(0x000000), content_mask));
            scene.insert_primitive(EffectQuad {
                effect_id: effect_id::COUNT,
                bounds: bounds(8., 8., 48., 48.),
                content_mask,
                opacity: 1.,
                scale: 1.,
                ..Default::default()
            });
            scene.finish();
            scene
        };
        let mut renderer = offscreen(64, 64)?;
        let center = |renderer: &mut DirectXRenderer| -> Result<[u8; 4]> {
            let image = renderer.render_to_image(&scene(), WindowBackgroundAppearance::Opaque)?;
            assert_eq!(image.get_pixel(2, 2).0, [0, 0, 0, 255]);
            Ok(image.get_pixel(32, 32).0)
        };
        install(&solid("1.0, 0.0, 0.0"))?;
        assert_eq!(center(&mut renderer)?, [255, 0, 0, 255]);
        install(&solid("0.0, 1.0, 0.0"))?;
        assert_eq!(center(&mut renderer)?, [0, 255, 0, 255]);
        install("float4 effect_fragment() : SV_Target { return oops; }")?;
        assert_eq!(
            center(&mut renderer)?,
            [0, 255, 0, 255],
            "a replacement that does not compile keeps the last good pipeline"
        );
        Ok(())
    }

    use crate::directx_renderer::effect_layer_tests::{
        install_layer_effects, layer_effect, stamped_element,
    };

    #[::core::prelude::v1::test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly for renderer qualification"]
    fn native_content_effect_draws_its_element_in_its_place_and_cross_fades_with_it() -> Result<()>
    {
        install_layer_effects()?;
        let content_mask = mask(64., 64.);
        let scene = |draws: Vec<VnEffectDraw>| {
            let mut scene = Scene::default();
            scene.insert_primitive(quad(bounds(0., 0., 64., 64.), rgb(0x000000), content_mask));
            scene.vn_push_effect(VnEffectPaint {
                draws,
                content: true,
                content_mask,
            });
            scene.insert_primitive(quad(
                bounds(16., 16., 32., 32.),
                rgb(0x00ff00),
                content_mask,
            ));
            scene.insert_primitive(quad(bounds(0., 0., 8., 8.), rgb(0xffffff), content_mask));
            scene.vn_pop_isolation();
            scene.finish();
            scene
        };
        let swap = |opacity| VnEffectDraw {
            quad: layer_effect(
                effect_id::COUNT,
                bounds(16., 16., 32., 32.),
                opacity,
                EFFECT_FLAG_NEEDS_CONTENT,
            ),
            history: 0,
            textures: None,
        };
        let mut renderer = offscreen(64, 64)?;
        let image = renderer.render_to_image(
            &scene(vec![swap(1.)]),
            WindowBackgroundAppearance::Transparent,
        )?;
        assert_eq!(
            image.get_pixel(32, 32).0,
            [0, 0, 255, 255],
            "the shaded box"
        );
        assert_eq!(
            image.get_pixel(4, 4).0,
            [0, 0, 0, 255],
            "content outside the quad does not show"
        );

        let plain = VnEffectDraw {
            quad: layer_effect(
                effect_id::CONTENT,
                bounds(0., 0., 64., 64.),
                0.5,
                EFFECT_FLAG_NEEDS_CONTENT,
            ),
            history: 0,
            textures: None,
        };
        let image = renderer.render_to_image(
            &scene(vec![plain, swap(0.5)]),
            WindowBackgroundAppearance::Transparent,
        )?;
        let shaded = image.get_pixel(32, 32).0;
        close_channel(shaded[1], 64);
        close_channel(shaded[2], 128);
        let corner = image.get_pixel(4, 4).0;
        close_channel(corner[0], 128);
        close_channel(corner[3], 255);
        Ok(())
    }

    #[::core::prelude::v1::test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly for renderer qualification"]
    fn native_history_steps_at_most_at_the_step_rate_and_starts_over() -> Result<()> {
        install_layer_effects()?;
        let content_mask = mask(64., 64.);
        let scene = |rect: Bounds<ScaledPixels>, time: f32| {
            let mut scene = Scene::default();
            scene.insert_primitive(quad(bounds(0., 0., 64., 64.), rgb(0x000000), content_mask));
            scene.vn_push_effect(VnEffectPaint {
                draws: vec![VnEffectDraw {
                    quad: EffectQuad {
                        time,
                        ..layer_effect(effect_id::COUNT + 1, rect, 1., EFFECT_FLAG_HISTORY)
                    },
                    history: 7,
                    textures: None,
                }],
                content: false,
                content_mask,
            });
            scene.vn_pop_isolation();
            scene.finish();
            scene
        };
        let mut renderer = offscreen(64, 64)?;
        let mut shows = |scene: Scene, red: u8| -> Result<()> {
            let image =
                renderer.render_to_image(&scene, WindowBackgroundAppearance::Transparent)?;
            assert_eq!(image.get_pixel(8, 8).0, [0, 0, 0, 255], "outside the quad");
            let center = image.get_pixel(32, 32).0;
            close_channel(center[0], red);
            close_channel(center[3], 255);
            Ok(())
        };
        let rect = bounds(16., 16., 32., 32.);
        shows(scene(rect, 0.), 0)?;
        shows(scene(rect, 1.), 64)?;
        shows(scene(rect, 1.001), 128)?;
        shows(scene(rect, 1.002), 128)?;
        shows(scene(bounds(16., 16., 30., 32.), 2.), 0)?;
        renderer.render_to_image(&Scene::default(), WindowBackgroundAppearance::Transparent)?;
        assert!(renderer.histories.is_empty(), "a frame without it frees it");
        Ok(())
    }

    #[::core::prelude::v1::test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly for renderer qualification"]
    fn native_content_effect_samples_its_own_images() -> Result<()> {
        install_layer_effects()?;
        let mut renderer = offscreen(64, 64)?;
        let transparent = WindowBackgroundAppearance::Transparent;
        let image = renderer.render_to_image(&stamped_element(true), transparent)?;
        assert_eq!(
            image.get_pixel(24, 32).0,
            [255, 255, 0, 255],
            "the first draw's"
        );
        assert_eq!(
            image.get_pixel(40, 32).0,
            [255, 0, 255, 255],
            "the second draw's"
        );
        let image = renderer.render_to_image(&stamped_element(false), transparent)?;
        assert_eq!(image.get_pixel(32, 32).0, [0, 0, 0, 255], "none bound");
        Ok(())
    }

    fn draw_repeatedly(renderer: &mut DirectXRenderer, scene: &Scene, frames: u32) -> Result<()> {
        for _ in 0..frames {
            renderer.render_to_image(scene, WindowBackgroundAppearance::Transparent)?;
        }
        Ok(())
    }

    #[::core::prelude::v1::test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly for renderer qualification"]
    fn native_backdrop_blur_region_matches_whole_window_blur() -> Result<()> {
        // A blur rect whose kernel reach stays inside the window takes the
        // region-limited path; one covering the window takes the full path.
        // Both must agree with a whole-window reference inside the rect.
        let content_mask = mask(256., 256.);
        let scene = |rect: Bounds<ScaledPixels>| {
            let mut scene = Scene::default();
            scene.insert_primitive(quad(bounds(0., 0., 256., 256.), rgb(0x000000), content_mask));
            for index in 0..8 {
                let x = 16. + index as f32 * 28.;
                scene.insert_primitive(quad(bounds(x, 0., 12., 256.), rgb(0xffffff), content_mask));
            }
            scene.insert_primitive(BackdropBlurRect {
                bounds: rect,
                content_mask,
                blur_radius: ScaledPixels(6.),
                ..Default::default()
            });
            scene.finish();
            scene
        };
        let small = bounds(96., 96., 64., 64.);
        let scene = scene(small);
        let levels = scene.backdrop_blur_rects[0].effective_kernel_levels() as usize;
        assert!(levels > 0, "the fixture must blur");
        assert!(
            intersect(round_out(&small, blur_reach(levels)), full_rect(256, 256))
                != full_rect(256, 256),
            "the fixture must exercise the region-limited path"
        );
        let mut limited = offscreen(256, 256)?;
        let limited_image =
            limited.render_to_image(&scene, WindowBackgroundAppearance::Transparent)?;
        FULL_WINDOW_BACKDROP.with(|full| full.set(true));
        let mut reference = offscreen(256, 256)?;
        let reference_image =
            reference.render_to_image(&scene, WindowBackgroundAppearance::Transparent);
        FULL_WINDOW_BACKDROP.with(|full| full.set(false));
        let reference_image = reference_image?;
        for y in 96..160 {
            for x in 96..160 {
                let a = limited_image.get_pixel(x, y).0;
                let b = reference_image.get_pixel(x, y).0;
                for channel in 0..4 {
                    close_channel(a[channel], b[channel]);
                }
            }
        }
        Ok(())
    }
    #[::core::prelude::v1::test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly for renderer qualification"]
    fn native_upload_pools_reuse_completed_resources_and_refresh_pixels() -> Result<()> {
        let mut renderer = offscreen(32, 32)?;
        let content_mask = mask(32., 32.);
        let target = Target::new(&renderer.gpu, 32, 32, 1)?;
        let mut identities: [Option<ID3D12Resource>; BUFFERS] = std::array::from_fn(|_| None);
        let mut guarded = UploadPool::default();
        assert!(
            guarded.begin(0, 1).is_err(),
            "unfinished frame uploads must not be reset"
        );
        for (frame, color) in [0xff0000, 0x00ff00, 0x0000ff, 0x0000ff, 0xff0000, 0x00ff00]
            .into_iter()
            .enumerate()
        {
            let slot = frame % BUFFERS;
            let previous = renderer.fences[slot];
            renderer.gpu.wait_for(previous)?;
            let mut uploads = std::mem::take(&mut renderer.uploads[slot]);
            uploads.begin(renderer.gpu.completed_value(), previous)?;
            let mut scene = Scene::default();
            scene.insert_primitive(quad(bounds(0., 0., 32., 32.), rgb(color), content_mask));
            let x = if frame < BUFFERS { 8.25 } else { 12.25 };
            let mut path = Path::new(point(px(x), px(8.25)));
            path.line_to(point(px(x + 16.), px(8.25)));
            path.line_to(point(px(x), px(24.25)));
            let mut path = path.scale(1.);
            path.content_mask = content_mask;
            path.color = rgb(0xffffff).into();
            scene.insert_primitive(path);
            scene.finish();
            let fence = renderer.render(
                &scene,
                WindowBackgroundAppearance::Transparent,
                target.clone(),
                false,
                &mut uploads,
            )?;
            uploads.finish();
            assert!(
                uploads.cursor >= 5 * 256,
                "globals, dummy, quads and both path streams must use the pool"
            );
            let resource = &uploads.buffer.as_ref().unwrap().resource;
            if let Some(previous) = &identities[slot] {
                assert_eq!(
                    previous.as_raw(),
                    resource.as_raw(),
                    "a completed slot must reuse its existing COM resource"
                );
            } else {
                identities[slot] = Some(resource.clone());
            }
            renderer.uploads[slot] = uploads;
            renderer.fences[slot] = fence;
            let image = renderer.capture_target(target.clone())?;
            let expected = [
                ((color >> 16) & 255) as u8,
                ((color >> 8) & 255) as u8,
                (color & 255) as u8,
                255,
            ];
            assert_eq!(
                image.get_pixel(1, 1).0,
                expected,
                "reused quad uploads must contain the current frame's color"
            );
            assert_eq!(image.get_pixel(16, 12).0, [255; 4]);
            assert_eq!(
                image.get_pixel(10, 10).0,
                if frame < BUFFERS { [255; 4] } else { expected },
                "reused path uploads must contain the current frame's vertices"
            );
        }
        drop(identities);
        for slot in 0..BUFFERS {
            let previous = renderer.fences[slot];
            renderer.gpu.wait_for(previous)?;
            let mut uploads = std::mem::take(&mut renderer.uploads[slot]);
            uploads.begin(renderer.gpu.completed_value(), previous)?;
            let fence = renderer.render(
                &Scene::default(),
                WindowBackgroundAppearance::Transparent,
                target.clone(),
                false,
                &mut uploads,
            )?;
            uploads.finish();
            assert_eq!(
                uploads.cursor,
                2 * 256,
                "an empty scene uploads only globals and the dummy binding"
            );
            renderer.uploads[slot] = uploads;
            renderer.fences[slot] = fence;
        }
        renderer.gpu.wait_idle()?;
        assert_eq!(renderer.capture_target(target)?.get_pixel(16, 12).0, [0; 4]);
        Ok(())
    }

    #[::core::prelude::v1::test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly for renderer qualification"]
    fn native_upload_pool_grows_within_a_frame_and_shrinks_when_sparse() -> Result<()> {
        let renderer = offscreen(8, 8)?;
        let retained = Retained::default();
        let mut pool = UploadPool::default();
        pool.begin(0, 0)?;
        let small = pool.write(&renderer.gpu, &retained, &[1u8; 16])?;
        let large_data = vec![7u8; 4 * MIN_UPLOAD_CAPACITY as usize];
        let large = pool.write(&renderer.gpu, &retained, &large_data)?;
        assert_ne!(small, large);
        let grown = pool.buffer.as_ref().unwrap().capacity;
        assert!(grown >= 4 * MIN_UPLOAD_CAPACITY);
        assert_eq!(
            retained.lock().unwrap().len(),
            2,
            "the outgrown buffer must stay alive until the frame's fence"
        );
        unsafe {
            let mapped = pool.buffer.as_ref().unwrap().mapped;
            assert_eq!(*mapped, 7);
            assert_eq!(*mapped.add(large_data.len() - 1), 7);
        }
        pool.finish();
        for frame in 0..UPLOAD_SHRINK_FRAMES {
            assert!(pool.buffer.is_some(), "shrunk after {frame} sparse frames");
            pool.begin(0, 0)?;
            pool.write(&renderer.gpu, &retained, &[0u8; 64])?;
            pool.finish();
        }
        assert!(pool.buffer.is_none(), "a sparse buffer must be released");
        assert_eq!(pool.capacity_hint, MIN_UPLOAD_CAPACITY);
        pool.begin(0, 0)?;
        pool.write(&renderer.gpu, &retained, &[0u8; 64])?;
        assert_eq!(pool.buffer.as_ref().unwrap().capacity, MIN_UPLOAD_CAPACITY);
        Ok(())
    }
}
