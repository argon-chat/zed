use super::{
    DirectXDevices,
    atlas::{Atlas, AtlasTexture},
    device::{Device, KeepAlive, TextureLocation, transition},
    pipelines::{
        Pipelines, ROOT_BATCH, ROOT_CLIP_SEGMENTS, ROOT_GLOBALS, ROOT_INSTANCES,
        ROOT_TEXT_BACKGROUNDS, ROOT_TEXTURES,
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
};
use windows::{
    Win32::{
        Foundation::HWND,
        Graphics::{
            Direct3D::*,
            Direct3D12::*,
            DirectComposition::*,
            Dxgi::{Common::*, *},
        },
    },
    core::Interface,
};

const FORMAT: DXGI_FORMAT = DXGI_FORMAT_B8G8R8A8_UNORM;
const BUFFERS: usize = 3;
const SHADER_READ: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0,
);
const MAX_UPLOAD_BYTES: usize = 256 * 1024 * 1024;
const MAX_LAYER_DEPTH: usize = 8;
const DESCRIPTORS_PER_PAGE: u32 = 256;
const _: () = assert!(std::mem::size_of::<GlobalParams>() == 48);
const _: () = assert!(std::mem::offset_of!(GlobalParams, is_bgr) == 32);
const _: () = assert!(std::mem::size_of::<PathRasterizationSprite>() == 292);
const _: () = assert!(std::mem::offset_of!(PathRasterizationSprite, color) == 16);
const _: () = assert!(std::mem::size_of::<PathSprite>() == 16);
type Retained = Arc<Mutex<Vec<KeepAlive>>>;

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
            Format: FORMAT,
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
            width,
            height,
            state: Mutex::new(state),
        }))
    }
    fn new(device: &Device, width: u32, height: u32, samples: u32) -> Result<Arc<Self>> {
        let resource = device.create_texture_multisampled(
            width,
            height,
            FORMAT,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COMMON,
            samples,
        )?;
        Self::from_resource(
            device,
            resource,
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
            format: FORMAT,
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
struct Composition {
    _device: IDCompositionDevice,
    _target: IDCompositionTarget,
    _visual: IDCompositionVisual,
}
impl Composition {
    fn new(hwnd: HWND, swap_chain: &IDXGISwapChain3) -> Result<Self> {
        let device: IDCompositionDevice =
            unsafe { DCompositionCreateDevice(None::<&IDXGIDevice>)? };
        let target = unsafe { device.CreateTargetForHwnd(hwnd, true)? };
        let visual = unsafe { device.CreateVisual()? };
        unsafe {
            visual.SetContent(swap_chain)?;
            target.SetRoot(&visual)?;
            device.Commit()?;
        }
        Ok(Self {
            _device: device,
            _target: target,
            _visual: visual,
        })
    }
}

pub(crate) struct DirectXRenderer {
    hwnd: HWND,
    gpu: Arc<Device>,
    atlas: Arc<Atlas>,
    pipelines: Pipelines,
    swap_chain: Option<IDXGISwapChain3>,
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
}
impl DirectXRenderer {
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
            Flags: 0,
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
        let composition = if disable_direct_composition {
            None
        } else {
            Some(Composition::new(hwnd, &swap_chain)?)
        };
        let pipelines = Pipelines::new(&gpu.device)?;
        let atlas = Arc::new(Atlas::new(gpu.clone()));
        let back_buffers = Self::buffers(&gpu, &swap_chain, 1, 1)?;
        Ok(Self {
            hwnd,
            gpu,
            atlas,
            pipelines,
            swap_chain: Some(swap_chain),
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
        self.swap_chain.take();
        self.layers.clear();
        self.path_msaa = None;
        self.path_resolved = None;
        self.snapshot = None;
        self.blur.clear();
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
                    DXGI_SWAP_CHAIN_FLAG(0),
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
        if self.skip_draws {
            return Ok(());
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
        if self.gpu.completed_value() < fence {
            if self.present_no_wait {
                return Ok(());
            }
            self.gpu.wait_for(fence)?;
        }
        let target = self
            .back_buffers
            .get(index)
            .context("D3D12 back buffer missing")?
            .clone();
        let completed = self.render(scene, appearance, target, true)?;
        self.fences[index] = completed;
        let flags = if self.present_no_wait {
            DXGI_PRESENT_DO_NOT_WAIT
        } else {
            DXGI_PRESENT(0)
        };
        let result = unsafe {
            self.swap_chain
                .as_ref()
                .context("D3D12 swap chain missing")?
                .Present(0, flags)
        };
        if self.present_no_wait && result == DXGI_ERROR_WAS_STILL_DRAWING {
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
    ) -> Result<u64> {
        let depth = layer_depth(scene, 0, 0)?;
        self.layers.truncate(depth);
        if scene.paths.is_empty() {
            self.path_msaa = None;
            self.path_resolved = None;
        }
        let needs_backdrop = self.prepare_pipelines(scene)?;
        if !needs_backdrop {
            self.snapshot = None;
            self.blur.clear();
        } else {
            let levels = scene
                .backdrop_blur_rects
                .iter()
                .map(BackdropBlurRect::effective_kernel_levels)
                .chain(
                    scene
                        .effect_quads
                        .iter()
                        .map(EffectQuad::effective_kernel_levels),
                )
                .max()
                .unwrap_or(0);
            self.blur.truncate(levels as usize);
        }
        let retained: Retained = Arc::new(Mutex::new(Vec::new()));
        let gpu = self.gpu.clone();
        let font = Self::get_font_info();
        let globals = upload(
            &gpu,
            &retained,
            &[GlobalParams {
                gamma_ratios: font.gamma_ratios,
                viewport_size: [self.width as f32, self.height as f32],
                grayscale_enhanced_contrast: font.grayscale_enhanced_contrast,
                subpixel_enhanced_contrast: font.subpixel_enhanced_contrast,
                is_bgr: font.is_bgr as u32,
                padding: [0; 3],
            }],
        )?;
        let data = SceneBuffers::new(&gpu, &retained, scene)?;
        let descriptors = Descriptors::new(gpu.clone(), retained.clone());
        // Atlas insertion takes the atlas lock before submitting uploads. Snapshot it
        // before submit takes the queue lock so rendering cannot reverse that order.
        let mut textures = HashMap::new();
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
            if let std::collections::hash_map::Entry::Vacant(entry) = textures.entry(texture_id) {
                let texture = self.atlas.get_texture(texture_id).with_context(|| {
                    format!("D3D12 scene references missing atlas texture {texture_id:?}")
                })?;
                entry.insert(texture);
            }
        }
        let mut changes = StateChanges::new();
        let result = gpu.submit(vec![retained.clone() as KeepAlive], |list| {
            let mut encoding = Encoding {
                gpu: &gpu,
                list,
                pipelines: &mut self.pipelines,
                textures: &textures,
                changes: &mut changes,
                scene,
                data,
                globals,
                retained,
                descriptors,
                target: target.clone(),
                width: self.width,
                height: self.height,
                layers: &mut self.layers,
                path_msaa: &mut self.path_msaa,
                path_resolved: &mut self.path_resolved,
                snapshot: &mut self.snapshot,
                blur: &mut self.blur,
            };
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
            Ok(())
        });
        if result.is_ok() {
            changes.commit();
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
        let mut needs_backdrop = !scene.backdrop_blur_rects.is_empty();
        let mut effects = HashSet::new();
        for effect in &scene.effect_quads {
            if effects.insert(effect.effect_id) {
                let (_, backdrop) = self.pipelines.effect(effect.effect_id, FORMAT, 1)?;
                needs_backdrop |= backdrop;
            }
        }
        if needs_backdrop {
            self.pipelines
                .pipeline(ShaderModule::BlurDownsample, FORMAT, 1)?;
            self.pipelines
                .pipeline(ShaderModule::BlurUpsample, FORMAT, 1)?;
        }
        Ok(needs_backdrop)
    }
    pub(crate) fn render_to_image(
        &mut self,
        scene: &Scene,
        appearance: WindowBackgroundAppearance,
    ) -> Result<image::RgbaImage> {
        ensure!(
            !self.skip_draws,
            "D3D12 capture unavailable during device recovery"
        );
        let target = Target::new(&self.gpu, self.width, self.height, 1)?;
        self.render(scene, appearance, target.clone(), false)?;
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
                required = required.max(layer_depth(scene, layer.group as usize, depth + 1)?);
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
fn upload<T>(gpu: &Device, retained: &Retained, data: &[T]) -> Result<ID3D12Resource> {
    let bytes = size_of_val(data);
    ensure!(
        bytes <= MAX_UPLOAD_BYTES,
        "D3D12 scene upload exceeds {MAX_UPLOAD_BYTES} bytes"
    );
    let resource = gpu.create_buffer(
        (bytes.max(1) as u64 + 255) & !255,
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
        if bytes > 0 {
            ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), mapped.cast::<u8>(), bytes);
        } else {
            ptr::write_bytes(mapped.cast::<u8>(), 0, 256);
        }
        resource.Unmap(
            0,
            Some(&D3D12_RANGE {
                Begin: 0,
                End: bytes.max(1),
            }),
        );
    }
    retain(retained, resource.clone());
    Ok(resource)
}
fn upload_or_dummy<T>(
    gpu: &Device,
    retained: &Retained,
    dummy: &ID3D12Resource,
    data: &[T],
) -> Result<ID3D12Resource> {
    if data.is_empty() {
        Ok(dummy.clone())
    } else {
        upload(gpu, retained, data)
    }
}

struct SceneBuffers {
    dummy: ID3D12Resource,
    shadows: ID3D12Resource,
    quads: ID3D12Resource,
    blur_rects: ID3D12Resource,
    effects: ID3D12Resource,
    underlines: ID3D12Resource,
    mono: ID3D12Resource,
    subpixel: ID3D12Resource,
    poly: ID3D12Resource,
    layers: ID3D12Resource,
    clips: ID3D12Resource,
    text_backgrounds: ID3D12Resource,
}
impl SceneBuffers {
    fn new(gpu: &Device, retained: &Retained, scene: &Scene) -> Result<Self> {
        let dummy = upload(gpu, retained, &[0u32; 64])?;
        Ok(Self {
            dummy: dummy.clone(),
            shadows: upload_or_dummy(gpu, retained, &dummy, &scene.shadows)?,
            quads: upload_or_dummy(gpu, retained, &dummy, &scene.quads)?,
            blur_rects: upload_or_dummy(gpu, retained, &dummy, &scene.backdrop_blur_rects)?,
            effects: upload_or_dummy(gpu, retained, &dummy, &scene.effect_quads)?,
            underlines: upload_or_dummy(gpu, retained, &dummy, &scene.underlines)?,
            mono: upload_or_dummy(gpu, retained, &dummy, &scene.monochrome_sprites)?,
            subpixel: upload_or_dummy(gpu, retained, &dummy, &scene.subpixel_sprites)?,
            poly: upload_or_dummy(gpu, retained, &dummy, &scene.polychrome_sprites)?,
            layers: upload_or_dummy(gpu, retained, &dummy, &scene.vn_layers)?,
            clips: upload_or_dummy(gpu, retained, &dummy, &scene.vn_clip_segments)?,
            text_backgrounds: upload_or_dummy(gpu, retained, &dummy, &scene.vn_text_backgrounds)?,
        })
    }
}
struct Descriptors {
    gpu: Arc<Device>,
    retained: Retained,
    heap: Option<ID3D12DescriptorHeap>,
    used: u32,
}
impl Descriptors {
    fn new(gpu: Arc<Device>, retained: Retained) -> Self {
        Self {
            gpu,
            retained,
            heap: None,
            used: DESCRIPTORS_PER_PAGE,
        }
    }
    fn textures(
        &mut self,
        list: &ID3D12GraphicsCommandList,
        first: Option<&AtlasTexture>,
        second: Option<&AtlasTexture>,
    ) -> Result<D3D12_GPU_DESCRIPTOR_HANDLE> {
        if self.used + 2 > DESCRIPTORS_PER_PAGE {
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
            retain(&self.retained, heap.clone());
            self.heap = Some(heap);
            self.used = 0;
        }
        let heap = self
            .heap
            .as_ref()
            .context("D3D12 descriptor page missing")?;
        let stride = unsafe {
            self.gpu
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };
        let cpu = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let gpu = unsafe { heap.GetGPUDescriptorHandleForHeapStart() };
        for (offset, texture) in [first, second].into_iter().enumerate() {
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
        unsafe {
            list.SetDescriptorHeaps(&[Some(heap.clone())]);
        }
        let table = D3D12_GPU_DESCRIPTOR_HANDLE {
            ptr: gpu.ptr + u64::from(self.used * stride),
        };
        self.used += 2;
        Ok(table)
    }
}

struct Encoding<'a> {
    gpu: &'a Arc<Device>,
    list: &'a ID3D12GraphicsCommandList,
    pipelines: &'a mut Pipelines,
    textures: &'a HashMap<AtlasTextureId, AtlasTexture>,
    changes: &'a mut StateChanges,
    scene: &'a Scene,
    data: SceneBuffers,
    globals: ID3D12Resource,
    retained: Retained,
    descriptors: Descriptors,
    target: Arc<Target>,
    width: u32,
    height: u32,
    layers: &'a mut Vec<Arc<Target>>,
    path_msaa: &'a mut Option<Arc<Target>>,
    path_resolved: &'a mut Option<Arc<Target>>,
    snapshot: &'a mut Option<Arc<Target>>,
    blur: &'a mut Vec<Arc<Target>>,
}
impl Encoding<'_> {
    fn bind_target(&mut self, target: &Arc<Target>) -> Result<()> {
        target.to(self.list, D3D12_RESOURCE_STATE_RENDER_TARGET, self.changes);
        retain(&self.retained, target.clone());
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
            self.list
                .RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                    left: 0,
                    top: 0,
                    right: target.width as i32,
                    bottom: target.height as i32,
                }]);
        }
        Ok(())
    }
    fn draw(
        &mut self,
        module: ShaderModule,
        buffer: &ID3D12Resource,
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
        buffer: &ID3D12Resource,
        start: u32,
        count: u32,
        vertices: u32,
        first: Option<&AtlasTexture>,
        second: Option<&AtlasTexture>,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let table = self.descriptors.textures(self.list, first, second)?;
        retain(&self.retained, pipeline.clone());
        retain(&self.retained, self.pipelines.root_signature.clone());
        let batch = [start, 0, 0, 0];
        unsafe {
            self.list.SetPipelineState(pipeline);
            self.list
                .SetGraphicsRootSignature(&self.pipelines.root_signature);
            self.list.SetGraphicsRootConstantBufferView(
                ROOT_GLOBALS,
                self.globals.GetGPUVirtualAddress(),
            );
            self.list
                .SetGraphicsRoot32BitConstants(ROOT_BATCH, 4, batch.as_ptr().cast(), 0);
            self.list
                .SetGraphicsRootShaderResourceView(ROOT_INSTANCES, buffer.GetGPUVirtualAddress());
            self.list.SetGraphicsRootShaderResourceView(
                ROOT_CLIP_SEGMENTS,
                self.data.clips.GetGPUVirtualAddress(),
            );
            self.list.SetGraphicsRootShaderResourceView(
                ROOT_TEXT_BACKGROUNDS,
                self.data.text_backgrounds.GetGPUVirtualAddress(),
            );
            self.list
                .SetGraphicsRootDescriptorTable(ROOT_TEXTURES, table);
            self.list.IASetPrimitiveTopology(if vertices == 4 {
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP
            } else {
                D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST
            });
            self.list.DrawInstanced(vertices, count, 0, 0);
        }
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
        self.target = target.clone();
        self.bind_target(&target)?;
        unsafe {
            self.list
                .ClearRenderTargetView(target.rtv(), &[0.0; 4], None);
        }
        self.group(layer.group as usize, depth + 1)?;
        self.target = parent.clone();
        self.bind_target(&parent)?;
        target.to(self.list, SHADER_READ, self.changes);
        let texture = target.sampled();
        let buffer = self.data.layers.clone();
        self.draw(
            ShaderModule::VnLayer,
            &buffer,
            index as u32,
            1,
            4,
            Some(&texture),
            None,
            1,
        )
    }
    fn batch(&mut self, batch: PrimitiveBatch) -> Result<()> {
        match batch {
            PrimitiveBatch::Quads(range) => {
                let buffer = self.data.quads.clone();
                self.draw(
                    ShaderModule::Quad,
                    &buffer,
                    range.start as u32,
                    range.len() as u32,
                    4,
                    None,
                    None,
                    1,
                )
            }
            PrimitiveBatch::Shadows(range) => {
                let buffer = self.data.shadows.clone();
                self.draw(
                    ShaderModule::Shadow,
                    &buffer,
                    range.start as u32,
                    range.len() as u32,
                    4,
                    None,
                    None,
                    1,
                )
            }
            PrimitiveBatch::Underlines(range) => {
                let buffer = self.data.underlines.clone();
                self.draw(
                    ShaderModule::Underline,
                    &buffer,
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
                let buffer = self.data.mono.clone();
                self.draw(
                    ShaderModule::MonochromeSprite,
                    &buffer,
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
                let buffer = self.data.subpixel.clone();
                self.draw(
                    ShaderModule::SubpixelSprite,
                    &buffer,
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
                let buffer = self.data.poly.clone();
                self.draw(
                    ShaderModule::PolychromeSprite,
                    &buffer,
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
        let buffer = upload(self.gpu, &self.retained, &vertices)?;
        self.bind_target(&msaa)?;
        unsafe {
            self.list.ClearRenderTargetView(msaa.rtv(), &[0.0; 4], None);
        }
        self.draw(
            ShaderModule::PathRasterization,
            &buffer,
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
        unsafe {
            self.list
                .ResolveSubresource(&resolved.resource, 0, &msaa.resource, 0, FORMAT);
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
            let mut bounds = first_path.clipped_bounds();
            for path in paths.iter().skip(1) {
                bounds = bounds.union(&path.clipped_bounds());
            }
            vec![PathSprite { bounds }]
        };
        let buffer = upload(self.gpu, &self.retained, &sprites)?;
        let target = self.target.clone();
        self.bind_target(&target)?;
        self.draw(
            ShaderModule::PathSprite,
            &buffer,
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
    fn backdrop(&mut self, levels: usize) -> Result<(AtlasTexture, AtlasTexture)> {
        let levels = levels.min(MAX_BACKDROP_BLUR_KERNEL_LEVELS as usize);
        self.ensure_blur(levels)?;
        let snapshot = self
            .snapshot
            .as_ref()
            .context("D3D12 backdrop snapshot missing")?
            .clone();
        self.target
            .to(self.list, D3D12_RESOURCE_STATE_COPY_SOURCE, self.changes);
        snapshot.to(self.list, D3D12_RESOURCE_STATE_COPY_DEST, self.changes);
        retain(&self.retained, snapshot.clone());
        unsafe {
            self.list
                .CopyResource(&snapshot.resource, &self.target.resource);
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
            self.blur_pass(ShaderModule::BlurDownsample, &source, &target)?;
            source = target.sampled();
        }
        for level in (1..levels).rev() {
            let target = self
                .blur
                .get(level - 1)
                .context("D3D12 blur pyramid target missing")?
                .clone();
            self.blur_pass(ShaderModule::BlurUpsample, &source, &target)?;
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
    ) -> Result<()> {
        self.bind_target(target)?;
        let buffer = self.data.dummy.clone();
        self.draw(module, &buffer, 0, 1, 4, Some(source), None, 1)?;
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
            let (blurred, original) = self.backdrop(levels)?;
            let buffer = self.data.blur_rects.clone();
            self.draw(
                ShaderModule::BlurRect,
                &buffer,
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
        let (pipeline, needs_backdrop) = self.pipelines.effect(effect, FORMAT, 1)?;
        let buffer = self.data.effects.clone();
        if !needs_backdrop {
            return self.draw_pipeline(
                &pipeline,
                &buffer,
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
            let (blurred, original) = self.backdrop(levels)?;
            self.draw_pipeline(
                &pipeline,
                &buffer,
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn offscreen(width: u32, height: u32) -> Result<DirectXRenderer> {
        let gpu = Device::new()?;
        Ok(DirectXRenderer {
            hwnd: HWND(ptr::null_mut()),
            atlas: Arc::new(Atlas::new(gpu.clone())),
            pipelines: Pipelines::new(&gpu.device)?,
            gpu,
            swap_chain: None,
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
        });
        scene.insert_primitive(MonochromeSprite {
            order: 0,
            pad: 0,
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
        renderer.render_to_image(&Scene::default(), WindowBackgroundAppearance::Transparent)?;
        assert!(
            renderer.layers.is_empty(),
            "closed isolated groups must release their scratch targets"
        );
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
            params0: [1., 0., 0., 0.],
            params1: [0., 0., 1., 1.],
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
        assert!(
            renderer.blur.is_empty(),
            "zero-radius effects must not retain blur levels"
        );
        assert_eq!(
            image.get_pixel(28, 28).0,
            [0, 255, 0, 255],
            "a later primitive must paint above the effect"
        );
        renderer.render_to_image(&Scene::default(), WindowBackgroundAppearance::Transparent)?;
        assert!(
            renderer.snapshot.is_none() && renderer.blur.is_empty(),
            "closing the backdrop scene must release its scratch textures"
        );
        Ok(())
    }
}
