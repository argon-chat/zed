use std::{borrow::Cow, sync::Arc};

use anyhow::{Context, Result, ensure};
use etagere::BucketedAtlasAllocator;
use gpui::{
    AtlasBackend, AtlasKey, AtlasState, AtlasStats, AtlasTextureId, AtlasTextureKind,
    AtlasTextureList, AtlasTile, Bounds, D3D12TextureHandle, DevicePixels, GpuTextureHandle,
    PlatformAtlas, Point, Size,
};
use parking_lot::Mutex;
use windows::{
    Win32::Graphics::{
        Direct3D11::{D3D11_TEXTURE2D_DESC, ID3D11Texture2D},
        Direct3D11on12::ID3D11On12Device2,
        Direct3D12::*,
        Dxgi::Common::*,
    },
    core::{IUnknown, Interface},
};

use super::device::{Device, KeepAlive};

pub(super) const ATLAS_STATE: D3D12_RESOURCE_STATES = D3D12_RESOURCE_STATES(
    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE.0,
);

pub(crate) struct Atlas(Mutex<AtlasState<Textures>>);

/// A snapshot owns the resource even if its last tile is removed before the
/// frame is submitted. Renderer submissions retain snapshots through completion.
#[derive(Clone)]
pub(super) struct AtlasTexture {
    pub(super) resource: ID3D12Resource,
    pub(super) format: DXGI_FORMAT,
    pub(super) width: u32,
    pub(super) height: u32,
}

struct Textures {
    device: Arc<Device>,
    monochrome: AtlasTextureList<Page>,
    polychrome: AtlasTextureList<Page>,
    subpixel: AtlasTextureList<Page>,
}

struct Page {
    id: AtlasTextureId,
    texture: AtlasTexture,
    bytes_per_pixel: u32,
    allocator: BucketedAtlasAllocator,
    live_tiles: u32,
}

impl Atlas {
    pub(super) fn new(device: Arc<Device>) -> Self {
        Self(Mutex::new(AtlasState::new(Textures::new(device))))
    }

    pub(super) fn get_texture(&self, id: AtlasTextureId) -> Option<AtlasTexture> {
        self.0
            .lock()
            .backend
            .texture(id)
            .map(|page| page.texture.clone())
    }

    pub(super) fn handle_device_lost(&self, device: Arc<Device>) {
        self.0
            .lock()
            .clear(|textures| *textures = Textures::new(device));
    }

    fn upload_d3d12(
        &self,
        key: &AtlasKey,
        size: Size<DevicePixels>,
        resource: &ID3D12Resource,
        source_state: D3D12_RESOURCE_STATES,
        ready: Option<(&ID3D12Fence, u64)>,
        owner: KeepAlive,
    ) -> Result<Option<AtlasTile>> {
        if size.width.0 <= 0
            || size.height.0 <= 0
            || key.texture_kind() != AtlasTextureKind::Polychrome
        {
            return Ok(None);
        }
        let descriptor = unsafe { resource.GetDesc() };
        if descriptor.Dimension != D3D12_RESOURCE_DIMENSION_TEXTURE2D
            || descriptor.Format != DXGI_FORMAT_B8G8R8A8_UNORM
            || descriptor.SampleDesc.Count != 1
            || descriptor.Width < size.width.0 as u64
            || descriptor.Height < size.height.0 as u32
        {
            return Ok(None);
        }
        let mut state = self.0.lock();
        // A producer can temporarily retain the preceding device generation.
        // Refuse it before allocating a destination tile so callers may rebuild.
        if state
            .backend
            .device
            .validate_resource_device(resource)
            .is_err()
        {
            return Ok(None);
        }
        let existing = state.tile(key);
        let (tile, fresh) = state
            .backend
            .tile_for_write(existing, size, key.texture_kind())?;
        let page = state
            .backend
            .texture(tile.texture_id)
            .context("Allocated atlas page disappeared")?;
        let copy = state.backend.device.copy_texture(
            resource,
            source_state,
            &page.texture.resource,
            (tile.bounds.left().0 as u32, tile.bounds.top().0 as u32),
            (size.width.0 as u32, size.height.0 as u32),
            ATLAS_STATE,
            owner,
            ready,
        );
        if let Err(error) = copy {
            if fresh {
                state.backend.remove(tile);
            }
            return Err(error);
        }
        if fresh {
            state.remove(key);
            state.insert_tile(key.clone(), tile);
        }
        Ok(Some(tile))
    }
}

impl PlatformAtlas for Atlas {
    fn get_or_insert_with<'a>(
        &self,
        key: AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        self.0.lock().get_or_insert_with(key, build)
    }

    fn upload_from_d3d12(
        &self,
        key: &AtlasKey,
        size: Size<DevicePixels>,
        texture: D3D12TextureHandle,
        owner: KeepAlive,
    ) -> Result<Option<AtlasTile>> {
        // The platform contract guarantees live COM pointers owned by `owner`.
        // The helper clones both interfaces before returning to the producer.
        let Some(resource) = (unsafe { ID3D12Resource::from_raw_borrowed(&texture.resource) })
        else {
            return Ok(None);
        };
        let Some(fence) = (unsafe { ID3D12Fence::from_raw_borrowed(&texture.ready_fence) }) else {
            return Ok(None);
        };
        self.upload_d3d12(
            key,
            size,
            resource,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            Some((fence, texture.ready_value)),
            owner,
        )
    }

    fn upload_from_gpu(
        &self,
        key: &AtlasKey,
        size: Size<DevicePixels>,
        texture: GpuTextureHandle,
    ) -> Result<Option<AtlasTile>> {
        let Some(source) = (unsafe { ID3D11Texture2D::from_raw_borrowed(&texture.0) }) else {
            return Ok(None);
        };
        let mut descriptor = D3D11_TEXTURE2D_DESC::default();
        unsafe {
            source.GetDesc(&mut descriptor);
        }
        if size.width.0 <= 0
            || size.height.0 <= 0
            || descriptor.Format != DXGI_FORMAT_B8G8R8A8_UNORM
            || descriptor.SampleDesc.Count != 1
            || descriptor.Width < size.width.0 as u32
            || descriptor.Height < size.height.0 as u32
        {
            return Ok(None);
        }
        let producer = unsafe { source.GetDevice()? };
        let Ok(on12) = producer.cast::<ID3D11On12Device2>() else {
            return Ok(None);
        };
        let producer_native: ID3D12Device = unsafe { on12.GetD3D12Device()? };
        let native = self.0.lock().backend.device.clone();
        if producer_native.cast::<IUnknown>()? != native.device.cast::<IUnknown>()? {
            return Ok(None);
        }
        let Some(bridge) = native.existing_video_bridge() else {
            return Ok(None);
        };
        if producer.cast::<IUnknown>()? != bridge.device.cast::<IUnknown>()? {
            return Ok(None);
        }
        // Checkout and check-in are one protected D3D11 context transaction.
        // Return supplies the copy fence, so the next decoder write waits on
        // the GPU instead of racing the atlas read or blocking the UI thread.
        let _guard = super::video::ContextGuard::lock(&producer)?;
        let context = unsafe { producer.GetImmediateContext()? };
        let resource: ID3D12Resource =
            unsafe { on12.UnwrapUnderlyingResource(source, &native.queue)? };
        unsafe {
            context.Flush();
        }
        let owner: KeepAlive = Arc::new((source.clone(), on12.clone()));
        let copied = self.upload_d3d12(
            key,
            size,
            &resource,
            D3D12_RESOURCE_STATE_COMMON,
            None,
            owner,
        );
        let (fence, value) = native.completion_point();
        let fences = [Some(fence)];
        let returned = unsafe { on12.ReturnUnderlyingResource(source, 1, &value, fences.as_ptr()) };
        returned.context("Returning video texture to D3D11On12")?;
        copied
    }

    fn remove(&self, key: &AtlasKey) {
        self.0.lock().remove(key);
    }

    fn tile(&self, key: &AtlasKey) -> Option<AtlasTile> {
        self.0.lock().tile(key)
    }

    fn write_bytes(
        &self,
        key: &AtlasKey,
        size: Size<DevicePixels>,
        bytes: &[u8],
    ) -> Result<Option<AtlasTile>> {
        if size.width.0 <= 0 || size.height.0 <= 0 {
            return Ok(None);
        }
        let kind = key.texture_kind();
        validate_bytes(kind, size, bytes)?;
        let mut state = self.0.lock();
        let existing = state.tile(key);
        let (tile, fresh) = state.backend.tile_for_write(existing, size, kind)?;
        if let Err(error) = state.backend.upload(tile, bytes) {
            if fresh {
                state.backend.remove(tile);
            }
            return Err(error);
        }
        if fresh {
            state.remove(key);
            state.insert_tile(key.clone(), tile);
        }
        Ok(Some(tile))
    }

    #[cfg(feature = "test-support")]
    fn contains(&self, key: &AtlasKey) -> bool {
        self.0.lock().contains(key)
    }

    fn stats(&self) -> Option<AtlasStats> {
        let state = self.0.lock();
        let mut stats = AtlasStats::default();
        state.count_keys(&mut stats);
        for kind in [
            AtlasTextureKind::Monochrome,
            AtlasTextureKind::Polychrome,
            AtlasTextureKind::Subpixel,
        ] {
            let output = &mut stats.kinds[kind as usize];
            for page in state.backend.list(kind).textures.iter().flatten() {
                output.pages += 1;
                output.bytes += u64::from(page.texture.width)
                    * u64::from(page.texture.height)
                    * u64::from(page.bytes_per_pixel);
                output.tiles += page.live_tiles;
            }
        }
        Some(stats)
    }
}

impl AtlasBackend for Textures {
    fn insert(
        &mut self,
        kind: AtlasTextureKind,
        size: Size<DevicePixels>,
        bytes: &[u8],
    ) -> Result<AtlasTile> {
        validate_bytes(kind, size, bytes)?;
        let tile = self.allocate(size, kind)?;
        if let Err(error) = self.upload(tile, bytes) {
            self.remove(tile);
            return Err(error);
        }
        Ok(tile)
    }

    fn remove(&mut self, tile: AtlasTile) {
        let list = self.list_mut(tile.texture_id.kind);
        let Some(slot) = list.textures.get_mut(tile.texture_id.index as usize) else {
            return;
        };
        let Some(page) = slot.as_mut() else {
            return;
        };
        page.allocator.deallocate(tile.tile_id.into());
        page.live_tiles -= 1;
        if page.live_tiles == 0 {
            *slot = None;
            list.free_list.push(tile.texture_id.index as usize);
        }
    }
}

impl Textures {
    fn new(device: Arc<Device>) -> Self {
        Self {
            device,
            monochrome: Default::default(),
            polychrome: Default::default(),
            subpixel: Default::default(),
        }
    }

    fn list(&self, kind: AtlasTextureKind) -> &AtlasTextureList<Page> {
        match kind {
            AtlasTextureKind::Monochrome => &self.monochrome,
            AtlasTextureKind::Polychrome => &self.polychrome,
            AtlasTextureKind::Subpixel => &self.subpixel,
        }
    }

    fn list_mut(&mut self, kind: AtlasTextureKind) -> &mut AtlasTextureList<Page> {
        match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome,
            AtlasTextureKind::Polychrome => &mut self.polychrome,
            AtlasTextureKind::Subpixel => &mut self.subpixel,
        }
    }

    fn texture(&self, id: AtlasTextureId) -> Option<&Page> {
        self.list(id.kind).textures.get(id.index as usize)?.as_ref()
    }

    fn tile_for_write(
        &mut self,
        existing: Option<AtlasTile>,
        size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> Result<(AtlasTile, bool)> {
        if let Some(tile) = existing.filter(|tile| tile.bounds.size == size) {
            return Ok((tile, false));
        }
        Ok((self.allocate(size, kind)?, true))
    }

    fn allocate(&mut self, size: Size<DevicePixels>, kind: AtlasTextureKind) -> Result<AtlasTile> {
        ensure!(
            size.width.0 > 0
                && size.height.0 > 0
                && size.width.0 <= 16384
                && size.height.0 <= 16384,
            "Atlas tile dimensions must be between 1 and 16384"
        );
        if let Some(tile) = self
            .list_mut(kind)
            .iter_mut()
            .rev()
            .find_map(|page| page.allocate(size))
        {
            return Ok(tile);
        }
        let width = size.width.0.max(1024) as u32;
        let height = size.height.0.max(1024) as u32;
        let (format, bytes_per_pixel) = pixel_layout(kind);
        let resource = self.device.create_texture(
            width,
            height,
            format,
            D3D12_RESOURCE_FLAG_NONE,
            ATLAS_STATE,
        )?;
        let list = self.list_mut(kind);
        let index = list.free_list.pop().unwrap_or(list.textures.len());
        let mut page = Page {
            id: AtlasTextureId {
                index: u32::try_from(index).context("Too many atlas pages")?,
                kind,
            },
            texture: AtlasTexture {
                resource,
                format,
                width,
                height,
            },
            bytes_per_pixel,
            allocator: BucketedAtlasAllocator::new(etagere::size2(width as i32, height as i32)),
            live_tiles: 0,
        };
        let tile = page
            .allocate(size)
            .context("New atlas page cannot fit requested tile")?;
        if index == list.textures.len() {
            list.textures.push(Some(page));
        } else {
            list.textures[index] = Some(page);
        }
        Ok(tile)
    }

    fn upload(&self, tile: AtlasTile, bytes: &[u8]) -> Result<()> {
        let page = self
            .texture(tile.texture_id)
            .context("Atlas tile refers to a missing page")?;
        self.device.upload_texture(
            &page.texture.resource,
            (tile.bounds.left().0 as u32, tile.bounds.top().0 as u32),
            (
                tile.bounds.size.width.0 as u32,
                tile.bounds.size.height.0 as u32,
            ),
            page.bytes_per_pixel,
            bytes,
            ATLAS_STATE,
        )?;
        Ok(())
    }
}

impl Page {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self
            .allocator
            .allocate(etagere::size2(size.width.0, size.height.0))?;
        self.live_tiles += 1;
        Some(AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            bounds: Bounds {
                origin: Point {
                    x: DevicePixels(allocation.rectangle.min.x),
                    y: DevicePixels(allocation.rectangle.min.y),
                },
                size,
            },
            padding: 0,
        })
    }
}

fn pixel_layout(kind: AtlasTextureKind) -> (DXGI_FORMAT, u32) {
    match kind {
        AtlasTextureKind::Monochrome => (DXGI_FORMAT_R8_UNORM, 1),
        AtlasTextureKind::Polychrome => (DXGI_FORMAT_B8G8R8A8_UNORM, 4),
        AtlasTextureKind::Subpixel => (DXGI_FORMAT_R8G8B8A8_UNORM, 4),
    }
}

fn validate_bytes(kind: AtlasTextureKind, size: Size<DevicePixels>, bytes: &[u8]) -> Result<()> {
    ensure!(
        size.width.0 > 0 && size.height.0 > 0,
        "Atlas tile must not be empty"
    );
    let required = u64::try_from(size.width.0)?
        * u64::try_from(size.height.0)?
        * u64::from(pixel_layout(kind).1);
    ensure!(
        required <= bytes.len() as u64,
        "Atlas upload requires {required} bytes; only {} provided",
        bytes.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{ImageId, RenderImageParams};
    use windows::Win32::Graphics::Direct3D11::{
        D3D11_BIND_RENDER_TARGET, D3D11_BIND_SHADER_RESOURCE, D3D11_USAGE_DEFAULT,
    };

    fn key() -> AtlasKey {
        AtlasKey::Image(RenderImageParams {
            image_id: ImageId(734),
            frame_index: 0,
        })
    }

    #[test]
    #[ignore = "requires a native D3D12 hardware adapter; run explicitly on Windows"]
    fn native_dx12_atlas_reuses_tile_and_rejects_short_upload_without_mutation() -> Result<()> {
        let device = Device::new()?;
        let atlas = Atlas::new(device.clone());
        let key = key();
        let size = Size {
            width: DevicePixels(5),
            height: DevicePixels(3),
        };
        let first = atlas
            .write_bytes(&key, size, &[17; 60])?
            .context("Native atlas refused CPU bytes")?;
        let second = atlas
            .write_bytes(&key, size, &[99; 60])?
            .context("Native atlas refused the second write")?;
        assert_eq!(first.texture_id, second.texture_id);
        assert_eq!(first.tile_id, second.tile_id);
        assert_eq!(first.bounds, second.bounds);
        let resized = Size {
            width: DevicePixels(7),
            height: DevicePixels(4),
        };
        assert!(atlas.write_bytes(&key, resized, &[0; 4]).is_err());
        assert_eq!(
            atlas.tile(&key).map(|tile| tile.bounds),
            Some(second.bounds)
        );
        let texture = atlas
            .get_texture(second.texture_id)
            .context("Atlas page missing")?;
        let pixels = device.read_texture_for_test(
            &texture.resource,
            (second.bounds.left().0 as u32, second.bounds.top().0 as u32),
            (5, 3),
            4,
            ATLAS_STATE,
        )?;
        assert_eq!(pixels, vec![99; 60]);
        let stats = atlas.stats().context("Atlas has no statistics")?;
        assert_eq!(stats.keys, 1);
        assert_eq!(stats.kinds[AtlasTextureKind::Polychrome as usize].tiles, 1);
        assert!(
            device.existing_video_bridge().is_none(),
            "CPU atlas writes created a video bridge"
        );
        atlas.remove(&key);
        let stats = atlas.stats().context("Atlas has no statistics")?;
        assert_eq!(stats.keys, 0);
        assert_eq!(stats.kinds[AtlasTextureKind::Polychrome as usize].pages, 0);
        Ok(())
    }

    #[test]
    #[ignore = "requires native D3D12 and D3D11On12 video support; run explicitly on Windows"]
    fn native_dx12_video_bridge_returns_source_before_decoder_reuses_it() -> Result<()> {
        let device = Device::new()?;
        assert!(device.existing_video_bridge().is_none());
        let bridge = device.video_bridge()?;
        let context = unsafe { bridge.device.GetImmediateContext()? };
        let mut source: Option<ID3D11Texture2D> = None;
        unsafe {
            bridge.device.CreateTexture2D(
                &D3D11_TEXTURE2D_DESC {
                    Width: 5,
                    Height: 3,
                    MipLevels: 1,
                    ArraySize: 1,
                    Format: DXGI_FORMAT_B8G8R8A8_UNORM,
                    SampleDesc: DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Usage: D3D11_USAGE_DEFAULT,
                    BindFlags: (D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE).0 as u32,
                    ..Default::default()
                },
                None,
                Some(&mut source),
            )?;
        }
        let source = source.context("D3D11On12 returned no source texture")?;
        let atlas = Atlas::new(device.clone());
        let key = key();
        let size = Size {
            width: DevicePixels(5),
            height: DevicePixels(3),
        };
        let mut first = None;
        // No CPU wait between decoder writes: ReturnUnderlyingResource must
        // make the second write wait for the preceding native atlas copy.
        for value in [37u8, 193] {
            let pixels = vec![value; 60];
            unsafe {
                context.UpdateSubresource(&source, 0, None, pixels.as_ptr().cast(), 20, 60);
            }
            let tile = atlas
                .upload_from_gpu(&key, size, GpuTextureHandle(source.as_raw()))?
                .context("Video GPU upload refused")?;
            if let Some(first) = first {
                assert_eq!(tile.bounds, first);
            } else {
                first = Some(tile.bounds);
            }
        }
        let tile = atlas.tile(&key).context("Video atlas tile missing")?;
        let texture = atlas
            .get_texture(tile.texture_id)
            .context("Video atlas page missing")?;
        let result = device.read_texture_for_test(
            &texture.resource,
            (tile.bounds.left().0 as u32, tile.bounds.top().0 as u32),
            (5, 3),
            4,
            ATLAS_STATE,
        )?;
        assert_eq!(result, vec![193; 60]);
        drop(source);
        drop(context);
        let weak = Arc::downgrade(&bridge);
        drop(bridge);
        assert!(
            weak.upgrade().is_none(),
            "renderer cache retained the last video bridge"
        );
        assert!(
            device.existing_video_bridge().is_none(),
            "closed video recreated its bridge"
        );
        Ok(())
    }
}
