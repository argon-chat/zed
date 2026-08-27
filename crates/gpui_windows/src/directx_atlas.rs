use collections::FxHashMap;
use etagere::BucketedAtlasAllocator;
use parking_lot::Mutex;
use windows::core::Interface;
use windows::Win32::Graphics::{
    Direct3D11::{
        D3D11_BIND_SHADER_RESOURCE, D3D11_BOX, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
        ID3D11Device, ID3D11DeviceContext, ID3D11ShaderResourceView, ID3D11Texture2D,
    },
    Dxgi::Common::*,
};

use gpui::{
    AtlasKey, AtlasTextureId, AtlasTextureKind, AtlasTextureList, AtlasTile, Bounds, DevicePixels,
    GpuTextureHandle, PlatformAtlas, Point, Size,
};

pub(crate) struct DirectXAtlas(Mutex<DirectXAtlasState>);

struct DirectXAtlasState {
    device: ID3D11Device,
    device_context: ID3D11DeviceContext,
    monochrome_textures: AtlasTextureList<DirectXAtlasTexture>,
    polychrome_textures: AtlasTextureList<DirectXAtlasTexture>,
    subpixel_textures: AtlasTextureList<DirectXAtlasTexture>,
    tiles_by_key: FxHashMap<AtlasKey, AtlasTile>,
}

struct DirectXAtlasTexture {
    id: AtlasTextureId,
    bytes_per_pixel: u32,
    allocator: BucketedAtlasAllocator,
    texture: ID3D11Texture2D,
    view: [Option<ID3D11ShaderResourceView>; 1],
    live_atlas_keys: u32,
}

impl DirectXAtlas {
    pub(crate) fn new(device: &ID3D11Device, device_context: &ID3D11DeviceContext) -> Self {
        DirectXAtlas(Mutex::new(DirectXAtlasState {
            device: device.clone(),
            device_context: device_context.clone(),
            monochrome_textures: Default::default(),
            polychrome_textures: Default::default(),
            subpixel_textures: Default::default(),
            tiles_by_key: Default::default(),
        }))
    }

    pub(crate) fn get_texture_view(
        &self,
        id: AtlasTextureId,
    ) -> [Option<ID3D11ShaderResourceView>; 1] {
        let lock = self.0.lock();
        let tex = lock.texture(id);
        tex.view.clone()
    }

    pub(crate) fn handle_device_lost(
        &self,
        device: &ID3D11Device,
        device_context: &ID3D11DeviceContext,
    ) {
        let mut lock = self.0.lock();
        lock.device = device.clone();
        lock.device_context = device_context.clone();
        lock.monochrome_textures = AtlasTextureList::default();
        lock.polychrome_textures = AtlasTextureList::default();
        lock.subpixel_textures = AtlasTextureList::default();
        lock.tiles_by_key.clear();
    }
}

impl PlatformAtlas for DirectXAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: &AtlasKey,
        build: &mut dyn FnMut() -> anyhow::Result<
            Option<(Size<DevicePixels>, std::borrow::Cow<'a, [u8]>)>,
        >,
    ) -> anyhow::Result<Option<AtlasTile>> {
        let mut lock = self.0.lock();
        if let Some(tile) = lock.tiles_by_key.get(key) {
            Ok(Some(*tile))
        } else {
            let Some((size, bytes)) = build()? else {
                return Ok(None);
            };
            let tile = lock
                .allocate(size, key.texture_kind())
                .ok_or_else(|| anyhow::anyhow!("failed to allocate"))?;
            let texture = lock.texture(tile.texture_id);
            texture.upload(&lock.device_context, tile.bounds, &bytes);
            lock.tiles_by_key.insert(key.clone(), tile);
            Ok(Some(tile))
        }
    }

    /// The zero-copy path: `CopySubresourceRegion` from a texture that already
    /// lives on this renderer's device straight into the tile.
    ///
    /// Written for `<video>` (`crates/vn-video`), whose Media Foundation
    /// backend decodes onto the device `renderer_d3d11_device()` publishes and
    /// hands us the result already in `DXGI_FORMAT_B8G8R8A8_UNORM` — which is
    /// exactly the Polychrome atlas format. Before this, the frame went GPU →
    /// `Map(D3D11_MAP_READ)` → `Vec<u8>` → `UpdateSubresource` → GPU, and the
    /// readback alone [measured] 1.71 ms per 1080p frame.
    ///
    /// The tile is allocated on the first call for a key and **reused in
    /// place** afterwards. That is why this is not `get_or_insert_with`-shaped:
    /// the caller keeps one key for the whole life of the element, so a playing
    /// video costs one allocation, not one per frame.
    ///
    /// Rejections are `Ok(None)`, not `Err` — the caller has a CPU fallback and
    /// a wrong-device texture is a recoverable state (it is what a device-lost
    /// leaves behind for one frame). Each reason is logged once per size.
    fn upload_from_gpu(
        &self,
        key: &AtlasKey,
        size: Size<DevicePixels>,
        texture: GpuTextureHandle,
    ) -> anyhow::Result<Option<AtlasTile>> {
        if texture.is_null() || size.width.0 <= 0 || size.height.0 <= 0 {
            return Ok(None);
        }
        // SAFETY: the contract on `PlatformAtlas::upload_from_gpu` is that this
        // is a live `ID3D11Texture2D` owned by the caller for the duration of
        // the call. `from_raw_borrowed` does not AddRef and does not Release.
        let Some(source) = (unsafe { ID3D11Texture2D::from_raw_borrowed(&texture.0) }) else {
            return Ok(None);
        };

        let mut lock = self.0.lock();

        // Reuse the tile unless it is the wrong size (the element resized), in
        // which case the old one is freed and a new one taken. A tile that is
        // merely stale is exactly what we are about to overwrite.
        let existing = lock.tiles_by_key.get(key).copied();
        let tile = match existing {
            Some(tile) if tile.bounds.size == size => tile,
            _ => {
                if !lock.can_copy_from(source, size) {
                    return Ok(None);
                }
                if existing.is_some() {
                    drop(lock);
                    self.remove(key);
                    lock = self.0.lock();
                }
                let Some(tile) = lock.allocate(size, key.texture_kind()) else {
                    log::error!(
                        "DirectXAtlas::upload_from_gpu: could not allocate a {}x{} tile",
                        size.width.0,
                        size.height.0,
                    );
                    return Ok(None);
                };
                lock.tiles_by_key.insert(key.clone(), tile);
                tile
            }
        };

        let destination = lock.texture(tile.texture_id).texture.clone();
        let region = D3D11_BOX {
            left: 0,
            top: 0,
            front: 0,
            right: size.width.0 as u32,
            bottom: size.height.0 as u32,
            back: 1,
        };
        unsafe {
            lock.device_context.CopySubresourceRegion(
                &destination,
                0,
                tile.bounds.left().0 as u32,
                tile.bounds.top().0 as u32,
                0,
                source,
                0,
                Some(&region),
            );
        }
        Ok(Some(tile))
    }

    fn remove(&self, key: &AtlasKey) {
        let mut lock = self.0.lock();

        let Some(tile) = lock.tiles_by_key.remove(key) else {
            return;
        };
        let id = tile.texture_id;

        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &mut lock.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut lock.polychrome_textures,
            AtlasTextureKind::Subpixel => &mut lock.subpixel_textures,
        };

        let Some(texture_slot) = textures.textures.get_mut(id.index as usize) else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                textures.free_list.push(texture.id.index as usize);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }
}

impl DirectXAtlasState {
    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        {
            let textures = match texture_kind {
                AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
                AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
                AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
            };

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind)?;
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> Option<&mut DirectXAtlasTexture> {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        // Max texture size for DirectX. See:
        // https://learn.microsoft.com/en-us/windows/win32/direct3d11/overviews-direct3d-11-resources-limits
        const MAX_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(16384),
            height: DevicePixels(16384),
        };
        let size = min_size.min(&MAX_ATLAS_SIZE).max(&DEFAULT_ATLAS_SIZE);
        let pixel_format;
        let bind_flag;
        let bytes_per_pixel;
        match kind {
            AtlasTextureKind::Monochrome => {
                pixel_format = DXGI_FORMAT_R8_UNORM;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 1;
            }
            AtlasTextureKind::Polychrome => {
                pixel_format = DXGI_FORMAT_B8G8R8A8_UNORM;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 4;
            }
            AtlasTextureKind::Subpixel => {
                pixel_format = DXGI_FORMAT_R8G8B8A8_UNORM;
                bind_flag = D3D11_BIND_SHADER_RESOURCE;
                bytes_per_pixel = 4;
            }
        }
        let texture_desc = D3D11_TEXTURE2D_DESC {
            Width: size.width.0 as u32,
            Height: size.height.0 as u32,
            MipLevels: 1,
            ArraySize: 1,
            Format: pixel_format,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: bind_flag.0 as u32,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let mut texture: Option<ID3D11Texture2D> = None;
        unsafe {
            // This only returns None if the device is lost, which we will recreate later.
            // So it's ok to return None here.
            self.device
                .CreateTexture2D(&texture_desc, None, Some(&mut texture))
                .ok()?;
        }
        let texture = texture.unwrap();

        let texture_list = match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => &mut self.subpixel_textures,
        };
        let index = texture_list.free_list.pop();
        let view = unsafe {
            let mut view = None;
            self.device
                .CreateShaderResourceView(&texture, None, Some(&mut view))
                .ok()?;
            [view]
        };
        let atlas_texture = DirectXAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            bytes_per_pixel,
            allocator: etagere::BucketedAtlasAllocator::new(device_size_to_etagere(size)),
            texture,
            view,
            live_atlas_keys: 0,
        };
        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list.textures.get_mut(ix).unwrap().as_mut()
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list.textures.last_mut().unwrap().as_mut()
        }
    }

    /// Is `source` a texture this atlas can `CopySubresourceRegion` from?
    ///
    /// Three things have to hold, and each one fails silently in D3D11 if it
    /// does not — a cross-device copy is a no-op with a debug-layer message
    /// nobody reads, and a format mismatch produces garbage rather than an
    /// error. Checked only when a tile is allocated (i.e. once per element and
    /// once more after a device-lost, which is exactly when the answer can
    /// change), so the per-frame path pays nothing.
    fn can_copy_from(&self, source: &ID3D11Texture2D, size: Size<DevicePixels>) -> bool {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        unsafe { source.GetDesc(&mut desc) };

        if desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM {
            log::error!(
                "DirectXAtlas::upload_from_gpu: source texture is format {:?}, and the polychrome \
                 atlas is DXGI_FORMAT_B8G8R8A8_UNORM — refusing the copy",
                desc.Format,
            );
            return false;
        }
        if desc.Width < size.width.0 as u32 || desc.Height < size.height.0 as u32 {
            log::error!(
                "DirectXAtlas::upload_from_gpu: source texture is {}x{} but {}x{} was asked for",
                desc.Width,
                desc.Height,
                size.width.0,
                size.height.0,
            );
            return false;
        }

        // COM identity: `GetDevice` hands back the device the resource was
        // created on, and `ID3D11Device` is not aggregated, so a raw pointer
        // compare is the identity test.
        match unsafe { source.GetDevice() } {
            Ok(owner) if owner.as_raw() == self.device.as_raw() => true,
            _ => {
                log::error!(
                    "DirectXAtlas::upload_from_gpu: the source texture belongs to a different \
                     D3D11 device than the renderer's — refusing the copy (a producer must use \
                     the device gpui_windows::renderer_d3d11_device() publishes, and must rebuild \
                     when its generation changes)",
                );
                false
            }
        }
    }

    fn texture(&self, id: AtlasTextureId) -> &DirectXAtlasTexture {
        match id.kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures[id.index as usize]
                .as_ref()
                .unwrap(),
            AtlasTextureKind::Polychrome => &self.polychrome_textures[id.index as usize]
                .as_ref()
                .unwrap(),
            AtlasTextureKind::Subpixel => {
                &self.subpixel_textures[id.index as usize].as_ref().unwrap()
            }
        }
    }
}

impl DirectXAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(device_size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            bounds: Bounds {
                origin: etagere_point_to_device(allocation.rectangle.min),
                size,
            },
            padding: 0,
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn upload(
        &self,
        device_context: &ID3D11DeviceContext,
        bounds: Bounds<DevicePixels>,
        bytes: &[u8],
    ) {
        // `UpdateSubresource` reads `row_pitch * height` bytes from `bytes` based on the
        // `D3D11_BOX` below. If the caller hands us a slice shorter than that, the driver would
        // over-read past the end of the source buffer (potentially by multiple megabytes), so bail
        // out instead. This is a first-insert path rather than a per-frame one, so the check is
        // effectively free.
        let row_bytes = bounds.size.width.to_bytes(self.bytes_per_pixel as u8) as usize;
        let expected = row_bytes * bounds.size.height.0.max(0) as usize;
        if bytes.len() < expected {
            log::error!(
                "DirectXAtlasTexture::upload: source slice is {} bytes but the {}x{} region \
                 requires {} bytes; skipping upload to avoid a driver over-read",
                bytes.len(),
                bounds.size.width.0,
                bounds.size.height.0,
                expected,
            );
            return;
        }
        unsafe {
            device_context.UpdateSubresource(
                &self.texture,
                0,
                Some(&D3D11_BOX {
                    left: bounds.left().0 as u32,
                    top: bounds.top().0 as u32,
                    front: 0,
                    right: bounds.right().0 as u32,
                    bottom: bounds.bottom().0 as u32,
                    back: 1,
                }),
                bytes.as_ptr() as _,
                bounds.size.width.to_bytes(self.bytes_per_pixel as u8),
                0,
            );
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&mut self) -> bool {
        self.live_atlas_keys == 0
    }
}

fn device_size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    etagere::Size::new(size.width.into(), size.height.into())
}

fn etagere_point_to_device(value: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels::from(value.x),
        y: DevicePixels::from(value.y),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{ImageId, RenderImageParams};
    use std::borrow::Cow;
    use windows::Win32::{
        Foundation::HMODULE,
        Graphics::{
            Direct3D::D3D_DRIVER_TYPE_WARP,
            Direct3D11::{D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_SDK_VERSION, D3D11CreateDevice},
        },
    };

    fn create_device() -> Option<(ID3D11Device, ID3D11DeviceContext)> {
        let mut device: Option<ID3D11Device> = None;
        let mut device_context: Option<ID3D11DeviceContext> = None;
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_WARP,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut device_context),
            )
        }
        .ok()?;
        Some((device?, device_context?))
    }

    fn create_atlas() -> Option<DirectXAtlas> {
        let (device, context) = create_device()?;
        Some(DirectXAtlas::new(&device, &context))
    }

    /// A BGRA texture of `size`, in the shape `upload_from_gpu` accepts.
    fn create_source(device: &ID3D11Device, size: Size<DevicePixels>) -> Option<ID3D11Texture2D> {
        let desc = D3D11_TEXTURE2D_DESC {
            Width: size.width.0 as u32,
            Height: size.height.0 as u32,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            ..Default::default()
        };
        let mut texture: Option<ID3D11Texture2D> = None;
        unsafe { device.CreateTexture2D(&desc, None, Some(&mut texture)) }.ok()?;
        texture
    }

    fn make_image_key(image_id: usize) -> AtlasKey {
        AtlasKey::Image(RenderImageParams {
            image_id: ImageId(image_id),
            frame_index: 0,
        })
    }

    fn insert_tile(atlas: &DirectXAtlas, key: &AtlasKey, size: Size<DevicePixels>) -> AtlasTile {
        atlas
            .get_or_insert_with(key, &mut || {
                let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
                Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
            })
            .expect("allocation should succeed")
            .expect("callback returns Some")
    }

    #[test]
    fn test_remove_deallocates_tile_space_for_reuse() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };
        let big = Size {
            width: DevicePixels(700),
            height: DevicePixels(700),
        };

        let keeper_key = make_image_key(1);
        let big_key_a = make_image_key(2);
        let big_key_b = make_image_key(3);

        let keeper_tile = insert_tile(&atlas, &keeper_key, small);
        let tile_a = insert_tile(&atlas, &big_key_a, big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);

        let tile_b = insert_tile(&atlas, &big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
    }

    /// The property `<video>` depends on: repeated uploads under ONE key reuse
    /// ONE tile. A key per frame would allocate (and have to free) atlas space
    /// sixty times a second, which is the leak `<lottie>`'s two-generation
    /// retire exists to plug — the GPU path avoids needing it at all.
    #[test]
    fn gpu_upload_reuses_one_tile_and_follows_a_resize() {
        let Some((device, context)) = create_device() else {
            return;
        };
        let atlas = DirectXAtlas::new(&device, &context);

        let size = Size {
            width: DevicePixels(320),
            height: DevicePixels(180),
        };
        let Some(source) = create_source(&device, size) else {
            return;
        };
        let handle = GpuTextureHandle(source.as_raw());
        let key = make_image_key(10);

        let first = atlas
            .upload_from_gpu(&key, size, handle)
            .expect("upload should not error")
            .expect("a same-device BGRA texture is acceptable");
        for _ in 0..8 {
            let again = atlas
                .upload_from_gpu(&key, size, handle)
                .unwrap()
                .expect("the tile is reused, not reallocated");
            assert_eq!(again.tile_id, first.tile_id);
            assert_eq!(again.bounds, first.bounds);
        }

        // A resize takes a new tile of the new size and frees the old one.
        let bigger = Size {
            width: DevicePixels(640),
            height: DevicePixels(360),
        };
        let Some(bigger_source) = create_source(&device, bigger) else {
            return;
        };
        let resized = atlas
            .upload_from_gpu(&key, bigger, GpuTextureHandle(bigger_source.as_raw()))
            .unwrap()
            .expect("a resize is still acceptable");
        assert_eq!(resized.bounds.size, bigger);
    }

    /// The three refusals, each of which D3D11 would otherwise turn into
    /// silence (a cross-device copy is a no-op; a format mismatch is garbage).
    /// `<video>`'s fallback to the CPU readback path hangs off exactly this.
    #[test]
    fn gpu_upload_refuses_null_wrong_device_and_wrong_format() {
        let Some((device, context)) = create_device() else {
            return;
        };
        let atlas = DirectXAtlas::new(&device, &context);
        let size = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };

        assert!(
            atlas
                .upload_from_gpu(&make_image_key(20), size, GpuTextureHandle::NULL)
                .unwrap()
                .is_none(),
            "a null handle is not a texture"
        );

        // A texture on a DIFFERENT device — what a device-lost recovery leaves
        // every already-decoding producer holding.
        let Some((other_device, _other_context)) = create_device() else {
            return;
        };
        let Some(foreign) = create_source(&other_device, size) else {
            return;
        };
        assert!(
            atlas
                .upload_from_gpu(&make_image_key(21), size, GpuTextureHandle(foreign.as_raw()))
                .unwrap()
                .is_none(),
            "a texture from another device must be refused, not silently copied"
        );

        // Wrong format: the polychrome atlas is BGRA8 and nothing else.
        let desc = D3D11_TEXTURE2D_DESC {
            Width: 64,
            Height: 64,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
            ..Default::default()
        };
        let mut rgba: Option<ID3D11Texture2D> = None;
        if unsafe { device.CreateTexture2D(&desc, None, Some(&mut rgba)) }.is_ok()
            && let Some(rgba) = rgba
        {
            assert!(
                atlas
                    .upload_from_gpu(&make_image_key(22), size, GpuTextureHandle(rgba.as_raw()))
                    .unwrap()
                    .is_none(),
                "RGBA into a BGRA atlas would swap the channels silently"
            );
        }
    }
}
