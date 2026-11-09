use anyhow::{Context as _, Result};
use derive_more::{Deref, DerefMut};
use etagere::BucketedAtlasAllocator;
use gpui::{
    AtlasBackend, AtlasKey, AtlasState, AtlasStats, AtlasTextureId, AtlasTextureKind,
    AtlasTextureList, AtlasTile, Bounds, DevicePixels, PlatformAtlas, Point, Size,
};
use metal::Device;
use parking_lot::Mutex;
use std::borrow::Cow;

pub struct MetalAtlas(Mutex<AtlasState<MetalAtlasTextures>>);

impl MetalAtlas {
    pub(crate) fn new(
        device: Device,
        command_queue: metal::CommandQueue,
        supports_shared_storage: bool,
    ) -> Self {
        MetalAtlas(Mutex::new(AtlasState::new(MetalAtlasTextures {
            device: AssertSend(device),
            command_queue: AssertSend(command_queue),
            supports_shared_storage,
            monochrome_textures: Default::default(),
            polychrome_textures: Default::default(),
        })))
    }

    /// Returns the GPU texture backing `id`, or `None` once every tile in it
    /// has been removed. A scene can still reference such a texture when a
    /// cached view replays a paint from before the image was dropped, so
    /// callers must skip those sprites rather than assume the texture exists.
    pub(crate) fn metal_texture(&self, id: AtlasTextureId) -> Option<metal::Texture> {
        Some(self.0.lock().backend.texture(id)?.metal_texture.clone())
    }
}

struct MetalAtlasTextures {
    device: AssertSend<Device>,
    command_queue: AssertSend<metal::CommandQueue>,
    supports_shared_storage: bool,
    monochrome_textures: AtlasTextureList<MetalAtlasTexture>,
    polychrome_textures: AtlasTextureList<MetalAtlasTexture>,
}

impl PlatformAtlas for MetalAtlas {
    fn get_or_insert_with<'a>(
        &self,
        key: AtlasKey,
        build: &mut dyn FnMut() -> Result<Option<(Size<DevicePixels>, Cow<'a, [u8]>)>>,
    ) -> Result<Option<AtlasTile>> {
        self.0.lock().get_or_insert_with(key, build)
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
        let Some(layout) = upload_layout(key.texture_kind(), size, bytes.len()) else {
            return Ok(None);
        };
        let mut state = self.0.lock();
        let existing = state.tile(key);
        let tile = match existing {
            Some(tile) if tile.bounds.size == size => tile,
            _ => {
                if existing.is_some() {
                    state.remove(key);
                }
                let Some(tile) = state.backend.allocate(size, key.texture_kind()) else {
                    return Ok(None);
                };
                state.insert_tile(key.clone(), tile);
                tile
            }
        };
        let Some(texture) = state.backend.texture(tile.texture_id) else {
            return Ok(None);
        };
        // Callers write before encoding the next scene. Uploads use the SAME
        // queue as that renderer, ordering prior reads -> blit -> next draw.
        // replace_region would instead race CPU writes against in-flight GPU
        // reads of this live tile on shared/managed atlas pages.
        texture.upload_queued(
            &state.backend.device,
            &state.backend.command_queue,
            tile.bounds,
            bytes,
            layout,
        );
        Ok(Some(tile))
    }

    #[cfg(any(test, feature = "test-support", feature = "bench-support"))]
    fn contains(&self, key: &AtlasKey) -> bool {
        self.0.lock().contains(key)
    }

    fn stats(&self) -> Option<AtlasStats> {
        let state = self.0.lock();
        let mut stats = AtlasStats::default();
        state.count_keys(&mut stats);
        for (kind, list) in [
            (
                AtlasTextureKind::Monochrome,
                &state.backend.monochrome_textures,
            ),
            (
                AtlasTextureKind::Polychrome,
                &state.backend.polychrome_textures,
            ),
        ] {
            let out = &mut stats.kinds[kind as usize];
            for texture in list.textures.iter().flatten() {
                let size = texture.allocator.size();
                out.pages += 1;
                out.bytes += (size.width.max(0) as u64)
                    * (size.height.max(0) as u64)
                    * u64::from(texture.bytes_per_pixel());
                out.tiles += texture.live_atlas_keys;
            }
        }
        Some(stats)
    }
}

impl AtlasBackend for MetalAtlasTextures {
    fn insert(
        &mut self,
        kind: AtlasTextureKind,
        size: Size<DevicePixels>,
        bytes: &[u8],
    ) -> Result<AtlasTile> {
        let tile = self.allocate(size, kind).context("failed to allocate")?;
        let texture = self
            .texture(tile.texture_id)
            .context("allocated tile refers to a missing texture")?;
        texture.upload(tile.bounds, bytes);
        Ok(tile)
    }

    fn remove(&mut self, tile: AtlasTile) {
        let id = tile.texture_id;

        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        };

        let Some(texture_slot) = textures
            .textures
            .iter_mut()
            .find(|texture| texture.as_ref().is_some_and(|v| v.id == id))
        else {
            return;
        };

        if let Some(mut texture) = texture_slot.take() {
            texture.allocator.deallocate(tile.tile_id.into());
            texture.decrement_ref_count();
            if texture.is_unreferenced() {
                textures.free_list.push(id.index as usize);
            } else {
                *texture_slot = Some(texture);
            }
        }
    }
}

impl MetalAtlasTextures {
    fn allocate(
        &mut self,
        size: Size<DevicePixels>,
        texture_kind: AtlasTextureKind,
    ) -> Option<AtlasTile> {
        {
            let textures = match texture_kind {
                AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
                AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
                AtlasTextureKind::Subpixel => unreachable!(),
            };

            if let Some(tile) = textures
                .iter_mut()
                .rev()
                .find_map(|texture| texture.allocate(size))
            {
                return Some(tile);
            }
        }

        let texture = self.push_texture(size, texture_kind);
        texture.allocate(size)
    }

    fn push_texture(
        &mut self,
        min_size: Size<DevicePixels>,
        kind: AtlasTextureKind,
    ) -> &mut MetalAtlasTexture {
        const DEFAULT_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(1024),
            height: DevicePixels(1024),
        };
        // Max texture size on all modern Apple GPUs. Anything bigger than that crashes in validateWithDevice.
        const MAX_ATLAS_SIZE: Size<DevicePixels> = Size {
            width: DevicePixels(16384),
            height: DevicePixels(16384),
        };
        let size = min_size.min(&MAX_ATLAS_SIZE).max(&DEFAULT_ATLAS_SIZE);
        let texture_descriptor = metal::TextureDescriptor::new();
        texture_descriptor.set_width(size.width.into());
        texture_descriptor.set_height(size.height.into());
        let pixel_format;
        let usage;
        match kind {
            AtlasTextureKind::Monochrome => {
                pixel_format = metal::MTLPixelFormat::A8Unorm;
                usage = metal::MTLTextureUsage::ShaderRead;
            }
            AtlasTextureKind::Polychrome => {
                pixel_format = metal::MTLPixelFormat::BGRA8Unorm;
                usage = metal::MTLTextureUsage::ShaderRead;
            }
            AtlasTextureKind::Subpixel => unreachable!(),
        }
        texture_descriptor.set_pixel_format(pixel_format);
        texture_descriptor.set_usage(usage);
        // Shared memory mode can be used only on Apple GPU families
        // https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodeshared
        texture_descriptor.set_storage_mode(if self.supports_shared_storage {
            metal::MTLStorageMode::Shared
        } else {
            metal::MTLStorageMode::Managed
        });
        let metal_texture = self.device.new_texture(&texture_descriptor);

        let texture_list = match kind {
            AtlasTextureKind::Monochrome => &mut self.monochrome_textures,
            AtlasTextureKind::Polychrome => &mut self.polychrome_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        };

        let index = texture_list.free_list.pop();

        let atlas_texture = MetalAtlasTexture {
            id: AtlasTextureId {
                index: index.unwrap_or(texture_list.textures.len()) as u32,
                kind,
            },
            allocator: etagere::BucketedAtlasAllocator::new(size_to_etagere(size)),
            metal_texture: AssertSend(metal_texture),
            live_atlas_keys: 0,
        };

        if let Some(ix) = index {
            texture_list.textures[ix] = Some(atlas_texture);
            texture_list.textures.get_mut(ix)
        } else {
            texture_list.textures.push(Some(atlas_texture));
            texture_list.textures.last_mut()
        }
        .unwrap()
        .as_mut()
        .unwrap()
    }

    fn texture(&self, id: AtlasTextureId) -> Option<&MetalAtlasTexture> {
        let textures = match id.kind {
            AtlasTextureKind::Monochrome => &self.monochrome_textures,
            AtlasTextureKind::Polychrome => &self.polychrome_textures,
            AtlasTextureKind::Subpixel => unreachable!(),
        };
        textures.textures.get(id.index as usize)?.as_ref()
    }
}

struct MetalAtlasTexture {
    id: AtlasTextureId,
    allocator: BucketedAtlasAllocator,
    metal_texture: AssertSend<metal::Texture>,
    live_atlas_keys: u32,
}

impl MetalAtlasTexture {
    fn allocate(&mut self, size: Size<DevicePixels>) -> Option<AtlasTile> {
        let allocation = self.allocator.allocate(size_to_etagere(size))?;
        let tile = AtlasTile {
            texture_id: self.id,
            tile_id: allocation.id.into(),
            bounds: Bounds {
                origin: point_from_etagere(allocation.rectangle.min),
                size,
            },
            padding: 0,
        };
        self.live_atlas_keys += 1;
        Some(tile)
    }

    fn upload(&self, bounds: Bounds<DevicePixels>, bytes: &[u8]) {
        let region = metal::MTLRegion::new_2d(
            bounds.origin.x.into(),
            bounds.origin.y.into(),
            bounds.size.width.into(),
            bounds.size.height.into(),
        );
        self.metal_texture.replace_region(
            region,
            0,
            bytes.as_ptr() as *const _,
            bounds.size.width.to_bytes(self.bytes_per_pixel()) as u64,
        );
    }

    fn upload_queued(
        &self,
        device: &metal::DeviceRef,
        queue: &metal::CommandQueueRef,
        bounds: Bounds<DevicePixels>,
        bytes: &[u8],
        layout: UploadLayout,
    ) {
        let mut padded = vec![0; layout.upload_len];
        for (input, output) in bytes[..layout.input_len]
            .chunks_exact(layout.row_bytes)
            .zip(padded.chunks_exact_mut(layout.row_stride))
        {
            output[..layout.row_bytes].copy_from_slice(input);
        }
        let staging = device.new_buffer_with_data(
            padded.as_ptr().cast(),
            padded.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        // commandBuffer (unlike commandBufferWithUnretainedReferences) retains
        // both staging and destination resources through GPU completion.
        let command = queue.new_command_buffer();
        let encoder = command.new_blit_command_encoder();
        encoder.copy_from_buffer_to_texture(
            &staging,
            0,
            layout.row_stride as u64,
            layout.upload_len as u64,
            metal::MTLSize::new(bounds.size.width.0 as u64, bounds.size.height.0 as u64, 1),
            &self.metal_texture,
            0,
            0,
            metal::MTLOrigin {
                x: bounds.origin.x.0 as u64,
                y: bounds.origin.y.0 as u64,
                z: 0,
            },
            metal::MTLBlitOption::None,
        );
        encoder.end_encoding();
        command.commit();
    }

    fn bytes_per_pixel(&self) -> u8 {
        use metal::MTLPixelFormat::*;
        match self.metal_texture.pixel_format() {
            A8Unorm | R8Unorm => 1,
            RGBA8Unorm | BGRA8Unorm => 4,
            _ => unimplemented!(),
        }
    }

    fn decrement_ref_count(&mut self) {
        self.live_atlas_keys -= 1;
    }

    fn is_unreferenced(&mut self) -> bool {
        self.live_atlas_keys == 0
    }
}

#[derive(Clone, Copy, Debug)]
struct UploadLayout {
    row_bytes: usize,
    row_stride: usize,
    input_len: usize,
    upload_len: usize,
}

fn upload_layout(
    kind: AtlasTextureKind,
    size: Size<DevicePixels>,
    bytes_len: usize,
) -> Option<UploadLayout> {
    // Mirrors the maximum page size; reject before mutating the old mapping.
    if size.width.0 <= 0 || size.height.0 <= 0 || size.width.0 > 16384 || size.height.0 > 16384 {
        return None;
    }
    let pixel_bytes = match kind {
        AtlasTextureKind::Monochrome => 1,
        AtlasTextureKind::Polychrome => 4,
        AtlasTextureKind::Subpixel => return None,
    };
    let row_bytes = (size.width.0 as usize).checked_mul(pixel_bytes)?;
    let input_len = row_bytes.checked_mul(size.height.0 as usize)?;
    if bytes_len < input_len {
        return None;
    }
    // 256-byte rows satisfy Metal's buffer-to-texture blit alignment on Macs,
    // including narrow monochrome glyphs and non-aligned BGRA image widths.
    let row_stride = row_bytes.checked_add(255)? & !255;
    let upload_len = row_stride.checked_mul(size.height.0 as usize)?;
    Some(UploadLayout {
        row_bytes,
        row_stride,
        input_len,
        upload_len,
    })
}

fn size_to_etagere(size: Size<DevicePixels>) -> etagere::Size {
    etagere::Size::new(size.width.into(), size.height.into())
}

fn point_from_etagere(value: etagere::Point) -> Point<DevicePixels> {
    Point {
        x: DevicePixels::from(value.x),
        y: DevicePixels::from(value.y),
    }
}

#[derive(Deref, DerefMut)]
struct AssertSend<T>(T);

unsafe impl<T> Send for AssertSend<T> {}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::PlatformAtlas;
    use std::borrow::Cow;

    fn create_atlas() -> Option<MetalAtlas> {
        let device = metal::Device::system_default()?;
        let queue = device.new_command_queue();
        let shared = device.supports_family(metal::MTLGPUFamily::Apple1);
        Some(MetalAtlas::new(device, queue, shared))
    }

    fn make_image_key(image_id: usize, frame_index: usize) -> AtlasKey {
        AtlasKey::Image(gpui::RenderImageParams {
            image_id: gpui::ImageId(image_id),
            frame_index,
        })
    }

    fn insert_tile(atlas: &MetalAtlas, key: AtlasKey, size: Size<DevicePixels>) -> AtlasTile {
        atlas
            .get_or_insert_with(key, &mut || {
                let byte_count = (size.width.0 as usize) * (size.height.0 as usize) * 4;
                Ok(Some((size, Cow::Owned(vec![0u8; byte_count]))))
            })
            .expect("allocation should succeed")
            .expect("callback returns Some")
    }

    #[test]
    fn test_remove_clears_stale_keys_from_tiles_by_key() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let small = Size {
            width: DevicePixels(64),
            height: DevicePixels(64),
        };

        let key_a = make_image_key(1, 0);
        let key_b = make_image_key(2, 0);
        let key_c = make_image_key(3, 0);

        let tile_a = insert_tile(&atlas, key_a.clone(), small);
        let tile_b = insert_tile(&atlas, key_b.clone(), small);
        let tile_c = insert_tile(&atlas, key_c.clone(), small);

        assert_eq!(tile_a.texture_id, tile_b.texture_id);
        assert_eq!(tile_b.texture_id, tile_c.texture_id);

        // Remove A: texture still has B and C, so it stays.
        // The key for A must be removed from tiles_by_key.
        atlas.remove(&key_a);

        // Remove B: texture still has C.
        atlas.remove(&key_b);

        // Remove C: texture becomes unreferenced and is deleted.
        atlas.remove(&key_c);

        // Re-inserting A must allocate a fresh tile on a new texture,
        // NOT return a stale tile referencing the deleted texture.
        let tile_a2 = insert_tile(&atlas, key_a, small);

        // The texture must actually exist — this would panic before the fix.
        assert!(atlas.metal_texture(tile_a2.texture_id).is_some());
    }

    #[test]
    fn test_metal_texture_is_none_after_last_tile_removed() {
        let Some(atlas) = create_atlas() else {
            return;
        };

        let key = make_image_key(1, 0);
        let tile = insert_tile(
            &atlas,
            key.clone(),
            Size {
                width: DevicePixels(64),
                height: DevicePixels(64),
            },
        );
        assert!(atlas.metal_texture(tile.texture_id).is_some());

        // A scene built before the removal may still carry `tile`; looking its
        // texture up must report the gap instead of panicking.
        atlas.remove(&key);
        assert!(atlas.metal_texture(tile.texture_id).is_none());
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

        let keeper_key = make_image_key(1, 0);
        let big_key_a = make_image_key(2, 0);
        let big_key_b = make_image_key(3, 0);

        let keeper_tile = insert_tile(&atlas, keeper_key, small);
        let tile_a = insert_tile(&atlas, big_key_a.clone(), big);
        assert_eq!(keeper_tile.texture_id, tile_a.texture_id);

        atlas.remove(&big_key_a);
        let tile_b = insert_tile(&atlas, big_key_b, big);
        assert_eq!(tile_b.texture_id, keeper_tile.texture_id);
    }

    #[test]
    fn test_remove_nonexistent_key_is_noop() {
        let Some(atlas) = create_atlas() else {
            return;
        };
        let key = make_image_key(999, 0);
        atlas.remove(&key);
    }
    fn snapshot(atlas: &MetalAtlas, tile: AtlasTile) -> metal::Buffer {
        let state = atlas.0.lock();
        let texture = state.backend.texture(tile.texture_id).unwrap();
        let layout = upload_layout(tile.texture_id.kind, tile.bounds.size, usize::MAX).unwrap();
        let buffer = state.backend.device.new_buffer(
            layout.upload_len as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let command = state.backend.command_queue.new_command_buffer();
        let encoder = command.new_blit_command_encoder();
        encoder.copy_from_texture_to_buffer(
            &texture.metal_texture,
            0,
            0,
            metal::MTLOrigin {
                x: tile.bounds.origin.x.0 as u64,
                y: tile.bounds.origin.y.0 as u64,
                z: 0,
            },
            metal::MTLSize::new(
                tile.bounds.size.width.0 as u64,
                tile.bounds.size.height.0 as u64,
                1,
            ),
            &buffer,
            0,
            layout.row_stride as u64,
            layout.upload_len as u64,
            metal::MTLBlitOption::None,
        );
        encoder.end_encoding();
        command.commit();
        buffer
    }

    fn finish_uploads(atlas: &MetalAtlas) {
        let state = atlas.0.lock();
        let command = state.backend.command_queue.new_command_buffer();
        command.commit();
        command.wait_until_completed();
        assert_eq!(command.status(), metal::MTLCommandBufferStatus::Completed);
    }

    fn assert_snapshot(
        buffer: &metal::BufferRef,
        kind: AtlasTextureKind,
        size: Size<DevicePixels>,
        value: u8,
    ) {
        let layout = upload_layout(kind, size, usize::MAX).unwrap();
        // The caller waited for the final command on the same queue; every
        // readback buffer is shared CPU/GPU storage and no writer remains.
        let bytes = unsafe {
            std::slice::from_raw_parts(buffer.contents().cast::<u8>(), layout.upload_len)
        };
        for row in bytes.chunks_exact(layout.row_stride) {
            assert!(row[..layout.row_bytes].iter().all(|byte| *byte == value));
        }
    }

    #[test]
    fn live_uploads_reuse_tiles_and_preserve_gpu_read_order() {
        let Some(atlas) = create_atlas() else { return };
        let size = Size {
            width: DevicePixels(3),
            height: DevicePixels(2),
        };
        for key in [
            make_image_key(500, 0),
            AtlasKey::Svg(gpui::RenderSvgParams {
                path: "test-live-mask".into(),
                size,
            }),
        ] {
            let kind = key.texture_kind();
            let len = upload_layout(kind, size, usize::MAX).unwrap().input_len;
            let first = atlas
                .write_bytes(&key, size, &vec![1; len])
                .unwrap()
                .unwrap();
            let mut snapshots = Vec::new();
            // A prior GPU read must see its frame even when later writes have
            // already been queued. Staging buffers are dropped by each call.
            snapshots.push((snapshot(&atlas, first), 1));
            for value in 2..=12 {
                let tile = atlas
                    .write_bytes(&key, size, &vec![value; len])
                    .unwrap()
                    .unwrap();
                assert_eq!(tile, first);
                snapshots.push((snapshot(&atlas, tile), value));
            }
            finish_uploads(&atlas);
            for (buffer, value) in snapshots {
                assert_snapshot(&buffer, kind, size, value);
            }
            atlas.remove(&key);
            assert!(atlas.tile(&key).is_none());
            assert!(atlas.metal_texture(first.texture_id).is_none());
        }
    }

    #[test]
    fn live_upload_validation_resize_and_stats_preserve_other_tiles() {
        let Some(atlas) = create_atlas() else { return };
        let key = make_image_key(600, 0);
        let neighbor = make_image_key(601, 0);
        let small = Size {
            width: DevicePixels(3),
            height: DevicePixels(2),
        };
        let big = Size {
            width: DevicePixels(7),
            height: DevicePixels(4),
        };
        let first = atlas.write_bytes(&key, small, &[7; 24]).unwrap().unwrap();
        let next = atlas
            .write_bytes(&neighbor, small, &[9; 24])
            .unwrap()
            .unwrap();
        assert_eq!(first.texture_id, next.texture_id);
        // Rejected input must not remove or mutate the working mapping.
        assert!(atlas.write_bytes(&key, small, &[0; 23]).unwrap().is_none());
        assert_eq!(atlas.tile(&key), Some(first));
        let before = snapshot(&atlas, first);
        let resized = atlas.write_bytes(&key, big, &[8; 112]).unwrap().unwrap();
        assert_eq!(resized.bounds.size, big);
        assert_ne!(resized, first);
        let after = snapshot(&atlas, resized);
        let beside = snapshot(&atlas, next);
        finish_uploads(&atlas);
        assert_snapshot(&before, AtlasTextureKind::Polychrome, small, 7);
        assert_snapshot(&after, AtlasTextureKind::Polychrome, big, 8);
        assert_snapshot(&beside, AtlasTextureKind::Polychrome, small, 9);
        let stats = atlas.stats().unwrap();
        assert_eq!(stats.image_keys, 2);
        assert_eq!(stats.image_pixels, 34);
        assert_eq!(stats.kinds[AtlasTextureKind::Polychrome as usize].tiles, 2);
        atlas.remove(&key);
        assert!(atlas.tile(&key).is_none());
        assert_eq!(atlas.tile(&neighbor), Some(next));
        assert_eq!(atlas.stats().unwrap().image_keys, 1);
        atlas.remove(&neighbor);
        let stats = atlas.stats().unwrap();
        assert_eq!(stats.keys, 0);
        assert_eq!(stats.kinds[AtlasTextureKind::Polychrome as usize].pages, 0);
    }

    #[test]
    fn live_upload_layout_rejects_invalid_sizes_and_short_rows() {
        let size = Size {
            width: DevicePixels(3),
            height: DevicePixels(2),
        };
        let mono = upload_layout(AtlasTextureKind::Monochrome, size, 6).unwrap();
        let color = upload_layout(AtlasTextureKind::Polychrome, size, 24).unwrap();
        assert_eq!(
            (mono.row_bytes, mono.row_stride, mono.upload_len),
            (3, 256, 512)
        );
        assert_eq!(
            (color.row_bytes, color.row_stride, color.upload_len),
            (12, 256, 512)
        );
        assert!(upload_layout(AtlasTextureKind::Monochrome, size, 5).is_none());
        assert!(upload_layout(AtlasTextureKind::Polychrome, size, 23).is_none());
        assert!(upload_layout(AtlasTextureKind::Subpixel, size, usize::MAX).is_none());
        for width in [-1, 0, 16385, i32::MAX] {
            assert!(
                upload_layout(
                    AtlasTextureKind::Polychrome,
                    Size {
                        width: DevicePixels(width),
                        ..size
                    },
                    usize::MAX
                )
                .is_none()
            );
        }
    }
}
