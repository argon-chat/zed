//! Bounded uploads for newly cached CPU atlas tiles. Mutable media writes stay immediate.

use std::sync::Arc;

use anyhow::{Context, Result, ensure};
use gpui::AtlasTile;
use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};

use super::{
    atlas::ATLAS_STATE,
    device::{Device, KeepAlive, TextureLocation, transition},
};

const MAX_BATCH_BYTES: usize = 1024 * 1024;
const MAX_BATCH_COPIES: usize = 256;
const COPY_ALIGNMENT: usize = D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT as usize;
const ROW_ALIGNMENT: usize = D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as usize;

struct Copy {
    tile: AtlasTile,
    destination: ID3D12Resource,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
}

#[derive(Default)]
pub(super) struct UploadBatch {
    bytes: Vec<u8>,
    copies: Vec<Copy>,
}

impl UploadBatch {
    pub(super) fn queue(
        &mut self,
        gpu: &Device,
        tile: AtlasTile,
        destination: &ID3D12Resource,
        bytes_per_pixel: u32,
        pixels: &[u8],
    ) -> Result<()> {
        let size = (
            tile.bounds.size.width.0 as u32,
            tile.bounds.size.height.0 as u32,
        );
        let origin = (tile.bounds.left().0 as u32, tile.bounds.top().0 as u32);
        let descriptor = unsafe { destination.GetDesc() };
        let expected_pixel_size = match descriptor.Format {
            DXGI_FORMAT_R8_UNORM => 1,
            DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM => 4,
            _ => 0,
        };
        ensure!(
            descriptor.Dimension == D3D12_RESOURCE_DIMENSION_TEXTURE2D
                && descriptor.SampleDesc.Count == 1
                && size.0 != 0
                && size.1 != 0
                && bytes_per_pixel != 0
                && bytes_per_pixel == expected_pixel_size
                && u64::from(origin.0) + u64::from(size.0) <= descriptor.Width
                && u64::from(origin.1) + u64::from(size.1) <= u64::from(descriptor.Height),
            "Invalid cached atlas upload dimensions or format"
        );
        let row_bytes = (size.0 as usize)
            .checked_mul(bytes_per_pixel as usize)
            .context("Atlas upload row size overflow")?;
        let packed_bytes = row_bytes
            .checked_mul(size.1 as usize)
            .context("Atlas upload pixel size overflow")?;
        ensure!(
            pixels.len() >= packed_bytes,
            "Atlas upload pixels are too short"
        );
        let row_pitch = align(row_bytes, ROW_ALIGNMENT)?;
        let length = row_pitch
            .checked_mul(size.1 as usize)
            .context("Atlas upload staging size overflow")?;
        if length > MAX_BATCH_BYTES {
            self.flush(gpu)?;
            gpu.upload_texture(
                destination,
                origin,
                size,
                bytes_per_pixel,
                pixels,
                ATLAS_STATE,
            )?;
            return Ok(());
        }
        let mut offset = align(self.bytes.len(), COPY_ALIGNMENT)?;
        if self.copies.len() == MAX_BATCH_COPIES || offset + length > MAX_BATCH_BYTES {
            self.flush(gpu)?;
            offset = 0;
        }
        let end = offset + length;
        // Grow geometrically without ever reserving past the batch's hard bound.
        if end > self.bytes.capacity() {
            let capacity = end.next_power_of_two().min(MAX_BATCH_BYTES);
            self.bytes
                .try_reserve_exact(capacity - self.bytes.len())
                .context("Allocating atlas CPU staging")?;
        }
        self.copies
            .try_reserve(1)
            .context("Allocating atlas copy records")?;
        self.bytes.resize(end, 0);
        for row in 0..size.1 as usize {
            self.bytes[offset + row * row_pitch..offset + row * row_pitch + row_bytes]
                .copy_from_slice(&pixels[row * row_bytes..(row + 1) * row_bytes]);
        }
        self.copies.push(Copy {
            tile,
            // This clone remains valid across page removal and texture-id reuse.
            destination: destination.clone(),
            footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: offset as u64,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: descriptor.Format,
                    Width: size.0,
                    Height: size.1,
                    Depth: 1,
                    RowPitch: row_pitch as u32,
                },
            },
        });
        Ok(())
    }

    pub(super) fn cancel(&mut self, tile: AtlasTile) {
        self.copies.retain(|copy| {
            copy.tile.texture_id != tile.texture_id || copy.tile.tile_id != tile.tile_id
        });
        if self.copies.is_empty() {
            self.bytes = Vec::new();
            self.copies = Vec::new();
        }
    }

    pub(super) fn flush(&mut self, gpu: &Device) -> Result<()> {
        if self.copies.is_empty() {
            return Ok(());
        }
        let _profile = gpui::native_profiling::current_span("native", "D3D12 flush atlas uploads");
        let upload = gpu.create_buffer(
            self.bytes.len() as u64,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let mut mapped = std::ptr::null_mut();
        let empty = D3D12_RANGE { Begin: 0, End: 0 };
        unsafe { upload.Map(0, Some(&empty), Some(&mut mapped))? };
        if mapped.is_null() {
            unsafe { upload.Unmap(0, Some(&empty)) };
            anyhow::bail!("Atlas staging buffer mapping returned a null pointer");
        }
        unsafe {
            std::ptr::copy_nonoverlapping(self.bytes.as_ptr(), mapped.cast(), self.bytes.len());
            upload.Unmap(
                0,
                Some(&D3D12_RANGE {
                    Begin: 0,
                    End: self.bytes.len(),
                }),
            );
        }
        let mut owners: Vec<KeepAlive> = Vec::with_capacity(self.copies.len() + 1);
        owners.push(Arc::new(upload.clone()));
        owners.extend(
            self.copies
                .iter()
                .map(|copy| Arc::new(copy.destination.clone()) as KeepAlive),
        );
        // Keep the pending CPU bytes on every failure. Allocation/Map failures may
        // retry; an Execute/Signal failure is gated by Device's existing failure
        // state, which also retains resources that the GPU may already consume.
        gpu.submit(owners, |commands| {
            for copy in &self.copies {
                transition(
                    commands,
                    &copy.destination,
                    ATLAS_STATE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                let target = TextureLocation::subresource(&copy.destination, 0);
                let source = TextureLocation::placed(&upload, copy.footprint);
                unsafe {
                    commands.CopyTextureRegion(
                        target.raw(),
                        copy.tile.bounds.left().0 as u32,
                        copy.tile.bounds.top().0 as u32,
                        0,
                        source.raw(),
                        None,
                    );
                }
                transition(
                    commands,
                    &copy.destination,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    ATLAS_STATE,
                );
            }
            Ok(())
        })?;
        // Neither CPU staging nor a new GPU upload pool remains on idle atlases.
        self.bytes = Vec::new();
        self.copies = Vec::new();
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn pending(&self) -> (usize, usize) {
        (self.copies.len(), self.bytes.capacity())
    }
}

fn align(value: usize, alignment: usize) -> Result<usize> {
    value
        .checked_add(alignment - 1)
        .map(|value| value & !(alignment - 1))
        .context("Atlas upload alignment overflow")
}
