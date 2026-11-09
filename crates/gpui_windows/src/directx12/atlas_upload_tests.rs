use super::*;
use gpui::{FontId, GlyphId, ImageId, RenderGlyphParams, RenderImageParams, px};

fn glyph(index: u32, subpixel: bool) -> AtlasKey {
    AtlasKey::Glyph(RenderGlyphParams {
        font_id: FontId(0),
        glyph_id: GlyphId(index),
        font_size: px(14.),
        subpixel_variant: Point { x: 0, y: 0 },
        scale_factor: 1.,
        is_emoji: false,
        subpixel_rendering: subpixel,
        dilation: 0,
    })
}

fn image(index: usize) -> AtlasKey {
    AtlasKey::Image(RenderImageParams {
        image_id: ImageId(index),
        frame_index: 0,
    })
}

fn size(width: i32, height: i32) -> Size<DevicePixels> {
    Size {
        width: DevicePixels(width),
        height: DevicePixels(height),
    }
}

fn cached(
    atlas: &Atlas,
    key: &AtlasKey,
    size: Size<DevicePixels>,
    bytes: &[u8],
) -> Result<AtlasTile> {
    atlas
        .get_or_insert_with(key.clone(), &mut || Ok(Some((size, Cow::Borrowed(bytes)))))?
        .context("Cached atlas tile refused")
}

fn submissions(device: &Device) -> u64 {
    let stats = device.command_stats();
    stats.created + stats.reused
}

fn read_tile(device: &Device, atlas: &Atlas, tile: AtlasTile, bpp: u32) -> Result<Vec<u8>> {
    let texture = atlas
        .get_texture(tile.texture_id)?
        .context("Atlas page missing")?;
    device.read_texture_for_test(
        &texture.resource,
        (tile.bounds.left().0 as u32, tile.bounds.top().0 as u32),
        (
            tile.bounds.size.width.0 as u32,
            tile.bounds.size.height.0 as u32,
        ),
        bpp,
        ATLAS_STATE,
    )
}

#[test]
#[ignore = "requires a native D3D12 hardware adapter; run explicitly on Windows"]
fn native_cold_atlas_batches_mixed_glyphs_with_aligned_rows_and_releases_staging() -> Result<()> {
    let device = Device::new()?;
    let atlas = Atlas::new(device.clone());
    let before = submissions(&device);
    let mut tiles = Vec::new();
    for index in 0..128u32 {
        let subpixel = index % 2 == 1;
        let value = (index + 1) as u8;
        let bytes = vec![value; if subpixel { 60 } else { 15 }];
        let tile = cached(&atlas, &glyph(index, subpixel), size(5, 3), &bytes)?;
        tiles.push((tile, value));
    }
    assert_eq!(
        submissions(&device),
        before,
        "cache misses must stage without per-glyph submits"
    );
    assert_eq!(atlas.0.lock().backend.uploads.pending().0, 128);
    let mono = atlas
        .get_texture(tiles[0].0.texture_id)?
        .context("Monochrome page missing")?;
    assert_eq!(
        submissions(&device),
        before + 1,
        "all cold glyphs must share one upload submit"
    );
    assert_eq!(
        atlas.0.lock().backend.uploads.pending(),
        (0, 0),
        "idle atlases retain no CPU staging"
    );
    let rgba = atlas
        .get_texture(tiles[1].0.texture_id)?
        .context("Subpixel page missing")?;
    let mono_bytes = device.read_texture_for_test(
        &mono.resource,
        (0, 0),
        (mono.width, mono.height),
        1,
        ATLAS_STATE,
    )?;
    let rgba_bytes = device.read_texture_for_test(
        &rgba.resource,
        (0, 0),
        (rgba.width, rgba.height),
        4,
        ATLAS_STATE,
    )?;
    for (tile, value) in tiles {
        let bpp = if tile.texture_id.kind == AtlasTextureKind::Monochrome {
            1
        } else {
            4
        };
        let bytes = if bpp == 1 { &mono_bytes } else { &rgba_bytes };
        let width = if bpp == 1 { mono.width } else { rgba.width } as usize;
        for row in 0..3 {
            let start = ((tile.bounds.top().0 as usize + row) * width
                + tile.bounds.left().0 as usize)
                * bpp;
            assert_eq!(&bytes[start..start + 5 * bpp], vec![value; 5 * bpp]);
        }
    }
    let before_hit = submissions(&device);
    atlas.get_or_insert_with(glyph(0, false), &mut || panic!("cache hit rebuilt glyph"))?;
    assert_eq!(submissions(&device), before_hit);
    assert!(device.existing_video_bridge().is_none());
    Ok(())
}

#[test]
#[ignore = "requires a native D3D12 hardware adapter; run explicitly on Windows"]
fn native_cold_atlas_flushes_before_cpu_gpu_writers_and_shared_device_readbacks() -> Result<()> {
    let device = Device::new()?;
    let atlas = Atlas::new(device.clone());
    let tile = cached(&atlas, &image(1), size(5, 3), &[17; 60])?;
    let before_write = submissions(&device);
    let rewritten = atlas
        .write_bytes(&image(1), size(5, 3), &[99; 60])?
        .context("CPU write refused")?;
    assert_eq!(tile.bounds, rewritten.bounds);
    assert_eq!(
        submissions(&device),
        before_write + 2,
        "pending cache upload precedes immediate writer"
    );
    assert_eq!(read_tile(&device, &atlas, tile, 4)?, vec![99; 60]);

    let gpu_tile = cached(&atlas, &image(2), size(5, 3), &[37; 60])?;
    let other = Atlas::new(device.clone());
    let other_tile = cached(&other, &image(2), size(5, 3), &[61; 60])?;
    let source = device.create_texture(
        5,
        3,
        DXGI_FORMAT_B8G8R8A8_UNORM,
        D3D12_RESOURCE_FLAG_NONE,
        ATLAS_STATE,
    )?;
    device.upload_texture(&source, (0, 0), (5, 3), 4, &[193; 60], ATLAS_STATE)?;
    let before_gpu = submissions(&device);
    let copied = atlas
        .upload_d3d12(
            &image(2),
            size(5, 3),
            &source,
            ATLAS_STATE,
            None,
            Arc::new(source.clone()),
        )?
        .context("GPU writer refused")?;
    assert_eq!(copied.bounds, gpu_tile.bounds);
    assert_eq!(
        submissions(&device),
        before_gpu + 2,
        "pending cache upload precedes GPU copy"
    );
    assert_eq!(read_tile(&device, &other, other_tile, 4)?, vec![61; 60]);
    assert_eq!(read_tile(&device, &atlas, gpu_tile, 4)?, vec![193; 60]);
    Ok(())
}

#[test]
#[ignore = "requires a native D3D12 hardware adapter; run explicitly on Windows"]
fn native_cold_atlas_bounds_batches_cancels_evictions_and_orders_oversized_tiles() -> Result<()> {
    let device = Device::new()?;
    let atlas = Atlas::new(device.clone());
    assert!(cached(&atlas, &image(99), size(5, 3), &[0; 4]).is_err());
    assert_eq!(
        atlas.stats().unwrap().keys,
        0,
        "failed staging must not publish a cache key"
    );
    let before = submissions(&device);
    let mut tiles = Vec::new();
    for index in 0..300 {
        tiles.push(cached(
            &atlas,
            &glyph(index, false),
            size(1, 1),
            &[index as u8],
        )?);
        let (copies, capacity) = atlas.0.lock().backend.uploads.pending();
        assert!(copies <= 256 && capacity <= 1024 * 1024);
    }
    assert_eq!(submissions(&device), before + 1);
    assert_eq!(atlas.0.lock().backend.uploads.pending().0, 44);
    for index in 0..300 {
        atlas.remove(&glyph(index, false));
    }
    assert_eq!(atlas.0.lock().backend.uploads.pending(), (0, 0));
    assert_eq!(
        atlas.stats().unwrap().kinds[AtlasTextureKind::Monochrome as usize].pages,
        0
    );
    let replacement = cached(&atlas, &glyph(1000, false), size(7, 5), &[52; 35])?;
    assert_eq!(
        replacement.texture_id, tiles[0].texture_id,
        "exercise texture-id reuse after page eviction"
    );
    let before_large = submissions(&device);
    let large_bytes = vec![166; 513 * 513 * 4];
    let large = cached(&atlas, &image(100), size(513, 513), &large_bytes)?;
    assert_eq!(
        submissions(&device),
        before_large + 2,
        "oversized tile follows flush via existing immediate upload"
    );
    assert_eq!(atlas.0.lock().backend.uploads.pending(), (0, 0));
    assert_eq!(read_tile(&device, &atlas, replacement, 1)?, vec![52; 35]);
    assert_eq!(read_tile(&device, &atlas, large, 4)?, large_bytes);
    Ok(())
}
