//! DirectWrite COLR layer composition for the native D3D12 backend.
//!
//! DirectWrite still produces the alpha masks. Their color composition and
//! gamma correction run on the CPU: the masks are a few hundred bytes, and a
//! GPU pass needed roughly twenty resource creations, a submit per layer and a
//! blocking fence wait on the UI thread for every new emoji. The arithmetic is
//! `color_text_raster.hlsl` and the `Premultiplied` blend into a B8G8R8A8_UNORM
//! target, including the 8-bit quantization between layers, so the BGRA
//! contract of the D3D11 text system is unchanged.

use anyhow::{Context as _, Result, ensure};
use gpui::{Bounds, DevicePixels, Rgba, Size};

use super::DirectXDevices;
use crate::directx_renderer::DirectXRenderer;

pub(crate) struct GPUState;

pub(crate) struct GlyphLayerTexture {
    run_color: Rgba,
    bounds: Bounds<i32>,
    alpha: Vec<u8>,
}

impl GPUState {
    pub(crate) fn new(_: &DirectXDevices) -> Result<Self> {
        Ok(Self)
    }

    pub(crate) fn composite_color_layers(
        &self,
        layers: &[GlyphLayerTexture],
        bitmap_size: Size<DevicePixels>,
    ) -> Result<Vec<u8>> {
        let width = usize::try_from(bitmap_size.width.0).context("Invalid color glyph width")?;
        let height = usize::try_from(bitmap_size.height.0).context("Invalid color glyph height")?;
        let byte_count = width
            .checked_mul(4)
            .and_then(|row| row.checked_mul(height))
            .context("Color glyph size overflow")?;
        let mut rasterized = vec![0u8; byte_count];
        if width == 0 || height == 0 || layers.is_empty() {
            return Ok(rasterized);
        }
        let font_info = DirectXRenderer::get_font_info();
        for layer in layers {
            composite_layer(
                &mut rasterized,
                (width, height),
                layer,
                font_info.gamma_ratios,
                font_info.grayscale_enhanced_contrast,
            );
        }
        // Preserve DirectWrite's straight-alpha BGRA output contract.
        for pixel in rasterized.chunks_exact_mut(4) {
            let alpha = pixel[3] as f32;
            if alpha > 0.0 {
                let inverse_alpha = 255.0 / alpha;
                for channel in &mut pixel[..3] {
                    *channel = (*channel as f32 * inverse_alpha).clamp(0.0, 255.0) as u8;
                }
            }
        }
        Ok(rasterized)
    }
}

/// One layer's draw: `emoji_rasterization_fragment` over the layer's
/// viewport, blended `ONE, INV_SRC_ALPHA` into premultiplied BGRA bytes.
fn composite_layer(
    target: &mut [u8],
    (width, height): (usize, usize),
    layer: &GlyphLayerTexture,
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
) {
    let color = [layer.run_color.r, layer.run_color.g, layer.run_color.b];
    // Per-layer constants of `apply_contrast_and_gamma_correction`.
    let brightness = color_brightness(color);
    let contrast = grayscale_enhanced_contrast * (4.0 * (0.75 - brightness)).clamp(0.0, 1.0);
    // The mask has one texel per viewport pixel, so the bilinear sample at a
    // pixel center is exactly that texel; one correction per mask value.
    let mut corrected = [0.0f32; 256];
    for (value, slot) in corrected.iter_mut().enumerate() {
        let sample = value as f32 / 255.0;
        let contrasted = sample * (contrast + 1.0) / (sample * contrast + 1.0);
        let alpha = alpha_correction(contrasted, brightness, gamma_ratios) * layer.run_color.a;
        *slot = alpha;
    }
    let origin_x = layer.bounds.origin.x as i64;
    let origin_y = layer.bounds.origin.y as i64;
    let layer_width = layer.bounds.size.width.max(0) as i64;
    let layer_height = layer.bounds.size.height.max(0) as i64;
    let x0 = origin_x.max(0);
    let y0 = origin_y.max(0);
    let x1 = (origin_x + layer_width).min(width as i64);
    let y1 = (origin_y + layer_height).min(height as i64);
    for y in y0..y1 {
        let mask_row = ((y - origin_y) * layer_width) as usize;
        let target_row = y as usize * width * 4;
        for x in x0..x1 {
            let alpha = corrected[layer.alpha[mask_row + (x - origin_x) as usize] as usize];
            // A UNORM target clamps the shader's output before blending.
            let source_alpha = alpha.clamp(0.0, 1.0);
            // B8G8R8A8: the shader's red lands in byte 2.
            let source = [
                (color[2] * alpha).clamp(0.0, 1.0),
                (color[1] * alpha).clamp(0.0, 1.0),
                (color[0] * alpha).clamp(0.0, 1.0),
                source_alpha,
            ];
            let pixel = &mut target[target_row + x as usize * 4..][..4];
            for (channel, source) in pixel.iter_mut().zip(source) {
                let destination = *channel as f32 / 255.0;
                let blended = source + destination * (1.0 - source_alpha);
                *channel = (blended.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
            }
        }
    }
}

/// `color_brightness` in `alpha_correction.hlsl`.
fn color_brightness(color: [f32; 3]) -> f32 {
    color[0] * 0.30 + color[1] * 0.59 + color[2] * 0.11
}

/// `apply_alpha_correction` in `alpha_correction.hlsl`.
fn alpha_correction(a: f32, b: f32, g: [f32; 4]) -> f32 {
    let brightness_adjustment = g[0] * b + g[1];
    let correction = brightness_adjustment * a + (g[2] * b + g[3]);
    a + a * (1.0 - a) * correction
}

impl GlyphLayerTexture {
    pub(crate) fn new(
        _: &GPUState,
        run_color: Rgba,
        bounds: Bounds<i32>,
        alpha_data: &[u8],
    ) -> Result<Self> {
        let width = usize::try_from(bounds.size.width).context("Invalid glyph layer width")?;
        let height = usize::try_from(bounds.size.height).context("Invalid glyph layer height")?;
        ensure!(
            width > 0 && height > 0,
            "Color glyph layers must not be empty"
        );
        let expected_bytes = width
            .checked_mul(height)
            .context("Color glyph layer size overflow")?;
        ensure!(
            alpha_data.len() == expected_bytes,
            "Invalid color glyph mask size"
        );
        Ok(Self {
            run_color,
            bounds,
            alpha: alpha_data.to_vec(),
        })
    }
}
