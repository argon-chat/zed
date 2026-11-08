//! DirectWrite COLR layer composition on the platform's native D3D12 queue.
//!
//! DirectWrite still produces the alpha masks. Only their color composition and
//! gamma correction run here, preserving the same shader and BGRA contract as
//! the D3D11 text system without creating a second graphics device.

use std::{ffi::c_void, sync::Arc};

use anyhow::{Context as _, Result, ensure};
use gpui::{Bounds, DevicePixels, Rgba, Size};
use parking_lot::Mutex;
use windows::Win32::{
    Foundation::RECT,
    Graphics::{
        Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
        Direct3D12::*,
        Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_R8_UNORM},
    },
};

use super::{
    DirectXDevices,
    device::{Device, TextureLocation, transition},
    pipelines::{Pipelines, ROOT_GLOBALS, ROOT_TEXTURES},
};
use crate::directx_renderer::{DirectXRenderer, shader_resources::ShaderModule};

pub(crate) struct GPUState {
    gpu: Arc<Device>,
    // Color glyphs may never be drawn (including headless/text-only use), so
    // creating the text system must not compile shaders or allocate a PSO.
    pipelines: Mutex<Option<Pipelines>>,
}

pub(crate) struct GlyphLayerTexture {
    run_color: Rgba,
    bounds: Bounds<i32>,
    texture: ID3D12Resource,
}

#[repr(C)]
struct GlyphLayerTextureParams {
    bounds: [i32; 4],
    run_color: [f32; 4],
    gamma_ratios: [f32; 4],
    grayscale_enhanced_contrast: f32,
    padding: [f32; 3],
}

const _: () = assert!(std::mem::size_of::<GlyphLayerTextureParams>() == 64);
const CONSTANT_STRIDE: usize = D3D12_CONSTANT_BUFFER_DATA_PLACEMENT_ALIGNMENT as usize;

struct SubmissionResources {
    output: ID3D12Resource,
    readback: ID3D12Resource,
    constants: ID3D12Resource,
    render_targets: ID3D12DescriptorHeap,
    textures: ID3D12DescriptorHeap,
    masks: Vec<ID3D12Resource>,
    pipeline: ID3D12PipelineState,
    root_signature: ID3D12RootSignature,
}

impl GPUState {
    pub(crate) fn new(directx_devices: &DirectXDevices) -> Result<Self> {
        Ok(Self {
            gpu: directx_devices.gpu.clone(),
            pipelines: Mutex::new(None),
        })
    }

    pub(crate) fn composite_color_layers(
        &self,
        layers: &[GlyphLayerTexture],
        bitmap_size: Size<DevicePixels>,
    ) -> Result<Vec<u8>> {
        let width = u32::try_from(bitmap_size.width.0).context("Invalid color glyph width")?;
        let height = u32::try_from(bitmap_size.height.0).context("Invalid color glyph height")?;
        let row_bytes = (width as usize)
            .checked_mul(4)
            .context("Color glyph row overflow")?;
        let byte_count = row_bytes
            .checked_mul(height as usize)
            .context("Color glyph size overflow")?;
        if width == 0 || height == 0 || layers.is_empty() {
            return Ok(vec![0; byte_count]);
        }
        let mut pipelines = self.pipelines.lock();
        if pipelines.is_none() {
            *pipelines = Some(Pipelines::new(&self.gpu.device)?);
        }
        let pipelines = pipelines
            .as_mut()
            .context("Color glyph pipelines were not initialized")?;
        let pipeline = pipelines.pipeline(
            ShaderModule::EmojiRasterization,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            1,
        )?;
        let root_signature = pipelines.root_signature.clone();

        let output = self.gpu.create_texture(
            width,
            height,
            DXGI_FORMAT_B8G8R8A8_UNORM,
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
        )?;
        let output_description = unsafe { output.GetDesc() };
        let mut footprint = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
        let mut readback_size = 0;
        unsafe {
            self.gpu.device.GetCopyableFootprints(
                &output_description,
                0,
                1,
                0,
                Some(&mut footprint),
                None,
                None,
                Some(&mut readback_size),
            );
        }
        let readback = self.gpu.create_buffer(
            readback_size,
            D3D12_HEAP_TYPE_READBACK,
            D3D12_RESOURCE_STATE_COPY_DEST,
        )?;
        let constants_size = layers
            .len()
            .checked_mul(CONSTANT_STRIDE)
            .context("Color glyph constants overflow")?;
        let constants = self.gpu.create_buffer(
            constants_size as u64,
            D3D12_HEAP_TYPE_UPLOAD,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        )?;
        let font_info = DirectXRenderer::get_font_info();
        let mut mapped_constants: *mut c_void = std::ptr::null_mut();
        unsafe {
            constants.Map(
                0,
                Some(&D3D12_RANGE::default()),
                Some(&mut mapped_constants),
            )?;
        }
        // Every b0 starts on its own 256-byte D3D12 CBV alignment. The HLSL
        // payload remains the same 64 bytes as color_text_raster.hlsl.
        for (index, layer) in layers.iter().enumerate() {
            let parameters = GlyphLayerTextureParams {
                bounds: [
                    layer.bounds.origin.x,
                    layer.bounds.origin.y,
                    layer.bounds.size.width,
                    layer.bounds.size.height,
                ],
                run_color: [
                    layer.run_color.r,
                    layer.run_color.g,
                    layer.run_color.b,
                    layer.run_color.a,
                ],
                gamma_ratios: font_info.gamma_ratios,
                grayscale_enhanced_contrast: font_info.grayscale_enhanced_contrast,
                padding: [0.0; 3],
            };
            unsafe {
                std::ptr::copy_nonoverlapping(
                    (&parameters as *const GlyphLayerTextureParams).cast::<u8>(),
                    mapped_constants.cast::<u8>().add(index * CONSTANT_STRIDE),
                    std::mem::size_of::<GlyphLayerTextureParams>(),
                );
            }
        }
        unsafe {
            constants.Unmap(0, None);
        }

        let render_targets: ID3D12DescriptorHeap = unsafe {
            self.gpu
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    ..Default::default()
                })?
        };
        let render_target = unsafe { render_targets.GetCPUDescriptorHandleForHeapStart() };
        unsafe {
            self.gpu
                .device
                .CreateRenderTargetView(&output, None, render_target);
        }
        let descriptor_count = u32::try_from(layers.len())
            .context("Too many color glyph layers")?
            .checked_mul(2)
            .context("Color glyph descriptors overflow")?;
        let textures: ID3D12DescriptorHeap = unsafe {
            self.gpu
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: descriptor_count,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })?
        };
        let descriptor_stride = unsafe {
            self.gpu
                .device
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
        };
        let texture_cpu = unsafe { textures.GetCPUDescriptorHandleForHeapStart() };
        let texture_gpu = unsafe { textures.GetGPUDescriptorHandleForHeapStart() };
        let mask_description = D3D12_SHADER_RESOURCE_VIEW_DESC {
            Format: DXGI_FORMAT_R8_UNORM,
            ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
            Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
            Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                Texture2D: D3D12_TEX2D_SRV {
                    MipLevels: 1,
                    ..Default::default()
                },
            },
        };
        for (index, layer) in layers.iter().enumerate() {
            let mask_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: texture_cpu.ptr + index * 2 * descriptor_stride as usize,
            };
            let unused_handle = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: mask_handle.ptr + descriptor_stride as usize,
            };
            unsafe {
                self.gpu.device.CreateShaderResourceView(
                    &layer.texture,
                    Some(&mask_description),
                    mask_handle,
                );
                // The shared signature has t0/t2. The glyph shader only reads
                // t0; initialize t2 to a valid null texture descriptor as well.
                self.gpu.device.CreateShaderResourceView(
                    None::<&ID3D12Resource>,
                    Some(&mask_description),
                    unused_handle,
                );
            }
        }
        let resources = Arc::new(SubmissionResources {
            output,
            readback,
            constants,
            render_targets,
            textures,
            masks: layers.iter().map(|layer| layer.texture.clone()).collect(),
            pipeline,
            root_signature,
        });
        let fence = self.gpu.submit(vec![resources.clone()], |commands| {
            // Keep all descriptors and COM resources alive until this exact
            // submission completes, even if a device error interrupts readback.
            let _retained = (&resources.render_targets, &resources.masks);
            unsafe {
                commands.SetPipelineState(&resources.pipeline);
                commands.SetGraphicsRootSignature(&resources.root_signature);
                commands.SetDescriptorHeaps(&[Some(resources.textures.clone())]);
                commands.OMSetRenderTargets(1, Some(&render_target), true, None);
                commands.ClearRenderTargetView(render_target, &[0.0; 4], None);
                commands.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP);
                commands.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                let constant_address = resources.constants.GetGPUVirtualAddress();
                for (index, layer) in layers.iter().enumerate() {
                    commands.SetGraphicsRootConstantBufferView(
                        ROOT_GLOBALS,
                        constant_address + (index * CONSTANT_STRIDE) as u64,
                    );
                    commands.SetGraphicsRootDescriptorTable(
                        ROOT_TEXTURES,
                        D3D12_GPU_DESCRIPTOR_HANDLE {
                            ptr: texture_gpu.ptr + index as u64 * 2 * descriptor_stride as u64,
                        },
                    );
                    commands.RSSetViewports(&[D3D12_VIEWPORT {
                        TopLeftX: layer.bounds.origin.x as f32,
                        TopLeftY: layer.bounds.origin.y as f32,
                        Width: layer.bounds.size.width as f32,
                        Height: layer.bounds.size.height as f32,
                        MinDepth: 0.0,
                        MaxDepth: 1.0,
                    }]);
                    commands.DrawInstanced(4, 1, 0, 0);
                }
                transition(
                    commands,
                    &resources.output,
                    D3D12_RESOURCE_STATE_RENDER_TARGET,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                );
                let source = TextureLocation::subresource(&resources.output, 0);
                let destination = TextureLocation::placed(&resources.readback, footprint);
                commands.CopyTextureRegion(destination.raw(), 0, 0, 0, source.raw(), None);
            }
            Ok(())
        })?;
        // Glyph rasterization is synchronous to DirectWrite's caller. Wait for
        // this submitted copy only, without globally idling the shared queue.
        self.gpu.wait_for(fence)?;
        let read_range = D3D12_RANGE {
            Begin: 0,
            End: readback_size as usize,
        };
        let mut mapped_readback: *mut c_void = std::ptr::null_mut();
        unsafe {
            resources
                .readback
                .Map(0, Some(&read_range), Some(&mut mapped_readback))?;
        }
        let mut rasterized = vec![0; byte_count];
        for row in 0..height as usize {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    mapped_readback.cast::<u8>().add(
                        footprint.Offset as usize + row * footprint.Footprint.RowPitch as usize,
                    ),
                    rasterized.as_mut_ptr().add(row * row_bytes),
                    row_bytes,
                );
            }
        }
        unsafe {
            resources.readback.Unmap(0, Some(&D3D12_RANGE::default()));
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

impl GlyphLayerTexture {
    pub(crate) fn new(
        gpu_state: &GPUState,
        run_color: Rgba,
        bounds: Bounds<i32>,
        alpha_data: &[u8],
    ) -> Result<Self> {
        let width = u32::try_from(bounds.size.width).context("Invalid glyph layer width")?;
        let height = u32::try_from(bounds.size.height).context("Invalid glyph layer height")?;
        ensure!(
            width > 0 && height > 0,
            "Color glyph layers must not be empty"
        );
        let expected_bytes = (width as usize)
            .checked_mul(height as usize)
            .context("Color glyph layer size overflow")?;
        ensure!(
            alpha_data.len() == expected_bytes,
            "Invalid color glyph mask size"
        );
        let texture = gpu_state.gpu.create_texture(
            width,
            height,
            DXGI_FORMAT_R8_UNORM,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        gpu_state.gpu.upload_texture(
            &texture,
            (0, 0),
            (width, height),
            1,
            alpha_data,
            D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
        )?;
        Ok(Self {
            run_color,
            bounds,
            texture,
        })
    }
}
