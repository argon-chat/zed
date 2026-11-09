#[cfg(test)]
use std::ffi::CString;
use std::{
    borrow::Cow,
    mem::ManuallyDrop,
    sync::Arc,
    time::Instant,
};

use parking_lot::Mutex;

use anyhow::{Context as _, Result, bail, ensure};
use windows::Win32::Graphics::{
    Direct3D::ID3DBlob,
    Direct3D12::*,
    Dxgi::Common::{DXGI_FORMAT, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC},
};
#[cfg(test)]
use windows::{
    Win32::Graphics::Direct3D::{
        Fxc::{D3DCOMPILE_ENABLE_STRICTNESS, D3DCOMPILE_OPTIMIZATION_LEVEL3, D3DCompile},
        ID3DInclude,
    },
    core::{PCSTR, s},
};

use crate::directx_renderer::shader_resources::{self, EffectShader, ShaderModule};

pub(super) const ROOT_GLOBALS: u32 = 0;
pub(super) const ROOT_BATCH: u32 = 1;
pub(super) const ROOT_INSTANCES: u32 = 2;
pub(super) const ROOT_CLIP_SEGMENTS: u32 = 3;
pub(super) const ROOT_TEXT_BACKGROUNDS: u32 = 4;
pub(super) const ROOT_TEXTURES: u32 = 5;

#[cfg(test)]
const ALPHA_CORRECTION: &str = include_str!("../alpha_correction.hlsl");
#[cfg(test)]
const PRIMITIVES: &str = include_str!("../shaders.hlsl");
#[cfg(test)]
const EFFECT_VERTEX: &str = include_str!("../effects.hlsl");
#[cfg(test)]
const EMOJI: &str = include_str!("../color_text_raster.hlsl");
#[cfg(test)]
const FROST: &str = include_str!(concat!(env!("GPUI_D3D12_EFFECTS_HLSL_DIR"), "/frost.hlsl"));
#[cfg(test)]
const NOISE: &str = include_str!(concat!(env!("GPUI_D3D12_EFFECTS_HLSL_DIR"), "/noise.hlsl"));
#[cfg(test)]
const GLOW: &str = include_str!(concat!(env!("GPUI_D3D12_EFFECTS_HLSL_DIR"), "/glow.hlsl"));

mod embedded {
    use super::{EffectShader, ShaderModule};
    include!(concat!(env!("OUT_DIR"), "/native_dx12_shaders.rs"));
}

struct CachedPipeline {
    key: PipelineKey,
    state: ID3D12PipelineState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PipelineKey {
    kind: PipelineKind,
    format: DXGI_FORMAT,
    samples: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PipelineKind {
    Primitive(ShaderModule),
    DynamicEffect(usize),
}

pub(super) struct Pipelines {
    pub root_signature: ID3D12RootSignature,
    device: ID3D12Device,
    cache: Vec<CachedPipeline>,
    /// Pipelines a background thread created ahead of their first use.
    warmed: Arc<Mutex<Vec<CachedPipeline>>>,
    dynamic_effects: Vec<shader_resources::DynamicEffect>,
    timings: bool,
}

/// The pipelines a window's frames use, most common first. Creating a PSO the
/// driver has not cached yet (a new build or driver) takes tens of
/// milliseconds, which on first use landed inside a frame.
const WARM_MODULES: [(ShaderModule, u32); 15] = [
    (ShaderModule::Quad, 1),
    (ShaderModule::MonochromeSprite, 1),
    (ShaderModule::Shadow, 1),
    (ShaderModule::PolychromeSprite, 1),
    (ShaderModule::Underline, 1),
    (ShaderModule::PathRasterization, 4),
    (ShaderModule::PathSprite, 1),
    (ShaderModule::VnLayer, 1),
    (ShaderModule::BlurRect, 1),
    (ShaderModule::BlurDownsample, 1),
    (ShaderModule::BlurUpsample, 1),
    (ShaderModule::SubpixelSprite, 1),
    (ShaderModule::Effect(EffectShader::Frost), 1),
    (ShaderModule::Effect(EffectShader::Noise), 1),
    (ShaderModule::Effect(EffectShader::Glow), 1),
];

impl Pipelines {
    pub fn new(device: &ID3D12Device) -> Result<Self> {
        Ok(Self {
            root_signature: create_root_signature(device)?,
            device: device.clone(),
            cache: Vec::new(),
            warmed: Arc::default(),
            dynamic_effects: shader_resources::dynamic_effects(),
            timings: std::env::var_os("VN_DX12_TIMINGS").is_some_and(|value| value == "1"),
        })
    }

    /// Creates the common pipelines for `format` on a background thread. A
    /// frame that needs one first takes it from there, or creates it itself
    /// when the thread has not reached it yet.
    pub fn prewarm(&self, format: DXGI_FORMAT) {
        let device = self.device.clone();
        let root = self.root_signature.clone();
        let warmed = Arc::downgrade(&self.warmed);
        let spawned = std::thread::Builder::new()
            .name("dx12-pso-warmup".into())
            .spawn(move || {
                for (module, samples) in WARM_MODULES {
                    // The renderer is gone (closed window, lost device).
                    let Some(warmed) = warmed.upgrade() else { return };
                    let key = PipelineKey {
                        kind: PipelineKind::Primitive(module),
                        format,
                        samples,
                    };
                    match create_primitive(&device, &root, module, format, samples) {
                        Ok(state) => warmed.lock().push(CachedPipeline { key, state }),
                        Err(error) => {
                            log::warn!("D3D12 pipeline warmup stopped at {module:?}: {error:#}");
                            return;
                        }
                    }
                }
            });
        if let Err(error) = spawned {
            log::warn!("D3D12 pipeline warmup thread failed to start: {error}");
        }
    }

    pub fn pipeline(
        &mut self,
        module: ShaderModule,
        format: DXGI_FORMAT,
        sample_count: u32,
    ) -> Result<ID3D12PipelineState> {
        self.get(PipelineKey {
            kind: PipelineKind::Primitive(module),
            format,
            samples: sample_count,
        })
    }

    pub fn effect(
        &mut self,
        index: u32,
        format: DXGI_FORMAT,
        sample_count: u32,
    ) -> Result<(ID3D12PipelineState, bool)> {
        let backdrop = self.needs_backdrop(index)?;
        let kind = if let Some(effect) = EffectShader::ALL.get(index as usize) {
            PipelineKind::Primitive(ShaderModule::Effect(*effect))
        } else {
            PipelineKind::DynamicEffect(index as usize - EffectShader::ALL.len())
        };
        Ok((
            self.get(PipelineKey {
                kind,
                format,
                samples: sample_count,
            })?,
            backdrop,
        ))
    }

    pub fn needs_backdrop(&self, index: u32) -> Result<bool> {
        if let Some(effect) = EffectShader::ALL.get(index as usize) {
            return Ok(effect.needs_backdrop());
        }
        self.dynamic_effects
            .get(index as usize - EffectShader::ALL.len())
            .map(|effect| effect.needs_backdrop)
            .with_context(|| format!("Unknown D3D12 effect id {index}"))
    }

    fn get(&mut self, key: PipelineKey) -> Result<ID3D12PipelineState> {
        if let Some(cached) = self.cache.iter().find(|cached| cached.key == key) {
            return Ok(cached.state.clone());
        }
        let warmed = self
            .warmed
            .lock()
            .iter()
            .find(|cached| cached.key == key)
            .map(|cached| cached.state.clone());
        if let Some(state) = warmed {
            self.cache.push(CachedPipeline {
                key,
                state: state.clone(),
            });
            return Ok(state);
        }
        ensure!(
            key.samples != 0,
            "D3D12 pipeline sample count must be positive"
        );
        let bytecode_started = self.timings.then(Instant::now);
        let (vertex, fragment, blend) = match key.kind {
            PipelineKind::Primitive(module) => {
                let (vertex, fragment) = embedded::module(module);
                (
                    Cow::Borrowed(vertex),
                    Cow::Borrowed(fragment),
                    module_blend(module),
                )
            }
            PipelineKind::DynamicEffect(index) => {
                let effect = self
                    .dynamic_effects
                    .get(index)
                    .context("D3D12 dynamic effect index is out of bounds")?;
                let vertex = embedded::effect_vertex();
                let fragment = shader_resources::compile_hlsl_file(
                    &effect.hlsl,
                    "effect_fragment\0",
                    "ps_5_1\0",
                )
                .with_context(|| format!("Compiling D3D12 effect {}", effect.name))?;
                (
                    Cow::Borrowed(vertex),
                    Cow::Owned(bytes(&fragment).to_vec()),
                    if effect.needs_backdrop {
                        Blend::Replace
                    } else {
                        Blend::Straight
                    },
                )
            }
        };
        let bytecode_time = bytecode_started.map(|started| started.elapsed());
        let pipeline_started = self.timings.then(Instant::now);
        let state = create_pipeline(
            &self.device,
            &self.root_signature,
            &vertex,
            &fragment,
            key.format,
            key.samples,
            blend,
        )?;
        if let Some(started) = pipeline_started {
            log::info!(
                "[DX12 pipeline] kind={:?} samples={} bytecode_ms={:.3} pso_ms={:.3}",
                key.kind,
                key.samples,
                bytecode_time.unwrap_or_default().as_secs_f64() * 1000.,
                started.elapsed().as_secs_f64() * 1000.
            );
        }
        self.cache.push(CachedPipeline {
            key,
            state: state.clone(),
        });
        Ok(state)
    }
}

fn create_root_signature(device: &ID3D12Device) -> Result<ID3D12RootSignature> {
    let texture_ranges = [
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 0,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 0,
        },
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 1,
            BaseShaderRegister: 2,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 1,
        },
    ];
    let descriptor = |kind, register| D3D12_ROOT_PARAMETER {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR {
                ShaderRegister: register,
                RegisterSpace: 0,
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
    };
    let parameters = [
        descriptor(D3D12_ROOT_PARAMETER_TYPE_CBV, 0),
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 1,
                    RegisterSpace: 0,
                    Num32BitValues: 4,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
        descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV, 1),
        descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV, 3),
        descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV, 4),
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: texture_ranges.len() as u32,
                    pDescriptorRanges: texture_ranges.as_ptr(),
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        },
    ];
    let sampler = |register, address| D3D12_STATIC_SAMPLER_DESC {
        Filter: D3D12_FILTER_MIN_MAG_MIP_LINEAR,
        AddressU: address,
        AddressV: address,
        AddressW: address,
        MaxAnisotropy: 1,
        ComparisonFunc: D3D12_COMPARISON_FUNC_NEVER,
        BorderColor: D3D12_STATIC_BORDER_COLOR_TRANSPARENT_BLACK,
        MinLOD: 0.0,
        MaxLOD: f32::MAX,
        ShaderRegister: register,
        ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        ..Default::default()
    };
    let samplers = [
        sampler(0, D3D12_TEXTURE_ADDRESS_MODE_WRAP),
        sampler(1, D3D12_TEXTURE_ADDRESS_MODE_CLAMP),
    ];
    let description = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: parameters.len() as u32,
        pParameters: parameters.as_ptr(),
        NumStaticSamplers: samplers.len() as u32,
        pStaticSamplers: samplers.as_ptr(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_DENY_HULL_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_DOMAIN_SHADER_ROOT_ACCESS
            | D3D12_ROOT_SIGNATURE_FLAG_DENY_GEOMETRY_SHADER_ROOT_ACCESS,
    };
    let mut serialized = None;
    let mut diagnostic = None;
    let result = unsafe {
        D3D12SerializeRootSignature(
            &description,
            D3D_ROOT_SIGNATURE_VERSION_1,
            &mut serialized,
            Some(&mut diagnostic),
        )
    };
    if let Err(error) = result {
        bail!(
            "Serializing GPUI D3D12 root signature: {error}; {}",
            diagnostic.as_ref().map(blob_message).unwrap_or_default()
        );
    }
    let serialized = serialized.context("D3D12 root signature serializer returned no bytecode")?;
    unsafe { device.CreateRootSignature(0, bytes(&serialized)) }
        .context("Creating GPUI D3D12 root signature")
}

#[derive(Clone, Copy)]
enum Blend {
    Straight,
    Premultiplied,
    Replace,
    Subpixel,
}

fn module_blend(module: ShaderModule) -> Blend {
    match module {
        ShaderModule::BlurDownsample | ShaderModule::BlurUpsample | ShaderModule::BlurRect => {
            Blend::Replace
        }
        ShaderModule::PathRasterization
        | ShaderModule::PathSprite
        | ShaderModule::VnLayer
        | ShaderModule::EmojiRasterization => Blend::Premultiplied,
        ShaderModule::SubpixelSprite => Blend::Subpixel,
        ShaderModule::Effect(effect) if effect.needs_backdrop() => Blend::Replace,
        _ => Blend::Straight,
    }
}

fn blend_description(blend: Blend) -> D3D12_BLEND_DESC {
    let mut description = D3D12_BLEND_DESC::default();
    let target = &mut description.RenderTarget[0];
    target.BlendEnable = (!matches!(blend, Blend::Replace)).into();
    target.LogicOpEnable = false.into();
    target.SrcBlend = match blend {
        Blend::Straight => D3D12_BLEND_SRC_ALPHA,
        Blend::Subpixel => D3D12_BLEND_SRC1_COLOR,
        _ => D3D12_BLEND_ONE,
    };
    target.DestBlend = if matches!(blend, Blend::Subpixel) {
        D3D12_BLEND_INV_SRC1_COLOR
    } else {
        D3D12_BLEND_INV_SRC_ALPHA
    };
    target.BlendOp = D3D12_BLEND_OP_ADD;
    target.SrcBlendAlpha = D3D12_BLEND_ONE;
    target.DestBlendAlpha = if matches!(blend, Blend::Subpixel) {
        D3D12_BLEND_ZERO
    } else {
        D3D12_BLEND_INV_SRC_ALPHA
    };
    target.BlendOpAlpha = D3D12_BLEND_OP_ADD;
    target.LogicOp = D3D12_LOGIC_OP_NOOP;
    target.RenderTargetWriteMask = if matches!(blend, Blend::Subpixel) {
        (D3D12_COLOR_WRITE_ENABLE_ALL.0 & !D3D12_COLOR_WRITE_ENABLE_ALPHA.0) as u8
    } else {
        D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8
    };
    description
}

fn create_primitive(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    module: ShaderModule,
    format: DXGI_FORMAT,
    samples: u32,
) -> Result<ID3D12PipelineState> {
    let (vertex, fragment) = embedded::module(module);
    create_pipeline(
        device,
        root,
        vertex,
        fragment,
        format,
        samples,
        module_blend(module),
    )
}

fn create_pipeline(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    vertex: &[u8],
    fragment: &[u8],
    format: DXGI_FORMAT,
    sample_count: u32,
    blend: Blend,
) -> Result<ID3D12PipelineState> {
    let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
    formats[0] = format;
    let mut description = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: ManuallyDrop::new(Some(root.clone())),
        VS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: vertex.as_ptr().cast(),
            BytecodeLength: vertex.len(),
        },
        PS: D3D12_SHADER_BYTECODE {
            pShaderBytecode: fragment.as_ptr().cast(),
            BytecodeLength: fragment.len(),
        },
        BlendState: blend_description(blend),
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            DepthClipEnable: true.into(),
            MultisampleEnable: (sample_count > 1).into(),
            ..Default::default()
        },
        DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
            DepthEnable: false.into(),
            DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
            DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
            StencilEnable: false.into(),
            ..Default::default()
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: formats,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: sample_count,
            Quality: 0,
        },
        ..Default::default()
    };
    let result = unsafe { device.CreateGraphicsPipelineState(&description) };
    // The generated descriptor uses ManuallyDrop for its borrowed COM field.
    unsafe {
        ManuallyDrop::drop(&mut description.pRootSignature);
    }
    result.context("Creating GPUI D3D12 graphics pipeline")
}

#[cfg(test)]
fn compile_module(module: ShaderModule) -> Result<(Vec<u8>, Vec<u8>)> {
    if let ShaderModule::Effect(effect) = module {
        let fragment = match effect {
            EffectShader::Frost => FROST,
            EffectShader::Noise => NOISE,
            EffectShader::Glow => GLOW,
        };
        return Ok((
            compile(EFFECT_VERTEX, "effect_vertex", true)?,
            compile(fragment, "effect_fragment", false)?,
        ));
    }
    let prefix = match module {
        ShaderModule::Quad => "quad",
        ShaderModule::Shadow => "shadow",
        ShaderModule::BlurDownsample => "blur_downsample",
        ShaderModule::BlurUpsample => "blur_upsample",
        ShaderModule::BlurRect => "blur_rect",
        ShaderModule::Underline => "underline",
        ShaderModule::PathRasterization => "path_rasterization",
        ShaderModule::PathSprite => "path_sprite",
        ShaderModule::MonochromeSprite => "monochrome_sprite",
        ShaderModule::SubpixelSprite => "subpixel_sprite",
        ShaderModule::PolychromeSprite => "polychrome_sprite",
        ShaderModule::EmojiRasterization => "emoji_rasterization",
        ShaderModule::VnLayer => "vn_layer",
        ShaderModule::Effect(_) => bail!("Effect shaders require their shared vertex stage"),
    };
    let source = if module == ShaderModule::EmojiRasterization {
        EMOJI
    } else {
        PRIMITIVES
    };
    Ok((
        compile(source, &format!("{prefix}_vertex"), true)?,
        compile(source, &format!("{prefix}_fragment"), false)?,
    ))
}

#[cfg(test)]
fn compile(source: &str, entry: &str, vertex: bool) -> Result<Vec<u8>> {
    let source = source.replace("#include \"alpha_correction.hlsl\"", ALPHA_CORRECTION);
    let entry = CString::new(entry).context("Shader entry contains a NUL byte")?;
    let mut bytecode = None;
    let mut diagnostic = None;
    let result = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("gpui-d3d12"),
            None,
            None::<&ID3DInclude>,
            PCSTR(entry.as_ptr().cast()),
            if vertex { s!("vs_5_1") } else { s!("ps_5_1") },
            D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut bytecode,
            Some(&mut diagnostic),
        )
    };
    if let Err(error) = result {
        bail!(
            "Compiling D3D12 shader {}: {error}; {}",
            entry.to_string_lossy(),
            diagnostic.as_ref().map(blob_message).unwrap_or_default()
        );
    }
    let bytecode = bytecode.context("D3DCompile returned no bytecode")?;
    Ok(bytes(&bytecode).to_vec())
}

fn bytes(blob: &ID3DBlob) -> &[u8] {
    // ID3DBlob owns this buffer for the lifetime of the returned borrow.
    unsafe { std::slice::from_raw_parts(blob.GetBufferPointer().cast(), blob.GetBufferSize()) }
}

fn blob_message(blob: &ID3DBlob) -> String {
    String::from_utf8_lossy(bytes(blob))
        .trim_end_matches('\0')
        .trim()
        .to_owned()
}

#[cfg(test)]
mod tests;
