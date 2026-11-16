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
    Dxgi::Common::{
        DXGI_FORMAT, DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_UNKNOWN, DXGI_SAMPLE_DESC,
    },
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
pub(super) const ROOT_EFFECT_TEXTURES: u32 = 6;
pub(super) const ROOT_PATH_CURVES: u32 = 7;
pub(super) const ROOT_PATH_PAINTS: u32 = 8;
/// The descriptors in a [`ROOT_TEXTURES`] table: t0, t2, and an effect
/// layer's content and history at t7 and t8.
pub(super) const TABLE_TEXTURES: usize = 4;
/// `[SkHistory]` state textures.
pub(super) const HISTORY_FORMAT: DXGI_FORMAT = DXGI_FORMAT_R16G16B16A16_FLOAT;

#[cfg(test)]
const ALPHA_CORRECTION: &str = include_str!("../alpha_correction.hlsl");
#[cfg(test)]
const PRIMITIVES: &str = include_str!("../shaders.hlsl");
#[cfg(test)]
const EFFECT_VERTEX: &str = include_str!("../effects.hlsl");
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
    /// vue-native: an app effect, by [`AppFragment::id`].
    AppEffect(u64),
    /// vue-native: `gpui::effect_id::PUNCH` — erases its quad (see [`PUNCH_FRAGMENT`]).
    Punch,
    /// vue-native: `gpui::effect_id::CONTENT` — an effect layer's content as it is.
    Content,
    /// vue-native: an app effect's `[SkHistory]` step, blending off.
    AppHistory(u64),
    /// vue-native: the layer composite blended `plus-lighter`.
    VnPlusLighter,
}

/// vue-native: an app effect's fragment stage, compiled when the app's effects
/// are installed or reloaded. A reload that changes the effect makes a new one,
/// so pipelines cached under the old `id` are never served for the new source.
struct AppFragment {
    id: u64,
    bytecode: Vec<u8>,
    needs_backdrop: bool,
    history: bool,
}

/// vue-native: the fragment stage of the punch. It shares the effects' vertex
/// stage and instance buffer and writes only a coverage; the pipeline's blend
/// ([`Blend::DestinationOut`]) multiplies what is in the target by one minus
/// it, so the quad's rounded rect, inside its content mask, becomes
/// transparent. The structs repeat `effects.hlsl`'s (the stride must match).
const PUNCH_FRAGMENT: &str = r#"
struct Bounds { float2 origin; float2 size; };
struct Corners { float top_left; float top_right; float bottom_right; float bottom_left; };
struct EffectQuad {
    uint order;
    uint effect_id;
    Bounds bounds;
    Bounds content_mask;
    Corners content_mask_radii;
    Corners corner_radii;
    float params[32];
    float opacity;
    float time;
    float bleed;
    float backdrop_blur_radius;
    float scale;
    uint flags;
};
StructuredBuffer<EffectQuad> effect_quads: register(t1);

float rounded_distance(float2 position, Bounds bounds, Corners radii) {
    float2 half_size = bounds.size * 0.5;
    float2 center = bounds.origin + half_size;
    float2 p = position - center;
    float radius = p.x < 0.0
        ? (p.y < 0.0 ? radii.top_left : radii.bottom_left)
        : (p.y < 0.0 ? radii.top_right : radii.bottom_right);
    radius = min(radius, min(half_size.x, half_size.y));
    float2 q = abs(p) - half_size + radius;
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - radius;
}

float4 punch_fragment(nointerpolation uint effect_quad_id: TEXCOORD0, float4 position: SV_Position) : SV_Target {
    EffectQuad quad = effect_quads[effect_quad_id];
    float inside = saturate(0.5 - rounded_distance(position.xy, quad.bounds, quad.corner_radii));
    float unmasked = saturate(0.5 - rounded_distance(position.xy, quad.content_mask, quad.content_mask_radii));
    return float4(0.0, 0.0, 0.0, inside * unmasked * quad.opacity);
}
"#;

/// vue-native: the fragment stage of `gpui::effect_id::CONTENT`, in the
/// effects' vertex source.
const CONTENT_FRAGMENT: &str = concat!(
    "#define SK_CONTENT_FRAGMENT\n",
    include_str!("../effects.hlsl")
);

/// vue-native: compiles [`PUNCH_FRAGMENT`]. At first use, like an app's own
/// effect: it is needed only by a window that shows an `<iframe>`.
fn compile_punch() -> Result<Vec<u8>> {
    compile_fragment(PUNCH_FRAGMENT, c"gpui-d3d12-punch", c"punch_fragment")
}

/// vue-native: compiles [`CONTENT_FRAGMENT`], at the first effect layer.
fn compile_content() -> Result<Vec<u8>> {
    compile_fragment(CONTENT_FRAGMENT, c"gpui-d3d12-content", c"content_fragment")
}

fn compile_fragment(
    source: &str,
    name: &std::ffi::CStr,
    entry: &std::ffi::CStr,
) -> Result<Vec<u8>> {
    use windows::Win32::Graphics::Direct3D::{
        Fxc::{D3DCOMPILE_ENABLE_STRICTNESS, D3DCOMPILE_OPTIMIZATION_LEVEL3, D3DCompile},
        ID3DInclude,
    };
    use windows::core::{PCSTR, s};
    let mut bytecode = None;
    let mut diagnostic = None;
    let result = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR(name.as_ptr().cast()),
            None,
            None::<&ID3DInclude>,
            PCSTR(entry.as_ptr().cast()),
            s!("ps_5_1"),
            D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
            0,
            &mut bytecode,
            Some(&mut diagnostic),
        )
    };
    if let Err(error) = result {
        bail!(
            "Compiling the D3D12 shader {}: {error}; {}",
            entry.to_string_lossy(),
            diagnostic.as_ref().map(blob_message).unwrap_or_default()
        );
    }
    let bytecode = bytecode.context("D3DCompile returned no bytecode")?;
    Ok(bytes(&bytecode).to_vec())
}

pub(super) struct Pipelines {
    pub root_signature: ID3D12RootSignature,
    device: ID3D12Device,
    cache: Vec<CachedPipeline>,
    /// Pipelines a background thread created ahead of their first use.
    warmed: Arc<Mutex<Vec<CachedPipeline>>>,
    /// Built at the first frame that draws an app effect.
    app_effects: gpui::AppEffectPipelines<AppFragment>,
    app_fragments_made: u64,
    timings: bool,
}

/// The pipelines a window's frames use, most common first. Creating a PSO the
/// driver has not cached yet (a new build or driver) takes tens of
/// milliseconds, which on first use landed inside a frame.
const WARM_MODULES: [(ShaderModule, u32); 14] = [
    (ShaderModule::Quad, 1),
    (ShaderModule::MonochromeSprite, 1),
    (ShaderModule::Shadow, 1),
    (ShaderModule::PolychromeSprite, 1),
    (ShaderModule::Underline, 1),
    (ShaderModule::PathTile, 1),
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
            app_effects: Default::default(),
            app_fragments_made: 0,
            timings: std::env::var_os("SK_DX12_TIMINGS").is_some_and(|value| value == "1"),
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

    /// The layer composite with `gpui::vn_layer_flags::PLUS_LIGHTER`'s blend.
    pub fn vn_plus_lighter(&mut self, format: DXGI_FORMAT) -> Result<ID3D12PipelineState> {
        self.get(PipelineKey {
            kind: PipelineKind::VnPlusLighter,
            format,
            samples: 1,
        })
    }

    /// The pipeline for `EffectQuad::effect_id` `index` and whether it reads the
    /// backdrop; None for an app effect that never built or an id past the
    /// app's effects, whose quads draw nothing.
    pub fn effect(
        &mut self,
        index: u32,
        format: DXGI_FORMAT,
        sample_count: u32,
    ) -> Result<Option<(ID3D12PipelineState, bool)>> {
        let (kind, backdrop) = if index == gpui::effect_id::PUNCH {
            (PipelineKind::Punch, false)
        } else if index == gpui::effect_id::CONTENT {
            (PipelineKind::Content, false)
        } else if let Some(effect) = EffectShader::ALL.get(index as usize) {
            (
                PipelineKind::Primitive(ShaderModule::Effect(*effect)),
                effect.needs_backdrop(),
            )
        } else {
            self.sync_app_effects(format);
            let Some(fragment) = self
                .app_effects
                .get(index as usize - EffectShader::ALL.len())
            else {
                return Ok(None);
            };
            (
                PipelineKind::AppEffect(fragment.id),
                fragment.needs_backdrop,
            )
        };
        let state = self.get(PipelineKey {
            kind,
            format,
            samples: sample_count,
        })?;
        Ok(Some((state, backdrop)))
    }

    /// The `[SkHistory]` step of app effect `index`, drawn into a
    /// [`HISTORY_FORMAT`] texture; None for an effect without history. Valid
    /// after [`Self::effect`] synced the app's effects this frame.
    pub fn history(&mut self, index: u32) -> Result<Option<ID3D12PipelineState>> {
        let Some(app_index) = (index as usize).checked_sub(EffectShader::ALL.len()) else {
            return Ok(None);
        };
        let Some(fragment) = self.app_effects.get(app_index) else {
            return Ok(None);
        };
        if !fragment.history {
            return Ok(None);
        }
        let kind = PipelineKind::AppHistory(fragment.id);
        self.get(PipelineKey {
            kind,
            format: HISTORY_FORMAT,
            samples: 1,
        })
        .map(Some)
    }

    /// Catches up with `gpui::reload_app_effects`. Each changed effect is
    /// compiled and its pipeline for `format` created here, so a shader that
    /// fails either step keeps the effect's previous pipeline.
    fn sync_app_effects(&mut self, format: DXGI_FORMAT) {
        let Self {
            device,
            root_signature,
            cache,
            app_effects,
            app_fragments_made,
            ..
        } = self;
        let synced = app_effects.sync(|effect| {
            let (file, source) = effect.hlsl.get("HLSL")?;
            let fragment = shader_resources::compile_hlsl_source(
                source,
                file,
                "effect_fragment\0",
                "ps_5_1\0",
            )
            .with_context(|| format!("compiling {file} failed"))?;
            let bytecode = bytes(&fragment).to_vec();
            let state = create_pipeline(
                device,
                root_signature,
                embedded::effect_vertex(),
                &bytecode,
                format,
                1,
                app_effect_blend(effect.needs_backdrop),
            )?;
            *app_fragments_made += 1;
            let id = *app_fragments_made;
            cache.push(CachedPipeline {
                key: PipelineKey {
                    kind: PipelineKind::AppEffect(id),
                    format,
                    samples: 1,
                },
                state,
            });
            if effect.history {
                let state = create_pipeline(
                    device,
                    root_signature,
                    embedded::effect_vertex(),
                    &bytecode,
                    HISTORY_FORMAT,
                    1,
                    Blend::Replace,
                )?;
                cache.push(CachedPipeline {
                    key: PipelineKey {
                        kind: PipelineKind::AppHistory(id),
                        format: HISTORY_FORMAT,
                        samples: 1,
                    },
                    state,
                });
            }
            Ok(AppFragment {
                id,
                bytecode,
                needs_backdrop: effect.needs_backdrop,
                history: effect.history,
            })
        });
        if synced {
            let live: Vec<u64> = app_effects
                .pipelines()
                .map(|fragment| fragment.id)
                .collect();
            cache.retain(|cached| match cached.key.kind {
                PipelineKind::AppEffect(id) | PipelineKind::AppHistory(id) => live.contains(&id),
                _ => true,
            });
        }
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
            PipelineKind::Punch => (
                Cow::Borrowed(embedded::effect_vertex()),
                Cow::Owned(compile_punch()?),
                Blend::DestinationOut,
            ),
            PipelineKind::Content => (
                Cow::Borrowed(embedded::effect_vertex()),
                Cow::Owned(compile_content()?),
                Blend::Premultiplied,
            ),
            PipelineKind::VnPlusLighter => {
                let (vertex, fragment) = embedded::module(ShaderModule::VnLayer);
                (Cow::Borrowed(vertex), Cow::Borrowed(fragment), Blend::PlusLighter)
            }
            PipelineKind::AppEffect(id) | PipelineKind::AppHistory(id) => {
                let fragment = self
                    .app_effects
                    .pipelines()
                    .find(|fragment| fragment.id == id)
                    .context("D3D12 app effect was replaced")?;
                let blend = match key.kind {
                    PipelineKind::AppHistory(_) => Blend::Replace,
                    _ => app_effect_blend(fragment.needs_backdrop),
                };
                (
                    Cow::Borrowed(embedded::effect_vertex()),
                    Cow::Borrowed(fragment.bytecode.as_slice()),
                    blend,
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
        D3D12_DESCRIPTOR_RANGE {
            RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
            NumDescriptors: 2,
            BaseShaderRegister: 7,
            RegisterSpace: 0,
            OffsetInDescriptorsFromTableStart: 2,
        },
    ];
    // t3..t6: an effect's `SkTexture` images.
    let effect_texture_range = D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: gpui::EFFECT_TEXTURE_SLOTS as u32,
        BaseShaderRegister: 3,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: 0,
    };
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
        descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV, 9),
        descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV, 10),
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
        D3D12_ROOT_PARAMETER {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
            Anonymous: D3D12_ROOT_PARAMETER_0 {
                DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                    NumDescriptorRanges: 1,
                    pDescriptorRanges: &effect_texture_range,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
        },
        descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV, 11),
        descriptor(D3D12_ROOT_PARAMETER_TYPE_SRV, 12),
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
        sampler(2, D3D12_TEXTURE_ADDRESS_MODE_WRAP),
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
    /// vue-native: the target times one minus the source's alpha — an eraser.
    DestinationOut,
    Straight,
    Premultiplied,
    Replace,
    Subpixel,
    /// vue-native: CSS `plus-lighter` over premultiplied colour.
    PlusLighter,
}

fn app_effect_blend(needs_backdrop: bool) -> Blend {
    if needs_backdrop {
        Blend::Replace
    } else {
        Blend::Premultiplied
    }
}

fn module_blend(module: ShaderModule) -> Blend {
    match module {
        ShaderModule::BlurDownsample | ShaderModule::BlurUpsample | ShaderModule::BlurRect => {
            Blend::Replace
        }
        ShaderModule::VnLayer => Blend::Premultiplied,
        ShaderModule::SubpixelSprite => Blend::Subpixel,
        ShaderModule::Effect(effect) if effect.needs_backdrop() => Blend::Replace,
        // `sk_finish` outputs premultiplied colour.
        ShaderModule::Effect(_) => Blend::Premultiplied,
        _ => Blend::Straight,
    }
}

fn blend_description(blend: Blend) -> D3D12_BLEND_DESC {
    let mut description = D3D12_BLEND_DESC::default();
    let target = &mut description.RenderTarget[0];
    target.BlendEnable = (!matches!(blend, Blend::Replace)).into();
    target.LogicOpEnable = false.into();
    target.SrcBlend = match blend {
        Blend::DestinationOut => D3D12_BLEND_ZERO,
        Blend::Straight => D3D12_BLEND_SRC_ALPHA,
        Blend::Subpixel => D3D12_BLEND_SRC1_COLOR,
        _ => D3D12_BLEND_ONE,
    };
    target.DestBlend = match blend {
        Blend::Subpixel => D3D12_BLEND_INV_SRC1_COLOR,
        Blend::PlusLighter => D3D12_BLEND_ONE,
        _ => D3D12_BLEND_INV_SRC_ALPHA,
    };
    target.BlendOp = D3D12_BLEND_OP_ADD;
    target.SrcBlendAlpha = if matches!(blend, Blend::DestinationOut) {
        D3D12_BLEND_ZERO
    } else {
        D3D12_BLEND_ONE
    };
    target.DestBlendAlpha = match blend {
        Blend::Subpixel => D3D12_BLEND_ZERO,
        Blend::PlusLighter => D3D12_BLEND_ONE,
        _ => D3D12_BLEND_INV_SRC_ALPHA,
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
        ShaderModule::PathTile => "path_tile",
        ShaderModule::MonochromeSprite => "monochrome_sprite",
        ShaderModule::SubpixelSprite => "subpixel_sprite",
        ShaderModule::PolychromeSprite => "polychrome_sprite",
        ShaderModule::VnLayer => "vn_layer",
        ShaderModule::Effect(_) => bail!("Effect shaders require their shared vertex stage"),
    };
    Ok((
        compile(PRIMITIVES, &format!("{prefix}_vertex"), true)?,
        compile(PRIMITIVES, &format!("{prefix}_fragment"), false)?,
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
