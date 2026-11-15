//! CSS `backdrop-filter: blur()` and the `--shading` effects, ported from the
//! DirectX and wgpu renderers: the same Dual-Kawase pyramid, the same
//! grouping, the same effect pipeline table.

use super::{
    BackdropInputIndex, BackdropTextureIndex, InstanceBinding, MetalRenderer,
    new_command_encoder_for_texture,
};
use anyhow::{Context as _, Result};
use gpui::{
    BackdropBlurRect, Bounds, DevicePixels, EffectQuad, MAX_BACKDROP_BLUR_KERNEL_LEVELS,
    ScaledPixels, Scene, Size, effect_id,
};
use objc2::{Message, rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlendFactor, MTLBlendOperation, MTLBlitCommandEncoder, MTLCommandBuffer, MTLCommandEncoder,
    MTLCompileOptions, MTLDevice, MTLFunction, MTLGPUFamily, MTLLibrary, MTLLoadAction, MTLOrigin,
    MTLPixelFormat, MTLPrimitiveType, MTLRegion, MTLRenderCommandEncoder, MTLRenderPassDescriptor,
    MTLRenderPipelineDescriptor, MTLRenderPipelineState, MTLResource, MTLSamplerAddressMode,
    MTLSamplerDescriptor, MTLSamplerMinMagFilter, MTLSamplerState, MTLScissorRect, MTLSize,
    MTLStorageMode, MTLStoreAction, MTLTexture, MTLTextureDescriptor, MTLTextureUsage,
};
use std::{mem, ops::Range, ptr};

include!(concat!(env!("OUT_DIR"), "/builtin_effects.rs"));

const _: () = {
    assert!(BUILTIN_EFFECT_COUNT == effect_id::COUNT as usize);
    assert!(effect_id::FROST == 0);
    assert!(effect_id::NOISE == 1);
    assert!(effect_id::GLOW == 2);
};

const BLUR_TEXTURE_LEVELS: usize = MAX_BACKDROP_BLUR_KERNEL_LEVELS as usize;

/// Frames the scratch targets may sit unused before release, as on D3D12:
/// recreating one is a window-sized allocation, so short gaps keep them.
const SCRATCH_RELEASE_FRAMES: u32 = 90;

/// The generated effect fragment stages' `[[sampler(1)]]`.
const CLAMP_SAMPLER_INDEX: usize = 1;
/// Their `[[sampler(2)]]`, which repeats an effect's images.
const REPEAT_SAMPLER_INDEX: usize = 2;
/// Their `[[texture(3)]]`..`[[texture(6)]]`: an effect's images.
const EFFECT_TEXTURE_INDEX: usize = 3;

/// Duplicated from the `sk-effects` registry, as in gpui_windows: gpui sits
/// below it. An app effect reads its manifest's `backdrop` field instead.
fn builtin_needs_backdrop(index: usize) -> bool {
    index == effect_id::FROST as usize
}

/// The DirectX `GlobalParams` constant buffer, which `modules/sk.slang`
/// declares for every target and the generated stages bind at `[[buffer(0)]]`.
/// They read only the viewport size; Metal has no text gamma to pass.
#[repr(C)]
struct EffectGlobals {
    gamma_ratios: [f32; 4],
    viewport_size: [f32; 2],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    is_bgr: u32,
    pad: [u32; 3],
}

const _: () = assert!(mem::size_of::<EffectGlobals>() == 48);

pub(super) type Texture = Retained<ProtocolObject<dyn MTLTexture>>;
pub(super) type Pipeline = Retained<ProtocolObject<dyn MTLRenderPipelineState>>;
pub(super) type Encoder = Retained<ProtocolObject<dyn MTLRenderCommandEncoder>>;

/// `[SkHistory]` state textures.
pub(super) const HISTORY_FORMAT: MTLPixelFormat = MTLPixelFormat::RGBA16Float;

#[derive(Default)]
pub(super) enum BackdropState {
    /// Built on the first frame that paints a backdrop primitive.
    #[default]
    Unbuilt,
    Ready(Box<MetalBackdrop>),
    Unavailable,
}

impl BackdropState {
    pub(super) fn ready(&self) -> Option<&MetalBackdrop> {
        match self {
            BackdropState::Ready(backdrop) => Some(backdrop),
            BackdropState::Unbuilt | BackdropState::Unavailable => None,
        }
    }
}

pub(super) struct MetalBackdrop {
    blur_downsample: Pipeline,
    blur_upsample: Pipeline,
    blur_rect: Pipeline,
    frame_copy: Pipeline,
    /// The built-ins, indexed by `EffectQuad::effect_id`. A hole keeps the
    /// numbering when one failed to build.
    effects: Vec<Option<Pipeline>>,
    /// The app's own, `effect_id::COUNT` on.
    app_effects: gpui::AppEffectPipelines<AppEffectPipeline>,
    effect_vertex: Retained<ProtocolObject<dyn MTLFunction>>,
    /// `gpui::effect_id::PUNCH`.
    punch: Option<Pipeline>,
    /// `gpui::effect_id::CONTENT`.
    content: Option<Pipeline>,
    clamp_sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    /// `[[sampler(2)]]`, for an effect's images.
    repeat_sampler: Retained<ProtocolObject<dyn MTLSamplerState>>,
    /// Bound for an effect that does not read the backdrop, or misses an
    /// image: its shader still declares the textures.
    placeholder: Texture,
    pub(super) histories: gpui::VnEffectHistories<Texture>,
    effect_textures: gpui::EffectTextureCache<Texture>,
    /// This frame's `Scene::effect_texture_sets`, uploaded.
    frame_effect_textures: Vec<[Option<Texture>; gpui::EFFECT_TEXTURE_SLOTS]>,
    targets: Option<BackdropTargets>,
    /// Consecutive frames that held more scratch targets than they needed.
    idle_frames: u32,
}

struct AppEffectPipeline {
    shade: Pipeline,
    /// `[SkHistory]`: the step, into a history texture with blending off.
    history: Option<Pipeline>,
}

/// Sized to the frame.
struct BackdropTargets {
    width: usize,
    height: usize,
    snapshot: Texture,
    /// Pyramid level `i` is the frame halved `i + 1` times, allocated up to the
    /// deepest level a frame has asked for.
    levels: Vec<Texture>,
    frame: Option<Texture>,
}

/// Where a frame renders. A backdrop copies what has been drawn so far, which
/// a framebuffer-only drawable does not allow; such a frame is drawn into an
/// intermediate and copied onto the drawable at the end.
pub(super) struct FrameTarget {
    pub(super) texture: Texture,
    copy_to: Option<Texture>,
    /// Whether this frame can draw backdrop blurs and effects at all.
    pub(super) backdrop: bool,
}

impl FrameTarget {
    pub(super) fn layer(texture: Texture, backdrop: bool) -> Self {
        Self {
            texture,
            copy_to: None,
            backdrop,
        }
    }
}

pub(super) enum EffectPass {
    Skip,
    /// Source-over inside the current pass.
    Blended,
    /// Breaks the pass for a snapshot, like a backdrop blur.
    Backdrop,
}

/// What a backdrop composite samples: the processed backdrop and the
/// untouched snapshot.
pub(super) struct BackdropTextures {
    processed: Texture,
    original: Texture,
}

impl MetalRenderer {
    pub(super) fn prepare_frame_target(
        &mut self,
        scene: &Scene,
        target: &ProtocolObject<dyn MTLTexture>,
    ) -> FrameTarget {
        let has_primitives = !scene.backdrop_blur_rects.is_empty()
            || !scene.effect_quads.is_empty()
            || !scene.vn_effect_draws.is_empty();
        let backdrop = has_primitives && self.ensure_backdrop();
        let mut frame = FrameTarget {
            texture: target.retain(),
            copy_to: None,
            backdrop,
        };
        let effect_quads = || {
            scene
                .effect_quads
                .iter()
                .chain(scene.vn_effect_draws.iter().map(|draw| &draw.quad))
        };
        let snapshots =
            !scene.backdrop_blur_rects.is_empty() || effect_quads().any(EffectQuad::needs_backdrop);
        if let BackdropState::Ready(state) = &mut self.backdrop {
            state.sync_app_effects(&self.device);
            state.upload_effect_textures(&self.device, scene);
            let kernel_levels = scene
                .backdrop_blur_rects
                .iter()
                .map(BackdropBlurRect::effective_kernel_levels)
                .chain(effect_quads().map(EffectQuad::effective_kernel_levels))
                .max()
                .unwrap_or(0) as usize;
            state.trim_targets(if backdrop && snapshots {
                kernel_levels.min(BLUR_TEXTURE_LEVELS) + 1
            } else {
                0
            });
            if state.targets.is_none() && self.framebuffer_only_lifted {
                self.framebuffer_only_lifted = false;
                if let Some(layer) = &self.layer {
                    layer.setFramebufferOnly(true);
                }
            }
        }
        if !backdrop || !snapshots {
            return frame;
        }

        let framebuffer_only = target.isFramebufferOnly();
        #[cfg(test)]
        let framebuffer_only = framebuffer_only || tests::FORCE_INTERMEDIATE_FRAME.get();
        if !framebuffer_only {
            return frame;
        }
        // Drawables vended from now on can be copied from; this one cannot.
        if let Some(layer) = &self.layer
            && layer.framebufferOnly()
        {
            layer.setFramebufferOnly(false);
            self.framebuffer_only_lifted = true;
        }
        let BackdropState::Ready(state) = &mut self.backdrop else {
            return frame;
        };
        match state.frame_texture(&self.device, target.width(), target.height()) {
            Ok(intermediate) => {
                frame.texture = intermediate;
                frame.copy_to = Some(target.retain());
            }
            Err(error) => {
                effects_advisory(&format!(
                    "`backdrop-filter` and `--shading` are skipped on a frame — its \
                     intermediate target could not be allocated: {error:#}"
                ));
                frame.backdrop = false;
            }
        }
        frame
    }

    /// Copies an intermediate frame onto the drawable it stood in for.
    pub(super) fn finish_frame_target(
        &self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        frame: &FrameTarget,
    ) -> Result<()> {
        let Some(target) = &frame.copy_to else {
            return Ok(());
        };
        let state = self
            .backdrop
            .ready()
            .context("backdrop pipelines missing")?;
        // Every pixel is overwritten with blending disabled.
        let encoder = render_pass(command_buffer, target, MTLLoadAction::DontCare)?;
        encoder.setRenderPipelineState(&state.frame_copy);
        // SAFETY: The retained intermediate matches the shader's texture slot,
        // and the strip is the four vertices the vertex stage derives.
        unsafe {
            encoder.setFragmentTexture_atIndex(
                Some(&frame.texture),
                BackdropTextureIndex::Processed as usize,
            );
            encoder.drawPrimitives_vertexStart_vertexCount(MTLPrimitiveType::TriangleStrip, 0, 4);
        }
        encoder.endEncoding();
        Ok(())
    }

    pub(super) fn ensure_backdrop(&mut self) -> bool {
        if matches!(self.backdrop, BackdropState::Unbuilt) {
            self.backdrop = match MetalBackdrop::new(&self.device, &self.library) {
                Ok(backdrop) => BackdropState::Ready(Box::new(backdrop)),
                Err(error) => {
                    effects_advisory(&format!(
                        "`backdrop-filter` and `--shading` are unavailable — their pipelines \
                         failed to build: {error:#}"
                    ));
                    BackdropState::Unavailable
                }
            };
        }
        matches!(self.backdrop, BackdropState::Ready(_))
    }

    /// CSS `backdrop-filter: blur()`, grouped as DirectX groups it: a group
    /// shares one kernel depth and holds no two overlapping rects, and each
    /// group snapshots the frame again, so glass over glass sees the glass
    /// below it.
    ///
    /// Takes the frame's open encoder and returns the one the frame continues
    /// in; on an error, no encoder is left open.
    pub(super) fn draw_backdrop_blur_rects(
        &mut self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        mut encoder: Encoder,
        frame: &FrameTarget,
        blur_rects: &[BackdropBlurRect],
        start: usize,
        instances: &InstanceBinding,
        viewport_size: Size<DevicePixels>,
    ) -> Result<Encoder> {
        for group in backdrop_groups(blur_rects, |rect| {
            (rect.effective_kernel_levels() as usize, rect.bounds)
        }) {
            let kernel_levels = blur_rects[group.start].effective_kernel_levels() as usize;
            let Some(union) = blur_rects[group.clone()]
                .iter()
                .map(|rect| rect.bounds)
                .reduce(|union, bounds| union.union(&bounds))
            else {
                continue;
            };
            let frame_rect = PixelRect::of(&frame.texture);
            if PixelRect::round_out(&union, 0)
                .intersect(frame_rect)
                .is_empty()
            {
                continue;
            }
            // Only the pixels the composite reads, plus the kernel's reach.
            let region = PixelRect::round_out(&union, blur_reach(kernel_levels));
            #[cfg(test)]
            let region = (!tests::FULL_WINDOW_BACKDROP.get()).then_some(region);
            #[cfg(not(test))]
            let region = Some(region);

            encoder.endEncoding();
            let textures = self.snapshot_backdrop(command_buffer, frame, kernel_levels, region)?;
            let state = self
                .backdrop
                .ready()
                .context("backdrop pipelines missing")?;
            encoder = new_command_encoder_for_texture(
                command_buffer,
                &frame.texture,
                viewport_size,
                None,
            )?;
            state.draw_instances(
                &encoder,
                &state.blur_rect,
                instances,
                viewport_size,
                Some(&textures),
                [None, None],
                None,
                start + group.start..start + group.end,
            );
        }
        Ok(encoder)
    }

    pub(super) fn effect_pass(
        &self,
        frame: &FrameTarget,
        effect_id: u32,
        effect_quads: &[EffectQuad],
    ) -> EffectPass {
        let Some(first) = effect_quads.first() else {
            return EffectPass::Skip;
        };
        // A missing pipeline was announced when the table was built; every
        // other style on the element still renders.
        let has_pipeline = frame.backdrop
            && self
                .backdrop
                .ready()
                .and_then(|state| state.effect(effect_id))
                .is_some();
        if !has_pipeline {
            EffectPass::Skip
        } else if first.needs_backdrop() {
            EffectPass::Backdrop
        } else {
            EffectPass::Blended
        }
    }

    pub(super) fn draw_blended_effects(
        &self,
        effect_id: u32,
        textures: Option<usize>,
        range: Range<usize>,
        instances: &InstanceBinding,
        viewport_size: Size<DevicePixels>,
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    ) {
        let Some(state) = self.backdrop.ready() else {
            return;
        };
        let Some(pipeline) = state.effect(effect_id) else {
            return;
        };
        state.draw_instances(
            encoder,
            pipeline,
            instances,
            viewport_size,
            None,
            [None, None],
            Some(state.images(textures)),
            range,
        );
    }

    /// An effect that reads the backdrop takes the backdrop blur's path:
    /// grouped, a snapshot and pyramid per group, then a composite with
    /// blending disabled. Encoders as in [`Self::draw_backdrop_blur_rects`].
    pub(super) fn draw_backdrop_effects(
        &mut self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        mut encoder: Encoder,
        frame: &FrameTarget,
        effect_id: u32,
        effect_quads: &[EffectQuad],
        start: usize,
        instances: &InstanceBinding,
        viewport_size: Size<DevicePixels>,
    ) -> Result<Encoder> {
        for group in backdrop_groups(effect_quads, |quad| {
            (quad.effective_kernel_levels() as usize, quad.bounds)
        }) {
            let kernel_levels = effect_quads[group.start].effective_kernel_levels() as usize;
            encoder.endEncoding();
            // An effect may sample its backdrop anywhere, as on D3D12.
            let textures = self.snapshot_backdrop(command_buffer, frame, kernel_levels, None)?;
            let state = self
                .backdrop
                .ready()
                .context("backdrop pipelines missing")?;
            let pipeline = state.effect(effect_id).context("effect pipeline missing")?;
            encoder = new_command_encoder_for_texture(
                command_buffer,
                &frame.texture,
                viewport_size,
                None,
            )?;
            state.draw_instances(
                &encoder,
                pipeline,
                instances,
                viewport_size,
                Some(&textures),
                [None, None],
                Some(state.images(effect_quads[0].texture_set())),
                start + group.start..start + group.end,
            );
        }
        Ok(encoder)
    }

    /// The backdrop of an effect layer's effect, which may sample it anywhere.
    /// Called between encoders.
    pub(super) fn snapshot_layer_backdrop(
        &mut self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        frame: &FrameTarget,
        kernel_levels: usize,
    ) -> Result<BackdropTextures> {
        self.snapshot_backdrop(command_buffer, frame, kernel_levels, None)
    }

    /// Copies what the frame holds so far and blurs it `kernel_levels` deep,
    /// within `region` (frame pixels, the kernel's reach included) or the
    /// whole frame. Called between encoders: it opens and ends its own.
    fn snapshot_backdrop(
        &mut self,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        frame: &FrameTarget,
        kernel_levels: usize,
        region: Option<PixelRect>,
    ) -> Result<BackdropTextures> {
        let kernel_levels = kernel_levels.min(BLUR_TEXTURE_LEVELS);
        let BackdropState::Ready(state) = &mut self.backdrop else {
            anyhow::bail!("backdrop pipelines missing");
        };
        state.ensure_targets(
            &self.device,
            frame.texture.width(),
            frame.texture.height(),
            kernel_levels,
        )?;
        let state = &**state;
        let targets = state.targets.as_ref().context("backdrop targets missing")?;
        let full = PixelRect::of(&frame.texture);
        let region = region.map_or(full, |region| region.intersect(full));
        anyhow::ensure!(!region.is_empty(), "empty backdrop region");

        let blit = command_buffer
            .blitCommandEncoder()
            .context("creating backdrop blit encoder")?;
        let origin = MTLOrigin {
            x: region.left as usize,
            y: region.top as usize,
            z: 0,
        };
        // SAFETY: Both textures are retained BGRA8Unorm 2D textures of the
        // frame's size, and the region lies within them.
        unsafe {
            blit.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin(
                &frame.texture,
                0,
                0,
                origin,
                MTLSize {
                    width: (region.right - region.left) as usize,
                    height: (region.bottom - region.top) as usize,
                    depth: 1,
                },
                &targets.snapshot,
                0,
                0,
                origin,
            );
        }
        blit.endEncoding();

        // Like D3D12, a pass writes only its scissor and leaves the rest of
        // the level as it was; the rounded-out rects keep every texel the
        // next pass reads inside what this one wrote.
        let blur_pass = |pipeline: &Pipeline,
                         source: &Texture,
                         destination: &Texture,
                         scale: usize|
         -> Result<()> {
            let rect = region
                .scale_down(scale)
                .intersect(PixelRect::of(destination));
            if rect.is_empty() {
                return Ok(());
            }
            #[cfg(test)]
            tests::BLUR_PASS_RECTS.with_borrow_mut(|rects| rects.push(rect));
            let encoder = render_pass(command_buffer, destination, MTLLoadAction::Load)?;
            encoder.setRenderPipelineState(pipeline);
            encoder.setScissorRect(MTLScissorRect {
                x: rect.left as usize,
                y: rect.top as usize,
                width: (rect.right - rect.left) as usize,
                height: (rect.bottom - rect.top) as usize,
            });
            // SAFETY: The retained source matches the shader's texture slot,
            // and the strip is the four vertices the vertex stage derives.
            unsafe {
                encoder.setFragmentTexture_atIndex(
                    Some(source),
                    BackdropTextureIndex::Processed as usize,
                );
                encoder.drawPrimitives_vertexStart_vertexCount(
                    MTLPrimitiveType::TriangleStrip,
                    0,
                    4,
                );
            }
            encoder.endEncoding();
            Ok(())
        };

        let processed = if kernel_levels == 0 {
            &targets.snapshot
        } else {
            let levels = targets
                .levels
                .get(..kernel_levels)
                .context("missing blur pyramid levels")?;
            blur_pass(&state.blur_downsample, &targets.snapshot, &levels[0], 1)?;
            for level in 1..kernel_levels {
                blur_pass(
                    &state.blur_downsample,
                    &levels[level - 1],
                    &levels[level],
                    level + 1,
                )?;
            }
            for level in (1..kernel_levels).rev() {
                blur_pass(
                    &state.blur_upsample,
                    &levels[level],
                    &levels[level - 1],
                    level,
                )?;
            }
            &levels[0]
        };
        Ok(BackdropTextures {
            processed: processed.clone(),
            original: targets.snapshot.clone(),
        })
    }
}

/// A render pass over all of `texture`, stored.
pub(super) fn render_pass(
    command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
    texture: &ProtocolObject<dyn MTLTexture>,
    load: MTLLoadAction,
) -> Result<Encoder> {
    let descriptor = MTLRenderPassDescriptor::new();
    // SAFETY: Index zero is within the render pass's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setTexture(Some(texture));
    color_attachment.setLoadAction(load);
    color_attachment.setStoreAction(MTLStoreAction::Store);
    command_buffer
        .renderCommandEncoderWithDescriptor(&descriptor)
        .context("creating backdrop render command encoder")
}

/// Whole device pixels, edges exclusive on the right and bottom; D3D12's
/// `RECT` helpers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PixelRect {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}

impl PixelRect {
    fn of(texture: &ProtocolObject<dyn MTLTexture>) -> Self {
        Self {
            left: 0,
            top: 0,
            right: texture.width() as i32,
            bottom: texture.height() as i32,
        }
    }

    /// The pixels covering `bounds`, grown by `margin` on every side.
    fn round_out(bounds: &Bounds<ScaledPixels>, margin: i32) -> Self {
        let clamp = |value: f32| value.clamp(i32::MIN as f32 / 2., i32::MAX as f32 / 2.) as i32;
        Self {
            left: clamp(bounds.origin.x.0.floor()).saturating_sub(margin),
            top: clamp(bounds.origin.y.0.floor()).saturating_sub(margin),
            right: clamp((bounds.origin.x.0 + bounds.size.width.0).ceil()).saturating_add(margin),
            bottom: clamp((bounds.origin.y.0 + bounds.size.height.0).ceil()).saturating_add(margin),
        }
    }

    fn intersect(self, other: Self) -> Self {
        Self {
            left: self.left.max(other.left),
            top: self.top.max(other.top),
            right: self.right.min(other.right),
            bottom: self.bottom.min(other.bottom),
        }
    }

    fn is_empty(&self) -> bool {
        self.left >= self.right || self.top >= self.bottom
    }

    /// This rect in a pyramid level `2^level` times smaller, rounded out by a
    /// texel.
    fn scale_down(self, level: usize) -> Self {
        let divisor = 1i32 << level;
        Self {
            left: self.left.div_euclid(divisor) - 1,
            top: self.top.div_euclid(divisor) - 1,
            right: (self.right + divisor - 1).div_euclid(divisor) + 1,
            bottom: (self.bottom + divisor - 1).div_euclid(divisor) + 1,
        }
    }
}

/// Frame pixels a blur of `levels` pyramid levels can carry into a pixel; the
/// D3D12 renderer's bound. Each pass reads a texel to either side of its own,
/// `2 * 2^level` frame pixels at level `level`, both chains together stay
/// below `4 * 2^levels`, and the composite's bilinear read adds a few more.
fn blur_reach(levels: usize) -> i32 {
    (4 << levels) + 8
}

/// Splits a batch into runs that share a kernel depth and do not overlap.
fn backdrop_groups<T>(
    primitives: &[T],
    key: impl Fn(&T) -> (usize, Bounds<ScaledPixels>),
) -> Vec<Range<usize>> {
    let mut groups = Vec::new();
    let mut group_start = 0;
    while group_start < primitives.len() {
        let (kernel_levels, _) = key(&primitives[group_start]);
        let mut group_end = group_start + 1;
        while group_end < primitives.len() {
            let (candidate_levels, candidate_bounds) = key(&primitives[group_end]);
            if candidate_levels != kernel_levels
                || primitives[group_start..group_end]
                    .iter()
                    .any(|previous| key(previous).1.intersects(&candidate_bounds))
            {
                break;
            }
            group_end += 1;
        }
        groups.push(group_start..group_end);
        group_start = group_end;
    }
    groups
}

#[derive(Clone, Copy)]
enum Blend {
    /// The pipeline rewrites every pixel it covers.
    Off,
    /// `sk_finish`'s premultiplied colour, drawn over: One, OneMinusSrcAlpha.
    Premultiplied,
    /// D3D12's `Blend::DestinationOut`: the target times one minus the
    /// source's alpha, an eraser.
    DestinationOut,
}

impl MetalBackdrop {
    fn new(
        device: &ProtocolObject<dyn MTLDevice>,
        library: &ProtocolObject<dyn MTLLibrary>,
    ) -> Result<Self> {
        let function = |name: &str| -> Result<Retained<ProtocolObject<dyn MTLFunction>>> {
            library
                .newFunctionWithName(&NSString::from_str(name))
                .with_context(|| format!("locating function {name}"))
        };
        let pass_vertex = function("backdrop_pass_vertex")?;
        let pass = |label: &str, fragment: &str| {
            let fragment = function(fragment)?;
            backdrop_pipeline(device, label, &pass_vertex, &fragment, Blend::Off)
        };
        let blur_downsample = pass("backdrop_blur_downsample", "blur_downsample_fragment")?;
        let blur_upsample = pass("backdrop_blur_upsample", "blur_upsample_fragment")?;
        let frame_copy = pass("backdrop_frame_copy", "frame_copy_fragment")?;
        let blur_rect_vertex = function("blur_rect_vertex")?;
        let blur_rect_fragment = function("blur_rect_fragment")?;
        let blur_rect = backdrop_pipeline(
            device,
            "backdrop_blur_rect",
            &blur_rect_vertex,
            &blur_rect_fragment,
            Blend::Off,
        )?;

        let effect_vertex = function("effect_vertex")?;
        let effects = build_builtin_effect_pipelines(device, &effect_vertex);
        let punch = function("punch_fragment")
            .and_then(|fragment| {
                backdrop_pipeline(
                    device,
                    "effect_punch",
                    &effect_vertex,
                    &fragment,
                    Blend::DestinationOut,
                )
            })
            .inspect_err(|error| {
                effects_advisory(&format!(
                    "`<iframe>` documents cannot show through — the punch pipeline failed to \
                     build: {error:#}"
                ));
            })
            .ok();
        let content = function("content_fragment")
            .and_then(|fragment| {
                backdrop_pipeline(
                    device,
                    "effect_content",
                    &effect_vertex,
                    &fragment,
                    Blend::Premultiplied,
                )
            })
            .inspect_err(|error| {
                effects_advisory(&format!(
                    "an effect whose pipeline is missing hides its element — the content \
                     pipeline failed to build: {error:#}"
                ));
            })
            .ok();

        let sampler_descriptor = MTLSamplerDescriptor::new();
        sampler_descriptor.setMinFilter(MTLSamplerMinMagFilter::Linear);
        sampler_descriptor.setMagFilter(MTLSamplerMinMagFilter::Linear);
        sampler_descriptor.setSAddressMode(MTLSamplerAddressMode::ClampToEdge);
        sampler_descriptor.setTAddressMode(MTLSamplerAddressMode::ClampToEdge);
        let clamp_sampler = device
            .newSamplerStateWithDescriptor(&sampler_descriptor)
            .context("creating the backdrop sampler")?;
        let placeholder = create_target_texture(
            device,
            "backdrop_placeholder",
            1,
            1,
            MTLTextureUsage::ShaderRead,
        )?;

        sampler_descriptor.setSAddressMode(MTLSamplerAddressMode::Repeat);
        sampler_descriptor.setTAddressMode(MTLSamplerAddressMode::Repeat);
        let repeat_sampler = device
            .newSamplerStateWithDescriptor(&sampler_descriptor)
            .context("creating the effect image sampler")?;

        let mut backdrop = Self {
            blur_downsample,
            blur_upsample,
            blur_rect,
            frame_copy,
            effects,
            app_effects: Default::default(),
            effect_vertex,
            punch,
            content,
            clamp_sampler,
            repeat_sampler,
            placeholder,
            histories: Default::default(),
            effect_textures: Default::default(),
            frame_effect_textures: Vec::new(),
            targets: None,
            idle_frames: 0,
        };
        backdrop.sync_app_effects(device);
        Ok(backdrop)
    }

    /// The images an effect batch binds at 3..6: its texture set's, the
    /// placeholder for a missing one.
    pub(super) fn images(&self, set: Option<usize>) -> [&Texture; gpui::EFFECT_TEXTURE_SLOTS] {
        let images = set.and_then(|set| self.frame_effect_textures.get(set));
        std::array::from_fn(|slot| {
            images
                .and_then(|images| images[slot].as_ref())
                .unwrap_or(&self.placeholder)
        })
    }

    /// Uploads the images this frame's effects sample that are not on the GPU yet.
    fn upload_effect_textures(&mut self, device: &ProtocolObject<dyn MTLDevice>, scene: &Scene) {
        self.effect_textures.begin_frame();
        let cache = &mut self.effect_textures;
        self.frame_effect_textures = scene
            .effect_texture_sets
            .iter()
            .map(|set| {
                std::array::from_fn(|slot| {
                    let texture = set[slot].as_ref()?;
                    cache
                        .get(texture, |width, height, texels| {
                            upload_effect_texture(device, width, height, texels)
                        })
                        .cloned()
                })
            })
            .collect();
    }

    pub(super) fn sync_app_effects(&mut self, device: &ProtocolObject<dyn MTLDevice>) {
        let vertex = &self.effect_vertex;
        self.app_effects.sync(|effect| {
            let (_, msl) = effect.metal.get("Metal shader")?;
            let library = effect_library(device, msl)?;
            let shade = effect_pipeline(
                device,
                vertex,
                &library,
                &effect.name,
                MTLPixelFormat::BGRA8Unorm,
                effect.needs_backdrop,
            )?;
            let history = effect
                .history
                .then(|| {
                    effect_pipeline(
                        device,
                        vertex,
                        &library,
                        &format!("{}_history", effect.name),
                        HISTORY_FORMAT,
                        true,
                    )
                })
                .transpose()?;
            Ok(AppEffectPipeline { shade, history })
        });
    }

    /// `needed` counts the snapshot as one more level below the pyramid.
    fn trim_targets(&mut self, needed: usize) {
        let held = self
            .targets
            .as_ref()
            .map_or(0, |targets| targets.levels.len() + 1);
        if needed >= held {
            self.idle_frames = 0;
            return;
        }
        self.idle_frames += 1;
        if self.idle_frames < SCRATCH_RELEASE_FRAMES {
            return;
        }
        self.idle_frames = 0;
        match (needed, &mut self.targets) {
            (0, targets) => *targets = None,
            (needed, Some(targets)) => targets.levels.truncate(needed - 1),
            (_, None) => {}
        }
    }

    pub(super) fn effect(&self, effect_id: u32) -> Option<&Pipeline> {
        match effect_id {
            effect_id::PUNCH => self.punch.as_ref(),
            effect_id::CONTENT => self.content.as_ref(),
            id if id < effect_id::COUNT => self.effects.get(id as usize)?.as_ref(),
            id => Some(
                &self
                    .app_effects
                    .get((id - effect_id::COUNT) as usize)?
                    .shade,
            ),
        }
    }

    pub(super) fn history_effect(&self, effect_id: u32) -> Option<&Pipeline> {
        let index = effect_id.checked_sub(effect_id::COUNT)?;
        self.app_effects.get(index as usize)?.history.as_ref()
    }

    fn ensure_targets(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        width: usize,
        height: usize,
        kernel_levels: usize,
    ) -> Result<&mut BackdropTargets> {
        let targets = match self
            .targets
            .take()
            .filter(|targets| targets.width == width && targets.height == height)
        {
            Some(targets) => targets,
            None => BackdropTargets {
                width,
                height,
                snapshot: create_target_texture(
                    device,
                    "backdrop_snapshot",
                    width,
                    height,
                    MTLTextureUsage::ShaderRead,
                )?,
                levels: Vec::new(),
                frame: None,
            },
        };
        let targets = self.targets.insert(targets);
        while targets.levels.len() < kernel_levels {
            let divisor = 1usize << (targets.levels.len() + 1);
            targets.levels.push(create_target_texture(
                device,
                "backdrop_blur_level",
                (width / divisor).max(1),
                (height / divisor).max(1),
                MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
            )?);
        }
        Ok(targets)
    }

    fn frame_texture(
        &mut self,
        device: &ProtocolObject<dyn MTLDevice>,
        width: usize,
        height: usize,
    ) -> Result<Texture> {
        let targets = self.ensure_targets(device, width, height, 0)?;
        if let Some(frame) = &targets.frame {
            return Ok(frame.clone());
        }
        let frame = create_target_texture(
            device,
            "backdrop_frame",
            width,
            height,
            MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
        )?;
        targets.frame = Some(frame.clone());
        Ok(frame)
    }

    /// Draws `range` of the frame's `instances` (effect quads or blur rects)
    /// with every slot the backdrop pipelines and the generated effect stages
    /// declare bound; `layer` is an effect layer's content and history.
    pub(super) fn draw_instances(
        &self,
        encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
        pipeline: &ProtocolObject<dyn MTLRenderPipelineState>,
        instances: &InstanceBinding,
        viewport_size: Size<DevicePixels>,
        textures: Option<&BackdropTextures>,
        layer: [Option<&Texture>; 2],
        images: Option<[&Texture; gpui::EFFECT_TEXTURE_SLOTS]>,
        range: Range<usize>,
    ) {
        if range.is_empty() {
            return;
        }
        let globals = EffectGlobals {
            gamma_ratios: [0.; 4],
            viewport_size: [viewport_size.width.0 as f32, viewport_size.height.0 as f32],
            grayscale_enhanced_contrast: 0.,
            subpixel_enhanced_contrast: 0.,
            is_bgr: 0,
            pad: [0; 3],
        };
        let (processed, original) = match textures {
            Some(textures) => (&textures.processed, &textures.original),
            None => (&self.placeholder, &self.placeholder),
        };
        encoder.setRenderPipelineState(pipeline);
        // SAFETY: The binding holds the scene's whole instance array at an
        // aligned offset and `range` lies within it; Metal retains the buffer,
        // textures and sampler, and copies the globals and viewport bytes from
        // live values into the slots the shaders declare.
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(
                Some(&instances.buffer),
                instances.offset,
                BackdropInputIndex::Instances as usize,
            );
            encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                BackdropInputIndex::ViewportSize as usize,
            );
            encoder.setFragmentBytes_length_atIndex(
                ptr::NonNull::from(&globals).cast(),
                mem::size_of_val(&globals),
                BackdropInputIndex::Globals as usize,
            );
            encoder.setFragmentBuffer_offset_atIndex(
                Some(&instances.buffer),
                instances.offset,
                BackdropInputIndex::Instances as usize,
            );
            encoder.setFragmentBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                BackdropInputIndex::ViewportSize as usize,
            );
            encoder.setFragmentTexture_atIndex(
                Some(processed),
                BackdropTextureIndex::Processed as usize,
            );
            encoder.setFragmentTexture_atIndex(
                Some(original),
                BackdropTextureIndex::Original as usize,
            );
            encoder.setFragmentTexture_atIndex(
                Some(layer[0].unwrap_or(&self.placeholder)),
                BackdropTextureIndex::Content as usize,
            );
            encoder.setFragmentTexture_atIndex(
                Some(layer[1].unwrap_or(&self.placeholder)),
                BackdropTextureIndex::History as usize,
            );
            encoder.setFragmentSamplerState_atIndex(Some(&self.clamp_sampler), CLAMP_SAMPLER_INDEX);
            if let Some(images) = images {
                for (slot, image) in images.into_iter().enumerate() {
                    encoder.setFragmentTexture_atIndex(Some(image), EFFECT_TEXTURE_INDEX + slot);
                }
                encoder.setFragmentSamplerState_atIndex(
                    Some(&self.repeat_sampler),
                    REPEAT_SAMPLER_INDEX,
                );
            }
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                range.len(),
                range.start,
            );
        }
    }
}

/// The built-in `--shading` pipelines in `gpui::effect_id` order. Infallible as
/// a whole, as on DirectX: one that does not build leaves a hole and a one-shot
/// advisory.
fn build_builtin_effect_pipelines(
    device: &ProtocolObject<dyn MTLDevice>,
    vertex: &ProtocolObject<dyn MTLFunction>,
) -> Vec<Option<Pipeline>> {
    let mut pipelines = Vec::new();
    for (index, &(name, msl)) in BUILTIN_EFFECT_MSL.iter().enumerate() {
        let pipeline = msl
            .context("this build carries no shader for it")
            .and_then(|msl| {
                frame_effect_pipeline(device, vertex, name, msl, builtin_needs_backdrop(index))
            });
        pipelines.push(match pipeline {
            Ok(pipeline) => Some(pipeline),
            Err(error) => {
                effects_advisory(&format!(
                    "`--shading: {name}(…)` is unavailable — its pipeline failed to build: {error:#}"
                ));
                None
            }
        });
    }
    pipelines
}

/// Compiles an effect's generated MSL at runtime: neither the built-ins nor
/// an app's effects need the Metal toolchain.
fn effect_library(
    device: &ProtocolObject<dyn MTLDevice>,
    msl: &str,
) -> Result<Retained<ProtocolObject<dyn MTLLibrary>>> {
    device
        .newLibraryWithSource_options_error(
            &NSString::from_str(msl),
            Some(&MTLCompileOptions::new()),
        )
        .context("its Metal source does not compile")
}

/// An effect drawn into the frame, from its MSL.
fn frame_effect_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vertex: &ProtocolObject<dyn MTLFunction>,
    name: &str,
    msl: &str,
    needs_backdrop: bool,
) -> Result<Pipeline> {
    let library = effect_library(device, msl)?;
    effect_pipeline(
        device,
        vertex,
        &library,
        name,
        MTLPixelFormat::BGRA8Unorm,
        needs_backdrop,
    )
}

fn effect_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    vertex: &ProtocolObject<dyn MTLFunction>,
    library: &ProtocolObject<dyn MTLLibrary>,
    name: &str,
    format: MTLPixelFormat,
    needs_backdrop: bool,
) -> Result<Pipeline> {
    let fragment = library
        .newFunctionWithName(&NSString::from_str("effect_fragment"))
        .context("its Metal source has no `effect_fragment`")?;
    // A backdrop effect rewrites every pixel of its quad from its own
    // snapshot. Any other is `sk_finish`'s premultiplied colour, drawn over.
    let blend = if needs_backdrop {
        Blend::Off
    } else {
        Blend::Premultiplied
    };
    pipeline_in(
        device,
        &format!("effect_{name}"),
        vertex,
        &fragment,
        format,
        blend,
    )
}

fn backdrop_pipeline(
    device: &ProtocolObject<dyn MTLDevice>,
    label: &str,
    vertex: &ProtocolObject<dyn MTLFunction>,
    fragment: &ProtocolObject<dyn MTLFunction>,
    blend: Blend,
) -> Result<Pipeline> {
    pipeline_in(
        device,
        label,
        vertex,
        fragment,
        MTLPixelFormat::BGRA8Unorm,
        blend,
    )
}

fn pipeline_in(
    device: &ProtocolObject<dyn MTLDevice>,
    label: &str,
    vertex: &ProtocolObject<dyn MTLFunction>,
    fragment: &ProtocolObject<dyn MTLFunction>,
    format: MTLPixelFormat,
    blend: Blend,
) -> Result<Pipeline> {
    let descriptor = MTLRenderPipelineDescriptor::new();
    descriptor.setLabel(Some(&NSString::from_str(label)));
    descriptor.setVertexFunction(Some(vertex));
    descriptor.setFragmentFunction(Some(fragment));
    // SAFETY: Index zero is within the pipeline's color attachment array.
    let color_attachment = unsafe { descriptor.colorAttachments().objectAtIndexedSubscript(0) };
    color_attachment.setPixelFormat(format);
    let source = match blend {
        Blend::Off => None,
        Blend::Premultiplied => Some(MTLBlendFactor::One),
        Blend::DestinationOut => Some(MTLBlendFactor::Zero),
    };
    color_attachment.setBlendingEnabled(source.is_some());
    if let Some(source) = source {
        color_attachment.setRgbBlendOperation(MTLBlendOperation::Add);
        color_attachment.setAlphaBlendOperation(MTLBlendOperation::Add);
        color_attachment.setSourceRGBBlendFactor(source);
        color_attachment.setSourceAlphaBlendFactor(source);
        color_attachment.setDestinationRGBBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
        color_attachment.setDestinationAlphaBlendFactor(MTLBlendFactor::OneMinusSourceAlpha);
    }
    device
        .newRenderPipelineStateWithDescriptor_error(&descriptor)
        .with_context(|| format!("creating {label} render pipeline"))
}

fn upload_effect_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    width: u32,
    height: u32,
    texels: &[u8],
) -> Result<Texture> {
    let descriptor = MTLTextureDescriptor::new();
    // SAFETY: `EffectTexture::texels` never yields an empty image, and this
    // descriptor creates a 2D texture.
    unsafe {
        descriptor.setWidth(width as usize);
        descriptor.setHeight(height as usize);
    }
    descriptor.setPixelFormat(MTLPixelFormat::BGRA8Unorm);
    descriptor.setUsage(MTLTextureUsage::ShaderRead);
    // Shared storage only on Apple GPU families, as in the atlas.
    descriptor.setStorageMode(if device.supportsFamily(MTLGPUFamily::Apple1) {
        MTLStorageMode::Shared
    } else {
        MTLStorageMode::Managed
    });
    let texture = device
        .newTextureWithDescriptor(&descriptor)
        .context("allocating an effect image")?;
    let region = MTLRegion {
        origin: MTLOrigin { x: 0, y: 0, z: 0 },
        size: MTLSize {
            width: width as usize,
            height: height as usize,
            depth: 1,
        },
    };
    // SAFETY: `texels` holds `width * height` BGRA8 texels, every row of the
    // region, and the texture is new: no GPU work reads it yet.
    unsafe {
        texture.replaceRegion_mipmapLevel_withBytes_bytesPerRow(
            region,
            0,
            ptr::NonNull::from(texels).cast(),
            width as usize * 4,
        );
    }
    Ok(texture)
}

pub(super) fn create_target_texture(
    device: &ProtocolObject<dyn MTLDevice>,
    label: &str,
    width: usize,
    height: usize,
    usage: MTLTextureUsage,
) -> Result<Texture> {
    create_texture_in(
        device,
        label,
        width,
        height,
        usage,
        MTLPixelFormat::BGRA8Unorm,
    )
}

pub(super) fn create_texture_in(
    device: &ProtocolObject<dyn MTLDevice>,
    label: &str,
    width: usize,
    height: usize,
    usage: MTLTextureUsage,
    format: MTLPixelFormat,
) -> Result<Texture> {
    let descriptor = MTLTextureDescriptor::new();
    // SAFETY: Both dimensions are at least one and this descriptor creates a 2D texture.
    unsafe {
        descriptor.setWidth(width.max(1));
        descriptor.setHeight(height.max(1));
    }
    descriptor.setPixelFormat(format);
    descriptor.setStorageMode(MTLStorageMode::Private);
    descriptor.setUsage(usage);
    let texture = device
        .newTextureWithDescriptor(&descriptor)
        .with_context(|| format!("allocating {label}"))?;
    texture.setLabel(Some(&NSString::from_str(label)));
    Ok(texture)
}

/// Degraded behaviour is announced, never silent, and each distinct message
/// once per process rather than every frame.
pub(super) fn effects_advisory(message: &str) {
    use std::sync::Mutex;
    static SEEN: Mutex<Vec<String>> = Mutex::new(Vec::new());
    let mut seen = match SEEN.lock() {
        Ok(seen) => seen,
        Err(poisoned) => poisoned.into_inner(),
    };
    if seen.iter().any(|seen| seen == message) {
        return;
    }
    seen.push(message.to_string());
    log::warn!("{message}");
    eprintln!("skathi: {message}");
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::metal_renderer::InstanceBufferPool;
    use gpui::{
        ContentMask, Corners, EFFECT_FLAG_NEEDS_BACKDROP, Hsla, Quad, Rgba, point, rgb, size,
    };
    use objc2_metal::MTLCommandBufferStatus;
    use objc2_quartz_core::CAMetalDrawable;
    use parking_lot::Mutex;
    use std::sync::Arc;

    thread_local! {
        /// Forces whole-frame blurs, the reference the region-limited path
        /// must reproduce.
        pub(super) static FULL_WINDOW_BACKDROP: std::cell::Cell<bool> =
            const { std::cell::Cell::new(false) };
        /// Renders through the intermediate a framebuffer-only drawable gets,
        /// into a target that can still be read back.
        pub(super) static FORCE_INTERMEDIATE_FRAME: std::cell::Cell<bool> =
            const { std::cell::Cell::new(false) };
        /// The scissor of every blur pass drawn on this thread.
        pub(super) static BLUR_PASS_RECTS: std::cell::RefCell<Vec<PixelRect>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    const SIDE: f32 = 64.;

    fn renderer() -> Result<MetalRenderer> {
        MetalRenderer::new_headless(Arc::new(Mutex::new(InstanceBufferPool::default())))
    }

    fn frame_size() -> Size<DevicePixels> {
        size(DevicePixels(SIDE as i32), DevicePixels(SIDE as i32))
    }

    fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(width), ScaledPixels(height)),
        )
    }

    fn mask(width: f32, height: f32) -> ContentMask<ScaledPixels> {
        ContentMask::rect(bounds(0., 0., width, height))
    }

    fn quad(rect: Bounds<ScaledPixels>, color: Rgba) -> Quad {
        Quad {
            bounds: rect,
            background: color.into(),
            content_mask: mask(SIDE, SIDE),
            ..Default::default()
        }
    }

    /// Black, with the left half white: one hard vertical edge at x = 32.
    fn split_scene() -> Scene {
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x000000)));
        scene.insert_primitive(quad(bounds(0., 0., SIDE / 2., SIDE), rgb(0xffffff)));
        scene
    }

    fn blur_rect() -> BackdropBlurRect {
        BackdropBlurRect {
            bounds: bounds(8., 8., 48., 48.),
            content_mask: mask(SIDE, SIDE),
            blur_radius: ScaledPixels(10.),
            ..Default::default()
        }
    }

    fn render(renderer: &mut MetalRenderer, mut scene: Scene) -> Result<image::RgbaImage> {
        scene.finish();
        renderer.render_scene_to_image(&scene, frame_size())
    }

    fn pixel(image: &image::RgbaImage, x: u32, y: u32) -> [u8; 4] {
        image.get_pixel(x, y).0
    }

    fn assert_close(actual: [u8; 4], expected: [u8; 4], what: &str) {
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(actual, expected)| actual.abs_diff(expected) <= 2),
            "{what}: {actual:?}, expected {expected:?}"
        );
    }

    fn backdrop(renderer: &MetalRenderer) -> Result<&MetalBackdrop> {
        renderer
            .backdrop
            .ready()
            .context("backdrop pipelines were not built")
    }

    fn held_levels(renderer: &MetalRenderer) -> Result<Option<usize>> {
        Ok(backdrop(renderer)?
            .targets
            .as_ref()
            .map(|targets| targets.levels.len()))
    }

    #[test]
    fn backdrop_blur_softens_the_edge_under_it_only() -> Result<()> {
        let mut renderer = renderer()?;
        let mut scene = split_scene();
        scene.insert_primitive(blur_rect());
        let image = render(&mut renderer, scene)?;

        let before_edge = pixel(&image, 30, 32)[0];
        let after_edge = pixel(&image, 33, 32)[0];
        assert!(
            (17..240).contains(&before_edge),
            "blur must soften the white side: {before_edge}"
        );
        assert!(
            (17..240).contains(&after_edge),
            "blur must soften the black side: {after_edge}"
        );
        assert!(before_edge > after_edge);
        assert_eq!(pixel(&image, 33, 32)[3], 255);
        assert_eq!(pixel(&image, 2, 32), [255; 4]);
        assert_eq!(pixel(&image, 61, 32), [0, 0, 0, 255]);
        assert_eq!(pixel(&image, 30, 4), [255; 4], "above the rect");
        assert_eq!(
            held_levels(&renderer)?,
            Some(2),
            "a 10px blur needs two pyramid levels, and only those are allocated"
        );
        Ok(())
    }

    #[test]
    fn backdrop_blur_respects_corners_mask_tint_and_opacity() -> Result<()> {
        let mut renderer = renderer()?;
        let blur = |tint: Hsla, opacity: f32| BackdropBlurRect {
            corner_radii: Corners::all(ScaledPixels(12.)),
            content_mask: mask(40., SIDE),
            tint,
            opacity,
            ..blur_rect()
        };
        let red = Hsla {
            a: 0.5,
            ..gpui::red()
        };

        let mut scene = split_scene();
        scene.insert_primitive(blur(red, 1.));
        let tinted = render(&mut renderer, scene)?;
        assert_eq!(pixel(&tinted, 9, 9), [255; 4], "outside the rounded corner");
        assert_eq!(
            pixel(&tinted, 50, 32),
            [0, 0, 0, 255],
            "outside the content mask"
        );
        let dark = pixel(&tinted, 36, 32);
        assert!(
            dark[0] > dark[1] + 100 && dark[1] == dark[2],
            "a half-alpha red tint over the dark side: {dark:?}"
        );
        let light = pixel(&tinted, 12, 32);
        assert!(
            light[0] > 240 && (100..200).contains(&light[1]),
            "a half-alpha red tint over the light side: {light:?}"
        );

        let mut scene = split_scene();
        scene.insert_primitive(blur(Hsla::default(), 1.));
        let opaque = render(&mut renderer, scene)?;
        let mut scene = split_scene();
        scene.insert_primitive(blur(Hsla::default(), 0.5));
        let half = render(&mut renderer, scene)?;
        let (full, faded) = (pixel(&opaque, 33, 32)[0], pixel(&half, 33, 32)[0]);
        assert!(full > 16, "the blur must reach the black side: {full}");
        assert!(
            faded.abs_diff(full / 2) <= 2,
            "opacity 0.5 must land halfway between the frame and the blur: {faded} vs {full}"
        );
        Ok(())
    }

    #[test]
    fn frost_replaces_its_backdrop_and_later_primitives_paint_above() -> Result<()> {
        let mut renderer = renderer()?;
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0xff0000)));
        scene.insert_primitive(EffectQuad {
            effect_id: effect_id::FROST,
            bounds: bounds(8., 8., 48., 48.),
            content_mask: mask(SIDE, SIDE),
            params: EffectQuad::params_from(&[1., 0., 0., 0., 0., 0., 1., 1.]),
            opacity: 1.,
            scale: 1.,
            flags: EFFECT_FLAG_NEEDS_BACKDROP,
            ..Default::default()
        });
        scene.insert_primitive(quad(bounds(24., 24., 8., 8.), rgb(0x00ff00)));
        let image = render(&mut renderer, scene)?;
        assert_eq!(pixel(&image, 2, 32), [255, 0, 0, 255]);
        assert_close(
            pixel(&image, 16, 16),
            [0, 0, 255, 255],
            "an opaque blue frost tint",
        );
        assert_eq!(
            pixel(&image, 28, 28),
            [0, 255, 0, 255],
            "a later primitive must paint above the effect"
        );
        assert!(
            backdrop(&renderer)?.effects.iter().all(Option::is_some),
            "every built-in effect pipeline must build"
        );
        Ok(())
    }

    #[test]
    fn frost_blurs_and_grains_its_backdrop() -> Result<()> {
        let mut renderer = renderer()?;
        let striped = || {
            let mut scene = Scene::default();
            scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x2040c0)));
            for stripe in 0..8 {
                let x = stripe as f32 * 8.;
                scene.insert_primitive(quad(bounds(x, 0., 4., SIDE), rgb(0xf0e0a0)));
            }
            scene
        };
        let plain = render(&mut renderer, striped())?;
        let mut scene = striped();
        scene.insert_primitive(EffectQuad {
            effect_id: effect_id::FROST,
            bounds: bounds(8., 8., 48., 48.),
            content_mask: mask(SIDE, SIDE),
            corner_radii: Corners::all(ScaledPixels(8.)),
            params: EffectQuad::params_from(&[1., 10., 0., 0., 1., 1., 1., 0.06]),
            opacity: 1.,
            backdrop_blur_radius: ScaledPixels(10.),
            scale: 1.,
            flags: EFFECT_FLAG_NEEDS_BACKDROP,
            ..Default::default()
        });
        let frosted = render(&mut renderer, scene)?;

        let inside: Vec<_> = (16..48)
            .flat_map(|y| (16..48).map(move |x| (x, y)))
            .collect();
        let changed = inside
            .iter()
            .filter(|&&(x, y)| {
                pixel(&plain, x, y)
                    .iter()
                    .zip(pixel(&frosted, x, y))
                    .any(|(plain, frosted)| plain.abs_diff(frosted) > 8)
            })
            .count();
        assert!(
            changed * 10 > inside.len() * 9,
            "frost must change its input: {changed} of {} pixels",
            inside.len()
        );
        let stripe_contrast = |image: &image::RgbaImage| {
            (16..48)
                .map(|x| i32::from(pixel(image, x, 32)[0]))
                .fold((i32::MAX, i32::MIN), |(low, high), red| {
                    (low.min(red), high.max(red))
                })
        };
        let (plain_low, plain_high) = stripe_contrast(&plain);
        let (frosted_low, frosted_high) = stripe_contrast(&frosted);
        assert!(
            frosted_high - frosted_low < (plain_high - plain_low) / 2,
            "frost must blur the stripes: {frosted_low}..{frosted_high} vs {plain_low}..{plain_high}"
        );
        assert_eq!(
            pixel(&frosted, 2, 2),
            pixel(&plain, 2, 2),
            "outside the quad"
        );
        Ok(())
    }

    fn glow(content_mask: ContentMask<ScaledPixels>) -> EffectQuad {
        EffectQuad {
            effect_id: effect_id::GLOW,
            // A 16px element at (24, 24), dilated by its 8px bleed.
            bounds: bounds(16., 16., 32., 32.),
            content_mask,
            params: EffectQuad::params_from(&[8., 1., 0., 0., 1., 0., 0., 1.]),
            opacity: 1.,
            bleed: ScaledPixels(8.),
            scale: 1.,
            ..Default::default()
        }
    }

    /// `glow.slang`'s lobe for [`glow`] over black, at the centre of pixel
    /// `x` on row 32: premultiplied red, drawn One, OneMinusSrcAlpha.
    fn expected_glow_red(x: u32) -> u8 {
        let (reach, peak) = (8_f32, 1_f32);
        let element_distance = (x as f32 + 0.5 - 32.).abs() - 8.;
        let signed_distance = element_distance.max(0.5 - 8.);
        let t = if signed_distance > 0. {
            signed_distance / reach
        } else {
            -signed_distance / (reach * 0.6)
        };
        let falloff = (1. - t).clamp(0., 1.);
        (falloff * falloff * peak * 255.).round() as u8
    }

    #[test]
    fn glow_bleeds_outside_the_element_and_stays_in_its_mask() -> Result<()> {
        let mut renderer = renderer()?;
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x000000)));
        scene.insert_primitive(glow(mask(SIDE, SIDE)));
        let image = render(&mut renderer, scene)?;
        for x in [12, 18, 22, 24, 26, 32, 38, 40, 42, 46] {
            let actual = pixel(&image, x, 32);
            let expected = expected_glow_red(x);
            assert!(
                actual[0].abs_diff(expected) <= 2 && actual[1..] == [0, 0, 255],
                "glow at x = {x}: {actual:?}, expected red {expected}"
            );
        }
        let outside = pixel(&image, 22, 32);
        assert!(outside[0] > 100, "the glow reaches outside the border box");

        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x000000)));
        scene.insert_primitive(glow(mask(32., SIDE)));
        let clipped = render(&mut renderer, scene)?;
        assert_eq!(pixel(&clipped, 22, 32), outside, "inside the mask");
        assert_eq!(pixel(&clipped, 42, 32), [0, 0, 0, 255], "outside the mask");
        Ok(())
    }

    fn punch(rect: Bounds<ScaledPixels>, content_mask: ContentMask<ScaledPixels>) -> EffectQuad {
        EffectQuad {
            effect_id: effect_id::PUNCH,
            bounds: rect,
            content_mask,
            corner_radii: Corners::all(ScaledPixels(8.)),
            opacity: 1.,
            scale: 1.,
            ..Default::default()
        }
    }

    #[test]
    fn punch_erases_its_rounded_rect_inside_the_mask() -> Result<()> {
        let mut renderer = renderer()?;
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0xff0000)));
        scene.insert_primitive(punch(bounds(16., 16., 32., 32.), mask(SIDE, 40.)));
        scene.insert_primitive(quad(bounds(36., 20., 8., 8.), rgb(0x00ff00)));
        let image = render(&mut renderer, scene)?;
        assert_eq!(pixel(&image, 24, 24), [0; 4], "the hole is transparent");
        assert_eq!(
            pixel(&image, 17, 17),
            [255, 0, 0, 255],
            "outside the corner"
        );
        assert_eq!(pixel(&image, 8, 8), [255, 0, 0, 255], "outside the rect");
        assert_eq!(pixel(&image, 24, 44), [255, 0, 0, 255], "outside the mask");
        assert_eq!(
            pixel(&image, 40, 24),
            [0, 255, 0, 255],
            "a later primitive paints over the hole"
        );

        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0xff0000)));
        scene.insert_primitive(EffectQuad {
            opacity: 0.5,
            ..punch(bounds(16., 16., 32., 32.), mask(SIDE, SIDE))
        });
        let half = render(&mut renderer, scene)?;
        assert_close(
            pixel(&half, 32, 32),
            [128, 0, 0, 128],
            "a half-opaque punch",
        );
        Ok(())
    }

    /// Stripes under a blurred `rect`.
    fn striped_blur(side: f32, rect: Bounds<ScaledPixels>) -> Scene {
        let mut scene = Scene::default();
        let content_mask = mask(side, side);
        scene.insert_primitive(Quad {
            content_mask,
            ..quad(bounds(0., 0., side, side), rgb(0x000000))
        });
        for index in 0..(side / 28.) as usize {
            let x = 16. + index as f32 * 28.;
            scene.insert_primitive(Quad {
                content_mask,
                ..quad(bounds(x, 0., 12., side), rgb(0xffffff))
            });
        }
        scene.insert_primitive(BackdropBlurRect {
            bounds: rect,
            content_mask,
            blur_radius: ScaledPixels(6.),
            ..Default::default()
        });
        scene.finish();
        scene
    }

    #[test]
    fn a_small_blur_processes_its_region_only_and_matches_the_whole_frame() -> Result<()> {
        const LARGE: i32 = 256;
        let size = size(DevicePixels(LARGE), DevicePixels(LARGE));
        let rect = bounds(96., 96., 64., 64.);
        let scene = striped_blur(LARGE as f32, rect);
        let levels = scene.backdrop_blur_rects[0].effective_kernel_levels() as usize;
        assert_eq!(levels, 2);

        BLUR_PASS_RECTS.with_borrow_mut(Vec::clear);
        let limited = renderer()?.render_scene_to_image(&scene, size)?;
        let limited_passes = BLUR_PASS_RECTS.take();

        FULL_WINDOW_BACKDROP.set(true);
        let reference =
            renderer().and_then(|mut renderer| renderer.render_scene_to_image(&scene, size));
        FULL_WINDOW_BACKDROP.set(false);
        let reference = reference?;
        let full_passes = BLUR_PASS_RECTS.take();

        let region = PixelRect::round_out(&rect, blur_reach(levels));
        let level_rect = |level: usize| PixelRect {
            left: 0,
            top: 0,
            right: LARGE >> level,
            bottom: LARGE >> level,
        };
        assert_eq!(
            limited_passes,
            [
                region.scale_down(1),
                region.scale_down(2),
                region.scale_down(1)
            ],
            "down to level 2 and back, each within the region"
        );
        assert_eq!(
            full_passes,
            [level_rect(1), level_rect(2), level_rect(1)],
            "the reference blurs every level whole"
        );

        let mut largest = 0;
        for y in 96..160 {
            for x in 96..160 {
                let (limited, reference) = (pixel(&limited, x, y), pixel(&reference, x, y));
                for channel in 0..4 {
                    largest = largest.max(limited[channel].abs_diff(reference[channel]));
                }
            }
        }
        assert!(
            largest <= 2,
            "inside the rect the region blur is off by {largest}"
        );
        assert!(
            (96..160).any(|x| {
                let red = pixel(&limited, x, 128)[0];
                red > 16 && red < 240
            }),
            "the fixture must blur"
        );
        Ok(())
    }

    fn frosted_scene() -> Scene {
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0xff0000)));
        scene.insert_primitive(EffectQuad {
            effect_id: effect_id::FROST,
            bounds: bounds(8., 8., 48., 48.),
            content_mask: mask(SIDE, SIDE),
            params: EffectQuad::params_from(&[1., 0., 0., 0.]),
            opacity: 1.,
            scale: 1.,
            flags: EFFECT_FLAG_NEEDS_BACKDROP,
            ..Default::default()
        });
        scene.finish();
        scene
    }

    #[test]
    fn idle_scratch_targets_are_released_as_on_d3d12() -> Result<()> {
        let mut renderer = renderer()?;
        let mut blurred = split_scene();
        blurred.insert_primitive(blur_rect());
        blurred.finish();
        renderer.render_scene(&blurred, frame_size())?;
        assert_eq!(held_levels(&renderer)?, Some(2));

        let frosted = frosted_scene();
        renderer.render_scene(&frosted, frame_size())?;
        assert_eq!(
            held_levels(&renderer)?,
            Some(2),
            "a short gap must not release levels a recent frame used"
        );
        for _ in 0..SCRATCH_RELEASE_FRAMES {
            renderer.render_scene(&frosted, frame_size())?;
        }
        assert_eq!(
            held_levels(&renderer)?,
            Some(0),
            "a zero-radius effect keeps the snapshot and no levels"
        );
        for _ in 0..SCRATCH_RELEASE_FRAMES {
            renderer.render_scene(&Scene::default(), frame_size())?;
        }
        assert_eq!(
            held_levels(&renderer)?,
            None,
            "a scene without backdrops releases the snapshot"
        );
        renderer.render_scene(&blurred, frame_size())?;
        assert_eq!(
            held_levels(&renderer)?,
            Some(2),
            "and a later blur gets them back"
        );
        Ok(())
    }

    #[test]
    fn noise_is_grain_not_a_flat_fill() -> Result<()> {
        let mut renderer = renderer()?;
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x808080)));
        scene.insert_primitive(EffectQuad {
            effect_id: effect_id::NOISE,
            bounds: bounds(8., 8., 48., 48.),
            content_mask: mask(SIDE, SIDE),
            params: EffectQuad::params_from(&[1., 1., 0., 0.]),
            opacity: 1.,
            scale: 1.,
            ..Default::default()
        });
        let image = render(&mut renderer, scene)?;
        let values: Vec<f64> = (8..56)
            .flat_map(|y| (8..56).map(move |x| (x, y)))
            .map(|(x, y)| f64::from(pixel(&image, x, y)[1]))
            .collect();
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let deviation = (values
            .iter()
            .map(|value| (value - mean).powi(2))
            .sum::<f64>()
            / values.len() as f64)
            .sqrt();
        let mut distinct: Vec<u8> = values.iter().map(|&value| value as u8).collect();
        distinct.sort_unstable();
        distinct.dedup();
        assert!(
            deviation > 10. && distinct.len() > 32,
            "grain must vary: deviation {deviation:.1}, {} distinct values",
            distinct.len()
        );
        assert_eq!(
            pixel(&image, 4, 4),
            [128, 128, 128, 255],
            "outside the quad"
        );
        Ok(())
    }

    fn blur_and_frost_scene() -> Scene {
        let mut scene = split_scene();
        scene.insert_primitive(blur_rect());
        scene.insert_primitive(EffectQuad {
            effect_id: effect_id::FROST,
            bounds: bounds(20., 20., 24., 24.),
            content_mask: mask(SIDE, SIDE),
            params: EffectQuad::params_from(&[0.8, 4., 0., 0., 0., 0., 1., 0.3]),
            opacity: 1.,
            backdrop_blur_radius: ScaledPixels(4.),
            scale: 1.,
            flags: EFFECT_FLAG_NEEDS_BACKDROP,
            ..Default::default()
        });
        scene.finish();
        scene
    }

    /// What a framebuffer-only drawable gets must be what a target that can
    /// be copied from shows.
    #[test]
    fn the_intermediate_frame_reproduces_the_direct_render() -> Result<()> {
        let mut renderer = renderer()?;
        let direct = renderer.render_scene_to_image(&blur_and_frost_scene(), frame_size())?;
        FORCE_INTERMEDIATE_FRAME.set(true);
        let through_intermediate =
            renderer.render_scene_to_image(&blur_and_frost_scene(), frame_size());
        FORCE_INTERMEDIATE_FRAME.set(false);
        let through_intermediate = through_intermediate?;
        assert!(
            backdrop(&renderer)?
                .targets
                .as_ref()
                .is_some_and(|targets| targets.frame.is_some()),
            "the frame must have gone through the intermediate"
        );
        assert!(
            direct.as_raw() == through_intermediate.as_raw(),
            "the intermediate frame must reproduce the direct render exactly"
        );
        Ok(())
    }

    /// A drawable that cannot be copied from must not be blitted (Metal's
    /// validation aborts on it); later drawables are made copyable, and the
    /// layer goes back to framebuffer-only with the scratch targets.
    #[test]
    fn a_framebuffer_only_drawable_lifts_the_flag_until_backdrops_go_idle() -> Result<()> {
        let mut renderer = MetalRenderer::new(Default::default(), false, false)?;
        let layer = renderer
            .layer()
            .context("window renderer has a layer")?
            .retain();
        layer.setFramebufferOnly(true);
        renderer.update_drawable_size(frame_size());
        let drawable = layer
            .nextDrawable()
            .context("the layer must vend a drawable")?;
        let texture = drawable.texture();
        assert!(texture.isFramebufferOnly());

        let command_buffer = objc2::rc::autoreleasepool(|_| {
            renderer.render_frame(&blur_and_frost_scene(), &texture, frame_size())
        })?;
        command_buffer.commit();
        command_buffer.waitUntilCompleted();
        assert_eq!(command_buffer.status(), MTLCommandBufferStatus::Completed);
        assert!(
            !layer.framebufferOnly(),
            "later drawables can be copied from"
        );

        for _ in 0..SCRATCH_RELEASE_FRAMES {
            renderer.render_scene(&Scene::default(), frame_size())?;
        }
        assert_eq!(held_levels(&renderer)?, None);
        assert!(
            layer.framebufferOnly(),
            "idle, the layer is framebuffer-only again"
        );
        Ok(())
    }

    #[test]
    fn a_broken_effect_shader_is_an_error_and_its_quads_a_hole() -> Result<()> {
        let mut renderer = renderer()?;
        let vertex = renderer
            .library
            .newFunctionWithName(&NSString::from_str("effect_vertex"))
            .context("effect_vertex")?;
        let broken = frame_effect_pipeline(
            &renderer.device,
            &vertex,
            "broken",
            "fragment float4 effect_fragment() { return oops; }",
            false,
        );
        assert!(
            broken.is_err(),
            "a shader that does not compile must be refused"
        );
        let mismatched = frame_effect_pipeline(
            &renderer.device,
            &vertex,
            "mismatched",
            "#include <metal_stdlib>\nusing namespace metal;\n\
             struct In { float4 tint [[user(SOMETHING_ELSE)]]; };\n\
             fragment float4 effect_fragment(In input [[stage_in]]) { return input.tint; }",
            false,
        );
        assert!(
            mismatched.is_err(),
            "a stage that does not link to the vertex half must be refused"
        );

        let missing = effect_id::COUNT + 7;
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x2040c0)));
        scene.insert_primitive(EffectQuad {
            effect_id: missing,
            ..glow(mask(SIDE, SIDE))
        });
        scene.insert_primitive(EffectQuad {
            effect_id: missing,
            flags: EFFECT_FLAG_NEEDS_BACKDROP,
            ..glow(mask(SIDE, SIDE))
        });
        let image = render(&mut renderer, scene)?;
        assert!(backdrop(&renderer)?.effect(missing).is_none());
        assert!(
            image
                .pixels()
                .all(|pixel| pixel.0 == [0x20, 0x40, 0xc0, 255]),
            "an effect without a pipeline draws nothing, and nothing else is lost"
        );
        Ok(())
    }

    /// The app's effects are process-wide; tests that install them take turns.
    pub(in crate::metal_renderer) static APP_EFFECTS: Mutex<()> = Mutex::new(());

    fn install_app_effect(msl: &str) -> Result<()> {
        gpui::reload_app_effects(
            gpui::AppEffects::new(
                "test",
                r#"{"schema": 10, "effects": [
                    {"name": "tint", "backdrop": false, "metal": "tint.metal"}
                ]}"#,
                [("tint.metal".to_string(), msl.to_string())],
            )
            .map_err(anyhow::Error::msg)?,
        );
        Ok(())
    }

    fn solid(rgb: &str) -> String {
        format!(
            "#include <metal_stdlib>\nusing namespace metal;\n\
             fragment float4 effect_fragment() {{ return float4({rgb}, 1.0); }}"
        )
    }

    #[test]
    fn a_reloaded_app_effect_draws_on_the_next_frame_and_a_broken_one_keeps_the_last() -> Result<()>
    {
        let _app_effects = APP_EFFECTS.lock();
        let scene = || {
            let mut scene = Scene::default();
            scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x000000)));
            scene.insert_primitive(EffectQuad {
                effect_id: effect_id::COUNT,
                bounds: bounds(8., 8., 48., 48.),
                content_mask: mask(SIDE, SIDE),
                opacity: 1.,
                scale: 1.,
                ..Default::default()
            });
            scene
        };
        let mut renderer = renderer()?;
        install_app_effect(&solid("1.0, 0.0, 0.0"))?;
        assert_eq!(
            pixel(&render(&mut renderer, scene())?, 32, 32),
            [255, 0, 0, 255]
        );

        install_app_effect(&solid("0.0, 1.0, 0.0"))?;
        let image = render(&mut renderer, scene())?;
        assert_eq!(pixel(&image, 32, 32), [0, 255, 0, 255]);
        assert_eq!(pixel(&image, 2, 2), [0, 0, 0, 255]);

        install_app_effect("fragment float4 effect_fragment() { return oops; }")?;
        assert_eq!(
            pixel(&render(&mut renderer, scene())?, 32, 32),
            [0, 255, 0, 255],
            "a replacement that does not compile keeps the last good pipeline"
        );
        Ok(())
    }

    #[test]
    fn effect_images_bind_per_quad_premultiplied_unless_linear() -> Result<()> {
        use gpui::{EffectTexture, Primitive, RenderImage};

        let _app_effects = APP_EFFECTS.lock();
        install_app_effect(
            "#include <metal_stdlib>\nusing namespace metal;\n\
             fragment float4 effect_fragment(float4 p [[position]],\n\
                 texture2d<float> t0 [[texture(3)]], texture2d<float> t1 [[texture(4)]],\n\
                 sampler wrap [[sampler(2)]]) {\n\
               if (p.y < 32.0) return float4(t0.sample(wrap, float2(0.25, 0.5), level(0.0)).rgb, 1.0);\n\
               return float4(t1.sample(wrap, float2(1.25, 0.5), level(0.0)).rgb, 1.0);\n\
             }",
        )?;
        let image = |bgra: &[[u8; 4]]| {
            let buffer = image::RgbaImage::from_raw(bgra.len() as u32, 1, bgra.concat()).unwrap();
            Arc::new(RenderImage::new(vec![image::Frame::new(buffer)]))
        };
        let halves = image(&[[0, 0, 255, 128], [255, 0, 0, 255]]);
        let green = image(&[[0, 255, 0, 255]]);
        let set = |first: &Arc<RenderImage>| {
            let mut set = gpui::EffectTextureSet::default();
            set[0] = Some(EffectTexture {
                image: first.clone(),
                linear: false,
            });
            set[1] = Some(EffectTexture {
                image: halves.clone(),
                linear: true,
            });
            Some(Arc::new(set))
        };
        let effect = |x: f32| EffectQuad {
            effect_id: effect_id::COUNT,
            bounds: bounds(x, 0., SIDE / 2., SIDE),
            content_mask: mask(SIDE, SIDE),
            opacity: 1.,
            scale: 1.,
            ..Default::default()
        };

        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x000000)));
        scene.insert_primitive(Primitive::EffectQuad(effect(0.), set(&halves)));
        scene.insert_primitive(Primitive::EffectQuad(effect(SIDE / 2.), set(&green)));
        let mut renderer = renderer()?;
        let image = render(&mut renderer, scene)?;
        assert_close(pixel(&image, 16, 16), [128, 0, 0, 255], "colour");
        assert_close(pixel(&image, 16, 48), [255, 0, 0, 255], "linear");
        assert_close(pixel(&image, 48, 16), [0, 255, 0, 255], "second");
        assert_close(pixel(&image, 48, 48), [255, 0, 0, 255], "shared");
        Ok(())
    }

    /// The app-effect path end to end: point `SK_EFFECTS_DIR` at an app's
    /// compiled effects (`crates/sk-effects/generated` serves).
    #[test]
    #[ignore = "needs SK_EFFECTS_DIR pointing at an app's compiled effects"]
    fn app_effects_from_the_effects_dir_compile_and_draw() -> Result<()> {
        let _app_effects = APP_EFFECTS.lock();
        let dir =
            std::env::var_os(gpui::APP_EFFECTS_DIR_ENV).context("SK_EFFECTS_DIR is not set")?;
        let app =
            gpui::AppEffects::from_dir(std::path::Path::new(&dir)).map_err(anyhow::Error::msg)?;
        let effects = app.effects().to_vec();
        gpui::reload_app_effects(app);
        anyhow::ensure!(!effects.is_empty(), "the manifest lists no effects");

        let mut renderer = renderer()?;
        for (index, effect) in effects.iter().enumerate() {
            let mut scene = Scene::default();
            scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x000000)));
            scene.insert_primitive(EffectQuad {
                effect_id: effect_id::COUNT + index as u32,
                bounds: bounds(8., 8., 48., 48.),
                content_mask: mask(SIDE, SIDE),
                params: EffectQuad::params_from(&[1., 24., 0.5, 0.5, 1., 1., 1., 1.]),
                opacity: 1.,
                time: 0.75,
                scale: 1.,
                flags: if effect.needs_backdrop {
                    EFFECT_FLAG_NEEDS_BACKDROP
                } else {
                    0
                },
                ..Default::default()
            });
            let image = render(&mut renderer, scene)?;
            assert!(
                backdrop(&renderer)?.app_effects.get(index).is_some(),
                "{} must build",
                effect.name
            );
            assert!(
                effect.content || (8..56).any(|x| pixel(&image, x, 32) != [0, 0, 0, 255]),
                "{} must draw",
                effect.name
            );
            assert_eq!(pixel(&image, 2, 2), [0, 0, 0, 255], "{}", effect.name);
        }
        Ok(())
    }
}
