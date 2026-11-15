//! CSS `backdrop-filter: blur()` and the `--shading` effects, ported from the
//! DirectX renderer: the same Dual-Kawase pyramid, the same grouping, the same
//! effect pipeline table.

use super::{
    BACKDROP_SHADERS, BLUR_SHADERS, EFFECT_VERTEX_SHADER, InstanceBinding, WgpuBindGroupLayouts,
    WgpuRendererCore,
};
use anyhow::{Context as _, Result};
use bytemuck::{Pod, Zeroable};
use gpui::{BackdropBlurRect, EffectQuad, MAX_BACKDROP_BLUR_KERNEL_LEVELS, Scene, effect_id};
use std::num::NonZeroU64;
use std::ops::Range;

include!(concat!(env!("OUT_DIR"), "/builtin_effects.rs"));

const _: () = {
    assert!(BUILTIN_EFFECT_COUNT == effect_id::COUNT as usize);
    assert!(effect_id::FROST == 0);
    assert!(effect_id::NOISE == 1);
    assert!(effect_id::GLOW == 2);
};

const BLUR_TEXTURE_LEVELS: usize = MAX_BACKDROP_BLUR_KERNEL_LEVELS as usize;

/// Duplicated from the `sk-effects` registry, as in gpui_windows: gpui sits
/// below it. An app effect reads its manifest's `backdrop` field instead.
fn builtin_needs_backdrop(index: usize) -> bool {
    index == effect_id::FROST as usize
}

/// The DirectX `GlobalParams` constant buffer, which `modules/sk.slang`
/// declares for every target. 12 words.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct EffectGlobals {
    gamma_ratios: [f32; 4],
    viewport_size: [f32; 2],
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    is_bgr: u32,
    pad: [u32; 3],
}

#[derive(Default)]
pub(super) enum BackdropState {
    /// Built on the first frame that paints a backdrop primitive.
    #[default]
    Unbuilt,
    Ready(Box<WgpuBackdrop>),
    Unavailable,
}

impl BackdropState {
    pub(super) fn invalidate_targets(&mut self) {
        if let BackdropState::Ready(backdrop) = self {
            backdrop.targets = None;
        }
    }

    pub(super) fn ready(&self) -> Option<&WgpuBackdrop> {
        match self {
            BackdropState::Ready(backdrop) => Some(backdrop),
            BackdropState::Unbuilt | BackdropState::Unavailable => None,
        }
    }
}

pub(super) struct WgpuBackdrop {
    blur_downsample: wgpu::RenderPipeline,
    blur_upsample: wgpu::RenderPipeline,
    blur_rect: wgpu::RenderPipeline,
    frame_copy: wgpu::RenderPipeline,
    /// The built-ins, indexed by `EffectQuad::effect_id`. A hole keeps the
    /// numbering when one failed to build.
    effects: Vec<Option<wgpu::RenderPipeline>>,
    /// The app's own, `effect_id::COUNT` on.
    app_effects: gpui::AppEffectPipelines<AppEffectPipeline>,
    effect_layout: wgpu::PipelineLayout,
    effect_vertex: wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    textures_layout: wgpu::BindGroupLayout,
    effect_globals_buffer: wgpu::Buffer,
    pub(super) effect_globals_bind_group: wgpu::BindGroup,
    clamp_sampler: wgpu::Sampler,
    /// At binding 9, for an effect's images.
    repeat_sampler: wgpu::Sampler,
    placeholder: TargetTexture,
    /// Group 2 of an effect that does not read the backdrop: its shader still
    /// declares the textures.
    placeholder_textures: wgpu::BindGroup,
    effect_images: gpui::EffectTextureCache<wgpu::TextureView>,
    /// This frame's `Scene::effect_texture_sets`, uploaded.
    frame_effect_images: Vec<[Option<wgpu::TextureView>; gpui::EFFECT_TEXTURE_SLOTS]>,
    /// `gpui::effect_id::PUNCH`.
    punch: Option<wgpu::RenderPipeline>,
    /// `gpui::effect_id::CONTENT`.
    content: Option<wgpu::RenderPipeline>,
    pub(super) histories: gpui::VnEffectHistories<TargetTexture>,
    targets: Option<BackdropTargets>,
    /// Consecutive frames that held more scratch targets than they needed.
    idle_frames: u32,
}

struct AppEffectPipeline {
    shade: wgpu::RenderPipeline,
    /// `[SkHistory]`: the step, into a history texture with blending off.
    history: Option<wgpu::RenderPipeline>,
}

/// Frames the scratch targets may sit unused before release, as on D3D12:
/// recreating one is a window-sized allocation, so short gaps keep them.
const SCRATCH_RELEASE_FRAMES: u32 = 90;

pub(super) const HISTORY_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

#[derive(Clone)]
pub(super) struct TargetTexture {
    pub(super) texture: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
}

/// Sized to the frame; dropped on resize.
struct BackdropTargets {
    width: u32,
    height: u32,
    snapshot: TargetTexture,
    /// Pyramid level `i` is the frame halved `i + 1` times, allocated up to the
    /// deepest level a frame has asked for.
    levels: Vec<TargetTexture>,
    frame: Option<TargetTexture>,
}

/// Where a frame renders. A backdrop needs to copy what has been drawn so far,
/// and a surface that is not a copy source (GL) gets an intermediate the frame
/// is drawn into and copied from at the end.
pub(super) struct FrameTarget {
    pub(super) texture: wgpu::Texture,
    pub(super) view: wgpu::TextureView,
    copy_to: Option<wgpu::TextureView>,
    /// Whether this frame can draw backdrop blurs and effects at all.
    pub(super) backdrop: bool,
}

impl FrameTarget {
    pub(super) fn layer(texture: wgpu::Texture, view: wgpu::TextureView, backdrop: bool) -> Self {
        Self {
            texture,
            view,
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

impl WgpuRendererCore {
    pub(super) fn prepare_frame_target(
        &mut self,
        scene: &Scene,
        target: &wgpu::Texture,
        target_view: &wgpu::TextureView,
    ) -> FrameTarget {
        let has_primitives = !scene.backdrop_blur_rects.is_empty()
            || !scene.effect_quads.is_empty()
            || !scene.vn_effect_draws.is_empty();
        let backdrop = has_primitives && self.ensure_backdrop();
        let mut frame = FrameTarget {
            texture: target.clone(),
            view: target_view.clone(),
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
        if let BackdropState::Ready(state) = &mut self.resources.backdrop {
            state.sync_app_effects(&self.resources.device);
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
        }
        if !backdrop {
            return frame;
        }

        let globals = EffectGlobals {
            gamma_ratios: self.rendering_params.gamma_ratios,
            viewport_size: [target.width() as f32, target.height() as f32],
            grayscale_enhanced_contrast: self.rendering_params.grayscale_enhanced_contrast,
            subpixel_enhanced_contrast: self.rendering_params.subpixel_enhanced_contrast,
            is_bgr: self.is_bgr as u32,
            pad: [0; 3],
        };
        let color_format = self.atlas.color_texture_format();
        let resources = &mut self.resources;
        let BackdropState::Ready(state) = &mut resources.backdrop else {
            return frame;
        };
        resources.queue.write_buffer(
            &state.effect_globals_buffer,
            0,
            bytemuck::bytes_of(&globals),
        );
        state.upload_effect_images(&resources.device, &resources.queue, color_format, scene);

        if snapshots && !target.usage().contains(wgpu::TextureUsages::COPY_SRC) {
            let targets = state.targets(
                &resources.device,
                self.target_format,
                target.width(),
                target.height(),
                0,
            );
            let intermediate = targets.frame.get_or_insert_with(|| {
                create_target_texture(
                    &resources.device,
                    "backdrop_frame",
                    self.target_format,
                    target.width(),
                    target.height(),
                    wgpu::TextureUsages::RENDER_ATTACHMENT
                        | wgpu::TextureUsages::TEXTURE_BINDING
                        | wgpu::TextureUsages::COPY_SRC,
                )
            });
            frame.texture = intermediate.texture.clone();
            frame.view = intermediate.view.clone();
            frame.copy_to = Some(target_view.clone());
        }
        frame
    }

    /// Copies an intermediate frame onto the target it stood in for.
    pub(super) fn finish_frame_target(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameTarget,
    ) {
        let (Some(target_view), Some(state)) = (&frame.copy_to, self.resources.backdrop.ready())
        else {
            return;
        };
        let bind_group = self.source_bind_group(state, &frame.view);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("backdrop_frame_copy"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });
        pass.set_pipeline(&state.frame_copy);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..4, 0..1);
    }

    pub(super) fn ensure_backdrop(&mut self) -> bool {
        if self.uses_webgl_instance_data {
            effects_advisory(
                "`backdrop-filter` and `--shading` need storage buffers, which WebGL2 lacks; \
                 they are not drawn and every other style renders normally",
            );
            return false;
        }
        let resources = &mut self.resources;
        if matches!(resources.backdrop, BackdropState::Unbuilt) {
            resources.backdrop = match WgpuBackdrop::new(
                &resources.device,
                &resources.bind_group_layouts,
                self.target_format,
            ) {
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
        matches!(resources.backdrop, BackdropState::Ready(_))
    }

    /// CSS `backdrop-filter: blur()`, grouped as DirectX groups it: a group
    /// shares one kernel depth and holds no two overlapping rects, and each
    /// group snapshots the frame again, so glass over glass sees the glass
    /// below it.
    pub(super) fn draw_backdrop_blur_rects(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameTarget,
        blur_rects: &[BackdropBlurRect],
        start: usize,
        instances: &InstanceBinding,
    ) -> Result<()> {
        for group in backdrop_groups(blur_rects, |rect| {
            (rect.effective_kernel_levels() as usize, rect.bounds)
        }) {
            let kernel_levels = blur_rects[group.start].effective_kernel_levels() as usize;
            let union = blur_rects[group.clone()]
                .iter()
                .map(|rect| rect.bounds)
                .reduce(|union, bounds| union.union(&bounds))
                .context("empty backdrop group")?;
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
            let textures =
                self.snapshot_backdrop(encoder, frame, kernel_levels, region, [None, None], None)?;
            let state = self
                .resources
                .backdrop
                .ready()
                .context("backdrop pipelines missing")?;
            let mut pass = super::continue_pass(encoder, &frame.view, "backdrop_blur_composite");
            pass.set_pipeline(&state.blur_rect);
            pass.set_bind_group(0, &self.resources.globals_bind_group, &[]);
            pass.set_bind_group(1, &instances.bind_group, &[]);
            pass.set_bind_group(2, &textures, &[]);
            pass.draw(0..4, instance_range(instances, start, group));
        }
        Ok(())
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
                .resources
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
        images: Option<usize>,
        instances: &InstanceBinding,
        range: Range<u32>,
        pass: &mut wgpu::RenderPass<'_>,
    ) {
        let Some(state) = self.resources.backdrop.ready() else {
            return;
        };
        let Some(pipeline) = state.effect(effect_id) else {
            return;
        };
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &state.effect_globals_bind_group, &[]);
        pass.set_bind_group(1, &instances.bind_group, &[]);
        match images {
            Some(_) => {
                let textures = state.effect_textures(
                    &self.resources.device,
                    [None, None],
                    [None, None],
                    images,
                );
                pass.set_bind_group(2, &textures, &[]);
            }
            None => pass.set_bind_group(2, &state.placeholder_textures, &[]),
        }
        pass.draw(
            0..4,
            instances.first_instance + range.start..instances.first_instance + range.end,
        );
    }

    /// An effect that reads the backdrop takes the backdrop blur's path:
    /// grouped, a snapshot and pyramid per group, then a composite with
    /// blending disabled.
    pub(super) fn draw_backdrop_effects(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameTarget,
        effect_id: u32,
        effect_quads: &[EffectQuad],
        start: usize,
        instances: &InstanceBinding,
    ) -> Result<()> {
        for group in backdrop_groups(effect_quads, |quad| {
            (quad.effective_kernel_levels() as usize, quad.bounds)
        }) {
            let kernel_levels = effect_quads[group.start].effective_kernel_levels() as usize;
            // An effect may sample its backdrop anywhere, as on D3D12.
            let images = effect_quads[0].texture_set();
            let textures =
                self.snapshot_backdrop(encoder, frame, kernel_levels, None, [None, None], images)?;
            let state = self
                .resources
                .backdrop
                .ready()
                .context("backdrop pipelines missing")?;
            let pipeline = state.effect(effect_id).context("effect pipeline missing")?;
            let mut pass = super::continue_pass(encoder, &frame.view, "backdrop_effect_composite");
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &state.effect_globals_bind_group, &[]);
            pass.set_bind_group(1, &instances.bind_group, &[]);
            pass.set_bind_group(2, &textures, &[]);
            pass.draw(0..4, instance_range(instances, start, group));
        }
        Ok(())
    }

    /// Copies what the frame holds so far and blurs it `kernel_levels` deep,
    /// within `region` (frame pixels, the kernel's reach included) or the
    /// whole frame. Returns group 2 for the composite: the processed backdrop,
    /// the clamping sampler, the untouched snapshot, `layer` (an effect
    /// layer's content and history) and the effect's `images`.
    fn snapshot_backdrop(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameTarget,
        kernel_levels: usize,
        region: Option<PixelRect>,
        layer: [Option<&wgpu::TextureView>; 2],
        images: Option<usize>,
    ) -> Result<wgpu::BindGroup> {
        let kernel_levels = kernel_levels.min(BLUR_TEXTURE_LEVELS);
        let resources = &mut self.resources;
        let BackdropState::Ready(state) = &mut resources.backdrop else {
            anyhow::bail!("backdrop pipelines missing");
        };
        let (width, height) = (frame.texture.width(), frame.texture.height());
        state.targets(
            &resources.device,
            self.target_format,
            width,
            height,
            kernel_levels,
        );
        let state = &**state;
        let targets = state.targets.as_ref().context("backdrop targets missing")?;
        let full = PixelRect::of(&frame.texture);
        let region = region.map_or(full, |region| region.intersect(full));
        if region.is_empty() {
            anyhow::bail!("empty backdrop region");
        }

        let origin = wgpu::Origin3d {
            x: region.left as u32,
            y: region.top as u32,
            z: 0,
        };
        encoder.copy_texture_to_texture(
            wgpu::TexelCopyTextureInfo {
                origin,
                ..frame.texture.as_image_copy()
            },
            wgpu::TexelCopyTextureInfo {
                origin,
                ..targets.snapshot.texture.as_image_copy()
            },
            wgpu::Extent3d {
                width: (region.right - region.left) as u32,
                height: (region.bottom - region.top) as u32,
                depth_or_array_layers: 1,
            },
        );

        // Like D3D12, a pass writes only its scissor and leaves the rest of
        // the level as it was; the rounded-out rects keep every texel the
        // next pass reads inside what this one wrote.
        let source_layout = &resources.bind_group_layouts.texture;
        let blur_pass = |encoder: &mut wgpu::CommandEncoder,
                         pipeline: &wgpu::RenderPipeline,
                         source: &wgpu::TextureView,
                         destination: &TargetTexture,
                         scale: usize| {
            let rect = region
                .scale_down(scale)
                .intersect(PixelRect::of(&destination.texture));
            if rect.is_empty() {
                return;
            }
            #[cfg(test)]
            tests::BLUR_PASS_RECTS.with_borrow_mut(|rects| rects.push(rect));
            let bind_group = source_bind_group(&resources.device, source_layout, state, source);
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("backdrop_blur_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &destination.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_scissor_rect(
                rect.left as u32,
                rect.top as u32,
                (rect.right - rect.left) as u32,
                (rect.bottom - rect.top) as u32,
            );
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.draw(0..4, 0..1);
        };

        let blurred = if kernel_levels == 0 {
            &targets.snapshot.view
        } else {
            let levels = &targets.levels[..kernel_levels];
            blur_pass(
                encoder,
                &state.blur_downsample,
                &targets.snapshot.view,
                &levels[0],
                1,
            );
            for level in 1..kernel_levels {
                blur_pass(
                    encoder,
                    &state.blur_downsample,
                    &levels[level - 1].view,
                    &levels[level],
                    level + 1,
                );
            }
            for level in (1..kernel_levels).rev() {
                blur_pass(
                    encoder,
                    &state.blur_upsample,
                    &levels[level].view,
                    &levels[level - 1],
                    level,
                );
            }
            &levels[0].view
        };

        Ok(state.effect_textures(
            &resources.device,
            [Some(blurred), Some(&targets.snapshot.view)],
            layer,
            images,
        ))
    }

    /// The backdrop of an effect layer's effect, which may sample it anywhere.
    pub(super) fn snapshot_layer_backdrop(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameTarget,
        kernel_levels: usize,
        layer: [Option<&wgpu::TextureView>; 2],
        images: Option<usize>,
    ) -> Result<wgpu::BindGroup> {
        self.snapshot_backdrop(encoder, frame, kernel_levels, None, layer, images)
    }

    fn source_bind_group(
        &self,
        state: &WgpuBackdrop,
        source: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        source_bind_group(
            &self.resources.device,
            &self.resources.bind_group_layouts.texture,
            state,
            source,
        )
    }
}

fn source_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    state: &WgpuBackdrop,
    source: &wgpu::TextureView,
) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("backdrop_pass_source"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(source),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&state.clamp_sampler),
            },
        ],
    })
}

fn instance_range(instances: &InstanceBinding, start: usize, group: Range<usize>) -> Range<u32> {
    let first = instances.first_instance + start as u32;
    first + group.start as u32..first + group.end as u32
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
    fn of(texture: &wgpu::Texture) -> Self {
        Self {
            left: 0,
            top: 0,
            right: texture.width() as i32,
            bottom: texture.height() as i32,
        }
    }

    /// The pixels covering `bounds`, grown by `margin` on every side.
    fn round_out(bounds: &gpui::Bounds<gpui::ScaledPixels>, margin: i32) -> Self {
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
    key: impl Fn(&T) -> (usize, gpui::Bounds<gpui::ScaledPixels>),
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

impl WgpuBackdrop {
    fn new(
        device: &wgpu::Device,
        layouts: &WgpuBindGroupLayouts,
        format: wgpu::TextureFormat,
    ) -> Result<Self> {
        let texture_entry = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let textures_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("backdrop_textures_layout"),
            entries: &[
                texture_entry(0),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                texture_entry(2),
                texture_entry(3),
                texture_entry(4),
                texture_entry(5),
                texture_entry(6),
                texture_entry(7),
                texture_entry(8),
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let effect_globals_size = std::mem::size_of::<EffectGlobals>() as u64;
        let effect_globals_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("effect_globals_layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: NonZeroU64::new(effect_globals_size),
                    },
                    count: None,
                }],
            });
        let effect_globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("effect_globals"),
            size: effect_globals_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let effect_globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("effect_globals"),
            layout: &effect_globals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: effect_globals_buffer.as_entire_binding(),
            }],
        });
        let clamp_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("backdrop_clamp_sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let repeat_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("effect_repeat_sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let placeholder = create_target_texture(
            device,
            "backdrop_placeholder",
            format,
            1,
            1,
            wgpu::TextureUsages::TEXTURE_BINDING,
        );
        let placeholder_textures = textures_bind_group(
            device,
            &textures_layout,
            (&clamp_sampler, &repeat_sampler),
            [&placeholder.view; 4],
            [&placeholder.view; gpui::EFFECT_TEXTURE_SLOTS],
        );

        let pass_layout = pipeline_layout(device, "backdrop_pass_layout", &[&layouts.texture]);
        let blur_rect_layout = pipeline_layout(
            device,
            "backdrop_blur_rect_layout",
            &[&layouts.globals, &layouts.instances, &textures_layout],
        );
        let effect_layout = pipeline_layout(
            device,
            "effect_layout",
            &[&effect_globals_layout, &layouts.instances, &textures_layout],
        );

        let (blur_downsample, blur_upsample, blur_rect, frame_copy, effect_vertex) =
            capture_validation(device, || {
                let blur = shader_module(device, "backdrop_blur_shaders", BLUR_SHADERS);
                let composite = shader_module(device, "backdrop_shaders", BACKDROP_SHADERS);
                let effect_vertex = shader_module(device, "effect_vertex", EFFECT_VERTEX_SHADER);
                let pass = |label, fragment| {
                    render_pipeline(
                        device,
                        label,
                        &pass_layout,
                        (&blur, "vs_blur_pass"),
                        (&blur, fragment),
                        format,
                        None,
                    )
                };
                (
                    pass("backdrop_blur_downsample", "fs_blur_downsample"),
                    pass("backdrop_blur_upsample", "fs_blur_upsample"),
                    render_pipeline(
                        device,
                        "backdrop_blur_rect",
                        &blur_rect_layout,
                        (&composite, "vs_blur_rect"),
                        (&composite, "fs_blur_rect"),
                        format,
                        None,
                    ),
                    pass("backdrop_frame_copy", "fs_frame_copy"),
                    effect_vertex,
                )
            })?;

        let effects =
            build_builtin_effect_pipelines(device, &effect_layout, &effect_vertex, format);
        let punch = capture_validation(device, || {
            render_pipeline(
                device,
                "effect_punch",
                &effect_layout,
                (&effect_vertex, "vs_effect"),
                (&effect_vertex, "fs_punch"),
                format,
                Some(DESTINATION_OUT),
            )
        })
        .inspect_err(|error| {
            effects_advisory(&format!(
                "`<iframe>` documents cannot show through — the punch pipeline failed to build: \
                 {error:#}"
            ));
        })
        .ok();
        let content = capture_validation(device, || {
            render_pipeline(
                device,
                "effect_content",
                &effect_layout,
                (&effect_vertex, "vs_effect"),
                (&effect_vertex, "fs_content"),
                format,
                Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
            )
        })
        .inspect_err(|error| {
            effects_advisory(&format!(
                "elements with an `[SkContent]` effect cannot cross-fade it — the content \
                 pipeline failed to build: {error:#}"
            ));
        })
        .ok();

        let mut backdrop = Self {
            blur_downsample,
            blur_upsample,
            blur_rect,
            frame_copy,
            effects,
            app_effects: Default::default(),
            effect_layout,
            effect_vertex,
            format,
            textures_layout,
            effect_globals_buffer,
            effect_globals_bind_group,
            clamp_sampler,
            repeat_sampler,
            placeholder,
            placeholder_textures,
            effect_images: Default::default(),
            frame_effect_images: Vec::new(),
            punch,
            content,
            histories: Default::default(),
            targets: None,
            idle_frames: 0,
        };
        backdrop.sync_app_effects(device);
        Ok(backdrop)
    }

    pub(super) fn sync_app_effects(&mut self, device: &wgpu::Device) {
        let (layout, vertex, format) = (&self.effect_layout, &self.effect_vertex, self.format);
        self.app_effects.sync(|effect| {
            let (_, wgsl) = effect.wgsl.get("WGSL")?;
            let shade = effect_pipeline(
                device,
                layout,
                vertex,
                &effect.name,
                wgsl,
                format,
                effect.needs_backdrop,
            )?;
            let history = effect
                .history
                .then(|| {
                    effect_pipeline(
                        device,
                        layout,
                        vertex,
                        &format!("{}_history", effect.name),
                        wgsl,
                        HISTORY_FORMAT,
                        true,
                    )
                })
                .transpose()?;
            Ok(AppEffectPipeline { shade, history })
        });
    }

    /// Group 2 of an effect: the backdrop as blurred and as snapshotted, an
    /// effect layer's content and history, and the images of texture set
    /// `images` (`EffectQuad::texture_set`); a placeholder for each absent.
    pub(super) fn effect_textures(
        &self,
        device: &wgpu::Device,
        backdrop: [Option<&wgpu::TextureView>; 2],
        layer: [Option<&wgpu::TextureView>; 2],
        images: Option<usize>,
    ) -> wgpu::BindGroup {
        let placeholder = &self.placeholder.view;
        let images = images.and_then(|set| self.frame_effect_images.get(set));
        textures_bind_group(
            device,
            &self.textures_layout,
            (&self.clamp_sampler, &self.repeat_sampler),
            [
                backdrop[0].unwrap_or(placeholder),
                backdrop[1].unwrap_or(placeholder),
                layer[0].unwrap_or(placeholder),
                layer[1].unwrap_or(placeholder),
            ],
            std::array::from_fn(|slot| {
                images
                    .and_then(|images| images[slot].as_ref())
                    .unwrap_or(placeholder)
            }),
        )
    }

    /// Uploads the images this frame's effects sample that are not on the GPU yet.
    fn upload_effect_images(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        format: wgpu::TextureFormat,
        scene: &Scene,
    ) {
        self.effect_images.begin_frame();
        let cache = &mut self.effect_images;
        self.frame_effect_images = scene
            .effect_texture_sets
            .iter()
            .map(|set| {
                std::array::from_fn(|slot| {
                    let texture = set[slot].as_ref()?;
                    cache
                        .get(texture, |width, height, texels| {
                            upload_effect_texture(device, queue, format, width, height, texels)
                        })
                        .cloned()
                })
            })
            .collect();
    }

    /// The `[SkHistory]` step of app effect `effect_id`.
    pub(super) fn history_effect(&self, effect_id: u32) -> Option<&wgpu::RenderPipeline> {
        let index = effect_id.checked_sub(effect_id::COUNT)?;
        self.app_effects.get(index as usize)?.history.as_ref()
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

    pub(super) fn effect(&self, effect_id: u32) -> Option<&wgpu::RenderPipeline> {
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

    fn targets(
        &mut self,
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
        kernel_levels: usize,
    ) -> &mut BackdropTargets {
        if self
            .targets
            .as_ref()
            .is_some_and(|targets| targets.width != width || targets.height != height)
        {
            self.targets = None;
        }
        let targets = self.targets.get_or_insert_with(|| BackdropTargets {
            width,
            height,
            snapshot: create_target_texture(
                device,
                "backdrop_snapshot",
                format,
                width,
                height,
                wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::TEXTURE_BINDING,
            ),
            levels: Vec::new(),
            frame: None,
        });
        while targets.levels.len() < kernel_levels {
            let divisor = 2u32.saturating_pow(targets.levels.len() as u32 + 1);
            targets.levels.push(create_target_texture(
                device,
                "backdrop_blur_level",
                format,
                (width / divisor).max(1),
                (height / divisor).max(1),
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            ));
        }
        targets
    }
}

/// The built-in `--shading` pipelines in `gpui::effect_id` order. Infallible as
/// a whole, as on DirectX: one that does not build leaves a hole and a one-shot
/// advisory.
fn build_builtin_effect_pipelines(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    vertex: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
) -> Vec<Option<wgpu::RenderPipeline>> {
    let mut pipelines = Vec::new();
    for (index, &(name, wgsl)) in BUILTIN_EFFECT_WGSL.iter().enumerate() {
        let pipeline = wgsl
            .context("this build carries no shader for it")
            .and_then(|wgsl| {
                effect_pipeline(
                    device,
                    layout,
                    vertex,
                    name,
                    wgsl,
                    format,
                    builtin_needs_backdrop(index),
                )
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

fn textures_bind_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    (clamp, repeat): (&wgpu::Sampler, &wgpu::Sampler),
    [blurred, snapshot, content, history]: [&wgpu::TextureView; 4],
    [image0, image1, image2, image3]: [&wgpu::TextureView; gpui::EFFECT_TEXTURE_SLOTS],
) -> wgpu::BindGroup {
    let texture = |binding, view| wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    };
    let sampler = |binding, sampler| wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Sampler(sampler),
    };
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("effect_textures"),
        layout,
        entries: &[
            texture(0, blurred),
            sampler(1, clamp),
            texture(2, snapshot),
            texture(3, image0),
            texture(4, image1),
            texture(5, image2),
            texture(6, image3),
            texture(7, content),
            texture(8, history),
            sampler(9, repeat),
        ],
    })
}

/// `format` is the atlas's colour format, `Bgra8Unorm` or `Rgba8Unorm`.
fn upload_effect_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    texels: &[u8],
) -> Result<wgpu::TextureView> {
    let limit = device.limits().max_texture_dimension_2d;
    anyhow::ensure!(
        width <= limit && height <= limit,
        "a {width}x{height} image is larger than this GPU's {limit}px textures"
    );
    let size = wgpu::Extent3d {
        width,
        height,
        depth_or_array_layers: 1,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("effect_texture"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let swizzled;
    let texels = if format == wgpu::TextureFormat::Rgba8Unorm {
        let mut rgba = texels.to_vec();
        for pixel in rgba.chunks_exact_mut(4) {
            pixel.swap(0, 2);
        }
        swizzled = rgba;
        &swizzled[..]
    } else {
        texels
    };
    queue.write_texture(
        texture.as_image_copy(),
        texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width * 4),
            rows_per_image: Some(height),
        },
        size,
    );
    Ok(texture.create_view(&wgpu::TextureViewDescriptor::default()))
}

/// D3D12's `Blend::DestinationOut`: the target times one minus the source's
/// alpha, an eraser.
const DESTINATION_OUT: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::Zero,
        dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
        operation: wgpu::BlendOperation::Add,
    },
};

fn effect_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    vertex: &wgpu::ShaderModule,
    name: &str,
    wgsl: &str,
    format: wgpu::TextureFormat,
    needs_backdrop: bool,
) -> Result<wgpu::RenderPipeline> {
    let label = format!("effect_{name}");
    capture_validation(device, || {
        let fragment = shader_module(device, &label, wgsl);
        // A backdrop effect rewrites every pixel of its quad from its own
        // snapshot. Any other is `sk_finish`'s premultiplied colour, drawn over.
        let blend = (!needs_backdrop).then_some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING);
        render_pipeline(
            device,
            &label,
            layout,
            (vertex, "vs_effect"),
            (&fragment, "effect_fragment"),
            format,
            blend,
        )
    })
}

pub(super) fn shader_module(
    device: &wgpu::Device,
    label: &str,
    source: &str,
) -> wgpu::ShaderModule {
    device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    })
}

pub(super) fn pipeline_layout(
    device: &wgpu::Device,
    label: &str,
    layouts: &[&wgpu::BindGroupLayout],
) -> wgpu::PipelineLayout {
    let layouts: Vec<_> = layouts.iter().copied().map(Some).collect();
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &layouts,
        immediate_size: 0,
    })
}

pub(super) fn render_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    vertex: (&wgpu::ShaderModule, &str),
    fragment: (&wgpu::ShaderModule, &str),
    format: wgpu::TextureFormat,
    blend: Option<wgpu::BlendState>,
) -> wgpu::RenderPipeline {
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: vertex.0,
            entry_point: Some(vertex.1),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: fragment.0,
            entry_point: Some(fragment.1),
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleStrip,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

pub(super) fn create_target_texture(
    device: &wgpu::Device,
    label: &str,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
    usage: wgpu::TextureUsages,
) -> TargetTexture {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    TargetTexture { texture, view }
}

/// Runs `build` inside a validation error scope, so a shader that does not
/// compile becomes an error here instead of a failed frame.
#[cfg(not(target_family = "wasm"))]
pub(super) fn capture_validation<T>(device: &wgpu::Device, build: impl FnOnce() -> T) -> Result<T> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let value = build();
    match gpui::block_on(scope.pop()) {
        Some(error) => Err(anyhow::anyhow!("{error}")),
        None => Ok(value),
    }
}

/// A browser resolves error scopes asynchronously; its errors reach the
/// device's uncaptured-error handler instead.
#[cfg(target_family = "wasm")]
pub(super) fn capture_validation<T>(
    _device: &wgpu::Device,
    build: impl FnOnce() -> T,
) -> Result<T> {
    Ok(build())
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
    use gpui::{Bounds, ContentMask, Corners, ScaledPixels, point, size};

    thread_local! {
        /// Forces whole-frame blurs, the reference the region-limited path
        /// must reproduce.
        pub(super) static FULL_WINDOW_BACKDROP: std::cell::Cell<bool> =
            const { std::cell::Cell::new(false) };
        /// The scissor of every blur pass drawn on this thread.
        pub(super) static BLUR_PASS_RECTS: std::cell::RefCell<Vec<PixelRect>> =
            const { std::cell::RefCell::new(Vec::new()) };
    }

    #[test]
    fn pixel_rects_match_the_d3d12_helpers() {
        let rect = PixelRect::round_out(&bounds(10.5, 20., 5.25, 4.), 3);
        assert_eq!(
            rect,
            PixelRect {
                left: 7,
                top: 17,
                right: 19,
                bottom: 27
            }
        );
        assert_eq!(
            rect.scale_down(2),
            PixelRect {
                left: 0,
                top: 3,
                right: 6,
                bottom: 8
            }
        );
        assert!(
            rect.intersect(PixelRect::round_out(&bounds(40., 0., 1., 1.), 0))
                .is_empty()
        );
        assert_eq!(blur_reach(2), 24);
    }

    fn module(source: &str) -> naga::Module {
        let module = naga::front::wgsl::parse_str(source).expect("shader should parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .expect("shader should validate");
        module
    }

    fn struct_layout(module: &naga::Module, name: &str) -> (Vec<u32>, u32) {
        module
            .types
            .iter()
            .find_map(|(_, ty)| match &ty.inner {
                naga::TypeInner::Struct { members, span } if ty.name.as_deref() == Some(name) => {
                    Some((members.iter().map(|member| member.offset).collect(), *span))
                }
                _ => None,
            })
            .unwrap_or_else(|| panic!("struct {name} missing"))
    }

    fn effect_quad_offsets() -> Vec<u32> {
        use std::mem::offset_of;
        let content_mask = offset_of!(EffectQuad, content_mask);
        [
            offset_of!(EffectQuad, order),
            offset_of!(EffectQuad, effect_id),
            offset_of!(EffectQuad, bounds),
            content_mask,
            content_mask + offset_of!(ContentMask<ScaledPixels>, corner_radii),
            offset_of!(EffectQuad, corner_radii),
            offset_of!(EffectQuad, params),
            offset_of!(EffectQuad, opacity),
            offset_of!(EffectQuad, time),
            offset_of!(EffectQuad, bleed),
            offset_of!(EffectQuad, backdrop_blur_radius),
            offset_of!(EffectQuad, scale),
            offset_of!(EffectQuad, flags),
        ]
        .into_iter()
        .map(|offset| offset as u32)
        .collect()
    }

    fn effect_globals_offsets() -> Vec<u32> {
        use std::mem::offset_of;
        let pad = offset_of!(EffectGlobals, pad);
        [
            offset_of!(EffectGlobals, gamma_ratios),
            offset_of!(EffectGlobals, viewport_size),
            offset_of!(EffectGlobals, grayscale_enhanced_contrast),
            offset_of!(EffectGlobals, subpixel_enhanced_contrast),
            offset_of!(EffectGlobals, is_bgr),
            pad,
            pad + 4,
            pad + 8,
        ]
        .into_iter()
        .map(|offset| offset as u32)
        .collect()
    }

    #[test]
    fn backdrop_and_blur_shaders_are_valid_wgsl() {
        module(BACKDROP_SHADERS);
        module(BLUR_SHADERS);
    }

    #[test]
    fn effect_vertex_stage_matches_the_rust_layouts() {
        let vertex = module(EFFECT_VERTEX_SHADER);
        assert_eq!(
            struct_layout(&vertex, "EffectQuad"),
            (
                effect_quad_offsets(),
                std::mem::size_of::<EffectQuad>() as u32
            )
        );
        assert_eq!(
            struct_layout(&vertex, "EffectGlobals"),
            (
                effect_globals_offsets(),
                std::mem::size_of::<EffectGlobals>() as u32
            )
        );
    }

    /// The generated fragment stages read the same buffers as the vertex
    /// stage and the renderer's bind groups, laid out by slangc.
    #[test]
    fn generated_effect_stages_are_valid_and_match_the_rust_layouts() {
        for (name, wgsl) in BUILTIN_EFFECT_WGSL {
            let wgsl = wgsl.unwrap_or_else(|| panic!("{name}.wgsl is not embedded"));
            let fragment = module(wgsl);
            assert_eq!(
                struct_layout(&fragment, "SkEffectQuad_std430_0"),
                (
                    effect_quad_offsets(),
                    std::mem::size_of::<EffectQuad>() as u32
                ),
                "{name}"
            );
            assert_eq!(
                struct_layout(&fragment, "SLANG_ParameterGroup_GlobalParams_std140_0"),
                (effect_globals_offsets(), 48),
                "{name}"
            );
            let mut bindings: Vec<_> = fragment
                .global_variables
                .iter()
                .filter_map(|(_, variable)| variable.binding.as_ref())
                .map(|binding| (binding.group, binding.binding))
                .collect();
            bindings.sort_unstable();
            assert_eq!(
                bindings,
                [(0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (2, 7), (2, 8)],
                "{name}"
            );
            assert!(
                fragment
                    .entry_points
                    .iter()
                    .any(|entry| entry.name == "effect_fragment"),
                "{name}"
            );
        }
    }

    #[test]
    fn blur_rects_are_the_words_the_shader_loads() {
        assert_eq!(std::mem::size_of::<BackdropBlurRect>(), 24 * 4);
    }

    fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(width), ScaledPixels(height)),
        )
    }

    #[test]
    fn backdrop_groups_split_on_overlap_and_kernel_depth() {
        let rects = [
            (2, bounds(0., 0., 10., 10.)),
            (2, bounds(20., 0., 10., 10.)),
            (2, bounds(5., 5., 10., 10.)),
            (3, bounds(40., 0., 10., 10.)),
        ];
        assert_eq!(
            backdrop_groups(&rects, |rect| *rect),
            vec![0..2, 2..3, 3..4]
        );
    }

    #[cfg(any(target_os = "linux", target_os = "windows"))]
    pub(in crate::wgpu_renderer) mod gpu {
        use super::*;
        use crate::WgpuHeadlessRenderer;
        use gpui::{
            DevicePixels, EFFECT_FLAG_NEEDS_BACKDROP, Hsla, PlatformHeadlessRenderer, Quad, Rgba,
            rgb,
        };

        const SIDE: f32 = 64.;

        fn frame_size() -> gpui::Size<DevicePixels> {
            size(DevicePixels(SIDE as i32), DevicePixels(SIDE as i32))
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

        fn render(
            renderer: &mut WgpuHeadlessRenderer,
            mut scene: Scene,
        ) -> Result<image::RgbaImage> {
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

        fn backdrop(renderer: &WgpuHeadlessRenderer) -> Result<&WgpuBackdrop> {
            renderer
                .core
                .resources
                .backdrop
                .ready()
                .context("backdrop pipelines were not built")
        }

        #[test]
        fn backdrop_blur_softens_the_edge_under_it_only() -> Result<()> {
            let mut renderer = WgpuHeadlessRenderer::new()?;
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
                backdrop(&renderer)?
                    .targets
                    .as_ref()
                    .map(|targets| targets.levels.len()),
                Some(2),
                "a 10px blur needs two pyramid levels, and only those are allocated"
            );
            Ok(())
        }

        #[test]
        fn backdrop_blur_respects_corners_mask_tint_and_opacity() -> Result<()> {
            let mut renderer = WgpuHeadlessRenderer::new()?;
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
            let mut renderer = WgpuHeadlessRenderer::new()?;
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
            let mut renderer = WgpuHeadlessRenderer::new()?;
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
            let mut renderer = WgpuHeadlessRenderer::new()?;
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
            assert_eq!(expected_glow_red(24), 205, "the peak sits at the edge");
            assert!(
                expected_glow_red(22) > 100,
                "and reaches outside the border box"
            );
            assert_eq!(expected_glow_red(12), 0, "but not beyond the bleed");
            let outside = pixel(&image, 22, 32);

            let mut scene = Scene::default();
            scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x000000)));
            scene.insert_primitive(glow(mask(32., SIDE)));
            let clipped = render(&mut renderer, scene)?;
            assert_eq!(pixel(&clipped, 22, 32), outside, "inside the mask");
            assert_eq!(pixel(&clipped, 42, 32), [0, 0, 0, 255], "outside the mask");
            Ok(())
        }

        fn punch(
            rect: Bounds<ScaledPixels>,
            content_mask: ContentMask<ScaledPixels>,
        ) -> EffectQuad {
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
            let mut renderer = WgpuHeadlessRenderer::new()?;
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

        /// Two scenes sharing stripes; `rect` is the one blurred.
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
            let limited = WgpuHeadlessRenderer::new()?.render_scene_to_image(&scene, size)?;
            let limited_passes = BLUR_PASS_RECTS.take();

            FULL_WINDOW_BACKDROP.set(true);
            let reference = WgpuHeadlessRenderer::new()?.render_scene_to_image(&scene, size);
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
            let area = |rects: &[PixelRect]| {
                rects
                    .iter()
                    .map(|rect| (rect.right - rect.left) * (rect.bottom - rect.top))
                    .sum::<i32>()
            };
            assert!(
                area(&limited_passes) * 4 < area(&full_passes),
                "{} of {} pyramid texels",
                area(&limited_passes),
                area(&full_passes)
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

        #[test]
        fn idle_scratch_targets_are_released_as_on_d3d12() -> Result<()> {
            let mut renderer = WgpuHeadlessRenderer::new()?;
            let held = |renderer: &WgpuHeadlessRenderer| -> Result<Option<usize>> {
                Ok(backdrop(renderer)?
                    .targets
                    .as_ref()
                    .map(|targets| targets.levels.len()))
            };
            let mut blurred = split_scene();
            blurred.insert_primitive(blur_rect());
            blurred.finish();
            renderer.render_scene(&blurred, frame_size())?;
            assert_eq!(held(&renderer)?, Some(2));

            let mut frosted = Scene::default();
            frosted.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0xff0000)));
            frosted.insert_primitive(EffectQuad {
                effect_id: effect_id::FROST,
                bounds: bounds(8., 8., 48., 48.),
                content_mask: mask(SIDE, SIDE),
                params: EffectQuad::params_from(&[1., 0., 0., 0.]),
                opacity: 1.,
                scale: 1.,
                flags: EFFECT_FLAG_NEEDS_BACKDROP,
                ..Default::default()
            });
            frosted.finish();
            renderer.render_scene(&frosted, frame_size())?;
            assert_eq!(
                held(&renderer)?,
                Some(2),
                "a short gap must not release levels a recent frame used"
            );
            for _ in 0..SCRATCH_RELEASE_FRAMES {
                renderer.render_scene(&frosted, frame_size())?;
            }
            assert_eq!(
                held(&renderer)?,
                Some(0),
                "a zero-radius effect keeps the snapshot and no levels"
            );
            for _ in 0..SCRATCH_RELEASE_FRAMES {
                renderer.render_scene(&Scene::default(), frame_size())?;
            }
            assert_eq!(
                held(&renderer)?,
                None,
                "a scene without backdrops releases the snapshot"
            );
            renderer.render_scene(&blurred, frame_size())?;
            assert_eq!(held(&renderer)?, Some(2), "and a later blur gets them back");
            Ok(())
        }

        #[test]
        fn noise_is_grain_not_a_flat_fill() -> Result<()> {
            let mut renderer = WgpuHeadlessRenderer::new()?;
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

        /// A surface that cannot be copied from (GL) gets an intermediate frame;
        /// the result must be what a copyable target shows.
        #[test]
        fn an_uncopyable_target_renders_through_an_intermediate_frame() -> Result<()> {
            let mut renderer = WgpuHeadlessRenderer::new()?;
            let scene = || {
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
            };
            let direct = renderer.render_scene_to_image(&scene(), frame_size())?;

            let device = renderer.core.resources.device.clone();
            let surface = create_target_texture(
                &device,
                "uncopyable_surface",
                renderer.core.target_format,
                SIDE as u32,
                SIDE as u32,
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            );
            renderer.core.render_frame(
                &scene(),
                &surface.texture,
                &surface.view,
                frame_size(),
                false,
                wgpu::Color::BLACK,
            )?;
            assert!(
                backdrop(&renderer)?
                    .targets
                    .as_ref()
                    .is_some_and(|targets| targets.frame.is_some()),
                "the frame must have gone through the intermediate"
            );

            let readback_view = renderer
                .render_target
                .as_ref()
                .map(|target| target.view.clone())
                .context("headless target missing")?;
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("uncopyable_surface_readback"),
            });
            renderer.core.finish_frame_target(
                &mut encoder,
                &FrameTarget {
                    texture: surface.texture.clone(),
                    view: surface.view.clone(),
                    copy_to: Some(readback_view),
                    backdrop: true,
                },
            );
            renderer
                .core
                .resources
                .queue
                .submit(std::iter::once(encoder.finish()));
            let through_intermediate = renderer.read_image()?;
            assert!(
                direct.as_raw() == through_intermediate.as_raw(),
                "the intermediate frame must reproduce the direct render exactly"
            );
            Ok(())
        }

        #[test]
        fn a_broken_effect_shader_is_an_error_not_a_device_fault() -> Result<()> {
            let mut renderer = WgpuHeadlessRenderer::new()?;
            let device = renderer.core.resources.device.clone();
            let layout = pipeline_layout(&device, "broken_effect_layout", &[]);
            let vertex = capture_validation(&device, || {
                shader_module(&device, "effect_vertex", EFFECT_VERTEX_SHADER)
            })?;
            let result = effect_pipeline(
                &device,
                &layout,
                &vertex,
                "broken",
                "@fragment fn effect_fragment() -> @location(0) vec4<f32> { return oops; }",
                renderer.core.target_format,
                false,
            );
            assert!(
                result.is_err(),
                "a shader that does not compile must be refused"
            );

            let mut scene = split_scene();
            scene.insert_primitive(blur_rect());
            render(&mut renderer, scene)?;
            Ok(())
        }

        /// The app's effects are process-wide; tests that install them take turns.
        pub(in crate::wgpu_renderer) static APP_EFFECTS: parking_lot::Mutex<()> =
            parking_lot::Mutex::new(());

        fn install_app_effect(wgsl: &str) -> Result<()> {
            gpui::reload_app_effects(
                gpui::AppEffects::new(
                    "test",
                    r#"{"schema": 10, "effects": [
                        {"name": "tint", "backdrop": false, "wgsl": "tint.wgsl"}
                    ]}"#,
                    [("tint.wgsl".to_string(), wgsl.to_string())],
                )
                .map_err(anyhow::Error::msg)?,
            );
            Ok(())
        }

        fn solid(rgb: &str) -> String {
            format!(
                "@fragment fn effect_fragment() -> @location(0) vec4<f32> {{ \
                 return vec4<f32>({rgb}, 1.0); }}"
            )
        }

        #[test]
        fn a_reloaded_app_effect_draws_on_the_next_frame_and_a_broken_one_keeps_the_last()
        -> Result<()> {
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
            let mut renderer = WgpuHeadlessRenderer::new()?;
            install_app_effect(&solid("1.0, 0.0, 0.0"))?;
            assert_eq!(
                pixel(&render(&mut renderer, scene())?, 32, 32),
                [255, 0, 0, 255]
            );

            install_app_effect(&solid("0.0, 1.0, 0.0"))?;
            let image = render(&mut renderer, scene())?;
            assert_eq!(pixel(&image, 32, 32), [0, 255, 0, 255]);
            assert_eq!(pixel(&image, 2, 2), [0, 0, 0, 255]);

            install_app_effect(
                "@fragment fn effect_fragment() -> @location(0) vec4<f32> { return oops; }",
            )?;
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
            use std::sync::Arc;

            let _app_effects = APP_EFFECTS.lock();
            install_app_effect(
                "@group(2) @binding(3) var t0: texture_2d<f32>;
                 @group(2) @binding(4) var t1: texture_2d<f32>;
                 @group(2) @binding(9) var wrap: sampler;
                 @fragment fn effect_fragment(@builtin(position) p: vec4<f32>) -> @location(0) vec4<f32> {
                     if (p.y < 32.0) {
                         return vec4<f32>(textureSampleLevel(t0, wrap, vec2<f32>(0.25, 0.5), 0.0).rgb, 1.0);
                     }
                     return vec4<f32>(textureSampleLevel(t1, wrap, vec2<f32>(1.25, 0.5), 0.0).rgb, 1.0);
                 }",
            )?;
            let image = |bgra: &[[u8; 4]]| {
                let buffer =
                    image::RgbaImage::from_raw(bgra.len() as u32, 1, bgra.concat()).unwrap();
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
            let mut renderer = WgpuHeadlessRenderer::new()?;
            let image = render(&mut renderer, scene)?;
            assert_close(pixel(&image, 16, 16), [128, 0, 0, 255], "colour");
            assert_close(pixel(&image, 16, 48), [255, 0, 0, 255], "linear");
            assert_close(pixel(&image, 48, 16), [0, 255, 0, 255], "second");
            assert_close(pixel(&image, 48, 48), [255, 0, 0, 255], "shared");
            Ok(())
        }

        /// The app-effect path end to end: `bun scripts/shaders.ts --app
        /// <app> --out <dir>`, then this with `SK_EFFECTS_DIR=<dir>`.
        #[test]
        #[ignore = "needs SK_EFFECTS_DIR pointing at an app's compiled effects"]
        fn app_effects_from_the_effects_dir_validate_and_draw() -> Result<()> {
            let _app_effects = APP_EFFECTS.lock();
            let dir =
                std::env::var_os(gpui::APP_EFFECTS_DIR_ENV).context("SK_EFFECTS_DIR is not set")?;
            let app = gpui::AppEffects::from_dir(std::path::Path::new(&dir))
                .map_err(anyhow::Error::msg)?;
            let effects = app.effects().to_vec();
            gpui::reload_app_effects(app);
            let effects = &effects;
            anyhow::ensure!(!effects.is_empty(), "the manifest lists no effects");
            for effect in effects {
                let (_, wgsl) = effect.wgsl.get("WGSL")?;
                let fragment = module(wgsl);
                assert_eq!(
                    struct_layout(&fragment, "SkEffectQuad_std430_0"),
                    (
                        effect_quad_offsets(),
                        std::mem::size_of::<EffectQuad>() as u32
                    ),
                    "{}",
                    effect.name
                );
            }

            let mut renderer = WgpuHeadlessRenderer::new()?;
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
}
