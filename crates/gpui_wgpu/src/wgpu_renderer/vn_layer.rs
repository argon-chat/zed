//! Isolated layers (`gpui::VnLayer`), ported from the DirectX renderer: each
//! group draws into a frame-sized target of its own, cleared to transparent,
//! which is then composited over its parent through the layer's clip shape,
//! color matrix and text background stack.

use super::backdrop::{
    FrameTarget, TargetTexture, capture_validation, create_target_texture, effects_advisory,
    pipeline_layout, render_pipeline, shader_module,
};
use super::{GroupDraw, InstanceBinding, VN_LAYER_SHADERS, WgpuRendererCore, continue_pass};
use anyhow::Result;
use gpui::{Scene, VnLayer, VnTextBackground, vn_layer_flags};
use std::{collections::HashMap, mem::size_of};

/// Deeper groups draw inline, unclipped by their shape, as on DirectX.
pub(super) const MAX_DEPTH: usize = 8;

/// Frames the layer targets may sit unused before release.
const TARGET_RELEASE_FRAMES: u32 = 90;

/// Renders a snapshot no frame captured or painted may sit before release.
const SNAPSHOT_IDLE_RENDERS: u32 = 4;

const LAYER_WORDS: usize = 56;
const SEGMENT_WORDS: usize = 4;
const TEXT_BACKGROUND_WORDS: usize = 69;

const _: () = {
    assert!(size_of::<VnLayer>() == LAYER_WORDS * 4);
    assert!(size_of::<[f32; 4]>() == SEGMENT_WORDS * 4);
    assert!(size_of::<VnTextBackground>() == TEXT_BACKGROUND_WORDS * 4);
};

#[derive(Default)]
pub(super) struct VnLayerState {
    pipeline: PipelineState,
    plus_lighter_pipeline: PipelineState,
    /// One per nesting depth.
    targets: Vec<TargetTexture>,
    depth_used: usize,
    idle_frames: u32,
    /// The captures' pixels by id (`Scene::vn_captures`); they outlive a
    /// resize and go with the device.
    snapshots: HashMap<u64, Snapshot>,
}

struct Snapshot {
    texture: TargetTexture,
    idle: u32,
}

#[derive(Default)]
enum PipelineState {
    #[default]
    Unbuilt,
    Ready(wgpu::RenderPipeline),
    Unavailable,
}

impl VnLayerState {
    pub(super) fn invalidate_targets(&mut self) {
        self.targets.clear();
    }

    fn trim_targets(&mut self) {
        if self.depth_used >= self.targets.len() {
            self.idle_frames = 0;
            return;
        }
        self.idle_frames += 1;
        if self.idle_frames >= TARGET_RELEASE_FRAMES {
            self.idle_frames = 0;
            self.targets.truncate(self.depth_used);
        }
    }
}

impl WgpuRendererCore {
    /// The frame's layers, then their clip segments and text backgrounds, as
    /// `shaders_vn_layer.wgsl` reads them. `None` when the frame has no layer
    /// or the transport cannot carry them; every group then draws inline.
    pub(super) fn write_vn_layers(
        &mut self,
        scene: &Scene,
        instance_offset: &mut u64,
    ) -> Result<Option<InstanceBinding>> {
        self.resources.vn_layers.depth_used = 0;
        if scene.vn_layers.is_empty() {
            return Ok(None);
        }
        if self.uses_webgl_instance_data {
            effects_advisory(
                "`clip-path` shapes, `filter` color matrices and `background-clip: text` need \
                 storage buffers, which WebGL2 lacks; their content paints unclipped and unfiltered",
            );
            return Ok(None);
        }
        let segments_word = scene.vn_layers.len() * LAYER_WORDS;
        let text_word = segments_word + scene.vn_clip_segments.len() * SEGMENT_WORDS;
        let mut layers = scene.vn_layers.clone();
        for layer in &mut layers {
            layer.segment_start =
                (segments_word + layer.segment_start as usize * SEGMENT_WORDS) as u32;
            layer.pad[1] = (text_word + layer.pad[1] as usize * TEXT_BACKGROUND_WORDS) as u32;
        }
        let mut bytes = Vec::with_capacity(
            (text_word + scene.vn_text_backgrounds.len() * TEXT_BACKGROUND_WORDS) * 4,
        );
        // SAFETY: all three are `#[repr(C)]` records of 4-byte fields with no
        // padding (the sizes are asserted above), read as plain bytes.
        unsafe {
            bytes.extend_from_slice(Self::instance_bytes(&layers));
            bytes.extend_from_slice(Self::instance_bytes(&scene.vn_clip_segments));
            bytes.extend_from_slice(Self::instance_bytes(&scene.vn_text_backgrounds));
        }
        self.write_instance_binding("vn_layers_bind_group", instance_offset, &bytes)
            .map(Some)
    }

    /// One isolated layer: its group into the target one level deeper, then
    /// the composite over `frame`.
    pub(super) fn draw_vn_layer(
        &mut self,
        draw: &GroupDraw<'_>,
        index: usize,
        depth: usize,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameTarget,
        instance_offset: &mut u64,
    ) -> Result<()> {
        if let Some(effect) = draw.scene.vn_effect_layer(index) {
            return self.draw_vn_effect_layer(
                draw,
                index,
                effect,
                depth,
                encoder,
                frame,
                instance_offset,
            );
        }
        let layer = draw.scene.vn_layers[index];
        let group = layer.group as usize;
        if let Some(id) = layer.snapshot_id() {
            let Some(layers) = draw.layers else {
                return Ok(());
            };
            let Some(view) = self
                .resources
                .vn_layers
                .snapshots
                .get(&id)
                .map(|snapshot| snapshot.texture.view.clone())
            else {
                return Ok(());
            };
            let Some(pipeline) = self.vn_layer_pipeline(layer.flags) else {
                return Ok(());
            };
            self.vn_composite(&layers.bind_group, &pipeline, &view, None, index, encoder, frame);
            return Ok(());
        }
        let Some(layers) = draw.layers else {
            return self.draw_group(draw, group, depth, encoder, frame, None, instance_offset);
        };
        let masked = layer.fill_rule >> 1 != 0;
        if depth + usize::from(masked) >= MAX_DEPTH {
            effects_advisory(&format!(
                "`clip-path` shapes and masks nested more than {MAX_DEPTH} deep are not applied \
                 below that depth — the content inside paints unclipped"
            ));
            return self.draw_group(draw, group, depth, encoder, frame, None, instance_offset);
        }
        let Some(pipeline) = self.vn_layer_pipeline(layer.flags) else {
            return self.draw_group(draw, group, depth, encoder, frame, None, instance_offset);
        };

        let target = self.draw_group_offscreen(draw, group, depth, encoder, frame, instance_offset)?;
        let mask = if masked {
            Some(self.draw_group_offscreen(draw, group + 1, depth + 1, encoder, frame, instance_offset)?)
        } else {
            None
        };
        let mask = mask.as_ref().map(|mask| &mask.view);
        self.vn_composite(&layers.bind_group, &pipeline, &target.view, mask, index, encoder, frame);
        Ok(())
    }

    /// Layer `index`'s composite from `source` over `frame`, through `mask`
    /// when it has one.
    #[allow(clippy::too_many_arguments)]
    fn vn_composite(
        &self,
        layers: &wgpu::BindGroup,
        pipeline: &wgpu::RenderPipeline,
        source: &wgpu::TextureView,
        mask: Option<&wgpu::TextureView>,
        index: usize,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameTarget,
    ) {
        let texture = self.create_texture_bind_group("vn_layer_texture", source);
        let mask = match mask {
            Some(mask) => self.create_texture_bind_group("vn_layer_mask", mask),
            None => texture.clone(),
        };
        let mut pass = continue_pass(encoder, &frame.view, "vn_layer_composite");
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &self.resources.globals_bind_group, &[]);
        pass.set_bind_group(1, layers, &[]);
        pass.set_bind_group(2, &texture, &[]);
        pass.set_bind_group(3, &mask, &[]);
        pass.draw(0..4, index as u32..index as u32 + 1);
    }

    pub(super) fn finish_vn_layers(&mut self) {
        self.resources.vn_layers.trim_targets();
    }

    /// Each capture's group into a cleared target, its ink copied into the
    /// capture's snapshot; then snapshots unused for a few renders go.
    pub(super) fn draw_vn_captures(
        &mut self,
        draw: &GroupDraw<'_>,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameTarget,
        instance_offset: &mut u64,
    ) -> Result<()> {
        if self.uses_webgl_instance_data {
            return Ok(());
        }
        let (width, height) = (frame.texture.width(), frame.texture.height());
        for capture in &draw.scene.vn_captures {
            let id = capture.info.id;
            let ink = capture.ink;
            let left = ink.origin.x.0.max(0.) as u32;
            let top = ink.origin.y.0.max(0.) as u32;
            let right = ((ink.origin.x.0 + ink.size.width.0).max(0.) as u32).min(width);
            let bottom = ((ink.origin.y.0 + ink.size.height.0).max(0.) as u32).min(height);
            if right <= left || bottom <= top {
                self.resources.vn_layers.snapshots.remove(&id);
                continue;
            }
            let target =
                self.draw_group_offscreen(draw, capture.group as usize, 0, encoder, frame, instance_offset)?;
            let (copy_width, copy_height) = (right - left, bottom - top);
            let reuse = self
                .resources
                .vn_layers
                .snapshots
                .get(&id)
                .is_some_and(|s| {
                    s.texture.texture.width() == copy_width
                        && s.texture.texture.height() == copy_height
                });
            if !reuse {
                let texture = create_target_texture(
                    &self.resources.device,
                    "vn_snapshot",
                    self.target_format,
                    copy_width,
                    copy_height,
                    wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                );
                self.resources
                    .vn_layers
                    .snapshots
                    .insert(id, Snapshot { texture, idle: 0 });
            }
            let Some(snapshot) = self.resources.vn_layers.snapshots.get_mut(&id) else {
                continue;
            };
            snapshot.idle = 0;
            encoder.copy_texture_to_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &target.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d {
                        x: left,
                        y: top,
                        z: 0,
                    },
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::TexelCopyTextureInfo {
                    texture: &snapshot.texture.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::Extent3d {
                    width: copy_width,
                    height: copy_height,
                    depth_or_array_layers: 1,
                },
            );
        }
        let state = &mut self.resources.vn_layers;
        if !state.snapshots.is_empty() {
            for layer in &draw.scene.vn_layers {
                if let Some(snapshot) = layer
                    .snapshot_id()
                    .and_then(|id| state.snapshots.get_mut(&id))
                {
                    snapshot.idle = 0;
                }
            }
            for snapshot in state.snapshots.values_mut() {
                snapshot.idle += 1;
            }
            state
                .snapshots
                .retain(|_, snapshot| snapshot.idle <= SNAPSHOT_IDLE_RENDERS);
        }
        Ok(())
    }

    /// The composite pipeline: source-over, or `plus-lighter`
    /// ([`vn_layer_flags::PLUS_LIGHTER`]).
    fn vn_layer_pipeline(&mut self, flags: u32) -> Option<wgpu::RenderPipeline> {
        let plus_lighter = flags & vn_layer_flags::PLUS_LIGHTER != 0;
        let format = self.target_format;
        let device = self.resources.device.clone();
        let layouts = &self.resources.bind_group_layouts;
        let state = &mut self.resources.vn_layers;
        let slot = if plus_lighter {
            &mut state.plus_lighter_pipeline
        } else {
            &mut state.pipeline
        };
        if let PipelineState::Unbuilt = slot {
            let layout = pipeline_layout(
                &device,
                "vn_layer_layout",
                &[
                    &layouts.globals,
                    &layouts.instances,
                    &layouts.texture,
                    &layouts.texture,
                ],
            );
            let blend = if plus_lighter {
                let add = wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::One,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                };
                wgpu::BlendState {
                    color: add,
                    alpha: add,
                }
            } else {
                wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING
            };
            *slot = match capture_validation(&device, || {
                let module = shader_module(&device, "vn_layer_shaders", VN_LAYER_SHADERS);
                render_pipeline(
                    &device,
                    "vn_layer",
                    &layout,
                    (&module, "vs_vn_layer"),
                    (&module, "fs_vn_layer"),
                    format,
                    Some(blend),
                )
            }) {
                Ok(pipeline) => PipelineState::Ready(pipeline),
                Err(error) => {
                    effects_advisory(&format!(
                        "`clip-path` shapes, `filter` color matrices and `background-clip: text` \
                         are unavailable — their pipeline failed to build: {error:#}"
                    ));
                    PipelineState::Unavailable
                }
            };
        }
        match slot {
            PipelineState::Ready(pipeline) => Some(pipeline.clone()),
            PipelineState::Unbuilt | PipelineState::Unavailable => None,
        }
    }

    /// The target for `depth`, the frame's size; a backdrop drawn inside the
    /// group copies from it, so the group is its Backdrop Root.
    pub(super) fn vn_layer_target(&mut self, depth: usize, frame: &FrameTarget) -> FrameTarget {
        let (width, height) = (frame.texture.width(), frame.texture.height());
        let state = &mut self.resources.vn_layers;
        if state.targets.first().is_some_and(|target| {
            target.texture.width() != width || target.texture.height() != height
        }) {
            state.targets.clear();
        }
        while state.targets.len() <= depth {
            state.targets.push(create_target_texture(
                &self.resources.device,
                "vn_layer_target",
                self.target_format,
                width,
                height,
                wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
            ));
        }
        state.depth_used = state.depth_used.max(depth + 1);
        let target = &state.targets[depth];
        FrameTarget::layer(target.texture.clone(), target.view.clone(), frame.backdrop)
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "windows")))]
mod tests {
    use super::*;
    use crate::WgpuHeadlessRenderer;
    use anyhow::Context as _;
    use gpui::{
        AtlasKey, BackdropBlurRect, Bounds, ContentMask, DevicePixels, MonochromeSprite,
        PlatformHeadlessRenderer, Quad, RenderSvgParams, Rgba, ScaledPixels, TransformationMatrix,
        VnCaptureInfo, VnClipGeometry, VnColorMatrix, VnFillRule, VnIsolation, VnSnapshotPaint,
        VnTextFill, VnTextPaint, linear_color_stop, linear_gradient, point, rgb, size,
    };
    use std::{borrow::Cow, sync::Arc};

    const SIDE: f32 = 64.;

    fn frame_size() -> gpui::Size<DevicePixels> {
        size(DevicePixels(SIDE as i32), DevicePixels(SIDE as i32))
    }

    fn bounds(x: f32, y: f32, width: f32, height: f32) -> Bounds<ScaledPixels> {
        Bounds::new(
            point(ScaledPixels(x), ScaledPixels(y)),
            size(ScaledPixels(width), ScaledPixels(height)),
        )
    }

    fn mask() -> ContentMask<ScaledPixels> {
        ContentMask::rect(bounds(0., 0., SIDE, SIDE))
    }

    fn quad(rect: Bounds<ScaledPixels>, color: Rgba) -> Quad {
        Quad {
            bounds: rect,
            background: color.into(),
            content_mask: mask(),
            ..Default::default()
        }
    }

    fn polygon(points: &[[f32; 2]]) -> Arc<VnClipGeometry> {
        let segments = (0..points.len())
            .map(|i| {
                let [x0, y0] = points[i];
                let [x1, y1] = points[(i + 1) % points.len()];
                [x0, y0, x1, y1]
            })
            .collect();
        let (mut min, mut max) = ([f32::MAX; 2], [f32::MIN; 2]);
        for [x, y] in points {
            min = [min[0].min(*x), min[1].min(*y)];
            max = [max[0].max(*x), max[1].max(*y)];
        }
        Arc::new(VnClipGeometry {
            fill_rule: VnFillRule::NonZero,
            segments,
            bounds: bounds(min[0], min[1], max[0] - min[0], max[1] - min[1]),
        })
    }

    fn rectangle(x: f32, y: f32, width: f32, height: f32) -> Arc<VnClipGeometry> {
        polygon(&[
            [x, y],
            [x + width, y],
            [x + width, y + height],
            [x, y + height],
        ])
    }

    fn isolation(
        geometry: Arc<VnClipGeometry>,
        matrix: Option<[f32; 20]>,
        linear_rgb: bool,
    ) -> VnIsolation {
        VnIsolation {
            geometry,
            color_matrix: matrix.map(|values| VnColorMatrix {
                group_id: 0,
                values,
                linear_rgb,
            }),
            text_paint: None,
            content_mask: mask(),
            capture: None,
        }
    }

    fn render(renderer: &mut WgpuHeadlessRenderer, mut scene: Scene) -> Result<image::RgbaImage> {
        scene.finish();
        renderer.render_scene_to_image(&scene, frame_size())
    }

    /// Over a transparent clear, as a transparent window draws.
    fn render_transparent(
        renderer: &mut WgpuHeadlessRenderer,
        scene: &Scene,
    ) -> Result<image::RgbaImage> {
        renderer.ensure_render_target(frame_size())?;
        let target = renderer
            .render_target
            .as_ref()
            .context("render target missing")?;
        let (texture, view) = (target.texture.clone(), target.view.clone());
        renderer.core.render_frame(
            scene,
            &texture,
            &view,
            frame_size(),
            false,
            wgpu::Color::TRANSPARENT,
        )?;
        renderer.read_image()
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

    const LUMINANCE: [f32; 20] = [
        0.2126, 0.7152, 0.0722, 0., 0., 0.2126, 0.7152, 0.0722, 0., 0., 0.2126, 0.7152, 0.0722, 0.,
        0., 0., 0., 0., 1., 0.,
    ];

    #[test]
    fn a_clip_path_cuts_its_group_to_the_shape() -> Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x0000ff)));
        scene.vn_push_isolation(isolation(
            polygon(&[[0., 0.], [SIDE, 0.], [0., SIDE]]),
            None,
            false,
        ));
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0xff0000)));
        scene.vn_pop_isolation();
        let image = render(&mut renderer, scene)?;

        assert_close(
            pixel(&image, 10, 10),
            [255, 0, 0, 255],
            "inside the triangle",
        );
        assert_close(
            pixel(&image, 40, 12),
            [255, 0, 0, 255],
            "inside, near the edge",
        );
        assert_close(
            pixel(&image, 54, 54),
            [0, 0, 255, 255],
            "outside the triangle",
        );
        let edge = pixel(&image, 31, 32);
        assert!(
            (100..156).contains(&edge[0]) && (100..156).contains(&edge[2]),
            "the edge is antialiased: {edge:?}"
        );
        Ok(())
    }

    #[test]
    fn a_color_matrix_filters_its_group_in_linear_or_srgb() -> Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let mut scene = Scene::default();
        for (x, linear_rgb) in [(0., true), (SIDE / 2., false)] {
            scene.vn_push_isolation(isolation(
                rectangle(x, 0., SIDE / 2., SIDE),
                Some(LUMINANCE),
                linear_rgb,
            ));
            scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0xff0000)));
            scene.vn_pop_isolation();
        }
        let image = render(&mut renderer, scene)?;

        // linear 0.2126 is sRGB 0.4985
        assert_close(pixel(&image, 16, 32), [127, 127, 127, 255], "linearRGB");
        assert_close(pixel(&image, 48, 32), [54, 54, 54, 255], "sRGB");
        Ok(())
    }

    #[test]
    fn nested_groups_clip_through_both_shapes_and_filter_once_each() -> Result<()> {
        let swap_red_blue = [
            0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
        ];
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x00ff00)));
        scene.vn_push_isolation(isolation(
            rectangle(0., 0., SIDE / 2., SIDE),
            Some(swap_red_blue),
            false,
        ));
        scene.vn_push_isolation(isolation(rectangle(0., 0., SIDE, SIDE / 2.), None, false));
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0xff0000)));
        scene.vn_pop_isolation();
        scene.vn_pop_isolation();
        let image = render(&mut renderer, scene)?;

        assert_close(pixel(&image, 16, 16), [0, 0, 255, 255], "inside both");
        assert_close(
            pixel(&image, 48, 16),
            [0, 255, 0, 255],
            "outside the outer shape",
        );
        assert_close(
            pixel(&image, 16, 48),
            [0, 255, 0, 255],
            "outside the inner shape",
        );
        Ok(())
    }

    #[test]
    fn a_backdrop_blur_in_a_group_reads_only_the_group() -> Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x000000)));
        scene.insert_primitive(quad(bounds(0., 0., SIDE / 2., SIDE), rgb(0xffffff)));
        scene.vn_push_isolation(isolation(rectangle(0., 0., SIDE, SIDE), None, false));
        scene.insert_primitive(BackdropBlurRect {
            bounds: bounds(8., 8., 48., 48.),
            content_mask: mask(),
            blur_radius: ScaledPixels(10.),
            ..Default::default()
        });
        scene.vn_pop_isolation();
        let image = render(&mut renderer, scene)?;

        assert_eq!(
            pixel(&image, 30, 32),
            [255; 4],
            "the page's edge stays sharp"
        );
        assert_eq!(
            pixel(&image, 33, 32),
            [0, 0, 0, 255],
            "the page's edge stays sharp"
        );
        Ok(())
    }

    fn snapshot_into(
        id: u64,
        rect: Bounds<ScaledPixels>,
        opacity: f32,
        plus_lighter: bool,
    ) -> VnSnapshotPaint {
        let (x, y) = (rect.origin.x.0, rect.origin.y.0);
        let (w, h) = (rect.size.width.0, rect.size.height.0);
        VnSnapshotPaint {
            id,
            to_unit: [1. / w, 0., -x / w, 0., 1. / h, -y / h],
            bounds: rect,
            content_mask: mask(),
            opacity,
            plus_lighter,
            in_place: false,
        }
    }

    #[test]
    fn a_capture_is_kept_and_composites_later_through_its_box() -> Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let info = VnCaptureInfo {
            id: 7,
            border_box: bounds(20., 20., 20., 20.),
            transform: TransformationMatrix::unit(),
        };
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0xffffff)));
        scene.vn_push_isolation(VnIsolation {
            capture: Some(info),
            ..isolation(rectangle(0., 0., SIDE, SIDE), None, false)
        });
        scene.insert_primitive(quad(bounds(20., 20., 20., 20.), rgb(0xff0000)));
        scene.vn_pop_isolation();
        scene.finish();
        let ink = scene.vn_captures[0].ink;
        let image = renderer.render_scene_to_image(&scene, frame_size())?;
        assert_close(
            pixel(&image, 30, 30),
            [255; 4],
            "a capture does not paint the page",
        );

        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0xffffff)));
        scene.vn_insert_snapshot(snapshot_into(7, bounds(34., 34., 28., 28.), 1., false));
        scene.finish();
        scene.vn_resolve_snapshots(|id| (id == 7).then_some((info, ink)));
        let image = renderer.render_scene_to_image(&scene, frame_size())?;
        assert_close(
            pixel(&image, 48, 48),
            [255, 0, 0, 255],
            "the snapshot in its new box",
        );
        assert_close(
            pixel(&image, 30, 30),
            [255; 4],
            "nothing where it was captured",
        );

        let mut scene = Scene::default();
        scene.vn_push_isolation(isolation(rectangle(0., 0., SIDE, SIDE), None, false));
        for _ in 0..2 {
            scene.vn_insert_snapshot(snapshot_into(7, bounds(0., 0., SIDE, SIDE), 0.5, true));
        }
        scene.vn_pop_isolation();
        scene.finish();
        scene.vn_resolve_snapshots(|id| (id == 7).then_some((info, ink)));
        let image = render_transparent(&mut renderer, &scene)?;
        assert_close(
            pixel(&image, 32, 32),
            [255, 0, 0, 255],
            "plus-lighter halves sum to opaque",
        );
        Ok(())
    }

    /// The scene of gpui_windows' `native_masks_multiply_by_alpha_or_luminance` and
    /// gpui_apple's `masks_multiply_by_alpha_or_luminance`.
    #[test]
    fn masks_multiply_by_alpha_or_luminance() -> Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let image = render(&mut renderer, mask_scene())?;

        assert_close(pixel(&image, 8, 16), [128, 0, 127, 255], "half alpha");
        assert_close(pixel(&image, 24, 16), [0, 0, 255, 255], "outside the mask");
        assert_close(pixel(&image, 40, 16), [182, 0, 73, 255], "green luminance");
        assert_close(pixel(&image, 8, 48), [0, 0, 255, 255], "an empty mask");
        assert_close(pixel(&image, 40, 40), [255, 0, 0, 255], "inside a clip");
        assert_close(pixel(&image, 56, 40), [0, 0, 255, 255], "masked inside a clip");
        assert_eq!(
            renderer.core.resources.vn_layers.targets.len(),
            3,
            "a nested mask draws two levels down"
        );
        Ok(())
    }

    fn mask_scene() -> Scene {
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x0000ff)));

        scene.vn_push_isolation(isolation(rectangle(0., 0., 32., 32.), None, false));
        scene.insert_primitive(quad(bounds(0., 0., 32., 32.), rgb(0xff0000)));
        scene.vn_begin_mask(false);
        scene.insert_primitive(quad(bounds(0., 0., 16., 32.), gpui::rgba(0xffffff80)));
        scene.vn_pop_isolation();

        scene.vn_push_isolation(isolation(rectangle(32., 0., 32., 32.), None, false));
        scene.insert_primitive(quad(bounds(32., 0., 32., 32.), rgb(0xff0000)));
        scene.vn_begin_mask(true);
        scene.insert_primitive(quad(bounds(32., 0., 32., 32.), rgb(0x00ff00)));
        scene.vn_pop_isolation();

        scene.vn_push_isolation(isolation(rectangle(0., 32., 32., 32.), None, false));
        scene.insert_primitive(quad(bounds(0., 32., 32., 32.), rgb(0xff0000)));
        scene.vn_begin_mask(false);
        scene.vn_pop_isolation();

        scene.vn_push_isolation(isolation(rectangle(32., 32., 32., 32.), None, false));
        scene.vn_push_isolation(isolation(rectangle(32., 32., 32., 32.), None, false));
        scene.insert_primitive(quad(bounds(32., 32., 32., 32.), rgb(0xff0000)));
        scene.vn_begin_mask(false);
        scene.insert_primitive(quad(bounds(32., 32., 16., 32.), rgb(0xffffff)));
        scene.vn_pop_isolation();
        scene.vn_pop_isolation();
        scene
    }

    /// The scene of gpui_windows' `native_nested_layers_apply_text_gradients_clips_and_color_matrix`.
    #[test]
    fn nested_layers_apply_text_gradients_clips_and_color_matrix() -> Result<()> {
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let tile_size = size(DevicePixels(8), DevicePixels(8));
        let mut coverage = vec![255u8; 64];
        for y in 2..6 {
            for x in 2..6 {
                coverage[y * 8 + x] = 0;
            }
        }
        let tile = renderer
            .sprite_atlas()
            .get_or_insert_with(
                AtlasKey::Svg(RenderSvgParams {
                    path: "vn-layer-gradient-mask".into(),
                    size: tile_size,
                }),
                &mut || Ok(Some((tile_size, Cow::Borrowed(coverage.as_slice())))),
            )?
            .context("glyph mask tile was not allocated")?;
        let triangle = polygon(&[[4., 4.], [60., 4.], [4., 60.]]);
        let text_paint = Arc::new(VnTextPaint {
            fill: VnTextFill {
                group_id: 1,
                backgrounds: vec![linear_gradient(
                    90.,
                    linear_color_stop(rgb(0xff0000), 0.),
                    linear_color_stop(rgb(0x0000ff), 1.),
                )],
            },
            bounds: [0., 0., SIDE, SIDE],
            inverse: [1., 0., 0., 1., 0., 0.],
            scale_factor: 1.,
            geometry: triangle.clone(),
            content_mask: mask(),
        });
        let mut half_alpha = VnColorMatrix::IDENTITY;
        half_alpha[18] = 0.5;
        let mut scene = Scene::default();
        scene.vn_push_isolation(VnIsolation {
            text_paint: Some(text_paint),
            ..isolation(triangle, None, false)
        });
        scene.vn_push_isolation(VnIsolation {
            color_matrix: Some(VnColorMatrix {
                group_id: 2,
                values: half_alpha,
                linear_rgb: false,
            }),
            ..isolation(rectangle(0., 0., SIDE, SIDE), None, false)
        });
        scene.insert_primitive(MonochromeSprite {
            order: 0,
            raw_coverage: false.into(),
            tile,
            bounds: bounds(0., 0., SIDE, SIDE),
            content_mask: mask(),
            color: rgb(0xffffff).into(),
            transformation: TransformationMatrix::unit(),
            animation: gpui::SpriteAnimationId::NONE,
            pad: 0,
        });
        scene.vn_pop_isolation();
        scene.vn_pop_isolation();
        scene.finish();
        let image = render_transparent(&mut renderer, &scene)?;

        let left = pixel(&image, 8, 8);
        let right = pixel(&image, 46, 8);
        assert!(left[3].abs_diff(128) <= 2, "{left:?}");
        assert!(right[3].abs_diff(128) <= 2, "{right:?}");
        assert!(
            left[0] > left[2] + 30,
            "left side of the masked gradient must be red: {left:?}"
        );
        assert!(
            right[2] > right[0] + 20,
            "right side of the masked gradient must be blue: {right:?}"
        );
        assert_eq!(
            pixel(&image, 24, 24),
            [0; 4],
            "the atlas mask's hole must remain transparent"
        );
        assert_eq!(
            pixel(&image, 55, 55),
            [0; 4],
            "pixels outside the triangle clip must remain transparent"
        );

        assert_eq!(renderer.core.resources.vn_layers.targets.len(), 2);
        for _ in 0..TARGET_RELEASE_FRAMES {
            render_transparent(&mut renderer, &Scene::default())?;
        }
        assert!(
            renderer.core.resources.vn_layers.targets.is_empty(),
            "closed isolated groups must release their targets"
        );
        Ok(())
    }
}
