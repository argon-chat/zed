//! Isolated layers (`gpui::VnLayer`), ported from the DirectX and wgpu
//! renderers: each group draws into a frame-sized target of its own, cleared
//! to transparent, which is then composited over its parent through the
//! layer's clip shape, color matrix and text background stack.

use super::backdrop::{FrameTarget, create_target_texture, effects_advisory};
use super::{
    GroupDraw, InstanceBinding, InstanceBufferWriter, MetalRenderer, VnLayerInputIndex,
    build_path_sprite_pipeline_state, new_command_encoder_for_texture,
};
use anyhow::{Context as _, Result};
use gpui::Scene;
use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_core_foundation::CFRetained;
use objc2_core_video::CVMetalTexture;
use objc2_metal::{
    MTLClearColor, MTLCommandBuffer, MTLCommandEncoder, MTLPixelFormat, MTLPrimitiveType,
    MTLRenderCommandEncoder, MTLRenderPipelineState, MTLTexture, MTLTextureUsage,
};
use std::{mem, ptr};

/// Deeper groups draw inline, unclipped by their shape, as on DirectX.
pub(super) const MAX_DEPTH: usize = 8;

/// Frames the layer targets may sit unused before release.
const TARGET_RELEASE_FRAMES: u32 = 90;

type Texture = Retained<ProtocolObject<dyn MTLTexture>>;

#[derive(Default)]
pub(super) struct VnLayerState {
    pipeline: PipelineState,
    /// One per nesting depth.
    targets: Vec<Texture>,
    depth_used: usize,
    idle_frames: u32,
}

#[derive(Default)]
enum PipelineState {
    #[default]
    Unbuilt,
    Ready(Retained<ProtocolObject<dyn MTLRenderPipelineState>>),
    Unavailable,
}

/// The frame's layers, clip segments and text backgrounds.
pub(super) struct VnLayerBindings {
    layers: InstanceBinding,
    segments: InstanceBinding,
    text_backgrounds: InstanceBinding,
}

impl MetalRenderer {
    /// `None` when the frame has no layer.
    pub(super) fn write_vn_layers(
        &mut self,
        scene: &Scene,
        writer: &mut InstanceBufferWriter,
    ) -> Result<Option<VnLayerBindings>> {
        self.vn_layers.depth_used = 0;
        if scene.vn_layers.is_empty() {
            return Ok(None);
        }
        Ok(Some(VnLayerBindings {
            layers: writer.write(&scene.vn_layers)?,
            segments: writer.write(&scene.vn_clip_segments)?,
            text_backgrounds: writer.write(&scene.vn_text_backgrounds)?,
        }))
    }

    /// One isolated layer: its group into the target one level deeper, then
    /// the composite over `frame`. Ends every encoder it opens.
    pub(super) fn draw_vn_layer(
        &mut self,
        draw: &GroupDraw<'_>,
        index: usize,
        depth: usize,
        command_buffer: &ProtocolObject<dyn MTLCommandBuffer>,
        frame: &FrameTarget,
        writer: &mut InstanceBufferWriter,
        surface_textures: &mut Vec<CFRetained<CVMetalTexture>>,
    ) -> Result<()> {
        if let Some(effect) = draw.scene.vn_effect_layer(index) {
            return self.draw_vn_effect_layer(
                draw,
                index,
                effect,
                depth,
                command_buffer,
                frame,
                writer,
                surface_textures,
            );
        }
        let layer = draw.scene.vn_layers[index];
        let group = layer.group as usize;
        let inline = |this: &mut Self,
                      writer: &mut InstanceBufferWriter,
                      surfaces: &mut Vec<CFRetained<CVMetalTexture>>| {
            this.draw_group(
                draw,
                group,
                depth,
                command_buffer,
                frame,
                None,
                writer,
                surfaces,
            )
        };
        let Some(bindings) = draw.layers else {
            return inline(self, writer, surface_textures);
        };
        if depth >= MAX_DEPTH {
            effects_advisory(&format!(
                "`clip-path` shapes nested more than {MAX_DEPTH} deep are not applied below that \
                 depth — the content inside paints unclipped"
            ));
            return inline(self, writer, surface_textures);
        }
        let Some(pipeline) = self.vn_layer_pipeline() else {
            return inline(self, writer, surface_textures);
        };

        let target = self.vn_layer_target(depth, frame)?;
        self.draw_group(
            draw,
            group,
            depth + 1,
            command_buffer,
            &target,
            Some(MTLClearColor {
                red: 0.,
                green: 0.,
                blue: 0.,
                alpha: 0.,
            }),
            writer,
            surface_textures,
        )?;
        let viewport_size = draw.viewport_size;
        let encoder =
            new_command_encoder_for_texture(command_buffer, &frame.texture, viewport_size, None)?;
        encoder.setRenderPipelineState(&pipeline);
        // SAFETY: the bindings hold the scene's layers, segments and text
        // backgrounds in the layouts shaders.metal declares through cbindgen,
        // at the slots it reads them from; `index` is one of the layers. Metal
        // retains the buffers and the target, and the viewport bytes are copied.
        unsafe {
            encoder.setVertexBuffer_offset_atIndex(
                Some(&bindings.layers.buffer),
                bindings.layers.offset,
                VnLayerInputIndex::Layers as usize,
            );
            encoder.setVertexBytes_length_atIndex(
                ptr::NonNull::from(&viewport_size).cast(),
                mem::size_of_val(&viewport_size),
                VnLayerInputIndex::ViewportSize as usize,
            );
            encoder.setFragmentBuffer_offset_atIndex(
                Some(&bindings.layers.buffer),
                bindings.layers.offset,
                VnLayerInputIndex::Layers as usize,
            );
            encoder.setFragmentBuffer_offset_atIndex(
                Some(&bindings.segments.buffer),
                bindings.segments.offset,
                VnLayerInputIndex::Segments as usize,
            );
            encoder.setFragmentBuffer_offset_atIndex(
                Some(&bindings.text_backgrounds.buffer),
                bindings.text_backgrounds.offset,
                VnLayerInputIndex::TextBackgrounds as usize,
            );
            encoder.setFragmentTexture_atIndex(
                Some(&target.texture),
                VnLayerInputIndex::Texture as usize,
            );
            encoder.drawPrimitives_vertexStart_vertexCount_instanceCount_baseInstance(
                MTLPrimitiveType::TriangleStrip,
                0,
                4,
                1,
                index,
            );
        }
        encoder.endEncoding();
        Ok(())
    }

    pub(super) fn finish_vn_layers(&mut self) {
        let state = &mut self.vn_layers;
        if state.depth_used >= state.targets.len() {
            state.idle_frames = 0;
            return;
        }
        state.idle_frames += 1;
        if state.idle_frames >= TARGET_RELEASE_FRAMES {
            state.idle_frames = 0;
            state.targets.truncate(state.depth_used);
        }
    }

    fn vn_layer_pipeline(
        &mut self,
    ) -> Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>> {
        if let PipelineState::Unbuilt = self.vn_layers.pipeline {
            self.vn_layers.pipeline = match build_path_sprite_pipeline_state(
                &self.device,
                &self.library,
                "vn_layers",
                "vn_layer_vertex",
                "vn_layer_fragment",
                MTLPixelFormat::BGRA8Unorm,
            ) {
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
        match &self.vn_layers.pipeline {
            PipelineState::Ready(pipeline) => Some(pipeline.clone()),
            PipelineState::Unbuilt | PipelineState::Unavailable => None,
        }
    }

    /// The target for `depth`, the frame's size; a backdrop drawn inside the
    /// group copies from it, so the group is its Backdrop Root.
    pub(super) fn vn_layer_target(
        &mut self,
        depth: usize,
        frame: &FrameTarget,
    ) -> Result<FrameTarget> {
        let (width, height) = (frame.texture.width(), frame.texture.height());
        let state = &mut self.vn_layers;
        if state
            .targets
            .first()
            .is_some_and(|target| target.width() != width || target.height() != height)
        {
            state.targets.clear();
        }
        while state.targets.len() <= depth {
            state.targets.push(create_target_texture(
                &self.device,
                "vn_layer_target",
                width,
                height,
                MTLTextureUsage::RenderTarget | MTLTextureUsage::ShaderRead,
            )?);
        }
        state.depth_used = state.depth_used.max(depth + 1);
        let target = state
            .targets
            .get(depth)
            .context("isolated layer target missing")?;
        Ok(FrameTarget::layer(target.clone(), frame.backdrop))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::{
        AtlasKey, BackdropBlurRect, Bounds, ContentMask, DevicePixels, MonochromeSprite,
        PlatformAtlas, Quad, RenderSvgParams, Rgba, ScaledPixels, TransformationMatrix,
        VnClipGeometry, VnColorMatrix, VnFillRule, VnIsolation, VnTextFill, VnTextPaint,
        linear_color_stop, linear_gradient, point, rgb, size,
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
        }
    }

    fn render(renderer: &mut MetalRenderer, mut scene: Scene) -> Result<image::RgbaImage> {
        scene.finish();
        renderer.render_scene_to_image(&scene, frame_size())
    }

    /// Over a transparent clear, as a transparent window draws.
    fn render_transparent(renderer: &mut MetalRenderer, scene: &Scene) -> Result<image::RgbaImage> {
        renderer.opaque = false;
        renderer.render_scene_to_image(scene, frame_size())
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
        let mut renderer = MetalRenderer::new_headless(Default::default())?;
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
        let mut renderer = MetalRenderer::new_headless(Default::default())?;
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
        let mut renderer = MetalRenderer::new_headless(Default::default())?;
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
        let mut renderer = MetalRenderer::new_headless(Default::default())?;
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

    /// The scene of gpui_windows' `native_nested_layers_apply_text_gradients_clips_and_color_matrix`.
    #[test]
    fn nested_layers_apply_text_gradients_clips_and_color_matrix() -> Result<()> {
        let mut renderer = MetalRenderer::new_headless(Default::default())?;
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

        assert_eq!(renderer.vn_layers.targets.len(), 2);
        for _ in 0..TARGET_RELEASE_FRAMES {
            render_transparent(&mut renderer, &Scene::default())?;
        }
        assert!(
            renderer.vn_layers.targets.is_empty(),
            "closed isolated groups must release their targets"
        );
        Ok(())
    }
}
