//! Effect layers (`gpui::VnEffectLayer`): an `[SkContent]` effect composites
//! its element's group in the group's place, and an `[SkHistory]` effect steps
//! its element's state before it draws.

use super::backdrop::{
    BackdropState, FrameTarget, HISTORY_FORMAT, create_target_texture, effects_advisory,
};
use super::vn_layer::MAX_DEPTH;
use super::{GroupDraw, WgpuRendererCore, continue_pass};
use anyhow::{Context as _, Result};
use gpui::{EFFECT_FLAG_HISTORY, EffectQuad, Scene, VnEffectLayer, effect_id};

/// One frame's effect quads as uploaded: the scene's own, then one per
/// `Scene::vn_effect_draws` entry, then the history steps.
#[derive(Default)]
pub(super) struct VnEffectFrame {
    pub(super) quads: Vec<EffectQuad>,
    histories: Vec<Option<DrawHistory>>,
}

struct DrawHistory {
    read: wgpu::TextureView,
    /// The texture this frame's step writes and the step's quad.
    step: Option<(wgpu::TextureView, u32)>,
}

impl VnEffectFrame {
    pub(super) fn effect_quads<'a>(&'a self, scene: &'a Scene) -> &'a [EffectQuad] {
        if self.quads.is_empty() {
            &scene.effect_quads
        } else {
            &self.quads
        }
    }
}

impl WgpuRendererCore {
    /// Takes this frame's history steps and the quads they draw.
    pub(super) fn prepare_vn_effects(&mut self, scene: &Scene) -> VnEffectFrame {
        let mut frame = VnEffectFrame::default();
        if scene.vn_effect_draws.is_empty() || !self.ensure_backdrop() {
            return frame;
        }
        let resources = &mut self.resources;
        let BackdropState::Ready(state) = &mut resources.backdrop else {
            return frame;
        };
        state.sync_app_effects(&resources.device);
        let first_step = scene.effect_quads.len() + scene.vn_effect_draws.len();
        frame.quads.reserve(first_step);
        frame.quads.extend_from_slice(&scene.effect_quads);
        let mut steps = Vec::new();
        for draw in &scene.vn_effect_draws {
            let mut quad = draw.quad;
            let history = if quad.has_history() && state.history_effect(quad.effect_id).is_some() {
                let device = &resources.device;
                state.histories.prepare(
                    draw.history,
                    quad.history_size(),
                    quad.time,
                    |width, height| {
                        Some(create_target_texture(
                            device,
                            "vn_effect_history",
                            HISTORY_FORMAT,
                            width,
                            height,
                            wgpu::TextureUsages::RENDER_ATTACHMENT
                                | wgpu::TextureUsages::TEXTURE_BINDING,
                        ))
                    },
                )
            } else {
                None
            };
            if history.is_none() {
                quad.flags &= !EFFECT_FLAG_HISTORY;
            }
            frame.quads.push(quad);
            // A new texture is zeroed by wgpu, so `fresh` needs no clear here.
            frame.histories.push(history.map(|history| DrawHistory {
                read: history.read.view,
                step: history.write.map(|write| {
                    steps.push(quad.history_pass());
                    (write.view, (first_step + steps.len() - 1) as u32)
                }),
            }));
        }
        frame.quads.extend(steps);
        frame
    }

    pub(super) fn finish_vn_effects(&mut self) {
        if let BackdropState::Ready(state) = &mut self.resources.backdrop {
            state.histories.finish_frame();
        }
    }

    /// Renders `group` (an element's subtree) into the offscreen target of
    /// `depth`, cleared to transparent and the size of `frame`; what draws
    /// inside it sees that target as its Backdrop Root.
    pub(super) fn draw_group_offscreen(
        &mut self,
        draw: &GroupDraw<'_>,
        group: usize,
        depth: usize,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameTarget,
        instance_offset: &mut u64,
    ) -> Result<FrameTarget> {
        let target = self.vn_layer_target(depth, frame);
        self.draw_group(
            draw,
            group,
            depth + 1,
            encoder,
            &target,
            Some(wgpu::Color::TRANSPARENT),
            instance_offset,
        )?;
        Ok(target)
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn draw_vn_effect_layer(
        &mut self,
        draw: &GroupDraw<'_>,
        index: usize,
        effect: VnEffectLayer,
        depth: usize,
        encoder: &mut wgpu::CommandEncoder,
        frame: &FrameTarget,
        instance_offset: &mut u64,
    ) -> Result<()> {
        let group = draw.scene.vn_layers[index].group as usize;
        let drawable = frame.backdrop
            && self.resources.backdrop.ready().is_some()
            && !draw.effects.quads.is_empty();
        if !drawable || depth >= MAX_DEPTH {
            if depth >= MAX_DEPTH {
                effects_advisory(&format!(
                    "`[SkContent]` effects nested more than {MAX_DEPTH} deep are not drawn \
                     below that depth — the element paints unshaded"
                ));
            }
            return match group {
                0 => Ok(()),
                group => self.draw_group(draw, group, depth, encoder, frame, None, instance_offset),
            };
        }
        let content = match group {
            0 => None,
            group => Some(
                self.draw_group_offscreen(draw, group, depth, encoder, frame, instance_offset)?
                    .view,
            ),
        };

        let instances = &draw.instances.effect_quads;
        let base = draw.scene.effect_quads.len();
        for index in effect.draws() {
            let quad = draw.effects.quads[base + index];
            let history = draw.effects.histories[index].as_ref();
            let layer = [content.as_ref(), history.map(|history| &history.read)];
            let images = quad.texture_set();
            let state = self
                .resources
                .backdrop
                .ready()
                .context("effect pipelines missing")?;

            if let Some((write, step)) = history.and_then(|history| history.step.as_ref())
                && let Some(pipeline) = state.history_effect(quad.effect_id)
            {
                let textures =
                    state.effect_textures(&self.resources.device, [None, None], layer, images);
                let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("vn_effect_history_step"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: write,
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
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &state.effect_globals_bind_group, &[]);
                pass.set_bind_group(1, &instances.bind_group, &[]);
                pass.set_bind_group(2, &textures, &[]);
                let step = instances.first_instance + step;
                pass.draw(0..4, step..step + 1);
            }

            let instance = instances.first_instance + (base + index) as u32;
            // An effect whose pipeline did not build leaves its share to the
            // content as it is.
            let (effect_id, backdrop) = match state.effect(quad.effect_id) {
                Some(_) => (quad.effect_id, quad.needs_backdrop()),
                None if quad.needs_content() => (effect_id::CONTENT, false),
                None => continue,
            };
            if backdrop {
                let textures = self.snapshot_layer_backdrop(
                    encoder,
                    frame,
                    quad.effective_kernel_levels() as usize,
                    layer,
                    images,
                )?;
                let state = self
                    .resources
                    .backdrop
                    .ready()
                    .context("effect pipelines missing")?;
                let pipeline = state.effect(effect_id).context("effect pipeline missing")?;
                let mut pass = continue_pass(encoder, &frame.view, "vn_effect_backdrop");
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &state.effect_globals_bind_group, &[]);
                pass.set_bind_group(1, &instances.bind_group, &[]);
                pass.set_bind_group(2, &textures, &[]);
                pass.draw(0..4, instance..instance + 1);
            } else {
                let Some(pipeline) = state.effect(effect_id) else {
                    continue;
                };
                let textures =
                    state.effect_textures(&self.resources.device, [None, None], layer, images);
                let mut pass = continue_pass(encoder, &frame.view, "vn_effect_composite");
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, &state.effect_globals_bind_group, &[]);
                pass.set_bind_group(1, &instances.bind_group, &[]);
                pass.set_bind_group(2, &textures, &[]);
                pass.draw(0..4, instance..instance + 1);
            }
        }
        Ok(())
    }
}

#[cfg(all(test, any(target_os = "linux", target_os = "windows")))]
mod tests {
    use super::super::backdrop::tests::gpu::APP_EFFECTS;
    use crate::WgpuHeadlessRenderer;
    use anyhow::{Context as _, Result};
    use gpui::{
        Bounds, ContentMask, DevicePixels, EFFECT_FLAG_HISTORY, EFFECT_FLAG_NEEDS_CONTENT,
        EffectQuad, PlatformHeadlessRenderer, Quad, Rgba, ScaledPixels, Scene, VnEffectDraw,
        VnEffectPaint, effect_id, point, rgb, size,
    };

    const SIDE: f32 = 64.;

    /// `swap` shows its content with the channels rotated (green turns blue),
    /// or with its first parameter set, its first image where the content is;
    /// `count` shows the red of its history, and each step adds 0.25 to it.
    const EFFECTS_WGSL: &str = r#"
        @group(0) @binding(0) var<uniform> globals: array<vec4<f32>, 3>;
        @group(1) @binding(0) var<storage, read> quads: array<array<f32, 56>>;
        @group(2) @binding(1) var clamp_sampler: sampler;
        @group(2) @binding(3) var image: texture_2d<f32>;
        @group(2) @binding(7) var content: texture_2d<f32>;
        @group(2) @binding(8) var history: texture_2d<f32>;

        struct Varying {
            @builtin(position) position: vec4<f32>,
            @location(0) @interpolate(flat) id: u32,
        }

        @fragment
        fn effect_fragment(input: Varying) -> @location(0) vec4<f32> {
            let quad = quads[input.id];
            let flags = bitcast<u32>(quad[55]);
            let origin = vec2<f32>(quad[2], quad[3]);
            let step = (flags & 8u) != 0u;
            let position = select(input.position.xy, origin + input.position.xy, step);
            var last = vec4<f32>(0.0);
            if (flags & 4u) != 0u {
                last = textureLoad(history, vec2<i32>(position - origin), 0);
            }
            if step {
                return last + vec4<f32>(0.25, 0.0, 0.0, 0.0);
            }
            if (flags & 4u) != 0u {
                return vec4<f32>(last.r, 0.0, 0.0, 1.0) * quad[50];
            }
            let shown = textureSampleLevel(content, clamp_sampler, position / globals[1].xy, 0.0);
            if quad[18] > 0.5 {
                return textureLoad(image, vec2<i32>(0, 0), 0) * shown.a * quad[50];
            }
            return vec4<f32>(shown.b, shown.r, shown.g, shown.a) * quad[50];
        }
    "#;

    const SWAP: u32 = effect_id::COUNT;
    const COUNT: u32 = effect_id::COUNT + 1;

    fn install_effects() -> Result<()> {
        gpui::reload_app_effects(
            gpui::AppEffects::new(
                "test",
                r#"{"schema": 10, "effects": [
                    {"name": "swap", "backdrop": false, "wgsl": "fx.wgsl"},
                    {"name": "count", "backdrop": false, "history": true, "wgsl": "fx.wgsl"}
                ]}"#,
                [("fx.wgsl".to_string(), EFFECTS_WGSL.to_string())],
            )
            .map_err(anyhow::Error::msg)?,
        );
        Ok(())
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

    fn effect(effect_id: u32, rect: Bounds<ScaledPixels>, opacity: f32, flags: u32) -> EffectQuad {
        EffectQuad {
            effect_id,
            bounds: rect,
            content_mask: mask(),
            opacity,
            scale: 1.,
            flags,
            ..Default::default()
        }
    }

    /// A black page, then an element whose painting is a green box and a
    /// white corner outside it, drawn by `draws`.
    fn element_scene(draws: Vec<VnEffectDraw>) -> Scene {
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x000000)));
        scene.vn_push_effect(VnEffectPaint {
            draws,
            content: true,
            content_mask: mask(),
        });
        scene.insert_primitive(quad(bounds(16., 16., 32., 32.), rgb(0x00ff00)));
        scene.insert_primitive(quad(bounds(0., 0., 8., 8.), rgb(0xffffff)));
        scene.vn_pop_isolation();
        scene.finish();
        scene
    }

    fn history_scene(rect: Bounds<ScaledPixels>, time: f32) -> Scene {
        let mut scene = Scene::default();
        scene.insert_primitive(quad(bounds(0., 0., SIDE, SIDE), rgb(0x000000)));
        scene.vn_push_effect(VnEffectPaint {
            draws: vec![VnEffectDraw {
                quad: EffectQuad {
                    time,
                    ..effect(COUNT, rect, 1., EFFECT_FLAG_HISTORY)
                },
                history: 7,
                textures: None,
            }],
            content: false,
            content_mask: mask(),
        });
        scene.vn_pop_isolation();
        scene.finish();
        scene
    }

    fn render(renderer: &mut WgpuHeadlessRenderer, scene: &Scene) -> Result<image::RgbaImage> {
        renderer.render_scene_to_image(scene, size(DevicePixels(64), DevicePixels(64)))
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

    #[test]
    fn a_content_effect_draws_its_element_in_its_place_and_cross_fades_with_it() -> Result<()> {
        let _app_effects = APP_EFFECTS.lock();
        install_effects()?;
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let swap = |opacity| VnEffectDraw {
            quad: effect(
                SWAP,
                bounds(16., 16., 32., 32.),
                opacity,
                EFFECT_FLAG_NEEDS_CONTENT,
            ),
            history: 0,
            textures: None,
        };

        let image = render(&mut renderer, &element_scene(vec![swap(1.)]))?;
        assert_eq!(pixel(&image, 32, 32), [0, 0, 255, 255], "the shaded box");
        assert_eq!(
            pixel(&image, 4, 4),
            [0, 0, 0, 255],
            "content outside the quad does not show"
        );

        let plain = VnEffectDraw {
            quad: effect(
                effect_id::CONTENT,
                bounds(0., 0., SIDE, SIDE),
                0.5,
                EFFECT_FLAG_NEEDS_CONTENT,
            ),
            history: 0,
            textures: None,
        };
        let image = render(&mut renderer, &element_scene(vec![plain, swap(0.5)]))?;
        assert_close(
            pixel(&image, 32, 32),
            [0, 64, 128, 255],
            "the shaded half over the plain half",
        );
        assert_close(
            pixel(&image, 4, 4),
            [128, 128, 128, 255],
            "the plain half shows all of the content",
        );
        Ok(())
    }

    #[test]
    fn a_history_steps_at_most_at_the_step_rate_and_starts_over() -> Result<()> {
        let _app_effects = APP_EFFECTS.lock();
        install_effects()?;
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let rect = bounds(16., 16., 32., 32.);
        let shows = |renderer: &mut WgpuHeadlessRenderer, scene: &Scene, red, what| {
            let image = render(renderer, scene)?;
            assert_eq!(pixel(&image, 8, 8), [0, 0, 0, 255], "outside the quad");
            assert_close(pixel(&image, 32, 32), [red, 0, 0, 255], what);
            anyhow::Ok(())
        };

        shows(&mut renderer, &history_scene(rect, 0.), 0, "starts at zero")?;
        shows(&mut renderer, &history_scene(rect, 1.), 64, "one step")?;
        shows(&mut renderer, &history_scene(rect, 1.001), 128, "two")?;
        shows(
            &mut renderer,
            &history_scene(rect, 1.002),
            128,
            "too soon for a third",
        )?;
        let histories = |renderer: &WgpuHeadlessRenderer| -> Result<usize> {
            Ok(renderer
                .core
                .resources
                .backdrop
                .ready()
                .context("effect pipelines missing")?
                .histories
                .len())
        };
        assert_eq!(histories(&renderer)?, 1);

        let resized = bounds(16., 16., 30., 32.);
        shows(
            &mut renderer,
            &history_scene(resized, 2.),
            0,
            "a new size starts over",
        )?;
        render(&mut renderer, &Scene::default())?;
        assert_eq!(histories(&renderer)?, 0, "a frame without it frees it");
        Ok(())
    }

    #[test]
    fn a_content_effect_samples_its_own_images() -> Result<()> {
        use gpui::{EffectTexture, EffectTextureSet, RenderImage};
        use std::sync::Arc;

        let _app_effects = APP_EFFECTS.lock();
        install_effects()?;
        let mut renderer = WgpuHeadlessRenderer::new()?;
        let image = |bgra: [u8; 4]| {
            let buffer = image::RgbaImage::from_raw(1, 1, bgra.to_vec()).unwrap();
            let mut set = EffectTextureSet::default();
            set[0] = Some(EffectTexture {
                image: Arc::new(RenderImage::new(vec![image::Frame::new(buffer)])),
                linear: false,
            });
            Some(Arc::new(set))
        };
        let stamp = |rect, textures| VnEffectDraw {
            quad: EffectQuad {
                params: EffectQuad::params_from(&[1.]),
                ..effect(SWAP, rect, 1., EFFECT_FLAG_NEEDS_CONTENT)
            },
            history: 0,
            textures,
        };

        let scene = element_scene(vec![
            stamp(bounds(0., 0., 32., SIDE), image([0, 255, 255, 255])),
            stamp(bounds(32., 0., 32., SIDE), image([255, 0, 255, 255])),
        ]);
        let image = render(&mut renderer, &scene)?;
        assert_eq!(
            pixel(&image, 24, 32),
            [255, 255, 0, 255],
            "the first draw's"
        );
        assert_eq!(
            pixel(&image, 40, 32),
            [255, 0, 255, 255],
            "the second draw's"
        );

        let image = render(
            &mut renderer,
            &element_scene(vec![stamp(mask().bounds, None)]),
        )?;
        assert_eq!(pixel(&image, 32, 32), [0, 0, 0, 255], "none bound");
        Ok(())
    }
}
