// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, Hsla, Pixels,
    Point, Radians, ScaledPixels, Size, bounds_tree::BoundsTree, point, px,
};
use std::{
    fmt::Debug,
    iter::Peekable,
    ops::{Add, Range, Sub},
    slice,
};

#[allow(non_camel_case_types, unused)]
#[expect(missing_docs)]
pub type PathVertex_ScaledPixels = PathVertex<ScaledPixels>;

#[expect(missing_docs)]
pub type DrawOrder = u32;

/// A boolean stored as a `u32` so that GPU-facing structs contain no
/// compiler-inserted padding bytes, which would be undefined behavior to
/// reinterpret as `&[u8]` when writing instance buffers. Guaranteed to be
/// `0` or `1` by construction; shaders read it as a `u32`/`uint`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PaddedBool32(u32);

impl From<bool> for PaddedBool32 {
    fn from(value: bool) -> Self {
        PaddedBool32(value as u32)
    }
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<Layer>,
    /// The lowest draw order any *newly opened layer* may take, raised whenever
    /// a backdrop blur rect is inserted. See [`Scene::draw_order_for_primitive`].
    backdrop_blur_floor: DrawOrder,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub backdrop_blur_rects: Vec<BackdropBlurRect>,
    pub effect_quads: Vec<EffectQuad>,
    pub paths: Vec<Path<ScaledPixels>>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
}

/// An entry on the scene's layer stack. The bounds are retained so that a
/// backdrop blur can re-insert the layer at a raised draw order.
struct Layer {
    order: DrawOrder,
    bounds: Bounds<ScaledPixels>,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.backdrop_blur_floor = 0;
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.backdrop_blur_rects.clear();
        self.effect_quads.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self, bounds: Bounds<ScaledPixels>) {
        // A layer opened after a backdrop blur must never sink below it, even
        // when the layer's bounds happen not to intersect the blur rect (its
        // *children* still might). See `draw_order_for_primitive`.
        let mut order = self.primitive_bounds.insert(bounds);
        if order < self.backdrop_blur_floor {
            order = self.backdrop_blur_floor;
            self.primitive_bounds.insert_with_order(bounds, order);
        }
        self.layer_stack.push(Layer { order, bounds });
        self.paint_operations
            .push(PaintOperation::StartLayer(bounds));
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        let mut primitive = primitive.into();
        let clipped_bounds = primitive
            .bounds()
            .intersect(&primitive.content_mask().bounds);

        if clipped_bounds.is_empty() {
            return;
        }

        let order = self.draw_order_for_primitive(&primitive, clipped_bounds);
        match &mut primitive {
            Primitive::Shadow(shadow) => {
                shadow.order = order;
                self.shadows.push(*shadow);
            }
            Primitive::Quad(quad) => {
                quad.order = order;
                self.quads.push(*quad);
            }
            Primitive::BackdropBlurRect(backdrop_blur_rect) => {
                backdrop_blur_rect.order = order;
                self.backdrop_blur_rects.push(*backdrop_blur_rect);
            }
            Primitive::EffectQuad(effect_quad) => {
                effect_quad.order = order;
                self.effect_quads.push(*effect_quad);
            }
            Primitive::Path(path) => {
                path.order = order;
                path.id = PathId(self.paths.len());
                self.paths.push(path.clone());
            }
            Primitive::Underline(underline) => {
                underline.order = order;
                self.underlines.push(*underline);
            }
            Primitive::MonochromeSprite(sprite) => {
                sprite.order = order;
                self.monochrome_sprites.push(*sprite);
            }
            Primitive::SubpixelSprite(sprite) => {
                sprite.order = order;
                self.subpixel_sprites.push(*sprite);
            }
            Primitive::PolychromeSprite(sprite) => {
                sprite.order = order;
                self.polychrome_sprites.push(*sprite);
            }
            Primitive::Surface(surface) => {
                surface.order = order;
                self.surfaces.push(surface.clone());
            }
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => self.insert_primitive(primitive.clone()),
                PaintOperation::StartLayer(bounds) => self.push_layer(*bounds),
                PaintOperation::EndLayer => self.pop_layer(),
            }
        }
    }

    pub fn finish(&mut self) {
        self.shadows.sort_by_key(|shadow| shadow.order);
        self.quads.sort_by_key(|quad| quad.order);
        self.backdrop_blur_rects
            .sort_by_key(|backdrop_blur_rect| backdrop_blur_rect.order);
        // Secondary key `effect_id`: the batch iterator breaks a run when the
        // id changes (each id is a different pipeline), so sorting by it keeps
        // same-effect quads at the same draw order in ONE instanced draw.
        self.effect_quads
            .sort_by_key(|effect_quad| (effect_quad.order, effect_quad.effect_id));
        self.paths.sort_by_key(|path| path.order);
        self.underlines.sort_by_key(|underline| underline.order);
        self.monochrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.subpixel_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.polychrome_sprites
            .sort_by_key(|sprite| (sprite.order, sprite.tile.tile_id));
        self.surfaces.sort_by_key(|surface| surface.order);
    }

    #[cfg_attr(
        all(
            any(target_os = "linux", target_os = "freebsd"),
            not(any(feature = "x11", feature = "wayland"))
        ),
        allow(dead_code)
    )]
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        BatchIterator {
            shadows_start: 0,
            shadows_iter: self.shadows.iter().peekable(),
            quads_start: 0,
            quads_iter: self.quads.iter().peekable(),
            backdrop_blur_rects_start: 0,
            backdrop_blur_rects_iter: self.backdrop_blur_rects.iter().peekable(),
            effect_quads_start: 0,
            effect_quads_iter: self.effect_quads.iter().peekable(),
            paths_start: 0,
            paths_iter: self.paths.iter().peekable(),
            underlines_start: 0,
            underlines_iter: self.underlines.iter().peekable(),
            monochrome_sprites_start: 0,
            monochrome_sprites_iter: self.monochrome_sprites.iter().peekable(),
            subpixel_sprites_start: 0,
            subpixel_sprites_iter: self.subpixel_sprites.iter().peekable(),
            polychrome_sprites_start: 0,
            polychrome_sprites_iter: self.polychrome_sprites.iter().peekable(),
            surfaces_start: 0,
            surfaces_iter: self.surfaces.iter().peekable(),
        }
    }

    /// Picks the draw order for a primitive about to be inserted.
    ///
    /// Everything except a backdrop blur takes the enclosing layer's order, or
    /// — with no layer open — an order derived from what it overlaps, which is
    /// exactly gpui's long-standing behaviour.
    ///
    /// A backdrop blur is different: the renderer snapshots the render target
    /// at the moment the blur draws, so *everything painted before it must draw
    /// before it, and everything painted after it must draw after it*, whether
    /// or not the bounds happen to intersect.
    ///
    /// * With no layer open, the [`BoundsTree`] already guarantees that for
    ///   anything that overlaps the blur (which is the only case that can be
    ///   seen), so the ordinary insert is correct. We additionally raise
    ///   `backdrop_blur_floor` so a *layer* opened later — whose children may
    ///   overlap even when the layer's own bounds do not — cannot sink below.
    /// * With layers open, every layer on the stack is re-inserted above the
    ///   blur. Upstream PR #59026 only bumped the innermost one, which meant a
    ///   blur painted inside a nested layer stopped forcing later siblings of
    ///   the *outer* layer above it — they would draw under the glass.
    ///
    /// An [`EffectQuad`] that reads the backdrop takes the identical treatment
    /// for the identical reason — it too is composited from a render-target
    /// snapshot. An effect that does *not* read the backdrop is an ordinary
    /// blended quad and takes the ordinary path, which is the whole point of
    /// [`EffectQuad::needs_backdrop`].
    fn draw_order_for_primitive(
        &mut self,
        primitive: &Primitive,
        clipped_bounds: Bounds<ScaledPixels>,
    ) -> DrawOrder {
        let snapshots_backdrop = match primitive {
            Primitive::BackdropBlurRect(_) => true,
            Primitive::EffectQuad(effect_quad) => effect_quad.needs_backdrop(),
            _ => false,
        };
        if !snapshots_backdrop {
            return self
                .layer_stack
                .last()
                .map(|layer| layer.order)
                .unwrap_or_else(|| self.primitive_bounds.insert(clipped_bounds));
        }

        let blur_order = match self.layer_stack.last() {
            Some(layer) => {
                let blur_order = layer.order.saturating_add(1);
                self.primitive_bounds
                    .insert_with_order(clipped_bounds, blur_order);
                blur_order
            }
            None => self.primitive_bounds.insert(clipped_bounds),
        };

        // Force everything painted after the blur to draw after it.
        let next_order = blur_order.saturating_add(1);
        for index in 0..self.layer_stack.len() {
            if self.layer_stack[index].order >= next_order {
                continue;
            }
            let bounds = self.layer_stack[index].bounds;
            self.primitive_bounds.insert_with_order(bounds, next_order);
            self.layer_stack[index].order = next_order;
        }
        self.backdrop_blur_floor = self.backdrop_blur_floor.max(next_order);

        blur_order
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Default)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
pub(crate) enum PrimitiveKind {
    Shadow,
    #[default]
    Quad,
    BackdropBlurRect,
    EffectQuad,
    Path,
    Underline,
    MonochromeSprite,
    SubpixelSprite,
    PolychromeSprite,
    Surface,
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer(Bounds<ScaledPixels>),
    EndLayer,
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    BackdropBlurRect(BackdropBlurRect),
    EffectQuad(EffectQuad),
    Path(Path<ScaledPixels>),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
}

#[expect(missing_docs)]
impl Primitive {
    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::BackdropBlurRect(backdrop_blur_rect) => &backdrop_blur_rect.bounds,
            Primitive::EffectQuad(effect_quad) => &effect_quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::BackdropBlurRect(backdrop_blur_rect) => &backdrop_blur_rect.content_mask,
            Primitive::EffectQuad(effect_quad) => &effect_quad.content_mask,
            Primitive::Path(path) => &path.content_mask,
            Primitive::Underline(underline) => &underline.content_mask,
            Primitive::MonochromeSprite(sprite) => &sprite.content_mask,
            Primitive::SubpixelSprite(sprite) => &sprite.content_mask,
            Primitive::PolychromeSprite(sprite) => &sprite.content_mask,
            Primitive::Surface(surface) => &surface.content_mask,
        }
    }
}

#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
struct BatchIterator<'a> {
    shadows_start: usize,
    shadows_iter: Peekable<slice::Iter<'a, Shadow>>,
    quads_start: usize,
    quads_iter: Peekable<slice::Iter<'a, Quad>>,
    backdrop_blur_rects_start: usize,
    backdrop_blur_rects_iter: Peekable<slice::Iter<'a, BackdropBlurRect>>,
    effect_quads_start: usize,
    effect_quads_iter: Peekable<slice::Iter<'a, EffectQuad>>,
    paths_start: usize,
    paths_iter: Peekable<slice::Iter<'a, Path<ScaledPixels>>>,
    underlines_start: usize,
    underlines_iter: Peekable<slice::Iter<'a, Underline>>,
    monochrome_sprites_start: usize,
    monochrome_sprites_iter: Peekable<slice::Iter<'a, MonochromeSprite>>,
    subpixel_sprites_start: usize,
    subpixel_sprites_iter: Peekable<slice::Iter<'a, SubpixelSprite>>,
    polychrome_sprites_start: usize,
    polychrome_sprites_iter: Peekable<slice::Iter<'a, PolychromeSprite>>,
    surfaces_start: usize,
    surfaces_iter: Peekable<slice::Iter<'a, PaintSurface>>,
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = PrimitiveBatch;

    fn next(&mut self) -> Option<Self::Item> {
        let mut orders_and_kinds = [
            (
                self.shadows_iter.peek().map(|s| s.order),
                PrimitiveKind::Shadow,
            ),
            (self.quads_iter.peek().map(|q| q.order), PrimitiveKind::Quad),
            (
                self.backdrop_blur_rects_iter.peek().map(|b| b.order),
                PrimitiveKind::BackdropBlurRect,
            ),
            (
                self.effect_quads_iter.peek().map(|e| e.order),
                PrimitiveKind::EffectQuad,
            ),
            (self.paths_iter.peek().map(|q| q.order), PrimitiveKind::Path),
            (
                self.underlines_iter.peek().map(|u| u.order),
                PrimitiveKind::Underline,
            ),
            (
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::MonochromeSprite,
            ),
            (
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::SubpixelSprite,
            ),
            (
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                PrimitiveKind::PolychromeSprite,
            ),
            (
                self.surfaces_iter.peek().map(|s| s.order),
                PrimitiveKind::Surface,
            ),
        ];
        orders_and_kinds.sort_by_key(|(order, kind)| (order.unwrap_or(u32::MAX), *kind));

        let first = orders_and_kinds[0];
        let second = orders_and_kinds[1];
        let (batch_kind, max_order_and_kind) = if first.0.is_some() {
            (first.1, (second.0.unwrap_or(u32::MAX), second.1))
        } else {
            return None;
        };

        match batch_kind {
            PrimitiveKind::Shadow => {
                let shadows_start = self.shadows_start;
                let mut shadows_end = shadows_start + 1;
                self.shadows_iter.next();
                while self
                    .shadows_iter
                    .next_if(|shadow| (shadow.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    shadows_end += 1;
                }
                self.shadows_start = shadows_end;
                Some(PrimitiveBatch::Shadows(shadows_start..shadows_end))
            }
            PrimitiveKind::Quad => {
                let quads_start = self.quads_start;
                let mut quads_end = quads_start + 1;
                self.quads_iter.next();
                while self
                    .quads_iter
                    .next_if(|quad| (quad.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    quads_end += 1;
                }
                self.quads_start = quads_end;
                Some(PrimitiveBatch::Quads(quads_start..quads_end))
            }
            PrimitiveKind::BackdropBlurRect => {
                let backdrop_blur_rects_start = self.backdrop_blur_rects_start;
                let mut backdrop_blur_rects_end = backdrop_blur_rects_start + 1;
                self.backdrop_blur_rects_iter.next();
                while self
                    .backdrop_blur_rects_iter
                    .next_if(|backdrop_blur_rect| {
                        (backdrop_blur_rect.order, batch_kind) < max_order_and_kind
                    })
                    .is_some()
                {
                    backdrop_blur_rects_end += 1;
                }
                self.backdrop_blur_rects_start = backdrop_blur_rects_end;
                Some(PrimitiveBatch::BackdropBlurRects(
                    backdrop_blur_rects_start..backdrop_blur_rects_end,
                ))
            }
            PrimitiveKind::EffectQuad => {
                // Each `effect_id` is a different fragment pipeline, so a run
                // breaks when it changes — the same shape as the sprite batches
                // breaking on `texture_id`.
                let effect_id = self.effect_quads_iter.peek().unwrap().effect_id;
                let effect_quads_start = self.effect_quads_start;
                let mut effect_quads_end = effect_quads_start + 1;
                self.effect_quads_iter.next();
                while self
                    .effect_quads_iter
                    .next_if(|effect_quad| {
                        (effect_quad.order, batch_kind) < max_order_and_kind
                            && effect_quad.effect_id == effect_id
                    })
                    .is_some()
                {
                    effect_quads_end += 1;
                }
                self.effect_quads_start = effect_quads_end;
                Some(PrimitiveBatch::EffectQuads {
                    effect_id,
                    range: effect_quads_start..effect_quads_end,
                })
            }
            PrimitiveKind::Path => {
                let paths_start = self.paths_start;
                let mut paths_end = paths_start + 1;
                self.paths_iter.next();
                while self
                    .paths_iter
                    .next_if(|path| (path.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    paths_end += 1;
                }
                self.paths_start = paths_end;
                Some(PrimitiveBatch::Paths(paths_start..paths_end))
            }
            PrimitiveKind::Underline => {
                let underlines_start = self.underlines_start;
                let mut underlines_end = underlines_start + 1;
                self.underlines_iter.next();
                while self
                    .underlines_iter
                    .next_if(|underline| (underline.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    underlines_end += 1;
                }
                self.underlines_start = underlines_end;
                Some(PrimitiveBatch::Underlines(underlines_start..underlines_end))
            }
            PrimitiveKind::MonochromeSprite => {
                let texture_id = self.monochrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.monochrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.monochrome_sprites_iter.next();
                while self
                    .monochrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.monochrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::MonochromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::SubpixelSprite => {
                let texture_id = self.subpixel_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.subpixel_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.subpixel_sprites_iter.next();
                while self
                    .subpixel_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.subpixel_sprites_start = sprites_end;
                Some(PrimitiveBatch::SubpixelSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::PolychromeSprite => {
                let texture_id = self.polychrome_sprites_iter.peek().unwrap().tile.texture_id;
                let sprites_start = self.polychrome_sprites_start;
                let mut sprites_end = sprites_start + 1;
                self.polychrome_sprites_iter.next();
                while self
                    .polychrome_sprites_iter
                    .next_if(|sprite| {
                        (sprite.order, batch_kind) < max_order_and_kind
                            && sprite.tile.texture_id == texture_id
                    })
                    .is_some()
                {
                    sprites_end += 1;
                }
                self.polychrome_sprites_start = sprites_end;
                Some(PrimitiveBatch::PolychromeSprites {
                    texture_id,
                    range: sprites_start..sprites_end,
                })
            }
            PrimitiveKind::Surface => {
                let surfaces_start = self.surfaces_start;
                let mut surfaces_end = surfaces_start + 1;
                self.surfaces_iter.next();
                while self
                    .surfaces_iter
                    .next_if(|surface| (surface.order, batch_kind) < max_order_and_kind)
                    .is_some()
                {
                    surfaces_end += 1;
                }
                self.surfaces_start = surfaces_end;
                Some(PrimitiveBatch::Surfaces(surfaces_start..surfaces_end))
            }
        }
    }
}

#[derive(Debug)]
#[cfg_attr(
    all(
        any(target_os = "linux", target_os = "freebsd"),
        not(any(feature = "x11", feature = "wayland"))
    ),
    allow(dead_code)
)]
#[allow(missing_docs)]
pub enum PrimitiveBatch {
    Shadows(Range<usize>),
    Quads(Range<usize>),
    BackdropBlurRects(Range<usize>),
    EffectQuads {
        effect_id: u32,
        range: Range<usize>,
    },
    Paths(Range<usize>),
    Underlines(Range<usize>),
    MonochromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    #[cfg_attr(target_os = "macos", allow(dead_code))]
    SubpixelSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    PolychromeSprites {
        texture_id: AtlasTextureId,
        range: Range<usize>,
    },
    Surfaces(Range<usize>),
}

impl PrimitiveBatch {
    #[expect(missing_docs)]
    pub fn label(&self) -> String {
        match self {
            Self::Shadows(range) => format!("shadows ({})", range.len()),
            Self::Quads(range) => format!("quads ({})", range.len()),
            Self::BackdropBlurRects(range) => {
                format!("backdrop blur rects ({})", range.len())
            }
            Self::EffectQuads { effect_id, range } => {
                format!("effect quads ({}) with effect {}", range.len(), effect_id)
            }
            Self::Paths(range) => format!("paths ({})", range.len()),
            Self::Underlines(range) => format!("underlines ({})", range.len()),
            Self::MonochromeSprites { texture_id, range } => {
                format!(
                    "monochrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::SubpixelSprites { texture_id, range } => {
                format!(
                    "subpixel sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::PolychromeSprites { texture_id, range } => {
                format!(
                    "polychrome sprites ({}) on atlas {}",
                    range.len(),
                    texture_id.index
                )
            }
            Self::Surfaces(range) => format!("surfaces ({})", range.len()),
        }
    }
}

/// Keep the field order in sync with `struct Quad` in
/// `gpui_windows/src/shaders.hlsl` (46 words / 184 bytes) — the DirectX
/// renderer uploads this straight into a `StructuredBuffer` and
/// `StructureByteStride` is `size_of::<Quad>()`, so the Rust layout IS the
/// shader layout.
#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Quad {
    pub order: DrawOrder,
    pub border_style: BorderStyle,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub background: Background,
    pub border_color: Hsla,
    pub corner_radii: Corners<ScaledPixels>,
    pub border_widths: Edges<ScaledPixels>,
    /// A paint-time transformation of the quad about its own centre — CSS
    /// `transform`.
    ///
    /// `bounds` stays the UNTRANSFORMED rect and is still what the scene sorts
    /// and culls by; the vertex shader applies this on top and the fragment
    /// shader keeps doing its SDF work in the untransformed local frame. That
    /// split is deliberate: it is what lets the rounded corners, the border
    /// widths and the dash phase stay correct under rotation without any of
    /// them learning about the matrix.
    pub transformation: TransformationMatrix,
}

impl From<Quad> for Primitive {
    fn from(quad: Quad) -> Self {
        Primitive::Quad(quad)
    }
}

/// Maximum number of GPU blur downsample/upsample levels a backdrop blur rect
/// may request.
pub const MAX_BACKDROP_BLUR_KERNEL_LEVELS: u32 = 5;

/// Scaled pixels of CSS `blur()` radius covered by one Dual-Kawase pyramid
/// level.
const BACKDROP_BLUR_RADIUS_PER_KERNEL_LEVEL: f32 = 5.;

/// A rounded rect that replaces the pixels already painted behind it with a
/// blurred copy of them — the primitive behind CSS `backdrop-filter: blur()`.
///
/// Keep the field order in sync with `struct BlurRect` in
/// `gpui_windows/src/shaders.hlsl` (20 words / 80 bytes).
#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct BackdropBlurRect {
    pub order: DrawOrder,
    pub pad: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub blur_radius: ScaledPixels,
    pub opacity: f32,
    pub tint: Hsla,
}

impl Default for BackdropBlurRect {
    fn default() -> Self {
        Self {
            order: Default::default(),
            pad: Default::default(),
            bounds: Default::default(),
            content_mask: Default::default(),
            corner_radii: Default::default(),
            blur_radius: Default::default(),
            opacity: 1.,
            tint: Default::default(),
        }
    }
}

impl BackdropBlurRect {
    /// Returns the clamped number of blur kernel levels required by this rect.
    ///
    /// # Known divergence from CSS `blur()`
    ///
    /// The radius is quantised to whole Dual-Kawase pyramid levels and then
    /// clamped to [`MAX_BACKDROP_BLUR_KERNEL_LEVELS`], so the mapping from CSS
    /// radius to visual blur **saturates**. It bites when
    ///
    /// ```text
    /// radius_css > MAX_BACKDROP_BLUR_KERNEL_LEVELS * BACKDROP_BLUR_RADIUS_PER_KERNEL_LEVEL / scale_factor
    ///            = 25px @ 100% DPI, 20px @ 125%, 16.7px @ 150%, 12.5px @ 200%
    /// ```
    ///
    /// Past that point `blur(30px)` and `blur(100px)` render identically, and
    /// below it the radius steps in 5-scaled-pixel increments rather than
    /// continuously. Browsers are continuous and unbounded.
    ///
    /// Fixing it properly means scaling the upsample tap offsets by the
    /// fractional remainder, which is a per-*group* uniform (the pyramid is
    /// built once per non-overlapping group) rather than a per-rect one — so it
    /// needs the group split to also break on fractional radius. Deferred: the
    /// UI range that matters (8–20px glass) is inside the linear part.
    pub fn effective_kernel_levels(&self) -> u32 {
        if self.blur_radius.0 <= 0. {
            0
        } else {
            let radius_levels = (self.blur_radius.0 / BACKDROP_BLUR_RADIUS_PER_KERNEL_LEVEL)
                .ceil()
                .max(1.) as u32;
            radius_levels.min(MAX_BACKDROP_BLUR_KERNEL_LEVELS)
        }
    }
}

impl From<BackdropBlurRect> for Primitive {
    fn from(backdrop_blur_rect: BackdropBlurRect) -> Self {
        Primitive::BackdropBlurRect(backdrop_blur_rect)
    }
}

/// The built-in effect a [`EffectQuad`] selects, and the index of its pipeline
/// in the renderer's effect table.
///
/// gpui deliberately does not know what these effects *do* — the shader
/// sources, the CSS parameter schemas and the defaults all live in
/// `crates/vn-effects`, which is above gpui in the dependency graph. What lives
/// here is only the numbering, because three separate places have to agree on
/// it: this enum, the pipeline table in `gpui_windows`, and the registry in
/// `vn-effects`.
pub mod effect_id {
    /// `frost(strength, radius?, tint?)` — blurred backdrop + crystalline
    /// grain + rim highlight. Reads the backdrop.
    pub const FROST: u32 = 0;
    /// `noise(amount, scale?, speed?)` — film grain over the element.
    pub const NOISE: u32 = 1;
    /// `glow(color?, radius, strength)` — soft halo around the border box.
    pub const GLOW: u32 = 2;
    /// How many built-in effects exist.
    pub const COUNT: u32 = 3;
    /// The id an [`super::EffectSpec`] carries when CSS said `--shading: none`.
    /// A refinement has no way to spell "explicitly nothing", so — exactly as
    /// `backdrop-filter: none` maps to a 0px radius — clearing is a *value*.
    pub const NONE: u32 = u32::MAX;
}

/// Bit 0 of [`EffectQuad::flags`]: this effect samples the backdrop, so the
/// renderer must snapshot the render target, run the blur pyramid, bind t0/t2
/// and draw with blending disabled.
pub const EFFECT_FLAG_NEEDS_BACKDROP: u32 = 1;

/// CSS `transform`, as the cascade computed it, carried on [`crate::Style`]
/// and turned into a [`TransformationMatrix`] at paint time by
/// [`crate::Style::paint`].
///
/// Deliberately the CSS *components* rather than a finished matrix, for two
/// reasons. The matrix needs the element's box to know where its centre is —
/// CSS's initial `transform-origin: 50% 50%` — and layout only settles at
/// paint. And an interpolable `transform` interpolates its components (a
/// browser lerps `rotate(0deg)` → `rotate(90deg)` through 45°, not through the
/// matrix entries), so a transition wants these numbers, not their product.
///
/// The rendering rule is CSS's: a transform moves the element's **painted
/// output and its whole subtree**, and changes no layout at all. `translate`
/// here is therefore *not* the same thing as a margin offset — siblings do not
/// move out of the way.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TransformSpec {
    /// `translate()`, per axis, in the unit it was authored in — see
    /// [`TransformLength`]. Kept unresolved because a percentage resolves
    /// against the element's OWN border box, which is only known at paint.
    pub translate: (TransformLength, TransformLength),
    /// `rotate()` about the element's centre, in RADIANS, clockwise — which is
    /// the direction a positive CSS angle turns on screen.
    pub rotate: f32,
    /// `scale()` about the element's centre. `(1.0, 1.0)` is no scale.
    pub scale: (f32, f32),
    /// `transform-origin`, per axis, relative to the element's own border box —
    /// the point `rotate()` and `scale()` act about. CSS's initial value is
    /// `50% 50%`, i.e. `(Fraction(0.5), Fraction(0.5))`, which is what
    /// [`TransformSpec::IDENTITY`] carries.
    ///
    /// Kept in the same authored unit as `translate` and for the same reason: a
    /// percentage origin resolves against the element's own box, which is only
    /// known at paint. Unlike `translate` it is measured from the box's
    /// TOP-LEFT, not from its centre, because that is how CSS writes it.
    pub origin: (TransformLength, TransformLength),
}

/// One axis of a [`TransformSpec`]'s `translate()`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum TransformLength {
    /// An absolute length in LOGICAL (CSS) pixels.
    Pixels(f32),
    /// A fraction of the element's own border box on this axis: `-50%` is
    /// `-0.5`. This is the unit `translate(-50%, -50%)` centring idiom is
    /// written in, and it cannot be resolved before layout.
    Fraction(f32),
}

impl TransformLength {
    /// Zero, in either unit — the identity translate.
    pub const ZERO: Self = Self::Pixels(0.0);

    /// Resolves against the element's own size along this axis.
    pub fn resolve(self, own: f32) -> f32 {
        match self {
            Self::Pixels(v) => v,
            Self::Fraction(f) => f * own,
        }
    }

    /// Is this zero, whatever unit it is written in?
    pub fn is_zero(self) -> bool {
        match self {
            Self::Pixels(v) | Self::Fraction(v) => v == 0.0,
        }
    }

    /// Interpolates between two authored lengths.
    ///
    /// Same units interpolate. Mixed units cannot without knowing the box, and
    /// CSS would produce a `calc()` there — except when one side is zero, which
    /// is the same length in both units and so adopts the other's. Anything
    /// else snaps at the halfway point, which is the discrete behaviour the
    /// rest of this animator already uses for uninterpolable pairs.
    pub fn interpolate(self, other: Self, t: f32) -> Self {
        let lerp = |a: f32, b: f32| a + (b - a) * t;
        match (self, other) {
            (Self::Pixels(a), Self::Pixels(b)) => Self::Pixels(lerp(a, b)),
            (Self::Fraction(a), Self::Fraction(b)) => Self::Fraction(lerp(a, b)),
            (a, b) if a.is_zero() => match b {
                Self::Pixels(v) => Self::Pixels(lerp(0.0, v)),
                Self::Fraction(v) => Self::Fraction(lerp(0.0, v)),
            },
            (a, b) if b.is_zero() => match a {
                Self::Pixels(v) => Self::Pixels(lerp(v, 0.0)),
                Self::Fraction(v) => Self::Fraction(lerp(v, 0.0)),
            },
            (a, b) => {
                if t < 0.5 {
                    a
                } else {
                    b
                }
            }
        }
    }
}

impl Default for TransformSpec {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl TransformSpec {
    /// `transform: none`.
    pub const IDENTITY: Self = Self {
        translate: (TransformLength::ZERO, TransformLength::ZERO),
        rotate: 0.0,
        scale: (1.0, 1.0),
        origin: Self::CENTRE_ORIGIN,
    };

    /// CSS's initial `transform-origin: 50% 50%`.
    pub const CENTRE_ORIGIN: (TransformLength, TransformLength) =
        (TransformLength::Fraction(0.5), TransformLength::Fraction(0.5));

    /// Is this `transform: none` — nothing to apply, nothing to pay for?
    ///
    /// `transform-origin` is deliberately NOT part of the answer: an origin on
    /// its own changes nothing, exactly as in CSS, and asking otherwise would
    /// make every element that merely declares one pay for a matrix.
    pub fn is_identity(&self) -> bool {
        self.rotate == 0.0
            && self.scale == (1.0, 1.0)
            && self.translate.0.is_zero()
            && self.translate.1.is_zero()
    }

    /// Componentwise interpolation, which is what a browser does when both
    /// sides are the same function list: `rotate(0deg)` → `rotate(90deg)`
    /// sweeps through 45°, not through the matrix entries (which would shear).
    pub fn interpolate(self, other: Self, t: f32) -> Self {
        let lerp = |a: f32, b: f32| a + (b - a) * t;
        Self {
            translate: (
                self.translate.0.interpolate(other.translate.0, t),
                self.translate.1.interpolate(other.translate.1, t),
            ),
            rotate: lerp(self.rotate, other.rotate),
            scale: (
                lerp(self.scale.0, other.scale.0),
                lerp(self.scale.1, other.scale.1),
            ),
            origin: (
                self.origin.0.interpolate(other.origin.0, t),
                self.origin.1.interpolate(other.origin.1, t),
            ),
        }
    }

    /// The device-space matrix for an element whose border box is `bounds`.
    ///
    /// Read bottom-up, as matrix products are: move the centre to the origin,
    /// scale, rotate, then move back — plus the translate, which rides on the
    /// recentring so it is applied in the element's own frame.
    ///
    /// The point rotate/scale act about is `origin`, resolved against the box
    /// and measured from its top-left — CSS's `transform-origin`, whose initial
    /// `50% 50%` lands exactly on the centre.
    pub fn to_matrix(self, bounds: Bounds<Pixels>, scale_factor: f32) -> TransformationMatrix {
        let anchor = bounds.origin
            + Point::new(
                px(self.origin.0.resolve(bounds.size.width.0)),
                px(self.origin.1.resolve(bounds.size.height.0)),
            );
        let translate = Point::new(
            px(self.translate.0.resolve(bounds.size.width.0)),
            px(self.translate.1.resolve(bounds.size.height.0)),
        );
        TransformationMatrix::unit()
            .translate((anchor + translate).scale(scale_factor))
            .rotate(Radians(self.rotate))
            .scale(Size {
                width: self.scale.0,
                height: self.scale.1,
            })
            .translate(anchor.scale(-scale_factor))
    }
}

/// The `--shading` value that came out of the cascade, carried on
/// [`crate::Style`] and painted by [`crate::Style::paint`].
///
/// The eight `params` are the effect's positional CSS arguments, already
/// converted to floats (a colour occupies four consecutive slots). Everything
/// else is schema metadata the *painter* needs and gpui must not have to look
/// up: which params are lengths (so they scale with the device factor), which
/// one drives the backdrop blur, and which one dilates the quad.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct EffectSpec {
    /// One of [`effect_id`].
    pub id: u32,
    /// [`EFFECT_FLAG_NEEDS_BACKDROP`], …
    pub flags: u32,
    /// Bit *i* set means `params[i]` is a length in LOGICAL pixels, to be
    /// multiplied by the device scale factor at paint time.
    pub length_mask: u32,
    /// `Some(i)`: `params[i]` (a logical-pixel length) is the backdrop blur
    /// radius. Only meaningful with [`EFFECT_FLAG_NEEDS_BACKDROP`].
    pub backdrop_radius_param: Option<u8>,
    /// `Some(i)`: `params[i]` (a logical-pixel length) dilates the painted
    /// quad, giving an outer glow somewhere to land.
    pub bleed_param: Option<u8>,
    /// Whether this effect's output depends on `time`, i.e. whether the window
    /// must keep repainting for it to animate.
    pub animated: bool,
    /// The positional parameters, in schema order.
    pub params: [f32; 8],
}

impl EffectSpec {
    /// `--shading: none` — a value that clears an effect inherited from an
    /// earlier rule in the cascade.
    pub const NONE: EffectSpec = EffectSpec {
        id: effect_id::NONE,
        flags: 0,
        length_mask: 0,
        backdrop_radius_param: None,
        bleed_param: None,
        animated: false,
        params: [0.; 8],
    };

    /// Whether this spec actually paints anything.
    pub fn is_some(&self) -> bool {
        self.id != effect_id::NONE
    }

    fn param(&self, index: Option<u8>) -> f32 {
        index
            .and_then(|i| self.params.get(i as usize))
            .copied()
            .unwrap_or(0.)
    }

    /// The quad dilation in logical pixels.
    pub fn bleed(&self) -> f32 {
        self.param(self.bleed_param).max(0.)
    }

    /// The backdrop blur radius in logical pixels.
    pub fn backdrop_blur_radius(&self) -> f32 {
        self.param(self.backdrop_radius_param).max(0.)
    }
}

/// A rounded rect shaded by a custom fragment pipeline — the primitive behind
/// the `--shading` CSS property.
///
/// Mechanically this is [`BackdropBlurRect`] with a swappable fragment shader
/// and a parameter block, and it reuses that primitive's whole machinery: the
/// render-target snapshot, the Dual-Kawase pyramid, the per-group re-snapshot,
/// and the "everything painted after must draw after" ordering guarantee.
///
/// `bounds`/`corner_radii` describe the *drawn* quad, which is the element's
/// border box dilated by `bleed`. Dilating a rounded rect is a Minkowski sum
/// with a disc, so the element's own SDF is exactly the drawn quad's plus
/// `bleed` — which is how the shader recovers one from the other with a single
/// add instead of a second bounds pair.
///
/// Keep the field order in sync with `struct EffectQuad` in
/// `gpui_windows/src/effects.hlsl` and with `struct VnEffectQuad` in the
/// generated wrapper (`packages/vue-native/shaders.ts`).
/// 32 words / 128 bytes.
#[derive(Debug, Copy, Clone, Default)]
#[repr(C)]
#[expect(missing_docs)]
pub struct EffectQuad {
    pub order: DrawOrder,
    /// Selects the pipeline; one of [`effect_id`].
    pub effect_id: u32,
    /// The element's border box, dilated by `bleed`.
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    /// The element's corner radii, dilated by `bleed`.
    pub corner_radii: Corners<ScaledPixels>,
    /// Effect parameters 0..3.
    pub params0: [f32; 4],
    /// Effect parameters 4..7.
    pub params1: [f32; 4],
    pub opacity: f32,
    /// Seconds since the process started.
    pub time: f32,
    /// How far outside the element's border box this quad reaches.
    pub bleed: ScaledPixels,
    /// The blur radius applied to the backdrop before the effect sees it.
    /// Ignored without [`EFFECT_FLAG_NEEDS_BACKDROP`].
    pub backdrop_blur_radius: ScaledPixels,
    /// Device pixels per logical pixel.
    pub scale: f32,
    pub flags: u32,
    pub pad: [u32; 4],
}

impl EffectQuad {
    /// Whether this quad reads the render-target snapshot, which decides both
    /// its draw ordering and whether the renderer has to break the pass.
    pub fn needs_backdrop(&self) -> bool {
        self.flags & EFFECT_FLAG_NEEDS_BACKDROP != 0
    }

    /// The clamped number of Dual-Kawase pyramid levels this quad's backdrop
    /// needs — the same quantisation (and the same saturation defect) as
    /// [`BackdropBlurRect::effective_kernel_levels`].
    pub fn effective_kernel_levels(&self) -> u32 {
        if !self.needs_backdrop() || self.backdrop_blur_radius.0 <= 0. {
            0
        } else {
            let radius_levels = (self.backdrop_blur_radius.0
                / BACKDROP_BLUR_RADIUS_PER_KERNEL_LEVEL)
                .ceil()
                .max(1.) as u32;
            radius_levels.min(MAX_BACKDROP_BLUR_KERNEL_LEVELS)
        }
    }
}

impl From<EffectQuad> for Primitive {
    fn from(effect_quad: EffectQuad) -> Self {
        Primitive::EffectQuad(effect_quad)
    }
}

const _: () = assert!(std::mem::size_of::<EffectQuad>() == 128);

// The four primitives that carry a CSS `transform`. Their Rust layout IS the
// shader layout on every backend — the DirectX renderer uses
// `size_of::<T>()` as the structured buffer's `StructureByteStride`, and
// `gpui_wgpu/src/shaders_webgl.wgsl` hardcodes the same numbers as word
// strides. A field added without updating the shaders reads the next record's
// bytes and produces garbage rather than an error, so the sizes are asserted
// here where a mismatch is a compile failure.
//
// Word counts, for the shader side: 46 / 34 / 22 / 30.
const _: () = assert!(std::mem::size_of::<Quad>() == 46 * 4);
const _: () = assert!(std::mem::size_of::<Shadow>() == 34 * 4);
const _: () = assert!(std::mem::size_of::<Underline>() == 22 * 4);
const _: () = assert!(std::mem::size_of::<PolychromeSprite>() == 30 * 4);
const _: () = assert!(std::mem::size_of::<TransformationMatrix>() == 6 * 4);

/// Seconds since the first effect was painted — the clock [`EffectQuad::time`]
/// carries.
///
/// Deliberately process-wide and monotonic rather than per-window: an f32 has
/// ~7 significant digits, so anchoring at zero keeps sub-millisecond precision
/// for the first couple of hours instead of quantising to ~64 ms the way a
/// UNIX-epoch value would.
pub fn effect_clock_seconds() -> f32 {
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    EPOCH
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f32()
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Underline {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub thickness: ScaledPixels,
    pub wavy: PaddedBool32,
    /// See [`Quad::transformation`]. An underline rides along with the text it
    /// decorates, so it takes the same ambient transform the glyph sprites do.
    pub transformation: TransformationMatrix,
}

impl From<Underline> for Primitive {
    fn from(underline: Underline) -> Self {
        Primitive::Underline(underline)
    }
}

#[derive(Debug, Copy, Clone)]
#[repr(C)]
#[expect(missing_docs)]
pub struct Shadow {
    pub order: DrawOrder,
    pub blur_radius: ScaledPixels,
    pub bounds: Bounds<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub element_bounds: Bounds<ScaledPixels>,
    pub element_corner_radii: Corners<ScaledPixels>,
    /// 0 = drop shadow (rendered outside the element), 1 = inset shadow (rendered inside).
    pub inset: u32,
    pub pad: u32, // align to 8 bytes
    /// See [`Quad::transformation`]. A shadow is part of the element's own
    /// painted content, so a `transform: rotate()` turns the shadow with the
    /// box — which is what a browser does.
    pub transformation: TransformationMatrix,
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[repr(C)]
pub enum BorderStyle {
    /// A solid border.
    #[default]
    Solid = 0,
    /// A dashed border.
    Dashed = 1,
}

/// A data type representing a 2 dimensional transformation that can be applied to an element.
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct TransformationMatrix {
    /// 2x2 matrix containing rotation and scale,
    /// stored row-major
    pub rotation_scale: [[f32; 2]; 2],
    /// translation vector
    pub translation: [f32; 2],
}

impl Eq for TransformationMatrix {}

impl TransformationMatrix {
    /// The unit matrix, has no effect.
    pub fn unit() -> Self {
        Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [0.0, 0.0],
        }
    }

    /// Move the origin by a given point
    pub fn translate(mut self, point: Point<ScaledPixels>) -> Self {
        self.compose(Self {
            rotation_scale: [[1.0, 0.0], [0.0, 1.0]],
            translation: [point.x.0, point.y.0],
        })
    }

    /// Clockwise rotation in radians around the origin
    pub fn rotate(self, angle: Radians) -> Self {
        self.compose(Self {
            rotation_scale: [
                [angle.0.cos(), -angle.0.sin()],
                [angle.0.sin(), angle.0.cos()],
            ],
            translation: [0.0, 0.0],
        })
    }

    /// Scale around the origin
    pub fn scale(self, size: Size<f32>) -> Self {
        self.compose(Self {
            rotation_scale: [[size.width, 0.0], [0.0, size.height]],
            translation: [0.0, 0.0],
        })
    }

    /// Perform matrix multiplication with another transformation
    /// to produce a new transformation that is the result of
    /// applying both transformations: first, `other`, then `self`.
    #[inline]
    pub fn compose(self, other: TransformationMatrix) -> TransformationMatrix {
        if other == Self::unit() {
            return self;
        }
        // Perform matrix multiplication
        TransformationMatrix {
            rotation_scale: [
                [
                    self.rotation_scale[0][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][0],
                    self.rotation_scale[0][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[0][1] * other.rotation_scale[1][1],
                ],
                [
                    self.rotation_scale[1][0] * other.rotation_scale[0][0]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][0],
                    self.rotation_scale[1][0] * other.rotation_scale[0][1]
                        + self.rotation_scale[1][1] * other.rotation_scale[1][1],
                ],
            ],
            translation: [
                self.translation[0]
                    + self.rotation_scale[0][0] * other.translation[0]
                    + self.rotation_scale[0][1] * other.translation[1],
                self.translation[1]
                    + self.rotation_scale[1][0] * other.translation[0]
                    + self.rotation_scale[1][1] * other.translation[1],
            ],
        }
    }

    /// Apply transformation to a point, mainly useful for debugging
    pub fn apply(&self, point: Point<Pixels>) -> Point<Pixels> {
        let input = [point.x.0, point.y.0];
        let mut output = self.translation;
        for (i, output_cell) in output.iter_mut().enumerate() {
            for (k, input_cell) in input.iter().enumerate() {
                *output_cell += self.rotation_scale[i][k] * *input_cell;
            }
        }
        Point::new(output[0].into(), output[1].into())
    }

    /// The inverse transformation, or `None` when the matrix is singular
    /// (`scale(0)` collapses the element to nothing, and nothing has no
    /// interior to map a point back into).
    ///
    /// This is what hit-testing needs: a hitbox stores its untransformed
    /// `bounds`, so testing a pointer against a *transformed* element means
    /// mapping the pointer back into the element's own frame rather than
    /// mapping four corners forward and testing a polygon.
    pub fn inverse(&self) -> Option<TransformationMatrix> {
        let [[a, b], [c, d]] = self.rotation_scale;
        let det = a * d - b * c;
        if det.abs() < f32::EPSILON {
            return None;
        }
        let inv = [[d / det, -b / det], [-c / det, a / det]];
        let [tx, ty] = self.translation;
        Some(TransformationMatrix {
            rotation_scale: inv,
            translation: [
                -(inv[0][0] * tx + inv[0][1] * ty),
                -(inv[1][0] * tx + inv[1][1] * ty),
            ],
        })
    }

    /// Is this the identity — i.e. is there nothing to apply?
    ///
    /// Worth asking before every push, because the whole point of the paint
    /// path is that an untransformed element costs exactly what it did before
    /// transforms existed.
    #[inline]
    pub fn is_unit(&self) -> bool {
        *self == Self::unit()
    }
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<MonochromeSprite> for Primitive {
    fn from(sprite: MonochromeSprite) -> Self {
        Primitive::MonochromeSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct SubpixelSprite {
    pub order: DrawOrder,
    pub pad: u32, // align to 8 bytes
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub tile: AtlasTile,
    pub transformation: TransformationMatrix,
}

impl From<SubpixelSprite> for Primitive {
    fn from(sprite: SubpixelSprite) -> Self {
        Primitive::SubpixelSprite(sprite)
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PolychromeSprite {
    pub order: DrawOrder,
    pub pad: u32,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    pub tile: AtlasTile,
    /// See [`Quad::transformation`]. The monochrome and subpixel sprites have
    /// carried one since gpui's SVG element; this is the one that was missing,
    /// and it is what makes an `<img>` (or a colour emoji) inside a rotated box
    /// rotate with it.
    pub transformation: TransformationMatrix,
}

impl From<PolychromeSprite> for Primitive {
    fn from(sprite: PolychromeSprite) -> Self {
        Primitive::PolychromeSprite(sprite)
    }
}

#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct PaintSurface {
    pub order: DrawOrder,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    #[cfg(target_os = "macos")]
    pub image_buffer: core_video::pixel_buffer::CVPixelBuffer,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[expect(missing_docs)]
pub struct PathId(pub usize);

/// A line made up of a series of vertices and control points.
#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct Path<P: Clone + Debug + Default + PartialEq> {
    pub id: PathId,
    pub order: DrawOrder,
    pub bounds: Bounds<P>,
    pub content_mask: ContentMask<P>,
    pub vertices: Vec<PathVertex<P>>,
    pub color: Background,
    start: Point<P>,
    current: Point<P>,
    contour_count: usize,
}

impl Path<Pixels> {
    /// Create a new path with the given starting point.
    pub fn new(start: Point<Pixels>) -> Self {
        Self {
            id: PathId(0),
            order: DrawOrder::default(),
            vertices: Vec::new(),
            start,
            current: start,
            bounds: Bounds {
                origin: start,
                size: Default::default(),
            },
            content_mask: Default::default(),
            color: Default::default(),
            contour_count: 0,
        }
    }

    /// Scale this path by the given factor.
    pub fn scale(&self, factor: f32) -> Path<ScaledPixels> {
        Path {
            id: self.id,
            order: self.order,
            bounds: self.bounds.scale(factor),
            content_mask: self.content_mask.scale(factor),
            vertices: self
                .vertices
                .iter()
                .map(|vertex| vertex.scale(factor))
                .collect(),
            start: self.start.map(|start| start.scale(factor)),
            current: self.current.scale(factor),
            contour_count: self.contour_count,
            color: self.color,
        }
    }

    /// Move the start, current point to the given point.
    pub fn move_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        self.start = to;
        self.current = to;
    }

    /// Draw a straight line from the current point to the given point.
    pub fn line_to(&mut self, to: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }
        self.current = to;
    }

    /// Draw a curve from the current point to the given point, using the given control point.
    pub fn curve_to(&mut self, to: Point<Pixels>, ctrl: Point<Pixels>) {
        self.contour_count += 1;
        if self.contour_count > 1 {
            self.push_triangle(
                (self.start, self.current, to),
                (point(0., 1.), point(0., 1.), point(0., 1.)),
            );
        }

        self.push_triangle(
            (self.current, ctrl, to),
            (point(0., 0.), point(0.5, 0.), point(1., 1.)),
        );
        self.current = to;
    }

    /// Push a triangle to the Path.
    pub fn push_triangle(
        &mut self,
        xy: (Point<Pixels>, Point<Pixels>, Point<Pixels>),
        st: (Point<f32>, Point<f32>, Point<f32>),
    ) {
        self.bounds = self
            .bounds
            .union(&Bounds {
                origin: xy.0,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.1,
                size: Default::default(),
            })
            .union(&Bounds {
                origin: xy.2,
                size: Default::default(),
            });

        self.vertices.push(PathVertex {
            xy_position: xy.0,
            st_position: st.0,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.1,
            st_position: st.1,
            content_mask: Default::default(),
        });
        self.vertices.push(PathVertex {
            xy_position: xy.2,
            st_position: st.2,
            content_mask: Default::default(),
        });
    }
}

impl<T> Path<T>
where
    T: Clone + Debug + Default + PartialEq + PartialOrd + Add<T, Output = T> + Sub<Output = T>,
{
    #[allow(unused)]
    #[expect(missing_docs)]
    pub fn clipped_bounds(&self) -> Bounds<T> {
        self.bounds.intersect(&self.content_mask.bounds)
    }
}

impl From<Path<ScaledPixels>> for Primitive {
    fn from(path: Path<ScaledPixels>) -> Self {
        Primitive::Path(path)
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathVertex<P: Clone + Debug + Default + PartialEq> {
    pub xy_position: Point<P>,
    pub st_position: Point<f32>,
    pub content_mask: ContentMask<P>,
}

#[expect(missing_docs)]
impl PathVertex<Pixels> {
    pub fn scale(&self, factor: f32) -> PathVertex<ScaledPixels> {
        PathVertex {
            xy_position: self.xy_position.scale(factor),
            st_position: self.st_position,
            content_mask: self.content_mask.scale(factor),
        }
    }
}
