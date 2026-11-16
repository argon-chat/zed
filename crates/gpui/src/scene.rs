// todo("windows"): remove
#![cfg_attr(windows, allow(dead_code))]

#[cfg(feature = "schemars")]
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::{
    AtlasTextureId, AtlasTile, Background, Bounds, ContentMask, Corners, Edges, Hsla, Path,
    PathTiles, Pixels, Point, Radians, ScaledPixels, Size, TimedTransformation,
    bounds_tree::BoundsTree, point, px,
};
use smallvec::SmallVec;
use std::{
    collections::{HashMap, VecDeque},
    fmt::Debug,
    iter::Peekable,
    mem,
    ops::Range,
    rc::Rc,
    slice,
    time::Duration,
};

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

impl PaddedBool32 {
    /// The wrapped boolean.
    pub fn as_bool(self) -> bool {
        self.0 != 0
    }
}

struct Layer {
    primitives: SmallVec<[Range<usize>; 1]>,
    /// vue-native fork: the indices into [`Scene::vn_layers`] of the
    /// composites closed inside this layer, which take its order.
    vn_composites: Vec<usize>,
    bounds: Bounds<ScaledPixels>,
    max_kind: PrimitiveKind,
    max_shared_kind: PrimitiveKind,
}

impl Layer {
    fn new() -> Self {
        Self {
            primitives: SmallVec::new(),
            vn_composites: Vec::new(),
            bounds: Bounds::default(),
            max_kind: PrimitiveKind::Shadow,
            max_shared_kind: PrimitiveKind::VnLayer,
        }
    }
}

#[derive(Default)]
#[expect(missing_docs)]
pub struct Scene {
    pub(crate) paint_operations: Vec<PaintOperation>,
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<usize>,
    layers: VecDeque<Layer>,
    pub shadows: Vec<Shadow>,
    pub quads: Vec<Quad>,
    pub backdrop_blur_rects: Vec<BackdropBlurRect>,
    pub effect_quads: Vec<EffectQuad>,
    /// vue-native fork: the images effect quads sample; [`EffectQuad::texture_set`]
    /// indexes it.
    pub effect_texture_sets: Vec<std::sync::Arc<crate::EffectTextureSet>>,
    effect_texture_set_index: HashMap<crate::EffectTextureKey, usize>,
    pub paths: Vec<PaintedPath>,
    pub underlines: Vec<Underline>,
    pub monochrome_sprites: Vec<MonochromeSprite>,
    pub subpixel_sprites: Vec<SubpixelSprite>,
    pub polychrome_sprites: Vec<PolychromeSprite>,
    pub surfaces: Vec<PaintSurface>,
    /// vue-native fork: the composites of isolated layers
    /// ([`Scene::vn_push_isolation`]), each drawing one group through its clip
    /// shape. After [`Scene::finish`] every vector above holds the root group
    /// first and then each isolated group's, in [`Scene::vn_group_ranges`].
    pub vn_layers: Vec<VnLayer>,
    /// vue-native fork: beside each of [`Scene::vn_layers`], the effects that
    /// composite it instead of its clip shape.
    pub vn_layer_effects: Vec<Option<VnEffectLayer>>,
    /// vue-native fork: the quads every [`VnEffectLayer`] draws, in no group's
    /// order (a layer names its run).
    pub vn_effect_draws: Vec<VnEffectDraw>,
    /// vue-native fork: every isolated layer's clip edges, `[x0, y0, x1, y1]`
    /// in device pixels; a [`VnLayer`] names its run.
    pub vn_clip_segments: Vec<[f32; 4]>,
    /// Gradient stacks used by text-mask composites, shared across source groups.
    pub vn_text_backgrounds: Vec<VnTextBackground>,
    /// vue-native fork: the groups still open, innermost last, each holding
    /// the lists of the group around it while it paints.
    vn_open: Vec<VnOpenGroup>,
    /// vue-native fork: the closed groups, until `finish` appends them.
    vn_groups: Vec<VnGroupLists>,
    // Reopened deferred descendants append to one source before the matrix
    // runs. Filtering separately changes translucent colors in linear RGB.
    vn_color_groups: HashMap<(u8, u64, usize), usize>,
    vn_contexts: HashMap<(usize, VnIsolationIdentity), usize>,
    /// vue-native fork: where each group's primitives are after `finish`:
    /// the root at 0, then the groups a [`VnLayer::group`] names.
    vn_group_ranges: Vec<VnGroupRanges>,
    /// vue-native fork: the groups painted for [`VnIsolation::capture`],
    /// which the renderer draws first and keeps as snapshots.
    pub vn_captures: Vec<VnCapture>,
    animations: Vec<SpriteAnimation>,
}

/// vue-native fork: what a captured group knows of the element it holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VnCaptureInfo {
    /// The snapshot this capture writes.
    pub id: u64,
    /// The element's border box in device pixels, as painted inside the
    /// capture (untransformed).
    pub border_box: Bounds<ScaledPixels>,
    /// The device-space transform the element had on the page, its own
    /// included.
    pub transform: TransformationMatrix,
}

/// vue-native fork: one capture of a finished scene. The renderer draws
/// `group` into a cleared target and keeps the `ink` rectangle of it as the
/// snapshot `info.id`.
#[derive(Clone, Copy, Debug)]
#[allow(missing_docs)]
pub struct VnCapture {
    pub info: VnCaptureInfo,
    pub group: u32,
    /// Device pixels the group painted, whole pixels; empty when nothing.
    pub ink: Bounds<ScaledPixels>,
}

/// vue-native fork: [`VnLayer::flags`].
pub mod vn_layer_flags {
    /// The source is the snapshot [`super::VnLayer::snapshot`], not a group.
    pub const SNAPSHOT: u32 = 1;
    /// Composited with `plus-lighter` (both factors one) instead of
    /// source-over.
    pub const PLUS_LIGHTER: u32 = 2;
    /// Drawn where the capture was taken; [`super::Scene::vn_resolve_snapshots`]
    /// clears it, so no renderer sees it.
    pub const IN_PLACE: u32 = 4;
}

/// vue-native fork: an isolated layer as it is opened: the clip it composites
/// through, in window device pixels, and the content mask around it.
#[derive(Clone, Debug)]
pub struct VnIsolation {
    /// The clip shape.
    pub geometry: std::sync::Arc<crate::VnClipGeometry>,
    /// A transform over the isolated source pixels before composition.
    pub color_matrix: Option<crate::VnColorMatrix>,
    /// A text-only alpha source, filled by this background stack.
    pub text_paint: Option<std::sync::Arc<crate::VnTextPaint>>,
    /// The content mask in force where the layer was opened; it cuts the
    /// composite as it cuts any primitive.
    pub content_mask: ContentMask<ScaledPixels>,
    /// A captured group: it composites nowhere, the renderer keeps it as a
    /// snapshot ([`Scene::vn_captures`]).
    pub capture: Option<VnCaptureInfo>,
}

impl VnIsolation {
    fn shared_key(&self, parent_context: usize) -> Option<(u8, u64, usize)> {
        if let Some(capture) = &self.capture {
            return Some((2, capture.id, 0));
        }
        if let Some(text) = &self.text_paint {
            return Some((1, text.fill.group_id, parent_context));
        }
        self.color_matrix.and_then(|matrix| (matrix.group_id != 0).then_some((0, matrix.group_id, 0)))
    }

    fn identity(&self) -> VnIsolationIdentity {
        if let Some(capture) = &self.capture { return VnIsolationIdentity::Capture(capture.id); }
        if let Some(text) = &self.text_paint { return VnIsolationIdentity::Text(text.fill.group_id); }
        if let Some(matrix) = self.color_matrix.filter(|matrix| matrix.group_id != 0) {
            return VnIsolationIdentity::Matrix(matrix.group_id);
        }
        VnIsolationIdentity::Clip(std::sync::Arc::as_ptr(&self.geometry) as usize)
    }
}

/// One text background and its positioning space. GPU layout matches HLSL.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct VnTextBackground {
    /// Gradient/solid and its CSS size, position and repeat settings.
    pub background: Background,
    /// Logical CSS background box (not device-scaled).
    pub bounds: Bounds<ScaledPixels>,
    /// Device-space inverse transform.
    pub inverse: TransformationMatrix,
    /// Device pixels per CSS pixel.
    pub scale_factor: f32,
    /// Explicit tail padding, initialized before GPU upload.
    pub pad: f32,
}

/// vue-native fork: the composite of one isolated layer. The renderer draws
/// group `group` into an offscreen target the size of the window, then
/// composites it over its parent's, multiplied by the coverage of the
/// segments `segment_start..segment_start + segment_count` of
/// [`Scene::vn_clip_segments`] under `fill_rule` (bit 0: 0 nonzero, 1
/// evenodd), and by the content mask. Bits 1–2 of `fill_rule` name a mask
/// ([`Scene::vn_begin_mask`]): 1 multiplies by the alpha, 2 by the luminance
/// of group `group + 1`, which the renderer draws into a target of its own.
///
/// A snapshot composite ([`vn_layer_flags::SNAPSHOT`]) samples the snapshot
/// instead: each device pixel maps through `source_transform` into the
/// capture's device space, where the snapshot covers `source_rect`.
///
/// Keep in sync with `struct VnLayer` in `gpui_windows/src/shaders.hlsl`
/// (56 words); the renderer uploads this vector as it is.
#[derive(Default, Debug, Copy, Clone)]
#[repr(C)]
#[allow(missing_docs)]
pub struct VnLayer {
    pub order: DrawOrder,
    /// The group, an index into [`Scene::vn_group_ranges`] (never 0, the root).
    pub group: u32,
    pub segment_start: u32,
    pub segment_count: u32,
    pub fill_rule: u32,
    pub pad: [u32; 3],
    /// Four coefficient rows and the bias column, matching the shader.
    pub color_matrix: [f32; 20],
    /// The device pixels the composite covers: the shape's box, out to whole
    /// pixels and one more for the antialiased edge, cut by the mask.
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    /// The snapshot id, low and high words.
    pub snapshot: [u32; 2],
    /// [`vn_layer_flags`].
    pub flags: u32,
    pub opacity: f32,
    /// `[x, y, width, height]` of the snapshot in capture device pixels.
    pub source_rect: [f32; 4],
    /// Rows `[a, b, tx, _]` and `[c, d, ty, _]`: device pixel to capture
    /// device pixel.
    pub source_transform: [f32; 8],
}

impl VnLayer {
    /// The snapshot a [`vn_layer_flags::SNAPSHOT`] composite samples.
    pub fn snapshot_id(&self) -> Option<u64> {
        (self.flags & vn_layer_flags::SNAPSHOT != 0)
            .then(|| self.snapshot[0] as u64 | (self.snapshot[1] as u64) << 32)
    }
}

/// vue-native fork: a snapshot drawn into a box ([`Scene::vn_insert_snapshot`]).
#[derive(Clone, Copy, Debug)]
#[allow(missing_docs)]
pub struct VnSnapshotPaint {
    pub id: u64,
    /// Device pixel to the box's unit square, `[a, b, tx, c, d, ty]`;
    /// [`Scene::vn_resolve_snapshots`] maps the square onto the capture's
    /// border box.
    pub to_unit: [f32; 6],
    /// The device pixels the box covers.
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub opacity: f32,
    pub plus_lighter: bool,
    /// Where the capture was taken, ignoring `to_unit`.
    pub in_place: bool,
}

/// vue-native fork: one `--shading` quad of a [`VnEffectLayer`], with the key
/// of its history (0 for none).
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct VnEffectDraw {
    pub quad: EffectQuad,
    pub history: u64,
    /// The images the quad samples; in [`Scene::vn_effect_draws`] they are
    /// interned like a primitive's ([`EffectQuad::texture_set`]) and this is None.
    pub textures: Option<std::sync::Arc<crate::EffectTextureSet>>,
}

/// vue-native fork: a [`VnLayer`] composited by its [`Scene::vn_effect_draws`]
/// in order, each reading the layer's group (none when it is 0) as content.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[allow(missing_docs)]
pub struct VnEffectLayer {
    pub start: u32,
    pub count: u32,
}

impl VnEffectLayer {
    /// The indices of its draws in [`Scene::vn_effect_draws`].
    pub fn draws(&self) -> Range<usize> {
        self.start as usize..(self.start + self.count) as usize
    }
}

/// vue-native fork: what [`Scene::vn_push_effect`] opens: the quads that
/// composite what is painted until the matching [`Scene::vn_pop_isolation`].
#[derive(Clone, Debug)]
#[allow(missing_docs)]
pub struct VnEffectPaint {
    pub draws: Vec<VnEffectDraw>,
    /// Whether the draws read what is painted inside; without, nothing is.
    pub content: bool,
    pub content_mask: ContentMask<ScaledPixels>,
}

/// vue-native fork: what the renderer iterates for one group: runs of
/// primitives as before, and runs of [`VnLayer`] composites.
#[derive(Debug)]
pub enum VnBatch {
    /// Primitives drawn into the group's target.
    Primitives(PrimitiveBatch),
    /// The composites `range` of [`Scene::vn_layers`], in order.
    Layers(Range<usize>),
}

/// vue-native fork: one group's primitives in each of the scene's vectors.
#[derive(Clone, Debug, Default)]
pub struct VnGroupRanges {
    shadows: Range<usize>,
    quads: Range<usize>,
    backdrop_blur_rects: Range<usize>,
    effect_quads: Range<usize>,
    paths: Range<usize>,
    underlines: Range<usize>,
    monochrome_sprites: Range<usize>,
    subpixel_sprites: Range<usize>,
    polychrome_sprites: Range<usize>,
    surfaces: Range<usize>,
    vn_layers: Range<usize>,
}

/// vue-native fork: everything a group gathers while it paints. The scene's
/// own fields are the open group's; opening another swaps them out.
#[derive(Default)]
struct VnGroupLists {
    primitive_bounds: BoundsTree<ScaledPixels>,
    layer_stack: Vec<usize>,
    layers: VecDeque<Layer>,
    shadows: Vec<Shadow>,
    quads: Vec<Quad>,
    backdrop_blur_rects: Vec<BackdropBlurRect>,
    effect_quads: Vec<EffectQuad>,
    paths: Vec<PaintedPath>,
    underlines: Vec<Underline>,
    monochrome_sprites: Vec<MonochromeSprite>,
    subpixel_sprites: Vec<SubpixelSprite>,
    polychrome_sprites: Vec<PolychromeSprite>,
    surfaces: Vec<PaintSurface>,
    vn_layers: Vec<VnLayer>,
    vn_layer_effects: Vec<Option<VnEffectLayer>>,
}

impl VnGroupLists {
    fn is_empty(&self) -> bool {
        self.shadows.is_empty()
            && self.quads.is_empty()
            && self.backdrop_blur_rects.is_empty()
            && self.effect_quads.is_empty()
            && self.paths.is_empty()
            && self.underlines.is_empty()
            && self.monochrome_sprites.is_empty()
            && self.subpixel_sprites.is_empty()
            && self.polychrome_sprites.is_empty()
            && self.surfaces.is_empty()
            && self.vn_layers.is_empty()
    }

    /// The device pixels everything in the group touches, before any clip of
    /// a composite it holds.
    fn ink(&self) -> Option<Bounds<ScaledPixels>> {
        fn seen(
            ink: &mut Option<Bounds<ScaledPixels>>,
            bounds: Bounds<ScaledPixels>,
            transform: Option<&TransformationMatrix>,
            mask: &ContentMask<ScaledPixels>,
        ) {
            let bounds = match transform {
                Some(matrix) if !matrix.is_unit() => transformed_bounds(bounds, *matrix),
                _ => bounds,
            }
            .intersect(&mask.bounds);
            if bounds.is_empty() {
                return;
            }
            *ink = Some(ink.map_or(bounds, |ink| ink.union(&bounds)));
        }
        let mut ink = None;
        for shadow in &self.shadows {
            let spread = shadow.blur_radius * 2.;
            let bounds = Bounds::from_corners(
                point(shadow.bounds.origin.x - spread, shadow.bounds.origin.y - spread),
                point(
                    shadow.bounds.origin.x + shadow.bounds.size.width + spread,
                    shadow.bounds.origin.y + shadow.bounds.size.height + spread,
                ),
            );
            seen(&mut ink, bounds, Some(&shadow.transformation), &shadow.content_mask);
        }
        for quad in &self.quads {
            seen(&mut ink, quad.bounds, Some(&quad.transformation), &quad.content_mask);
        }
        for rect in &self.backdrop_blur_rects {
            seen(&mut ink, rect.bounds, None, &rect.content_mask);
        }
        for quad in &self.effect_quads {
            seen(&mut ink, quad.bounds, None, &quad.content_mask);
        }
        for path in &self.paths {
            seen(&mut ink, path.bounds, None, &path.paint.content_mask);
        }
        for underline in &self.underlines {
            seen(&mut ink, underline.bounds, Some(&underline.transformation), &underline.content_mask);
        }
        for sprite in &self.monochrome_sprites {
            seen(&mut ink, sprite.bounds, Some(&sprite.transformation), &sprite.content_mask);
        }
        for sprite in &self.subpixel_sprites {
            seen(&mut ink, sprite.bounds, Some(&sprite.transformation), &sprite.content_mask);
        }
        for sprite in &self.polychrome_sprites {
            seen(&mut ink, sprite.bounds, Some(&sprite.transformation), &sprite.content_mask);
        }
        for surface in &self.surfaces {
            seen(&mut ink, surface.bounds, None, &surface.content_mask);
        }
        for layer in &self.vn_layers {
            seen(&mut ink, layer.bounds, None, &layer.content_mask);
        }
        ink
    }

    /// The draw-order sort `Scene::finish` gives every group.
    fn sort(&mut self) {
        let keys = &mut Vec::new();
        sort_in_place(&mut self.shadows, keys, |shadow| (shadow.order, 0));
        sort_in_place(&mut self.quads, keys, |quad| (quad.order, 0));
        sort_in_place(&mut self.backdrop_blur_rects, keys, |rect| (rect.order, 0));
        // Secondary key `effect_id`: the batch iterator breaks a run when the
        // id changes (each id is a different pipeline), so sorting by it keeps
        // same-effect quads at the same draw order in ONE instanced draw. The
        // texture set breaks a run the same way.
        self.effect_quads.sort_by_key(|effect_quad| {
            (
                effect_quad.order,
                effect_quad.effect_id,
                effect_quad.texture_set(),
            )
        });
        self.paths.sort_by_key(|path| path.order);
        sort_in_place(&mut self.underlines, keys, |underline| (underline.order, 0));
        sort_in_place(&mut self.monochrome_sprites, keys, |sprite| {
            (sprite.order, sprite.tile.texture_id.index)
        });
        sort_in_place(&mut self.subpixel_sprites, keys, |sprite| {
            (sprite.order, sprite.tile.texture_id.index)
        });
        sort_in_place(&mut self.polychrome_sprites, keys, |sprite| {
            (sprite.order, sprite.tile.texture_id.index)
        });
        self.surfaces.sort_by_key(|surface| surface.order);
        if self.vn_layer_effects.iter().any(Option::is_some) {
            let mut layers: Vec<_> = mem::take(&mut self.vn_layers)
                .into_iter()
                .zip(mem::take(&mut self.vn_layer_effects))
                .collect();
            layers.sort_by_key(|(layer, _)| layer.order);
            (self.vn_layers, self.vn_layer_effects) = layers.into_iter().unzip();
        } else {
            self.vn_layers.sort_by_key(|layer| layer.order);
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum VnIsolationIdentity {
    Matrix(u64),
    Text(u64),
    Clip(usize),
    Effect(usize),
    Capture(u64),
}

struct VnOpenGroup {
    /// The lists of the group around this one.
    saved: VnGroupLists,
    isolation: VnIsolation,
    effect: Option<VnEffectPaint>,
    existing_group: Option<usize>,
    shared_key: Option<(u8, u64, usize)>,
    context: usize,
    /// Once the mask is painting: the group's own lists, and whether the mask
    /// is luminance.
    mask: Option<(VnGroupLists, bool)>,
}

#[expect(missing_docs)]
impl Scene {
    pub fn clear(&mut self) {
        self.paint_operations.clear();
        self.primitive_bounds.clear();
        self.layer_stack.clear();
        self.layers.clear();
        self.paths.clear();
        self.shadows.clear();
        self.quads.clear();
        self.backdrop_blur_rects.clear();
        self.effect_quads.clear();
        self.effect_texture_sets.clear();
        self.effect_texture_set_index.clear();
        self.underlines.clear();
        self.monochrome_sprites.clear();
        self.subpixel_sprites.clear();
        self.polychrome_sprites.clear();
        self.surfaces.clear();
        self.vn_layers.clear();
        self.vn_layer_effects.clear();
        self.vn_effect_draws.clear();
        self.vn_clip_segments.clear();
        self.vn_text_backgrounds.clear();
        self.vn_open.clear();
        self.vn_groups.clear();
        self.vn_color_groups.clear();
        self.vn_contexts.clear();
        self.vn_group_ranges.clear();
        self.vn_captures.clear();
        self.animations.clear();
    }

    /// vue-native fork: swaps the open group's lists with `lists`.
    fn vn_swap_lists(&mut self, lists: &mut VnGroupLists) {
        mem::swap(&mut self.primitive_bounds, &mut lists.primitive_bounds);
        mem::swap(&mut self.layer_stack, &mut lists.layer_stack);
        mem::swap(&mut self.layers, &mut lists.layers);
        mem::swap(&mut self.shadows, &mut lists.shadows);
        mem::swap(&mut self.quads, &mut lists.quads);
        mem::swap(&mut self.backdrop_blur_rects, &mut lists.backdrop_blur_rects);
        mem::swap(&mut self.effect_quads, &mut lists.effect_quads);
        mem::swap(&mut self.paths, &mut lists.paths);
        mem::swap(&mut self.underlines, &mut lists.underlines);
        mem::swap(&mut self.monochrome_sprites, &mut lists.monochrome_sprites);
        mem::swap(&mut self.subpixel_sprites, &mut lists.subpixel_sprites);
        mem::swap(&mut self.polychrome_sprites, &mut lists.polychrome_sprites);
        mem::swap(&mut self.surfaces, &mut lists.surfaces);
        mem::swap(&mut self.vn_layers, &mut lists.vn_layers);
        mem::swap(&mut self.vn_layer_effects, &mut lists.vn_layer_effects);
    }

    /// vue-native fork: opens an isolated layer. Every primitive inserted
    /// until the matching [`Scene::vn_pop_isolation`] goes into a group of its
    /// own, with its own draw orders: the renderer draws the group into an
    /// offscreen target and composites it through `isolation`'s clip, at the
    /// draw order the composite takes among its parent's primitives.
    ///
    /// The group is what CSS calls an isolated group: a backdrop filter in it
    /// sees what the group painted under it, not the page (the element is a
    /// Backdrop Root, Filter Effects 2 §3.1). The same group is where filters
    /// on an element's own painting would apply.
    pub fn vn_push_isolation(&mut self, isolation: VnIsolation) {
        self.paint_operations
            .push(PaintOperation::VnStartIsolation(isolation.clone()));
        let mut saved = VnGroupLists::default();
        self.vn_swap_lists(&mut saved);
        let parent_context = self.vn_open.last().map_or(0, |group| group.context);
        let shared_key = isolation.shared_key(parent_context);
        let next_context = self.vn_contexts.len() + 1;
        let context = *self.vn_contexts.entry((parent_context, isolation.identity())).or_insert(next_context);
        let existing_group = shared_key.and_then(|key| self.vn_color_groups.get(&key).copied());
        if let Some(index) = existing_group {
            let mut source = mem::take(&mut self.vn_groups[index]);
            self.vn_swap_lists(&mut source);
        }
        self.vn_open.push(VnOpenGroup {
            saved,
            isolation,
            effect: None,
            existing_group,
            shared_key,
            context,
            mask: None,
        });
    }

    /// vue-native fork: opens the group `effect`'s draws read as their content;
    /// the matching [`Scene::vn_pop_isolation`] inserts them as one [`VnEffectLayer`].
    pub fn vn_push_effect(&mut self, effect: VnEffectPaint) {
        self.paint_operations
            .push(PaintOperation::VnStartEffect(effect.clone()));
        let mut saved = VnGroupLists::default();
        self.vn_swap_lists(&mut saved);
        let next_context = self.vn_contexts.len() + 1;
        let parent_context = self.vn_open.last().map_or(0, |group| group.context);
        let context = *self
            .vn_contexts
            .entry((parent_context, VnIsolationIdentity::Effect(next_context)))
            .or_insert(next_context);
        let isolation = VnIsolation {
            geometry: std::sync::Arc::new(crate::VnClipGeometry {
                fill_rule: crate::VnFillRule::NonZero,
                segments: Vec::new(),
                bounds: effect.content_mask.bounds,
            }),
            color_matrix: None,
            text_paint: None,
            content_mask: effect.content_mask,
            capture: None,
        };
        self.vn_open.push(VnOpenGroup {
            saved,
            isolation,
            effect: Some(effect),
            existing_group: None,
            shared_key: None,
            context,
            mask: None,
        });
    }

    /// vue-native fork: what the innermost isolated layer paints from here to
    /// its [`Scene::vn_pop_isolation`] is its mask: the group composites
    /// multiplied by the mask's alpha, or its luminance.
    pub fn vn_begin_mask(&mut self, luminance: bool) {
        self.paint_operations.push(PaintOperation::VnBeginMask(luminance));
        let Some(open) = self.vn_open.last() else {
            return;
        };
        if open.mask.is_some()
            || open.effect.is_some()
            || open.existing_group.is_some()
            || open.shared_key.is_some()
        {
            return;
        }
        let mut content = VnGroupLists::default();
        self.vn_swap_lists(&mut content);
        if let Some(open) = self.vn_open.last_mut() {
            open.mask = Some((content, luminance));
        }
    }

    /// vue-native fork: whether an isolated layer is open.
    pub fn vn_is_isolating(&self) -> bool {
        !self.vn_open.is_empty()
    }

    /// vue-native fork: closes the innermost isolated layer
    /// ([`Scene::vn_push_isolation`]) and inserts its composite.
    pub fn vn_pop_isolation(&mut self) {
        self.paint_operations.push(PaintOperation::VnEndIsolation);
        self.vn_close_group();
    }

    fn vn_close_group(&mut self) {
        let Some(open) = self.vn_open.pop() else {
            return;
        };
        // Layers the group left open (`finish` closing it) still paint.
        self.layer_stack.clear();
        self.insert_layers();
        let mut group = open.saved;
        self.vn_swap_lists(&mut group);
        if let Some(effect) = open.effect {
            self.vn_close_effect(group, effect);
            return;
        }
        let (mut group, mask_group) = match open.mask {
            Some((content, luminance)) => (content, Some((group, luminance))),
            None => (group, None),
        };
        if let Some(index) = open.existing_group {
            // Its composite was inserted on the first paint. Appending source
            // pixels here must neither filter them alone nor composite twice.
            self.vn_groups[index] = group;
            return;
        }
        let color_group = open.shared_key;
        if let Some(info) = open.isolation.capture {
            if let Some(key) = color_group {
                self.vn_color_groups.insert(key, self.vn_groups.len());
            }
            group.layer_stack = Vec::new();
            self.vn_groups.push(group);
            self.vn_captures.push(VnCapture {
                info,
                group: self.vn_groups.len() as u32,
                ink: Bounds::default(),
            });
            return;
        }
        let geometry = &open.isolation.geometry;
        let mask = open.isolation.content_mask;
        // The antialiased edge reaches half a pixel past the shape.
        let shape = geometry.bounds;
        let bounds = Bounds::from_corners(
            point(
                ScaledPixels((shape.origin.x.0 - 1.).floor()),
                ScaledPixels((shape.origin.y.0 - 1.).floor()),
            ),
            point(
                ScaledPixels((shape.origin.x.0 + shape.size.width.0 + 1.).ceil()),
                ScaledPixels((shape.origin.y.0 + shape.size.height.0 + 1.).ceil()),
            ),
        )
        .intersect(&mask.bounds);
        // An empty mask hides everything, as CSS's transparent black layer does.
        let hidden = mask_group.as_ref().is_some_and(|(lists, _)| lists.is_empty());
        if (group.is_empty() && color_group.is_none()) || hidden || geometry.segments.is_empty() || bounds.is_empty() {
            return;
        }
        if color_group.is_none() { group.primitive_bounds = BoundsTree::default(); }
        group.layer_stack = Vec::new();
        if let Some(key) = color_group { self.vn_color_groups.insert(key, self.vn_groups.len()); }
        self.vn_groups.push(group);
        let mask_mode = match mask_group {
            Some((mut lists, luminance)) => {
                lists.primitive_bounds = BoundsTree::default();
                lists.layer_stack = Vec::new();
                self.vn_groups.push(lists);
                if luminance { 2 } else { 1 }
            }
            None => 0,
        };
        let group_index = self.vn_groups.len() as u32 - u32::from(mask_mode != 0);
        let segment_start = self.vn_clip_segments.len() as u32;
        self.vn_clip_segments.extend_from_slice(&geometry.segments);
        let text_start = self.vn_text_backgrounds.len() as u32;
        if let Some(text) = &open.isolation.text_paint {
            let m = text.inverse;
            self.vn_text_backgrounds.extend(text.fill.backgrounds.iter().map(|background| VnTextBackground {
                background: *background,
                bounds: Bounds::new(
                    point(ScaledPixels(text.bounds[0]), ScaledPixels(text.bounds[1])),
                    crate::size(ScaledPixels(text.bounds[2]), ScaledPixels(text.bounds[3]))),
                inverse: TransformationMatrix { rotation_scale: [[m[0], m[1]], [m[2], m[3]]], translation: [m[4], m[5]] },
                scale_factor: text.scale_factor, pad: 0.,
            }));
        }
        let text_count = self.vn_text_backgrounds.len() as u32 - text_start;
        self.vn_layers.push(VnLayer {
            group: group_index,
            segment_start,
            segment_count: geometry.segments.len() as u32,
            fill_rule: (geometry.fill_rule == crate::VnFillRule::EvenOdd) as u32 | mask_mode << 1,
            pad: [open.isolation.color_matrix.map_or(0, |m| if m.linear_rgb { 1 } else { 2 }), text_start, text_count],
            color_matrix: open.isolation.color_matrix.map_or([0.; 20], |matrix| matrix.gpu_values()),
            bounds,
            content_mask: mask,
            opacity: 1.,
            ..VnLayer::default()
        });
        self.vn_layer_effects.push(None);
        self.vn_place_composite(self.vn_layers.len() - 1, bounds, false);
    }

    fn vn_close_effect(&mut self, mut group: VnGroupLists, effect: VnEffectPaint) {
        let mask = effect.content_mask;
        let Some(reach) = effect
            .draws
            .iter()
            .map(|draw| draw.quad.bounds)
            .reduce(|union, bounds| union.union(&bounds))
        else {
            return;
        };
        let bounds = Bounds::from_corners(
            point(
                ScaledPixels(reach.origin.x.0.floor()),
                ScaledPixels(reach.origin.y.0.floor()),
            ),
            point(
                ScaledPixels((reach.origin.x.0 + reach.size.width.0).ceil()),
                ScaledPixels((reach.origin.y.0 + reach.size.height.0).ceil()),
            ),
        )
        .intersect(&mask.bounds);
        if bounds.is_empty() {
            return;
        }
        let group_index = if effect.content && !group.is_empty() {
            group.primitive_bounds = BoundsTree::default();
            group.layer_stack = Vec::new();
            self.vn_groups.push(group);
            self.vn_groups.len() as u32
        } else {
            0
        };
        let snapshots = effect.draws.iter().any(|draw| draw.quad.needs_backdrop());
        let start = self.vn_effect_draws.len() as u32;
        for draw in &effect.draws {
            let mut quad = draw.quad;
            self.vn_set_effect_textures(&mut quad, draw.textures.as_ref());
            self.vn_effect_draws.push(VnEffectDraw {
                quad,
                history: draw.history,
                textures: None,
            });
        }
        self.vn_layers.push(VnLayer {
            group: group_index,
            bounds,
            content_mask: mask,
            opacity: 1.,
            ..Default::default()
        });
        self.vn_layer_effects.push(Some(VnEffectLayer {
            start,
            count: effect.draws.len() as u32,
        }));
        self.vn_place_composite(self.vn_layers.len() - 1, bounds, snapshots);
    }

    /// vue-native fork: the [`VnEffectLayer`] that composites
    /// [`Scene::vn_layers`]`[index]`, if effects do.
    pub fn vn_effect_layer(&self, index: usize) -> Option<VnEffectLayer> {
        self.vn_layer_effects.get(index).copied().flatten()
    }

    /// vue-native fork: composites the snapshot `paint.id` at the current
    /// draw position. Its mapping is settled after `finish`
    /// ([`Scene::vn_resolve_snapshots`]), once every capture is known.
    pub fn vn_insert_snapshot(&mut self, paint: VnSnapshotPaint) {
        let bounds = paint.bounds.intersect(&paint.content_mask.bounds);
        if bounds.is_empty() || paint.opacity <= 0. {
            return;
        }
        self.paint_operations.push(PaintOperation::VnSnapshot(paint));
        let [a, b, tx, c, d, ty] = paint.to_unit;
        self.vn_layers.push(VnLayer {
            bounds,
            content_mask: paint.content_mask,
            snapshot: [paint.id as u32, (paint.id >> 32) as u32],
            flags: vn_layer_flags::SNAPSHOT
                | if paint.plus_lighter { vn_layer_flags::PLUS_LIGHTER } else { 0 }
                | if paint.in_place { vn_layer_flags::IN_PLACE } else { 0 },
            opacity: paint.opacity,
            source_transform: [a, b, tx, 0., c, d, ty, 0.],
            ..VnLayer::default()
        });
        self.vn_layer_effects.push(None);
        self.vn_place_composite(self.vn_layers.len() - 1, bounds, false);
    }

    /// vue-native fork: maps each snapshot composite's unit square onto its
    /// capture's border box and names the snapshot's rectangle. `known` finds
    /// a capture of an earlier frame; one of this scene wins. A composite
    /// whose capture is unknown draws nothing.
    pub fn vn_resolve_snapshots(&mut self, known: impl Fn(u64) -> Option<(VnCaptureInfo, Bounds<ScaledPixels>)>) {
        let current: HashMap<u64, (VnCaptureInfo, Bounds<ScaledPixels>)> =
            self.vn_captures.iter().map(|c| (c.info.id, (c.info, c.ink))).collect();
        for layer in &mut self.vn_layers {
            let Some(id) = layer.snapshot_id() else { continue };
            let Some((info, ink)) = current.get(&id).copied().or_else(|| known(id)) else {
                layer.opacity = 0.;
                continue;
            };
            if ink.is_empty() {
                layer.opacity = 0.;
                continue;
            }
            if layer.flags & vn_layer_flags::IN_PLACE != 0 {
                layer.flags &= !vn_layer_flags::IN_PLACE;
                let Some(inverse) = info.transform.inverse() else {
                    layer.opacity = 0.;
                    continue;
                };
                let [[a, b], [c, d]] = inverse.rotation_scale;
                let [tx, ty] = inverse.translation;
                layer.source_transform = [a, b, tx, 0., c, d, ty, 0.];
                let drawn = transformed_bounds(ink, info.transform);
                let drawn = Bounds::from_corners(
                    point(ScaledPixels(drawn.origin.x.0.floor()), ScaledPixels(drawn.origin.y.0.floor())),
                    point(
                        ScaledPixels((drawn.origin.x.0 + drawn.size.width.0).ceil()),
                        ScaledPixels((drawn.origin.y.0 + drawn.size.height.0).ceil()),
                    ),
                );
                layer.bounds = layer.bounds.intersect(&drawn);
            } else {
                let bb = info.border_box;
                let t = layer.source_transform;
                let (w, h) = (bb.size.width.0, bb.size.height.0);
                layer.source_transform = [
                    t[0] * w, t[1] * w, t[2] * w + bb.origin.x.0, 0.,
                    t[4] * h, t[5] * h, t[6] * h + bb.origin.y.0, 0.,
                ];
            }
            layer.source_rect = [ink.origin.x.0, ink.origin.y.0, ink.size.width.0, ink.size.height.0];
        }
    }

    /// vue-native fork: appends a closed group to the scene's vectors.
    fn vn_append(&mut self, group: VnGroupLists) -> VnGroupRanges {
        fn append<T>(into: &mut Vec<T>, from: Vec<T>) -> Range<usize> {
            let start = into.len();
            into.extend(from);
            start..into.len()
        }
        append(&mut self.vn_layer_effects, group.vn_layer_effects);
        VnGroupRanges {
            shadows: append(&mut self.shadows, group.shadows),
            quads: append(&mut self.quads, group.quads),
            backdrop_blur_rects: append(&mut self.backdrop_blur_rects, group.backdrop_blur_rects),
            effect_quads: append(&mut self.effect_quads, group.effect_quads),
            paths: append(&mut self.paths, group.paths),
            underlines: append(&mut self.underlines, group.underlines),
            monochrome_sprites: append(&mut self.monochrome_sprites, group.monochrome_sprites),
            subpixel_sprites: append(&mut self.subpixel_sprites, group.subpixel_sprites),
            polychrome_sprites: append(&mut self.polychrome_sprites, group.polychrome_sprites),
            surfaces: append(&mut self.surfaces, group.surfaces),
            vn_layers: append(&mut self.vn_layers, group.vn_layers),
        }
    }

    /// vue-native fork: the ranges that hold everything the vectors hold —
    /// the root group before `finish` appended any other.
    fn vn_whole_ranges(&self) -> VnGroupRanges {
        VnGroupRanges {
            shadows: 0..self.shadows.len(),
            quads: 0..self.quads.len(),
            backdrop_blur_rects: 0..self.backdrop_blur_rects.len(),
            effect_quads: 0..self.effect_quads.len(),
            paths: 0..self.paths.len(),
            underlines: 0..self.underlines.len(),
            monochrome_sprites: 0..self.monochrome_sprites.len(),
            subpixel_sprites: 0..self.subpixel_sprites.len(),
            polychrome_sprites: 0..self.polychrome_sprites.len(),
            surfaces: 0..self.surfaces.len(),
            vn_layers: 0..self.vn_layers.len(),
        }
    }

    /// vue-native fork: how many groups a finished scene holds, the root
    /// included.
    pub fn vn_group_count(&self) -> usize {
        self.vn_group_ranges.len().max(1)
    }

    /// vue-native fork: the batches of one group (0 is the root), with its
    /// isolated layers' composites as [`VnBatch::Layers`]; a renderer that
    /// isolates draws each composite's [`VnLayer::group`] through these too.
    pub fn vn_batches(&self, group: usize) -> impl Iterator<Item = VnBatch> + '_ {
        let ranges = self
            .vn_group_ranges
            .get(group)
            .cloned()
            .unwrap_or_else(|| if group == 0 { self.vn_whole_ranges() } else { VnGroupRanges::default() });
        BatchIterator::new(self, ranges)
    }

    fn vn_flatten_into(&self, group: usize, depth: usize, out: &mut Vec<PrimitiveBatch>) {
        for batch in self.vn_batches(group) {
            match batch {
                VnBatch::Primitives(batch) => out.push(batch),
                VnBatch::Layers(range) => {
                    // A cycle cannot be built, but a corrupt index must not
                    // recurse for ever.
                    if depth > 64 {
                        continue;
                    }
                    for layer in &self.vn_layers[range] {
                        if layer.group != 0 && layer.snapshot_id().is_none() {
                            self.vn_flatten_into(layer.group as usize, depth + 1, out);
                        }
                    }
                }
            }
        }
    }

    /// Inserts a sprite that [`Self::advance_animations`] keeps moving. The
    /// sprite's color is taken as its color before the animation's alpha.
    pub(crate) fn insert_animated_sprite(
        &mut self,
        mut sprite: MonochromeSprite,
        mut animation: SpriteAnimation,
    ) {
        sprite.animation = SpriteAnimationId(self.animations.len() as u32);
        animation.base_color = sprite.color;
        if self.try_insert_primitive(sprite.into()) {
            self.animations.push(animation);
        }
    }

    /// How long the window may wait between presents of this scene while only
    /// its animations need a frame: `None` when nothing in it animates, and
    /// zero when an animation wants every frame.
    pub(crate) fn animation_frame_interval(&self) -> Option<Duration> {
        self.animations
            .iter()
            .map(|animation| {
                animation
                    .transformation
                    .min_frame_interval()
                    .unwrap_or_default()
            })
            .min()
    }

    /// Moves every animated sprite to its pose at `elapsed` time on the app's
    /// synced animation clock, without the scene being rebuilt.
    pub(crate) fn advance_animations(&mut self, elapsed: Duration) {
        for animation in &self.animations {
            let Some(sprite) = self.monochrome_sprites.get_mut(animation.sprite_index) else {
                continue;
            };
            let pose = animation.transformation.at(elapsed);
            sprite.transformation = animation
                .outer
                .compose(pose.into_matrix(animation.center, animation.scale_factor));
            sprite.color = animation.base_color.opacity(pose.alpha_factor());
        }
    }

    pub fn len(&self) -> usize {
        self.paint_operations.len()
    }

    pub fn push_layer(&mut self) {
        self.layer_stack.push(self.layers.len());
        self.layers.push_back(Layer::new());
        self.paint_operations.push(PaintOperation::StartLayer);
    }

    pub fn pop_layer(&mut self) {
        self.layer_stack.pop();
        if self.layer_stack.is_empty() {
            self.insert_layers();
        }
        self.paint_operations.push(PaintOperation::EndLayer);
    }

    fn insert_layers(&mut self) {
        while let Some(layer) = self.layers.pop_front() {
            self.insert_layer(layer);
        }
    }

    fn insert_layer(&mut self, layer: Layer) {
        if layer.primitives.is_empty() && layer.vn_composites.is_empty() {
            return;
        }
        let order =
            self.primitive_bounds
                .insert(layer.bounds, layer.max_kind, layer.max_shared_kind);
        for index in layer.primitives.into_iter().flatten() {
            if let PaintOperation::Primitive(primitive) = &self.paint_operations[index] {
                let primitive = primitive.clone();
                self.push_primitive(primitive, order);
            }
        }
        for index in layer.vn_composites {
            self.vn_set_composite_order(index, order);
        }
    }

    /// vue-native fork: a primitive that samples the render target draws
    /// after everything painted before it and before everything painted after
    /// it. The open layers would give it their one order, so they are inserted
    /// as they stand and reopened empty.
    fn vn_split_layers(&mut self) {
        let open = self.layer_stack.len();
        if open == 0 {
            return;
        }
        self.insert_layers();
        self.layer_stack.clear();
        for index in 0..open {
            self.layer_stack.push(index);
            self.layers.push_back(Layer::new());
        }
    }

    /// vue-native fork: gives the composite [`Scene::vn_layers`]`[index]`,
    /// which covers `bounds`, an order: the open layer's once that is
    /// inserted, or one of its own.
    fn vn_place_composite(
        &mut self,
        index: usize,
        bounds: Bounds<ScaledPixels>,
        snapshots_backdrop: bool,
    ) {
        let kind = PrimitiveKind::VnLayer;
        if snapshots_backdrop {
            self.vn_split_layers();
        } else if let Some(&layer) = self.layer_stack.last()
            && let Some(layer) = self.layers.get_mut(layer)
        {
            layer.bounds = layer.bounds.join(&bounds);
            layer.max_kind = layer.max_kind.max(kind);
            layer.max_shared_kind = layer.max_shared_kind.min(kind.max_shared_kind());
            layer.vn_composites.push(index);
            return;
        }
        // A composite that samples its backdrop cannot share an order with
        // another composite: they draw in one run.
        let max_shared_kind = if snapshots_backdrop {
            PrimitiveKind::Surface
        } else {
            kind.max_shared_kind()
        };
        let order = self.primitive_bounds.insert(bounds, kind, max_shared_kind);
        self.vn_set_composite_order(index, order);
    }

    fn vn_set_composite_order(&mut self, index: usize, order: DrawOrder) {
        self.vn_layers[index].order = order;
        if let Some(effect) = self.vn_layer_effects[index] {
            for draw in &mut self.vn_effect_draws[effect.draws()] {
                draw.quad.order = order;
            }
        }
    }

    /// vue-native fork: points `quad` at `textures` in
    /// [`Self::effect_texture_sets`], which holds each distinct set of images once.
    fn vn_set_effect_textures(
        &mut self,
        quad: &mut EffectQuad,
        textures: Option<&std::sync::Arc<crate::EffectTextureSet>>,
    ) {
        quad.flags &= (1 << crate::EFFECT_FLAG_TEXTURE_SET_SHIFT) - 1;
        let Some(textures) = textures else {
            return;
        };
        let key = crate::effect_texture_key(textures);
        let index = match self.effect_texture_set_index.get(&key) {
            Some(index) => *index,
            None => {
                let index = self.effect_texture_sets.len();
                self.effect_texture_sets.push(textures.clone());
                self.effect_texture_set_index.insert(key, index);
                index
            }
        };
        quad.flags |= (index as u32 + 1) << crate::EFFECT_FLAG_TEXTURE_SET_SHIFT;
    }

    pub fn insert_primitive(&mut self, primitive: impl Into<Primitive>) {
        self.try_insert_primitive(primitive.into());
    }

    /// Returns whether the primitive was kept, rather than clipped away.
    fn try_insert_primitive(&mut self, primitive: Primitive) -> bool {
        let clipped_bounds = primitive.clipped_bounds();
        if clipped_bounds.is_empty() {
            return false;
        }

        let kind = primitive.kind();
        let snapshots_backdrop = primitive.snapshots_backdrop();
        if snapshots_backdrop {
            self.vn_split_layers();
        }
        if !snapshots_backdrop
            && let Some(&layer) = self.layer_stack.last()
            && let Some(layer) = self.layers.get_mut(layer)
        {
            layer.bounds = layer.bounds.join(&clipped_bounds);
            layer.max_kind = layer.max_kind.max(kind);
            layer.max_shared_kind = layer.max_shared_kind.min(kind.max_shared_kind());
            let index = self.paint_operations.len();
            match layer.primitives.last_mut() {
                Some(range) if range.end == index => range.end += 1,
                _ => layer.primitives.push(index..index + 1),
            }
        } else {
            let order = self
                .primitive_bounds
                .insert(clipped_bounds, kind, kind.max_shared_kind());
            self.push_primitive(primitive.clone(), order);
        }
        self.paint_operations
            .push(PaintOperation::Primitive(primitive));
        true
    }

    fn push_primitive(&mut self, primitive: Primitive, order: DrawOrder) {
        match primitive {
            Primitive::Shadow(shadow) => self.shadows.push(Shadow { order, ..shadow }),
            Primitive::Quad(quad) => self.quads.push(Quad { order, ..quad }),
            Primitive::BackdropBlurRect(rect) => self
                .backdrop_blur_rects
                .push(BackdropBlurRect { order, ..rect }),
            Primitive::EffectQuad(effect_quad, textures) => {
                let mut effect_quad = EffectQuad {
                    order,
                    ..effect_quad
                };
                self.vn_set_effect_textures(&mut effect_quad, textures.as_ref());
                self.effect_quads.push(effect_quad);
            }
            Primitive::Path(mut path) => {
                path.order = order;
                self.paths.push(path);
            }
            Primitive::Underline(underline) => {
                self.underlines.push(Underline { order, ..underline })
            }
            Primitive::MonochromeSprite(sprite) => self
                .monochrome_sprites
                .push(MonochromeSprite { order, ..sprite }),
            Primitive::SubpixelSprite(sprite) => self
                .subpixel_sprites
                .push(SubpixelSprite { order, ..sprite }),
            Primitive::PolychromeSprite(sprite) => self
                .polychrome_sprites
                .push(PolychromeSprite { order, ..sprite }),
            Primitive::Surface(mut surface) => {
                surface.order = order;
                self.surfaces.push(surface);
            }
        }
    }

    pub fn replay(&mut self, range: Range<usize>, prev_scene: &Scene) {
        for operation in &prev_scene.paint_operations[range] {
            match operation {
                PaintOperation::Primitive(primitive) => {
                    // Animation ids index the scene they were inserted into.
                    if let Primitive::MonochromeSprite(sprite) = primitive
                        && let Some(animation) =
                            prev_scene.animations.get(sprite.animation.0 as usize)
                    {
                        self.insert_animated_sprite(*sprite, *animation);
                    } else {
                        self.insert_primitive(primitive.clone());
                    }
                }
                PaintOperation::StartLayer => self.push_layer(),
                PaintOperation::EndLayer => self.pop_layer(),
                PaintOperation::VnStartIsolation(isolation) => {
                    self.vn_push_isolation(isolation.clone())
                }
                PaintOperation::VnStartEffect(effect) => self.vn_push_effect(effect.clone()),
                PaintOperation::VnEndIsolation => self.vn_pop_isolation(),
                PaintOperation::VnSnapshot(paint) => self.vn_insert_snapshot(*paint),
                PaintOperation::VnBeginMask(luminance) => self.vn_begin_mask(*luminance),
            }
        }
    }

    pub fn finish(&mut self) {
        // vue-native fork: once the groups are appended, sorting the vectors
        // again would mix them; a scene is finished once.
        if !self.vn_group_ranges.is_empty() {
            return;
        }
        // An isolated layer left open closes here.
        while !self.vn_open.is_empty() {
            self.vn_close_group();
        }
        self.layer_stack.clear();
        self.insert_layers();
        let mut root = VnGroupLists::default();
        self.vn_swap_lists(&mut root);
        root.sort();
        self.vn_swap_lists(&mut root);
        // Then each isolated group, sorted on its own, after the root.
        if self.vn_groups.is_empty() {
            self.vn_group_ranges.clear();
        } else {
            let mut ranges = vec![self.vn_whole_ranges()];
            for (index, mut group) in mem::take(&mut self.vn_groups).into_iter().enumerate() {
                if let Some(capture) = self.vn_captures.iter_mut().find(|c| c.group as usize == index + 1) {
                    capture.ink = group.ink().map_or_else(Bounds::default, |ink| {
                        Bounds::from_corners(
                            point(ScaledPixels(ink.origin.x.0.floor()), ScaledPixels(ink.origin.y.0.floor())),
                            point(
                                ScaledPixels((ink.origin.x.0 + ink.size.width.0).ceil()),
                                ScaledPixels((ink.origin.y.0 + ink.size.height.0).ceil()),
                            ),
                        )
                    });
                }
                group.sort();
                ranges.push(self.vn_append(group));
            }
            self.vn_group_ranges = ranges;
        }

        // Sorting moved the sprites, so animations learn where theirs ended up.
        if !self.animations.is_empty() {
            for (index, sprite) in self.monochrome_sprites.iter().enumerate() {
                if let Some(animation) = self.animations.get_mut(sprite.animation.0 as usize) {
                    animation.sprite_index = index;
                }
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
    ///
    /// vue-native fork: an isolated layer's group comes inline where its
    /// composite would draw, unclipped by its shape (cut to the shape's box by
    /// the content mask). That is what a renderer without isolated layers
    /// draws; the DirectX, Metal and wgpu renderers walk [`Scene::vn_batches`].
    pub fn batches(&self) -> impl Iterator<Item = PrimitiveBatch> + '_ {
        let mut out = Vec::new();
        self.vn_flatten_into(0, 0, &mut out);
        out.into_iter()
    }
}

/// Sorts small `(key, index)` pairs instead of the primitives themselves, then moves each
/// primitive into place once. Primitives are large, so a comparison sort that shuffles them
/// directly spends most of its time copying. Ties keep insertion order, as `sort_by_key`
/// does, because the index is part of what is sorted.
fn sort_in_place<T: Copy>(
    primitives: &mut [T],
    keys: &mut Vec<(u64, u32)>,
    key: impl Fn(&T) -> (u32, u32),
) {
    if primitives.is_sorted_by_key(&key) {
        return;
    }
    keys.clear();
    keys.extend(primitives.iter().enumerate().map(|(index, primitive)| {
        let (major, minor) = key(primitive);
        ((major as u64) << 32 | minor as u64, index as u32)
    }));
    keys.sort_unstable();

    // `keys[target].1` is the source index for `target`. Follow each cycle of that
    // permutation, marking visited slots by pointing them at themselves.
    for start in 0..primitives.len() {
        if keys[start].1 as usize == start {
            continue;
        }
        let displaced = primitives[start];
        let mut target = start;
        loop {
            let source = keys[target].1 as usize;
            keys[target].1 = target as u32;
            if source == start {
                primitives[target] = displaced;
                break;
            }
            primitives[target] = primitives[source];
            target = source;
        }
    }
}

/// The depth step between consecutive quads in a frame with `quad_count` quads.
pub fn quad_depth_scale(quad_count: usize) -> f32 {
    let denominator = (quad_count.min((1 << 24) - 1) + 1).next_power_of_two();
    1.0 / denominator as f32
}

/// Maps a quad's index in [`Scene::quads`] to depth, with greater values closer.
pub fn quad_depth(quad_id: u32, scale: f32) -> f32 {
    ((quad_id + 1) as f32 * scale).min(1.0)
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
    /// vue-native fork: an isolated layer's composite.
    VnLayer,
}

impl PrimitiveKind {
    /// We guarantee that overlapping primitives are drawn in paint order.
    /// But, to improve GPU draw batching, the renderer sorts primitives that
    /// share a draw order by kind, then by atlas texture, and only then in
    /// paint order. So, a primitive can only share the draw order of the
    /// topmost primitive it overlaps if that primitive's kind is at most the
    /// one returned here. Otherwise, it needs a new draw order.
    fn max_shared_kind(self) -> Self {
        match self {
            // vue-native fork: one batch of backdrop blurs reads one snapshot,
            // so a blur cannot share the order of a blur it overlaps.
            Self::BackdropBlurRect => Self::Quad,
            // vue-native fork: effect quads are sorted by effect and texture
            // set within an order, as sprites are by texture.
            Self::EffectQuad => Self::BackdropBlurRect,
            Self::MonochromeSprite => Self::Underline,
            Self::SubpixelSprite => Self::MonochromeSprite,
            Self::PolychromeSprite => Self::SubpixelSprite,
            kind => kind,
        }
    }
}

pub(crate) enum PaintOperation {
    Primitive(Primitive),
    StartLayer,
    EndLayer,
    /// vue-native fork: [`Scene::vn_push_isolation`].
    VnStartIsolation(VnIsolation),
    /// vue-native fork: [`Scene::vn_push_effect`].
    VnStartEffect(VnEffectPaint),
    /// vue-native fork: [`Scene::vn_pop_isolation`].
    VnEndIsolation,
    /// vue-native fork: [`Scene::vn_insert_snapshot`].
    VnSnapshot(VnSnapshotPaint),
    /// vue-native fork: [`Scene::vn_begin_mask`].
    VnBeginMask(bool),
}

#[derive(Clone)]
#[expect(missing_docs)]
pub enum Primitive {
    Shadow(Shadow),
    Quad(Quad),
    BackdropBlurRect(BackdropBlurRect),
    /// vue-native fork: with the images it samples.
    EffectQuad(EffectQuad, Option<std::sync::Arc<crate::EffectTextureSet>>),
    Path(PaintedPath),
    Underline(Underline),
    MonochromeSprite(MonochromeSprite),
    SubpixelSprite(SubpixelSprite),
    PolychromeSprite(PolychromeSprite),
    Surface(PaintSurface),
}

#[expect(missing_docs)]
impl Primitive {
    pub(crate) fn kind(&self) -> PrimitiveKind {
        match self {
            Primitive::Shadow(_) => PrimitiveKind::Shadow,
            Primitive::Quad(_) => PrimitiveKind::Quad,
            Primitive::BackdropBlurRect(_) => PrimitiveKind::BackdropBlurRect,
            Primitive::EffectQuad(..) => PrimitiveKind::EffectQuad,
            Primitive::Path(_) => PrimitiveKind::Path,
            Primitive::Underline(_) => PrimitiveKind::Underline,
            Primitive::MonochromeSprite(_) => PrimitiveKind::MonochromeSprite,
            Primitive::SubpixelSprite(_) => PrimitiveKind::SubpixelSprite,
            Primitive::PolychromeSprite(_) => PrimitiveKind::PolychromeSprite,
            Primitive::Surface(_) => PrimitiveKind::Surface,
        }
    }

    pub fn bounds(&self) -> &Bounds<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.bounds,
            Primitive::Quad(quad) => &quad.bounds,
            Primitive::BackdropBlurRect(backdrop_blur_rect) => &backdrop_blur_rect.bounds,
            Primitive::EffectQuad(effect_quad, _) => &effect_quad.bounds,
            Primitive::Path(path) => &path.bounds,
            Primitive::Underline(underline) => &underline.bounds,
            Primitive::MonochromeSprite(sprite) => &sprite.bounds,
            Primitive::SubpixelSprite(sprite) => &sprite.bounds,
            Primitive::PolychromeSprite(sprite) => &sprite.bounds,
            Primitive::Surface(surface) => &surface.bounds,
        }
    }

    /// The CSS `transform` this primitive carries, if its kind has one.
    ///
    /// `bounds` is the UNTRANSFORMED rect — the shader applies the matrix — so
    /// anything comparing a primitive's rectangle with a window-space rect (the
    /// content mask, the draw-order tree) has to map it first.
    pub fn transformation(&self) -> Option<TransformationMatrix> {
        match self {
            Primitive::Shadow(shadow) => Some(shadow.transformation),
            Primitive::Quad(quad) => Some(quad.transformation),
            Primitive::Underline(underline) => Some(underline.transformation),
            Primitive::MonochromeSprite(sprite) => Some(sprite.transformation),
            Primitive::SubpixelSprite(sprite) => Some(sprite.transformation),
            Primitive::PolychromeSprite(sprite) => Some(sprite.transformation),
            Primitive::BackdropBlurRect(_)
            | Primitive::EffectQuad(..)
            | Primitive::Path(_)
            | Primitive::Surface(_) => None,
        }
    }

    fn clipped_bounds(&self) -> Bounds<ScaledPixels> {
        self.transformation()
            .unwrap_or_default()
            .transform_bounds(*self.bounds())
            .intersect(&self.content_mask().bounds)
    }

    /// vue-native fork: whether the renderer composites this primitive from a
    /// snapshot of the render target under it.
    fn snapshots_backdrop(&self) -> bool {
        match self {
            Primitive::BackdropBlurRect(_) => true,
            Primitive::EffectQuad(effect_quad, _) => effect_quad.needs_backdrop(),
            _ => false,
        }
    }

    pub fn content_mask(&self) -> &ContentMask<ScaledPixels> {
        match self {
            Primitive::Shadow(shadow) => &shadow.content_mask,
            Primitive::Quad(quad) => &quad.content_mask,
            Primitive::BackdropBlurRect(backdrop_blur_rect) => &backdrop_blur_rect.content_mask,
            Primitive::EffectQuad(effect_quad, _) => &effect_quad.content_mask,
            Primitive::Path(path) => &path.paint.content_mask,
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
    paths_iter: Peekable<slice::Iter<'a, PaintedPath>>,
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
    vn_layers_start: usize,
    vn_layers_iter: Peekable<slice::Iter<'a, VnLayer>>,
}

impl<'a> BatchIterator<'a> {
    /// vue-native fork: the batches of one group's ranges of `scene`'s
    /// vectors. The ranges it yields index the whole vectors.
    fn new(scene: &'a Scene, r: VnGroupRanges) -> Self {
        BatchIterator {
            shadows_start: r.shadows.start,
            shadows_iter: scene.shadows[r.shadows].iter().peekable(),
            quads_start: r.quads.start,
            quads_iter: scene.quads[r.quads].iter().peekable(),
            backdrop_blur_rects_start: r.backdrop_blur_rects.start,
            backdrop_blur_rects_iter: scene.backdrop_blur_rects[r.backdrop_blur_rects]
                .iter()
                .peekable(),
            effect_quads_start: r.effect_quads.start,
            effect_quads_iter: scene.effect_quads[r.effect_quads].iter().peekable(),
            paths_start: r.paths.start,
            paths_iter: scene.paths[r.paths].iter().peekable(),
            underlines_start: r.underlines.start,
            underlines_iter: scene.underlines[r.underlines].iter().peekable(),
            monochrome_sprites_start: r.monochrome_sprites.start,
            monochrome_sprites_iter: scene.monochrome_sprites[r.monochrome_sprites]
                .iter()
                .peekable(),
            subpixel_sprites_start: r.subpixel_sprites.start,
            subpixel_sprites_iter: scene.subpixel_sprites[r.subpixel_sprites].iter().peekable(),
            polychrome_sprites_start: r.polychrome_sprites.start,
            polychrome_sprites_iter: scene.polychrome_sprites[r.polychrome_sprites]
                .iter()
                .peekable(),
            surfaces_start: r.surfaces.start,
            surfaces_iter: scene.surfaces[r.surfaces].iter().peekable(),
            vn_layers_start: r.vn_layers.start,
            vn_layers_iter: scene.vn_layers[r.vn_layers].iter().peekable(),
        }
    }
}

impl<'a> Iterator for BatchIterator<'a> {
    type Item = VnBatch;

    fn next(&mut self) -> Option<Self::Item> {
        // vue-native fork: a run of composites at the lowest order goes
        // first; otherwise the primitive kinds, as upstream picks them.
        let layer_order = self.vn_layers_iter.peek().map(|l| l.order);
        if let Some(layer_order) = layer_order {
            let lowest_primitive = [
                self.shadows_iter.peek().map(|s| s.order),
                self.quads_iter.peek().map(|q| q.order),
                self.backdrop_blur_rects_iter.peek().map(|b| b.order),
                self.effect_quads_iter.peek().map(|e| e.order),
                self.paths_iter.peek().map(|p| p.order),
                self.underlines_iter.peek().map(|u| u.order),
                self.monochrome_sprites_iter.peek().map(|s| s.order),
                self.subpixel_sprites_iter.peek().map(|s| s.order),
                self.polychrome_sprites_iter.peek().map(|s| s.order),
                self.surfaces_iter.peek().map(|s| s.order),
            ]
            .into_iter()
            .flatten()
            .min();
            // At equal orders the primitives go first, as `PrimitiveKind`
            // sorts the composite last.
            if lowest_primitive.is_none_or(|order| layer_order < order) {
                let start = self.vn_layers_start;
                let mut end = start + 1;
                self.vn_layers_iter.next();
                while self
                    .vn_layers_iter
                    .next_if(|layer| lowest_primitive.is_none_or(|order| layer.order < order))
                    .is_some()
                {
                    end += 1;
                }
                self.vn_layers_start = end;
                return Some(VnBatch::Layers(start..end));
            }
        }
        self.next_primitives(layer_order).map(VnBatch::Primitives)
    }
}

impl<'a> BatchIterator<'a> {
    /// The next run of one primitive kind, which ends before `layer_order`
    /// (the next composite's) as before any other kind's next order.
    fn next_primitives(&mut self, layer_order: Option<DrawOrder>) -> Option<PrimitiveBatch> {
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
            (
                self.paths_iter.peek().map(|path| path.order),
                PrimitiveKind::Path,
            ),
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
            (layer_order, PrimitiveKind::VnLayer),
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
                // breaking on `texture_id`. Its images break it too: a batch
                // binds one texture set, its first quad's.
                let first = self.effect_quads_iter.peek().unwrap();
                let (effect_id, textures) = (first.effect_id, first.texture_set());
                let effect_quads_start = self.effect_quads_start;
                let mut effect_quads_end = effect_quads_start + 1;
                self.effect_quads_iter.next();
                while self
                    .effect_quads_iter
                    .next_if(|effect_quad| {
                        (effect_quad.order, batch_kind) < max_order_and_kind
                            && effect_quad.effect_id == effect_id
                            && effect_quad.texture_set() == textures
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
            // `next` takes a composite that comes first.
            PrimitiveKind::VnLayer => None,
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
/// `gpui_windows/src/shaders.hlsl` (85 words / 340 bytes) — the DirectX
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
/// `crates/sk-effects`, which is above gpui in the dependency graph. What lives
/// here is only the numbering, because three separate places have to agree on
/// it: this enum, the pipeline table in `gpui_windows`, and the registry in
/// `sk-effects`.
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
    /// vue-native: not an effect a stylesheet can name. An [`super::EffectQuad`]
    /// with this id ERASES what was painted under it — its rounded rect, cut
    /// by its content mask, becomes transparent — so a surface composited
    /// under the window (an `<iframe>`'s webview) shows through there. See
    /// `Window::vn_paint_punch`. A renderer without it draws nothing.
    pub const PUNCH: u32 = u32::MAX - 1;
    /// vue-native: not an effect a stylesheet can name. Draws the content an
    /// [`super::VnEffectLayer`] reads as it is, at the quad's opacity: what
    /// shows of the element while its `[SkContent]` effect fades in or out,
    /// or when the effect cannot be drawn.
    pub const CONTENT: u32 = u32::MAX - 2;
    /// The id an [`super::EffectSpec`] carries when CSS said `--shading: none`.
    /// A refinement has no way to spell "explicitly nothing", so — exactly as
    /// `backdrop-filter: none` maps to a 0px radius — clearing is a *value*.
    pub const NONE: u32 = u32::MAX;
}

/// Bit 0 of [`EffectQuad::flags`]: this effect samples the backdrop, so the
/// renderer must snapshot the render target, run the blur pyramid, bind t0/t2
/// and draw with blending disabled.
pub const EFFECT_FLAG_NEEDS_BACKDROP: u32 = 1;

/// Bit 1 of [`EffectQuad::flags`]: the effect reads its element's painted
/// subtree (`[SkContent]`), so the element paints into an isolated group the
/// effect samples at t7 and composites in its place.
pub const EFFECT_FLAG_NEEDS_CONTENT: u32 = 2;

/// Bit 2 of [`EffectQuad::flags`]: the effect keeps its own state from frame
/// to frame (`[SkHistory]`), at t8. A renderer clears it on a quad it has no
/// history for, which then reads zeros.
pub const EFFECT_FLAG_HISTORY: u32 = 4;

/// Bit 3 of [`EffectQuad::flags`], set by the renderer only: this draw writes
/// the next history state instead of the element's pixels.
pub const EFFECT_FLAG_HISTORY_PASS: u32 = 8;

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
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
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
    /// Upper-right coefficient of the linear matrix before rotation:
    /// `R * [[scale.x, shear], [0, scale.y]]`. Unlike a tangent multiplied by
    /// scale.y this also represents finite rank-one/zero-scale matrices.
    #[serde(default)]
    pub shear: f32,
    /// Ordered transform lists can mix pixel translations with percentages of
    /// either box axis. When present, each row is `[pixels, width, height]`
    /// and replaces `translate` at paint time. Ordinary TRS lists need none.
    #[serde(default)]
    pub translation_basis: Option<[[f32; 3]; 2]>,
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
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
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
        shear: 0.0,
        translation_basis: None,
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
            && self.shear == 0.0
            && match self.translation_basis {
                Some(basis) => basis.iter().flatten().all(|v| *v == 0.0),
                None => self.translate.0.is_zero() && self.translate.1.is_zero(),
            }
    }

    fn translation_coefficients(&self) -> [[f32; 3]; 2] {
        self.translation_basis.unwrap_or_else(|| {
            let x = match self.translate.0 {
                TransformLength::Pixels(v) => [v, 0.0, 0.0],
                TransformLength::Fraction(v) => [0.0, v, 0.0],
            };
            let y = match self.translate.1 {
                TransformLength::Pixels(v) => [v, 0.0, 0.0],
                TransformLength::Fraction(v) => [0.0, 0.0, v],
            };
            [x, y]
        })
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
            shear: lerp(self.shear, other.shear),
            translation_basis: if self.translation_basis.is_some() || other.translation_basis.is_some() {
                let a = self.translation_coefficients();
                let b = other.translation_coefficients();
                Some(std::array::from_fn(|row| std::array::from_fn(|column| lerp(a[row][column], b[row][column]))))
            } else { None },
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
        let coefficients = self.translation_coefficients();
        let resolve = |row: [f32; 3]| px(row[0] + row[1] * bounds.size.width.0 + row[2] * bounds.size.height.0);
        let translate = Point::new(resolve(coefficients[0]), resolve(coefficients[1]));
        TransformationMatrix::unit()
            .translate((anchor + translate).scale(scale_factor))
            .rotate(Radians(self.rotate))
            .compose(TransformationMatrix {
                rotation_scale: [[self.scale.0, self.shear], [0.0, self.scale.1]],
                translation: [0.0, 0.0],
            })
            .translate(anchor.scale(-scale_factor))
    }
}

/// How many floats an effect's parameter block holds ([`EffectSpec::params`],
/// [`EffectQuad::params`]).
pub const EFFECT_PARAM_SLOTS: usize = 32;

/// The `--shading` value that came out of the cascade, carried on
/// [`crate::Style`] and painted by [`crate::Style::paint`].
///
/// The `params` are the effect's positional CSS arguments, already
/// converted to floats (a colour occupies four consecutive slots), plus the
/// slots the host fills at paint time. Everything else is schema metadata the
/// *painter* needs and gpui must not have to look up: which params are lengths
/// (so they scale with the device factor), which one drives the backdrop blur,
/// which one dilates the quad, and which ones the host writes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
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
    /// `Some(i)`: `params[i]` is a speed `time` is multiplied by, which the
    /// painter integrates per element so a change of it does not jump.
    #[serde(default)]
    pub animated_param: Option<u8>,
    /// `Some(i)`: the painter writes the pointer's position in the element's
    /// 0..1 box units into `params[i..i + 2]`.
    pub pointer_param: Option<u8>,
    /// `Some(i)`: the painter writes the pointer's last move, entry into the
    /// window and exit from it, on the effect clock, into `params[i..i + 3]`.
    pub pointer_times_param: Option<u8>,
    /// Bit *i* set means `params[i]` is a time on the effect clock; the
    /// element keeps repainting until `clock_window` seconds after the latest.
    pub clock_mask: u32,
    /// Seconds an effect keeps repainting after the times in `clock_mask`.
    pub clock_window: f32,
    /// The positional parameters, in schema order.
    pub params: [f32; EFFECT_PARAM_SLOTS],
    /// The `SkTexture` parameters by binding: `None` past the ones the effect
    /// declares.
    #[serde(default)]
    pub textures: [Option<crate::EffectTextureParam>; crate::EFFECT_TEXTURE_SLOTS],
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
        animated_param: None,
        pointer_param: None,
        pointer_times_param: None,
        clock_mask: 0,
        clock_window: 0.,
        params: [0.; EFFECT_PARAM_SLOTS],
        textures: [None; crate::EFFECT_TEXTURE_SLOTS],
    };

    /// Whether this spec actually paints anything.
    pub fn is_some(&self) -> bool {
        self.id != effect_id::NONE
    }

    /// Whether a transition can interpolate from `self` to `other` slot by
    /// slot: the same effect sampling the same images. Otherwise it
    /// cross-fades, since an image does not interpolate.
    pub fn interpolates_with(&self, other: &EffectSpec) -> bool {
        self.id == other.id && self.textures == other.textures
    }

    /// Whether the painter fills a pointer parameter, so the element repaints
    /// when the pointer moves.
    pub fn follows_pointer(&self) -> bool {
        self.pointer_param.is_some() || self.pointer_times_param.is_some()
    }

    /// Whether `now` (effect-clock seconds) is within `clock_window` of the
    /// latest time in a `clock_mask` slot.
    pub fn clock_running(&self, now: f32) -> bool {
        if self.clock_mask == 0 {
            return false;
        }
        self.params
            .iter()
            .enumerate()
            .filter(|(slot, _)| self.clock_mask & (1 << slot) != 0)
            .any(|(_, start)| now < start + self.clock_window)
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

/// A `--shading` transition between two different effects: `from` paints at
/// `1 - progress` beneath the style's own effect at `progress`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
pub struct EffectCrossFade {
    /// The effect being faded out.
    pub from: EffectSpec,
    /// 0 shows only `from`, 1 only the style's effect.
    pub progress: f32,
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
/// generated wrapper (`scripts/shaders.ts`).
/// 56 words / 224 bytes.
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
    /// The effect's parameter block, [`EFFECT_PARAM_SLOTS`] long (a literal for
    /// the Metal header cbindgen writes).
    pub params: [f32; 32],
    pub opacity: f32,
    /// Seconds on [`effect_clock_seconds`].
    pub time: f32,
    /// How far outside the element's border box this quad reaches.
    pub bleed: ScaledPixels,
    /// The blur radius applied to the backdrop before the effect sees it.
    /// Ignored without [`EFFECT_FLAG_NEEDS_BACKDROP`].
    pub backdrop_blur_radius: ScaledPixels,
    /// Device pixels per logical pixel.
    pub scale: f32,
    pub flags: u32,
}

impl EffectQuad {
    /// A parameter block holding `head` in its first slots and zeros after.
    pub fn params_from(head: &[f32]) -> [f32; EFFECT_PARAM_SLOTS] {
        let mut params = [0.; EFFECT_PARAM_SLOTS];
        params[..head.len()].copy_from_slice(head);
        params
    }

    /// Whether this quad reads the render-target snapshot, which decides both
    /// its draw ordering and whether the renderer has to break the pass.
    pub fn needs_backdrop(&self) -> bool {
        self.flags & EFFECT_FLAG_NEEDS_BACKDROP != 0
    }

    /// Whether this quad reads its element's content ([`EFFECT_FLAG_NEEDS_CONTENT`]).
    pub fn needs_content(&self) -> bool {
        self.flags & EFFECT_FLAG_NEEDS_CONTENT != 0
    }

    /// Whether this quad reads and steps a history ([`EFFECT_FLAG_HISTORY`]).
    pub fn has_history(&self) -> bool {
        self.flags & EFFECT_FLAG_HISTORY != 0
    }

    /// The device-pixel size of this quad's history texture.
    pub fn history_size(&self) -> (u32, u32) {
        let side = |value: f32| (value.round().max(1.)) as u32;
        (
            side(self.bounds.size.width.0),
            side(self.bounds.size.height.0),
        )
    }

    /// The copy a renderer draws into the history target to step it.
    pub fn history_pass(mut self) -> Self {
        self.flags = (self.flags & !EFFECT_FLAG_NEEDS_BACKDROP) | EFFECT_FLAG_HISTORY_PASS;
        self
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
        Primitive::EffectQuad(effect_quad, None)
    }
}

const _: () = assert!(std::mem::size_of::<EffectQuad>() == 224);
const _: () = assert!(EFFECT_PARAM_SLOTS == 32);

// The four primitives that carry a CSS `transform`. Their Rust layout IS the
// shader layout on every backend — the DirectX renderer uses
// `size_of::<T>()` as the structured buffer's `StructureByteStride`, and
// `gpui_wgpu/src/shaders_webgl.wgsl` hardcodes the same numbers as word
// strides. A field added without updating the shaders reads the next record's
// bytes and produces garbage rather than an error, so the sizes are asserted
// here where a mismatch is a compile failure.
//
// Word counts, for the shader side: 89 / 42 / 26 / 38. (vue-native fork: each
// grew by four, the content mask's corner radii; `PolychromeSprite` by four
// more, its `tint`, and `Shadow` by four more, its `element_bounds`.)
//
// `Quad` carries a `Background`, which is 57 words since it grew from a
// two-stop linear ramp to an eight-stop linear / radial / conic one with
// repeat and tiling (`color.rs`). `PathRasterizationVertex` carries one too
// and is asserted next to the WebGL word strides in `gpui_wgpu`.
const _: () = assert!(std::mem::size_of::<Quad>() == 89 * 4);
const _: () = assert!(std::mem::size_of::<Background>() == 57 * 4);
const _: () = assert!(std::mem::size_of::<Shadow>() == 42 * 4);
const _: () = assert!(std::mem::size_of::<Underline>() == 26 * 4);
const _: () = assert!(std::mem::size_of::<PolychromeSprite>() == 38 * 4);
const _: () = assert!(std::mem::size_of::<TransformationMatrix>() == 6 * 4);

static EFFECT_CLOCK_ORIGIN: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();

/// Seconds since the effect clock's origin — the clock [`EffectQuad::time`]
/// carries. The origin is [`set_effect_clock_origin`]'s, or the first read.
///
/// Deliberately process-wide and monotonic rather than per-window: an f32 has
/// ~7 significant digits, so anchoring at zero keeps sub-millisecond precision
/// for the first couple of hours instead of quantising to ~64 ms the way a
/// UNIX-epoch value would.
pub fn effect_clock_seconds() -> f32 {
    EFFECT_CLOCK_ORIGIN
        .get_or_init(std::time::Instant::now)
        .elapsed()
        .as_secs_f32()
}

/// Anchors the effect clock, so an embedder can make it share a zero with its
/// own (a JS `performance.now()`). Returns `false` once the clock has been read.
pub fn set_effect_clock_origin(origin: std::time::Instant) -> bool {
    EFFECT_CLOCK_ORIGIN.set(origin).is_ok()
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
    pub shape_bounds: Bounds<ScaledPixels>,
    pub shape_corner_radii: Corners<ScaledPixels>,
    /// vue-native fork: `SHADOW_OUTSIDE_BOX`, `SHADOW_INSET` or
    /// `SHADOW_UNDER_BOX` (gpui's window.rs).
    pub inset: u32,
    pub pad: u32, // align to 8 bytes
    /// See [`Quad::transformation`]. A shadow is part of the element's own
    /// painted content, so a `transform: rotate()` turns the shadow with the
    /// box — which is what a browser does.
    pub transformation: TransformationMatrix,
    /// vue-native fork: the element's border box, with `corner_radii`. An
    /// outer shadow is cut out of it; an inset one is clipped to it.
    pub element_bounds: Bounds<ScaledPixels>,
}

impl From<Shadow> for Primitive {
    fn from(shadow: Shadow) -> Self {
        Primitive::Shadow(shadow)
    }
}

/// The style of a border.
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[cfg_attr(feature = "schemars", derive(JsonSchema))]
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
    pub fn translate(self, point: Point<ScaledPixels>) -> Self {
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

    /// The smallest bounds that contain `bounds` once it is transformed.
    pub(crate) fn transform_bounds(&self, bounds: Bounds<ScaledPixels>) -> Bounds<ScaledPixels> {
        if *self == Self::unit() {
            return bounds;
        }
        let [[xx, xy], [yx, yy]] = self.rotation_scale;
        let [translate_x, translate_y] = self.translation;
        let corners = [
            bounds.origin,
            bounds.top_right(),
            bounds.bottom_left(),
            bounds.bottom_right(),
        ]
        .map(|corner| {
            Point::new(
                ScaledPixels(xx * corner.x.0 + xy * corner.y.0 + translate_x),
                ScaledPixels(yx * corner.x.0 + yy * corner.y.0 + translate_y),
            )
        });
        let [first, rest @ ..] = corners;
        let top_left = rest.iter().fold(first, |min, corner| min.min(corner));
        let bottom_right = rest.iter().fold(first, |max, corner| max.max(corner));
        Bounds::from_corners(top_left, bottom_right)
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
        // Zero, not "small": `scale(0.0003)` has a determinant under
        // `f32::EPSILON` and an inverse all the same, and an element part way
        // into a scale-in could not be hit.
        if det == 0.0 || !det.is_finite() || !(1.0 / det).is_finite() {
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

/// The axis-aligned bounding box of `bounds` after `matrix`, in the same
/// (device) space both are already expressed in.
///
/// Exact for a translation; a superset for a rotation or a scale, which is the
/// safe direction for a culling test.
pub(crate) fn transformed_bounds(
    bounds: Bounds<ScaledPixels>,
    matrix: TransformationMatrix,
) -> Bounds<ScaledPixels> {
    matrix.transform_bounds(bounds)
}

/// vue-native fork: `bounds` carried through `matrix` EXACTLY, for the
/// primitives that have no `transformation` of their own and are drawn as an
/// axis-aligned rect — the backdrop blur and the `--shading` effect quad, which
/// sample the render target under that rect. A translation and an axis-aligned
/// scale (how every popper is positioned) fold into the rect itself; the `f32`
/// is the factor the corner radii scale by. `None` under a rotation or a skew,
/// which no axis-aligned rect can express.
pub(crate) fn axis_aligned_through(
    bounds: Bounds<ScaledPixels>,
    matrix: TransformationMatrix,
) -> Option<(Bounds<ScaledPixels>, f32)> {
    if matrix.is_unit() {
        return Some((bounds, 1.0));
    }
    let [[a, b], [c, d]] = matrix.rotation_scale;
    const EPS: f32 = 1e-6;
    if b.abs() > EPS || c.abs() > EPS || a.abs() <= EPS || d.abs() <= EPS {
        return None;
    }
    Some((transformed_bounds(bounds, matrix), a.abs().min(d.abs())))
}

impl Default for TransformationMatrix {
    fn default() -> Self {
        Self::unit()
    }
}

#[cfg(test)]
mod vn_axis_aligned_tests {
    use super::*;
    use crate::{Radians, point, size};

    fn rect(x: f32, y: f32, w: f32, h: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(y)),
            size: size(ScaledPixels(w), ScaledPixels(h)),
        }
    }

    #[test]
    fn a_translate_moves_the_rect_and_a_rotation_has_no_rect() {
        let moved =
            TransformationMatrix::unit().translate(point(ScaledPixels(300.), ScaledPixels(120.)));
        assert_eq!(
            axis_aligned_through(rect(0., 0., 100., 50.), moved),
            Some((rect(300., 120., 100., 50.), 1.0))
        );
        // A popper's open animation: scale 0.95 about the centre.
        let zoomed = TransformationMatrix::unit()
            .translate(point(ScaledPixels(50.), ScaledPixels(25.)))
            .scale(size(0.5, 0.5))
            .translate(point(ScaledPixels(-50.), ScaledPixels(-25.)));
        assert_eq!(
            axis_aligned_through(rect(0., 0., 100., 50.), zoomed),
            Some((rect(25., 12.5, 50., 25.), 0.5))
        );
        let turned = TransformationMatrix::unit().rotate(Radians(0.3));
        assert_eq!(axis_aligned_through(rect(0., 0., 100., 50.), turned), None);
        assert_eq!(
            axis_aligned_through(rect(1., 2., 3., 4.), TransformationMatrix::unit()),
            Some((rect(1., 2., 3., 4.), 1.0))
        );
    }

    /// A very small scale still has an inverse; only a flat one has none.
    #[test]
    fn a_tiny_scale_is_still_invertible() {
        let tiny = TransformationMatrix { rotation_scale: [[0.0003, 0.0], [0.0, 0.0003]], translation: [5.0, 7.0] };
        let back = tiny.inverse().expect("an inverse");
        assert!((back.rotation_scale[0][0] - 1.0 / 0.0003).abs() < 1.0);
        let flat = TransformationMatrix { rotation_scale: [[1.0, 0.0], [0.0, 0.0]], translation: [0.0, 0.0] };
        assert!(flat.inverse().is_none());
    }

    /// A path has no transformation field, so its curves move: a chevron in a
    /// translated dialog is drawn where the dialog is.
    #[test]
    fn a_transform_moves_a_paths_curves() {
        let mut builder = crate::PathBuilder::fill();
        builder.move_to(point(Pixels(0.), Pixels(0.)));
        builder.line_to(point(Pixels(10.), Pixels(0.)));
        builder.curve_to(point(Pixels(10.), Pixels(20.)), point(Pixels(20.), Pixels(10.)));
        builder.close();
        let path = builder.build().unwrap();
        let moved =
            TransformationMatrix::unit().translate(point(ScaledPixels(300.), ScaledPixels(-120.)));
        let still = path.bin(2.0).bounds();
        assert_eq!(
            path.transformed(moved, 2.0).bin(2.0).bounds(),
            Bounds::new(still.origin + point(ScaledPixels(300.), ScaledPixels(-120.)), still.size)
        );
        // A quarter turn swaps the box's sides, with the curve split where it
        // turns back on the new axes.
        let turned = TransformationMatrix::unit().rotate(Radians(std::f32::consts::FRAC_PI_2));
        let turned = path.transformed(turned, 2.0).bin(2.0).bounds();
        assert!((turned.size.width.0 - still.size.height.0).abs() < 1e-3);
        assert!((turned.size.height.0 - still.size.width.0).abs() < 1e-3);
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct MonochromeSprite {
    pub order: DrawOrder,
    /// Skips the glyph contrast and gamma correction: the tile is linear coverage (a blurred shadow).
    pub raw_coverage: PaddedBool32,
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
    pub transformation: TransformationMatrix,
    /// vue-native fork: upstream keeps this in the padding word that
    /// `raw_coverage` holds here. Renderers ignore it.
    pub animation: SpriteAnimationId,
    pub pad: u32,
}

/// Refers to an animation of the scene a sprite is inserted into.
///
/// A single `u32` so it fits the sprite's existing padding and leaves the
/// instance layout unchanged on every backend; renderers ignore it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(transparent)]
pub struct SpriteAnimationId(u32);

impl SpriteAnimationId {
    /// No animation.
    pub const NONE: Self = Self(u32::MAX);
}

impl Default for SpriteAnimationId {
    fn default() -> Self {
        Self::NONE
    }
}

/// A [`TimedTransformation`] the window applies to a sprite right before
/// every present, so the scene can be presented again without being redrawn.
#[derive(Copy, Clone, Debug)]
pub(crate) struct SpriteAnimation {
    transformation: TimedTransformation,
    center: Point<Pixels>,
    scale_factor: f32,
    base_color: Hsla,
    sprite_index: usize,
    /// vue-native fork: the CSS `transform` the pose composes under.
    outer: TransformationMatrix,
}

impl SpriteAnimation {
    pub(crate) fn new(
        transformation: TimedTransformation,
        center: Point<Pixels>,
        scale_factor: f32,
    ) -> Self {
        Self {
            transformation,
            center,
            scale_factor,
            base_color: Hsla::default(),
            sprite_index: usize::MAX,
            outer: TransformationMatrix::unit(),
        }
    }

    /// vue-native fork: this animation, posed under the CSS `transform` `outer`.
    pub(crate) fn under(self, outer: TransformationMatrix) -> Self {
        Self { outer, ..self }
    }
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
    pub tile: AtlasTile,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Hsla,
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
    pub tile: AtlasTile,
    pub grayscale: PaddedBool32,
    pub opacity: f32,
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub corner_radii: Corners<ScaledPixels>,
    /// See [`Quad::transformation`]. The monochrome and subpixel sprites have
    /// carried one since gpui's SVG element; this is the one that was missing,
    /// and it is what makes an `<img>` (or a colour emoji) inside a rotated box
    /// rotate with it.
    ///
    /// vue-native fork: kept LAST. Upstream (zed#65814) moved `tile` up to
    /// offset 8 so the texture sort key sits beside `order`; every shader's
    /// `PolychromeSprite` mirrors this order — tile, grayscale, opacity,
    /// bounds, content_mask, corner_radii, transformation, tint.
    pub transformation: TransformationMatrix,
    /// vue-native fork: the colour the sprite's pixels take by its alpha,
    /// keeping their own alpha (`Window::with_image_tint`); transparent for
    /// none. After `transformation`, so every shader mirrors it last.
    pub tint: Hsla,
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
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    pub image_buffer: objc2_core_foundation::CFRetained<objc2_core_video::CVPixelBuffer>,
}

impl From<PaintSurface> for Primitive {
    fn from(surface: PaintSurface) -> Self {
        Primitive::Surface(surface)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathCurve {
    pub p0: Point<f32>,
    pub p1: Point<f32>,
    pub a: Point<f32>,
    pub b: Point<f32>,
    pub sign: f32,
    pub pad: u32, // align to 8 bytes
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathPaint {
    /// The path's bounds, in device pixels.
    pub bounds: Bounds<ScaledPixels>,
    pub content_mask: ContentMask<ScaledPixels>,
    pub color: Background,
    pub even_odd: PaddedBool32,
    pub pad: u32, // align to 8 bytes
}

#[derive(Clone, Copy, Debug, PartialEq)]
#[repr(C)]
#[expect(missing_docs)]
pub struct PathTile {
    pub paint: u32,
    pub curve_start: u32,
    pub curve_count: u32,
    pub backdrop: i32,
    pub top_left: Point<f32>,
    pub bottom_right: Point<f32>,
    pub reference: Point<f32>,
    pub curves_top_left: Point<f32>,
    pub curves_bottom_right: Point<f32>,
}

#[derive(Clone, Debug)]
#[expect(missing_docs)]
pub struct PaintedPath {
    order: DrawOrder,
    bounds: Bounds<ScaledPixels>,
    tiles: Rc<PathTiles>,
    paint: PathPaint,
}

#[expect(missing_docs)]
impl PaintedPath {
    /// Bins `path` into tiles, to be filled with `color` within `content_mask`.
    pub fn new(
        path: Path,
        scale_factor: f32,
        content_mask: ContentMask<Pixels>,
        color: Background,
    ) -> Self {
        let content_mask = content_mask.scale(scale_factor);
        let mut tiles = path.bin(scale_factor);
        tiles.retain_visible(content_mask.bounds);
        let paint = PathPaint {
            bounds: tiles.bounds(),
            content_mask,
            color,
            even_odd: path.even_odd().into(),
            pad: 0,
        };
        Self {
            order: 0,
            bounds: tiles.painted_bounds(),
            tiles: Rc::new(tiles),
            paint,
        }
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.tiles().len()
    }

    pub fn append_to(
        &self,
        paints: &mut Vec<PathPaint>,
        curves: &mut Vec<PathCurve>,
        tiles: &mut Vec<PathTile>,
    ) {
        let paint_index = paints.len() as u32;
        let curve_start = curves.len() as u32;
        paints.push(self.paint);
        curves.extend_from_slice(self.tiles.curves());
        tiles.extend(self.tiles.tiles().iter().map(|tile| PathTile {
            paint: paint_index,
            curve_start: curve_start + tile.curve_start,
            ..*tile
        }));
    }
}

impl From<PaintedPath> for Primitive {
    fn from(path: PaintedPath) -> Self {
        Primitive::Path(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{point, size};

    fn square(x: f32) -> Bounds<ScaledPixels> {
        Bounds {
            origin: point(ScaledPixels(x), ScaledPixels(0.)),
            size: size(ScaledPixels(10.), ScaledPixels(10.)),
        }
    }

    fn quad(x: f32) -> Quad {
        Quad {
            bounds: square(x),
            content_mask: ContentMask::rect(square(x)),
            ..Default::default()
        }
    }

    fn underline(x: f32) -> Underline {
        Underline {
            order: 0,
            pad: 0,
            bounds: square(x),
            content_mask: ContentMask::rect(square(x)),
            color: Hsla::default(),
            thickness: ScaledPixels(1.),
            wavy: false.into(),
        }
    }

    fn batch_labels(scene: &mut Scene) -> Vec<String> {
        scene.finish();
        scene.batches().map(|batch| batch.label()).collect()
    }

    #[test]
    fn layers_are_drawn_in_paint_order() {
        // A quad painted after a layer goes above it.
        let mut scene = Scene::default();
        scene.insert_primitive(quad(0.));
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.insert_primitive(underline(5.));
        scene.pop_layer();
        scene.insert_primitive(quad(5.));
        assert_eq!(
            batch_labels(&mut scene),
            ["quads (1)", "underlines (2)", "quads (1)"]
        );

        // A layer goes above the layer painted before it.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.pop_layer();
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (1)", "quads (1)"]);
    }

    #[test]
    fn nested_layers_are_drawn_above_their_parents() {
        // A layer has one draw order, so what it paints after a nested layer
        // still goes below that layer.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        scene.insert_primitive(underline(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (2)", "quads (1)"]);

        // The same holds when the nested layer shares its parent's order.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.insert_primitive(quad(0.));
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        scene.insert_primitive(quad(8.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["quads (3)"]);
        assert_eq!(
            scene
                .quads
                .iter()
                .map(|quad| quad.bounds)
                .collect::<Vec<_>>(),
            [square(0.), square(8.), square(5.)]
        );

        // A layer goes above the layers nested in the layer before it.
        let mut scene = Scene::default();
        scene.push_layer();
        scene.push_layer();
        scene.insert_primitive(underline(0.));
        scene.pop_layer();
        scene.pop_layer();
        scene.push_layer();
        scene.insert_primitive(quad(5.));
        scene.pop_layer();
        assert_eq!(batch_labels(&mut scene), ["underlines (1)", "quads (1)"]);
    }
}


#[cfg(test)]
mod vn_color_group_tests {
    use super::*;
    use crate::{VnColorMatrix, VnClipGeometry, VnFillRule, size};
    use std::sync::Arc;

    fn rectangle() -> Bounds<ScaledPixels> {
        Bounds { origin: point(ScaledPixels(0.), ScaledPixels(0.)), size: size(ScaledPixels(100.), ScaledPixels(100.)) }
    }

    fn filter(id: u64) -> VnIsolation {
        VnIsolation {
            text_paint: None,
            geometry: Arc::new(VnClipGeometry {
                fill_rule: VnFillRule::NonZero,
                segments: vec![[0., 0., 100., 0.], [100., 0., 100., 100.], [100., 100., 0., 100.], [0., 100., 0., 0.]],
                bounds: rectangle(),
            }),
            color_matrix: Some(VnColorMatrix { group_id: id, values: VnColorMatrix::IDENTITY, linear_rgb: true }),
            content_mask: ContentMask { bounds: rectangle(), ..Default::default() },
            capture: None,
        }
    }

    fn paint(scene: &mut Scene) {
        scene.insert_primitive(Quad { bounds: rectangle(), content_mask: ContentMask { bounds: rectangle(), ..Default::default() }, ..Default::default() });
    }

    fn text_mask(id: u64) -> VnIsolation {
        let mut isolation = filter(id);
        isolation.color_matrix = None;
        isolation.text_paint = Some(Arc::new(crate::VnTextPaint {
            fill: crate::VnTextFill { group_id: id, backgrounds: vec![crate::red().into(), crate::blue().into()] },
            bounds: [10., 20., 40., 30.],
            inverse: [0., 1., -1., 0., 50., 60.], scale_factor: 1.5,
            geometry: isolation.geometry.clone(), content_mask: isolation.content_mask,
        }));
        isolation
    }

    #[test]
    fn text_glyphs_and_deferred_replay_share_one_gradient_stack_and_source() {
        let mut scene = Scene::default();
        for _ in 0..3 {
            scene.vn_push_isolation(text_mask(7));
            paint(&mut scene);
            scene.vn_pop_isolation();
        }
        assert_eq!(scene.vn_layers.len(), 1, "a text scope composites once, not per glyph");
        assert_eq!(scene.vn_groups[0].quads.len(), 3);
        assert_eq!(scene.vn_text_backgrounds.len(), 2, "metadata belongs to a text scope, not each glyph");
        assert_eq!(scene.vn_layers[0].pad, [0, 0, 2]);
        let brush = scene.vn_text_backgrounds[0];
        assert_eq!(brush.bounds.origin.x.0, 10.);
        assert_eq!(brush.inverse.apply(point(px(2.), px(3.))), point(px(53.), px(58.)));
        assert_eq!(brush.scale_factor, 1.5);
        let operations = 0..scene.len();
        scene.finish();
        let mut replay = Scene::default();
        replay.replay(operations, &scene);
        assert_eq!(replay.vn_layers.len(), 1);
        assert_eq!(replay.vn_text_backgrounds.len(), 2, "cached operations rebuild valid metadata indices");
        assert_eq!(replay.vn_groups[0].quads.len(), 3);
        replay.clear();
        assert!(replay.vn_text_backgrounds.is_empty());
        replay.vn_push_isolation(text_mask(7));
        paint(&mut replay);
        replay.vn_pop_isolation();
        assert_eq!(replay.vn_text_backgrounds.len(), 2);
        assert_eq!(replay.vn_groups[0].quads.len(), 1, "new frames cannot retain old mask coverage");
    }

    #[test]
    fn text_and_color_filter_group_identities_do_not_alias() {
        let mut scene = Scene::default();
        scene.vn_push_isolation(filter(7));
        scene.vn_push_isolation(text_mask(7));
        paint(&mut scene);
        scene.vn_pop_isolation();
        scene.vn_pop_isolation();
        assert_eq!(scene.vn_groups.len(), 2);
        assert_eq!(scene.vn_groups[1].vn_layers.len(), 1);
        assert_eq!(scene.vn_layers[0].pad[0], 1);
        assert_eq!(scene.vn_groups[1].vn_layers[0].pad[2], 2);
        assert_eq!(std::mem::size_of::<VnLayer>(), 224, "keep the HLSL layer ABI");
        assert_eq!(std::mem::size_of::<VnTextBackground>(), 276, "match the HLSL structured buffer stride");
        assert_eq!(std::mem::offset_of!(VnTextBackground, bounds), 228);
        assert_eq!(std::mem::offset_of!(VnTextBackground, inverse), 244);
        assert_eq!(std::mem::offset_of!(VnTextBackground, scale_factor), 268);
    }

    #[test]
    fn text_sources_partition_child_clip_and_filter_contexts_and_rejoin_deferred_siblings() {
        for matrix in [false, true] {
            let mut first = filter(11);
            let mut second = filter(12);
            if !matrix { first.color_matrix = None; second.color_matrix = None; }
            let mut scene = Scene::default();
            for parent in [first.clone(), second, first] {
                scene.vn_push_isolation(parent);
                scene.vn_push_isolation(text_mask(7));
                paint(&mut scene);
                scene.vn_pop_isolation();
                scene.vn_pop_isolation();
            }
            // Clip-path groups themselves may close separately; their text
            // sources must still partition by stable parent geometry identity.
            let masks: Vec<_> = scene.vn_groups.iter().filter(|group| !group.quads.is_empty()).collect();
            assert_eq!(masks.len(), 2, "text cannot be captured by the first child's clip or filter");
            assert_eq!(masks[0].quads.len(), 2, "deferred text in the same parent reuses its source");
            assert_eq!(masks[1].quads.len(), 1);
            assert_eq!(scene.vn_text_backgrounds.len(), 4, "two brushes per distinct isolation context");
            let operations = 0..scene.len();
            scene.finish();
            let mut replay = Scene::default();
            replay.replay(operations, &scene);
            assert_eq!(replay.vn_text_backgrounds.len(), 4, "paint-cache replay interns equivalent parent contexts");
            let masks: Vec<_> = replay.vn_groups.iter().filter(|group| !group.quads.is_empty()).collect();
            assert_eq!(masks.len(), 2);
            assert_eq!(masks[0].quads.len(), 2);
        }
    }

    #[test]
    fn deferred_source_is_composited_with_its_background_before_one_color_matrix() {
        let mut scene = Scene::default();
        scene.vn_push_isolation(filter(1));
        paint(&mut scene);
        scene.vn_pop_isolation();
        scene.vn_push_isolation(filter(1));
        paint(&mut scene);
        scene.vn_pop_isolation();
        assert_eq!(scene.vn_layers.len(), 1, "one composite, not separately filtered translucent pieces");
        assert_eq!(scene.vn_groups.len(), 1);
        assert_eq!(scene.vn_groups[0].quads.len(), 2);
        assert!(scene.vn_groups[0].quads[1].order > scene.vn_groups[0].quads[0].order, "deferred source blends over its earlier background");
        let operations = 0..scene.len();
        scene.finish();
        assert_eq!(scene.vn_group_count(), 2);
        let mut replay = Scene::default();
        replay.replay(operations, &scene);
        assert_eq!(replay.vn_layers.len(), 1, "cached operations rebuild the same keyed source");
        assert_eq!(replay.vn_groups[0].quads.len(), 2);
        replay.clear();
        replay.vn_push_isolation(filter(1));
        paint(&mut replay);
        replay.vn_pop_isolation();
        assert_eq!(replay.vn_groups[0].quads.len(), 1, "a new frame has no old source pixels");
    }

    #[test]
    fn an_initially_empty_filter_keeps_its_composite_for_deferred_descendants() {
        let mut scene = Scene::default();
        scene.vn_push_isolation(filter(1));
        scene.vn_pop_isolation();
        scene.vn_push_isolation(filter(1));
        paint(&mut scene);
        scene.vn_pop_isolation();
        assert_eq!(scene.vn_layers.len(), 1);
        assert_eq!(scene.vn_groups.len(), 1);
        assert_eq!(scene.vn_groups[0].quads.len(), 1);
    }

    #[test]
    fn nested_filtered_sources_reopen_without_duplicate_inner_composites() {
        let mut scene = Scene::default();
        scene.vn_push_isolation(filter(1));
        paint(&mut scene);
        scene.vn_push_isolation(filter(2));
        paint(&mut scene);
        scene.vn_pop_isolation();
        scene.vn_pop_isolation();
        scene.vn_push_isolation(filter(1));
        scene.vn_push_isolation(filter(2));
        paint(&mut scene);
        scene.vn_pop_isolation();
        scene.vn_pop_isolation();
        assert_eq!(scene.vn_layers.len(), 1, "only the outer composite belongs to the page");
        assert_eq!(scene.vn_groups.len(), 2);
        assert_eq!(scene.vn_groups[0].quads.len(), 2, "both inner source pieces share one filter");
        assert_eq!(scene.vn_groups[1].vn_layers.len(), 1, "the inner filter composites once into the outer source");
        scene.finish();
        assert_eq!(scene.vn_group_count(), 3);
        assert_eq!(scene.vn_layers.len(), 2);
    }

    #[test]
    fn an_in_place_snapshot_samples_through_the_captured_transform() {
        let mut capture = filter(0);
        capture.color_matrix = None;
        capture.capture = Some(VnCaptureInfo {
            id: 3,
            border_box: rectangle(),
            transform: crate::TransformationMatrix { rotation_scale: [[2., 0.], [0., 2.]], translation: [100., 5.] },
        });
        let mut scene = Scene::default();
        scene.vn_push_isolation(capture);
        paint(&mut scene);
        scene.vn_pop_isolation();
        let viewport = Bounds { origin: point(ScaledPixels(0.), ScaledPixels(0.)), size: size(ScaledPixels(400.), ScaledPixels(400.)) };
        scene.vn_insert_snapshot(VnSnapshotPaint {
            id: 3,
            to_unit: [0.; 6],
            bounds: viewport,
            content_mask: ContentMask { bounds: viewport, ..Default::default() },
            opacity: 1.,
            plus_lighter: false,
            in_place: true,
        });
        scene.finish();
        scene.vn_resolve_snapshots(|_| None);
        let layer = scene.vn_layers.iter().find(|l| l.snapshot_id() == Some(3)).copied().unwrap();
        assert_eq!(layer.flags, vn_layer_flags::SNAPSHOT);
        assert_eq!(layer.source_transform, [0.5, 0., -50., 0., 0., 0.5, -2.5, 0.]);
        assert_eq!(layer.bounds.origin, point(ScaledPixels(100.), ScaledPixels(5.)));
        assert_eq!(layer.bounds.size, size(ScaledPixels(200.), ScaledPixels(200.)));
    }
}
