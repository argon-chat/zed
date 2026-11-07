use std::{
    hash::{Hash, Hasher},
    iter, mem,
    ops::Range,
    sync::Arc,
};

use crate::{
    AbsoluteLength, App, Background, BackgroundTag, BorderStyle, Bounds, ContentMask, Corners,
    CornersRefinement, CursorStyle, DefiniteLength, DevicePixels, Edges, EdgesRefinement,
    EffectSpec, Font,
    FontFallbacks, FontFeatures, FontStyle, FontWeight, GridLocation, Hsla, Length, Pixels, Point,
    PointRefinement, Rgba, SharedString, Size, SizeRefinement, Styled, TextRun, Window, black, phi,
    TransformSpec, TransformationMatrix, VnClipOutline, VnFillRule, VnPathVerb, point, px, quad,
    rems, size,
};
use collections::HashSet;
use refineable::Refineable;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Use this struct for interfacing with the 'debug_below' styling from your own elements.
/// If a parent element has this style set on it, then this struct will be set as a global in
/// GPUI.
#[cfg(debug_assertions)]
pub struct DebugBelow;

#[cfg(debug_assertions)]
impl crate::Global for DebugBelow {}

/// How to fit the image into the bounds of the element.
pub enum ObjectFit {
    /// The image will be stretched to fill the bounds of the element.
    Fill,
    /// The image will be scaled to fit within the bounds of the element.
    Contain,
    /// The image will be scaled to cover the bounds of the element.
    Cover,
    /// The image will be scaled down to fit within the bounds of the element.
    ScaleDown,
    /// The image will maintain its original size.
    None,
}

impl ObjectFit {
    /// Get the bounds of the image within the given bounds.
    pub fn get_bounds(
        &self,
        bounds: Bounds<Pixels>,
        image_size: Size<DevicePixels>,
    ) -> Bounds<Pixels> {
        let image_size = image_size.map(|dimension| Pixels::from(u32::from(dimension)));
        let image_ratio = image_size.width / image_size.height;
        let bounds_ratio = bounds.size.width / bounds.size.height;

        match self {
            ObjectFit::Fill => bounds,
            ObjectFit::Contain => {
                let new_size = if bounds_ratio > image_ratio {
                    size(
                        image_size.width * (bounds.size.height / image_size.height),
                        bounds.size.height,
                    )
                } else {
                    size(
                        bounds.size.width,
                        image_size.height * (bounds.size.width / image_size.width),
                    )
                };

                Bounds {
                    origin: point(
                        bounds.origin.x + (bounds.size.width - new_size.width) / 2.0,
                        bounds.origin.y + (bounds.size.height - new_size.height) / 2.0,
                    ),
                    size: new_size,
                }
            }
            ObjectFit::ScaleDown => {
                // Check if the image is larger than the bounds in either dimension.
                if image_size.width > bounds.size.width || image_size.height > bounds.size.height {
                    // If the image is larger, use the same logic as Contain to scale it down.
                    let new_size = if bounds_ratio > image_ratio {
                        size(
                            image_size.width * (bounds.size.height / image_size.height),
                            bounds.size.height,
                        )
                    } else {
                        size(
                            bounds.size.width,
                            image_size.height * (bounds.size.width / image_size.width),
                        )
                    };

                    Bounds {
                        origin: point(
                            bounds.origin.x + (bounds.size.width - new_size.width) / 2.0,
                            bounds.origin.y + (bounds.size.height - new_size.height) / 2.0,
                        ),
                        size: new_size,
                    }
                } else {
                    // If the image is smaller than or equal to the container, display it at its original size,
                    // centered within the container.
                    let original_size = size(image_size.width, image_size.height);
                    Bounds {
                        origin: point(
                            bounds.origin.x + (bounds.size.width - original_size.width) / 2.0,
                            bounds.origin.y + (bounds.size.height - original_size.height) / 2.0,
                        ),
                        size: original_size,
                    }
                }
            }
            ObjectFit::Cover => {
                let new_size = if bounds_ratio > image_ratio {
                    size(
                        bounds.size.width,
                        image_size.height * (bounds.size.width / image_size.width),
                    )
                } else {
                    size(
                        image_size.width * (bounds.size.height / image_size.height),
                        bounds.size.height,
                    )
                };

                Bounds {
                    origin: point(
                        bounds.origin.x + (bounds.size.width - new_size.width) / 2.0,
                        bounds.origin.y + (bounds.size.height - new_size.height) / 2.0,
                    ),
                    size: new_size,
                }
            }
            ObjectFit::None => Bounds {
                origin: bounds.origin,
                size: image_size,
            },
        }
    }
}

/// The minimum size of a column or row in a grid layout
#[derive(
    Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug, Default, JsonSchema, Serialize, Deserialize,
)]
pub enum GridTemplateMinSize {
    /// The column or row size may be 0
    #[default]
    Zero,
    /// The column or row size can be determined by the min content
    MinContent,
    /// The column or row size can be determined by the max content
    MaxContent,
}

/// A simplified representation of the grid-template-* value
#[derive(
    Copy,
    Clone,
    Refineable,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Debug,
    Default,
    JsonSchema,
    Serialize,
    Deserialize,
)]
pub struct GridTemplate {
    /// How this template directive should be repeated
    pub repeat: u16,
    /// The minimum size in the repeat(<>, minmax(_, 1fr)) equation
    pub min_size: GridTemplateMinSize,
}

/// One end of a CSS grid track's sizing function — the `a` or the `b` in
/// `minmax(a, b)`.
///
/// This is the full set taffy can express for a *definite* track. `fit-content()`
/// and `calc()` are absent: the first has no caller and the second is resolved
/// before it reaches here.
#[derive(Copy, Clone, PartialEq, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub enum GridTrackSize {
    /// `auto`
    #[default]
    Auto,
    /// `min-content`
    MinContent,
    /// `max-content`
    MaxContent,
    /// An absolute length, in logical pixels.
    Px(f32),
    /// A percentage of the container's track-axis size, as a FRACTION (`50%`
    /// is `0.5`) — taffy's unit, not CSS's.
    Percent(f32),
    /// `<flex>` — `1fr` is `Fr(1.0)`. Only valid as the MAX of a track; a `fr`
    /// minimum is invalid CSS and is normalised to `Auto` by the caller.
    Fr(f32),
}

/// One track in a `grid-template-columns` / `grid-template-rows` list.
///
/// Always stored in its `minmax()` normal form, which is what CSS says every
/// track sizing function means: a bare `1fr` is `minmax(auto, 1fr)`, a bare
/// `200px` is `minmax(200px, 200px)`.
#[derive(Copy, Clone, PartialEq, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub struct GridTrack {
    /// The track's minimum size.
    pub min: GridTrackSize,
    /// The track's maximum size.
    pub max: GridTrackSize,
}

impl GridTrack {
    /// `minmax(size, size)` — the normal form of a bare, definite track.
    pub fn fixed(size: GridTrackSize) -> Self {
        Self {
            min: size,
            max: size,
        }
    }

    /// `minmax(auto, <flex>)` — the normal form of a bare `<flex>` track.
    pub fn fr(flex: f32) -> Self {
        Self {
            min: GridTrackSize::Auto,
            max: GridTrackSize::Fr(flex),
        }
    }
}

/// vue-native fork: CSS `repeat(auto-fill | auto-fit, <tracks>)` inside a
/// track list — as many repetitions as fit the container, which taffy counts
/// at layout time. The repeated tracks are a run of the track list it belongs
/// to ([`Style::vn_grid_cols`] / [`Style::vn_grid_rows`]); CSS allows one such
/// repetition per list, and every other track in that list must then have a
/// fixed size, which the caller checks.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Serialize, Deserialize, JsonSchema)]
pub struct GridAutoRepeat {
    /// `auto-fit` (the empty repetitions collapse) rather than `auto-fill`
    /// (they keep their size).
    pub fit: bool,
    /// The index in the track list of the first repeated track.
    pub start: u16,
    /// How many tracks of the list one repetition holds.
    pub len: u16,
}

/// CSS `grid-auto-flow`: which axis the auto-placement algorithm fills, and
/// whether it backfills holes.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default, Serialize, Deserialize, JsonSchema)]
pub enum GridAutoFlow {
    /// Fill each row in turn, adding rows as needed.
    #[default]
    Row,
    /// Fill each column in turn, adding columns as needed.
    Column,
    /// `row dense`
    RowDense,
    /// `column dense`
    ColumnDense,
}

/// CSS `outline`: a ring painted just outside the border box, taking no layout
/// space. See [`Style::vn_outline`].
#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize, JsonSchema)]
pub struct Outline {
    /// `outline-width`. Zero paints nothing.
    pub width: Pixels,
    /// `outline-offset` — the gap between the border box and the ring's INNER
    /// edge. May be negative, which pulls the ring inside the box.
    pub offset: Pixels,
    /// `outline-color`.
    pub color: Hsla,
    /// `outline-style`, reduced to what gpui's quad can draw.
    pub style: BorderStyle,
}

/// vue-native fork: one CSS `border-*-radius` corner, as its two radii. Each
/// is a length, a percentage of the box (`Fraction`), or a linear `calc()`
/// with one. See [`Style::vn_corner_radii`].
#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize, JsonSchema)]
pub struct VnCornerRadius {
    /// The horizontal radius; a percentage is of the box's width.
    pub x: DefiniteLength,
    /// The vertical radius; a percentage is of the box's height.
    pub y: DefiniteLength,
}

static VN_ELLIPSE_DRAWN_AS_CIRCLE: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// vue-native fork: whether a corner was painted with a circle in place of
/// the ellipse CSS asks for, since the last call. vn-tree reads it to advise
/// once.
pub fn vn_take_ellipse_drawn_as_circle() -> bool {
    VN_ELLIPSE_DRAWN_AS_CIRCLE.swap(false, std::sync::atomic::Ordering::Relaxed)
}

/// vue-native fork: a box's painted corner radii, as CSS resolves them.
///
/// - A corner with a `relative` radius resolves it against `size`: the
///   horizontal radius against the width, the vertical against the height.
///   Any other corner takes its `lengths` radius on both axes.
/// - CSS's overlap rule (css-backgrounds-3 §5.5) applies: when the radii
///   along a side sum past its length, every radius is scaled by the
///   smallest such ratio. That is what makes Tailwind's `rounded-full`
///   (9999px) a pill.
/// - gpui's corners are circles, so a corner whose two radii still differ
///   takes the smaller one. `50%` on a box that is not square comes out as a
///   pill where a browser draws an ellipse, and
///   [`vn_take_ellipse_drawn_as_circle`] reports it.
/// - Each radius is finally clamped to half the shorter side, which the
///   shaders need because they pick a corner by quadrant. CSS lets a lone
///   large corner reach past the middle of the box.
pub fn vn_resolve_corner_radii(
    lengths: &Corners<AbsoluteLength>,
    relative: &Corners<Option<VnCornerRadius>>,
    size: Size<Pixels>,
    rem_size: Pixels,
) -> Corners<Pixels> {
    let (width, height) = (f32::from(size.width).max(0.), f32::from(size.height).max(0.));
    let pair = |length: &AbsoluteLength, relative: &Option<VnCornerRadius>| -> (f32, f32) {
        match relative {
            Some(radius) => (
                f32::from(radius.x.to_pixels(px(width).into(), rem_size)).max(0.),
                f32::from(radius.y.to_pixels(px(height).into(), rem_size)).max(0.),
            ),
            None => {
                let radius = f32::from(length.to_pixels(rem_size)).max(0.);
                (radius, radius)
            }
        }
    };
    let top_left = pair(&lengths.top_left, &relative.top_left);
    let top_right = pair(&lengths.top_right, &relative.top_right);
    let bottom_right = pair(&lengths.bottom_right, &relative.bottom_right);
    let bottom_left = pair(&lengths.bottom_left, &relative.bottom_left);

    let ratio = |side: f32, a: f32, b: f32| if a + b > side { side / (a + b) } else { 1. };
    let factor = ratio(width, top_left.0, top_right.0)
        .min(ratio(width, bottom_left.0, bottom_right.0))
        .min(ratio(height, top_left.1, bottom_left.1))
        .min(ratio(height, top_right.1, bottom_right.1));

    let circle = |(x, y): (f32, f32)| {
        let (x, y) = (x * factor, y * factor);
        let radius = x.min(y);
        if radius > 0. && (x - y).abs() > 0.5 {
            VN_ELLIPSE_DRAWN_AS_CIRCLE.store(true, std::sync::atomic::Ordering::Relaxed);
        }
        px(radius)
    };
    Corners {
        top_left: circle(top_left),
        top_right: circle(top_right),
        bottom_right: circle(bottom_right),
        bottom_left: circle(bottom_left),
    }
    .clamp_radii_for_quad_size(size)
}

/// vue-native fork: the box a `clip-path` is laid against (CSS Masking 1
/// §5.1, `<geometry-box>`). `fill-box` is the content box and `stroke-box` and
/// `view-box` the border box, as for any element with a CSS box.
#[derive(Copy, Clone, Default, PartialEq, Debug, Serialize, Deserialize, JsonSchema)]
pub enum VnClipBox {
    /// The margin box.
    Margin,
    /// The border box, the default.
    #[default]
    Border,
    /// The padding box.
    Padding,
    /// The content box.
    Content,
}

/// vue-native fork: a `<shape-radius>` of `circle()` or `ellipse()`.
#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize, JsonSchema)]
pub enum VnShapeRadius {
    /// A length. A percentage is of the reference box's width for an
    /// ellipse's horizontal radius, of its height for the vertical one, and
    /// of `sqrt(width² + height²) / sqrt(2)` for a circle's.
    Length(DefiniteLength),
    /// The distance from the centre to the reference box's closest side
    /// (in the radius's own axis, for an ellipse).
    ClosestSide,
    /// The distance to the farthest side.
    FarthestSide,
}

/// vue-native fork: CSS `clip-path` (CSS Masking 1 §5). The shapes a rounded
/// content mask can carry are one ([`Style::vn_clip_mask`]); `polygon()`,
/// `path()` and an ellipse that is no circle are an isolated layer composited
/// through the shape ([`Style::vn_clip_outline`], [`crate::VnClipGeometry`]).
#[derive(Clone, Default, PartialEq, Debug, Serialize, Deserialize, JsonSchema)]
pub enum VnClipPath {
    /// `none`: no clip.
    #[default]
    None,
    /// A `<geometry-box>` alone: that box, rounded as `border-radius` rounds
    /// it.
    Box(VnClipBox),
    /// `inset()`: the reference box less `insets`, with `radii` (`round`).
    /// A percentage inset is of the box's width (left and right) or height
    /// (top and bottom), and a percentage radius of the box's size, as
    /// Chromium resolves them.
    Inset {
        /// The box the insets are taken from.
        reference: VnClipBox,
        /// The insets: top, right, bottom, left.
        insets: [DefiniteLength; 4],
        /// The `round` radii: top left, top right, bottom right, bottom
        /// left. Zero for a square corner.
        radii: [VnCornerRadius; 4],
    },
    /// `circle()`, or `ellipse()`: a centre in the reference box and two
    /// radii. A circle is a rounded mask; an ellipse whose radii differ is
    /// its outline, four cubics, in an isolated layer. Both are exact.
    Ellipse {
        /// The box the centre and the radii are resolved against.
        reference: VnClipBox,
        /// The centre, from the reference box's top left corner: x, y.
        center: [DefiniteLength; 2],
        /// The horizontal and vertical radii; the same for a circle.
        radius: [VnShapeRadius; 2],
        /// `circle()`: a percentage radius is of the box's diagonal over
        /// `sqrt(2)`, and the closest or farthest side is of all four.
        circle: bool,
    },
    /// `polygon()`: its vertices, x and y from the reference box's top left
    /// corner, a percentage of its width and height.
    Polygon {
        /// The box the vertices are laid against.
        reference: VnClipBox,
        /// `nonzero` or `evenodd`.
        fill_rule: VnFillRule,
        /// The vertices, in order.
        points: Arc<[[DefiniteLength; 2]]>,
    },
    /// `path()`: SVG path data, in pixels from the reference box's top left
    /// corner (CSS Shapes 1 §3.1).
    Path {
        /// The box the path is laid against.
        reference: VnClipBox,
        /// `nonzero` or `evenodd`.
        fill_rule: VnFillRule,
        /// The path, absolute ([`VnPathVerb`]).
        verbs: Arc<[VnPathVerb]>,
    },
}

/// vue-native fork: a `circle()` or `ellipse()` resolved against its reference
/// box `rect`: the centre and the two radii (CSS Shapes 1 §3.1).
fn vn_clip_ellipse(
    rect: Bounds<Pixels>,
    center: &[DefiniteLength; 2],
    radius: &[VnShapeRadius; 2],
    circle: bool,
    rem_size: Pixels,
) -> (Point<Pixels>, Pixels, Pixels) {
    let (w, h) = (rect.size.width, rect.size.height);
    let resolve = |length: &DefiniteLength, base: Pixels| length.to_pixels(AbsoluteLength::Pixels(base), rem_size);
    let cx = rect.left() + resolve(&center[0], w);
    let cy = rect.top() + resolve(&center[1], h);
    let (near_x, far_x) = ((cx - rect.left()).abs(), (rect.right() - cx).abs());
    let (near_y, far_y) = ((cy - rect.top()).abs(), (rect.bottom() - cy).abs());
    // A circle's sides are all four, and its percentage is of the box's
    // diagonal over sqrt(2).
    let diagonal = px((f32::from(w).powi(2) + f32::from(h).powi(2)).sqrt() / 2f32.sqrt());
    let one = |r: &VnShapeRadius, horizontal: bool| -> Pixels {
        let (a, b) = if circle {
            (near_x.min(far_x).min(near_y.min(far_y)), near_x.max(far_x).max(near_y.max(far_y)))
        } else if horizontal {
            (near_x.min(far_x), near_x.max(far_x))
        } else {
            (near_y.min(far_y), near_y.max(far_y))
        };
        match r {
            VnShapeRadius::ClosestSide => a,
            VnShapeRadius::FarthestSide => b,
            VnShapeRadius::Length(l) => resolve(l, if circle { diagonal } else if horizontal { w } else { h }).max(px(0.)),
        }
    };
    (point(cx, cy), one(&radius[0], true), one(&radius[1], false))
}

/// The CSS styling that can be applied to an element via the `Styled` trait
#[derive(Clone, Refineable, Debug)]
#[refineable(Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Style {
    /// What layout strategy should be used?
    pub display: Display,

    /// Should the element be painted on screen?
    pub visibility: Visibility,

    // Overflow properties
    /// How children overflowing their container should affect layout
    #[refineable]
    pub overflow: Point<Overflow>,
    /// How much space (in points) should be reserved for the scrollbars of `Overflow::Scroll` and `Overflow::Auto` nodes.
    pub scrollbar_width: AbsoluteLength,
    /// Whether both x and y axis should be scrollable at the same time.
    pub allow_concurrent_scroll: bool,
    /// Whether scrolling should be restricted to the input gesture's axis.
    ///
    /// Pixel-based scroll gestures are locked to their initially dominant axis. The lock may be
    /// released when the gesture changes direction strongly. Touch phases delimit gestures when
    /// available, with a timeout fallback for platforms that only emit moved events.
    ///
    /// This also prevents input from being remapped to another axis. For example, horizontal input
    /// will not scroll a container that only has vertical overflow enabled. Mouse wheel platforms
    /// typically report ordinary wheel input on the Y axis and Shift-modified input on the X axis.
    ///
    /// ## Motivation
    ///
    /// On the web when scrolling with the mouse wheel, scrolling up and down will always scroll the Y axis, even when
    /// the mouse is over a horizontally-scrollable element.
    ///
    /// The only way to scroll horizontally is to hold down `Shift` while scrolling, which then changes the scroll axis
    /// to the X axis.
    ///
    /// Currently, GPUI operates differently from the web in that it will scroll an element in either the X or Y axis
    /// when scrolling with just the mouse wheel. This causes problems when scrolling in a vertical list that contains
    /// horizontally-scrollable elements, as when you get to the horizontally-scrollable elements the scroll will be
    /// hijacked.
    ///
    /// Ideally we would match the web's behavior and not have a need for this, but right now we're adding this opt-in
    /// style property to limit the potential blast radius.
    pub restrict_scroll_to_axis: bool,

    // Position properties
    /// What should the `position` value of this struct use as a base offset?
    pub position: Position,
    /// How should the position of this element be tweaked relative to the layout defined?
    #[refineable]
    pub inset: Edges<Length>,

    // Size properties
    /// Sets the initial size of the item
    #[refineable]
    pub size: Size<Length>,
    /// Controls the minimum size of the item
    #[refineable]
    pub min_size: Size<Length>,
    /// Controls the maximum size of the item
    #[refineable]
    pub max_size: Size<Length>,
    /// Sets the preferred aspect ratio for the item. The ratio is calculated as width divided by height.
    pub aspect_ratio: Option<f32>,

    // Spacing Properties
    /// How large should the margin be on each side?
    #[refineable]
    pub margin: Edges<Length>,
    /// How large should the padding be on each side?
    #[refineable]
    pub padding: Edges<DefiniteLength>,
    /// How large should the border be on each side?
    #[refineable]
    pub border_widths: Edges<AbsoluteLength>,

    // Alignment properties
    /// How this node's children aligned in the cross/block axis?
    pub align_items: Option<AlignItems>,
    /// How this node should be aligned in the cross/block axis. Falls back to the parents [`AlignItems`] if not set
    pub align_self: Option<AlignSelf>,
    /// How should content contained within this item be aligned in the cross/block axis
    pub align_content: Option<AlignContent>,
    /// How should contained within this item be aligned in the main/inline axis
    pub justify_content: Option<JustifyContent>,
    /// How large should the gaps between items in a flex container be?
    #[refineable]
    pub gap: Size<DefiniteLength>,

    // Flexbox properties
    /// Which direction does the main axis flow in?
    pub flex_direction: FlexDirection,
    /// Should elements wrap, or stay in a single line?
    pub flex_wrap: FlexWrap,
    /// Sets the initial main axis size of the item
    pub flex_basis: Length,
    /// The relative rate at which this item grows when it is expanding to fill space, 0.0 is the default value, and this value must be positive.
    pub flex_grow: f32,
    /// The relative rate at which this item shrinks when it is contracting to fit into space, 1.0 is the default value, and this value must be positive.
    pub flex_shrink: f32,

    /// The fill color of this element
    pub background: Option<Fill>,

    /// Extra background layers painted OVER [`Style::background`], bottom layer
    /// first — CSS's `background-image` stack, which sits above
    /// `background-color`.
    ///
    /// One quad per layer rather than a layer array inside the quad: painter's
    /// order already composites them in the right sequence, the scene's sort
    /// and cull need no new concept, and nothing about the shader changes. The
    /// cost is one extra primitive per layer, which is why a caller is expected
    /// to keep the stack short.
    ///
    /// CSS lists layers top-first; this vector is bottom-first, so the mapper
    /// reverses.
    pub background_layers: Vec<Background>,

    /// The border color of this element
    pub border_color: Option<Hsla>,

    /// CSS side colours override the shared colour independently.
    #[refineable]
    pub vn_border_colors: Edges<Option<Hsla>>,

    /// The border style of this element
    pub border_style: BorderStyle,

    /// The radius of the corners of this element
    #[refineable]
    pub corner_radii: Corners<AbsoluteLength>,

    /// vue-native fork: a CSS corner radius `corner_radii` cannot hold. That is a
    /// percentage of the box, a `calc()` with one, or an elliptical `a / b`
    /// corner. Where a corner has one, it replaces that corner's
    /// `corner_radii`, and it resolves against the box at paint
    /// ([`Style::corner_radii_for`]). vn-css writes both fields for every
    /// corner it maps, so the later declaration wins whichever it used.
    #[refineable]
    pub vn_corner_radii: Corners<Option<VnCornerRadius>>,

    /// Box shadow of the element
    pub box_shadow: Vec<BoxShadow>,

    /// vue-native fork: draw the outer shadows under the border box too.
    ///
    /// CSS clips an outer `box-shadow` to the outside of the border box (CSS
    /// Backgrounds 3 §7.1): the shadow is cast as if the box were opaque, so
    /// a translucent background shows what is behind the element, not the
    /// shadow. [`Window::paint_drop_shadows`] does that by default. This
    /// switch is for a shadow that stands in for the element's own painting:
    /// vue-native draws `filter: blur()` on a solid box as the box's own
    /// Gaussian shadow, whose middle must stay filled.
    pub vn_shadow_under_box: bool,

    /// CSS `backdrop-filter: blur(<length>)`: blur everything already painted
    /// behind this element's border box, clipped to `corner_radii`.
    ///
    /// `None` (and a non-positive radius) means no backdrop filter. Only the
    /// `blur()` filter function is modelled; the rest of the CSS filter
    /// function list is not.
    pub backdrop_blur: Option<Pixels>,

    /// The `--shading: <effect>(<args>)` custom property: shade this element's
    /// border box with one of the built-in effect pipelines.
    ///
    /// `None` means the property was never declared. `--shading: none` is
    /// [`EffectSpec::NONE`] — a *value*, because a refinement cannot spell
    /// "explicitly nothing" and that is what has to cancel an effect declared
    /// by an earlier rule in the cascade. Same shape as `backdrop-filter: none`
    /// mapping to a 0px radius.
    ///
    /// Deliberate divergence from CSS, documented for authors: a browser
    /// *inherits* custom properties down the subtree. vue-native applies
    /// `--shading` to the declaring element only — inheriting a GPU effect into
    /// every descendant is never what an author means, and a browser ignores
    /// the property entirely anyway, so nothing renders differently there.
    pub vn_effect: Option<EffectSpec>,

    /// CSS `transform`: rotate, scale and translate this element's PAINTED
    /// output — its background, borders, shadows, text and its whole subtree —
    /// about its own centre, changing no layout at all.
    ///
    /// `None` means the property was never declared; [`TransformSpec::IDENTITY`]
    /// is `transform: none`, a *value*, because a refinement cannot spell
    /// "explicitly nothing" and that is what has to cancel a transform declared
    /// by an earlier rule in the cascade. Same shape as `vn_effect` above, and
    /// as `backdrop-filter: none` mapping to a 0px radius.
    ///
    /// Layout is untouched on purpose, and that is the browser's rule rather
    /// than a shortcut: a transformed element still occupies exactly the box it
    /// laid out in, its siblings do not move, and its scroll container does not
    /// grow. It also means `translate()` here is NOT interchangeable with a
    /// margin offset — which is the divergence this field exists to retire.
    ///
    /// Hit-testing follows the picture: a hitbox records the matrix in force
    /// when it was inserted and the pointer is mapped back through the inverse,
    /// so a rotated button is clickable where it is *drawn*.
    pub vn_transform: Option<TransformSpec>,

    /// The text style of this element
    #[refineable]
    pub text: TextStyleRefinement,

    /// The mouse cursor style shown when the mouse pointer is over an element.
    pub mouse_cursor: Option<CursorStyle>,

    /// The opacity of this element
    pub opacity: Option<f32>,

    /// The grid columns of this element
    /// Roughly equivalent to the Tailwind `grid-cols-<number>`
    pub grid_cols: Option<GridTemplate>,

    /// The row span of this element
    /// Equivalent to the Tailwind `grid-rows-<number>`
    pub grid_rows: Option<GridTemplate>,

    /// The grid location of this element
    pub grid_location: Option<GridLocation>,

    /// CSS `grid-template-columns` as an ARBITRARY track list — `1fr auto`,
    /// `minmax(0, 1fr) 200px`, `repeat(3, 1fr)` (expanded by the caller).
    ///
    /// Takes precedence over [`Style::grid_cols`], which stays as the cheap
    /// `repeat(N, …)` shorthand the `Styled` helpers write. `None` means "this
    /// element declared no track list", not "no columns".
    pub vn_grid_cols: Option<Vec<GridTrack>>,

    /// CSS `grid-template-rows` as an arbitrary track list. See
    /// [`Style::vn_grid_cols`].
    pub vn_grid_rows: Option<Vec<GridTrack>>,

    /// The `repeat(auto-fill | auto-fit, …)` run of [`Style::vn_grid_cols`],
    /// if it has one. Meaningless without that list.
    pub vn_grid_cols_auto: Option<GridAutoRepeat>,

    /// The `repeat(auto-fill | auto-fit, …)` run of [`Style::vn_grid_rows`].
    pub vn_grid_rows_auto: Option<GridAutoRepeat>,

    /// CSS `grid-auto-flow` — the direction the auto-placement algorithm fills,
    /// and whether it packs densely.
    pub vn_grid_auto_flow: Option<GridAutoFlow>,

    /// CSS `justify-items` — the inline-axis alignment this container gives its
    /// children. (`align_items` is the block axis; taffy honours both, gpui only
    /// ever exposed the second.)
    pub vn_justify_items: Option<AlignItems>,

    /// CSS `justify-self` — this item's own inline-axis alignment, overriding
    /// its container's `justify-items`.
    pub vn_justify_self: Option<AlignSelf>,

    /// CSS `outline` — a ring painted OUTSIDE the border box that takes no
    /// layout space at all, which is what makes it (and not a border) the right
    /// primitive for a focus indicator.
    ///
    /// gpui has no outline in its scene; this is painted as one extra
    /// border-only quad at dilated bounds, after the element's own border, so it
    /// lands on top of the element and outside it. `None` = never declared.
    pub vn_outline: Option<Outline>,

    /// vue-native fork: CSS `mask` in the one shape the paint path can cut,
    /// the gradient-border idiom. That is two opaque layers, the top one
    /// clipped to the content box, composited with `exclude`:
    /// `mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    /// mask-composite: exclude`. What is left visible is the ring of padding
    /// and border around the content box.
    ///
    /// When set, the background and its layers paint only in that ring, as one
    /// quad each whose border widths are the border plus the padding
    /// (`color::BG_RING`). The border lies inside the ring and paints as
    /// usual. Not painted at all: the outer shadows and the outline, which lie
    /// outside the border box and which the mask cuts entirely, and the inset
    /// shadows, backdrop blur and effect quads, which the ring would show only
    /// a strip of. The children are NOT masked. vn-tree says so when there
    /// are any. A
    /// percentage padding resolves against the element's own width here,
    /// not its containing block's.
    pub vn_mask_ring: bool,

    /// vue-native fork: CSS `clip-path`, in the shapes a rounded content
    /// mask carries ([`VnClipPath`]). It clips the element's whole painting,
    /// its shadows, background, border, outline and subtree, and its hit
    /// test, inside its `transform`.
    pub vn_clip_path: VnClipPath,

    /// SVG color filtering of the whole painted subtree.
    pub vn_color_matrix: Option<crate::VnColorMatrix>,

    /// CSS backgrounds painted through the descendant glyph alpha, not the box.
    pub vn_text_fill: Option<VnTextFill>,

    /// Whether to draw a red debugging outline around this element
    #[cfg(debug_assertions)]
    pub debug: bool,

    /// Whether to draw a red debugging outline around this element and all of its conforming children
    #[cfg(debug_assertions)]
    pub debug_below: bool,
}

impl Styled for StyleRefinement {
    fn style(&mut self) -> &mut StyleRefinement {
        self
    }
}

impl StyleRefinement {
    /// The grid location of this element
    pub fn grid_location_mut(&mut self) -> &mut GridLocation {
        self.grid_location.get_or_insert_default()
    }
}

/// The value of the visibility property, similar to the CSS property `visibility`
#[derive(Default, Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum Visibility {
    /// The element should be drawn as normal.
    #[default]
    Visible,
    /// The element should not be drawn, but should still take up space in the layout.
    Hidden,
}

/// The possible values of the box-shadow property
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct BoxShadow {
    /// What color should the shadow have?
    pub color: Hsla,
    /// How should it be offset from its element?
    pub offset: Point<Pixels>,
    /// How much should the shadow be blurred?
    pub blur_radius: Pixels,
    /// How much should the shadow spread?
    pub spread_radius: Pixels,
    /// Whether this is an inset shadow (drawn inside the element's bounds).
    pub inset: bool,
}

impl BoxShadow {
    /// Creates a new [`BoxShadow`] with the given offset and color, matching the order
    /// of the CSS `box-shadow` property. Use the builder methods to set blur radius,
    /// spread radius, and inset.
    pub fn new(offset_x: Pixels, offset_y: Pixels, color: Hsla) -> Self {
        Self {
            color,
            offset: point(offset_x, offset_y),
            blur_radius: px(0.),
            spread_radius: px(0.),
            inset: false,
        }
    }

    /// Sets the shadow blur radius.
    pub fn blur_radius(mut self, blur_radius: Pixels) -> Self {
        self.blur_radius = blur_radius;
        self
    }

    /// Sets the shadow spread radius.
    pub fn spread_radius(mut self, spread_radius: Pixels) -> Self {
        self.spread_radius = spread_radius;
        self
    }

    /// Marks the shadow as inset (drawn inside the element's bounds).
    pub fn inset(mut self) -> Self {
        self.inset = true;
        self
    }
}

/// How to handle whitespace in text
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum WhiteSpace {
    /// Normal line wrapping when text overflows the width of the element
    #[default]
    Normal,
    /// No line wrapping, text will overflow the width of the element
    Nowrap,
}

/// vue-native fork: CSS `overflow-wrap` (CSS Text 3 §5.5). May a line break
/// inside a word that does not fit on a line of its own?
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema,
)]
pub enum OverflowWrap {
    /// Lines break only at soft wrap opportunities. A word wider than the
    /// line overflows it.
    #[default]
    Normal,
    /// A word that does not fit on a line of its own is broken where it
    /// overflows. The min-content size still counts the word whole.
    BreakWord,
    /// As `BreakWord`, and the min-content size counts those breaks too, so a
    /// flex item or a `min-content` box can be as narrow as one character.
    Anywhere,
}

/// vue-native fork: CSS `word-break` (CSS Text 3 §5.2).
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema,
)]
pub enum WordBreak {
    /// The usual soft wrap opportunities.
    #[default]
    Normal,
    /// A line may break between any two letters, and the min-content size
    /// counts those breaks.
    BreakAll,
    /// CJK letters are kept together like the letters of any other word: no
    /// break between two of them.
    KeepAll,
    /// The legacy keyword: `normal` with `overflow-wrap: anywhere`, whatever
    /// `overflow-wrap` says.
    BreakWord,
}

/// vue-native fork: CSS `hyphens` (CSS Text 3 §6.1). There is no hyphenation
/// dictionary, so `auto` is the caller's to map to `Manual`.
#[derive(
    Copy, Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema,
)]
pub enum Hyphens {
    /// A soft hyphen (U+00AD) is no break opportunity.
    None,
    /// A line may break after a soft hyphen, and shows a hyphen there.
    #[default]
    Manual,
}

/// vue-native fork: the rules a text breaks its lines under, from its style
/// ([`TextStyle::wrap`]). Part of the line layout cache's key.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TextWrap {
    /// CSS `overflow-wrap`.
    pub overflow_wrap: OverflowWrap,
    /// CSS `word-break`.
    pub word_break: WordBreak,
    /// CSS `hyphens`.
    pub hyphens: Hyphens,
}

impl TextWrap {
    /// gpui's rules before the fork took CSS's: a word that does not fit on
    /// a line is broken where it overflows. What [`crate::WindowTextSystem::shape_text`]
    /// keeps for its callers, a text field among them (a `<textarea>` is
    /// `overflow-wrap: break-word` in every browser's UA sheet).
    pub const BREAK_WORD: Self = Self {
        overflow_wrap: OverflowWrap::BreakWord,
        word_break: WordBreak::Normal,
        hyphens: Hyphens::Manual,
    };

    /// May a word that does not fit be broken where it overflows?
    pub(crate) fn breaks_words(&self) -> bool {
        self.overflow_wrap != OverflowWrap::Normal || self.word_break == WordBreak::BreakWord
    }

    /// Do those breaks count for the min-content size?
    pub(crate) fn min_content_breaks_words(&self) -> bool {
        self.overflow_wrap == OverflowWrap::Anywhere || self.word_break == WordBreak::BreakWord
    }
}

/// How to truncate text that overflows the width of the element
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum TextOverflow {
    /// Truncate the text at the end when it doesn't fit, and represent this truncation by
    /// displaying the provided string (e.g., "very long te…").
    Truncate(SharedString),
    /// Truncate the text at the start when it doesn't fit, and represent this truncation by
    /// displaying the provided string at the beginning (e.g., "…ong text here").
    /// Typically more adequate for file paths where the end is more important than the beginning.
    TruncateStart(SharedString),
    /// Truncate the text in the middle when it doesn't fit, preserving both the start and end
    /// of the string (e.g., "long fi…name.rs"). Useful for filenames where both the prefix
    /// and the extension are important context.
    TruncateMiddle(SharedString),
}

/// How to align text within the element
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub enum TextAlign {
    /// Align the text to the left of the element
    #[default]
    Left,

    /// Center the text within the element
    Center,

    /// Align the text to the right of the element
    Right,
}

/// A CSS background stack clipped to the element's descendant text.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct VnTextFill {
    /// Stable element identity; reopened/deferred text shares one alpha mask.
    pub group_id: u64,
    /// Backgrounds, bottom first, retaining their gradient and tile parameters.
    pub backgrounds: Vec<Background>,
}

/// A text background resolved against its element, in logical CSS pixels.
/// Carried in TextStyle so deferred drawing and paint-cache keys retain it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct VnTextPaint {
    /// The authored stack.
    pub fill: VnTextFill,
    /// The background positioning box.
    pub bounds: [f32; 4],
    /// Inverse device-space element transform, row major followed by translation.
    pub inverse: [f32; 6],
    /// Device pixels per logical CSS pixel.
    pub scale_factor: f32,
    /// The owner scope's geometry, shared by every descendant glyph.
    #[serde(skip)]
    #[schemars(skip)]
    pub geometry: Arc<crate::VnClipGeometry>,
    /// The owner scope's outer mask. Glyphs retain their local child masks.
    #[serde(skip)]
    #[schemars(skip)]
    pub content_mask: ContentMask<crate::ScaledPixels>,
}

/// The properties that can be used to style text in GPUI
#[derive(Refineable, Clone, Debug, PartialEq)]
#[refineable(Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct TextStyle {
    /// The color of the text
    pub color: Hsla,

    /// Background clipped to transparent glyphs in this text scope.
    pub vn_text_paint: Option<Arc<VnTextPaint>>,

    /// The font family to use
    pub font_family: SharedString,

    /// The font features to use
    pub font_features: FontFeatures,

    /// The fallback fonts to use
    pub font_fallbacks: Option<FontFallbacks>,

    /// The font size to use, in pixels or rems.
    pub font_size: AbsoluteLength,

    /// The line height to use, in pixels or fractions
    pub line_height: DefiniteLength,

    /// The font weight, e.g. bold
    pub font_weight: FontWeight,

    /// The font style, e.g. italic
    pub font_style: FontStyle,

    /// The background color of the text
    pub background_color: Option<Hsla>,

    /// The underline style of the text
    pub underline: Option<UnderlineStyle>,

    /// The strikethrough style of the text
    pub strikethrough: Option<StrikethroughStyle>,

    /// How to handle whitespace in the text
    pub white_space: WhiteSpace,

    /// The text should be truncated if it overflows the width of the element
    pub text_overflow: Option<TextOverflow>,

    /// How the text should be aligned within the element
    pub text_align: TextAlign,

    /// The number of lines to display before truncating the text
    pub line_clamp: Option<usize>,

    /// Extra space after each typographic character unit — CSS
    /// `letter-spacing`. Already resolved to pixels: `em`-relative tracking is
    /// a function of the font size, which the caller has and this does not.
    pub letter_spacing: Pixels,

    /// Extra space after each word separator — CSS `word-spacing`.
    pub word_spacing: Pixels,

    /// vue-native fork: CSS `overflow-wrap`. Initial `normal`, as in CSS.
    pub overflow_wrap: OverflowWrap,

    /// vue-native fork: CSS `word-break`.
    pub word_break: WordBreak,

    /// vue-native fork: CSS `hyphens`.
    pub hyphens: Hyphens,
}

impl Default for TextStyle {
    fn default() -> Self {
        TextStyle {
            color: black(),
            vn_text_paint: None,
            // todo(linux) make this configurable or choose better default
            font_family: ".SystemUIFont".into(),
            font_features: FontFeatures::default(),
            font_fallbacks: None,
            font_size: rems(1.).into(),
            line_height: phi(),
            font_weight: FontWeight::default(),
            font_style: FontStyle::default(),
            background_color: None,
            underline: None,
            strikethrough: None,
            white_space: WhiteSpace::Normal,
            text_overflow: None,
            text_align: TextAlign::default(),
            line_clamp: None,
            letter_spacing: px(0.),
            word_spacing: px(0.),
            overflow_wrap: OverflowWrap::default(),
            word_break: WordBreak::default(),
            hyphens: Hyphens::default(),
        }
    }
}

impl TextStyle {
    /// vue-native fork: the line-breaking rules of this style.
    pub fn wrap(&self) -> TextWrap {
        TextWrap {
            overflow_wrap: self.overflow_wrap,
            word_break: self.word_break,
            hyphens: self.hyphens,
        }
    }

    /// Create a new text style with the given highlighting applied.
    pub fn highlight(mut self, style: impl Into<HighlightStyle>) -> Self {
        let style = style.into();
        if let Some(weight) = style.font_weight {
            self.font_weight = weight;
        }
        if let Some(style) = style.font_style {
            self.font_style = style;
        }

        if let Some(color) = style.color {
            self.color = self.color.blend(color);
        }

        if let Some(factor) = style.fade_out {
            self.color.fade_out(factor);
        }

        if let Some(background_color) = style.background_color {
            self.background_color = Some(background_color);
        }

        if let Some(underline) = style.underline {
            self.underline = Some(underline);
        }

        if let Some(strikethrough) = style.strikethrough {
            self.strikethrough = Some(strikethrough);
        }

        self
    }

    /// Get the font configured for this text style.
    pub fn font(&self) -> Font {
        Font {
            family: self.font_family.clone(),
            features: self.font_features.clone(),
            fallbacks: self.font_fallbacks.clone(),
            weight: self.font_weight,
            style: self.font_style,
        }
    }

    /// Returns the rounded line height in pixels.
    pub fn line_height_in_pixels(&self, rem_size: Pixels) -> Pixels {
        self.line_height.to_pixels(self.font_size, rem_size).round()
    }

    /// Convert this text style into a [`TextRun`], for the given length of the text.
    pub fn to_run(&self, len: usize) -> TextRun {
        TextRun {
            len,
            font: Font {
                family: self.font_family.clone(),
                features: self.font_features.clone(),
                fallbacks: self.font_fallbacks.clone(),
                weight: self.font_weight,
                style: self.font_style,
            },
            color: self.color,
            background_color: self.background_color,
            underline: self.underline,
            strikethrough: self.strikethrough,
            tracking: self.letter_spacing,
            word_spacing: self.word_spacing,
        }
    }
}

/// A highlight style to apply, similar to a `TextStyle` except
/// for a single font, uniformly sized and spaced text.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct HighlightStyle {
    /// The color of the text
    pub color: Option<Hsla>,

    /// The font weight, e.g. bold
    pub font_weight: Option<FontWeight>,

    /// The font style, e.g. italic
    pub font_style: Option<FontStyle>,

    /// The background color of the text
    pub background_color: Option<Hsla>,

    /// The underline style of the text
    pub underline: Option<UnderlineStyle>,

    /// The underline style of the text
    pub strikethrough: Option<StrikethroughStyle>,

    /// Similar to the CSS `opacity` property, this will cause the text to be less vibrant.
    pub fade_out: Option<f32>,
}

impl Eq for HighlightStyle {}

impl Hash for HighlightStyle {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.color.hash(state);
        self.font_weight.hash(state);
        self.font_style.hash(state);
        self.background_color.hash(state);
        self.underline.hash(state);
        self.strikethrough.hash(state);
        state.write_u32(u32::from_be_bytes(
            self.fade_out.map(|f| f.to_be_bytes()).unwrap_or_default(),
        ));
    }
}

/// The one colour that best stands in for a background — the fill itself when
/// it is solid, and a gradient's first live stop otherwise. Used for the
/// transparent border that keeps a fill's antialiased corner from seaming.
fn representative_color(background: &Background) -> Hsla {
    match background.tag {
        BackgroundTag::Solid | BackgroundTag::PatternSlash | BackgroundTag::Checkerboard => {
            background.solid
        }
        BackgroundTag::LinearGradient
        | BackgroundTag::RadialGradient
        | BackgroundTag::ConicGradient => background
            .live_stops()
            .first()
            .map(|stop| stop.color)
            .unwrap_or_default(),
    }
}

impl Style {
    /// Returns true if the style is visible and the background is opaque.
    pub fn has_opaque_background(&self) -> bool {
        self.background
            .as_ref()
            .is_some_and(|fill| fill.color().is_some_and(|color| !color.is_transparent()))
    }

    /// Get the text style in this element style.
    pub fn text_style(&self) -> Option<&TextStyleRefinement> {
        if self.text.is_some() {
            Some(&self.text)
        } else {
            None
        }
    }

    /// Text refinements with the background positioning box resolved for paint.
    /// Layout uses `text_style` because this scope must not change glyph shaping.
    pub fn text_style_for_bounds(&self, bounds: Bounds<Pixels>, window: &Window) -> Option<TextStyleRefinement> {
        let mut text = self.text.clone();
        if let Some(fill) = &self.vn_text_fill {
            text.vn_text_paint = Some(window.vn_text_paint(fill.clone(), bounds));
        }
        text.is_some().then_some(text)
    }

    /// vue-native fork: the corner radii this style paints on a box of `size`,
    /// percentages, the overlap rule and all. See [`vn_resolve_corner_radii`].
    pub fn corner_radii_for(&self, size: Size<Pixels>, rem_size: Pixels) -> Corners<Pixels> {
        vn_resolve_corner_radii(&self.corner_radii, &self.vn_corner_radii, size, rem_size)
    }

    /// Get the content mask for this element style, based on the given bounds.
    /// If the element does not hide its overflow, this will return `None`.
    pub fn overflow_mask(
        &self,
        bounds: Bounds<Pixels>,
        rem_size: Pixels,
    ) -> Option<ContentMask<Pixels>> {
        match self.overflow {
            Point {
                x: Overflow::Visible,
                y: Overflow::Visible,
            } => None,
            _ => {
                let mut min = bounds.origin;
                let mut max = bounds.bottom_right();
                let border_box = bounds;

                if self
                    .border_color
                    .is_some_and(|color| !color.is_transparent())
                {
                    min.x += self.border_widths.left.to_pixels(rem_size);
                    max.x -= self.border_widths.right.to_pixels(rem_size);
                    min.y += self.border_widths.top.to_pixels(rem_size);
                    max.y -= self.border_widths.bottom.to_pixels(rem_size);
                }

                let bounds = match (
                    self.overflow.x == Overflow::Visible,
                    self.overflow.y == Overflow::Visible,
                ) {
                    // x and y both visible
                    (true, true) => return None,
                    // x visible, y hidden
                    (true, false) => Bounds::from_corners(
                        point(min.x, bounds.origin.y),
                        point(max.x, bounds.bottom_right().y),
                    ),
                    // x hidden, y visible
                    (false, true) => Bounds::from_corners(
                        point(bounds.origin.x, min.y),
                        point(bounds.bottom_right().x, max.y),
                    ),
                    // both hidden
                    (false, false) => Bounds::from_corners(min, max),
                };

                // vue-native fork: CSS clips overflow to the padding box's
                // ROUNDED shape, whose radii are the border radii less the
                // border widths. Each corner takes the larger of its two
                // borders off, a circle inside the elliptical corner CSS draws
                // for unequal ones. The insets are read off the mask's own
                // bounds, so a border the mask does not inset (a transparent
                // one, above) takes nothing off.
                let insets = Edges {
                    top: bounds.top() - border_box.top(),
                    right: border_box.right() - bounds.right(),
                    bottom: border_box.bottom() - bounds.bottom(),
                    left: bounds.left() - border_box.left(),
                };
                let outer = self.corner_radii_for(border_box.size, rem_size);
                let inner = |radius: Pixels, a: Pixels, b: Pixels| (radius - a.max(b)).max(px(0.));
                let corner_radii = Corners {
                    top_left: inner(outer.top_left, insets.top, insets.left),
                    top_right: inner(outer.top_right, insets.top, insets.right),
                    bottom_right: inner(outer.bottom_right, insets.bottom, insets.right),
                    bottom_left: inner(outer.bottom_left, insets.bottom, insets.left),
                }
                .clamp_radii_for_quad_size(bounds.size);

                Some(ContentMask {
                    bounds,
                    corner_radii,
                })
            }
        }
    }

    /// vue-native fork: the mask `clip-path` cuts this element's painting and
    /// hit test to ([`Style::vn_clip_path`]), for the border box `bounds`, in
    /// the element's own frame. `None` without a clip.
    ///
    /// A shape that is no rounded rectangle ([`Style::vn_clip_outline`])
    /// gives the rectangle around its outline here, which culls and cuts the
    /// isolated layer the outline is then composited through.
    pub fn vn_clip_mask(&self, bounds: Bounds<Pixels>, rem_size: Pixels) -> Option<ContentMask<Pixels>> {
        let clip = &self.vn_clip_path;
        let reference = match clip {
            VnClipPath::None => return None,
            VnClipPath::Polygon { .. } | VnClipPath::Path { .. } => {
                let outline = self.vn_clip_outline(bounds, rem_size)?;
                let rect = outline.control_bounds().unwrap_or(Bounds { origin: bounds.origin, size: Size::default() });
                return Some(ContentMask::rect(rect));
            }
            VnClipPath::Box(reference)
            | VnClipPath::Inset { reference, .. }
            | VnClipPath::Ellipse { reference, .. } => *reference,
        };
        let (rect, rounded) = self.vn_clip_box(reference, bounds, rem_size);
        let (w, h) = (rect.size.width, rect.size.height);
        let resolve = |length: &DefiniteLength, base: Pixels| length.to_pixels(AbsoluteLength::Pixels(base), rem_size);
        Some(match clip {
            VnClipPath::None | VnClipPath::Polygon { .. } | VnClipPath::Path { .. } => return None,
            VnClipPath::Box(_) => ContentMask { bounds: rect, corner_radii: rounded },
            VnClipPath::Inset { insets, radii, .. } => {
                // Chromium's `BasicShapeInset::GetPath`: the insets come off
                // the reference box, an empty side stays empty, and the radii
                // resolve against the reference box and are then scaled down
                // to fit the inset rectangle (css-backgrounds-3 §5.5).
                let [top, right, bottom, left] = insets;
                let (top, right) = (resolve(top, h), resolve(right, w));
                let (bottom, left) = (resolve(bottom, h), resolve(left, w));
                let shape = Bounds {
                    origin: point(rect.left() + left, rect.top() + top),
                    size: size((w - left - right).max(px(0.)), (h - top - bottom).max(px(0.))),
                };
                let pairs = radii.map(|r| {
                    let x = f32::from(resolve(&r.x, w)).max(0.);
                    let y = f32::from(resolve(&r.y, h)).max(0.);
                    (x, y)
                });
                let [tl, tr, br, bl] = pairs;
                let (sw, sh) = (f32::from(shape.size.width), f32::from(shape.size.height));
                let ratio = |side: f32, a: f32, b: f32| if a + b > side { side / (a + b) } else { 1. };
                let factor = ratio(sw, tl.0, tr.0)
                    .min(ratio(sw, bl.0, br.0))
                    .min(ratio(sh, tl.1, bl.1))
                    .min(ratio(sh, tr.1, br.1));
                let circle = |(x, y): (f32, f32)| px(x.min(y) * factor);
                let corner_radii = Corners {
                    top_left: circle(tl),
                    top_right: circle(tr),
                    bottom_right: circle(br),
                    bottom_left: circle(bl),
                }
                .clamp_radii_for_quad_size(shape.size);
                ContentMask { bounds: shape, corner_radii }
            }
            VnClipPath::Ellipse { center, radius, circle, .. } => {
                let (c, rx, ry) = vn_clip_ellipse(rect, center, radius, *circle, rem_size);
                let shape = Bounds { origin: point(c.x - rx, c.y - ry), size: size(rx * 2., ry * 2.) };
                // A circle is the rounded mask; an ellipse is its box here,
                // and its outline in an isolated layer.
                let corner_radii = if rx == ry { Corners::all(rx) } else { Corners::default() };
                ContentMask { bounds: shape, corner_radii }
            }
        })
    }

    /// vue-native fork: the outline of a `clip-path` that is no rounded
    /// rectangle (`polygon()`, `path()`, an ellipse whose radii differ), laid
    /// against its reference box for the border box `bounds`, in the
    /// element's own frame. `None` for no clip and for a shape
    /// [`Style::vn_clip_mask`] carries alone. The element paints into an
    /// isolated layer composited through it ([`Window::vn_with_clip_shape`]).
    pub fn vn_clip_outline(&self, bounds: Bounds<Pixels>, rem_size: Pixels) -> Option<VnClipOutline> {
        let resolve = |length: &DefiniteLength, base: Pixels| {
            f32::from(length.to_pixels(AbsoluteLength::Pixels(base), rem_size))
        };
        match &self.vn_clip_path {
            VnClipPath::Polygon { reference, fill_rule, points } => {
                let (rect, _) = self.vn_clip_box(*reference, bounds, rem_size);
                let (x0, y0) = (f32::from(rect.left()), f32::from(rect.top()));
                let (w, h) = (rect.size.width, rect.size.height);
                let mut verbs = Vec::with_capacity(points.len() + 1);
                for (i, [x, y]) in points.iter().enumerate() {
                    let p = [x0 + resolve(x, w), y0 + resolve(y, h)];
                    verbs.push(if i == 0 { VnPathVerb::MoveTo(p) } else { VnPathVerb::LineTo(p) });
                }
                if !verbs.is_empty() {
                    verbs.push(VnPathVerb::Close);
                }
                Some(VnClipOutline { fill_rule: *fill_rule, verbs })
            }
            VnClipPath::Path { reference, fill_rule, verbs } => {
                let (rect, _) = self.vn_clip_box(*reference, bounds, rem_size);
                let (x0, y0) = (f32::from(rect.left()), f32::from(rect.top()));
                let at = |p: [f32; 2]| [p[0] + x0, p[1] + y0];
                let verbs = verbs
                    .iter()
                    .map(|v| match *v {
                        VnPathVerb::MoveTo(p) => VnPathVerb::MoveTo(at(p)),
                        VnPathVerb::LineTo(p) => VnPathVerb::LineTo(at(p)),
                        VnPathVerb::QuadTo(c, p) => VnPathVerb::QuadTo(at(c), at(p)),
                        VnPathVerb::CubicTo(c1, c2, p) => VnPathVerb::CubicTo(at(c1), at(c2), at(p)),
                        VnPathVerb::Close => VnPathVerb::Close,
                    })
                    .collect();
                Some(VnClipOutline { fill_rule: *fill_rule, verbs })
            }
            VnClipPath::Ellipse { reference, center, radius, circle } => {
                let (rect, _) = self.vn_clip_box(*reference, bounds, rem_size);
                let (c, rx, ry) = vn_clip_ellipse(rect, center, radius, *circle, rem_size);
                if rx == ry {
                    return None;
                }
                // Four cubics, one per quadrant, k = 4/3 (sqrt(2) - 1).
                let (cx, cy, rx, ry) = (f32::from(c.x), f32::from(c.y), f32::from(rx), f32::from(ry));
                let (kx, ky) = (0.552_284_8 * rx, 0.552_284_8 * ry);
                use VnPathVerb::*;
                Some(VnClipOutline {
                    fill_rule: VnFillRule::NonZero,
                    verbs: vec![
                        MoveTo([cx + rx, cy]),
                        CubicTo([cx + rx, cy + ky], [cx + kx, cy + ry], [cx, cy + ry]),
                        CubicTo([cx - kx, cy + ry], [cx - rx, cy + ky], [cx - rx, cy]),
                        CubicTo([cx - rx, cy - ky], [cx - kx, cy - ry], [cx, cy - ry]),
                        CubicTo([cx + kx, cy - ry], [cx + rx, cy - ky], [cx + rx, cy]),
                        Close,
                    ],
                })
            }
            VnClipPath::None | VnClipPath::Box(_) | VnClipPath::Inset { .. } => None,
        }
    }

    /// vue-native fork: a `clip-path` reference box and its rounded corners,
    /// for the border box `bounds`. Inside the border box a corner loses the
    /// larger of its two insets, as `overflow_mask` rounds the padding box;
    /// the margin box's grows by the larger margin, where it is rounded at
    /// all. A percentage padding or margin resolves against the element's own
    /// width, not its containing block's.
    fn vn_clip_box(&self, reference: VnClipBox, bounds: Bounds<Pixels>, rem_size: Pixels) -> (Bounds<Pixels>, Corners<Pixels>) {
        let outer = self.corner_radii_for(bounds.size, rem_size);
        let width = AbsoluteLength::Pixels(bounds.size.width);
        let edges = match reference {
            VnClipBox::Border => return (bounds, outer),
            VnClipBox::Margin => {
                let margin = |l: &Length| match l {
                    Length::Definite(l) => -l.to_pixels(width, rem_size),
                    Length::Auto => px(0.),
                };
                Edges {
                    top: margin(&self.margin.top),
                    right: margin(&self.margin.right),
                    bottom: margin(&self.margin.bottom),
                    left: margin(&self.margin.left),
                }
            }
            VnClipBox::Padding | VnClipBox::Content => {
                let border = self.border_widths.to_pixels(rem_size);
                if reference == VnClipBox::Padding {
                    border
                } else {
                    let padding = self.padding.to_pixels(size(width, width), rem_size);
                    Edges {
                        top: border.top + padding.top,
                        right: border.right + padding.right,
                        bottom: border.bottom + padding.bottom,
                        left: border.left + padding.left,
                    }
                }
            }
        };
        let rect = Bounds::from_corners(
            point(bounds.left() + edges.left, bounds.top() + edges.top),
            point(
                (bounds.right() - edges.right).max(bounds.left() + edges.left),
                (bounds.bottom() - edges.bottom).max(bounds.top() + edges.top),
            ),
        );
        // A negative inset is an outset (the margin box).
        let corner = |radius: Pixels, a: Pixels, b: Pixels| {
            if a < px(0.) || b < px(0.) {
                if radius > px(0.) { radius - a.min(b) } else { px(0.) }
            } else {
                (radius - a.max(b)).max(px(0.))
            }
        };
        let corner_radii = Corners {
            top_left: corner(outer.top_left, edges.top, edges.left),
            top_right: corner(outer.top_right, edges.top, edges.right),
            bottom_right: corner(outer.bottom_right, edges.bottom, edges.right),
            bottom_left: corner(outer.bottom_left, edges.bottom, edges.left),
        }
        .clamp_radii_for_quad_size(rect.size);
        (rect, corner_radii)
    }

    /// The device-space matrix this style's CSS `transform` produces for an
    /// element laid out at `bounds`, or the identity when there is none.
    ///
    /// Shared by paint (which pushes it onto the primitives) and prepaint
    /// (which records it on the hitbox), so the picture and the hit test can
    /// never disagree about where the element is.
    pub fn transform_matrix(&self, bounds: Bounds<Pixels>, scale_factor: f32) -> TransformationMatrix
    {
        self.vn_transform
            .filter(|t| !t.is_identity())
            .map(|t| t.to_matrix(bounds, scale_factor))
            .unwrap_or_else(TransformationMatrix::unit)
    }

    /// Paints the background of an element styled with this style.
    pub fn paint(
        &self,
        bounds: Bounds<Pixels>,
        window: &mut Window,
        cx: &mut App,
        continuation: impl FnOnce(&mut Window, &mut App),
    ) {
        // CSS `transform` applies to this element's whole painted output AND
        // to its subtree, so it is pushed around everything below — including
        // `continuation`, which is where the children paint. Layout has already
        // happened at this point and is deliberately not consulted again: a
        // transform never moves a box, only its picture.
        //
        // vue-native fork: `clip-path` cuts everything below too, in the
        // element's transformed frame (`Style::vn_clip_mask`).
        //
        // A shape that is no rounded rectangle paints into an isolated layer
        // composited through it (`Window::vn_with_clip_shape`), inside its
        // box's mask.
        let transformation = self.transform_matrix(bounds, window.scale_factor());
        let clip = self.vn_clip_mask(bounds, window.rem_size());
        let outline = self.vn_clip_outline(bounds, window.rem_size());
        if !transformation.is_unit() {
            return window.with_element_transform(transformation, |window| {
                window.with_content_mask(clip, |window| {
                    window.vn_with_clip_shape(outline.as_ref(), |window| {
                        window.vn_with_color_matrix(self.vn_color_matrix, |window| {
                            self.paint_untransformed(bounds, window, cx, continuation)
                        })
                    })
                })
            });
        }
        window.with_content_mask(clip, |window| {
            window.vn_with_clip_shape(outline.as_ref(), |window| {
                window.vn_with_color_matrix(self.vn_color_matrix, |window| {
                    self.paint_untransformed(bounds, window, cx, continuation)
                })
            })
        })
    }

    fn paint_untransformed(
        &self,
        bounds: Bounds<Pixels>,
        window: &mut Window,
        cx: &mut App,
        continuation: impl FnOnce(&mut Window, &mut App),
    ) {
        #[cfg(debug_assertions)]
        if self.debug_below {
            cx.set_global(DebugBelow)
        }

        #[cfg(debug_assertions)]
        if self.debug || cx.has_global::<DebugBelow>() {
            window.paint_quad(crate::outline(bounds, crate::red(), BorderStyle::default()));
        }

        let rem_size = window.rem_size();
        let corner_radii = self.corner_radii_for(bounds.size, rem_size);

        // vue-native fork: `vn_mask_ring` paints the background only in the
        // padding and border area. These are the border widths of the quads
        // that do it: the border plus the padding. A percentage padding
        // resolves against this box's width (see the field's doc).
        let ring_widths = self.vn_mask_ring.then(|| {
            let border = self.border_widths.to_pixels(rem_size);
            let width = AbsoluteLength::Pixels(bounds.size.width);
            let padding = self.padding.to_pixels(size(width, width), rem_size);
            Edges {
                top: border.top + padding.top,
                right: border.right + padding.right,
                bottom: border.bottom + padding.bottom,
                left: border.left + padding.left,
            }
        });
        let ring = ring_widths.is_some();

        if !ring {
            if self.vn_shadow_under_box {
                window.paint_drop_shadows_under_box(bounds, corner_radii, &self.box_shadow);
            } else {
                window.paint_drop_shadows(bounds, corner_radii, &self.box_shadow);
            }
        }

        // CSS paints the filtered backdrop underneath the element's own
        // background, so the blur goes in between the drop shadows and the
        // background quad. The tint stays transparent — `background-color` is
        // the quad below.
        if let Some(radius) = self.backdrop_blur
            && radius > crate::px(0.)
            && !ring
        {
            window.paint_backdrop_blur_rect(
                bounds,
                corner_radii,
                crate::BackdropBlurEffect::new(radius),
            );
        }

        // `--shading` splits on whether the effect consumes the backdrop:
        //
        // * a BACKDROP effect (`frost`) replaces the pixels behind the element
        //   and so paints exactly where `backdrop-filter` does — under the
        //   element's own background. It also *has* to: it draws with blending
        //   disabled and owns every pixel of its quad, so painting it later
        //   would erase the background quad.
        // * a non-backdrop effect (`noise`, `glow`) is an ordinary source-over
        //   layer and paints over the background but under the content, which
        //   is the only slot where film grain is visible on an opaque panel
        //   and where a glow's inner rim survives.
        //
        // An effect whose output moves with `time` (an animated grain) keeps
        // asking for the next frame. That is decorative motion, so it honours
        // `App::reduce_motion` here — clearing the flag stops the repaint
        // request, which freezes the effect on its current frame rather than
        // removing it. Same behaviour vn-css's animator gives transitions.
        let effect = self
            .vn_effect
            .filter(|spec| spec.is_some() && !ring)
            .map(|mut spec| {
                spec.animated &= !cx.reduce_motion();
                spec
            });
        // The background quads: the whole box, or only the ring.
        let fill = |background: Background| match ring_widths {
            Some(widths) => (background.ring(), widths),
            None => (background, Edges::default()),
        };
        if let Some(spec) = effect
            && spec.flags & crate::EFFECT_FLAG_NEEDS_BACKDROP != 0
        {
            window.paint_effect_quad(bounds, corner_radii, spec);
        }

        let background_color = self.background.as_ref().and_then(Fill::color);
        if background_color.is_some_and(|color| !color.is_transparent()) {
            let color = background_color.unwrap_or_default();
            let mut border_color = representative_color(&color);
            border_color.a = 0.;
            let (color, widths) = fill(color);
            window.paint_quad(quad(
                bounds,
                corner_radii,
                color,
                widths,
                border_color,
                self.border_style,
            ));
        }

        // The `background-image` stack, bottom layer first, each one its own
        // quad over the colour below it. The AA-matching transparent border
        // trick above is deliberately NOT repeated per layer: it exists to stop
        // the bottom fill's antialiased corner showing a seam against the
        // border, and a second quad carrying it would double-blend that seam.
        for layer in &self.background_layers {
            if layer.is_transparent() {
                continue;
            }
            let mut border_color = representative_color(layer);
            border_color.a = 0.;
            let (layer, widths) = fill(*layer);
            window.paint_quad(quad(
                bounds,
                corner_radii,
                layer,
                widths,
                border_color,
                self.border_style,
            ));
        }

        if !ring {
            window.paint_inset_shadows(bounds, corner_radii, &self.box_shadow);
        }

        if let Some(spec) = effect
            && spec.flags & crate::EFFECT_FLAG_NEEDS_BACKDROP == 0
        {
            window.paint_effect_quad(bounds, corner_radii, spec);
        }

        continuation(window, cx);

        if self.is_border_visible() {
            let border_widths = self.border_widths.to_pixels(rem_size);
            let colors = self.vn_resolved_border_colors();
            if colors.top == colors.right && colors.top == colors.bottom && colors.top == colors.left {
                window.paint_quad(quad(
                    bounds, corner_radii, Hsla::transparent_black(), border_widths,
                    colors.top, self.border_style,
                ));
            } else {
                // Reuse the rounded/dashed border shader, partitioned along
                // the joins between outer and inner corners. Rectangular
                // strips would give rounded corners the wrong side colour.
                let clips = Self::vn_border_side_clips(bounds, border_widths);
                for (points, (color, width)) in clips.into_iter().zip([
                    (colors.top, border_widths.top), (colors.right, border_widths.right),
                    (colors.bottom, border_widths.bottom), (colors.left, border_widths.left),
                ]) {
                    if width <= px(0.) || color.is_transparent() || points.len() < 3 { continue; }
                    let mut verbs: Vec<_> = points.into_iter().enumerate().map(|(index, point)| {
                        if index == 0 { VnPathVerb::MoveTo(point) } else { VnPathVerb::LineTo(point) }
                    }).collect();
                    verbs.push(VnPathVerb::Close);
                    let outline = VnClipOutline { fill_rule: VnFillRule::NonZero, verbs };
                    window.vn_with_clip_shape(Some(&outline), |window| {
                        window.paint_quad(quad(bounds, corner_radii, Hsla::transparent_black(),
                            border_widths, color, self.border_style));
                    });
                }
            }
        }

        // `outline` LAST, and at dilated bounds: CSS paints it outside the
        // border box, over everything, and it takes no layout space — so it is
        // one extra border-only quad rather than anything the box itself knows
        // about. `dilate` by (offset + width) puts the ring's INNER edge at
        // `offset` from the border box and its outer edge `width` beyond that,
        // which is what `outline-offset` means.
        //
        // The corner radius grows with the dilation, so the ring stays
        // concentric with a rounded box — the same rule Chromium applies.
        if let Some(outline) = self
            .vn_outline
            .filter(|o| o.width > px(0.) && !o.color.is_transparent() && !ring)
        {
            let grow = outline.offset + outline.width;
            let ring = bounds.dilate(grow);
            // Per corner: a square corner stays square (CSS does not round an
            // outline the box did not round), a rounded one grows with the ring.
            let radii = corner_radii.map(|r| {
                if r.0 <= 0.0 {
                    px(0.)
                } else {
                    px((r.0 + grow.0).max(0.0))
                }
            });
            window.paint_quad(quad(
                ring,
                radii,
                Hsla::transparent_black(),
                Edges::all(outline.width),
                outline.color,
                outline.style,
            ));
        }

        #[cfg(debug_assertions)]
        if self.debug_below {
            cx.remove_global::<DebugBelow>();
        }
    }

    fn vn_resolved_border_colors(&self) -> Edges<Hsla> {
        let shared = self.border_color.unwrap_or_default();
        self.vn_border_colors.map(|color| color.unwrap_or(shared))
    }

    fn vn_border_side_clips(bounds: Bounds<Pixels>, widths: Edges<Pixels>) -> [Vec<[f32; 2]>; 4] {
        let (left, top, right, bottom) = (f32::from(bounds.left()), f32::from(bounds.top()),
            f32::from(bounds.right()), f32::from(bounds.bottom()));
        let widths = [widths.top, widths.right, widths.bottom, widths.left].map(|width| f32::from(width).max(0.));
        let distances = [[0., 1., -top], [-1., 0., right], [0., -1., bottom], [1., 0., -left]];
        std::array::from_fn(|side| {
            if widths[side] <= 0. { return Vec::new(); }
            let mut polygon = vec![[left, top], [right, top], [right, bottom], [left, bottom]];
            // Partition the FULL quad by distance / border width. Stopping
            // at inner rectangular corners cuts holes out of rounded arcs.
            // These bisectors keep the CSS joins through the inner corners
            // while extending far enough to cover circles and large radii.
            for other in 0..4 {
                if other == side || widths[other] <= 0. { continue; }
                let plane: [f32; 3] = std::array::from_fn(|index|
                    distances[side][index] * widths[other] - distances[other][index] * widths[side]);
                let distance = |point: [f32; 2]| plane[0] * point[0] + plane[1] * point[1] + plane[2];
                let Some(mut previous) = polygon.last().copied() else { break };
                let mut previous_distance = distance(previous);
                let mut clipped = Vec::with_capacity(polygon.len() + 1);
                for current in polygon {
                    let current_distance = distance(current);
                    if (previous_distance <= 0.) != (current_distance <= 0.) {
                        let fraction = previous_distance / (previous_distance - current_distance);
                        clipped.push([previous[0] + fraction * (current[0] - previous[0]),
                            previous[1] + fraction * (current[1] - previous[1])]);
                    }
                    if current_distance <= 0. { clipped.push(current); }
                    previous = current;
                    previous_distance = current_distance;
                }
                polygon = clipped;
            }
            polygon
        })
    }

    fn is_border_visible(&self) -> bool {
        self.vn_resolved_border_colors().any(|color| !color.is_transparent())
            && self.border_widths.any(|length| !length.is_zero())
    }
}

impl Default for Style {
    fn default() -> Self {
        Style {
            display: Display::Block,
            visibility: Visibility::Visible,
            overflow: Point {
                x: Overflow::Visible,
                y: Overflow::Visible,
            },
            allow_concurrent_scroll: false,
            restrict_scroll_to_axis: false,
            scrollbar_width: AbsoluteLength::default(),
            position: Position::Relative,
            inset: Edges::auto(),
            margin: Edges::<Length>::zero(),
            padding: Edges::<DefiniteLength>::zero(),
            border_widths: Edges::<AbsoluteLength>::zero(),
            size: Size::auto(),
            min_size: Size::auto(),
            max_size: Size::auto(),
            aspect_ratio: None,
            gap: Size::default(),
            // Alignment
            align_items: None,
            align_self: None,
            align_content: None,
            justify_content: None,
            // Flexbox
            flex_direction: FlexDirection::Row,
            flex_wrap: FlexWrap::NoWrap,
            flex_grow: 0.0,
            flex_shrink: 1.0,
            flex_basis: Length::Auto,
            background: None,
            background_layers: Vec::new(),
            border_color: None,
            vn_border_colors: Edges::default(),
            border_style: BorderStyle::default(),
            corner_radii: Corners::default(),
            box_shadow: Default::default(),
            vn_shadow_under_box: false,
            backdrop_blur: None,
            vn_effect: None,
            vn_transform: None,
            text: TextStyleRefinement::default(),
            mouse_cursor: None,
            opacity: None,
            grid_rows: None,
            grid_cols: None,
            grid_location: None,
            vn_grid_cols: None,
            vn_grid_rows: None,
            vn_grid_cols_auto: None,
            vn_grid_rows_auto: None,
            vn_grid_auto_flow: None,
            vn_justify_items: None,
            vn_justify_self: None,
            vn_outline: None,
            vn_mask_ring: false,
            vn_clip_path: VnClipPath::None,
            vn_color_matrix: None,
            vn_text_fill: None,
            vn_corner_radii: Corners::default(),

            #[cfg(debug_assertions)]
            debug: false,
            #[cfg(debug_assertions)]
            debug_below: false,
        }
    }
}

/// The properties that can be applied to an underline.
#[derive(
    Refineable, Copy, Clone, Default, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct UnderlineStyle {
    /// The thickness of the underline.
    pub thickness: Pixels,

    /// The color of the underline.
    pub color: Option<Hsla>,

    /// Whether the underline should be wavy, like in a spell checker.
    pub wavy: bool,

    /// How far BELOW its default position the line sits — CSS
    /// `text-underline-offset`. Zero keeps the position the font's own metrics
    /// put it at, which is what every existing caller gets.
    pub offset: Pixels,
}

/// The properties that can be applied to a strikethrough.
#[derive(
    Refineable, Copy, Clone, Default, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema,
)]
pub struct StrikethroughStyle {
    /// The thickness of the strikethrough.
    pub thickness: Pixels,

    /// The color of the strikethrough.
    pub color: Option<Hsla>,
}

/// The kinds of fill that can be applied to a shape.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum Fill {
    /// A solid color fill.
    Color(Background),
}

impl Fill {
    /// Unwrap this fill into a solid color, if it is one.
    ///
    /// If the fill is not a solid color, this method returns `None`.
    pub fn color(&self) -> Option<Background> {
        match self {
            Fill::Color(color) => Some(*color),
        }
    }
}

impl Default for Fill {
    fn default() -> Self {
        Self::Color(Background::default())
    }
}

impl From<Hsla> for Fill {
    fn from(color: Hsla) -> Self {
        Self::Color(color.into())
    }
}

impl From<Rgba> for Fill {
    fn from(color: Rgba) -> Self {
        Self::Color(color.into())
    }
}

impl From<Background> for Fill {
    fn from(background: Background) -> Self {
        Self::Color(background)
    }
}

impl From<TextStyle> for HighlightStyle {
    fn from(other: TextStyle) -> Self {
        Self::from(&other)
    }
}

impl From<&TextStyle> for HighlightStyle {
    fn from(other: &TextStyle) -> Self {
        Self {
            color: Some(other.color),
            font_weight: Some(other.font_weight),
            font_style: Some(other.font_style),
            background_color: other.background_color,
            underline: other.underline,
            strikethrough: other.strikethrough,
            fade_out: None,
        }
    }
}

impl HighlightStyle {
    /// Create a highlight style with just a color
    pub fn color(color: Hsla) -> Self {
        Self {
            color: Some(color),
            ..Default::default()
        }
    }
    /// Blend this highlight style with another.
    /// Non-continuous properties, like font_weight and font_style, are overwritten.
    #[must_use]
    pub fn highlight(self, other: HighlightStyle) -> Self {
        Self {
            color: other
                .color
                .map(|other_color| {
                    if let Some(color) = self.color {
                        color.blend(other_color)
                    } else {
                        other_color
                    }
                })
                .or(self.color),
            font_weight: other.font_weight.or(self.font_weight),
            font_style: other.font_style.or(self.font_style),
            background_color: other.background_color.or(self.background_color),
            underline: other.underline.or(self.underline),
            strikethrough: other.strikethrough.or(self.strikethrough),
            fade_out: other
                .fade_out
                .map(|source_fade| {
                    self.fade_out
                        .map(|dest_fade| (dest_fade * (1. + source_fade)).clamp(0., 1.))
                        .unwrap_or(source_fade)
                })
                .or(self.fade_out),
        }
    }
}

impl From<Hsla> for HighlightStyle {
    fn from(color: Hsla) -> Self {
        Self {
            color: Some(color),
            ..Default::default()
        }
    }
}

impl From<FontWeight> for HighlightStyle {
    fn from(font_weight: FontWeight) -> Self {
        Self {
            font_weight: Some(font_weight),
            ..Default::default()
        }
    }
}

impl From<FontStyle> for HighlightStyle {
    fn from(font_style: FontStyle) -> Self {
        Self {
            font_style: Some(font_style),
            ..Default::default()
        }
    }
}

impl From<Rgba> for HighlightStyle {
    fn from(color: Rgba) -> Self {
        Self {
            color: Some(color.into()),
            ..Default::default()
        }
    }
}

/// Combine and merge the highlights and ranges in the two iterators.
pub fn combine_highlights(
    a: impl IntoIterator<Item = (Range<usize>, HighlightStyle)>,
    b: impl IntoIterator<Item = (Range<usize>, HighlightStyle)>,
) -> impl Iterator<Item = (Range<usize>, HighlightStyle)> {
    let mut endpoints = Vec::new();
    let mut highlights = Vec::new();
    for (range, highlight) in a.into_iter().chain(b) {
        if !range.is_empty() {
            let highlight_id = highlights.len();
            endpoints.push((range.start, highlight_id, true));
            endpoints.push((range.end, highlight_id, false));
            highlights.push(highlight);
        }
    }
    endpoints.sort_unstable_by_key(|(position, _, _)| *position);
    let mut endpoints = endpoints.into_iter().peekable();

    let mut active_styles = HashSet::default();
    let mut ix = 0;
    iter::from_fn(move || {
        while let Some((endpoint_ix, highlight_id, is_start)) = endpoints.peek() {
            let prev_index = mem::replace(&mut ix, *endpoint_ix);
            if ix > prev_index && !active_styles.is_empty() {
                let current_style = active_styles
                    .iter()
                    .fold(HighlightStyle::default(), |acc, highlight_id| {
                        acc.highlight(highlights[*highlight_id])
                    });
                return Some((prev_index..ix, current_style));
            }

            if *is_start {
                active_styles.insert(*highlight_id);
            } else {
                active_styles.remove(highlight_id);
            }
            endpoints.next();
        }
        None
    })
}

/// Used to control how child nodes are aligned.
/// For Flexbox it controls alignment in the cross axis
/// For Grid it controls alignment in the block axis
///
/// [MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/align-items)
#[derive(Copy, Clone, PartialEq, Eq, Debug, Serialize, Deserialize, JsonSchema)]
// Copy of taffy::style type of the same name, to derive JsonSchema.
pub enum AlignItems {
    /// Items are packed toward the start of the axis
    Start,
    /// Items are packed toward the end of the axis
    End,
    /// Items are packed towards the flex-relative start of the axis.
    ///
    /// For flex containers with flex_direction RowReverse or ColumnReverse this is equivalent
    /// to End. In all other cases it is equivalent to Start.
    FlexStart,
    /// Items are packed towards the flex-relative end of the axis.
    ///
    /// For flex containers with flex_direction RowReverse or ColumnReverse this is equivalent
    /// to Start. In all other cases it is equivalent to End.
    FlexEnd,
    /// Items are packed along the center of the cross axis
    Center,
    /// Items are aligned such as their baselines align
    Baseline,
    /// Stretch to fill the container
    Stretch,
}
/// Used to control how child nodes are aligned.
/// Does not apply to Flexbox, and will be ignored if specified on a flex container
/// For Grid it controls alignment in the inline axis
///
/// [MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/justify-items)
pub type JustifyItems = AlignItems;
/// Used to control how the specified nodes is aligned.
/// Overrides the parent Node's `AlignItems` property.
/// For Flexbox it controls alignment in the cross axis
/// For Grid it controls alignment in the block axis
///
/// [MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/align-self)
pub type AlignSelf = AlignItems;
/// Used to control how the specified nodes is aligned.
/// Overrides the parent Node's `JustifyItems` property.
/// Does not apply to Flexbox, and will be ignored if specified on a flex child
/// For Grid it controls alignment in the inline axis
///
/// [MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/justify-self)
pub type JustifySelf = AlignItems;

/// Sets the distribution of space between and around content items
/// For Flexbox it controls alignment in the cross axis
/// For Grid it controls alignment in the block axis
///
/// [MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/align-content)
#[derive(Copy, Clone, PartialEq, Eq, Debug, Serialize, Deserialize, JsonSchema)]
// Copy of taffy::style type of the same name, to derive JsonSchema.
pub enum AlignContent {
    /// Items are packed toward the start of the axis
    Start,
    /// Items are packed toward the end of the axis
    End,
    /// Items are packed towards the flex-relative start of the axis.
    ///
    /// For flex containers with flex_direction RowReverse or ColumnReverse this is equivalent
    /// to End. In all other cases it is equivalent to Start.
    FlexStart,
    /// Items are packed towards the flex-relative end of the axis.
    ///
    /// For flex containers with flex_direction RowReverse or ColumnReverse this is equivalent
    /// to Start. In all other cases it is equivalent to End.
    FlexEnd,
    /// Items are centered around the middle of the axis
    Center,
    /// Items are stretched to fill the container
    Stretch,
    /// The first and last items are aligned flush with the edges of the container (no gap)
    /// The gap between items is distributed evenly.
    SpaceBetween,
    /// The gap between the first and last items is exactly THE SAME as the gap between items.
    /// The gaps are distributed evenly
    SpaceEvenly,
    /// The gap between the first and last items is exactly HALF the gap between items.
    /// The gaps are distributed evenly in proportion to these ratios.
    SpaceAround,
    /// vue-native fork: CSS `safe center`. Centred while the content fits, and
    /// packed toward the start when it would overflow. HTML lays a `<button>`'s
    /// content out this way.
    SafeCenter,
}

/// Sets the distribution of space between and around content items
/// For Flexbox it controls alignment in the main axis
/// For Grid it controls alignment in the inline axis
///
/// [MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/justify-content)
pub type JustifyContent = AlignContent;

/// Sets the layout used for the children of this node
///
/// The default values depends on on which feature flags are enabled. The order of precedence is: Flex, Grid, Block, None.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default, Serialize, Deserialize, JsonSchema)]
// Copy of taffy::style type of the same name, to derive JsonSchema.
pub enum Display {
    /// The children will follow the block layout algorithm
    Block,
    /// The children will follow the flexbox layout algorithm
    #[default]
    Flex,
    /// The children will follow the CSS Grid layout algorithm
    Grid,
    /// The children will not be laid out, and will follow absolute positioning
    None,
}

/// Controls whether flex items are forced onto one line or can wrap onto multiple lines.
///
/// Defaults to [`FlexWrap::NoWrap`]
///
/// [Specification](https://www.w3.org/TR/css-flexbox-1/#flex-wrap-property)
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default, Serialize, Deserialize, JsonSchema)]
// Copy of taffy::style type of the same name, to derive JsonSchema.
pub enum FlexWrap {
    /// Items will not wrap and stay on a single line
    #[default]
    NoWrap,
    /// Items will wrap according to this item's [`FlexDirection`]
    Wrap,
    /// Items will wrap in the opposite direction to this item's [`FlexDirection`]
    WrapReverse,
}

/// The direction of the flexbox layout main axis.
///
/// There are always two perpendicular layout axes: main (or primary) and cross (or secondary).
/// Adding items will cause them to be positioned adjacent to each other along the main axis.
/// By varying this value throughout your tree, you can create complex axis-aligned layouts.
///
/// Items are always aligned relative to the cross axis, and justified relative to the main axis.
///
/// The default behavior is [`FlexDirection::Row`].
///
/// [Specification](https://www.w3.org/TR/css-flexbox-1/#flex-direction-property)
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default, Serialize, Deserialize, JsonSchema)]
// Copy of taffy::style type of the same name, to derive JsonSchema.
pub enum FlexDirection {
    /// Defines +x as the main axis
    ///
    /// Items will be added from left to right in a row.
    #[default]
    Row,
    /// Defines +y as the main axis
    ///
    /// Items will be added from top to bottom in a column.
    Column,
    /// Defines -x as the main axis
    ///
    /// Items will be added from right to left in a row.
    RowReverse,
    /// Defines -y as the main axis
    ///
    /// Items will be added from bottom to top in a column.
    ColumnReverse,
}

/// How children overflowing their container should affect layout
///
/// In CSS the primary effect of this property is to control whether contents of a parent container that overflow that container should
/// be displayed anyway, be clipped, or trigger the container to become a scroll container. However it also has secondary effects on layout,
/// the main ones being:
///
///   - The automatic minimum size Flexbox/CSS Grid items with non-`Visible` overflow is `0` rather than being content based
///   - `Overflow::Scroll` nodes have space in the layout reserved for a scrollbar (width controlled by the `scrollbar_width` property)
///
/// In Taffy, we only implement the layout related secondary effects as we are not concerned with drawing/painting. The amount of space reserved for
/// a scrollbar is controlled by the `scrollbar_width` property. If this is `0` then `Scroll` behaves identically to `Hidden`.
///
/// <https://developer.mozilla.org/en-US/docs/Web/CSS/overflow>
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default, Serialize, Deserialize, JsonSchema)]
// Copy of taffy::style type of the same name, to derive JsonSchema.
pub enum Overflow {
    /// The automatic minimum size of this node as a flexbox/grid item should be based on the size of its content.
    /// Content that overflows this node *should* contribute to the scroll region of its parent.
    #[default]
    Visible,
    /// The automatic minimum size of this node as a flexbox/grid item should be based on the size of its content.
    /// Content that overflows this node should *not* contribute to the scroll region of its parent.
    Clip,
    /// The automatic minimum size of this node as a flexbox/grid item should be `0`.
    /// Content that overflows this node should *not* contribute to the scroll region of its parent.
    Hidden,
    /// The automatic minimum size of this node as a flexbox/grid item should be `0`. Additionally, space should be reserved
    /// for a scrollbar. The amount of space reserved is controlled by the `scrollbar_width` property.
    /// Content that overflows this node should *not* contribute to the scroll region of its parent.
    Scroll,
}

/// The positioning strategy for this item.
///
/// This controls both how the origin is determined for the [`Style::position`] field,
/// and whether or not the item will be controlled by flexbox's layout algorithm.
///
/// WARNING: this enum follows the behavior of [CSS's `position` property](https://developer.mozilla.org/en-US/docs/Web/CSS/position),
/// which can be unintuitive.
///
/// [`Position::Relative`] is the default value, in contrast to the default behavior in CSS.
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default, Serialize, Deserialize, JsonSchema)]
// Copy of taffy::style type of the same name, to derive JsonSchema.
pub enum Position {
    /// The offset is computed relative to the final position given by the layout algorithm.
    /// Offsets do not affect the position of any other items; they are effectively a correction factor applied at the end.
    #[default]
    Relative,
    /// The offset is computed relative to this item's closest positioned ancestor, if any.
    /// Otherwise, it is placed relative to the origin.
    /// No space is created for the item in the page layout, and its size will not be altered.
    ///
    /// WARNING: to opt-out of layouting entirely, you must use [`Display::None`] instead on your [`Style`] object.
    Absolute,
}

impl From<AlignItems> for taffy::style::AlignItems {
    fn from(value: AlignItems) -> Self {
        match value {
            AlignItems::Start => Self::START,
            AlignItems::End => Self::END,
            AlignItems::FlexStart => Self::FLEX_START,
            AlignItems::FlexEnd => Self::FLEX_END,
            AlignItems::Center => Self::CENTER,
            AlignItems::Baseline => Self::BASELINE,
            AlignItems::Stretch => Self::STRETCH,
        }
    }
}

impl From<AlignContent> for taffy::style::AlignContent {
    fn from(value: AlignContent) -> Self {
        match value {
            AlignContent::Start => Self::START,
            AlignContent::End => Self::END,
            AlignContent::FlexStart => Self::FLEX_START,
            AlignContent::FlexEnd => Self::FLEX_END,
            AlignContent::Center => Self::CENTER,
            AlignContent::Stretch => Self::STRETCH,
            AlignContent::SpaceBetween => Self::SPACE_BETWEEN,
            AlignContent::SpaceEvenly => Self::SPACE_EVENLY,
            AlignContent::SpaceAround => Self::SPACE_AROUND,
            AlignContent::SafeCenter => Self::SAFE_CENTER,
        }
    }
}

impl From<Display> for taffy::style::Display {
    fn from(value: Display) -> Self {
        match value {
            Display::Block => Self::Block,
            Display::Flex => Self::Flex,
            Display::Grid => Self::Grid,
            Display::None => Self::None,
        }
    }
}

impl From<FlexWrap> for taffy::style::FlexWrap {
    fn from(value: FlexWrap) -> Self {
        match value {
            FlexWrap::NoWrap => Self::NoWrap,
            FlexWrap::Wrap => Self::Wrap,
            FlexWrap::WrapReverse => Self::WrapReverse,
        }
    }
}

impl From<FlexDirection> for taffy::style::FlexDirection {
    fn from(value: FlexDirection) -> Self {
        match value {
            FlexDirection::Row => Self::Row,
            FlexDirection::Column => Self::Column,
            FlexDirection::RowReverse => Self::RowReverse,
            FlexDirection::ColumnReverse => Self::ColumnReverse,
        }
    }
}

impl From<Overflow> for taffy::style::Overflow {
    fn from(value: Overflow) -> Self {
        match value {
            Overflow::Visible => Self::Visible,
            Overflow::Clip => Self::Clip,
            Overflow::Hidden => Self::Hidden,
            Overflow::Scroll => Self::Scroll,
        }
    }
}

impl From<Position> for taffy::style::Position {
    fn from(value: Position) -> Self {
        match value {
            Position::Relative => Self::Relative,
            Position::Absolute => Self::Absolute,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{blue, green, px, red, yellow};

    use super::*;

    use util_macros::perf;

    #[perf]
    fn test_basic_highlight_style_combination() {
        let style_a = HighlightStyle::default();
        let style_b = HighlightStyle::default();
        let style_a = style_a.highlight(style_b);
        assert_eq!(
            style_a,
            HighlightStyle::default(),
            "Combining empty styles should not produce a non-empty style."
        );

        let mut style_b = HighlightStyle {
            color: Some(red()),
            strikethrough: Some(StrikethroughStyle {
                thickness: px(2.),
                color: Some(blue()),
            }),
            fade_out: Some(0.),
            font_style: Some(FontStyle::Italic),
            font_weight: Some(FontWeight(300.)),
            background_color: Some(yellow()),
            underline: Some(UnderlineStyle {
                thickness: px(2.),
                color: Some(red()),
                wavy: true,
                offset: px(0.),
            }),
        };
        let expected_style = style_b;

        let style_a = style_a.highlight(style_b);
        assert_eq!(
            style_a, expected_style,
            "Blending an empty style with another style should return the other style"
        );

        let style_b = style_b.highlight(Default::default());
        assert_eq!(
            style_b, expected_style,
            "Blending a style with an empty style should not change the style."
        );

        let mut style_c = expected_style;

        let style_d = HighlightStyle {
            color: Some(blue().alpha(0.7)),
            strikethrough: Some(StrikethroughStyle {
                thickness: px(4.),
                color: Some(crate::red()),
            }),
            fade_out: Some(0.),
            font_style: Some(FontStyle::Oblique),
            font_weight: Some(FontWeight(800.)),
            background_color: Some(green()),
            underline: Some(UnderlineStyle {
                thickness: px(4.),
                color: None,
                wavy: false,
                offset: px(0.),
            }),
        };

        let expected_style = HighlightStyle {
            color: Some(red().blend(blue().alpha(0.7))),
            strikethrough: Some(StrikethroughStyle {
                thickness: px(4.),
                color: Some(red()),
            }),
            // TODO this does not seem right
            fade_out: Some(0.),
            font_style: Some(FontStyle::Oblique),
            font_weight: Some(FontWeight(800.)),
            background_color: Some(green()),
            underline: Some(UnderlineStyle {
                thickness: px(4.),
                color: None,
                wavy: false,
                offset: px(0.),
            }),
        };

        let style_c = style_c.highlight(style_d);
        assert_eq!(
            style_c, expected_style,
            "Blending styles should blend properties where possible and override all others"
        );
    }

    #[perf]
    fn test_combine_highlights() {
        assert_eq!(
            combine_highlights(
                [
                    (0..5, green().into()),
                    (4..10, FontWeight::BOLD.into()),
                    (15..20, yellow().into()),
                ],
                [
                    (2..6, FontStyle::Italic.into()),
                    (1..3, blue().into()),
                    (21..23, red().into()),
                ]
            )
            .collect::<Vec<_>>(),
            [
                (
                    0..1,
                    HighlightStyle {
                        color: Some(green()),
                        ..Default::default()
                    }
                ),
                (
                    1..2,
                    HighlightStyle {
                        color: Some(blue()),
                        ..Default::default()
                    }
                ),
                (
                    2..3,
                    HighlightStyle {
                        color: Some(blue()),
                        font_style: Some(FontStyle::Italic),
                        ..Default::default()
                    }
                ),
                (
                    3..4,
                    HighlightStyle {
                        color: Some(green()),
                        font_style: Some(FontStyle::Italic),
                        ..Default::default()
                    }
                ),
                (
                    4..5,
                    HighlightStyle {
                        color: Some(green()),
                        font_weight: Some(FontWeight::BOLD),
                        font_style: Some(FontStyle::Italic),
                        ..Default::default()
                    }
                ),
                (
                    5..6,
                    HighlightStyle {
                        font_weight: Some(FontWeight::BOLD),
                        font_style: Some(FontStyle::Italic),
                        ..Default::default()
                    }
                ),
                (
                    6..10,
                    HighlightStyle {
                        font_weight: Some(FontWeight::BOLD),
                        ..Default::default()
                    }
                ),
                (
                    15..20,
                    HighlightStyle {
                        color: Some(yellow()),
                        ..Default::default()
                    }
                ),
                (
                    21..23,
                    HighlightStyle {
                        color: Some(red()),
                        ..Default::default()
                    }
                )
            ]
        );
    }

    #[perf]
    fn test_text_style_refinement() {
        let mut style = Style::default();
        style.refine(&StyleRefinement::default().text_size(px(20.0)));
        style.refine(&StyleRefinement::default().font_weight(FontWeight::SEMIBOLD));

        assert_eq!(
            Some(AbsoluteLength::from(px(20.0))),
            style.text_style().unwrap().font_size
        );

        assert_eq!(
            Some(FontWeight::SEMIBOLD),
            style.text_style().unwrap().font_weight
        );
    }
}

/// vue-native fork: `AlignContent::SafeCenter` reaches taffy as `safe center`.
/// A block container centres content that fits and starts content that
/// overflows.
#[cfg(test)]
mod vn_safe_center_tests {
    use crate::{AlignContent, Context, TestAppContext, Window, div, prelude::*, px, size};

    struct Boxes;

    impl Render for Boxes {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let block = |name: &'static str, child_h: f32| {
                let mut d = div().w(px(100.)).h(px(100.));
                d.style().align_content = Some(AlignContent::SafeCenter);
                d.child(div().debug_selector(move || name.into()).w(px(10.)).h(px(child_h)))
            };
            div().child(block("FITS", 20.)).child(block("OVERFLOWS", 200.))
        }
    }

    #[gpui::test]
    fn safe_center_centres_what_fits_and_starts_what_overflows(cx: &mut TestAppContext) {
        let window = cx.open_window(size(px(400.), px(600.)), |_, _| Boxes);
        cx.run_until_parked();
        window
            .update(cx, |_, window, _| {
                let at = |name: &str| window.rendered_frame.debug_bounds.get(name).copied().unwrap();
                assert_eq!(at("FITS").origin.y, px(40.), "(100 - 20) / 2");
                assert_eq!(at("OVERFLOWS").origin.y, px(100.), "at the start of the second box");
            })
            .unwrap();
    }
}

/// vue-native fork: `Style::vn_mask_ring`.
#[cfg(test)]
mod vn_mask_ring_tests {
    use crate::{
        BoxShadow, Context, Hsla, TestAppContext, Window, color::BG_RING, div, point,
        prelude::*, px, size,
    };

    struct Card {
        ring: bool,
    }

    impl Render for Card {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let mut d = div()
                .w(px(100.))
                .h(px(60.))
                .p(px(2.))
                .border_1()
                .border_color(Hsla::white())
                .bg(Hsla::red())
                .shadow(vec![BoxShadow {
                    color: Hsla::black(),
                    offset: point(px(0.), px(4.)),
                    blur_radius: px(8.),
                    spread_radius: px(0.),
                    inset: false,
                }]);
            d.style().vn_mask_ring = Some(self.ring);
            d
        }
    }

    /// The painted quads, and the frame's shadow count.
    fn paint(ring: bool, cx: &mut TestAppContext) -> (Vec<crate::Quad>, usize, f32) {
        let window = cx.open_window(size(px(300.), px(300.)), move |_, _| Card { ring });
        cx.run_until_parked();
        window
            .update(cx, |_, window, _| {
                let shadows = window.rendered_frame.scene.shadows.len();
                (window.painted_quads(), shadows, window.scale_factor())
            })
            .unwrap()
    }

    #[gpui::test]
    fn a_ring_mask_paints_the_background_in_the_padding_and_border_only(
        cx: &mut TestAppContext,
    ) {
        let (quads, shadows, scale) = paint(false, cx);
        let background = quads.iter().find(|q| q.background.as_solid() == Some(Hsla::red()));
        let background = background.expect("the background quad");
        assert_eq!(background.background.flags & BG_RING, 0);
        assert_eq!(background.border_widths.top.0, 0.0, "a plain background fills the box");
        assert_eq!(shadows, 1);

        let (quads, shadows, _) = paint(true, cx);
        let background = quads
            .iter()
            .find(|q| q.background.flags & BG_RING != 0)
            .expect("the background quad, flagged");
        assert_eq!(background.background.as_solid(), Some(Hsla::red()));
        for width in [
            background.border_widths.top,
            background.border_widths.right,
            background.border_widths.bottom,
            background.border_widths.left,
        ] {
            assert_eq!(width.0, 3.0 * scale, "1px border + 2px padding");
        }
        let border = quads
            .iter()
            .find(|q| q.border_color == Hsla::white())
            .expect("the border still paints");
        assert_eq!(border.border_widths.top.0, 1.0 * scale);
        assert_eq!(shadows, 0, "the mask cuts the outer shadow");
    }
}

/// vue-native fork: `overflow` under a `border-radius` clips to the rounded
/// padding box, and what is painted inside carries that mask.
#[cfg(test)]
mod vn_rounded_clip_tests {
    use crate::{
        Bounds, Context, Corners, DefiniteLength, Edges, Hsla, Overflow, Style, TestAppContext,
        VnCornerRadius, Window, div, point, prelude::*, px, size, vn_resolve_corner_radii,
        vn_take_ellipse_drawn_as_circle,
    };

    fn clipping(radius: f32, border: f32, border_color: Option<Hsla>) -> Style {
        Style {
            overflow: point(Overflow::Hidden, Overflow::Hidden),
            corner_radii: Corners::all(px(radius)).map(|radius| (*radius).into()),
            border_widths: Edges::all(px(border).into()),
            border_color,
            ..Style::default()
        }
    }

    #[test]
    fn an_overflow_clip_takes_the_padding_box_curve() {
        let rem = px(16.);
        let bounds = Bounds::new(point(px(10.), px(20.)), size(px(200.), px(160.)));

        let mask = clipping(24., 4., Some(Hsla::black())).overflow_mask(bounds, rem).unwrap();
        assert_eq!(mask.bounds, Bounds::new(point(px(14.), px(24.)), size(px(192.), px(152.))));
        assert_eq!(mask.corner_radii, Corners::all(px(20.)), "the radius less the border");

        let mask = clipping(24., 4., None).overflow_mask(bounds, rem).unwrap();
        assert_eq!(mask.bounds, bounds, "a transparent border is not inset");
        assert_eq!(mask.corner_radii, Corners::all(px(24.)), "so it takes nothing off");

        let mask = clipping(3., 4., Some(Hsla::black())).overflow_mask(bounds, rem).unwrap();
        assert_eq!(mask.corner_radii, Corners::all(px(0.)), "a border wider than the radius");

        let avatar = Bounds::new(point(px(0.), px(0.)), size(px(64.), px(64.)));
        let mask = clipping(9999., 0., None).overflow_mask(avatar, rem).unwrap();
        assert_eq!(mask.corner_radii, Corners::all(px(32.)), "a pill is a circle");

        let mask = clipping(0., 0., None).overflow_mask(bounds, rem).unwrap();
        assert!(!mask.is_rounded(), "no radius, the plain rect");
    }

    fn resolve(lengths: [f32; 4], relative: [Option<(DefiniteLength, DefiniteLength)>; 4], w: f32, h: f32) -> [f32; 4] {
        let corner = |r: Option<(DefiniteLength, DefiniteLength)>| r.map(|(x, y)| VnCornerRadius { x, y });
        let radii = vn_resolve_corner_radii(
            &Corners {
                top_left: px(lengths[0]).into(),
                top_right: px(lengths[1]).into(),
                bottom_right: px(lengths[2]).into(),
                bottom_left: px(lengths[3]).into(),
            },
            &Corners {
                top_left: corner(relative[0]),
                top_right: corner(relative[1]),
                bottom_right: corner(relative[2]),
                bottom_left: corner(relative[3]),
            },
            size(px(w), px(h)),
            px(16.),
        );
        [radii.top_left, radii.top_right, radii.bottom_right, radii.bottom_left].map(f32::from)
    }

    #[test]
    fn percentages_resolve_against_the_box_and_overlaps_scale() {
        let half = Some((DefiniteLength::Fraction(0.5), DefiniteLength::Fraction(0.5)));
        let _ = vn_take_ellipse_drawn_as_circle();

        assert_eq!(resolve([0.; 4], [half; 4], 64., 64.), [32.; 4], "50% of a square: a circle");
        assert!(!vn_take_ellipse_drawn_as_circle());

        // 50% of 120x48 is a 60x24 ellipse at each corner; gpui draws the
        // 24px circle, a pill, and says so.
        assert_eq!(resolve([0.; 4], [half; 4], 120., 48.), [24.; 4]);
        assert!(vn_take_ellipse_drawn_as_circle());

        // `rounded-full`: 9999px on every corner of 100x40 scales by 40/19998.
        assert_eq!(resolve([9999.; 4], [None; 4], 100., 40.), [20.; 4]);
        // `rounded-l-full`: only the left side overruns; both its corners
        // scale, and the right stays square.
        assert_eq!(resolve([9999., 0., 0., 9999.], [None; 4], 100., 40.), [20., 0., 0., 20.]);
        // Radii that fit stay as written.
        assert_eq!(resolve([8., 4., 0., 12.], [None; 4], 100., 40.), [8., 4., 0., 12.]);
        // 40 + 40 across a 60px top: both scale by 60/80, to 30.
        assert_eq!(resolve([40., 40., 0., 0.], [None; 4], 60., 200.), [30., 30., 0., 0.]);

        // `calc(50% - 4px)` and an elliptical `10px / 20px`.
        let calc = DefiniteLength::Calc { fraction: 0.5, pixels: px(-4.) };
        assert_eq!(resolve([0.; 4], [Some((calc, calc)); 4], 64., 64.), [28.; 4]);
        let ellipse = Some((DefiniteLength::Absolute(px(10.).into()), DefiniteLength::Absolute(px(20.).into())));
        assert_eq!(resolve([0.; 4], [ellipse; 4], 100., 100.), [10.; 4]);
        assert!(vn_take_ellipse_drawn_as_circle());
    }

    #[test]
    fn a_percentage_radius_clips_overflow_too() {
        let mut avatar = clipping(0., 0., None);
        let half = Some(VnCornerRadius {
            x: DefiniteLength::Fraction(0.5),
            y: DefiniteLength::Fraction(0.5),
        });
        avatar.vn_corner_radii = Corners {
            top_left: half,
            top_right: half,
            bottom_right: half,
            bottom_left: half,
        };
        let bounds = Bounds::new(point(px(0.), px(0.)), size(px(40.), px(40.)));
        let mask = avatar.overflow_mask(bounds, px(16.)).unwrap();
        assert_eq!(mask.corner_radii, Corners::all(px(20.)));
    }

    /// reka's AvatarRoot around its image, and a rounded card around a list
    /// that scrolls.
    struct Clips;

    impl Render for Clips {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            div()
                .child(
                    div()
                        .size(px(64.))
                        .rounded_full()
                        .overflow_hidden()
                        .child(div().size_full().bg(Hsla::red())),
                )
                .child(
                    div()
                        .w(px(200.))
                        .h(px(160.))
                        .rounded(px(24.))
                        .overflow_hidden()
                        .child(
                            div()
                                .id("list")
                                .size_full()
                                .overflow_y_scroll()
                                .child(div().w_full().h(px(400.)).bg(Hsla::green())),
                        ),
                )
        }
    }

    #[gpui::test]
    fn what_paints_inside_carries_the_rounded_mask(cx: &mut TestAppContext) {
        let window = cx.open_window(size(px(400.), px(400.)), |_, _| Clips);
        cx.run_until_parked();
        let (quads, scale) = window
            .update(cx, |_, window, _| (window.painted_quads(), window.scale_factor()))
            .unwrap();
        let quad = |color: Hsla| {
            quads
                .iter()
                .find(|q| q.background.as_solid() == Some(color))
                .copied()
                .expect("the quad")
        };

        let avatar = quad(Hsla::red());
        assert_eq!(avatar.content_mask.corner_radii, Corners::all(px(32.)).scale(scale));
        assert_eq!(avatar.corner_radii, Corners::default(), "the image itself is square");

        // The scroller's own mask is square and fills the card, so its
        // corners are the card's: the rows clip at the card's curve.
        let rows = quad(Hsla::green());
        assert_eq!(rows.content_mask.corner_radii, Corners::all(px(24.)).scale(scale));
        assert_eq!(rows.content_mask.bounds.size, size(px(200.), px(160.)).scale(scale));
    }
}

/// vue-native fork: `clip-path` is a rounded mask around the element's whole
/// painting (`Style::vn_clip_path`, `Style::vn_clip_mask`).
#[cfg(test)]
mod vn_clip_path_tests {
    use crate::{
        Bounds, BoxShadow, Context, Corners, DefiniteLength, Edges, Hsla, Length, Pixels, Style,
        TestAppContext, VnClipBox, VnClipPath, VnCornerRadius, VnShapeRadius, Window, div, point,
        prelude::*, px, relative, size,
    };

    const BOX: VnClipBox = VnClipBox::Border;

    fn len(v: f32) -> DefiniteLength {
        px(v).into()
    }

    fn round(x: DefiniteLength, y: DefiniteLength) -> [VnCornerRadius; 4] {
        [VnCornerRadius { x, y }; 4]
    }

    /// The mask's bounds, as `[x, y, w, h]`, and its four radii, to 0.01px.
    fn mask(clip: VnClipPath, style: Style, bounds: [f32; 4]) -> ([f32; 4], [f32; 4]) {
        let bounds = Bounds::new(point(px(bounds[0]), px(bounds[1])), size(px(bounds[2]), px(bounds[3])));
        let m = Style { vn_clip_path: clip, ..style }.vn_clip_mask(bounds, px(16.)).expect("a mask");
        let (b, r) = (m.bounds, m.corner_radii);
        let at = |v: Pixels| (f32::from(v) * 100.).round() / 100.;
        (
            [b.origin.x, b.origin.y, b.size.width, b.size.height].map(at),
            [r.top_left, r.top_right, r.bottom_right, r.bottom_left].map(at),
        )
    }

    fn inset(insets: [DefiniteLength; 4], radii: [VnCornerRadius; 4]) -> VnClipPath {
        VnClipPath::Inset { reference: BOX, insets, radii }
    }

    fn ellipse(center: [DefiniteLength; 2], radius: [VnShapeRadius; 2], circle: bool) -> VnClipPath {
        VnClipPath::Ellipse { reference: BOX, center, radius, circle }
    }

    #[test]
    fn inset_takes_its_insets_off_the_box_and_fits_its_radii() {
        let none = Style::default;
        let square = round(len(0.), len(0.));
        assert_eq!(Style::default().vn_clip_mask(Bounds::default(), px(16.)), None, "none is no mask");
        // `inset(10px 20% round 8px)` on a 200x100 box at (10, 20).
        let insets = [len(10.), relative(0.2), len(10.), relative(0.2)];
        assert_eq!(
            mask(inset(insets, round(len(8.), len(8.))), none(), [10., 20., 200., 100.]),
            ([50., 30., 120., 80.], [8.; 4])
        );
        // The screen-reader-only `inset(50%)` encloses nothing.
        let half = relative(0.5);
        assert_eq!(mask(inset([half; 4], square), none(), [0., 0., 30., 10.]).0, [15., 5., 0., 0.]);
        // Insets past the box leave it empty, not inside out.
        assert_eq!(mask(inset([len(80.); 4], square), none(), [0., 0., 100., 100.]).0, [80., 80., 0., 0.]);
        // `round 30px` on a 200x20 strip scales by 20 / (30 + 30). A
        // percentage radius is of the REFERENCE box: 10% of 200x100 is 20 by
        // 10, drawn as the 10px circle.
        let strip = [len(40.), len(0.), len(40.), len(0.)];
        assert_eq!(mask(inset(strip, round(len(30.), len(30.))), none(), [0., 0., 200., 100.]).1, [10.; 4]);
        let ten = relative(0.1);
        assert_eq!(mask(inset([len(0.); 4], round(ten, ten)), none(), [0., 0., 200., 100.]).1, [10.; 4]);
    }

    #[test]
    fn circle_and_ellipse_resolve_against_the_box() {
        let none = Style::default;
        let mid = [relative(0.5), relative(0.5)];
        // `circle(50%)` on a square is its disc.
        let circle = ellipse(mid, [VnShapeRadius::Length(relative(0.5)); 2], true);
        assert_eq!(mask(circle.clone(), none(), [0., 0., 60., 60.]), ([0., 0., 60., 60.], [30.; 4]));
        // On 200x100 the percentage is of sqrt(200^2 + 100^2) / sqrt(2), 158.11.
        assert_eq!(mask(circle, none(), [0., 0., 200., 100.]).1, [79.06; 4]);
        // `circle(closest-side at 25% 50%)`: 50 to the left, top and bottom.
        let at = [relative(0.25), relative(0.5)];
        assert_eq!(
            mask(ellipse(at, [VnShapeRadius::ClosestSide; 2], true), none(), [0., 0., 200., 100.]),
            ([0., 0., 100., 100.], [50.; 4])
        );
        let far = ellipse(at, [VnShapeRadius::FarthestSide; 2], true);
        assert_eq!(mask(far, none(), [0., 0., 200., 100.]).1, [150.; 4]);
        // `ellipse(closest-side farthest-side)`: each radius in its own axis.
        // The mask is its box; the ellipse is the outline (an isolated
        // layer), so it is exact.
        let e = ellipse(mid, [VnShapeRadius::ClosestSide, VnShapeRadius::FarthestSide], false);
        assert_eq!(mask(e.clone(), none(), [0., 0., 200., 100.]), ([0., 0., 200., 100.], [0.; 4]));
        let bounds = Bounds::new(point(px(0.), px(0.)), size(px(200.), px(100.)));
        let outline = Style { vn_clip_path: e, ..none() }.vn_clip_outline(bounds, px(16.)).expect("an outline");
        let g = crate::VnClipGeometry::new(&outline, 1., crate::TransformationMatrix::unit());
        assert!(g.contains([100., 50.]) && g.contains([2., 50.]) && g.contains([100., 2.]));
        assert!(!g.contains([20., 10.]), "the corner a capsule would keep");
        // `ellipse(10px 20% at right 10px center)`: `right 10px` is 100% - 10px.
        let right = [DefiniteLength::Calc { fraction: 1., pixels: px(-10.) }, relative(0.5)];
        let e = ellipse(right, [VnShapeRadius::Length(len(10.)), VnShapeRadius::Length(relative(0.2))], false);
        assert_eq!(mask(e, none(), [0., 0., 200., 100.]), ([180., 30., 20., 40.], [0.; 4]));
        // A circle has no outline: the rounded mask is all of it.
        let c = Style { vn_clip_path: ellipse(mid, [VnShapeRadius::ClosestSide; 2], true), ..none() };
        assert_eq!(c.vn_clip_outline(bounds, px(16.)), None);
    }

    #[test]
    fn a_box_alone_is_rounded_as_the_border_radius_rounds_it() {
        let style = || Style {
            corner_radii: Corners::all(px(24.)).map(|r| (*r).into()),
            border_widths: Edges::all(px(4.).into()),
            padding: Edges::all(px(6.).into()),
            margin: Edges { top: Length::Auto, ..Edges::all(len(8.).into()) },
            ..Style::default()
        };
        let at = [0., 0., 100., 80.];
        assert_eq!(mask(VnClipPath::Box(VnClipBox::Border), style(), at), ([0., 0., 100., 80.], [24.; 4]));
        assert_eq!(mask(VnClipPath::Box(VnClipBox::Padding), style(), at), ([4., 4., 92., 72.], [20.; 4]));
        assert_eq!(mask(VnClipPath::Box(VnClipBox::Content), style(), at), ([10., 10., 80., 60.], [14.; 4]));
        // An `auto` margin is none; a rounded corner grows by the margin.
        assert_eq!(mask(VnClipPath::Box(VnClipBox::Margin), style(), at), ([-8., 0., 116., 88.], [32.; 4]));
        // An inset's reference box: the content box less 5px.
        let clip = VnClipPath::Inset { reference: VnClipBox::Content, insets: [len(5.); 4], radii: round(len(0.), len(0.)) };
        assert_eq!(mask(clip, style(), at), ([15., 15., 70., 50.], [0.; 4]));
    }

    /// A clipped box with a border and a shadow, and a child that overflows
    /// it.
    struct Clipped;

    impl Render for Clipped {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let mut d = div()
                .size(px(60.))
                .bg(Hsla::red())
                .border_2()
                .border_color(Hsla::white())
                .shadow(vec![BoxShadow {
                    color: Hsla::black(),
                    offset: point(px(0.), px(4.)),
                    blur_radius: px(8.),
                    spread_radius: px(0.),
                    inset: false,
                }])
                .child(div().size(px(100.)).bg(Hsla::green()));
            d.style().vn_clip_path = Some(VnClipPath::Ellipse {
                reference: VnClipBox::Border,
                center: [relative(0.5), relative(0.5)],
                radius: [VnShapeRadius::Length(relative(0.5)); 2],
                circle: true,
            });
            div().child(d)
        }
    }

    #[gpui::test]
    fn everything_the_element_paints_carries_the_mask(cx: &mut TestAppContext) {
        let window = cx.open_window(size(px(200.), px(200.)), |_, _| Clipped);
        cx.run_until_parked();
        let (quads, shadows, scale) = window
            .update(cx, |_, window, _| {
                (window.painted_quads(), window.rendered_frame.scene.shadows.clone(), window.scale_factor())
            })
            .unwrap();
        let disc = Bounds::new(point(px(0.), px(0.)), size(px(60.), px(60.))).scale(scale);
        let radius = Corners::all(px(30.)).scale(scale);
        for color in [Hsla::red(), Hsla::green()] {
            let q = quads.iter().find(|q| q.background.as_solid() == Some(color)).expect("the quad");
            assert_eq!((q.content_mask.bounds, q.content_mask.corner_radii), (disc, radius), "{color:?}");
        }
        let border = quads.iter().find(|q| q.border_color == Hsla::white()).expect("the border");
        assert_eq!(border.content_mask.bounds, disc);
        assert_eq!(shadows.len(), 1);
        assert_eq!((shadows[0].content_mask.bounds, shadows[0].content_mask.corner_radii), (disc, radius));
    }

    fn polygon(reference: VnClipBox, fill_rule: crate::VnFillRule, points: &[[DefiniteLength; 2]]) -> VnClipPath {
        VnClipPath::Polygon { reference, fill_rule, points: points.to_vec().into() }
    }

    fn outline(clip: VnClipPath, style: Style, at: [f32; 4]) -> crate::VnClipOutline {
        let bounds = Bounds::new(point(px(at[0]), px(at[1])), size(px(at[2]), px(at[3])));
        Style { vn_clip_path: clip, ..style }.vn_clip_outline(bounds, px(16.)).expect("an outline")
    }

    #[test]
    fn a_polygon_resolves_its_percentages_against_the_reference_box() {
        use crate::{VnFillRule, VnPathVerb::*};
        // `polygon(50% 0, 100% 100%, 0 100%)` on a 200x100 box at (10, 20).
        let triangle = [[relative(0.5), len(0.)], [relative(1.), relative(1.)], [len(0.), relative(1.)]];
        let o = outline(polygon(BOX, VnFillRule::NonZero, &triangle), Style::default(), [10., 20., 200., 100.]);
        assert_eq!(o.verbs, vec![MoveTo([110., 20.]), LineTo([210., 120.]), LineTo([10., 120.]), Close]);
        // The mask is the box around it, which culls.
        assert_eq!(
            mask(polygon(BOX, VnFillRule::NonZero, &triangle), Style::default(), [10., 20., 200., 100.]),
            ([10., 20., 200., 100.], [0.; 4])
        );
        // Against the content box: 4px border and 6px padding in.
        let style = || Style {
            border_widths: Edges::all(px(4.).into()),
            padding: Edges::all(px(6.).into()),
            ..Style::default()
        };
        let o = outline(polygon(VnClipBox::Content, VnFillRule::EvenOdd, &triangle), style(), [0., 0., 100., 100.]);
        assert_eq!(o.fill_rule, VnFillRule::EvenOdd);
        assert_eq!(o.verbs[0], MoveTo([50., 10.]));
        assert_eq!(o.verbs[1], LineTo([90., 90.]));
        // `calc()` and a value past the box.
        let past = [[DefiniteLength::Calc { fraction: 1., pixels: px(-10.) }, relative(1.5)]];
        assert_eq!(
            outline(polygon(BOX, VnFillRule::NonZero, &past), Style::default(), [0., 0., 100., 40.]).verbs[0],
            MoveTo([90., 60.])
        );
    }

    #[test]
    fn a_path_is_laid_from_the_reference_box_corner() {
        use crate::{VnFillRule, VnPathVerb::*};
        let verbs = vec![MoveTo([0., 0.]), QuadTo([10., 20.], [20., 0.]), Close];
        let path = |reference| VnClipPath::Path { reference, fill_rule: VnFillRule::NonZero, verbs: verbs.clone().into() };
        let o = outline(path(BOX), Style::default(), [5., 7., 50., 50.]);
        assert_eq!(o.verbs, vec![MoveTo([5., 7.]), QuadTo([15., 27.], [25., 7.]), Close]);
        let style = Style { padding: Edges::all(px(3.).into()), ..Style::default() };
        let o = outline(path(VnClipBox::Content), style, [0., 0., 50., 50.]);
        assert_eq!(o.verbs[0], MoveTo([3., 3.]));
        // The mask is the control points' box, the curve inside it.
        assert_eq!(mask(path(BOX), Style::default(), [5., 7., 50., 50.]).0, [5., 7., 20., 20.]);
    }

    /// A box clipped to a triangle, with a child in flow and a deferred one.
    struct Triangle;

    impl Render for Triangle {
        fn render(&mut self, _window: &mut Window, _cx: &mut Context<Self>) -> impl IntoElement {
            let mut d = div()
                .size(px(100.))
                .bg(Hsla::red())
                .cursor_pointer()
                .child(div().size(px(100.)).bg(Hsla::green()))
                .child(crate::deferred(div().absolute().top_0().left_0().size(px(50.)).bg(Hsla::blue())).clip_to_ancestors(true));
            d.style().vn_clip_path = Some(polygon(
                BOX,
                crate::VnFillRule::NonZero,
                &[[len(0.), len(0.)], [len(100.), len(0.)], [len(0.), len(100.)]],
            ));
            div().child(d)
        }
    }

    #[gpui::test]
    fn a_polygon_paints_into_an_isolated_layer_and_cuts_the_hit_test(cx: &mut TestAppContext) {
        let window = cx.open_window(size(px(200.), px(200.)), |_, _| Triangle);
        cx.run_until_parked();
        window
            .update(cx, |_, window, _| {
                let scale = window.scale_factor();
                let scene = &window.rendered_frame.scene;
                // One layer for the box and its child in flow, one re-opened
                // for the deferred child; each composites one group.
                assert_eq!(scene.vn_layers.len(), 2, "{:?}", scene.vn_layers);
                assert_eq!(scene.vn_group_count(), 3);
                let root: Vec<_> = scene.vn_batches(0).collect();
                assert!(root.iter().all(|b| matches!(b, crate::VnBatch::Layers(_))), "{root:?}");
                let colors = |group: usize| -> Vec<Hsla> {
                    scene
                        .vn_batches(group)
                        .filter_map(|b| match b {
                            crate::VnBatch::Primitives(crate::PrimitiveBatch::Quads(r)) => Some(r),
                            _ => None,
                        })
                        .flat_map(|r| scene.quads[r].iter().filter_map(|q| q.background.as_solid()).collect::<Vec<_>>())
                        .collect()
                };
                let mut groups: Vec<Vec<Hsla>> = scene.vn_layers.iter().map(|l| colors(l.group as usize)).collect();
                groups.sort_by_key(|g| g.len());
                assert_eq!(groups, vec![vec![Hsla::blue()], vec![Hsla::red(), Hsla::green()]]);
                // Each composite is the triangle's box out to whole pixels
                // and a pixel more for the edge, with the triangle's edges.
                for layer in &scene.vn_layers {
                    assert_eq!(layer.segment_count, 3);
                    assert_eq!(layer.fill_rule, 0);
                    assert!(layer.bounds.size.width.0 >= 100. * scale);
                }
                assert_eq!(scene.vn_clip_segments.len(), 6);
                // A renderer without layers still gets every primitive.
                let flat = scene.batches().count();
                assert!(flat >= 2, "{flat}");

                // The hitbox: in the triangle, and not past its hypotenuse.
                let frame = &window.rendered_frame;
                let clipped: Vec<_> = frame.hitboxes.iter().filter(|h| h.vn_clip.is_some()).collect();
                assert_eq!(clipped.len(), 1);
                let id = clipped[0].id;
                let hits = |x: f32, y: f32| frame.hit_test(point(px(x), px(y)), scale).ids.contains(&id);
                assert!(hits(10., 10.));
                assert!(hits(40., 40.));
                assert!(!hits(60., 60.), "inside the box, outside the triangle");
                assert!(!hits(150., 10.));
            })
            .unwrap();
    }
}

#[cfg(test)]
mod vn_border_color_tests {
    use super::*;

    #[test]
    fn independent_border_colors_keep_shared_fallback_and_transparent_sides() {
        let mut style = Style { border_color: Some(Hsla::red()), ..Default::default() };
        style.vn_border_colors.left = Some(Hsla::transparent_black());
        style.vn_border_colors.top = Some(Hsla::blue());
        let colors = style.vn_resolved_border_colors();
        assert_eq!(colors.top, Hsla::blue());
        assert_eq!(colors.right, Hsla::red());
        assert!(colors.left.is_transparent());
    }

    #[test]
    fn colored_border_partitions_cover_rounded_arcs_with_width_weighted_joins() {
        let clips = Style::vn_border_side_clips(
            Bounds::new(point(px(0.), px(0.)), size(px(80.), px(80.))), Edges::all(px(8.)),
        );
        let contains = |polygon: &Vec<[f32; 2]>, point: [f32; 2]| {
            let mut sign = 0.;
            for (a, b) in polygon.iter().zip(polygon.iter().cycle().skip(1)) {
                let cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0]);
                if cross.abs() < 0.0001 { continue; }
                if sign != 0. && cross.signum() != sign { return false; }
                sign = cross.signum();
            }
            polygon.len() >= 3
        };
        assert!(contains(&clips[0], [15., 13.]), "top rounded arc extends below the rectangular inner top");
        assert!(contains(&clips[3], [13., 15.]), "left rounded arc extends beyond the rectangular inner left");
        for y in 0..80 {
            for x in 0..80 {
                assert!(clips.iter().any(|polygon| contains(polygon, [x as f32 + 0.25, y as f32 + 0.25])),
                    "coloured side clips must cover the entire quad, including rounded corners");
            }
        }
        let clips = Style::vn_border_side_clips(
            Bounds::new(point(px(0.), px(0.)), size(px(100.), px(60.))),
            Edges { top: px(4.), right: px(8.), bottom: px(6.), left: px(2.) },
        );
        assert!(contains(&clips[0], [2., 3.]));
        assert!(contains(&clips[3], [2., 5.]), "top/left join uses their2:4width ratio");
    }
}
