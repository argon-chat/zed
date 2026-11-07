//! vue-native fork: the CSS `clip-path` shapes that are no rounded rectangle,
//! `polygon()` and `path()` (CSS Shapes 1 §3.1, CSS Masking 1 §5.1).
//!
//! A rounded shape (`inset()`, `circle()`, a box) is a content mask
//! ([`crate::Style::vn_clip_mask`]), which every shader already applies. These
//! are not: the element's painting goes into an isolated layer instead, an
//! offscreen texture the renderer composites through the shape's coverage
//! ([`crate::Scene::vn_push_isolation`]). The shape travels three ways:
//!
//! - [`VnPathVerb`]s, the parsed `path()` data (absolute, arcs as cubics), in
//!   the style;
//! - a [`VnClipOutline`], the shape laid out against its reference box in the
//!   element's own frame, logical pixels;
//! - a [`VnClipGeometry`], the outline flattened into line segments in WINDOW
//!   device pixels, through the element's whole transform. The renderer
//!   uploads its segments and evaluates the winding number and the distance
//!   to the nearest edge per pixel; the hit test asks [`VnClipGeometry::contains`].
//!   An affine map takes a Bézier's control points to the mapped curve's, so
//!   the geometry is exact under rotation and skew too.

use crate::{Bounds, ScaledPixels, TransformationMatrix, point, px, size};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

/// vue-native fork: the `<fill-rule>` of a `polygon()` or `path()`.
#[derive(Copy, Clone, Default, PartialEq, Eq, Debug, Serialize, Deserialize, JsonSchema)]
pub enum VnFillRule {
    /// `nonzero`, the default: inside where the winding number is not zero.
    #[default]
    NonZero,
    /// `evenodd`: inside where the winding number is odd.
    EvenOdd,
}

/// vue-native fork: one verb of a `path()`, in absolute pixels from the
/// reference box's top left corner. A parser resolves the relative commands,
/// the `H`/`V` lines, the `S`/`T` reflections and the arcs
/// ([`VnPathVerb::push_arc`]) into these five.
#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize, JsonSchema)]
pub enum VnPathVerb {
    /// `M`: starts a subpath.
    MoveTo([f32; 2]),
    /// `L`, `H`, `V`.
    LineTo([f32; 2]),
    /// `Q`, `T`: a control point and the end.
    QuadTo([f32; 2], [f32; 2]),
    /// `C`, `S`, `A`: two control points and the end.
    CubicTo([f32; 2], [f32; 2], [f32; 2]),
    /// `Z`: closes the subpath. A fill closes every subpath anyway.
    Close,
}

impl VnPathVerb {
    /// Appends SVG's elliptical arc from `from` to `to` (SVG 2 §9.3.8 and
    /// its appendix B.2.5: out-of-range radii scale up, a zero radius is a
    /// line, an arc to its own start is nothing) as cubics.
    pub fn push_arc(
        out: &mut Vec<VnPathVerb>,
        from: [f32; 2],
        radii: [f32; 2],
        x_rotation_degrees: f32,
        large_arc: bool,
        sweep: bool,
        to: [f32; 2],
    ) {
        use lyon::geom::{Angle, ArcFlags, SvgArc, point as lpoint, vector};
        if from == to {
            return;
        }
        let arc = SvgArc {
            from: lpoint(from[0], from[1]),
            to: lpoint(to[0], to[1]),
            radii: vector(radii[0].abs(), radii[1].abs()),
            x_rotation: Angle::degrees(x_rotation_degrees),
            flags: ArcFlags { large_arc, sweep },
        };
        if arc.is_straight_line() {
            out.push(VnPathVerb::LineTo(to));
            return;
        }
        arc.for_each_cubic_bezier(&mut |c| {
            out.push(VnPathVerb::CubicTo([c.ctrl1.x, c.ctrl1.y], [c.ctrl2.x, c.ctrl2.y], [c.to.x, c.to.y]));
        });
        // The last cubic ends on `to` up to rounding; make it exact, so a
        // following relative command starts where the arc was asked to end.
        if let Some(VnPathVerb::CubicTo(_, _, end)) = out.last_mut() {
            *end = to;
        }
    }
}

/// vue-native fork: a `polygon()` or `path()` laid out against its reference
/// box, in the element's own frame and logical pixels
/// ([`crate::Style::vn_clip_outline`]).
#[derive(Clone, PartialEq, Debug, Default)]
pub struct VnClipOutline {
    /// How the winding number decides inside.
    pub fill_rule: VnFillRule,
    /// The shape.
    pub verbs: Vec<VnPathVerb>,
}

impl VnClipOutline {
    /// The bounding box of every point and control point, which holds the
    /// shape (a Bézier lies in the hull of its control points). `None` for
    /// an outline with no point.
    pub fn control_bounds(&self) -> Option<Bounds<crate::Pixels>> {
        let mut points = self.verbs.iter().flat_map(|v| match v {
            VnPathVerb::MoveTo(a) | VnPathVerb::LineTo(a) => vec![*a],
            VnPathVerb::QuadTo(a, b) => vec![*a, *b],
            VnPathVerb::CubicTo(a, b, c) => vec![*a, *b, *c],
            VnPathVerb::Close => vec![],
        });
        let first = points.next()?;
        let (mut min, mut max) = (first, first);
        for p in points {
            min = [min[0].min(p[0]), min[1].min(p[1])];
            max = [max[0].max(p[0]), max[1].max(p[1])];
        }
        Some(Bounds::from_corners(point(px(min[0]), px(min[1])), point(px(max[0]), px(max[1]))))
    }
}

/// The largest distance, in device pixels, between a curve and the line
/// segments that stand for it. A quarter pixel is below what the shader's
/// one-pixel antialiased edge shows.
pub const VN_CLIP_TOLERANCE: f32 = 0.25;

/// Most segments one curve becomes, whatever its size.
const MAX_CURVE_SEGMENTS: u32 = 128;

/// vue-native fork: a clip shape flattened into line segments, in window
/// device pixels. What the renderer evaluates per pixel and the hit test
/// asks.
#[derive(Clone, PartialEq, Debug, Default)]
pub struct VnClipGeometry {
    /// How the winding number decides inside.
    pub fill_rule: VnFillRule,
    /// The edges, `[x0, y0, x1, y1]`; every subpath is closed.
    pub segments: Vec<[f32; 4]>,
    /// The box around the segments.
    pub bounds: Bounds<ScaledPixels>,
}

impl VnClipGeometry {
    /// Flattens `outline` (logical pixels, the element's own frame) into
    /// window device pixels: scaled by `scale_factor`, then through
    /// `transform` (a device-space matrix, as
    /// [`crate::Window::element_transform`]). Curves are cut to
    /// [`VN_CLIP_TOLERANCE`] in device pixels.
    pub fn new(outline: &VnClipOutline, scale_factor: f32, transform: TransformationMatrix) -> Self {
        let unit = transform.is_unit();
        let map = |p: [f32; 2]| -> [f32; 2] {
            let d = [p[0] * scale_factor, p[1] * scale_factor];
            if unit {
                return d;
            }
            let t = transform.apply(point(px(d[0]), px(d[1])));
            [t.x.0, t.y.0]
        };
        let mut segments = Vec::new();
        let mut start = [0.; 2];
        let mut current = [0.; 2];
        let mut open = false;
        let mut close = |segments: &mut Vec<[f32; 4]>, current: [f32; 2], start: [f32; 2]| {
            if current != start {
                segments.push([current[0], current[1], start[0], start[1]]);
            }
        };
        for verb in &outline.verbs {
            match *verb {
                VnPathVerb::MoveTo(p) => {
                    if open {
                        close(&mut segments, current, start);
                    }
                    start = map(p);
                    current = start;
                    open = true;
                }
                VnPathVerb::LineTo(p) => {
                    let p = map(p);
                    if p != current {
                        segments.push([current[0], current[1], p[0], p[1]]);
                    }
                    current = p;
                    open = true;
                }
                VnPathVerb::QuadTo(c, p) => {
                    let (c, p) = (map(c), map(p));
                    flatten_quad(&mut segments, current, c, p);
                    current = p;
                    open = true;
                }
                VnPathVerb::CubicTo(c1, c2, p) => {
                    let (c1, c2, p) = (map(c1), map(c2), map(p));
                    flatten_cubic(&mut segments, current, c1, c2, p);
                    current = p;
                    open = true;
                }
                VnPathVerb::Close => {
                    if open {
                        close(&mut segments, current, start);
                    }
                    // SVG: the next segment starts where the subpath did.
                    current = start;
                    open = false;
                }
            }
        }
        if open {
            close(&mut segments, current, start);
        }
        let bounds = segment_bounds(&segments);
        VnClipGeometry { fill_rule: outline.fill_rule, segments, bounds }
    }

    /// The winding number of the shape around `p` (device pixels): the
    /// signed count of edges crossing the ray from `p` to the right, upward
    /// edges counting one way and downward ones the other.
    pub fn winding(&self, p: [f32; 2]) -> i32 {
        let mut winding = 0;
        for s in &self.segments {
            let (a, b) = ([s[0], s[1]], [s[2], s[3]]);
            // Half-open in y, so a vertex on the ray counts once.
            if (a[1] <= p[1]) != (b[1] <= p[1]) {
                let t = (p[1] - a[1]) / (b[1] - a[1]);
                let x = a[0] + t * (b[0] - a[0]);
                if p[0] < x {
                    winding += if b[1] > a[1] { 1 } else { -1 };
                }
            }
        }
        winding
    }

    /// Whether `p` (device pixels) is inside the shape under its fill rule.
    pub fn contains(&self, p: [f32; 2]) -> bool {
        let w = self.winding(p);
        match self.fill_rule {
            VnFillRule::NonZero => w != 0,
            VnFillRule::EvenOdd => w & 1 == 1,
        }
    }

    /// The antialiased coverage at the pixel centre `p`, as the composite
    /// shader computes it (`vn_layer_fragment` in gpui_windows'
    /// shaders.hlsl): inside or outside by the fill rule, blended over one
    /// pixel across the nearest edge.
    pub fn coverage(&self, p: [f32; 2]) -> f32 {
        let mut nearest = f32::MAX;
        for s in &self.segments {
            nearest = nearest.min(distance_to_segment(p, *s));
        }
        let d = if self.contains(p) { nearest } else { -nearest };
        (0.5 + d).clamp(0., 1.)
    }
}

/// An SVG feColorMatrix over unpremultiplied RGBA components.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct VnColorMatrix {
    /// Stable identity of the filtered element within this window's scene.
    /// Zero keeps independently composited layers for callers without an ID.
    pub group_id: u64,
    /// Four rows of five coefficients, in SVG order.
    pub values: [f32; 20],
    /// SVG filters use linear RGB unless color-interpolation-filters is sRGB.
    pub linear_rgb: bool,
}

impl VnColorMatrix {
    /// The identity transform.
    pub const IDENTITY: [f32; 20] = [
        1., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
        0., 0., 1., 0., 0., 0., 0., 0., 1., 0.,
    ];

    /// Four float4 coefficient rows followed by a float4 bias, for the GPU.
    pub fn gpu_values(self) -> [f32; 20] {
        let mut packed = [0.; 20];
        for row in 0..4 {
            packed[row * 4..row * 4 + 4].copy_from_slice(&self.values[row * 5..row * 5 + 4]);
            packed[16 + row] = self.values[row * 5 + 4];
        }
        packed
    }
}

/// vue-native fork: a clip shape in force, and the ones around it — what a
/// hitbox, a deferred draw and the window's clip stack carry, so a press
/// outside any of them misses (CSS Masking 1 §5).
#[derive(Debug)]
pub struct VnClipNode {
    /// This clip's shape, in window device pixels.
    pub geometry: Arc<VnClipGeometry>,
    /// A color transform applied when this isolated layer is composited.
    pub color_matrix: Option<VnColorMatrix>,
    /// The clip around this one, if any.
    pub parent: Option<Arc<VnClipNode>>,
}

impl VnClipNode {
    /// Whether the window device pixel `p` is inside this clip and every
    /// one around it.
    pub fn contains(&self, p: [f32; 2]) -> bool {
        let mut node = Some(self);
        while let Some(n) = node {
            if !n.geometry.contains(p) {
                return false;
            }
            node = n.parent.as_deref();
        }
        true
    }

    /// The chain from the outermost clip in.
    pub fn outermost_first(this: &Arc<VnClipNode>) -> Vec<Arc<VnClipNode>> {
        let mut chain = vec![this.clone()];
        while let Some(parent) = chain.last().and_then(|n| n.parent.clone()) {
            chain.push(parent);
        }
        chain.reverse();
        chain
    }

    /// Fixed-position descendants can escape geometric clips, but remain in
    /// their ancestors' color-filter groups until those groups composite.
    pub fn color_filters(clip: &Option<Arc<VnClipNode>>) -> Option<Arc<VnClipNode>> {
        let Some(clip) = clip else { return None };
        let mut parent = None;
        for node in Self::outermost_first(clip) {
            if node.color_matrix.is_some() {
                parent = Some(Arc::new(Self {
                    geometry: node.geometry.clone(),
                    color_matrix: node.color_matrix,
                    parent,
                }));
            }
        }
        parent
    }

    /// Two chains are the same clip: the same shapes, nested the same way.
    pub fn same(a: &Option<Arc<VnClipNode>>, b: &Option<Arc<VnClipNode>>) -> bool {
        match (a, b) {
            (None, None) => true,
            (Some(a), Some(b)) => {
                Arc::ptr_eq(a, b) || (a.geometry == b.geometry && a.color_matrix == b.color_matrix && VnClipNode::same(&a.parent, &b.parent))
            }
            _ => false,
        }
    }
}

fn segment_bounds(segments: &[[f32; 4]]) -> Bounds<ScaledPixels> {
    let Some(first) = segments.first() else {
        return Bounds::default();
    };
    let (mut min, mut max) = ([first[0], first[1]], [first[0], first[1]]);
    for s in segments {
        for p in [[s[0], s[1]], [s[2], s[3]]] {
            min = [min[0].min(p[0]), min[1].min(p[1])];
            max = [max[0].max(p[0]), max[1].max(p[1])];
        }
    }
    Bounds {
        origin: point(ScaledPixels(min[0]), ScaledPixels(min[1])),
        size: size(ScaledPixels(max[0] - min[0]), ScaledPixels(max[1] - min[1])),
    }
}

fn distance_to_segment(p: [f32; 2], s: [f32; 4]) -> f32 {
    let (ax, ay, bx, by) = (s[0], s[1], s[2], s[3]);
    let (dx, dy) = (bx - ax, by - ay);
    let len2 = dx * dx + dy * dy;
    let t = if len2 > 0. { (((p[0] - ax) * dx + (p[1] - ay) * dy) / len2).clamp(0., 1.) } else { 0. };
    let (qx, qy) = (ax + t * dx - p[0], ay + t * dy - p[1]);
    (qx * qx + qy * qy).sqrt()
}

/// Wang's formula: the segments a Bézier of `degree` needs for its
/// approximation to stay within the tolerance, from its largest second
/// difference.
fn segment_count(degree: f32, second_difference: f32) -> u32 {
    let n = (degree * (degree - 1.) / 8. * second_difference / VN_CLIP_TOLERANCE).sqrt().ceil();
    (n as u32).clamp(1, MAX_CURVE_SEGMENTS)
}

fn flatten_quad(out: &mut Vec<[f32; 4]>, p0: [f32; 2], p1: [f32; 2], p2: [f32; 2]) {
    let dd = ((p0[0] - 2. * p1[0] + p2[0]).powi(2) + (p0[1] - 2. * p1[1] + p2[1]).powi(2)).sqrt();
    let n = segment_count(2., dd);
    let mut prev = p0;
    for i in 1..=n {
        let t = i as f32 / n as f32;
        let u = 1. - t;
        let p = if i == n {
            p2
        } else {
            [u * u * p0[0] + 2. * u * t * p1[0] + t * t * p2[0], u * u * p0[1] + 2. * u * t * p1[1] + t * t * p2[1]]
        };
        if p != prev {
            out.push([prev[0], prev[1], p[0], p[1]]);
        }
        prev = p;
    }
}

fn flatten_cubic(out: &mut Vec<[f32; 4]>, p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], p3: [f32; 2]) {
    let d1 = ((p0[0] - 2. * p1[0] + p2[0]).powi(2) + (p0[1] - 2. * p1[1] + p2[1]).powi(2)).sqrt();
    let d2 = ((p1[0] - 2. * p2[0] + p3[0]).powi(2) + (p1[1] - 2. * p2[1] + p3[1]).powi(2)).sqrt();
    let n = segment_count(3., d1.max(d2));
    let mut prev = p0;
    for i in 1..=n {
        let t = i as f32 / n as f32;
        let u = 1. - t;
        let p = if i == n {
            p3
        } else {
            let (a, b, c, d) = (u * u * u, 3. * u * u * t, 3. * u * t * t, t * t * t);
            [
                a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0],
                a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1],
            ]
        };
        if p != prev {
            out.push([prev[0], prev[1], p[0], p[1]]);
        }
        prev = p;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{TransformationMatrix, point};

    fn outline(fill_rule: VnFillRule, verbs: Vec<VnPathVerb>) -> VnClipOutline {
        VnClipOutline { fill_rule, verbs }
    }

    fn square(x: f32, y: f32, side: f32, clockwise: bool) -> Vec<VnPathVerb> {
        use VnPathVerb::*;
        let mut v = vec![MoveTo([x, y])];
        if clockwise {
            v.extend([LineTo([x + side, y]), LineTo([x + side, y + side]), LineTo([x, y + side])]);
        } else {
            v.extend([LineTo([x, y + side]), LineTo([x + side, y + side]), LineTo([x + side, y])]);
        }
        v.push(Close);
        v
    }

    #[test]
    fn color_matrix_gpu_rows_and_bias_match_the_layer_shader() {
        let matrix = VnColorMatrix {
            group_id: 1,
            values: std::array::from_fn(|index| index as f32),
            linear_rgb: true,
        };
        assert_eq!(matrix.gpu_values(), [
            0., 1., 2., 3., 5., 6., 7., 8., 10., 11., 12., 13., 15., 16., 17., 18.,
            4., 9., 14., 19.,
        ]);
        assert_eq!(std::mem::size_of::<crate::VnLayer>(), 160);
        assert_eq!(std::mem::offset_of!(crate::VnLayer, color_matrix), 32);
        assert_eq!(std::mem::offset_of!(crate::VnLayer, bounds), 112);
    }

    #[test]
    fn a_triangle_is_inside_where_it_is_and_closes_itself() {
        use VnPathVerb::*;
        // No `Z`: a fill closes the subpath.
        let g = VnClipGeometry::new(
            &outline(VnFillRule::NonZero, vec![MoveTo([0., 0.]), LineTo([100., 0.]), LineTo([0., 100.])]),
            1.,
            TransformationMatrix::unit(),
        );
        assert_eq!(g.segments.len(), 3);
        assert!(g.contains([10., 10.]));
        assert!(!g.contains([60., 60.]), "beyond the hypotenuse");
        assert!(!g.contains([-1., 5.]));
        assert_eq!(g.coverage([10.5, 10.5]), 1.);
        assert_eq!(g.coverage([80.5, 80.5]), 0.);
        // On the hypotenuse x + y = 100 the pixel is half covered.
        assert!((g.coverage([50., 50.]) - 0.5).abs() < 0.01);
        assert_eq!(
            [g.bounds.origin.x.0, g.bounds.origin.y.0, g.bounds.size.width.0, g.bounds.size.height.0],
            [0., 0., 100., 100.]
        );
    }

    #[test]
    fn the_fill_rule_decides_a_hole() {
        // A square inside a square, both clockwise: winding 2 in the middle.
        let mut verbs = square(0., 0., 100., true);
        verbs.extend(square(25., 25., 50., true));
        let nonzero = VnClipGeometry::new(&outline(VnFillRule::NonZero, verbs.clone()), 1., TransformationMatrix::unit());
        let evenodd = VnClipGeometry::new(&outline(VnFillRule::EvenOdd, verbs), 1., TransformationMatrix::unit());
        assert_eq!(nonzero.winding([50., 50.]).abs(), 2);
        assert!(nonzero.contains([50., 50.]) && nonzero.contains([10., 10.]));
        assert!(!evenodd.contains([50., 50.]) && evenodd.contains([10., 10.]));
        // The inner one counter-clockwise: a hole under nonzero too.
        let mut verbs = square(0., 0., 100., true);
        verbs.extend(square(25., 25., 50., false));
        let g = VnClipGeometry::new(&outline(VnFillRule::NonZero, verbs), 1., TransformationMatrix::unit());
        assert!(!g.contains([50., 50.]) && g.contains([10., 10.]));
    }

    #[test]
    fn curves_flatten_within_the_tolerance() {
        use VnPathVerb::*;
        // A circle of radius 40 from four cubics (k = 0.5523).
        let (c, r, k) = (50., 40., 0.5523 * 40.);
        let verbs = vec![
            MoveTo([c + r, c]),
            CubicTo([c + r, c + k], [c + k, c + r], [c, c + r]),
            CubicTo([c - k, c + r], [c - r, c + k], [c - r, c]),
            CubicTo([c - r, c - k], [c - k, c - r], [c, c - r]),
            CubicTo([c + k, c - r], [c + r, c - k], [c + r, c]),
            Close,
        ];
        let g = VnClipGeometry::new(&outline(VnFillRule::NonZero, verbs), 2., TransformationMatrix::unit());
        // At scale 2 the radius is 80 device pixels: every vertex on it.
        for s in &g.segments {
            let d = ((s[0] - 100.).powi(2) + (s[1] - 100.).powi(2)).sqrt();
            assert!((d - 80.).abs() < 0.1, "{d}");
        }
        // Few enough segments to evaluate per pixel, enough to be round.
        assert!(g.segments.len() >= 16 && g.segments.len() <= 64, "{}", g.segments.len());
        assert!(g.contains([100., 100.]) && !g.contains([100. + 80. * 0.75, 100. + 80. * 0.75]));
        // A quadratic too.
        let q = VnClipGeometry::new(
            &outline(VnFillRule::NonZero, vec![MoveTo([0., 0.]), QuadTo([50., 100.], [100., 0.]), Close]),
            1.,
            TransformationMatrix::unit(),
        );
        assert!(q.contains([50., 40.]) && !q.contains([50., 60.]), "the apex is at y = 50");
    }

    #[test]
    fn the_shape_follows_the_whole_transform() {
        // A 90 degree rotation about the origin, then a translation: exact,
        // where a rounded mask would fall back to its bounding box.
        let rotate = TransformationMatrix::unit()
            .translate(point(ScaledPixels(200.), ScaledPixels(0.)))
            .rotate(crate::Radians(std::f32::consts::FRAC_PI_2));
        use VnPathVerb::*;
        let triangle = vec![MoveTo([0., 0.]), LineTo([100., 0.]), LineTo([0., 50.]), Close];
        let g = VnClipGeometry::new(&outline(VnFillRule::NonZero, triangle.clone()), 1., rotate);
        let probe = |p: [f32; 2]| {
            let q = rotate.apply(point(px(p[0]), px(p[1])));
            [q.x.0, q.y.0]
        };
        assert!(g.contains(probe([10., 10.])));
        assert!(!g.contains(probe([90., 40.])));
        assert!(!g.contains([10., 10.]), "not where the untransformed shape was");
    }

    #[test]
    fn arcs_become_cubics_that_end_where_asked() {
        let mut verbs = vec![VnPathVerb::MoveTo([0., 50.])];
        // A half circle of radius 50 from (0, 50) to (100, 50), over the top.
        VnPathVerb::push_arc(&mut verbs, [0., 50.], [50., 50.], 0., false, true, [100., 50.]);
        assert!(verbs.len() >= 3, "{verbs:?}");
        assert_eq!(verbs.last(), Some(&VnPathVerb::CubicTo(
            match verbs.last() { Some(VnPathVerb::CubicTo(a, _, _)) => *a, _ => unreachable!() },
            match verbs.last() { Some(VnPathVerb::CubicTo(_, b, _)) => *b, _ => unreachable!() },
            [100., 50.],
        )));
        let g = VnClipGeometry::new(&outline(VnFillRule::NonZero, verbs), 1., TransformationMatrix::unit());
        assert!(g.contains([50., 10.]), "the top of the half disc");
        assert!(!g.contains([50., 60.]), "below the diameter");
        assert!(!g.contains([5., 5.]), "outside the curve in the corner");
        // A zero radius is a line; an arc to its own start is nothing.
        let mut line = vec![];
        VnPathVerb::push_arc(&mut line, [0., 0.], [0., 10.], 0., false, false, [10., 0.]);
        assert_eq!(line, vec![VnPathVerb::LineTo([10., 0.])]);
        let mut none = vec![];
        VnPathVerb::push_arc(&mut none, [3., 3.], [10., 10.], 0., false, false, [3., 3.]);
        assert!(none.is_empty());
    }

    #[test]
    fn a_clip_chain_contains_what_every_link_does() {
        let left = Arc::new(VnClipGeometry::new(
            &outline(VnFillRule::NonZero, square(0., 0., 60., true)),
            1.,
            TransformationMatrix::unit(),
        ));
        let right = Arc::new(VnClipGeometry::new(
            &outline(VnFillRule::NonZero, square(40., 0., 60., true)),
            1.,
            TransformationMatrix::unit(),
        ));
        let outer = Arc::new(VnClipNode { geometry: left.clone(), color_matrix: None, parent: None });
        let inner = Arc::new(VnClipNode { geometry: right.clone(), color_matrix: None, parent: Some(outer.clone()) });
        assert!(inner.contains([50., 10.]));
        assert!(!inner.contains([10., 10.]) && !inner.contains([90., 10.]));
        assert_eq!(VnClipNode::outermost_first(&inner).len(), 2);
        assert!(Arc::ptr_eq(&VnClipNode::outermost_first(&inner)[0], &outer));
        // The same shapes rebuilt are the same clip; a different one is not.
        let rebuilt = Some(Arc::new(VnClipNode {
            geometry: Arc::new((*right).clone()),
            color_matrix: None,
            parent: Some(Arc::new(VnClipNode { geometry: Arc::new((*left).clone()), color_matrix: None, parent: None })),
        }));
        assert!(VnClipNode::same(&Some(inner.clone()), &rebuilt));
        assert!(!VnClipNode::same(&Some(inner), &Some(outer)));
    }

    #[test]
    fn fixed_escapes_preserve_color_groups_while_dropping_geometric_clips() {
        let geometry = Arc::new(VnClipGeometry::new(
            &outline(VnFillRule::NonZero, square(0., 0., 100., true)),
            1., TransformationMatrix::unit(),
        ));
        let clip = Arc::new(VnClipNode { geometry: geometry.clone(), color_matrix: None, parent: None });
        let filter = Arc::new(VnClipNode {
            geometry: geometry.clone(),
            color_matrix: Some(VnColorMatrix { group_id: 7, values: VnColorMatrix::IDENTITY, linear_rgb: true }),
            parent: Some(clip),
        });
        let inside = Some(Arc::new(VnClipNode { geometry, color_matrix: None, parent: Some(filter) }));
        let retained = VnClipNode::color_filters(&inside).expect("filter retained");
        assert_eq!(retained.color_matrix.expect("matrix").group_id, 7);
        assert!(retained.parent.is_none(), "geometric clipping is escaped");
    }

}
