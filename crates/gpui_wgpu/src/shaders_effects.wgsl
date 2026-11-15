// The engine-owned VERTEX half of every `--shading` effect; the fragment half
// is generated from the effect's `.slang` source (crates/sk-effects). A port of
// gpui_windows' effects.hlsl. The `<iframe>` punch's fragment stage is here too.
//
// The bindings are the ones `modules/sk.slang` declares with
// `[[vk::binding]]`. The globals keep the DirectX constant-buffer layout,
// because the generated fragment stages read `sk_viewport_size` at offset 16.

struct EffectGlobals {
    gamma_ratios: vec4<f32>,
    viewport_size: vec2<f32>,
    grayscale_enhanced_contrast: f32,
    subpixel_enhanced_contrast: f32,
    is_bgr: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

struct EffectBounds {
    origin: vec2<f32>,
    size: vec2<f32>,
}

struct EffectCorners {
    top_left: f32,
    top_right: f32,
    bottom_right: f32,
    bottom_left: f32,
}

// 56 words / 224 bytes, `gpui::EffectQuad`.
struct EffectQuad {
    order: u32,
    effect_id: u32,
    bounds: EffectBounds,
    content_mask: EffectBounds,
    content_mask_radii: EffectCorners,
    corner_radii: EffectCorners,
    params: array<f32, 32>,
    opacity: f32,
    time: f32,
    bleed: f32,
    backdrop_blur_radius: f32,
    scale: f32,
    flags: u32,
}

@group(0) @binding(0) var<uniform> effect_globals: EffectGlobals;
@group(1) @binding(0) var<storage, read> effect_quads: array<EffectQuad>;

// Exactly the varyings the generated fragment stage declares.
struct EffectVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) effect_quad_id: u32,
}

// One instance draws one effect quad, already dilated by its bleed. The quad
// is cut to its content mask's rectangle here because the generated fragment
// stage cannot discard; with no transform the intersection rejects exactly the
// pixels DirectX's clip distances reject.
@vertex
fn vs_effect(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> EffectVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let quad = effect_quads[instance_id];
    let origin = max(quad.bounds.origin, quad.content_mask.origin);
    let corner = min(quad.bounds.origin + quad.bounds.size, quad.content_mask.origin + quad.content_mask.size);
    let position = origin + unit_vertex * max(corner - origin, vec2<f32>(0.0));

    var out = EffectVarying();
    out.position = vec4<f32>(position / effect_globals.viewport_size * vec2<f32>(2.0, -2.0) + vec2<f32>(-1.0, 1.0), 0.0, 1.0);
    out.effect_quad_id = instance_id;
    return out;
}

fn punch_distance(position: vec2<f32>, bounds: EffectBounds, radii: EffectCorners) -> f32 {
    let half_size = bounds.size * 0.5;
    let p = position - (bounds.origin + half_size);
    let left = select(radii.bottom_left, radii.top_left, p.y < 0.0);
    let right = select(radii.bottom_right, radii.top_right, p.y < 0.0);
    let radius = min(select(right, left, p.x < 0.0), min(half_size.x, half_size.y));
    let q = abs(p) - half_size + radius;
    return length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0) - radius;
}

// `gpui::effect_id::PUNCH`, gpui_windows' `punch_fragment`: only a coverage,
// which the pipeline's destination-out blend erases from the target.
@fragment
fn fs_punch(input: EffectVarying) -> @location(0) vec4<f32> {
    let quad = effect_quads[input.effect_quad_id];
    let inside = saturate(0.5 - punch_distance(input.position.xy, quad.bounds, quad.corner_radii));
    let unmasked = saturate(0.5 - punch_distance(input.position.xy, quad.content_mask, quad.content_mask_radii));
    return vec4<f32>(0.0, 0.0, 0.0, inside * unmasked * quad.opacity);
}
