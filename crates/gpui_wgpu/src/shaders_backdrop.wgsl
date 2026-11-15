// CSS `backdrop-filter: blur()`: the blurred pyramid composited through a
// rounded rect. Concatenated after the storage-buffer variant of `shaders.wgsl`;
// a port of `blur_rect_*` in gpui_windows' shaders.hlsl.
//
// Group 2 is shared with the `--shading` effects: the processed backdrop at
// binding 0 (`t_sprite`), the clamping sampler at 1 (`s_sprite`) and the
// untouched snapshot at 2.

@group(2) @binding(2) var t_backdrop_original: texture_2d<f32>;

struct BlurRect {
    order: u32,
    pad: u32,
    bounds: Bounds,
    content_mask: Bounds,
    content_mask_radii: Corners,
    corner_radii: Corners,
    blur_radius: f32,
    opacity: f32,
    tint: Hsla,
}

// 24 words, in `gpui::BackdropBlurRect` order.
fn load_blur_rect(instance_id: u32) -> BlurRect {
    var cursor = instance_cursor(instance_id * 24u);
    return BlurRect(
        read_word(&cursor),
        read_word(&cursor),
        read_bounds(&cursor),
        read_bounds(&cursor),
        read_corners(&cursor),
        read_corners(&cursor),
        read_f32(&cursor),
        read_f32(&cursor),
        read_hsla(&cursor),
    );
}

// Keeps rgb <= a, so a premultiplied target never holds an out-of-gamut pixel.
fn clamp_premultiplied(color: vec4<f32>) -> vec4<f32> {
    let alpha = saturate(color.a);
    return vec4<f32>(clamp(color.rgb, vec3<f32>(0.0), vec3<f32>(alpha)), alpha);
}

// `above` arrives with straight alpha, `below` is already premultiplied.
fn over_straight_on_premultiplied(below: vec4<f32>, above: vec4<f32>) -> vec4<f32> {
    let premultiplied = vec4<f32>(above.rgb * above.a, above.a);
    return clamp_premultiplied(premultiplied + below * (1.0 - above.a));
}

// The rect cut to its content mask's rectangle. Nothing here is transformed,
// so drawing the intersection rejects exactly the pixels that DirectX's clip
// distances reject.
fn clip_bounds(bounds: Bounds, mask: Bounds) -> Bounds {
    let origin = max(bounds.origin, mask.origin);
    let corner = min(bounds.origin + bounds.size, mask.origin + mask.size);
    return Bounds(origin, max(corner - origin, vec2<f32>(0.0)));
}

struct BlurRectVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) blur_rect_id: u32,
}

@vertex
fn vs_blur_rect(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> BlurRectVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let blur_rect = load_blur_rect(instance_id);
    var out = BlurRectVarying();
    out.position = to_device_position(unit_vertex, clip_bounds(blur_rect.bounds, blur_rect.content_mask));
    out.blur_rect_id = instance_id;
    return out;
}

// Drawn with blending disabled: outside the rounded rect the untouched
// snapshot is written back, which keeps the antialiased edge correct.
@fragment
fn fs_blur_rect(input: BlurRectVarying) -> @location(0) vec4<f32> {
    let blur_rect = load_blur_rect(input.blur_rect_id);
    let position = input.position.xy;
    let texture_position = position / globals.viewport_size;

    let sdf = quad_sdf(position, blur_rect.bounds, blur_rect.corner_radii);
    let shape_alpha = saturate(0.5 - sdf);

    let original = textureSampleLevel(t_backdrop_original, s_sprite, texture_position, 0.0);
    let blurred = textureSampleLevel(t_sprite, s_sprite, texture_position, 0.0);
    let tint = hsla_to_rgba(blur_rect.tint);
    let tinted = select(blurred, over_straight_on_premultiplied(blurred, tint), tint.a > 0.0);
    let coverage = shape_alpha * saturate(blur_rect.opacity) *
        content_mask_coverage(position, blur_rect.content_mask, blur_rect.content_mask_radii);
    return clamp_premultiplied(mix(original, tinted, vec4<f32>(coverage)));
}
