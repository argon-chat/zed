// The composite of one isolated layer (`gpui::VnLayer`): the group drawn into
// an offscreen target the size of the frame, put back over its parent through
// a clip shape, a color matrix and a text background stack. A port of
// `vn_layer_*` in gpui_windows' shaders.hlsl.
//
// The instance buffer holds the frame's layers (56 words each), then every
// clip segment (4 words), then every text background (69 words). The renderer
// rewrites `segment_start` and `text_start` to the word where each run begins.
// Group 2 binding 0 (`t_sprite`) is the layer's target, or the snapshot a
// snapshot composite samples.

struct VnLayer {
    segment_start: u32,
    segment_count: u32,
    fill_rule: u32,
    matrix_mode: u32,
    text_start: u32,
    text_count: u32,
    color_rows: mat4x4<f32>,
    color_bias: vec4<f32>,
    bounds: Bounds,
    content_mask: Bounds,
    content_mask_radii: Corners,
    flags: u32,
    opacity: f32,
    source_rect: vec4<f32>,
    source_row0: vec4<f32>,
    source_row1: vec4<f32>,
}

fn read_vec4_f32(cursor: ptr<function, InstanceCursor>) -> vec4<f32> {
    return vec4<f32>(read_f32(cursor), read_f32(cursor), read_f32(cursor), read_f32(cursor));
}

fn load_vn_layer(layer_id: u32) -> VnLayer {
    // `order` and `group` are the scene's, not the shader's.
    var cursor = instance_cursor(layer_id * 56u + 2u);
    let segment_start = read_word(&cursor);
    let segment_count = read_word(&cursor);
    let fill_rule = read_word(&cursor);
    let matrix_mode = read_word(&cursor);
    let text_start = read_word(&cursor);
    let text_count = read_word(&cursor);
    let row0 = read_vec4_f32(&cursor);
    let row1 = read_vec4_f32(&cursor);
    let row2 = read_vec4_f32(&cursor);
    let row3 = read_vec4_f32(&cursor);
    let bias = read_vec4_f32(&cursor);
    let bounds = read_bounds(&cursor);
    let content_mask = read_bounds(&cursor);
    let content_mask_radii = read_corners(&cursor);
    // The snapshot id is the renderer's.
    _ = read_word(&cursor);
    _ = read_word(&cursor);
    let flags = read_word(&cursor);
    let opacity = read_f32(&cursor);
    let source_rect = read_vec4_f32(&cursor);
    let source_row0 = read_vec4_f32(&cursor);
    let source_row1 = read_vec4_f32(&cursor);
    return VnLayer(
        segment_start,
        segment_count,
        fill_rule,
        matrix_mode,
        text_start,
        text_count,
        mat4x4<f32>(row0, row1, row2, row3),
        bias,
        bounds,
        content_mask,
        content_mask_radii,
        flags,
        opacity,
        source_rect,
        source_row0,
        source_row1,
    );
}

struct VnTextBackground {
    background: Background,
    bounds: Bounds,
    inverse: TransformationMatrix,
    scale_factor: f32,
}

fn load_vn_text_background(word: u32) -> VnTextBackground {
    var cursor = instance_cursor(word);
    let background = read_background(&cursor);
    let bounds = read_bounds(&cursor);
    let inverse = read_transformation(&cursor);
    let scale_factor = read_f32(&cursor);
    return VnTextBackground(background, bounds, inverse, scale_factor);
}

struct VnLayerVarying {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) layer_id: u32,
}

// `bounds` is already cut to the content mask's rectangle.
@vertex
fn vs_vn_layer(@builtin(vertex_index) vertex_id: u32, @builtin(instance_index) instance_id: u32) -> VnLayerVarying {
    let unit_vertex = vec2<f32>(f32(vertex_id & 1u), 0.5 * f32(vertex_id & 2u));
    let layer = load_vn_layer(instance_id);
    var out = VnLayerVarying();
    out.position = to_device_position(unit_vertex, layer.bounds);
    out.layer_id = instance_id;
    return out;
}

// Winding number of the edges crossing the ray to the right (half-open in y,
// so a vertex on the ray counts once), blended across the nearest edge. The
// CPU twin is `gpui::VnClipGeometry::coverage`.
fn vn_clip_coverage(p: vec2<f32>, start: u32, count: u32, fill_rule: u32) -> f32 {
    var winding = 0i;
    var nearest = 3.0e38;
    for (var i = 0u; i < count; i = i + 1u) {
        var cursor = instance_cursor(start + i * 4u);
        let a = read_vec2_f32(&cursor);
        let b = read_vec2_f32(&cursor);
        if ((a.y <= p.y) != (b.y <= p.y)) {
            let x = a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x);
            if (p.x < x) {
                winding += select(-1i, 1i, b.y > a.y);
            }
        }
        let ab = b - a;
        let ap = p - a;
        let len2 = dot(ab, ab);
        let t = select(0.0, saturate(dot(ap, ab) / len2), len2 > 0.0);
        let q = ap - t * ab;
        nearest = min(nearest, dot(q, q));
    }
    let inside = select(winding != 0i, (winding & 1i) != 0i, fill_rule == 1u);
    let d = sqrt(nearest);
    return saturate(0.5 + select(-d, d, inside));
}

// A snapshot composite: the pixel maps into the capture's device space, where
// the snapshot covers `source_rect`.
fn vn_snapshot_color(layer: VnLayer, position: vec2<f32>) -> vec4<f32> {
    let q = vec2<f32>(dot(layer.source_row0.xy, position) + layer.source_row0.z,
        dot(layer.source_row1.xy, position) + layer.source_row1.z);
    let local = q - layer.source_rect.xy;
    let edge = min(min(local.x, local.y),
        min(layer.source_rect.z - local.x, layer.source_rect.w - local.y));
    let coverage = saturate(edge + 0.5)
        * content_mask_coverage(position, layer.content_mask, layer.content_mask_radii);
    let color = textureSampleLevel(t_sprite, s_sprite, local / layer.source_rect.zw, 0.0);
    return color * (coverage * layer.opacity);
}

@fragment
fn fs_vn_layer(input: VnLayerVarying) -> @location(0) vec4<f32> {
    let layer = load_vn_layer(input.layer_id);
    let position = input.position.xy;
    if ((layer.flags & 1u) != 0u) {
        return vn_snapshot_color(layer, position);
    }
    let coverage = vn_clip_coverage(position, layer.segment_start, layer.segment_count, layer.fill_rule)
        * content_mask_coverage(position, layer.content_mask, layer.content_mask_radii);
    var color = textureLoad(t_sprite, vec2<i32>(position), 0);
    if (layer.text_count != 0u) {
        var fill = vec4<f32>(0.0);
        for (var i = 0u; i < layer.text_count; i = i + 1u) {
            let text = load_vn_text_background(layer.text_start + i * 69u);
            var local = transpose(text.inverse.rotation_scale) * position + text.inverse.translation;
            local /= text.scale_factor;
            let prepared = prepare_gradient_color(text.background);
            let layer_color = gradient_color(text.background, local, text.bounds,
                prepared.solid, prepared.color0, prepared.color1);
            let premultiplied = vec4<f32>(layer_color.rgb * layer_color.a, layer_color.a);
            fill = premultiplied + fill * (1.0 - premultiplied.a);
        }
        // The whole stack is masked once, which keeps the glyphs' antialiasing.
        color = fill * color.a;
    }
    if (layer.matrix_mode != 0u) {
        var straight = vec4<f32>(0.0);
        if (color.a > 0.0) {
            straight = vec4<f32>(color.rgb / color.a, color.a);
        }
        if (layer.matrix_mode == 1u) {
            straight = vec4<f32>(srgb_to_linear(straight.rgb), straight.a);
        }
        var filtered = saturate(straight * layer.color_rows + layer.color_bias);
        if (layer.matrix_mode == 1u) {
            filtered = vec4<f32>(linear_to_srgb(filtered.rgb), filtered.a);
        }
        color = vec4<f32>(filtered.rgb * filtered.a, filtered.a);
    }
    return color * (coverage * layer.opacity);
}
