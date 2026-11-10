// Instance record decoding, shared by both transports. Concatenated after
// `shaders_storage.wgsl` or `shaders_webgl.wgsl`, which each define
// `InstanceCursor`, `instance_cursor` and `read_word`.
//
// vue-native fork: the records are read word by word on both transports,
// because WGSL cannot declare them. A `vec2` or `mat2x2` member is 8-byte
// aligned in WGSL and the Rust `#[repr(C)]` records are 4-byte aligned, so
// `Quad` (89 words) and `PathRasterizationVertex` (73) have no WGSL struct
// with their stride and field offsets. `array<Quad>` read every quad past
// the first from the wrong bytes, and wgpu refused a buffer of one quad.

fn read_f32(cursor: ptr<function, InstanceCursor>) -> f32 {
    return bitcast<f32>(read_word(cursor));
}

fn read_i32(cursor: ptr<function, InstanceCursor>) -> i32 {
    return bitcast<i32>(read_word(cursor));
}

fn read_vec2_f32(cursor: ptr<function, InstanceCursor>) -> vec2<f32> {
    return vec2<f32>(read_f32(cursor), read_f32(cursor));
}

fn read_vec2_i32(cursor: ptr<function, InstanceCursor>) -> vec2<i32> {
    return vec2<i32>(read_i32(cursor), read_i32(cursor));
}

fn read_hsla(cursor: ptr<function, InstanceCursor>) -> Hsla {
    return Hsla(
        read_f32(cursor),
        read_f32(cursor),
        read_f32(cursor),
        read_f32(cursor),
    );
}

fn read_bounds(cursor: ptr<function, InstanceCursor>) -> Bounds {
    return Bounds(read_vec2_f32(cursor), read_vec2_f32(cursor));
}

fn read_corners(cursor: ptr<function, InstanceCursor>) -> Corners {
    return Corners(
        read_f32(cursor),
        read_f32(cursor),
        read_f32(cursor),
        read_f32(cursor),
    );
}

fn read_edges(cursor: ptr<function, InstanceCursor>) -> Edges {
    return Edges(
        read_f32(cursor),
        read_f32(cursor),
        read_f32(cursor),
        read_f32(cursor),
    );
}

fn read_color_stop(cursor: ptr<function, InstanceCursor>) -> LinearColorStop {
    return LinearColorStop(read_hsla(cursor), read_f32(cursor));
}

fn read_f32x4(cursor: ptr<function, InstanceCursor>) -> array<f32, 4> {
    return array<f32, 4>(
        read_f32(cursor),
        read_f32(cursor),
        read_f32(cursor),
        read_f32(cursor),
    );
}

// 57 words: 2 + 4 + 1 + 8*5 + 1 + 1 + 4 + 4. This is the ONE place in the
// renderer where a stride mistake is silent rather than a compile error — the
// cursor would simply start reading the next record's bytes — so it moves in
// the same commit as `Background` in `gpui/src/color.rs`.
fn read_background(cursor: ptr<function, InstanceCursor>) -> Background {
    return Background(
        read_word(cursor),
        read_word(cursor),
        read_hsla(cursor),
        read_f32(cursor),
        array<LinearColorStop, 8>(
            read_color_stop(cursor),
            read_color_stop(cursor),
            read_color_stop(cursor),
            read_color_stop(cursor),
            read_color_stop(cursor),
            read_color_stop(cursor),
            read_color_stop(cursor),
            read_color_stop(cursor),
        ),
        read_word(cursor),
        read_word(cursor),
        read_f32x4(cursor),
        read_f32x4(cursor),
    );
}

fn read_atlas_tile(cursor: ptr<function, InstanceCursor>) -> AtlasTile {
    return AtlasTile(
        AtlasTextureId(
            read_word(cursor),
            read_word(cursor),
        ),
        read_word(cursor),
        read_word(cursor),
        AtlasBounds(
            read_vec2_i32(cursor),
            read_vec2_i32(cursor),
        ),
    );
}

fn read_transformation(cursor: ptr<function, InstanceCursor>) -> TransformationMatrix {
    return TransformationMatrix(
        mat2x2<f32>(
            read_vec2_f32(cursor),
            read_vec2_f32(cursor),
        ),
        read_vec2_f32(cursor),
    );
}

fn load_quad(instance_id: u32) -> Quad {
    // 89 words: 26 + the 57 of `Background` + the 6 of the vue-native
    // `transformation` field. (vue-native fork: the 26 include the content
    // mask's corner radii.)
    //
    // vue-native fork: upstream (zed#64337) replaced this with a fixed
    // 10-texel decoder for ITS 40-word quad. Our quad is 85 words with an
    // 8-stop `Background`, so the cursor stays; gpui_web is not in our build.
    var cursor = instance_cursor(instance_id * 89u);
    return Quad(
        read_word(&cursor),
        read_word(&cursor),
        read_bounds(&cursor),
        read_bounds(&cursor),
        read_corners(&cursor),
        read_background(&cursor),
        read_hsla(&cursor),
        read_corners(&cursor),
        read_edges(&cursor),
        read_transformation(&cursor),
    );
}

fn load_shadow(instance_id: u32) -> Shadow {
    var cursor = instance_cursor(instance_id * 38u);
    return Shadow(
        read_word(&cursor),
        read_f32(&cursor),
        read_bounds(&cursor),
        read_corners(&cursor),
        read_bounds(&cursor),
        read_corners(&cursor),
        read_hsla(&cursor),
        read_bounds(&cursor),
        read_corners(&cursor),
        read_word(&cursor),
        read_word(&cursor),
        read_transformation(&cursor),
    );
}

fn load_path_vertex(vertex_id: u32) -> PathRasterizationVertex {
    // 73 words: 8 + the 57 of `Background` + the vue-native content mask's
    // rect and radii.
    var cursor = instance_cursor(vertex_id * 73u);
    return PathRasterizationVertex(
        read_vec2_f32(&cursor),
        read_vec2_f32(&cursor),
        read_background(&cursor),
        read_bounds(&cursor),
        read_bounds(&cursor),
        read_corners(&cursor),
    );
}

fn load_path_sprite(instance_id: u32) -> PathSprite {
    var cursor = instance_cursor(instance_id * 4u);
    return PathSprite(read_bounds(&cursor));
}

fn load_underline(instance_id: u32) -> Underline {
    var cursor = instance_cursor(instance_id * 26u);
    return Underline(
        read_word(&cursor),
        read_word(&cursor),
        read_bounds(&cursor),
        read_bounds(&cursor),
        read_corners(&cursor),
        read_hsla(&cursor),
        read_f32(&cursor),
        read_word(&cursor),
        read_transformation(&cursor),
    );
}

fn load_mono_sprite(instance_id: u32) -> MonochromeSprite {
    var cursor = instance_cursor(instance_id * 32u);
    return MonochromeSprite(
        read_word(&cursor),
        read_word(&cursor),
        read_atlas_tile(&cursor),
        read_bounds(&cursor),
        read_bounds(&cursor),
        read_corners(&cursor),
        read_hsla(&cursor),
        read_transformation(&cursor),
    );
}

fn load_poly_sprite(instance_id: u32) -> PolychromeSprite {
    var cursor = instance_cursor(instance_id * 38u);
    return PolychromeSprite(
        read_word(&cursor),
        read_word(&cursor),
        read_atlas_tile(&cursor),
        read_word(&cursor),
        read_f32(&cursor),
        read_bounds(&cursor),
        read_bounds(&cursor),
        read_corners(&cursor),
        read_corners(&cursor),
        read_transformation(&cursor),
        read_hsla(&cursor),
    );
}
