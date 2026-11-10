@group(1) @binding(0) var t_instances: texture_2d<u32>;

// Each texel of `t_instances` packs four 32-bit words of instance data. Most
// records are read strictly front to back through a cursor that keeps the
// most recently fetched texel and only touches the texture again when the next
// word crosses a texel boundary. This fetches each texel exactly once per
// record load instead of once per word. The `read_*` functions must therefore
// consume their words in struct declaration order; skipping or re-reading words
// would still return correct data (the cursor re-fetches whenever the texel
// index changes) but would forfeit the single-fetch-per-texel guarantee.
struct InstanceCursor {
    word_index: u32,
    texel_index: u32,
    texel: vec4<u32>,
    width: u32,
}

fn fetch_instance_texel(texel_index: u32, width: u32) -> vec4<u32> {
    let coordinate = vec2<i32>(
        i32(texel_index % width),
        i32(texel_index / width),
    );
    return textureLoad(t_instances, coordinate, 0);
}

fn instance_cursor(word_index: u32) -> InstanceCursor {
    let width = textureDimensions(t_instances).x;
    let texel_index = word_index / 4u;
    return InstanceCursor(
        word_index,
        texel_index,
        fetch_instance_texel(texel_index, width),
        width,
    );
}

fn read_word(cursor: ptr<function, InstanceCursor>) -> u32 {
    let word_index = (*cursor).word_index;
    let texel_index = word_index / 4u;
    if texel_index != (*cursor).texel_index {
        (*cursor).texel = fetch_instance_texel(texel_index, (*cursor).width);
        (*cursor).texel_index = texel_index;
    }
    (*cursor).word_index = word_index + 1u;
    return (*cursor).texel[word_index % 4u];
}
