// Storage-buffer instance transport. Concatenated after `shaders.wgsl` on backends
// with storage buffer support; `shaders_webgl.wgsl` is the WebGL2 counterpart that
// decodes the same records from a texture instead.
//
// vue-native fork: the buffer is plain words and `shaders_decode.wgsl` reads the
// records out of it, as it does on WebGL. See that file for why no
// `array<Quad>` can describe the Rust layout.
//
// Every pipeline binds its instance buffer at `@group(1) @binding(0)`.

@group(1) @binding(0) var<storage, read> b_instances: array<u32>;

struct InstanceCursor {
    word_index: u32,
}

fn instance_cursor(word_index: u32) -> InstanceCursor {
    return InstanceCursor(word_index);
}

fn read_word(cursor: ptr<function, InstanceCursor>) -> u32 {
    let word_index = (*cursor).word_index;
    (*cursor).word_index = word_index + 1u;
    return b_instances[word_index];
}
