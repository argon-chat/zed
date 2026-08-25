/*
**
**              Custom effects — the engine-owned VERTEX half
**
**  The `--shading` primitive family. One instanced quad per element, shaded by
**  a swappable fragment pipeline; mechanically this is the backdrop blur with a
**  parameter block, and it reuses that machinery wholesale (snapshot,
**  Dual-Kawase pyramid, per-group re-snapshot, replace blend).
**
**  WHY THIS FILE EXISTS AT ALL — i.e. why the vertex stage is not authored in
**  Slang like everything else under crates/vn-effects/shaders:
**
**    gpui clips every primitive with `float4 clip_distance : SV_ClipDistance`.
**    Slang rejects that outright:
**      error[E30701]: type 'vector<float,4>' is not valid for system value
**      semantic 'SV_ClipDistance'; expected 'float' or 'float[]'
**    Spelled as `float clip_distance[4] : SV_ClipDistance` Slang compiles it,
**    but expands the array to SV_ClipDistance0..3, which fxc then rejects:
**      error X4502: invalid output semantic 'SV_ClipDistance': Legal indices
**      are in [0,1]
**    `float2 cd0 : SV_ClipDistance0; float2 cd1 : SV_ClipDistance1;` — which
**    WOULD be legal for fxc — goes back to the E30701 above.
**
**    RE-PROBED at vs_5_0 (2026-08-25, slangc 2026.12 + fxc 10.1): the X4502
**    limit is a Shader Model 4 *and* 5 rule, so raising the effect profile did
**    not lift it. There is no spelling that satisfies both compilers at any
**    profile we can target, so the vertex entry point stays hand-written HLSL.
**
**  That is not a wart we tolerate — it is strictly better. An effect author
**  cannot break the instance fetch, the device-position transform or the
**  content-mask clip, because they never see them.
**
**  ONE vertex shader serves EVERY effect: the fragment half differs per effect,
**  the vertex half never does.
**
**  Compiled at vs_5_0. The rest of gpui stays at 4_1 — only the effect
**  pipelines require D3D feature level 11, and the renderer skips effect draws
**  with a one-shot advisory below it.
*/

cbuffer GlobalParams: register(b0) {
    float4 gamma_ratios;
    float2 global_viewport_size;
    float grayscale_enhanced_contrast;
    float subpixel_enhanced_contrast;
    uint is_bgr;
    uint3 global_pad;
};

cbuffer BatchParams: register(b1) {
    uint batch_start_index;
    uint3 batch_pad;
};

struct Bounds {
    float2 origin;
    float2 size;
};

struct Corners {
    float top_left;
    float top_right;
    float bottom_right;
    float bottom_left;
};

// Keep in sync with Rust `EffectQuad` (crates/gpui/src/scene.rs) and with
// `struct VnEffectQuad` in the generated fragment wrapper
// (packages/vue-native/shaders.ts). 32 words / 128 bytes.
//
// The whole struct is declared even though the vertex stage reads only three
// fields: D3D11 validates a StructuredBuffer's declared stride against the
// view's StructureByteStride and refuses the draw when they disagree.
struct EffectQuad {
    uint order;
    uint effect_id;
    Bounds bounds;
    Bounds content_mask;
    Corners corner_radii;
    float4 params0;
    float4 params1;
    float opacity;
    float time;
    float bleed;
    float backdrop_blur_radius;
    float scale;
    uint flags;
    uint4 pad;
};

StructuredBuffer<EffectQuad> effect_quads: register(t1);

float4 effect_to_device_position(float2 position) {
    float2 device_position = position / global_viewport_size * float2(2.0, -2.0) + float2(-1.0, 1.0);
    return float4(device_position, 0.0, 1.0);
}

float4 effect_distance_from_clip_rect(float2 position, Bounds clip_bounds) {
    float2 tl = position - clip_bounds.origin;
    float2 br = clip_bounds.origin + clip_bounds.size - position;
    return float4(tl.x, br.x, tl.y, br.y);
}

struct EffectVertexOutput {
    nointerpolation uint effect_quad_id: TEXCOORD0;
    float4 position: SV_Position;
    float4 clip_distance: SV_ClipDistance;
};

// One instance draws one effect quad. `bounds` is already the element's border
// box dilated by `bleed`, so nothing here has to know about the dilation.
//
// The two varyings this emits are exactly what the generated fragment stage
// declares as its input (`nointerpolation uint : TEXCOORD0` + `SV_Position`);
// SV_ClipDistance is consumed by the rasterizer and never reaches it.
EffectVertexOutput effect_vertex(uint vertex_id: SV_VertexID, uint instance_id: SV_InstanceID) {
    float2 unit_vertex = float2(float(vertex_id & 1u), 0.5 * float(vertex_id & 2u));
    uint effect_quad_id = batch_start_index + instance_id;
    EffectQuad effect_quad = effect_quads[effect_quad_id];
    float2 position = unit_vertex * effect_quad.bounds.size + effect_quad.bounds.origin;

    EffectVertexOutput output;
    output.position = effect_to_device_position(position);
    output.effect_quad_id = effect_quad_id;
    output.clip_distance = effect_distance_from_clip_rect(position, effect_quad.content_mask);
    return output;
}
