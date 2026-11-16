#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

// vue-native fork: Background::flags, the background fills only the quad's
// border area. Defined up here, ahead of the other BG_* flags, because
// quad_fragment reads it.
#define BG_RING 2048u

float4 hsla_to_rgba(Hsla hsla);
float3 srgb_to_linear(float3 color);
float3 linear_to_srgb(float3 color);
float4 srgb_to_oklab(float4 color);
float4 oklab_to_srgb(float4 color);
float4 to_device_position(float2 unit_vertex, Bounds_ScaledPixels bounds,
                          constant Size_DevicePixels *viewport_size);
float4 to_device_position_transformed(float2 unit_vertex, Bounds_ScaledPixels bounds,
                          TransformationMatrix transformation,
                          constant Size_DevicePixels *input_viewport_size);

float2 to_tile_position(float2 unit_vertex, AtlasTile tile,
                        constant Size_DevicePixels *atlas_size);
float4 distance_from_clip_rect(float2 unit_vertex, Bounds_ScaledPixels bounds,
                               Bounds_ScaledPixels clip_bounds);
float4 distance_from_clip_rect_transformed(float2 unit_vertex, Bounds_ScaledPixels bounds,
                               Bounds_ScaledPixels clip_bounds, TransformationMatrix transformation);
float corner_dash_velocity(float dv1, float dv2);
float dash_alpha(float t, float period, float length, float dash_velocity,
                 float antialias_threshold);
float quarter_ellipse_sdf(float2 point, float2 radii);
float pick_corner_radius(float2 center_to_point, Corners_ScaledPixels corner_radii);
float quad_sdf(float2 point, Bounds_ScaledPixels bounds,
               Corners_ScaledPixels corner_radii);
float quad_sdf_impl(float2 center_to_point, float corner_radius);
float content_mask_coverage(float2 position, Bounds_ScaledPixels mask,
                            Corners_ScaledPixels radii);
float flat_mask_coverage(float2 position, float4 mask_bounds, float4 mask_radii);
float4 over(float4 below, float4 above);
float radians(float degrees);
float4 fill_color(Background background, float2 position, Bounds_ScaledPixels bounds,
  float4 solid_color, float4 color0, float4 color1);

struct GradientColor {
  float4 solid;
  float4 color0;
  float4 color1;
};
GradientColor prepare_fill_color(uint tag, uint color_space, Hsla solid, Hsla color0, Hsla color1);

struct QuadVertexOutput {
  uint quad_id [[flat]];
  float4 position [[position]];
  // vue-native fork: the pre-transform position, in the same space as
  // `quad.bounds`. Equal to `position.xy` when there is no transform. The
  // fragment shader does all of its SDF work (corner radii, border widths,
  // dash phase) in this untransformed local frame; see `Quad::transformation`.
  float2 local_position;
  float4 border_color [[flat]];
  float4 background_solid [[flat]];
  float4 background_color0 [[flat]];
  float4 background_color1 [[flat]];
  float clip_distance [[clip_distance]][4];
};

struct QuadFragmentInput {
  uint quad_id [[flat]];
  float4 position [[position]];
  float2 local_position;
  float4 border_color [[flat]];
  float4 background_solid [[flat]];
  float4 background_color0 [[flat]];
  float4 background_color1 [[flat]];
};

vertex QuadVertexOutput quad_vertex(uint unit_vertex_id [[vertex_id]],
                                    uint quad_id [[instance_id]],
                                    constant float2 *unit_vertices
                                    [[buffer(QuadInputIndex_Vertices)]],
                                    constant Quad *quads
                                    [[buffer(QuadInputIndex_Quads)]],
                                    constant Size_DevicePixels *viewport_size
                                    [[buffer(QuadInputIndex_ViewportSize)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  Quad quad = quads[quad_id];
  // vue-native fork: CSS `transform` (`Quad::transformation`), applied here
  // only. The clip distances are measured from the TRANSFORMED vertex, since
  // the content mask is in screen space.
  float4 device_position = to_device_position_transformed(
      unit_vertex, quad.bounds, quad.transformation, viewport_size);
  float4 clip_distance = distance_from_clip_rect_transformed(
      unit_vertex, quad.bounds, quad.content_mask.bounds, quad.transformation);
  // Affine in `unit_vertex`, so the rasteriser's interpolation reproduces it
  // exactly across the quad.
  float2 local_position =
      unit_vertex * float2(quad.bounds.size.width, quad.bounds.size.height) +
      float2(quad.bounds.origin.x, quad.bounds.origin.y);
  float4 border_color = hsla_to_rgba(quad.border_color);

  GradientColor gradient = prepare_fill_color(
    quad.background.tag,
    quad.background.color_space,
    quad.background.solid,
    quad.background.colors[0].color,
    quad.background.colors[7].color
  );

  return QuadVertexOutput{
      quad_id,
      device_position,
      local_position,
      border_color,
      gradient.solid,
      gradient.color0,
      gradient.color1,
      {clip_distance.x, clip_distance.y, clip_distance.z, clip_distance.w}};
}

// vue-native fork: `quad_fragment` before the content mask's corners; it
// returns from a handful of places, and the wrapper below takes them all. Its
// inputs are the fragment's varyings, passed one by one; `local_position` is
// the untransformed position (see `QuadVertexOutput`), where the old body read
// `input.position.xy`.
float4 quad_color(Quad quad, float2 local_position, float4 input_border_color,
                  float4 background_solid, float4 background_color0,
                  float4 background_color1) {
  float4 background_color = fill_color(quad.background, local_position, quad.bounds,
    background_solid, background_color0, background_color1);
  // vue-native fork: what fills the inside of the inner border edge. A
  // BG_RING background paints only in the border area, so its inside is
  // transparent.
  float4 interior_color = (quad.background.flags & BG_RING) != 0u
    ? float4(0.0)
    : background_color;

  bool unrounded = quad.corner_radii.top_left == 0.0 &&
    quad.corner_radii.bottom_left == 0.0 &&
    quad.corner_radii.top_right == 0.0 &&
    quad.corner_radii.bottom_right == 0.0;

  // Fast path when the quad is not rounded and doesn't have any border
  if (quad.border_widths.top == 0.0 &&
      quad.border_widths.left == 0.0 &&
      quad.border_widths.right == 0.0 &&
      quad.border_widths.bottom == 0.0 &&
      unrounded) {
    return interior_color;
  }

  float2 size = float2(quad.bounds.size.width, quad.bounds.size.height);
  float2 half_size = size / 2.0;
  float2 point = local_position - float2(quad.bounds.origin.x, quad.bounds.origin.y);
  float2 center_to_point = point - half_size;

  // Signed distance field threshold for inclusion of pixels. 0.5 is the
  // minimum distance between the center of the pixel and the edge.
  const float antialias_threshold = 0.5;

  // Radius of the nearest corner
  float corner_radius = pick_corner_radius(center_to_point, quad.corner_radii);

  // Width of the nearest borders
  float2 border = float2(
    center_to_point.x < 0.0 ? quad.border_widths.left : quad.border_widths.right,
    center_to_point.y < 0.0 ? quad.border_widths.top : quad.border_widths.bottom
  );

  // 0-width borders are reduced so that `inner_sdf >= antialias_threshold`.
  // The purpose of this is to not draw antialiasing pixels in this case.
  float2 reduced_border = float2(
    border.x == 0.0 ? -antialias_threshold : border.x,
    border.y == 0.0 ? -antialias_threshold : border.y);

  // Vector from the corner of the quad bounds to the point, after mirroring
  // the point into the bottom right quadrant. Both components are <= 0.
  float2 corner_to_point = fabs(center_to_point) - half_size;

  // Vector from the point to the center of the rounded corner's circle, also
  // mirrored into bottom right quadrant.
  float2 corner_center_to_point = corner_to_point + corner_radius;

  // Whether the nearest point on the border is rounded
  bool is_near_rounded_corner =
    corner_center_to_point.x >= 0.0 &&
    corner_center_to_point.y >= 0.0;

  // Vector from straight border inner corner to point.
  //
  // 0-width borders are turned into width -1 so that inner_sdf is > 1.0 near
  // the border. Without this, antialiasing pixels would be drawn.
  float2 straight_border_inner_corner_to_point = corner_to_point + reduced_border;

  // Whether the point is beyond the inner edge of the straight border
  bool is_beyond_inner_straight_border =
    straight_border_inner_corner_to_point.x > 0.0 ||
    straight_border_inner_corner_to_point.y > 0.0;


  // Whether the point is far enough inside the quad, such that the pixels are
  // not affected by the straight border.
  bool is_within_inner_straight_border =
    straight_border_inner_corner_to_point.x < -antialias_threshold &&
    straight_border_inner_corner_to_point.y < -antialias_threshold;

  // Fast path for points that must be part of the background
  if (is_within_inner_straight_border && !is_near_rounded_corner) {
    return interior_color;
  }

  // Signed distance of the point to the outside edge of the quad's border
  float outer_sdf = quad_sdf_impl(corner_center_to_point, corner_radius);

  // Approximate signed distance of the point to the inside edge of the quad's
  // border. It is negative outside this edge (within the border), and
  // positive inside.
  //
  // This is not always an accurate signed distance:
  // * The rounded portions with varying border width use an approximation of
  //   nearest-point-on-ellipse.
  // * When it is quickly known to be outside the edge, -1.0 is used.
  float inner_sdf = 0.0;
  if (corner_center_to_point.x <= 0.0 || corner_center_to_point.y <= 0.0) {
    // Fast paths for straight borders
    inner_sdf = -max(straight_border_inner_corner_to_point.x,
                     straight_border_inner_corner_to_point.y);
  } else if (is_beyond_inner_straight_border) {
    // Fast path for points that must be outside the inner edge
    inner_sdf = -1.0;
  } else if (reduced_border.x == reduced_border.y) {
    // Fast path for circular inner edge.
    inner_sdf = -(outer_sdf + reduced_border.x);
  } else {
    float2 ellipse_radii = max(float2(0.0), float2(corner_radius) - reduced_border);
    inner_sdf = quarter_ellipse_sdf(corner_center_to_point, ellipse_radii);
  }

  // Negative when inside the border
  float border_sdf = max(inner_sdf, outer_sdf);

  float4 color = interior_color;
  if (border_sdf < antialias_threshold) {
    float4 border_color = input_border_color;

    // Dashed border logic when border_style == 1
    if (quad.border_style == 1) {
      // Position along the perimeter in "dash space", where each dash
      // period has length 1
      float t = 0.0;

      // Total number of dash periods, so that the dash spacing can be
      // adjusted to evenly divide it
      float max_t = 0.0;

      // Border width is proportional to dash size. This is the behavior
      // used by browsers, but also avoids dashes from different segments
      // overlapping when dash size is smaller than the border width.
      //
      // Dash pattern: (2 * border width) dash, (1 * border width) gap
      const float dash_length_per_width = 2.0;
      const float dash_gap_per_width = 1.0;
      const float dash_period_per_width = dash_length_per_width + dash_gap_per_width;

      // Since the dash size is determined by border width, the density of
      // dashes varies. Multiplying a pixel distance by this returns a
      // position in dash space - it has units (dash period / pixels). So
      // a dash velocity of (1 / 10) is 1 dash every 10 pixels.
      float dash_velocity = 0.0;

      // Dividing this by the border width gives the dash velocity
      const float dv_numerator = 1.0 / dash_period_per_width;

      if (unrounded) {
        // When corners aren't rounded, the dashes are separately laid
        // out on each straight line, rather than around the whole
        // perimeter. This way each line starts and ends with a dash.
        bool is_horizontal = corner_center_to_point.x < corner_center_to_point.y;

        // Choosing the right border width for dashed borders.
        // TODO: A better solution exists taking a look at the whole file.
        // this does not fix single dashed borders at the corners
        float2 dashed_border = float2(
        fmax(quad.border_widths.bottom, quad.border_widths.top),
        fmax(quad.border_widths.right, quad.border_widths.left));

        float border_width = is_horizontal ? dashed_border.x : dashed_border.y;
        dash_velocity = dv_numerator / border_width;
        t = is_horizontal ? point.x : point.y;
        t *= dash_velocity;
        max_t = is_horizontal ? size.x : size.y;
        max_t *= dash_velocity;
      } else {
        // When corners are rounded, the dashes are laid out clockwise
        // around the whole perimeter.

        float r_tr = quad.corner_radii.top_right;
        float r_br = quad.corner_radii.bottom_right;
        float r_bl = quad.corner_radii.bottom_left;
        float r_tl = quad.corner_radii.top_left;

        float w_t = quad.border_widths.top;
        float w_r = quad.border_widths.right;
        float w_b = quad.border_widths.bottom;
        float w_l = quad.border_widths.left;

        // Straight side dash velocities
        float dv_t = w_t <= 0.0 ? 0.0 : dv_numerator / w_t;
        float dv_r = w_r <= 0.0 ? 0.0 : dv_numerator / w_r;
        float dv_b = w_b <= 0.0 ? 0.0 : dv_numerator / w_b;
        float dv_l = w_l <= 0.0 ? 0.0 : dv_numerator / w_l;

        // Straight side lengths in dash space
        float s_t = (size.x - r_tl - r_tr) * dv_t;
        float s_r = (size.y - r_tr - r_br) * dv_r;
        float s_b = (size.x - r_br - r_bl) * dv_b;
        float s_l = (size.y - r_bl - r_tl) * dv_l;

        float corner_dash_velocity_tr = corner_dash_velocity(dv_t, dv_r);
        float corner_dash_velocity_br = corner_dash_velocity(dv_b, dv_r);
        float corner_dash_velocity_bl = corner_dash_velocity(dv_b, dv_l);
        float corner_dash_velocity_tl = corner_dash_velocity(dv_t, dv_l);

        // Corner lengths in dash space
        float c_tr = r_tr * (M_PI_F / 2.0) * corner_dash_velocity_tr;
        float c_br = r_br * (M_PI_F / 2.0) * corner_dash_velocity_br;
        float c_bl = r_bl * (M_PI_F / 2.0) * corner_dash_velocity_bl;
        float c_tl = r_tl * (M_PI_F / 2.0) * corner_dash_velocity_tl;

        // Cumulative dash space upto each segment
        float upto_tr = s_t;
        float upto_r = upto_tr + c_tr;
        float upto_br = upto_r + s_r;
        float upto_b = upto_br + c_br;
        float upto_bl = upto_b + s_b;
        float upto_l = upto_bl + c_bl;
        float upto_tl = upto_l + s_l;
        max_t = upto_tl + c_tl;

        if (is_near_rounded_corner) {
          float radians = atan2(corner_center_to_point.y, corner_center_to_point.x);
          float corner_t = radians * corner_radius;

          if (center_to_point.x >= 0.0) {
            if (center_to_point.y < 0.0) {
              dash_velocity = corner_dash_velocity_tr;
              // Subtracted because radians is pi/2 to 0 when
              // going clockwise around the top right corner,
              // since the y axis has been flipped
              t = upto_r - corner_t * dash_velocity;
            } else {
              dash_velocity = corner_dash_velocity_br;
              // Added because radians is 0 to pi/2 when going
              // clockwise around the bottom-right corner
              t = upto_br + corner_t * dash_velocity;
            }
          } else {
            if (center_to_point.y >= 0.0) {
              dash_velocity = corner_dash_velocity_bl;
              // Subtracted because radians is pi/1 to 0 when
              // going clockwise around the bottom-left corner,
              // since the x axis has been flipped
              t = upto_l - corner_t * dash_velocity;
            } else {
              dash_velocity = corner_dash_velocity_tl;
              // Added because radians is 0 to pi/2 when going
              // clockwise around the top-left corner, since both
              // axis were flipped
              t = upto_tl + corner_t * dash_velocity;
            }
          }
        } else {
          // Straight borders
          bool is_horizontal = corner_center_to_point.x < corner_center_to_point.y;
          if (is_horizontal) {
            if (center_to_point.y < 0.0) {
              dash_velocity = dv_t;
              t = (point.x - r_tl) * dash_velocity;
            } else {
              dash_velocity = dv_b;
              t = upto_bl - (point.x - r_bl) * dash_velocity;
            }
          } else {
            if (center_to_point.x < 0.0) {
              dash_velocity = dv_l;
              t = upto_tl - (point.y - r_tl) * dash_velocity;
            } else {
              dash_velocity = dv_r;
              t = upto_r + (point.y - r_tr) * dash_velocity;
            }
          }
        }
      }

      float dash_length = dash_length_per_width / dash_period_per_width;
      float desired_dash_gap = dash_gap_per_width / dash_period_per_width;

      // Straight borders should start and end with a dash, so max_t is
      // reduced to cause this.
      max_t -= unrounded ? dash_length : 0.0;
      if (max_t >= 1.0) {
        // Adjust dash gap to evenly divide max_t
        float dash_count = floor(max_t);
        float dash_period = max_t / dash_count;
        border_color.a *= dash_alpha(t, dash_period, dash_length, dash_velocity,
                                     antialias_threshold);
      } else if (unrounded) {
        // When there isn't enough space for the full gap between the
        // two start / end dashes of a straight border, reduce gap to
        // make them fit.
        float dash_gap = max_t - dash_length;
        if (dash_gap > 0.0) {
          float dash_period = dash_length + dash_gap;
          border_color.a *= dash_alpha(t, dash_period, dash_length, dash_velocity,
                                       antialias_threshold);
        }
      }
    }

    float inner_coverage = saturate(antialias_threshold - inner_sdf);
    float4 below = background_color;
    if ((quad.background.flags & BG_RING) != 0u) {
      below.a *= inner_coverage;
    }
    border_color.a *= inner_coverage;
    color = over(below, border_color);
  }

  return color * float4(1.0, 1.0, 1.0, saturate(antialias_threshold - outer_sdf));
}

fragment float4 quad_fragment(QuadFragmentInput input [[stage_in]],
                              constant Quad *quads
                              [[buffer(QuadInputIndex_Quads)]]) {
  Quad quad = quads[input.quad_id];
  float4 color = quad_color(quad, input.local_position, input.border_color,
                            input.background_solid, input.background_color0,
                            input.background_color1);
  // vue-native fork: the content mask's rounded corners, in screen space.
  // Straight alpha, like everything `quad_color` returns.
  color.a *= content_mask_coverage(input.position.xy, quad.content_mask.bounds,
                                   quad.content_mask.corner_radii);
  return color;
}

// Returns the dash velocity of a corner given the dash velocity of the two
// sides, by returning the slower velocity (larger dashes).
//
// Since 0 is used for dash velocity when the border width is 0 (instead of
// +inf), this returns the other dash velocity in that case.
//
// An alternative to this might be to appropriately interpolate the dash
// velocity around the corner, but that seems overcomplicated.
float corner_dash_velocity(float dv1, float dv2) {
  if (dv1 == 0.0) {
    return dv2;
  } else if (dv2 == 0.0) {
    return dv1;
  } else {
    return min(dv1, dv2);
  }
}

// Returns alpha used to render antialiased dashes.
// `t` is within the dash when `fmod(t, period) < length`.
float dash_alpha(
    float t, float period, float length, float dash_velocity,
    float antialias_threshold) {
  float half_period = period / 2.0;
  float half_length = length / 2.0;
  // Value in [-half_period, half_period]
  // The dash is in [-half_length, half_length]
  float centered = fmod(t + half_period - half_length, period) - half_period;
  // Signed distance for the dash, negative values are inside the dash
  float signed_distance = abs(centered) - half_length;
  // Antialiased alpha based on the signed distance
  return saturate(antialias_threshold - signed_distance / dash_velocity);
}

// This approximates distance to the nearest point to a quarter ellipse in a way
// that is sufficient for anti-aliasing when the ellipse is not very eccentric.
// The components of `point` are expected to be positive.
//
// Negative on the outside and positive on the inside.
float quarter_ellipse_sdf(float2 point, float2 radii) {
  // Scale the space to treat the ellipse like a unit circle
  float2 circle_vec = point / radii;
  float unit_circle_sdf = length(circle_vec) - 1.0;
  // Approximate up-scaling of the length by using the average of the radii.
  //
  // TODO: A better solution would be to use the gradient of the implicit
  // function for an ellipse to approximate a scaling factor.
  return unit_circle_sdf * (radii.x + radii.y) * -0.5;
}

struct ShadowVertexOutput {
  float4 position [[position]];
  // vue-native fork: the pre-transform position, the frame the shadow shape
  // is evaluated in. See `QuadVertexOutput::local_position`.
  float2 local_position;
  float4 color [[flat]];
  uint shadow_id [[flat]];
  float4 blur_shape [[flat]];
  float4 blur_r1 [[flat]];
  float4 blur_exponent [[flat]];
  float4 blur_scale [[flat]];
  float clip_distance [[clip_distance]][4];
};

struct ShadowFragmentInput {
  float4 position [[position]];
  float2 local_position;
  float4 color [[flat]];
  uint shadow_id [[flat]];
  float4 blur_shape [[flat]];
  float4 blur_r1 [[flat]];
  float4 blur_exponent [[flat]];
  float4 blur_scale [[flat]];
};

struct PreparedBlurredRoundedRect {
  float4 shape;
  float4 r1;
  float4 exponent;
  float4 scale;
};

float erf7(float x) {
  x = clamp(x, -10., 10.) * (2. / sqrt(M_PI_F));
  float xx = x * x;
  x = x + (0.24295 + (0.03395 + 0.0104 * xx) * xx) * (x * xx);
  return x / sqrt(1. + x * x);
}

// Raph Levien's blurred rounded rectangle approximation, whose blur s is sigma * sqrt(2):
// https://raphlinus.github.io/graphics/2020/04/21/blurred-rounded-rects.html
PreparedBlurredRoundedRect prepare_blurred_rounded_rect(float2 size, Corners_ScaledPixels corner_radii, float sigma) {
  float s = max(sigma * sqrt(2.), 1e-6);
  float s_inv = 1. / s;
  float min_edge = min(size.x, size.y);
  float r_max = 0.5 * min_edge;
  float4 radii = float4(corner_radii.top_left, corner_radii.top_right,
                        corner_radii.bottom_right, corner_radii.bottom_left);
  float4 r0 = min(sqrt(radii * radii + 1.15 * 1.15 * s * s), r_max);
  float4 r1 = min(sqrt(radii * radii + 4. * s * s), r_max);

  float2 half_extent = 0.5 * s_inv * size;
  float delta = 1.25 * s * (exp(-half_extent.x * half_extent.x) - exp(-half_extent.y * half_extent.y));
  float2 adjusted_size = float2(size.x + min(delta, 0.), size.y - max(delta, 0.));
  float4 peak = s_inv * 0.5 * (max(adjusted_size.x, adjusted_size.y) - 0.5 * radii);

  PreparedBlurredRoundedRect output;
  output.shape = float4(0.5 * adjusted_size, s_inv, min_edge);
  output.r1 = r1;
  output.exponent = 2. * r1 / r0;
  output.scale = 0.5 * float4(erf7(peak.x), erf7(peak.y), erf7(peak.z), erf7(peak.w));
  return output;
}

float pick_corner(float2 center_to_point, float4 values) {
  Corners_ScaledPixels corners = {values.x, values.y, values.z, values.w};
  return pick_corner_radius(center_to_point, corners);
}

float blurred_rounded_rect(float2 center_to_point, PreparedBlurredRoundedRect prepared) {
  float r1 = pick_corner(center_to_point, prepared.r1);
  float exponent = pick_corner(center_to_point, prepared.exponent);
  float scale = pick_corner(center_to_point, prepared.scale);
  float s_inv = prepared.shape.z;
  float min_edge = prepared.shape.w;

  float2 p0 = abs(center_to_point) - (prepared.shape.xy - r1);
  float2 p1 = max(p0, 0.);
  float d_pos;
  if (p1.x > 0. && p1.y > 0.) {
    d_pos = pow(pow(p1.x, exponent) + pow(p1.y, exponent), 1. / exponent);
  } else {
    d_pos = p1.x + p1.y;
  }

  float d_neg = min(max(p0.x, p0.y), 0.);
  float d = d_pos + d_neg - r1;
  return scale * (erf7(s_inv * (min_edge + d)) - erf7(s_inv * d));
}

vertex ShadowVertexOutput shadow_vertex(
    uint unit_vertex_id [[vertex_id]], uint shadow_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(ShadowInputIndex_Vertices)]],
    constant Shadow *shadows [[buffer(ShadowInputIndex_Shadows)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(ShadowInputIndex_ViewportSize)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  Shadow shadow = shadows[shadow_id];
  Bounds_ScaledPixels bounds = shadow.bounds;

  // vue-native fork: CSS `transform`; a shadow is part of the element's
  // painted output, so it turns with the box. See `quad_vertex`.
  float4 device_position = to_device_position_transformed(
      unit_vertex, bounds, shadow.transformation, viewport_size);
  float4 clip_distance = distance_from_clip_rect_transformed(
      unit_vertex, bounds, shadow.content_mask.bounds, shadow.transformation);
  float2 local_position =
      unit_vertex * float2(bounds.size.width, bounds.size.height) +
      float2(bounds.origin.x, bounds.origin.y);
  float4 color = hsla_to_rgba(shadow.color);
  PreparedBlurredRoundedRect blur = prepare_blurred_rounded_rect(
      float2(shadow.shape_bounds.size.width, shadow.shape_bounds.size.height),
      shadow.shape_corner_radii, shadow.blur_radius);

  return ShadowVertexOutput{
      device_position,
      local_position,
      color,
      shadow_id,
      blur.shape,
      blur.r1,
      blur.exponent,
      blur.scale,
      {clip_distance.x, clip_distance.y, clip_distance.z, clip_distance.w}};
}

fragment float4 shadow_fragment(ShadowFragmentInput input [[stage_in]],
                                constant Shadow *shadows
                                [[buffer(ShadowInputIndex_Shadows)]]) {
  Shadow shadow = shadows[input.shadow_id];

  // vue-native fork: every SDF below runs in the untransformed local frame.
  float alpha;
  if (shadow.blur_radius == 0.) {
    float distance = quad_sdf(input.local_position, shadow.shape_bounds,
                              shadow.shape_corner_radii);
    alpha = saturate(0.5 - distance);
  } else {
    float2 origin = float2(shadow.shape_bounds.origin.x, shadow.shape_bounds.origin.y);
    float2 size = float2(shadow.shape_bounds.size.width, shadow.shape_bounds.size.height);
    float2 center = origin + size / 2.;
    PreparedBlurredRoundedRect blur = {input.blur_shape, input.blur_r1,
                                       input.blur_exponent, input.blur_scale};
    alpha = blurred_rounded_rect(input.local_position - center, blur);
  }

  if (shadow.inset == 1u) {
    // The inset shadow is the complement of the (blurred) hole rect, clipped to the element.
    // `saturate(0.5 - d)` gives a 1-pixel antialiased edge: d <= -0.5 -> 1, d >= 0.5 -> 0.
    alpha = 1. - alpha;
    float element_distance = quad_sdf(input.local_position, shadow.element_bounds,
                                      shadow.corner_radii);
    alpha *= saturate(0.5 - element_distance);
  } else if (shadow.inset == 0u) {
    // vue-native fork: an outer shadow is clipped to the outside of the element's
    // border box (CSS Backgrounds 3 §7.1), so a translucent background shows what is
    // behind the element. inset == 2 is the unclipped variant.
    float element_distance = quad_sdf(input.local_position, shadow.element_bounds,
                                      shadow.corner_radii);
    alpha *= saturate(0.5 + element_distance);
  }

  // vue-native fork: the content mask's rounded corners, in screen space.
  alpha *= content_mask_coverage(input.position.xy, shadow.content_mask.bounds,
                                 shadow.content_mask.corner_radii);
  return input.color * float4(1., 1., 1., alpha);
}

struct UnderlineVertexOutput {
  float4 position [[position]];
  // vue-native fork: the pre-transform position. The wave below is
  // parameterised along the underline's own axis, not the screen's.
  float2 local_position;
  float4 color [[flat]];
  uint underline_id [[flat]];
  float clip_distance [[clip_distance]][4];
};

struct UnderlineFragmentInput {
  float4 position [[position]];
  float2 local_position;
  float4 color [[flat]];
  uint underline_id [[flat]];
};

vertex UnderlineVertexOutput underline_vertex(
    uint unit_vertex_id [[vertex_id]], uint underline_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(UnderlineInputIndex_Vertices)]],
    constant Underline *underlines [[buffer(UnderlineInputIndex_Underlines)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(ShadowInputIndex_ViewportSize)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  Underline underline = underlines[underline_id];
  // vue-native fork: CSS `transform`; an underline rides with the text it
  // decorates. See `quad_vertex`.
  float4 device_position = to_device_position_transformed(
      unit_vertex, underline.bounds, underline.transformation, viewport_size);
  float4 clip_distance = distance_from_clip_rect_transformed(
      unit_vertex, underline.bounds, underline.content_mask.bounds,
      underline.transformation);
  float2 local_position =
      unit_vertex * float2(underline.bounds.size.width, underline.bounds.size.height) +
      float2(underline.bounds.origin.x, underline.bounds.origin.y);
  float4 color = hsla_to_rgba(underline.color);
  return UnderlineVertexOutput{
      device_position,
      local_position,
      color,
      underline_id,
      {clip_distance.x, clip_distance.y, clip_distance.z, clip_distance.w}};
}

fragment float4 underline_fragment(UnderlineFragmentInput input [[stage_in]],
                                   constant Underline *underlines
                                   [[buffer(UnderlineInputIndex_Underlines)]]) {
  const float WAVE_FREQUENCY = 2.0;
  const float WAVE_HEIGHT_RATIO = 0.8;

  Underline underline = underlines[input.underline_id];
  // vue-native fork: the content mask's rounded corners, in screen space.
  float mask = content_mask_coverage(input.position.xy,
                                     underline.content_mask.bounds,
                                     underline.content_mask.corner_radii);
  if (underline.wavy) {
    float half_thickness = underline.thickness * 0.5;
    float2 origin =
        float2(underline.bounds.origin.x, underline.bounds.origin.y);

    float2 st = ((input.local_position - origin) / underline.bounds.size.height) -
                float2(0., 0.5);
    float frequency = (M_PI_F * WAVE_FREQUENCY * underline.thickness) / underline.bounds.size.height;
    float amplitude = (underline.thickness * WAVE_HEIGHT_RATIO) / underline.bounds.size.height;

    float sine = sin(st.x * frequency) * amplitude;
    float dSine = cos(st.x * frequency) * amplitude * frequency;
    float distance = (st.y - sine) / sqrt(1. + dSine * dSine);
    float distance_in_pixels = distance * underline.bounds.size.height;
    float distance_from_top_border = distance_in_pixels - half_thickness;
    float distance_from_bottom_border = distance_in_pixels + half_thickness;
    float alpha = saturate(
        0.5 - max(-distance_from_bottom_border, distance_from_top_border));
    return input.color * float4(1., 1., 1., alpha * mask);
  } else {
    return input.color * float4(1., 1., 1., mask);
  }
}

// vue-native fork: `sprite_id` carries the sprite to the fragment shader for
// its content mask's rounded corners. (gpui_windows and gpui_wgpu pass the
// mask as flat varyings instead; here the sprites buffer is already bound to
// the fragment stage, so the id is enough.)
struct MonochromeSpriteVertexOutput {
  float4 position [[position]];
  float2 tile_position;
  float4 color [[flat]];
  float4 clip_distance;
  uint sprite_id [[flat]];
};

struct MonochromeSpriteFragmentInput {
  float4 position [[position]];
  float2 tile_position;
  float4 color [[flat]];
  float4 clip_distance;
  uint sprite_id [[flat]];
};

vertex MonochromeSpriteVertexOutput monochrome_sprite_vertex(
    uint unit_vertex_id [[vertex_id]], uint sprite_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(SpriteInputIndex_Vertices)]],
    constant MonochromeSprite *sprites [[buffer(SpriteInputIndex_Sprites)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(SpriteInputIndex_ViewportSize)]],
    constant Size_DevicePixels *atlas_size
    [[buffer(SpriteInputIndex_AtlasTextureSize)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  MonochromeSprite sprite = sprites[sprite_id];
  float4 device_position =
      to_device_position_transformed(unit_vertex, sprite.bounds, sprite.transformation, viewport_size);
  float4 clip_distance = distance_from_clip_rect_transformed(unit_vertex, sprite.bounds,
                                                 sprite.content_mask.bounds, sprite.transformation);
  float2 tile_position = to_tile_position(unit_vertex, sprite.tile, atlas_size);
  float4 color = hsla_to_rgba(sprite.color);
  return MonochromeSpriteVertexOutput{
      device_position,
      tile_position,
      color,
      {clip_distance.x, clip_distance.y, clip_distance.z, clip_distance.w},
      sprite_id};
}

// Glyph contrast and gamma correction, as gpui_windows' alpha_correction.hlsl
// and gpui_wgpu apply it. Adapted from
// https://github.com/microsoft/terminal/blob/1283c0f5b99a2961673249fa77c6b986efb5086c/src/renderer/atlas/dwrite.hlsl
// Copyright (c) Microsoft Corporation. Licensed under the MIT license.
struct SpriteGammaParams {
  float4 gamma_ratios;
  float grayscale_enhanced_contrast;
};

float glyph_alpha_correction(float sample, float3 color,
                             constant SpriteGammaParams *params) {
  float brightness = dot(color, float3(0.30, 0.59, 0.11));
  float k = params->grayscale_enhanced_contrast *
            saturate(4.0 * (0.75 - brightness));
  float a = sample * (k + 1.0) / (sample * k + 1.0);
  float4 g = params->gamma_ratios;
  float correction = (g.x * brightness + g.y) * a + (g.z * brightness + g.w);
  return a + a * (1.0 - a) * correction;
}

fragment float4 monochrome_sprite_fragment(
    MonochromeSpriteFragmentInput input [[stage_in]],
    constant MonochromeSprite *sprites [[buffer(SpriteInputIndex_Sprites)]],
    constant SpriteGammaParams *gamma_params
    [[buffer(SpriteInputIndex_GammaParams)]],
    texture2d<float> atlas_texture [[texture(SpriteInputIndex_AtlasTexture)]]) {
  if (any(input.clip_distance < float4(0.0))) {
    return float4(0.0);
  }

  constexpr sampler atlas_texture_sampler(mag_filter::linear,
                                          min_filter::linear);
  float4 sample =
      atlas_texture.sample(atlas_texture_sampler, input.tile_position);
  float4 color = input.color;
  // vue-native fork: the content mask's rounded corners, in screen space.
  MonochromeSprite sprite = sprites[input.sprite_id];
  float coverage = sprite.raw_coverage
                       ? sample.a
                       : glyph_alpha_correction(sample.a, color.rgb, gamma_params);
  color.a *= coverage *
             content_mask_coverage(input.position.xy,
                                              sprite.content_mask.bounds,
                                              sprite.content_mask.corner_radii);
  return color;
}

struct PolychromeSpriteVertexOutput {
  float4 position [[position]];
  float2 tile_position;
  // vue-native fork: the pre-transform position. The corner-radius SDF is
  // evaluated against the untransformed `sprite.bounds`.
  float2 local_position;
  uint sprite_id [[flat]];
  float clip_distance [[clip_distance]][4];
};

struct PolychromeSpriteFragmentInput {
  float4 position [[position]];
  float2 tile_position;
  float2 local_position;
  uint sprite_id [[flat]];
};

vertex PolychromeSpriteVertexOutput polychrome_sprite_vertex(
    uint unit_vertex_id [[vertex_id]], uint sprite_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(SpriteInputIndex_Vertices)]],
    constant PolychromeSprite *sprites [[buffer(SpriteInputIndex_Sprites)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(SpriteInputIndex_ViewportSize)]],
    constant Size_DevicePixels *atlas_size
    [[buffer(SpriteInputIndex_AtlasTextureSize)]]) {

  float2 unit_vertex = unit_vertices[unit_vertex_id];
  PolychromeSprite sprite = sprites[sprite_id];
  // vue-native fork: CSS `transform`; an <img> (or a colour emoji) inside a
  // rotated box rotates with it. See `quad_vertex`.
  float4 device_position = to_device_position_transformed(
      unit_vertex, sprite.bounds, sprite.transformation, viewport_size);
  float4 clip_distance = distance_from_clip_rect_transformed(
      unit_vertex, sprite.bounds, sprite.content_mask.bounds, sprite.transformation);
  float2 tile_position = to_tile_position(unit_vertex, sprite.tile, atlas_size);
  float2 local_position =
      unit_vertex * float2(sprite.bounds.size.width, sprite.bounds.size.height) +
      float2(sprite.bounds.origin.x, sprite.bounds.origin.y);
  return PolychromeSpriteVertexOutput{
      device_position,
      tile_position,
      local_position,
      sprite_id,
      {clip_distance.x, clip_distance.y, clip_distance.z, clip_distance.w}};
}

fragment float4 polychrome_sprite_fragment(
    PolychromeSpriteFragmentInput input [[stage_in]],
    constant PolychromeSprite *sprites [[buffer(SpriteInputIndex_Sprites)]],
    texture2d<float> atlas_texture [[texture(SpriteInputIndex_AtlasTexture)]]) {
  PolychromeSprite sprite = sprites[input.sprite_id];
  constexpr sampler atlas_texture_sampler(mag_filter::linear,
                                          min_filter::linear);
  float4 sample =
      atlas_texture.sample(atlas_texture_sampler, input.tile_position);
  // vue-native fork: the sprite's own corners in its untransformed frame.
  float distance =
      quad_sdf(input.local_position, sprite.bounds, sprite.corner_radii);

  float4 color = sample;
  // vue-native fork: the tint takes the pixels by its alpha, keeping theirs
  // (canvas `source-atop`, in straight alpha). `PolychromeSprite` comes from
  // cbindgen (build.rs), so its `tint` is already there.
  if (sprite.tint.a > 0.0) {
    float4 tint = hsla_to_rgba(sprite.tint);
    color.rgb = mix(color.rgb, tint.rgb, tint.a);
  }
  if (sprite.grayscale) {
    float grayscale = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
    color.r = grayscale;
    color.g = grayscale;
    color.b = grayscale;
  }
  // vue-native fork: times the content mask's rounded corners, in screen space.
  color.a *= sprite.opacity * saturate(0.5 - distance) *
      content_mask_coverage(input.position.xy, sprite.content_mask.bounds,
                            sprite.content_mask.corner_radii);
  return color;
}

struct PathTileFragmentInput {
  float4 position [[position]];
  uint tile_id [[flat]];
  float4 curve_box [[flat]];
  float4 background_solid [[flat]];
  float4 background_color0 [[flat]];
  float4 background_color1 [[flat]];
};

vertex PathTileFragmentInput path_tile_vertex(
    uint vertex_id [[vertex_id]], uint tile_id [[instance_id]],
    constant PathTile *tiles [[buffer(PathTileInputIndex_Tiles)]],
    constant PathPaint *paints [[buffer(PathTileInputIndex_Paints)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(PathTileInputIndex_ViewportSize)]]) {
  PathTile tile = tiles[tile_id];
  PathPaint paint = paints[tile.paint];

  // Neighboring tiles store their shared edge as the same number, so clipping
  // each corner on its own gives them bit-identical edges, which the
  // rasterizer neither overlaps nor leaves gaps between.
  float2 mask_top_left = float2(paint.content_mask.bounds.origin.x,
                                paint.content_mask.bounds.origin.y);
  float2 mask_bottom_right =
      mask_top_left + float2(paint.content_mask.bounds.size.width,
                             paint.content_mask.bounds.size.height);
  float2 top_left = clamp(float2(tile.top_left.x, tile.top_left.y),
                          mask_top_left, mask_bottom_right);
  float2 bottom_right = clamp(float2(tile.bottom_right.x, tile.bottom_right.y),
                              top_left, mask_bottom_right);
  float2 position =
      float2((vertex_id & 1u) != 0u ? bottom_right.x : top_left.x,
             (vertex_id & 2u) != 0u ? bottom_right.y : top_left.y);
  float4 device_position = float4(
      position * float2(2. / viewport_size->width, -2. / viewport_size->height) +
          float2(-1., 1.),
      0., 1.);

  GradientColor gradient = prepare_fill_color(
      paint.color.tag, paint.color.color_space, paint.color.solid,
      paint.color.colors[0].color, paint.color.colors[7].color);

  return PathTileFragmentInput{
      device_position,
      tile_id,
      float4(tile.curves_top_left.x, tile.curves_top_left.y,
             tile.curves_bottom_right.x, tile.curves_bottom_right.y),
      gradient.solid,
      gradient.color0,
      gradient.color1};
}

// Solves a*t^2 + b*t + c = 0 for the root in [0, 1] on a monotone piece.
float monotone_quadratic_root(float a, float b, float c, float direction) {
  float discriminant = max(b * b - 4.0 * a * c, 0.0);
  float denominator = b + direction * sqrt(discriminant);
  if (denominator == 0.0) {
    return 0.0;
  }
  return saturate(-2.0 * c / denominator);
}

// Coverage estimate adapted from https://github.com/EricLengyel/Slug/blob/be3c13eb7d63f9e8aa5c583e42d92c374cb91d98/SlugPixelShader.hlsl
// Copyright (c) 2017 Eric Lengyel.
// Licensed under the MIT license.

// Accumulates the curve's winding along the pixel's leftward and upward rays.
// Like Slug, a crossing within half a pixel of the center counts fractionally.
void add_curve_winding(PathCurve curve, float2 center, thread float2 &winding,
                       thread float2 &weight) {
  float2 p0 = float2(curve.p0.x, curve.p0.y);
  float2 p1 = float2(curve.p1.x, curve.p1.y);
  float2 a = float2(curve.a.x, curve.a.y);
  float2 b = float2(curve.b.x, curve.b.y);
  float min_x = min(p0.x, p1.x);
  float max_x = max(p0.x, p1.x);

  if (p0.y <= center.y && center.y < p1.y) {
    if (max_x <= center.x - 0.5) {
      winding.x += curve.sign;
    } else {
      float t = monotone_quadratic_root(a.y, b.y, p0.y - center.y, 1.0);
      float x = (a.x * t + b.x) * t + p0.x;
      float distance = center.x - x;
      winding.x += curve.sign * saturate(distance + 0.5);
      weight.x = max(weight.x, saturate(1.0 - 2.0 * abs(distance)));
    }
  }

  if (min_x <= center.x && center.x < max_x) {
    float direction = p1.x > p0.x ? 1.0 : -1.0;
    if (p1.y <= center.y - 0.5) {
      winding.y -= curve.sign * direction;
    } else if (p0.y < center.y + 0.5) {
      float t = monotone_quadratic_root(a.x, b.x, p0.x - center.x, direction);
      float y = (a.y * t + b.y) * t + p0.y;
      float distance = center.y - y;
      winding.y -= curve.sign * direction * saturate(distance + 0.5);
      weight.y = max(weight.y, saturate(1.0 - 2.0 * abs(distance)));
    }
  }
}

// The backdrop gives the unlisted curves' winding along a leftward ray from
// the tile's reference corner. We need it at the current pixel, along both
// leftward and upward rays. Because paths are closed, chains of unlisted
// curves can only begin or end where a listed curve does. So, we can compute
// the winding at the current pixel by correcting the backdrop for each listed
// endpoint between its ray and the pixel.
void correct_for_endpoint(float2 endpoint, float step, float2 reference,
                          float2 center, thread float2 &winding) {
  if (endpoint.x <= reference.x) {
    winding.x -= step * (float(endpoint.y <= center.y) -
                         float(endpoint.y <= reference.y));
  }
  if (endpoint.y <= reference.y && endpoint.x <= center.x) {
    winding.y += step;
  }
}

fragment float4 path_tile_fragment(
    PathTileFragmentInput input [[stage_in]],
    constant PathTile *tiles [[buffer(PathTileInputIndex_Tiles)]],
    constant PathCurve *curves [[buffer(PathTileInputIndex_Curves)]],
    constant PathPaint *paints [[buffer(PathTileInputIndex_Paints)]]) {
  PathTile tile = tiles[input.tile_id];
  PathPaint paint = paints[tile.paint];
  float2 center = floor(input.position.xy) + 0.5;

  float coverage = 1.0;
  if (all(center >= input.curve_box.xy) && all(center <= input.curve_box.zw)) {
    float2 reference = float2(tile.reference.x, tile.reference.y);
    float2 winding = float2(float(tile.backdrop));
    float2 weight = float2(0.0);
    for (uint i = 0u; i < tile.curve_count; i++) {
      PathCurve curve = curves[tile.curve_start + i];
      float2 p0 = float2(curve.p0.x, curve.p0.y);
      float2 p1 = float2(curve.p1.x, curve.p1.y);
      if (min(p0.x, p1.x) >= center.x + 0.5) {
        break;
      }
      if (p0.y < center.y + 0.5) {
        add_curve_winding(curve, center, winding, weight);
        correct_for_endpoint(p0, curve.sign, reference, center, winding);
        correct_for_endpoint(p1, -curve.sign, reference, center, winding);
      }
    }

    // Weight the winding computed by each ray by its nearest crossing's
    // distance from the pixel center, using Slug's technique.
    coverage =
        max(abs(dot(winding, weight)) / max(weight.x + weight.y, 1.0 / 65536.0),
            min(abs(winding.x), abs(winding.y)));
    coverage = paint.even_odd != 0u
                   ? 1.0 - abs(1.0 - fract(coverage * 0.5) * 2.0)
                   : saturate(coverage);
  }

  // vue-native fork: the content mask's rounded corners, in screen space.
  coverage *= content_mask_coverage(input.position.xy, paint.content_mask.bounds,
                                    paint.content_mask.corner_radii);
  if (coverage == 0.0) {
    return float4(0.0);
  }

  float4 color = fill_color(paint.color, input.position.xy, paint.bounds,
                            input.background_solid, input.background_color0,
                            input.background_color1);
  return float4(color.rgb, color.a * coverage);
}

// vue-native fork: the surfaces buffer is not bound to the fragment stage, so
// the content mask's rect and radii ride along as flat varyings (as the
// glyph pipelines' do in gpui_windows' shaders.hlsl); see `flat_mask_coverage`.
struct SurfaceVertexOutput {
  float4 position [[position]];
  float2 texture_position;
  float4 mask_bounds [[flat]];
  float4 mask_radii [[flat]];
  float clip_distance [[clip_distance]][4];
};

struct SurfaceFragmentInput {
  float4 position [[position]];
  float2 texture_position;
  float4 mask_bounds [[flat]];
  float4 mask_radii [[flat]];
};

vertex SurfaceVertexOutput surface_vertex(
    uint unit_vertex_id [[vertex_id]], uint surface_id [[instance_id]],
    constant float2 *unit_vertices [[buffer(SurfaceInputIndex_Vertices)]],
    constant SurfaceBounds *surfaces [[buffer(SurfaceInputIndex_Surfaces)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(SurfaceInputIndex_ViewportSize)]],
    constant Size_DevicePixels *texture_size
    [[buffer(SurfaceInputIndex_TextureSize)]]) {
  float2 unit_vertex = unit_vertices[unit_vertex_id];
  SurfaceBounds surface = surfaces[surface_id];
  float4 device_position =
      to_device_position(unit_vertex, surface.bounds, viewport_size);
  float4 clip_distance = distance_from_clip_rect(unit_vertex, surface.bounds,
                                                 surface.content_mask.bounds);
  // We are going to copy the whole texture, so the texture position corresponds
  // to the current vertex of the unit triangle.
  float2 texture_position = unit_vertex;
  float4 mask_bounds = float4(
      surface.content_mask.bounds.origin.x, surface.content_mask.bounds.origin.y,
      surface.content_mask.bounds.size.width, surface.content_mask.bounds.size.height);
  float4 mask_radii = float4(
      surface.content_mask.corner_radii.top_left,
      surface.content_mask.corner_radii.top_right,
      surface.content_mask.corner_radii.bottom_right,
      surface.content_mask.corner_radii.bottom_left);
  return SurfaceVertexOutput{
      device_position,
      texture_position,
      mask_bounds,
      mask_radii,
      {clip_distance.x, clip_distance.y, clip_distance.z, clip_distance.w}};
}

fragment float4 surface_fragment(SurfaceFragmentInput input [[stage_in]],
                                 texture2d<float> y_texture
                                 [[texture(SurfaceInputIndex_YTexture)]],
                                 texture2d<float> cb_cr_texture
                                 [[texture(SurfaceInputIndex_CbCrTexture)]]) {
  constexpr sampler texture_sampler(mag_filter::linear, min_filter::linear);
  const float4x4 ycbcrToRGBTransform =
      float4x4(float4(+1.0000f, +1.0000f, +1.0000f, +0.0000f),
               float4(+0.0000f, -0.3441f, +1.7720f, +0.0000f),
               float4(+1.4020f, -0.7141f, +0.0000f, +0.0000f),
               float4(-0.7010f, +0.5291f, -0.8860f, +1.0000f));
  float4 ycbcr = float4(
      y_texture.sample(texture_sampler, input.texture_position).r,
      cb_cr_texture.sample(texture_sampler, input.texture_position).rg, 1.0);

  float4 color = ycbcrToRGBTransform * ycbcr;
  // vue-native fork: the content mask's rounded corners, in screen space.
  color.a *= flat_mask_coverage(input.position.xy, input.mask_bounds,
                                input.mask_radii);
  return color;
}

float4 hsla_to_rgba(Hsla hsla) {
  float h = hsla.h * 6.0; // Now, it's an angle but scaled in [0, 6) range
  float s = hsla.s;
  float l = hsla.l;
  float a = hsla.a;

  float c = (1.0 - fabs(2.0 * l - 1.0)) * s;
  float x = c * (1.0 - fabs(fmod(h, 2.0) - 1.0));
  float m = l - c / 2.0;

  float r = 0.0;
  float g = 0.0;
  float b = 0.0;

  if (h >= 0.0 && h < 1.0) {
    r = c;
    g = x;
    b = 0.0;
  } else if (h >= 1.0 && h < 2.0) {
    r = x;
    g = c;
    b = 0.0;
  } else if (h >= 2.0 && h < 3.0) {
    r = 0.0;
    g = c;
    b = x;
  } else if (h >= 3.0 && h < 4.0) {
    r = 0.0;
    g = x;
    b = c;
  } else if (h >= 4.0 && h < 5.0) {
    r = x;
    g = 0.0;
    b = c;
  } else {
    r = c;
    g = 0.0;
    b = x;
  }

  float4 rgba;
  rgba.x = (r + m);
  rgba.y = (g + m);
  rgba.z = (b + m);
  rgba.w = a;
  return rgba;
}

// The exact sRGB curves, as browsers and gpui_wgpu use for Oklab gradients.
float3 srgb_to_linear(float3 color) {
  return select(pow((color + 0.055) / 1.055, float3(2.4)), color / 12.92,
                color < 0.04045);
}

float3 linear_to_srgb(float3 color) {
  return select(1.055 * pow(abs(color), float3(1.0 / 2.4)) - 0.055,
                color * 12.92, color < 0.0031308);
}

// Converts a sRGB color to the Oklab color space.
// Reference: https://bottosson.github.io/posts/oklab/#converting-from-linear-srgb-to-oklab
float4 srgb_to_oklab(float4 color) {
  // Convert non-linear sRGB to linear sRGB
  color = float4(srgb_to_linear(color.rgb), color.a);

  float l = 0.4122214708 * color.r + 0.5363325363 * color.g + 0.0514459929 * color.b;
  float m = 0.2119034982 * color.r + 0.6806995451 * color.g + 0.1073969566 * color.b;
  float s = 0.0883024619 * color.r + 0.2817188376 * color.g + 0.6299787005 * color.b;

  float l_ = pow(l, 1.0/3.0);
  float m_ = pow(m, 1.0/3.0);
  float s_ = pow(s, 1.0/3.0);

  return float4(
   	0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_,
   	1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_,
   	0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_,
   	color.a
  );
}

// Converts an Oklab color to the sRGB color space.
float4 oklab_to_srgb(float4 color) {
  float l_ = color.r + 0.3963377774 * color.g + 0.2158037573 * color.b;
  float m_ = color.r - 0.1055613458 * color.g - 0.0638541728 * color.b;
  float s_ = color.r - 0.0894841775 * color.g - 1.2914855480 * color.b;

  float l = l_ * l_ * l_;
  float m = m_ * m_ * m_;
  float s = s_ * s_ * s_;

  float3 linear_rgb = float3(
   	4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
   	-1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
   	-0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
  );

  // Convert linear sRGB to non-linear sRGB
  return float4(linear_to_srgb(linear_rgb), color.a);
}

float4 to_device_position(float2 unit_vertex, Bounds_ScaledPixels bounds,
                          constant Size_DevicePixels *input_viewport_size) {
  float2 position =
      unit_vertex * float2(bounds.size.width, bounds.size.height) +
      float2(bounds.origin.x, bounds.origin.y);
  float2 viewport_size = float2((float)input_viewport_size->width,
                                (float)input_viewport_size->height);
  float2 device_position =
      position / viewport_size * float2(2., -2.) + float2(-1., 1.);
  return float4(device_position, 0., 1.);
}

float4 to_device_position_transformed(float2 unit_vertex, Bounds_ScaledPixels bounds,
                          TransformationMatrix transformation,
                          constant Size_DevicePixels *input_viewport_size) {
  float2 position =
      unit_vertex * float2(bounds.size.width, bounds.size.height) +
      float2(bounds.origin.x, bounds.origin.y);

  // Apply the transformation matrix to the position via matrix multiplication.
  float2 transformed_position = float2(0, 0);
  transformed_position[0] = position[0] * transformation.rotation_scale[0][0] + position[1] * transformation.rotation_scale[0][1];
  transformed_position[1] = position[0] * transformation.rotation_scale[1][0] + position[1] * transformation.rotation_scale[1][1];

  // Add in the translation component of the transformation matrix.
  transformed_position[0] += transformation.translation[0];
  transformed_position[1] += transformation.translation[1];

  float2 viewport_size = float2((float)input_viewport_size->width,
                                (float)input_viewport_size->height);
  float2 device_position =
      transformed_position / viewport_size * float2(2., -2.) + float2(-1., 1.);
  return float4(device_position, 0., 1.);
}


float2 to_tile_position(float2 unit_vertex, AtlasTile tile,
                        constant Size_DevicePixels *atlas_size) {
  float2 tile_origin = float2(tile.bounds.origin.x, tile.bounds.origin.y);
  float2 tile_size = float2(tile.bounds.size.width, tile.bounds.size.height);
  return (tile_origin + unit_vertex * tile_size) /
         float2((float)atlas_size->width, (float)atlas_size->height);
}

// Selects corner radius based on quadrant.
float pick_corner_radius(float2 center_to_point, Corners_ScaledPixels corner_radii) {
  if (center_to_point.x < 0.) {
    if (center_to_point.y < 0.) {
      return corner_radii.top_left;
    } else {
      return corner_radii.bottom_left;
    }
  } else {
    if (center_to_point.y < 0.) {
      return corner_radii.top_right;
    } else {
      return corner_radii.bottom_right;
    }
  }
}

// Signed distance of the point to the quad's border - positive outside the
// border, and negative inside.
float quad_sdf(float2 point, Bounds_ScaledPixels bounds,
               Corners_ScaledPixels corner_radii) {
    float2 half_size = float2(bounds.size.width, bounds.size.height) / 2.0;
    float2 center = float2(bounds.origin.x, bounds.origin.y) + half_size;
    float2 center_to_point = point - center;
    float corner_radius = pick_corner_radius(center_to_point, corner_radii);
    float2 corner_to_point = fabs(center_to_point) - half_size;
    float2 corner_center_to_point = corner_to_point + corner_radius;
    return quad_sdf_impl(corner_center_to_point, corner_radius);
}

// Implementation of quad signed distance field
float quad_sdf_impl(float2 corner_center_to_point, float corner_radius) {
    if (corner_radius == 0.0) {
        // Fast path for unrounded corners
        return max(corner_center_to_point.x, corner_center_to_point.y);
    } else {
        // Signed distance of the point from a quad that is inset by corner_radius
        // It is negative inside this quad, and positive outside
        float signed_distance_to_inset_quad =
            // 0 inside the inset quad, and positive outside
            length(max(float2(0.0), corner_center_to_point)) +
            // 0 outside the inset quad, and negative inside
            min(0.0, max(corner_center_to_point.x, corner_center_to_point.y));

        return signed_distance_to_inset_quad - corner_radius;
    }
}

// vue-native fork: the coverage of a ROUNDED content mask at `position`, the
// fragment's [[position]] (its pixel centre, in the device pixels the mask is
// in). The clip distances already cut the mask's rectangle, so all-zero radii,
// the common case, return 1 before any SDF math: one uniform branch per
// fragment. A rounded mask is the antialiased SDF the polychrome sprite's own
// corners use. Mirrors gpui_windows' shaders.hlsl and gpui_wgpu's shaders.wgsl.
float content_mask_coverage(float2 position, Bounds_ScaledPixels mask,
                            Corners_ScaledPixels radii) {
  if (radii.top_left == 0. && radii.top_right == 0. &&
      radii.bottom_right == 0. && radii.bottom_left == 0.) {
    return 1.;
  }
  return saturate(0.5 - quad_sdf(position, mask, radii));
}

// vue-native fork: `content_mask_coverage` for a mask passed as flat varyings:
// `mask_bounds` is (origin.x, origin.y, width, height) and `mask_radii` is
// (top_left, top_right, bottom_right, bottom_left).
float flat_mask_coverage(float2 position, float4 mask_bounds, float4 mask_radii) {
  Bounds_ScaledPixels mask;
  mask.origin.x = mask_bounds.x;
  mask.origin.y = mask_bounds.y;
  mask.size.width = mask_bounds.z;
  mask.size.height = mask_bounds.w;
  Corners_ScaledPixels radii;
  radii.top_left = mask_radii.x;
  radii.top_right = mask_radii.y;
  radii.bottom_right = mask_radii.z;
  radii.bottom_left = mask_radii.w;
  return content_mask_coverage(position, mask, radii);
}

float4 distance_from_clip_rect(float2 unit_vertex, Bounds_ScaledPixels bounds,
                               Bounds_ScaledPixels clip_bounds) {
  float2 position =
      unit_vertex * float2(bounds.size.width, bounds.size.height) +
      float2(bounds.origin.x, bounds.origin.y);
  return float4(position.x - clip_bounds.origin.x,
                clip_bounds.origin.x + clip_bounds.size.width - position.x,
                position.y - clip_bounds.origin.y,
                clip_bounds.origin.y + clip_bounds.size.height - position.y);
}

float4 distance_from_clip_rect_transformed(float2 unit_vertex, Bounds_ScaledPixels bounds,
                               Bounds_ScaledPixels clip_bounds, TransformationMatrix transformation) {
  float2 position =
      unit_vertex * float2(bounds.size.width, bounds.size.height) +
      float2(bounds.origin.x, bounds.origin.y);
  float2 transformed_position = float2(0, 0);
  transformed_position[0] = position[0] * transformation.rotation_scale[0][0] + position[1] * transformation.rotation_scale[0][1];
  transformed_position[1] = position[0] * transformation.rotation_scale[1][0] + position[1] * transformation.rotation_scale[1][1];
  transformed_position[0] += transformation.translation[0];
  transformed_position[1] += transformation.translation[1];

  return float4(transformed_position.x - clip_bounds.origin.x,
                clip_bounds.origin.x + clip_bounds.size.width - transformed_position.x,
                transformed_position.y - clip_bounds.origin.y,
                clip_bounds.origin.y + clip_bounds.size.height - transformed_position.y);
}

float4 over(float4 below, float4 above) {
  float alpha = above.a + below.a * (1.0 - above.a);
  float3 premultiplied =
      above.rgb * above.a + below.rgb * below.a * (1.0 - above.a);
  return float4(premultiplied / (alpha == 0.0 ? 1.0 : alpha), alpha);
}

// Only the SOLID colour can still be resolved in the vertex shader: which pair
// of stops a fragment falls between is a per-fragment question once a gradient
// carries up to eight of them (`GRADIENT_STOPS` in `gpui/src/color.rs`; the
// literal is spelled out here so it cannot clash with cbindgen's own constant).
// `color0`/`color1` keep carrying the first and last stop.
GradientColor prepare_fill_color(uint tag, uint color_space, Hsla solid,
                                     Hsla color0, Hsla color1) {
  GradientColor out;
  out.solid = hsla_to_rgba(solid);
  out.color0 = hsla_to_rgba(color0);
  out.color1 = hsla_to_rgba(color1);
  if (color_space == 1) {
    out.color0 = srgb_to_oklab(out.color0);
    out.color1 = srgb_to_oklab(out.color1);
  }
  return out;
}

// Background::flags
#define BG_REPEATING      1u
#define BG_TILED          2u
#define BG_CORNER         4u
#define BG_NO_REPEAT_X    8u
#define BG_NO_REPEAT_Y    16u
#define BG_CENTER_PX        32u
#define BG_RADIUS_PX        64u
#define BG_TILE_W_FRACTION  128u
#define BG_TILE_H_FRACTION  256u
#define BG_TILE_X_FRACTION  512u
#define BG_TILE_Y_FRACTION  1024u
#define BG_STOP_PX_SHIFT    16u

float4 gradient_stop_color(Background background, uint i) {
  float4 c = hsla_to_rgba(background.colors[i].color);
  if (background.color_space == 1) {
    c = srgb_to_oklab(c);
  }
  return c;
}

// A stop flagged BG_STOP_PX carries an absolute length along the gradient line
// rather than a fraction of it — `white 1px, transparent 1px` has no percentage
// form and the line's length is only known here.
float gradient_stop_position(Background background, uint i, float line_length) {
  float p = background.colors[i].percentage;
  if (background.flags & (1u << (BG_STOP_PX_SHIFT + i))) {
    p = (line_length > 1e-6) ? (p / line_length) : 0.0;
  }
  return p;
}

// See the HLSL twin in `gpui_windows/src/shaders.hlsl` for the reasoning; a
// coincident pair of stops (a hard stop) falls out of the same arithmetic.
float4 gradient_ramp(Background background, float t, float line_length) {
  uint n = min(background.stop_count, 8u);
  if (n == 0u) {
    return float4(0.0);
  }

  float first = gradient_stop_position(background, 0u, line_length);
  float last = gradient_stop_position(background, n - 1u, line_length);
  if (background.flags & BG_REPEATING) {
    float span = last - first;
    if (span > 1e-6) {
      t = first + fract((t - first) / span) * span;
    }
  }

  float4 color = gradient_stop_color(background, 0u);
  float prev = first;
  for (uint i = 1u; i < 8u; i++) {
    if (i >= n) {
      break;
    }
    float p = gradient_stop_position(background, i, line_length);
    if (t > prev) {
      float seg = p - prev;
      float local = (seg > 1e-6) ? saturate((t - prev) / seg) : ((t >= p) ? 1.0 : 0.0);
      color = mix(color, gradient_stop_color(background, i), local);
    }
    prev = p;
  }

  if (background.color_space == 1) {
    color = oklab_to_srgb(color);
  }
  return color;
}

float2 gradient_center(Background background, float2 box_origin, float2 box_size) {
  float2 c = float2(background.params[0], background.params[1]);
  return box_origin + ((background.flags & BG_CENTER_PX) ? c : (c * box_size));
}

float4 gradient_dither(float4 color, float2 position) {
  // A stop list can resolve to EXACTLY transparent — that is how a gradient
  // split into layered bands masks the part another band owns — and dithering
  // that would spray faint noise over the whole box.
  if (color.a <= 0.0) {
    return color;
  }
  float2 seed = position * 0.6180339887;
  float r1 = fract(sin(dot(seed, float2(12.9898, 78.233))) * 43758.5453);
  float r2 = fract(sin(dot(seed, float2(39.3460, 11.135))) * 24634.6345);
  float tri = r1 + r2 - 1.0;
  color.rgb += tri * 2.0 / 255.0;
  color.a   += tri * 3.0 / 255.0;
  return color;
}

// CSS aims a `to <corner>` gradient's line at the actual corner, so its angle
// depends on the box's aspect ratio.
float gradient_corner_angle(uint corner, float2 size) {
  float a = (atan2(size.x, size.y)) * (180.0 / M_PI_F);
  if (corner == 0u) return a;
  if (corner == 1u) return 180.0 - a;
  if (corner == 2u) return 180.0 + a;
  return 360.0 - a;
}

float2 radial_radii(Background background, float2 center, float2 origin, float2 size) {
  if (background.params[2] >= 0.0) {
    float2 r = float2(background.params[2], background.params[3]);
    return (background.flags & BG_RADIUS_PX) ? r : (r * size);
  }
  uint keyword = (uint)(-background.params[2] + 0.5);
  bool ellipse = background.params[3] > 0.5;
  float2 to_start = abs(center - origin);
  float2 to_end = abs(origin + size - center);
  float2 nearest = min(to_start, to_end);
  float2 farthest = max(to_start, to_end);

  if (keyword == 1u) {
    return ellipse ? nearest : float2(min(nearest.x, nearest.y));
  }
  if (keyword == 2u) {
    return ellipse ? farthest : float2(max(farthest.x, farthest.y));
  }
  float2 corner = (keyword == 3u) ? nearest : farthest;
  if (!ellipse) {
    return float2(length(corner));
  }
  float2 sides = max(corner, float2(1e-6));
  float scale = sqrt(dot(corner / sides, corner / sides));
  return sides * scale;
}

float2x2 rotate2d(float angle) {
    float s = sin(angle);
    float c = cos(angle);
    return float2x2(c, -s, s, c);
}

float4 fill_color(Background background,
                      float2 position,
                      Bounds_ScaledPixels bounds,
                      float4 solid_color, float4 color0, float4 color1) {
  float4 color;

  // `background-size` / `background-position` / `background-repeat`: the
  // gradient is measured against ONE tile and the box is paved with copies of
  // it, so every tag below works in tile space without knowing about tiling.
  float2 box_origin = float2(bounds.origin.x, bounds.origin.y);
  float2 box_size = float2(bounds.size.width, bounds.size.height);
  if (background.flags & BG_TILED) {
    float2 tile_size = float2(background.tile[0], background.tile[1]);
    if (background.flags & BG_TILE_W_FRACTION) tile_size.x *= box_size.x;
    if (background.flags & BG_TILE_H_FRACTION) tile_size.y *= box_size.y;
    tile_size = max(tile_size, float2(1e-3));
    float2 tile_offset = float2(background.tile[2], background.tile[3]);
    // A PERCENTAGE background-position aligns the tile inside the space it
    // leaves — 50% centres it — rather than translating by a fraction of the
    // box.
    if (background.flags & BG_TILE_X_FRACTION) tile_offset.x *= (box_size.x - tile_size.x);
    if (background.flags & BG_TILE_Y_FRACTION) tile_offset.y *= (box_size.y - tile_size.y);
    float2 relative = position - box_origin - tile_offset;
    // `background-repeat: no-repeat` on an axis: outside the single tile there
    // is nothing, rather than another copy of it.
    if (((background.flags & BG_NO_REPEAT_X) &&
         (relative.x < 0.0 || relative.x >= tile_size.x)) ||
        ((background.flags & BG_NO_REPEAT_Y) &&
         (relative.y < 0.0 || relative.y >= tile_size.y))) {
      return float4(0.0);
    }
    position = box_origin + (relative - tile_size * floor(relative / tile_size));
    box_size = tile_size;
  }

  switch (background.tag) {
    case 0:
      color = solid_color;
      break;
    case 1: {
      float gradient_angle = background.gradient_angle_or_pattern_height;
      if (background.flags & BG_CORNER) {
        gradient_angle = gradient_corner_angle((uint)gradient_angle, box_size);
      }
      // CSS: 0deg points at the top and the angle turns clockwise. The
      // window's y axis points down, so "to top" is (0, -1).
      float rad = fmod(gradient_angle, 360.0) * (M_PI_F / 180.0);
      float2 direction = float2(sin(rad), -cos(rad));
      float line_length = abs(box_size.x * sin(rad)) + abs(box_size.y * cos(rad));
      float2 center = box_origin + box_size * 0.5;
      float t = dot(position - center, direction) / max(line_length, 1e-6) + 0.5;
      color = gradient_dither(gradient_ramp(background, t, line_length), position);
      break;
    }
    case 4: {
      float2 center = gradient_center(background, box_origin, box_size);
      float2 radii = max(radial_radii(background, center, box_origin, box_size), float2(1e-6));
      float t = length((position - center) / radii);
      color = gradient_dither(gradient_ramp(background, t, max(radii.x, radii.y)), position);
      break;
    }
    case 5: {
      float2 center = gradient_center(background, box_origin, box_size);
      float2 spoke = position - center;
      float angle = atan2(spoke.x, -spoke.y);
      float from = background.gradient_angle_or_pattern_height * (M_PI_F / 180.0);
      float t = fract((angle - from) / (2.0 * M_PI_F) + 1.0);
      color = gradient_dither(gradient_ramp(background, t, 1.0), position);
      break;
    }
    case 2: {
        float gradient_angle_or_pattern_height = background.gradient_angle_or_pattern_height;
        float pattern_width = (gradient_angle_or_pattern_height / 65535.0f) / 255.0f;
        float pattern_interval = fmod(gradient_angle_or_pattern_height, 65535.0f) / 255.0f;
        float pattern_height = pattern_width + pattern_interval;
        float stripe_angle = M_PI_F / 4.0;
        float pattern_period = pattern_height * sin(stripe_angle);
        float2x2 rotation = rotate2d(stripe_angle);
        float2 relative_position = position - float2(bounds.origin.x, bounds.origin.y);
        float2 rotated_point = rotation * relative_position;
        float pattern = fmod(rotated_point.x, pattern_period);
        float distance = min(pattern, pattern_period - pattern) - pattern_period * (pattern_width / pattern_height) /  2.0f;
        color = solid_color;
        color.a *= saturate(0.5 - distance);
        break;
    }
    case 3: {
        // checkerboard
        float size = background.gradient_angle_or_pattern_height;
        float2 relative_position = position - float2(bounds.origin.x, bounds.origin.y);

        float x_index = floor(relative_position.x / size);
        float y_index = floor(relative_position.y / size);
        float should_be_colored = fmod(x_index + y_index, 2.0);

        color = solid_color;
        color.a *= saturate(should_be_colored);
        break;
    }
  }

  return color;
}

// vue-native: CSS `backdrop-filter: blur()` and the `--shading` effects, a port
// of gpui_windows' shaders.hlsl / effects.hlsl and gpui_wgpu's
// shaders_{blur,backdrop,effects}.wgsl. Every entry point draws a four-vertex
// triangle strip.

constexpr sampler backdrop_clamp_sampler(address::clamp_to_edge,
                                         mag_filter::linear,
                                         min_filter::linear);

float2 strip_unit_vertex(uint vertex_id) {
  return float2(float(vertex_id & 1u), 0.5 * float(vertex_id & 2u));
}

// Keeps rgb <= a, so a premultiplied target never holds an out-of-gamut pixel.
float4 clamp_premultiplied(float4 color) {
  float alpha = saturate(color.a);
  return float4(clamp(color.rgb, float3(0.), float3(alpha)), alpha);
}

// The rect cut to its content mask's rectangle. Nothing here is transformed,
// so drawing the intersection rejects exactly the pixels that the other
// pipelines' clip distances reject.
Bounds_ScaledPixels clip_to_mask(Bounds_ScaledPixels bounds,
                                 Bounds_ScaledPixels mask) {
  float2 origin = max(float2(bounds.origin.x, bounds.origin.y),
                      float2(mask.origin.x, mask.origin.y));
  float2 corner = min(
      float2(bounds.origin.x + bounds.size.width,
             bounds.origin.y + bounds.size.height),
      float2(mask.origin.x + mask.size.width, mask.origin.y + mask.size.height));
  float2 size = max(corner - origin, float2(0.));
  Bounds_ScaledPixels clipped;
  clipped.origin.x = origin.x;
  clipped.origin.y = origin.y;
  clipped.size.width = size.x;
  clipped.size.height = size.y;
  return clipped;
}

struct BackdropPassVertexOutput {
  float4 position [[position]];
  float2 texture_position;
};

// The whole target, for the blur pyramid passes and the intermediate frame's
// copy; drawn with blending disabled.
vertex BackdropPassVertexOutput backdrop_pass_vertex(uint vertex_id [[vertex_id]]) {
  float2 unit_vertex = strip_unit_vertex(vertex_id);
  return BackdropPassVertexOutput{
      float4(unit_vertex * float2(2., -2.) + float2(-1., 1.), 0., 1.),
      unit_vertex};
}

fragment float4 blur_downsample_fragment(
    BackdropPassVertexOutput input [[stage_in]],
    texture2d<float> source [[texture(BackdropTextureIndex_Processed)]]) {
  float2 texel = 1. / float2(source.get_width(), source.get_height());
  float2 uv = input.texture_position;
  float4 color = source.sample(backdrop_clamp_sampler, uv) * 0.5;
  color += source.sample(backdrop_clamp_sampler, uv + texel * float2(-1., -1.)) * 0.125;
  color += source.sample(backdrop_clamp_sampler, uv + texel * float2(1., -1.)) * 0.125;
  color += source.sample(backdrop_clamp_sampler, uv + texel * float2(-1., 1.)) * 0.125;
  color += source.sample(backdrop_clamp_sampler, uv + texel * float2(1., 1.)) * 0.125;
  return clamp_premultiplied(color);
}

fragment float4 blur_upsample_fragment(
    BackdropPassVertexOutput input [[stage_in]],
    texture2d<float> source [[texture(BackdropTextureIndex_Processed)]]) {
  float2 texel = 1. / float2(source.get_width(), source.get_height());
  float2 uv = input.texture_position;
  float4 color = float4(0.);
  color += source.sample(backdrop_clamp_sampler, uv + texel * float2(-1., 0.)) * 0.125;
  color += source.sample(backdrop_clamp_sampler, uv + texel * float2(1., 0.)) * 0.125;
  color += source.sample(backdrop_clamp_sampler, uv + texel * float2(0., -1.)) * 0.125;
  color += source.sample(backdrop_clamp_sampler, uv + texel * float2(0., 1.)) * 0.125;
  color += source.sample(backdrop_clamp_sampler, uv + texel * float2(-1., -1.)) * 0.125;
  color += source.sample(backdrop_clamp_sampler, uv + texel * float2(1., -1.)) * 0.125;
  color += source.sample(backdrop_clamp_sampler, uv + texel * float2(-1., 1.)) * 0.125;
  color += source.sample(backdrop_clamp_sampler, uv + texel * float2(1., 1.)) * 0.125;
  return clamp_premultiplied(color);
}

fragment float4 frame_copy_fragment(
    BackdropPassVertexOutput input [[stage_in]],
    texture2d<float> source [[texture(BackdropTextureIndex_Processed)]]) {
  return source.read(uint2(input.position.xy));
}

struct BlurRectVertexOutput {
  float4 position [[position]];
  uint blur_rect_id [[flat]];
};

vertex BlurRectVertexOutput blur_rect_vertex(
    uint vertex_id [[vertex_id]], uint blur_rect_id [[instance_id]],
    constant BackdropBlurRect *blur_rects [[buffer(BackdropInputIndex_Instances)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(BackdropInputIndex_ViewportSize)]]) {
  BackdropBlurRect blur_rect = blur_rects[blur_rect_id];
  float4 position = to_device_position(
      strip_unit_vertex(vertex_id),
      clip_to_mask(blur_rect.bounds, blur_rect.content_mask.bounds),
      viewport_size);
  return BlurRectVertexOutput{position, blur_rect_id};
}

// `above` arrives with straight alpha, `below` is already premultiplied.
float4 over_straight_on_premultiplied(float4 below, float4 above) {
  float4 premultiplied = float4(above.rgb * above.a, above.a);
  return clamp_premultiplied(premultiplied + below * (1. - above.a));
}

// Drawn with blending disabled: outside the rounded rect the untouched
// snapshot is written back, which keeps the antialiased edge correct.
fragment float4 blur_rect_fragment(
    BlurRectVertexOutput input [[stage_in]],
    constant BackdropBlurRect *blur_rects [[buffer(BackdropInputIndex_Instances)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(BackdropInputIndex_ViewportSize)]],
    texture2d<float> processed [[texture(BackdropTextureIndex_Processed)]],
    texture2d<float> original [[texture(BackdropTextureIndex_Original)]]) {
  BackdropBlurRect blur_rect = blur_rects[input.blur_rect_id];
  float2 position = input.position.xy;
  float2 texture_position =
      position / float2(viewport_size->width, viewport_size->height);

  float sdf = quad_sdf(position, blur_rect.bounds, blur_rect.corner_radii);
  float shape_alpha = saturate(0.5 - sdf);

  float4 untouched =
      original.sample(backdrop_clamp_sampler, texture_position, level(0.));
  float4 blurred =
      processed.sample(backdrop_clamp_sampler, texture_position, level(0.));
  float4 tint = hsla_to_rgba(blur_rect.tint);
  float4 tinted =
      tint.a > 0. ? over_straight_on_premultiplied(blurred, tint) : blurred;
  float coverage = shape_alpha * saturate(blur_rect.opacity) *
                   content_mask_coverage(position, blur_rect.content_mask.bounds,
                                         blur_rect.content_mask.corner_radii);
  return clamp_premultiplied(mix(untouched, tinted, float4(coverage)));
}

// The generated fragment stages (crates/sk-effects/generated/*.metal) read
// the quad's index as `[[flat]] uint [[user(TEXCOORD)]]`.
struct EffectVertexOutput {
  float4 position [[position]];
  uint effect_quad_id [[user(TEXCOORD), flat]];
};

// The VERTEX half of every `--shading` effect. One instance draws one effect
// quad, already dilated by its bleed, cut to its content mask's rectangle
// because the generated fragment stage cannot discard.
vertex EffectVertexOutput effect_vertex(
    uint vertex_id [[vertex_id]], uint effect_quad_id [[instance_id]],
    constant EffectQuad *effect_quads [[buffer(BackdropInputIndex_Instances)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(BackdropInputIndex_ViewportSize)]]) {
  EffectQuad quad = effect_quads[effect_quad_id];
  // `gpui::EFFECT_FLAG_HISTORY_PASS`: a history step covers its whole
  // target, the quad's own texels.
  if ((quad.flags & 8u) != 0u) {
    float2 unit_vertex = strip_unit_vertex(vertex_id);
    return EffectVertexOutput{
        float4(unit_vertex * float2(2., -2.) + float2(-1., 1.), 0., 1.),
        effect_quad_id};
  }
  float4 position =
      to_device_position(strip_unit_vertex(vertex_id),
                         clip_to_mask(quad.bounds, quad.content_mask.bounds),
                         viewport_size);
  return EffectVertexOutput{position, effect_quad_id};
}

float punch_distance(float2 position, Bounds_ScaledPixels bounds,
                     Corners_ScaledPixels radii) {
  float2 half_size = float2(bounds.size.width, bounds.size.height) * 0.5;
  float2 p = position - (float2(bounds.origin.x, bounds.origin.y) + half_size);
  float left = p.y < 0. ? radii.top_left : radii.bottom_left;
  float right = p.y < 0. ? radii.top_right : radii.bottom_right;
  float radius = min(p.x < 0. ? left : right, min(half_size.x, half_size.y));
  float2 q = abs(p) - half_size + radius;
  return length(max(q, float2(0.))) + min(max(q.x, q.y), 0.) - radius;
}

// `gpui::effect_id::PUNCH`: only a coverage, which the pipeline's
// destination-out blend erases from the target.
fragment float4 punch_fragment(
    EffectVertexOutput input [[stage_in]],
    constant EffectQuad *effect_quads [[buffer(BackdropInputIndex_Instances)]]) {
  EffectQuad quad = effect_quads[input.effect_quad_id];
  float inside = saturate(
      0.5 - punch_distance(input.position.xy, quad.bounds, quad.corner_radii));
  float unmasked = saturate(0.5 - punch_distance(input.position.xy,
                                                 quad.content_mask.bounds,
                                                 quad.content_mask.corner_radii));
  return float4(0., 0., 0., inside * unmasked * quad.opacity);
}

// `gpui::effect_id::CONTENT`: an effect layer's content as it is, at the
// quad's opacity, cut by its content mask's rounded corners.
fragment float4 content_fragment(
    EffectVertexOutput input [[stage_in]],
    constant EffectQuad *effect_quads [[buffer(BackdropInputIndex_Instances)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(BackdropInputIndex_ViewportSize)]],
    texture2d<float> content [[texture(BackdropTextureIndex_Content)]]) {
  EffectQuad quad = effect_quads[input.effect_quad_id];
  float unmasked = saturate(0.5 - punch_distance(input.position.xy,
                                                 quad.content_mask.bounds,
                                                 quad.content_mask.corner_radii));
  float2 uv = input.position.xy /
              float2(viewport_size->width, viewport_size->height);
  return content.sample(backdrop_clamp_sampler, uv, level(0.)) *
         (quad.opacity * unmasked);
}

// vue-native: the composite of one isolated layer (`gpui::VnLayer`), a port of
// `vn_layer_*` in gpui_windows' shaders.hlsl: the group drawn into an
// offscreen target the size of the frame, put back over its parent through a
// clip shape, a color matrix and a text background stack.

struct VnLayerVertexOutput {
  float4 position [[position]];
  uint layer_id [[flat]];
};

vertex VnLayerVertexOutput vn_layer_vertex(
    uint vertex_id [[vertex_id]], uint layer_id [[instance_id]],
    constant VnLayer *layers [[buffer(VnLayerInputIndex_Layers)]],
    constant Size_DevicePixels *viewport_size
    [[buffer(VnLayerInputIndex_ViewportSize)]]) {
  // `bounds` is already cut to the content mask's rectangle.
  float4 position = to_device_position(strip_unit_vertex(vertex_id),
                                       layers[layer_id].bounds, viewport_size);
  return VnLayerVertexOutput{position, layer_id};
}

// Winding number of the edges crossing the ray to the right (half-open in y,
// so a vertex on the ray counts once), blended across the nearest edge. The
// CPU twin is `gpui::VnClipGeometry::coverage`.
float vn_clip_coverage(float2 p, constant float4 *segments, uint start,
                       uint count, uint fill_rule) {
  int winding = 0;
  float nearest = 3.0e38;
  for (uint i = 0; i < count; i++) {
    float4 s = segments[start + i];
    float2 a = s.xy;
    float2 b = s.zw;
    if ((a.y <= p.y) != (b.y <= p.y)) {
      float x = a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x);
      if (p.x < x) {
        winding += b.y > a.y ? 1 : -1;
      }
    }
    float2 ab = b - a;
    float2 ap = p - a;
    float len2 = dot(ab, ab);
    float t = len2 > 0. ? saturate(dot(ap, ab) / len2) : 0.;
    float2 q = ap - t * ab;
    nearest = min(nearest, dot(q, q));
  }
  bool inside = fill_rule == 1u ? (winding & 1) != 0 : winding != 0;
  float d = sqrt(nearest);
  return saturate(0.5 + (inside ? d : -d));
}

// The exact sRGB curves, as DirectX and wgpu apply a color matrix in
// linearRGB.
float3 vn_srgb_to_linear(float3 c) {
  return select(pow((c + 0.055) / 1.055, float3(2.4)), c / 12.92, c < 0.04045);
}

float3 vn_linear_to_srgb(float3 c) {
  return select(1.055 * pow(c, float3(1. / 2.4)) - 0.055, c * 12.92, c < 0.0031308);
}

fragment float4 vn_layer_fragment(
    VnLayerVertexOutput input [[stage_in]],
    constant VnLayer *layers [[buffer(VnLayerInputIndex_Layers)]],
    constant float4 *segments [[buffer(VnLayerInputIndex_Segments)]],
    constant VnTextBackground *text_backgrounds
    [[buffer(VnLayerInputIndex_TextBackgrounds)]],
    texture2d<float> source [[texture(VnLayerInputIndex_Texture)]],
    texture2d<float> mask_source [[texture(VnLayerInputIndex_MaskTexture)]]) {
  VnLayer layer = layers[input.layer_id];
  float2 p = input.position.xy;
  if ((layer.flags & 1u) != 0u) {
    // A snapshot: the pixel maps into the capture's device space, where the
    // snapshot covers `source_rect`.
    thread const float *t = layer.source_transform;
    float2 q = float2(t[0] * p.x + t[1] * p.y + t[2], t[4] * p.x + t[5] * p.y + t[6]);
    float4 rect = float4(layer.source_rect[0], layer.source_rect[1],
                         layer.source_rect[2], layer.source_rect[3]);
    float2 local = q - rect.xy;
    float edge = min(min(local.x, local.y), min(rect.z - local.x, rect.w - local.y));
    float snapshot_coverage = saturate(edge + 0.5) *
        content_mask_coverage(p, layer.content_mask.bounds, layer.content_mask.corner_radii);
    constexpr sampler clamped(mag_filter::linear, min_filter::linear,
                              address::clamp_to_edge);
    float4 sampled = source.sample(clamped, local / rect.zw);
    return sampled * (snapshot_coverage * layer.opacity);
  }
  float coverage = vn_clip_coverage(p, segments, layer.segment_start,
                                    layer.segment_count, layer.fill_rule & 1u) *
                   content_mask_coverage(p, layer.content_mask.bounds,
                                         layer.content_mask.corner_radii);
  float4 color = source.read(uint2(p));
  uint matrix_mode = layer.pad[0];
  uint text_start = layer.pad[1];
  uint text_count = layer.pad[2];
  if (text_count != 0u) {
    float4 fill = float4(0.);
    for (uint i = 0u; i < text_count; i++) {
      VnTextBackground text = text_backgrounds[text_start + i];
      float2 local = float2(
          p.x * text.inverse.rotation_scale[0][0] + p.y * text.inverse.rotation_scale[0][1],
          p.x * text.inverse.rotation_scale[1][0] + p.y * text.inverse.rotation_scale[1][1]);
      local += float2(text.inverse.translation[0], text.inverse.translation[1]);
      local /= text.scale_factor;
      Background background = text.background;
      GradientColor prepared = prepare_fill_color(
          background.tag, background.color_space, background.solid,
          background.colors[0].color, background.colors[7].color);
      float4 layer_color = fill_color(background, local, text.bounds, prepared.solid,
                                      prepared.color0, prepared.color1);
      float4 premultiplied = float4(layer_color.rgb * layer_color.a, layer_color.a);
      fill = premultiplied + fill * (1. - premultiplied.a);
    }
    // The whole stack is masked once, which keeps the glyphs' antialiasing.
    color = fill * color.a;
  }
  if (matrix_mode != 0u) {
    float4 straight = color.a > 0. ? float4(color.rgb / color.a, color.a) : float4(0.);
    if (matrix_mode == 1u) {
      straight.rgb = vn_srgb_to_linear(straight.rgb);
    }
    thread const float *m = layer.color_matrix;
    float4 filtered = saturate(
        float4(dot(float4(m[0], m[1], m[2], m[3]), straight),
               dot(float4(m[4], m[5], m[6], m[7]), straight),
               dot(float4(m[8], m[9], m[10], m[11]), straight),
               dot(float4(m[12], m[13], m[14], m[15]), straight)) +
        float4(m[16], m[17], m[18], m[19]));
    if (matrix_mode == 1u) {
      filtered.rgb = vn_linear_to_srgb(filtered.rgb);
    }
    color = float4(filtered.rgb * filtered.a, filtered.a);
  }
  uint mask_mode = layer.fill_rule >> 1;
  if (mask_mode != 0u) {
    float4 mask_texel = mask_source.read(uint2(p));
    // Premultiplied, so this is the luminance times the alpha.
    float luminance = dot(mask_texel.rgb, float3(0.2126, 0.7152, 0.0722));
    color *= mask_mode == 2u ? luminance : mask_texel.a;
  }
  return color * (coverage * layer.opacity);
}
