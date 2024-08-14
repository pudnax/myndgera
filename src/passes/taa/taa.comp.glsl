#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_samplerless_texture_functions : require

#include <textures.glsl>

layout(scalar, push_constant) uniform PushConstant {
    uint src_img;
    uint dst_img;
    uint history_img;
    uint motion_img;
}
pc;

layout(set = 0, binding = 0) uniform sampler gsamplers[];
layout(set = 0, binding = 1) uniform texture2D gtextures[];
layout(set = 1, binding = 0) coherent restrict uniform image2D gstorage[];

// Controls how much to blend between the current and past samples
// Lower numbers = less of the current col and more of the past col = more
// smoothing Values chosen empirically
const float DEFAULT_HISTORY_BLEND_RATE =
    0.1; // Default blend rate to use when no confidence in history
const float MIN_HISTORY_BLEND_RATE =
    0.015; // Minimum blend rate allowed, to ensure at least some of the current
           // col is used

// TAA is ideally applied after tonemapping, but before post processing
// Post processing wants to go before tonemapping, which conflicts
// Solution: Put TAA before tonemapping, tonemap TAA input, apply TAA,
// invert-tonemap TAA output
// https://advances.realtimerendering.com/s2014/index.html#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING,
// slide 20
// https://gpuopen.com/learn/optimized-reversible-tonemapper-for-resolve
float max3(vec3 x) { return max(x.r, max(x.g, x.b)); }
vec3 tonemap(vec3 color) { return color * 1. / (max3(color) + 1.0); }
vec3 reverse_tonemap(vec3 color) { return color * 1. / (1.0 - max3(color)); }

// The following 3 functions are from Playdead (MIT-licensed)
// https://github.com/playdeadgames/temporal/blob/master/Assets/Shaders/TemporalReprojection.shader
vec3 RGB_to_YCoCg(vec3 rgb) {
    float y = (rgb.r / 4.0) + (rgb.g / 2.0) + (rgb.b / 4.0);
    float co = (rgb.r / 2.0) - (rgb.b / 2.0);
    float cg = (-rgb.r / 4.0) + (rgb.g / 2.0) - (rgb.b / 4.0);
    return vec3(y, co, cg);
}

vec3 YCoCg_to_RGB(vec3 ycocg) {
    float r = ycocg.x + ycocg.y - ycocg.z;
    float g = ycocg.x + ycocg.z;
    float b = ycocg.x - ycocg.y - ycocg.z;
    return clamp(vec3(r, g, b), 0., 1.);
}

vec3 clip_towards_aabb_center(vec3 history_color, vec3 current_color,
                              vec3 aabb_min, vec3 aabb_max) {
    vec3 p_clip = 0.5 * (aabb_max + aabb_min);
    vec3 e_clip = 0.5 * (aabb_max - aabb_min) + 0.00000001;
    vec3 v_clip = history_color - p_clip;
    vec3 v_unit = v_clip / e_clip;
    vec3 a_unit = abs(v_unit);
    float ma_unit = max3(a_unit);
    if (ma_unit > 1.0) {
        return p_clip + (v_clip / ma_unit);
    } else {
        return history_color;
    }
}

vec3 sample_history(int u, int v) {
    return imageLoad(gstorage[pc.history_img], ivec2(u, v)).rgb;
}

vec3 sample_view_target(ivec2 gid) {
    vec3 col = imageLoad(gstorage[pc.src_img], gid).rgb;
    col = tonemap(col);
    return RGB_to_YCoCg(col);
}

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main() {
    const ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    uvec2 dims = imageSize(gstorage[pc.src_img]);
    if (any(greaterThanEqual(gid, dims))) { return; }
    vec2 uv = (vec2(gid) + 0.5) / vec2(dims);

    vec4 original_color = imageLoad(gstorage[pc.src_img], gid);
    vec3 current_color = original_color.rgb;
    current_color = tonemap(current_color);

    vec3 velocity = imageLoad(gstorage[pc.motion_img], gid).rgb;
    vec2 motion_vector = velocity.xy;

    // Reproject to find the equivalent sample from the past
    // Uses 5-sample Catmull-Rom filtering (reduces blurriness)
    // Catmull-Rom filtering:
    // https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
    // Ignoring corners:
    // https://www.activision.com/cdn/research/Dynamic_Temporal_Antialiasing_and_Upsampling_in_Call_of_Duty_v4.pdf#page=68
    // Technically we should renormalize the weights since we're skipping the
    // corners, but it's basically the same result
    vec2 history_uv = uv - motion_vector;
    vec2 sample_position = history_uv * dims;
    vec2 texel_center = floor(sample_position - 0.5) + 0.5;
    vec2 f = sample_position - texel_center;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    ivec2 texel_position_0 = ivec2(texel_center - 1.0);
    ivec2 texel_position_3 = ivec2(texel_center + 2.0);
    ivec2 texel_position_12 = ivec2(texel_center + (w2 / w12));
    vec3 history_color =
        sample_history(texel_position_12.x, texel_position_0.y) * w12.x * w0.y;
    history_color +=
        sample_history(texel_position_0.x, texel_position_12.y) * w0.x * w12.y;
    history_color += sample_history(texel_position_12.x, texel_position_12.y) *
                     w12.x * w12.y;
    history_color +=
        sample_history(texel_position_3.x, texel_position_12.y) * w3.x * w12.y;
    history_color +=
        sample_history(texel_position_12.x, texel_position_3.y) * w12.x * w3.y;

    // Constrain past sample with 3x3 YCoCg variance clipping (reduces ghosting)
    // YCoCg:
    // https://advances.realtimerendering.com/s2014/index.html#_HIGH-QUALITY_TEMPORAL_SUPERSAMPLING,
    // slide 33 Variance clipping:
    // https://developer.download.nvidia.com/gameworks/events/GDC2016/msalvi_temporal_supersampling.pdf
    vec3 s_tl = sample_view_target(gid + ivec2(-1, 1.));
    vec3 s_tm = sample_view_target(gid + ivec2(0.0, 1.));
    vec3 s_tr = sample_view_target(gid + ivec2(1., 1.));
    vec3 s_ml = sample_view_target(gid + ivec2(-1., 0.0));
    vec3 s_mm = RGB_to_YCoCg(current_color);
    vec3 s_mr = sample_view_target(gid + ivec2(1, 0.0));
    vec3 s_bl = sample_view_target(gid + ivec2(-1, -1));
    vec3 s_bm = sample_view_target(gid + ivec2(0.0, -1));
    vec3 s_br = sample_view_target(gid + ivec2(1, -1));
    vec3 moment_1 =
        s_tl + s_tm + s_tr + s_ml + s_mm + s_mr + s_bl + s_bm + s_br;
    vec3 moment_2 = (s_tl * s_tl) + (s_tm * s_tm) + (s_tr * s_tr) +
                    (s_ml * s_ml) + (s_mm * s_mm) + (s_mr * s_mr) +
                    (s_bl * s_bl) + (s_bm * s_bm) + (s_br * s_br);
    vec3 mean = moment_1 / 9.0;
    vec3 variance = (moment_2 / 9.0) - (mean * mean);
    vec3 std_deviation = sqrt(max(variance, vec3(0.0)));
    history_color = RGB_to_YCoCg(history_color);
    history_color = clip_towards_aabb_center(
        history_color, s_mm, mean - std_deviation, mean + std_deviation);
    history_color = YCoCg_to_RGB(history_color);

    float history_confidence = 0.12;

    // Blend current and past sample
    // Use more of the history if we're confident in it (reduces noise when
    // there is no motion) https://hhoppe.com/supersample.pdf, section 4.1
    float current_color_factor =
        clamp(1.0 / history_confidence, MIN_HISTORY_BLEND_RATE,
              DEFAULT_HISTORY_BLEND_RATE);

    // Reject history when motion vectors point off screen
    if (any(bvec2(clamp(history_uv, 0., 1.) != history_uv))) {
        current_color_factor = 1.0;
        history_confidence = 1.0;
    }

    current_color = mix(history_color, current_color, current_color_factor);

    imageStore(gstorage[pc.history_img], gid,
               vec4(current_color, history_confidence));
    current_color = reverse_tonemap(current_color);
    imageStore(gstorage[pc.dst_img], gid,
               vec4(current_color, original_color.a));
}
