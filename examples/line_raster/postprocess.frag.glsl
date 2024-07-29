#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_shader_image_load_formatted : require

#include "shared.glsl"
#include <camera.glsl>
#include <prelude.glsl>

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler gsamplers[];
layout(set = 0, binding = 1) uniform texture2D gtextures[];
vec4 Tex(uint tex_id, uint smp_id, vec2 uv) {
    return texture(
        nonuniformEXT(sampler2D(gtextures[tex_id], gsamplers[smp_id])), uv);
}
vec4 Tex(uint tex_id, uint smp_id) { return Tex(tex_id, smp_id, in_uv); }
vec4 TexLinear(uint tex_id, vec2 uv) { return Tex(tex_id, LINEAR_SAMPL, uv); }
vec4 TexLinear(uint tex_id) { return Tex(tex_id, LINEAR_SAMPL, in_uv); }
vec4 TexNear(uint tex_id, vec2 uv) { return Tex(tex_id, NEAREST_SAMPL, uv); }
vec4 TexNear(uint tex_id) { return Tex(tex_id, NEAREST_SAMPL, in_uv); }

layout(set = 1, binding = 0) uniform readonly image2D gstorage[];

layout(scalar, push_constant) uniform PushConstant {
    uint idx;
    uint hdr_sampled;
    uint hdr_storage;
}
pc;

vec3 linear_to_nonlinear_srgb(vec3 linear_color) {
    bvec3 cutoff = lessThan(linear_color, vec3(0.0031308));
    vec3 higher =
        vec3(1.055) * pow(linear_color, vec3(1.0 / 2.4)) - vec3(0.055);
    vec3 lower = linear_color * vec3(12.92);

    return mix(higher, lower, cutoff);
}

vec3 aces_tonemap(vec3 color) {
    mat3 m1 = mat3(0.59719, 0.07600, 0.02840, 0.35458, 0.90834, 0.13383,
                   0.04823, 0.01566, 0.83777);
    mat3 m2 = mat3(1.60475, -0.10208, -0.00327, -0.53108, 1.10813, -0.07276,
                   -0.07367, -0.00605, 1.07602);
    vec3 v = m1 * color;
    vec3 a = v * (v + 0.0245786) - 0.000090537;
    vec3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return pow(clamp(m2 * (a / b), 0.0, 1.0), vec3(1.0 / 2.2));
}

vec3 PBRNeutralToneMapping(vec3 color) {
    const float startCompression = 0.8 - 0.04;
    const float desaturation = 0.15;

    float x = min(color.r, min(color.g, color.b));
    float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
    color -= offset;

    float peak = max(color.r, max(color.g, color.b));
    if (peak < startCompression) return color;

    const float d = 1. - startCompression;
    float newPeak = 1. - d * d / (peak + d - startCompression);
    color *= newPeak / peak;

    float g = 1. - 1. / (desaturation * (peak - newPeak) + 1.);
    return mix(color, newPeak * vec3(1, 1, 1), g);
}

void main() {
    vec2 dims = vec2(imageSize(gstorage[pc.hdr_storage]));
    vec2 uv = (in_uv * 2. - 1.) * vec2(dims.x / dims.y, 1);
    vec3 col = vec3(0.);

    // col += TexLinear(pc.hdr_sampled, vec2(in_uv.x, 1. - in_uv.y)).rgb;
    ivec2 pix = ivec2(vec2(in_uv.x, 1. - in_uv.y) * dims);
    col += imageLoad(gstorage[pc.hdr_storage], pix).rgb;

    col = PBRNeutralToneMapping(col);
    col = linear_to_nonlinear_srgb(col);

    out_color = vec4(col, 1.0);
}
