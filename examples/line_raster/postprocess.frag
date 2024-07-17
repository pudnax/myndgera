#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

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

layout(set = 1, binding = 0,
       r32ui) uniform readonly uimage2D gstorage_textures[];

layout(std430, push_constant) uniform PushConstant { uint idx; }
pc;

void main() {
    vec2 uv = (in_uv * 2. - 1.); // * vec2(dims.x / dims.y, 1);

    vec3 col = vec3(0.);

    col = Tex(pc.idx, 0).rgb;

    col = pow(col, vec3(0.4545));

    out_color = vec4(col, 1.0);
}
