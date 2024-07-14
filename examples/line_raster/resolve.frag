#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

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

layout(std430, push_constant) uniform PushConstant {
    uint idx;
    uint red_img;
    uint green_img;
    uint blue_img;
}
pc;

void main() {
    vec2 dims = vec2(imageSize(gstorage_textures[pc.red_img]));
    vec2 uv = (in_uv + -0.5) * vec2(dims.x / dims.y, 1);

    vec3 col = vec3(.1);

    vec2 u = in_uv + vec2(0., -1.);
    u.y *= -1.;
    ivec2 pix = ivec2(u * dims);
    col.r += imageLoad(gstorage_textures[pc.red_img], pix).x / 255.;
    col.g += imageLoad(gstorage_textures[pc.green_img], pix).x / 255.;
    col.b += imageLoad(gstorage_textures[pc.blue_img], pix).x / 255.;

    col = pow(col, vec3(0.4545));

    out_color = vec4(col, 1.0);
}
