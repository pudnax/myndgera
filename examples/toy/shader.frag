#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 final_color;

layout(set = 0, binding = 0) uniform sampler gsamplers[];
layout(set = 0, binding = 1) uniform texture2D gtextures[];
#include "textures.glsl"
vec4 Tex(uint tex_id, uint smp_id) { return Tex(tex_id, smp_id, in_uv); }
vec4 TexLinear(uint tex_id) { return Tex(tex_id, LINEAR_SAMPL, in_uv); }
vec4 TexNear(uint tex_id) { return Tex(tex_id, NEAREST_SAMPL, in_uv); }

layout(scalar, push_constant) uniform PushConstant {
    vec2 resolution;
    vec3 pos;
    vec2 mouse;
    bool mouse_pressed;
    float time;
    float dt;
    uint frame;
}
pc;

void main() {
    vec2 uv = (in_uv + -0.5) * vec2(pc.resolution.x / pc.resolution.y, 1.);
    vec2 m = pc.mouse * vec2(pc.resolution.x / pc.resolution.y, 1.);

    vec3 col = vec3(in_uv, 0.);
    uv.x += 0.1 * cos(pc.time * 5.);
    float d = length(uv - m) - 0.25;
    col = mix(col, vec3(0.25), step(d, 0.));
    col *= TexLinear(BLUE_TEX).rgb;

    final_color = vec4(col, 1.);
}
