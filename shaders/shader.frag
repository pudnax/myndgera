#version 460
#extension GL_EXT_buffer_reference : require

// In the beginning, colours never existed. There's nothing that was done before
// you...

#include <prelude.glsl>

layout(set = 0, binding = 0) uniform sampler gsamplers[];
layout(set = 0, binding = 1) uniform texture2D gtextures[];

layout(std430, push_constant) uniform PushConstant {
    vec3 pos;
    float time;
    vec2 resolution;
    vec2 mouse;
    bool mouse_pressed;
    uint frame;
    float time_delta;
    float record_time;
}
pc;

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

void main() {
    vec2 uv = (in_uv + -0.5) * vec2(pc.resolution.x / pc.resolution.y, 1);

    uv *= rotate(pc.time);

    float d = length(uv) - 0.25;

    vec3 col = vec3(uv, 1.);
    col *= d;

    vec4 tex = texture(sampler2D(gtextures[DITHER_TEX], gsamplers[0]), in_uv);
    col *= tex.rgb;

    out_color = vec4(col, 1.0);
}
