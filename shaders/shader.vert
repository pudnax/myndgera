#version 460
#extension GL_EXT_buffer_reference : require

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

layout(location = 0) out vec2 out_uv;

void main() {
    out_uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(out_uv * 2.0f + -1.0f, 0.0, 1.0);
}