#version 460
#extension GL_EXT_buffer_reference : require

#include "math.glsl"
#include "utils.glsl"

layout(std430, buffer_reference,
       buffer_reference_align = 8) readonly buffer Transform {
    mat2 transform;
};

layout(push_constant) uniform _ { Transform tr_ptr; }
pc;

layout(location = 0) out vec2 out_uv;
layout(location = 1) out vec3 out_color;

const vec3 colors[3] = vec3[3](vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f),
                               vec3(0.0f, 0.0f, 1.0f));

void main() {
    const vec2 positions[3] =
        vec2[3](vec2(0.25, 0.25), vec2(-0.25, 0.25), vec2(0., -0.25));

    mat2 trans = pc.tr_ptr.transform;
    gl_Position = vec4(trans * positions[gl_VertexIndex], 0., 1.0);
    out_color = colors[gl_VertexIndex];
}
