#version 460
#extension GL_EXT_buffer_reference : require

#include "utils.glsl"

layout(location = 0) in vec2 in_uv;
layout(location = 1) in vec3 in_colot;

layout(location = 0) out vec4 out_color;

void main() { out_color = vec4(in_colot, 1.0); }
