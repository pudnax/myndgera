#version 460

#include "shared.glsl"
#include <extensions.glsl>

layout(set = 0, binding = 0, r32ui) writeonly uniform uimage2D gtextures[];
layout(set = 0, binding = 0, r16f) writeonly uniform image2D gtexturesf[];

layout(scalar, push_constant) uniform PushConstant {
    uint red_img;
    uint green_img;
    uint blue_img;
    uint depth_img;
}
pc;

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main() {
    vec2 dims = imageSize(gtextures[pc.red_img]);
    vec2 pix = gl_GlobalInvocationID.xy;
    if (pix.x >= dims.x || pix.y >= dims.y) {
        return;
    }

    imageStore(gtextures[pc.red_img], ivec2(pix), uvec4(0));
    imageStore(gtextures[pc.green_img], ivec2(pix), uvec4(0));
    imageStore(gtextures[pc.blue_img], ivec2(pix), uvec4(0));
    imageStore(gtexturesf[pc.depth_img], ivec2(pix), uvec4(0));
}
