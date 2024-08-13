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
}
pc;

layout(set = 0, binding = 0) uniform sampler gsamplers[];
layout(set = 0, binding = 1) uniform texture2D gtextures[];
layout(set = 1, binding = 0) coherent restrict uniform image2D gstorage[];

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main() {
    const ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    uvec2 dims = imageSize(gstorage[pc.src_img]);

    vec4 pix = imageLoad(gstorage[pc.src_img], gid);
    imageStore(gstorage[pc.dst_img], gid, pix);
}
