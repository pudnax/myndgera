#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_samplerless_texture_functions : require

layout(scalar, push_constant) uniform PushConstant {
    uvec2 source_dim;
    uvec2 target_dim;
    uint source_img;
    uint target_img_sampled;
    uint target_img_storage;
    float width;
    float strength;
    uint num_passes;
    bool is_final_pass;
}
pc;

layout(set = 0, binding = 0) uniform sampler gsamplers[];
layout(set = 0, binding = 1) uniform texture2D gtextures[];
layout(set = 1, binding = 0) writeonly coherent
    restrict uniform image2D gstorage[];

#include <textures.glsl>

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

vec4 tex_lod(uint tex_id, vec2 uv) {
    return textureLod(
        nonuniformEXT(sampler2D(gtextures[tex_id], gsamplers[LINEAR_SAMPL])),
        uv, 0);
}

void main() {
    const ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    if (any(greaterThanEqual(gid, pc.target_dim))) {
        return;
    }

    vec2 texel = 1.0 / pc.source_dim;

    // Center of written pixel
    const vec2 uv = (vec2(gid) + 0.5) / pc.target_dim;

    vec4 rgba =
        texelFetch(nonuniformEXT(sampler2D(gtextures[pc.target_img_sampled],
                                           gsamplers[LINEAR_SAMPL])),
                   gid, int(0));

    uint source_img = pc.source_img;
    vec2 width = texel * pc.width;
    vec4 blur_sum = vec4(0);
    blur_sum += tex_lod(source_img, uv + vec2(-1, -1) * width) * 1.0 / 16.0;
    blur_sum += tex_lod(source_img, uv + vec2(0, -1) * width) * 2.0 / 16.0;
    blur_sum += tex_lod(source_img, uv + vec2(1, -1) * width) * 1.0 / 16.0;
    blur_sum += tex_lod(source_img, uv + vec2(-1, 0) * width) * 2.0 / 16.0;
    blur_sum += tex_lod(source_img, uv + vec2(0, 0) * width) * 4.0 / 16.0;
    blur_sum += tex_lod(source_img, uv + vec2(1, 0) * width) * 2.0 / 16.0;
    blur_sum += tex_lod(source_img, uv + vec2(-1, 1) * width) * 1.0 / 16.0;
    blur_sum += tex_lod(source_img, uv + vec2(0, 1) * width) * 2.0 / 16.0;
    blur_sum += tex_lod(source_img, uv + vec2(1, 1) * width) * 1.0 / 16.0;

    if (pc.is_final_pass) {
        // Conserve energy
        rgba = mix(rgba, blur_sum / pc.num_passes, pc.strength);
    } else {
        // Accumulate
        rgba += blur_sum;
    }

    imageStore(gstorage[pc.target_img_storage], gid, rgba);
}
