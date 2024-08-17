#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_samplerless_texture_functions : require

#define LOCAL_SIZE_X 8
#define LOCAL_SIZE_Y 8
#include "bloom_downsample_common.glsl"

void main() {
    const ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 lid = ivec2(gl_LocalInvocationID.xy);

    // Center of written pixel
    const vec2 uv = (vec2(gid) + 0.5) / pc.target_dim;

    InitializeSharedMemory(ivec2(pc.target_dim), ivec2(pc.source_dim));

    barrier();

    if (any(greaterThanEqual(gid, pc.target_dim))) { return; }

    vec3 filterSum = vec3(0);
    filterSum += sh_coarseSamples[lid.x + 0][lid.y + 0] * (1.0 / 32.0);
    filterSum += sh_coarseSamples[lid.x + 2][lid.y + 0] * (1.0 / 32.0);
    filterSum += sh_coarseSamples[lid.x + 0][lid.y + 2] * (1.0 / 32.0);
    filterSum += sh_coarseSamples[lid.x + 2][lid.y + 2] * (1.0 / 32.0);

    filterSum += sh_coarseSamples[lid.x + 1][lid.y + 2] * (2.0 / 32.0);
    filterSum += sh_coarseSamples[lid.x + 1][lid.y + 0] * (2.0 / 32.0);
    filterSum += sh_coarseSamples[lid.x + 2][lid.y + 1] * (2.0 / 32.0);
    filterSum += sh_coarseSamples[lid.x + 0][lid.y + 1] * (2.0 / 32.0);

    filterSum += sh_coarseSamples[lid.x + 1][lid.y + 1] * (4.0 / 32.0);

    filterSum += sh_preciseSamples[lid.x + 0][lid.y + 0] * (4.0 / 32.0);
    filterSum += sh_preciseSamples[lid.x + 1][lid.y + 0] * (4.0 / 32.0);
    filterSum += sh_preciseSamples[lid.x + 0][lid.y + 1] * (4.0 / 32.0);
    filterSum += sh_preciseSamples[lid.x + 1][lid.y + 1] * (4.0 / 32.0);

    imageStore(gstorage[pc.target_img], gid, vec4(filterSum, 1.0));
}
