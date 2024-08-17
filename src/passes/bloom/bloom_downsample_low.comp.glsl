#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_samplerless_texture_functions : require

#define LOCAL_SIZE_X 16
#define LOCAL_SIZE_Y 16
#include "bloom_downsample_common.glsl"

layout(set = 0, binding = 0) uniform sampler gsamplers[];
layout(set = 0, binding = 1) uniform texture2D gtextures[];
layout(set = 1, binding = 0) uniform readonly image2D gstorage[];

float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// Reduce the dynamic range of the input samples
vec3 karis_average(vec3 c1, vec3 c2, vec3 c3, vec3 c4) {
    float w1 = 1.0 / (luminance(c1.rgb) + 1.0);
    float w2 = 1.0 / (luminance(c2.rgb) + 1.0);
    float w3 = 1.0 / (luminance(c3.rgb) + 1.0);
    float w4 = 1.0 / (luminance(c4.rgb) + 1.0);

    return (c1 * w1 + c2 * w2 + c3 * w3 + c4 * w4) / (w1 + w2 + w3 + w4);
}

void main() {
    const ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 lid = ivec2(gl_LocalInvocationID.xy);

    // Center of written pixel
    const vec2 uv = (vec2(gid) + 0.5) / pc.target_dim;

    InitializeSharedMemory(ivec2(pc.target_dim), ivec2(pc.source_dim));

    barrier();

    if (any(greaterThanEqual(gid, pc.target_dim))) { return; }

    vec3 samples[13];
    samples[0] = sh_coarseSamples[lid.x + 0][lid.y + 0];  //  (-2, -2)
    samples[1] = sh_coarseSamples[lid.x + 1][lid.y + 0];  //  (0, -2)
    samples[2] = sh_coarseSamples[lid.x + 2][lid.y + 0];  //  (2, -2)
    samples[3] = sh_preciseSamples[lid.x + 0][lid.y + 0]; //  (-1, -1)
    samples[4] = sh_preciseSamples[lid.x + 1][lid.y + 0]; //  (1, -1)
    samples[5] = sh_coarseSamples[lid.x + 0][lid.y + 1];  //  (-2, 0)
    samples[6] = sh_coarseSamples[lid.x + 1][lid.y + 1];  //  (0, 0)
    samples[7] = sh_coarseSamples[lid.x + 2][lid.y + 1];  //  (2, 0)
    samples[8] = sh_preciseSamples[lid.x + 0][lid.y + 1]; //  (-1, 1)
    samples[9] = sh_preciseSamples[lid.x + 1][lid.y + 1]; //  (1, 1)
    samples[10] = sh_coarseSamples[lid.x + 0][lid.y + 2]; //  (-2, 2)
    samples[11] = sh_coarseSamples[lid.x + 1][lid.y + 2]; //  (0, 2)
    samples[12] = sh_coarseSamples[lid.x + 2][lid.y + 2]; //  (2, 2)

    vec3 filterSum = vec3(0);
    filterSum +=
        karis_average(samples[3], samples[4], samples[8], samples[9]) * 0.5;
    filterSum +=
        karis_average(samples[0], samples[1], samples[5], samples[6]) * 0.125;
    filterSum +=
        karis_average(samples[1], samples[2], samples[6], samples[7]) * 0.125;
    filterSum +=
        karis_average(samples[5], samples[6], samples[10], samples[11]) * 0.125;
    filterSum +=
        karis_average(samples[6], samples[7], samples[11], samples[12]) * 0.125;

    imageStore(gstorage[pc.target_img], gid, vec4(filterSum, 1.0));
}
