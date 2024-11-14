layout(set = 0, binding = 0) uniform sampler gsamplers[];
layout(set = 0, binding = 1) uniform texture2D gtextures[];
layout(set = 1, binding = 0) writeonly coherent
    restrict uniform image2D gstorage[];

#include <textures.glsl>

layout(scalar, push_constant) uniform PushConstant {
    uint source_img;
    uint target_img;
    uvec2 source_dim;
    uvec2 target_dim;
}
pc;

layout(local_size_x = LOCAL_SIZE_X, local_size_y = LOCAL_SIZE_Y) in;

/*
We take 13 bilinear samples of the source texture as such:

 O   O   O
   o   o
 O   X   O
   o   o
 O   O   O

 where X is the position of the pixel we are computing.
 Samples are intentionally staggered.
*/

vec3 tex_lod(uint tex_id, vec2 uv) {
    vec4 tex =
        textureLod(nonuniformEXT(sampler2D(gtextures[tex_id],
                                           gsamplers[LINEAR_BORDER_SAMPL])),
                   uv, 0);
    return tex.rgb;
}

// Cached samples corresponding to the large 'O's in the above image
shared vec3 sh_coarseSamples[gl_WorkGroupSize.x + 2][gl_WorkGroupSize.y + 2];

// Cached samples corresponding to the small 'o's in the above image
shared vec3 sh_preciseSamples[gl_WorkGroupSize.x + 1][gl_WorkGroupSize.y + 1];

void InitializeSharedMemory(ivec2 target_dim, ivec2 source_dim) {
    const ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    const ivec2 lid = ivec2(gl_LocalInvocationID.xy);

    // Center of written pixel
    const vec2 uv = (vec2(gid) + 0.5) / target_dim;

    // Minimum caching for each output pixel
    sh_coarseSamples[lid.x + 1][lid.y + 1] =
        tex_lod(pc.source_img, uv + vec2(0, 0) / source_dim);
    sh_preciseSamples[lid.x + 0][lid.y + 0] =
        tex_lod(pc.source_img, uv + vec2(-1, -1) / source_dim);

    // Pixels on the edge of the thread group
    // Left
    if (lid.x == 0) {
        sh_coarseSamples[lid.x + 0][lid.y + 1] =
            tex_lod(pc.source_img, uv + vec2(-2, 0) / source_dim);
    }

    // Right
    if (lid.x == gl_WorkGroupSize.x - 1) {
        sh_coarseSamples[lid.x + 2][lid.y + 1] =
            tex_lod(pc.source_img, uv + vec2(2, 0) / source_dim);
        sh_preciseSamples[lid.x + 1][lid.y + 0] =
            tex_lod(pc.source_img, uv + vec2(1, -1) / source_dim);
    }

    // Bottom
    if (lid.y == 0) {
        sh_coarseSamples[lid.x + 1][lid.y + 0] =
            tex_lod(pc.source_img, uv + vec2(0, -2) / source_dim);
    }

    // Top
    if (lid.y == gl_WorkGroupSize.y - 1) {
        sh_coarseSamples[lid.x + 1][lid.y + 2] =
            tex_lod(pc.source_img, uv + vec2(0, 2) / source_dim);
        sh_preciseSamples[lid.x + 0][lid.y + 1] =
            tex_lod(pc.source_img, uv + vec2(-1, 1) / source_dim);
    }

    // Bottom-left
    if (lid.x == 0 && lid.y == 0) {
        sh_coarseSamples[lid.x + 0][lid.y + 0] =
            tex_lod(pc.source_img, uv + vec2(-2, -2) / source_dim);
    }

    // Bottom-right
    if (lid.x == gl_WorkGroupSize.x - 1 && lid.y == 0) {
        sh_coarseSamples[lid.x + 2][lid.y + 0] =
            tex_lod(pc.source_img, uv + vec2(2, -2) / source_dim);
    }

    // Top-left
    if (lid.x == 0 && lid.y == gl_WorkGroupSize.y - 1) {
        sh_coarseSamples[lid.x + 0][lid.y + 2] =
            tex_lod(pc.source_img, uv + vec2(-2, 2) / source_dim);
    }

    // Top-right
    if (lid == gl_WorkGroupSize.xy - 1) {
        sh_coarseSamples[lid.x + 2][lid.y + 2] =
            tex_lod(pc.source_img, uv + vec2(2, 2) / source_dim);
        sh_preciseSamples[lid.x + 1][lid.y + 1] =
            tex_lod(pc.source_img, uv + vec2(1, 1) / source_dim);
    }
}
