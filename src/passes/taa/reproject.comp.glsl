#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_image_load_formatted : require
#extension GL_EXT_samplerless_texture_functions : require

#include <camera.glsl>
#include <textures.glsl>

layout(scalar, push_constant) uniform PushConstant {
    uint depth_img;
    uint motion_img;
    CameraBuf camera_ptr;
}
pc;

layout(set = 0, binding = 0) uniform sampler gsamplers[];
layout(set = 0, binding = 1) uniform texture2D gtextures[];
layout(set = 1, binding = 0) coherent restrict uniform image2D gstorage[];

vec3 ndc_from_uv_raw_depth(vec2 uv, float raw_depth) {
    return vec3(uv.x * 2. - 1., (1. - uv.y) * 2. - 1., raw_depth);
}

vec3 world_position_from_depth(vec2 uv, float raw_depth,
                               mat4 inverse_projection_view) {
    vec4 clip = vec4(ndc_from_uv_raw_depth(uv, raw_depth), 1.0);
    vec4 world_w = inverse_projection_view * clip;

    return world_w.xyz / world_w.w;
}

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main() {
    const ivec2 gid = ivec2(gl_GlobalInvocationID.xy);
    uvec2 dims = imageSize(gstorage[pc.motion_img]);
    if (any(greaterThanEqual(gid, dims))) { return; }

    vec2 uv = (vec2(gid) + 0.5) / vec2(dims);

    float depth = 0.;
    for (int y = -1; y <= 1; ++y) {
        for (int x = -1; x <= 1; ++x) {
            float d = imageLoad(gstorage[pc.depth_img], gid + ivec2(x, y)).x;
            depth = max(depth, d);
        }
    }

    vec4 curr_position_ndc = vec4(ndc_from_uv_raw_depth(uv, depth), 1.);

    Camera camera = pc.camera_ptr.cam;
    vec3 pos_ws = world_position_from_depth(uv, depth, camera.clip_to_world);
    vec4 prev_position_ndc_w = camera.prev_world_to_clip * vec4(pos_ws, 1.);
    vec3 prev_position_ndc = prev_position_ndc_w.xyz / prev_position_ndc_w.w;

    vec2 velocity = (curr_position_ndc.xy + camera.jitter) -
                    (prev_position_ndc.xy + camera.prev_jitter);
    velocity = velocity * vec2(1., -1.) * 0.5;

    vec2 inv_dims = 1.0 / vec2(dims);
    bool limits =
        all(bvec3(prev_position_ndc.xy ==
                  clamp(prev_position_ndc.xy, -1. + inv_dims, 1. - inv_dims)));
    imageStore(gstorage[pc.motion_img], gid, vec4(velocity, float(limits), 1.));
}
