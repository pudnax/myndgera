#version 460

#include <extensions.glsl>

layout(set = 0, binding = 0) uniform sampler gsamplers[];
layout(set = 0, binding = 1) uniform texture2D gtextures[];
layout(set = 1, binding = 0, r32ui) uniform coherent
    restrict uimage2D gstorage[];
layout(set = 1, binding = 0, r16f) uniform coherent
    restrict image2D gstoragef[];

#include "./shared.glsl"
#include <camera.glsl>
#include <textures.glsl>

layout(scalar, push_constant) uniform PushConstant {
    uint red_img;
    uint green_img;
    uint blue_img;
    uint depth_img;
    vec2 noise_offset;
    CameraBuf camera;
    Rays rays_ptr;
}
pc;

vec4 read_blue_noise(ivec2 loc) {
    vec2 tex_size = vec2(1024);
    ivec2 offset = ivec2(pc.noise_offset * tex_size);
    ivec2 wrappedloc = (loc + offset) % ivec2(tex_size);
    vec4 sampleValue = texelFetch(gtextures[BLUE_TEX], wrappedloc, 0);
    return sampleValue;
}

vec4 project(mat4 world_to_clip, vec3 pos) {
    vec4 screen_pos = world_to_clip * vec4(pos, 1.);
    screen_pos.xyz /= screen_pos.w;
    return screen_pos;
}

bool in_bounds(vec2 pos, vec2 dims) {
    return all(greaterThanEqual(pos, ivec2(0))) && all(lessThan(pos, dims));
}

bool in_clip_space(vec4 pos) {
    return !(pos.w <= 0.0 || pos.x < -1.0 || pos.x > 1.0 || pos.y < -1.0 ||
             pos.y > 1.0);
}

vec2 ndc_to_raster(vec4 ndc, vec2 dims) {
    vec2 res = (ndc.xy * 0.5 + 0.5);
    res.y = 1. - res.y;
    return res * dims;
}

void draw_point(ivec2 pix, vec3 col) {
    uvec3 ucol = uvec3(floor(col * RAY_COLOR_RANGE));
    imageAtomicAdd(gstorage[pc.red_img], pix, ucol.r);
    imageAtomicAdd(gstorage[pc.green_img], pix, ucol.g);
    imageAtomicAdd(gstorage[pc.blue_img], pix, ucol.b);
}

void naive(vec2 ray_start, vec2 ray_end, Ray ray, vec2 dims, float start_depth,
           float end_depth) {
    vec2 ray_dir = ray_end - ray_start;
    float ray_len = length(ray_dir);
    ray_dir /= ray_len;

    vec4 blue_noise = read_blue_noise(ivec2(gl_GlobalInvocationID.x, 0.));

    float n = 25;
    float step = ray_len / n;
    step = step + (blue_noise.x * 2. - 1.) * step / 2.;
    step = max(step, sqrt(1.));
    for (float s = 0; s < ray_len; s += step) {
        vec2 p = ray_start + ray_dir * s;
        ivec2 pix = ivec2(p);
        if (!in_bounds(pix, dims)) {
            continue;
        }
        float curr_depth = mix(start_depth, end_depth, s / ray_len);
        float depth = imageLoad(gstoragef[pc.depth_img], pix).x;
        if (curr_depth > depth) {
            imageStore(gstoragef[pc.depth_img], pix,
                       vec4(curr_depth, 0., 0., 0.));
        }
        vec3 col = ray.color.rgb * ray.color.w;
        draw_point(pix, col);
    }
}

void dda(vec2 ray_start, vec2 ray_end, Ray ray, vec2 dims) {
    vec2 ray_dir = ray_end - ray_start;
    float ray_len_squared = dot(ray_dir, ray_dir);
    ray_dir /= ray_len_squared * ray_len_squared;

    ivec2 map_pos = ivec2(floor(ray_start + 0.5));

    vec2 delta_dist = 1.0 / abs(ray_dir);
    ivec2 ray_step = ivec2(sign(ray_dir));
    vec2 side_dist = (sign(ray_dir) * (vec2(map_pos) - ray_start) +
                      (sign(ray_dir) * 0.5) + 0.5) *
                     delta_dist;

    for (int i = 0; i < 2500; i++) {
        vec2 len = ray_start - map_pos;
        if (dot(len, len) > ray_len_squared) {
            break;
        }

        bvec2 mask = lessThanEqual(side_dist.xy, side_dist.yx);
        side_dist += vec2(mask) * delta_dist;
        map_pos += ivec2(vec2(mask)) * ray_step;
        if (!in_bounds(map_pos, dims)) {
            continue;
        }

        vec3 col = ray.color.rgb * ray.color.w;
        draw_point(map_pos, col);
    }
}

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

void main() {
    uvec2 dims = imageSize(gstorage[pc.red_img]);

    uint idx = gl_GlobalInvocationID.x;
    if (idx > pc.rays_ptr.len) {
        return;
    }

    Ray ray = pc.rays_ptr.rays[idx];
    vec3 gray_start = ray.start.xyz;
    vec3 gray_end = ray.end.xyz;
    if (gray_start == vec3(0) && gray_end == vec3(0.)) {
        return;
    }

    vec4 cray_start = project(pc.camera.cam.world_to_clip, gray_start);
    vec4 cray_end = project(pc.camera.cam.world_to_clip, gray_end);

    if (!in_clip_space(cray_start) && !in_clip_space(cray_end)) {
        return;
    }

    vec2 ray_start = ndc_to_raster(cray_start, dims);
    vec2 ray_end = ndc_to_raster(cray_end, dims);

    naive(ray_start, ray_end, ray, vec2(dims), cray_start.z, cray_end.z);
    // dda(ray_start, ray_end, ray, vec2(dims));
}
