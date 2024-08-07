#version 460
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_samplerless_texture_functions : require

#include "shared.glsl"
#include <hashes.glsl>
#include <textures.glsl>

layout(set = 0, binding = 0) uniform sampler gsamplers[];
layout(set = 0, binding = 1) uniform texture2D gtextures[];

layout(scalar, push_constant) uniform PushConstant {
    uint num_rays;
    uint num_bounces;
    float time;
    vec2 noise_offset;
    Rays rays_ptr;
    Lights lights_ptr;
}
pc;

const float PI = acos(-1.);

float ihash11(float p) {
    p = fract(p * .1031);
    p *= p + 33.33;
    p *= p + p;
    return fract(p);
}

vec3 hash13(float p) {
    vec3 p3 = fract(vec3(p) * vec3(.1031, .1030, .0973));
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.xxy + p3.yzz) * p3.zyx);
}

vec4 read_blue_noise(ivec2 loc) {
    vec2 tex_size = vec2(1024);
    ivec2 offset = ivec2(pc.noise_offset * tex_size);
    ivec2 wrappedloc = (loc + offset) % ivec2(tex_size);
    vec4 sampleValue = texelFetch(gtextures[BLUE_TEX], wrappedloc, 0);
    return sampleValue;
}

vec3 ortho(vec3 v) {
    return abs(v.x) > abs(v.z) ? vec3(-v.y, v.x, 0.0) : vec3(0.0, -v.z, v.y);
}

vec3 get_cosine_weighted_sample(vec3 dir, float radius) {
    vec3 o1 = normalize(ortho(dir));
    vec3 o2 = normalize(cross(dir, o1));
    vec2 r = vec2(hash1(), hash1());
    r.x = r.x * 2.0 * PI;
    r.y = pow(r.y, radius);
    float oneminus = sqrt(abs(1.0 - r.y * r.y));
    return cos(r.x) * oneminus * o1 + sin(r.x) * oneminus * o2 + r.y * dir;
}

vec3 rand_cone_direction(const float i, const float angularRadius,
                         const int steps) {
    float x = i * 2.0 - 1.0;
    float y = i * float(steps) * 16.0 * 16.0 * goldenAngle;

    float angle = acos(x) * radians(angularRadius) * 1. / PI;
    float c = cos(angle);
    float s = sin(angle);

    return vec3(cos(y) * s, sin(y) * s, c);
}

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

void main() {
    uint idx = gl_GlobalInvocationID.x;
    if (idx == 0) { pc.rays_ptr.len = pc.num_rays * pc.num_bounces; }
    if (idx > pc.num_rays) { return; }
    init_seed(idx, pc.time);
    uint ray_idx = idx * pc.num_bounces;
    for (int i = 0; i < pc.num_bounces; ++i) {
        pc.rays_ptr.rays[ray_idx + i] = Ray(vec4(0.), vec3(0.), vec3(0.));
    }

    uint light_idx = uint(floor(hash11(int(idx)) * pc.lights_ptr.len));
    vec4 blue_noise = read_blue_noise(ivec2(idx, pc.time));

    Light light = pc.lights_ptr.lights[light_idx];

    vec3 col = light.color.rgb;
    vec3 origin = (light.transform * vec4(vec3(0., 0., 0.), 1.)).xyz;
    vec3 dir =
        rand_cone_direction((float(idx) + blue_noise.x) / float(pc.num_rays),
                            10., int(pc.num_rays));
    dir = (light.transform * vec4(dir, 0.)).xyz;

    float throughput = 1.;
    for (int i = 0; i < pc.num_bounces; ++i) {
        vec2 hit = trace(origin, dir);
        if (hit.y <= 0.) { break; }
        vec3 end = origin + dir * hit.x;
        vec3 nor = get_norm(end);

        // float throughput = 1. - float(i) / float(pc.num_bounces);
        Ray ray = Ray(vec4(col * 0.025, throughput), vec3(origin), vec3(end));
        pc.rays_ptr.rays[ray_idx + i] = ray;
        throughput *= 0.97;

        float t = pc.time * 0.5;
        t = floor(t) + pow(fract(t), 15.);
        if (hash1() < 0.5 + sin(t) * 0.5) {
            dir = get_cosine_weighted_sample(refract(dir, nor, 0.), 0.5);
        } else {
            dir = get_cosine_weighted_sample(nor, 0.5);
        }
        origin = end + nor * 0.02;
    }
}
