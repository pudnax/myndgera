#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

#include "shared.glsl"
#include <camera.glsl>
#include <prelude.glsl>

layout(location = 0) in vec2 in_uv;
layout(location = 0) out vec4 out_color;

layout(set = 0, binding = 0) uniform sampler gsamplers[];
layout(set = 0, binding = 1) uniform texture2D gtextures[];
vec4 Tex(uint tex_id, uint smp_id, vec2 uv) {
    return texture(
        nonuniformEXT(sampler2D(gtextures[tex_id], gsamplers[smp_id])), uv);
}
vec4 Tex(uint tex_id, uint smp_id) { return Tex(tex_id, smp_id, in_uv); }
vec4 TexLinear(uint tex_id, vec2 uv) { return Tex(tex_id, LINEAR_SAMPL, uv); }
vec4 TexLinear(uint tex_id) { return Tex(tex_id, LINEAR_SAMPL, in_uv); }
vec4 TexNear(uint tex_id, vec2 uv) { return Tex(tex_id, NEAREST_SAMPL, uv); }
vec4 TexNear(uint tex_id) { return Tex(tex_id, NEAREST_SAMPL, in_uv); }

layout(set = 1, binding = 0,
       r32ui) uniform readonly uimage2D gstorage_textures[];

layout(std430, push_constant) uniform PushConstant {
    uint idx;
    uint red_img;
    uint green_img;
    uint blue_img;
    CameraBuf camera;
}
pc;

vec2 RESOLUTION;

float map(vec3 p) { return sd_box(p, vec3(0.1, 0.25, 0.1)); }

vec3 get_norm(vec3 p, float eps) {
    mat3 k = mat3(p, p, p) - mat3(eps);
    return normalize(vec3(sdf_model(p)) -
                     vec3(sdf_model(k[0]), sdf_model(k[1]), sdf_model(k[2])));
}

vec3 wire_trace(vec3 eye, vec3 dir) {
    vec3 wire_color = vec3(1., 0., 1.);
    vec3 col = vec3(0.);
    float s = 1.;
    float t = 0.;
    for (int i = 0; i < 200; i++) {
        vec3 pos = eye + dir * t;
        float d = sdf_model(pos);
        if (abs(d) < 0.001) {
            float edge = 0.00015 * t * clamp(fwidth(1. * in_uv).x, 1., 2.5);
            float edge_amount =
                length(get_norm(pos, 0.012) - get_norm(pos, edge));
            col += wire_color * smoothstep(0.0, 1.0, edge_amount) * 0.5;
            s *= -1.;
            d = 0.25 * s;
        }
        t += d * s;
        if (t > 500.) { break; }
    }
    return max(col, vec3(0.));
}

void main() {
    vec2 dims = vec2(imageSize(gstorage_textures[pc.red_img]));
    RESOLUTION = dims;
    vec2 uv = (in_uv * 2. - 1.) * vec2(dims.x / dims.y, 1);

    vec2 screen_point = in_uv * 2. - 1.0;

    vec4 view_pos = pc.camera.cam.clip_to_world * vec4(screen_point, 1., 1.);
    vec4 view_dir = pc.camera.cam.clip_to_world * vec4(screen_point, 0., 1.);

    vec3 eye = view_pos.xyz / view_pos.w;
    vec3 dir = normalize(view_dir.xyz);

    vec3 col = vec3(0.);
    vec2 hit = trace(eye, dir);
    if (hit.y > 0.) {
        vec3 pos = eye + dir * hit.x;
        vec3 nor = get_norm(pos);
        // col = 0.03 * (nor * 0.5 + 0.5);
    }

    col += wire_trace(eye, dir);

    vec2 u = in_uv;
    u.y = 1. - u.y;
    ivec2 pix = ivec2(u * dims);
    col.r += imageLoad(gstorage_textures[pc.red_img], pix).x / 255.;
    col.g += imageLoad(gstorage_textures[pc.green_img], pix).x / 255.;
    col.b += imageLoad(gstorage_textures[pc.blue_img], pix).x / 255.;

    col = pow(col, vec3(0.4545));

    out_color = vec4(col, 1.0);
}
