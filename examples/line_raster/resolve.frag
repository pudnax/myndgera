#version 460
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_nonuniform_qualifier : require

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

float sd_box(vec3 p, vec3 h) {
    p = abs(p) - h;
    return length(max(p, 0.)) + min(0., max(p.x, max(p.y, p.z)));
}

float map(vec3 p) { return sd_box(p, vec3(0.1, 0.25, 0.1)); }

vec2 trace(vec3 eye, vec3 dir) {
    float t = 0.;
    for (int i = 0; i < 100; i++) {
        vec3 pos = eye + dir * t;
        float d = map(pos);
        if (d < 0.001) { return vec2(t, 1.); }
        t += d;
        if (t > 500.) { break; }
    }
    return vec2(-1.);
}

vec3 get_norm(vec3 p) {
    mat3 k = mat3(p, p, p) - mat3(0.0001);
    return normalize(map(p) - vec3(map(k[0]), map(k[1]), map(k[2])));
}

void main() {
    vec2 dims = vec2(imageSize(gstorage_textures[pc.red_img]));
    vec2 uv = (in_uv * 2. - 1.) * vec2(dims.x / dims.y, 1);

    vec4 screen_point = vec4(in_uv * 2. - 1., 0., 1.);
    vec4 screen_tangent = screen_point + vec4(0., 0., 1., 0.);

    vec4 view_pos = pc.camera.cam.clip_to_world * screen_point;
    vec4 view_tang = pc.camera.cam.clip_to_world * screen_tangent;

    vec3 eye = vec3(0., 0., -3);
    vec3 dir = normalize(vec3(uv, 1.));

    eye = view_pos.xyz / view_pos.w;
    eye = pc.camera.cam.pos.xyz;
    dir = normalize(view_tang.xyz / view_tang.w - eye);

    vec3 col = vec3(0.);
    vec2 hit = trace(eye, dir);
    if (hit.y > 0.) {
        vec3 pos = eye + dir * hit.x;
        vec3 nor = get_norm(pos);
        col = nor * 0.5 + 0.5;
    }

    vec2 u = in_uv;
    u.y = 1. - u.y;
    ivec2 pix = ivec2(u * dims);
    col.r += imageLoad(gstorage_textures[pc.red_img], pix).x / 255.;
    col.g += imageLoad(gstorage_textures[pc.green_img], pix).x / 255.;
    col.b += imageLoad(gstorage_textures[pc.blue_img], pix).x / 255.;

    col = pow(col, vec3(0.4545));

    out_color = vec4(col, 1.0);
}
