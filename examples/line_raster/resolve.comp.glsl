#version 460

#include <extensions.glsl>

layout(set = 0, binding = 0) uniform sampler gsamplers[];
layout(set = 0, binding = 1) uniform texture2D gtextures[];
layout(set = 1, binding = 0) coherent restrict uniform uimage2D gstorage_read[];
layout(set = 1, binding = 0) writeonly coherent
    restrict uniform image2D gstorage_write[];

#include "shared.glsl"
#include <camera.glsl>

layout(scalar, push_constant) uniform PushConstant {
    uint target_img;
    uint red_img;
    uint green_img;
    uint blue_img;
    uint depth_img;
    CameraBuf camera;
}
pc;

vec2 cs_to_uv(vec2 pix, vec2 dims) {
    vec2 uv = (pix / dims);
    uv.y = 1. - uv.y;
    uv = uv * 2. - 1.0;
    return uv;
}

vec2 RESOLUTION;

float sdf_model1(vec3 p) { return sd_box(p, vec3(4.)); }

vec3 get_norm(vec3 p, float s) {
    vec2 off = vec2(s, 0);
    return normalize(vec3(sdf_model(p + off.xyy).x, sdf_model(p + off.yxy).x,
                          sdf_model(p + off.yyx).x) -
                     vec3(sdf_model(p - off.xyy).x, sdf_model(p - off.yxy).x,
                          sdf_model(p - off.yyx).x));
}

vec3 wire_trace(vec3 eye, vec3 dir) {
    vec3 wire_color = vec3(1., 0., 1.);
    float wire = 0.;
    float s = 1.;
    float t = 0.;
    for (int i = 0; i < 50; i++) {
        vec3 pos = eye + dir * t;
        float d = sdf_model(pos);
        if (abs(d) < 0.001) {
            float edge = 0.0015 * clamp(800. / RESOLUTION.x, 1., 2.5);
            float edge_amount =
                length(get_norm(pos, 0.05) - get_norm(pos, edge));
            wire += smoothstep(0.0, .10, edge_amount) * 0.5;
            s *= -1.;
            d = 0.025 * s;
        }
        t += d * s; // * 0.85;
        if (t > 500.) {
            break;
        }
    }
    return clamp(wire * wire_color, 0.0, 1.0);
}

layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;

void main() {
    vec2 dims = imageSize(gstorage_read[pc.red_img]);
    RESOLUTION = dims;
    vec2 pix = gl_GlobalInvocationID.xy;
    vec2 uv = cs_to_uv(pix, dims);
    if (pix.x >= dims.x || pix.y >= dims.y) {
        return;
    }
    ivec2 ipix = ivec2(pix);

    vec4 view_pos = pc.camera.cam.clip_to_world * vec4(uv, 1., 1.);
    vec4 view_dir = pc.camera.cam.clip_to_world * vec4(uv, 0., 1.);

    vec3 eye = view_pos.xyz / view_pos.w;
    vec3 dir = normalize(view_dir.xyz);

    vec3 col = vec3(0.);
    // col += wire_trace(eye, dir) * 0.25;

    float red = imageLoad(gstorage_read[pc.red_img], ipix).x;
    float green = imageLoad(gstorage_read[pc.green_img], ipix).x;
    float blue = imageLoad(gstorage_read[pc.blue_img], ipix).x;
    col += vec3(red, green, blue) / RAY_COLOR_RANGE;

    // float d = imageLoad(gstorage_read[pc.depth_img], ipix).x;
    // d = smoothstep(d, 0., 1.);
    // col = vec3(d);

    imageStore(gstorage_write[pc.target_img], ipix, vec4(col, 1.));
}
