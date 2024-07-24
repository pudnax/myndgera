const float goldenAngle = 2.3999632297286533;
const float RAY_COLOR_RANGE = 2500.;

struct Ray {
    vec4 color;
    vec3 start;
    vec3 end;
};

layout(std430, buffer_reference, buffer_reference_align = 8) buffer Rays {
    uint len;
    Ray rays[];
};

float sd_box(vec3 p, vec3 h) {
    p = abs(p) - h;
    return length(max(p, 0.)) + min(0., max(p.x, max(p.y, p.z)));
}

float sdf_model(vec3 p) {
    float width = 2.5;
    float box = sd_box(p, vec3(width));
    box = abs(box) - 0.01;
    return box;
}

vec3 get_norm(vec3 p) {
    mat3 k = mat3(p, p, p) - mat3(0.0001);
    return normalize(sdf_model(p) -
                     vec3(sdf_model(k[0]), sdf_model(k[1]), sdf_model(k[2])));
}

vec2 trace(vec3 eye, vec3 dir) {
    float t = 0.;
    for (int i = 0; i < 50; i++) {
        vec3 pos = eye + dir * t;
        float d = sdf_model(pos);
        if ((d) < 0.01) { return vec2(t, 1.); }
        t += d;
        if (t > 500.) { break; }
    }
    return vec2(-1.);
}
