const float goldenAngle = 2.3999632297286533;
const float RAY_COLOR_RANGE = 5000.;

struct Light {
    vec4 pos;
    vec4 color;
    mat4 transform;
};

layout(std430, buffer_reference, buffer_reference_align = 8) buffer Lights {
    uint len;
    Light lights[];
};

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

#define ITERS 5
#define SCALE 3.
#define MR2 0.25
vec4 scalevec = vec4(SCALE, SCALE, SCALE, abs(SCALE)) / MR2;
float C1 = abs(SCALE - 1.0), C2 = pow(abs(SCALE), float(1 - ITERS));

float mandelbox(vec3 position) {
    vec4 p = vec4(position.xyz, 1.0),
         p0 = vec4(position.xyz, 1.0); // p.w is knighty's DEfactor
    for (int i = 0; i < ITERS; i++) {
        p.xyz =
            clamp(p.xyz, -1.0, 1.0) * 2.0 - p.xyz; // box fold: min3, max3, mad3
        float r2 = dot(p.xyz, p.xyz);              // dp3
        p.xyzw *= clamp(max(MR2 / r2, MR2), 0.0,
                        1.0);       // sphere fold: div1, max1.sat, mul4
        p.xyzw = p * scalevec + p0; // mad4
    }
    return (length(p.xyz) - C1) / p.w - C2;
}

float sdf_model(vec3 p) {
    float width = 2.5;
    float d = sd_box(p, vec3(width));
    // d = length(p) - width;
    d = mandelbox(p);
    // d = abs(d) - 0.01;

    d = min(d, abs(sd_box(p, vec3(4.))) - 0.01);
    return d;
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
        if (t > 500.) { return vec2(t, 0.); }
    }
    return vec2(-1.);
}
