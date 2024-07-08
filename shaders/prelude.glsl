const float PI = acos(-1.);
const float TAU = 2. * PI;

const uint DUMMY_TEX = 0;
const uint PREV_FRAME_TEX = 1;
const uint GENERIC_TEX1 = 2;
const uint GENERIC_TEX2 = 3;
const uint DITHER_TEX = 4;
const uint NOISE_TEX = 5;
const uint BLUE_TEX = 6;

const uint LINEAR_SAMPL = 0;
const uint NEAREST_SAMPL = 1;

const float HALF_WIDTH = 1.0;

vec4 ASSERT_COL = vec4(0.);
void assert(bool cond, int v) {
    if (!(cond)) {
        if (v == 0)
            ASSERT_COL.x = -1.0;
        else if (v == 1)
            ASSERT_COL.y = -1.0;
        else if (v == 2)
            ASSERT_COL.z = -1.0;
        else
            ASSERT_COL.w = -1.0;
    }
}
void assert(bool cond) { assert(cond, 0); }
#define catch_assert(out)                                                      \
    if (ASSERT_COL.x < 0.0)                                                    \
        out = vec4(1.0, 0.0, 0.0, 1.0);                                        \
    if (ASSERT_COL.y < 0.0)                                                    \
        out = vec4(0.0, 1.0, 0.0, 1.0);                                        \
    if (ASSERT_COL.z < 0.0)                                                    \
        out = vec4(0.0, 0.0, 1.0, 1.0);                                        \
    if (ASSERT_COL.w < 0.0)                                                    \
        out = vec4(1.0, 1.0, 0.0, 1.0);

float AAstep(float threshold, float val) {
    return smoothstep(-.5, .5,
                      (val - threshold) / min(0.005, fwidth(val - threshold)));
}
float AAstep(float val) { return AAstep(val, 0.); }

float worldsdf(vec3 rayPos);

vec2 ray_march(vec3 rayPos, vec3 rayDir) {
    const vec3 EPS = vec3(0., 0.001, 0.0001);
    const float HIT_DIST = EPS.y;
    const int MAX_STEPS = 100;
    const float MISS_DIST = 10.0;
    float dist = 0.0;

    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 pos = rayPos + (dist * rayDir);
        float posToScene = worldsdf(pos);
        dist += posToScene;
        if (abs(posToScene) < HIT_DIST)
            return vec2(dist, i);
        if (posToScene > MISS_DIST)
            break;
    }

    return vec2(-dist, MAX_STEPS);
}

mat2 rotate(float angle) {
    float sine = sin(angle);
    float cosine = cos(angle);
    return mat2(cosine, -sine, sine, cosine);
}

vec3 enlight(in vec3 at, vec3 normal, vec3 diffuse, vec3 l_color, vec3 l_pos) {
    vec3 l_dir = l_pos - at;
    return diffuse * l_color * max(0., dot(normal, normalize(l_dir))) /
           dot(l_dir, l_dir);
}

vec3 wnormal(in vec3 p) {
    const vec3 EPS = vec3(0., 0.01, 0.0001);
    return normalize(vec3(worldsdf(p + EPS.yxx) - worldsdf(p - EPS.yxx),
                          worldsdf(p + EPS.xyx) - worldsdf(p - EPS.xyx),
                          worldsdf(p + EPS.xxy) - worldsdf(p - EPS.xxy)));
}
