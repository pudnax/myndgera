uint _SEED = 111425u;
uint wang(uint a) {
    a = (a ^ 61U) ^ (a >> 16U);
    a = a * 9U;
    a = a ^ (a >> 4);
    a = a * 0x27d4eb2dU;
    a = a ^ (a >> 15);
    return a;
}

float hash1() { return float(_SEED = wang(_SEED)) / float(0xffffffffU); }
float hash11(int x) { return float(wang(uint(x))) / float(0xffffffffU); }
// float hash11(float x) {
//     return float(wang(floatBitsToUint(x))) / float(0xffffffffU);
// }
vec2 hash2() { return vec2(hash1(), hash1()); }
vec2 hash21(vec2 x) { return vec2(hash1(), hash1()); }
// vec2 hash22(vec2 v) {
//     uvec2 vbits = floatBitsToUint(v);
//     uvec2 pcg = wang(vbits.y ^ vbits.x, vbits.y);
//     return vec2(pcg) * (1.0 / float(0xffffffffU));
// }
vec3 hash3() { return vec3(hash1(), hash1(), hash1()); }
vec4 hash4() { return vec4(hash1(), hash1(), hash1(), hash1()); }

void init_seed(uint idx, float time) {
    _SEED = wang(idx + wang(uint(time * 1500)) * 250u);
}

void init_seed(vec2 uv, float time) {
    uvec2 V = uvec2(uv);
    _SEED = wang(V.x + wang(V.y + wang(uint(time * 1500)) * 200u) +
                 wang(uint(time * 1500)) * 250u);
}
