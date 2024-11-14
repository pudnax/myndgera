const uint DUMMY_TEX = 0;
const uint DITHER_TEX = 1;
const uint NOISE_TEX = 2;
const uint BLUE_TEX = 3;

const uint LINEAR_SAMPL = 0;
const uint LINEAR_BORDER_SAMPL = 1;
const uint NEAREST_SAMPL = 2;

vec4 Tex(uint tex_id, uint smp_id, vec2 uv) {
    return texture(
        nonuniformEXT(sampler2D(gtextures[tex_id], gsamplers[smp_id])), uv);
}
vec4 TexLinear(uint tex_id, vec2 uv) { return Tex(tex_id, LINEAR_SAMPL, uv); }
vec4 TexNear(uint tex_id, vec2 uv) { return Tex(tex_id, NEAREST_SAMPL, uv); }
