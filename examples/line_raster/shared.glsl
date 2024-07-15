const float PI = acos(-1.);
const float goldenAngle = 2.3999632297286533;

struct Line {
    vec3 start;
    vec3 end;
};

layout(std430, buffer_reference, buffer_reference_align = 8) buffer Lines {
    uint len;
    Line lines[];
};
