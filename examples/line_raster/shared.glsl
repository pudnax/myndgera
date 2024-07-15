const float PI = acos(-1.);
const float goldenAngle = 2.3999632297286533;

struct Line {
    vec4 start;
    vec4 end;
};

layout(std430, buffer_reference, buffer_reference_align = 16) buffer Lines {
    Line lines[];
};
