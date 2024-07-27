#extension GL_EXT_scalar_block_layout : require

struct Camera {
    vec4 pos;
    mat4 world_to_clip;
    mat4 clip_to_world;
};

layout(scalar, buffer_reference,
       buffer_reference_align = 16) readonly buffer CameraBuf {
    Camera cam;
};
