use dolly::{
    drivers::{Position, Smooth, YawPitch},
    rig::CameraRig,
};
use glam::{Mat4, Quat, Vec3};

#[repr(C)]
#[derive(Copy, Default, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    pub view_position: [f32; 4],
    pub world_to_clip: Mat4,
    pub clip_to_world: Mat4,
}

#[derive(Debug)]
pub struct Camera {
    pub rig: CameraRig,
    pub position: Vec3,
    pub rotation: Quat,
    pub aspect: f32,
}

impl Camera {
    pub const ZNEAR: f32 = 0.001;
    pub const FOVY: f32 = std::f32::consts::PI / 2.0;

    pub fn new(position: Vec3, yaw: f32, pitch: f32) -> Self {
        let rig: CameraRig = CameraRig::builder()
            .with(Position::new(position))
            .with(YawPitch::new().yaw_degrees(yaw).pitch_degrees(pitch))
            .with(Smooth::new_position_rotation(1.0, 1.5))
            .build();
        Self {
            rig,
            aspect: 1.25,
            position,
            rotation: Quat::IDENTITY,
        }
    }

    pub fn build_projection_view_matrix(&self) -> (Mat4, Mat4) {
        let tr = self.rig.final_transform;
        let pos: Vec3 = tr.position.into();
        let view = Mat4::look_at_rh(pos, pos + tr.forward::<Vec3>(), tr.up());
        let proj = Mat4::perspective_infinite_reverse_rh(Self::FOVY, self.aspect, Self::ZNEAR);
        (proj, view)
    }

    pub fn get_uniform(&self) -> CameraUniform {
        let pos = Vec3::from(self.rig.final_transform.position).extend(1.);
        let (projection, view) = self.build_projection_view_matrix();
        let world_to_clip = projection * view;

        CameraUniform {
            view_position: pos.to_array(),
            world_to_clip,
            clip_to_world: world_to_clip.inverse(),
        }
    }
}
