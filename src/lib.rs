mod device;
mod instance;
mod pipeline_arena;
mod shader_compiler;
mod surface;
mod swapchain;
mod watcher;

pub use self::{
    device::{Device, HostBuffer},
    instance::Instance,
    pipeline_arena::*,
    shader_compiler::ShaderCompiler,
    surface::Surface,
    swapchain::Swapchain,
    watcher::Watcher,
};

#[derive(Debug)]
pub enum UserEvent {
    Glsl { path: std::path::PathBuf },
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ShaderSource {
    pub path: std::path::PathBuf,
    pub kind: ShaderKind,
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub enum ShaderKind {
    Fragment,
    Vertex,
    Compute,
}

impl From<ShaderKind> for shaderc::ShaderKind {
    fn from(value: ShaderKind) -> Self {
        match value {
            ShaderKind::Compute => shaderc::ShaderKind::Compute,
            ShaderKind::Vertex => shaderc::ShaderKind::Vertex,
            ShaderKind::Fragment => shaderc::ShaderKind::Fragment,
        }
    }
}

pub fn align_to(size: u64, alignment: u64) -> u64 {
    (size + alignment - 1) & !(alignment - 1)
}
