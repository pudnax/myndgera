mod device;
mod instance;
mod pipeline_arena;
mod recorder;
mod shader_compiler;
mod surface;
mod swapchain;
mod watcher;

use std::sync::Arc;

pub use self::{
    device::{Device, HostBuffer},
    instance::Instance,
    pipeline_arena::*,
    recorder::{RecordEvent, Recorder},
    shader_compiler::ShaderCompiler,
    surface::Surface,
    swapchain::Swapchain,
    watcher::Watcher,
};

use ash::{prelude::VkResult, vk};

pub fn align_to(size: u64, alignment: u64) -> u64 {
    (size + alignment - 1) & !(alignment - 1)
}

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

#[derive(Clone, Copy)]
pub struct ImageDimensions {
    pub width: usize,
    pub height: usize,
    pub padded_bytes_per_row: usize,
    pub unpadded_bytes_per_row: usize,
}

impl ImageDimensions {
    fn new(width: usize, height: usize, row_pitch: usize) -> Self {
        let channel_width = std::mem::size_of::<[u8; 4]>();
        let unpadded_bytes_per_row = width * channel_width;
        Self {
            width,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row: row_pitch,
        }
    }
}

pub struct ManagedImage {
    image: vk::Image,
    memory: vk::DeviceMemory,
    memory_reqs: vk::MemoryRequirements,
    image_dimensions: ImageDimensions,
    ptr: Option<&'static mut [u8]>,
    device: Arc<ash::Device>,
}

impl ManagedImage {
    pub fn new(
        device: &Device,
        info: &vk::ImageCreateInfo,
        memory_props: vk::MemoryPropertyFlags,
    ) -> anyhow::Result<Self> {
        let image = unsafe { device.create_image(&info, None)? };
        let memory_reqs = unsafe { device.get_image_memory_requirements(image) };
        let memory = device.alloc_memory(memory_reqs, memory_props)?;
        unsafe { device.bind_image_memory(image, memory, 0) }?;
        let subresource = vk::ImageSubresource::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .mip_level(0)
            .array_layer(0);
        let image_subresource = unsafe { device.get_image_subresource_layout(image, subresource) };
        let image_dimensions = ImageDimensions::new(
            info.extent.width as _,
            info.extent.height as _,
            image_subresource.row_pitch as _,
        );
        Ok(Self {
            image,
            memory,
            memory_reqs,
            image_dimensions,
            ptr: None,
            device: device.device.clone(),
        })
    }

    pub fn map_memory(&mut self) -> VkResult<()> {
        if self.ptr.is_some() {
            return Ok(());
        }
        let ptr = unsafe {
            self.device.map_memory(
                self.memory,
                0,
                self.memory_reqs.size,
                vk::MemoryMapFlags::empty(),
            )?
        };
        self.ptr =
            Some(unsafe { std::slice::from_raw_parts_mut(ptr.cast(), self.memory_reqs.size as _) });
        Ok(())
    }
}

impl Drop for ManagedImage {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image(self.image, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
