#![allow(clippy::new_without_default)]

mod device;
mod instance;
mod pipeline_arena;
mod recorder;
mod shader_compiler;
mod surface;
mod swapchain;
mod watcher;

use std::{mem::ManuallyDrop, sync::Arc};

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

use ash::vk::{self, DeviceMemory};
use gpu_alloc::{GpuAllocator, MapError, MemoryBlock};
use gpu_alloc_ash::AshMemoryDevice;
use parking_lot::Mutex;

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
    memory: ManuallyDrop<MemoryBlock<DeviceMemory>>,
    image_dimensions: ImageDimensions,
    data: Option<&'static mut [u8]>,
    device: Arc<ash::Device>,
    allocator: Arc<Mutex<GpuAllocator<DeviceMemory>>>,
}

impl ManagedImage {
    pub fn new(
        device: &Device,
        info: &vk::ImageCreateInfo,
        usage: gpu_alloc::UsageFlags,
    ) -> anyhow::Result<Self> {
        let image = unsafe { device.create_image(info, None)? };
        let memory_reqs = unsafe { device.get_image_memory_requirements(image) };
        let memory = device.alloc_memory(memory_reqs, usage)?;
        unsafe { device.bind_image_memory(image, *memory.memory(), memory.offset()) }?;
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
            memory: ManuallyDrop::new(memory),
            image_dimensions,
            data: None,
            device: device.device.clone(),
            allocator: device.allocator.clone(),
        })
    }

    pub fn map_memory(&mut self) -> Result<(), MapError> {
        if self.data.is_some() {
            return Ok(());
        }
        let size = self.memory.size() as usize;
        let offset = self.memory.offset();
        let ptr = unsafe {
            self.memory
                .map(AshMemoryDevice::wrap(&self.device), offset, size)?
        };
        self.data = Some(unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr().cast(), size) });
        Ok(())
    }
}

impl Drop for ManagedImage {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_image(self.image, None);
            {
                let mut allocator = self.allocator.lock();
                let memory = ManuallyDrop::take(&mut self.memory);
                allocator.dealloc(AshMemoryDevice::wrap(&self.device), memory);
            }
        }
    }
}

pub fn find_memory_type_index(
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    memory_type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_type_bits != 0 && (memory_type.property_flags & flags) == flags
        })
        .map(|(index, _memory_type)| index as _)
}
