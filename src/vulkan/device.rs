use anyhow::Result;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator},
    MemoryLocation,
};
use parking_lot::Mutex;
use std::marker::PhantomData;
use std::{
    ffi::{CStr, CString},
    mem::ManuallyDrop,
    sync::Arc,
};

use ash::{
    ext, khr,
    prelude::VkResult,
    vk::{self, Handle},
};

use crate::{align_to, ManagedImage};

pub struct Device {
    pub physical_device: vk::PhysicalDevice,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub device_properties: vk::PhysicalDeviceProperties,
    pub descriptor_indexing_props: vk::PhysicalDeviceDescriptorIndexingProperties<'static>,
    pub command_pool: vk::CommandPool,
    pub main_queue_family_idx: u32,
    pub transfer_queue_family_idx: u32,
    pub allocator: Mutex<Allocator>,
    pub device: ash::Device,
    pub dynamic_rendering: khr::dynamic_rendering::Device,
    pub(crate) dbg_utils: ext::debug_utils::Device,
}

impl std::ops::Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Device {
    pub fn name_object(&self, handle: impl Handle, name: &str) {
        let name = CString::new(name).unwrap();
        let _ = unsafe {
            self.dbg_utils.set_debug_utils_object_name(
                &vk::DebugUtilsObjectNameInfoEXT::default()
                    .object_handle(handle)
                    .object_name(&name),
            )
        };
    }

    pub fn begin_debug_marker(&self, &cbuff: &vk::CommandBuffer, label: &str) {
        let label = CString::new(label).unwrap_or_default();
        let label = vk::DebugUtilsLabelEXT::default().label_name(&label);
        unsafe {
            self.dbg_utils.cmd_begin_debug_utils_label(cbuff, &label);
        }
    }
    pub fn end_debug_marker(&self, &cbuff: &vk::CommandBuffer) {
        unsafe { self.dbg_utils.cmd_end_debug_utils_label(cbuff) }
    }
    pub fn create_scoped_marker<'buff>(
        self: &Arc<Self>,
        command_buffer: &'buff vk::CommandBuffer,
        label: &str,
    ) -> ScopedMarker<'buff> {
        let label = CString::new(label).unwrap_or_default();
        let label = vk::DebugUtilsLabelEXT::default().label_name(&label);
        unsafe {
            self.dbg_utils
                .cmd_begin_debug_utils_label(*command_buffer, &label);
        }

        ScopedMarker {
            command_buffer,
            device: Arc::clone(self),
        }
    }

    pub fn create_image(
        &self,
        info: &vk::ImageCreateInfo,
        usage: MemoryLocation,
    ) -> Result<(vk::Image, Allocation)> {
        let image = unsafe { self.device.create_image(info, None)? };
        let memory_reqs = unsafe { self.get_image_memory_requirements(image) };
        let linear = info.tiling == vk::ImageTiling::LINEAR;
        let memory =
            self.alloc_memory(memory_reqs, usage, linear, AllocationResource::Image(image))?;
        unsafe { self.bind_image_memory(image, memory.memory(), memory.offset()) }?;
        Ok((image, memory))
    }

    pub fn image_transition(
        &self,
        command_buffer: &vk::CommandBuffer,
        image: &vk::Image,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    ) {
        let aspect_mask = if old_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
            || new_layout == vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL
            || new_layout == vk::ImageLayout::DEPTH_READ_ONLY_STENCIL_ATTACHMENT_OPTIMAL
        {
            vk::ImageAspectFlags::DEPTH
        } else {
            vk::ImageAspectFlags::COLOR
        };
        let subresource = vk::ImageSubresourceRange {
            aspect_mask,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        };
        let src_stage_mask = get_pipeline_stage_flags(old_layout);
        let dst_stage_mask = get_pipeline_stage_flags(new_layout);
        let src_access_mask = get_access_flags(old_layout);
        let dst_access_mask = get_access_flags(new_layout);
        let image_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(src_stage_mask)
            .dst_stage_mask(dst_stage_mask)
            .src_access_mask(src_access_mask)
            .dst_access_mask(dst_access_mask)
            .subresource_range(subresource)
            .image(*image)
            .old_layout(old_layout)
            .new_layout(new_layout);
        let dependency_info = vk::DependencyInfo::default()
            .image_memory_barriers(std::slice::from_ref(&image_barrier));
        unsafe { self.cmd_pipeline_barrier2(*command_buffer, &dependency_info) }
    }

    pub fn destroy_image(&self, image: vk::Image, memory: Allocation) {
        unsafe { self.device.destroy_image(image, None) };
        self.dealloc_memory(memory);
    }

    pub fn create_2d_view(
        &self,
        image: &vk::Image,
        format: vk::Format,
        base_mip_level: u32,
    ) -> VkResult<vk::ImageView> {
        let view = unsafe {
            self.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .image(*image)
                    .format(format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(base_mip_level)
                            .level_count(vk::REMAINING_MIP_LEVELS)
                            .base_array_layer(0)
                            .layer_count(vk::REMAINING_ARRAY_LAYERS),
                    ),
                None,
            )?
        };
        Ok(view)
    }

    pub fn start_command_buffer(&self) -> Result<vk::CommandBuffer> {
        let command_buffer = unsafe {
            self.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY),
            )?[0]
        };

        unsafe {
            self.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        Ok(command_buffer)
    }

    pub fn end_command_buffer(&self, &cbuff: &vk::CommandBuffer) -> VkResult<()> {
        unsafe { self.device.end_command_buffer(cbuff) }
    }

    pub fn one_time_submit(
        self: &Arc<Self>,
        queue: &vk::Queue,
        callbk: impl FnOnce(&Arc<Self>, vk::CommandBuffer) -> anyhow::Result<()>,
    ) -> Result<()> {
        let fence = unsafe { self.create_fence(&vk::FenceCreateInfo::default(), None)? };
        let command_buffer = unsafe {
            self.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(self.command_pool)
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY),
            )?[0]
        };

        unsafe {
            self.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;

            callbk(self, command_buffer)?;

            self.end_command_buffer(&command_buffer)?;

            let submit_info =
                vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));

            self.queue_submit(*queue, &[submit_info], fence)?;
            self.wait_for_fences(&[fence], true, u64::MAX)?;

            self.destroy_fence(fence, None);
            self.free_command_buffers(self.command_pool, &[command_buffer]);
        }

        Ok(())
    }

    pub fn alloc_memory(
        &self,
        memory_reqs: vk::MemoryRequirements,
        usage: MemoryLocation,
        linear: bool,
        resource: AllocationResource,
    ) -> Result<Allocation, gpu_allocator::AllocationError> {
        let mut allocator = self.allocator.lock();
        let allocation_scheme = match resource {
            AllocationResource::Image(image) => AllocationScheme::DedicatedImage(image),
            AllocationResource::Buffer(buffer) => AllocationScheme::DedicatedBuffer(buffer),
            AllocationResource::None => AllocationScheme::GpuAllocatorManaged,
        };
        allocator.allocate(&AllocationCreateDesc {
            name: &format!("Memory: {usage:?}"),
            requirements: memory_reqs,
            location: usage,
            linear,
            allocation_scheme,
        })
    }

    pub fn dealloc_memory(&self, memory: Allocation) {
        let mut allocator = self.allocator.lock();
        let _ = allocator.free(memory);
    }

    pub fn blit_image(
        &self,
        command_buffer: &vk::CommandBuffer,
        src_image: &vk::Image,
        src_extent: vk::Extent2D,
        src_orig_layout: vk::ImageLayout,
        dst_image: &vk::Image,
        dst_extent: vk::Extent2D,
        dst_orig_layout: vk::ImageLayout,
    ) {
        self.image_transition(
            command_buffer,
            src_image,
            src_orig_layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        self.image_transition(
            command_buffer,
            dst_image,
            dst_orig_layout,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        let src_offsets = [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: src_extent.width as _,
                y: src_extent.height as _,
                z: 1,
            },
        ];
        let dst_offsets = [
            vk::Offset3D { x: 0, y: 0, z: 0 },
            vk::Offset3D {
                x: dst_extent.width as _,
                y: dst_extent.height as _,
                z: 1,
            },
        ];
        let subresource_layer = vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_array_layer: 0,
            layer_count: 1,
            mip_level: 0,
        };
        let regions = [vk::ImageBlit2::default()
            .src_offsets(src_offsets)
            .dst_offsets(dst_offsets)
            .src_subresource(subresource_layer)
            .dst_subresource(subresource_layer)];
        let blit_info = vk::BlitImageInfo2::default()
            .src_image(*src_image)
            .src_image_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .dst_image(*dst_image)
            .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .regions(&regions)
            .filter(vk::Filter::LINEAR);
        unsafe { self.cmd_blit_image2(*command_buffer, &blit_info) };

        self.image_transition(
            command_buffer,
            src_image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            src_orig_layout,
        );
        self.image_transition(
            command_buffer,
            dst_image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            match dst_orig_layout {
                vk::ImageLayout::UNDEFINED => vk::ImageLayout::GENERAL,
                _ => dst_orig_layout,
            },
        );
    }

    pub fn capture_image_data(
        self: &Arc<Self>,
        queue: &vk::Queue,
        src_image: &vk::Image,
        extent: vk::Extent2D,
        callback: impl FnOnce(ManagedImage),
    ) -> Result<()> {
        let dst_image = ManagedImage::new(
            self,
            &vk::ImageCreateInfo::default()
                .extent(vk::Extent3D {
                    width: align_to(extent.width, 2),
                    height: align_to(extent.height, 2),
                    depth: 1,
                })
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_UNORM)
                .usage(vk::ImageUsageFlags::TRANSFER_DST)
                .samples(vk::SampleCountFlags::TYPE_1)
                .mip_levels(1)
                .array_layers(1)
                .tiling(vk::ImageTiling::LINEAR),
            MemoryLocation::GpuToCpu,
        )?;

        self.one_time_submit(queue, |device, command_buffer| {
            device.blit_image(
                &command_buffer,
                src_image,
                extent,
                vk::ImageLayout::PRESENT_SRC_KHR,
                &dst_image.image,
                extent,
                vk::ImageLayout::UNDEFINED,
            );
            Ok(())
        })?;

        callback(dst_image);

        Ok(())
    }

    pub fn create_buffer(
        self: &Arc<Self>,
        size: u64,
        usage: vk::BufferUsageFlags,
        memory_usage: MemoryLocation,
    ) -> Result<Buffer> {
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(size)
                    .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
                None,
            )?
        };
        let mem_requirements = unsafe { self.get_buffer_memory_requirements(buffer) };

        let memory = self.alloc_memory(
            mem_requirements,
            memory_usage,
            true,
            AllocationResource::Buffer(buffer),
        )?;
        unsafe { self.bind_buffer_memory(buffer, memory.memory(), memory.offset()) }?;

        let address = unsafe {
            self.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };

        Ok(Buffer {
            address,
            size,
            buffer,
            memory: ManuallyDrop::new(memory),
            device: self.clone(),
        })
    }

    pub fn create_buffer_typed<T>(
        self: &Arc<Self>,
        usage: vk::BufferUsageFlags,
        memory_usage: MemoryLocation,
    ) -> Result<BufferTyped<T>> {
        let byte_size = (size_of::<T>()) as vk::DeviceSize;
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(byte_size)
                    .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
                None,
            )?
        };
        let mem_requirements = unsafe { self.get_buffer_memory_requirements(buffer) };

        let memory = self.alloc_memory(
            mem_requirements,
            memory_usage,
            true,
            AllocationResource::Buffer(buffer),
        )?;
        unsafe { self.bind_buffer_memory(buffer, memory.memory(), memory.offset()) }?;

        let address = unsafe {
            self.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        self.name_object(
            buffer,
            &format!("BufferTyped<{}>", pretty_type_name::pretty_type_name::<T>()),
        );

        Ok(BufferTyped {
            address,
            buffer,
            memory: ManuallyDrop::new(memory),
            device: self.clone(),
            _marker: PhantomData,
        })
    }

    pub fn get_info(&self) -> RendererInfo {
        RendererInfo {
            device_name: self.get_device_name().unwrap().to_string(),
            device_type: self.get_device_type().to_string(),
            vendor_name: self.get_vendor_name().to_string(),
        }
    }
    pub fn get_device_name(&self) -> Result<&str, std::str::Utf8Error> {
        unsafe { CStr::from_ptr(self.device_properties.device_name.as_ptr()) }.to_str()
    }
    pub fn get_device_type(&self) -> &str {
        match self.device_properties.device_type {
            vk::PhysicalDeviceType::CPU => "CPU",
            vk::PhysicalDeviceType::INTEGRATED_GPU => "INTEGRATED_GPU",
            vk::PhysicalDeviceType::DISCRETE_GPU => "DISCRETE_GPU",
            vk::PhysicalDeviceType::VIRTUAL_GPU => "VIRTUAL_GPU",
            _ => "OTHER",
        }
    }
    pub fn get_vendor_name(&self) -> &str {
        match self.device_properties.vendor_id {
            0x1002 => "AMD",
            0x1010 => "ImgTec",
            0x10DE => "NVIDIA Corporation",
            0x13B5 => "ARM",
            0x5143 => "Qualcomm",
            0x8086 => "INTEL Corporation",
            _ => "Unknown vendor",
        }
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
        }
    }
}

pub enum AllocationResource {
    Image(vk::Image),
    Buffer(vk::Buffer),
    None,
}

pub struct Buffer {
    pub address: u64,
    pub size: u64,
    pub buffer: vk::Buffer,
    pub memory: ManuallyDrop<Allocation>,
    device: Arc<Device>,
}

impl Buffer {
    pub fn map_memory(&mut self) -> Option<&mut [u8]> {
        self.memory.mapped_slice_mut()
    }
}

impl Drop for Buffer {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            let memory = ManuallyDrop::take(&mut self.memory);
            self.device.dealloc_memory(memory);
        }
    }
}

pub struct BufferTyped<T: 'static> {
    pub address: u64,
    pub buffer: vk::Buffer,
    pub memory: ManuallyDrop<Allocation>,
    device: Arc<Device>,
    _marker: PhantomData<*mut T>,
}

impl<T> BufferTyped<T> {
    pub fn map_memory(&mut self) -> Option<&mut T> {
        self.memory
            .mapped_slice_mut()
            .map(|slice| crate::from_bytes::<T>(slice))
    }
}

impl<T> Drop for BufferTyped<T> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            let memory = ManuallyDrop::take(&mut self.memory);
            self.device.dealloc_memory(memory);
        }
    }
}

#[derive(Debug)]
pub struct RendererInfo {
    pub device_name: String,
    pub device_type: String,
    pub vendor_name: String,
}

impl std::fmt::Display for RendererInfo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Vendor name: {}", self.vendor_name)?;
        writeln!(f, "Device name: {}", self.device_name)?;
        writeln!(f, "Device type: {}", self.device_type)?;
        Ok(())
    }
}

pub struct ScopedMarker<'a> {
    command_buffer: &'a vk::CommandBuffer,
    device: Arc<Device>,
}

impl Drop for ScopedMarker<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device
                .dbg_utils
                .cmd_end_debug_utils_label(*self.command_buffer)
        };
    }
}

fn get_pipeline_stage_flags(layout: vk::ImageLayout) -> vk::PipelineStageFlags2 {
    match layout {
        vk::ImageLayout::UNDEFINED => vk::PipelineStageFlags2::TOP_OF_PIPE,
        vk::ImageLayout::PREINITIALIZED => vk::PipelineStageFlags2::HOST,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL | vk::ImageLayout::TRANSFER_SRC_OPTIMAL => {
            vk::PipelineStageFlags2::TRANSFER
        }
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => {
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
        }
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => {
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS
        }
        vk::ImageLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR => {
            vk::PipelineStageFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR
        }
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
            vk::PipelineStageFlags2::VERTEX_SHADER | vk::PipelineStageFlags2::FRAGMENT_SHADER
        }
        vk::ImageLayout::PRESENT_SRC_KHR => vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
        vk::ImageLayout::GENERAL => vk::PipelineStageFlags2::ALL_COMMANDS,
        _ => panic!("Unknown layout for pipeline stage: {layout:?}!"),
    }
}

fn get_access_flags(layout: vk::ImageLayout) -> vk::AccessFlags2 {
    match layout {
        vk::ImageLayout::UNDEFINED | vk::ImageLayout::PRESENT_SRC_KHR => vk::AccessFlags2::empty(),
        vk::ImageLayout::PREINITIALIZED => vk::AccessFlags2::HOST_WRITE,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL => {
            vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
        }
        vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL => {
            vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ
                | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
        }
        vk::ImageLayout::FRAGMENT_SHADING_RATE_ATTACHMENT_OPTIMAL_KHR => {
            vk::AccessFlags2::FRAGMENT_SHADING_RATE_ATTACHMENT_READ_KHR
        }
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL => {
            vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::INPUT_ATTACHMENT_READ
        }
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL => vk::AccessFlags2::TRANSFER_READ,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL => vk::AccessFlags2::TRANSFER_WRITE,
        vk::ImageLayout::GENERAL => vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
        _ => panic!("Unknown layout for access mask: {layout:?}"),
    }
}
