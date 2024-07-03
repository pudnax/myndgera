use anyhow::{Context, Result};
use std::sync::Arc;

use ash::{khr, prelude::VkResult, vk};

use crate::ManagedImage;

pub struct Device {
    pub physical_device: vk::PhysicalDevice,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub main_queue_family_idx: u32,
    pub transfer_queue_family_idx: u32,
    pub device: Arc<ash::Device>,
    pub ext: Arc<DeviceExt>,
}

pub struct DeviceExt {
    pub dynamic_rendering: khr::dynamic_rendering::Device,
}

impl std::ops::Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl Device {
    pub fn get_buffer_address<T>(&self, buffer: vk::Buffer) -> u64 {
        unsafe {
            self.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        }
    }

    pub fn alloc_memory(
        &self,
        memory_reqs: vk::MemoryRequirements,
        usage: vk::MemoryPropertyFlags,
    ) -> anyhow::Result<vk::DeviceMemory> {
        let memory_type_index =
            find_memory_type_index(&self.memory_properties, memory_reqs.memory_type_bits, usage)
                .context("Failed to find suitable memory")?;

        let mut alloc_flag =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let allocate_info = vk::MemoryAllocateInfo::default()
            .allocation_size(memory_reqs.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut alloc_flag);
        Ok(unsafe { self.allocate_memory(&allocate_info, None)? })
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
        let subresource_range = vk::ImageSubresourceRange {
            aspect_mask: vk::ImageAspectFlags::COLOR,
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        };
        let src_barrier = vk::ImageMemoryBarrier2::default()
            .subresource_range(subresource_range)
            .image(*src_image)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
            .old_layout(src_orig_layout)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        let dst_barrier = vk::ImageMemoryBarrier2::default()
            .subresource_range(subresource_range)
            .image(*dst_image)
            .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(dst_orig_layout)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL);
        let image_memory_barriers = &[src_barrier, dst_barrier];
        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(image_memory_barriers);
        unsafe { self.cmd_pipeline_barrier2(*command_buffer, &dependency_info) };

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
            .filter(vk::Filter::NEAREST);
        unsafe { self.cmd_blit_image2(*command_buffer, &blit_info) };

        let src_barrier = src_barrier
            .src_access_mask(vk::AccessFlags2::MEMORY_READ)
            .old_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .new_layout(src_orig_layout);
        let dst_barrier = dst_barrier
            .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(match dst_orig_layout {
                vk::ImageLayout::UNDEFINED => vk::ImageLayout::GENERAL,
                _ => dst_orig_layout,
            });
        let image_memory_barriers = &[src_barrier, dst_barrier];
        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(image_memory_barriers);
        unsafe { self.cmd_pipeline_barrier2(*command_buffer, &dependency_info) };
    }

    pub fn capture_image_data(
        &self,
        queue: &vk::Queue,
        src_image: &vk::Image,
        extent: vk::Extent2D,
        callback: impl FnOnce(ManagedImage) + Send + 'static,
    ) -> Result<()> {
        let now = std::time::Instant::now();
        let dst_image = ManagedImage::new(
            self,
            &vk::ImageCreateInfo::default()
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_SRGB)
                .usage(vk::ImageUsageFlags::TRANSFER_DST)
                .samples(vk::SampleCountFlags::TYPE_1)
                .mip_levels(1)
                .array_layers(1)
                .tiling(vk::ImageTiling::LINEAR),
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        )?;

        let fence = unsafe { self.create_fence(&vk::FenceCreateInfo::default(), None)? };

        let command_pool = unsafe {
            self.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .queue_family_index(self.main_queue_family_idx),
                None,
            )?
        };
        let command_buffer = unsafe {
            self.allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(command_pool)
                    .command_buffer_count(1)
                    .level(vk::CommandBufferLevel::PRIMARY),
            )?[0]
        };

        unsafe {
            self.begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?
        };

        self.blit_image(
            &command_buffer,
            src_image,
            extent,
            vk::ImageLayout::PRESENT_SRC_KHR,
            &dst_image.image,
            extent,
            vk::ImageLayout::UNDEFINED,
        );

        unsafe { self.end_command_buffer(command_buffer) }?;

        let submit_info =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
        unsafe { self.queue_submit(*queue, &[submit_info], fence)? };
        unsafe { self.wait_for_fences(&[fence], true, u64::MAX)? };

        println!("Blit image: {:?}", now.elapsed());

        callback(dst_image);

        unsafe {
            self.destroy_fence(fence, None);
            self.free_command_buffers(command_pool, &[command_buffer]);
            self.destroy_command_pool(command_pool, None);
        }

        Ok(())
    }

    pub fn create_host_buffer<T>(
        &self,
        usage: vk::BufferUsageFlags,
        memory_prop_flags: vk::MemoryPropertyFlags,
    ) -> VkResult<HostBuffer<T>> {
        let byte_size = (size_of::<T>()) as vk::DeviceSize;
        let buffer = unsafe {
            self.device.create_buffer(
                &vk::BufferCreateInfo::default()
                    .size(byte_size)
                    .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS),
                None,
            )?
        };
        let requirements = unsafe { self.get_buffer_memory_requirements(buffer) };
        let memory_type_index = find_memory_type_index(
            &self.memory_properties,
            requirements.memory_type_bits,
            memory_prop_flags | vk::MemoryPropertyFlags::HOST_VISIBLE,
        )
        .expect("Failed to find suitable memory index for buffer memory");

        let mut alloc_flag =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let alloc_info = vk::MemoryAllocateInfo::default()
            .allocation_size(requirements.size)
            .memory_type_index(memory_type_index)
            .push_next(&mut alloc_flag);
        let memory = unsafe { self.device.allocate_memory(&alloc_info, None) }?;
        unsafe { self.bind_buffer_memory(buffer, memory, 0) }?;

        let address = unsafe {
            self.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };

        let ptr = unsafe { self.map_memory(memory, 0, byte_size, vk::MemoryMapFlags::empty())? };
        let ptr = unsafe { &mut *ptr.cast::<T>() };

        Ok(HostBuffer {
            address,
            buffer,
            memory,
            ptr,
            device: self.device.clone(),
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
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

pub struct HostBuffer<T: 'static> {
    pub address: u64,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub ptr: &'static mut T,
    device: Arc<ash::Device>,
}

impl<T> std::ops::Deref for HostBuffer<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.ptr
    }
}

impl<T> std::ops::DerefMut for HostBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.ptr
    }
}

impl<T> Drop for HostBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}
