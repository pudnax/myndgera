use anyhow::Result;
use gpu_alloc::{GpuAllocator, MemoryBlock, Request, UsageFlags};
use gpu_alloc_ash::AshMemoryDevice;
use parking_lot::Mutex;
use std::{mem::ManuallyDrop, sync::Arc};

use ash::{
    khr,
    vk::{self, DeviceMemory},
};

use crate::ManagedImage;

pub struct Device {
    pub physical_device: vk::PhysicalDevice,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub main_queue_family_idx: u32,
    pub transfer_queue_family_idx: u32,
    pub allocator: Arc<Mutex<GpuAllocator<DeviceMemory>>>,
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
        usage: UsageFlags,
    ) -> Result<gpu_alloc::MemoryBlock<DeviceMemory>, gpu_alloc::AllocationError> {
        let mut allocator = self.allocator.lock();
        let memory_block = unsafe {
            allocator.alloc(
                AshMemoryDevice::wrap(&self.device),
                Request {
                    size: memory_reqs.size,
                    align_mask: memory_reqs.alignment - 1,
                    usage: usage | UsageFlags::DEVICE_ADDRESS,
                    memory_types: memory_reqs.memory_type_bits,
                },
            )
        };
        memory_block
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
            UsageFlags::DOWNLOAD,
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
        memory_usage: gpu_alloc::UsageFlags,
    ) -> Result<HostBuffer<T>> {
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

        let mut memory =
            self.alloc_memory(mem_requirements, memory_usage | UsageFlags::HOST_ACCESS)?;
        unsafe { self.bind_buffer_memory(buffer, *memory.memory(), memory.offset()) }?;

        let address = unsafe {
            self.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };

        let ptr = unsafe {
            memory.map(
                AshMemoryDevice::wrap(&self.device),
                memory.offset(),
                memory.size() as usize,
            )?
        };
        let ptr = unsafe { &mut *ptr.as_ptr().cast::<T>() };

        Ok(HostBuffer {
            address,
            buffer,
            memory: ManuallyDrop::new(memory),
            data: ptr,
            device: self.device.clone(),
            allocator: self.allocator.clone(),
        })
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            {
                let mut allocator = self.allocator.lock();
                allocator.cleanup(AshMemoryDevice::wrap(&self.device));
            }
            self.device.destroy_device(None);
        }
    }
}

pub struct HostBuffer<T: 'static> {
    pub address: u64,
    pub buffer: vk::Buffer,
    pub memory: ManuallyDrop<MemoryBlock<DeviceMemory>>,
    pub data: &'static mut T,
    device: Arc<ash::Device>,
    allocator: Arc<Mutex<GpuAllocator<DeviceMemory>>>,
}

impl<T> std::ops::Deref for HostBuffer<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<T> std::ops::DerefMut for HostBuffer<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}

impl<T> Drop for HostBuffer<T> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_buffer(self.buffer, None);
            {
                let mut allocator = self.allocator.lock();
                let memory = ManuallyDrop::take(&mut self.memory);
                allocator.dealloc(AshMemoryDevice::wrap(&self.device), memory);
            }
        }
    }
}
