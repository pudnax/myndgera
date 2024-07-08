use std::sync::Arc;

use anyhow::Result;
use ash::{
    prelude::VkResult,
    vk::{self, DeviceMemory},
};
use gpu_alloc::{MemoryBlock, UsageFlags};

use crate::{Device, Swapchain, COLOR_SUBRESOURCE_MASK};

pub const LINEAR_SAMPLER_IDX: usize = 0;
pub const NEAREST_SAMPLER_IDX: usize = 1;

pub const PREV_FRAME_IMAGE_IDX: usize = 0;
pub const GENERIC_IMAGE1_IDX: usize = 1;
pub const GENERIC_IMAGE2_IDX: usize = 2;
pub const DITHER_IMAGE_IDX: usize = 3;
pub const NOISE_IMAGE_IDX: usize = 4;
pub const BLUE_IMAGE_IDX: usize = 5;

pub const SCREENSIZED_IMAGE_INDICES: [usize; 3] =
    [PREV_FRAME_IMAGE_IDX, GENERIC_IMAGE1_IDX, GENERIC_IMAGE2_IDX];

const IMAGES_COUNT: u32 = 2048;
const SAMPLER_COUNT: u32 = 8;

pub struct TextureArena {
    pub images: Vec<vk::Image>,
    pub memories: Vec<Option<MemoryBlock<DeviceMemory>>>,
    pub infos: Vec<vk::ImageCreateInfo<'static>>,
    pub views: Vec<vk::ImageView>,
    pub samplers: [vk::Sampler; SAMPLER_COUNT as usize],
    descriptor_pool: vk::DescriptorPool,
    pub images_set: vk::DescriptorSet,
    pub images_set_layout: vk::DescriptorSetLayout,
    device: Arc<Device>,
}

impl TextureArena {
    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    pub fn new(device: &Arc<Device>, swapchain: &Swapchain, queue: &vk::Queue) -> Result<Self> {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(IMAGES_COUNT),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(SAMPLER_COUNT),
        ];
        let descriptor_pool = unsafe {
            device.create_descriptor_pool(
                &vk::DescriptorPoolCreateInfo::default()
                    .flags(
                        vk::DescriptorPoolCreateFlags::UPDATE_AFTER_BIND
                            | vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
                    )
                    .pool_sizes(&pool_sizes)
                    .max_sets(1),
                None,
            )?
        };

        let binding_flags = vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING;
        let binding_flags = [
            binding_flags,
            binding_flags | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
        ];
        let mut binding_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
        let sampler_set_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE)
            .descriptor_count(
                device
                    .descriptor_indexing_props
                    .max_descriptor_set_update_after_bind_samplers,
            );
        let image_set_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(1)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE)
            .descriptor_count(
                device
                    .descriptor_indexing_props
                    .max_descriptor_set_update_after_bind_sampled_images,
            );
        let bindings = [sampler_set_layout_binding, image_set_layout_binding];
        let images_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&bindings)
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut binding_flags),
                None,
            )?
        };

        let mut variable_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&[IMAGES_COUNT]);
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&images_set_layout))
            .push_next(&mut variable_info);
        let images_set = unsafe { device.allocate_descriptor_sets(&allocate_info)? }[0];

        let image_infos: [_; 3] = std::array::from_fn(|_| {
            vk::ImageCreateInfo::default()
                .extent(vk::Extent3D {
                    width: swapchain.extent.width,
                    height: swapchain.extent.height,
                    depth: 1,
                })
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_SRGB)
                .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
                .samples(vk::SampleCountFlags::TYPE_1)
                .mip_levels(1)
                .array_layers(1)
                .tiling(vk::ImageTiling::OPTIMAL)
        });

        let (images, memories) = image_infos
            .iter()
            .map(|info| {
                let (image, memory) = device.create_image(info, UsageFlags::FAST_DEVICE_ACCESS)?;
                Ok((image, Some(memory)))
            })
            .collect::<Result<(Vec<_>, Vec<_>)>>()?;

        let views = images
            .iter()
            .zip(image_infos)
            .map(|(image, info)| device.create_2d_view(image, info.format))
            .collect::<VkResult<Vec<_>>>()?;

        for (i, view) in views.iter().enumerate() {
            let image_info = vk::DescriptorImageInfo::default()
                .image_view(*view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(images_set)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .dst_binding(1)
                .image_info(std::slice::from_ref(&image_info))
                .dst_array_element(i as _);
            unsafe { device.update_descriptor_sets(&[write], &[]) };
        }

        let mut samplers = [vk::Sampler::null(); SAMPLER_COUNT as usize];
        let mut sampler_create_info = vk::SamplerCreateInfo::default()
            .min_filter(vk::Filter::LINEAR)
            .mag_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::MIRRORED_REPEAT)
            .address_mode_v(vk::SamplerAddressMode::MIRRORED_REPEAT)
            .address_mode_w(vk::SamplerAddressMode::MIRRORED_REPEAT)
            .max_lod(vk::LOD_CLAMP_NONE);
        let sampler = unsafe { device.create_sampler(&sampler_create_info, None)? };
        let descriptor_image_info = vk::DescriptorImageInfo::default().sampler(sampler);
        let mut desc_write = vk::WriteDescriptorSet::default()
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .dst_set(images_set)
            .dst_binding(0)
            .image_info(std::slice::from_ref(&descriptor_image_info))
            .dst_array_element(0);
        unsafe { device.update_descriptor_sets(&[desc_write], &[]) };
        samplers[0] = sampler;

        sampler_create_info = sampler_create_info
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST);
        let sampler = unsafe { device.create_sampler(&sampler_create_info, None)? };
        let descriptor_image_info = vk::DescriptorImageInfo::default().sampler(sampler);
        desc_write = desc_write.image_info(std::slice::from_ref(&descriptor_image_info));
        unsafe { device.update_descriptor_sets(&[desc_write], &[]) };
        samplers[1] = sampler;

        let mut texture_arena = Self {
            images,
            memories,
            infos: image_infos.to_vec(),
            views,
            samplers,
            descriptor_pool,
            images_set,
            images_set_layout,
            device: device.clone(),
        };

        texture_arena.device.name_object(
            texture_arena.images[PREV_FRAME_IMAGE_IDX],
            "Previous Frame Image",
        );
        texture_arena.device.name_object(
            texture_arena.views[PREV_FRAME_IMAGE_IDX],
            "Previous Frame Image View",
        );

        let bytes = include_bytes!("../assets/dither.dds");
        let dds = ddsfile::Dds::read(&bytes[..])?;
        let mut extent = vk::Extent3D {
            width: dds.get_width(),
            height: dds.get_height(),
            depth: 1,
        };
        let mut info = vk::ImageCreateInfo::default()
            .extent(extent)
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .samples(vk::SampleCountFlags::TYPE_1)
            .mip_levels(1)
            .array_layers(1)
            .tiling(vk::ImageTiling::OPTIMAL);
        texture_arena.push_image(device, queue, info, dds.get_data(0)?)?;
        texture_arena
            .device
            .name_object(texture_arena.images[DITHER_IMAGE_IDX], "Dither Image");
        texture_arena
            .device
            .name_object(texture_arena.views[DITHER_IMAGE_IDX], "Dither Image View");

        let bytes = include_bytes!("../assets/noise.dds");
        let dds = ddsfile::Dds::read(&bytes[..])?;
        extent.width = dds.get_width();
        extent.height = dds.get_height();
        info.extent = extent;
        texture_arena.push_image(device, queue, info, dds.get_data(0)?)?;
        texture_arena
            .device
            .name_object(texture_arena.images[NOISE_IMAGE_IDX], "Noise Image");
        texture_arena
            .device
            .name_object(texture_arena.views[NOISE_IMAGE_IDX], "Noise Image View");

        let bytes = include_bytes!("../assets/BLUE_RGBA_0.dds");
        let dds = ddsfile::Dds::read(&bytes[..])?;
        extent.width = dds.get_width();
        extent.height = dds.get_height();
        info.extent = extent;
        texture_arena.push_image(device, queue, info, dds.get_data(0)?)?;
        texture_arena
            .device
            .name_object(texture_arena.images[BLUE_IMAGE_IDX], "Blue Noise Image");
        texture_arena
            .device
            .name_object(texture_arena.views[BLUE_IMAGE_IDX], "Blue Noise Image View");

        Ok(texture_arena)
    }

    pub fn push_image(
        &mut self,
        device: &Arc<Device>,
        queue: &vk::Queue,
        info: vk::ImageCreateInfo<'static>,
        data: &[u8],
    ) -> Result<u32> {
        let (image, memory) = device.create_image(&info, UsageFlags::FAST_DEVICE_ACCESS)?;

        let mut staging = device.create_host_buffer(
            memory.size(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            UsageFlags::UPLOAD,
        )?;
        staging[..data.len()].copy_from_slice(data);

        device.one_time_submit(queue, |device, cbuff| unsafe {
            let mut image_barrier = vk::ImageMemoryBarrier2::default()
                .subresource_range(COLOR_SUBRESOURCE_MASK)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .image(image);
            let dependency_info = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&image_barrier));
            device.cmd_pipeline_barrier2(cbuff, &dependency_info);
            let regions = vk::BufferImageCopy::default()
                .image_extent(info.extent)
                .image_subresource(vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_array_layer: 0,
                    layer_count: 1,
                    mip_level: 0,
                });
            device.cmd_copy_buffer_to_image(
                cbuff,
                staging.buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[regions],
            );
            image_barrier.old_layout = vk::ImageLayout::TRANSFER_DST_OPTIMAL;
            image_barrier.new_layout = vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL;
            let dependency_info = vk::DependencyInfo::default()
                .image_memory_barriers(std::slice::from_ref(&image_barrier));
            device.cmd_pipeline_barrier2(cbuff, &dependency_info);
        })?;

        let view = self.device.create_2d_view(&image, info.format)?;
        let idx = self.images.len() as u32;

        let image_info = vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.images_set)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .dst_binding(1)
            .image_info(std::slice::from_ref(&image_info))
            .dst_array_element(idx);
        unsafe { device.update_descriptor_sets(&[write], &[]) };

        self.images.push(image);
        self.infos.push(info);
        self.memories.push(Some(memory));
        self.views.push(view);

        Ok(idx)
    }

    pub fn update_images(&mut self, indices: &[usize]) -> Result<()> {
        for (i, info) in indices.iter().map(|&i| (i, &self.infos[i])) {
            let (image, memory) = self
                .device
                .create_image(info, UsageFlags::FAST_DEVICE_ACCESS)?;
            let view = self.device.create_2d_view(&image, info.format)?;

            let image_info = vk::DescriptorImageInfo::default()
                .image_view(view)
                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
            let write = vk::WriteDescriptorSet::default()
                .dst_set(self.images_set)
                .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
                .dst_binding(1)
                .image_info(std::slice::from_ref(&image_info))
                .dst_array_element(i as _);
            unsafe { self.device.update_descriptor_sets(&[write], &[]) };

            if let Some(old_memory) = self.memories[i].take() {
                self.device.destroy_image(self.images[i], old_memory);
            }
            unsafe { self.device.destroy_image_view(self.views[i], None) };
            self.images[i] = image;
            self.memories[i] = Some(memory);
            self.views[i] = view;
        }

        Ok(())
    }
}

impl Drop for TextureArena {
    fn drop(&mut self) {
        unsafe {
            self.images
                .iter_mut()
                .zip(self.memories.iter_mut())
                .filter_map(|(img, mem)| mem.take().map(|mem| (img, mem)))
                .for_each(|(image, memory)| {
                    self.device.destroy_image(*image, memory);
                });

            self.views
                .iter()
                .for_each(|&view| self.device.destroy_image_view(view, None));
            self.samplers
                .iter()
                .for_each(|&sampler| self.device.destroy_sampler(sampler, None));
            let _ = self
                .device
                .free_descriptor_sets(self.descriptor_pool, &[self.images_set]);
            self.device
                .destroy_descriptor_set_layout(self.images_set_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
