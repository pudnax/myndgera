use std::{mem::ManuallyDrop, sync::Arc};

use anyhow::Result;
use ash::vk::{self, DeviceMemory, Handle};
use gpu_alloc::{MapError, MemoryBlock, UsageFlags};
use gpu_alloc_ash::AshMemoryDevice;

use crate::align_to;

use super::{Device, Swapchain};

const SAMPLER_SET: u32 = 0;
const IMAGE_SET: u32 = 1;

const LINEAR_SAMPLER_IDX: u32 = 0;
const NEAREST_SAMPLER_IDX: u32 = 1;

pub const DUMMY_IMAGE_IDX: usize = 0;
pub const PREV_FRAME_IDX: usize = 1;
pub const DITHER_IMAGE_IDX: usize = 2;
pub const NOISE_IMAGE_IDX: usize = 3;
pub const BLUE_IMAGE_IDX: usize = 4;

pub const SCREENSIZED_IMAGE_INDICES: [usize; 1] = [PREV_FRAME_IDX];

const IMAGES_COUNT: u32 = 2048;
const STORAGE_COUNT: u32 = 2048;
const SAMPLER_COUNT: u32 = 8;

#[derive(Debug, Clone, Copy)]
pub struct ImageDimensions {
    pub width: usize,
    pub height: usize,
    pub padded_bytes_per_row: usize,
    pub unpadded_bytes_per_row: usize,
}

impl ImageDimensions {
    pub fn new(width: usize, height: usize, alignment: u64) -> Self {
        let channel_width = std::mem::size_of::<[u8; 4]>();
        let unpadded_bytes_per_row = width * channel_width;
        let padded_bytes_per_row = align_to(unpadded_bytes_per_row, alignment as usize);
        Self {
            width,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row,
        }
    }
}

pub struct ManagedImage {
    pub image: vk::Image,
    pub memory: ManuallyDrop<MemoryBlock<DeviceMemory>>,
    pub image_dimensions: ImageDimensions,
    pub data: Option<&'static mut [u8]>,
    pub format: vk::Format,
    device: Arc<Device>,
}

impl ManagedImage {
    pub fn new(
        device: &Arc<Device>,
        info: &vk::ImageCreateInfo,
        usage: gpu_alloc::UsageFlags,
    ) -> anyhow::Result<Self> {
        let (image, memory) = device.create_image(info, usage)?;
        let memory_reqs = unsafe { device.get_image_memory_requirements(image) };
        let image_dimensions = ImageDimensions::new(
            info.extent.width as _,
            info.extent.height as _,
            memory_reqs.alignment,
        );
        Ok(Self {
            image,
            memory: ManuallyDrop::new(memory),
            image_dimensions,
            format: info.format,
            data: None,
            device: device.clone(),
        })
    }

    pub fn map_memory(&mut self) -> Result<&mut [u8], MapError> {
        if self.data.is_some() {
            return Err(MapError::AlreadyMapped);
        }
        let size = self.memory.size() as usize;
        let offset = self.memory.offset();
        let ptr = unsafe {
            self.memory
                .map(AshMemoryDevice::wrap(&self.device), offset, size)?
        };

        let data = unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr().cast(), size) };
        self.data = Some(data);

        Ok(self.data.as_mut().unwrap())
    }
}

impl Drop for ManagedImage {
    fn drop(&mut self) {
        unsafe {
            let memory = ManuallyDrop::take(&mut self.memory);
            self.device.destroy_image(self.image, memory);
        }
    }
}

pub struct TextureArena {
    pub sampled_images: Vec<vk::Image>,
    pub sampled_memories: Vec<Option<MemoryBlock<DeviceMemory>>>,
    pub sampled_infos: Vec<Option<vk::ImageCreateInfo<'static>>>,
    pub sampled_views: Vec<vk::ImageView>,
    pub sampled_set: vk::DescriptorSet,
    pub sampled_set_layout: vk::DescriptorSetLayout,

    pub storage_images: Vec<vk::Image>,
    pub storage_memory: Vec<Option<MemoryBlock<DeviceMemory>>>,
    pub storage_infos: Vec<Option<vk::ImageCreateInfo<'static>>>,
    pub storage_views: Vec<vk::ImageView>,
    pub storage_set: vk::DescriptorSet,
    pub storage_set_layout: vk::DescriptorSetLayout,

    pub external_sampled_img_idx: Vec<u32>,
    pub external_storage_img_idx: Vec<u32>,

    pub samplers: [vk::Sampler; SAMPLER_COUNT as usize],

    descriptor_pool: vk::DescriptorPool,
    device: Arc<Device>,
}

impl TextureArena {
    pub fn image_count(&self) -> usize {
        self.sampled_images.len()
    }

    pub fn new(device: &Arc<Device>, swapchain: &Swapchain, queue: &vk::Queue) -> Result<Self> {
        let pool_sizes = [
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLED_IMAGE)
                .descriptor_count(IMAGES_COUNT),
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::SAMPLER)
                .descriptor_count(STORAGE_COUNT),
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
                    .max_sets(2),
                None,
            )?
        };

        // Sampled textures
        let binding_flags = vk::DescriptorBindingFlags::PARTIALLY_BOUND;
        let binding_flags = [
            binding_flags,
            binding_flags
                | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
                | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT,
        ];
        let mut binding_flags =
            vk::DescriptorSetLayoutBindingFlagsCreateInfo::default().binding_flags(&binding_flags);
        let sampler_set_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(SAMPLER_SET)
            .descriptor_type(vk::DescriptorType::SAMPLER)
            .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE)
            .descriptor_count(SAMPLER_COUNT);
        let image_set_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(IMAGE_SET)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE)
            .descriptor_count(
                device
                    .descriptor_indexing_props
                    .max_descriptor_set_update_after_bind_sampled_images,
            );
        let bindings = [sampler_set_layout_binding, image_set_layout_binding];
        let sampled_set_layout = unsafe {
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
            .set_layouts(std::slice::from_ref(&sampled_set_layout))
            .push_next(&mut variable_info);
        let sampled_set = unsafe { device.allocate_descriptor_sets(&allocate_info)? }[0];

        // Storage textures
        let binding_flags = vk::DescriptorBindingFlags::PARTIALLY_BOUND
            | vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
            | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING
            | vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT;

        let mut binding_flags = vk::DescriptorSetLayoutBindingFlagsCreateInfo::default()
            .binding_flags(std::slice::from_ref(&binding_flags));
        let storage_set_layout_binding = vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::COMPUTE)
            .descriptor_count(
                device
                    .descriptor_indexing_props
                    .max_descriptor_set_update_after_bind_storage_images,
            );
        let bindings = [storage_set_layout_binding];
        let storage_set_layout = unsafe {
            device.create_descriptor_set_layout(
                &vk::DescriptorSetLayoutCreateInfo::default()
                    .bindings(&bindings)
                    .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                    .push_next(&mut binding_flags),
                None,
            )?
        };
        let mut variable_info = vk::DescriptorSetVariableDescriptorCountAllocateInfo::default()
            .descriptor_counts(&[STORAGE_COUNT]);
        let allocate_info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(descriptor_pool)
            .set_layouts(std::slice::from_ref(&storage_set_layout))
            .push_next(&mut variable_info);
        let storage_set = unsafe { device.allocate_descriptor_sets(&allocate_info)? }[0];

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
            .dst_set(sampled_set)
            .dst_binding(SAMPLER_SET)
            .image_info(std::slice::from_ref(&descriptor_image_info))
            .dst_array_element(LINEAR_SAMPLER_IDX);
        unsafe { device.update_descriptor_sets(&[desc_write], &[]) };
        samplers[LINEAR_SAMPLER_IDX as usize] = sampler;

        sampler_create_info = sampler_create_info
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST);
        let sampler = unsafe { device.create_sampler(&sampler_create_info, None)? };
        let descriptor_image_info = vk::DescriptorImageInfo::default().sampler(sampler);
        desc_write = desc_write
            .dst_array_element(NEAREST_SAMPLER_IDX)
            .image_info(std::slice::from_ref(&descriptor_image_info));
        unsafe { device.update_descriptor_sets(&[desc_write], &[]) };
        samplers[NEAREST_SAMPLER_IDX as usize] = sampler;

        let mut texture_arena = Self {
            sampled_images: vec![],
            sampled_memories: vec![],
            sampled_infos: vec![],
            sampled_views: vec![],
            samplers,
            descriptor_pool,
            sampled_set,
            sampled_set_layout,
            storage_images: vec![],
            storage_views: vec![],
            storage_memory: vec![],
            storage_infos: vec![],
            storage_set,
            storage_set_layout,
            external_storage_img_idx: vec![],
            external_sampled_img_idx: vec![],
            device: device.clone(),
        };

        let image_info = vk::ImageCreateInfo::default()
            .extent(vk::Extent3D {
                width: 1,
                height: 1,
                depth: 1,
            })
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_SRGB)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
            .samples(vk::SampleCountFlags::TYPE_1)
            .mip_levels(1)
            .array_layers(1)
            .tiling(vk::ImageTiling::OPTIMAL);
        texture_arena.push_image(queue, image_info, &[255, 255, 0, 255])?;

        let image_infos: [_; SCREENSIZED_IMAGE_INDICES.len()] = std::array::from_fn(|_| {
            image_info.extent(vk::Extent3D {
                width: swapchain.extent.width,
                height: swapchain.extent.height,
                depth: 1,
            })
        });

        for info in image_infos {
            let (image, memory) = device.create_image(&info, UsageFlags::FAST_DEVICE_ACCESS)?;
            let view = device.create_2d_view(&image, info.format, 0)?;
            texture_arena.push_sampled_image(image, view, Some(memory), Some(info));
        }

        let bytes = include_bytes!("../../assets/dither.dds");
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
        texture_arena.push_image(queue, info, dds.get_data(0)?)?;
        texture_arena.device.name_object(
            texture_arena.sampled_images[DITHER_IMAGE_IDX],
            "Dither Image",
        );
        texture_arena.device.name_object(
            texture_arena.sampled_views[DITHER_IMAGE_IDX],
            "Dither Image View",
        );

        let bytes = include_bytes!("../../assets/noise.dds");
        let dds = ddsfile::Dds::read(&bytes[..])?;
        extent.width = dds.get_width();
        extent.height = dds.get_height();
        info.extent = extent;
        texture_arena.push_image(queue, info, dds.get_data(0)?)?;
        texture_arena
            .device
            .name_object(texture_arena.sampled_images[NOISE_IMAGE_IDX], "Noise Image");
        texture_arena.device.name_object(
            texture_arena.sampled_views[NOISE_IMAGE_IDX],
            "Noise Image View",
        );

        let bytes = include_bytes!("../../assets/BLUE_RGBA_0.dds");
        let dds = ddsfile::Dds::read(&bytes[..])?;
        extent.width = dds.get_width();
        extent.height = dds.get_height();
        info.extent = extent;
        texture_arena.push_image(queue, info, dds.get_data(0)?)?;
        texture_arena.device.name_object(
            texture_arena.sampled_images[BLUE_IMAGE_IDX],
            "Blue Noise Image",
        );
        texture_arena.device.name_object(
            texture_arena.sampled_views[BLUE_IMAGE_IDX],
            "Blue Noise Image View",
        );

        for (image, view) in swapchain.images.iter().zip(&swapchain.views) {
            texture_arena.push_sampled_image(*image, *view, None, None);
            texture_arena.push_storage_image(*image, *view, None, None);
        }

        Ok(texture_arena)
    }

    pub fn push_sampled_image(
        &mut self,
        image: vk::Image,
        view: vk::ImageView,
        memory: Option<MemoryBlock<DeviceMemory>>,
        info: Option<vk::ImageCreateInfo<'static>>,
    ) -> u32 {
        if let (Some(_), None) | (None, Some(_)) = (&info, &memory) {
            panic!("Both image info and memory have to be present or elided")
        }
        let idx = self.sampled_images.len() as u32;
        self.sampled_images.push(image);
        self.sampled_views.push(view);
        self.sampled_memories.push(memory);
        self.sampled_infos.push(info);

        let image_info = vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.sampled_set)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .dst_binding(IMAGE_SET)
            .image_info(std::slice::from_ref(&image_info))
            .dst_array_element(idx);
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };

        if info.is_none() {
            self.external_sampled_img_idx.push(idx);
        }

        idx
    }

    pub fn push_storage_image(
        &mut self,
        image: vk::Image,
        view: vk::ImageView,
        memory: Option<MemoryBlock<DeviceMemory>>,
        info: Option<vk::ImageCreateInfo<'static>>,
    ) -> u32 {
        if let (Some(_), None) | (None, Some(_)) = (&info, &memory) {
            panic!("Both image info and memory have to be present or elided")
        }
        let idx = self.storage_images.len() as u32;
        self.storage_images.push(image);
        self.storage_views.push(view);
        self.storage_memory.push(memory);
        self.storage_infos.push(info);

        let image_info = vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::GENERAL);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.storage_set)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .image_info(std::slice::from_ref(&image_info))
            .dst_array_element(idx);
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };

        if info.is_none() {
            self.external_storage_img_idx.push(idx);
        }

        idx
    }

    pub fn push_image(
        &mut self,
        queue: &vk::Queue,
        info: vk::ImageCreateInfo<'static>,
        data: &[u8],
    ) -> Result<u32> {
        let (image, memory) = self
            .device
            .create_image(&info, UsageFlags::FAST_DEVICE_ACCESS)?;

        let mut staging = self.device.create_buffer(
            memory.size(),
            vk::BufferUsageFlags::TRANSFER_SRC,
            UsageFlags::UPLOAD | UsageFlags::TRANSIENT,
        )?;
        let mapped = staging.map_memory()?;
        mapped[..data.len()].copy_from_slice(data);

        self.device.one_time_submit(queue, |device, cbuff| unsafe {
            device.image_transition(
                &cbuff,
                &image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            );
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
            device.image_transition(
                &cbuff,
                &image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            );
            Ok(())
        })?;

        let view = self.device.create_2d_view(&image, info.format, 0)?;
        let idx = self.sampled_images.len() as u32;

        let image_info = vk::DescriptorImageInfo::default()
            .image_view(view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.sampled_set)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .dst_binding(IMAGE_SET)
            .image_info(std::slice::from_ref(&image_info))
            .dst_array_element(idx);
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };

        self.sampled_images.push(image);
        self.sampled_infos.push(Some(info));
        self.sampled_memories.push(Some(memory));
        self.sampled_views.push(view);

        Ok(idx)
    }

    pub fn update_sampled_image(&self, idx: u32, view: &vk::ImageView) {
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(*view)
            .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.sampled_set)
            .descriptor_type(vk::DescriptorType::SAMPLED_IMAGE)
            .dst_binding(IMAGE_SET)
            .image_info(std::slice::from_ref(&image_info))
            .dst_array_element(idx);
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
    }

    pub fn update_storage_image(&self, idx: u32, view: &vk::ImageView) {
        let image_info = vk::DescriptorImageInfo::default()
            .image_view(*view)
            .image_layout(vk::ImageLayout::GENERAL);
        let write = vk::WriteDescriptorSet::default()
            .dst_set(self.storage_set)
            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
            .dst_binding(0)
            .image_info(std::slice::from_ref(&image_info))
            .dst_array_element(idx);
        unsafe { self.device.update_descriptor_sets(&[write], &[]) };
    }

    pub fn update_sampled_images_by_idx(&mut self, indices: &[usize]) -> Result<()> {
        for (i, info) in indices
            .iter()
            .filter_map(|&i| self.sampled_infos[i].map(|info| (i, info)))
        {
            let (image, memory) = self
                .device
                .create_image(&info, UsageFlags::FAST_DEVICE_ACCESS)?;
            let view = self.device.create_2d_view(&image, info.format, 0)?;

            self.update_sampled_image(i as u32, &view);

            if let Some(old_memory) = self.sampled_memories[i].take() {
                self.device
                    .destroy_image(self.sampled_images[i], old_memory);
            }
            unsafe { self.device.destroy_image_view(self.sampled_views[i], None) };
            self.sampled_images[i] = image;
            self.sampled_memories[i] = Some(memory);
            self.sampled_views[i] = view;
        }

        Ok(())
    }

    pub fn update_storage_images_by_idx(&mut self, indices: &[usize]) -> Result<()> {
        for (i, info) in indices
            .iter()
            .filter_map(|&i| self.storage_infos[i].map(|info| (i, info)))
        {
            let (image, memory) = self
                .device
                .create_image(&info, UsageFlags::FAST_DEVICE_ACCESS)?;
            let view = self.device.create_2d_view(&image, info.format, 0)?;

            self.update_storage_image(i as u32, &view);

            if let Some(old_memory) = self.storage_memory[i].take() {
                self.device
                    .destroy_image(self.storage_images[i], old_memory);
            }
            unsafe { self.device.destroy_image_view(self.storage_views[i], None) };
            self.storage_images[i] = image;
            self.storage_memory[i] = Some(memory);
            self.storage_views[i] = view;
        }

        Ok(())
    }
}

impl Drop for TextureArena {
    fn drop(&mut self) {
        unsafe {
            self.sampled_views
                .iter()
                .enumerate()
                .filter(|(_, view)| !view.is_null())
                .filter(|(i, _)| !self.external_sampled_img_idx.contains(&(*i as u32)))
                .for_each(|(_, &view)| self.device.destroy_image_view(view, None));
            self.sampled_images
                .iter_mut()
                .zip(self.sampled_memories.iter_mut())
                .filter_map(|(img, mem)| mem.take().map(|mem| (img, mem)))
                .for_each(|(image, memory)| {
                    self.device.destroy_image(*image, memory);
                });

            self.storage_views
                .iter()
                .enumerate()
                .filter(|(_, view)| !view.is_null())
                .filter(|(i, _)| !self.external_storage_img_idx.contains(&(*i as u32)))
                .for_each(|(_, &view)| self.device.destroy_image_view(view, None));
            self.storage_images
                .iter_mut()
                .zip(self.storage_memory.iter_mut())
                .filter_map(|(img, mem)| mem.take().map(|mem| (img, mem)))
                .for_each(|(image, memory)| {
                    self.device.destroy_image(*image, memory);
                });

            self.samplers
                .iter()
                .for_each(|&sampler| self.device.destroy_sampler(sampler, None));
            let _ = self
                .device
                .free_descriptor_sets(self.descriptor_pool, &[self.sampled_set, self.storage_set]);
            self.device
                .destroy_descriptor_set_layout(self.sampled_set_layout, None);
            self.device
                .destroy_descriptor_set_layout(self.storage_set_layout, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
        }
    }
}
