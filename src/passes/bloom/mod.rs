use std::{mem, sync::Arc};

use anyhow::Result;
use ash::vk;
use glam::{uvec2, UVec2};
use gpu_alloc::UsageFlags;

use crate::{
    dispatch_optimal, AppState, ComputeHandle, Device, FrameGuard, ManagedImage, RenderContext,
    COLOR_SUBRESOURCE_MASK,
};

#[derive(Clone, Copy, Debug)]
pub struct BloomParams<'a> {
    pub target_image: &'a vk::Image,
    pub target_image_sampled: u32,
    pub target_image_storage: u32,
    pub target_current_layout: vk::ImageLayout,
    pub strength: f32,
    pub width: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct DownsamplePC {
    source_sampled_img_idx: u32,
    target_storage_img_idx: u32,
    source_dims: UVec2,
    target_dims: UVec2,
    source_lod: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct UpsamplePC {
    source_sampled_img_idx: u32,
    target_sampled_img_idx: u32,
    target_storage_img_idx: u32,
    source_dims: UVec2,
    target_dims: UVec2,
    width: f32,
    strength: f32,
    source_lod: u32,
    target_lod: u32,
    num_passes: u32,
    is_final_pass: u32,
}

pub struct Bloom {
    downsample_pass: ComputeHandle,
    downsample_low_pass: ComputeHandle,
    upsample_pass: ComputeHandle,
    texture_info: vk::ImageCreateInfo<'static>,
    miplevel_count: u32,
    accum_texture: ManagedImage,
    accum_views: Vec<vk::ImageView>,
    accum_texture_sampled_idx: u32,
    accum_texture_storage_idx: Vec<u32>,
    device: Arc<Device>,
}

impl Drop for Bloom {
    fn drop(&mut self) {
        unsafe {
            self.accum_views
                .iter()
                .for_each(|&view| self.device.destroy_image_view(view, None))
        };
    }
}

impl Bloom {
    pub fn new(ctx: &RenderContext, state: &mut AppState) -> Result<Self> {
        let miplevel_count = 6;
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(mem::size_of::<DownsamplePC>() as u32);
        let desc_layouts = [
            state.texture_arena.sampled_set_layout,
            state.texture_arena.storage_set_layout,
        ];
        let downsample_pass = state.pipeline_arena.create_compute_pipeline(
            "src/passes/bloom/bloom_downsample.comp",
            &[push_constant_range],
            &desc_layouts,
        )?;
        let downsample_low_pass = state.pipeline_arena.create_compute_pipeline(
            "src/passes/bloom/bloom_downsample_low.comp",
            &[push_constant_range],
            &desc_layouts,
        )?;
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(mem::size_of::<UpsamplePC>() as u32);
        let upsample_pass = state.pipeline_arena.create_compute_pipeline(
            "src/passes/bloom/bloom_upsample.comp",
            &[push_constant_range],
            &desc_layouts,
        )?;
        let texture_info = vk::ImageCreateInfo::default()
            .extent(vk::Extent3D {
                width: ctx.swapchain.extent.width / 2,
                height: ctx.swapchain.extent.height / 2,
                depth: 1,
            })
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::B10G11R11_UFLOAT_PACK32)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
            .samples(vk::SampleCountFlags::TYPE_1)
            .mip_levels(miplevel_count)
            .array_layers(1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let accum_texture =
            ManagedImage::new(&ctx.device, &texture_info, UsageFlags::FAST_DEVICE_ACCESS)?;

        let mut accum_views = vec![];
        let mut accum_texture_storage_idx = vec![];
        for i in 0..miplevel_count {
            let view = ctx
                .device
                .create_2d_view(&accum_texture.image, texture_info.format, i)?;

            accum_texture_storage_idx.push(state.texture_arena.push_storage_image(
                accum_texture.image,
                view,
                None,
                None,
            ));
            accum_views.push(view);
        }
        let accum_texture_sampled_idx =
            state
                .texture_arena
                .push_sampled_image(accum_texture.image, accum_views[0], None, None);
        ctx.device.name_object(accum_texture.image, "Bloom Texture");

        Ok(Self {
            downsample_pass,
            downsample_low_pass,
            upsample_pass,
            miplevel_count,
            texture_info,
            accum_texture,
            accum_views,
            accum_texture_sampled_idx,
            accum_texture_storage_idx,
            device: ctx.device.clone(),
        })
    }

    pub fn resize(&mut self, ctx: &RenderContext, state: &mut AppState) -> Result<()> {
        self.texture_info.extent.width = ctx.swapchain.extent.width / 2;
        self.texture_info.extent.height = ctx.swapchain.extent.height / 2;
        self.accum_texture = ManagedImage::new(
            &ctx.device,
            &self.texture_info,
            UsageFlags::FAST_DEVICE_ACCESS,
        )?;
        ctx.device
            .name_object(self.accum_texture.image, "Bloom Texture");
        unsafe {
            self.accum_views
                .iter()
                .for_each(|&view| self.device.destroy_image_view(view, None))
        };
        for (i, view) in self.accum_views.iter_mut().enumerate() {
            let new_view = self.device.create_2d_view(
                &self.accum_texture.image,
                self.texture_info.format,
                i as u32,
            )?;
            state
                .texture_arena
                .update_storage_image(self.accum_texture_storage_idx[i], &new_view);
            *view = new_view;
        }
        state
            .texture_arena
            .update_sampled_image(self.accum_texture_sampled_idx, &self.accum_views[0]);
        Ok(())
    }

    pub fn apply(
        &self,
        ctx: &RenderContext,
        state: &mut AppState,
        frame: &FrameGuard,
        params: BloomParams,
    ) {
        let _marker = self
            .device
            .create_scoped_marker(frame.command_buffer(), "Bloom Pass");
        let screen_extent = ctx.swapchain.extent;
        let source_image = params.target_image;

        ctx.device.image_transition(
            frame.command_buffer(),
            source_image,
            params.target_current_layout,
            vk::ImageLayout::GENERAL,
        );
        ctx.device.image_transition(
            frame.command_buffer(),
            &self.accum_texture.image,
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::GENERAL,
        );

        let image_barriers = {
            let image_barrier = vk::ImageMemoryBarrier2::default()
                .subresource_range(COLOR_SUBRESOURCE_MASK)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER);
            [
                image_barrier.image(*source_image),
                image_barrier.image(self.accum_texture.image),
            ]
        };

        for i in 0..self.miplevel_count {
            unsafe {
                ctx.device.cmd_pipeline_barrier2(
                    *frame.command_buffer(),
                    &vk::DependencyInfo::default().image_memory_barriers(&image_barriers),
                )
            };

            let target_dims = uvec2(screen_extent.width, screen_extent.height) >> (i + 1);

            let (pipeline, source_lod, source_texture, source_dims, workgroup_size);
            if i == 0 {
                pipeline = self.downsample_low_pass;
                source_lod = 0;
                source_texture = params.target_image_sampled;
                source_dims = uvec2(screen_extent.width, screen_extent.height);
                workgroup_size = 16;
            } else {
                pipeline = self.downsample_pass;
                source_lod = i - 1;
                source_texture = self.accum_texture_sampled_idx;
                source_dims = uvec2(screen_extent.width, screen_extent.height) >> i;
                workgroup_size = 8;
            }

            let downsample_pipeline = state.pipeline_arena.get_pipeline(pipeline);
            frame.bind_pipeline(vk::PipelineBindPoint::COMPUTE, downsample_pipeline);
            frame.push_constant(
                downsample_pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                &[DownsamplePC {
                    source_sampled_img_idx: source_texture,
                    target_storage_img_idx: self.accum_texture_storage_idx[i as usize],
                    source_dims,
                    target_dims,
                    source_lod,
                }],
            );
            frame.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                downsample_pipeline.layout,
                &[
                    state.texture_arena.sampled_set,
                    state.texture_arena.storage_set,
                ],
            );

            frame.dispatch(
                dispatch_optimal(target_dims.x, workgroup_size),
                dispatch_optimal(target_dims.y, workgroup_size),
                1,
            );
        }

        let upsample_pipeline = state.pipeline_arena.get_pipeline(self.upsample_pass);
        frame.bind_pipeline(vk::PipelineBindPoint::COMPUTE, upsample_pipeline);
        frame.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            upsample_pipeline.layout,
            &[
                state.texture_arena.sampled_set,
                state.texture_arena.storage_set,
            ],
        );

        for i in (0..self.miplevel_count).rev() {
            unsafe {
                ctx.device.cmd_pipeline_barrier2(
                    *frame.command_buffer(),
                    &vk::DependencyInfo::default().image_memory_barriers(&image_barriers),
                )
            };

            let source_dims = uvec2(screen_extent.width, screen_extent.height) >> (i + 1);
            let (target_lod, target_texture_sampled, target_texture_storage, target_dims);
            if i == 0 {
                target_lod = 0;
                target_texture_sampled = params.target_image_sampled;
                target_texture_storage = params.target_image_storage;
                target_dims = uvec2(screen_extent.width, screen_extent.height);
            } else {
                target_lod = i - 1;
                target_texture_sampled = self.accum_texture_sampled_idx;
                target_texture_storage = self.accum_texture_storage_idx[target_lod as usize];
                target_dims = uvec2(screen_extent.width, screen_extent.height) >> i;
            }

            frame.push_constant(
                upsample_pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                &[UpsamplePC {
                    source_sampled_img_idx: self.accum_texture_sampled_idx,
                    target_sampled_img_idx: target_texture_sampled,
                    target_storage_img_idx: target_texture_storage,
                    source_dims,
                    target_dims,
                    width: params.width,
                    strength: params.strength,
                    source_lod: i,
                    target_lod,
                    num_passes: self.miplevel_count,
                    is_final_pass: (i == 0) as u32,
                }],
            );
            frame.dispatch(
                dispatch_optimal(target_dims.x, 16),
                dispatch_optimal(target_dims.y, 16),
                1,
            );
        }
        ctx.device.image_transition(
            frame.command_buffer(),
            source_image,
            vk::ImageLayout::GENERAL,
            params.target_current_layout,
        );
    }
}
