use anyhow::Result;
use ash::vk;
use glam::{uvec2, vec2, UVec2, Vec2};
use gpu_alloc::UsageFlags;

use crate::{dispatch_optimal, AppState, ComputeHandle, FrameGuard, ManagedImage, RenderContext};

#[derive(Clone, Copy, Debug)]
pub struct BloomParams {
    target_image_sampled: u32,
    target_image_storage: u32,
    target_current_layout: vk::ImageLayout,
    strength: f32,
    width: f32,
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
    upsample_pass: ComputeHandle,
    texture_info: vk::ImageCreateInfo<'static>,
    miplevel_count: u32,
    accum_texture: ManagedImage,
    accum_view: vk::ImageView,
    accum_texture_sampled_idx: u32,
    accum_texture_storage_idx: u32,
}

impl Bloom {
    pub fn new(ctx: &RenderContext, state: &mut AppState) -> Result<Self> {
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .size(std::mem::size_of::<DownsamplePC>() as u32);
        let desc_layouts = [
            state.texture_arena.sampled_set_layout,
            state.texture_arena.storage_set_layout,
        ];
        let downsample_pass = state.pipeline_arena.create_compute_pipeline(
            "src/passes/bloom/bloom_upsample.vert",
            &[push_constant_range],
            &desc_layouts,
        )?;
        let upsample_pass = state.pipeline_arena.create_compute_pipeline(
            "src/passes/bloom/bloom_upsample.frag",
            &[push_constant_range],
            &desc_layouts,
        )?;
        let miplevel_count = 8;
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
            .initial_layout(vk::ImageLayout::GENERAL);
        let accum_texture =
            ManagedImage::new(&ctx.device, &texture_info, UsageFlags::FAST_DEVICE_ACCESS)?;
        let accum_view = unsafe {
            ctx.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .image(accum_texture.image)
                    .format(accum_texture.format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(texture_info.mip_levels)
                            .base_array_layer(0)
                            .layer_count(texture_info.array_layers),
                    ),
                None,
            )?
        };
        let accum_texture_sampled_idx =
            state
                .texture_arena
                .push_sampled_image(accum_texture.image, accum_view, None, None);
        let accum_texture_storage_idx =
            state
                .texture_arena
                .push_sampled_image(accum_texture.image, accum_view, None, None);

        Ok(Self {
            downsample_pass,
            upsample_pass,
            miplevel_count,
            texture_info,
            accum_texture,
            accum_view,
            accum_texture_sampled_idx,
            accum_texture_storage_idx,
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
        self.accum_view = unsafe {
            ctx.device.create_image_view(
                &vk::ImageViewCreateInfo::default()
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .image(self.accum_texture.image)
                    .format(self.accum_texture.format)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .base_mip_level(0)
                            .level_count(self.texture_info.mip_levels)
                            .base_array_layer(0)
                            .layer_count(self.texture_info.array_layers),
                    ),
                None,
            )?
        };
        state
            .texture_arena
            .update_sampled_image(self.accum_texture_sampled_idx, &self.accum_view);
        state
            .texture_arena
            .update_storage_image(self.accum_texture_sampled_idx, &self.accum_view);
        Ok(())
    }

    pub fn apply(
        &self,
        ctx: &RenderContext,
        state: &mut AppState,
        frame: &FrameGuard,
        params: BloomParams,
    ) {
        let screen_extent = ctx.swapchain.extent;
        let source_image = state.texture_arena.images[params.target_image_sampled as usize];
        ctx.device.image_transition(
            frame.command_buffer(),
            &source_image,
            params.target_current_layout,
            vk::ImageLayout::GENERAL,
        );

        let downsample_pipeline = state.pipeline_arena.get_pipeline(self.downsample_pass);
        frame.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            downsample_pipeline.layout,
            &[
                state.texture_arena.sampled_set,
                state.texture_arena.storage_set,
            ],
        );

        for i in 0..self.miplevel_count {
            let image_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER);
            let image_barriers = [
                image_barrier.image(source_image),
                image_barrier.image(self.accum_texture.image),
            ];
            let dependency_info =
                vk::DependencyInfo::default().image_memory_barriers(&image_barriers);
            unsafe {
                ctx.device
                    .cmd_pipeline_barrier2(*frame.command_buffer(), &dependency_info)
            };

            let target_dims = uvec2(screen_extent.width, screen_extent.height) >> (i + 1);

            let (source_lod, source_texture, source_dims);
            if i == 0 {
                source_lod = 0;
                source_texture = params.target_image_sampled;
                source_dims = uvec2(screen_extent.width, screen_extent.height);
            } else {
                source_lod = i - 1;
                source_texture = self.accum_texture_sampled_idx;
                source_dims = uvec2(screen_extent.width, screen_extent.height) >> i;
            }

            frame.push_constant(
                downsample_pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                &[DownsamplePC {
                    source_sampled_img_idx: source_texture,
                    target_storage_img_idx: self.accum_texture_storage_idx,
                    source_dims,
                    target_dims,
                    source_lod,
                }],
            );

            frame.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, downsample_pipeline);
            frame.dispatch(
                dispatch_optimal(screen_extent.width, 8),
                dispatch_optimal(screen_extent.height, 8),
                1,
            );
        }

        let upsample_pipeline = state.pipeline_arena.get_pipeline(self.upsample_pass);
        frame.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            upsample_pipeline.layout,
            &[
                state.texture_arena.sampled_set,
                state.texture_arena.storage_set,
            ],
        );

        for i in (0..self.miplevel_count).rev() {
            let image_barrier = vk::ImageMemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER);
            let image_barriers = [
                image_barrier.image(source_image),
                image_barrier.image(self.accum_texture.image),
            ];
            let dependency_info =
                vk::DependencyInfo::default().image_memory_barriers(&image_barriers);
            unsafe {
                ctx.device
                    .cmd_pipeline_barrier2(*frame.command_buffer(), &dependency_info)
            };

            let source_dims = uvec2(screen_extent.width, screen_extent.height) >> (i + 1);
            let (target_lod, target_texture, target_dims);
            if i == 0 {
                target_lod = 0;
                target_texture = params.target_image_sampled;
                target_dims = uvec2(screen_extent.width, screen_extent.height);
            } else {
                target_lod = i - 1;
                target_texture = self.accum_texture_sampled_idx;
                target_dims = uvec2(screen_extent.width, screen_extent.height) >> i;
            }

            frame.push_constant(
                downsample_pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                &[UpsamplePC {
                    source_sampled_img_idx: self.accum_texture_sampled_idx,
                    target_sampled_img_idx: self.accum_texture_sampled_idx,
                    target_storage_img_idx: self.accum_texture_storage_idx,
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
            frame.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, upsample_pipeline);
            frame.dispatch(
                dispatch_optimal(screen_extent.width, 16),
                dispatch_optimal(screen_extent.height, 16),
                1,
            );
        }
        ctx.device.image_transition(
            frame.command_buffer(),
            &source_image,
            vk::ImageLayout::GENERAL,
            params.target_current_layout,
        );
    }
}
