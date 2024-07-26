use std::{mem, sync::Arc};

use anyhow::{Ok, Result};
use ash::vk;
use bytemuck::{Pod, Zeroable};
use glam::{vec3, vec4, Mat4, Vec2, Vec3, Vec4};
use gpu_allocator::MemoryLocation;
use myndgera::*;

use self::bloom::{Bloom, BloomParams};

#[repr(C)]
#[derive(Clone, Copy)]
struct GpuBuffer<T: Copy, const N: usize = 1> {
    size: u32,
    data: [T; N],
}

#[repr(C)]
#[derive(Default, Clone, Copy, Debug, Pod, Zeroable)]
struct Light {
    transform: Mat4,
    color: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Ray {
    color: Vec4,
    start: Vec3,
    end: Vec3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SpawnPC {
    num_rays: u32,
    num_bounces: u32,
    time: f32,
    line_buffer: u64,
    lights_buffer: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RasterPC {
    red_image: u32,
    green_image: u32,
    blue_image: u32,
    noise_offset: Vec2,
    camera_buffer: u64,
    line_buffer: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ResolvePC {
    target_image: u32,
    red_image: u32,
    green_image: u32,
    blue_image: u32,
    camera_buffer: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PostProcessPC {
    current_image: u32,
    hdr_sampled: u32,
    hdr_storage: u32,
}

const NUM_LIGHTS: u32 = 3;
const NUM_BOUNCES: u32 = 10;
const NUM_RAYS: u32 = 150;

struct LineRaster {
    lights_buffer: Buffer,
    lines_buffer: Buffer,
    spawn_pass: ComputeHandle,
    clear_pass: ComputeHandle,
    raster_pass: ComputeHandle,
    resolve_pass: ComputeHandle,
    postprocess_pass: RenderHandle,
    accumulate_images: Vec<usize>,
    hdr_target: ManagedImage,
    hdr_target_info: vk::ImageCreateInfo<'static>,
    hdr_target_view: vk::ImageView,
    hdr_sampled_idx: u32,
    hdr_storage_idx: u32,
    bloom: Bloom,
    device: Arc<Device>,
}

impl Drop for LineRaster {
    fn drop(&mut self) {
        unsafe { self.device.destroy_image_view(self.hdr_target_view, None) };
    }
}

impl Example for LineRaster {
    fn name() -> &'static str {
        "Line Rasteriazation"
    }
    fn init(ctx: &RenderContext, state: &mut AppState) -> Result<Self> {
        let size = mem::size_of::<GpuBuffer<Ray, { (NUM_RAYS * NUM_BOUNCES) as usize }>>();
        let lines_buffer = ctx.device.create_buffer(
            size as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            MemoryLocation::GpuOnly,
        )?;
        let size = mem::size_of::<GpuBuffer<Light, { NUM_LIGHTS as usize }>>();
        let lights_buffer = ctx.device.create_buffer(
            size as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            MemoryLocation::GpuOnly,
        )?;
        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<SpawnPC>() as _)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let spawn_pass = state.pipeline_arena.create_compute_pipeline(
            "examples/line_raster/spawn.comp",
            &[push_constant_range],
            &[],
        )?;

        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<RasterPC>() as _)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let clear_pass = state.pipeline_arena.create_compute_pipeline(
            "examples/line_raster/clear.comp",
            &[push_constant_range],
            &[state.texture_arena.storage_set_layout],
        )?;
        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<RasterPC>() as _)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let raster_pass = state.pipeline_arena.create_compute_pipeline(
            "examples/line_raster/raster.comp",
            &[push_constant_range],
            &[
                state.texture_arena.sampled_set_layout,
                state.texture_arena.storage_set_layout,
            ],
        )?;

        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<ResolvePC>() as _)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let resolve_pass = state.pipeline_arena.create_compute_pipeline(
            "examples/line_raster/resolve.comp",
            &[push_constant_range],
            &[
                state.texture_arena.sampled_set_layout,
                state.texture_arena.storage_set_layout,
            ],
        )?;

        let vertex_shader_desc = VertexShaderDesc {
            shader_path: "shaders/screen_trig.vert".into(),
            ..Default::default()
        };
        let fragment_shader_desc = FragmentShaderDesc {
            shader_path: "examples/line_raster/postprocess.frag".into(),
        };
        let fragment_output_desc = FragmentOutputDesc {
            surface_format: ctx.swapchain.format(),
            ..Default::default()
        };
        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<PostProcessPC>() as _)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
        let postprocess_pass = state.pipeline_arena.create_render_pipeline(
            &VertexInputDesc::default(),
            &vertex_shader_desc,
            &fragment_shader_desc,
            &fragment_output_desc,
            &[push_constant_range],
            &[
                state.texture_arena.sampled_set_layout,
                state.texture_arena.storage_set_layout,
            ],
        )?;

        let mut accumulate_images = vec![];
        let extent = vk::Extent3D {
            width: ctx.swapchain.extent.width,
            height: ctx.swapchain.extent.height,
            depth: 1,
        };
        for i in 0..3 {
            let device = &ctx.device;
            let info = vk::ImageCreateInfo::default()
                .extent(extent)
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R32_UINT)
                .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
                .samples(vk::SampleCountFlags::TYPE_1)
                .mip_levels(1)
                .array_layers(1)
                .tiling(vk::ImageTiling::OPTIMAL);
            let (image, memory) = ctx.device.create_image(&info, MemoryLocation::GpuOnly)?;
            let view = ctx.device.create_2d_view(&image, vk::Format::R32_UINT, 0)?;
            let idx = state
                .texture_arena
                .push_storage_image(image, view, Some(memory), Some(info));
            accumulate_images.push(idx as usize);
            device.name_object(view, &format!("Storage img view: {i}"));
            device.name_object(image, &format!("Storage img: {i}"));
        }

        let hdr_target_info = vk::ImageCreateInfo::default()
            .extent(vk::Extent3D {
                width: ctx.swapchain.extent.width,
                height: ctx.swapchain.extent.height,
                depth: 1,
            })
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::B10G11R11_UFLOAT_PACK32)
            .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::STORAGE)
            .samples(vk::SampleCountFlags::TYPE_1)
            .mip_levels(1)
            .array_layers(1)
            .tiling(vk::ImageTiling::OPTIMAL);
        let hdr_target = ManagedImage::new(&ctx.device, &hdr_target_info, MemoryLocation::GpuOnly)?;
        let hdr_target_view =
            ctx.device
                .create_2d_view(&hdr_target.image, hdr_target_info.format, 0)?;
        let hdr_sampled_idx =
            state
                .texture_arena
                .push_sampled_image(hdr_target.image, hdr_target_view, None, None);
        let hdr_storage_idx =
            state
                .texture_arena
                .push_storage_image(hdr_target.image, hdr_target_view, None, None);
        ctx.device.name_object(hdr_target.image, "Hdr Target");

        let bloom = Bloom::new(ctx, state)?;

        Ok(Self {
            lights_buffer,
            lines_buffer,
            spawn_pass,
            clear_pass,
            raster_pass,
            resolve_pass,
            postprocess_pass,
            accumulate_images,
            hdr_target,
            hdr_target_info,
            hdr_target_view,
            hdr_sampled_idx,
            hdr_storage_idx,
            bloom,
            device: ctx.device.clone(),
        })
    }

    fn update(
        &mut self,
        ctx: &RenderContext,
        state: &mut AppState,
        &cbuff: &vk::CommandBuffer,
    ) -> Result<()> {
        let mut lights = [Light::default(); NUM_LIGHTS as usize];
        lights[0] = Light {
            transform: Mat4::look_at_rh(vec3(1., 1., 1.), vec3(0., 0., 0.), Vec3::Z),
            color: vec4(1., 0., 0., 0.),
        };
        lights[1] = Light {
            transform: Mat4::look_at_rh(vec3(-1., 1., 1.), vec3(0., 0., 0.), Vec3::Z),
            color: vec4(0., 1., 0., 0.),
        };
        lights[2] = Light {
            transform: Mat4::look_at_rh(vec3(-1., 1., -1.), vec3(0., 0., 0.), Vec3::Z),
            color: vec4(0., 0., 1., 0.),
        };

        let buffer_data = GpuBuffer {
            size: lights.len() as u32,
            data: lights,
        };

        state
            .staging_write
            .write_buffer(self.lights_buffer.buffer, bytes_of(&buffer_data));

        let pipeline = state.pipeline_arena.get_pipeline(self.spawn_pass);
        let spawn_push_constant = SpawnPC {
            num_rays: NUM_RAYS,
            num_bounces: NUM_BOUNCES,
            time: state.stats.time,
            lights_buffer: self.lights_buffer.address,
            line_buffer: self.lines_buffer.address,
        };
        unsafe {
            let ptr = core::ptr::from_ref(&spawn_push_constant);
            let bytes =
                core::slice::from_raw_parts(ptr.cast(), mem::size_of_val(&spawn_push_constant));
            ctx.device.cmd_push_constants(
                cbuff,
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytes,
            );
            ctx.device
                .cmd_bind_pipeline(cbuff, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            ctx.device
                .cmd_dispatch(cbuff, dispatch_optimal(NUM_RAYS, 256), 1, 1);
        }

        Ok(())
    }

    fn resize(&mut self, ctx: &RenderContext, state: &mut AppState) -> Result<()> {
        let extent = ctx.swapchain.extent;
        let texture_arena = &mut state.texture_arena;
        for &i in &self.accumulate_images {
            if let Some(info) = &mut texture_arena.storage_infos[i] {
                info.extent.width = extent.width;
                info.extent.height = extent.height;
            }
        }
        texture_arena.update_storage_images_by_idx(&self.accumulate_images)?;

        self.hdr_target_info.extent.width = extent.width;
        self.hdr_target_info.extent.height = extent.height;
        self.hdr_target =
            ManagedImage::new(&ctx.device, &self.hdr_target_info, MemoryLocation::GpuOnly)?;
        ctx.device.name_object(self.hdr_target.image, "Hdr Target");
        unsafe { self.device.destroy_image_view(self.hdr_target_view, None) };
        self.hdr_target_view =
            ctx.device
                .create_2d_view(&self.hdr_target.image, self.hdr_target_info.format, 0)?;
        texture_arena.update_sampled_image(self.hdr_sampled_idx, &self.hdr_target_view);
        texture_arena.update_storage_image(self.hdr_storage_idx, &self.hdr_target_view);

        self.bloom.resize(ctx, state)?;
        Ok(())
    }

    fn render(
        &mut self,
        ctx: &RenderContext,
        state: &mut AppState,
        frame: &mut FrameGuard,
    ) -> Result<()> {
        let idx = frame.image_idx;

        let raster_push_const = RasterPC {
            red_image: self.accumulate_images[0] as u32,
            green_image: self.accumulate_images[1] as u32,
            blue_image: self.accumulate_images[2] as u32,
            noise_offset: rand::random::<Vec2>(),
            camera_buffer: state.camera_uniform.address,
            line_buffer: self.lines_buffer.address,
        };

        {
            let pipeline = state.pipeline_arena.get_pipeline(self.clear_pass);
            frame.push_constant(
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                &[raster_push_const],
            );
            frame.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                &[state.texture_arena.storage_set],
            );
            frame.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline.pipeline);
            const SUBGROUP_SIZE: u32 = 16;
            let extent = ctx.swapchain.extent();
            frame.dispatch(
                dispatch_optimal(extent.width, SUBGROUP_SIZE),
                dispatch_optimal(extent.height, SUBGROUP_SIZE),
                1,
            );
        }

        unsafe {
            let mem_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER);
            self.device.cmd_pipeline_barrier2(
                *frame.command_buffer(),
                &vk::DependencyInfo::default().memory_barriers(&[mem_barrier]),
            )
        };

        {
            let pipeline = state.pipeline_arena.get_pipeline(self.raster_pass);
            frame.push_constant(
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                &[raster_push_const],
            );
            frame.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                &[
                    state.texture_arena.sampled_set,
                    state.texture_arena.storage_set,
                ],
            );
            frame.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);
            frame.dispatch(dispatch_optimal(NUM_RAYS * NUM_BOUNCES, 256), 1, 1);
        }

        unsafe {
            let mem_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER);
            self.device.cmd_pipeline_barrier2(
                *frame.command_buffer(),
                &vk::DependencyInfo::default().memory_barriers(&[mem_barrier]),
            )
        };

        {
            self.device.image_transition(
                frame.command_buffer(),
                &self.hdr_target.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::GENERAL,
            );
            let pipeline = state.pipeline_arena.get_pipeline(self.resolve_pass);
            frame.push_constant(
                pipeline.layout,
                vk::ShaderStageFlags::COMPUTE,
                &[ResolvePC {
                    target_image: self.hdr_storage_idx,
                    red_image: self.accumulate_images[0] as u32,
                    green_image: self.accumulate_images[1] as u32,
                    blue_image: self.accumulate_images[2] as u32,
                    camera_buffer: state.camera_uniform.address,
                }],
            );
            frame.bind_descriptor_sets(
                vk::PipelineBindPoint::COMPUTE,
                pipeline.layout,
                &[
                    state.texture_arena.sampled_set,
                    state.texture_arena.storage_set,
                ],
            );
            frame.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline.pipeline);
            const SUBGROUP_SIZE: u32 = 16;
            let extent = ctx.swapchain.extent();
            frame.dispatch(
                dispatch_optimal(extent.width, SUBGROUP_SIZE),
                dispatch_optimal(extent.height, SUBGROUP_SIZE),
                1,
            );
        }

        self.bloom.apply(
            ctx,
            state,
            frame,
            BloomParams {
                target_image: &self.hdr_target.image,
                target_image_sampled: self.hdr_sampled_idx,
                target_image_storage: self.hdr_storage_idx,
                target_current_layout: vk::ImageLayout::GENERAL,
                strength: 4. / 16.,
                width: 2.,
            },
        );

        {
            frame.begin_rendering(
                ctx.swapchain.get_current_image_view(),
                vk::AttachmentLoadOp::CLEAR,
                [0., 0.025, 0.025, 1.0],
            );
            let pipeline = state.pipeline_arena.get_pipeline(self.postprocess_pass);
            frame.push_constant(
                pipeline.layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                &[PostProcessPC {
                    current_image: state.texture_arena.external_sampled_img_idx[idx as usize],
                    hdr_sampled: self.hdr_sampled_idx,
                    hdr_storage: self.hdr_storage_idx,
                }],
            );
            frame.bind_descriptor_sets(
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.layout,
                &[
                    state.texture_arena.sampled_set,
                    state.texture_arena.storage_set,
                ],
            );
            frame.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &pipeline.pipeline);
            frame.draw(3, 0, 1, 0);
            frame.end_rendering();
        }

        Ok(())
    }
}

fn main() -> Result<()> {
    let event_loop = winit::event_loop::EventLoop::with_user_event().build()?;

    let camera = Camera::new(vec3(0., 0., 10.), 0., 0.);
    let mut app = App::<LineRaster>::new(event_loop.create_proxy(), Some(camera));
    event_loop.run_app(&mut app)?;
    Ok(())
}
