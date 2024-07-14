use anyhow::{Ok, Result};
use ash::vk;
use glam::Vec4;
use gpu_alloc::UsageFlags;
use myndgera::*;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Line {
    start: Vec4,
    end: Vec4,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct SpawnPC {
    line_buffer: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RasterPC {
    red_image: u32,
    green_image: u32,
    blue_image: u32,
    camera_buffer: u64,
    line_buffer: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ResolvePC {
    current_image: u32,
    red_image: u32,
    green_image: u32,
    bluew_image: u32,
}

const NUM_LINES: u32 = 50;

struct LineRaster {
    spawn_pass: ComputeHandle,
    lines_buffer: Buffer,
    fill_pass: ComputeHandle,
    raster_pass: ComputeHandle,
    resolve_pass: RenderHandle,
    accumulate_images: Vec<usize>,
}

impl Example for LineRaster {
    fn name() -> &'static str {
        "Line Rasteriazation"
    }
    fn init(ctx: RenderContext) -> Result<Self> {
        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<SpawnPC>() as _)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let spawn_pass = ctx.pipeline_arena.create_compute_pipeline(
            "examples/line_raster/spawn.comp",
            &[push_constant_range],
            &[],
        )?;
        let size = std::mem::size_of::<Line>() * NUM_LINES as usize;
        let lines_buffer = ctx.device.create_buffer(
            size as u64,
            vk::BufferUsageFlags::STORAGE_BUFFER,
            UsageFlags::FAST_DEVICE_ACCESS,
        )?;

        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<RasterPC>() as _)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let fill_pass = ctx.pipeline_arena.create_compute_pipeline(
            "examples/line_raster/fill.comp",
            &[push_constant_range],
            &[ctx.texture_arena.storage_set_layout],
        )?;
        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<RasterPC>() as _)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let raster_pass = ctx.pipeline_arena.create_compute_pipeline(
            "examples/line_raster/raster.comp",
            &[push_constant_range],
            &[ctx.texture_arena.storage_set_layout],
        )?;

        let vertex_shader_desc = VertexShaderDesc {
            shader_path: "examples/line_raster/resolve.vert".into(),
            ..Default::default()
        };
        let fragment_shader_desc = FragmentShaderDesc {
            shader_path: "examples/line_raster/resolve.frag".into(),
        };
        let fragment_output_desc = FragmentOutputDesc {
            surface_format: ctx.swapchain.format(),
            ..Default::default()
        };
        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<ResolvePC>() as _)
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT);
        let resolve_pass = ctx.pipeline_arena.create_render_pipeline(
            &VertexInputDesc::default(),
            &vertex_shader_desc,
            &fragment_shader_desc,
            &fragment_output_desc,
            &[push_constant_range],
            &[
                ctx.texture_arena.images_set_layout,
                ctx.texture_arena.storage_set_layout,
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
            let (image, memory) = ctx
                .device
                .create_image(&info, UsageFlags::FAST_DEVICE_ACCESS)?;
            let view = ctx.device.create_2d_view(&image, vk::Format::R32_UINT)?;
            let idx = ctx
                .texture_arena
                .push_storage_image(image, view, Some(memory), Some(info));
            accumulate_images.push(idx as usize);
            device.name_object(view, &format!("Storage img view: {i}"));
            device.name_object(image, &format!("Storage img: {i}"));
        }

        Ok(Self {
            spawn_pass,
            lines_buffer,
            fill_pass,
            raster_pass,
            resolve_pass,
            accumulate_images,
        })
    }

    fn update(&mut self, ctx: RenderContext) -> Result<()> {
        let pipeline = ctx.pipeline_arena.get_pipeline(self.spawn_pass);
        let spawn_push_constant = SpawnPC {
            line_buffer: self.lines_buffer.address,
        };
        ctx.device.one_time_submit(ctx.queue, |device, cbuff| {
            unsafe {
                let ptr = core::ptr::from_ref(&spawn_push_constant);
                let bytes = core::slice::from_raw_parts(
                    ptr.cast(),
                    std::mem::size_of_val(&spawn_push_constant),
                );
                device.cmd_push_constants(
                    cbuff,
                    pipeline.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    bytes,
                );
                device.cmd_bind_pipeline(cbuff, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
                device.cmd_dispatch(cbuff, NUM_LINES, 1, 1);
            }
            Ok(())
        })
    }

    fn resize(&mut self, ctx: RenderContext) -> Result<()> {
        let extent = ctx.swapchain.extent;
        for &i in &self.accumulate_images {
            if let Some(info) = &mut ctx.texture_arena.storage_infos[i] {
                info.extent.width = extent.width;
                info.extent.height = extent.height;
            }
        }
        ctx.texture_arena
            .update_storage_images_by_idx(&self.accumulate_images)?;
        Ok(())
    }

    fn render(&mut self, ctx: RenderContext, frame: &mut FrameGuard) -> Result<()> {
        let idx = frame.image_idx;

        let raster_push_const = RasterPC {
            red_image: self.accumulate_images[0] as u32,
            green_image: self.accumulate_images[1] as u32,
            blue_image: self.accumulate_images[2] as u32,
            camera_buffer: ctx.camera_uniform.address,
            line_buffer: self.lines_buffer.address,
        };

        let pipeline = ctx.pipeline_arena.get_pipeline(self.fill_pass);
        frame.push_constant(
            pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            &[raster_push_const],
        );
        frame.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            &[ctx.texture_arena.storage_set],
        );
        frame.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline.pipeline);
        const SUBGROUP_SIZE: u32 = 16;
        let extent = ctx.swapchain.extent();
        frame.dispatch(
            dispatch_optimal(extent.width, SUBGROUP_SIZE),
            dispatch_optimal(extent.height, SUBGROUP_SIZE),
            1,
        );

        let pipeline = ctx.pipeline_arena.get_pipeline(self.raster_pass);
        frame.push_constant(
            pipeline.layout,
            vk::ShaderStageFlags::COMPUTE,
            &[raster_push_const],
        );
        frame.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            &[ctx.texture_arena.storage_set],
        );
        frame.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);
        frame.dispatch(NUM_LINES, 1, 1);

        frame.begin_rendering(
            ctx.swapchain.get_current_image_view(),
            [0., 0.025, 0.025, 1.0],
        );
        let pipeline = ctx.pipeline_arena.get_pipeline(self.resolve_pass);
        frame.push_constant(
            pipeline.layout,
            vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
            &[ResolvePC {
                current_image: idx as u32,
                red_image: self.accumulate_images[0] as u32,
                green_image: self.accumulate_images[1] as u32,
                bluew_image: self.accumulate_images[2] as u32,
            }],
        );
        frame.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.layout,
            &[ctx.texture_arena.images_set, ctx.texture_arena.storage_set],
        );
        frame.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &pipeline.pipeline);

        frame.draw(3, 0, 1, 0);
        frame.end_rendering();

        Ok(())
    }
}

fn main() -> Result<()> {
    let event_loop = winit::event_loop::EventLoop::with_user_event().build()?;

    let mut app = App::<LineRaster>::new(event_loop.create_proxy());
    event_loop.run_app(&mut app)?;
    Ok(())
}
