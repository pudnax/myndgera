use std::path::PathBuf;

use anyhow::Result;
use ash::vk;
use myndgera::*;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct RasterPC {
    current_image: u32,
    line_buffer: u64,
}

struct LineRaster {
    compute_raster: ComputeHandle,
}

impl Example for LineRaster {
    fn name() -> &'static str {
        "Line Rasteriazation"
    }
    fn init(ctx: &mut RenderContext) -> Result<Self> {
        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<RasterPC>() as _)
            .stage_flags(vk::ShaderStageFlags::COMPUTE);
        let compute_raster = ctx.pipeline_arena.create_compute_pipeline(
            "examples/line_raster/raster.comp",
            &[push_constant_range],
            &[ctx.texture_arena.storage_set_layout],
        )?;
        Ok(Self { compute_raster })
    }

    fn render(&mut self, ctx: &mut RenderContext, frame: &mut FrameGuard) -> Result<()> {
        let idx = frame.image_idx;
        let image = ctx.swapchain.images[idx];
        ctx.device.image_transition(
            frame.command_buffer(),
            &image,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            vk::ImageLayout::GENERAL,
        );

        let pipeline = ctx.pipeline_arena.get_pipeline(self.compute_raster);
        let pc = RasterPC {
            current_image: idx as u32,
            line_buffer: 5,
        };
        frame.push_constant(pipeline.layout, vk::ShaderStageFlags::COMPUTE, &[pc]);
        frame.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            &[ctx.texture_arena.storage_set],
        );
        frame.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline);
        const SUBGROUP_SIZE: u32 = 16;
        frame.dispatch(
            dispatch_optimal(frame.extent.width, SUBGROUP_SIZE),
            dispatch_optimal(frame.extent.height, SUBGROUP_SIZE),
            1,
        );

        ctx.device.image_transition(
            frame.command_buffer(),
            &image,
            vk::ImageLayout::GENERAL,
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        );

        Ok(())
    }
}

fn main() -> Result<()> {
    let event_loop = winit::event_loop::EventLoop::with_user_event().build()?;

    let shader_dir = PathBuf::new().join(SHADER_FOLDER);
    if !shader_dir.is_dir() {
        default_shaders::create_default_shaders(&shader_dir)?;
    }

    let mut app = App::<LineRaster>::new(event_loop.create_proxy());
    event_loop.run_app(&mut app)?;
    Ok(())
}
