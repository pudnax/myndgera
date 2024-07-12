use std::path::PathBuf;

use anyhow::Result;
use ash::vk;
use myndgera::*;

struct Toy {
    render_pipeline: RenderHandle,
    compute_pipeline: ComputeHandle,
}

impl Example for Toy {
    fn name() -> &'static str {
        "Toy"
    }
    fn init(ctx: RenderContext) -> Result<Self> {
        let vertex_shader_desc = VertexShaderDesc {
            shader_path: "examples/toy/shader.vert".into(),
            ..Default::default()
        };
        let fragment_shader_desc = FragmentShaderDesc {
            shader_path: "examples/toy/shader.frag".into(),
        };
        let fragment_output_desc = FragmentOutputDesc {
            surface_format: ctx.swapchain.format(),
            ..Default::default()
        };
        let push_constant_range = vk::PushConstantRange::default()
            .size(size_of::<GlobalStats>() as _)
            .stage_flags(
                vk::ShaderStageFlags::VERTEX
                    | vk::ShaderStageFlags::FRAGMENT
                    | vk::ShaderStageFlags::COMPUTE,
            );
        let render_pipeline = ctx.pipeline_arena.create_render_pipeline(
            &VertexInputDesc::default(),
            &vertex_shader_desc,
            &fragment_shader_desc,
            &fragment_output_desc,
            &[push_constant_range],
            &[ctx.texture_arena.images_set_layout],
        )?;

        let compute_pipeline = ctx.pipeline_arena.create_compute_pipeline(
            "examples/toy/shader.comp",
            &[push_constant_range],
            &[ctx.texture_arena.images_set_layout],
        )?;
        Ok(Self {
            render_pipeline,
            compute_pipeline,
        })
    }

    fn render(&mut self, ctx: RenderContext, frame: &mut FrameGuard) -> Result<()> {
        let stages = vk::ShaderStageFlags::VERTEX
            | vk::ShaderStageFlags::FRAGMENT
            | vk::ShaderStageFlags::COMPUTE;
        let pipeline = ctx.pipeline_arena.get_pipeline(self.compute_pipeline);
        frame.push_constant(pipeline.layout, stages, &[*ctx.stats]);
        frame.bind_descriptor_sets(
            vk::PipelineBindPoint::COMPUTE,
            pipeline.layout,
            &[ctx.texture_arena.images_set],
        );
        frame.bind_pipeline(vk::PipelineBindPoint::COMPUTE, &pipeline.pipeline);
        const SUBGROUP_SIZE: u32 = 16;
        let extent = ctx.swapchain.extent();
        frame.dispatch(
            dispatch_optimal(extent.width, SUBGROUP_SIZE),
            dispatch_optimal(extent.height, SUBGROUP_SIZE),
            1,
        );

        unsafe {
            let image_barrier = vk::ImageMemoryBarrier2::default()
                .subresource_range(COLOR_SUBRESOURCE_MASK)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS)
                .image(ctx.texture_arena.images[PREV_FRAME_IDX]);
            ctx.device.cmd_pipeline_barrier2(
                *frame.command_buffer(),
                &vk::DependencyInfo::default()
                    .image_memory_barriers(std::slice::from_ref(&image_barrier)),
            )
        };

        frame.begin_rendering(
            ctx.swapchain.get_current_image_view(),
            [0., 0.025, 0.025, 1.0],
        );
        let pipeline = ctx.pipeline_arena.get_pipeline(self.render_pipeline);
        frame.push_constant(
            pipeline.layout,
            vk::ShaderStageFlags::VERTEX
                | vk::ShaderStageFlags::FRAGMENT
                | vk::ShaderStageFlags::COMPUTE,
            &[*ctx.stats],
        );
        frame.bind_descriptor_sets(
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.layout,
            &[ctx.texture_arena.images_set],
        );
        frame.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &pipeline.pipeline);

        frame.draw(3, 0, 1, 0);
        frame.end_rendering();

        Ok(())
    }
}

fn main() -> Result<()> {
    let event_loop = winit::event_loop::EventLoop::with_user_event().build()?;

    let shader_dir = PathBuf::new().join(SHADER_FOLDER);
    if !shader_dir.is_dir() {
        default_shaders::create_default_shaders(&shader_dir)?;
    }

    let mut app = App::<Toy>::new(event_loop.create_proxy());
    event_loop.run_app(&mut app)?;
    Ok(())
}