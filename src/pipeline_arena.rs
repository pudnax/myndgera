use ahash::{AHashMap, AHashSet};
use anyhow::Result;
use either::Either;
use shaderc::ShaderKind;
use slotmap::SlotMap;
use std::{
    path::{Path, PathBuf},
    sync::Arc,
};

use ash::{
    prelude::VkResult,
    vk::{self},
};

use crate::{Device, ShaderCompiler, Watcher};

pub struct ComputePipeline {
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    device: Arc<ash::Device>,
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

pub struct RenderPipeline {
    pub layout: vk::PipelineLayout,
    pub pipeline: vk::Pipeline,
    vertex_input_lib: vk::Pipeline,
    vertex_shader_lib: vk::Pipeline,
    fragment_shader_lib: vk::Pipeline,
    fragment_output_lib: vk::Pipeline,
    device: Arc<ash::Device>,
}

impl RenderPipeline {
    pub fn new(
        device: &crate::Device,
        shader_compiler: &ShaderCompiler,
        watcher: &mut Watcher,
        surface_format: vk::Format,
        cache: &vk::PipelineCache,
    ) -> Result<Self> {
        let vs_bytes = shader_compiler.compile("shaders/trig.vert.glsl", ShaderKind::Vertex)?;
        let fs_bytes = shader_compiler.compile("shaders/trig.frag.glsl", ShaderKind::Fragment)?;
        watcher.watch_file("shaders/trig.vert.glsl")?;
        watcher.watch_file("shaders/trig.frag.glsl")?;

        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX)
            .size(size_of::<u64>() as _);
        let pipeline_layout = unsafe {
            device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .push_constant_ranges(std::slice::from_ref(&push_constant_range)),
                None,
            )?
        };

        use vk::GraphicsPipelineLibraryFlagsEXT as GPF;
        let vertex_input_lib = {
            let input_ass = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST)
                .primitive_restart_enable(false);
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

            create_library(device, cache, GPF::VERTEX_INPUT_INTERFACE, |desc| {
                desc.vertex_input_state(&vertex_input)
                    .input_assembly_state(&input_ass)
            })?
        };

        let vertex_shader_lib = {
            let mut shader_module =
                vk::ShaderModuleCreateInfo::default().code(vs_bytes.as_binary());
            let shader_stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .name(c"main")
                .push_next(&mut shader_module);
            let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
            let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
                .line_width(1.0)
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);

            create_library(device, cache, GPF::PRE_RASTERIZATION_SHADERS, |desc| {
                desc.layout(pipeline_layout)
                    .stages(std::slice::from_ref(&shader_stage))
                    .dynamic_state(&dynamic_state)
                    .viewport_state(&viewport_state)
                    .rasterization_state(&rasterization_state)
            })?
        };

        let fragment_shader_lib = {
            let mut shader_module =
                vk::ShaderModuleCreateInfo::default().code(fs_bytes.as_binary());
            let shader_stage = vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .name(c"main")
                .push_next(&mut shader_module);

            let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default();

            create_library(device, cache, GPF::FRAGMENT_SHADER, |desc| {
                desc.layout(pipeline_layout)
                    .stages(std::slice::from_ref(&shader_stage))
                    .depth_stencil_state(&depth_stencil_state)
            })?
        };

        let fragment_output_lib = {
            let color_attachment_formats = [surface_format];
            let mut dyn_render = vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(&color_attachment_formats);

            let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);

            create_library(device, cache, GPF::FRAGMENT_OUTPUT_INTERFACE, |desc| {
                desc.multisample_state(&multisample_state)
                    .push_next(&mut dyn_render)
            })?
        };

        let libraries = [
            vertex_input_lib,
            vertex_shader_lib,
            fragment_shader_lib,
            fragment_output_lib,
        ];
        let pipeline = {
            let mut linking_info =
                vk::PipelineLibraryCreateInfoKHR::default().libraries(&libraries);
            let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .flags(vk::PipelineCreateFlags::LINK_TIME_OPTIMIZATION_EXT)
                .layout(pipeline_layout)
                .push_next(&mut linking_info);
            let pipeline =
                unsafe { device.create_graphics_pipelines(*cache, &[pipeline_info], None) };
            pipeline.map_err(|(_, err)| err)?[0]
        };

        Ok(Self {
            device: device.device.clone(),
            layout: pipeline_layout,
            pipeline,
            vertex_input_lib,
            vertex_shader_lib,
            fragment_shader_lib,
            fragment_output_lib,
        })
    }

    pub fn reload_vertex_lib(
        &mut self,
        shader_compiler: &ShaderCompiler,
        pipeline_cache: &vk::PipelineCache,
        shader_path: impl AsRef<Path>,
    ) -> Result<()> {
        let vs_bytes = shader_compiler.compile(shader_path, ShaderKind::Vertex)?;

        unsafe { self.device.destroy_pipeline(self.vertex_shader_lib, None) };

        let mut shader_module = vk::ShaderModuleCreateInfo::default().code(vs_bytes.as_binary());
        let shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .name(c"main")
            .push_next(&mut shader_module);
        let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
            .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
        let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
            .line_width(1.0)
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::BACK)
            .front_face(vk::FrontFace::COUNTER_CLOCKWISE);
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let vertex_shader_lib = create_library(
            &self.device,
            pipeline_cache,
            vk::GraphicsPipelineLibraryFlagsEXT::PRE_RASTERIZATION_SHADERS,
            |desc| {
                desc.layout(self.layout)
                    .stages(std::slice::from_ref(&shader_stage))
                    .dynamic_state(&dynamic_state)
                    .viewport_state(&viewport_state)
                    .rasterization_state(&rasterization_state)
            },
        )?;

        self.vertex_shader_lib = vertex_shader_lib;

        unsafe { self.device.destroy_pipeline(self.pipeline, None) };
        self.pipeline = Self::link_libraries(
            &self.device,
            pipeline_cache,
            &self.layout,
            &self.vertex_input_lib,
            &self.vertex_shader_lib,
            &self.fragment_shader_lib,
            &self.fragment_output_lib,
        )?;

        Ok(())
    }

    pub fn reload_fragment_lib(
        &mut self,
        shader_compiler: &ShaderCompiler,
        pipeline_cache: &vk::PipelineCache,
        shader_path: impl AsRef<Path>,
    ) -> Result<()> {
        let fs_bytes = shader_compiler.compile(shader_path, ShaderKind::Fragment)?;

        unsafe { self.device.destroy_pipeline(self.fragment_shader_lib, None) };

        let mut shader_module = vk::ShaderModuleCreateInfo::default().code(fs_bytes.as_binary());
        let shader_stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .name(c"main")
            .push_next(&mut shader_module);

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default();

        let fragment_shader_lib = create_library(
            &self.device,
            pipeline_cache,
            vk::GraphicsPipelineLibraryFlagsEXT::FRAGMENT_SHADER,
            |desc| {
                desc.layout(self.layout)
                    .stages(std::slice::from_ref(&shader_stage))
                    .depth_stencil_state(&depth_stencil_state)
            },
        )?;

        self.fragment_shader_lib = fragment_shader_lib;

        unsafe { self.device.destroy_pipeline(self.pipeline, None) };
        self.pipeline = Self::link_libraries(
            &self.device,
            pipeline_cache,
            &self.layout,
            &self.vertex_input_lib,
            &self.vertex_shader_lib,
            &self.fragment_shader_lib,
            &self.fragment_output_lib,
        )?;

        Ok(())
    }

    fn link_libraries(
        device: &ash::Device,
        cache: &vk::PipelineCache,
        layout: &vk::PipelineLayout,
        vertex_input_lib: &vk::Pipeline,
        vertex_shader_lib: &vk::Pipeline,
        fragment_shader_lib: &vk::Pipeline,
        fragment_output_lib: &vk::Pipeline,
    ) -> Result<vk::Pipeline> {
        let libraries = [
            *vertex_input_lib,
            *vertex_shader_lib,
            *fragment_shader_lib,
            *fragment_output_lib,
        ];
        let pipeline = {
            let mut linking_info =
                vk::PipelineLibraryCreateInfoKHR::default().libraries(&libraries);
            let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .flags(vk::PipelineCreateFlags::LINK_TIME_OPTIMIZATION_EXT)
                .layout(*layout)
                .push_next(&mut linking_info);
            let pipeline =
                unsafe { device.create_graphics_pipelines(*cache, &[pipeline_info], None) };
            pipeline.map_err(|(_, err)| err)?[0]
        };

        Ok(pipeline)
    }
}

impl Drop for RenderPipeline {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.vertex_input_lib, None);
            self.device.destroy_pipeline(self.vertex_shader_lib, None);
            self.device.destroy_pipeline(self.fragment_shader_lib, None);
            self.device.destroy_pipeline(self.fragment_output_lib, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.layout, None);
        }
    }
}

fn create_library<'a, F>(
    device: &ash::Device,
    cache: &vk::PipelineCache,
    kind: vk::GraphicsPipelineLibraryFlagsEXT,
    f: F,
) -> VkResult<vk::Pipeline>
where
    F: FnOnce(vk::GraphicsPipelineCreateInfo<'a>) -> vk::GraphicsPipelineCreateInfo<'a>,
{
    let mut library_type = vk::GraphicsPipelineLibraryCreateInfoEXT::default().flags(kind);
    let pipeline = unsafe {
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default().flags(
            vk::PipelineCreateFlags::LIBRARY_KHR
                | vk::PipelineCreateFlags::RETAIN_LINK_TIME_OPTIMIZATION_INFO_EXT,
        );

        // TODO: `let` introduces implicit copy on the struct that contains pointers
        let pipeline_info = f(pipeline_info).push_next(&mut library_type);

        device.create_graphics_pipelines(*cache, std::slice::from_ref(&pipeline_info), None)
    };

    Ok(pipeline.map_err(|(_, err)| err)?[0])
}

slotmap::new_key_type! {
    pub struct RenderHandle;
    pub struct ComputeHandle;
}

pub struct PipelineArena {
    pub render: RenderArena,
    compute: ComputeArena,
    pub path_mapping: AHashMap<PathBuf, AHashSet<Either<RenderHandle, ComputeHandle>>>,
    shader_compiler: ShaderCompiler,
    file_watcher: Watcher,
    device: Arc<ash::Device>,
}

impl PipelineArena {
    pub fn new(device: &Device, file_watcher: Watcher) -> Result<Self> {
        Ok(Self {
            render: RenderArena {
                pipelines: SlotMap::with_key(),
            },
            compute: ComputeArena {
                pipelines: SlotMap::with_key(),
            },
            shader_compiler: ShaderCompiler::new(file_watcher.clone())?,
            file_watcher,
            path_mapping: AHashMap::new(),
            device: device.device.clone(),
        })
    }

    pub fn get_pipeline<H: Handle>(&self, handle: H) -> &H::Pipeline {
        handle.get_pipeline(self)
    }

    pub fn get_pipeline_mut<H: Handle>(&mut self, handle: H) -> &mut H::Pipeline {
        handle.get_pipeline_mut(self)
    }
}

pub struct RenderArena {
    pub pipelines: SlotMap<RenderHandle, RenderPipeline>,
    //     descriptors: SecondaryMap<RenderHandle, RenderPipelineDescriptor>,
}

struct ComputeArena {
    pipelines: SlotMap<ComputeHandle, ComputePipeline>,
    //     descriptors: SecondaryMap<ComputeHandle, ComputePipelineDescriptor>,
}

pub trait Handle {
    type Pipeline;
    // type Descriptor;
    fn get_pipeline(self, arena: &PipelineArena) -> &Self::Pipeline;
    fn get_pipeline_mut(self, arena: &mut PipelineArena) -> &mut Self::Pipeline;
    // fn get_descriptor(self, arena: &PipelineArena) -> &Self::Descriptor;
}

impl Handle for RenderHandle {
    type Pipeline = RenderPipeline;
    // type Descriptor = RenderPipelineDescriptor;

    fn get_pipeline(self, arena: &PipelineArena) -> &Self::Pipeline {
        &arena.render.pipelines[self]
    }

    fn get_pipeline_mut(self, arena: &mut PipelineArena) -> &mut Self::Pipeline {
        &mut arena.render.pipelines[self]
    }

    // fn get_descriptor(self, arena: &PipelineArena) -> &Self::Descriptor {
    //     &arena.render.descriptors[self]
    // }
}

impl Handle for ComputeHandle {
    type Pipeline = ComputePipeline;
    // type Descriptor = ComputePipelineDescriptor;
    fn get_pipeline(self, arena: &PipelineArena) -> &Self::Pipeline {
        &arena.compute.pipelines[self]
    }

    fn get_pipeline_mut(self, arena: &mut PipelineArena) -> &mut Self::Pipeline {
        &mut arena.compute.pipelines[self]
    }

    // fn get_descriptor(self, arena: &PipelineArena) -> &Self::Descriptor {
    //     &arena.compute.descriptors[self]
    // }
}
