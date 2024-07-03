use std::path::PathBuf;

use anyhow::{bail, Result};
use ash::{khr, vk};
use either::Either;
use myndgera::{
    Device, FragmentOutputDesc, FragmentShaderDesc, HostBuffer, Instance, PipelineArena,
    RenderHandle, ShaderKind, ShaderSource, Surface, Swapchain, UserEvent, VertexInputDesc,
    VertexShaderDesc, Watcher,
};
use winit::{
    application::ApplicationHandler,
    event::{KeyEvent, WindowEvent},
    event_loop::EventLoopProxy,
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

#[allow(dead_code)]
struct AppInit {
    time: std::time::Instant,
    window: Window,
    watcher: Watcher,

    host_buffer: HostBuffer<[f32; 4]>,

    pipeline_handle: RenderHandle,
    pipeline_arena: PipelineArena,

    queue: vk::Queue,
    transfer_queue: vk::Queue,

    swapchain: Swapchain,
    surface: Surface,
    device: Device,
    instance: Instance,
}

impl AppInit {
    fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        proxy: EventLoopProxy<UserEvent>,
        window_attributes: WindowAttributes,
    ) -> Result<Self> {
        let window = event_loop.create_window(window_attributes)?;
        let watcher = Watcher::new(proxy)?;

        let instance = Instance::new(Some(&window))?;
        let surface = instance.create_surface(&window)?;
        let (device, queue, transfer_queue) = instance.create_device_and_queues(&surface)?;

        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);
        let swapchain = Swapchain::new(&device, &surface, swapchain_loader)?;

        let host_buffer = device.create_host_buffer(
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let mut pipeline_arena = PipelineArena::new(&device, watcher.clone())?;

        let vertex_shader_desc = VertexShaderDesc {
            shader_path: "shaders/trig.vert.glsl".into(),
            ..Default::default()
        };
        let fragment_shader_desc = FragmentShaderDesc {
            shader_path: "shaders/trig.frag.glsl".into(),
        };
        let fragment_output_desc = FragmentOutputDesc {
            surface_format: swapchain.format(),
            ..Default::default()
        };
        let push_constant_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT)
            .size(size_of::<u64>() as _);
        let pipeline_handle = pipeline_arena.create_render_pipeline(
            &VertexInputDesc::default(),
            &vertex_shader_desc,
            &fragment_shader_desc,
            &fragment_output_desc,
            &[push_constant_range],
            &[],
        )?;

        Ok(Self {
            watcher,
            time: std::time::Instant::now(),
            host_buffer,
            pipeline_arena,
            pipeline_handle,
            window,
            surface,
            swapchain,
            device,
            queue,
            transfer_queue,
            instance,
        })
    }

    fn reload_shaders(&mut self, path: PathBuf) -> Result<()> {
        if let Some(frame) = self.swapchain.get_current_frame() {
            let fences = std::slice::from_ref(&frame.present_finished);
            unsafe { self.device.wait_for_fences(fences, true, u64::MAX)? };
        }

        let resolved = {
            let mapping = self.watcher.include_mapping.lock();
            mapping[&path].clone()
        };

        for ShaderSource { path, kind } in resolved {
            let handles = &self.pipeline_arena.path_mapping[&path];
            for handle in handles {
                let cache = &self.pipeline_arena.pipeline_cache;
                let compiler = &self.pipeline_arena.shader_compiler;
                match handle {
                    Either::Left(handle) => {
                        let pipeline = &mut self.pipeline_arena.render.pipelines[*handle];
                        match kind {
                            ShaderKind::Vertex => {
                                pipeline.reload_vertex_lib(compiler, cache, &path)?
                            }
                            ShaderKind::Fragment => {
                                pipeline.reload_fragment_lib(compiler, cache, &path)?
                            }
                            ShaderKind::Compute => {
                                bail!("Supplied compute shader into render pipeline!")
                            }
                        }
                        pipeline.link(cache)?;
                    }
                    Either::Right(_compute) => {}
                }
            }
        }
        Ok(())
    }

    fn recreate_swapchain(&mut self) {
        self.swapchain
            .recreate(&self.device, &self.surface)
            .expect("Failed to recreate swapchain")
    }
}

impl ApplicationHandler<UserEvent> for AppInit {
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested
            | WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(NamedKey::Escape),
                        ..
                    },
                ..
            } => event_loop.exit(),
            WindowEvent::RedrawRequested => {
                let t = self.time.elapsed().as_secs_f32();
                let (c, s) = (t.cos(), t.sin());
                self.host_buffer.copy_from_slice(&[c, -s, s, c]);

                let mut frame = match self.swapchain.acquire_next_image() {
                    Ok(frame) => frame,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        self.recreate_swapchain();
                        self.window.request_redraw();
                        return;
                    }
                    Err(e) => panic!("error: {e}\n"),
                };

                frame.begin_rendering(
                    self.swapchain.get_current_image_view(),
                    [0., 0.025, 0.025, 1.0],
                );
                let pipeline = self.pipeline_arena.get_pipeline(self.pipeline_handle);
                frame.push_constant(
                    pipeline.layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                    &[self.host_buffer.address],
                );
                frame.bind_pipeline(vk::PipelineBindPoint::GRAPHICS, &pipeline.pipeline);

                frame.draw(3, 0);
                frame.end_rendering();

                match self.swapchain.submit_image(&self.queue, frame) {
                    Ok(_) => {}
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => self.recreate_swapchain(),
                    Err(e) => panic!("error: {e}\n"),
                }

                self.window.request_redraw();
            }
            _ => {}
        }
    }

    fn user_event(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Glsl { path } => {
                let time = std::time::Instant::now();

                if let Err(err) = self.reload_shaders(path) {
                    eprintln!("{err}");
                };

                println!("Elapsed: {:?}", time.elapsed());
            }
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        let _ = unsafe { self.device.device_wait_idle() };
    }

    fn resumed(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        panic!("On native platforms `resumed` can be called only once.")
    }
}

fn main() -> Result<()> {
    let event_loop = winit::event_loop::EventLoop::with_user_event().build()?;

    let mut app = App::new(event_loop.create_proxy());
    event_loop.run_app(&mut app)?;
    Ok(())
}

struct App {
    proxy: EventLoopProxy<UserEvent>,
    inner: AppEnum,
}

impl App {
    fn new(proxy: EventLoopProxy<UserEvent>) -> Self {
        Self {
            proxy,
            inner: AppEnum::Uninitialized,
        }
    }
}

#[derive(Default)]
enum AppEnum {
    #[default]
    Uninitialized,
    Init(AppInit),
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes = WindowAttributes::default();
        match self.inner {
            AppEnum::Uninitialized => {
                self.inner = AppEnum::Init(
                    AppInit::new(event_loop, self.proxy.clone(), window_attributes)
                        .expect("Failed to create application"),
                )
            }
            AppEnum::Init(_) => {}
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        if let AppEnum::Init(app) = &mut self.inner {
            app.window_event(event_loop, window_id, event);
        }
    }

    fn new_events(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        if let AppEnum::Init(app) = &mut self.inner {
            app.new_events(event_loop, cause);
        }
    }

    fn user_event(&mut self, event_loop: &winit::event_loop::ActiveEventLoop, event: UserEvent) {
        if let AppEnum::Init(app) = &mut self.inner {
            app.user_event(event_loop, event)
        }
    }

    fn device_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let AppEnum::Init(app) = &mut self.inner {
            app.device_event(event_loop, device_id, event)
        }
    }

    fn about_to_wait(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let AppEnum::Init(app) = &mut self.inner {
            app.about_to_wait(event_loop)
        }
    }

    fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let AppEnum::Init(app) = &mut self.inner {
            app.suspended(event_loop)
        }
    }

    fn exiting(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let AppEnum::Init(app) = &mut self.inner {
            app.exiting(event_loop)
        }
    }

    fn memory_warning(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if let AppEnum::Init(app) = &mut self.inner {
            app.memory_warning(event_loop)
        }
    }
}
