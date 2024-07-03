use std::path::PathBuf;

use ahash::AHashSet;
use anyhow::{bail, Result};
use ash::{khr, vk};
use either::Either;
use myndgera::{
    Device, HostBuffer, Instance, PipelineArena, RenderHandle, RenderPipeline, ShaderCompiler,
    Surface, Swapchain, UserEvent, Watcher,
};
use shaderc::ShaderKind;
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
    shader_compiler: ShaderCompiler,

    host_buffer: HostBuffer<[f32; 4]>,

    pipeline_handle: RenderHandle,
    pipeline_arena: PipelineArena,
    pipeline_cache: vk::PipelineCache,

    queue: vk::Queue,
    transfer_queue: vk::Queue,

    swapchain: Swapchain,
    surface: Surface,
    device: Device,
    instance: Instance,
}

impl Drop for AppInit {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_pipeline_cache(self.pipeline_cache, None);
        }
    }
}

impl AppInit {
    fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        proxy: EventLoopProxy<UserEvent>,
        window_attributes: WindowAttributes,
    ) -> Result<Self> {
        let window = event_loop.create_window(window_attributes)?;
        let mut watcher = Watcher::new(proxy)?;

        let instance = Instance::new(Some(&window))?;
        let surface = instance.create_surface(&window)?;
        let (device, queue, transfer_queue) = instance.create_device_and_queues(&surface)?;

        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);
        let swapchain = Swapchain::new(&device, &surface, swapchain_loader)?;

        let shader_compiler = ShaderCompiler::new(watcher.clone())?;

        let pipeline_cache =
            unsafe { device.create_pipeline_cache(&vk::PipelineCacheCreateInfo::default(), None)? };

        let host_buffer = device.create_host_buffer(
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        )?;

        let rpipeline = RenderPipeline::new(
            &device,
            &shader_compiler,
            &mut watcher,
            swapchain.format(),
            &pipeline_cache,
        )?;
        let mut pipeline_arena = PipelineArena::new(&device, watcher.clone())?;
        let pipeline_handle = pipeline_arena.render.pipelines.insert(rpipeline);
        let path = PathBuf::new()
            .join("shaders/trig.vert.glsl")
            .canonicalize()
            .unwrap();
        pipeline_arena
            .path_mapping
            .entry(path.clone())
            .or_insert_with_key(|path| {
                let _ = watcher.watch_file(path);
                AHashSet::new()
            })
            .insert(Either::Left(pipeline_handle));
        {
            let mut mapping = watcher.include_mapping.lock();
            mapping
                .entry(path.clone())
                .or_default()
                .insert(path.clone());
        }
        let path = PathBuf::new()
            .join("shaders/trig.frag.glsl")
            .canonicalize()
            .unwrap();
        pipeline_arena
            .path_mapping
            .entry(path.clone())
            .or_insert_with_key(|path| {
                let _ = watcher.watch_file(path);
                AHashSet::new()
            })
            .insert(Either::Left(pipeline_handle));
        {
            let mut mapping = watcher.include_mapping.lock();
            mapping
                .entry(path.clone())
                .or_default()
                .insert(path.clone());
        }

        dbg!(&watcher.include_mapping.lock());

        Ok(Self {
            shader_compiler,
            watcher,
            time: std::time::Instant::now(),
            host_buffer,
            pipeline_arena,
            pipeline_handle,
            pipeline_cache,
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

        for res in resolved {
            let Some(x) = self.pipeline_arena.path_mapping.get(&res) else {
                bail!("No such shader: {res:?}");
            };
            let kind = match res.file_stem().and_then(|s| s.to_str()) {
                Some(s) if s.ends_with("frag") => shaderc::ShaderKind::Fragment,
                Some(s) if s.ends_with("vert") => shaderc::ShaderKind::Vertex,
                Some(s) if s.ends_with("comp") => shaderc::ShaderKind::Compute,
                None | Some(_) => {
                    bail!("Unsupported shader!\n\tpath: {:?}", res);
                }
            };

            for handles in x.iter() {
                match handles {
                    Either::Left(handle) => {
                        let pip = &mut self.pipeline_arena.render.pipelines[*handle];
                        if kind == ShaderKind::Fragment {
                            pip.reload_fragment_lib(
                                &self.shader_compiler,
                                &self.pipeline_cache,
                                &res,
                            )?;
                        }
                        if kind == ShaderKind::Vertex {
                            pip.reload_vertex_lib(
                                &self.shader_compiler,
                                &self.pipeline_cache,
                                &res,
                            )?;
                        }
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
                    vk::ShaderStageFlags::VERTEX,
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
