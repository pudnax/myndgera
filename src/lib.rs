#![allow(clippy::new_without_default)]
#![allow(clippy::too_many_arguments)]

pub mod default_shaders;
mod input;
mod recorder;
mod shader_compiler;
mod utils;
mod vulkan;
mod watcher;

use either::Either;
use std::{
    io::Write,
    path::{Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};
use winit::{
    application::ApplicationHandler,
    dpi::{LogicalSize, PhysicalPosition, PhysicalSize},
    event::{ElementState, KeyEvent, MouseButton, StartCause, WindowEvent},
    event_loop::EventLoopProxy,
    keyboard::{Key, NamedKey},
    window::{Window, WindowAttributes},
};

pub use self::{
    input::Input,
    recorder::{RecordEvent, Recorder},
    shader_compiler::ShaderCompiler,
    utils::*,
    vulkan::*,
    watcher::Watcher,
};

use anyhow::bail;
use anyhow::Result;
use ash::{khr, vk};

pub const UPDATES_PER_SECOND: u32 = 60;
pub const FIXED_TIME_STEP: f64 = 1. / UPDATES_PER_SECOND as f64;
pub const MAX_FRAME_TIME: f64 = 15. * FIXED_TIME_STEP; // 0.25;

pub const SHADER_DUMP_FOLDER: &str = "shader_dump";
pub const SHADER_FOLDER: &str = "shaders";
pub const VIDEO_FOLDER: &str = "recordings";
pub const SCREENSHOT_FOLDER: &str = "screenshots";

pub const COLOR_SUBRESOURCE_MASK: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: vk::REMAINING_MIP_LEVELS,
    base_array_layer: 0,
    layer_count: vk::REMAINING_ARRAY_LAYERS,
};

pub struct RenderContext<'a> {
    pub window: &'a mut Window,
    pub device: &'a Device,
    pub swapchain: &'a mut Swapchain,
    pub texture_arena: &'a mut TextureArena,
    pub pipeline_arena: &'a mut PipelineArena,
    pub queue: &'a vk::Queue,
    pub transfer_queue: &'a vk::Queue,
    pub stats: &'a mut GlobalStats,
}

pub trait Example: 'static + Sized {
    fn name() -> &'static str {
        "Myndgera"
    }
    fn init(ctx: &mut RenderContext) -> Result<Self>;
    fn resize(&mut self, _ctx: &mut RenderContext) -> Result<()> {
        Ok(())
    }
    fn update(&mut self, _ctx: &mut RenderContext) -> Result<()> {
        Ok(())
    }
    fn render(&mut self, _ctx: &mut RenderContext, _frame: &mut FrameGuard) -> Result<()> {
        Ok(())
    }
    fn input(&mut self, _key_event: &KeyEvent) {}
}

#[allow(dead_code)]
struct AppInit<E> {
    example: E,
    window: Window,
    input: Input,

    pause: bool,
    timeline: Instant,
    backup_time: Duration,
    frame_instant: Instant,
    frame_accumulated_time: f64,

    texture_arena: TextureArena,

    file_watcher: Watcher,
    recorder: Recorder,
    video_recording: bool,
    record_time: Option<Duration>,

    stats: GlobalStats,
    pipeline_arena: PipelineArena,

    queue: vk::Queue,
    transfer_queue: vk::Queue,

    swapchain: Swapchain,
    surface: Surface,
    device: Arc<Device>,
    instance: Instance,
}

impl<E: Example> AppInit<E> {
    fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        proxy: EventLoopProxy<UserEvent>,
        window_attributes: WindowAttributes,
        record_time: Option<Duration>,
    ) -> Result<Self> {
        let mut window = event_loop.create_window(window_attributes)?;
        let watcher = Watcher::new(proxy)?;
        let mut recorder = Recorder::new();

        let instance = Instance::new(Some(&window))?;
        let surface = instance.create_surface(&window)?;
        let (device, queue, transfer_queue) = instance.create_device_and_queues(&surface)?;
        let device = Arc::new(device);

        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);
        let mut swapchain = Swapchain::new(&device, &surface, swapchain_loader)?;

        let mut pipeline_arena = PipelineArena::new(&device, watcher.clone())?;

        let extent = swapchain.extent();
        let video_recording = record_time.is_some();
        let mut stats = GlobalStats {
            wh: [extent.width as f32, extent.height as f32],
            record_time: record_time.map(|t| t.as_secs_f32()).unwrap_or(10.),
            ..Default::default()
        };

        let mut texture_arena = TextureArena::new(&device, &swapchain, &queue)?;

        let mut ctx = RenderContext {
            window: &mut window,
            device: &device,
            swapchain: &mut swapchain,
            texture_arena: &mut texture_arena,
            pipeline_arena: &mut pipeline_arena,
            queue: &queue,
            transfer_queue: &transfer_queue,
            stats: &mut stats,
        };
        let example = E::init(&mut ctx)?;

        if record_time.is_some() {
            let mut image_dimensions = swapchain.image_dimensions;
            image_dimensions.width = align_to(image_dimensions.width, 2);
            image_dimensions.height = align_to(image_dimensions.height, 2);
            recorder.start(image_dimensions);
        }

        Ok(Self {
            example,
            window,
            input: Input::default(),

            pause: false,
            timeline: Instant::now(),
            backup_time: Duration::from_secs(0),
            frame_instant: Instant::now(),
            frame_accumulated_time: 0.,

            texture_arena,

            file_watcher: watcher,
            video_recording,
            record_time,
            recorder,

            stats,
            pipeline_arena,

            queue,
            transfer_queue,

            surface,
            swapchain,
            device,
            instance,
        })
    }

    fn update(&mut self) {
        self.input.process_position(&mut self.stats.pos);
        let mut ctx = RenderContext {
            window: &mut self.window,
            swapchain: &mut self.swapchain,
            device: &self.device,
            texture_arena: &mut self.texture_arena,
            pipeline_arena: &mut self.pipeline_arena,
            queue: &mut self.queue,
            transfer_queue: &mut self.transfer_queue,
            stats: &mut self.stats,
        };
        let _ = self
            .example
            .update(&mut ctx)
            .map_err(|err| log::error!("{err}"));
    }

    fn reload_shaders(&mut self, path: PathBuf) -> Result<()> {
        if let Some(frame) = self.swapchain.get_current_frame() {
            let fences = std::slice::from_ref(&frame.present_finished);
            unsafe { self.device.wait_for_fences(fences, true, u64::MAX)? };
        }

        let resolved = {
            let mapping = self.file_watcher.include_mapping.lock();
            mapping[Path::new(path.file_name().unwrap())].clone()
        };

        for ShaderSource { path, kind } in resolved {
            let handles = &self.pipeline_arena.path_mapping[&path];
            for handle in handles {
                let compiler = &self.pipeline_arena.shader_compiler;
                match handle {
                    Either::Left(handle) => {
                        let pipeline = &mut self.pipeline_arena.render.pipelines[*handle];
                        match kind {
                            ShaderKind::Vertex => pipeline.reload_vertex_lib(compiler, &path),
                            ShaderKind::Fragment => pipeline.reload_fragment_lib(compiler, &path),
                            ShaderKind::Compute => {
                                bail!("Supplied compute shader into the render pipeline!")
                            }
                        }?;
                        pipeline.link()?;
                    }
                    Either::Right(handle) => {
                        let pipeline = &mut self.pipeline_arena.compute.pipelines[*handle];
                        pipeline.reload(compiler)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        if let Some(frame) = self.swapchain.get_current_frame() {
            let fences = std::slice::from_ref(&frame.present_finished);
            unsafe { self.device.wait_for_fences(fences, true, u64::MAX)? };
        }

        self.swapchain
            .recreate(&self.device, &self.surface)
            .expect("Failed to recreate swapchain");
        let extent = self.swapchain.extent();
        self.stats.wh = [extent.width as f32, extent.height as f32];

        for (idx, view) in self.swapchain.views.iter().enumerate() {
            self.texture_arena.update_storage_image(idx as u32, view)
        }

        for i in SCREENSIZED_IMAGE_INDICES {
            if let Some(info) = &mut self.texture_arena.infos[i] {
                info.extent.width = extent.width;
                info.extent.height = extent.height;
            }
        }
        self.texture_arena
            .update_images_by_idx(&SCREENSIZED_IMAGE_INDICES)?;

        let mut ctx = RenderContext {
            window: &mut self.window,
            device: &mut self.device,
            swapchain: &mut self.swapchain,
            texture_arena: &mut self.texture_arena,
            pipeline_arena: &mut self.pipeline_arena,
            queue: &mut self.queue,
            transfer_queue: &mut self.transfer_queue,
            stats: &mut self.stats,
        };
        self.example.resize(&mut ctx)?;

        Ok(())
    }
}

impl<E: Example> ApplicationHandler<UserEvent> for AppInit<E> {
    fn new_events(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        self.stats.time = if !self.pause {
            self.timeline.elapsed().as_secs_f32()
        } else {
            self.backup_time.as_secs_f32()
        };
        if let StartCause::WaitCancelled { .. } = cause {
            let new_instant = Instant::now();
            let frame_time = new_instant
                .duration_since(self.frame_instant)
                .as_secs_f64()
                .min(MAX_FRAME_TIME);
            self.frame_instant = new_instant;
            self.stats.time_delta = frame_time as _;

            self.frame_accumulated_time += frame_time;
            while self.frame_accumulated_time >= FIXED_TIME_STEP {
                self.update();

                self.frame_accumulated_time -= FIXED_TIME_STEP;
            }
        }

        if let Some(limit) = self.record_time {
            if self.timeline.elapsed() >= limit && self.recorder.is_active() {
                self.recorder.finish();
                event_loop.exit();
            }
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let winit::event::DeviceEvent::Key(key_event) = event {
            self.input.update_device_input(key_event);
        }
    }

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
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => event_loop.exit(),

            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: Key::Named(key),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => {
                let dt = Duration::from_secs_f32(1. / 60.);
                match key {
                    NamedKey::F1 => print_help(),
                    NamedKey::F2 => {
                        if !self.pause {
                            self.backup_time = self.timeline.elapsed();
                        } else {
                            self.timeline = Instant::now() - self.backup_time;
                        }
                        self.pause = !self.pause;
                    }
                    NamedKey::F3 => {
                        if !self.pause {
                            self.backup_time = self.timeline.elapsed();
                            self.pause = true;
                        }
                        self.backup_time = self.backup_time.saturating_sub(dt);
                    }
                    NamedKey::F4 => {
                        if !self.pause {
                            self.backup_time = self.timeline.elapsed();
                            self.pause = true;
                        }
                        self.backup_time += dt;
                    }
                    NamedKey::F5 => {
                        self.stats.pos = [0.; 3];
                        self.stats.time = 0.;
                        self.stats.frame = 0;
                        self.timeline = Instant::now();
                        self.backup_time = self.timeline.elapsed();
                    }
                    NamedKey::F6 => {
                        println!("{}", self.stats);
                    }
                    NamedKey::F10 => {
                        let _ = save_shaders(SHADER_FOLDER).map_err(|err| log::error!("{err}"));
                    }
                    NamedKey::F11 => {
                        let _ = self
                            .device
                            .capture_image_data(
                                &self.queue,
                                self.swapchain.get_current_image(),
                                self.swapchain.extent(),
                                |tex| self.recorder.screenshot(tex),
                            )
                            .map_err(|err| log::error!("{err}"));
                    }
                    NamedKey::F12 => {
                        if !self.video_recording {
                            let mut image_dimensions = self.swapchain.image_dimensions;
                            image_dimensions.width = align_to(image_dimensions.width, 2);
                            image_dimensions.height = align_to(image_dimensions.height, 2);
                            self.recorder.start(image_dimensions);
                        } else {
                            self.recorder.finish();
                        }
                        self.video_recording = !self.video_recording;
                    }
                    _ => {}
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.input.update_window_input(&event);
                self.example.input(&event);
            }

            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.stats.mouse_pressed = (ElementState::Pressed == state) as u32;
            }
            WindowEvent::CursorMoved {
                position: PhysicalPosition { x, y },
                ..
            } => {
                if !self.pause {
                    let PhysicalSize { width, height } = self.window.inner_size();
                    let x = (x as f32 / width as f32 - 0.5) * 2.;
                    let y = -(y as f32 / height as f32 - 0.5) * 2.;
                    self.stats.mouse = [x, y];
                }
            }
            WindowEvent::RedrawRequested => {
                let mut frame = match self.swapchain.acquire_next_image() {
                    Ok(frame) => frame,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        let _ = self.recreate_swapchain().map_err(|err| log::warn!("{err}"));
                        self.window.request_redraw();
                        return;
                    }
                    Err(e) => panic!("error: {e}\n"),
                };

                {
                    let mut ctx = RenderContext {
                        window: &mut self.window,
                        swapchain: &mut self.swapchain,
                        device: &mut self.device,
                        texture_arena: &mut self.texture_arena,
                        pipeline_arena: &mut self.pipeline_arena,
                        queue: &mut self.queue,
                        transfer_queue: &mut self.transfer_queue,
                        stats: &mut self.stats,
                    };
                    let _ = self
                        .example
                        .render(&mut ctx, &mut frame)
                        .map_err(|err| log::error!("{err}"));
                }

                self.device.blit_image(
                    frame.command_buffer(),
                    self.swapchain.get_current_image(),
                    self.swapchain.extent(),
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    &self.texture_arena.images[PREV_FRAME_IDX],
                    self.swapchain.extent(),
                    vk::ImageLayout::UNDEFINED,
                );

                match self.swapchain.submit_image(&self.queue, frame) {
                    Ok(_) => {}
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        let _ = self.recreate_swapchain().map_err(|err| log::warn!("{err}"));
                    }
                    Err(e) => panic!("error: {e}\n"),
                }

                self.window.request_redraw();

                if self.video_recording && self.recorder.ffmpeg_installed() {
                    let res = self.device.capture_image_data(
                        &self.queue,
                        self.swapchain.get_current_image(),
                        self.swapchain.extent(),
                        |tex| self.recorder.record(tex),
                    );
                    if let Err(err) = res {
                        log::error!("{err}");
                        self.video_recording = false;
                    }
                }

                self.stats.frame = self.stats.frame.saturating_add(1);
            }
            _ => {}
        }
    }

    fn user_event(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Glsl { path } => {
                match self.reload_shaders(path) {
                    Err(err) => eprintln!("{err}"),
                    Ok(()) => {
                        const ESC: &str = "\x1B[";
                        const RESET: &str = "\x1B[0m";
                        eprint!("\r{}42m{}K{}\r", ESC, ESC, RESET);
                        std::io::stdout().flush().unwrap();
                        std::thread::spawn(|| {
                            std::thread::sleep(std::time::Duration::from_millis(50));
                            eprint!("\r{}40m{}K{}\r", ESC, ESC, RESET);
                            std::io::stdout().flush().unwrap();
                        });
                    }
                };
            }
        }
    }

    fn exiting(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        self.recorder.close_thread();
        if let Some(handle) = self.recorder.thread_handle.take() {
            let _ = handle.join();
        }
        let _ = unsafe { self.device.device_wait_idle() };
        println!("// End from the loop. Bye bye~⏎ ");
    }

    fn resumed(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop) {
        panic!("On native platforms `resumed` can be called only once.")
    }
}

pub struct App<E> {
    proxy: EventLoopProxy<UserEvent>,
    inner: AppEnum<E>,
}

impl<E> App<E> {
    pub fn new(proxy: EventLoopProxy<UserEvent>) -> Self {
        Self {
            proxy,
            inner: AppEnum::Uninitialized,
        }
    }
}

#[derive(Default)]
enum AppEnum<E> {
    #[default]
    Uninitialized,
    Init(AppInit<E>),
}

impl<E: Example> ApplicationHandler<UserEvent> for App<E> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        env_logger::init();

        let Args {
            record_time,
            inner_size,
        } = parse_args().unwrap();

        let mut window_attributes = WindowAttributes::default().with_title(E::name());
        if let Some(size) = inner_size {
            window_attributes = window_attributes
                .with_resizable(false)
                .with_inner_size(LogicalSize::<u32>::from(size));
        }
        match self.inner {
            AppEnum::Uninitialized => {
                let app = AppInit::new(
                    event_loop,
                    self.proxy.clone(),
                    window_attributes,
                    record_time,
                )
                .expect("Failed to create application");

                println!("{}", app.device.get_info());
                println!("{}", app.recorder.ffmpeg_version);
                println!(
                    "Default shader path:\n\t{}",
                    Path::new(SHADER_FOLDER).canonicalize().unwrap().display()
                );
                print_help();

                println!("// Set up our new world⏎ ");
                println!("// And let's begin the⏎ ");
                println!("\tSIMULATION⏎ \n");

                self.inner = AppEnum::Init(app);
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
