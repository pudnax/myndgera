#![allow(clippy::new_without_default)]
#![allow(clippy::too_many_arguments)]

mod camera;
pub mod default_shaders;
mod input;
mod recorder;
mod shader_compiler;
mod utils;
mod vulkan;
mod watcher;

use dolly::drivers::YawPitch;
use either::Either;
use glam::{vec2, vec3, Vec3};
use gpu_alloc::UsageFlags;
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

use self::camera::{Camera, CameraUniform};
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

pub trait Example: 'static + Sized {
    fn name() -> &'static str {
        "Myndgera"
    }
    fn init(_ctx: &RenderContext, _state: &mut AppState) -> Result<Self>;
    fn resize(&mut self, _ctx: &RenderContext, _state: &mut AppState) -> Result<()> {
        Ok(())
    }
    fn update(
        &mut self,
        _ctx: &RenderContext,
        _state: &mut AppState,
        _cbuff: &vk::CommandBuffer,
    ) -> Result<()> {
        Ok(())
    }
    fn render(
        &mut self,
        _ctx: &RenderContext,
        _state: &mut AppState,
        _frame: &mut FrameGuard,
    ) -> Result<()> {
        Ok(())
    }
    fn input(&mut self, _state: &AppState) {}
}

pub struct AppState {
    pub input: Input,
    pub camera: Camera,
    pub camera_uniform: BufferTyped<CameraUniform>,

    pub pause: bool,
    pub timeline: Instant,
    backup_time: Duration,
    frame_instant: Instant,
    frame_accumulated_time: f64,

    pub texture_arena: TextureArena,

    recorder: Recorder,
    video_recording: bool,
    record_time: Option<Duration>,

    pub stats: GlobalStats,
    pub pipeline_arena: PipelineArena,
}

impl AppState {
    pub fn new(
        ctx: &RenderContext,
        proxy: EventLoopProxy<UserEvent>,
        record_time: Option<Duration>,
    ) -> anyhow::Result<Self> {
        let watcher = Watcher::new(proxy)?;
        let recorder = Recorder::new();

        let pipeline_arena = PipelineArena::new(&ctx.device, watcher.clone())?;

        let extent = ctx.swapchain.extent();
        let video_recording = record_time.is_some();
        let stats = GlobalStats {
            wh: [extent.width as f32, extent.height as f32],
            record_time: record_time.map(|t| t.as_secs_f32()).unwrap_or(10.),
            ..Default::default()
        };

        let texture_arena = TextureArena::new(&ctx.device, &ctx.swapchain, &ctx.queue)?;
        let camera = Camera::new(vec3(0., -10., 18.), 0., 0.);
        let camera_uniform = ctx.device.create_buffer_typed(
            vk::BufferUsageFlags::TRANSFER_DST | vk::BufferUsageFlags::UNIFORM_BUFFER,
            UsageFlags::FAST_DEVICE_ACCESS,
        )?;

        Ok(Self {
            stats,
            input: Input::new(),
            camera,
            camera_uniform,

            pause: false,
            timeline: Instant::now(),
            backup_time: Duration::from_secs(0),
            frame_instant: Instant::now(),
            frame_accumulated_time: 0.,

            texture_arena,

            recorder,
            video_recording,
            record_time,

            pipeline_arena,
        })
    }

    fn update(&mut self, ctx: &RenderContext, &cbuff: &vk::CommandBuffer) -> Result<()> {
        self.input.process_position(&mut self.stats.pos);

        if self.input.mouse_state.left_held() {
            let sensitivity = 0.5;
            self.camera.rig.driver_mut::<YawPitch>().rotate_yaw_pitch(
                -sensitivity * self.input.mouse_state.delta.x,
                -sensitivity * self.input.mouse_state.delta.y,
            );
        }

        let dt = self.stats.time_delta;
        let move_right = self.input.move_right - self.input.move_left;
        let move_up = self.input.move_up - self.input.move_down;
        let move_fwd = self.input.move_backward - self.input.move_forward;

        let rotation: glam::Quat = self.camera.rig.final_transform.rotation.into();
        let move_vec = rotation
            * Vec3::new(move_right, move_up, move_fwd).clamp_length_max(1.0)
            * 4.0f32.powf(self.input.boost);

        self.camera
            .rig
            .driver_mut::<dolly::drivers::Position>()
            .translate(move_vec * dt * 5.0);

        self.camera.rig.update(dt as _);

        self.camera.position = self.camera.rig.final_transform.position.into();
        self.camera.rotation = self.camera.rig.final_transform.rotation.into();

        let mut staging = ctx.device.create_buffer_typed::<CameraUniform>(
            vk::BufferUsageFlags::TRANSFER_SRC,
            UsageFlags::UPLOAD | UsageFlags::TRANSIENT,
        )?;
        let mapped = staging.map_memory()?;
        *mapped = self.camera.get_uniform();
        let region = vk::BufferCopy2::default().size(std::mem::size_of::<CameraUniform>() as _);
        let copy_info = vk::CopyBufferInfo2::default()
            .src_buffer(staging.buffer)
            .dst_buffer(self.camera_uniform.buffer)
            .regions(std::slice::from_ref(&region));
        unsafe { ctx.device.cmd_copy_buffer2(cbuff, &copy_info) };

        Ok(())
    }

    fn reload_shaders(
        &mut self,
        RenderContext {
            swapchain, device, ..
        }: &RenderContext,
        path: PathBuf,
    ) -> Result<()> {
        if let Some(frame) = swapchain.get_current_frame() {
            let fences = std::slice::from_ref(&frame.present_finished);
            unsafe { device.wait_for_fences(fences, true, u64::MAX)? };
        }

        let resolved = {
            let mapping = self.pipeline_arena.file_watcher.include_mapping.lock();
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
}

#[allow(dead_code)]
pub struct RenderContext {
    pub window: Window,

    pub queue: vk::Queue,
    pub transfer_queue: vk::Queue,

    pub swapchain: Swapchain,
    pub surface: Surface,
    pub device: Arc<Device>,
    pub instance: Instance,
}

impl RenderContext {
    pub fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_attributes: WindowAttributes,
    ) -> Result<Self> {
        let window = event_loop.create_window(window_attributes)?;

        let instance = Instance::new(Some(&window))?;
        let surface = instance.create_surface(&window)?;
        let (device, queue, transfer_queue) = instance.create_device_and_queues(&surface)?;
        let device = Arc::new(device);

        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);
        let swapchain = Swapchain::new(&device, &surface, swapchain_loader)?;

        Ok(Self {
            window,
            queue,
            transfer_queue,
            swapchain,
            surface,
            device,
            instance,
        })
    }
}

struct AppInit<E> {
    example: E,
    state: AppState,
    update_fence: vk::Fence,
    device: Arc<Device>,
    render: RenderContext,
}

impl<E> Drop for AppInit<E> {
    fn drop(&mut self) {
        unsafe { self.device.destroy_fence(self.update_fence, None) };
    }
}

impl<E: Example> AppInit<E> {
    pub fn new(
        event_loop: &winit::event_loop::ActiveEventLoop,
        proxy: EventLoopProxy<UserEvent>,
        window_attributes: WindowAttributes,
        record_time: Option<Duration>,
    ) -> anyhow::Result<Self> {
        let render = RenderContext::new(event_loop, window_attributes)?;
        let mut state = AppState::new(&render, proxy, record_time)?;
        let example = E::init(&render, &mut state)?;

        let update_fence = unsafe {
            render
                .device
                .create_fence(&vk::FenceCreateInfo::default(), None)?
        };

        Ok(Self {
            update_fence,
            device: render.device.clone(),
            render,
            state,
            example,
        })
    }

    pub fn update(&mut self) -> Result<()> {
        let state = &mut self.state;
        let new_instant = Instant::now();
        let frame_time = new_instant
            .duration_since(state.frame_instant)
            .as_secs_f64()
            .min(MAX_FRAME_TIME);
        state.frame_instant = new_instant;
        state.stats.time_delta = frame_time as _;

        state.frame_accumulated_time += frame_time;
        while state.frame_accumulated_time >= FIXED_TIME_STEP {
            self.device
                .one_time_submit(&self.render.queue, |_, cbuff| {
                    state.update(&self.render, &cbuff)?;
                    self.example.update(&self.render, state, &cbuff)?;
                    Ok(())
                })?;

            state.input.mouse_state.refresh();

            state.frame_accumulated_time -= FIXED_TIME_STEP;
        }
        Ok(())
    }

    fn recreate_swapchain(&mut self) -> Result<()> {
        if let Some(frame) = self.render.swapchain.get_current_frame() {
            let fences = std::slice::from_ref(&frame.present_finished);
            unsafe { self.render.device.wait_for_fences(fences, true, u64::MAX)? };
        }

        self.render
            .swapchain
            .recreate(&self.device, &self.render.surface)
            .expect("Failed to recreate swapchain");
        let extent = self.render.swapchain.extent();
        self.state.stats.wh = [extent.width as f32, extent.height as f32];
        self.state.camera.aspect = extent.width as f32 / extent.height as f32;

        for (idx, view) in self.render.swapchain.views.iter().enumerate() {
            self.state
                .texture_arena
                .update_storage_image(self.state.texture_arena.external_storage_img_idx[idx], view);
            self.state
                .texture_arena
                .update_sampled_image(self.state.texture_arena.external_sampled_img_idx[idx], view)
        }

        for i in SCREENSIZED_IMAGE_INDICES {
            if let Some(info) = &mut self.state.texture_arena.infos[i] {
                info.extent.width = extent.width;
                info.extent.height = extent.height;
            }
        }
        self.state
            .texture_arena
            .update_sampled_images_by_idx(&SCREENSIZED_IMAGE_INDICES)?;

        self.example.resize(&self.render, &mut self.state)?;

        Ok(())
    }
}

impl<E: Example> ApplicationHandler<UserEvent> for AppInit<E> {
    fn new_events(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        cause: winit::event::StartCause,
    ) {
        self.state.stats.time = if !self.state.pause {
            self.state.timeline.elapsed().as_secs_f32()
        } else {
            self.state.backup_time.as_secs_f32()
        };
        if let StartCause::WaitCancelled { .. } = cause {
            let _ = self.update().map_err(|err| log::error!("{err}"));
        }

        if let Some(limit) = self.state.record_time {
            if self.state.timeline.elapsed() >= limit && self.state.recorder.is_active() {
                self.state.recorder.finish();
                event_loop.exit();
            }
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
                let state = &mut self.state;
                match key {
                    NamedKey::F1 => print_help(),
                    NamedKey::F2 => {
                        if !state.pause {
                            state.backup_time = state.timeline.elapsed();
                        } else {
                            state.timeline = Instant::now() - state.backup_time;
                        }
                        state.pause = !state.pause;
                    }
                    NamedKey::F3 => {
                        if !state.pause {
                            state.backup_time = state.timeline.elapsed();
                            state.pause = true;
                        }
                        state.backup_time = state.backup_time.saturating_sub(dt);
                    }
                    NamedKey::F4 => {
                        if !state.pause {
                            state.backup_time = state.timeline.elapsed();
                            state.pause = true;
                        }
                        state.backup_time += dt;
                    }
                    NamedKey::F5 => {
                        state.stats.pos = [0.; 3];
                        state.stats.time = 0.;
                        state.stats.frame = 0;
                        state.timeline = Instant::now();
                        state.backup_time = state.timeline.elapsed();
                    }
                    NamedKey::F6 => {
                        println!("{}", state.stats);
                    }
                    NamedKey::F10 => {
                        let _ = save_shaders(SHADER_FOLDER).map_err(|err| log::error!("{err}"));
                    }
                    NamedKey::F11 => {
                        let _ = self
                            .device
                            .capture_image_data(
                                &self.render.queue,
                                self.render.swapchain.get_current_image(),
                                self.render.swapchain.extent(),
                                |tex| state.recorder.screenshot(tex),
                            )
                            .map_err(|err| log::error!("{err}"));
                    }
                    NamedKey::F12 => {
                        if !state.video_recording {
                            let mut image_dimensions = self.render.swapchain.image_dimensions;
                            image_dimensions.width = align_to(image_dimensions.width, 2);
                            image_dimensions.height = align_to(image_dimensions.height, 2);
                            state.recorder.start(image_dimensions);
                        } else {
                            state.recorder.finish();
                        }
                        state.video_recording = !state.video_recording;
                    }
                    _ => {}
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                self.state.input.update_window_input(&event);
                self.example.input(&self.state);
            }

            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => {
                self.state.stats.mouse_pressed = (ElementState::Pressed == state) as u32;
            }
            WindowEvent::CursorMoved {
                position: PhysicalPosition { x, y },
                ..
            } => {
                if !self.state.pause {
                    let PhysicalSize { width, height } = self.render.window.inner_size();
                    let x = (x as f32 / width as f32 - 0.5) * 2.;
                    let y = -(y as f32 / height as f32 - 0.5) * 2.;
                    self.state.stats.mouse = [x, y];
                    self.state.input.mouse_state.screen_position = vec2(x, y);
                }
            }
            WindowEvent::RedrawRequested => {
                let mut frame = match self.render.swapchain.acquire_next_image() {
                    Ok(frame) => frame,
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        let _ = self.recreate_swapchain().map_err(|err| log::warn!("{err}"));
                        self.render.window.request_redraw();
                        return;
                    }
                    Err(e) => panic!("error: {e}\n"),
                };

                {
                    let _ = self
                        .example
                        .render(&self.render, &mut self.state, &mut frame)
                        .map_err(|err| log::error!("{err}"));
                }

                self.device.blit_image(
                    frame.command_buffer(),
                    self.render.swapchain.get_current_image(),
                    self.render.swapchain.extent(),
                    vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    &self.state.texture_arena.images[PREV_FRAME_IDX],
                    self.render.swapchain.extent(),
                    vk::ImageLayout::UNDEFINED,
                );

                match self
                    .render
                    .swapchain
                    .submit_image(&self.render.queue, frame)
                {
                    Ok(_) => {}
                    Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                        let _ = self.recreate_swapchain().map_err(|err| log::warn!("{err}"));
                    }
                    Err(e) => panic!("error: {e}\n"),
                }

                self.render.window.request_redraw();

                if self.state.video_recording && self.state.recorder.ffmpeg_installed() {
                    let res = self.device.capture_image_data(
                        &self.render.queue,
                        self.render.swapchain.get_current_image(),
                        self.render.swapchain.extent(),
                        |tex| self.state.recorder.record(tex),
                    );
                    if let Err(err) = res {
                        log::error!("{err}");
                        self.state.video_recording = false;
                    }
                }

                self.state.stats.frame = self.state.stats.frame.saturating_add(1);
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        self.state.input.update_device_input(event);
    }

    fn user_event(&mut self, _event_loop: &winit::event_loop::ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::Glsl { path } => {
                match self.state.reload_shaders(&self.render, path) {
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
        self.state.recorder.close_thread();
        if let Some(handle) = self.state.recorder.thread_handle.take() {
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
                println!("{}", app.state.recorder.ffmpeg_version);
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
