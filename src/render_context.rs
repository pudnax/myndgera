use crate::vulkan::{self, Device, Instance, Surface, Swapchain};
use anyhow::Result;
use ash::khr::{self};
use std::sync::Arc;
use winit::{dpi::PhysicalSize, window::Window};

pub struct RenderContext {
    pub(crate) is_swapchain_dirty: bool,
    pub swapchain: Swapchain,
    pub surface: Surface,

    pub device: Arc<Device>,
    _instance: Instance,
    pub window: Window,
}

impl RenderContext {
    pub fn new(window: Window) -> Result<Self> {
        let instance = Instance::new(Some(&window))?;

        let surface = Surface::new(&instance, &window)?;

        let (device, _transfer_queue) = instance.create_device_and_queues(&surface)?;
        let device = Arc::new(device);

        let PhysicalSize { width, height } = window.inner_size();
        let swapchain_loader = khr::swapchain::Device::new(&instance, &device);
        let swapchain = vulkan::Swapchain::new(&device, &surface, swapchain_loader, width, height)?;

        Ok(Self {
            window,
            _instance: instance,
            surface,
            device,
            swapchain,
            is_swapchain_dirty: false,
        })
    }
}
