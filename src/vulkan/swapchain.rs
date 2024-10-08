use std::{collections::VecDeque, slice, sync::Arc};

use ash::{
    khr,
    prelude::VkResult,
    vk::{self, CompositeAlphaFlagsKHR},
};

use crate::ImageDimensions;

use super::{device::Device, surface::Surface};

pub struct Frame {
    command_buffer: vk::CommandBuffer,
    image_available_semaphore: vk::Semaphore,
    render_finished_semaphore: vk::Semaphore,
    pub present_finished: vk::Fence,
    device: Arc<Device>,
}

impl Frame {
    fn destroy(&mut self, pool: &vk::CommandPool) {
        unsafe {
            self.device.destroy_fence(self.present_finished, None);
            self.device
                .destroy_semaphore(self.image_available_semaphore, None);
            self.device
                .destroy_semaphore(self.render_finished_semaphore, None);
            self.device
                .free_command_buffers(*pool, &[self.command_buffer]);
        }
    }
}

impl Frame {
    fn new(device: &Arc<Device>, command_pool: &vk::CommandPool) -> VkResult<Self> {
        let present_finished = unsafe {
            device.create_fence(
                &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::default()),
                None,
            )
        }?;
        let image_available_semaphore =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };
        let render_finished_semaphore =
            unsafe { device.create_semaphore(&vk::SemaphoreCreateInfo::default(), None)? };
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(*command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&alloc_info) }?[0];
        Ok(Self {
            command_buffer,
            image_available_semaphore,
            render_finished_semaphore,
            present_finished,
            device: device.clone(),
        })
    }
}

pub struct FrameGuard {
    frame: Frame,
    pub extent: vk::Extent2D,
    pub image_idx: usize,
    device: Arc<Device>,
}

pub struct Swapchain {
    pub images: Vec<vk::Image>,
    pub views: Vec<vk::ImageView>,
    pub frames: VecDeque<Frame>,
    command_pool: vk::CommandPool,
    pub current_image: usize,
    pub format: vk::SurfaceFormatKHR,
    pub extent: vk::Extent2D,
    pub image_dimensions: ImageDimensions,
    inner: vk::SwapchainKHR,
    loader: khr::swapchain::Device,
    device: Arc<Device>,
}

impl Swapchain {
    const SUBRANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: 1,
        base_array_layer: 0,
        layer_count: 1,
    };

    pub fn format(&self) -> vk::Format {
        self.format.format
    }

    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    pub fn new(
        device: &Arc<Device>,
        surface: &Surface,
        swapchain_loader: khr::swapchain::Device,
    ) -> VkResult<Self> {
        let info = surface.info(device);
        let capabilities = info.capabilities;
        let format = info
            .formats
            .iter()
            .find(|format| matches!(format.format, vk::Format::B8G8R8A8_UNORM))
            .unwrap_or(&info.formats[0]);

        let image_count = capabilities
            .max_image_count
            .min(3)
            .max(capabilities.min_image_count);

        let queue_family_index = [device.main_queue_family_idx];

        let mut extent = capabilities.current_extent;
        //Sadly _current_extent_ can be outside the min/max capabilities :(.
        extent.width = extent.width.min(capabilities.max_image_extent.width);
        extent.height = extent.height.min(capabilities.max_image_extent.height);

        assert!(capabilities
            .supported_composite_alpha
            .contains(CompositeAlphaFlagsKHR::OPAQUE));
        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(**surface)
            .image_format(format.format)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .image_extent(extent)
            .image_color_space(format.color_space)
            .min_image_count(image_count)
            .image_array_layers(capabilities.max_image_array_layers)
            .queue_family_indices(&queue_family_index)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);
        let swapchain = unsafe { swapchain_loader.create_swapchain(&swapchain_create_info, None)? };

        let images = unsafe { swapchain_loader.get_swapchain_images(swapchain)? };
        let views = images
            .iter()
            .map(|img| device.create_2d_view(img, format.format, 0))
            .collect::<VkResult<Vec<_>>>()?;
        views
            .iter()
            .enumerate()
            .for_each(|(i, &view)| device.name_object(view, &format!("Swapchain View {i}")));

        let frames = VecDeque::new();

        let command_pool = unsafe {
            device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                    .queue_family_index(device.main_queue_family_idx),
                None,
            )?
        };

        let memory_reqs = unsafe { device.get_image_memory_requirements(images[0]) };
        let image_dimensions =
            ImageDimensions::new(extent.width as _, extent.height as _, memory_reqs.alignment);

        Ok(Self {
            images,
            views,
            frames,
            command_pool,
            current_image: 0,
            image_dimensions,
            format: *format,
            extent,
            inner: swapchain,
            loader: swapchain_loader,
            device: device.clone(),
        })
    }

    pub fn destroy(&self) {
        for view in self.views.iter() {
            unsafe { self.device.destroy_image_view(*view, None) };
        }
        unsafe { self.loader.destroy_swapchain(self.inner, None) };
    }

    pub fn recreate(&mut self, device: &Device, surface: &Surface) -> VkResult<()> {
        let info = surface.info(device);
        let capabilities = info.capabilities;

        for view in self.views.iter() {
            unsafe { self.device.destroy_image_view(*view, None) };
        }
        let old_swapchain = self.inner;

        let queue_family_index = [device.main_queue_family_idx];

        let extent = capabilities.current_extent;
        self.extent.width = extent.width.min(capabilities.max_image_extent.width);
        self.extent.height = extent.height.min(capabilities.max_image_extent.height);

        let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
            .surface(**surface)
            .old_swapchain(old_swapchain)
            .image_format(self.format.format)
            .image_usage(
                vk::ImageUsageFlags::COLOR_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE
                    | vk::ImageUsageFlags::TRANSFER_SRC,
            )
            .image_extent(self.extent)
            .image_color_space(self.format.color_space)
            .min_image_count(self.images.len() as u32)
            .image_array_layers(capabilities.max_image_array_layers)
            .queue_family_indices(&queue_family_index)
            .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
            .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
            .composite_alpha(CompositeAlphaFlagsKHR::OPAQUE)
            .present_mode(vk::PresentModeKHR::FIFO)
            .clipped(true);
        self.inner = unsafe { self.loader.create_swapchain(&swapchain_create_info, None)? };

        unsafe { self.loader.destroy_swapchain(old_swapchain, None) };

        self.images = unsafe { self.loader.get_swapchain_images(self.inner)? };
        self.views = self
            .images
            .iter()
            .map(|img| device.create_2d_view(img, self.format.format, 0))
            .collect::<VkResult<Vec<_>>>()?;
        self.views
            .iter()
            .enumerate()
            .for_each(|(i, &view)| device.name_object(view, &format!("Swapchain View {i}")));

        let memory_reqs = unsafe { device.get_image_memory_requirements(self.images[0]) };
        self.image_dimensions =
            ImageDimensions::new(extent.width as _, extent.height as _, memory_reqs.alignment);

        Ok(())
    }

    pub fn get_current_frame(&self) -> Option<&Frame> {
        self.frames.back()
    }
    pub fn get_current_image(&self) -> &vk::Image {
        &self.images[self.current_image]
    }
    pub fn get_current_image_view(&self) -> &vk::ImageView {
        &self.views[self.current_image]
    }
    pub fn get_prev_image_idx(&self) -> usize {
        self.current_image
            .checked_sub(1)
            .unwrap_or(self.images.len() - 1)
    }
    pub fn get_prev_image(&self) -> &vk::Image {
        &self.images[self.get_prev_image_idx()]
    }
    pub fn get_prev_image_view(&self) -> &vk::ImageView {
        &self.views[self.get_prev_image_idx()]
    }

    pub fn acquire_next_image(&mut self) -> VkResult<FrameGuard> {
        self.frames.retain_mut(|frame| {
            let status = unsafe { self.device.get_fence_status(frame.present_finished) };
            // Frame finished execution
            if status == Ok(true) {
                frame.destroy(&self.command_pool);
                false
            } else {
                true
            }
        });

        let mut frame = Frame::new(&self.device, &self.command_pool)?;

        let idx = match unsafe {
            self.loader.acquire_next_image(
                self.inner,
                u64::MAX,
                frame.image_available_semaphore,
                vk::Fence::null(),
            )
        } {
            Ok((idx, false)) => idx,
            Ok((_, true)) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                frame.destroy(&self.command_pool);
                return VkResult::Err(vk::Result::ERROR_OUT_OF_DATE_KHR);
            }
            Err(e) => return Err(e),
        };

        self.current_image = idx as usize;
        unsafe {
            self.device.begin_command_buffer(
                frame.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?
        };

        let image_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .subresource_range(Self::SUBRANGE)
            .image(self.images[self.current_image])
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL);
        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&image_barrier));
        self.device
            .pipeline_barrier(&frame.command_buffer, &dependency_info);

        Ok(FrameGuard {
            frame,
            extent: self.extent,
            image_idx: self.current_image,
            device: self.device.clone(),
        })
    }

    pub fn submit_image(&mut self, frame_guard: FrameGuard) -> VkResult<()> {
        let frame = frame_guard.frame;

        let image_barrier = vk::ImageMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT)
            .dst_stage_mask(vk::PipelineStageFlags2::BOTTOM_OF_PIPE)
            .src_access_mask(vk::AccessFlags2::COLOR_ATTACHMENT_WRITE)
            .subresource_range(Self::SUBRANGE)
            .image(self.images[frame_guard.image_idx])
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::PRESENT_SRC_KHR);
        let dependency_info =
            vk::DependencyInfo::default().image_memory_barriers(slice::from_ref(&image_barrier));
        self.device
            .pipeline_barrier(&frame.command_buffer, &dependency_info);

        self.device.end_command_buffer(&frame.command_buffer)?;

        let wait_semaphores = [frame.image_available_semaphore];
        let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
        let signal_semaphores = [frame.render_finished_semaphore];
        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(&wait_semaphores)
            .wait_dst_stage_mask(&wait_stages)
            .command_buffers(slice::from_ref(&frame.command_buffer))
            .signal_semaphores(&signal_semaphores);
        unsafe {
            self.device
                .queue_submit(self.device.queue, &[submit_info], frame.present_finished)?
        };

        self.frames.push_back(frame);

        let image_indices = [frame_guard.image_idx as u32];
        let present_info = vk::PresentInfoKHR::default()
            .wait_semaphores(&signal_semaphores)
            .swapchains(slice::from_ref(&self.inner))
            .image_indices(&image_indices);
        match unsafe { self.loader.queue_present(self.device.queue, &present_info) } {
            Ok(false) => Ok(()),
            Ok(true) | Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                VkResult::Err(vk::Result::ERROR_OUT_OF_DATE_KHR)
            }
            Err(e) => Err(e),
        }
    }
}

impl Drop for Swapchain {
    fn drop(&mut self) {
        unsafe {
            for view in self.views.iter() {
                self.device.destroy_image_view(*view, None);
            }
            self.loader.destroy_swapchain(self.inner, None);
            self.frames
                .iter_mut()
                .for_each(|f| f.destroy(&self.command_pool));
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

impl FrameGuard {
    pub fn command_buffer(&self) -> &vk::CommandBuffer {
        &self.frame.command_buffer
    }

    pub fn begin_rendering(
        &mut self,
        view: &vk::ImageView,
        load_op: vk::AttachmentLoadOp,
        color: [f32; 4],
    ) {
        let clear_color = vk::ClearValue {
            color: vk::ClearColorValue { float32: color },
        };
        let color_attachments = [vk::RenderingAttachmentInfo::default()
            .image_view(*view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .resolve_image_layout(vk::ImageLayout::PRESENT_SRC_KHR)
            .load_op(load_op)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(clear_color)];
        let rendering_info = vk::RenderingInfo::default()
            .render_area(self.extent.into())
            .layer_count(1)
            .color_attachments(&color_attachments);
        self.device
            .begin_rendering(self.command_buffer(), &rendering_info);

        let viewport = vk::Viewport {
            x: 0.0,
            y: self.extent.height as f32,
            width: self.extent.width as f32,
            height: -(self.extent.height as f32),
            min_depth: 0.0,
            max_depth: 1.0,
        };
        self.set_viewports(&[viewport]);
        self.set_scissors(&[vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: self.extent,
        }]);
    }

    pub fn draw(
        &self,
        vertex_count: u32,
        instance_count: u32,
        first_vertex: u32,
        first_instance: u32,
    ) {
        self.device.draw(
            self.command_buffer(),
            vertex_count,
            instance_count,
            first_vertex,
            first_instance,
        );
    }

    pub fn draw_indexed(
        &self,
        index_count: u32,
        instance_count: u32,
        first_index: u32,
        vertex_offset: i32,
        first_instance: u32,
    ) {
        self.device.draw_indexed(
            self.command_buffer(),
            index_count,
            instance_count,
            first_index,
            vertex_offset,
            first_instance,
        );
    }

    pub fn bind_index_buffer(&self, buffer: vk::Buffer, offset: u64) {
        self.device
            .bind_index_buffer(self.command_buffer(), buffer, offset);
    }

    pub fn bind_vertex_buffer(&self, buffer: vk::Buffer) {
        self.device
            .bind_vertex_buffer(self.command_buffer(), buffer);
    }

    pub fn bind_descriptor_sets(
        &self,
        bind_point: vk::PipelineBindPoint,
        pipeline_layout: vk::PipelineLayout,
        descriptor_sets: &[vk::DescriptorSet],
    ) {
        self.device.bind_descriptor_sets(
            self.command_buffer(),
            bind_point,
            pipeline_layout,
            descriptor_sets,
        );
    }

    pub fn bind_push_constants<T>(
        &self,
        pipeline_layout: vk::PipelineLayout,
        stages: vk::ShaderStageFlags,
        data: &[T],
    ) {
        self.device
            .bind_push_constants(self.command_buffer(), pipeline_layout, stages, data);
    }

    pub fn set_viewports(&self, viewports: &[vk::Viewport]) {
        self.device.set_viewports(self.command_buffer(), viewports)
    }

    pub fn set_scissors(&self, viewports: &[vk::Rect2D]) {
        self.device.set_scissors(self.command_buffer(), viewports)
    }

    pub fn bind_pipeline(&self, bind_point: vk::PipelineBindPoint, pipeline: &vk::Pipeline) {
        self.device
            .bind_pipeline(self.command_buffer(), bind_point, pipeline)
    }

    pub fn dispatch(&self, x: u32, y: u32, z: u32) {
        self.device.dispatch(self.command_buffer(), x, y, z)
    }

    pub fn end_rendering(&mut self) {
        self.device.end_rendering(self.command_buffer());
    }
}
