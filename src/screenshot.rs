use std::{
    fs::File,
    io::{BufWriter, Write},
    path::Path,
    time::Instant,
};

use crate::Device;
use anyhow::Result;
use ash::vk;

pub const SCREENSHOTS_FOLDER: &str = "screenshots";

pub fn make_screenshot(
    device: &Device,
    queue: &vk::Queue,
    src_image: &vk::Image,
    extent: vk::Extent2D,
) -> Result<()> {
    let now = std::time::Instant::now();
    let dst_image = unsafe {
        device.create_image(
            &vk::ImageCreateInfo::default()
                .extent(vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                })
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::R8G8B8A8_SRGB)
                .usage(vk::ImageUsageFlags::TRANSFER_DST)
                .samples(vk::SampleCountFlags::TYPE_1)
                .mip_levels(1)
                .array_layers(1)
                .tiling(vk::ImageTiling::LINEAR),
            None,
        )?
    };
    let memory_reqs = unsafe { device.get_image_memory_requirements(dst_image) };
    let memory = device.alloc_memory(memory_reqs, vk::MemoryPropertyFlags::HOST_VISIBLE)?;
    unsafe { device.bind_image_memory(dst_image, memory, 0) }?;

    let fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None)? };

    let command_pool = unsafe {
        device.create_command_pool(
            &vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                .queue_family_index(device.main_queue_family_idx),
            None,
        )?
    };
    let command_buffer = unsafe {
        device.allocate_command_buffers(
            &vk::CommandBufferAllocateInfo::default()
                .command_pool(command_pool)
                .command_buffer_count(1)
                .level(vk::CommandBufferLevel::PRIMARY),
        )?[0]
    };

    unsafe {
        device.begin_command_buffer(
            command_buffer,
            &vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
        )?
    };

    device.blit_image(&command_buffer, src_image, extent, &dst_image, extent);

    unsafe { device.end_command_buffer(command_buffer) }?;

    let submit_info =
        vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&command_buffer));
    unsafe { device.queue_submit(*queue, &[submit_info], fence)? };
    unsafe { device.wait_for_fences(&[fence], true, u64::MAX)? };

    let subresource = vk::ImageSubresource::default()
        .aspect_mask(vk::ImageAspectFlags::COLOR)
        .mip_level(0)
        .array_layer(0);
    let image_subresource = unsafe { device.get_image_subresource_layout(dst_image, subresource) };
    let image_dimensions = ImageDimensions::new(
        extent.width as _,
        extent.height as _,
        image_subresource.row_pitch as _,
    );

    let ptr =
        unsafe { device.map_memory(memory, 0, memory_reqs.size, vk::MemoryMapFlags::default())? };
    let frame = unsafe { std::slice::from_raw_parts_mut(ptr.cast(), memory_reqs.size as _) };

    println!("Blit image: {:?}", now.elapsed());

    save_screenshot(frame, image_dimensions)?;

    unsafe {
        device.destroy_fence(fence, None);
        device.destroy_image(dst_image, None);
        device.free_memory(memory, None);
        device.free_command_buffers(command_pool, &[command_buffer]);
        device.destroy_command_pool(command_pool, None);
    }

    Ok(())
}

pub fn create_folder(name: impl AsRef<Path>) -> std::io::Result<()> {
    match std::fs::create_dir(name) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e),
    }

    Ok(())
}

pub struct ImageDimensions {
    pub width: usize,
    pub height: usize,
    pub padded_bytes_per_row: usize,
    pub unpadded_bytes_per_row: usize,
}

impl ImageDimensions {
    fn new(width: usize, height: usize, row_pitch: usize) -> Self {
        let channel_width = std::mem::size_of::<[u8; 4]>();
        let unpadded_bytes_per_row = width * channel_width;
        Self {
            width,
            height,
            unpadded_bytes_per_row,
            padded_bytes_per_row: row_pitch,
        }
    }
}

pub fn save_screenshot(frame: &[u8], image_dimensions: ImageDimensions) -> Result<()> {
    let now = Instant::now();
    let screenshots_folder = Path::new(SCREENSHOTS_FOLDER);
    create_folder(screenshots_folder)?;
    let path = screenshots_folder.join(format!(
        "screenshot-{}.png",
        chrono::Local::now().format("%Y-%m-%d_%H-%M-%S%.9f")
    ));
    let file = File::create(path)?;
    let w = BufWriter::new(file);
    let mut encoder =
        png::Encoder::new(w, image_dimensions.width as _, image_dimensions.height as _);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    let padded_bytes = image_dimensions.padded_bytes_per_row;
    let unpadded_bytes = image_dimensions.unpadded_bytes_per_row;
    let mut writer = encoder
        .write_header()?
        .into_stream_writer_with_size(unpadded_bytes)?;
    writer.set_filter(png::FilterType::Paeth);
    writer.set_adaptive_filter(png::AdaptiveFilterType::Adaptive);
    for chunk in frame
        .chunks(padded_bytes)
        .map(|chunk| &chunk[..unpadded_bytes])
    {
        writer.write_all(chunk)?;
    }
    writer.finish()?;
    eprintln!("Encode image: {:#.2?}", now.elapsed());
    Ok(())
}
