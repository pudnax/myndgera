mod buffers;
mod device;
mod frame;
mod instance;
mod pipeline_arena;
mod staging;
mod surface;
mod swapchain;
mod texture_arena;
mod view_target;

use ash::vk;

pub use buffers::*;
pub use device::*;
pub use frame::*;
pub use instance::Instance;
pub use pipeline_arena::*;
pub use staging::*;
pub use surface::Surface;
pub use swapchain::*;
pub use texture_arena::*;
pub use view_target::*;

pub const BASE_IMAGE_RANGE: vk::ImageSubresourceRange = vk::ImageSubresourceRange {
    aspect_mask: vk::ImageAspectFlags::COLOR,
    base_mip_level: 0,
    level_count: 1,
    base_array_layer: 0,
    layer_count: 1,
};
