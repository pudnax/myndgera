use std::sync::atomic::{AtomicU8, Ordering};

use anyhow::Result;
use ash::vk;

use crate::{AppState, ImageHandle};

pub struct PostProcessWrite<'a> {
    pub source: &'a ImageHandle,
    pub destination: &'a ImageHandle,
}

pub struct ViewTarget {
    images: [ImageHandle; 2],
    main_image: AtomicU8,
}

impl ViewTarget {
    pub fn new(state: &mut AppState, format: vk::Format) -> Result<Self> {
        let [width, height] = state.stats.wh;
        let image_info = vk::ImageCreateInfo::default()
            .format(format)
            .extent(vk::Extent3D {
                width: width as u32,
                height: height as u32,
                depth: 1,
            })
            .image_type(vk::ImageType::TYPE_2D)
            .usage(vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::SAMPLED)
            .samples(vk::SampleCountFlags::TYPE_1)
            .mip_levels(1)
            .array_layers(1)
            .tiling(vk::ImageTiling::OPTIMAL);
        let a = state.texture_arena.push_image(
            image_info,
            crate::ScreenRelation::Identity,
            &[],
            Some("View Target A"),
        )?;
        let b = state.texture_arena.push_image(
            image_info,
            crate::ScreenRelation::Identity,
            &[],
            Some("View Target B"),
        )?;

        Ok(Self {
            images: [a, b],
            main_image: AtomicU8::new(0),
        })
    }

    pub fn main_image(&self) -> &ImageHandle {
        let idx = self.main_image.load(Ordering::Relaxed);
        &self.images[idx as usize]
    }

    pub fn post_process_write(&self) -> PostProcessWrite {
        let old_target = self.main_image.fetch_xor(1, Ordering::Relaxed);
        if old_target == 0 {
            PostProcessWrite {
                source: &self.images[0],
                destination: &self.images[1],
            }
        } else {
            PostProcessWrite {
                source: &self.images[1],
                destination: &self.images[0],
            }
        }
    }
}
