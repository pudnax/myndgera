use std::{collections::HashSet, sync::Arc};

use crate::{
    device::{Device, DeviceExt},
    surface::Surface,
};

use anyhow::{Context, Result};
use ash::{ext, khr, vk, Entry};
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

pub struct Instance {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
}

impl std::ops::Deref for Instance {
    type Target = ash::Instance;

    fn deref(&self) -> &Self::Target {
        &self.instance
    }
}

impl Instance {
    pub fn new(display_handle: Option<&impl HasDisplayHandle>) -> Result<Self> {
        let entry = unsafe { Entry::load() }?;
        let layers = [
            #[cfg(debug_assertions)]
            c"VK_LAYER_KHRONOS_validation".as_ptr(),
        ];
        let mut extensions = vec![
            khr::surface::NAME.as_ptr(),
            khr::display::NAME.as_ptr(),
            khr::get_physical_device_properties2::NAME.as_ptr(),
        ];
        if let Some(handle) = display_handle {
            extensions.extend(ash_window::enumerate_required_extensions(
                handle.display_handle()?.as_raw(),
            )?);
        }

        let appinfo = vk::ApplicationInfo::default()
            .application_name(c"Modern Vulkan")
            .api_version(vk::API_VERSION_1_3);
        let instance_info = vk::InstanceCreateInfo::default()
            .application_info(&appinfo)
            .flags(vk::InstanceCreateFlags::default())
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        let instance = unsafe { entry.create_instance(&instance_info, None) }?;
        Ok(Self { entry, instance })
    }

    pub fn create_device_and_queues(
        &self,
        surface: &Surface,
    ) -> Result<(Device, vk::Queue, vk::Queue)> {
        let required_device_extensions = [
            khr::swapchain::NAME,
            ext::graphics_pipeline_library::NAME,
            khr::pipeline_library::NAME,
            khr::dynamic_rendering::NAME,
            ext::extended_dynamic_state2::NAME,
            ext::extended_dynamic_state::NAME,
            khr::synchronization2::NAME,
            khr::buffer_device_address::NAME,
            khr::create_renderpass2::NAME,
            ext::descriptor_indexing::NAME,
        ];
        let required_device_extensions_set = HashSet::from(required_device_extensions);

        let devices = unsafe { self.enumerate_physical_devices() }?;
        let (pdevice, main_queue_family_idx, transfer_queue_family_idx) =
            devices
                .into_iter()
                .find_map(|device| {
                    let extensions =
                        unsafe { self.enumerate_device_extension_properties(device) }.ok()?;
                    let extensions: HashSet<_> = extensions
                        .iter()
                        .map(|x| x.extension_name_as_c_str().unwrap())
                        .collect();
                    let missing = required_device_extensions_set.difference(&extensions);
                    if missing.count() > 0 {
                        return None;
                    }

                    use vk::QueueFlags as QF;
                    let queue_properties =
                        unsafe { self.get_physical_device_queue_family_properties(device) };
                    let main_queue_idx =
                        queue_properties
                            .iter()
                            .enumerate()
                            .find_map(|(family_idx, properties)| {
                                let family_idx = family_idx as u32;

                                let queue_support =
                                    properties.queue_flags.contains(QF::GRAPHICS | QF::TRANSFER);
                                let surface_support =
                                    surface.get_device_surface_support(device, family_idx);
                                (queue_support && surface_support).then_some(family_idx)
                            });

                    let transfer_queue_idx = queue_properties.iter().enumerate().find_map(
                        |(family_idx, properties)| {
                            let family_idx = family_idx as u32;
                            let queue_support = properties.queue_flags.contains(QF::TRANSFER)
                                && !properties.queue_flags.contains(QF::GRAPHICS);
                            (Some(family_idx) != main_queue_idx && queue_support)
                                .then_some(family_idx)
                        },
                    )?;

                    Some((device, main_queue_idx?, transfer_queue_idx))
                })
                .context("Failed to find suitable device.")?;

        let queue_infos = [
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(main_queue_family_idx)
                .queue_priorities(&[1.0]),
            vk::DeviceQueueCreateInfo::default()
                .queue_family_index(transfer_queue_family_idx)
                .queue_priorities(&[0.5]),
        ];

        let required_device_extensions = required_device_extensions.map(|x| x.as_ptr());

        let mut feature_dynamic_state =
            vk::PhysicalDeviceExtendedDynamicState2FeaturesEXT::default();
        let mut feature_descriptor_indexing =
            vk::PhysicalDeviceDescriptorIndexingFeatures::default()
                .runtime_descriptor_array(true)
                .descriptor_binding_variable_descriptor_count(true)
                .descriptor_binding_partially_bound(true)
                .descriptor_binding_update_unused_while_pending(true)
                .descriptor_binding_sampled_image_update_after_bind(true);
        let mut feature_buffer_device_address =
            vk::PhysicalDeviceBufferDeviceAddressFeatures::default().buffer_device_address(true);
        let mut feature_synchronization2 =
            vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
        let mut feature_pipeline_library =
            vk::PhysicalDeviceGraphicsPipelineLibraryFeaturesEXT::default()
                .graphics_pipeline_library(true);
        let mut feature_dynamic_rendering =
            vk::PhysicalDeviceDynamicRenderingFeatures::default().dynamic_rendering(true);

        let mut features = vk::PhysicalDeviceFeatures::default().shader_int64(true);
        if cfg!(debug_assertions) {
            features.robust_buffer_access = 1;
        }

        let mut default_features = vk::PhysicalDeviceFeatures2::default()
            .features(features)
            .push_next(&mut feature_descriptor_indexing)
            .push_next(&mut feature_buffer_device_address)
            .push_next(&mut feature_synchronization2)
            .push_next(&mut feature_dynamic_state)
            .push_next(&mut feature_pipeline_library)
            .push_next(&mut feature_dynamic_rendering);

        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .enabled_extension_names(&required_device_extensions)
            .push_next(&mut default_features);
        let device = unsafe { self.instance.create_device(pdevice, &device_info, None) }?;

        fn fmt_size(n: u64) -> String {
            if n < 1_000 {
                format!("{:>3} B", n)
            } else if n < 1_000_000 {
                format!("{:>3} kB", n >> 10)
            } else if n < 1_000_000_000 {
                format!("{:>3} MB", n >> 20)
            } else {
                format!("{:>3} GB", n >> 30)
            }
        }
        let memory_properties = unsafe { self.get_physical_device_memory_properties(pdevice) };

        for mp in &memory_properties.memory_types[..memory_properties.memory_type_count as _] {
            if !mp.property_flags.is_empty() {
                println!("Memory: {:?}", mp.property_flags);
                let heap = memory_properties.memory_heaps[mp.heap_index as usize];
                println!(
                    "\tMemory Heap {}: Size {:?} | Type {:?}",
                    mp.heap_index,
                    fmt_size(heap.size),
                    heap.flags
                )
            }
        }

        let dynamic_rendering = khr::dynamic_rendering::Device::new(self, &device);

        let device = Device {
            physical_device: pdevice,
            main_queue_family_idx,
            transfer_queue_family_idx,
            memory_properties,
            device: Arc::new(device),
            ext: Arc::new(DeviceExt { dynamic_rendering }),
        };
        let main_queue = unsafe { device.get_device_queue(main_queue_family_idx, 0) };
        let transfer_queue = unsafe { device.get_device_queue(transfer_queue_family_idx, 0) };

        Ok((device, main_queue, transfer_queue))
    }

    pub fn create_surface(
        &self,
        handle: &(impl HasDisplayHandle + HasWindowHandle),
    ) -> Result<Surface> {
        Surface::new(&self.entry, &self.instance, handle)
    }
}

impl Drop for Instance {
    fn drop(&mut self) {
        unsafe { self.instance.destroy_instance(None) };
    }
}
