use std::sync::Arc;

use anyhow::{Context, Result};
use ash::vk;
use gpu_allocator::MemoryLocation;

use super::Device;

struct PendingWrite {
    region: vk::BufferCopy2<'static>,
    buffer: vk::Buffer,
}

pub struct StagingWrite {
    buffer: crate::Buffer,
    intermediate_data: Vec<u8>,
    pending_writes: Vec<PendingWrite>,
    size: usize,
    device: Arc<Device>,
}

impl StagingWrite {
    pub fn new(device: &Arc<Device>) -> Result<Self> {
        let size = 4;
        let buffer = device.create_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        Ok(Self {
            buffer,
            intermediate_data: vec![],
            pending_writes: vec![],
            size: size as _,
            device: device.clone(),
        })
    }

    pub fn clear(&mut self) {
        self.intermediate_data.clear();
        self.pending_writes.clear();
    }

    pub fn write_buffer(&mut self, buffer: vk::Buffer, data: &[u8]) {
        let offset = self.intermediate_data.len();
        self.pending_writes.push(PendingWrite {
            buffer,
            region: vk::BufferCopy2::default()
                .src_offset(offset as u64)
                .dst_offset(0)
                .size(data.len() as u64),
        });
        self.intermediate_data.extend(data);
    }

    pub fn reserve_buffer(&mut self) -> Result<bool> {
        let offset = self.intermediate_data.len();
        if offset < self.size {
            return Ok(false);
        }

        let max_buffer_size = self
            .device
            .device_properties
            .limits
            .max_storage_buffer_range;
        let new_size = offset
            .checked_next_power_of_two()
            .unwrap_or(offset)
            .min(max_buffer_size as usize);
        let new_buffer = self.device.create_buffer(
            new_size as u64,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
        )?;
        self.buffer = new_buffer;
        self.size = new_size;

        Ok(true)
    }

    pub fn consume_pending_writes(&mut self, &cbuff: &vk::CommandBuffer) -> Result<()> {
        if self.pending_writes.is_empty() {
            return Ok(());
        }
        let mapped = self.buffer.map_memory().context("Failed to map memory")?;
        mapped[..self.intermediate_data.len()].copy_from_slice(&self.intermediate_data);
        for write in &self.pending_writes {
            let copy_info = vk::CopyBufferInfo2::default()
                .src_buffer(self.buffer.buffer)
                .dst_buffer(write.buffer)
                .regions(std::slice::from_ref(&write.region));
            unsafe { self.device.cmd_copy_buffer2(cbuff, &copy_info) };
        }
        Ok(())
    }
}
