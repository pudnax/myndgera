use std::{
    io,
    ops::{Add, BitAnd, Not, Sub},
    path::Path,
    time::Duration,
};

use anyhow::Context;
use ash::vk;

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ShaderSource {
    pub path: std::path::PathBuf,
    pub kind: ShaderKind,
}

#[derive(Debug, Copy, Clone, Hash, Eq, PartialEq)]
pub enum ShaderKind {
    Fragment,
    Vertex,
    Compute,
}

impl From<ShaderKind> for shaderc::ShaderKind {
    fn from(value: ShaderKind) -> Self {
        match value {
            ShaderKind::Compute => shaderc::ShaderKind::Compute,
            ShaderKind::Vertex => shaderc::ShaderKind::Vertex,
            ShaderKind::Fragment => shaderc::ShaderKind::Fragment,
        }
    }
}

pub fn create_folder<P: AsRef<Path>>(name: P) -> io::Result<()> {
    match std::fs::create_dir(name) {
        Ok(_) => {}
        Err(e) if e.kind() == io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e),
    }

    Ok(())
}

#[derive(Debug)]
pub struct Args {
    pub inner_size: Option<(u32, u32)>,
    pub recording_time: Option<Duration>,
}

pub fn parse_args() -> anyhow::Result<Args> {
    let mut inner_size = None;
    let mut record_time = None;
    let args = std::env::args().skip(1).step_by(2);
    for (flag, value) in args.zip(std::env::args().skip(2).step_by(2)) {
        match flag.trim() {
            "--record" => {
                let time = match value.split_once('.') {
                    Some((sec, ms)) => {
                        let seconds = sec.parse()?;
                        let millis: u32 = ms.parse()?;
                        Duration::new(seconds, millis * 1_000_000)
                    }
                    None => Duration::from_secs(value.parse()?),
                };
                record_time = Some(time)
            }
            "--size" => {
                let (w, h) = value
                    .split_once('x')
                    .context("Failed to parse window size: Missing 'x' delimiter")?;
                inner_size = Some((w.parse()?, h.parse()?));
            }
            _ => {}
        }
    }

    Ok(Args {
        recording_time: record_time,
        inner_size,
    })
}

pub fn bytes_of<T: Copy>(data: &T) -> &[u8] {
    let ptr: *const T = data;
    unsafe { std::slice::from_raw_parts(ptr.cast(), std::mem::size_of_val(data)) }
}

pub fn from_bytes<T>(bytes: &mut [u8]) -> &mut T {
    unsafe { &mut *(bytes as *mut [u8] as *mut T) }
}

pub fn dispatch_optimal(len: u32, workgroup_size: u32) -> u32 {
    len.div_ceil(workgroup_size)
}

pub fn align_to<T>(value: T, alignment: T) -> T
where
    T: Add<Output = T> + Copy + IntMath + Not<Output = T> + BitAnd<Output = T> + Sub<Output = T>,
{
    let mask = alignment.saturating_sub(T::one());
    (value + mask) & !mask
}

pub trait IntMath {
    fn one() -> Self;
    fn saturating_sub(self, other: Self) -> Self;
}
macro_rules! missing_math {
    ( $($t:ty),+) => {
        $(impl IntMath for $t {
            fn one() -> $t { 1 }
            fn saturating_sub(self, other: $t) -> $t {
                self.saturating_sub(other)
            }
        })*
    }
}
missing_math!(i32, u32, i64, u64, usize);

pub fn find_memory_type_index(
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    memory_type_bits: u32,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as _]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_type_bits != 0 && (memory_type.property_flags & flags) == flags
        })
        .map(|(index, _memory_type)| index as _)
}
