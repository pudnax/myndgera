use std::{
    fs::File,
    io,
    ops::{Add, BitAnd, Not, Sub},
    path::Path,
    time::Duration,
};

use anyhow::{bail, Context};
use ash::vk;

use crate::{SHADER_DUMP_FOLDER, SHADER_FOLDER};

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GlobalStats {
    pub pos: [f32; 3],
    pub time: f32,
    pub wh: [f32; 2],
    pub mouse: [f32; 2],
    pub mouse_pressed: u32,
    pub frame: u32,
    pub time_delta: f32,
    pub record_time: f32,
}

impl Default for GlobalStats {
    fn default() -> Self {
        Self {
            pos: [0.; 3],
            time: 0.,
            wh: [1920.0, 1020.],
            mouse: [0.; 2],
            mouse_pressed: false as _,
            frame: 0,
            time_delta: 1. / 60.,
            record_time: 10.,
        }
    }
}

impl std::fmt::Display for GlobalStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let time = Duration::from_secs_f32(self.time);
        let time_delta = Duration::from_secs_f32(self.time_delta);
        write!(
            f,
            "position:\t{:?}\n\
             time:\t\t{:#.2?}\n\
             time delta:\t{:#.3?}, fps: {:#.2?}\n\
             width, height:\t{:?}\nmouse:\t\t{:.2?}\n\
             frame:\t\t{}\nrecord_period:\t{}\n",
            self.pos,
            time,
            time_delta,
            1. / self.time_delta,
            self.wh,
            self.mouse,
            self.frame,
            self.record_time
        )
    }
}

pub fn print_help() {
    println!();
    println!("- `F1`:   Print help");
    println!("- `F2`:   Toggle play/pause");
    println!("- `F3`:   Pause and step back one frame");
    println!("- `F4`:   Pause and step forward one frame");
    println!("- `F5`:   Restart playback at frame 0 (`Time` and `Pos` = 0)");
    println!("- `F6`:   Print parameters");
    println!("- `F10`:  Save shaders");
    println!("- `F11`:  Take Screenshot");
    println!("- `F12`:  Start/Stop record video");
    println!("- `ESC`:  Exit the application");
    println!("- `Arrows`: Change `Pos`\n");
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
    pub record_time: Option<Duration>,
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
        record_time,
        inner_size,
    })
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

pub fn dispatch_optimal(len: u32, subgroup_size: u32) -> u32 {
    let padded_size = (subgroup_size - len % subgroup_size) % subgroup_size;
    (len + padded_size) / subgroup_size
}

pub fn save_shaders<P: AsRef<Path>>(path: P) -> anyhow::Result<()> {
    let dump_folder = Path::new(SHADER_DUMP_FOLDER);
    create_folder(dump_folder)?;
    let dump_folder =
        dump_folder.join(chrono::Local::now().format("%Y-%m-%d_%H-%M-%S").to_string());
    create_folder(&dump_folder)?;
    let dump_folder = dump_folder.join(SHADER_FOLDER);
    create_folder(&dump_folder)?;

    if !path.as_ref().is_dir() {
        bail!("Folder wasn't supplied");
    }
    let shaders = path.as_ref().read_dir()?;

    for shader in shaders {
        let shader = shader?.path();
        let to = dump_folder.join(shader.file_name().and_then(|s| s.to_str()).unwrap());
        if !to.exists() {
            std::fs::create_dir_all(&to.parent().unwrap().canonicalize()?)?;
            File::create(&to)?;
        }
        std::fs::copy(shader, &to)?;
        println!("Saved: {}", &to.display());
    }

    Ok(())
}

#[derive(Debug)]
pub enum UserEvent {
    Glsl { path: std::path::PathBuf },
}

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