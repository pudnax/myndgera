use std::path::{Path, PathBuf};

use crate::{device, Watcher};
use ahash::AHashSet;
use anyhow::{Context, Result};
use shaderc::{CompilationArtifact, ShaderKind};

const SHADER_PATH: &str = "shaders";

pub struct ShaderCompiler {
    compiler: shaderc::Compiler,
    options: shaderc::CompileOptions<'static>,
    watcher: Watcher,
}

impl ShaderCompiler {
    pub fn new(watcher: Watcher) -> Result<Self> {
        let mut options =
            shaderc::CompileOptions::new().context("Failed to create shader compiler options")?;
        options.set_target_env(
            shaderc::TargetEnv::Vulkan,
            shaderc::EnvVersion::Vulkan1_3 as u32,
        );
        options.set_optimization_level(shaderc::OptimizationLevel::Performance);
        options.set_target_spirv(shaderc::SpirvVersion::V1_6);
        options.set_generate_debug_info();

        let watcher_copy = watcher.clone();
        options.set_include_callback(move |name, include_type, source_file, depth| {
            let path = if include_type == shaderc::IncludeType::Relative {
                Path::new(Path::new(source_file).parent().unwrap()).join(name)
            } else {
                Path::new(SHADER_PATH).join(name)
            };
            match std::fs::read_to_string(&path) {
                Ok(glsl_code) => {
                    let include_path = path.canonicalize().unwrap();
                    let source_path = Path::new(SHADER_PATH)
                        .join(source_file)
                        .canonicalize()
                        .unwrap();
                    {
                        let mut watcher_guard = watcher_copy.watcher.lock();
                        let watcher = watcher_guard.watcher();
                        let mut mapping = watcher_copy.include_mapping.lock();
                        if depth > 1 {
                            let sources: Vec<_> = mapping[&source_path].iter().cloned().collect();
                            for source in sources {
                                mapping
                                    .entry(include_path.clone())
                                    .or_insert_with_key(|path| {
                                        // TODO: move up and recreate
                                        let _ = watcher
                                            .watch(path, notify::RecursiveMode::NonRecursive);
                                        AHashSet::new()
                                    })
                                    .insert(source);
                            }
                        } else {
                            mapping
                                .entry(include_path)
                                .or_insert_with_key(|path| {
                                    let _ =
                                        watcher.watch(path, notify::RecursiveMode::NonRecursive);
                                    AHashSet::new()
                                })
                                .insert(source_path);
                        }
                    }
                    Ok(shaderc::ResolvedInclude {
                        resolved_name: String::from(name),
                        content: glsl_code,
                    })
                }
                Err(err) => Err(format!(
                    "Failed to resolve include to {} in {} (was looking for {:?}): {}",
                    name, source_file, path, err
                )),
            }
        });

        Ok(Self {
            compiler: shaderc::Compiler::new().unwrap(),
            options,
            watcher,
        })
    }

    pub fn compile(&self, path: impl AsRef<Path>, kind: ShaderKind) -> Result<CompilationArtifact> {
        let source = std::fs::read_to_string(path.as_ref())?;
        Ok(self.compiler.compile_into_spirv(
            &source,
            kind,
            path.as_ref().file_name().and_then(|s| s.to_str()).unwrap(),
            "main",
            Some(&self.options),
        )?)
    }
}
