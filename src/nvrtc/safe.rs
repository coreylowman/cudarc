//! Safe abstractions around [crate::nvrtc::result] for compiling PTX files.
//!
//! Call [compile_ptx()] or [compile_ptx_with_opts()].

use super::{result, sys};

use core::ffi::{c_char, CStr};
use std::ffi::CString;
use std::fs;
use std::path::Path;
use std::{borrow::ToOwned, path::PathBuf, string::String, vec::Vec};

/// An opaque structure representing a compiled PTX program
/// output from [compile_ptx()] or [compile_ptx_with_opts()].
///
/// Can also be created from a [Ptx::from_file] and [Ptx::from_src]
#[derive(Debug, Clone)]
pub struct Ptx(pub(crate) PtxKind);

impl Ptx {
    /// Creates a Ptx from a pre-compiled .ptx file.
    pub fn from_file<P: Into<PathBuf>>(path: P) -> Self {
        Self(PtxKind::File(path.into()))
    }

    /// Creates a Ptx from the source string of a pre-compiled .ptx
    /// file.
    pub fn from_src<S: Into<String>>(src: S) -> Self {
        Self(PtxKind::Src(src.into()))
    }
}

impl<S: Into<String>> From<S> for Ptx {
    fn from(value: S) -> Self {
        Self::from_src(value)
    }
}

#[derive(Debug, Clone)]
pub(crate) enum PtxKind {
    /// An image created by [compile_ptx]
    Image(Vec<c_char>),

    /// Content of a pre compiled ptx file
    Src(String),

    /// Path to a compiled ptx
    File(PathBuf),
}

/// Calls [compile_ptx_with_opts] with no options. `src` is the source string
/// of a `.cu` file.
///
/// Example:
/// ```rust
/// # use cudarc::nvrtc::*;
/// let ptx = compile_ptx("extern \"C\" __global__ void kernel() { }").unwrap();
/// ```
pub fn compile_ptx<S: AsRef<str>>(src: S) -> Result<Ptx, CompileError> {
    compile_ptx_with_opts(src, Default::default())
}

/// Compiles `src` with the given `opts`. `src` is the source string of a `.cu` file.
///
/// Example:
/// ```rust
/// # use cudarc::nvrtc::*;
/// let opts = CompileOptions {
///     ftz: Some(true),
///     maxrregcount: Some(10),
///     ..Default::default()
/// };
/// let ptx = compile_ptx_with_opts("extern \"C\" __global__ void kernel() { }", opts).unwrap();
/// ```
pub fn compile_ptx_with_opts<S: AsRef<str>>(
    src: S,
    opts: CompileOptions,
) -> Result<Ptx, CompileError> {
    let prog = Program::create(src)?;
    prog.compile(opts)
}

pub(crate) struct Program {
    prog: sys::nvrtcProgram,
}

impl Program {
    pub(crate) fn create<S: AsRef<str>>(src: S) -> Result<Self, CompileError> {
        let prog = result::create_program(src).map_err(CompileError::CreationError)?;
        Ok(Self { prog })
    }

    pub(crate) fn compile(self, opts: CompileOptions) -> Result<Ptx, CompileError> {
        let options = opts.build();

        unsafe { result::compile_program(self.prog, &options) }.map_err(|e| {
            let log_raw = unsafe { result::get_program_log(self.prog) }.unwrap();
            let log_ptr = log_raw.as_ptr();
            let log = unsafe { CStr::from_ptr(log_ptr) }.to_owned();
            CompileError::CompileError {
                nvrtc: e,
                options,
                log,
            }
        })?;

        let image = unsafe { result::get_ptx(self.prog) }.map_err(CompileError::GetPtxError)?;

        Ok(Ptx(PtxKind::Image(image)))
    }
}

impl Drop for Program {
    fn drop(&mut self) {
        let prog = std::mem::replace(&mut self.prog, std::ptr::null_mut());
        if !prog.is_null() {
            unsafe { result::destroy_program(prog) }.unwrap()
        }
    }
}

/// Represents an error that happens during nvrtc compilation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    /// Error happened during [result::create_program()]
    CreationError(result::NvrtcError),

    /// Error happened during [result::compile_program()]
    CompileError {
        nvrtc: result::NvrtcError,
        options: Vec<String>,
        log: CString,
    },

    /// Error happened during [result::get_program_log()]
    GetLogError(result::NvrtcError),

    /// Error happened during [result::get_ptx()]
    GetPtxError(result::NvrtcError),

    /// Error happened during [result::destroy_program()]
    DestroyError(result::NvrtcError),
}

#[cfg(feature = "std")]
impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CompileError {}

/// Flags you can pass to the nvrtc compiler.
/// See <https://docs.nvidia.com/cuda/nvrtc/index.html#group__options>
/// for all available flags and documentation for what they do.
///
/// All fields of this struct match one of the flags in the documentation.
/// if a field is `None` it will not be passed to the compiler.
///
/// All fields default to `None`.
///
/// *NOTE*: not all flags are currently supported.
///
/// Example:
/// ```rust
/// # use cudarc::nvrtc::*;
/// // "--ftz=true" will be passed to the compiler
/// let opts = CompileOptions {
///     ftz: Some(true),
///     ..Default::default()
/// };
/// ```
#[derive(Clone, Debug, Default, Hash, PartialEq, Eq)]
pub struct CompileOptions {
    pub ftz: Option<bool>,
    pub prec_sqrt: Option<bool>,
    pub prec_div: Option<bool>,
    pub fmad: Option<bool>,
    pub use_fast_math: Option<bool>,
    pub maxrregcount: Option<usize>,
    pub include_paths: Vec<String>,
    pub arch: Option<&'static str>,
}

impl CompileOptions {
    pub(crate) fn build(self) -> Vec<String> {
        let mut options: Vec<String> = Vec::new();

        if let Some(v) = self.ftz {
            options.push(std::format!("--ftz={v}"));
        }

        if let Some(v) = self.prec_sqrt {
            options.push(std::format!("--prec-sqrt={v}"));
        }

        if let Some(v) = self.prec_div {
            options.push(std::format!("--prec-div={v}"));
        }

        if let Some(v) = self.fmad {
            options.push(std::format!("--fmad={v}"));
        }

        if let Some(true) = self.use_fast_math {
            options.push("--fmad=true".into());
        }

        if let Some(count) = self.maxrregcount {
            options.push(std::format!("--maxrregcount={count}"));
        }

        for path in self.include_paths {
            options.push(std::format!("--include-path={path}"));
        }

        if let Some(arch) = self.arch {
            options.push(std::format!("--gpu-architecture={arch}"))
        }

        options
    }
}

#[derive(Debug)]
pub enum PtxCrateKind {
    Cargo{ project_dir: PathBuf },      // rust ptx project dir w/ Cargo.toml & kernel(s) in src/lib.rs
    Standalone{standalone: PathBuf },    // todo!("standalone rs file compiled with rustc")
}

impl TryFrom<PathBuf> for PtxCrateKind {
    type Error = String;

    fn try_from(value: PathBuf) -> Result<Self, Self::Error> {
        // if value = path/to/files
        if let Some(name) = value
            .file_name()
            .map(|name| name.to_str())
        {
            match name {
                Some("Cargo.toml") =>   // value = path/to/project/Cargo.toml
                    return Ok( Self::Cargo { project_dir: value.parent().unwrap().into() } ),
                Some("lib.rs") => {     // value = path/to/project/src/lib.rs
                    let src = value.parent().unwrap();
                    if let Some(project_dir) = src.parent() {
                        let manifest = project_dir.join("Cargo.toml");
                        if manifest.exists() {
                            return Ok( Self::Cargo { project_dir: project_dir.into() } )
                        } else {
                            todo!()
                        }
                    } else {
                        return Err(format!("could not find parent of {src:?}"))
                    }
                },
                Some(name) =>     // value = path/to/project/unsupported_name
                    return Err(format!("unsupported file name: {name}")),
                None =>                 // name.to_str() failed to parse as unicode
                    return Err(format!("failed to parse {name:?} as valid unicode"))
            }
        }

        // if value = path/to/project/ containing Cargo.toml
        if value.join("Cargo.toml").exists() {
            Ok( Self::Cargo { project_dir: value } )
        } else {
            Err(format!("{value:?}/Cargo.toml missing"))
        }
    }
}

#[derive(Debug)]
pub struct PtxCrate {
    project: PtxCrateKind,
    kernels: Option<Vec<Ptx>>,
}

impl TryInto<Vec<Ptx>> for PtxCrate {
    type Error = String;
    fn try_into(self) -> Result<Vec<Ptx>, Self::Error> {
        self.kernels.ok_or("kernels not built".into())
    }
}

impl TryFrom<PathBuf> for PtxCrate {
    type Error = String;

    fn try_from(value: PathBuf) -> Result<Self, Self::Error> {
        let project = value.try_into()?;
        Ok(PtxCrate { project, kernels: None })
    }
}

use std::process::Command;
impl PtxCrate {
    
    pub fn compile_crate_to_ptx<S: AsRef<Path>>(kernel_path: S) -> Result<Vec<Ptx>, String> {
        let kernel_path: PathBuf = kernel_path.as_ref().into();
        let mut rust_ptx: Self = kernel_path.try_into()?;
        rust_ptx.build_ptx()?;
        Ok(rust_ptx.take_kernels().unwrap())
    }

    pub fn take_kernels(self) -> Option<Vec<Ptx>> {
        self.kernels
    }
    
    pub fn ptx_files(&self) -> Option<&Vec<Ptx>> {
        self.kernels.as_ref()
    }

    pub fn clean(&mut self) -> Result<(), String> {
        match &self.project {
            PtxCrateKind::Cargo { project_dir } => {
                let manifest_path = project_dir.join("Cargo.toml");

                let output = Command::new("cargo")
                    .arg("clean")
                    .arg("--manifest-path")
                    .arg(manifest_path)
                    .output()
                    .map_err(|e| format!("Failed to execute command: {}", e))?;
                if output.status.success() {
                    self.kernels = None;
                    Ok(())
                } else {
                    Err(format!(
                        "Failed to build PTX file: {}",
                        String::from_utf8_lossy(&output.stderr)
                    ))
                }
            },
            PtxCrateKind::Standalone { standalone: _standalone } => todo!("rm -rf target/ ?"),
        }
    }
    
    pub fn build_ptx(&mut self) -> Result<(), String> {
        match &self.project {
            PtxCrateKind::Cargo { project_dir } => {
                let manifest_path = project_dir.join("Cargo.toml");

                let output = Command::new("cargo")
                    .arg("+nightly")
                    .arg("rustc")
                    .arg("--manifest-path")
                    .arg(manifest_path)
                    .arg("--lib")
                    .arg("--target")
                    .arg("nvptx64-nvidia-cuda")
                    .arg("--release")
                    .arg("--")
                    .arg("--emit")
                    .arg("asm")
                    .output()
                    .map_err(|e| format!("Failed to execute command: {}", e))?;

                if output.status.success() {
                    let ptx_path = project_dir.join("target/nvptx64-nvidia-cuda/release");
                    if ptx_path.exists() {
                        let ptx_files: Vec<Ptx> = fs::read_dir(ptx_path)
                        .map_err(|e| e.to_string())?
                        .filter_map(|entry| {
                            let path = entry.unwrap().path();
                            if let Some("ptx") = path.extension().and_then(|ext| ext.to_str()) {
                                Some(Ptx::from_file(path))
                            } else {
                                None
                            }
                        })
                        .collect();
                        self.kernels = Some(ptx_files);
                        Ok(())
                    } else {
                        Err(format!(
                            "Could not find {ptx_path:?}"
                        ))
                    }
                } else {
                    Err(format!(
                        "Failed to build PTX file: {}",
                        String::from_utf8_lossy(&output.stderr)
                    ))
                }       
            },
            PtxCrateKind::Standalone { standalone: _ } => todo!("standalone rs -> ptx"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_no_opts() {
        const SRC: &str =
            "extern \"C\" __global__ void sin_kernel(float *out, const float *inp, int numel) {
            int i = blockIdx.x * blockDim.x + threadIdx.x;
            if (i < numel) {
                out[i] = sin(inp[i]);
            }
        }";
        compile_ptx_with_opts(SRC, Default::default()).unwrap();
    }

    #[test]
    fn test_compile_options_build_none() {
        let opts: CompileOptions = Default::default();
        assert!(opts.build().is_empty());
    }

    #[test]
    fn test_compile_options_build_ftz() {
        let opts = CompileOptions {
            ftz: Some(true),
            ..Default::default()
        };
        assert_eq!(&opts.build(), &["--ftz=true"]);
    }

    #[test]
    fn test_compile_options_build_multi() {
        let opts = CompileOptions {
            prec_div: Some(false),
            maxrregcount: Some(60),
            ..Default::default()
        };
        assert_eq!(&opts.build(), &["--prec-div=false", "--maxrregcount=60"]);
    }
}
