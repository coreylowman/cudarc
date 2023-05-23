//! Safe abstractions around [crate::nvrtc::result] for compiling PTX files.
//!
//! Call [compile_ptx()] or [compile_ptx_with_opts()].

use super::{result, sys};

use core::ffi::{c_char, CStr};
use std::ffi::CString;
use std::{fs, fs::File};
use std::io::Read;
use std::path::Path;
use std::{borrow::ToOwned, path::PathBuf, string::String, vec::Vec};
use std::process::Command;

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

    
/// Compiles a Rust crate at the specified path into PTX code and returns a vector of `Ptx` objects.
///
/// # Arguments
///
/// * `kernel_path` - The path to the Rust crate to be compiled.
///
/// # Returns
///
/// A `Result` object containing either a vector of `Ptx` objects representing the compiled kernels
/// from the Rust crate, or an error message as a `String` if the compilation process failed.
///
/// # Examples
///
/// ```
/// use ptx_builder::{PtxCrate, Ptx};
/// use std::path::PathBuf;
///
/// /* alternatively, 
///     let kernel_path: PathBuf = "examples/rust-kernel/".into();
///     let kernel_path: PathBuf = "examples/rust-kernel/Cargo.toml".into();
/// */
/// let kernel_path: PathBuf = "examples/rust-kernel/src/lib.rs".into();
/// let kernels: Vec<Ptx> = PtxCrate::compile_crate_to_ptx(&kernel_path).unwrap();
/// let kernel = kernels.first().unwrap();
/// ```
pub fn compile_crate_to_ptx<S: AsRef<Path>>(kernel_path: S) -> Result<Vec<Ptx>, String> {
    let kernel_path: PathBuf = kernel_path.as_ref().into();
    let mut rust_ptx: PtxCrate = kernel_path.try_into()?;
    rust_ptx.build_ptx()?;
    Ok(rust_ptx.take_kernels().unwrap())
}


/// `PtxCrate` provides methods to compile a Rust crate to CUDA PTX code,
/// as well as to extract and manipulate the resulting PTX kernels.
/// It requires a path to the Rust crate containing the kernel to be compiled.
///
/// # Examples
///
/// ```
/// use std::convert::TryInto;
/// use ptx_builder::{PtxCrate, Ptx};
/// use std::path::PathBuf;
///
/// let kernel_path: PathBuf = "examples/rust-kernel/src/lib.rs".into();
/// let mut rust_ptx: PtxCrate = kernel_path.try_into().unwrap();
/// rust_ptx.build_ptx().unwrap();
/// let _kernel: &Ptx = rust_ptx.peek_kernels().unwrap().first().unwrap();
/// println!("Cleaned successfully? {:?}", rust_ptx.clean());
/// ```
#[derive(Debug)]
pub struct PtxCrate {
    project_dir: PathBuf,
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
        if !value.exists() {
            return Err(format!("{value:?} does not exist"))
        }
        
        // if value = path/to/files
        if let Some(name) = value
            .file_name()
            .map(|name| name.to_str())
        {
            let project_dir = match name {
                // value = path/to/project/Cargo.toml
                Some("Cargo.toml") =>
                    value.parent(),
                // value = path/to/project/src/lib.rs
                Some("lib.rs") => {
                    let src = value.parent().ok_or("err")?;
                    if let Some(project_dir) = src.parent() {
                        let manifest = project_dir.join("Cargo.toml");
                        if manifest.exists() {
                            project_dir.into()
                        } else {
                            return Err("missing Cargo.toml".to_string())
                        }
                    } else {
                        return Err(format!("could not find parent of {src:?}"))
                    }
                },
                // value = path/to/project/unsupported_name
                Some(name) =>
                    return Err(format!("unsupported file name: {name}")),
                // name.to_str() failed to parse as unicode
                None =>
                    return Err(format!("failed to parse {name:?} as valid unicode"))
            };
            return Ok(Self { project_dir: project_dir.unwrap().into(), kernels: None })
        }

        // if value = path/to/project/ containing Cargo.toml
        if value.join("Cargo.toml").exists() {
            Ok(PtxCrate { project_dir: value, kernels: None } )
        } else {
            Err(format!("{value:?}/Cargo.toml missing"))
        }
    }
}
impl PtxCrate {
    /// Takes ownership of the PTX kernels stored in this `PtxCrate`.
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to a vector of `Ptx` objects if the `PtxCrate` contains
    /// any PTX kernels. If the `PtxCrate` has not been built or has been cleaned, `None` is returned.
    pub fn take_kernels(self) -> Option<Vec<Ptx>> {
        self.kernels
    }
    
    /// Returns a reference to the PTX kernels stored in this `PtxCrate`.
    ///
    /// # Returns
    ///
    /// An `Option` containing a reference to a vector of `Ptx` objects if the `PtxCrate` contains
    /// any PTX kernels. If the `PtxCrate` has not been built or has been cleaned, `None` is returned.
    pub fn peek_kernels(&self) -> Option<&Vec<Ptx>> {
        self.kernels.as_ref()
    }

    /// Removes any compiled PTX kernels from this `PtxCrate`.
    ///
    /// # Returns
    ///
    /// An `Ok(())` value if the operation succeeds.
    ///
    /// # Errors
    ///
    /// Returns an error if there is a problem cleaning the `PtxCrate`.
    pub fn clean(&mut self) -> Result<(), String> {
        let manifest_path = self.project_dir.join("Cargo.toml");
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
    }
    
    /// Builds the PTX code for the currently set kernels.
    ///
    /// # Arguments
    ///
    /// * `self` - A mutable reference to the `PtxCrate` instance.
    ///
    /// # Returns
    ///
    /// Returns a reference to a vector of `Ptx` instances containing the compiled PTX code
    /// for the kernels if the build is successful, otherwise an error message as a `String`.
    ///
    /// # Errors
    ///
    /// This method will return an error if the build process fails.
    pub fn build_ptx(&mut self) -> Result<&Vec<Ptx>, String> {
        let manifest_path = self.project_dir.join("Cargo.toml");
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
            let ptx_path = self.project_dir.join("target/nvptx64-nvidia-cuda/release");
            if ptx_path.exists() {
                let ptx_files: Vec<Ptx> = fs::read_dir(ptx_path)
                .map_err(|e| e.to_string())?
                .filter_map(|entry| {
                    let path = entry.unwrap().path();
                    if let Some("ptx") = path.extension().and_then(|ext| ext.to_str()) {
                        let mut src = String::new();
                        File::open(path)
                            .unwrap()
                            .read_to_string(&mut src)
                            .unwrap();
                        Some(Ptx::from_src(src))
                    } else {
                        None
                    }
                })
                .collect();
                self.kernels = Some(ptx_files);
                Ok(self.peek_kernels().unwrap())
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
