use super::result;
use std::ffi::{CStr, CString};

#[derive(Debug)]
pub struct Ptx {
    pub(crate) image: Vec<std::ffi::c_char>,
}

pub fn compile_ptx<S: AsRef<str>>(src: S) -> Result<Ptx, CompilationError> {
    let prog = result::create_program(src).map_err(CompilationError::CreationError)?;
    unsafe {
        result::compile_program(prog, &[]).map_err(|error| {
            let log = result::get_program_log(prog).unwrap();
            CompilationError::CompileError {
                error,
                log: CStr::from_ptr(log.as_ptr()).to_owned(),
            }
        })?;
        let image = result::get_ptx(prog).map_err(CompilationError::GetPtxError)?;
        Ok(Ptx { image })
    }
}

#[derive(Debug)]
pub enum CompilationError {
    CreationError(result::NvrtcError),
    CompileError {
        error: result::NvrtcError,
        log: CString,
    },
    GetPtxError(result::NvrtcError),
}

impl std::fmt::Display for CompilationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CompilationError {}

/// See https://docs.nvidia.com/cuda/nvrtc/index.html#group__options
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub struct CompileOptions {
    gpu_arch: GpuArchitecture,
    relocatable_device_code: bool,
    extensible_whole_program: bool,
    device_debug: bool,
    generate_line_info: bool,
    maxrregcount: Option<usize>,
    ftz: bool,
    prec_sqrt: bool,
    prec_div: bool,
    fmad: bool,
    extra_device_vectorization: bool,
    modify_stack_limit: bool,
    dlink_time_opt: bool,
    std: Option<LanguageDialect>,
    builtin_move_forward: bool,
    builtin_initializer_list: bool,
    disable_warnings: bool,
    restrict: bool,
    device_as_default_execution_space: bool,
    device_int128: bool,
    version_ident: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            gpu_arch: GpuArchitecture::Compute52,
            relocatable_device_code: false,
            extensible_whole_program: false,
            device_debug: false,
            generate_line_info: false,
            maxrregcount: None,
            ftz: false,
            prec_sqrt: true,
            prec_div: true,
            fmad: true,
            extra_device_vectorization: false,
            modify_stack_limit: true,
            dlink_time_opt: false,
            std: None,
            builtin_move_forward: true,
            builtin_initializer_list: true,
            disable_warnings: false,
            restrict: false,
            device_as_default_execution_space: false,
            device_int128: false,
            version_ident: false,
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum GpuArchitecture {
    Compute35,
    Compute37,
    Compute50,
    Compute52,
    Compute53,
    Compute60,
    Compute61,
    Compute62,
    Compute70,
    Compute72,
    Compute75,
    Compute80,
    Sm35,
    Sm37,
    Sm50,
    Sm52,
    Sm53,
    Sm60,
    Sm61,
    Sm62,
    Sm70,
    Sm72,
    Sm75,
    Sm80,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
pub enum LanguageDialect {
    Cpp03,
    Cpp11,
    Cpp14,
    Cpp17,
    Cpp20,
}
