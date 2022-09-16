use super::result;
use std::ffi::{CStr, CString};

/// TODO
#[derive(Debug, Clone)]
pub struct Ptx {
    pub(crate) image: Vec<std::os::raw::c_char>,
}

/// Calls [compile_ptx_with_opts] with no options.
pub fn compile_ptx<S: AsRef<str>>(src: S) -> Result<Ptx, CompileError> {
    compile_ptx_with_opts(src, Default::default())
}

/// TODO
pub fn compile_ptx_with_opts<S: AsRef<str>>(
    src: S,
    opts: CompileOptions,
) -> Result<Ptx, CompileError> {
    let options = opts.build();
    let prog = result::create_program(src).map_err(CompileError::CreationError)?;
    unsafe {
        result::compile_program(prog, &options).map_err(|error| {
            let log = result::get_program_log(prog).unwrap();
            CompileError::CompileError {
                error,
                log: CStr::from_ptr(log.as_ptr()).to_owned(),
                options,
            }
        })?;
        let image = result::get_ptx(prog).map_err(CompileError::GetPtxError)?;
        Ok(Ptx { image })
    }
}

/// TODO
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompileError {
    CreationError(result::NvrtcError),
    CompileError {
        error: result::NvrtcError,
        log: CString,
        options: Vec<&'static str>,
    },
    GetPtxError(result::NvrtcError),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for CompileError {}

/// TODO add more of the options
///
/// See <https://docs.nvidia.com/cuda/nvrtc/index.html#group__options>
#[derive(Clone, Copy, Debug, Default, Hash, PartialEq, Eq)]
pub struct CompileOptions {
    pub ftz: Option<bool>,
    pub prec_sqrt: Option<bool>,
    pub prec_div: Option<bool>,
    pub fmad: Option<bool>,
    pub use_fast_math: Option<bool>,
}

impl CompileOptions {
    pub(crate) fn build(self) -> Vec<&'static str> {
        let mut options = Vec::with_capacity(4);

        match self.ftz {
            Some(true) => options.push("--ftz=true"),
            Some(false) => options.push("--ftz=false"),
            None => {}
        }

        match self.prec_sqrt {
            Some(true) => options.push("--prec-sqrt=true"),
            Some(false) => options.push("--prec-sqrt=false"),
            None => {}
        }

        match self.prec_div {
            Some(true) => options.push("--prec-div=true"),
            Some(false) => options.push("--prec-div=false"),
            None => {}
        }

        match self.fmad {
            Some(true) => options.push("--fmad=true"),
            Some(false) => options.push("--fmad=false"),
            None => {}
        }

        match self.use_fast_math {
            Some(true) => options.push("--fmad=true"),
            _ => {}
        }

        options
    }
}
