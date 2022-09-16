//! A thin wrapper around [sys] providing [Result]s with [NvrtcError].

use super::sys;
use std::{
    ffi::{CStr, CString},
    mem::MaybeUninit,
};

/// TODO
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvrtcError(pub sys::nvrtcResult);

impl sys::nvrtcResult {
    /// Transforms into a [Result] of [NvrtcError]
    pub fn result(self) -> Result<(), NvrtcError> {
        match self {
            sys::nvrtcResult::NVRTC_SUCCESS => Ok(()),
            _ => Err(NvrtcError(self)),
        }
    }
}

impl std::fmt::Display for NvrtcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for NvrtcError {}

/// Creates a program
pub fn create_program<S: AsRef<str>>(src: S) -> Result<sys::nvrtcProgram, NvrtcError> {
    let src_c = CString::new(src.as_ref()).unwrap();
    let mut prog = MaybeUninit::uninit();
    unsafe {
        sys::nvrtcCreateProgram(
            prog.as_mut_ptr(),
            src_c.as_c_str().as_ptr(),
            std::ptr::null(),
            0,
            std::ptr::null(),
            std::ptr::null(),
        )
        .result()?;
        Ok(prog.assume_init())
    }
}

/// Compiles something
pub unsafe fn compile_program<'a>(
    prog: sys::nvrtcProgram,
    options: &[&'a str],
) -> Result<(), NvrtcError> {
    let c_strings: Vec<CString> = options.iter().map(|&o| CString::new(o).unwrap()).collect();
    let c_strs: Vec<&CStr> = c_strings.iter().map(CString::as_c_str).collect();
    let opts: Vec<*const std::os::raw::c_char> = c_strs.iter().cloned().map(CStr::as_ptr).collect();
    sys::nvrtcCompileProgram(prog, options.len() as i32, opts.as_ptr()).result()
}

/// Release
pub unsafe fn destroy_program(prog: sys::nvrtcProgram) -> Result<(), NvrtcError> {
    sys::nvrtcDestroyProgram(&prog as *const _ as *mut _).result()
}

/// Extract ptx
pub unsafe fn get_ptx(prog: sys::nvrtcProgram) -> Result<Vec<std::os::raw::c_char>, NvrtcError> {
    let mut size: usize = 0;
    sys::nvrtcGetPTXSize(prog, &mut size as *mut _).result()?;

    let mut ptx_src: Vec<std::os::raw::c_char> = vec![0i8; size];
    sys::nvrtcGetPTX(prog, ptx_src.as_mut_ptr()).result()?;
    Ok(ptx_src)
}

/// Extract log
pub unsafe fn get_program_log(
    prog: sys::nvrtcProgram,
) -> Result<Vec<std::os::raw::c_char>, NvrtcError> {
    let mut size: usize = 0;
    sys::nvrtcGetProgramLogSize(prog, &mut size as *mut _).result()?;

    let mut log_src: Vec<std::os::raw::c_char> = vec![0; size];
    sys::nvrtcGetProgramLog(prog, log_src.as_mut_ptr()).result()?;
    Ok(log_src)
}
