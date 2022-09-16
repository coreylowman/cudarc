use super::sys as nvrtc;
use std::{
    ffi::{CStr, CString},
    mem::MaybeUninit,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NvrtcError(nvrtc::nvrtcResult);

impl nvrtc::nvrtcResult {
    pub fn result(self) -> Result<(), NvrtcError> {
        match self {
            nvrtc::nvrtcResult::NVRTC_SUCCESS => Ok(()),
            _ => Err(NvrtcError(self)),
        }
    }
}

impl std::fmt::Display for NvrtcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }
}

impl std::error::Error for NvrtcError {}

pub unsafe fn compile_program<'a>(
    prog: nvrtc::nvrtcProgram,
    options: &[&'a str],
) -> Result<(), NvrtcError> {
    let c_strings: Vec<CString> = options.iter().map(|&o| CString::new(o).unwrap()).collect();
    let c_strs: Vec<&CStr> = c_strings.iter().map(CString::as_c_str).collect();
    let opts: Vec<*const std::ffi::c_char> = c_strs.iter().cloned().map(CStr::as_ptr).collect();
    nvrtc::nvrtcCompileProgram(prog, options.len() as i32, opts.as_ptr()).result()
}

pub fn create_program<S: AsRef<str>>(src: S) -> Result<nvrtc::nvrtcProgram, NvrtcError> {
    let src_c = CString::new(src.as_ref()).unwrap();
    let mut prog = MaybeUninit::uninit();
    unsafe {
        nvrtc::nvrtcCreateProgram(
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

pub unsafe fn destroy_program(prog: nvrtc::nvrtcProgram) -> Result<(), NvrtcError> {
    nvrtc::nvrtcDestroyProgram(&prog as *const _ as *mut _).result()
}

pub unsafe fn get_ptx(prog: nvrtc::nvrtcProgram) -> Result<Vec<std::ffi::c_char>, NvrtcError> {
    let mut size: usize = 0;
    nvrtc::nvrtcGetPTXSize(prog, &mut size as *mut _).result()?;

    let mut ptx_src: Vec<std::ffi::c_char> = vec![0i8; size];
    nvrtc::nvrtcGetPTX(prog, ptx_src.as_mut_ptr()).result()?;
    Ok(ptx_src)
}

pub unsafe fn get_program_log(
    prog: nvrtc::nvrtcProgram,
) -> Result<Vec<std::ffi::c_char>, NvrtcError> {
    let mut size: usize = 0;
    nvrtc::nvrtcGetProgramLogSize(prog, &mut size as *mut _).result()?;

    let mut log_src: Vec<std::ffi::c_char> = vec![0; size];
    nvrtc::nvrtcGetProgramLog(prog, log_src.as_mut_ptr()).result()?;
    Ok(log_src)
}
