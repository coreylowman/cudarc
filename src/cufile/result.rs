use std::mem::MaybeUninit;

use super::sys;

/// Wrapper around [sys::CUfileOpError] and [std::io::Error]
#[derive(Debug)]
pub enum CufileError {
    Cufile(sys::CUfileOpError),
    IO(std::io::Error),
}

impl sys::CUfileOpError {
    pub fn result(self) -> Result<(), CufileError> {
        match self {
            sys::CUfileOpError::CU_FILE_SUCCESS => Ok(()),
            _ => Err(CufileError::Cufile(self)),
        }
    }
}

impl sys::CUfileError {
    pub fn result(self) -> Result<(), CufileError> {
        match self.err {
            sys::CUfileOpError::CU_FILE_SUCCESS => Ok(()),
            _ => Err(CufileError::Cufile(self.err)),
        }
    }
}

impl std::fmt::Display for CufileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::error::Error for CufileError {}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufiledriveropen)
pub fn driver_open() -> Result<(), CufileError> {
    unsafe { sys::cuFileDriverOpen() }.result()
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufiledriverclose)
pub fn driver_close() -> Result<(), CufileError> {
    unsafe { sys::cuFileDriverClose_v2() }.result()
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufiledrivergetproperties)
pub fn driver_get_properties() -> Result<sys::CUfileDrvProps, CufileError> {
    let mut props = MaybeUninit::uninit();
    unsafe { sys::cuFileDriverGetProperties(props.as_mut_ptr()) }.result()?;
    Ok(unsafe { props.assume_init() })
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufiledriversetpollmode-bool-poll-size-t-poll-threshold-size)
pub fn driver_set_poll_mode(poll: bool, poll_threshold_size: usize) -> Result<(), CufileError> {
    unsafe { sys::cuFileDriverSetPollMode(poll, poll_threshold_size) }.result()
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufiledriversetmaxdirectiosize-size-t-max-direct-io-size)
pub fn driver_set_max_direct_io_size(max_direct_io_size: usize) -> Result<(), CufileError> {
    unsafe { sys::cuFileDriverSetMaxDirectIOSize(max_direct_io_size) }.result()
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#size-t-max-cache-size)
pub fn driver_set_max_cache_size(max_cache_size: usize) -> Result<(), CufileError> {
    unsafe { sys::cuFileDriverSetMaxCacheSize(max_cache_size) }.result()
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufiledriversetmaxpinnedmemsize-size-t-max-pinned-memory-size)
pub fn driver_set_max_pinned_mem_size(max_pinned_size: usize) -> Result<(), CufileError> {
    unsafe { sys::cuFileDriverSetMaxPinnedMemSize(max_pinned_size) }.result()
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilehandleregister)
pub fn handle_register(descr: &sys::CUfileDescr_t) -> Result<sys::CUfileHandle_t, CufileError> {
    let mut fh = MaybeUninit::uninit();
    unsafe { sys::cuFileHandleRegister(fh.as_mut_ptr(), descr as *const _ as _) }.result()?;
    Ok(unsafe { fh.assume_init() })
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilehandlederegister)
///
/// # Safety
/// `fh` must be valid (not deregistered already)
pub unsafe fn handle_deregister(fh: sys::CUfileHandle_t) -> Result<(), CufileError> {
    sys::cuFileHandleDeregister(fh);
    Ok(())
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufileread)
///
/// # Safety
/// Ensure data ranges are valid & pointers are valid
pub unsafe fn read(
    fh: sys::CUfileHandle_t,
    buf_ptr_base: *mut ::core::ffi::c_void,
    size: usize,
    file_offset: i64,
    buf_ptr_offset: i64,
) -> Result<isize, CufileError> {
    let bytes_read = sys::cuFileRead(fh, buf_ptr_base, size, file_offset, buf_ptr_offset);

    if bytes_read == -1 {
        Err(CufileError::IO(std::io::Error::last_os_error()))
    } else if bytes_read < -1 {
        let errno = -bytes_read;
        let errno = errno as u32;
        Err(CufileError::Cufile(std::mem::transmute(errno)))
    } else {
        Ok(bytes_read)
    }
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilewrite)
/// # Safety
/// Ensure data rangefs are valid & pointers are valid
pub unsafe fn write(
    fh: sys::CUfileHandle_t,
    buf_ptr_base: *mut ::core::ffi::c_void,
    size: usize,
    file_offset: i64,
    buf_ptr_offset: i64,
) -> Result<isize, CufileError> {
    let bytes_written = sys::cuFileWrite(fh, buf_ptr_base, size, file_offset, buf_ptr_offset);

    if bytes_written == -1 {
        Err(CufileError::IO(std::io::Error::last_os_error()))
    } else if bytes_written < -1 {
        let errno = -bytes_written;
        let errno = errno as u32;
        Err(CufileError::Cufile(std::mem::transmute(errno)))
    } else {
        Ok(bytes_written)
    }
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilebufregister)
///
/// # Safety
/// Pointers & lengths must be valid
pub unsafe fn buf_register(
    buf_ptr_base: *const ::core::ffi::c_void,
    length: usize,
    flags: i32,
) -> Result<(), CufileError> {
    sys::cuFileBufRegister(buf_ptr_base, length, flags).result()
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilebufderegister)
///
/// # Safety
/// Pointers & lengths must be valid
pub unsafe fn buf_deregister(buf_ptr_base: *const ::core::ffi::c_void) -> Result<(), CufileError> {
    sys::cuFileBufDeregister(buf_ptr_base).result()
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilestreamregister)
///
/// # Safety
/// `stream` must be valid
pub unsafe fn stream_register(stream: sys::CUstream, flags: u32) -> Result<(), CufileError> {
    sys::cuFileStreamRegister(stream, flags).result()
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilestreamderegister)
///
/// # Safety
/// `stream` must be valid
pub unsafe fn stream_deregister(stream: sys::CUstream) -> Result<(), CufileError> {
    sys::cuFileStreamDeregister(stream).result()
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufileread)
///
/// # Safety
/// Ensure data ranges are valid & pointers are valid
pub unsafe fn read_async(
    fh: sys::CUfileHandle_t,
    buf_ptr_base: *mut ::core::ffi::c_void,
    size: &usize,
    file_offset: &i64,
    buf_ptr_offset: &i64,
    bytes_read: &mut isize,
    stream: sys::CUstream,
) -> Result<(), CufileError> {
    sys::cuFileReadAsync(
        fh,
        buf_ptr_base,
        size as *const _ as _,
        file_offset as *const _ as _,
        buf_ptr_offset as *const _ as _,
        bytes_read as _,
        stream,
    )
    .result()
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilewriteasync)
/// # Safety
/// Ensure data rangefs are valid & pointers are valid
pub unsafe fn write_async(
    fh: sys::CUfileHandle_t,
    buf_ptr_base: *mut ::core::ffi::c_void,
    size: &usize,
    file_offset: &i64,
    buf_ptr_offset: &i64,
    bytes_written: &mut isize,
    stream: sys::CUstream,
) -> Result<(), CufileError> {
    sys::cuFileWriteAsync(
        fh,
        buf_ptr_base,
        size as *const _ as _,
        file_offset as *const _ as _,
        buf_ptr_offset as *const _ as _,
        bytes_written as _,
        stream,
    )
    .result()
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilebatchiosetup)
pub fn batch_io_setup(max_nr: u32) -> Result<sys::CUfileBatchHandle_t, CufileError> {
    let mut batch_idp = MaybeUninit::uninit();
    unsafe { sys::cuFileBatchIOSetUp(batch_idp.as_mut_ptr(), max_nr) }.result()?;
    Ok(unsafe { batch_idp.assume_init() })
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilebatchiosubmit)
pub unsafe fn batch_io_submit(
    handle: sys::CUfileBatchHandle_t,
    items: &[sys::CUfileIOParams_t],
    flags: u32,
) -> Result<(), CufileError> {
    sys::cuFileBatchIOSubmit(handle, items.len() as u32, items.as_ptr() as _, flags).result()
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilebatchiogetstatus)
///
/// # Safety
/// A lot of weirdness in this api. Just check cuda docs
pub unsafe fn batch_io_get_status(
    handle: sys::CUfileBatchHandle_t,
    min_nr: u32,
    nr: &mut u32,
    events: &mut [sys::CUfileIOEvents_t],
    timeout: &sys::timespec,
) -> Result<(), CufileError> {
    sys::cuFileBatchIOGetStatus(
        handle,
        min_nr,
        nr as _,
        events.as_mut_ptr(),
        timeout as *const _ as _,
    )
    .result()
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilebatchiocancel)
///
/// # Safety
/// handle must be valid
pub unsafe fn batch_io_cancel(handle: sys::CUfileBatchHandle_t) -> Result<(), CufileError> {
    sys::cuFileBatchIOCancel(handle).result()
}

/// See [cuda docs](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html#cufilebatchiodestroy)
///
/// # Safety
/// Must not already be destroyed
pub unsafe fn batch_io_destroy(handle: sys::CUfileBatchHandle_t) -> Result<(), CufileError> {
    sys::cuFileBatchIODestroy(handle);
    Ok(())
}
