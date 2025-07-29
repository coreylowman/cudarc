use super::sys;

pub struct CufileError(pub sys::CUfileOpError);

impl sys::CUfileOpError {
    pub fn result(self) -> Result<(), CufileError> {
        match self {
            sys::CUfileOpError::CU_FILE_SUCCESS => Ok(()),
            _ => Err(CufileError(self)),
        }
    }
}

impl sys::CUfileError {
    pub fn result(self) -> Result<(), CufileError> {
        match self.err {
            sys::CUfileOpError::CU_FILE_SUCCESS => Ok(()),
            _ => Err(CufileError(self.err)),
        }
    }
}

impl std::fmt::Debug for CufileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CufileError({:?})", self.0)
    }
}

impl std::fmt::Display for CufileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "CufileError({:?})", self.0)
    }
}

impl std::error::Error for CufileError {}

pub unsafe fn driver_open() -> Result<(), CufileError> {
    sys::cuFileDriverOpen().result()
}

pub unsafe fn driver_get_properties() -> Result<sys::CUfileDrvProps, CufileError> {
    let mut props = sys::CUfileDrvProps::default();
    sys::cuFileDriverGetProperties(&mut props).result()?;
    Ok(props)
}

pub unsafe fn driver_close() -> Result<(), CufileError> {
    sys::cuFileDriverClose_v2().result()
}

pub unsafe fn handle_register(fd: i32) -> Result<sys::CUfileHandle_t, CufileError> {
    let mut cuda_file = sys::CUfileDescr_t::default();
    cuda_file.type_ = sys::CUfileFileHandleType::CU_FILE_HANDLE_TYPE_OPAQUE_FD;
    cuda_file.handle = sys::CUfileDescr_t__bindgen_ty_1::default();
    cuda_file.handle.fd = fd;

    let mut fh: sys::CUfileHandle_t = std::ptr::null_mut();

    sys::cuFileHandleRegister(&mut fh, &mut cuda_file).result()?;

    Ok(fh)
}

pub unsafe fn read(
    fh: sys::CUfileHandle_t,
    buf_ptr_base: *mut ::core::ffi::c_void,
    size: usize,
    file_offset: usize,
    buf_ptr_offset: usize,
) -> Result<isize, CufileError> {
    let bytes_read = sys::cuFileRead(
        fh,
        buf_ptr_base,
        size,
        file_offset as i64,
        buf_ptr_offset as i64,
    );

    Ok(bytes_read)
}
