use std::mem::MaybeUninit;

use super::sys;

/// Wrapper around [sys::cusolverStatus_t]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CusolverMgError(pub sys::cusolverStatus_t);

impl sys::cusolverStatus_t {
    pub fn result(self) -> Result<(), CusolverMgError> {
        match self {
            sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS => Ok(()),
            _ => Err(CusolverMgError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CusolverMgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CusolverMgError {}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgcreate)
pub fn mg_create() -> Result<sys::cusolverMgHandle_t, CusolverMgError> {
    let mut handle = MaybeUninit::uninit();
    unsafe { sys::cusolverMgCreate(handle.as_mut_ptr()) }.result()?;
    Ok(unsafe { handle.assume_init() })
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgdestroy)
///
/// # Safety
/// `handle` must be valid (not destroyed)
pub unsafe fn mg_destroy(handle: sys::cusolverMgHandle_t) -> Result<(), CusolverMgError> {
    sys::cusolverMgDestroy(handle).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgdeviceselect)
///
/// # Safety
/// `handle` must be valid (not destroyed)
pub unsafe fn mg_device_select(
    handle: sys::cusolverMgHandle_t,
    devices: &[i32],
) -> Result<(), CusolverMgError> {
    sys::cusolverMgDeviceSelect(handle, devices.len() as i32, devices.as_ptr() as _).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgcreatedevicegrid)
pub fn mg_create_device_grid(
    col_devices: &[i32],
    mapping: sys::cusolverMgGridMapping_t,
) -> Result<sys::cudaLibMgGrid_t, CusolverMgError> {
    let mut grid = MaybeUninit::uninit();
    unsafe {
        sys::cusolverMgCreateDeviceGrid(
            grid.as_mut_ptr(),
            1,
            col_devices.len() as i32,
            col_devices.as_ptr(),
            mapping,
        )
        .result()?;
        Ok(grid.assume_init())
    }
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgdestroygrid)
///
/// # Safety
/// `grid` msut be valid (not destroyed)
pub unsafe fn mg_destroy_grid(grid: sys::cudaLibMgGrid_t) -> Result<(), CusolverMgError> {
    sys::cusolverMgDestroyGrid(grid).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgcreatematrixdesc)
///
/// # Safety
/// `grid` must be valid (not destroyed)
pub unsafe fn mg_create_matrix_desc(
    num_rows: i64,
    num_cols: i64,
    row_block_size: i64,
    col_block_size: i64,
    dtype: sys::cudaDataType,
    grid: sys::cudaLibMgGrid_t,
) -> Result<sys::cudaLibMgMatrixDesc_t, CusolverMgError> {
    let mut desc = MaybeUninit::uninit();
    sys::cusolverMgCreateMatrixDesc(
        desc.as_mut_ptr(),
        num_rows,
        num_cols,
        row_block_size,
        col_block_size,
        dtype,
        grid,
    )
    .result()?;
    Ok(desc.assume_init())
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgdestroymatrixdesc)
///
/// # Safety
/// `desc` must be valid (not destroyed)
pub unsafe fn mg_destroy_matrix_desc(
    desc: sys::cudaLibMgMatrixDesc_t,
) -> Result<(), CusolverMgError> {
    sys::cusolverMgDestroyMatrixDesc(desc).result()
}
