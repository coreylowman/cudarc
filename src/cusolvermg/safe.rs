use super::{result, sys};

pub use super::result::CusolverMgError;

/// CusolverMG wrapper handle
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#using-the-cusolvermg-api)
///
/// This is [NOT thread safe](https://docs.nvidia.com/cuda/cusolver/index.html#mg-thread-safety) in the rust sense:
/// > The library is thread-safe only if there is one cuSolverMG context per thread.
#[derive(Debug)]
pub struct MgHandle {
    handle: sys::cusolverMgHandle_t,
}

impl Drop for MgHandle {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::mg_destroy(handle) }.unwrap();
        }
    }
}

impl MgHandle {
    /// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgcreate)
    pub fn new() -> Result<Self, CusolverMgError> {
        let handle = result::mg_create()?;
        Ok(Self { handle })
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgdeviceselect)
    pub fn device_select(&self, devices: &[i32]) -> Result<(), CusolverMgError> {
        unsafe { result::mg_device_select(self.handle, devices) }
    }

    pub fn cu(&self) -> sys::cusolverMgHandle_t {
        self.handle
    }
}

/// A multi nod grid object - used to create a [MatrixDesc]
#[derive(Debug)]
pub struct Grid {
    grid: sys::cudaLibMgGrid_t,
}

impl Drop for Grid {
    fn drop(&mut self) {
        let grid = std::mem::replace(&mut self.grid, std::ptr::null_mut());
        if !grid.is_null() {
            unsafe { result::mg_destroy_grid(grid) }.unwrap();
        }
    }
}

impl Grid {
    /// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgcreatedevicegrid)
    pub fn new(
        col_devices: &[i32],
        mapping: sys::cusolverMgGridMapping_t,
    ) -> Result<Self, CusolverMgError> {
        let grid = result::mg_create_device_grid(col_devices, mapping)?;
        Ok(Self { grid })
    }

    pub fn cu(&self) -> sys::cudaLibMgGrid_t {
        self.grid
    }
}

/// Matrix description, used to invoke functions in conjunction with [MgHandle]
#[derive(Debug)]
pub struct MatrixDesc {
    desc: sys::cudaLibMgMatrixDesc_t,
}

impl Drop for MatrixDesc {
    fn drop(&mut self) {
        let desc = std::mem::replace(&mut self.desc, std::ptr::null_mut());
        if !desc.is_null() {
            unsafe { result::mg_destroy_matrix_desc(desc) }.unwrap();
        }
    }
}

impl MatrixDesc {
    /// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgcreatematrixdesc)
    pub fn new(
        grid: &Grid,
        num_rows: i64,
        num_cols: i64,
        row_block_size: i64,
        col_block_size: i64,
        dtype: sys::cudaDataType_t,
    ) -> Result<Self, CusolverMgError> {
        let desc = unsafe {
            result::mg_create_matrix_desc(
                num_rows,
                num_cols,
                row_block_size,
                col_block_size,
                dtype,
                grid.cu(),
            )
        }?;
        Ok(Self { desc })
    }
}
