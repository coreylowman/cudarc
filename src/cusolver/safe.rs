use std::sync::Arc;

use crate::driver::CudaStream;

use super::{result, sys};

pub use super::result::CusolverError;

/// Handle for Dense LAPACK functions
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdn-dense-lapack-function-reference)
///
/// This is [thread safe](https://docs.nvidia.com/cuda/cusolver/index.html#thread-safety)
#[derive(Debug)]
pub struct DnHandle {
    handle: sys::cusolverDnHandle_t,
    stream: Arc<CudaStream>,
}

unsafe impl Send for DnHandle {}
unsafe impl Sync for DnHandle {}

impl Drop for DnHandle {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::dn_destroy(self.handle) }.unwrap();
        }
    }
}

impl DnHandle {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CusolverError> {
        let handle = result::dn_create()?;
        unsafe { result::dn_set_stream(handle, stream.cu_stream() as _) }?;
        Ok(Self { handle, stream })
    }

    pub fn cu(&self) -> sys::cusolverDnHandle_t {
        self.handle
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub unsafe fn set_stream(&mut self, stream: Arc<CudaStream>) -> Result<(), CusolverError> {
        self.stream = stream;
        result::dn_set_stream(self.handle, self.stream.cu_stream() as _)
    }

    #[cfg(feature = "gte-12020")]
    pub fn set_deterministic_mode(
        &self,
        mode: sys::cusolverDeterministicMode_t,
    ) -> Result<(), CusolverError> {
        unsafe { result::dn_set_deterministic_mode(self.handle, mode) }
    }

    #[cfg(feature = "gte-12020")]
    pub fn get_deterministic_mode(&self) -> sys::cusolverDeterministicMode_t {
        // NOTE: the possible errors here are `CUSOLVER_STATUS_NOT_INITIALIZED`, which is not possible
        // since we have `&self`, and `CUSOLVER_STATUS_INVALID_VALUE` when mode is a null pointer, which
        // is handled via result level. So we can safely unwrap
        unsafe { result::dn_get_deterministic_mode(self.handle) }.unwrap()
    }
}

#[derive(Debug)]
pub struct DnParams {
    params: sys::cusolverDnParams_t,
}

impl Drop for DnParams {
    fn drop(&mut self) {
        let params = std::mem::replace(&mut self.params, std::ptr::null_mut());
        if !params.is_null() {
            unsafe { result::dn_destroy_params(params) }.unwrap();
        }
    }
}

impl DnParams {
    pub fn new(
        function: sys::cusolverDnFunction_t,
        algo: sys::cusolverAlgMode_t,
    ) -> Result<Self, CusolverError> {
        let params = result::dn_create_params()?;
        unsafe { result::dn_set_adv_options(params, function, algo) }?;
        Ok(Self { params })
    }

    pub fn cu(&self) -> sys::cusolverDnParams_t {
        self.params
    }
}

/// Handle for Sparse LAPACK functions
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolversp-sparse-lapack-function-reference)
///
/// This is [thread safe](https://docs.nvidia.com/cuda/cusolver/index.html#thread-safety)
#[derive(Debug)]
pub struct SpHandle {
    handle: sys::cusolverSpHandle_t,
    stream: Arc<CudaStream>,
}

unsafe impl Send for SpHandle {}
unsafe impl Sync for SpHandle {}

impl Drop for SpHandle {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::sp_destroy(self.handle) }.unwrap();
        }
    }
}

impl SpHandle {
    pub fn new(stream: Arc<CudaStream>) -> Result<Self, CusolverError> {
        let handle = result::sp_create()?;
        unsafe { result::sp_set_stream(handle, stream.cu_stream() as _) }?;
        Ok(Self { handle, stream })
    }

    pub fn cu(&self) -> sys::cusolverSpHandle_t {
        self.handle
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub unsafe fn set_stream(&mut self, stream: Arc<CudaStream>) -> Result<(), CusolverError> {
        self.stream = stream;
        result::sp_set_stream(self.handle, self.stream.cu_stream() as _)
    }
}

/// Handle for refactorization functions
///
/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrf-refactorization-reference)
///
/// This is [thread safe](https://docs.nvidia.com/cuda/cusolver/index.html#thread-safety)
#[derive(Debug)]
pub struct RfHandle {
    handle: sys::cusolverRfHandle_t,
}

unsafe impl Send for RfHandle {}
unsafe impl Sync for RfHandle {}

impl Drop for RfHandle {
    fn drop(&mut self) {
        let handle = std::mem::replace(&mut self.handle, std::ptr::null_mut());
        if !handle.is_null() {
            unsafe { result::rf_destroy(self.handle) }.unwrap();
        }
    }
}

impl RfHandle {
    pub fn new() -> Result<Self, CusolverError> {
        let handle = result::rf_create()?;
        Ok(Self { handle })
    }

    pub fn cu(&self) -> sys::cusolverRfHandle_t {
        self.handle
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfsetmatrixformat)
    pub fn set_matrix_format(
        &self,
        format: sys::cusolverRfMatrixFormat_t,
        diag: sys::cusolverRfUnitDiagonal_t,
    ) {
        unsafe { result::rf_set_matrix_format(self.handle, format, diag) }.unwrap()
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfsetnumericproperties)
    pub fn set_numeric_properties(&self, zero: f64, boost: f64) {
        unsafe { result::rf_set_numeric_properties(self.handle, zero, boost) }.unwrap()
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfsetresetvaluesfastmode)
    pub fn set_reset_values_fast_mode(&self, fast_mode: sys::cusolverRfResetValuesFastMode_t) {
        unsafe { result::rf_set_reset_values_fast_mode(self.handle, fast_mode) }.unwrap()
    }

    /// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfsetalgs)
    pub fn set_algs(
        &self,
        fact_alg: sys::cusolverRfFactorization_t,
        alg: sys::cusolverRfTriangularSolve_t,
    ) {
        unsafe { result::rf_set_algs(self.handle, fact_alg, alg) }.unwrap();
    }
}
