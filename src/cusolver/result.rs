use std::mem::MaybeUninit;

use super::sys;

/// Wrapper around [sys::cusolverStatus_t]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CusolverError(pub sys::cusolverStatus_t);

impl sys::cusolverStatus_t {
    pub fn result(self) -> Result<(), CusolverError> {
        match self {
            sys::cusolverStatus_t::CUSOLVER_STATUS_SUCCESS => Ok(()),
            _ => Err(CusolverError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CusolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CusolverError {}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdncreate)
pub fn dn_create() -> Result<sys::cusolverDnHandle_t, CusolverError> {
    let mut handle = MaybeUninit::uninit();
    unsafe { sys::cusolverDnCreate(handle.as_mut_ptr()) }.result()?;
    Ok(unsafe { handle.assume_init() })
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdndestroy)
///
/// # Safety
/// Make sure `handle` has not already been freed
pub unsafe fn dn_destroy(handle: sys::cusolverDnHandle_t) -> Result<(), CusolverError> {
    sys::cusolverDnDestroy(handle).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnsetstream)
///
/// # Safety
/// Make sure `handle` and `stream` are valid (not destroyed)
pub unsafe fn dn_set_stream(
    handle: sys::cusolverDnHandle_t,
    stream: sys::cudaStream_t,
) -> Result<(), CusolverError> {
    sys::cusolverDnSetStream(handle, stream).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdngetdeterministicmode)
///
/// # Safety
/// Make sure `handle` is valid (not destroyed)
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000",
))]
pub unsafe fn dn_get_deterministic_mode(
    handle: sys::cusolverDnHandle_t,
) -> Result<sys::cusolverDeterministicMode_t, CusolverError> {
    let mut mode = MaybeUninit::uninit();
    sys::cusolverDnGetDeterministicMode(handle, mode.as_mut_ptr()).result()?;
    Ok(unsafe { mode.assume_init() })
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnsetdeterministicmode)
///
/// # Safety
/// Make sure `handle` is valid (not destroyed)
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000",
))]
pub unsafe fn dn_set_deterministic_mode(
    handle: sys::cusolverDnHandle_t,
    mode: sys::cusolverDeterministicMode_t,
) -> Result<(), CusolverError> {
    sys::cusolverDnSetDeterministicMode(handle, mode).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdncreateparams)
pub fn dn_create_params() -> Result<sys::cusolverDnParams_t, CusolverError> {
    let mut params = MaybeUninit::uninit();
    unsafe { sys::cusolverDnCreateParams(params.as_mut_ptr()) }.result()?;
    Ok(unsafe { params.assume_init() })
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnsetadvoptions)
///
/// # Safety
/// Make sure `params` is valid (not destroyed)
pub unsafe fn dn_set_adv_options(
    params: sys::cusolverDnParams_t,
    function: sys::cusolverDnFunction_t,
    algo: sys::cusolverAlgMode_t,
) -> Result<(), CusolverError> {
    sys::cusolverDnSetAdvOptions(params, function, algo).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdndestroyparams)
///
/// # Safety
/// Make sure `params` is valid (not destroyed)
pub unsafe fn dn_destroy_params(params: sys::cusolverDnParams_t) -> Result<(), CusolverError> {
    sys::cusolverDnDestroyParams(params).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverspcreate)
pub fn sp_create() -> Result<sys::cusolverSpHandle_t, CusolverError> {
    let mut handle = MaybeUninit::uninit();
    unsafe { sys::cusolverSpCreate(handle.as_mut_ptr()) }.result()?;
    Ok(unsafe { handle.assume_init() })
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverspdestroy)
///
/// # Safety
/// Make sure `handle` is valid (not destroyed)
pub unsafe fn sp_destroy(handle: sys::cusolverSpHandle_t) -> Result<(), CusolverError> {
    sys::cusolverSpDestroy(handle).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverspsetstream)
///
/// # Safety
/// Make sure `handle` and `stream` are valid (not destroyed)
pub unsafe fn sp_set_stream(
    handle: sys::cusolverSpHandle_t,
    stream: sys::cudaStream_t,
) -> Result<(), CusolverError> {
    sys::cusolverSpSetStream(handle, stream).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfcreate)
pub fn rf_create() -> Result<sys::cusolverRfHandle_t, CusolverError> {
    let mut handle = MaybeUninit::uninit();
    unsafe { sys::cusolverRfCreate(handle.as_mut_ptr()) }.result()?;
    Ok(unsafe { handle.assume_init() })
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfdestroy)
///
/// # Safety
/// Make sure `handle` is valid (not destroyed)
pub unsafe fn rf_destroy(handle: sys::cusolverRfHandle_t) -> Result<(), CusolverError> {
    sys::cusolverRfDestroy(handle).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfsetmatrixformat)
pub unsafe fn rf_set_matrix_format(
    handle: sys::cusolverRfHandle_t,
    format: sys::cusolverRfMatrixFormat_t,
    diag: sys::cusolverRfUnitDiagonal_t,
) -> Result<(), CusolverError> {
    sys::cusolverRfSetMatrixFormat(handle, format, diag).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfsetnumericproperties)
pub unsafe fn rf_set_numeric_properties(
    handle: sys::cusolverRfHandle_t,
    zero: f64,
    boost: f64,
) -> Result<(), CusolverError> {
    sys::cusolverRfSetNumericProperties(handle, zero, boost).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfsetresetvaluesfastmode)
pub unsafe fn rf_set_reset_values_fast_mode(
    handle: sys::cusolverRfHandle_t,
    fast_mode: sys::cusolverRfResetValuesFastMode_t,
) -> Result<(), CusolverError> {
    sys::cusolverRfSetResetValuesFastMode(handle, fast_mode).result()
}

/// See [cuda docs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolverrfsetalgs)
pub unsafe fn rf_set_algs(
    handle: sys::cusolverRfHandle_t,
    fact_alg: sys::cusolverRfFactorization_t,
    alg: sys::cusolverRfTriangularSolve_t,
) -> Result<(), CusolverError> {
    sys::cusolverRfSetAlgs(handle, fact_alg, alg).result()
}
