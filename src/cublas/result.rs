use super::sys;
use core::ffi::c_int;
use core::mem::MaybeUninit;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CublasError(pub sys::cublasStatus_t);

impl sys::cublasStatus_t {
    pub fn result(self) -> Result<(), CublasError> {
        match self {
            sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            _ => Err(CublasError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CublasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CublasError {}

pub fn create_handle() -> Result<sys::cublasHandle_t, CublasError> {
    let mut handle = MaybeUninit::uninit();
    unsafe {
        sys::cublasCreate_v2(handle.as_mut_ptr()).result()?;
        Ok(handle.assume_init())
    }
}

pub unsafe fn destroy_handle(handle: sys::cublasHandle_t) -> Result<(), CublasError> {
    sys::cublasDestroy_v2(handle).result()
}

pub unsafe fn set_stream(
    handle: sys::cublasHandle_t,
    stream: sys::cudaStream_t,
) -> Result<(), CublasError> {
    sys::cublasSetStream_v2(handle, stream).result()
}

pub unsafe fn sgemv(
    handle: sys::cublasHandle_t,
    trans: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    x: *const f32,
    incx: c_int,
    beta: *const f32,
    y: *mut f32,
    incy: c_int,
) -> Result<(), CublasError> {
    sys::cublasSgemv_v2(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy).result()
}

pub unsafe fn dgemv(
    handle: sys::cublasHandle_t,
    trans: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    x: *const f64,
    incx: c_int,
    beta: *const f64,
    y: *mut f64,
    incy: c_int,
) -> Result<(), CublasError> {
    sys::cublasDgemv_v2(handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy).result()
}

pub unsafe fn sgemm(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    b: *const f32,
    ldb: c_int,
    beta: *const f32,
    c: *mut f32,
    ldc: c_int,
) -> Result<(), CublasError> {
    sys::cublasSgemm_v2(
        handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    )
    .result()
}

pub unsafe fn dgemm(
    handle: sys::cublasHandle_t,
    transa: sys::cublasOperation_t,
    transb: sys::cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    b: *const f64,
    ldb: c_int,
    beta: *const f64,
    c: *mut f64,
    ldc: c_int,
) -> Result<(), CublasError> {
    sys::cublasDgemm_v2(
        handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
    )
    .result()
}
