#![cfg_attr(feature = "no-std", no_std)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;
pub use self::cudaDataType_t as cudaDataType;
pub type cudaLibMgGrid_t = *mut ::core::ffi::c_void;
pub type cudaLibMgMatrixDesc_t = *mut ::core::ffi::c_void;
pub type cusolverMgHandle_t = *mut cusolverMgContext;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasFillMode_t {
    CUBLAS_FILL_MODE_LOWER = 0,
    CUBLAS_FILL_MODE_UPPER = 1,
    CUBLAS_FILL_MODE_FULL = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasOperation_t {
    CUBLAS_OP_N = 0,
    CUBLAS_OP_T = 1,
    CUBLAS_OP_C = 2,
    CUBLAS_OP_CONJG = 3,
}
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_C_16F = 6,
    CUDA_R_16BF = 14,
    CUDA_C_16BF = 15,
    CUDA_R_32F = 0,
    CUDA_C_32F = 4,
    CUDA_R_64F = 1,
    CUDA_C_64F = 5,
    CUDA_R_4I = 16,
    CUDA_C_4I = 17,
    CUDA_R_4U = 18,
    CUDA_C_4U = 19,
    CUDA_R_8I = 3,
    CUDA_C_8I = 7,
    CUDA_R_8U = 8,
    CUDA_C_8U = 9,
    CUDA_R_16I = 20,
    CUDA_C_16I = 21,
    CUDA_R_16U = 22,
    CUDA_C_16U = 23,
    CUDA_R_32I = 10,
    CUDA_C_32I = 11,
    CUDA_R_32U = 12,
    CUDA_C_32U = 13,
    CUDA_R_64I = 24,
    CUDA_C_64I = 25,
    CUDA_R_64U = 26,
    CUDA_C_64U = 27,
}
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_C_16F = 6,
    CUDA_R_16BF = 14,
    CUDA_C_16BF = 15,
    CUDA_R_32F = 0,
    CUDA_C_32F = 4,
    CUDA_R_64F = 1,
    CUDA_C_64F = 5,
    CUDA_R_4I = 16,
    CUDA_C_4I = 17,
    CUDA_R_4U = 18,
    CUDA_C_4U = 19,
    CUDA_R_8I = 3,
    CUDA_C_8I = 7,
    CUDA_R_8U = 8,
    CUDA_C_8U = 9,
    CUDA_R_16I = 20,
    CUDA_C_16I = 21,
    CUDA_R_16U = 22,
    CUDA_C_16U = 23,
    CUDA_R_32I = 10,
    CUDA_C_32I = 11,
    CUDA_R_32U = 12,
    CUDA_C_32U = 13,
    CUDA_R_64I = 24,
    CUDA_C_64I = 25,
    CUDA_R_64U = 26,
    CUDA_C_64U = 27,
    CUDA_R_8F_E4M3 = 28,
    CUDA_R_8F_E5M2 = 29,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDataType_t {
    CUDA_R_16F = 2,
    CUDA_C_16F = 6,
    CUDA_R_16BF = 14,
    CUDA_C_16BF = 15,
    CUDA_R_32F = 0,
    CUDA_C_32F = 4,
    CUDA_R_64F = 1,
    CUDA_C_64F = 5,
    CUDA_R_4I = 16,
    CUDA_C_4I = 17,
    CUDA_R_4U = 18,
    CUDA_C_4U = 19,
    CUDA_R_8I = 3,
    CUDA_C_8I = 7,
    CUDA_R_8U = 8,
    CUDA_C_8U = 9,
    CUDA_R_16I = 20,
    CUDA_C_16I = 21,
    CUDA_R_16U = 22,
    CUDA_C_16U = 23,
    CUDA_R_32I = 10,
    CUDA_C_32I = 11,
    CUDA_R_32U = 12,
    CUDA_C_32U = 13,
    CUDA_R_64I = 24,
    CUDA_C_64I = 25,
    CUDA_R_64U = 26,
    CUDA_C_64U = 27,
    CUDA_R_8F_E4M3 = 28,
    CUDA_R_8F_E5M2 = 29,
    CUDA_R_8F_UE8M0 = 30,
    CUDA_R_6F_E2M3 = 31,
    CUDA_R_6F_E3M2 = 32,
    CUDA_R_4F_E2M1 = 33,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverEigMode_t {
    CUSOLVER_EIG_MODE_NOVECTOR = 0,
    CUSOLVER_EIG_MODE_VECTOR = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverMgGridMapping_t {
    CUDALIBMG_GRID_MAPPING_ROW_MAJOR = 1,
    CUDALIBMG_GRID_MAPPING_COL_MAJOR = 0,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverStatus_t {
    CUSOLVER_STATUS_SUCCESS = 0,
    CUSOLVER_STATUS_NOT_INITIALIZED = 1,
    CUSOLVER_STATUS_ALLOC_FAILED = 2,
    CUSOLVER_STATUS_INVALID_VALUE = 3,
    CUSOLVER_STATUS_ARCH_MISMATCH = 4,
    CUSOLVER_STATUS_MAPPING_ERROR = 5,
    CUSOLVER_STATUS_EXECUTION_FAILED = 6,
    CUSOLVER_STATUS_INTERNAL_ERROR = 7,
    CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    CUSOLVER_STATUS_NOT_SUPPORTED = 9,
    CUSOLVER_STATUS_ZERO_PIVOT = 10,
    CUSOLVER_STATUS_INVALID_LICENSE = 11,
    CUSOLVER_STATUS_IRS_PARAMS_NOT_INITIALIZED = 12,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID = 13,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC = 14,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_REFINE = 15,
    CUSOLVER_STATUS_IRS_PARAMS_INVALID_MAXITER = 16,
    CUSOLVER_STATUS_IRS_INTERNAL_ERROR = 20,
    CUSOLVER_STATUS_IRS_NOT_SUPPORTED = 21,
    CUSOLVER_STATUS_IRS_OUT_OF_RANGE = 22,
    CUSOLVER_STATUS_IRS_NRHS_NOT_SUPPORTED_FOR_REFINE_GMRES = 23,
    CUSOLVER_STATUS_IRS_INFOS_NOT_INITIALIZED = 25,
    CUSOLVER_STATUS_IRS_INFOS_NOT_DESTROYED = 26,
    CUSOLVER_STATUS_IRS_MATRIX_SINGULAR = 30,
    CUSOLVER_STATUS_INVALID_WORKSPACE = 31,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusolverMgContext {
    _unused: [u8; 0],
}
impl cublasOperation_t {
    pub const CUBLAS_OP_HERMITAN: cublasOperation_t = cublasOperation_t::CUBLAS_OP_C;
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
impl cudaDataType_t {
    pub const CUDA_R_8F_UE4M3: cudaDataType_t = cudaDataType_t::CUDA_R_8F_E4M3;
}
#[cfg(not(feature = "dynamic-loading"))]
extern "C" {
    pub fn cusolverMgCreate(handle: *mut cusolverMgHandle_t) -> cusolverStatus_t;
    pub fn cusolverMgCreateDeviceGrid(
        grid: *mut cudaLibMgGrid_t,
        numRowDevices: i32,
        numColDevices: i32,
        deviceId: *const i32,
        mapping: cusolverMgGridMapping_t,
    ) -> cusolverStatus_t;
    pub fn cusolverMgCreateMatrixDesc(
        desc: *mut cudaLibMgMatrixDesc_t,
        numRows: i64,
        numCols: i64,
        rowBlockSize: i64,
        colBlockSize: i64,
        dataType: cudaDataType,
        grid: cudaLibMgGrid_t,
    ) -> cusolverStatus_t;
    pub fn cusolverMgDestroy(handle: cusolverMgHandle_t) -> cusolverStatus_t;
    pub fn cusolverMgDestroyGrid(grid: cudaLibMgGrid_t) -> cusolverStatus_t;
    pub fn cusolverMgDestroyMatrixDesc(desc: cudaLibMgMatrixDesc_t) -> cusolverStatus_t;
    pub fn cusolverMgDeviceSelect(
        handle: cusolverMgHandle_t,
        nbDevices: ::core::ffi::c_int,
        deviceId: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverMgGetrf(
        handle: cusolverMgHandle_t,
        M: ::core::ffi::c_int,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        array_d_IPIV: *mut *mut ::core::ffi::c_int,
        computeType: cudaDataType,
        array_d_work: *mut *mut ::core::ffi::c_void,
        lwork: i64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverMgGetrf_bufferSize(
        handle: cusolverMgHandle_t,
        M: ::core::ffi::c_int,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        array_d_IPIV: *mut *mut ::core::ffi::c_int,
        computeType: cudaDataType,
        lwork: *mut i64,
    ) -> cusolverStatus_t;
    pub fn cusolverMgGetrs(
        handle: cusolverMgHandle_t,
        TRANS: cublasOperation_t,
        N: ::core::ffi::c_int,
        NRHS: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        array_d_IPIV: *mut *mut ::core::ffi::c_int,
        array_d_B: *mut *mut ::core::ffi::c_void,
        IB: ::core::ffi::c_int,
        JB: ::core::ffi::c_int,
        descrB: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        array_d_work: *mut *mut ::core::ffi::c_void,
        lwork: i64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverMgGetrs_bufferSize(
        handle: cusolverMgHandle_t,
        TRANS: cublasOperation_t,
        N: ::core::ffi::c_int,
        NRHS: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        array_d_IPIV: *mut *mut ::core::ffi::c_int,
        array_d_B: *mut *mut ::core::ffi::c_void,
        IB: ::core::ffi::c_int,
        JB: ::core::ffi::c_int,
        descrB: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        lwork: *mut i64,
    ) -> cusolverStatus_t;
    pub fn cusolverMgPotrf(
        handle: cusolverMgHandle_t,
        uplo: cublasFillMode_t,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        array_d_work: *mut *mut ::core::ffi::c_void,
        lwork: i64,
        h_info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverMgPotrf_bufferSize(
        handle: cusolverMgHandle_t,
        uplo: cublasFillMode_t,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        lwork: *mut i64,
    ) -> cusolverStatus_t;
    pub fn cusolverMgPotri(
        handle: cusolverMgHandle_t,
        uplo: cublasFillMode_t,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        array_d_work: *mut *mut ::core::ffi::c_void,
        lwork: i64,
        h_info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverMgPotri_bufferSize(
        handle: cusolverMgHandle_t,
        uplo: cublasFillMode_t,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        lwork: *mut i64,
    ) -> cusolverStatus_t;
    pub fn cusolverMgPotrs(
        handle: cusolverMgHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        array_d_B: *mut *mut ::core::ffi::c_void,
        IB: ::core::ffi::c_int,
        JB: ::core::ffi::c_int,
        descrB: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        array_d_work: *mut *mut ::core::ffi::c_void,
        lwork: i64,
        h_info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverMgPotrs_bufferSize(
        handle: cusolverMgHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        array_d_B: *mut *mut ::core::ffi::c_void,
        IB: ::core::ffi::c_int,
        JB: ::core::ffi::c_int,
        descrB: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        lwork: *mut i64,
    ) -> cusolverStatus_t;
    pub fn cusolverMgSyevd(
        handle: cusolverMgHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        W: *mut ::core::ffi::c_void,
        dataTypeW: cudaDataType,
        computeType: cudaDataType,
        array_d_work: *mut *mut ::core::ffi::c_void,
        lwork: i64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverMgSyevd_bufferSize(
        handle: cusolverMgHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        W: *mut ::core::ffi::c_void,
        dataTypeW: cudaDataType,
        computeType: cudaDataType,
        lwork: *mut i64,
    ) -> cusolverStatus_t;
}
#[cfg(feature = "dynamic-loading")]
mod loaded {
    use super::*;
    pub unsafe fn cusolverMgCreate(handle: *mut cusolverMgHandle_t) -> cusolverStatus_t {
        (culib().cusolverMgCreate)(handle)
    }
    pub unsafe fn cusolverMgCreateDeviceGrid(
        grid: *mut cudaLibMgGrid_t,
        numRowDevices: i32,
        numColDevices: i32,
        deviceId: *const i32,
        mapping: cusolverMgGridMapping_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverMgCreateDeviceGrid)(
            grid,
            numRowDevices,
            numColDevices,
            deviceId,
            mapping,
        )
    }
    pub unsafe fn cusolverMgCreateMatrixDesc(
        desc: *mut cudaLibMgMatrixDesc_t,
        numRows: i64,
        numCols: i64,
        rowBlockSize: i64,
        colBlockSize: i64,
        dataType: cudaDataType,
        grid: cudaLibMgGrid_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverMgCreateMatrixDesc)(
            desc,
            numRows,
            numCols,
            rowBlockSize,
            colBlockSize,
            dataType,
            grid,
        )
    }
    pub unsafe fn cusolverMgDestroy(handle: cusolverMgHandle_t) -> cusolverStatus_t {
        (culib().cusolverMgDestroy)(handle)
    }
    pub unsafe fn cusolverMgDestroyGrid(grid: cudaLibMgGrid_t) -> cusolverStatus_t {
        (culib().cusolverMgDestroyGrid)(grid)
    }
    pub unsafe fn cusolverMgDestroyMatrixDesc(
        desc: cudaLibMgMatrixDesc_t,
    ) -> cusolverStatus_t {
        (culib().cusolverMgDestroyMatrixDesc)(desc)
    }
    pub unsafe fn cusolverMgDeviceSelect(
        handle: cusolverMgHandle_t,
        nbDevices: ::core::ffi::c_int,
        deviceId: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverMgDeviceSelect)(handle, nbDevices, deviceId)
    }
    pub unsafe fn cusolverMgGetrf(
        handle: cusolverMgHandle_t,
        M: ::core::ffi::c_int,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        array_d_IPIV: *mut *mut ::core::ffi::c_int,
        computeType: cudaDataType,
        array_d_work: *mut *mut ::core::ffi::c_void,
        lwork: i64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverMgGetrf)(
            handle,
            M,
            N,
            array_d_A,
            IA,
            JA,
            descrA,
            array_d_IPIV,
            computeType,
            array_d_work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverMgGetrf_bufferSize(
        handle: cusolverMgHandle_t,
        M: ::core::ffi::c_int,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        array_d_IPIV: *mut *mut ::core::ffi::c_int,
        computeType: cudaDataType,
        lwork: *mut i64,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverMgGetrf_bufferSize)(
            handle,
            M,
            N,
            array_d_A,
            IA,
            JA,
            descrA,
            array_d_IPIV,
            computeType,
            lwork,
        )
    }
    pub unsafe fn cusolverMgGetrs(
        handle: cusolverMgHandle_t,
        TRANS: cublasOperation_t,
        N: ::core::ffi::c_int,
        NRHS: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        array_d_IPIV: *mut *mut ::core::ffi::c_int,
        array_d_B: *mut *mut ::core::ffi::c_void,
        IB: ::core::ffi::c_int,
        JB: ::core::ffi::c_int,
        descrB: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        array_d_work: *mut *mut ::core::ffi::c_void,
        lwork: i64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverMgGetrs)(
            handle,
            TRANS,
            N,
            NRHS,
            array_d_A,
            IA,
            JA,
            descrA,
            array_d_IPIV,
            array_d_B,
            IB,
            JB,
            descrB,
            computeType,
            array_d_work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverMgGetrs_bufferSize(
        handle: cusolverMgHandle_t,
        TRANS: cublasOperation_t,
        N: ::core::ffi::c_int,
        NRHS: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        array_d_IPIV: *mut *mut ::core::ffi::c_int,
        array_d_B: *mut *mut ::core::ffi::c_void,
        IB: ::core::ffi::c_int,
        JB: ::core::ffi::c_int,
        descrB: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        lwork: *mut i64,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverMgGetrs_bufferSize)(
            handle,
            TRANS,
            N,
            NRHS,
            array_d_A,
            IA,
            JA,
            descrA,
            array_d_IPIV,
            array_d_B,
            IB,
            JB,
            descrB,
            computeType,
            lwork,
        )
    }
    pub unsafe fn cusolverMgPotrf(
        handle: cusolverMgHandle_t,
        uplo: cublasFillMode_t,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        array_d_work: *mut *mut ::core::ffi::c_void,
        lwork: i64,
        h_info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverMgPotrf)(
            handle,
            uplo,
            N,
            array_d_A,
            IA,
            JA,
            descrA,
            computeType,
            array_d_work,
            lwork,
            h_info,
        )
    }
    pub unsafe fn cusolverMgPotrf_bufferSize(
        handle: cusolverMgHandle_t,
        uplo: cublasFillMode_t,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        lwork: *mut i64,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverMgPotrf_bufferSize)(
            handle,
            uplo,
            N,
            array_d_A,
            IA,
            JA,
            descrA,
            computeType,
            lwork,
        )
    }
    pub unsafe fn cusolverMgPotri(
        handle: cusolverMgHandle_t,
        uplo: cublasFillMode_t,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        array_d_work: *mut *mut ::core::ffi::c_void,
        lwork: i64,
        h_info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverMgPotri)(
            handle,
            uplo,
            N,
            array_d_A,
            IA,
            JA,
            descrA,
            computeType,
            array_d_work,
            lwork,
            h_info,
        )
    }
    pub unsafe fn cusolverMgPotri_bufferSize(
        handle: cusolverMgHandle_t,
        uplo: cublasFillMode_t,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        lwork: *mut i64,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverMgPotri_bufferSize)(
            handle,
            uplo,
            N,
            array_d_A,
            IA,
            JA,
            descrA,
            computeType,
            lwork,
        )
    }
    pub unsafe fn cusolverMgPotrs(
        handle: cusolverMgHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        array_d_B: *mut *mut ::core::ffi::c_void,
        IB: ::core::ffi::c_int,
        JB: ::core::ffi::c_int,
        descrB: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        array_d_work: *mut *mut ::core::ffi::c_void,
        lwork: i64,
        h_info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverMgPotrs)(
            handle,
            uplo,
            n,
            nrhs,
            array_d_A,
            IA,
            JA,
            descrA,
            array_d_B,
            IB,
            JB,
            descrB,
            computeType,
            array_d_work,
            lwork,
            h_info,
        )
    }
    pub unsafe fn cusolverMgPotrs_bufferSize(
        handle: cusolverMgHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        array_d_B: *mut *mut ::core::ffi::c_void,
        IB: ::core::ffi::c_int,
        JB: ::core::ffi::c_int,
        descrB: cudaLibMgMatrixDesc_t,
        computeType: cudaDataType,
        lwork: *mut i64,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverMgPotrs_bufferSize)(
            handle,
            uplo,
            n,
            nrhs,
            array_d_A,
            IA,
            JA,
            descrA,
            array_d_B,
            IB,
            JB,
            descrB,
            computeType,
            lwork,
        )
    }
    pub unsafe fn cusolverMgSyevd(
        handle: cusolverMgHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        W: *mut ::core::ffi::c_void,
        dataTypeW: cudaDataType,
        computeType: cudaDataType,
        array_d_work: *mut *mut ::core::ffi::c_void,
        lwork: i64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverMgSyevd)(
            handle,
            jobz,
            uplo,
            N,
            array_d_A,
            IA,
            JA,
            descrA,
            W,
            dataTypeW,
            computeType,
            array_d_work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverMgSyevd_bufferSize(
        handle: cusolverMgHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        N: ::core::ffi::c_int,
        array_d_A: *mut *mut ::core::ffi::c_void,
        IA: ::core::ffi::c_int,
        JA: ::core::ffi::c_int,
        descrA: cudaLibMgMatrixDesc_t,
        W: *mut ::core::ffi::c_void,
        dataTypeW: cudaDataType,
        computeType: cudaDataType,
        lwork: *mut i64,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverMgSyevd_bufferSize)(
            handle,
            jobz,
            uplo,
            N,
            array_d_A,
            IA,
            JA,
            descrA,
            W,
            dataTypeW,
            computeType,
            lwork,
        )
    }
    pub struct Lib {
        __library: ::libloading::Library,
        pub cusolverMgCreate: unsafe extern "C" fn(
            handle: *mut cusolverMgHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverMgCreateDeviceGrid: unsafe extern "C" fn(
            grid: *mut cudaLibMgGrid_t,
            numRowDevices: i32,
            numColDevices: i32,
            deviceId: *const i32,
            mapping: cusolverMgGridMapping_t,
        ) -> cusolverStatus_t,
        pub cusolverMgCreateMatrixDesc: unsafe extern "C" fn(
            desc: *mut cudaLibMgMatrixDesc_t,
            numRows: i64,
            numCols: i64,
            rowBlockSize: i64,
            colBlockSize: i64,
            dataType: cudaDataType,
            grid: cudaLibMgGrid_t,
        ) -> cusolverStatus_t,
        pub cusolverMgDestroy: unsafe extern "C" fn(
            handle: cusolverMgHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverMgDestroyGrid: unsafe extern "C" fn(
            grid: cudaLibMgGrid_t,
        ) -> cusolverStatus_t,
        pub cusolverMgDestroyMatrixDesc: unsafe extern "C" fn(
            desc: cudaLibMgMatrixDesc_t,
        ) -> cusolverStatus_t,
        pub cusolverMgDeviceSelect: unsafe extern "C" fn(
            handle: cusolverMgHandle_t,
            nbDevices: ::core::ffi::c_int,
            deviceId: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverMgGetrf: unsafe extern "C" fn(
            handle: cusolverMgHandle_t,
            M: ::core::ffi::c_int,
            N: ::core::ffi::c_int,
            array_d_A: *mut *mut ::core::ffi::c_void,
            IA: ::core::ffi::c_int,
            JA: ::core::ffi::c_int,
            descrA: cudaLibMgMatrixDesc_t,
            array_d_IPIV: *mut *mut ::core::ffi::c_int,
            computeType: cudaDataType,
            array_d_work: *mut *mut ::core::ffi::c_void,
            lwork: i64,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverMgGetrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverMgHandle_t,
            M: ::core::ffi::c_int,
            N: ::core::ffi::c_int,
            array_d_A: *mut *mut ::core::ffi::c_void,
            IA: ::core::ffi::c_int,
            JA: ::core::ffi::c_int,
            descrA: cudaLibMgMatrixDesc_t,
            array_d_IPIV: *mut *mut ::core::ffi::c_int,
            computeType: cudaDataType,
            lwork: *mut i64,
        ) -> cusolverStatus_t,
        pub cusolverMgGetrs: unsafe extern "C" fn(
            handle: cusolverMgHandle_t,
            TRANS: cublasOperation_t,
            N: ::core::ffi::c_int,
            NRHS: ::core::ffi::c_int,
            array_d_A: *mut *mut ::core::ffi::c_void,
            IA: ::core::ffi::c_int,
            JA: ::core::ffi::c_int,
            descrA: cudaLibMgMatrixDesc_t,
            array_d_IPIV: *mut *mut ::core::ffi::c_int,
            array_d_B: *mut *mut ::core::ffi::c_void,
            IB: ::core::ffi::c_int,
            JB: ::core::ffi::c_int,
            descrB: cudaLibMgMatrixDesc_t,
            computeType: cudaDataType,
            array_d_work: *mut *mut ::core::ffi::c_void,
            lwork: i64,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverMgGetrs_bufferSize: unsafe extern "C" fn(
            handle: cusolverMgHandle_t,
            TRANS: cublasOperation_t,
            N: ::core::ffi::c_int,
            NRHS: ::core::ffi::c_int,
            array_d_A: *mut *mut ::core::ffi::c_void,
            IA: ::core::ffi::c_int,
            JA: ::core::ffi::c_int,
            descrA: cudaLibMgMatrixDesc_t,
            array_d_IPIV: *mut *mut ::core::ffi::c_int,
            array_d_B: *mut *mut ::core::ffi::c_void,
            IB: ::core::ffi::c_int,
            JB: ::core::ffi::c_int,
            descrB: cudaLibMgMatrixDesc_t,
            computeType: cudaDataType,
            lwork: *mut i64,
        ) -> cusolverStatus_t,
        pub cusolverMgPotrf: unsafe extern "C" fn(
            handle: cusolverMgHandle_t,
            uplo: cublasFillMode_t,
            N: ::core::ffi::c_int,
            array_d_A: *mut *mut ::core::ffi::c_void,
            IA: ::core::ffi::c_int,
            JA: ::core::ffi::c_int,
            descrA: cudaLibMgMatrixDesc_t,
            computeType: cudaDataType,
            array_d_work: *mut *mut ::core::ffi::c_void,
            lwork: i64,
            h_info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverMgPotrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverMgHandle_t,
            uplo: cublasFillMode_t,
            N: ::core::ffi::c_int,
            array_d_A: *mut *mut ::core::ffi::c_void,
            IA: ::core::ffi::c_int,
            JA: ::core::ffi::c_int,
            descrA: cudaLibMgMatrixDesc_t,
            computeType: cudaDataType,
            lwork: *mut i64,
        ) -> cusolverStatus_t,
        pub cusolverMgPotri: unsafe extern "C" fn(
            handle: cusolverMgHandle_t,
            uplo: cublasFillMode_t,
            N: ::core::ffi::c_int,
            array_d_A: *mut *mut ::core::ffi::c_void,
            IA: ::core::ffi::c_int,
            JA: ::core::ffi::c_int,
            descrA: cudaLibMgMatrixDesc_t,
            computeType: cudaDataType,
            array_d_work: *mut *mut ::core::ffi::c_void,
            lwork: i64,
            h_info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverMgPotri_bufferSize: unsafe extern "C" fn(
            handle: cusolverMgHandle_t,
            uplo: cublasFillMode_t,
            N: ::core::ffi::c_int,
            array_d_A: *mut *mut ::core::ffi::c_void,
            IA: ::core::ffi::c_int,
            JA: ::core::ffi::c_int,
            descrA: cudaLibMgMatrixDesc_t,
            computeType: cudaDataType,
            lwork: *mut i64,
        ) -> cusolverStatus_t,
        pub cusolverMgPotrs: unsafe extern "C" fn(
            handle: cusolverMgHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            array_d_A: *mut *mut ::core::ffi::c_void,
            IA: ::core::ffi::c_int,
            JA: ::core::ffi::c_int,
            descrA: cudaLibMgMatrixDesc_t,
            array_d_B: *mut *mut ::core::ffi::c_void,
            IB: ::core::ffi::c_int,
            JB: ::core::ffi::c_int,
            descrB: cudaLibMgMatrixDesc_t,
            computeType: cudaDataType,
            array_d_work: *mut *mut ::core::ffi::c_void,
            lwork: i64,
            h_info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverMgPotrs_bufferSize: unsafe extern "C" fn(
            handle: cusolverMgHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            array_d_A: *mut *mut ::core::ffi::c_void,
            IA: ::core::ffi::c_int,
            JA: ::core::ffi::c_int,
            descrA: cudaLibMgMatrixDesc_t,
            array_d_B: *mut *mut ::core::ffi::c_void,
            IB: ::core::ffi::c_int,
            JB: ::core::ffi::c_int,
            descrB: cudaLibMgMatrixDesc_t,
            computeType: cudaDataType,
            lwork: *mut i64,
        ) -> cusolverStatus_t,
        pub cusolverMgSyevd: unsafe extern "C" fn(
            handle: cusolverMgHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            N: ::core::ffi::c_int,
            array_d_A: *mut *mut ::core::ffi::c_void,
            IA: ::core::ffi::c_int,
            JA: ::core::ffi::c_int,
            descrA: cudaLibMgMatrixDesc_t,
            W: *mut ::core::ffi::c_void,
            dataTypeW: cudaDataType,
            computeType: cudaDataType,
            array_d_work: *mut *mut ::core::ffi::c_void,
            lwork: i64,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverMgSyevd_bufferSize: unsafe extern "C" fn(
            handle: cusolverMgHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            N: ::core::ffi::c_int,
            array_d_A: *mut *mut ::core::ffi::c_void,
            IA: ::core::ffi::c_int,
            JA: ::core::ffi::c_int,
            descrA: cudaLibMgMatrixDesc_t,
            W: *mut ::core::ffi::c_void,
            dataTypeW: cudaDataType,
            computeType: cudaDataType,
            lwork: *mut i64,
        ) -> cusolverStatus_t,
    }
    impl Lib {
        pub unsafe fn new<P>(path: P) -> Result<Self, ::libloading::Error>
        where
            P: AsRef<::std::ffi::OsStr>,
        {
            let library = ::libloading::Library::new(path)?;
            Self::from_library(library)
        }
        pub unsafe fn from_library<L>(library: L) -> Result<Self, ::libloading::Error>
        where
            L: Into<::libloading::Library>,
        {
            let __library = library.into();
            let cusolverMgCreate = __library
                .get(b"cusolverMgCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgCreateDeviceGrid = __library
                .get(b"cusolverMgCreateDeviceGrid\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgCreateMatrixDesc = __library
                .get(b"cusolverMgCreateMatrixDesc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgDestroy = __library
                .get(b"cusolverMgDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgDestroyGrid = __library
                .get(b"cusolverMgDestroyGrid\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgDestroyMatrixDesc = __library
                .get(b"cusolverMgDestroyMatrixDesc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgDeviceSelect = __library
                .get(b"cusolverMgDeviceSelect\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgGetrf = __library
                .get(b"cusolverMgGetrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgGetrf_bufferSize = __library
                .get(b"cusolverMgGetrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgGetrs = __library
                .get(b"cusolverMgGetrs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgGetrs_bufferSize = __library
                .get(b"cusolverMgGetrs_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgPotrf = __library
                .get(b"cusolverMgPotrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgPotrf_bufferSize = __library
                .get(b"cusolverMgPotrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgPotri = __library
                .get(b"cusolverMgPotri\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgPotri_bufferSize = __library
                .get(b"cusolverMgPotri_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgPotrs = __library
                .get(b"cusolverMgPotrs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgPotrs_bufferSize = __library
                .get(b"cusolverMgPotrs_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgSyevd = __library
                .get(b"cusolverMgSyevd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverMgSyevd_bufferSize = __library
                .get(b"cusolverMgSyevd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            Ok(Self {
                __library,
                cusolverMgCreate,
                cusolverMgCreateDeviceGrid,
                cusolverMgCreateMatrixDesc,
                cusolverMgDestroy,
                cusolverMgDestroyGrid,
                cusolverMgDestroyMatrixDesc,
                cusolverMgDeviceSelect,
                cusolverMgGetrf,
                cusolverMgGetrf_bufferSize,
                cusolverMgGetrs,
                cusolverMgGetrs_bufferSize,
                cusolverMgPotrf,
                cusolverMgPotrf_bufferSize,
                cusolverMgPotri,
                cusolverMgPotri_bufferSize,
                cusolverMgPotrs,
                cusolverMgPotrs_bufferSize,
                cusolverMgSyevd,
                cusolverMgSyevd_bufferSize,
            })
        }
    }
    pub unsafe fn culib() -> &'static Lib {
        static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
        LIB.get_or_init(|| {
            let lib_names = std::vec!["cusolverMg"];
            let choices: std::vec::Vec<_> = lib_names
                .iter()
                .map(|l| crate::get_lib_name_candidates(l))
                .flatten()
                .collect();
            for choice in choices.iter() {
                if let Ok(lib) = Lib::new(choice) {
                    return lib;
                }
            }
            crate::panic_no_lib_found(lib_names[0], &choices);
        })
    }
}
#[cfg(feature = "dynamic-loading")]
pub use loaded::*;
