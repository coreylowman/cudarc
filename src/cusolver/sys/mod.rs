#![cfg_attr(feature = "no-std", no_std)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;
pub use self::cudaDataType_t as cudaDataType;
pub use self::libraryPropertyType_t as libraryPropertyType;
#[cfg(
    any(
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type FILE = _IO_FILE;
#[cfg(
    any(
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type _IO_lock_t = ::core::ffi::c_void;
#[cfg(
    any(
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type __off64_t = ::core::ffi::c_long;
#[cfg(
    any(
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type __off_t = ::core::ffi::c_long;
pub type csrqrInfo_t = *mut csrqrInfo;
pub type cuComplex = cuFloatComplex;
pub type cuDoubleComplex = double2;
pub type cuFloatComplex = float2;
pub type cudaLibMgGrid_t = *mut ::core::ffi::c_void;
pub type cudaLibMgMatrixDesc_t = *mut ::core::ffi::c_void;
pub type cudaStream_t = *mut CUstream_st;
pub type cusolverDnHandle_t = *mut cusolverDnContext;
pub type cusolverDnIRSInfos_t = *mut cusolverDnIRSInfos;
pub type cusolverDnIRSParams_t = *mut cusolverDnIRSParams;
#[cfg(
    any(
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type cusolverDnLoggerCallback_t = ::core::option::Option<
    unsafe extern "C" fn(
        logLevel: ::core::ffi::c_int,
        functionName: *const ::core::ffi::c_char,
        message: *const ::core::ffi::c_char,
    ),
>;
pub type cusolverDnParams_t = *mut cusolverDnParams;
pub type cusolverMgHandle_t = *mut cusolverMgContext;
pub type cusolverRfHandle_t = *mut cusolverRfCommon;
pub type cusolverSpHandle_t = *mut cusolverSpContext;
pub type cusolver_int_t = ::core::ffi::c_int;
pub type cusparseMatDescr_t = *mut cusparseMatDescr;
pub type gesvdjInfo_t = *mut gesvdjInfo;
pub type syevjInfo_t = *mut syevjInfo;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasDiagType_t {
    CUBLAS_DIAG_NON_UNIT = 0,
    CUBLAS_DIAG_UNIT = 1,
}
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
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cublasSideMode_t {
    CUBLAS_SIDE_LEFT = 0,
    CUBLAS_SIDE_RIGHT = 1,
}
#[cfg(any(feature = "cuda-11050", feature = "cuda-11060", feature = "cuda-11070"))]
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
#[cfg(any(feature = "cuda-12080"))]
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
pub enum cusolverAlgMode_t {
    CUSOLVER_ALG_0 = 0,
    CUSOLVER_ALG_1 = 1,
    CUSOLVER_ALG_2 = 2,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverDeterministicMode_t {
    CUSOLVER_DETERMINISTIC_RESULTS = 1,
    CUSOLVER_ALLOW_NON_DETERMINISTIC_RESULTS = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverDirectMode_t {
    CUBLAS_DIRECT_FORWARD = 0,
    CUBLAS_DIRECT_BACKWARD = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverDnFunction_t {
    CUSOLVERDN_GETRF = 0,
    CUSOLVERDN_POTRF = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverEigMode_t {
    CUSOLVER_EIG_MODE_NOVECTOR = 0,
    CUSOLVER_EIG_MODE_VECTOR = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverEigRange_t {
    CUSOLVER_EIG_RANGE_ALL = 1001,
    CUSOLVER_EIG_RANGE_I = 1002,
    CUSOLVER_EIG_RANGE_V = 1003,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverEigType_t {
    CUSOLVER_EIG_TYPE_1 = 1,
    CUSOLVER_EIG_TYPE_2 = 2,
    CUSOLVER_EIG_TYPE_3 = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverIRSRefinement_t {
    CUSOLVER_IRS_REFINE_NOT_SET = 1100,
    CUSOLVER_IRS_REFINE_NONE = 1101,
    CUSOLVER_IRS_REFINE_CLASSICAL = 1102,
    CUSOLVER_IRS_REFINE_CLASSICAL_GMRES = 1103,
    CUSOLVER_IRS_REFINE_GMRES = 1104,
    CUSOLVER_IRS_REFINE_GMRES_GMRES = 1105,
    CUSOLVER_IRS_REFINE_GMRES_NOPCOND = 1106,
    CUSOLVER_PREC_DD = 1150,
    CUSOLVER_PREC_SS = 1151,
    CUSOLVER_PREC_SHT = 1152,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverMgGridMapping_t {
    CUDALIBMG_GRID_MAPPING_ROW_MAJOR = 1,
    CUDALIBMG_GRID_MAPPING_COL_MAJOR = 0,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverNorm_t {
    CUSOLVER_INF_NORM = 104,
    CUSOLVER_MAX_NORM = 105,
    CUSOLVER_ONE_NORM = 106,
    CUSOLVER_FRO_NORM = 107,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverPrecType_t {
    CUSOLVER_R_8I = 1201,
    CUSOLVER_R_8U = 1202,
    CUSOLVER_R_64F = 1203,
    CUSOLVER_R_32F = 1204,
    CUSOLVER_R_16F = 1205,
    CUSOLVER_R_16BF = 1206,
    CUSOLVER_R_TF32 = 1207,
    CUSOLVER_R_AP = 1208,
    CUSOLVER_C_8I = 1211,
    CUSOLVER_C_8U = 1212,
    CUSOLVER_C_64F = 1213,
    CUSOLVER_C_32F = 1214,
    CUSOLVER_C_16F = 1215,
    CUSOLVER_C_16BF = 1216,
    CUSOLVER_C_TF32 = 1217,
    CUSOLVER_C_AP = 1218,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverRfFactorization_t {
    CUSOLVERRF_FACTORIZATION_ALG0 = 0,
    CUSOLVERRF_FACTORIZATION_ALG1 = 1,
    CUSOLVERRF_FACTORIZATION_ALG2 = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverRfMatrixFormat_t {
    CUSOLVERRF_MATRIX_FORMAT_CSR = 0,
    CUSOLVERRF_MATRIX_FORMAT_CSC = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverRfNumericBoostReport_t {
    CUSOLVERRF_NUMERIC_BOOST_NOT_USED = 0,
    CUSOLVERRF_NUMERIC_BOOST_USED = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverRfResetValuesFastMode_t {
    CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF = 0,
    CUSOLVERRF_RESET_VALUES_FAST_MODE_ON = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverRfTriangularSolve_t {
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG1 = 1,
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG2 = 2,
    CUSOLVERRF_TRIANGULAR_SOLVE_ALG3 = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverRfUnitDiagonal_t {
    CUSOLVERRF_UNIT_DIAGONAL_STORED_L = 0,
    CUSOLVERRF_UNIT_DIAGONAL_STORED_U = 1,
    CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L = 2,
    CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U = 3,
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
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cusolverStorevMode_t {
    CUBLAS_STOREV_COLUMNWISE = 0,
    CUBLAS_STOREV_ROWWISE = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum libraryPropertyType_t {
    MAJOR_VERSION = 0,
    MINOR_VERSION = 1,
    PATCH_LEVEL = 2,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct _IO_FILE {
    pub _flags: ::core::ffi::c_int,
    pub _IO_read_ptr: *mut ::core::ffi::c_char,
    pub _IO_read_end: *mut ::core::ffi::c_char,
    pub _IO_read_base: *mut ::core::ffi::c_char,
    pub _IO_write_base: *mut ::core::ffi::c_char,
    pub _IO_write_ptr: *mut ::core::ffi::c_char,
    pub _IO_write_end: *mut ::core::ffi::c_char,
    pub _IO_buf_base: *mut ::core::ffi::c_char,
    pub _IO_buf_end: *mut ::core::ffi::c_char,
    pub _IO_save_base: *mut ::core::ffi::c_char,
    pub _IO_backup_base: *mut ::core::ffi::c_char,
    pub _IO_save_end: *mut ::core::ffi::c_char,
    pub _markers: *mut _IO_marker,
    pub _chain: *mut _IO_FILE,
    pub _fileno: ::core::ffi::c_int,
    pub _flags2: ::core::ffi::c_int,
    pub _old_offset: __off_t,
    pub _cur_column: ::core::ffi::c_ushort,
    pub _vtable_offset: ::core::ffi::c_schar,
    pub _shortbuf: [::core::ffi::c_char; 1usize],
    pub _lock: *mut _IO_lock_t,
    pub _offset: __off64_t,
    pub _codecvt: *mut _IO_codecvt,
    pub _wide_data: *mut _IO_wide_data,
    pub _freeres_list: *mut _IO_FILE,
    pub _freeres_buf: *mut ::core::ffi::c_void,
    pub __pad5: usize,
    pub _mode: ::core::ffi::c_int,
    pub _unused2: [::core::ffi::c_char; 20usize],
}
#[cfg(
    any(
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_codecvt {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_marker {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _IO_wide_data {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct csrqrInfo {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusolverDnContext {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusolverDnIRSInfos {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusolverDnIRSParams {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusolverDnParams {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusolverMgContext {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusolverRfCommon {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusolverSpContext {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cusparseMatDescr {
    _unused: [u8; 0],
}
#[repr(C)]
#[repr(align(16))]
#[derive(Debug, Default, Copy, Clone, PartialOrd, PartialEq)]
pub struct double2 {
    pub x: f64,
    pub y: f64,
}
#[repr(C)]
#[repr(align(8))]
#[derive(Debug, Default, Copy, Clone, PartialOrd, PartialEq)]
pub struct float2 {
    pub x: f32,
    pub y: f32,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct gesvdjInfo {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct syevjInfo {
    _unused: [u8; 0],
}
impl cublasOperation_t {
    pub const CUBLAS_OP_HERMITAN: cublasOperation_t = cublasOperation_t::CUBLAS_OP_C;
}
#[cfg(any(feature = "cuda-12080"))]
impl cudaDataType_t {
    pub const CUDA_R_8F_UE4M3: cudaDataType_t = cudaDataType_t::CUDA_R_8F_E4M3;
}
#[cfg(
    any(
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for _IO_FILE {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(not(feature = "dynamic-loading"))]
extern "C" {
    pub fn cusolverDnCCgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCCgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCCgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCCgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCEgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCEgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCEgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCEgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCKgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCKgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCKgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCKgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCYgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCYgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCYgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCYgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCgebrd(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        D: *mut f32,
        E: *mut f32,
        TAUQ: *mut cuComplex,
        TAUP: *mut cuComplex,
        Work: *mut cuComplex,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCgebrd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCgeqrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        TAU: *mut cuComplex,
        Workspace: *mut cuComplex,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCgeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCgesvd(
        handle: cusolverDnHandle_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        S: *mut f32,
        U: *mut cuComplex,
        ldu: ::core::ffi::c_int,
        VT: *mut cuComplex,
        ldvt: ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        rwork: *mut f32,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCgesvd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCgesvdaStridedBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const cuComplex,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *mut f32,
        strideS: ::core::ffi::c_longlong,
        d_U: *mut cuComplex,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *mut cuComplex,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        d_work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        h_R_nrmF: *mut f64,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCgesvdaStridedBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const cuComplex,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *const f32,
        strideS: ::core::ffi::c_longlong,
        d_U: *const cuComplex,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *const cuComplex,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        lwork: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCgesvdj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        S: *mut f32,
        U: *mut cuComplex,
        ldu: ::core::ffi::c_int,
        V: *mut cuComplex,
        ldv: ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCgesvdjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        S: *mut f32,
        U: *mut cuComplex,
        ldu: ::core::ffi::c_int,
        V: *mut cuComplex,
        ldv: ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCgesvdjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        S: *const f32,
        U: *const cuComplex,
        ldu: ::core::ffi::c_int,
        V: *const cuComplex,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCgesvdj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        S: *const f32,
        U: *const cuComplex,
        ldu: ::core::ffi::c_int,
        V: *const cuComplex,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCgetrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        Workspace: *mut cuComplex,
        devIpiv: *mut ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCgetrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCgetrs(
        handle: cusolverDnHandle_t,
        trans: cublasOperation_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        B: *mut cuComplex,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCheevd(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCheevd_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCheevdx(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f32,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCheevdx_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCheevj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCheevjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCheevjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCheevj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnChegvd(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuComplex,
        ldb: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnChegvd_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        B: *const cuComplex,
        ldb: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnChegvdx(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuComplex,
        ldb: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f32,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnChegvdx_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        B: *const cuComplex,
        ldb: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnChegvj(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuComplex,
        ldb: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnChegvj_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        B: *const cuComplex,
        ldb: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnChetrd(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        d: *mut f32,
        e: *mut f32,
        tau: *mut cuComplex,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnChetrd_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        d: *const f32,
        e: *const f32,
        tau: *const cuComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnClaswp(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        k1: ::core::ffi::c_int,
        k2: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        incx: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnClauum(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnClauum_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCpotrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        Workspace: *mut cuComplex,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCpotrfBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        Aarray: *mut *mut cuComplex,
        lda: ::core::ffi::c_int,
        infoArray: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCpotrf_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCpotri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCpotri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCpotrs(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuComplex,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCpotrsBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *mut *mut cuComplex,
        lda: ::core::ffi::c_int,
        B: *mut *mut cuComplex,
        ldb: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCreate(handle: *mut cusolverDnHandle_t) -> cusolverStatus_t;
    pub fn cusolverDnCreateGesvdjInfo(info: *mut gesvdjInfo_t) -> cusolverStatus_t;
    pub fn cusolverDnCreateParams(params: *mut cusolverDnParams_t) -> cusolverStatus_t;
    pub fn cusolverDnCreateSyevjInfo(info: *mut syevjInfo_t) -> cusolverStatus_t;
    pub fn cusolverDnCsytrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        ipiv: *mut ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCsytrf_bufferSize(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCsytri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCsytri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCungbr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCungbr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCungqr(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCungqr_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCungtr(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCungtr_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCunmqr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        C: *mut cuComplex,
        ldc: ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCunmqr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        C: *const cuComplex,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCunmtr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        tau: *mut cuComplex,
        C: *mut cuComplex,
        ldc: ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnCunmtr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        C: *const cuComplex,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDBgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDBgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDBgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDBgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDDgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDDgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDDgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDDgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDHgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDHgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDHgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDHgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDSgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDSgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDSgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDSgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDXgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDXgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDXgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDXgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDestroy(handle: cusolverDnHandle_t) -> cusolverStatus_t;
    pub fn cusolverDnDestroyGesvdjInfo(info: gesvdjInfo_t) -> cusolverStatus_t;
    pub fn cusolverDnDestroyParams(params: cusolverDnParams_t) -> cusolverStatus_t;
    pub fn cusolverDnDestroySyevjInfo(info: syevjInfo_t) -> cusolverStatus_t;
    pub fn cusolverDnDgebrd(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        D: *mut f64,
        E: *mut f64,
        TAUQ: *mut f64,
        TAUP: *mut f64,
        Work: *mut f64,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDgebrd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDgeqrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        TAU: *mut f64,
        Workspace: *mut f64,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDgeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDgesvd(
        handle: cusolverDnHandle_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        S: *mut f64,
        U: *mut f64,
        ldu: ::core::ffi::c_int,
        VT: *mut f64,
        ldvt: ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        rwork: *mut f64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDgesvd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDgesvdaStridedBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const f64,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *mut f64,
        strideS: ::core::ffi::c_longlong,
        d_U: *mut f64,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *mut f64,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        d_work: *mut f64,
        lwork: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        h_R_nrmF: *mut f64,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDgesvdaStridedBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const f64,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *const f64,
        strideS: ::core::ffi::c_longlong,
        d_U: *const f64,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *const f64,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        lwork: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDgesvdj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        S: *mut f64,
        U: *mut f64,
        ldu: ::core::ffi::c_int,
        V: *mut f64,
        ldv: ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDgesvdjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        S: *mut f64,
        U: *mut f64,
        ldu: ::core::ffi::c_int,
        V: *mut f64,
        ldv: ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDgesvdjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        S: *const f64,
        U: *const f64,
        ldu: ::core::ffi::c_int,
        V: *const f64,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDgesvdj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        S: *const f64,
        U: *const f64,
        ldu: ::core::ffi::c_int,
        V: *const f64,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDgetrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        Workspace: *mut f64,
        devIpiv: *mut ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDgetrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDgetrs(
        handle: cusolverDnHandle_t,
        trans: cublasOperation_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        B: *mut f64,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDlaswp(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        k1: ::core::ffi::c_int,
        k2: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        incx: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDlauum(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDlauum_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDorgbr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDorgbr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDorgqr(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDorgqr_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDorgtr(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDorgtr_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDormqr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        C: *mut f64,
        ldc: ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDormqr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        C: *const f64,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDormtr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        tau: *mut f64,
        C: *mut f64,
        ldc: ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDormtr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        C: *const f64,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDpotrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        Workspace: *mut f64,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDpotrfBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        Aarray: *mut *mut f64,
        lda: ::core::ffi::c_int,
        infoArray: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDpotrf_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDpotri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDpotri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDpotrs(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        B: *mut f64,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDpotrsBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *mut *mut f64,
        lda: ::core::ffi::c_int,
        B: *mut *mut f64,
        ldb: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsyevd(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsyevd_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsyevdx(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsyevdx_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsyevj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsyevjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsyevjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsyevj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsygvd(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        B: *mut f64,
        ldb: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsygvd_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        B: *const f64,
        ldb: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsygvdx(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        B: *mut f64,
        ldb: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsygvdx_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        B: *const f64,
        ldb: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsygvj(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        B: *mut f64,
        ldb: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsygvj_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        B: *const f64,
        ldb: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsytrd(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        d: *mut f64,
        e: *mut f64,
        tau: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsytrd_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        d: *const f64,
        e: *const f64,
        tau: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsytrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        ipiv: *mut ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsytrf_bufferSize(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsytri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnDsytri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnGeqrf(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeTau: cudaDataType,
        tau: *mut ::core::ffi::c_void,
        computeType: cudaDataType,
        pBuffer: *mut ::core::ffi::c_void,
        workspaceInBytes: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnGeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeTau: cudaDataType,
        tau: *const ::core::ffi::c_void,
        computeType: cudaDataType,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnGesvd(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeS: cudaDataType,
        S: *mut ::core::ffi::c_void,
        dataTypeU: cudaDataType,
        U: *mut ::core::ffi::c_void,
        ldu: i64,
        dataTypeVT: cudaDataType,
        VT: *mut ::core::ffi::c_void,
        ldvt: i64,
        computeType: cudaDataType,
        pBuffer: *mut ::core::ffi::c_void,
        workspaceInBytes: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnGesvd_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeS: cudaDataType,
        S: *const ::core::ffi::c_void,
        dataTypeU: cudaDataType,
        U: *const ::core::ffi::c_void,
        ldu: i64,
        dataTypeVT: cudaDataType,
        VT: *const ::core::ffi::c_void,
        ldvt: i64,
        computeType: cudaDataType,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t;
    #[cfg(
        any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cusolverDnGetDeterministicMode(
        handle: cusolverDnHandle_t,
        mode: *mut cusolverDeterministicMode_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnGetStream(
        handle: cusolverDnHandle_t,
        streamId: *mut cudaStream_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnGetrf(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        ipiv: *mut i64,
        computeType: cudaDataType,
        pBuffer: *mut ::core::ffi::c_void,
        workspaceInBytes: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnGetrf_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        computeType: cudaDataType,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnGetrs(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        trans: cublasOperation_t,
        n: i64,
        nrhs: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        ipiv: *const i64,
        dataTypeB: cudaDataType,
        B: *mut ::core::ffi::c_void,
        ldb: i64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSInfosCreate(
        infos_ptr: *mut cusolverDnIRSInfos_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSInfosDestroy(infos: cusolverDnIRSInfos_t) -> cusolverStatus_t;
    pub fn cusolverDnIRSInfosGetMaxIters(
        infos: cusolverDnIRSInfos_t,
        maxiters: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSInfosGetNiters(
        infos: cusolverDnIRSInfos_t,
        niters: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSInfosGetOuterNiters(
        infos: cusolverDnIRSInfos_t,
        outer_niters: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSInfosGetResidualHistory(
        infos: cusolverDnIRSInfos_t,
        residual_history: *mut *mut ::core::ffi::c_void,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSInfosRequestResidual(
        infos: cusolverDnIRSInfos_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSParamsCreate(
        params_ptr: *mut cusolverDnIRSParams_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSParamsDestroy(params: cusolverDnIRSParams_t) -> cusolverStatus_t;
    pub fn cusolverDnIRSParamsDisableFallback(
        params: cusolverDnIRSParams_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSParamsEnableFallback(
        params: cusolverDnIRSParams_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSParamsGetMaxIters(
        params: cusolverDnIRSParams_t,
        maxiters: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSParamsSetMaxIters(
        params: cusolverDnIRSParams_t,
        maxiters: cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSParamsSetMaxItersInner(
        params: cusolverDnIRSParams_t,
        maxiters_inner: cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSParamsSetRefinementSolver(
        params: cusolverDnIRSParams_t,
        refinement_solver: cusolverIRSRefinement_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSParamsSetSolverLowestPrecision(
        params: cusolverDnIRSParams_t,
        solver_lowest_precision: cusolverPrecType_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSParamsSetSolverMainPrecision(
        params: cusolverDnIRSParams_t,
        solver_main_precision: cusolverPrecType_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSParamsSetSolverPrecisions(
        params: cusolverDnIRSParams_t,
        solver_main_precision: cusolverPrecType_t,
        solver_lowest_precision: cusolverPrecType_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSParamsSetTol(
        params: cusolverDnIRSParams_t,
        val: f64,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSParamsSetTolInner(
        params: cusolverDnIRSParams_t,
        val: f64,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSXgels(
        handle: cusolverDnHandle_t,
        gels_irs_params: cusolverDnIRSParams_t,
        gels_irs_infos: cusolverDnIRSInfos_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut ::core::ffi::c_void,
        ldda: cusolver_int_t,
        dB: *mut ::core::ffi::c_void,
        lddb: cusolver_int_t,
        dX: *mut ::core::ffi::c_void,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        niters: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSXgels_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnIRSParams_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSXgesv(
        handle: cusolverDnHandle_t,
        gesv_irs_params: cusolverDnIRSParams_t,
        gesv_irs_infos: cusolverDnIRSInfos_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut ::core::ffi::c_void,
        ldda: cusolver_int_t,
        dB: *mut ::core::ffi::c_void,
        lddb: cusolver_int_t,
        dX: *mut ::core::ffi::c_void,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        niters: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnIRSXgesv_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnIRSParams_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    #[cfg(
        any(
            feature = "cuda-11070",
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cusolverDnLoggerForceDisable() -> cusolverStatus_t;
    #[cfg(
        any(
            feature = "cuda-11070",
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cusolverDnLoggerOpenFile(
        logFile: *const ::core::ffi::c_char,
    ) -> cusolverStatus_t;
    #[cfg(
        any(
            feature = "cuda-11070",
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cusolverDnLoggerSetCallback(
        callback: cusolverDnLoggerCallback_t,
    ) -> cusolverStatus_t;
    #[cfg(
        any(
            feature = "cuda-11070",
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cusolverDnLoggerSetFile(file: *mut FILE) -> cusolverStatus_t;
    #[cfg(
        any(
            feature = "cuda-11070",
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cusolverDnLoggerSetLevel(level: ::core::ffi::c_int) -> cusolverStatus_t;
    #[cfg(
        any(
            feature = "cuda-11070",
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cusolverDnLoggerSetMask(mask: ::core::ffi::c_int) -> cusolverStatus_t;
    pub fn cusolverDnPotrf(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        computeType: cudaDataType,
        pBuffer: *mut ::core::ffi::c_void,
        workspaceInBytes: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnPotrf_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        computeType: cudaDataType,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnPotrs(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        uplo: cublasFillMode_t,
        n: i64,
        nrhs: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeB: cudaDataType,
        B: *mut ::core::ffi::c_void,
        ldb: i64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSBgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSBgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSBgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSBgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSHgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSHgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSHgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSHgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSSgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSSgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSSgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSSgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSXgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSXgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSXgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSXgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSetAdvOptions(
        params: cusolverDnParams_t,
        function: cusolverDnFunction_t,
        algo: cusolverAlgMode_t,
    ) -> cusolverStatus_t;
    #[cfg(
        any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cusolverDnSetDeterministicMode(
        handle: cusolverDnHandle_t,
        mode: cusolverDeterministicMode_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSetStream(
        handle: cusolverDnHandle_t,
        streamId: cudaStream_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSgebrd(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        D: *mut f32,
        E: *mut f32,
        TAUQ: *mut f32,
        TAUP: *mut f32,
        Work: *mut f32,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSgebrd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSgeqrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        TAU: *mut f32,
        Workspace: *mut f32,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSgeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSgesvd(
        handle: cusolverDnHandle_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        S: *mut f32,
        U: *mut f32,
        ldu: ::core::ffi::c_int,
        VT: *mut f32,
        ldvt: ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        rwork: *mut f32,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSgesvd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSgesvdaStridedBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const f32,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *mut f32,
        strideS: ::core::ffi::c_longlong,
        d_U: *mut f32,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *mut f32,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        d_work: *mut f32,
        lwork: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        h_R_nrmF: *mut f64,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSgesvdaStridedBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const f32,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *const f32,
        strideS: ::core::ffi::c_longlong,
        d_U: *const f32,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *const f32,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        lwork: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSgesvdj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        S: *mut f32,
        U: *mut f32,
        ldu: ::core::ffi::c_int,
        V: *mut f32,
        ldv: ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSgesvdjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        S: *mut f32,
        U: *mut f32,
        ldu: ::core::ffi::c_int,
        V: *mut f32,
        ldv: ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSgesvdjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        S: *const f32,
        U: *const f32,
        ldu: ::core::ffi::c_int,
        V: *const f32,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSgesvdj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        S: *const f32,
        U: *const f32,
        ldu: ::core::ffi::c_int,
        V: *const f32,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSgetrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        Workspace: *mut f32,
        devIpiv: *mut ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSgetrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSgetrs(
        handle: cusolverDnHandle_t,
        trans: cublasOperation_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        B: *mut f32,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSlaswp(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        k1: ::core::ffi::c_int,
        k2: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        incx: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSlauum(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSlauum_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSorgbr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSorgbr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSorgqr(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSorgqr_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSorgtr(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSorgtr_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSormqr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        C: *mut f32,
        ldc: ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSormqr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        C: *const f32,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSormtr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        tau: *mut f32,
        C: *mut f32,
        ldc: ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSormtr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        C: *const f32,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSpotrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        Workspace: *mut f32,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSpotrfBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        Aarray: *mut *mut f32,
        lda: ::core::ffi::c_int,
        infoArray: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSpotrf_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSpotri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSpotri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSpotrs(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        B: *mut f32,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSpotrsBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *mut *mut f32,
        lda: ::core::ffi::c_int,
        B: *mut *mut f32,
        ldb: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsyevd(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsyevd_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsyevdx(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsyevdx_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsyevj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsyevjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsyevjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsyevj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsygvd(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        B: *mut f32,
        ldb: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsygvd_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        B: *const f32,
        ldb: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsygvdx(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        B: *mut f32,
        ldb: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsygvdx_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        B: *const f32,
        ldb: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsygvj(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        B: *mut f32,
        ldb: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsygvj_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        B: *const f32,
        ldb: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsytrd(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        d: *mut f32,
        e: *mut f32,
        tau: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsytrd_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        d: *const f32,
        e: *const f32,
        tau: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsytrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        ipiv: *mut ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsytrf_bufferSize(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsytri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSsytri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSyevd(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *mut ::core::ffi::c_void,
        computeType: cudaDataType,
        pBuffer: *mut ::core::ffi::c_void,
        workspaceInBytes: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSyevd_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *const ::core::ffi::c_void,
        computeType: cudaDataType,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSyevdx(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        vl: *mut ::core::ffi::c_void,
        vu: *mut ::core::ffi::c_void,
        il: i64,
        iu: i64,
        meig64: *mut i64,
        dataTypeW: cudaDataType,
        W: *mut ::core::ffi::c_void,
        computeType: cudaDataType,
        pBuffer: *mut ::core::ffi::c_void,
        workspaceInBytes: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnSyevdx_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        vl: *mut ::core::ffi::c_void,
        vu: *mut ::core::ffi::c_void,
        il: i64,
        iu: i64,
        h_meig: *mut i64,
        dataTypeW: cudaDataType,
        W: *const ::core::ffi::c_void,
        computeType: cudaDataType,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t;
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
    pub fn cusolverDnXgeev(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobvl: cusolverEigMode_t,
        jobvr: cusolverEigMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *mut ::core::ffi::c_void,
        dataTypeVL: cudaDataType,
        VL: *mut ::core::ffi::c_void,
        ldvl: i64,
        dataTypeVR: cudaDataType,
        VR: *mut ::core::ffi::c_void,
        ldvr: i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
    pub fn cusolverDnXgeev_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobvl: cusolverEigMode_t,
        jobvr: cusolverEigMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *const ::core::ffi::c_void,
        dataTypeVL: cudaDataType,
        VL: *const ::core::ffi::c_void,
        ldvl: i64,
        dataTypeVR: cudaDataType,
        VR: *const ::core::ffi::c_void,
        ldvr: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgeqrf(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeTau: cudaDataType,
        tau: *mut ::core::ffi::c_void,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeTau: cudaDataType,
        tau: *const ::core::ffi::c_void,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgesvd(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeS: cudaDataType,
        S: *mut ::core::ffi::c_void,
        dataTypeU: cudaDataType,
        U: *mut ::core::ffi::c_void,
        ldu: i64,
        dataTypeVT: cudaDataType,
        VT: *mut ::core::ffi::c_void,
        ldvt: i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgesvd_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeS: cudaDataType,
        S: *const ::core::ffi::c_void,
        dataTypeU: cudaDataType,
        U: *const ::core::ffi::c_void,
        ldu: i64,
        dataTypeVT: cudaDataType,
        VT: *const ::core::ffi::c_void,
        ldvt: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgesvdjGetResidual(
        handle: cusolverDnHandle_t,
        info: gesvdjInfo_t,
        residual: *mut f64,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgesvdjGetSweeps(
        handle: cusolverDnHandle_t,
        info: gesvdjInfo_t,
        executed_sweeps: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgesvdjSetMaxSweeps(
        info: gesvdjInfo_t,
        max_sweeps: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgesvdjSetSortEig(
        info: gesvdjInfo_t,
        sort_svd: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgesvdjSetTolerance(
        info: gesvdjInfo_t,
        tolerance: f64,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgesvdp(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeS: cudaDataType,
        S: *mut ::core::ffi::c_void,
        dataTypeU: cudaDataType,
        U: *mut ::core::ffi::c_void,
        ldu: i64,
        dataTypeV: cudaDataType,
        V: *mut ::core::ffi::c_void,
        ldv: i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        d_info: *mut ::core::ffi::c_int,
        h_err_sigma: *mut f64,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgesvdp_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeS: cudaDataType,
        S: *const ::core::ffi::c_void,
        dataTypeU: cudaDataType,
        U: *const ::core::ffi::c_void,
        ldu: i64,
        dataTypeV: cudaDataType,
        V: *const ::core::ffi::c_void,
        ldv: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgesvdr(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobu: ::core::ffi::c_schar,
        jobv: ::core::ffi::c_schar,
        m: i64,
        n: i64,
        k: i64,
        p: i64,
        niters: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeSrand: cudaDataType,
        Srand: *mut ::core::ffi::c_void,
        dataTypeUrand: cudaDataType,
        Urand: *mut ::core::ffi::c_void,
        ldUrand: i64,
        dataTypeVrand: cudaDataType,
        Vrand: *mut ::core::ffi::c_void,
        ldVrand: i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        d_info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgesvdr_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobu: ::core::ffi::c_schar,
        jobv: ::core::ffi::c_schar,
        m: i64,
        n: i64,
        k: i64,
        p: i64,
        niters: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeSrand: cudaDataType,
        Srand: *const ::core::ffi::c_void,
        dataTypeUrand: cudaDataType,
        Urand: *const ::core::ffi::c_void,
        ldUrand: i64,
        dataTypeVrand: cudaDataType,
        Vrand: *const ::core::ffi::c_void,
        ldVrand: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgetrf(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        ipiv: *mut i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgetrf_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXgetrs(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        trans: cublasOperation_t,
        n: i64,
        nrhs: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        ipiv: *const i64,
        dataTypeB: cudaDataType,
        B: *mut ::core::ffi::c_void,
        ldb: i64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    #[cfg(any(feature = "cuda-12040"))]
    pub fn cusolverDnXlarft(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        direct: cusolverDirectMode_t,
        storev: cusolverStorevMode_t,
        N: i64,
        K: i64,
        dataTypeV: cudaDataType,
        d_V: *const ::core::ffi::c_void,
        ldv: i64,
        dataTypeTau: cudaDataType,
        d_tau: *const ::core::ffi::c_void,
        dataTypeT: cudaDataType,
        d_T: *mut ::core::ffi::c_void,
        ldt: i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
    ) -> cusolverStatus_t;
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub fn cusolverDnXlarft(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        direct: cusolverDirectMode_t,
        storev: cusolverStorevMode_t,
        n: i64,
        k: i64,
        dataTypeV: cudaDataType,
        V: *const ::core::ffi::c_void,
        ldv: i64,
        dataTypeTau: cudaDataType,
        tau: *const ::core::ffi::c_void,
        dataTypeT: cudaDataType,
        T: *mut ::core::ffi::c_void,
        ldt: i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
    ) -> cusolverStatus_t;
    #[cfg(any(feature = "cuda-12040"))]
    pub fn cusolverDnXlarft_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        direct: cusolverDirectMode_t,
        storev: cusolverStorevMode_t,
        N: i64,
        K: i64,
        dataTypeV: cudaDataType,
        d_V: *const ::core::ffi::c_void,
        ldv: i64,
        dataTypeTau: cudaDataType,
        d_tau: *const ::core::ffi::c_void,
        dataTypeT: cudaDataType,
        d_T: *mut ::core::ffi::c_void,
        ldt: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t;
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub fn cusolverDnXlarft_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        direct: cusolverDirectMode_t,
        storev: cusolverStorevMode_t,
        n: i64,
        k: i64,
        dataTypeV: cudaDataType,
        V: *const ::core::ffi::c_void,
        ldv: i64,
        dataTypeTau: cudaDataType,
        tau: *const ::core::ffi::c_void,
        dataTypeT: cudaDataType,
        T: *mut ::core::ffi::c_void,
        ldt: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXpotrf(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXpotrf_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXpotrs(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        uplo: cublasFillMode_t,
        n: i64,
        nrhs: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeB: cudaDataType,
        B: *mut ::core::ffi::c_void,
        ldb: i64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
    pub fn cusolverDnXsyevBatched(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *mut ::core::ffi::c_void,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
        batchSize: i64,
    ) -> cusolverStatus_t;
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
    pub fn cusolverDnXsyevBatched_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *const ::core::ffi::c_void,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
        batchSize: i64,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXsyevd(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *mut ::core::ffi::c_void,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXsyevd_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *const ::core::ffi::c_void,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXsyevdx(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        vl: *mut ::core::ffi::c_void,
        vu: *mut ::core::ffi::c_void,
        il: i64,
        iu: i64,
        meig64: *mut i64,
        dataTypeW: cudaDataType,
        W: *mut ::core::ffi::c_void,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXsyevdx_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        vl: *mut ::core::ffi::c_void,
        vu: *mut ::core::ffi::c_void,
        il: i64,
        iu: i64,
        h_meig: *mut i64,
        dataTypeW: cudaDataType,
        W: *const ::core::ffi::c_void,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXsyevjGetResidual(
        handle: cusolverDnHandle_t,
        info: syevjInfo_t,
        residual: *mut f64,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXsyevjGetSweeps(
        handle: cusolverDnHandle_t,
        info: syevjInfo_t,
        executed_sweeps: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXsyevjSetMaxSweeps(
        info: syevjInfo_t,
        max_sweeps: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXsyevjSetSortEig(
        info: syevjInfo_t,
        sort_eig: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXsyevjSetTolerance(
        info: syevjInfo_t,
        tolerance: f64,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXsytrs(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: i64,
        nrhs: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        ipiv: *const i64,
        dataTypeB: cudaDataType,
        B: *mut ::core::ffi::c_void,
        ldb: i64,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXsytrs_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: i64,
        nrhs: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        ipiv: *const i64,
        dataTypeB: cudaDataType,
        B: *mut ::core::ffi::c_void,
        ldb: i64,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXtrtri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        diag: cublasDiagType_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnXtrtri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        diag: cublasDiagType_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZCgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZCgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZCgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZCgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZEgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZEgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZEgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZEgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZKgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZKgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZKgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZKgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZYgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZYgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZYgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZYgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZZgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZZgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZZgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZZgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZgebrd(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        D: *mut f64,
        E: *mut f64,
        TAUQ: *mut cuDoubleComplex,
        TAUP: *mut cuDoubleComplex,
        Work: *mut cuDoubleComplex,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZgebrd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZgeqrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        TAU: *mut cuDoubleComplex,
        Workspace: *mut cuDoubleComplex,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZgeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZgesvd(
        handle: cusolverDnHandle_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        S: *mut f64,
        U: *mut cuDoubleComplex,
        ldu: ::core::ffi::c_int,
        VT: *mut cuDoubleComplex,
        ldvt: ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        rwork: *mut f64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZgesvd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZgesvdaStridedBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *mut f64,
        strideS: ::core::ffi::c_longlong,
        d_U: *mut cuDoubleComplex,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *mut cuDoubleComplex,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        d_work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        h_R_nrmF: *mut f64,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZgesvdaStridedBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *const f64,
        strideS: ::core::ffi::c_longlong,
        d_U: *const cuDoubleComplex,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *const cuDoubleComplex,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        lwork: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZgesvdj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        S: *mut f64,
        U: *mut cuDoubleComplex,
        ldu: ::core::ffi::c_int,
        V: *mut cuDoubleComplex,
        ldv: ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZgesvdjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        S: *mut f64,
        U: *mut cuDoubleComplex,
        ldu: ::core::ffi::c_int,
        V: *mut cuDoubleComplex,
        ldv: ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZgesvdjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        S: *const f64,
        U: *const cuDoubleComplex,
        ldu: ::core::ffi::c_int,
        V: *const cuDoubleComplex,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZgesvdj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        S: *const f64,
        U: *const cuDoubleComplex,
        ldu: ::core::ffi::c_int,
        V: *const cuDoubleComplex,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZgetrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        Workspace: *mut cuDoubleComplex,
        devIpiv: *mut ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZgetrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZgetrs(
        handle: cusolverDnHandle_t,
        trans: cublasOperation_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        B: *mut cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZheevd(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZheevd_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZheevdx(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f64,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZheevdx_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZheevj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZheevjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZheevjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZheevj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZhegvd(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZhegvd_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *const cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZhegvdx(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f64,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZhegvdx_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *const cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZhegvj(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZhegvj_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *const cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZhetrd(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        d: *mut f64,
        e: *mut f64,
        tau: *mut cuDoubleComplex,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZhetrd_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        d: *const f64,
        e: *const f64,
        tau: *const cuDoubleComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZlaswp(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        k1: ::core::ffi::c_int,
        k2: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        incx: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZlauum(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZlauum_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZpotrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        Workspace: *mut cuDoubleComplex,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZpotrfBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        Aarray: *mut *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        infoArray: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZpotrf_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZpotri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZpotri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZpotrs(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZpotrsBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *mut *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *mut *mut cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZsytrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        ipiv: *mut ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZsytrf_bufferSize(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZsytri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZsytri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZungbr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZungbr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZungqr(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZungqr_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZungtr(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZungtr_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZunmqr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        C: *mut cuDoubleComplex,
        ldc: ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZunmqr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        C: *const cuDoubleComplex,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZunmtr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *mut cuDoubleComplex,
        C: *mut cuDoubleComplex,
        ldc: ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverDnZunmtr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        C: *const cuDoubleComplex,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverGetProperty(
        type_: libraryPropertyType,
        value: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverGetVersion(version: *mut ::core::ffi::c_int) -> cusolverStatus_t;
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
    pub fn cusolverRfAccessBundledFactorsDevice(
        handle: cusolverRfHandle_t,
        nnzM: *mut ::core::ffi::c_int,
        Mp: *mut *mut ::core::ffi::c_int,
        Mi: *mut *mut ::core::ffi::c_int,
        Mx: *mut *mut f64,
    ) -> cusolverStatus_t;
    pub fn cusolverRfAnalyze(handle: cusolverRfHandle_t) -> cusolverStatus_t;
    pub fn cusolverRfBatchAnalyze(handle: cusolverRfHandle_t) -> cusolverStatus_t;
    pub fn cusolverRfBatchRefactor(handle: cusolverRfHandle_t) -> cusolverStatus_t;
    pub fn cusolverRfBatchResetValues(
        batchSize: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        csrRowPtrA: *mut ::core::ffi::c_int,
        csrColIndA: *mut ::core::ffi::c_int,
        csrValA_array: *mut *mut f64,
        P: *mut ::core::ffi::c_int,
        Q: *mut ::core::ffi::c_int,
        handle: cusolverRfHandle_t,
    ) -> cusolverStatus_t;
    pub fn cusolverRfBatchSetupHost(
        batchSize: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        h_csrRowPtrA: *mut ::core::ffi::c_int,
        h_csrColIndA: *mut ::core::ffi::c_int,
        h_csrValA_array: *mut *mut f64,
        nnzL: ::core::ffi::c_int,
        h_csrRowPtrL: *mut ::core::ffi::c_int,
        h_csrColIndL: *mut ::core::ffi::c_int,
        h_csrValL: *mut f64,
        nnzU: ::core::ffi::c_int,
        h_csrRowPtrU: *mut ::core::ffi::c_int,
        h_csrColIndU: *mut ::core::ffi::c_int,
        h_csrValU: *mut f64,
        h_P: *mut ::core::ffi::c_int,
        h_Q: *mut ::core::ffi::c_int,
        handle: cusolverRfHandle_t,
    ) -> cusolverStatus_t;
    pub fn cusolverRfBatchSolve(
        handle: cusolverRfHandle_t,
        P: *mut ::core::ffi::c_int,
        Q: *mut ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        Temp: *mut f64,
        ldt: ::core::ffi::c_int,
        XF_array: *mut *mut f64,
        ldxf: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverRfBatchZeroPivot(
        handle: cusolverRfHandle_t,
        position: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverRfCreate(handle: *mut cusolverRfHandle_t) -> cusolverStatus_t;
    pub fn cusolverRfDestroy(handle: cusolverRfHandle_t) -> cusolverStatus_t;
    pub fn cusolverRfExtractBundledFactorsHost(
        handle: cusolverRfHandle_t,
        h_nnzM: *mut ::core::ffi::c_int,
        h_Mp: *mut *mut ::core::ffi::c_int,
        h_Mi: *mut *mut ::core::ffi::c_int,
        h_Mx: *mut *mut f64,
    ) -> cusolverStatus_t;
    pub fn cusolverRfExtractSplitFactorsHost(
        handle: cusolverRfHandle_t,
        h_nnzL: *mut ::core::ffi::c_int,
        h_csrRowPtrL: *mut *mut ::core::ffi::c_int,
        h_csrColIndL: *mut *mut ::core::ffi::c_int,
        h_csrValL: *mut *mut f64,
        h_nnzU: *mut ::core::ffi::c_int,
        h_csrRowPtrU: *mut *mut ::core::ffi::c_int,
        h_csrColIndU: *mut *mut ::core::ffi::c_int,
        h_csrValU: *mut *mut f64,
    ) -> cusolverStatus_t;
    pub fn cusolverRfGetAlgs(
        handle: cusolverRfHandle_t,
        factAlg: *mut cusolverRfFactorization_t,
        solveAlg: *mut cusolverRfTriangularSolve_t,
    ) -> cusolverStatus_t;
    pub fn cusolverRfGetMatrixFormat(
        handle: cusolverRfHandle_t,
        format: *mut cusolverRfMatrixFormat_t,
        diag: *mut cusolverRfUnitDiagonal_t,
    ) -> cusolverStatus_t;
    pub fn cusolverRfGetNumericBoostReport(
        handle: cusolverRfHandle_t,
        report: *mut cusolverRfNumericBoostReport_t,
    ) -> cusolverStatus_t;
    pub fn cusolverRfGetNumericProperties(
        handle: cusolverRfHandle_t,
        zero: *mut f64,
        boost: *mut f64,
    ) -> cusolverStatus_t;
    pub fn cusolverRfGetResetValuesFastMode(
        handle: cusolverRfHandle_t,
        fastMode: *mut cusolverRfResetValuesFastMode_t,
    ) -> cusolverStatus_t;
    pub fn cusolverRfRefactor(handle: cusolverRfHandle_t) -> cusolverStatus_t;
    pub fn cusolverRfResetValues(
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        csrRowPtrA: *mut ::core::ffi::c_int,
        csrColIndA: *mut ::core::ffi::c_int,
        csrValA: *mut f64,
        P: *mut ::core::ffi::c_int,
        Q: *mut ::core::ffi::c_int,
        handle: cusolverRfHandle_t,
    ) -> cusolverStatus_t;
    pub fn cusolverRfSetAlgs(
        handle: cusolverRfHandle_t,
        factAlg: cusolverRfFactorization_t,
        solveAlg: cusolverRfTriangularSolve_t,
    ) -> cusolverStatus_t;
    pub fn cusolverRfSetMatrixFormat(
        handle: cusolverRfHandle_t,
        format: cusolverRfMatrixFormat_t,
        diag: cusolverRfUnitDiagonal_t,
    ) -> cusolverStatus_t;
    pub fn cusolverRfSetNumericProperties(
        handle: cusolverRfHandle_t,
        zero: f64,
        boost: f64,
    ) -> cusolverStatus_t;
    pub fn cusolverRfSetResetValuesFastMode(
        handle: cusolverRfHandle_t,
        fastMode: cusolverRfResetValuesFastMode_t,
    ) -> cusolverStatus_t;
    pub fn cusolverRfSetupDevice(
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        csrRowPtrA: *mut ::core::ffi::c_int,
        csrColIndA: *mut ::core::ffi::c_int,
        csrValA: *mut f64,
        nnzL: ::core::ffi::c_int,
        csrRowPtrL: *mut ::core::ffi::c_int,
        csrColIndL: *mut ::core::ffi::c_int,
        csrValL: *mut f64,
        nnzU: ::core::ffi::c_int,
        csrRowPtrU: *mut ::core::ffi::c_int,
        csrColIndU: *mut ::core::ffi::c_int,
        csrValU: *mut f64,
        P: *mut ::core::ffi::c_int,
        Q: *mut ::core::ffi::c_int,
        handle: cusolverRfHandle_t,
    ) -> cusolverStatus_t;
    pub fn cusolverRfSetupHost(
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        h_csrRowPtrA: *mut ::core::ffi::c_int,
        h_csrColIndA: *mut ::core::ffi::c_int,
        h_csrValA: *mut f64,
        nnzL: ::core::ffi::c_int,
        h_csrRowPtrL: *mut ::core::ffi::c_int,
        h_csrColIndL: *mut ::core::ffi::c_int,
        h_csrValL: *mut f64,
        nnzU: ::core::ffi::c_int,
        h_csrRowPtrU: *mut ::core::ffi::c_int,
        h_csrColIndU: *mut ::core::ffi::c_int,
        h_csrValU: *mut f64,
        h_P: *mut ::core::ffi::c_int,
        h_Q: *mut ::core::ffi::c_int,
        handle: cusolverRfHandle_t,
    ) -> cusolverStatus_t;
    pub fn cusolverRfSolve(
        handle: cusolverRfHandle_t,
        P: *mut ::core::ffi::c_int,
        Q: *mut ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        Temp: *mut f64,
        ldt: ::core::ffi::c_int,
        XF: *mut f64,
        ldxf: ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpCcsreigsHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        left_bottom_corner: cuComplex,
        right_upper_corner: cuComplex,
        num_eigs: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpCcsreigvsi(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: cuComplex,
        x0: *const cuComplex,
        maxite: ::core::ffi::c_int,
        eps: f32,
        mu: *mut cuComplex,
        x: *mut cuComplex,
    ) -> cusolverStatus_t;
    pub fn cusolverSpCcsreigvsiHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: cuComplex,
        x0: *const cuComplex,
        maxite: ::core::ffi::c_int,
        tol: f32,
        mu: *mut cuComplex,
        x: *mut cuComplex,
    ) -> cusolverStatus_t;
    pub fn cusolverSpCcsrlsqvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuComplex,
        tol: f32,
        rankA: *mut ::core::ffi::c_int,
        x: *mut cuComplex,
        p: *mut ::core::ffi::c_int,
        min_norm: *mut f32,
    ) -> cusolverStatus_t;
    pub fn cusolverSpCcsrlsvchol(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const cuComplex,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut cuComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpCcsrlsvcholHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const cuComplex,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut cuComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpCcsrlsvluHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuComplex,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut cuComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpCcsrlsvqr(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const cuComplex,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut cuComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpCcsrlsvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuComplex,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut cuComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpCcsrqrBufferInfoBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        internalDataInBytes: *mut usize,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverSpCcsrqrsvBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuComplex,
        x: *mut cuComplex,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        pBuffer: *mut ::core::ffi::c_void,
    ) -> cusolverStatus_t;
    pub fn cusolverSpCcsrzfdHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        P: *mut ::core::ffi::c_int,
        numnz: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpCreate(handle: *mut cusolverSpHandle_t) -> cusolverStatus_t;
    pub fn cusolverSpCreateCsrqrInfo(info: *mut csrqrInfo_t) -> cusolverStatus_t;
    pub fn cusolverSpDcsreigsHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        left_bottom_corner: cuDoubleComplex,
        right_upper_corner: cuDoubleComplex,
        num_eigs: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpDcsreigvsi(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: f64,
        x0: *const f64,
        maxite: ::core::ffi::c_int,
        eps: f64,
        mu: *mut f64,
        x: *mut f64,
    ) -> cusolverStatus_t;
    pub fn cusolverSpDcsreigvsiHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: f64,
        x0: *const f64,
        maxite: ::core::ffi::c_int,
        tol: f64,
        mu: *mut f64,
        x: *mut f64,
    ) -> cusolverStatus_t;
    pub fn cusolverSpDcsrlsqvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f64,
        tol: f64,
        rankA: *mut ::core::ffi::c_int,
        x: *mut f64,
        p: *mut ::core::ffi::c_int,
        min_norm: *mut f64,
    ) -> cusolverStatus_t;
    pub fn cusolverSpDcsrlsvchol(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f64,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const f64,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut f64,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpDcsrlsvcholHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f64,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const f64,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut f64,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpDcsrlsvluHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f64,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut f64,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpDcsrlsvqr(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f64,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const f64,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut f64,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpDcsrlsvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f64,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut f64,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpDcsrqrBufferInfoBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f64,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        internalDataInBytes: *mut usize,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverSpDcsrqrsvBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f64,
        x: *mut f64,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        pBuffer: *mut ::core::ffi::c_void,
    ) -> cusolverStatus_t;
    pub fn cusolverSpDcsrzfdHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        P: *mut ::core::ffi::c_int,
        numnz: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpDestroy(handle: cusolverSpHandle_t) -> cusolverStatus_t;
    pub fn cusolverSpDestroyCsrqrInfo(info: csrqrInfo_t) -> cusolverStatus_t;
    pub fn cusolverSpGetStream(
        handle: cusolverSpHandle_t,
        streamId: *mut cudaStream_t,
    ) -> cusolverStatus_t;
    pub fn cusolverSpScsreigsHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        left_bottom_corner: cuComplex,
        right_upper_corner: cuComplex,
        num_eigs: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpScsreigvsi(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: f32,
        x0: *const f32,
        maxite: ::core::ffi::c_int,
        eps: f32,
        mu: *mut f32,
        x: *mut f32,
    ) -> cusolverStatus_t;
    pub fn cusolverSpScsreigvsiHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: f32,
        x0: *const f32,
        maxite: ::core::ffi::c_int,
        tol: f32,
        mu: *mut f32,
        x: *mut f32,
    ) -> cusolverStatus_t;
    pub fn cusolverSpScsrlsqvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f32,
        tol: f32,
        rankA: *mut ::core::ffi::c_int,
        x: *mut f32,
        p: *mut ::core::ffi::c_int,
        min_norm: *mut f32,
    ) -> cusolverStatus_t;
    pub fn cusolverSpScsrlsvchol(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f32,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const f32,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut f32,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpScsrlsvcholHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f32,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const f32,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut f32,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpScsrlsvluHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f32,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut f32,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpScsrlsvqr(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f32,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const f32,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut f32,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpScsrlsvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f32,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut f32,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpScsrqrBufferInfoBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f32,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        internalDataInBytes: *mut usize,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverSpScsrqrsvBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f32,
        x: *mut f32,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        pBuffer: *mut ::core::ffi::c_void,
    ) -> cusolverStatus_t;
    pub fn cusolverSpScsrzfdHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        P: *mut ::core::ffi::c_int,
        numnz: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpSetStream(
        handle: cusolverSpHandle_t,
        streamId: cudaStream_t,
    ) -> cusolverStatus_t;
    pub fn cusolverSpXcsrissymHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrEndPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        issym: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpXcsrmetisndHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        options: *const i64,
        p: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpXcsrpermHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *mut ::core::ffi::c_int,
        csrColIndA: *mut ::core::ffi::c_int,
        p: *const ::core::ffi::c_int,
        q: *const ::core::ffi::c_int,
        map: *mut ::core::ffi::c_int,
        pBuffer: *mut ::core::ffi::c_void,
    ) -> cusolverStatus_t;
    pub fn cusolverSpXcsrperm_bufferSizeHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        p: *const ::core::ffi::c_int,
        q: *const ::core::ffi::c_int,
        bufferSizeInBytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverSpXcsrqrAnalysisBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        info: csrqrInfo_t,
    ) -> cusolverStatus_t;
    pub fn cusolverSpXcsrsymamdHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        p: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpXcsrsymmdqHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        p: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpXcsrsymrcmHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        p: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpZcsreigsHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        left_bottom_corner: cuDoubleComplex,
        right_upper_corner: cuDoubleComplex,
        num_eigs: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpZcsreigvsi(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: cuDoubleComplex,
        x0: *const cuDoubleComplex,
        maxite: ::core::ffi::c_int,
        eps: f64,
        mu: *mut cuDoubleComplex,
        x: *mut cuDoubleComplex,
    ) -> cusolverStatus_t;
    pub fn cusolverSpZcsreigvsiHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: cuDoubleComplex,
        x0: *const cuDoubleComplex,
        maxite: ::core::ffi::c_int,
        tol: f64,
        mu: *mut cuDoubleComplex,
        x: *mut cuDoubleComplex,
    ) -> cusolverStatus_t;
    pub fn cusolverSpZcsrlsqvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuDoubleComplex,
        tol: f64,
        rankA: *mut ::core::ffi::c_int,
        x: *mut cuDoubleComplex,
        p: *mut ::core::ffi::c_int,
        min_norm: *mut f64,
    ) -> cusolverStatus_t;
    pub fn cusolverSpZcsrlsvchol(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuDoubleComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const cuDoubleComplex,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut cuDoubleComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpZcsrlsvcholHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuDoubleComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const cuDoubleComplex,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut cuDoubleComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpZcsrlsvluHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuDoubleComplex,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut cuDoubleComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpZcsrlsvqr(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuDoubleComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const cuDoubleComplex,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut cuDoubleComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpZcsrlsvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuDoubleComplex,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut cuDoubleComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
    pub fn cusolverSpZcsrqrBufferInfoBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuDoubleComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        internalDataInBytes: *mut usize,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t;
    pub fn cusolverSpZcsrqrsvBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuDoubleComplex,
        x: *mut cuDoubleComplex,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        pBuffer: *mut ::core::ffi::c_void,
    ) -> cusolverStatus_t;
    pub fn cusolverSpZcsrzfdHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        P: *mut ::core::ffi::c_int,
        numnz: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t;
}
#[cfg(feature = "dynamic-loading")]
mod loaded {
    use super::*;
    pub unsafe fn cusolverDnCCgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCCgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnCCgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCCgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnCCgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCCgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnCCgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCCgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnCEgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCEgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnCEgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCEgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnCEgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCEgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnCEgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCEgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnCKgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCKgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnCKgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCKgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnCKgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCKgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnCKgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCKgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnCYgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCYgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnCYgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCYgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnCYgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCYgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnCYgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuComplex,
        lddb: cusolver_int_t,
        dX: *mut cuComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCYgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnCgebrd(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        D: *mut f32,
        E: *mut f32,
        TAUQ: *mut cuComplex,
        TAUP: *mut cuComplex,
        Work: *mut cuComplex,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCgebrd)(
            handle,
            m,
            n,
            A,
            lda,
            D,
            E,
            TAUQ,
            TAUP,
            Work,
            Lwork,
            devInfo,
        )
    }
    pub unsafe fn cusolverDnCgebrd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCgebrd_bufferSize)(handle, m, n, Lwork)
    }
    pub unsafe fn cusolverDnCgeqrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        TAU: *mut cuComplex,
        Workspace: *mut cuComplex,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCgeqrf)(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    }
    pub unsafe fn cusolverDnCgeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCgeqrf_bufferSize)(handle, m, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnCgesvd(
        handle: cusolverDnHandle_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        S: *mut f32,
        U: *mut cuComplex,
        ldu: ::core::ffi::c_int,
        VT: *mut cuComplex,
        ldvt: ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        rwork: *mut f32,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCgesvd)(
            handle,
            jobu,
            jobvt,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            VT,
            ldvt,
            work,
            lwork,
            rwork,
            info,
        )
    }
    pub unsafe fn cusolverDnCgesvd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCgesvd_bufferSize)(handle, m, n, lwork)
    }
    pub unsafe fn cusolverDnCgesvdaStridedBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const cuComplex,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *mut f32,
        strideS: ::core::ffi::c_longlong,
        d_U: *mut cuComplex,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *mut cuComplex,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        d_work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        h_R_nrmF: *mut f64,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCgesvdaStridedBatched)(
            handle,
            jobz,
            rank,
            m,
            n,
            d_A,
            lda,
            strideA,
            d_S,
            strideS,
            d_U,
            ldu,
            strideU,
            d_V,
            ldv,
            strideV,
            d_work,
            lwork,
            d_info,
            h_R_nrmF,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnCgesvdaStridedBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const cuComplex,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *const f32,
        strideS: ::core::ffi::c_longlong,
        d_U: *const cuComplex,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *const cuComplex,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        lwork: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCgesvdaStridedBatched_bufferSize)(
            handle,
            jobz,
            rank,
            m,
            n,
            d_A,
            lda,
            strideA,
            d_S,
            strideS,
            d_U,
            ldu,
            strideU,
            d_V,
            ldv,
            strideV,
            lwork,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnCgesvdj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        S: *mut f32,
        U: *mut cuComplex,
        ldu: ::core::ffi::c_int,
        V: *mut cuComplex,
        ldv: ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCgesvdj)(
            handle,
            jobz,
            econ,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            work,
            lwork,
            info,
            params,
        )
    }
    pub unsafe fn cusolverDnCgesvdjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        S: *mut f32,
        U: *mut cuComplex,
        ldu: ::core::ffi::c_int,
        V: *mut cuComplex,
        ldv: ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCgesvdjBatched)(
            handle,
            jobz,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            work,
            lwork,
            info,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnCgesvdjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        S: *const f32,
        U: *const cuComplex,
        ldu: ::core::ffi::c_int,
        V: *const cuComplex,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCgesvdjBatched_bufferSize)(
            handle,
            jobz,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            lwork,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnCgesvdj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        S: *const f32,
        U: *const cuComplex,
        ldu: ::core::ffi::c_int,
        V: *const cuComplex,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCgesvdj_bufferSize)(
            handle,
            jobz,
            econ,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            lwork,
            params,
        )
    }
    pub unsafe fn cusolverDnCgetrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        Workspace: *mut cuComplex,
        devIpiv: *mut ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCgetrf)(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    }
    pub unsafe fn cusolverDnCgetrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCgetrf_bufferSize)(handle, m, n, A, lda, Lwork)
    }
    pub unsafe fn cusolverDnCgetrs(
        handle: cusolverDnHandle_t,
        trans: cublasOperation_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        B: *mut cuComplex,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCgetrs)(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    }
    pub unsafe fn cusolverDnCheevd(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCheevd)(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    }
    pub unsafe fn cusolverDnCheevd_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCheevd_bufferSize)(handle, jobz, uplo, n, A, lda, W, lwork)
    }
    pub unsafe fn cusolverDnCheevdx(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f32,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCheevdx)(
            handle,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnCheevdx_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCheevdx_bufferSize)(
            handle,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            lwork,
        )
    }
    pub unsafe fn cusolverDnCheevj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCheevj)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            work,
            lwork,
            info,
            params,
        )
    }
    pub unsafe fn cusolverDnCheevjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCheevjBatched)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            work,
            lwork,
            info,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnCheevjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCheevjBatched_bufferSize)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            lwork,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnCheevj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCheevj_bufferSize)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            lwork,
            params,
        )
    }
    pub unsafe fn cusolverDnChegvd(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuComplex,
        ldb: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnChegvd)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnChegvd_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        B: *const cuComplex,
        ldb: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnChegvd_bufferSize)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            lwork,
        )
    }
    pub unsafe fn cusolverDnChegvdx(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuComplex,
        ldb: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f32,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnChegvdx)(
            handle,
            itype,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnChegvdx_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        B: *const cuComplex,
        ldb: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnChegvdx_bufferSize)(
            handle,
            itype,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            lwork,
        )
    }
    pub unsafe fn cusolverDnChegvj(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuComplex,
        ldb: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnChegvj)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            work,
            lwork,
            info,
            params,
        )
    }
    pub unsafe fn cusolverDnChegvj_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        B: *const cuComplex,
        ldb: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnChegvj_bufferSize)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            lwork,
            params,
        )
    }
    pub unsafe fn cusolverDnChetrd(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        d: *mut f32,
        e: *mut f32,
        tau: *mut cuComplex,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnChetrd)(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnChetrd_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        d: *const f32,
        e: *const f32,
        tau: *const cuComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnChetrd_bufferSize)(handle, uplo, n, A, lda, d, e, tau, lwork)
    }
    pub unsafe fn cusolverDnClaswp(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        k1: ::core::ffi::c_int,
        k2: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        incx: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnClaswp)(handle, n, A, lda, k1, k2, devIpiv, incx)
    }
    pub unsafe fn cusolverDnClauum(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnClauum)(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    pub unsafe fn cusolverDnClauum_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnClauum_bufferSize)(handle, uplo, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnCpotrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        Workspace: *mut cuComplex,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCpotrf)(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    }
    pub unsafe fn cusolverDnCpotrfBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        Aarray: *mut *mut cuComplex,
        lda: ::core::ffi::c_int,
        infoArray: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCpotrfBatched)(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    }
    pub unsafe fn cusolverDnCpotrf_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCpotrf_bufferSize)(handle, uplo, n, A, lda, Lwork)
    }
    pub unsafe fn cusolverDnCpotri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCpotri)(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    pub unsafe fn cusolverDnCpotri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCpotri_bufferSize)(handle, uplo, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnCpotrs(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuComplex,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCpotrs)(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    }
    pub unsafe fn cusolverDnCpotrsBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *mut *mut cuComplex,
        lda: ::core::ffi::c_int,
        B: *mut *mut cuComplex,
        ldb: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCpotrsBatched)(
            handle,
            uplo,
            n,
            nrhs,
            A,
            lda,
            B,
            ldb,
            d_info,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnCreate(handle: *mut cusolverDnHandle_t) -> cusolverStatus_t {
        (culib().cusolverDnCreate)(handle)
    }
    pub unsafe fn cusolverDnCreateGesvdjInfo(
        info: *mut gesvdjInfo_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCreateGesvdjInfo)(info)
    }
    pub unsafe fn cusolverDnCreateParams(
        params: *mut cusolverDnParams_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCreateParams)(params)
    }
    pub unsafe fn cusolverDnCreateSyevjInfo(info: *mut syevjInfo_t) -> cusolverStatus_t {
        (culib().cusolverDnCreateSyevjInfo)(info)
    }
    pub unsafe fn cusolverDnCsytrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        ipiv: *mut ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCsytrf)(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    pub unsafe fn cusolverDnCsytrf_bufferSize(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCsytrf_bufferSize)(handle, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnCsytri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCsytri)(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    pub unsafe fn cusolverDnCsytri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCsytri_bufferSize)(handle, uplo, n, A, lda, ipiv, lwork)
    }
    pub unsafe fn cusolverDnCungbr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCungbr)(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnCungbr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCungbr_bufferSize)(handle, side, m, n, k, A, lda, tau, lwork)
    }
    pub unsafe fn cusolverDnCungqr(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCungqr)(handle, m, n, k, A, lda, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnCungqr_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCungqr_bufferSize)(handle, m, n, k, A, lda, tau, lwork)
    }
    pub unsafe fn cusolverDnCungtr(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCungtr)(handle, uplo, n, A, lda, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnCungtr_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnCungtr_bufferSize)(handle, uplo, n, A, lda, tau, lwork)
    }
    pub unsafe fn cusolverDnCunmqr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        C: *mut cuComplex,
        ldc: ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCunmqr)(
            handle,
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            tau,
            C,
            ldc,
            work,
            lwork,
            devInfo,
        )
    }
    pub unsafe fn cusolverDnCunmqr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        C: *const cuComplex,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCunmqr_bufferSize)(
            handle,
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            tau,
            C,
            ldc,
            lwork,
        )
    }
    pub unsafe fn cusolverDnCunmtr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuComplex,
        lda: ::core::ffi::c_int,
        tau: *mut cuComplex,
        C: *mut cuComplex,
        ldc: ::core::ffi::c_int,
        work: *mut cuComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCunmtr)(
            handle,
            side,
            uplo,
            trans,
            m,
            n,
            A,
            lda,
            tau,
            C,
            ldc,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnCunmtr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const cuComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuComplex,
        C: *const cuComplex,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnCunmtr_bufferSize)(
            handle,
            side,
            uplo,
            trans,
            m,
            n,
            A,
            lda,
            tau,
            C,
            ldc,
            lwork,
        )
    }
    pub unsafe fn cusolverDnDBgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDBgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnDBgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDBgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnDBgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDBgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnDBgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDBgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnDDgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDDgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnDDgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDDgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnDDgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDDgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnDDgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDDgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnDHgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDHgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnDHgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDHgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnDHgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDHgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnDHgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDHgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnDSgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDSgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnDSgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDSgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnDSgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDSgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnDSgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDSgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnDXgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDXgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnDXgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDXgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnDXgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDXgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnDXgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f64,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f64,
        lddb: cusolver_int_t,
        dX: *mut f64,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDXgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnDestroy(handle: cusolverDnHandle_t) -> cusolverStatus_t {
        (culib().cusolverDnDestroy)(handle)
    }
    pub unsafe fn cusolverDnDestroyGesvdjInfo(info: gesvdjInfo_t) -> cusolverStatus_t {
        (culib().cusolverDnDestroyGesvdjInfo)(info)
    }
    pub unsafe fn cusolverDnDestroyParams(
        params: cusolverDnParams_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDestroyParams)(params)
    }
    pub unsafe fn cusolverDnDestroySyevjInfo(info: syevjInfo_t) -> cusolverStatus_t {
        (culib().cusolverDnDestroySyevjInfo)(info)
    }
    pub unsafe fn cusolverDnDgebrd(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        D: *mut f64,
        E: *mut f64,
        TAUQ: *mut f64,
        TAUP: *mut f64,
        Work: *mut f64,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDgebrd)(
            handle,
            m,
            n,
            A,
            lda,
            D,
            E,
            TAUQ,
            TAUP,
            Work,
            Lwork,
            devInfo,
        )
    }
    pub unsafe fn cusolverDnDgebrd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDgebrd_bufferSize)(handle, m, n, Lwork)
    }
    pub unsafe fn cusolverDnDgeqrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        TAU: *mut f64,
        Workspace: *mut f64,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDgeqrf)(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    }
    pub unsafe fn cusolverDnDgeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDgeqrf_bufferSize)(handle, m, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnDgesvd(
        handle: cusolverDnHandle_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        S: *mut f64,
        U: *mut f64,
        ldu: ::core::ffi::c_int,
        VT: *mut f64,
        ldvt: ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        rwork: *mut f64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDgesvd)(
            handle,
            jobu,
            jobvt,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            VT,
            ldvt,
            work,
            lwork,
            rwork,
            info,
        )
    }
    pub unsafe fn cusolverDnDgesvd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDgesvd_bufferSize)(handle, m, n, lwork)
    }
    pub unsafe fn cusolverDnDgesvdaStridedBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const f64,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *mut f64,
        strideS: ::core::ffi::c_longlong,
        d_U: *mut f64,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *mut f64,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        d_work: *mut f64,
        lwork: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        h_R_nrmF: *mut f64,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDgesvdaStridedBatched)(
            handle,
            jobz,
            rank,
            m,
            n,
            d_A,
            lda,
            strideA,
            d_S,
            strideS,
            d_U,
            ldu,
            strideU,
            d_V,
            ldv,
            strideV,
            d_work,
            lwork,
            d_info,
            h_R_nrmF,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnDgesvdaStridedBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const f64,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *const f64,
        strideS: ::core::ffi::c_longlong,
        d_U: *const f64,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *const f64,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        lwork: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDgesvdaStridedBatched_bufferSize)(
            handle,
            jobz,
            rank,
            m,
            n,
            d_A,
            lda,
            strideA,
            d_S,
            strideS,
            d_U,
            ldu,
            strideU,
            d_V,
            ldv,
            strideV,
            lwork,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnDgesvdj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        S: *mut f64,
        U: *mut f64,
        ldu: ::core::ffi::c_int,
        V: *mut f64,
        ldv: ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDgesvdj)(
            handle,
            jobz,
            econ,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            work,
            lwork,
            info,
            params,
        )
    }
    pub unsafe fn cusolverDnDgesvdjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        S: *mut f64,
        U: *mut f64,
        ldu: ::core::ffi::c_int,
        V: *mut f64,
        ldv: ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDgesvdjBatched)(
            handle,
            jobz,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            work,
            lwork,
            info,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnDgesvdjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        S: *const f64,
        U: *const f64,
        ldu: ::core::ffi::c_int,
        V: *const f64,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDgesvdjBatched_bufferSize)(
            handle,
            jobz,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            lwork,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnDgesvdj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        S: *const f64,
        U: *const f64,
        ldu: ::core::ffi::c_int,
        V: *const f64,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDgesvdj_bufferSize)(
            handle,
            jobz,
            econ,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            lwork,
            params,
        )
    }
    pub unsafe fn cusolverDnDgetrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        Workspace: *mut f64,
        devIpiv: *mut ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDgetrf)(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    }
    pub unsafe fn cusolverDnDgetrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDgetrf_bufferSize)(handle, m, n, A, lda, Lwork)
    }
    pub unsafe fn cusolverDnDgetrs(
        handle: cusolverDnHandle_t,
        trans: cublasOperation_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        B: *mut f64,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDgetrs)(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    }
    pub unsafe fn cusolverDnDlaswp(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        k1: ::core::ffi::c_int,
        k2: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        incx: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDlaswp)(handle, n, A, lda, k1, k2, devIpiv, incx)
    }
    pub unsafe fn cusolverDnDlauum(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDlauum)(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    pub unsafe fn cusolverDnDlauum_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDlauum_bufferSize)(handle, uplo, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnDorgbr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDorgbr)(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnDorgbr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDorgbr_bufferSize)(handle, side, m, n, k, A, lda, tau, lwork)
    }
    pub unsafe fn cusolverDnDorgqr(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDorgqr)(handle, m, n, k, A, lda, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnDorgqr_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDorgqr_bufferSize)(handle, m, n, k, A, lda, tau, lwork)
    }
    pub unsafe fn cusolverDnDorgtr(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDorgtr)(handle, uplo, n, A, lda, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnDorgtr_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDorgtr_bufferSize)(handle, uplo, n, A, lda, tau, lwork)
    }
    pub unsafe fn cusolverDnDormqr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        C: *mut f64,
        ldc: ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDormqr)(
            handle,
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            tau,
            C,
            ldc,
            work,
            lwork,
            devInfo,
        )
    }
    pub unsafe fn cusolverDnDormqr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        C: *const f64,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDormqr_bufferSize)(
            handle,
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            tau,
            C,
            ldc,
            lwork,
        )
    }
    pub unsafe fn cusolverDnDormtr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        tau: *mut f64,
        C: *mut f64,
        ldc: ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDormtr)(
            handle,
            side,
            uplo,
            trans,
            m,
            n,
            A,
            lda,
            tau,
            C,
            ldc,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnDormtr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        tau: *const f64,
        C: *const f64,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDormtr_bufferSize)(
            handle,
            side,
            uplo,
            trans,
            m,
            n,
            A,
            lda,
            tau,
            C,
            ldc,
            lwork,
        )
    }
    pub unsafe fn cusolverDnDpotrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        Workspace: *mut f64,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDpotrf)(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    }
    pub unsafe fn cusolverDnDpotrfBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        Aarray: *mut *mut f64,
        lda: ::core::ffi::c_int,
        infoArray: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDpotrfBatched)(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    }
    pub unsafe fn cusolverDnDpotrf_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDpotrf_bufferSize)(handle, uplo, n, A, lda, Lwork)
    }
    pub unsafe fn cusolverDnDpotri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDpotri)(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    pub unsafe fn cusolverDnDpotri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDpotri_bufferSize)(handle, uplo, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnDpotrs(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        B: *mut f64,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDpotrs)(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    }
    pub unsafe fn cusolverDnDpotrsBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *mut *mut f64,
        lda: ::core::ffi::c_int,
        B: *mut *mut f64,
        ldb: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDpotrsBatched)(
            handle,
            uplo,
            n,
            nrhs,
            A,
            lda,
            B,
            ldb,
            d_info,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnDsyevd(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDsyevd)(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    }
    pub unsafe fn cusolverDnDsyevd_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDsyevd_bufferSize)(handle, jobz, uplo, n, A, lda, W, lwork)
    }
    pub unsafe fn cusolverDnDsyevdx(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDsyevdx)(
            handle,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnDsyevdx_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDsyevdx_bufferSize)(
            handle,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            lwork,
        )
    }
    pub unsafe fn cusolverDnDsyevj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDsyevj)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            work,
            lwork,
            info,
            params,
        )
    }
    pub unsafe fn cusolverDnDsyevjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDsyevjBatched)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            work,
            lwork,
            info,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnDsyevjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDsyevjBatched_bufferSize)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            lwork,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnDsyevj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDsyevj_bufferSize)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            lwork,
            params,
        )
    }
    pub unsafe fn cusolverDnDsygvd(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        B: *mut f64,
        ldb: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDsygvd)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnDsygvd_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        B: *const f64,
        ldb: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDsygvd_bufferSize)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            lwork,
        )
    }
    pub unsafe fn cusolverDnDsygvdx(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        B: *mut f64,
        ldb: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDsygvdx)(
            handle,
            itype,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnDsygvdx_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        B: *const f64,
        ldb: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDsygvdx_bufferSize)(
            handle,
            itype,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            lwork,
        )
    }
    pub unsafe fn cusolverDnDsygvj(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        B: *mut f64,
        ldb: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDsygvj)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            work,
            lwork,
            info,
            params,
        )
    }
    pub unsafe fn cusolverDnDsygvj_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        B: *const f64,
        ldb: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnDsygvj_bufferSize)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            lwork,
            params,
        )
    }
    pub unsafe fn cusolverDnDsytrd(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        d: *mut f64,
        e: *mut f64,
        tau: *mut f64,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDsytrd)(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnDsytrd_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f64,
        lda: ::core::ffi::c_int,
        d: *const f64,
        e: *const f64,
        tau: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDsytrd_bufferSize)(handle, uplo, n, A, lda, d, e, tau, lwork)
    }
    pub unsafe fn cusolverDnDsytrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        ipiv: *mut ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDsytrf)(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    pub unsafe fn cusolverDnDsytrf_bufferSize(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDsytrf_bufferSize)(handle, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnDsytri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        work: *mut f64,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDsytri)(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    pub unsafe fn cusolverDnDsytri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f64,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnDsytri_bufferSize)(handle, uplo, n, A, lda, ipiv, lwork)
    }
    pub unsafe fn cusolverDnGeqrf(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeTau: cudaDataType,
        tau: *mut ::core::ffi::c_void,
        computeType: cudaDataType,
        pBuffer: *mut ::core::ffi::c_void,
        workspaceInBytes: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnGeqrf)(
            handle,
            params,
            m,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeTau,
            tau,
            computeType,
            pBuffer,
            workspaceInBytes,
            info,
        )
    }
    pub unsafe fn cusolverDnGeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeTau: cudaDataType,
        tau: *const ::core::ffi::c_void,
        computeType: cudaDataType,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnGeqrf_bufferSize)(
            handle,
            params,
            m,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeTau,
            tau,
            computeType,
            workspaceInBytes,
        )
    }
    pub unsafe fn cusolverDnGesvd(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeS: cudaDataType,
        S: *mut ::core::ffi::c_void,
        dataTypeU: cudaDataType,
        U: *mut ::core::ffi::c_void,
        ldu: i64,
        dataTypeVT: cudaDataType,
        VT: *mut ::core::ffi::c_void,
        ldvt: i64,
        computeType: cudaDataType,
        pBuffer: *mut ::core::ffi::c_void,
        workspaceInBytes: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnGesvd)(
            handle,
            params,
            jobu,
            jobvt,
            m,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeS,
            S,
            dataTypeU,
            U,
            ldu,
            dataTypeVT,
            VT,
            ldvt,
            computeType,
            pBuffer,
            workspaceInBytes,
            info,
        )
    }
    pub unsafe fn cusolverDnGesvd_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeS: cudaDataType,
        S: *const ::core::ffi::c_void,
        dataTypeU: cudaDataType,
        U: *const ::core::ffi::c_void,
        ldu: i64,
        dataTypeVT: cudaDataType,
        VT: *const ::core::ffi::c_void,
        ldvt: i64,
        computeType: cudaDataType,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnGesvd_bufferSize)(
            handle,
            params,
            jobu,
            jobvt,
            m,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeS,
            S,
            dataTypeU,
            U,
            ldu,
            dataTypeVT,
            VT,
            ldvt,
            computeType,
            workspaceInBytes,
        )
    }
    #[cfg(
        any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cusolverDnGetDeterministicMode(
        handle: cusolverDnHandle_t,
        mode: *mut cusolverDeterministicMode_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnGetDeterministicMode)(handle, mode)
    }
    pub unsafe fn cusolverDnGetStream(
        handle: cusolverDnHandle_t,
        streamId: *mut cudaStream_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnGetStream)(handle, streamId)
    }
    pub unsafe fn cusolverDnGetrf(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        ipiv: *mut i64,
        computeType: cudaDataType,
        pBuffer: *mut ::core::ffi::c_void,
        workspaceInBytes: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnGetrf)(
            handle,
            params,
            m,
            n,
            dataTypeA,
            A,
            lda,
            ipiv,
            computeType,
            pBuffer,
            workspaceInBytes,
            info,
        )
    }
    pub unsafe fn cusolverDnGetrf_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        computeType: cudaDataType,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnGetrf_bufferSize)(
            handle,
            params,
            m,
            n,
            dataTypeA,
            A,
            lda,
            computeType,
            workspaceInBytes,
        )
    }
    pub unsafe fn cusolverDnGetrs(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        trans: cublasOperation_t,
        n: i64,
        nrhs: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        ipiv: *const i64,
        dataTypeB: cudaDataType,
        B: *mut ::core::ffi::c_void,
        ldb: i64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnGetrs)(
            handle,
            params,
            trans,
            n,
            nrhs,
            dataTypeA,
            A,
            lda,
            ipiv,
            dataTypeB,
            B,
            ldb,
            info,
        )
    }
    pub unsafe fn cusolverDnIRSInfosCreate(
        infos_ptr: *mut cusolverDnIRSInfos_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSInfosCreate)(infos_ptr)
    }
    pub unsafe fn cusolverDnIRSInfosDestroy(
        infos: cusolverDnIRSInfos_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSInfosDestroy)(infos)
    }
    pub unsafe fn cusolverDnIRSInfosGetMaxIters(
        infos: cusolverDnIRSInfos_t,
        maxiters: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSInfosGetMaxIters)(infos, maxiters)
    }
    pub unsafe fn cusolverDnIRSInfosGetNiters(
        infos: cusolverDnIRSInfos_t,
        niters: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSInfosGetNiters)(infos, niters)
    }
    pub unsafe fn cusolverDnIRSInfosGetOuterNiters(
        infos: cusolverDnIRSInfos_t,
        outer_niters: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSInfosGetOuterNiters)(infos, outer_niters)
    }
    pub unsafe fn cusolverDnIRSInfosGetResidualHistory(
        infos: cusolverDnIRSInfos_t,
        residual_history: *mut *mut ::core::ffi::c_void,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSInfosGetResidualHistory)(infos, residual_history)
    }
    pub unsafe fn cusolverDnIRSInfosRequestResidual(
        infos: cusolverDnIRSInfos_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSInfosRequestResidual)(infos)
    }
    pub unsafe fn cusolverDnIRSParamsCreate(
        params_ptr: *mut cusolverDnIRSParams_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSParamsCreate)(params_ptr)
    }
    pub unsafe fn cusolverDnIRSParamsDestroy(
        params: cusolverDnIRSParams_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSParamsDestroy)(params)
    }
    pub unsafe fn cusolverDnIRSParamsDisableFallback(
        params: cusolverDnIRSParams_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSParamsDisableFallback)(params)
    }
    pub unsafe fn cusolverDnIRSParamsEnableFallback(
        params: cusolverDnIRSParams_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSParamsEnableFallback)(params)
    }
    pub unsafe fn cusolverDnIRSParamsGetMaxIters(
        params: cusolverDnIRSParams_t,
        maxiters: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSParamsGetMaxIters)(params, maxiters)
    }
    pub unsafe fn cusolverDnIRSParamsSetMaxIters(
        params: cusolverDnIRSParams_t,
        maxiters: cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSParamsSetMaxIters)(params, maxiters)
    }
    pub unsafe fn cusolverDnIRSParamsSetMaxItersInner(
        params: cusolverDnIRSParams_t,
        maxiters_inner: cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSParamsSetMaxItersInner)(params, maxiters_inner)
    }
    pub unsafe fn cusolverDnIRSParamsSetRefinementSolver(
        params: cusolverDnIRSParams_t,
        refinement_solver: cusolverIRSRefinement_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSParamsSetRefinementSolver)(params, refinement_solver)
    }
    pub unsafe fn cusolverDnIRSParamsSetSolverLowestPrecision(
        params: cusolverDnIRSParams_t,
        solver_lowest_precision: cusolverPrecType_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnIRSParamsSetSolverLowestPrecision)(
            params,
            solver_lowest_precision,
        )
    }
    pub unsafe fn cusolverDnIRSParamsSetSolverMainPrecision(
        params: cusolverDnIRSParams_t,
        solver_main_precision: cusolverPrecType_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnIRSParamsSetSolverMainPrecision)(params, solver_main_precision)
    }
    pub unsafe fn cusolverDnIRSParamsSetSolverPrecisions(
        params: cusolverDnIRSParams_t,
        solver_main_precision: cusolverPrecType_t,
        solver_lowest_precision: cusolverPrecType_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnIRSParamsSetSolverPrecisions)(
            params,
            solver_main_precision,
            solver_lowest_precision,
        )
    }
    pub unsafe fn cusolverDnIRSParamsSetTol(
        params: cusolverDnIRSParams_t,
        val: f64,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSParamsSetTol)(params, val)
    }
    pub unsafe fn cusolverDnIRSParamsSetTolInner(
        params: cusolverDnIRSParams_t,
        val: f64,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSParamsSetTolInner)(params, val)
    }
    pub unsafe fn cusolverDnIRSXgels(
        handle: cusolverDnHandle_t,
        gels_irs_params: cusolverDnIRSParams_t,
        gels_irs_infos: cusolverDnIRSInfos_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut ::core::ffi::c_void,
        ldda: cusolver_int_t,
        dB: *mut ::core::ffi::c_void,
        lddb: cusolver_int_t,
        dX: *mut ::core::ffi::c_void,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        niters: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnIRSXgels)(
            handle,
            gels_irs_params,
            gels_irs_infos,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            niters,
            d_info,
        )
    }
    pub unsafe fn cusolverDnIRSXgels_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnIRSParams_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSXgels_bufferSize)(handle, params, m, n, nrhs, lwork_bytes)
    }
    pub unsafe fn cusolverDnIRSXgesv(
        handle: cusolverDnHandle_t,
        gesv_irs_params: cusolverDnIRSParams_t,
        gesv_irs_infos: cusolverDnIRSInfos_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut ::core::ffi::c_void,
        ldda: cusolver_int_t,
        dB: *mut ::core::ffi::c_void,
        lddb: cusolver_int_t,
        dX: *mut ::core::ffi::c_void,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        niters: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnIRSXgesv)(
            handle,
            gesv_irs_params,
            gesv_irs_infos,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            niters,
            d_info,
        )
    }
    pub unsafe fn cusolverDnIRSXgesv_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnIRSParams_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib().cusolverDnIRSXgesv_bufferSize)(handle, params, n, nrhs, lwork_bytes)
    }
    #[cfg(
        any(
            feature = "cuda-11070",
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cusolverDnLoggerForceDisable() -> cusolverStatus_t {
        (culib().cusolverDnLoggerForceDisable)()
    }
    #[cfg(
        any(
            feature = "cuda-11070",
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cusolverDnLoggerOpenFile(
        logFile: *const ::core::ffi::c_char,
    ) -> cusolverStatus_t {
        (culib().cusolverDnLoggerOpenFile)(logFile)
    }
    #[cfg(
        any(
            feature = "cuda-11070",
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cusolverDnLoggerSetCallback(
        callback: cusolverDnLoggerCallback_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnLoggerSetCallback)(callback)
    }
    #[cfg(
        any(
            feature = "cuda-11070",
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cusolverDnLoggerSetFile(file: *mut FILE) -> cusolverStatus_t {
        (culib().cusolverDnLoggerSetFile)(file)
    }
    #[cfg(
        any(
            feature = "cuda-11070",
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cusolverDnLoggerSetLevel(
        level: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnLoggerSetLevel)(level)
    }
    #[cfg(
        any(
            feature = "cuda-11070",
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cusolverDnLoggerSetMask(mask: ::core::ffi::c_int) -> cusolverStatus_t {
        (culib().cusolverDnLoggerSetMask)(mask)
    }
    pub unsafe fn cusolverDnPotrf(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        computeType: cudaDataType,
        pBuffer: *mut ::core::ffi::c_void,
        workspaceInBytes: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnPotrf)(
            handle,
            params,
            uplo,
            n,
            dataTypeA,
            A,
            lda,
            computeType,
            pBuffer,
            workspaceInBytes,
            info,
        )
    }
    pub unsafe fn cusolverDnPotrf_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        computeType: cudaDataType,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnPotrf_bufferSize)(
            handle,
            params,
            uplo,
            n,
            dataTypeA,
            A,
            lda,
            computeType,
            workspaceInBytes,
        )
    }
    pub unsafe fn cusolverDnPotrs(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        uplo: cublasFillMode_t,
        n: i64,
        nrhs: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeB: cudaDataType,
        B: *mut ::core::ffi::c_void,
        ldb: i64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnPotrs)(
            handle,
            params,
            uplo,
            n,
            nrhs,
            dataTypeA,
            A,
            lda,
            dataTypeB,
            B,
            ldb,
            info,
        )
    }
    pub unsafe fn cusolverDnSBgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSBgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnSBgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSBgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnSBgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSBgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnSBgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSBgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnSHgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSHgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnSHgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSHgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnSHgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSHgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnSHgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSHgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnSSgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSSgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnSSgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSSgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnSSgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSSgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnSSgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSSgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnSXgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSXgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnSXgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSXgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnSXgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSXgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnSXgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut f32,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut f32,
        lddb: cusolver_int_t,
        dX: *mut f32,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSXgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnSetAdvOptions(
        params: cusolverDnParams_t,
        function: cusolverDnFunction_t,
        algo: cusolverAlgMode_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSetAdvOptions)(params, function, algo)
    }
    #[cfg(
        any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cusolverDnSetDeterministicMode(
        handle: cusolverDnHandle_t,
        mode: cusolverDeterministicMode_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSetDeterministicMode)(handle, mode)
    }
    pub unsafe fn cusolverDnSetStream(
        handle: cusolverDnHandle_t,
        streamId: cudaStream_t,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSetStream)(handle, streamId)
    }
    pub unsafe fn cusolverDnSgebrd(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        D: *mut f32,
        E: *mut f32,
        TAUQ: *mut f32,
        TAUP: *mut f32,
        Work: *mut f32,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSgebrd)(
            handle,
            m,
            n,
            A,
            lda,
            D,
            E,
            TAUQ,
            TAUP,
            Work,
            Lwork,
            devInfo,
        )
    }
    pub unsafe fn cusolverDnSgebrd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSgebrd_bufferSize)(handle, m, n, Lwork)
    }
    pub unsafe fn cusolverDnSgeqrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        TAU: *mut f32,
        Workspace: *mut f32,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSgeqrf)(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    }
    pub unsafe fn cusolverDnSgeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSgeqrf_bufferSize)(handle, m, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnSgesvd(
        handle: cusolverDnHandle_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        S: *mut f32,
        U: *mut f32,
        ldu: ::core::ffi::c_int,
        VT: *mut f32,
        ldvt: ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        rwork: *mut f32,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSgesvd)(
            handle,
            jobu,
            jobvt,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            VT,
            ldvt,
            work,
            lwork,
            rwork,
            info,
        )
    }
    pub unsafe fn cusolverDnSgesvd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSgesvd_bufferSize)(handle, m, n, lwork)
    }
    pub unsafe fn cusolverDnSgesvdaStridedBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const f32,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *mut f32,
        strideS: ::core::ffi::c_longlong,
        d_U: *mut f32,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *mut f32,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        d_work: *mut f32,
        lwork: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        h_R_nrmF: *mut f64,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSgesvdaStridedBatched)(
            handle,
            jobz,
            rank,
            m,
            n,
            d_A,
            lda,
            strideA,
            d_S,
            strideS,
            d_U,
            ldu,
            strideU,
            d_V,
            ldv,
            strideV,
            d_work,
            lwork,
            d_info,
            h_R_nrmF,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnSgesvdaStridedBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const f32,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *const f32,
        strideS: ::core::ffi::c_longlong,
        d_U: *const f32,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *const f32,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        lwork: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSgesvdaStridedBatched_bufferSize)(
            handle,
            jobz,
            rank,
            m,
            n,
            d_A,
            lda,
            strideA,
            d_S,
            strideS,
            d_U,
            ldu,
            strideU,
            d_V,
            ldv,
            strideV,
            lwork,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnSgesvdj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        S: *mut f32,
        U: *mut f32,
        ldu: ::core::ffi::c_int,
        V: *mut f32,
        ldv: ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSgesvdj)(
            handle,
            jobz,
            econ,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            work,
            lwork,
            info,
            params,
        )
    }
    pub unsafe fn cusolverDnSgesvdjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        S: *mut f32,
        U: *mut f32,
        ldu: ::core::ffi::c_int,
        V: *mut f32,
        ldv: ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSgesvdjBatched)(
            handle,
            jobz,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            work,
            lwork,
            info,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnSgesvdjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        S: *const f32,
        U: *const f32,
        ldu: ::core::ffi::c_int,
        V: *const f32,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSgesvdjBatched_bufferSize)(
            handle,
            jobz,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            lwork,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnSgesvdj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        S: *const f32,
        U: *const f32,
        ldu: ::core::ffi::c_int,
        V: *const f32,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSgesvdj_bufferSize)(
            handle,
            jobz,
            econ,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            lwork,
            params,
        )
    }
    pub unsafe fn cusolverDnSgetrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        Workspace: *mut f32,
        devIpiv: *mut ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSgetrf)(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    }
    pub unsafe fn cusolverDnSgetrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSgetrf_bufferSize)(handle, m, n, A, lda, Lwork)
    }
    pub unsafe fn cusolverDnSgetrs(
        handle: cusolverDnHandle_t,
        trans: cublasOperation_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        B: *mut f32,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSgetrs)(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    }
    pub unsafe fn cusolverDnSlaswp(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        k1: ::core::ffi::c_int,
        k2: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        incx: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSlaswp)(handle, n, A, lda, k1, k2, devIpiv, incx)
    }
    pub unsafe fn cusolverDnSlauum(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSlauum)(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    pub unsafe fn cusolverDnSlauum_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSlauum_bufferSize)(handle, uplo, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnSorgbr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSorgbr)(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnSorgbr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSorgbr_bufferSize)(handle, side, m, n, k, A, lda, tau, lwork)
    }
    pub unsafe fn cusolverDnSorgqr(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSorgqr)(handle, m, n, k, A, lda, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnSorgqr_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSorgqr_bufferSize)(handle, m, n, k, A, lda, tau, lwork)
    }
    pub unsafe fn cusolverDnSorgtr(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSorgtr)(handle, uplo, n, A, lda, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnSorgtr_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSorgtr_bufferSize)(handle, uplo, n, A, lda, tau, lwork)
    }
    pub unsafe fn cusolverDnSormqr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        C: *mut f32,
        ldc: ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSormqr)(
            handle,
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            tau,
            C,
            ldc,
            work,
            lwork,
            devInfo,
        )
    }
    pub unsafe fn cusolverDnSormqr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        C: *const f32,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSormqr_bufferSize)(
            handle,
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            tau,
            C,
            ldc,
            lwork,
        )
    }
    pub unsafe fn cusolverDnSormtr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        tau: *mut f32,
        C: *mut f32,
        ldc: ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSormtr)(
            handle,
            side,
            uplo,
            trans,
            m,
            n,
            A,
            lda,
            tau,
            C,
            ldc,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnSormtr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        tau: *const f32,
        C: *const f32,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSormtr_bufferSize)(
            handle,
            side,
            uplo,
            trans,
            m,
            n,
            A,
            lda,
            tau,
            C,
            ldc,
            lwork,
        )
    }
    pub unsafe fn cusolverDnSpotrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        Workspace: *mut f32,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSpotrf)(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    }
    pub unsafe fn cusolverDnSpotrfBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        Aarray: *mut *mut f32,
        lda: ::core::ffi::c_int,
        infoArray: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSpotrfBatched)(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    }
    pub unsafe fn cusolverDnSpotrf_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSpotrf_bufferSize)(handle, uplo, n, A, lda, Lwork)
    }
    pub unsafe fn cusolverDnSpotri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSpotri)(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    pub unsafe fn cusolverDnSpotri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSpotri_bufferSize)(handle, uplo, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnSpotrs(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        B: *mut f32,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSpotrs)(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    }
    pub unsafe fn cusolverDnSpotrsBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *mut *mut f32,
        lda: ::core::ffi::c_int,
        B: *mut *mut f32,
        ldb: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSpotrsBatched)(
            handle,
            uplo,
            n,
            nrhs,
            A,
            lda,
            B,
            ldb,
            d_info,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnSsyevd(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSsyevd)(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    }
    pub unsafe fn cusolverDnSsyevd_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSsyevd_bufferSize)(handle, jobz, uplo, n, A, lda, W, lwork)
    }
    pub unsafe fn cusolverDnSsyevdx(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSsyevdx)(
            handle,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnSsyevdx_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSsyevdx_bufferSize)(
            handle,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            lwork,
        )
    }
    pub unsafe fn cusolverDnSsyevj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSsyevj)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            work,
            lwork,
            info,
            params,
        )
    }
    pub unsafe fn cusolverDnSsyevjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSsyevjBatched)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            work,
            lwork,
            info,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnSsyevjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSsyevjBatched_bufferSize)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            lwork,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnSsyevj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSsyevj_bufferSize)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            lwork,
            params,
        )
    }
    pub unsafe fn cusolverDnSsygvd(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        B: *mut f32,
        ldb: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSsygvd)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnSsygvd_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        B: *const f32,
        ldb: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSsygvd_bufferSize)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            lwork,
        )
    }
    pub unsafe fn cusolverDnSsygvdx(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        B: *mut f32,
        ldb: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSsygvdx)(
            handle,
            itype,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnSsygvdx_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        B: *const f32,
        ldb: ::core::ffi::c_int,
        vl: f32,
        vu: f32,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSsygvdx_bufferSize)(
            handle,
            itype,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            lwork,
        )
    }
    pub unsafe fn cusolverDnSsygvj(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        B: *mut f32,
        ldb: ::core::ffi::c_int,
        W: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSsygvj)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            work,
            lwork,
            info,
            params,
        )
    }
    pub unsafe fn cusolverDnSsygvj_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        B: *const f32,
        ldb: ::core::ffi::c_int,
        W: *const f32,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSsygvj_bufferSize)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            lwork,
            params,
        )
    }
    pub unsafe fn cusolverDnSsytrd(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        d: *mut f32,
        e: *mut f32,
        tau: *mut f32,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSsytrd)(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnSsytrd_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const f32,
        lda: ::core::ffi::c_int,
        d: *const f32,
        e: *const f32,
        tau: *const f32,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSsytrd_bufferSize)(handle, uplo, n, A, lda, d, e, tau, lwork)
    }
    pub unsafe fn cusolverDnSsytrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        ipiv: *mut ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSsytrf)(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    pub unsafe fn cusolverDnSsytrf_bufferSize(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSsytrf_bufferSize)(handle, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnSsytri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        work: *mut f32,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSsytri)(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    pub unsafe fn cusolverDnSsytri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut f32,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnSsytri_bufferSize)(handle, uplo, n, A, lda, ipiv, lwork)
    }
    pub unsafe fn cusolverDnSyevd(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *mut ::core::ffi::c_void,
        computeType: cudaDataType,
        pBuffer: *mut ::core::ffi::c_void,
        workspaceInBytes: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSyevd)(
            handle,
            params,
            jobz,
            uplo,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeW,
            W,
            computeType,
            pBuffer,
            workspaceInBytes,
            info,
        )
    }
    pub unsafe fn cusolverDnSyevd_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *const ::core::ffi::c_void,
        computeType: cudaDataType,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSyevd_bufferSize)(
            handle,
            params,
            jobz,
            uplo,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeW,
            W,
            computeType,
            workspaceInBytes,
        )
    }
    pub unsafe fn cusolverDnSyevdx(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        vl: *mut ::core::ffi::c_void,
        vu: *mut ::core::ffi::c_void,
        il: i64,
        iu: i64,
        meig64: *mut i64,
        dataTypeW: cudaDataType,
        W: *mut ::core::ffi::c_void,
        computeType: cudaDataType,
        pBuffer: *mut ::core::ffi::c_void,
        workspaceInBytes: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSyevdx)(
            handle,
            params,
            jobz,
            range,
            uplo,
            n,
            dataTypeA,
            A,
            lda,
            vl,
            vu,
            il,
            iu,
            meig64,
            dataTypeW,
            W,
            computeType,
            pBuffer,
            workspaceInBytes,
            info,
        )
    }
    pub unsafe fn cusolverDnSyevdx_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        vl: *mut ::core::ffi::c_void,
        vu: *mut ::core::ffi::c_void,
        il: i64,
        iu: i64,
        h_meig: *mut i64,
        dataTypeW: cudaDataType,
        W: *const ::core::ffi::c_void,
        computeType: cudaDataType,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnSyevdx_bufferSize)(
            handle,
            params,
            jobz,
            range,
            uplo,
            n,
            dataTypeA,
            A,
            lda,
            vl,
            vu,
            il,
            iu,
            h_meig,
            dataTypeW,
            W,
            computeType,
            workspaceInBytes,
        )
    }
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
    pub unsafe fn cusolverDnXgeev(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobvl: cusolverEigMode_t,
        jobvr: cusolverEigMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *mut ::core::ffi::c_void,
        dataTypeVL: cudaDataType,
        VL: *mut ::core::ffi::c_void,
        ldvl: i64,
        dataTypeVR: cudaDataType,
        VR: *mut ::core::ffi::c_void,
        ldvr: i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXgeev)(
            handle,
            params,
            jobvl,
            jobvr,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeW,
            W,
            dataTypeVL,
            VL,
            ldvl,
            dataTypeVR,
            VR,
            ldvr,
            computeType,
            bufferOnDevice,
            workspaceInBytesOnDevice,
            bufferOnHost,
            workspaceInBytesOnHost,
            info,
        )
    }
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
    pub unsafe fn cusolverDnXgeev_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobvl: cusolverEigMode_t,
        jobvr: cusolverEigMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *const ::core::ffi::c_void,
        dataTypeVL: cudaDataType,
        VL: *const ::core::ffi::c_void,
        ldvl: i64,
        dataTypeVR: cudaDataType,
        VR: *const ::core::ffi::c_void,
        ldvr: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXgeev_bufferSize)(
            handle,
            params,
            jobvl,
            jobvr,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeW,
            W,
            dataTypeVL,
            VL,
            ldvl,
            dataTypeVR,
            VR,
            ldvr,
            computeType,
            workspaceInBytesOnDevice,
            workspaceInBytesOnHost,
        )
    }
    pub unsafe fn cusolverDnXgeqrf(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeTau: cudaDataType,
        tau: *mut ::core::ffi::c_void,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXgeqrf)(
            handle,
            params,
            m,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeTau,
            tau,
            computeType,
            bufferOnDevice,
            workspaceInBytesOnDevice,
            bufferOnHost,
            workspaceInBytesOnHost,
            info,
        )
    }
    pub unsafe fn cusolverDnXgeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeTau: cudaDataType,
        tau: *const ::core::ffi::c_void,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXgeqrf_bufferSize)(
            handle,
            params,
            m,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeTau,
            tau,
            computeType,
            workspaceInBytesOnDevice,
            workspaceInBytesOnHost,
        )
    }
    pub unsafe fn cusolverDnXgesvd(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeS: cudaDataType,
        S: *mut ::core::ffi::c_void,
        dataTypeU: cudaDataType,
        U: *mut ::core::ffi::c_void,
        ldu: i64,
        dataTypeVT: cudaDataType,
        VT: *mut ::core::ffi::c_void,
        ldvt: i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXgesvd)(
            handle,
            params,
            jobu,
            jobvt,
            m,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeS,
            S,
            dataTypeU,
            U,
            ldu,
            dataTypeVT,
            VT,
            ldvt,
            computeType,
            bufferOnDevice,
            workspaceInBytesOnDevice,
            bufferOnHost,
            workspaceInBytesOnHost,
            info,
        )
    }
    pub unsafe fn cusolverDnXgesvd_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeS: cudaDataType,
        S: *const ::core::ffi::c_void,
        dataTypeU: cudaDataType,
        U: *const ::core::ffi::c_void,
        ldu: i64,
        dataTypeVT: cudaDataType,
        VT: *const ::core::ffi::c_void,
        ldvt: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXgesvd_bufferSize)(
            handle,
            params,
            jobu,
            jobvt,
            m,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeS,
            S,
            dataTypeU,
            U,
            ldu,
            dataTypeVT,
            VT,
            ldvt,
            computeType,
            workspaceInBytesOnDevice,
            workspaceInBytesOnHost,
        )
    }
    pub unsafe fn cusolverDnXgesvdjGetResidual(
        handle: cusolverDnHandle_t,
        info: gesvdjInfo_t,
        residual: *mut f64,
    ) -> cusolverStatus_t {
        (culib().cusolverDnXgesvdjGetResidual)(handle, info, residual)
    }
    pub unsafe fn cusolverDnXgesvdjGetSweeps(
        handle: cusolverDnHandle_t,
        info: gesvdjInfo_t,
        executed_sweeps: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnXgesvdjGetSweeps)(handle, info, executed_sweeps)
    }
    pub unsafe fn cusolverDnXgesvdjSetMaxSweeps(
        info: gesvdjInfo_t,
        max_sweeps: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnXgesvdjSetMaxSweeps)(info, max_sweeps)
    }
    pub unsafe fn cusolverDnXgesvdjSetSortEig(
        info: gesvdjInfo_t,
        sort_svd: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnXgesvdjSetSortEig)(info, sort_svd)
    }
    pub unsafe fn cusolverDnXgesvdjSetTolerance(
        info: gesvdjInfo_t,
        tolerance: f64,
    ) -> cusolverStatus_t {
        (culib().cusolverDnXgesvdjSetTolerance)(info, tolerance)
    }
    pub unsafe fn cusolverDnXgesvdp(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeS: cudaDataType,
        S: *mut ::core::ffi::c_void,
        dataTypeU: cudaDataType,
        U: *mut ::core::ffi::c_void,
        ldu: i64,
        dataTypeV: cudaDataType,
        V: *mut ::core::ffi::c_void,
        ldv: i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        d_info: *mut ::core::ffi::c_int,
        h_err_sigma: *mut f64,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXgesvdp)(
            handle,
            params,
            jobz,
            econ,
            m,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeS,
            S,
            dataTypeU,
            U,
            ldu,
            dataTypeV,
            V,
            ldv,
            computeType,
            bufferOnDevice,
            workspaceInBytesOnDevice,
            bufferOnHost,
            workspaceInBytesOnHost,
            d_info,
            h_err_sigma,
        )
    }
    pub unsafe fn cusolverDnXgesvdp_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeS: cudaDataType,
        S: *const ::core::ffi::c_void,
        dataTypeU: cudaDataType,
        U: *const ::core::ffi::c_void,
        ldu: i64,
        dataTypeV: cudaDataType,
        V: *const ::core::ffi::c_void,
        ldv: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXgesvdp_bufferSize)(
            handle,
            params,
            jobz,
            econ,
            m,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeS,
            S,
            dataTypeU,
            U,
            ldu,
            dataTypeV,
            V,
            ldv,
            computeType,
            workspaceInBytesOnDevice,
            workspaceInBytesOnHost,
        )
    }
    pub unsafe fn cusolverDnXgesvdr(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobu: ::core::ffi::c_schar,
        jobv: ::core::ffi::c_schar,
        m: i64,
        n: i64,
        k: i64,
        p: i64,
        niters: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeSrand: cudaDataType,
        Srand: *mut ::core::ffi::c_void,
        dataTypeUrand: cudaDataType,
        Urand: *mut ::core::ffi::c_void,
        ldUrand: i64,
        dataTypeVrand: cudaDataType,
        Vrand: *mut ::core::ffi::c_void,
        ldVrand: i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        d_info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXgesvdr)(
            handle,
            params,
            jobu,
            jobv,
            m,
            n,
            k,
            p,
            niters,
            dataTypeA,
            A,
            lda,
            dataTypeSrand,
            Srand,
            dataTypeUrand,
            Urand,
            ldUrand,
            dataTypeVrand,
            Vrand,
            ldVrand,
            computeType,
            bufferOnDevice,
            workspaceInBytesOnDevice,
            bufferOnHost,
            workspaceInBytesOnHost,
            d_info,
        )
    }
    pub unsafe fn cusolverDnXgesvdr_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobu: ::core::ffi::c_schar,
        jobv: ::core::ffi::c_schar,
        m: i64,
        n: i64,
        k: i64,
        p: i64,
        niters: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeSrand: cudaDataType,
        Srand: *const ::core::ffi::c_void,
        dataTypeUrand: cudaDataType,
        Urand: *const ::core::ffi::c_void,
        ldUrand: i64,
        dataTypeVrand: cudaDataType,
        Vrand: *const ::core::ffi::c_void,
        ldVrand: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXgesvdr_bufferSize)(
            handle,
            params,
            jobu,
            jobv,
            m,
            n,
            k,
            p,
            niters,
            dataTypeA,
            A,
            lda,
            dataTypeSrand,
            Srand,
            dataTypeUrand,
            Urand,
            ldUrand,
            dataTypeVrand,
            Vrand,
            ldVrand,
            computeType,
            workspaceInBytesOnDevice,
            workspaceInBytesOnHost,
        )
    }
    pub unsafe fn cusolverDnXgetrf(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        ipiv: *mut i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXgetrf)(
            handle,
            params,
            m,
            n,
            dataTypeA,
            A,
            lda,
            ipiv,
            computeType,
            bufferOnDevice,
            workspaceInBytesOnDevice,
            bufferOnHost,
            workspaceInBytesOnHost,
            info,
        )
    }
    pub unsafe fn cusolverDnXgetrf_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        m: i64,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXgetrf_bufferSize)(
            handle,
            params,
            m,
            n,
            dataTypeA,
            A,
            lda,
            computeType,
            workspaceInBytesOnDevice,
            workspaceInBytesOnHost,
        )
    }
    pub unsafe fn cusolverDnXgetrs(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        trans: cublasOperation_t,
        n: i64,
        nrhs: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        ipiv: *const i64,
        dataTypeB: cudaDataType,
        B: *mut ::core::ffi::c_void,
        ldb: i64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXgetrs)(
            handle,
            params,
            trans,
            n,
            nrhs,
            dataTypeA,
            A,
            lda,
            ipiv,
            dataTypeB,
            B,
            ldb,
            info,
        )
    }
    #[cfg(any(feature = "cuda-12040"))]
    pub unsafe fn cusolverDnXlarft(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        direct: cusolverDirectMode_t,
        storev: cusolverStorevMode_t,
        N: i64,
        K: i64,
        dataTypeV: cudaDataType,
        d_V: *const ::core::ffi::c_void,
        ldv: i64,
        dataTypeTau: cudaDataType,
        d_tau: *const ::core::ffi::c_void,
        dataTypeT: cudaDataType,
        d_T: *mut ::core::ffi::c_void,
        ldt: i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXlarft)(
            handle,
            params,
            direct,
            storev,
            N,
            K,
            dataTypeV,
            d_V,
            ldv,
            dataTypeTau,
            d_tau,
            dataTypeT,
            d_T,
            ldt,
            computeType,
            bufferOnDevice,
            workspaceInBytesOnDevice,
            bufferOnHost,
            workspaceInBytesOnHost,
        )
    }
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub unsafe fn cusolverDnXlarft(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        direct: cusolverDirectMode_t,
        storev: cusolverStorevMode_t,
        n: i64,
        k: i64,
        dataTypeV: cudaDataType,
        V: *const ::core::ffi::c_void,
        ldv: i64,
        dataTypeTau: cudaDataType,
        tau: *const ::core::ffi::c_void,
        dataTypeT: cudaDataType,
        T: *mut ::core::ffi::c_void,
        ldt: i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXlarft)(
            handle,
            params,
            direct,
            storev,
            n,
            k,
            dataTypeV,
            V,
            ldv,
            dataTypeTau,
            tau,
            dataTypeT,
            T,
            ldt,
            computeType,
            bufferOnDevice,
            workspaceInBytesOnDevice,
            bufferOnHost,
            workspaceInBytesOnHost,
        )
    }
    #[cfg(any(feature = "cuda-12040"))]
    pub unsafe fn cusolverDnXlarft_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        direct: cusolverDirectMode_t,
        storev: cusolverStorevMode_t,
        N: i64,
        K: i64,
        dataTypeV: cudaDataType,
        d_V: *const ::core::ffi::c_void,
        ldv: i64,
        dataTypeTau: cudaDataType,
        d_tau: *const ::core::ffi::c_void,
        dataTypeT: cudaDataType,
        d_T: *mut ::core::ffi::c_void,
        ldt: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXlarft_bufferSize)(
            handle,
            params,
            direct,
            storev,
            N,
            K,
            dataTypeV,
            d_V,
            ldv,
            dataTypeTau,
            d_tau,
            dataTypeT,
            d_T,
            ldt,
            computeType,
            workspaceInBytesOnDevice,
            workspaceInBytesOnHost,
        )
    }
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub unsafe fn cusolverDnXlarft_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        direct: cusolverDirectMode_t,
        storev: cusolverStorevMode_t,
        n: i64,
        k: i64,
        dataTypeV: cudaDataType,
        V: *const ::core::ffi::c_void,
        ldv: i64,
        dataTypeTau: cudaDataType,
        tau: *const ::core::ffi::c_void,
        dataTypeT: cudaDataType,
        T: *mut ::core::ffi::c_void,
        ldt: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXlarft_bufferSize)(
            handle,
            params,
            direct,
            storev,
            n,
            k,
            dataTypeV,
            V,
            ldv,
            dataTypeTau,
            tau,
            dataTypeT,
            T,
            ldt,
            computeType,
            workspaceInBytesOnDevice,
            workspaceInBytesOnHost,
        )
    }
    pub unsafe fn cusolverDnXpotrf(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXpotrf)(
            handle,
            params,
            uplo,
            n,
            dataTypeA,
            A,
            lda,
            computeType,
            bufferOnDevice,
            workspaceInBytesOnDevice,
            bufferOnHost,
            workspaceInBytesOnHost,
            info,
        )
    }
    pub unsafe fn cusolverDnXpotrf_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXpotrf_bufferSize)(
            handle,
            params,
            uplo,
            n,
            dataTypeA,
            A,
            lda,
            computeType,
            workspaceInBytesOnDevice,
            workspaceInBytesOnHost,
        )
    }
    pub unsafe fn cusolverDnXpotrs(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        uplo: cublasFillMode_t,
        n: i64,
        nrhs: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeB: cudaDataType,
        B: *mut ::core::ffi::c_void,
        ldb: i64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXpotrs)(
            handle,
            params,
            uplo,
            n,
            nrhs,
            dataTypeA,
            A,
            lda,
            dataTypeB,
            B,
            ldb,
            info,
        )
    }
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
    pub unsafe fn cusolverDnXsyevBatched(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *mut ::core::ffi::c_void,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
        batchSize: i64,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXsyevBatched)(
            handle,
            params,
            jobz,
            uplo,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeW,
            W,
            computeType,
            bufferOnDevice,
            workspaceInBytesOnDevice,
            bufferOnHost,
            workspaceInBytesOnHost,
            info,
            batchSize,
        )
    }
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
    pub unsafe fn cusolverDnXsyevBatched_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *const ::core::ffi::c_void,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
        batchSize: i64,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXsyevBatched_bufferSize)(
            handle,
            params,
            jobz,
            uplo,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeW,
            W,
            computeType,
            workspaceInBytesOnDevice,
            workspaceInBytesOnHost,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnXsyevd(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *mut ::core::ffi::c_void,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXsyevd)(
            handle,
            params,
            jobz,
            uplo,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeW,
            W,
            computeType,
            bufferOnDevice,
            workspaceInBytesOnDevice,
            bufferOnHost,
            workspaceInBytesOnHost,
            info,
        )
    }
    pub unsafe fn cusolverDnXsyevd_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        dataTypeW: cudaDataType,
        W: *const ::core::ffi::c_void,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXsyevd_bufferSize)(
            handle,
            params,
            jobz,
            uplo,
            n,
            dataTypeA,
            A,
            lda,
            dataTypeW,
            W,
            computeType,
            workspaceInBytesOnDevice,
            workspaceInBytesOnHost,
        )
    }
    pub unsafe fn cusolverDnXsyevdx(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        vl: *mut ::core::ffi::c_void,
        vu: *mut ::core::ffi::c_void,
        il: i64,
        iu: i64,
        meig64: *mut i64,
        dataTypeW: cudaDataType,
        W: *mut ::core::ffi::c_void,
        computeType: cudaDataType,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXsyevdx)(
            handle,
            params,
            jobz,
            range,
            uplo,
            n,
            dataTypeA,
            A,
            lda,
            vl,
            vu,
            il,
            iu,
            meig64,
            dataTypeW,
            W,
            computeType,
            bufferOnDevice,
            workspaceInBytesOnDevice,
            bufferOnHost,
            workspaceInBytesOnHost,
            info,
        )
    }
    pub unsafe fn cusolverDnXsyevdx_bufferSize(
        handle: cusolverDnHandle_t,
        params: cusolverDnParams_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        vl: *mut ::core::ffi::c_void,
        vu: *mut ::core::ffi::c_void,
        il: i64,
        iu: i64,
        h_meig: *mut i64,
        dataTypeW: cudaDataType,
        W: *const ::core::ffi::c_void,
        computeType: cudaDataType,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXsyevdx_bufferSize)(
            handle,
            params,
            jobz,
            range,
            uplo,
            n,
            dataTypeA,
            A,
            lda,
            vl,
            vu,
            il,
            iu,
            h_meig,
            dataTypeW,
            W,
            computeType,
            workspaceInBytesOnDevice,
            workspaceInBytesOnHost,
        )
    }
    pub unsafe fn cusolverDnXsyevjGetResidual(
        handle: cusolverDnHandle_t,
        info: syevjInfo_t,
        residual: *mut f64,
    ) -> cusolverStatus_t {
        (culib().cusolverDnXsyevjGetResidual)(handle, info, residual)
    }
    pub unsafe fn cusolverDnXsyevjGetSweeps(
        handle: cusolverDnHandle_t,
        info: syevjInfo_t,
        executed_sweeps: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnXsyevjGetSweeps)(handle, info, executed_sweeps)
    }
    pub unsafe fn cusolverDnXsyevjSetMaxSweeps(
        info: syevjInfo_t,
        max_sweeps: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnXsyevjSetMaxSweeps)(info, max_sweeps)
    }
    pub unsafe fn cusolverDnXsyevjSetSortEig(
        info: syevjInfo_t,
        sort_eig: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnXsyevjSetSortEig)(info, sort_eig)
    }
    pub unsafe fn cusolverDnXsyevjSetTolerance(
        info: syevjInfo_t,
        tolerance: f64,
    ) -> cusolverStatus_t {
        (culib().cusolverDnXsyevjSetTolerance)(info, tolerance)
    }
    pub unsafe fn cusolverDnXsytrs(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: i64,
        nrhs: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        ipiv: *const i64,
        dataTypeB: cudaDataType,
        B: *mut ::core::ffi::c_void,
        ldb: i64,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXsytrs)(
            handle,
            uplo,
            n,
            nrhs,
            dataTypeA,
            A,
            lda,
            ipiv,
            dataTypeB,
            B,
            ldb,
            bufferOnDevice,
            workspaceInBytesOnDevice,
            bufferOnHost,
            workspaceInBytesOnHost,
            info,
        )
    }
    pub unsafe fn cusolverDnXsytrs_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: i64,
        nrhs: i64,
        dataTypeA: cudaDataType,
        A: *const ::core::ffi::c_void,
        lda: i64,
        ipiv: *const i64,
        dataTypeB: cudaDataType,
        B: *mut ::core::ffi::c_void,
        ldb: i64,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXsytrs_bufferSize)(
            handle,
            uplo,
            n,
            nrhs,
            dataTypeA,
            A,
            lda,
            ipiv,
            dataTypeB,
            B,
            ldb,
            workspaceInBytesOnDevice,
            workspaceInBytesOnHost,
        )
    }
    pub unsafe fn cusolverDnXtrtri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        diag: cublasDiagType_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        bufferOnDevice: *mut ::core::ffi::c_void,
        workspaceInBytesOnDevice: usize,
        bufferOnHost: *mut ::core::ffi::c_void,
        workspaceInBytesOnHost: usize,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXtrtri)(
            handle,
            uplo,
            diag,
            n,
            dataTypeA,
            A,
            lda,
            bufferOnDevice,
            workspaceInBytesOnDevice,
            bufferOnHost,
            workspaceInBytesOnHost,
            devInfo,
        )
    }
    pub unsafe fn cusolverDnXtrtri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        diag: cublasDiagType_t,
        n: i64,
        dataTypeA: cudaDataType,
        A: *mut ::core::ffi::c_void,
        lda: i64,
        workspaceInBytesOnDevice: *mut usize,
        workspaceInBytesOnHost: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnXtrtri_bufferSize)(
            handle,
            uplo,
            diag,
            n,
            dataTypeA,
            A,
            lda,
            workspaceInBytesOnDevice,
            workspaceInBytesOnHost,
        )
    }
    pub unsafe fn cusolverDnZCgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZCgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnZCgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZCgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnZCgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZCgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnZCgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZCgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnZEgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZEgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnZEgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZEgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnZEgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZEgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnZEgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZEgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnZKgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZKgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnZKgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZKgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnZKgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZKgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnZKgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZKgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnZYgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZYgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnZYgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZYgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnZYgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZYgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnZYgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZYgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnZZgels(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZZgels)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnZZgels_bufferSize(
        handle: cusolverDnHandle_t,
        m: cusolver_int_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZZgels_bufferSize)(
            handle,
            m,
            n,
            nrhs,
            dA,
            ldda,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnZZgesv(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: usize,
        iter: *mut cusolver_int_t,
        d_info: *mut cusolver_int_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZZgesv)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
            iter,
            d_info,
        )
    }
    pub unsafe fn cusolverDnZZgesv_bufferSize(
        handle: cusolverDnHandle_t,
        n: cusolver_int_t,
        nrhs: cusolver_int_t,
        dA: *mut cuDoubleComplex,
        ldda: cusolver_int_t,
        dipiv: *mut cusolver_int_t,
        dB: *mut cuDoubleComplex,
        lddb: cusolver_int_t,
        dX: *mut cuDoubleComplex,
        lddx: cusolver_int_t,
        dWorkspace: *mut ::core::ffi::c_void,
        lwork_bytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZZgesv_bufferSize)(
            handle,
            n,
            nrhs,
            dA,
            ldda,
            dipiv,
            dB,
            lddb,
            dX,
            lddx,
            dWorkspace,
            lwork_bytes,
        )
    }
    pub unsafe fn cusolverDnZgebrd(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        D: *mut f64,
        E: *mut f64,
        TAUQ: *mut cuDoubleComplex,
        TAUP: *mut cuDoubleComplex,
        Work: *mut cuDoubleComplex,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZgebrd)(
            handle,
            m,
            n,
            A,
            lda,
            D,
            E,
            TAUQ,
            TAUP,
            Work,
            Lwork,
            devInfo,
        )
    }
    pub unsafe fn cusolverDnZgebrd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZgebrd_bufferSize)(handle, m, n, Lwork)
    }
    pub unsafe fn cusolverDnZgeqrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        TAU: *mut cuDoubleComplex,
        Workspace: *mut cuDoubleComplex,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZgeqrf)(handle, m, n, A, lda, TAU, Workspace, Lwork, devInfo)
    }
    pub unsafe fn cusolverDnZgeqrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZgeqrf_bufferSize)(handle, m, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnZgesvd(
        handle: cusolverDnHandle_t,
        jobu: ::core::ffi::c_schar,
        jobvt: ::core::ffi::c_schar,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        S: *mut f64,
        U: *mut cuDoubleComplex,
        ldu: ::core::ffi::c_int,
        VT: *mut cuDoubleComplex,
        ldvt: ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        rwork: *mut f64,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZgesvd)(
            handle,
            jobu,
            jobvt,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            VT,
            ldvt,
            work,
            lwork,
            rwork,
            info,
        )
    }
    pub unsafe fn cusolverDnZgesvd_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZgesvd_bufferSize)(handle, m, n, lwork)
    }
    pub unsafe fn cusolverDnZgesvdaStridedBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *mut f64,
        strideS: ::core::ffi::c_longlong,
        d_U: *mut cuDoubleComplex,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *mut cuDoubleComplex,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        d_work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        h_R_nrmF: *mut f64,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZgesvdaStridedBatched)(
            handle,
            jobz,
            rank,
            m,
            n,
            d_A,
            lda,
            strideA,
            d_S,
            strideS,
            d_U,
            ldu,
            strideU,
            d_V,
            ldv,
            strideV,
            d_work,
            lwork,
            d_info,
            h_R_nrmF,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnZgesvdaStridedBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        rank: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        d_A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        strideA: ::core::ffi::c_longlong,
        d_S: *const f64,
        strideS: ::core::ffi::c_longlong,
        d_U: *const cuDoubleComplex,
        ldu: ::core::ffi::c_int,
        strideU: ::core::ffi::c_longlong,
        d_V: *const cuDoubleComplex,
        ldv: ::core::ffi::c_int,
        strideV: ::core::ffi::c_longlong,
        lwork: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZgesvdaStridedBatched_bufferSize)(
            handle,
            jobz,
            rank,
            m,
            n,
            d_A,
            lda,
            strideA,
            d_S,
            strideS,
            d_U,
            ldu,
            strideU,
            d_V,
            ldv,
            strideV,
            lwork,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnZgesvdj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        S: *mut f64,
        U: *mut cuDoubleComplex,
        ldu: ::core::ffi::c_int,
        V: *mut cuDoubleComplex,
        ldv: ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZgesvdj)(
            handle,
            jobz,
            econ,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            work,
            lwork,
            info,
            params,
        )
    }
    pub unsafe fn cusolverDnZgesvdjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        S: *mut f64,
        U: *mut cuDoubleComplex,
        ldu: ::core::ffi::c_int,
        V: *mut cuDoubleComplex,
        ldv: ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZgesvdjBatched)(
            handle,
            jobz,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            work,
            lwork,
            info,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnZgesvdjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        S: *const f64,
        U: *const cuDoubleComplex,
        ldu: ::core::ffi::c_int,
        V: *const cuDoubleComplex,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZgesvdjBatched_bufferSize)(
            handle,
            jobz,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            lwork,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnZgesvdj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        econ: ::core::ffi::c_int,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        S: *const f64,
        U: *const cuDoubleComplex,
        ldu: ::core::ffi::c_int,
        V: *const cuDoubleComplex,
        ldv: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
        params: gesvdjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZgesvdj_bufferSize)(
            handle,
            jobz,
            econ,
            m,
            n,
            A,
            lda,
            S,
            U,
            ldu,
            V,
            ldv,
            lwork,
            params,
        )
    }
    pub unsafe fn cusolverDnZgetrf(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        Workspace: *mut cuDoubleComplex,
        devIpiv: *mut ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZgetrf)(handle, m, n, A, lda, Workspace, devIpiv, devInfo)
    }
    pub unsafe fn cusolverDnZgetrf_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZgetrf_bufferSize)(handle, m, n, A, lda, Lwork)
    }
    pub unsafe fn cusolverDnZgetrs(
        handle: cusolverDnHandle_t,
        trans: cublasOperation_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        B: *mut cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZgetrs)(handle, trans, n, nrhs, A, lda, devIpiv, B, ldb, devInfo)
    }
    pub unsafe fn cusolverDnZheevd(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZheevd)(handle, jobz, uplo, n, A, lda, W, work, lwork, info)
    }
    pub unsafe fn cusolverDnZheevd_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZheevd_bufferSize)(handle, jobz, uplo, n, A, lda, W, lwork)
    }
    pub unsafe fn cusolverDnZheevdx(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f64,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZheevdx)(
            handle,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnZheevdx_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZheevdx_bufferSize)(
            handle,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            lwork,
        )
    }
    pub unsafe fn cusolverDnZheevj(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZheevj)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            work,
            lwork,
            info,
            params,
        )
    }
    pub unsafe fn cusolverDnZheevjBatched(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZheevjBatched)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            work,
            lwork,
            info,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnZheevjBatched_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZheevjBatched_bufferSize)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            lwork,
            params,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnZheevj_bufferSize(
        handle: cusolverDnHandle_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZheevj_bufferSize)(
            handle,
            jobz,
            uplo,
            n,
            A,
            lda,
            W,
            lwork,
            params,
        )
    }
    pub unsafe fn cusolverDnZhegvd(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZhegvd)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnZhegvd_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *const cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZhegvd_bufferSize)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            lwork,
        )
    }
    pub unsafe fn cusolverDnZhegvdx(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *mut f64,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZhegvdx)(
            handle,
            itype,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnZhegvdx_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        range: cusolverEigRange_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *const cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        vl: f64,
        vu: f64,
        il: ::core::ffi::c_int,
        iu: ::core::ffi::c_int,
        meig: *mut ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZhegvdx_bufferSize)(
            handle,
            itype,
            jobz,
            range,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            vl,
            vu,
            il,
            iu,
            meig,
            W,
            lwork,
        )
    }
    pub unsafe fn cusolverDnZhegvj(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        W: *mut f64,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZhegvj)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            work,
            lwork,
            info,
            params,
        )
    }
    pub unsafe fn cusolverDnZhegvj_bufferSize(
        handle: cusolverDnHandle_t,
        itype: cusolverEigType_t,
        jobz: cusolverEigMode_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *const cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        W: *const f64,
        lwork: *mut ::core::ffi::c_int,
        params: syevjInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZhegvj_bufferSize)(
            handle,
            itype,
            jobz,
            uplo,
            n,
            A,
            lda,
            B,
            ldb,
            W,
            lwork,
            params,
        )
    }
    pub unsafe fn cusolverDnZhetrd(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        d: *mut f64,
        e: *mut f64,
        tau: *mut cuDoubleComplex,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZhetrd)(handle, uplo, n, A, lda, d, e, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnZhetrd_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        d: *const f64,
        e: *const f64,
        tau: *const cuDoubleComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZhetrd_bufferSize)(handle, uplo, n, A, lda, d, e, tau, lwork)
    }
    pub unsafe fn cusolverDnZlaswp(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        k1: ::core::ffi::c_int,
        k2: ::core::ffi::c_int,
        devIpiv: *const ::core::ffi::c_int,
        incx: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZlaswp)(handle, n, A, lda, k1, k2, devIpiv, incx)
    }
    pub unsafe fn cusolverDnZlauum(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZlauum)(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    pub unsafe fn cusolverDnZlauum_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZlauum_bufferSize)(handle, uplo, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnZpotrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        Workspace: *mut cuDoubleComplex,
        Lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZpotrf)(handle, uplo, n, A, lda, Workspace, Lwork, devInfo)
    }
    pub unsafe fn cusolverDnZpotrfBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        Aarray: *mut *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        infoArray: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZpotrfBatched)(handle, uplo, n, Aarray, lda, infoArray, batchSize)
    }
    pub unsafe fn cusolverDnZpotrf_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        Lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZpotrf_bufferSize)(handle, uplo, n, A, lda, Lwork)
    }
    pub unsafe fn cusolverDnZpotri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZpotri)(handle, uplo, n, A, lda, work, lwork, devInfo)
    }
    pub unsafe fn cusolverDnZpotri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZpotri_bufferSize)(handle, uplo, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnZpotrs(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *mut cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZpotrs)(handle, uplo, n, nrhs, A, lda, B, ldb, devInfo)
    }
    pub unsafe fn cusolverDnZpotrsBatched(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        A: *mut *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        B: *mut *mut cuDoubleComplex,
        ldb: ::core::ffi::c_int,
        d_info: *mut ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZpotrsBatched)(
            handle,
            uplo,
            n,
            nrhs,
            A,
            lda,
            B,
            ldb,
            d_info,
            batchSize,
        )
    }
    pub unsafe fn cusolverDnZsytrf(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        ipiv: *mut ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZsytrf)(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    pub unsafe fn cusolverDnZsytrf_bufferSize(
        handle: cusolverDnHandle_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZsytrf_bufferSize)(handle, n, A, lda, lwork)
    }
    pub unsafe fn cusolverDnZsytri(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZsytri)(handle, uplo, n, A, lda, ipiv, work, lwork, info)
    }
    pub unsafe fn cusolverDnZsytri_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        ipiv: *const ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZsytri_bufferSize)(handle, uplo, n, A, lda, ipiv, lwork)
    }
    pub unsafe fn cusolverDnZungbr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZungbr)(handle, side, m, n, k, A, lda, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnZungbr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZungbr_bufferSize)(handle, side, m, n, k, A, lda, tau, lwork)
    }
    pub unsafe fn cusolverDnZungqr(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZungqr)(handle, m, n, k, A, lda, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnZungqr_bufferSize(
        handle: cusolverDnHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZungqr_bufferSize)(handle, m, n, k, A, lda, tau, lwork)
    }
    pub unsafe fn cusolverDnZungtr(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZungtr)(handle, uplo, n, A, lda, tau, work, lwork, info)
    }
    pub unsafe fn cusolverDnZungtr_bufferSize(
        handle: cusolverDnHandle_t,
        uplo: cublasFillMode_t,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverDnZungtr_bufferSize)(handle, uplo, n, A, lda, tau, lwork)
    }
    pub unsafe fn cusolverDnZunmqr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        C: *mut cuDoubleComplex,
        ldc: ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        devInfo: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZunmqr)(
            handle,
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            tau,
            C,
            ldc,
            work,
            lwork,
            devInfo,
        )
    }
    pub unsafe fn cusolverDnZunmqr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        k: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        C: *const cuDoubleComplex,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZunmqr_bufferSize)(
            handle,
            side,
            trans,
            m,
            n,
            k,
            A,
            lda,
            tau,
            C,
            ldc,
            lwork,
        )
    }
    pub unsafe fn cusolverDnZunmtr(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *mut cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *mut cuDoubleComplex,
        C: *mut cuDoubleComplex,
        ldc: ::core::ffi::c_int,
        work: *mut cuDoubleComplex,
        lwork: ::core::ffi::c_int,
        info: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZunmtr)(
            handle,
            side,
            uplo,
            trans,
            m,
            n,
            A,
            lda,
            tau,
            C,
            ldc,
            work,
            lwork,
            info,
        )
    }
    pub unsafe fn cusolverDnZunmtr_bufferSize(
        handle: cusolverDnHandle_t,
        side: cublasSideMode_t,
        uplo: cublasFillMode_t,
        trans: cublasOperation_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        A: *const cuDoubleComplex,
        lda: ::core::ffi::c_int,
        tau: *const cuDoubleComplex,
        C: *const cuDoubleComplex,
        ldc: ::core::ffi::c_int,
        lwork: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverDnZunmtr_bufferSize)(
            handle,
            side,
            uplo,
            trans,
            m,
            n,
            A,
            lda,
            tau,
            C,
            ldc,
            lwork,
        )
    }
    pub unsafe fn cusolverGetProperty(
        type_: libraryPropertyType,
        value: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverGetProperty)(type_, value)
    }
    pub unsafe fn cusolverGetVersion(
        version: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverGetVersion)(version)
    }
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
    pub unsafe fn cusolverRfAccessBundledFactorsDevice(
        handle: cusolverRfHandle_t,
        nnzM: *mut ::core::ffi::c_int,
        Mp: *mut *mut ::core::ffi::c_int,
        Mi: *mut *mut ::core::ffi::c_int,
        Mx: *mut *mut f64,
    ) -> cusolverStatus_t {
        (culib().cusolverRfAccessBundledFactorsDevice)(handle, nnzM, Mp, Mi, Mx)
    }
    pub unsafe fn cusolverRfAnalyze(handle: cusolverRfHandle_t) -> cusolverStatus_t {
        (culib().cusolverRfAnalyze)(handle)
    }
    pub unsafe fn cusolverRfBatchAnalyze(
        handle: cusolverRfHandle_t,
    ) -> cusolverStatus_t {
        (culib().cusolverRfBatchAnalyze)(handle)
    }
    pub unsafe fn cusolverRfBatchRefactor(
        handle: cusolverRfHandle_t,
    ) -> cusolverStatus_t {
        (culib().cusolverRfBatchRefactor)(handle)
    }
    pub unsafe fn cusolverRfBatchResetValues(
        batchSize: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        csrRowPtrA: *mut ::core::ffi::c_int,
        csrColIndA: *mut ::core::ffi::c_int,
        csrValA_array: *mut *mut f64,
        P: *mut ::core::ffi::c_int,
        Q: *mut ::core::ffi::c_int,
        handle: cusolverRfHandle_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverRfBatchResetValues)(
            batchSize,
            n,
            nnzA,
            csrRowPtrA,
            csrColIndA,
            csrValA_array,
            P,
            Q,
            handle,
        )
    }
    pub unsafe fn cusolverRfBatchSetupHost(
        batchSize: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        h_csrRowPtrA: *mut ::core::ffi::c_int,
        h_csrColIndA: *mut ::core::ffi::c_int,
        h_csrValA_array: *mut *mut f64,
        nnzL: ::core::ffi::c_int,
        h_csrRowPtrL: *mut ::core::ffi::c_int,
        h_csrColIndL: *mut ::core::ffi::c_int,
        h_csrValL: *mut f64,
        nnzU: ::core::ffi::c_int,
        h_csrRowPtrU: *mut ::core::ffi::c_int,
        h_csrColIndU: *mut ::core::ffi::c_int,
        h_csrValU: *mut f64,
        h_P: *mut ::core::ffi::c_int,
        h_Q: *mut ::core::ffi::c_int,
        handle: cusolverRfHandle_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverRfBatchSetupHost)(
            batchSize,
            n,
            nnzA,
            h_csrRowPtrA,
            h_csrColIndA,
            h_csrValA_array,
            nnzL,
            h_csrRowPtrL,
            h_csrColIndL,
            h_csrValL,
            nnzU,
            h_csrRowPtrU,
            h_csrColIndU,
            h_csrValU,
            h_P,
            h_Q,
            handle,
        )
    }
    pub unsafe fn cusolverRfBatchSolve(
        handle: cusolverRfHandle_t,
        P: *mut ::core::ffi::c_int,
        Q: *mut ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        Temp: *mut f64,
        ldt: ::core::ffi::c_int,
        XF_array: *mut *mut f64,
        ldxf: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverRfBatchSolve)(handle, P, Q, nrhs, Temp, ldt, XF_array, ldxf)
    }
    pub unsafe fn cusolverRfBatchZeroPivot(
        handle: cusolverRfHandle_t,
        position: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverRfBatchZeroPivot)(handle, position)
    }
    pub unsafe fn cusolverRfCreate(handle: *mut cusolverRfHandle_t) -> cusolverStatus_t {
        (culib().cusolverRfCreate)(handle)
    }
    pub unsafe fn cusolverRfDestroy(handle: cusolverRfHandle_t) -> cusolverStatus_t {
        (culib().cusolverRfDestroy)(handle)
    }
    pub unsafe fn cusolverRfExtractBundledFactorsHost(
        handle: cusolverRfHandle_t,
        h_nnzM: *mut ::core::ffi::c_int,
        h_Mp: *mut *mut ::core::ffi::c_int,
        h_Mi: *mut *mut ::core::ffi::c_int,
        h_Mx: *mut *mut f64,
    ) -> cusolverStatus_t {
        (culib().cusolverRfExtractBundledFactorsHost)(handle, h_nnzM, h_Mp, h_Mi, h_Mx)
    }
    pub unsafe fn cusolverRfExtractSplitFactorsHost(
        handle: cusolverRfHandle_t,
        h_nnzL: *mut ::core::ffi::c_int,
        h_csrRowPtrL: *mut *mut ::core::ffi::c_int,
        h_csrColIndL: *mut *mut ::core::ffi::c_int,
        h_csrValL: *mut *mut f64,
        h_nnzU: *mut ::core::ffi::c_int,
        h_csrRowPtrU: *mut *mut ::core::ffi::c_int,
        h_csrColIndU: *mut *mut ::core::ffi::c_int,
        h_csrValU: *mut *mut f64,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverRfExtractSplitFactorsHost)(
            handle,
            h_nnzL,
            h_csrRowPtrL,
            h_csrColIndL,
            h_csrValL,
            h_nnzU,
            h_csrRowPtrU,
            h_csrColIndU,
            h_csrValU,
        )
    }
    pub unsafe fn cusolverRfGetAlgs(
        handle: cusolverRfHandle_t,
        factAlg: *mut cusolverRfFactorization_t,
        solveAlg: *mut cusolverRfTriangularSolve_t,
    ) -> cusolverStatus_t {
        (culib().cusolverRfGetAlgs)(handle, factAlg, solveAlg)
    }
    pub unsafe fn cusolverRfGetMatrixFormat(
        handle: cusolverRfHandle_t,
        format: *mut cusolverRfMatrixFormat_t,
        diag: *mut cusolverRfUnitDiagonal_t,
    ) -> cusolverStatus_t {
        (culib().cusolverRfGetMatrixFormat)(handle, format, diag)
    }
    pub unsafe fn cusolverRfGetNumericBoostReport(
        handle: cusolverRfHandle_t,
        report: *mut cusolverRfNumericBoostReport_t,
    ) -> cusolverStatus_t {
        (culib().cusolverRfGetNumericBoostReport)(handle, report)
    }
    pub unsafe fn cusolverRfGetNumericProperties(
        handle: cusolverRfHandle_t,
        zero: *mut f64,
        boost: *mut f64,
    ) -> cusolverStatus_t {
        (culib().cusolverRfGetNumericProperties)(handle, zero, boost)
    }
    pub unsafe fn cusolverRfGetResetValuesFastMode(
        handle: cusolverRfHandle_t,
        fastMode: *mut cusolverRfResetValuesFastMode_t,
    ) -> cusolverStatus_t {
        (culib().cusolverRfGetResetValuesFastMode)(handle, fastMode)
    }
    pub unsafe fn cusolverRfRefactor(handle: cusolverRfHandle_t) -> cusolverStatus_t {
        (culib().cusolverRfRefactor)(handle)
    }
    pub unsafe fn cusolverRfResetValues(
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        csrRowPtrA: *mut ::core::ffi::c_int,
        csrColIndA: *mut ::core::ffi::c_int,
        csrValA: *mut f64,
        P: *mut ::core::ffi::c_int,
        Q: *mut ::core::ffi::c_int,
        handle: cusolverRfHandle_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverRfResetValues)(
            n,
            nnzA,
            csrRowPtrA,
            csrColIndA,
            csrValA,
            P,
            Q,
            handle,
        )
    }
    pub unsafe fn cusolverRfSetAlgs(
        handle: cusolverRfHandle_t,
        factAlg: cusolverRfFactorization_t,
        solveAlg: cusolverRfTriangularSolve_t,
    ) -> cusolverStatus_t {
        (culib().cusolverRfSetAlgs)(handle, factAlg, solveAlg)
    }
    pub unsafe fn cusolverRfSetMatrixFormat(
        handle: cusolverRfHandle_t,
        format: cusolverRfMatrixFormat_t,
        diag: cusolverRfUnitDiagonal_t,
    ) -> cusolverStatus_t {
        (culib().cusolverRfSetMatrixFormat)(handle, format, diag)
    }
    pub unsafe fn cusolverRfSetNumericProperties(
        handle: cusolverRfHandle_t,
        zero: f64,
        boost: f64,
    ) -> cusolverStatus_t {
        (culib().cusolverRfSetNumericProperties)(handle, zero, boost)
    }
    pub unsafe fn cusolverRfSetResetValuesFastMode(
        handle: cusolverRfHandle_t,
        fastMode: cusolverRfResetValuesFastMode_t,
    ) -> cusolverStatus_t {
        (culib().cusolverRfSetResetValuesFastMode)(handle, fastMode)
    }
    pub unsafe fn cusolverRfSetupDevice(
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        csrRowPtrA: *mut ::core::ffi::c_int,
        csrColIndA: *mut ::core::ffi::c_int,
        csrValA: *mut f64,
        nnzL: ::core::ffi::c_int,
        csrRowPtrL: *mut ::core::ffi::c_int,
        csrColIndL: *mut ::core::ffi::c_int,
        csrValL: *mut f64,
        nnzU: ::core::ffi::c_int,
        csrRowPtrU: *mut ::core::ffi::c_int,
        csrColIndU: *mut ::core::ffi::c_int,
        csrValU: *mut f64,
        P: *mut ::core::ffi::c_int,
        Q: *mut ::core::ffi::c_int,
        handle: cusolverRfHandle_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverRfSetupDevice)(
            n,
            nnzA,
            csrRowPtrA,
            csrColIndA,
            csrValA,
            nnzL,
            csrRowPtrL,
            csrColIndL,
            csrValL,
            nnzU,
            csrRowPtrU,
            csrColIndU,
            csrValU,
            P,
            Q,
            handle,
        )
    }
    pub unsafe fn cusolverRfSetupHost(
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        h_csrRowPtrA: *mut ::core::ffi::c_int,
        h_csrColIndA: *mut ::core::ffi::c_int,
        h_csrValA: *mut f64,
        nnzL: ::core::ffi::c_int,
        h_csrRowPtrL: *mut ::core::ffi::c_int,
        h_csrColIndL: *mut ::core::ffi::c_int,
        h_csrValL: *mut f64,
        nnzU: ::core::ffi::c_int,
        h_csrRowPtrU: *mut ::core::ffi::c_int,
        h_csrColIndU: *mut ::core::ffi::c_int,
        h_csrValU: *mut f64,
        h_P: *mut ::core::ffi::c_int,
        h_Q: *mut ::core::ffi::c_int,
        handle: cusolverRfHandle_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverRfSetupHost)(
            n,
            nnzA,
            h_csrRowPtrA,
            h_csrColIndA,
            h_csrValA,
            nnzL,
            h_csrRowPtrL,
            h_csrColIndL,
            h_csrValL,
            nnzU,
            h_csrRowPtrU,
            h_csrColIndU,
            h_csrValU,
            h_P,
            h_Q,
            handle,
        )
    }
    pub unsafe fn cusolverRfSolve(
        handle: cusolverRfHandle_t,
        P: *mut ::core::ffi::c_int,
        Q: *mut ::core::ffi::c_int,
        nrhs: ::core::ffi::c_int,
        Temp: *mut f64,
        ldt: ::core::ffi::c_int,
        XF: *mut f64,
        ldxf: ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib().cusolverRfSolve)(handle, P, Q, nrhs, Temp, ldt, XF, ldxf)
    }
    pub unsafe fn cusolverSpCcsreigsHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        left_bottom_corner: cuComplex,
        right_upper_corner: cuComplex,
        num_eigs: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpCcsreigsHost)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            left_bottom_corner,
            right_upper_corner,
            num_eigs,
        )
    }
    pub unsafe fn cusolverSpCcsreigvsi(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: cuComplex,
        x0: *const cuComplex,
        maxite: ::core::ffi::c_int,
        eps: f32,
        mu: *mut cuComplex,
        x: *mut cuComplex,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpCcsreigvsi)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            mu0,
            x0,
            maxite,
            eps,
            mu,
            x,
        )
    }
    pub unsafe fn cusolverSpCcsreigvsiHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: cuComplex,
        x0: *const cuComplex,
        maxite: ::core::ffi::c_int,
        tol: f32,
        mu: *mut cuComplex,
        x: *mut cuComplex,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpCcsreigvsiHost)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            mu0,
            x0,
            maxite,
            tol,
            mu,
            x,
        )
    }
    pub unsafe fn cusolverSpCcsrlsqvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuComplex,
        tol: f32,
        rankA: *mut ::core::ffi::c_int,
        x: *mut cuComplex,
        p: *mut ::core::ffi::c_int,
        min_norm: *mut f32,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpCcsrlsqvqrHost)(
            handle,
            m,
            n,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            tol,
            rankA,
            x,
            p,
            min_norm,
        )
    }
    pub unsafe fn cusolverSpCcsrlsvchol(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const cuComplex,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut cuComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpCcsrlsvchol)(
            handle,
            m,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpCcsrlsvcholHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const cuComplex,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut cuComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpCcsrlsvcholHost)(
            handle,
            m,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpCcsrlsvluHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuComplex,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut cuComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpCcsrlsvluHost)(
            handle,
            n,
            nnzA,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpCcsrlsvqr(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const cuComplex,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut cuComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpCcsrlsvqr)(
            handle,
            m,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpCcsrlsvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuComplex,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut cuComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpCcsrlsvqrHost)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpCcsrqrBufferInfoBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        internalDataInBytes: *mut usize,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpCcsrqrBufferInfoBatched)(
            handle,
            m,
            n,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            batchSize,
            info,
            internalDataInBytes,
            workspaceInBytes,
        )
    }
    pub unsafe fn cusolverSpCcsrqrsvBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuComplex,
        x: *mut cuComplex,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        pBuffer: *mut ::core::ffi::c_void,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpCcsrqrsvBatched)(
            handle,
            m,
            n,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            x,
            batchSize,
            info,
            pBuffer,
        )
    }
    pub unsafe fn cusolverSpCcsrzfdHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        P: *mut ::core::ffi::c_int,
        numnz: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpCcsrzfdHost)(
            handle,
            n,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            P,
            numnz,
        )
    }
    pub unsafe fn cusolverSpCreate(handle: *mut cusolverSpHandle_t) -> cusolverStatus_t {
        (culib().cusolverSpCreate)(handle)
    }
    pub unsafe fn cusolverSpCreateCsrqrInfo(info: *mut csrqrInfo_t) -> cusolverStatus_t {
        (culib().cusolverSpCreateCsrqrInfo)(info)
    }
    pub unsafe fn cusolverSpDcsreigsHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        left_bottom_corner: cuDoubleComplex,
        right_upper_corner: cuDoubleComplex,
        num_eigs: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpDcsreigsHost)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            left_bottom_corner,
            right_upper_corner,
            num_eigs,
        )
    }
    pub unsafe fn cusolverSpDcsreigvsi(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: f64,
        x0: *const f64,
        maxite: ::core::ffi::c_int,
        eps: f64,
        mu: *mut f64,
        x: *mut f64,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpDcsreigvsi)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            mu0,
            x0,
            maxite,
            eps,
            mu,
            x,
        )
    }
    pub unsafe fn cusolverSpDcsreigvsiHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: f64,
        x0: *const f64,
        maxite: ::core::ffi::c_int,
        tol: f64,
        mu: *mut f64,
        x: *mut f64,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpDcsreigvsiHost)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            mu0,
            x0,
            maxite,
            tol,
            mu,
            x,
        )
    }
    pub unsafe fn cusolverSpDcsrlsqvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f64,
        tol: f64,
        rankA: *mut ::core::ffi::c_int,
        x: *mut f64,
        p: *mut ::core::ffi::c_int,
        min_norm: *mut f64,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpDcsrlsqvqrHost)(
            handle,
            m,
            n,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            tol,
            rankA,
            x,
            p,
            min_norm,
        )
    }
    pub unsafe fn cusolverSpDcsrlsvchol(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f64,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const f64,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut f64,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpDcsrlsvchol)(
            handle,
            m,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpDcsrlsvcholHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f64,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const f64,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut f64,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpDcsrlsvcholHost)(
            handle,
            m,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpDcsrlsvluHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f64,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut f64,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpDcsrlsvluHost)(
            handle,
            n,
            nnzA,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpDcsrlsvqr(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f64,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const f64,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut f64,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpDcsrlsvqr)(
            handle,
            m,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpDcsrlsvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f64,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut f64,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpDcsrlsvqrHost)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpDcsrqrBufferInfoBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f64,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        internalDataInBytes: *mut usize,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpDcsrqrBufferInfoBatched)(
            handle,
            m,
            n,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            batchSize,
            info,
            internalDataInBytes,
            workspaceInBytes,
        )
    }
    pub unsafe fn cusolverSpDcsrqrsvBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f64,
        x: *mut f64,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        pBuffer: *mut ::core::ffi::c_void,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpDcsrqrsvBatched)(
            handle,
            m,
            n,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            x,
            batchSize,
            info,
            pBuffer,
        )
    }
    pub unsafe fn cusolverSpDcsrzfdHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f64,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        P: *mut ::core::ffi::c_int,
        numnz: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpDcsrzfdHost)(
            handle,
            n,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            P,
            numnz,
        )
    }
    pub unsafe fn cusolverSpDestroy(handle: cusolverSpHandle_t) -> cusolverStatus_t {
        (culib().cusolverSpDestroy)(handle)
    }
    pub unsafe fn cusolverSpDestroyCsrqrInfo(info: csrqrInfo_t) -> cusolverStatus_t {
        (culib().cusolverSpDestroyCsrqrInfo)(info)
    }
    pub unsafe fn cusolverSpGetStream(
        handle: cusolverSpHandle_t,
        streamId: *mut cudaStream_t,
    ) -> cusolverStatus_t {
        (culib().cusolverSpGetStream)(handle, streamId)
    }
    pub unsafe fn cusolverSpScsreigsHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        left_bottom_corner: cuComplex,
        right_upper_corner: cuComplex,
        num_eigs: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpScsreigsHost)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            left_bottom_corner,
            right_upper_corner,
            num_eigs,
        )
    }
    pub unsafe fn cusolverSpScsreigvsi(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: f32,
        x0: *const f32,
        maxite: ::core::ffi::c_int,
        eps: f32,
        mu: *mut f32,
        x: *mut f32,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpScsreigvsi)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            mu0,
            x0,
            maxite,
            eps,
            mu,
            x,
        )
    }
    pub unsafe fn cusolverSpScsreigvsiHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: f32,
        x0: *const f32,
        maxite: ::core::ffi::c_int,
        tol: f32,
        mu: *mut f32,
        x: *mut f32,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpScsreigvsiHost)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            mu0,
            x0,
            maxite,
            tol,
            mu,
            x,
        )
    }
    pub unsafe fn cusolverSpScsrlsqvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f32,
        tol: f32,
        rankA: *mut ::core::ffi::c_int,
        x: *mut f32,
        p: *mut ::core::ffi::c_int,
        min_norm: *mut f32,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpScsrlsqvqrHost)(
            handle,
            m,
            n,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            tol,
            rankA,
            x,
            p,
            min_norm,
        )
    }
    pub unsafe fn cusolverSpScsrlsvchol(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f32,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const f32,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut f32,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpScsrlsvchol)(
            handle,
            m,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpScsrlsvcholHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f32,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const f32,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut f32,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpScsrlsvcholHost)(
            handle,
            m,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpScsrlsvluHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f32,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut f32,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpScsrlsvluHost)(
            handle,
            n,
            nnzA,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpScsrlsvqr(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f32,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const f32,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut f32,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpScsrlsvqr)(
            handle,
            m,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpScsrlsvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f32,
        tol: f32,
        reorder: ::core::ffi::c_int,
        x: *mut f32,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpScsrlsvqrHost)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpScsrqrBufferInfoBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const f32,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        internalDataInBytes: *mut usize,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpScsrqrBufferInfoBatched)(
            handle,
            m,
            n,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            batchSize,
            info,
            internalDataInBytes,
            workspaceInBytes,
        )
    }
    pub unsafe fn cusolverSpScsrqrsvBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const f32,
        x: *mut f32,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        pBuffer: *mut ::core::ffi::c_void,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpScsrqrsvBatched)(
            handle,
            m,
            n,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            x,
            batchSize,
            info,
            pBuffer,
        )
    }
    pub unsafe fn cusolverSpScsrzfdHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const f32,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        P: *mut ::core::ffi::c_int,
        numnz: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpScsrzfdHost)(
            handle,
            n,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            P,
            numnz,
        )
    }
    pub unsafe fn cusolverSpSetStream(
        handle: cusolverSpHandle_t,
        streamId: cudaStream_t,
    ) -> cusolverStatus_t {
        (culib().cusolverSpSetStream)(handle, streamId)
    }
    pub unsafe fn cusolverSpXcsrissymHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrEndPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        issym: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpXcsrissymHost)(
            handle,
            m,
            nnzA,
            descrA,
            csrRowPtrA,
            csrEndPtrA,
            csrColIndA,
            issym,
        )
    }
    pub unsafe fn cusolverSpXcsrmetisndHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        options: *const i64,
        p: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpXcsrmetisndHost)(
            handle,
            n,
            nnzA,
            descrA,
            csrRowPtrA,
            csrColIndA,
            options,
            p,
        )
    }
    pub unsafe fn cusolverSpXcsrpermHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *mut ::core::ffi::c_int,
        csrColIndA: *mut ::core::ffi::c_int,
        p: *const ::core::ffi::c_int,
        q: *const ::core::ffi::c_int,
        map: *mut ::core::ffi::c_int,
        pBuffer: *mut ::core::ffi::c_void,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpXcsrpermHost)(
            handle,
            m,
            n,
            nnzA,
            descrA,
            csrRowPtrA,
            csrColIndA,
            p,
            q,
            map,
            pBuffer,
        )
    }
    pub unsafe fn cusolverSpXcsrperm_bufferSizeHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        p: *const ::core::ffi::c_int,
        q: *const ::core::ffi::c_int,
        bufferSizeInBytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpXcsrperm_bufferSizeHost)(
            handle,
            m,
            n,
            nnzA,
            descrA,
            csrRowPtrA,
            csrColIndA,
            p,
            q,
            bufferSizeInBytes,
        )
    }
    pub unsafe fn cusolverSpXcsrqrAnalysisBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        info: csrqrInfo_t,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpXcsrqrAnalysisBatched)(
            handle,
            m,
            n,
            nnzA,
            descrA,
            csrRowPtrA,
            csrColIndA,
            info,
        )
    }
    pub unsafe fn cusolverSpXcsrsymamdHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        p: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpXcsrsymamdHost)(
            handle,
            n,
            nnzA,
            descrA,
            csrRowPtrA,
            csrColIndA,
            p,
        )
    }
    pub unsafe fn cusolverSpXcsrsymmdqHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        p: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpXcsrsymmdqHost)(
            handle,
            n,
            nnzA,
            descrA,
            csrRowPtrA,
            csrColIndA,
            p,
        )
    }
    pub unsafe fn cusolverSpXcsrsymrcmHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        p: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpXcsrsymrcmHost)(
            handle,
            n,
            nnzA,
            descrA,
            csrRowPtrA,
            csrColIndA,
            p,
        )
    }
    pub unsafe fn cusolverSpZcsreigsHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        left_bottom_corner: cuDoubleComplex,
        right_upper_corner: cuDoubleComplex,
        num_eigs: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpZcsreigsHost)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            left_bottom_corner,
            right_upper_corner,
            num_eigs,
        )
    }
    pub unsafe fn cusolverSpZcsreigvsi(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: cuDoubleComplex,
        x0: *const cuDoubleComplex,
        maxite: ::core::ffi::c_int,
        eps: f64,
        mu: *mut cuDoubleComplex,
        x: *mut cuDoubleComplex,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpZcsreigvsi)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            mu0,
            x0,
            maxite,
            eps,
            mu,
            x,
        )
    }
    pub unsafe fn cusolverSpZcsreigvsiHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        mu0: cuDoubleComplex,
        x0: *const cuDoubleComplex,
        maxite: ::core::ffi::c_int,
        tol: f64,
        mu: *mut cuDoubleComplex,
        x: *mut cuDoubleComplex,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpZcsreigvsiHost)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            mu0,
            x0,
            maxite,
            tol,
            mu,
            x,
        )
    }
    pub unsafe fn cusolverSpZcsrlsqvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuDoubleComplex,
        tol: f64,
        rankA: *mut ::core::ffi::c_int,
        x: *mut cuDoubleComplex,
        p: *mut ::core::ffi::c_int,
        min_norm: *mut f64,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpZcsrlsqvqrHost)(
            handle,
            m,
            n,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            tol,
            rankA,
            x,
            p,
            min_norm,
        )
    }
    pub unsafe fn cusolverSpZcsrlsvchol(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuDoubleComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const cuDoubleComplex,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut cuDoubleComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpZcsrlsvchol)(
            handle,
            m,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpZcsrlsvcholHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuDoubleComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const cuDoubleComplex,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut cuDoubleComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpZcsrlsvcholHost)(
            handle,
            m,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpZcsrlsvluHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnzA: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuDoubleComplex,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut cuDoubleComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpZcsrlsvluHost)(
            handle,
            n,
            nnzA,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpZcsrlsvqr(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuDoubleComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        b: *const cuDoubleComplex,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut cuDoubleComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpZcsrlsvqr)(
            handle,
            m,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpZcsrlsvqrHost(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuDoubleComplex,
        tol: f64,
        reorder: ::core::ffi::c_int,
        x: *mut cuDoubleComplex,
        singularity: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpZcsrlsvqrHost)(
            handle,
            m,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            tol,
            reorder,
            x,
            singularity,
        )
    }
    pub unsafe fn cusolverSpZcsrqrBufferInfoBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrVal: *const cuDoubleComplex,
        csrRowPtr: *const ::core::ffi::c_int,
        csrColInd: *const ::core::ffi::c_int,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        internalDataInBytes: *mut usize,
        workspaceInBytes: *mut usize,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpZcsrqrBufferInfoBatched)(
            handle,
            m,
            n,
            nnz,
            descrA,
            csrVal,
            csrRowPtr,
            csrColInd,
            batchSize,
            info,
            internalDataInBytes,
            workspaceInBytes,
        )
    }
    pub unsafe fn cusolverSpZcsrqrsvBatched(
        handle: cusolverSpHandle_t,
        m: ::core::ffi::c_int,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        b: *const cuDoubleComplex,
        x: *mut cuDoubleComplex,
        batchSize: ::core::ffi::c_int,
        info: csrqrInfo_t,
        pBuffer: *mut ::core::ffi::c_void,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpZcsrqrsvBatched)(
            handle,
            m,
            n,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            b,
            x,
            batchSize,
            info,
            pBuffer,
        )
    }
    pub unsafe fn cusolverSpZcsrzfdHost(
        handle: cusolverSpHandle_t,
        n: ::core::ffi::c_int,
        nnz: ::core::ffi::c_int,
        descrA: cusparseMatDescr_t,
        csrValA: *const cuDoubleComplex,
        csrRowPtrA: *const ::core::ffi::c_int,
        csrColIndA: *const ::core::ffi::c_int,
        P: *mut ::core::ffi::c_int,
        numnz: *mut ::core::ffi::c_int,
    ) -> cusolverStatus_t {
        (culib()
            .cusolverSpZcsrzfdHost)(
            handle,
            n,
            nnz,
            descrA,
            csrValA,
            csrRowPtrA,
            csrColIndA,
            P,
            numnz,
        )
    }
    pub struct Lib {
        __library: ::libloading::Library,
        pub cusolverDnCCgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnCCgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnCCgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnCCgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnCEgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnCEgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnCEgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnCEgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnCKgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnCKgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnCKgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnCKgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnCYgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnCYgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnCYgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnCYgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuComplex,
            lddb: cusolver_int_t,
            dX: *mut cuComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnCgebrd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            D: *mut f32,
            E: *mut f32,
            TAUQ: *mut cuComplex,
            TAUP: *mut cuComplex,
            Work: *mut cuComplex,
            Lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCgebrd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            Lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCgeqrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            TAU: *mut cuComplex,
            Workspace: *mut cuComplex,
            Lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCgeqrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCgesvd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobu: ::core::ffi::c_schar,
            jobvt: ::core::ffi::c_schar,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            S: *mut f32,
            U: *mut cuComplex,
            ldu: ::core::ffi::c_int,
            VT: *mut cuComplex,
            ldvt: ::core::ffi::c_int,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            rwork: *mut f32,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCgesvd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCgesvdaStridedBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            rank: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            d_A: *const cuComplex,
            lda: ::core::ffi::c_int,
            strideA: ::core::ffi::c_longlong,
            d_S: *mut f32,
            strideS: ::core::ffi::c_longlong,
            d_U: *mut cuComplex,
            ldu: ::core::ffi::c_int,
            strideU: ::core::ffi::c_longlong,
            d_V: *mut cuComplex,
            ldv: ::core::ffi::c_int,
            strideV: ::core::ffi::c_longlong,
            d_work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            d_info: *mut ::core::ffi::c_int,
            h_R_nrmF: *mut f64,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCgesvdaStridedBatched_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            rank: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            d_A: *const cuComplex,
            lda: ::core::ffi::c_int,
            strideA: ::core::ffi::c_longlong,
            d_S: *const f32,
            strideS: ::core::ffi::c_longlong,
            d_U: *const cuComplex,
            ldu: ::core::ffi::c_int,
            strideU: ::core::ffi::c_longlong,
            d_V: *const cuComplex,
            ldv: ::core::ffi::c_int,
            strideV: ::core::ffi::c_longlong,
            lwork: *mut ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCgesvdj: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            econ: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            S: *mut f32,
            U: *mut cuComplex,
            ldu: ::core::ffi::c_int,
            V: *mut cuComplex,
            ldv: ::core::ffi::c_int,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnCgesvdjBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            S: *mut f32,
            U: *mut cuComplex,
            ldu: ::core::ffi::c_int,
            V: *mut cuComplex,
            ldv: ::core::ffi::c_int,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCgesvdjBatched_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            S: *const f32,
            U: *const cuComplex,
            ldu: ::core::ffi::c_int,
            V: *const cuComplex,
            ldv: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCgesvdj_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            econ: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            S: *const f32,
            U: *const cuComplex,
            ldu: ::core::ffi::c_int,
            V: *const cuComplex,
            ldv: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnCgetrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            Workspace: *mut cuComplex,
            devIpiv: *mut ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCgetrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            Lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCgetrs: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            trans: cublasOperation_t,
            n: ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            devIpiv: *const ::core::ffi::c_int,
            B: *mut cuComplex,
            ldb: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCheevd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            W: *mut f32,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCheevd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            W: *const f32,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCheevdx: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            vl: f32,
            vu: f32,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *mut f32,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCheevdx_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            vl: f32,
            vu: f32,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *const f32,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCheevj: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            W: *mut f32,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnCheevjBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            W: *mut f32,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCheevjBatched_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            W: *const f32,
            lwork: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCheevj_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            W: *const f32,
            lwork: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnChegvd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            B: *mut cuComplex,
            ldb: ::core::ffi::c_int,
            W: *mut f32,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnChegvd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            B: *const cuComplex,
            ldb: ::core::ffi::c_int,
            W: *const f32,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnChegvdx: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            B: *mut cuComplex,
            ldb: ::core::ffi::c_int,
            vl: f32,
            vu: f32,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *mut f32,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnChegvdx_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            B: *const cuComplex,
            ldb: ::core::ffi::c_int,
            vl: f32,
            vu: f32,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *const f32,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnChegvj: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            B: *mut cuComplex,
            ldb: ::core::ffi::c_int,
            W: *mut f32,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnChegvj_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            B: *const cuComplex,
            ldb: ::core::ffi::c_int,
            W: *const f32,
            lwork: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnChetrd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            d: *mut f32,
            e: *mut f32,
            tau: *mut cuComplex,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnChetrd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            d: *const f32,
            e: *const f32,
            tau: *const cuComplex,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnClaswp: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            k1: ::core::ffi::c_int,
            k2: ::core::ffi::c_int,
            devIpiv: *const ::core::ffi::c_int,
            incx: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnClauum: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnClauum_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCpotrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            Workspace: *mut cuComplex,
            Lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCpotrfBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            Aarray: *mut *mut cuComplex,
            lda: ::core::ffi::c_int,
            infoArray: *mut ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCpotrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            Lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCpotri: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCpotri_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCpotrs: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            B: *mut cuComplex,
            ldb: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCpotrsBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            A: *mut *mut cuComplex,
            lda: ::core::ffi::c_int,
            B: *mut *mut cuComplex,
            ldb: ::core::ffi::c_int,
            d_info: *mut ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCreate: unsafe extern "C" fn(
            handle: *mut cusolverDnHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverDnCreateGesvdjInfo: unsafe extern "C" fn(
            info: *mut gesvdjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnCreateParams: unsafe extern "C" fn(
            params: *mut cusolverDnParams_t,
        ) -> cusolverStatus_t,
        pub cusolverDnCreateSyevjInfo: unsafe extern "C" fn(
            info: *mut syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnCsytrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            ipiv: *mut ::core::ffi::c_int,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCsytrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCsytri: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            ipiv: *const ::core::ffi::c_int,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCsytri_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            ipiv: *const ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCungbr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuComplex,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCungbr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuComplex,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCungqr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuComplex,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCungqr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuComplex,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCungtr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuComplex,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCungtr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuComplex,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCunmqr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuComplex,
            C: *mut cuComplex,
            ldc: ::core::ffi::c_int,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCunmqr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuComplex,
            C: *const cuComplex,
            ldc: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCunmtr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            uplo: cublasFillMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuComplex,
            lda: ::core::ffi::c_int,
            tau: *mut cuComplex,
            C: *mut cuComplex,
            ldc: ::core::ffi::c_int,
            work: *mut cuComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnCunmtr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            uplo: cublasFillMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *const cuComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuComplex,
            C: *const cuComplex,
            ldc: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDBgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDBgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnDBgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDBgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnDDgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDDgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnDDgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDDgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnDHgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDHgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnDHgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDHgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnDSgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDSgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnDSgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDSgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnDXgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDXgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnDXgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDXgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f64,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f64,
            lddb: cusolver_int_t,
            dX: *mut f64,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnDestroy: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDestroyGesvdjInfo: unsafe extern "C" fn(
            info: gesvdjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDestroyParams: unsafe extern "C" fn(
            params: cusolverDnParams_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDestroySyevjInfo: unsafe extern "C" fn(
            info: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDgebrd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            D: *mut f64,
            E: *mut f64,
            TAUQ: *mut f64,
            TAUP: *mut f64,
            Work: *mut f64,
            Lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDgebrd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            Lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDgeqrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            TAU: *mut f64,
            Workspace: *mut f64,
            Lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDgeqrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDgesvd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobu: ::core::ffi::c_schar,
            jobvt: ::core::ffi::c_schar,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            S: *mut f64,
            U: *mut f64,
            ldu: ::core::ffi::c_int,
            VT: *mut f64,
            ldvt: ::core::ffi::c_int,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            rwork: *mut f64,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDgesvd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDgesvdaStridedBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            rank: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            d_A: *const f64,
            lda: ::core::ffi::c_int,
            strideA: ::core::ffi::c_longlong,
            d_S: *mut f64,
            strideS: ::core::ffi::c_longlong,
            d_U: *mut f64,
            ldu: ::core::ffi::c_int,
            strideU: ::core::ffi::c_longlong,
            d_V: *mut f64,
            ldv: ::core::ffi::c_int,
            strideV: ::core::ffi::c_longlong,
            d_work: *mut f64,
            lwork: ::core::ffi::c_int,
            d_info: *mut ::core::ffi::c_int,
            h_R_nrmF: *mut f64,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDgesvdaStridedBatched_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            rank: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            d_A: *const f64,
            lda: ::core::ffi::c_int,
            strideA: ::core::ffi::c_longlong,
            d_S: *const f64,
            strideS: ::core::ffi::c_longlong,
            d_U: *const f64,
            ldu: ::core::ffi::c_int,
            strideU: ::core::ffi::c_longlong,
            d_V: *const f64,
            ldv: ::core::ffi::c_int,
            strideV: ::core::ffi::c_longlong,
            lwork: *mut ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDgesvdj: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            econ: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            S: *mut f64,
            U: *mut f64,
            ldu: ::core::ffi::c_int,
            V: *mut f64,
            ldv: ::core::ffi::c_int,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDgesvdjBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            S: *mut f64,
            U: *mut f64,
            ldu: ::core::ffi::c_int,
            V: *mut f64,
            ldv: ::core::ffi::c_int,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDgesvdjBatched_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            S: *const f64,
            U: *const f64,
            ldu: ::core::ffi::c_int,
            V: *const f64,
            ldv: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDgesvdj_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            econ: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            S: *const f64,
            U: *const f64,
            ldu: ::core::ffi::c_int,
            V: *const f64,
            ldv: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDgetrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            Workspace: *mut f64,
            devIpiv: *mut ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDgetrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            Lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDgetrs: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            trans: cublasOperation_t,
            n: ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            devIpiv: *const ::core::ffi::c_int,
            B: *mut f64,
            ldb: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDlaswp: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            k1: ::core::ffi::c_int,
            k2: ::core::ffi::c_int,
            devIpiv: *const ::core::ffi::c_int,
            incx: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDlauum: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDlauum_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDorgbr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            tau: *const f64,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDorgbr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            tau: *const f64,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDorgqr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            tau: *const f64,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDorgqr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            tau: *const f64,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDorgtr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            tau: *const f64,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDorgtr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            tau: *const f64,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDormqr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            tau: *const f64,
            C: *mut f64,
            ldc: ::core::ffi::c_int,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDormqr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            tau: *const f64,
            C: *const f64,
            ldc: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDormtr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            uplo: cublasFillMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            tau: *mut f64,
            C: *mut f64,
            ldc: ::core::ffi::c_int,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDormtr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            uplo: cublasFillMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            tau: *const f64,
            C: *const f64,
            ldc: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDpotrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            Workspace: *mut f64,
            Lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDpotrfBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            Aarray: *mut *mut f64,
            lda: ::core::ffi::c_int,
            infoArray: *mut ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDpotrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            Lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDpotri: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDpotri_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDpotrs: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            B: *mut f64,
            ldb: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDpotrsBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            A: *mut *mut f64,
            lda: ::core::ffi::c_int,
            B: *mut *mut f64,
            ldb: ::core::ffi::c_int,
            d_info: *mut ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsyevd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            W: *mut f64,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsyevd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            W: *const f64,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsyevdx: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            vl: f64,
            vu: f64,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *mut f64,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsyevdx_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            vl: f64,
            vu: f64,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *const f64,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsyevj: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            W: *mut f64,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDsyevjBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            W: *mut f64,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsyevjBatched_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            W: *const f64,
            lwork: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsyevj_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            W: *const f64,
            lwork: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDsygvd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            B: *mut f64,
            ldb: ::core::ffi::c_int,
            W: *mut f64,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsygvd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            B: *const f64,
            ldb: ::core::ffi::c_int,
            W: *const f64,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsygvdx: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            B: *mut f64,
            ldb: ::core::ffi::c_int,
            vl: f64,
            vu: f64,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *mut f64,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsygvdx_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            B: *const f64,
            ldb: ::core::ffi::c_int,
            vl: f64,
            vu: f64,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *const f64,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsygvj: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            B: *mut f64,
            ldb: ::core::ffi::c_int,
            W: *mut f64,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDsygvj_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            B: *const f64,
            ldb: ::core::ffi::c_int,
            W: *const f64,
            lwork: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnDsytrd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            d: *mut f64,
            e: *mut f64,
            tau: *mut f64,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsytrd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f64,
            lda: ::core::ffi::c_int,
            d: *const f64,
            e: *const f64,
            tau: *const f64,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsytrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            ipiv: *mut ::core::ffi::c_int,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsytrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsytri: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            ipiv: *const ::core::ffi::c_int,
            work: *mut f64,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnDsytri_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f64,
            lda: ::core::ffi::c_int,
            ipiv: *const ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnGeqrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            m: i64,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            dataTypeTau: cudaDataType,
            tau: *mut ::core::ffi::c_void,
            computeType: cudaDataType,
            pBuffer: *mut ::core::ffi::c_void,
            workspaceInBytes: usize,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnGeqrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            m: i64,
            n: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            dataTypeTau: cudaDataType,
            tau: *const ::core::ffi::c_void,
            computeType: cudaDataType,
            workspaceInBytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnGesvd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobu: ::core::ffi::c_schar,
            jobvt: ::core::ffi::c_schar,
            m: i64,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            dataTypeS: cudaDataType,
            S: *mut ::core::ffi::c_void,
            dataTypeU: cudaDataType,
            U: *mut ::core::ffi::c_void,
            ldu: i64,
            dataTypeVT: cudaDataType,
            VT: *mut ::core::ffi::c_void,
            ldvt: i64,
            computeType: cudaDataType,
            pBuffer: *mut ::core::ffi::c_void,
            workspaceInBytes: usize,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnGesvd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobu: ::core::ffi::c_schar,
            jobvt: ::core::ffi::c_schar,
            m: i64,
            n: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            dataTypeS: cudaDataType,
            S: *const ::core::ffi::c_void,
            dataTypeU: cudaDataType,
            U: *const ::core::ffi::c_void,
            ldu: i64,
            dataTypeVT: cudaDataType,
            VT: *const ::core::ffi::c_void,
            ldvt: i64,
            computeType: cudaDataType,
            workspaceInBytes: *mut usize,
        ) -> cusolverStatus_t,
        #[cfg(
            any(
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cusolverDnGetDeterministicMode: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            mode: *mut cusolverDeterministicMode_t,
        ) -> cusolverStatus_t,
        pub cusolverDnGetStream: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            streamId: *mut cudaStream_t,
        ) -> cusolverStatus_t,
        pub cusolverDnGetrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            m: i64,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            ipiv: *mut i64,
            computeType: cudaDataType,
            pBuffer: *mut ::core::ffi::c_void,
            workspaceInBytes: usize,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnGetrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            m: i64,
            n: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            computeType: cudaDataType,
            workspaceInBytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnGetrs: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            trans: cublasOperation_t,
            n: i64,
            nrhs: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            ipiv: *const i64,
            dataTypeB: cudaDataType,
            B: *mut ::core::ffi::c_void,
            ldb: i64,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSInfosCreate: unsafe extern "C" fn(
            infos_ptr: *mut cusolverDnIRSInfos_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSInfosDestroy: unsafe extern "C" fn(
            infos: cusolverDnIRSInfos_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSInfosGetMaxIters: unsafe extern "C" fn(
            infos: cusolverDnIRSInfos_t,
            maxiters: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSInfosGetNiters: unsafe extern "C" fn(
            infos: cusolverDnIRSInfos_t,
            niters: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSInfosGetOuterNiters: unsafe extern "C" fn(
            infos: cusolverDnIRSInfos_t,
            outer_niters: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSInfosGetResidualHistory: unsafe extern "C" fn(
            infos: cusolverDnIRSInfos_t,
            residual_history: *mut *mut ::core::ffi::c_void,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSInfosRequestResidual: unsafe extern "C" fn(
            infos: cusolverDnIRSInfos_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSParamsCreate: unsafe extern "C" fn(
            params_ptr: *mut cusolverDnIRSParams_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSParamsDestroy: unsafe extern "C" fn(
            params: cusolverDnIRSParams_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSParamsDisableFallback: unsafe extern "C" fn(
            params: cusolverDnIRSParams_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSParamsEnableFallback: unsafe extern "C" fn(
            params: cusolverDnIRSParams_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSParamsGetMaxIters: unsafe extern "C" fn(
            params: cusolverDnIRSParams_t,
            maxiters: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSParamsSetMaxIters: unsafe extern "C" fn(
            params: cusolverDnIRSParams_t,
            maxiters: cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSParamsSetMaxItersInner: unsafe extern "C" fn(
            params: cusolverDnIRSParams_t,
            maxiters_inner: cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSParamsSetRefinementSolver: unsafe extern "C" fn(
            params: cusolverDnIRSParams_t,
            refinement_solver: cusolverIRSRefinement_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSParamsSetSolverLowestPrecision: unsafe extern "C" fn(
            params: cusolverDnIRSParams_t,
            solver_lowest_precision: cusolverPrecType_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSParamsSetSolverMainPrecision: unsafe extern "C" fn(
            params: cusolverDnIRSParams_t,
            solver_main_precision: cusolverPrecType_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSParamsSetSolverPrecisions: unsafe extern "C" fn(
            params: cusolverDnIRSParams_t,
            solver_main_precision: cusolverPrecType_t,
            solver_lowest_precision: cusolverPrecType_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSParamsSetTol: unsafe extern "C" fn(
            params: cusolverDnIRSParams_t,
            val: f64,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSParamsSetTolInner: unsafe extern "C" fn(
            params: cusolverDnIRSParams_t,
            val: f64,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSXgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            gels_irs_params: cusolverDnIRSParams_t,
            gels_irs_infos: cusolverDnIRSInfos_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut ::core::ffi::c_void,
            ldda: cusolver_int_t,
            dB: *mut ::core::ffi::c_void,
            lddb: cusolver_int_t,
            dX: *mut ::core::ffi::c_void,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            niters: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSXgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnIRSParams_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSXgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            gesv_irs_params: cusolverDnIRSParams_t,
            gesv_irs_infos: cusolverDnIRSInfos_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut ::core::ffi::c_void,
            ldda: cusolver_int_t,
            dB: *mut ::core::ffi::c_void,
            lddb: cusolver_int_t,
            dX: *mut ::core::ffi::c_void,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            niters: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnIRSXgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnIRSParams_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        #[cfg(
            any(
                feature = "cuda-11070",
                feature = "cuda-11080",
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cusolverDnLoggerForceDisable: unsafe extern "C" fn() -> cusolverStatus_t,
        #[cfg(
            any(
                feature = "cuda-11070",
                feature = "cuda-11080",
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cusolverDnLoggerOpenFile: unsafe extern "C" fn(
            logFile: *const ::core::ffi::c_char,
        ) -> cusolverStatus_t,
        #[cfg(
            any(
                feature = "cuda-11070",
                feature = "cuda-11080",
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cusolverDnLoggerSetCallback: unsafe extern "C" fn(
            callback: cusolverDnLoggerCallback_t,
        ) -> cusolverStatus_t,
        #[cfg(
            any(
                feature = "cuda-11070",
                feature = "cuda-11080",
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cusolverDnLoggerSetFile: unsafe extern "C" fn(
            file: *mut FILE,
        ) -> cusolverStatus_t,
        #[cfg(
            any(
                feature = "cuda-11070",
                feature = "cuda-11080",
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cusolverDnLoggerSetLevel: unsafe extern "C" fn(
            level: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        #[cfg(
            any(
                feature = "cuda-11070",
                feature = "cuda-11080",
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cusolverDnLoggerSetMask: unsafe extern "C" fn(
            mask: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnPotrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            uplo: cublasFillMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            computeType: cudaDataType,
            pBuffer: *mut ::core::ffi::c_void,
            workspaceInBytes: usize,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnPotrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            uplo: cublasFillMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            computeType: cudaDataType,
            workspaceInBytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnPotrs: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            uplo: cublasFillMode_t,
            n: i64,
            nrhs: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            dataTypeB: cudaDataType,
            B: *mut ::core::ffi::c_void,
            ldb: i64,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSBgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSBgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnSBgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSBgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnSHgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSHgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnSHgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSHgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnSSgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSSgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnSSgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSSgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnSXgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSXgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnSXgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSXgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut f32,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut f32,
            lddb: cusolver_int_t,
            dX: *mut f32,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnSetAdvOptions: unsafe extern "C" fn(
            params: cusolverDnParams_t,
            function: cusolverDnFunction_t,
            algo: cusolverAlgMode_t,
        ) -> cusolverStatus_t,
        #[cfg(
            any(
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cusolverDnSetDeterministicMode: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            mode: cusolverDeterministicMode_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSetStream: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            streamId: cudaStream_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSgebrd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            D: *mut f32,
            E: *mut f32,
            TAUQ: *mut f32,
            TAUP: *mut f32,
            Work: *mut f32,
            Lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSgebrd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            Lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSgeqrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            TAU: *mut f32,
            Workspace: *mut f32,
            Lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSgeqrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSgesvd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobu: ::core::ffi::c_schar,
            jobvt: ::core::ffi::c_schar,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            S: *mut f32,
            U: *mut f32,
            ldu: ::core::ffi::c_int,
            VT: *mut f32,
            ldvt: ::core::ffi::c_int,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            rwork: *mut f32,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSgesvd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSgesvdaStridedBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            rank: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            d_A: *const f32,
            lda: ::core::ffi::c_int,
            strideA: ::core::ffi::c_longlong,
            d_S: *mut f32,
            strideS: ::core::ffi::c_longlong,
            d_U: *mut f32,
            ldu: ::core::ffi::c_int,
            strideU: ::core::ffi::c_longlong,
            d_V: *mut f32,
            ldv: ::core::ffi::c_int,
            strideV: ::core::ffi::c_longlong,
            d_work: *mut f32,
            lwork: ::core::ffi::c_int,
            d_info: *mut ::core::ffi::c_int,
            h_R_nrmF: *mut f64,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSgesvdaStridedBatched_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            rank: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            d_A: *const f32,
            lda: ::core::ffi::c_int,
            strideA: ::core::ffi::c_longlong,
            d_S: *const f32,
            strideS: ::core::ffi::c_longlong,
            d_U: *const f32,
            ldu: ::core::ffi::c_int,
            strideU: ::core::ffi::c_longlong,
            d_V: *const f32,
            ldv: ::core::ffi::c_int,
            strideV: ::core::ffi::c_longlong,
            lwork: *mut ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSgesvdj: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            econ: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            S: *mut f32,
            U: *mut f32,
            ldu: ::core::ffi::c_int,
            V: *mut f32,
            ldv: ::core::ffi::c_int,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSgesvdjBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            S: *mut f32,
            U: *mut f32,
            ldu: ::core::ffi::c_int,
            V: *mut f32,
            ldv: ::core::ffi::c_int,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSgesvdjBatched_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            S: *const f32,
            U: *const f32,
            ldu: ::core::ffi::c_int,
            V: *const f32,
            ldv: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSgesvdj_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            econ: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            S: *const f32,
            U: *const f32,
            ldu: ::core::ffi::c_int,
            V: *const f32,
            ldv: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSgetrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            Workspace: *mut f32,
            devIpiv: *mut ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSgetrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            Lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSgetrs: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            trans: cublasOperation_t,
            n: ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            devIpiv: *const ::core::ffi::c_int,
            B: *mut f32,
            ldb: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSlaswp: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            k1: ::core::ffi::c_int,
            k2: ::core::ffi::c_int,
            devIpiv: *const ::core::ffi::c_int,
            incx: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSlauum: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSlauum_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSorgbr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            tau: *const f32,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSorgbr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            tau: *const f32,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSorgqr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            tau: *const f32,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSorgqr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            tau: *const f32,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSorgtr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            tau: *const f32,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSorgtr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            tau: *const f32,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSormqr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            tau: *const f32,
            C: *mut f32,
            ldc: ::core::ffi::c_int,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSormqr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            tau: *const f32,
            C: *const f32,
            ldc: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSormtr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            uplo: cublasFillMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            tau: *mut f32,
            C: *mut f32,
            ldc: ::core::ffi::c_int,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSormtr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            uplo: cublasFillMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            tau: *const f32,
            C: *const f32,
            ldc: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSpotrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            Workspace: *mut f32,
            Lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSpotrfBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            Aarray: *mut *mut f32,
            lda: ::core::ffi::c_int,
            infoArray: *mut ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSpotrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            Lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSpotri: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSpotri_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSpotrs: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            B: *mut f32,
            ldb: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSpotrsBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            A: *mut *mut f32,
            lda: ::core::ffi::c_int,
            B: *mut *mut f32,
            ldb: ::core::ffi::c_int,
            d_info: *mut ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsyevd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            W: *mut f32,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsyevd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            W: *const f32,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsyevdx: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            vl: f32,
            vu: f32,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *mut f32,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsyevdx_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            vl: f32,
            vu: f32,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *const f32,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsyevj: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            W: *mut f32,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSsyevjBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            W: *mut f32,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsyevjBatched_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            W: *const f32,
            lwork: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsyevj_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            W: *const f32,
            lwork: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSsygvd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            B: *mut f32,
            ldb: ::core::ffi::c_int,
            W: *mut f32,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsygvd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            B: *const f32,
            ldb: ::core::ffi::c_int,
            W: *const f32,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsygvdx: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            B: *mut f32,
            ldb: ::core::ffi::c_int,
            vl: f32,
            vu: f32,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *mut f32,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsygvdx_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            B: *const f32,
            ldb: ::core::ffi::c_int,
            vl: f32,
            vu: f32,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *const f32,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsygvj: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            B: *mut f32,
            ldb: ::core::ffi::c_int,
            W: *mut f32,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSsygvj_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            B: *const f32,
            ldb: ::core::ffi::c_int,
            W: *const f32,
            lwork: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnSsytrd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            d: *mut f32,
            e: *mut f32,
            tau: *mut f32,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsytrd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const f32,
            lda: ::core::ffi::c_int,
            d: *const f32,
            e: *const f32,
            tau: *const f32,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsytrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            ipiv: *mut ::core::ffi::c_int,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsytrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsytri: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            ipiv: *const ::core::ffi::c_int,
            work: *mut f32,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSsytri_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut f32,
            lda: ::core::ffi::c_int,
            ipiv: *const ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSyevd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            dataTypeW: cudaDataType,
            W: *mut ::core::ffi::c_void,
            computeType: cudaDataType,
            pBuffer: *mut ::core::ffi::c_void,
            workspaceInBytes: usize,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSyevd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            dataTypeW: cudaDataType,
            W: *const ::core::ffi::c_void,
            computeType: cudaDataType,
            workspaceInBytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnSyevdx: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            vl: *mut ::core::ffi::c_void,
            vu: *mut ::core::ffi::c_void,
            il: i64,
            iu: i64,
            meig64: *mut i64,
            dataTypeW: cudaDataType,
            W: *mut ::core::ffi::c_void,
            computeType: cudaDataType,
            pBuffer: *mut ::core::ffi::c_void,
            workspaceInBytes: usize,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnSyevdx_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            vl: *mut ::core::ffi::c_void,
            vu: *mut ::core::ffi::c_void,
            il: i64,
            iu: i64,
            h_meig: *mut i64,
            dataTypeW: cudaDataType,
            W: *const ::core::ffi::c_void,
            computeType: cudaDataType,
            workspaceInBytes: *mut usize,
        ) -> cusolverStatus_t,
        #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
        pub cusolverDnXgeev: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobvl: cusolverEigMode_t,
            jobvr: cusolverEigMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            dataTypeW: cudaDataType,
            W: *mut ::core::ffi::c_void,
            dataTypeVL: cudaDataType,
            VL: *mut ::core::ffi::c_void,
            ldvl: i64,
            dataTypeVR: cudaDataType,
            VR: *mut ::core::ffi::c_void,
            ldvr: i64,
            computeType: cudaDataType,
            bufferOnDevice: *mut ::core::ffi::c_void,
            workspaceInBytesOnDevice: usize,
            bufferOnHost: *mut ::core::ffi::c_void,
            workspaceInBytesOnHost: usize,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
        pub cusolverDnXgeev_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobvl: cusolverEigMode_t,
            jobvr: cusolverEigMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            dataTypeW: cudaDataType,
            W: *const ::core::ffi::c_void,
            dataTypeVL: cudaDataType,
            VL: *const ::core::ffi::c_void,
            ldvl: i64,
            dataTypeVR: cudaDataType,
            VR: *const ::core::ffi::c_void,
            ldvr: i64,
            computeType: cudaDataType,
            workspaceInBytesOnDevice: *mut usize,
            workspaceInBytesOnHost: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnXgeqrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            m: i64,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            dataTypeTau: cudaDataType,
            tau: *mut ::core::ffi::c_void,
            computeType: cudaDataType,
            bufferOnDevice: *mut ::core::ffi::c_void,
            workspaceInBytesOnDevice: usize,
            bufferOnHost: *mut ::core::ffi::c_void,
            workspaceInBytesOnHost: usize,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnXgeqrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            m: i64,
            n: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            dataTypeTau: cudaDataType,
            tau: *const ::core::ffi::c_void,
            computeType: cudaDataType,
            workspaceInBytesOnDevice: *mut usize,
            workspaceInBytesOnHost: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnXgesvd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobu: ::core::ffi::c_schar,
            jobvt: ::core::ffi::c_schar,
            m: i64,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            dataTypeS: cudaDataType,
            S: *mut ::core::ffi::c_void,
            dataTypeU: cudaDataType,
            U: *mut ::core::ffi::c_void,
            ldu: i64,
            dataTypeVT: cudaDataType,
            VT: *mut ::core::ffi::c_void,
            ldvt: i64,
            computeType: cudaDataType,
            bufferOnDevice: *mut ::core::ffi::c_void,
            workspaceInBytesOnDevice: usize,
            bufferOnHost: *mut ::core::ffi::c_void,
            workspaceInBytesOnHost: usize,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnXgesvd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobu: ::core::ffi::c_schar,
            jobvt: ::core::ffi::c_schar,
            m: i64,
            n: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            dataTypeS: cudaDataType,
            S: *const ::core::ffi::c_void,
            dataTypeU: cudaDataType,
            U: *const ::core::ffi::c_void,
            ldu: i64,
            dataTypeVT: cudaDataType,
            VT: *const ::core::ffi::c_void,
            ldvt: i64,
            computeType: cudaDataType,
            workspaceInBytesOnDevice: *mut usize,
            workspaceInBytesOnHost: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnXgesvdjGetResidual: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            info: gesvdjInfo_t,
            residual: *mut f64,
        ) -> cusolverStatus_t,
        pub cusolverDnXgesvdjGetSweeps: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            info: gesvdjInfo_t,
            executed_sweeps: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnXgesvdjSetMaxSweeps: unsafe extern "C" fn(
            info: gesvdjInfo_t,
            max_sweeps: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnXgesvdjSetSortEig: unsafe extern "C" fn(
            info: gesvdjInfo_t,
            sort_svd: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnXgesvdjSetTolerance: unsafe extern "C" fn(
            info: gesvdjInfo_t,
            tolerance: f64,
        ) -> cusolverStatus_t,
        pub cusolverDnXgesvdp: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobz: cusolverEigMode_t,
            econ: ::core::ffi::c_int,
            m: i64,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            dataTypeS: cudaDataType,
            S: *mut ::core::ffi::c_void,
            dataTypeU: cudaDataType,
            U: *mut ::core::ffi::c_void,
            ldu: i64,
            dataTypeV: cudaDataType,
            V: *mut ::core::ffi::c_void,
            ldv: i64,
            computeType: cudaDataType,
            bufferOnDevice: *mut ::core::ffi::c_void,
            workspaceInBytesOnDevice: usize,
            bufferOnHost: *mut ::core::ffi::c_void,
            workspaceInBytesOnHost: usize,
            d_info: *mut ::core::ffi::c_int,
            h_err_sigma: *mut f64,
        ) -> cusolverStatus_t,
        pub cusolverDnXgesvdp_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobz: cusolverEigMode_t,
            econ: ::core::ffi::c_int,
            m: i64,
            n: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            dataTypeS: cudaDataType,
            S: *const ::core::ffi::c_void,
            dataTypeU: cudaDataType,
            U: *const ::core::ffi::c_void,
            ldu: i64,
            dataTypeV: cudaDataType,
            V: *const ::core::ffi::c_void,
            ldv: i64,
            computeType: cudaDataType,
            workspaceInBytesOnDevice: *mut usize,
            workspaceInBytesOnHost: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnXgesvdr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobu: ::core::ffi::c_schar,
            jobv: ::core::ffi::c_schar,
            m: i64,
            n: i64,
            k: i64,
            p: i64,
            niters: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            dataTypeSrand: cudaDataType,
            Srand: *mut ::core::ffi::c_void,
            dataTypeUrand: cudaDataType,
            Urand: *mut ::core::ffi::c_void,
            ldUrand: i64,
            dataTypeVrand: cudaDataType,
            Vrand: *mut ::core::ffi::c_void,
            ldVrand: i64,
            computeType: cudaDataType,
            bufferOnDevice: *mut ::core::ffi::c_void,
            workspaceInBytesOnDevice: usize,
            bufferOnHost: *mut ::core::ffi::c_void,
            workspaceInBytesOnHost: usize,
            d_info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnXgesvdr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobu: ::core::ffi::c_schar,
            jobv: ::core::ffi::c_schar,
            m: i64,
            n: i64,
            k: i64,
            p: i64,
            niters: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            dataTypeSrand: cudaDataType,
            Srand: *const ::core::ffi::c_void,
            dataTypeUrand: cudaDataType,
            Urand: *const ::core::ffi::c_void,
            ldUrand: i64,
            dataTypeVrand: cudaDataType,
            Vrand: *const ::core::ffi::c_void,
            ldVrand: i64,
            computeType: cudaDataType,
            workspaceInBytesOnDevice: *mut usize,
            workspaceInBytesOnHost: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnXgetrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            m: i64,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            ipiv: *mut i64,
            computeType: cudaDataType,
            bufferOnDevice: *mut ::core::ffi::c_void,
            workspaceInBytesOnDevice: usize,
            bufferOnHost: *mut ::core::ffi::c_void,
            workspaceInBytesOnHost: usize,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnXgetrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            m: i64,
            n: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            computeType: cudaDataType,
            workspaceInBytesOnDevice: *mut usize,
            workspaceInBytesOnHost: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnXgetrs: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            trans: cublasOperation_t,
            n: i64,
            nrhs: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            ipiv: *const i64,
            dataTypeB: cudaDataType,
            B: *mut ::core::ffi::c_void,
            ldb: i64,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        #[cfg(any(feature = "cuda-12040"))]
        pub cusolverDnXlarft: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            direct: cusolverDirectMode_t,
            storev: cusolverStorevMode_t,
            N: i64,
            K: i64,
            dataTypeV: cudaDataType,
            d_V: *const ::core::ffi::c_void,
            ldv: i64,
            dataTypeTau: cudaDataType,
            d_tau: *const ::core::ffi::c_void,
            dataTypeT: cudaDataType,
            d_T: *mut ::core::ffi::c_void,
            ldt: i64,
            computeType: cudaDataType,
            bufferOnDevice: *mut ::core::ffi::c_void,
            workspaceInBytesOnDevice: usize,
            bufferOnHost: *mut ::core::ffi::c_void,
            workspaceInBytesOnHost: usize,
        ) -> cusolverStatus_t,
        #[cfg(
            any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080")
        )]
        pub cusolverDnXlarft: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            direct: cusolverDirectMode_t,
            storev: cusolverStorevMode_t,
            n: i64,
            k: i64,
            dataTypeV: cudaDataType,
            V: *const ::core::ffi::c_void,
            ldv: i64,
            dataTypeTau: cudaDataType,
            tau: *const ::core::ffi::c_void,
            dataTypeT: cudaDataType,
            T: *mut ::core::ffi::c_void,
            ldt: i64,
            computeType: cudaDataType,
            bufferOnDevice: *mut ::core::ffi::c_void,
            workspaceInBytesOnDevice: usize,
            bufferOnHost: *mut ::core::ffi::c_void,
            workspaceInBytesOnHost: usize,
        ) -> cusolverStatus_t,
        #[cfg(any(feature = "cuda-12040"))]
        pub cusolverDnXlarft_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            direct: cusolverDirectMode_t,
            storev: cusolverStorevMode_t,
            N: i64,
            K: i64,
            dataTypeV: cudaDataType,
            d_V: *const ::core::ffi::c_void,
            ldv: i64,
            dataTypeTau: cudaDataType,
            d_tau: *const ::core::ffi::c_void,
            dataTypeT: cudaDataType,
            d_T: *mut ::core::ffi::c_void,
            ldt: i64,
            computeType: cudaDataType,
            workspaceInBytesOnDevice: *mut usize,
            workspaceInBytesOnHost: *mut usize,
        ) -> cusolverStatus_t,
        #[cfg(
            any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080")
        )]
        pub cusolverDnXlarft_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            direct: cusolverDirectMode_t,
            storev: cusolverStorevMode_t,
            n: i64,
            k: i64,
            dataTypeV: cudaDataType,
            V: *const ::core::ffi::c_void,
            ldv: i64,
            dataTypeTau: cudaDataType,
            tau: *const ::core::ffi::c_void,
            dataTypeT: cudaDataType,
            T: *mut ::core::ffi::c_void,
            ldt: i64,
            computeType: cudaDataType,
            workspaceInBytesOnDevice: *mut usize,
            workspaceInBytesOnHost: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnXpotrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            uplo: cublasFillMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            computeType: cudaDataType,
            bufferOnDevice: *mut ::core::ffi::c_void,
            workspaceInBytesOnDevice: usize,
            bufferOnHost: *mut ::core::ffi::c_void,
            workspaceInBytesOnHost: usize,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnXpotrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            uplo: cublasFillMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            computeType: cudaDataType,
            workspaceInBytesOnDevice: *mut usize,
            workspaceInBytesOnHost: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnXpotrs: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            uplo: cublasFillMode_t,
            n: i64,
            nrhs: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            dataTypeB: cudaDataType,
            B: *mut ::core::ffi::c_void,
            ldb: i64,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
        pub cusolverDnXsyevBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            dataTypeW: cudaDataType,
            W: *mut ::core::ffi::c_void,
            computeType: cudaDataType,
            bufferOnDevice: *mut ::core::ffi::c_void,
            workspaceInBytesOnDevice: usize,
            bufferOnHost: *mut ::core::ffi::c_void,
            workspaceInBytesOnHost: usize,
            info: *mut ::core::ffi::c_int,
            batchSize: i64,
        ) -> cusolverStatus_t,
        #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
        pub cusolverDnXsyevBatched_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            dataTypeW: cudaDataType,
            W: *const ::core::ffi::c_void,
            computeType: cudaDataType,
            workspaceInBytesOnDevice: *mut usize,
            workspaceInBytesOnHost: *mut usize,
            batchSize: i64,
        ) -> cusolverStatus_t,
        pub cusolverDnXsyevd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            dataTypeW: cudaDataType,
            W: *mut ::core::ffi::c_void,
            computeType: cudaDataType,
            bufferOnDevice: *mut ::core::ffi::c_void,
            workspaceInBytesOnDevice: usize,
            bufferOnHost: *mut ::core::ffi::c_void,
            workspaceInBytesOnHost: usize,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnXsyevd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            dataTypeW: cudaDataType,
            W: *const ::core::ffi::c_void,
            computeType: cudaDataType,
            workspaceInBytesOnDevice: *mut usize,
            workspaceInBytesOnHost: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnXsyevdx: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            vl: *mut ::core::ffi::c_void,
            vu: *mut ::core::ffi::c_void,
            il: i64,
            iu: i64,
            meig64: *mut i64,
            dataTypeW: cudaDataType,
            W: *mut ::core::ffi::c_void,
            computeType: cudaDataType,
            bufferOnDevice: *mut ::core::ffi::c_void,
            workspaceInBytesOnDevice: usize,
            bufferOnHost: *mut ::core::ffi::c_void,
            workspaceInBytesOnHost: usize,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnXsyevdx_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            params: cusolverDnParams_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            vl: *mut ::core::ffi::c_void,
            vu: *mut ::core::ffi::c_void,
            il: i64,
            iu: i64,
            h_meig: *mut i64,
            dataTypeW: cudaDataType,
            W: *const ::core::ffi::c_void,
            computeType: cudaDataType,
            workspaceInBytesOnDevice: *mut usize,
            workspaceInBytesOnHost: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnXsyevjGetResidual: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            info: syevjInfo_t,
            residual: *mut f64,
        ) -> cusolverStatus_t,
        pub cusolverDnXsyevjGetSweeps: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            info: syevjInfo_t,
            executed_sweeps: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnXsyevjSetMaxSweeps: unsafe extern "C" fn(
            info: syevjInfo_t,
            max_sweeps: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnXsyevjSetSortEig: unsafe extern "C" fn(
            info: syevjInfo_t,
            sort_eig: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnXsyevjSetTolerance: unsafe extern "C" fn(
            info: syevjInfo_t,
            tolerance: f64,
        ) -> cusolverStatus_t,
        pub cusolverDnXsytrs: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: i64,
            nrhs: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            ipiv: *const i64,
            dataTypeB: cudaDataType,
            B: *mut ::core::ffi::c_void,
            ldb: i64,
            bufferOnDevice: *mut ::core::ffi::c_void,
            workspaceInBytesOnDevice: usize,
            bufferOnHost: *mut ::core::ffi::c_void,
            workspaceInBytesOnHost: usize,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnXsytrs_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: i64,
            nrhs: i64,
            dataTypeA: cudaDataType,
            A: *const ::core::ffi::c_void,
            lda: i64,
            ipiv: *const i64,
            dataTypeB: cudaDataType,
            B: *mut ::core::ffi::c_void,
            ldb: i64,
            workspaceInBytesOnDevice: *mut usize,
            workspaceInBytesOnHost: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnXtrtri: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            diag: cublasDiagType_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            bufferOnDevice: *mut ::core::ffi::c_void,
            workspaceInBytesOnDevice: usize,
            bufferOnHost: *mut ::core::ffi::c_void,
            workspaceInBytesOnHost: usize,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnXtrtri_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            diag: cublasDiagType_t,
            n: i64,
            dataTypeA: cudaDataType,
            A: *mut ::core::ffi::c_void,
            lda: i64,
            workspaceInBytesOnDevice: *mut usize,
            workspaceInBytesOnHost: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnZCgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZCgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnZCgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZCgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnZEgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZEgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnZEgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZEgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnZKgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZKgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnZKgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZKgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnZYgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZYgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnZYgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZYgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnZZgels: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZZgels_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: cusolver_int_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnZZgesv: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: usize,
            iter: *mut cusolver_int_t,
            d_info: *mut cusolver_int_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZZgesv_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: cusolver_int_t,
            nrhs: cusolver_int_t,
            dA: *mut cuDoubleComplex,
            ldda: cusolver_int_t,
            dipiv: *mut cusolver_int_t,
            dB: *mut cuDoubleComplex,
            lddb: cusolver_int_t,
            dX: *mut cuDoubleComplex,
            lddx: cusolver_int_t,
            dWorkspace: *mut ::core::ffi::c_void,
            lwork_bytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverDnZgebrd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            D: *mut f64,
            E: *mut f64,
            TAUQ: *mut cuDoubleComplex,
            TAUP: *mut cuDoubleComplex,
            Work: *mut cuDoubleComplex,
            Lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZgebrd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            Lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZgeqrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            TAU: *mut cuDoubleComplex,
            Workspace: *mut cuDoubleComplex,
            Lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZgeqrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZgesvd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobu: ::core::ffi::c_schar,
            jobvt: ::core::ffi::c_schar,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            S: *mut f64,
            U: *mut cuDoubleComplex,
            ldu: ::core::ffi::c_int,
            VT: *mut cuDoubleComplex,
            ldvt: ::core::ffi::c_int,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            rwork: *mut f64,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZgesvd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZgesvdaStridedBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            rank: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            d_A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            strideA: ::core::ffi::c_longlong,
            d_S: *mut f64,
            strideS: ::core::ffi::c_longlong,
            d_U: *mut cuDoubleComplex,
            ldu: ::core::ffi::c_int,
            strideU: ::core::ffi::c_longlong,
            d_V: *mut cuDoubleComplex,
            ldv: ::core::ffi::c_int,
            strideV: ::core::ffi::c_longlong,
            d_work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            d_info: *mut ::core::ffi::c_int,
            h_R_nrmF: *mut f64,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZgesvdaStridedBatched_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            rank: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            d_A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            strideA: ::core::ffi::c_longlong,
            d_S: *const f64,
            strideS: ::core::ffi::c_longlong,
            d_U: *const cuDoubleComplex,
            ldu: ::core::ffi::c_int,
            strideU: ::core::ffi::c_longlong,
            d_V: *const cuDoubleComplex,
            ldv: ::core::ffi::c_int,
            strideV: ::core::ffi::c_longlong,
            lwork: *mut ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZgesvdj: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            econ: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            S: *mut f64,
            U: *mut cuDoubleComplex,
            ldu: ::core::ffi::c_int,
            V: *mut cuDoubleComplex,
            ldv: ::core::ffi::c_int,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZgesvdjBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            S: *mut f64,
            U: *mut cuDoubleComplex,
            ldu: ::core::ffi::c_int,
            V: *mut cuDoubleComplex,
            ldv: ::core::ffi::c_int,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZgesvdjBatched_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            S: *const f64,
            U: *const cuDoubleComplex,
            ldu: ::core::ffi::c_int,
            V: *const cuDoubleComplex,
            ldv: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZgesvdj_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            econ: ::core::ffi::c_int,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            S: *const f64,
            U: *const cuDoubleComplex,
            ldu: ::core::ffi::c_int,
            V: *const cuDoubleComplex,
            ldv: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
            params: gesvdjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZgetrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            Workspace: *mut cuDoubleComplex,
            devIpiv: *mut ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZgetrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            Lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZgetrs: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            trans: cublasOperation_t,
            n: ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            devIpiv: *const ::core::ffi::c_int,
            B: *mut cuDoubleComplex,
            ldb: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZheevd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            W: *mut f64,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZheevd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            W: *const f64,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZheevdx: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            vl: f64,
            vu: f64,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *mut f64,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZheevdx_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            vl: f64,
            vu: f64,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *const f64,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZheevj: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            W: *mut f64,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZheevjBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            W: *mut f64,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZheevjBatched_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            W: *const f64,
            lwork: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZheevj_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            W: *const f64,
            lwork: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZhegvd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            B: *mut cuDoubleComplex,
            ldb: ::core::ffi::c_int,
            W: *mut f64,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZhegvd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            B: *const cuDoubleComplex,
            ldb: ::core::ffi::c_int,
            W: *const f64,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZhegvdx: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            B: *mut cuDoubleComplex,
            ldb: ::core::ffi::c_int,
            vl: f64,
            vu: f64,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *mut f64,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZhegvdx_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            range: cusolverEigRange_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            B: *const cuDoubleComplex,
            ldb: ::core::ffi::c_int,
            vl: f64,
            vu: f64,
            il: ::core::ffi::c_int,
            iu: ::core::ffi::c_int,
            meig: *mut ::core::ffi::c_int,
            W: *const f64,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZhegvj: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            B: *mut cuDoubleComplex,
            ldb: ::core::ffi::c_int,
            W: *mut f64,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZhegvj_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            itype: cusolverEigType_t,
            jobz: cusolverEigMode_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            B: *const cuDoubleComplex,
            ldb: ::core::ffi::c_int,
            W: *const f64,
            lwork: *mut ::core::ffi::c_int,
            params: syevjInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverDnZhetrd: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            d: *mut f64,
            e: *mut f64,
            tau: *mut cuDoubleComplex,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZhetrd_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            d: *const f64,
            e: *const f64,
            tau: *const cuDoubleComplex,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZlaswp: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            k1: ::core::ffi::c_int,
            k2: ::core::ffi::c_int,
            devIpiv: *const ::core::ffi::c_int,
            incx: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZlauum: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZlauum_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZpotrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            Workspace: *mut cuDoubleComplex,
            Lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZpotrfBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            Aarray: *mut *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            infoArray: *mut ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZpotrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            Lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZpotri: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZpotri_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZpotrs: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            B: *mut cuDoubleComplex,
            ldb: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZpotrsBatched: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            A: *mut *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            B: *mut *mut cuDoubleComplex,
            ldb: ::core::ffi::c_int,
            d_info: *mut ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZsytrf: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            ipiv: *mut ::core::ffi::c_int,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZsytrf_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZsytri: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            ipiv: *const ::core::ffi::c_int,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZsytri_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            ipiv: *const ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZungbr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuDoubleComplex,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZungbr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuDoubleComplex,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZungqr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuDoubleComplex,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZungqr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuDoubleComplex,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZungtr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuDoubleComplex,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZungtr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            uplo: cublasFillMode_t,
            n: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuDoubleComplex,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZunmqr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuDoubleComplex,
            C: *mut cuDoubleComplex,
            ldc: ::core::ffi::c_int,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            devInfo: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZunmqr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            k: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuDoubleComplex,
            C: *const cuDoubleComplex,
            ldc: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZunmtr: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            uplo: cublasFillMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *mut cuDoubleComplex,
            lda: ::core::ffi::c_int,
            tau: *mut cuDoubleComplex,
            C: *mut cuDoubleComplex,
            ldc: ::core::ffi::c_int,
            work: *mut cuDoubleComplex,
            lwork: ::core::ffi::c_int,
            info: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverDnZunmtr_bufferSize: unsafe extern "C" fn(
            handle: cusolverDnHandle_t,
            side: cublasSideMode_t,
            uplo: cublasFillMode_t,
            trans: cublasOperation_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            A: *const cuDoubleComplex,
            lda: ::core::ffi::c_int,
            tau: *const cuDoubleComplex,
            C: *const cuDoubleComplex,
            ldc: ::core::ffi::c_int,
            lwork: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverGetProperty: unsafe extern "C" fn(
            type_: libraryPropertyType,
            value: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverGetVersion: unsafe extern "C" fn(
            version: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
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
        pub cusolverRfAccessBundledFactorsDevice: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
            nnzM: *mut ::core::ffi::c_int,
            Mp: *mut *mut ::core::ffi::c_int,
            Mi: *mut *mut ::core::ffi::c_int,
            Mx: *mut *mut f64,
        ) -> cusolverStatus_t,
        pub cusolverRfAnalyze: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverRfBatchAnalyze: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverRfBatchRefactor: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverRfBatchResetValues: unsafe extern "C" fn(
            batchSize: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            csrRowPtrA: *mut ::core::ffi::c_int,
            csrColIndA: *mut ::core::ffi::c_int,
            csrValA_array: *mut *mut f64,
            P: *mut ::core::ffi::c_int,
            Q: *mut ::core::ffi::c_int,
            handle: cusolverRfHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverRfBatchSetupHost: unsafe extern "C" fn(
            batchSize: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            h_csrRowPtrA: *mut ::core::ffi::c_int,
            h_csrColIndA: *mut ::core::ffi::c_int,
            h_csrValA_array: *mut *mut f64,
            nnzL: ::core::ffi::c_int,
            h_csrRowPtrL: *mut ::core::ffi::c_int,
            h_csrColIndL: *mut ::core::ffi::c_int,
            h_csrValL: *mut f64,
            nnzU: ::core::ffi::c_int,
            h_csrRowPtrU: *mut ::core::ffi::c_int,
            h_csrColIndU: *mut ::core::ffi::c_int,
            h_csrValU: *mut f64,
            h_P: *mut ::core::ffi::c_int,
            h_Q: *mut ::core::ffi::c_int,
            handle: cusolverRfHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverRfBatchSolve: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
            P: *mut ::core::ffi::c_int,
            Q: *mut ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            Temp: *mut f64,
            ldt: ::core::ffi::c_int,
            XF_array: *mut *mut f64,
            ldxf: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverRfBatchZeroPivot: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
            position: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverRfCreate: unsafe extern "C" fn(
            handle: *mut cusolverRfHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverRfDestroy: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverRfExtractBundledFactorsHost: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
            h_nnzM: *mut ::core::ffi::c_int,
            h_Mp: *mut *mut ::core::ffi::c_int,
            h_Mi: *mut *mut ::core::ffi::c_int,
            h_Mx: *mut *mut f64,
        ) -> cusolverStatus_t,
        pub cusolverRfExtractSplitFactorsHost: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
            h_nnzL: *mut ::core::ffi::c_int,
            h_csrRowPtrL: *mut *mut ::core::ffi::c_int,
            h_csrColIndL: *mut *mut ::core::ffi::c_int,
            h_csrValL: *mut *mut f64,
            h_nnzU: *mut ::core::ffi::c_int,
            h_csrRowPtrU: *mut *mut ::core::ffi::c_int,
            h_csrColIndU: *mut *mut ::core::ffi::c_int,
            h_csrValU: *mut *mut f64,
        ) -> cusolverStatus_t,
        pub cusolverRfGetAlgs: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
            factAlg: *mut cusolverRfFactorization_t,
            solveAlg: *mut cusolverRfTriangularSolve_t,
        ) -> cusolverStatus_t,
        pub cusolverRfGetMatrixFormat: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
            format: *mut cusolverRfMatrixFormat_t,
            diag: *mut cusolverRfUnitDiagonal_t,
        ) -> cusolverStatus_t,
        pub cusolverRfGetNumericBoostReport: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
            report: *mut cusolverRfNumericBoostReport_t,
        ) -> cusolverStatus_t,
        pub cusolverRfGetNumericProperties: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
            zero: *mut f64,
            boost: *mut f64,
        ) -> cusolverStatus_t,
        pub cusolverRfGetResetValuesFastMode: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
            fastMode: *mut cusolverRfResetValuesFastMode_t,
        ) -> cusolverStatus_t,
        pub cusolverRfRefactor: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverRfResetValues: unsafe extern "C" fn(
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            csrRowPtrA: *mut ::core::ffi::c_int,
            csrColIndA: *mut ::core::ffi::c_int,
            csrValA: *mut f64,
            P: *mut ::core::ffi::c_int,
            Q: *mut ::core::ffi::c_int,
            handle: cusolverRfHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverRfSetAlgs: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
            factAlg: cusolverRfFactorization_t,
            solveAlg: cusolverRfTriangularSolve_t,
        ) -> cusolverStatus_t,
        pub cusolverRfSetMatrixFormat: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
            format: cusolverRfMatrixFormat_t,
            diag: cusolverRfUnitDiagonal_t,
        ) -> cusolverStatus_t,
        pub cusolverRfSetNumericProperties: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
            zero: f64,
            boost: f64,
        ) -> cusolverStatus_t,
        pub cusolverRfSetResetValuesFastMode: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
            fastMode: cusolverRfResetValuesFastMode_t,
        ) -> cusolverStatus_t,
        pub cusolverRfSetupDevice: unsafe extern "C" fn(
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            csrRowPtrA: *mut ::core::ffi::c_int,
            csrColIndA: *mut ::core::ffi::c_int,
            csrValA: *mut f64,
            nnzL: ::core::ffi::c_int,
            csrRowPtrL: *mut ::core::ffi::c_int,
            csrColIndL: *mut ::core::ffi::c_int,
            csrValL: *mut f64,
            nnzU: ::core::ffi::c_int,
            csrRowPtrU: *mut ::core::ffi::c_int,
            csrColIndU: *mut ::core::ffi::c_int,
            csrValU: *mut f64,
            P: *mut ::core::ffi::c_int,
            Q: *mut ::core::ffi::c_int,
            handle: cusolverRfHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverRfSetupHost: unsafe extern "C" fn(
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            h_csrRowPtrA: *mut ::core::ffi::c_int,
            h_csrColIndA: *mut ::core::ffi::c_int,
            h_csrValA: *mut f64,
            nnzL: ::core::ffi::c_int,
            h_csrRowPtrL: *mut ::core::ffi::c_int,
            h_csrColIndL: *mut ::core::ffi::c_int,
            h_csrValL: *mut f64,
            nnzU: ::core::ffi::c_int,
            h_csrRowPtrU: *mut ::core::ffi::c_int,
            h_csrColIndU: *mut ::core::ffi::c_int,
            h_csrValU: *mut f64,
            h_P: *mut ::core::ffi::c_int,
            h_Q: *mut ::core::ffi::c_int,
            handle: cusolverRfHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverRfSolve: unsafe extern "C" fn(
            handle: cusolverRfHandle_t,
            P: *mut ::core::ffi::c_int,
            Q: *mut ::core::ffi::c_int,
            nrhs: ::core::ffi::c_int,
            Temp: *mut f64,
            ldt: ::core::ffi::c_int,
            XF: *mut f64,
            ldxf: ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpCcsreigsHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            left_bottom_corner: cuComplex,
            right_upper_corner: cuComplex,
            num_eigs: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpCcsreigvsi: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            mu0: cuComplex,
            x0: *const cuComplex,
            maxite: ::core::ffi::c_int,
            eps: f32,
            mu: *mut cuComplex,
            x: *mut cuComplex,
        ) -> cusolverStatus_t,
        pub cusolverSpCcsreigvsiHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            mu0: cuComplex,
            x0: *const cuComplex,
            maxite: ::core::ffi::c_int,
            tol: f32,
            mu: *mut cuComplex,
            x: *mut cuComplex,
        ) -> cusolverStatus_t,
        pub cusolverSpCcsrlsqvqrHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const cuComplex,
            tol: f32,
            rankA: *mut ::core::ffi::c_int,
            x: *mut cuComplex,
            p: *mut ::core::ffi::c_int,
            min_norm: *mut f32,
        ) -> cusolverStatus_t,
        pub cusolverSpCcsrlsvchol: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const cuComplex,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            b: *const cuComplex,
            tol: f32,
            reorder: ::core::ffi::c_int,
            x: *mut cuComplex,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpCcsrlsvcholHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const cuComplex,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            b: *const cuComplex,
            tol: f32,
            reorder: ::core::ffi::c_int,
            x: *mut cuComplex,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpCcsrlsvluHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const cuComplex,
            tol: f32,
            reorder: ::core::ffi::c_int,
            x: *mut cuComplex,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpCcsrlsvqr: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const cuComplex,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            b: *const cuComplex,
            tol: f32,
            reorder: ::core::ffi::c_int,
            x: *mut cuComplex,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpCcsrlsvqrHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const cuComplex,
            tol: f32,
            reorder: ::core::ffi::c_int,
            x: *mut cuComplex,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpCcsrqrBufferInfoBatched: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const cuComplex,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
            info: csrqrInfo_t,
            internalDataInBytes: *mut usize,
            workspaceInBytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverSpCcsrqrsvBatched: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const cuComplex,
            x: *mut cuComplex,
            batchSize: ::core::ffi::c_int,
            info: csrqrInfo_t,
            pBuffer: *mut ::core::ffi::c_void,
        ) -> cusolverStatus_t,
        pub cusolverSpCcsrzfdHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            P: *mut ::core::ffi::c_int,
            numnz: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpCreate: unsafe extern "C" fn(
            handle: *mut cusolverSpHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverSpCreateCsrqrInfo: unsafe extern "C" fn(
            info: *mut csrqrInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverSpDcsreigsHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f64,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            left_bottom_corner: cuDoubleComplex,
            right_upper_corner: cuDoubleComplex,
            num_eigs: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpDcsreigvsi: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f64,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            mu0: f64,
            x0: *const f64,
            maxite: ::core::ffi::c_int,
            eps: f64,
            mu: *mut f64,
            x: *mut f64,
        ) -> cusolverStatus_t,
        pub cusolverSpDcsreigvsiHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f64,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            mu0: f64,
            x0: *const f64,
            maxite: ::core::ffi::c_int,
            tol: f64,
            mu: *mut f64,
            x: *mut f64,
        ) -> cusolverStatus_t,
        pub cusolverSpDcsrlsqvqrHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f64,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const f64,
            tol: f64,
            rankA: *mut ::core::ffi::c_int,
            x: *mut f64,
            p: *mut ::core::ffi::c_int,
            min_norm: *mut f64,
        ) -> cusolverStatus_t,
        pub cusolverSpDcsrlsvchol: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const f64,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            b: *const f64,
            tol: f64,
            reorder: ::core::ffi::c_int,
            x: *mut f64,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpDcsrlsvcholHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const f64,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            b: *const f64,
            tol: f64,
            reorder: ::core::ffi::c_int,
            x: *mut f64,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpDcsrlsvluHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f64,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const f64,
            tol: f64,
            reorder: ::core::ffi::c_int,
            x: *mut f64,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpDcsrlsvqr: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const f64,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            b: *const f64,
            tol: f64,
            reorder: ::core::ffi::c_int,
            x: *mut f64,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpDcsrlsvqrHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f64,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const f64,
            tol: f64,
            reorder: ::core::ffi::c_int,
            x: *mut f64,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpDcsrqrBufferInfoBatched: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const f64,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
            info: csrqrInfo_t,
            internalDataInBytes: *mut usize,
            workspaceInBytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverSpDcsrqrsvBatched: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f64,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const f64,
            x: *mut f64,
            batchSize: ::core::ffi::c_int,
            info: csrqrInfo_t,
            pBuffer: *mut ::core::ffi::c_void,
        ) -> cusolverStatus_t,
        pub cusolverSpDcsrzfdHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f64,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            P: *mut ::core::ffi::c_int,
            numnz: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpDestroy: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
        ) -> cusolverStatus_t,
        pub cusolverSpDestroyCsrqrInfo: unsafe extern "C" fn(
            info: csrqrInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverSpGetStream: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            streamId: *mut cudaStream_t,
        ) -> cusolverStatus_t,
        pub cusolverSpScsreigsHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f32,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            left_bottom_corner: cuComplex,
            right_upper_corner: cuComplex,
            num_eigs: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpScsreigvsi: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f32,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            mu0: f32,
            x0: *const f32,
            maxite: ::core::ffi::c_int,
            eps: f32,
            mu: *mut f32,
            x: *mut f32,
        ) -> cusolverStatus_t,
        pub cusolverSpScsreigvsiHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f32,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            mu0: f32,
            x0: *const f32,
            maxite: ::core::ffi::c_int,
            tol: f32,
            mu: *mut f32,
            x: *mut f32,
        ) -> cusolverStatus_t,
        pub cusolverSpScsrlsqvqrHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f32,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const f32,
            tol: f32,
            rankA: *mut ::core::ffi::c_int,
            x: *mut f32,
            p: *mut ::core::ffi::c_int,
            min_norm: *mut f32,
        ) -> cusolverStatus_t,
        pub cusolverSpScsrlsvchol: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const f32,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            b: *const f32,
            tol: f32,
            reorder: ::core::ffi::c_int,
            x: *mut f32,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpScsrlsvcholHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const f32,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            b: *const f32,
            tol: f32,
            reorder: ::core::ffi::c_int,
            x: *mut f32,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpScsrlsvluHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f32,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const f32,
            tol: f32,
            reorder: ::core::ffi::c_int,
            x: *mut f32,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpScsrlsvqr: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const f32,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            b: *const f32,
            tol: f32,
            reorder: ::core::ffi::c_int,
            x: *mut f32,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpScsrlsvqrHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f32,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const f32,
            tol: f32,
            reorder: ::core::ffi::c_int,
            x: *mut f32,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpScsrqrBufferInfoBatched: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const f32,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
            info: csrqrInfo_t,
            internalDataInBytes: *mut usize,
            workspaceInBytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverSpScsrqrsvBatched: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f32,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const f32,
            x: *mut f32,
            batchSize: ::core::ffi::c_int,
            info: csrqrInfo_t,
            pBuffer: *mut ::core::ffi::c_void,
        ) -> cusolverStatus_t,
        pub cusolverSpScsrzfdHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const f32,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            P: *mut ::core::ffi::c_int,
            numnz: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpSetStream: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            streamId: cudaStream_t,
        ) -> cusolverStatus_t,
        pub cusolverSpXcsrissymHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrEndPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            issym: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpXcsrmetisndHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            options: *const i64,
            p: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpXcsrpermHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrRowPtrA: *mut ::core::ffi::c_int,
            csrColIndA: *mut ::core::ffi::c_int,
            p: *const ::core::ffi::c_int,
            q: *const ::core::ffi::c_int,
            map: *mut ::core::ffi::c_int,
            pBuffer: *mut ::core::ffi::c_void,
        ) -> cusolverStatus_t,
        pub cusolverSpXcsrperm_bufferSizeHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            p: *const ::core::ffi::c_int,
            q: *const ::core::ffi::c_int,
            bufferSizeInBytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverSpXcsrqrAnalysisBatched: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            info: csrqrInfo_t,
        ) -> cusolverStatus_t,
        pub cusolverSpXcsrsymamdHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            p: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpXcsrsymmdqHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            p: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpXcsrsymrcmHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            p: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpZcsreigsHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuDoubleComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            left_bottom_corner: cuDoubleComplex,
            right_upper_corner: cuDoubleComplex,
            num_eigs: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpZcsreigvsi: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuDoubleComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            mu0: cuDoubleComplex,
            x0: *const cuDoubleComplex,
            maxite: ::core::ffi::c_int,
            eps: f64,
            mu: *mut cuDoubleComplex,
            x: *mut cuDoubleComplex,
        ) -> cusolverStatus_t,
        pub cusolverSpZcsreigvsiHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuDoubleComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            mu0: cuDoubleComplex,
            x0: *const cuDoubleComplex,
            maxite: ::core::ffi::c_int,
            tol: f64,
            mu: *mut cuDoubleComplex,
            x: *mut cuDoubleComplex,
        ) -> cusolverStatus_t,
        pub cusolverSpZcsrlsqvqrHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuDoubleComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const cuDoubleComplex,
            tol: f64,
            rankA: *mut ::core::ffi::c_int,
            x: *mut cuDoubleComplex,
            p: *mut ::core::ffi::c_int,
            min_norm: *mut f64,
        ) -> cusolverStatus_t,
        pub cusolverSpZcsrlsvchol: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const cuDoubleComplex,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            b: *const cuDoubleComplex,
            tol: f64,
            reorder: ::core::ffi::c_int,
            x: *mut cuDoubleComplex,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpZcsrlsvcholHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const cuDoubleComplex,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            b: *const cuDoubleComplex,
            tol: f64,
            reorder: ::core::ffi::c_int,
            x: *mut cuDoubleComplex,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpZcsrlsvluHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            n: ::core::ffi::c_int,
            nnzA: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuDoubleComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const cuDoubleComplex,
            tol: f64,
            reorder: ::core::ffi::c_int,
            x: *mut cuDoubleComplex,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpZcsrlsvqr: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const cuDoubleComplex,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            b: *const cuDoubleComplex,
            tol: f64,
            reorder: ::core::ffi::c_int,
            x: *mut cuDoubleComplex,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpZcsrlsvqrHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuDoubleComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const cuDoubleComplex,
            tol: f64,
            reorder: ::core::ffi::c_int,
            x: *mut cuDoubleComplex,
            singularity: *mut ::core::ffi::c_int,
        ) -> cusolverStatus_t,
        pub cusolverSpZcsrqrBufferInfoBatched: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrVal: *const cuDoubleComplex,
            csrRowPtr: *const ::core::ffi::c_int,
            csrColInd: *const ::core::ffi::c_int,
            batchSize: ::core::ffi::c_int,
            info: csrqrInfo_t,
            internalDataInBytes: *mut usize,
            workspaceInBytes: *mut usize,
        ) -> cusolverStatus_t,
        pub cusolverSpZcsrqrsvBatched: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            m: ::core::ffi::c_int,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuDoubleComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            b: *const cuDoubleComplex,
            x: *mut cuDoubleComplex,
            batchSize: ::core::ffi::c_int,
            info: csrqrInfo_t,
            pBuffer: *mut ::core::ffi::c_void,
        ) -> cusolverStatus_t,
        pub cusolverSpZcsrzfdHost: unsafe extern "C" fn(
            handle: cusolverSpHandle_t,
            n: ::core::ffi::c_int,
            nnz: ::core::ffi::c_int,
            descrA: cusparseMatDescr_t,
            csrValA: *const cuDoubleComplex,
            csrRowPtrA: *const ::core::ffi::c_int,
            csrColIndA: *const ::core::ffi::c_int,
            P: *mut ::core::ffi::c_int,
            numnz: *mut ::core::ffi::c_int,
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
            let cusolverDnCCgels = __library
                .get(b"cusolverDnCCgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCCgels_bufferSize = __library
                .get(b"cusolverDnCCgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCCgesv = __library
                .get(b"cusolverDnCCgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCCgesv_bufferSize = __library
                .get(b"cusolverDnCCgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCEgels = __library
                .get(b"cusolverDnCEgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCEgels_bufferSize = __library
                .get(b"cusolverDnCEgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCEgesv = __library
                .get(b"cusolverDnCEgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCEgesv_bufferSize = __library
                .get(b"cusolverDnCEgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCKgels = __library
                .get(b"cusolverDnCKgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCKgels_bufferSize = __library
                .get(b"cusolverDnCKgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCKgesv = __library
                .get(b"cusolverDnCKgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCKgesv_bufferSize = __library
                .get(b"cusolverDnCKgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCYgels = __library
                .get(b"cusolverDnCYgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCYgels_bufferSize = __library
                .get(b"cusolverDnCYgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCYgesv = __library
                .get(b"cusolverDnCYgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCYgesv_bufferSize = __library
                .get(b"cusolverDnCYgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCgebrd = __library
                .get(b"cusolverDnCgebrd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCgebrd_bufferSize = __library
                .get(b"cusolverDnCgebrd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCgeqrf = __library
                .get(b"cusolverDnCgeqrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCgeqrf_bufferSize = __library
                .get(b"cusolverDnCgeqrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCgesvd = __library
                .get(b"cusolverDnCgesvd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCgesvd_bufferSize = __library
                .get(b"cusolverDnCgesvd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCgesvdaStridedBatched = __library
                .get(b"cusolverDnCgesvdaStridedBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCgesvdaStridedBatched_bufferSize = __library
                .get(b"cusolverDnCgesvdaStridedBatched_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCgesvdj = __library
                .get(b"cusolverDnCgesvdj\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCgesvdjBatched = __library
                .get(b"cusolverDnCgesvdjBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCgesvdjBatched_bufferSize = __library
                .get(b"cusolverDnCgesvdjBatched_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCgesvdj_bufferSize = __library
                .get(b"cusolverDnCgesvdj_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCgetrf = __library
                .get(b"cusolverDnCgetrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCgetrf_bufferSize = __library
                .get(b"cusolverDnCgetrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCgetrs = __library
                .get(b"cusolverDnCgetrs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCheevd = __library
                .get(b"cusolverDnCheevd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCheevd_bufferSize = __library
                .get(b"cusolverDnCheevd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCheevdx = __library
                .get(b"cusolverDnCheevdx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCheevdx_bufferSize = __library
                .get(b"cusolverDnCheevdx_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCheevj = __library
                .get(b"cusolverDnCheevj\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCheevjBatched = __library
                .get(b"cusolverDnCheevjBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCheevjBatched_bufferSize = __library
                .get(b"cusolverDnCheevjBatched_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCheevj_bufferSize = __library
                .get(b"cusolverDnCheevj_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnChegvd = __library
                .get(b"cusolverDnChegvd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnChegvd_bufferSize = __library
                .get(b"cusolverDnChegvd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnChegvdx = __library
                .get(b"cusolverDnChegvdx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnChegvdx_bufferSize = __library
                .get(b"cusolverDnChegvdx_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnChegvj = __library
                .get(b"cusolverDnChegvj\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnChegvj_bufferSize = __library
                .get(b"cusolverDnChegvj_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnChetrd = __library
                .get(b"cusolverDnChetrd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnChetrd_bufferSize = __library
                .get(b"cusolverDnChetrd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnClaswp = __library
                .get(b"cusolverDnClaswp\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnClauum = __library
                .get(b"cusolverDnClauum\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnClauum_bufferSize = __library
                .get(b"cusolverDnClauum_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCpotrf = __library
                .get(b"cusolverDnCpotrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCpotrfBatched = __library
                .get(b"cusolverDnCpotrfBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCpotrf_bufferSize = __library
                .get(b"cusolverDnCpotrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCpotri = __library
                .get(b"cusolverDnCpotri\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCpotri_bufferSize = __library
                .get(b"cusolverDnCpotri_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCpotrs = __library
                .get(b"cusolverDnCpotrs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCpotrsBatched = __library
                .get(b"cusolverDnCpotrsBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCreate = __library
                .get(b"cusolverDnCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCreateGesvdjInfo = __library
                .get(b"cusolverDnCreateGesvdjInfo\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCreateParams = __library
                .get(b"cusolverDnCreateParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCreateSyevjInfo = __library
                .get(b"cusolverDnCreateSyevjInfo\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCsytrf = __library
                .get(b"cusolverDnCsytrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCsytrf_bufferSize = __library
                .get(b"cusolverDnCsytrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCsytri = __library
                .get(b"cusolverDnCsytri\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCsytri_bufferSize = __library
                .get(b"cusolverDnCsytri_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCungbr = __library
                .get(b"cusolverDnCungbr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCungbr_bufferSize = __library
                .get(b"cusolverDnCungbr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCungqr = __library
                .get(b"cusolverDnCungqr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCungqr_bufferSize = __library
                .get(b"cusolverDnCungqr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCungtr = __library
                .get(b"cusolverDnCungtr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCungtr_bufferSize = __library
                .get(b"cusolverDnCungtr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCunmqr = __library
                .get(b"cusolverDnCunmqr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCunmqr_bufferSize = __library
                .get(b"cusolverDnCunmqr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCunmtr = __library
                .get(b"cusolverDnCunmtr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnCunmtr_bufferSize = __library
                .get(b"cusolverDnCunmtr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDBgels = __library
                .get(b"cusolverDnDBgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDBgels_bufferSize = __library
                .get(b"cusolverDnDBgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDBgesv = __library
                .get(b"cusolverDnDBgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDBgesv_bufferSize = __library
                .get(b"cusolverDnDBgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDDgels = __library
                .get(b"cusolverDnDDgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDDgels_bufferSize = __library
                .get(b"cusolverDnDDgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDDgesv = __library
                .get(b"cusolverDnDDgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDDgesv_bufferSize = __library
                .get(b"cusolverDnDDgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDHgels = __library
                .get(b"cusolverDnDHgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDHgels_bufferSize = __library
                .get(b"cusolverDnDHgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDHgesv = __library
                .get(b"cusolverDnDHgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDHgesv_bufferSize = __library
                .get(b"cusolverDnDHgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDSgels = __library
                .get(b"cusolverDnDSgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDSgels_bufferSize = __library
                .get(b"cusolverDnDSgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDSgesv = __library
                .get(b"cusolverDnDSgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDSgesv_bufferSize = __library
                .get(b"cusolverDnDSgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDXgels = __library
                .get(b"cusolverDnDXgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDXgels_bufferSize = __library
                .get(b"cusolverDnDXgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDXgesv = __library
                .get(b"cusolverDnDXgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDXgesv_bufferSize = __library
                .get(b"cusolverDnDXgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDestroy = __library
                .get(b"cusolverDnDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDestroyGesvdjInfo = __library
                .get(b"cusolverDnDestroyGesvdjInfo\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDestroyParams = __library
                .get(b"cusolverDnDestroyParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDestroySyevjInfo = __library
                .get(b"cusolverDnDestroySyevjInfo\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDgebrd = __library
                .get(b"cusolverDnDgebrd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDgebrd_bufferSize = __library
                .get(b"cusolverDnDgebrd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDgeqrf = __library
                .get(b"cusolverDnDgeqrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDgeqrf_bufferSize = __library
                .get(b"cusolverDnDgeqrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDgesvd = __library
                .get(b"cusolverDnDgesvd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDgesvd_bufferSize = __library
                .get(b"cusolverDnDgesvd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDgesvdaStridedBatched = __library
                .get(b"cusolverDnDgesvdaStridedBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDgesvdaStridedBatched_bufferSize = __library
                .get(b"cusolverDnDgesvdaStridedBatched_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDgesvdj = __library
                .get(b"cusolverDnDgesvdj\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDgesvdjBatched = __library
                .get(b"cusolverDnDgesvdjBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDgesvdjBatched_bufferSize = __library
                .get(b"cusolverDnDgesvdjBatched_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDgesvdj_bufferSize = __library
                .get(b"cusolverDnDgesvdj_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDgetrf = __library
                .get(b"cusolverDnDgetrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDgetrf_bufferSize = __library
                .get(b"cusolverDnDgetrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDgetrs = __library
                .get(b"cusolverDnDgetrs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDlaswp = __library
                .get(b"cusolverDnDlaswp\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDlauum = __library
                .get(b"cusolverDnDlauum\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDlauum_bufferSize = __library
                .get(b"cusolverDnDlauum_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDorgbr = __library
                .get(b"cusolverDnDorgbr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDorgbr_bufferSize = __library
                .get(b"cusolverDnDorgbr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDorgqr = __library
                .get(b"cusolverDnDorgqr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDorgqr_bufferSize = __library
                .get(b"cusolverDnDorgqr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDorgtr = __library
                .get(b"cusolverDnDorgtr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDorgtr_bufferSize = __library
                .get(b"cusolverDnDorgtr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDormqr = __library
                .get(b"cusolverDnDormqr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDormqr_bufferSize = __library
                .get(b"cusolverDnDormqr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDormtr = __library
                .get(b"cusolverDnDormtr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDormtr_bufferSize = __library
                .get(b"cusolverDnDormtr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDpotrf = __library
                .get(b"cusolverDnDpotrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDpotrfBatched = __library
                .get(b"cusolverDnDpotrfBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDpotrf_bufferSize = __library
                .get(b"cusolverDnDpotrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDpotri = __library
                .get(b"cusolverDnDpotri\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDpotri_bufferSize = __library
                .get(b"cusolverDnDpotri_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDpotrs = __library
                .get(b"cusolverDnDpotrs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDpotrsBatched = __library
                .get(b"cusolverDnDpotrsBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsyevd = __library
                .get(b"cusolverDnDsyevd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsyevd_bufferSize = __library
                .get(b"cusolverDnDsyevd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsyevdx = __library
                .get(b"cusolverDnDsyevdx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsyevdx_bufferSize = __library
                .get(b"cusolverDnDsyevdx_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsyevj = __library
                .get(b"cusolverDnDsyevj\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsyevjBatched = __library
                .get(b"cusolverDnDsyevjBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsyevjBatched_bufferSize = __library
                .get(b"cusolverDnDsyevjBatched_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsyevj_bufferSize = __library
                .get(b"cusolverDnDsyevj_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsygvd = __library
                .get(b"cusolverDnDsygvd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsygvd_bufferSize = __library
                .get(b"cusolverDnDsygvd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsygvdx = __library
                .get(b"cusolverDnDsygvdx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsygvdx_bufferSize = __library
                .get(b"cusolverDnDsygvdx_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsygvj = __library
                .get(b"cusolverDnDsygvj\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsygvj_bufferSize = __library
                .get(b"cusolverDnDsygvj_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsytrd = __library
                .get(b"cusolverDnDsytrd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsytrd_bufferSize = __library
                .get(b"cusolverDnDsytrd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsytrf = __library
                .get(b"cusolverDnDsytrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsytrf_bufferSize = __library
                .get(b"cusolverDnDsytrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsytri = __library
                .get(b"cusolverDnDsytri\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnDsytri_bufferSize = __library
                .get(b"cusolverDnDsytri_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnGeqrf = __library
                .get(b"cusolverDnGeqrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnGeqrf_bufferSize = __library
                .get(b"cusolverDnGeqrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnGesvd = __library
                .get(b"cusolverDnGesvd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnGesvd_bufferSize = __library
                .get(b"cusolverDnGesvd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cusolverDnGetDeterministicMode = __library
                .get(b"cusolverDnGetDeterministicMode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnGetStream = __library
                .get(b"cusolverDnGetStream\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnGetrf = __library
                .get(b"cusolverDnGetrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnGetrf_bufferSize = __library
                .get(b"cusolverDnGetrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnGetrs = __library
                .get(b"cusolverDnGetrs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSInfosCreate = __library
                .get(b"cusolverDnIRSInfosCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSInfosDestroy = __library
                .get(b"cusolverDnIRSInfosDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSInfosGetMaxIters = __library
                .get(b"cusolverDnIRSInfosGetMaxIters\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSInfosGetNiters = __library
                .get(b"cusolverDnIRSInfosGetNiters\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSInfosGetOuterNiters = __library
                .get(b"cusolverDnIRSInfosGetOuterNiters\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSInfosGetResidualHistory = __library
                .get(b"cusolverDnIRSInfosGetResidualHistory\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSInfosRequestResidual = __library
                .get(b"cusolverDnIRSInfosRequestResidual\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSParamsCreate = __library
                .get(b"cusolverDnIRSParamsCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSParamsDestroy = __library
                .get(b"cusolverDnIRSParamsDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSParamsDisableFallback = __library
                .get(b"cusolverDnIRSParamsDisableFallback\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSParamsEnableFallback = __library
                .get(b"cusolverDnIRSParamsEnableFallback\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSParamsGetMaxIters = __library
                .get(b"cusolverDnIRSParamsGetMaxIters\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSParamsSetMaxIters = __library
                .get(b"cusolverDnIRSParamsSetMaxIters\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSParamsSetMaxItersInner = __library
                .get(b"cusolverDnIRSParamsSetMaxItersInner\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSParamsSetRefinementSolver = __library
                .get(b"cusolverDnIRSParamsSetRefinementSolver\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSParamsSetSolverLowestPrecision = __library
                .get(b"cusolverDnIRSParamsSetSolverLowestPrecision\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSParamsSetSolverMainPrecision = __library
                .get(b"cusolverDnIRSParamsSetSolverMainPrecision\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSParamsSetSolverPrecisions = __library
                .get(b"cusolverDnIRSParamsSetSolverPrecisions\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSParamsSetTol = __library
                .get(b"cusolverDnIRSParamsSetTol\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSParamsSetTolInner = __library
                .get(b"cusolverDnIRSParamsSetTolInner\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSXgels = __library
                .get(b"cusolverDnIRSXgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSXgels_bufferSize = __library
                .get(b"cusolverDnIRSXgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSXgesv = __library
                .get(b"cusolverDnIRSXgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnIRSXgesv_bufferSize = __library
                .get(b"cusolverDnIRSXgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11070",
                    feature = "cuda-11080",
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cusolverDnLoggerForceDisable = __library
                .get(b"cusolverDnLoggerForceDisable\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11070",
                    feature = "cuda-11080",
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cusolverDnLoggerOpenFile = __library
                .get(b"cusolverDnLoggerOpenFile\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11070",
                    feature = "cuda-11080",
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cusolverDnLoggerSetCallback = __library
                .get(b"cusolverDnLoggerSetCallback\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11070",
                    feature = "cuda-11080",
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cusolverDnLoggerSetFile = __library
                .get(b"cusolverDnLoggerSetFile\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11070",
                    feature = "cuda-11080",
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cusolverDnLoggerSetLevel = __library
                .get(b"cusolverDnLoggerSetLevel\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11070",
                    feature = "cuda-11080",
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cusolverDnLoggerSetMask = __library
                .get(b"cusolverDnLoggerSetMask\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnPotrf = __library
                .get(b"cusolverDnPotrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnPotrf_bufferSize = __library
                .get(b"cusolverDnPotrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnPotrs = __library
                .get(b"cusolverDnPotrs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSBgels = __library
                .get(b"cusolverDnSBgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSBgels_bufferSize = __library
                .get(b"cusolverDnSBgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSBgesv = __library
                .get(b"cusolverDnSBgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSBgesv_bufferSize = __library
                .get(b"cusolverDnSBgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSHgels = __library
                .get(b"cusolverDnSHgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSHgels_bufferSize = __library
                .get(b"cusolverDnSHgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSHgesv = __library
                .get(b"cusolverDnSHgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSHgesv_bufferSize = __library
                .get(b"cusolverDnSHgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSSgels = __library
                .get(b"cusolverDnSSgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSSgels_bufferSize = __library
                .get(b"cusolverDnSSgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSSgesv = __library
                .get(b"cusolverDnSSgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSSgesv_bufferSize = __library
                .get(b"cusolverDnSSgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSXgels = __library
                .get(b"cusolverDnSXgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSXgels_bufferSize = __library
                .get(b"cusolverDnSXgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSXgesv = __library
                .get(b"cusolverDnSXgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSXgesv_bufferSize = __library
                .get(b"cusolverDnSXgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSetAdvOptions = __library
                .get(b"cusolverDnSetAdvOptions\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cusolverDnSetDeterministicMode = __library
                .get(b"cusolverDnSetDeterministicMode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSetStream = __library
                .get(b"cusolverDnSetStream\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSgebrd = __library
                .get(b"cusolverDnSgebrd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSgebrd_bufferSize = __library
                .get(b"cusolverDnSgebrd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSgeqrf = __library
                .get(b"cusolverDnSgeqrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSgeqrf_bufferSize = __library
                .get(b"cusolverDnSgeqrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSgesvd = __library
                .get(b"cusolverDnSgesvd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSgesvd_bufferSize = __library
                .get(b"cusolverDnSgesvd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSgesvdaStridedBatched = __library
                .get(b"cusolverDnSgesvdaStridedBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSgesvdaStridedBatched_bufferSize = __library
                .get(b"cusolverDnSgesvdaStridedBatched_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSgesvdj = __library
                .get(b"cusolverDnSgesvdj\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSgesvdjBatched = __library
                .get(b"cusolverDnSgesvdjBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSgesvdjBatched_bufferSize = __library
                .get(b"cusolverDnSgesvdjBatched_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSgesvdj_bufferSize = __library
                .get(b"cusolverDnSgesvdj_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSgetrf = __library
                .get(b"cusolverDnSgetrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSgetrf_bufferSize = __library
                .get(b"cusolverDnSgetrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSgetrs = __library
                .get(b"cusolverDnSgetrs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSlaswp = __library
                .get(b"cusolverDnSlaswp\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSlauum = __library
                .get(b"cusolverDnSlauum\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSlauum_bufferSize = __library
                .get(b"cusolverDnSlauum_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSorgbr = __library
                .get(b"cusolverDnSorgbr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSorgbr_bufferSize = __library
                .get(b"cusolverDnSorgbr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSorgqr = __library
                .get(b"cusolverDnSorgqr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSorgqr_bufferSize = __library
                .get(b"cusolverDnSorgqr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSorgtr = __library
                .get(b"cusolverDnSorgtr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSorgtr_bufferSize = __library
                .get(b"cusolverDnSorgtr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSormqr = __library
                .get(b"cusolverDnSormqr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSormqr_bufferSize = __library
                .get(b"cusolverDnSormqr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSormtr = __library
                .get(b"cusolverDnSormtr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSormtr_bufferSize = __library
                .get(b"cusolverDnSormtr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSpotrf = __library
                .get(b"cusolverDnSpotrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSpotrfBatched = __library
                .get(b"cusolverDnSpotrfBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSpotrf_bufferSize = __library
                .get(b"cusolverDnSpotrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSpotri = __library
                .get(b"cusolverDnSpotri\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSpotri_bufferSize = __library
                .get(b"cusolverDnSpotri_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSpotrs = __library
                .get(b"cusolverDnSpotrs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSpotrsBatched = __library
                .get(b"cusolverDnSpotrsBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsyevd = __library
                .get(b"cusolverDnSsyevd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsyevd_bufferSize = __library
                .get(b"cusolverDnSsyevd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsyevdx = __library
                .get(b"cusolverDnSsyevdx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsyevdx_bufferSize = __library
                .get(b"cusolverDnSsyevdx_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsyevj = __library
                .get(b"cusolverDnSsyevj\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsyevjBatched = __library
                .get(b"cusolverDnSsyevjBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsyevjBatched_bufferSize = __library
                .get(b"cusolverDnSsyevjBatched_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsyevj_bufferSize = __library
                .get(b"cusolverDnSsyevj_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsygvd = __library
                .get(b"cusolverDnSsygvd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsygvd_bufferSize = __library
                .get(b"cusolverDnSsygvd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsygvdx = __library
                .get(b"cusolverDnSsygvdx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsygvdx_bufferSize = __library
                .get(b"cusolverDnSsygvdx_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsygvj = __library
                .get(b"cusolverDnSsygvj\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsygvj_bufferSize = __library
                .get(b"cusolverDnSsygvj_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsytrd = __library
                .get(b"cusolverDnSsytrd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsytrd_bufferSize = __library
                .get(b"cusolverDnSsytrd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsytrf = __library
                .get(b"cusolverDnSsytrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsytrf_bufferSize = __library
                .get(b"cusolverDnSsytrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsytri = __library
                .get(b"cusolverDnSsytri\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSsytri_bufferSize = __library
                .get(b"cusolverDnSsytri_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSyevd = __library
                .get(b"cusolverDnSyevd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSyevd_bufferSize = __library
                .get(b"cusolverDnSyevd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSyevdx = __library
                .get(b"cusolverDnSyevdx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnSyevdx_bufferSize = __library
                .get(b"cusolverDnSyevdx_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
            let cusolverDnXgeev = __library
                .get(b"cusolverDnXgeev\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
            let cusolverDnXgeev_bufferSize = __library
                .get(b"cusolverDnXgeev_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgeqrf = __library
                .get(b"cusolverDnXgeqrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgeqrf_bufferSize = __library
                .get(b"cusolverDnXgeqrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgesvd = __library
                .get(b"cusolverDnXgesvd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgesvd_bufferSize = __library
                .get(b"cusolverDnXgesvd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgesvdjGetResidual = __library
                .get(b"cusolverDnXgesvdjGetResidual\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgesvdjGetSweeps = __library
                .get(b"cusolverDnXgesvdjGetSweeps\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgesvdjSetMaxSweeps = __library
                .get(b"cusolverDnXgesvdjSetMaxSweeps\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgesvdjSetSortEig = __library
                .get(b"cusolverDnXgesvdjSetSortEig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgesvdjSetTolerance = __library
                .get(b"cusolverDnXgesvdjSetTolerance\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgesvdp = __library
                .get(b"cusolverDnXgesvdp\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgesvdp_bufferSize = __library
                .get(b"cusolverDnXgesvdp_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgesvdr = __library
                .get(b"cusolverDnXgesvdr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgesvdr_bufferSize = __library
                .get(b"cusolverDnXgesvdr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgetrf = __library
                .get(b"cusolverDnXgetrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgetrf_bufferSize = __library
                .get(b"cusolverDnXgetrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXgetrs = __library
                .get(b"cusolverDnXgetrs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12040"))]
            let cusolverDnXlarft = __library
                .get(b"cusolverDnXlarft\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cusolverDnXlarft = __library
                .get(b"cusolverDnXlarft\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12040"))]
            let cusolverDnXlarft_bufferSize = __library
                .get(b"cusolverDnXlarft_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cusolverDnXlarft_bufferSize = __library
                .get(b"cusolverDnXlarft_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXpotrf = __library
                .get(b"cusolverDnXpotrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXpotrf_bufferSize = __library
                .get(b"cusolverDnXpotrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXpotrs = __library
                .get(b"cusolverDnXpotrs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
            let cusolverDnXsyevBatched = __library
                .get(b"cusolverDnXsyevBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
            let cusolverDnXsyevBatched_bufferSize = __library
                .get(b"cusolverDnXsyevBatched_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXsyevd = __library
                .get(b"cusolverDnXsyevd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXsyevd_bufferSize = __library
                .get(b"cusolverDnXsyevd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXsyevdx = __library
                .get(b"cusolverDnXsyevdx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXsyevdx_bufferSize = __library
                .get(b"cusolverDnXsyevdx_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXsyevjGetResidual = __library
                .get(b"cusolverDnXsyevjGetResidual\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXsyevjGetSweeps = __library
                .get(b"cusolverDnXsyevjGetSweeps\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXsyevjSetMaxSweeps = __library
                .get(b"cusolverDnXsyevjSetMaxSweeps\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXsyevjSetSortEig = __library
                .get(b"cusolverDnXsyevjSetSortEig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXsyevjSetTolerance = __library
                .get(b"cusolverDnXsyevjSetTolerance\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXsytrs = __library
                .get(b"cusolverDnXsytrs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXsytrs_bufferSize = __library
                .get(b"cusolverDnXsytrs_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXtrtri = __library
                .get(b"cusolverDnXtrtri\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnXtrtri_bufferSize = __library
                .get(b"cusolverDnXtrtri_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZCgels = __library
                .get(b"cusolverDnZCgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZCgels_bufferSize = __library
                .get(b"cusolverDnZCgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZCgesv = __library
                .get(b"cusolverDnZCgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZCgesv_bufferSize = __library
                .get(b"cusolverDnZCgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZEgels = __library
                .get(b"cusolverDnZEgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZEgels_bufferSize = __library
                .get(b"cusolverDnZEgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZEgesv = __library
                .get(b"cusolverDnZEgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZEgesv_bufferSize = __library
                .get(b"cusolverDnZEgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZKgels = __library
                .get(b"cusolverDnZKgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZKgels_bufferSize = __library
                .get(b"cusolverDnZKgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZKgesv = __library
                .get(b"cusolverDnZKgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZKgesv_bufferSize = __library
                .get(b"cusolverDnZKgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZYgels = __library
                .get(b"cusolverDnZYgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZYgels_bufferSize = __library
                .get(b"cusolverDnZYgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZYgesv = __library
                .get(b"cusolverDnZYgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZYgesv_bufferSize = __library
                .get(b"cusolverDnZYgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZZgels = __library
                .get(b"cusolverDnZZgels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZZgels_bufferSize = __library
                .get(b"cusolverDnZZgels_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZZgesv = __library
                .get(b"cusolverDnZZgesv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZZgesv_bufferSize = __library
                .get(b"cusolverDnZZgesv_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZgebrd = __library
                .get(b"cusolverDnZgebrd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZgebrd_bufferSize = __library
                .get(b"cusolverDnZgebrd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZgeqrf = __library
                .get(b"cusolverDnZgeqrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZgeqrf_bufferSize = __library
                .get(b"cusolverDnZgeqrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZgesvd = __library
                .get(b"cusolverDnZgesvd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZgesvd_bufferSize = __library
                .get(b"cusolverDnZgesvd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZgesvdaStridedBatched = __library
                .get(b"cusolverDnZgesvdaStridedBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZgesvdaStridedBatched_bufferSize = __library
                .get(b"cusolverDnZgesvdaStridedBatched_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZgesvdj = __library
                .get(b"cusolverDnZgesvdj\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZgesvdjBatched = __library
                .get(b"cusolverDnZgesvdjBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZgesvdjBatched_bufferSize = __library
                .get(b"cusolverDnZgesvdjBatched_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZgesvdj_bufferSize = __library
                .get(b"cusolverDnZgesvdj_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZgetrf = __library
                .get(b"cusolverDnZgetrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZgetrf_bufferSize = __library
                .get(b"cusolverDnZgetrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZgetrs = __library
                .get(b"cusolverDnZgetrs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZheevd = __library
                .get(b"cusolverDnZheevd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZheevd_bufferSize = __library
                .get(b"cusolverDnZheevd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZheevdx = __library
                .get(b"cusolverDnZheevdx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZheevdx_bufferSize = __library
                .get(b"cusolverDnZheevdx_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZheevj = __library
                .get(b"cusolverDnZheevj\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZheevjBatched = __library
                .get(b"cusolverDnZheevjBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZheevjBatched_bufferSize = __library
                .get(b"cusolverDnZheevjBatched_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZheevj_bufferSize = __library
                .get(b"cusolverDnZheevj_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZhegvd = __library
                .get(b"cusolverDnZhegvd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZhegvd_bufferSize = __library
                .get(b"cusolverDnZhegvd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZhegvdx = __library
                .get(b"cusolverDnZhegvdx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZhegvdx_bufferSize = __library
                .get(b"cusolverDnZhegvdx_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZhegvj = __library
                .get(b"cusolverDnZhegvj\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZhegvj_bufferSize = __library
                .get(b"cusolverDnZhegvj_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZhetrd = __library
                .get(b"cusolverDnZhetrd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZhetrd_bufferSize = __library
                .get(b"cusolverDnZhetrd_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZlaswp = __library
                .get(b"cusolverDnZlaswp\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZlauum = __library
                .get(b"cusolverDnZlauum\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZlauum_bufferSize = __library
                .get(b"cusolverDnZlauum_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZpotrf = __library
                .get(b"cusolverDnZpotrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZpotrfBatched = __library
                .get(b"cusolverDnZpotrfBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZpotrf_bufferSize = __library
                .get(b"cusolverDnZpotrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZpotri = __library
                .get(b"cusolverDnZpotri\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZpotri_bufferSize = __library
                .get(b"cusolverDnZpotri_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZpotrs = __library
                .get(b"cusolverDnZpotrs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZpotrsBatched = __library
                .get(b"cusolverDnZpotrsBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZsytrf = __library
                .get(b"cusolverDnZsytrf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZsytrf_bufferSize = __library
                .get(b"cusolverDnZsytrf_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZsytri = __library
                .get(b"cusolverDnZsytri\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZsytri_bufferSize = __library
                .get(b"cusolverDnZsytri_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZungbr = __library
                .get(b"cusolverDnZungbr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZungbr_bufferSize = __library
                .get(b"cusolverDnZungbr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZungqr = __library
                .get(b"cusolverDnZungqr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZungqr_bufferSize = __library
                .get(b"cusolverDnZungqr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZungtr = __library
                .get(b"cusolverDnZungtr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZungtr_bufferSize = __library
                .get(b"cusolverDnZungtr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZunmqr = __library
                .get(b"cusolverDnZunmqr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZunmqr_bufferSize = __library
                .get(b"cusolverDnZunmqr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZunmtr = __library
                .get(b"cusolverDnZunmtr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverDnZunmtr_bufferSize = __library
                .get(b"cusolverDnZunmtr_bufferSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverGetProperty = __library
                .get(b"cusolverGetProperty\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverGetVersion = __library
                .get(b"cusolverGetVersion\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
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
            let cusolverRfAccessBundledFactorsDevice = __library
                .get(b"cusolverRfAccessBundledFactorsDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfAnalyze = __library
                .get(b"cusolverRfAnalyze\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfBatchAnalyze = __library
                .get(b"cusolverRfBatchAnalyze\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfBatchRefactor = __library
                .get(b"cusolverRfBatchRefactor\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfBatchResetValues = __library
                .get(b"cusolverRfBatchResetValues\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfBatchSetupHost = __library
                .get(b"cusolverRfBatchSetupHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfBatchSolve = __library
                .get(b"cusolverRfBatchSolve\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfBatchZeroPivot = __library
                .get(b"cusolverRfBatchZeroPivot\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfCreate = __library
                .get(b"cusolverRfCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfDestroy = __library
                .get(b"cusolverRfDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfExtractBundledFactorsHost = __library
                .get(b"cusolverRfExtractBundledFactorsHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfExtractSplitFactorsHost = __library
                .get(b"cusolverRfExtractSplitFactorsHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfGetAlgs = __library
                .get(b"cusolverRfGetAlgs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfGetMatrixFormat = __library
                .get(b"cusolverRfGetMatrixFormat\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfGetNumericBoostReport = __library
                .get(b"cusolverRfGetNumericBoostReport\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfGetNumericProperties = __library
                .get(b"cusolverRfGetNumericProperties\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfGetResetValuesFastMode = __library
                .get(b"cusolverRfGetResetValuesFastMode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfRefactor = __library
                .get(b"cusolverRfRefactor\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfResetValues = __library
                .get(b"cusolverRfResetValues\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfSetAlgs = __library
                .get(b"cusolverRfSetAlgs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfSetMatrixFormat = __library
                .get(b"cusolverRfSetMatrixFormat\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfSetNumericProperties = __library
                .get(b"cusolverRfSetNumericProperties\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfSetResetValuesFastMode = __library
                .get(b"cusolverRfSetResetValuesFastMode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfSetupDevice = __library
                .get(b"cusolverRfSetupDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfSetupHost = __library
                .get(b"cusolverRfSetupHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverRfSolve = __library
                .get(b"cusolverRfSolve\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpCcsreigsHost = __library
                .get(b"cusolverSpCcsreigsHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpCcsreigvsi = __library
                .get(b"cusolverSpCcsreigvsi\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpCcsreigvsiHost = __library
                .get(b"cusolverSpCcsreigvsiHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpCcsrlsqvqrHost = __library
                .get(b"cusolverSpCcsrlsqvqrHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpCcsrlsvchol = __library
                .get(b"cusolverSpCcsrlsvchol\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpCcsrlsvcholHost = __library
                .get(b"cusolverSpCcsrlsvcholHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpCcsrlsvluHost = __library
                .get(b"cusolverSpCcsrlsvluHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpCcsrlsvqr = __library
                .get(b"cusolverSpCcsrlsvqr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpCcsrlsvqrHost = __library
                .get(b"cusolverSpCcsrlsvqrHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpCcsrqrBufferInfoBatched = __library
                .get(b"cusolverSpCcsrqrBufferInfoBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpCcsrqrsvBatched = __library
                .get(b"cusolverSpCcsrqrsvBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpCcsrzfdHost = __library
                .get(b"cusolverSpCcsrzfdHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpCreate = __library
                .get(b"cusolverSpCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpCreateCsrqrInfo = __library
                .get(b"cusolverSpCreateCsrqrInfo\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpDcsreigsHost = __library
                .get(b"cusolverSpDcsreigsHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpDcsreigvsi = __library
                .get(b"cusolverSpDcsreigvsi\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpDcsreigvsiHost = __library
                .get(b"cusolverSpDcsreigvsiHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpDcsrlsqvqrHost = __library
                .get(b"cusolverSpDcsrlsqvqrHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpDcsrlsvchol = __library
                .get(b"cusolverSpDcsrlsvchol\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpDcsrlsvcholHost = __library
                .get(b"cusolverSpDcsrlsvcholHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpDcsrlsvluHost = __library
                .get(b"cusolverSpDcsrlsvluHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpDcsrlsvqr = __library
                .get(b"cusolverSpDcsrlsvqr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpDcsrlsvqrHost = __library
                .get(b"cusolverSpDcsrlsvqrHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpDcsrqrBufferInfoBatched = __library
                .get(b"cusolverSpDcsrqrBufferInfoBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpDcsrqrsvBatched = __library
                .get(b"cusolverSpDcsrqrsvBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpDcsrzfdHost = __library
                .get(b"cusolverSpDcsrzfdHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpDestroy = __library
                .get(b"cusolverSpDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpDestroyCsrqrInfo = __library
                .get(b"cusolverSpDestroyCsrqrInfo\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpGetStream = __library
                .get(b"cusolverSpGetStream\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpScsreigsHost = __library
                .get(b"cusolverSpScsreigsHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpScsreigvsi = __library
                .get(b"cusolverSpScsreigvsi\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpScsreigvsiHost = __library
                .get(b"cusolverSpScsreigvsiHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpScsrlsqvqrHost = __library
                .get(b"cusolverSpScsrlsqvqrHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpScsrlsvchol = __library
                .get(b"cusolverSpScsrlsvchol\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpScsrlsvcholHost = __library
                .get(b"cusolverSpScsrlsvcholHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpScsrlsvluHost = __library
                .get(b"cusolverSpScsrlsvluHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpScsrlsvqr = __library
                .get(b"cusolverSpScsrlsvqr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpScsrlsvqrHost = __library
                .get(b"cusolverSpScsrlsvqrHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpScsrqrBufferInfoBatched = __library
                .get(b"cusolverSpScsrqrBufferInfoBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpScsrqrsvBatched = __library
                .get(b"cusolverSpScsrqrsvBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpScsrzfdHost = __library
                .get(b"cusolverSpScsrzfdHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpSetStream = __library
                .get(b"cusolverSpSetStream\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpXcsrissymHost = __library
                .get(b"cusolverSpXcsrissymHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpXcsrmetisndHost = __library
                .get(b"cusolverSpXcsrmetisndHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpXcsrpermHost = __library
                .get(b"cusolverSpXcsrpermHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpXcsrperm_bufferSizeHost = __library
                .get(b"cusolverSpXcsrperm_bufferSizeHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpXcsrqrAnalysisBatched = __library
                .get(b"cusolverSpXcsrqrAnalysisBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpXcsrsymamdHost = __library
                .get(b"cusolverSpXcsrsymamdHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpXcsrsymmdqHost = __library
                .get(b"cusolverSpXcsrsymmdqHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpXcsrsymrcmHost = __library
                .get(b"cusolverSpXcsrsymrcmHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpZcsreigsHost = __library
                .get(b"cusolverSpZcsreigsHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpZcsreigvsi = __library
                .get(b"cusolverSpZcsreigvsi\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpZcsreigvsiHost = __library
                .get(b"cusolverSpZcsreigvsiHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpZcsrlsqvqrHost = __library
                .get(b"cusolverSpZcsrlsqvqrHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpZcsrlsvchol = __library
                .get(b"cusolverSpZcsrlsvchol\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpZcsrlsvcholHost = __library
                .get(b"cusolverSpZcsrlsvcholHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpZcsrlsvluHost = __library
                .get(b"cusolverSpZcsrlsvluHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpZcsrlsvqr = __library
                .get(b"cusolverSpZcsrlsvqr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpZcsrlsvqrHost = __library
                .get(b"cusolverSpZcsrlsvqrHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpZcsrqrBufferInfoBatched = __library
                .get(b"cusolverSpZcsrqrBufferInfoBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpZcsrqrsvBatched = __library
                .get(b"cusolverSpZcsrqrsvBatched\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cusolverSpZcsrzfdHost = __library
                .get(b"cusolverSpZcsrzfdHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            Ok(Self {
                __library,
                cusolverDnCCgels,
                cusolverDnCCgels_bufferSize,
                cusolverDnCCgesv,
                cusolverDnCCgesv_bufferSize,
                cusolverDnCEgels,
                cusolverDnCEgels_bufferSize,
                cusolverDnCEgesv,
                cusolverDnCEgesv_bufferSize,
                cusolverDnCKgels,
                cusolverDnCKgels_bufferSize,
                cusolverDnCKgesv,
                cusolverDnCKgesv_bufferSize,
                cusolverDnCYgels,
                cusolverDnCYgels_bufferSize,
                cusolverDnCYgesv,
                cusolverDnCYgesv_bufferSize,
                cusolverDnCgebrd,
                cusolverDnCgebrd_bufferSize,
                cusolverDnCgeqrf,
                cusolverDnCgeqrf_bufferSize,
                cusolverDnCgesvd,
                cusolverDnCgesvd_bufferSize,
                cusolverDnCgesvdaStridedBatched,
                cusolverDnCgesvdaStridedBatched_bufferSize,
                cusolverDnCgesvdj,
                cusolverDnCgesvdjBatched,
                cusolverDnCgesvdjBatched_bufferSize,
                cusolverDnCgesvdj_bufferSize,
                cusolverDnCgetrf,
                cusolverDnCgetrf_bufferSize,
                cusolverDnCgetrs,
                cusolverDnCheevd,
                cusolverDnCheevd_bufferSize,
                cusolverDnCheevdx,
                cusolverDnCheevdx_bufferSize,
                cusolverDnCheevj,
                cusolverDnCheevjBatched,
                cusolverDnCheevjBatched_bufferSize,
                cusolverDnCheevj_bufferSize,
                cusolverDnChegvd,
                cusolverDnChegvd_bufferSize,
                cusolverDnChegvdx,
                cusolverDnChegvdx_bufferSize,
                cusolverDnChegvj,
                cusolverDnChegvj_bufferSize,
                cusolverDnChetrd,
                cusolverDnChetrd_bufferSize,
                cusolverDnClaswp,
                cusolverDnClauum,
                cusolverDnClauum_bufferSize,
                cusolverDnCpotrf,
                cusolverDnCpotrfBatched,
                cusolverDnCpotrf_bufferSize,
                cusolverDnCpotri,
                cusolverDnCpotri_bufferSize,
                cusolverDnCpotrs,
                cusolverDnCpotrsBatched,
                cusolverDnCreate,
                cusolverDnCreateGesvdjInfo,
                cusolverDnCreateParams,
                cusolverDnCreateSyevjInfo,
                cusolverDnCsytrf,
                cusolverDnCsytrf_bufferSize,
                cusolverDnCsytri,
                cusolverDnCsytri_bufferSize,
                cusolverDnCungbr,
                cusolverDnCungbr_bufferSize,
                cusolverDnCungqr,
                cusolverDnCungqr_bufferSize,
                cusolverDnCungtr,
                cusolverDnCungtr_bufferSize,
                cusolverDnCunmqr,
                cusolverDnCunmqr_bufferSize,
                cusolverDnCunmtr,
                cusolverDnCunmtr_bufferSize,
                cusolverDnDBgels,
                cusolverDnDBgels_bufferSize,
                cusolverDnDBgesv,
                cusolverDnDBgesv_bufferSize,
                cusolverDnDDgels,
                cusolverDnDDgels_bufferSize,
                cusolverDnDDgesv,
                cusolverDnDDgesv_bufferSize,
                cusolverDnDHgels,
                cusolverDnDHgels_bufferSize,
                cusolverDnDHgesv,
                cusolverDnDHgesv_bufferSize,
                cusolverDnDSgels,
                cusolverDnDSgels_bufferSize,
                cusolverDnDSgesv,
                cusolverDnDSgesv_bufferSize,
                cusolverDnDXgels,
                cusolverDnDXgels_bufferSize,
                cusolverDnDXgesv,
                cusolverDnDXgesv_bufferSize,
                cusolverDnDestroy,
                cusolverDnDestroyGesvdjInfo,
                cusolverDnDestroyParams,
                cusolverDnDestroySyevjInfo,
                cusolverDnDgebrd,
                cusolverDnDgebrd_bufferSize,
                cusolverDnDgeqrf,
                cusolverDnDgeqrf_bufferSize,
                cusolverDnDgesvd,
                cusolverDnDgesvd_bufferSize,
                cusolverDnDgesvdaStridedBatched,
                cusolverDnDgesvdaStridedBatched_bufferSize,
                cusolverDnDgesvdj,
                cusolverDnDgesvdjBatched,
                cusolverDnDgesvdjBatched_bufferSize,
                cusolverDnDgesvdj_bufferSize,
                cusolverDnDgetrf,
                cusolverDnDgetrf_bufferSize,
                cusolverDnDgetrs,
                cusolverDnDlaswp,
                cusolverDnDlauum,
                cusolverDnDlauum_bufferSize,
                cusolverDnDorgbr,
                cusolverDnDorgbr_bufferSize,
                cusolverDnDorgqr,
                cusolverDnDorgqr_bufferSize,
                cusolverDnDorgtr,
                cusolverDnDorgtr_bufferSize,
                cusolverDnDormqr,
                cusolverDnDormqr_bufferSize,
                cusolverDnDormtr,
                cusolverDnDormtr_bufferSize,
                cusolverDnDpotrf,
                cusolverDnDpotrfBatched,
                cusolverDnDpotrf_bufferSize,
                cusolverDnDpotri,
                cusolverDnDpotri_bufferSize,
                cusolverDnDpotrs,
                cusolverDnDpotrsBatched,
                cusolverDnDsyevd,
                cusolverDnDsyevd_bufferSize,
                cusolverDnDsyevdx,
                cusolverDnDsyevdx_bufferSize,
                cusolverDnDsyevj,
                cusolverDnDsyevjBatched,
                cusolverDnDsyevjBatched_bufferSize,
                cusolverDnDsyevj_bufferSize,
                cusolverDnDsygvd,
                cusolverDnDsygvd_bufferSize,
                cusolverDnDsygvdx,
                cusolverDnDsygvdx_bufferSize,
                cusolverDnDsygvj,
                cusolverDnDsygvj_bufferSize,
                cusolverDnDsytrd,
                cusolverDnDsytrd_bufferSize,
                cusolverDnDsytrf,
                cusolverDnDsytrf_bufferSize,
                cusolverDnDsytri,
                cusolverDnDsytri_bufferSize,
                cusolverDnGeqrf,
                cusolverDnGeqrf_bufferSize,
                cusolverDnGesvd,
                cusolverDnGesvd_bufferSize,
                #[cfg(
                    any(
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cusolverDnGetDeterministicMode,
                cusolverDnGetStream,
                cusolverDnGetrf,
                cusolverDnGetrf_bufferSize,
                cusolverDnGetrs,
                cusolverDnIRSInfosCreate,
                cusolverDnIRSInfosDestroy,
                cusolverDnIRSInfosGetMaxIters,
                cusolverDnIRSInfosGetNiters,
                cusolverDnIRSInfosGetOuterNiters,
                cusolverDnIRSInfosGetResidualHistory,
                cusolverDnIRSInfosRequestResidual,
                cusolverDnIRSParamsCreate,
                cusolverDnIRSParamsDestroy,
                cusolverDnIRSParamsDisableFallback,
                cusolverDnIRSParamsEnableFallback,
                cusolverDnIRSParamsGetMaxIters,
                cusolverDnIRSParamsSetMaxIters,
                cusolverDnIRSParamsSetMaxItersInner,
                cusolverDnIRSParamsSetRefinementSolver,
                cusolverDnIRSParamsSetSolverLowestPrecision,
                cusolverDnIRSParamsSetSolverMainPrecision,
                cusolverDnIRSParamsSetSolverPrecisions,
                cusolverDnIRSParamsSetTol,
                cusolverDnIRSParamsSetTolInner,
                cusolverDnIRSXgels,
                cusolverDnIRSXgels_bufferSize,
                cusolverDnIRSXgesv,
                cusolverDnIRSXgesv_bufferSize,
                #[cfg(
                    any(
                        feature = "cuda-11070",
                        feature = "cuda-11080",
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cusolverDnLoggerForceDisable,
                #[cfg(
                    any(
                        feature = "cuda-11070",
                        feature = "cuda-11080",
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cusolverDnLoggerOpenFile,
                #[cfg(
                    any(
                        feature = "cuda-11070",
                        feature = "cuda-11080",
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cusolverDnLoggerSetCallback,
                #[cfg(
                    any(
                        feature = "cuda-11070",
                        feature = "cuda-11080",
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cusolverDnLoggerSetFile,
                #[cfg(
                    any(
                        feature = "cuda-11070",
                        feature = "cuda-11080",
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cusolverDnLoggerSetLevel,
                #[cfg(
                    any(
                        feature = "cuda-11070",
                        feature = "cuda-11080",
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cusolverDnLoggerSetMask,
                cusolverDnPotrf,
                cusolverDnPotrf_bufferSize,
                cusolverDnPotrs,
                cusolverDnSBgels,
                cusolverDnSBgels_bufferSize,
                cusolverDnSBgesv,
                cusolverDnSBgesv_bufferSize,
                cusolverDnSHgels,
                cusolverDnSHgels_bufferSize,
                cusolverDnSHgesv,
                cusolverDnSHgesv_bufferSize,
                cusolverDnSSgels,
                cusolverDnSSgels_bufferSize,
                cusolverDnSSgesv,
                cusolverDnSSgesv_bufferSize,
                cusolverDnSXgels,
                cusolverDnSXgels_bufferSize,
                cusolverDnSXgesv,
                cusolverDnSXgesv_bufferSize,
                cusolverDnSetAdvOptions,
                #[cfg(
                    any(
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cusolverDnSetDeterministicMode,
                cusolverDnSetStream,
                cusolverDnSgebrd,
                cusolverDnSgebrd_bufferSize,
                cusolverDnSgeqrf,
                cusolverDnSgeqrf_bufferSize,
                cusolverDnSgesvd,
                cusolverDnSgesvd_bufferSize,
                cusolverDnSgesvdaStridedBatched,
                cusolverDnSgesvdaStridedBatched_bufferSize,
                cusolverDnSgesvdj,
                cusolverDnSgesvdjBatched,
                cusolverDnSgesvdjBatched_bufferSize,
                cusolverDnSgesvdj_bufferSize,
                cusolverDnSgetrf,
                cusolverDnSgetrf_bufferSize,
                cusolverDnSgetrs,
                cusolverDnSlaswp,
                cusolverDnSlauum,
                cusolverDnSlauum_bufferSize,
                cusolverDnSorgbr,
                cusolverDnSorgbr_bufferSize,
                cusolverDnSorgqr,
                cusolverDnSorgqr_bufferSize,
                cusolverDnSorgtr,
                cusolverDnSorgtr_bufferSize,
                cusolverDnSormqr,
                cusolverDnSormqr_bufferSize,
                cusolverDnSormtr,
                cusolverDnSormtr_bufferSize,
                cusolverDnSpotrf,
                cusolverDnSpotrfBatched,
                cusolverDnSpotrf_bufferSize,
                cusolverDnSpotri,
                cusolverDnSpotri_bufferSize,
                cusolverDnSpotrs,
                cusolverDnSpotrsBatched,
                cusolverDnSsyevd,
                cusolverDnSsyevd_bufferSize,
                cusolverDnSsyevdx,
                cusolverDnSsyevdx_bufferSize,
                cusolverDnSsyevj,
                cusolverDnSsyevjBatched,
                cusolverDnSsyevjBatched_bufferSize,
                cusolverDnSsyevj_bufferSize,
                cusolverDnSsygvd,
                cusolverDnSsygvd_bufferSize,
                cusolverDnSsygvdx,
                cusolverDnSsygvdx_bufferSize,
                cusolverDnSsygvj,
                cusolverDnSsygvj_bufferSize,
                cusolverDnSsytrd,
                cusolverDnSsytrd_bufferSize,
                cusolverDnSsytrf,
                cusolverDnSsytrf_bufferSize,
                cusolverDnSsytri,
                cusolverDnSsytri_bufferSize,
                cusolverDnSyevd,
                cusolverDnSyevd_bufferSize,
                cusolverDnSyevdx,
                cusolverDnSyevdx_bufferSize,
                #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
                cusolverDnXgeev,
                #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
                cusolverDnXgeev_bufferSize,
                cusolverDnXgeqrf,
                cusolverDnXgeqrf_bufferSize,
                cusolverDnXgesvd,
                cusolverDnXgesvd_bufferSize,
                cusolverDnXgesvdjGetResidual,
                cusolverDnXgesvdjGetSweeps,
                cusolverDnXgesvdjSetMaxSweeps,
                cusolverDnXgesvdjSetSortEig,
                cusolverDnXgesvdjSetTolerance,
                cusolverDnXgesvdp,
                cusolverDnXgesvdp_bufferSize,
                cusolverDnXgesvdr,
                cusolverDnXgesvdr_bufferSize,
                cusolverDnXgetrf,
                cusolverDnXgetrf_bufferSize,
                cusolverDnXgetrs,
                #[cfg(any(feature = "cuda-12040"))]
                cusolverDnXlarft,
                #[cfg(
                    any(
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cusolverDnXlarft,
                #[cfg(any(feature = "cuda-12040"))]
                cusolverDnXlarft_bufferSize,
                #[cfg(
                    any(
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cusolverDnXlarft_bufferSize,
                cusolverDnXpotrf,
                cusolverDnXpotrf_bufferSize,
                cusolverDnXpotrs,
                #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
                cusolverDnXsyevBatched,
                #[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
                cusolverDnXsyevBatched_bufferSize,
                cusolverDnXsyevd,
                cusolverDnXsyevd_bufferSize,
                cusolverDnXsyevdx,
                cusolverDnXsyevdx_bufferSize,
                cusolverDnXsyevjGetResidual,
                cusolverDnXsyevjGetSweeps,
                cusolverDnXsyevjSetMaxSweeps,
                cusolverDnXsyevjSetSortEig,
                cusolverDnXsyevjSetTolerance,
                cusolverDnXsytrs,
                cusolverDnXsytrs_bufferSize,
                cusolverDnXtrtri,
                cusolverDnXtrtri_bufferSize,
                cusolverDnZCgels,
                cusolverDnZCgels_bufferSize,
                cusolverDnZCgesv,
                cusolverDnZCgesv_bufferSize,
                cusolverDnZEgels,
                cusolverDnZEgels_bufferSize,
                cusolverDnZEgesv,
                cusolverDnZEgesv_bufferSize,
                cusolverDnZKgels,
                cusolverDnZKgels_bufferSize,
                cusolverDnZKgesv,
                cusolverDnZKgesv_bufferSize,
                cusolverDnZYgels,
                cusolverDnZYgels_bufferSize,
                cusolverDnZYgesv,
                cusolverDnZYgesv_bufferSize,
                cusolverDnZZgels,
                cusolverDnZZgels_bufferSize,
                cusolverDnZZgesv,
                cusolverDnZZgesv_bufferSize,
                cusolverDnZgebrd,
                cusolverDnZgebrd_bufferSize,
                cusolverDnZgeqrf,
                cusolverDnZgeqrf_bufferSize,
                cusolverDnZgesvd,
                cusolverDnZgesvd_bufferSize,
                cusolverDnZgesvdaStridedBatched,
                cusolverDnZgesvdaStridedBatched_bufferSize,
                cusolverDnZgesvdj,
                cusolverDnZgesvdjBatched,
                cusolverDnZgesvdjBatched_bufferSize,
                cusolverDnZgesvdj_bufferSize,
                cusolverDnZgetrf,
                cusolverDnZgetrf_bufferSize,
                cusolverDnZgetrs,
                cusolverDnZheevd,
                cusolverDnZheevd_bufferSize,
                cusolverDnZheevdx,
                cusolverDnZheevdx_bufferSize,
                cusolverDnZheevj,
                cusolverDnZheevjBatched,
                cusolverDnZheevjBatched_bufferSize,
                cusolverDnZheevj_bufferSize,
                cusolverDnZhegvd,
                cusolverDnZhegvd_bufferSize,
                cusolverDnZhegvdx,
                cusolverDnZhegvdx_bufferSize,
                cusolverDnZhegvj,
                cusolverDnZhegvj_bufferSize,
                cusolverDnZhetrd,
                cusolverDnZhetrd_bufferSize,
                cusolverDnZlaswp,
                cusolverDnZlauum,
                cusolverDnZlauum_bufferSize,
                cusolverDnZpotrf,
                cusolverDnZpotrfBatched,
                cusolverDnZpotrf_bufferSize,
                cusolverDnZpotri,
                cusolverDnZpotri_bufferSize,
                cusolverDnZpotrs,
                cusolverDnZpotrsBatched,
                cusolverDnZsytrf,
                cusolverDnZsytrf_bufferSize,
                cusolverDnZsytri,
                cusolverDnZsytri_bufferSize,
                cusolverDnZungbr,
                cusolverDnZungbr_bufferSize,
                cusolverDnZungqr,
                cusolverDnZungqr_bufferSize,
                cusolverDnZungtr,
                cusolverDnZungtr_bufferSize,
                cusolverDnZunmqr,
                cusolverDnZunmqr_bufferSize,
                cusolverDnZunmtr,
                cusolverDnZunmtr_bufferSize,
                cusolverGetProperty,
                cusolverGetVersion,
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
                cusolverRfAccessBundledFactorsDevice,
                cusolverRfAnalyze,
                cusolverRfBatchAnalyze,
                cusolverRfBatchRefactor,
                cusolverRfBatchResetValues,
                cusolverRfBatchSetupHost,
                cusolverRfBatchSolve,
                cusolverRfBatchZeroPivot,
                cusolverRfCreate,
                cusolverRfDestroy,
                cusolverRfExtractBundledFactorsHost,
                cusolverRfExtractSplitFactorsHost,
                cusolverRfGetAlgs,
                cusolverRfGetMatrixFormat,
                cusolverRfGetNumericBoostReport,
                cusolverRfGetNumericProperties,
                cusolverRfGetResetValuesFastMode,
                cusolverRfRefactor,
                cusolverRfResetValues,
                cusolverRfSetAlgs,
                cusolverRfSetMatrixFormat,
                cusolverRfSetNumericProperties,
                cusolverRfSetResetValuesFastMode,
                cusolverRfSetupDevice,
                cusolverRfSetupHost,
                cusolverRfSolve,
                cusolverSpCcsreigsHost,
                cusolverSpCcsreigvsi,
                cusolverSpCcsreigvsiHost,
                cusolverSpCcsrlsqvqrHost,
                cusolverSpCcsrlsvchol,
                cusolverSpCcsrlsvcholHost,
                cusolverSpCcsrlsvluHost,
                cusolverSpCcsrlsvqr,
                cusolverSpCcsrlsvqrHost,
                cusolverSpCcsrqrBufferInfoBatched,
                cusolverSpCcsrqrsvBatched,
                cusolverSpCcsrzfdHost,
                cusolverSpCreate,
                cusolverSpCreateCsrqrInfo,
                cusolverSpDcsreigsHost,
                cusolverSpDcsreigvsi,
                cusolverSpDcsreigvsiHost,
                cusolverSpDcsrlsqvqrHost,
                cusolverSpDcsrlsvchol,
                cusolverSpDcsrlsvcholHost,
                cusolverSpDcsrlsvluHost,
                cusolverSpDcsrlsvqr,
                cusolverSpDcsrlsvqrHost,
                cusolverSpDcsrqrBufferInfoBatched,
                cusolverSpDcsrqrsvBatched,
                cusolverSpDcsrzfdHost,
                cusolverSpDestroy,
                cusolverSpDestroyCsrqrInfo,
                cusolverSpGetStream,
                cusolverSpScsreigsHost,
                cusolverSpScsreigvsi,
                cusolverSpScsreigvsiHost,
                cusolverSpScsrlsqvqrHost,
                cusolverSpScsrlsvchol,
                cusolverSpScsrlsvcholHost,
                cusolverSpScsrlsvluHost,
                cusolverSpScsrlsvqr,
                cusolverSpScsrlsvqrHost,
                cusolverSpScsrqrBufferInfoBatched,
                cusolverSpScsrqrsvBatched,
                cusolverSpScsrzfdHost,
                cusolverSpSetStream,
                cusolverSpXcsrissymHost,
                cusolverSpXcsrmetisndHost,
                cusolverSpXcsrpermHost,
                cusolverSpXcsrperm_bufferSizeHost,
                cusolverSpXcsrqrAnalysisBatched,
                cusolverSpXcsrsymamdHost,
                cusolverSpXcsrsymmdqHost,
                cusolverSpXcsrsymrcmHost,
                cusolverSpZcsreigsHost,
                cusolverSpZcsreigvsi,
                cusolverSpZcsreigvsiHost,
                cusolverSpZcsrlsqvqrHost,
                cusolverSpZcsrlsvchol,
                cusolverSpZcsrlsvcholHost,
                cusolverSpZcsrlsvluHost,
                cusolverSpZcsrlsvqr,
                cusolverSpZcsrlsvqrHost,
                cusolverSpZcsrqrBufferInfoBatched,
                cusolverSpZcsrqrsvBatched,
                cusolverSpZcsrzfdHost,
            })
        }
    }
    pub unsafe fn culib() -> &'static Lib {
        static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
        LIB.get_or_init(|| {
            let lib_names = std::vec!["cusolver"];
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
