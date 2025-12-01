#![cfg_attr(feature = "no-std", no_std)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;
pub use self::cufftCompatibility_t as cufftCompatibility;
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000"
))]
pub use self::cufftProperty_t as cufftProperty;
pub use self::cufftResult_t as cufftResult;
pub use self::cufftType_t as cufftType;
pub use self::libraryPropertyType_t as libraryPropertyType;
pub type cuComplex = cuFloatComplex;
pub type cuDoubleComplex = double2;
pub type cuFloatComplex = float2;
pub type cudaStream_t = *mut CUstream_st;
pub type cufftComplex = cuComplex;
pub type cufftDoubleComplex = cuDoubleComplex;
pub type cufftDoubleReal = f64;
pub type cufftHandle = ::core::ffi::c_int;
pub type cufftReal = f32;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cufftCompatibility_t {
    CUFFT_COMPATIBILITY_FFTW_PADDING = 1,
}
#[cfg(any(feature = "cuda-12040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cufftProperty_t {
    NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT = 1,
}
#[cfg(any(
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cufftProperty_t {
    NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT = 1,
    NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS = 2,
}
#[cfg(any(
    feature = "cuda-12000",
    feature = "cuda-12010",
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cufftResult_t {
    CUFFT_SUCCESS = 0,
    CUFFT_INVALID_PLAN = 1,
    CUFFT_ALLOC_FAILED = 2,
    CUFFT_INVALID_TYPE = 3,
    CUFFT_INVALID_VALUE = 4,
    CUFFT_INTERNAL_ERROR = 5,
    CUFFT_EXEC_FAILED = 6,
    CUFFT_SETUP_FAILED = 7,
    CUFFT_INVALID_SIZE = 8,
    CUFFT_UNALIGNED_DATA = 9,
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10,
    CUFFT_INVALID_DEVICE = 11,
    CUFFT_PARSE_ERROR = 12,
    CUFFT_NO_WORKSPACE = 13,
    CUFFT_NOT_IMPLEMENTED = 14,
    CUFFT_LICENSE_ERROR = 15,
    CUFFT_NOT_SUPPORTED = 16,
}
#[cfg(any(feature = "cuda-13000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cufftResult_t {
    CUFFT_SUCCESS = 0,
    CUFFT_INVALID_PLAN = 1,
    CUFFT_ALLOC_FAILED = 2,
    CUFFT_INVALID_TYPE = 3,
    CUFFT_INVALID_VALUE = 4,
    CUFFT_INTERNAL_ERROR = 5,
    CUFFT_EXEC_FAILED = 6,
    CUFFT_SETUP_FAILED = 7,
    CUFFT_INVALID_SIZE = 8,
    CUFFT_UNALIGNED_DATA = 9,
    CUFFT_INVALID_DEVICE = 11,
    CUFFT_NO_WORKSPACE = 13,
    CUFFT_NOT_IMPLEMENTED = 14,
    CUFFT_NOT_SUPPORTED = 16,
    CUFFT_MISSING_DEPENDENCY = 17,
    CUFFT_NVRTC_FAILURE = 18,
    CUFFT_NVJITLINK_FAILURE = 19,
    CUFFT_NVSHMEM_FAILURE = 20,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cufftType_t {
    CUFFT_R2C = 42,
    CUFFT_C2R = 44,
    CUFFT_C2C = 41,
    CUFFT_D2Z = 106,
    CUFFT_Z2D = 108,
    CUFFT_Z2Z = 105,
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
#[repr(C)]
#[repr(align(16))]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct double2 {
    pub x: f64,
    pub y: f64,
}
#[repr(C)]
#[repr(align(8))]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct float2 {
    pub x: f32,
    pub y: f32,
}
#[cfg(not(feature = "dynamic-loading"))]
extern "C" {
    pub fn cufftCreate(handle: *mut cufftHandle) -> cufftResult;
    pub fn cufftDestroy(plan: cufftHandle) -> cufftResult;
    pub fn cufftEstimate1d(
        nx: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
        workSize: *mut usize,
    ) -> cufftResult;
    pub fn cufftEstimate2d(
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        type_: cufftType,
        workSize: *mut usize,
    ) -> cufftResult;
    pub fn cufftEstimate3d(
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        nz: ::core::ffi::c_int,
        type_: cufftType,
        workSize: *mut usize,
    ) -> cufftResult;
    pub fn cufftEstimateMany(
        rank: ::core::ffi::c_int,
        n: *mut ::core::ffi::c_int,
        inembed: *mut ::core::ffi::c_int,
        istride: ::core::ffi::c_int,
        idist: ::core::ffi::c_int,
        onembed: *mut ::core::ffi::c_int,
        ostride: ::core::ffi::c_int,
        odist: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
        workSize: *mut usize,
    ) -> cufftResult;
    pub fn cufftExecC2C(
        plan: cufftHandle,
        idata: *mut cufftComplex,
        odata: *mut cufftComplex,
        direction: ::core::ffi::c_int,
    ) -> cufftResult;
    pub fn cufftExecC2R(
        plan: cufftHandle,
        idata: *mut cufftComplex,
        odata: *mut cufftReal,
    ) -> cufftResult;
    pub fn cufftExecD2Z(
        plan: cufftHandle,
        idata: *mut cufftDoubleReal,
        odata: *mut cufftDoubleComplex,
    ) -> cufftResult;
    pub fn cufftExecR2C(
        plan: cufftHandle,
        idata: *mut cufftReal,
        odata: *mut cufftComplex,
    ) -> cufftResult;
    pub fn cufftExecZ2D(
        plan: cufftHandle,
        idata: *mut cufftDoubleComplex,
        odata: *mut cufftDoubleReal,
    ) -> cufftResult;
    pub fn cufftExecZ2Z(
        plan: cufftHandle,
        idata: *mut cufftDoubleComplex,
        odata: *mut cufftDoubleComplex,
        direction: ::core::ffi::c_int,
    ) -> cufftResult;
    #[cfg(any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000"
    ))]
    pub fn cufftGetPlanPropertyInt64(
        plan: cufftHandle,
        property: cufftProperty,
        returnPtrValue: *mut ::core::ffi::c_longlong,
    ) -> cufftResult;
    pub fn cufftGetProperty(
        type_: libraryPropertyType,
        value: *mut ::core::ffi::c_int,
    ) -> cufftResult;
    pub fn cufftGetSize(handle: cufftHandle, workSize: *mut usize) -> cufftResult;
    pub fn cufftGetSize1d(
        handle: cufftHandle,
        nx: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
        workSize: *mut usize,
    ) -> cufftResult;
    pub fn cufftGetSize2d(
        handle: cufftHandle,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        type_: cufftType,
        workSize: *mut usize,
    ) -> cufftResult;
    pub fn cufftGetSize3d(
        handle: cufftHandle,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        nz: ::core::ffi::c_int,
        type_: cufftType,
        workSize: *mut usize,
    ) -> cufftResult;
    pub fn cufftGetSizeMany(
        handle: cufftHandle,
        rank: ::core::ffi::c_int,
        n: *mut ::core::ffi::c_int,
        inembed: *mut ::core::ffi::c_int,
        istride: ::core::ffi::c_int,
        idist: ::core::ffi::c_int,
        onembed: *mut ::core::ffi::c_int,
        ostride: ::core::ffi::c_int,
        odist: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
        workArea: *mut usize,
    ) -> cufftResult;
    pub fn cufftGetSizeMany64(
        plan: cufftHandle,
        rank: ::core::ffi::c_int,
        n: *mut ::core::ffi::c_longlong,
        inembed: *mut ::core::ffi::c_longlong,
        istride: ::core::ffi::c_longlong,
        idist: ::core::ffi::c_longlong,
        onembed: *mut ::core::ffi::c_longlong,
        ostride: ::core::ffi::c_longlong,
        odist: ::core::ffi::c_longlong,
        type_: cufftType,
        batch: ::core::ffi::c_longlong,
        workSize: *mut usize,
    ) -> cufftResult;
    pub fn cufftGetVersion(version: *mut ::core::ffi::c_int) -> cufftResult;
    pub fn cufftMakePlan1d(
        plan: cufftHandle,
        nx: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
        workSize: *mut usize,
    ) -> cufftResult;
    pub fn cufftMakePlan2d(
        plan: cufftHandle,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        type_: cufftType,
        workSize: *mut usize,
    ) -> cufftResult;
    pub fn cufftMakePlan3d(
        plan: cufftHandle,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        nz: ::core::ffi::c_int,
        type_: cufftType,
        workSize: *mut usize,
    ) -> cufftResult;
    pub fn cufftMakePlanMany(
        plan: cufftHandle,
        rank: ::core::ffi::c_int,
        n: *mut ::core::ffi::c_int,
        inembed: *mut ::core::ffi::c_int,
        istride: ::core::ffi::c_int,
        idist: ::core::ffi::c_int,
        onembed: *mut ::core::ffi::c_int,
        ostride: ::core::ffi::c_int,
        odist: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
        workSize: *mut usize,
    ) -> cufftResult;
    pub fn cufftMakePlanMany64(
        plan: cufftHandle,
        rank: ::core::ffi::c_int,
        n: *mut ::core::ffi::c_longlong,
        inembed: *mut ::core::ffi::c_longlong,
        istride: ::core::ffi::c_longlong,
        idist: ::core::ffi::c_longlong,
        onembed: *mut ::core::ffi::c_longlong,
        ostride: ::core::ffi::c_longlong,
        odist: ::core::ffi::c_longlong,
        type_: cufftType,
        batch: ::core::ffi::c_longlong,
        workSize: *mut usize,
    ) -> cufftResult;
    pub fn cufftPlan1d(
        plan: *mut cufftHandle,
        nx: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
    ) -> cufftResult;
    pub fn cufftPlan2d(
        plan: *mut cufftHandle,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        type_: cufftType,
    ) -> cufftResult;
    pub fn cufftPlan3d(
        plan: *mut cufftHandle,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        nz: ::core::ffi::c_int,
        type_: cufftType,
    ) -> cufftResult;
    pub fn cufftPlanMany(
        plan: *mut cufftHandle,
        rank: ::core::ffi::c_int,
        n: *mut ::core::ffi::c_int,
        inembed: *mut ::core::ffi::c_int,
        istride: ::core::ffi::c_int,
        idist: ::core::ffi::c_int,
        onembed: *mut ::core::ffi::c_int,
        ostride: ::core::ffi::c_int,
        odist: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
    ) -> cufftResult;
    #[cfg(any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000"
    ))]
    pub fn cufftResetPlanProperty(plan: cufftHandle, property: cufftProperty) -> cufftResult;
    pub fn cufftSetAutoAllocation(
        plan: cufftHandle,
        autoAllocate: ::core::ffi::c_int,
    ) -> cufftResult;
    #[cfg(any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000"
    ))]
    pub fn cufftSetPlanPropertyInt64(
        plan: cufftHandle,
        property: cufftProperty,
        inputValueInt: ::core::ffi::c_longlong,
    ) -> cufftResult;
    pub fn cufftSetStream(plan: cufftHandle, stream: cudaStream_t) -> cufftResult;
    pub fn cufftSetWorkArea(plan: cufftHandle, workArea: *mut ::core::ffi::c_void) -> cufftResult;
}
#[cfg(feature = "dynamic-loading")]
mod loaded {
    use super::*;
    pub unsafe fn cufftCreate(handle: *mut cufftHandle) -> cufftResult {
        (culib().cufftCreate)(handle)
    }
    pub unsafe fn cufftDestroy(plan: cufftHandle) -> cufftResult {
        (culib().cufftDestroy)(plan)
    }
    pub unsafe fn cufftEstimate1d(
        nx: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
        workSize: *mut usize,
    ) -> cufftResult {
        (culib().cufftEstimate1d)(nx, type_, batch, workSize)
    }
    pub unsafe fn cufftEstimate2d(
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        type_: cufftType,
        workSize: *mut usize,
    ) -> cufftResult {
        (culib().cufftEstimate2d)(nx, ny, type_, workSize)
    }
    pub unsafe fn cufftEstimate3d(
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        nz: ::core::ffi::c_int,
        type_: cufftType,
        workSize: *mut usize,
    ) -> cufftResult {
        (culib().cufftEstimate3d)(nx, ny, nz, type_, workSize)
    }
    pub unsafe fn cufftEstimateMany(
        rank: ::core::ffi::c_int,
        n: *mut ::core::ffi::c_int,
        inembed: *mut ::core::ffi::c_int,
        istride: ::core::ffi::c_int,
        idist: ::core::ffi::c_int,
        onembed: *mut ::core::ffi::c_int,
        ostride: ::core::ffi::c_int,
        odist: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
        workSize: *mut usize,
    ) -> cufftResult {
        (culib().cufftEstimateMany)(
            rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workSize,
        )
    }
    pub unsafe fn cufftExecC2C(
        plan: cufftHandle,
        idata: *mut cufftComplex,
        odata: *mut cufftComplex,
        direction: ::core::ffi::c_int,
    ) -> cufftResult {
        (culib().cufftExecC2C)(plan, idata, odata, direction)
    }
    pub unsafe fn cufftExecC2R(
        plan: cufftHandle,
        idata: *mut cufftComplex,
        odata: *mut cufftReal,
    ) -> cufftResult {
        (culib().cufftExecC2R)(plan, idata, odata)
    }
    pub unsafe fn cufftExecD2Z(
        plan: cufftHandle,
        idata: *mut cufftDoubleReal,
        odata: *mut cufftDoubleComplex,
    ) -> cufftResult {
        (culib().cufftExecD2Z)(plan, idata, odata)
    }
    pub unsafe fn cufftExecR2C(
        plan: cufftHandle,
        idata: *mut cufftReal,
        odata: *mut cufftComplex,
    ) -> cufftResult {
        (culib().cufftExecR2C)(plan, idata, odata)
    }
    pub unsafe fn cufftExecZ2D(
        plan: cufftHandle,
        idata: *mut cufftDoubleComplex,
        odata: *mut cufftDoubleReal,
    ) -> cufftResult {
        (culib().cufftExecZ2D)(plan, idata, odata)
    }
    pub unsafe fn cufftExecZ2Z(
        plan: cufftHandle,
        idata: *mut cufftDoubleComplex,
        odata: *mut cufftDoubleComplex,
        direction: ::core::ffi::c_int,
    ) -> cufftResult {
        (culib().cufftExecZ2Z)(plan, idata, odata, direction)
    }
    #[cfg(any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000"
    ))]
    pub unsafe fn cufftGetPlanPropertyInt64(
        plan: cufftHandle,
        property: cufftProperty,
        returnPtrValue: *mut ::core::ffi::c_longlong,
    ) -> cufftResult {
        (culib().cufftGetPlanPropertyInt64)(plan, property, returnPtrValue)
    }
    pub unsafe fn cufftGetProperty(
        type_: libraryPropertyType,
        value: *mut ::core::ffi::c_int,
    ) -> cufftResult {
        (culib().cufftGetProperty)(type_, value)
    }
    pub unsafe fn cufftGetSize(handle: cufftHandle, workSize: *mut usize) -> cufftResult {
        (culib().cufftGetSize)(handle, workSize)
    }
    pub unsafe fn cufftGetSize1d(
        handle: cufftHandle,
        nx: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
        workSize: *mut usize,
    ) -> cufftResult {
        (culib().cufftGetSize1d)(handle, nx, type_, batch, workSize)
    }
    pub unsafe fn cufftGetSize2d(
        handle: cufftHandle,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        type_: cufftType,
        workSize: *mut usize,
    ) -> cufftResult {
        (culib().cufftGetSize2d)(handle, nx, ny, type_, workSize)
    }
    pub unsafe fn cufftGetSize3d(
        handle: cufftHandle,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        nz: ::core::ffi::c_int,
        type_: cufftType,
        workSize: *mut usize,
    ) -> cufftResult {
        (culib().cufftGetSize3d)(handle, nx, ny, nz, type_, workSize)
    }
    pub unsafe fn cufftGetSizeMany(
        handle: cufftHandle,
        rank: ::core::ffi::c_int,
        n: *mut ::core::ffi::c_int,
        inembed: *mut ::core::ffi::c_int,
        istride: ::core::ffi::c_int,
        idist: ::core::ffi::c_int,
        onembed: *mut ::core::ffi::c_int,
        ostride: ::core::ffi::c_int,
        odist: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
        workArea: *mut usize,
    ) -> cufftResult {
        (culib().cufftGetSizeMany)(
            handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch,
            workArea,
        )
    }
    pub unsafe fn cufftGetSizeMany64(
        plan: cufftHandle,
        rank: ::core::ffi::c_int,
        n: *mut ::core::ffi::c_longlong,
        inembed: *mut ::core::ffi::c_longlong,
        istride: ::core::ffi::c_longlong,
        idist: ::core::ffi::c_longlong,
        onembed: *mut ::core::ffi::c_longlong,
        ostride: ::core::ffi::c_longlong,
        odist: ::core::ffi::c_longlong,
        type_: cufftType,
        batch: ::core::ffi::c_longlong,
        workSize: *mut usize,
    ) -> cufftResult {
        (culib().cufftGetSizeMany64)(
            plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workSize,
        )
    }
    pub unsafe fn cufftGetVersion(version: *mut ::core::ffi::c_int) -> cufftResult {
        (culib().cufftGetVersion)(version)
    }
    pub unsafe fn cufftMakePlan1d(
        plan: cufftHandle,
        nx: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
        workSize: *mut usize,
    ) -> cufftResult {
        (culib().cufftMakePlan1d)(plan, nx, type_, batch, workSize)
    }
    pub unsafe fn cufftMakePlan2d(
        plan: cufftHandle,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        type_: cufftType,
        workSize: *mut usize,
    ) -> cufftResult {
        (culib().cufftMakePlan2d)(plan, nx, ny, type_, workSize)
    }
    pub unsafe fn cufftMakePlan3d(
        plan: cufftHandle,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        nz: ::core::ffi::c_int,
        type_: cufftType,
        workSize: *mut usize,
    ) -> cufftResult {
        (culib().cufftMakePlan3d)(plan, nx, ny, nz, type_, workSize)
    }
    pub unsafe fn cufftMakePlanMany(
        plan: cufftHandle,
        rank: ::core::ffi::c_int,
        n: *mut ::core::ffi::c_int,
        inembed: *mut ::core::ffi::c_int,
        istride: ::core::ffi::c_int,
        idist: ::core::ffi::c_int,
        onembed: *mut ::core::ffi::c_int,
        ostride: ::core::ffi::c_int,
        odist: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
        workSize: *mut usize,
    ) -> cufftResult {
        (culib().cufftMakePlanMany)(
            plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workSize,
        )
    }
    pub unsafe fn cufftMakePlanMany64(
        plan: cufftHandle,
        rank: ::core::ffi::c_int,
        n: *mut ::core::ffi::c_longlong,
        inembed: *mut ::core::ffi::c_longlong,
        istride: ::core::ffi::c_longlong,
        idist: ::core::ffi::c_longlong,
        onembed: *mut ::core::ffi::c_longlong,
        ostride: ::core::ffi::c_longlong,
        odist: ::core::ffi::c_longlong,
        type_: cufftType,
        batch: ::core::ffi::c_longlong,
        workSize: *mut usize,
    ) -> cufftResult {
        (culib().cufftMakePlanMany64)(
            plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch, workSize,
        )
    }
    pub unsafe fn cufftPlan1d(
        plan: *mut cufftHandle,
        nx: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
    ) -> cufftResult {
        (culib().cufftPlan1d)(plan, nx, type_, batch)
    }
    pub unsafe fn cufftPlan2d(
        plan: *mut cufftHandle,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        type_: cufftType,
    ) -> cufftResult {
        (culib().cufftPlan2d)(plan, nx, ny, type_)
    }
    pub unsafe fn cufftPlan3d(
        plan: *mut cufftHandle,
        nx: ::core::ffi::c_int,
        ny: ::core::ffi::c_int,
        nz: ::core::ffi::c_int,
        type_: cufftType,
    ) -> cufftResult {
        (culib().cufftPlan3d)(plan, nx, ny, nz, type_)
    }
    pub unsafe fn cufftPlanMany(
        plan: *mut cufftHandle,
        rank: ::core::ffi::c_int,
        n: *mut ::core::ffi::c_int,
        inembed: *mut ::core::ffi::c_int,
        istride: ::core::ffi::c_int,
        idist: ::core::ffi::c_int,
        onembed: *mut ::core::ffi::c_int,
        ostride: ::core::ffi::c_int,
        odist: ::core::ffi::c_int,
        type_: cufftType,
        batch: ::core::ffi::c_int,
    ) -> cufftResult {
        (culib().cufftPlanMany)(
            plan, rank, n, inembed, istride, idist, onembed, ostride, odist, type_, batch,
        )
    }
    #[cfg(any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000"
    ))]
    pub unsafe fn cufftResetPlanProperty(
        plan: cufftHandle,
        property: cufftProperty,
    ) -> cufftResult {
        (culib().cufftResetPlanProperty)(plan, property)
    }
    pub unsafe fn cufftSetAutoAllocation(
        plan: cufftHandle,
        autoAllocate: ::core::ffi::c_int,
    ) -> cufftResult {
        (culib().cufftSetAutoAllocation)(plan, autoAllocate)
    }
    #[cfg(any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090",
        feature = "cuda-13000"
    ))]
    pub unsafe fn cufftSetPlanPropertyInt64(
        plan: cufftHandle,
        property: cufftProperty,
        inputValueInt: ::core::ffi::c_longlong,
    ) -> cufftResult {
        (culib().cufftSetPlanPropertyInt64)(plan, property, inputValueInt)
    }
    pub unsafe fn cufftSetStream(plan: cufftHandle, stream: cudaStream_t) -> cufftResult {
        (culib().cufftSetStream)(plan, stream)
    }
    pub unsafe fn cufftSetWorkArea(
        plan: cufftHandle,
        workArea: *mut ::core::ffi::c_void,
    ) -> cufftResult {
        (culib().cufftSetWorkArea)(plan, workArea)
    }
    pub struct Lib {
        __library: ::libloading::Library,
        pub cufftCreate: unsafe extern "C" fn(handle: *mut cufftHandle) -> cufftResult,
        pub cufftDestroy: unsafe extern "C" fn(plan: cufftHandle) -> cufftResult,
        pub cufftEstimate1d: unsafe extern "C" fn(
            nx: ::core::ffi::c_int,
            type_: cufftType,
            batch: ::core::ffi::c_int,
            workSize: *mut usize,
        ) -> cufftResult,
        pub cufftEstimate2d: unsafe extern "C" fn(
            nx: ::core::ffi::c_int,
            ny: ::core::ffi::c_int,
            type_: cufftType,
            workSize: *mut usize,
        ) -> cufftResult,
        pub cufftEstimate3d: unsafe extern "C" fn(
            nx: ::core::ffi::c_int,
            ny: ::core::ffi::c_int,
            nz: ::core::ffi::c_int,
            type_: cufftType,
            workSize: *mut usize,
        ) -> cufftResult,
        pub cufftEstimateMany: unsafe extern "C" fn(
            rank: ::core::ffi::c_int,
            n: *mut ::core::ffi::c_int,
            inembed: *mut ::core::ffi::c_int,
            istride: ::core::ffi::c_int,
            idist: ::core::ffi::c_int,
            onembed: *mut ::core::ffi::c_int,
            ostride: ::core::ffi::c_int,
            odist: ::core::ffi::c_int,
            type_: cufftType,
            batch: ::core::ffi::c_int,
            workSize: *mut usize,
        ) -> cufftResult,
        pub cufftExecC2C: unsafe extern "C" fn(
            plan: cufftHandle,
            idata: *mut cufftComplex,
            odata: *mut cufftComplex,
            direction: ::core::ffi::c_int,
        ) -> cufftResult,
        pub cufftExecC2R: unsafe extern "C" fn(
            plan: cufftHandle,
            idata: *mut cufftComplex,
            odata: *mut cufftReal,
        ) -> cufftResult,
        pub cufftExecD2Z: unsafe extern "C" fn(
            plan: cufftHandle,
            idata: *mut cufftDoubleReal,
            odata: *mut cufftDoubleComplex,
        ) -> cufftResult,
        pub cufftExecR2C: unsafe extern "C" fn(
            plan: cufftHandle,
            idata: *mut cufftReal,
            odata: *mut cufftComplex,
        ) -> cufftResult,
        pub cufftExecZ2D: unsafe extern "C" fn(
            plan: cufftHandle,
            idata: *mut cufftDoubleComplex,
            odata: *mut cufftDoubleReal,
        ) -> cufftResult,
        pub cufftExecZ2Z: unsafe extern "C" fn(
            plan: cufftHandle,
            idata: *mut cufftDoubleComplex,
            odata: *mut cufftDoubleComplex,
            direction: ::core::ffi::c_int,
        ) -> cufftResult,
        #[cfg(any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090",
            feature = "cuda-13000"
        ))]
        pub cufftGetPlanPropertyInt64: unsafe extern "C" fn(
            plan: cufftHandle,
            property: cufftProperty,
            returnPtrValue: *mut ::core::ffi::c_longlong,
        ) -> cufftResult,
        pub cufftGetProperty: unsafe extern "C" fn(
            type_: libraryPropertyType,
            value: *mut ::core::ffi::c_int,
        ) -> cufftResult,
        pub cufftGetSize:
            unsafe extern "C" fn(handle: cufftHandle, workSize: *mut usize) -> cufftResult,
        pub cufftGetSize1d: unsafe extern "C" fn(
            handle: cufftHandle,
            nx: ::core::ffi::c_int,
            type_: cufftType,
            batch: ::core::ffi::c_int,
            workSize: *mut usize,
        ) -> cufftResult,
        pub cufftGetSize2d: unsafe extern "C" fn(
            handle: cufftHandle,
            nx: ::core::ffi::c_int,
            ny: ::core::ffi::c_int,
            type_: cufftType,
            workSize: *mut usize,
        ) -> cufftResult,
        pub cufftGetSize3d: unsafe extern "C" fn(
            handle: cufftHandle,
            nx: ::core::ffi::c_int,
            ny: ::core::ffi::c_int,
            nz: ::core::ffi::c_int,
            type_: cufftType,
            workSize: *mut usize,
        ) -> cufftResult,
        pub cufftGetSizeMany: unsafe extern "C" fn(
            handle: cufftHandle,
            rank: ::core::ffi::c_int,
            n: *mut ::core::ffi::c_int,
            inembed: *mut ::core::ffi::c_int,
            istride: ::core::ffi::c_int,
            idist: ::core::ffi::c_int,
            onembed: *mut ::core::ffi::c_int,
            ostride: ::core::ffi::c_int,
            odist: ::core::ffi::c_int,
            type_: cufftType,
            batch: ::core::ffi::c_int,
            workArea: *mut usize,
        ) -> cufftResult,
        pub cufftGetSizeMany64: unsafe extern "C" fn(
            plan: cufftHandle,
            rank: ::core::ffi::c_int,
            n: *mut ::core::ffi::c_longlong,
            inembed: *mut ::core::ffi::c_longlong,
            istride: ::core::ffi::c_longlong,
            idist: ::core::ffi::c_longlong,
            onembed: *mut ::core::ffi::c_longlong,
            ostride: ::core::ffi::c_longlong,
            odist: ::core::ffi::c_longlong,
            type_: cufftType,
            batch: ::core::ffi::c_longlong,
            workSize: *mut usize,
        ) -> cufftResult,
        pub cufftGetVersion: unsafe extern "C" fn(version: *mut ::core::ffi::c_int) -> cufftResult,
        pub cufftMakePlan1d: unsafe extern "C" fn(
            plan: cufftHandle,
            nx: ::core::ffi::c_int,
            type_: cufftType,
            batch: ::core::ffi::c_int,
            workSize: *mut usize,
        ) -> cufftResult,
        pub cufftMakePlan2d: unsafe extern "C" fn(
            plan: cufftHandle,
            nx: ::core::ffi::c_int,
            ny: ::core::ffi::c_int,
            type_: cufftType,
            workSize: *mut usize,
        ) -> cufftResult,
        pub cufftMakePlan3d: unsafe extern "C" fn(
            plan: cufftHandle,
            nx: ::core::ffi::c_int,
            ny: ::core::ffi::c_int,
            nz: ::core::ffi::c_int,
            type_: cufftType,
            workSize: *mut usize,
        ) -> cufftResult,
        pub cufftMakePlanMany: unsafe extern "C" fn(
            plan: cufftHandle,
            rank: ::core::ffi::c_int,
            n: *mut ::core::ffi::c_int,
            inembed: *mut ::core::ffi::c_int,
            istride: ::core::ffi::c_int,
            idist: ::core::ffi::c_int,
            onembed: *mut ::core::ffi::c_int,
            ostride: ::core::ffi::c_int,
            odist: ::core::ffi::c_int,
            type_: cufftType,
            batch: ::core::ffi::c_int,
            workSize: *mut usize,
        ) -> cufftResult,
        pub cufftMakePlanMany64: unsafe extern "C" fn(
            plan: cufftHandle,
            rank: ::core::ffi::c_int,
            n: *mut ::core::ffi::c_longlong,
            inembed: *mut ::core::ffi::c_longlong,
            istride: ::core::ffi::c_longlong,
            idist: ::core::ffi::c_longlong,
            onembed: *mut ::core::ffi::c_longlong,
            ostride: ::core::ffi::c_longlong,
            odist: ::core::ffi::c_longlong,
            type_: cufftType,
            batch: ::core::ffi::c_longlong,
            workSize: *mut usize,
        ) -> cufftResult,
        pub cufftPlan1d: unsafe extern "C" fn(
            plan: *mut cufftHandle,
            nx: ::core::ffi::c_int,
            type_: cufftType,
            batch: ::core::ffi::c_int,
        ) -> cufftResult,
        pub cufftPlan2d: unsafe extern "C" fn(
            plan: *mut cufftHandle,
            nx: ::core::ffi::c_int,
            ny: ::core::ffi::c_int,
            type_: cufftType,
        ) -> cufftResult,
        pub cufftPlan3d: unsafe extern "C" fn(
            plan: *mut cufftHandle,
            nx: ::core::ffi::c_int,
            ny: ::core::ffi::c_int,
            nz: ::core::ffi::c_int,
            type_: cufftType,
        ) -> cufftResult,
        pub cufftPlanMany: unsafe extern "C" fn(
            plan: *mut cufftHandle,
            rank: ::core::ffi::c_int,
            n: *mut ::core::ffi::c_int,
            inembed: *mut ::core::ffi::c_int,
            istride: ::core::ffi::c_int,
            idist: ::core::ffi::c_int,
            onembed: *mut ::core::ffi::c_int,
            ostride: ::core::ffi::c_int,
            odist: ::core::ffi::c_int,
            type_: cufftType,
            batch: ::core::ffi::c_int,
        ) -> cufftResult,
        #[cfg(any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090",
            feature = "cuda-13000"
        ))]
        pub cufftResetPlanProperty:
            unsafe extern "C" fn(plan: cufftHandle, property: cufftProperty) -> cufftResult,
        pub cufftSetAutoAllocation: unsafe extern "C" fn(
            plan: cufftHandle,
            autoAllocate: ::core::ffi::c_int,
        ) -> cufftResult,
        #[cfg(any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090",
            feature = "cuda-13000"
        ))]
        pub cufftSetPlanPropertyInt64: unsafe extern "C" fn(
            plan: cufftHandle,
            property: cufftProperty,
            inputValueInt: ::core::ffi::c_longlong,
        ) -> cufftResult,
        pub cufftSetStream:
            unsafe extern "C" fn(plan: cufftHandle, stream: cudaStream_t) -> cufftResult,
        pub cufftSetWorkArea: unsafe extern "C" fn(
            plan: cufftHandle,
            workArea: *mut ::core::ffi::c_void,
        ) -> cufftResult,
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
            let cufftCreate = __library
                .get(b"cufftCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftDestroy = __library
                .get(b"cufftDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftEstimate1d = __library
                .get(b"cufftEstimate1d\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftEstimate2d = __library
                .get(b"cufftEstimate2d\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftEstimate3d = __library
                .get(b"cufftEstimate3d\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftEstimateMany = __library
                .get(b"cufftEstimateMany\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftExecC2C = __library
                .get(b"cufftExecC2C\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftExecC2R = __library
                .get(b"cufftExecC2R\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftExecD2Z = __library
                .get(b"cufftExecD2Z\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftExecR2C = __library
                .get(b"cufftExecR2C\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftExecZ2D = __library
                .get(b"cufftExecZ2D\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftExecZ2Z = __library
                .get(b"cufftExecZ2Z\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090",
                feature = "cuda-13000"
            ))]
            let cufftGetPlanPropertyInt64 = __library
                .get(b"cufftGetPlanPropertyInt64\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftGetProperty = __library
                .get(b"cufftGetProperty\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftGetSize = __library
                .get(b"cufftGetSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftGetSize1d = __library
                .get(b"cufftGetSize1d\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftGetSize2d = __library
                .get(b"cufftGetSize2d\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftGetSize3d = __library
                .get(b"cufftGetSize3d\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftGetSizeMany = __library
                .get(b"cufftGetSizeMany\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftGetSizeMany64 = __library
                .get(b"cufftGetSizeMany64\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftGetVersion = __library
                .get(b"cufftGetVersion\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftMakePlan1d = __library
                .get(b"cufftMakePlan1d\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftMakePlan2d = __library
                .get(b"cufftMakePlan2d\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftMakePlan3d = __library
                .get(b"cufftMakePlan3d\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftMakePlanMany = __library
                .get(b"cufftMakePlanMany\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftMakePlanMany64 = __library
                .get(b"cufftMakePlanMany64\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftPlan1d = __library
                .get(b"cufftPlan1d\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftPlan2d = __library
                .get(b"cufftPlan2d\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftPlan3d = __library
                .get(b"cufftPlan3d\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftPlanMany = __library
                .get(b"cufftPlanMany\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090",
                feature = "cuda-13000"
            ))]
            let cufftResetPlanProperty = __library
                .get(b"cufftResetPlanProperty\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftSetAutoAllocation = __library
                .get(b"cufftSetAutoAllocation\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090",
                feature = "cuda-13000"
            ))]
            let cufftSetPlanPropertyInt64 = __library
                .get(b"cufftSetPlanPropertyInt64\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftSetStream = __library
                .get(b"cufftSetStream\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cufftSetWorkArea = __library
                .get(b"cufftSetWorkArea\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            Ok(Self {
                __library,
                cufftCreate,
                cufftDestroy,
                cufftEstimate1d,
                cufftEstimate2d,
                cufftEstimate3d,
                cufftEstimateMany,
                cufftExecC2C,
                cufftExecC2R,
                cufftExecD2Z,
                cufftExecR2C,
                cufftExecZ2D,
                cufftExecZ2Z,
                #[cfg(any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090",
                    feature = "cuda-13000"
                ))]
                cufftGetPlanPropertyInt64,
                cufftGetProperty,
                cufftGetSize,
                cufftGetSize1d,
                cufftGetSize2d,
                cufftGetSize3d,
                cufftGetSizeMany,
                cufftGetSizeMany64,
                cufftGetVersion,
                cufftMakePlan1d,
                cufftMakePlan2d,
                cufftMakePlan3d,
                cufftMakePlanMany,
                cufftMakePlanMany64,
                cufftPlan1d,
                cufftPlan2d,
                cufftPlan3d,
                cufftPlanMany,
                #[cfg(any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090",
                    feature = "cuda-13000"
                ))]
                cufftResetPlanProperty,
                cufftSetAutoAllocation,
                #[cfg(any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090",
                    feature = "cuda-13000"
                ))]
                cufftSetPlanPropertyInt64,
                cufftSetStream,
                cufftSetWorkArea,
            })
        }
    }
    pub unsafe fn is_culib_present() -> bool {
        let lib_names = ["cufft"];
        let choices = lib_names
            .iter()
            .map(|l| crate::get_lib_name_candidates(l))
            .flatten();
        for choice in choices {
            if Lib::new(choice).is_ok() {
                return true;
            }
        }
        false
    }
    pub unsafe fn culib() -> &'static Lib {
        static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
        LIB.get_or_init(|| {
            let lib_names = std::vec!["cufft"];
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
