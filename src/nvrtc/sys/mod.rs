#![cfg_attr(feature = "no-std", no_std)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;
pub type nvrtcProgram = *mut _nvrtcProgram;
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080",
    feature = "cuda-12000"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvrtcResult {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
}
#[cfg(any(
    feature = "cuda-12010",
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvrtcResult {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
    NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12,
}
#[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvrtcResult {
    NVRTC_SUCCESS = 0,
    NVRTC_ERROR_OUT_OF_MEMORY = 1,
    NVRTC_ERROR_PROGRAM_CREATION_FAILURE = 2,
    NVRTC_ERROR_INVALID_INPUT = 3,
    NVRTC_ERROR_INVALID_PROGRAM = 4,
    NVRTC_ERROR_INVALID_OPTION = 5,
    NVRTC_ERROR_COMPILATION = 6,
    NVRTC_ERROR_BUILTIN_OPERATION_FAILURE = 7,
    NVRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION = 8,
    NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION = 9,
    NVRTC_ERROR_NAME_EXPRESSION_NOT_VALID = 10,
    NVRTC_ERROR_INTERNAL_ERROR = 11,
    NVRTC_ERROR_TIME_FILE_WRITE_FAILED = 12,
    NVRTC_ERROR_NO_PCH_CREATE_ATTEMPTED = 13,
    NVRTC_ERROR_PCH_CREATE_HEAP_EXHAUSTED = 14,
    NVRTC_ERROR_PCH_CREATE = 15,
    NVRTC_ERROR_CANCELLED = 16,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _nvrtcProgram {
    _unused: [u8; 0],
}
#[cfg(not(feature = "dynamic-loading"))]
extern "C" {
    pub fn nvrtcAddNameExpression(
        prog: nvrtcProgram,
        name_expression: *const ::core::ffi::c_char,
    ) -> nvrtcResult;
    pub fn nvrtcCompileProgram(
        prog: nvrtcProgram,
        numOptions: ::core::ffi::c_int,
        options: *const *const ::core::ffi::c_char,
    ) -> nvrtcResult;
    pub fn nvrtcCreateProgram(
        prog: *mut nvrtcProgram,
        src: *const ::core::ffi::c_char,
        name: *const ::core::ffi::c_char,
        numHeaders: ::core::ffi::c_int,
        headers: *const *const ::core::ffi::c_char,
        includeNames: *const *const ::core::ffi::c_char,
    ) -> nvrtcResult;
    pub fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;
    pub fn nvrtcGetCUBIN(prog: nvrtcProgram, cubin: *mut ::core::ffi::c_char) -> nvrtcResult;
    pub fn nvrtcGetCUBINSize(prog: nvrtcProgram, cubinSizeRet: *mut usize) -> nvrtcResult;
    pub fn nvrtcGetErrorString(result: nvrtcResult) -> *const ::core::ffi::c_char;
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
    pub fn nvrtcGetLTOIR(prog: nvrtcProgram, LTOIR: *mut ::core::ffi::c_char) -> nvrtcResult;
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
    pub fn nvrtcGetLTOIRSize(prog: nvrtcProgram, LTOIRSizeRet: *mut usize) -> nvrtcResult;
    pub fn nvrtcGetLoweredName(
        prog: nvrtcProgram,
        name_expression: *const ::core::ffi::c_char,
        lowered_name: *mut *const ::core::ffi::c_char,
    ) -> nvrtcResult;
    pub fn nvrtcGetNVVM(prog: nvrtcProgram, nvvm: *mut ::core::ffi::c_char) -> nvrtcResult;
    pub fn nvrtcGetNVVMSize(prog: nvrtcProgram, nvvmSizeRet: *mut usize) -> nvrtcResult;
    pub fn nvrtcGetNumSupportedArchs(numArchs: *mut ::core::ffi::c_int) -> nvrtcResult;
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
    pub fn nvrtcGetOptiXIR(prog: nvrtcProgram, optixir: *mut ::core::ffi::c_char) -> nvrtcResult;
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
    pub fn nvrtcGetOptiXIRSize(prog: nvrtcProgram, optixirSizeRet: *mut usize) -> nvrtcResult;
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn nvrtcGetPCHCreateStatus(prog: nvrtcProgram) -> nvrtcResult;
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn nvrtcGetPCHHeapSize(ret: *mut usize) -> nvrtcResult;
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn nvrtcGetPCHHeapSizeRequired(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;
    pub fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut ::core::ffi::c_char) -> nvrtcResult;
    pub fn nvrtcGetPTXSize(prog: nvrtcProgram, ptxSizeRet: *mut usize) -> nvrtcResult;
    pub fn nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut ::core::ffi::c_char) -> nvrtcResult;
    pub fn nvrtcGetProgramLogSize(prog: nvrtcProgram, logSizeRet: *mut usize) -> nvrtcResult;
    pub fn nvrtcGetSupportedArchs(supportedArchs: *mut ::core::ffi::c_int) -> nvrtcResult;
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn nvrtcSetFlowCallback(
        prog: nvrtcProgram,
        callback: ::core::option::Option<
            unsafe extern "C" fn(
                arg1: *mut ::core::ffi::c_void,
                arg2: *mut ::core::ffi::c_void,
            ) -> ::core::ffi::c_int,
        >,
        payload: *mut ::core::ffi::c_void,
    ) -> nvrtcResult;
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn nvrtcSetPCHHeapSize(size: usize) -> nvrtcResult;
    pub fn nvrtcVersion(
        major: *mut ::core::ffi::c_int,
        minor: *mut ::core::ffi::c_int,
    ) -> nvrtcResult;
}
#[cfg(feature = "dynamic-loading")]
mod loaded {
    use super::*;
    pub unsafe fn nvrtcAddNameExpression(
        prog: nvrtcProgram,
        name_expression: *const ::core::ffi::c_char,
    ) -> nvrtcResult {
        (culib().nvrtcAddNameExpression)(prog, name_expression)
    }
    pub unsafe fn nvrtcCompileProgram(
        prog: nvrtcProgram,
        numOptions: ::core::ffi::c_int,
        options: *const *const ::core::ffi::c_char,
    ) -> nvrtcResult {
        (culib().nvrtcCompileProgram)(prog, numOptions, options)
    }
    pub unsafe fn nvrtcCreateProgram(
        prog: *mut nvrtcProgram,
        src: *const ::core::ffi::c_char,
        name: *const ::core::ffi::c_char,
        numHeaders: ::core::ffi::c_int,
        headers: *const *const ::core::ffi::c_char,
        includeNames: *const *const ::core::ffi::c_char,
    ) -> nvrtcResult {
        (culib().nvrtcCreateProgram)(prog, src, name, numHeaders, headers, includeNames)
    }
    pub unsafe fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult {
        (culib().nvrtcDestroyProgram)(prog)
    }
    pub unsafe fn nvrtcGetCUBIN(
        prog: nvrtcProgram,
        cubin: *mut ::core::ffi::c_char,
    ) -> nvrtcResult {
        (culib().nvrtcGetCUBIN)(prog, cubin)
    }
    pub unsafe fn nvrtcGetCUBINSize(prog: nvrtcProgram, cubinSizeRet: *mut usize) -> nvrtcResult {
        (culib().nvrtcGetCUBINSize)(prog, cubinSizeRet)
    }
    pub unsafe fn nvrtcGetErrorString(result: nvrtcResult) -> *const ::core::ffi::c_char {
        (culib().nvrtcGetErrorString)(result)
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
    pub unsafe fn nvrtcGetLTOIR(
        prog: nvrtcProgram,
        LTOIR: *mut ::core::ffi::c_char,
    ) -> nvrtcResult {
        (culib().nvrtcGetLTOIR)(prog, LTOIR)
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
    pub unsafe fn nvrtcGetLTOIRSize(prog: nvrtcProgram, LTOIRSizeRet: *mut usize) -> nvrtcResult {
        (culib().nvrtcGetLTOIRSize)(prog, LTOIRSizeRet)
    }
    pub unsafe fn nvrtcGetLoweredName(
        prog: nvrtcProgram,
        name_expression: *const ::core::ffi::c_char,
        lowered_name: *mut *const ::core::ffi::c_char,
    ) -> nvrtcResult {
        (culib().nvrtcGetLoweredName)(prog, name_expression, lowered_name)
    }
    pub unsafe fn nvrtcGetNVVM(prog: nvrtcProgram, nvvm: *mut ::core::ffi::c_char) -> nvrtcResult {
        (culib().nvrtcGetNVVM)(prog, nvvm)
    }
    pub unsafe fn nvrtcGetNVVMSize(prog: nvrtcProgram, nvvmSizeRet: *mut usize) -> nvrtcResult {
        (culib().nvrtcGetNVVMSize)(prog, nvvmSizeRet)
    }
    pub unsafe fn nvrtcGetNumSupportedArchs(numArchs: *mut ::core::ffi::c_int) -> nvrtcResult {
        (culib().nvrtcGetNumSupportedArchs)(numArchs)
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
    pub unsafe fn nvrtcGetOptiXIR(
        prog: nvrtcProgram,
        optixir: *mut ::core::ffi::c_char,
    ) -> nvrtcResult {
        (culib().nvrtcGetOptiXIR)(prog, optixir)
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
    pub unsafe fn nvrtcGetOptiXIRSize(
        prog: nvrtcProgram,
        optixirSizeRet: *mut usize,
    ) -> nvrtcResult {
        (culib().nvrtcGetOptiXIRSize)(prog, optixirSizeRet)
    }
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn nvrtcGetPCHCreateStatus(prog: nvrtcProgram) -> nvrtcResult {
        (culib().nvrtcGetPCHCreateStatus)(prog)
    }
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn nvrtcGetPCHHeapSize(ret: *mut usize) -> nvrtcResult {
        (culib().nvrtcGetPCHHeapSize)(ret)
    }
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn nvrtcGetPCHHeapSizeRequired(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult {
        (culib().nvrtcGetPCHHeapSizeRequired)(prog, size)
    }
    pub unsafe fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut ::core::ffi::c_char) -> nvrtcResult {
        (culib().nvrtcGetPTX)(prog, ptx)
    }
    pub unsafe fn nvrtcGetPTXSize(prog: nvrtcProgram, ptxSizeRet: *mut usize) -> nvrtcResult {
        (culib().nvrtcGetPTXSize)(prog, ptxSizeRet)
    }
    pub unsafe fn nvrtcGetProgramLog(
        prog: nvrtcProgram,
        log: *mut ::core::ffi::c_char,
    ) -> nvrtcResult {
        (culib().nvrtcGetProgramLog)(prog, log)
    }
    pub unsafe fn nvrtcGetProgramLogSize(
        prog: nvrtcProgram,
        logSizeRet: *mut usize,
    ) -> nvrtcResult {
        (culib().nvrtcGetProgramLogSize)(prog, logSizeRet)
    }
    pub unsafe fn nvrtcGetSupportedArchs(supportedArchs: *mut ::core::ffi::c_int) -> nvrtcResult {
        (culib().nvrtcGetSupportedArchs)(supportedArchs)
    }
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn nvrtcSetFlowCallback(
        prog: nvrtcProgram,
        callback: ::core::option::Option<
            unsafe extern "C" fn(
                arg1: *mut ::core::ffi::c_void,
                arg2: *mut ::core::ffi::c_void,
            ) -> ::core::ffi::c_int,
        >,
        payload: *mut ::core::ffi::c_void,
    ) -> nvrtcResult {
        (culib().nvrtcSetFlowCallback)(prog, callback, payload)
    }
    #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn nvrtcSetPCHHeapSize(size: usize) -> nvrtcResult {
        (culib().nvrtcSetPCHHeapSize)(size)
    }
    pub unsafe fn nvrtcVersion(
        major: *mut ::core::ffi::c_int,
        minor: *mut ::core::ffi::c_int,
    ) -> nvrtcResult {
        (culib().nvrtcVersion)(major, minor)
    }
    pub struct Lib {
        __library: ::libloading::Library,
        pub nvrtcAddNameExpression: unsafe extern "C" fn(
            prog: nvrtcProgram,
            name_expression: *const ::core::ffi::c_char,
        ) -> nvrtcResult,
        pub nvrtcCompileProgram: unsafe extern "C" fn(
            prog: nvrtcProgram,
            numOptions: ::core::ffi::c_int,
            options: *const *const ::core::ffi::c_char,
        ) -> nvrtcResult,
        pub nvrtcCreateProgram: unsafe extern "C" fn(
            prog: *mut nvrtcProgram,
            src: *const ::core::ffi::c_char,
            name: *const ::core::ffi::c_char,
            numHeaders: ::core::ffi::c_int,
            headers: *const *const ::core::ffi::c_char,
            includeNames: *const *const ::core::ffi::c_char,
        ) -> nvrtcResult,
        pub nvrtcDestroyProgram: unsafe extern "C" fn(prog: *mut nvrtcProgram) -> nvrtcResult,
        pub nvrtcGetCUBIN: unsafe extern "C" fn(
            prog: nvrtcProgram,
            cubin: *mut ::core::ffi::c_char,
        ) -> nvrtcResult,
        pub nvrtcGetCUBINSize:
            unsafe extern "C" fn(prog: nvrtcProgram, cubinSizeRet: *mut usize) -> nvrtcResult,
        pub nvrtcGetErrorString:
            unsafe extern "C" fn(result: nvrtcResult) -> *const ::core::ffi::c_char,
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
        pub nvrtcGetLTOIR: unsafe extern "C" fn(
            prog: nvrtcProgram,
            LTOIR: *mut ::core::ffi::c_char,
        ) -> nvrtcResult,
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
        pub nvrtcGetLTOIRSize:
            unsafe extern "C" fn(prog: nvrtcProgram, LTOIRSizeRet: *mut usize) -> nvrtcResult,
        pub nvrtcGetLoweredName: unsafe extern "C" fn(
            prog: nvrtcProgram,
            name_expression: *const ::core::ffi::c_char,
            lowered_name: *mut *const ::core::ffi::c_char,
        ) -> nvrtcResult,
        pub nvrtcGetNVVM:
            unsafe extern "C" fn(prog: nvrtcProgram, nvvm: *mut ::core::ffi::c_char) -> nvrtcResult,
        pub nvrtcGetNVVMSize:
            unsafe extern "C" fn(prog: nvrtcProgram, nvvmSizeRet: *mut usize) -> nvrtcResult,
        pub nvrtcGetNumSupportedArchs:
            unsafe extern "C" fn(numArchs: *mut ::core::ffi::c_int) -> nvrtcResult,
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
        pub nvrtcGetOptiXIR: unsafe extern "C" fn(
            prog: nvrtcProgram,
            optixir: *mut ::core::ffi::c_char,
        ) -> nvrtcResult,
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
        pub nvrtcGetOptiXIRSize:
            unsafe extern "C" fn(prog: nvrtcProgram, optixirSizeRet: *mut usize) -> nvrtcResult,
        #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
        pub nvrtcGetPCHCreateStatus: unsafe extern "C" fn(prog: nvrtcProgram) -> nvrtcResult,
        #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
        pub nvrtcGetPCHHeapSize: unsafe extern "C" fn(ret: *mut usize) -> nvrtcResult,
        #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
        pub nvrtcGetPCHHeapSizeRequired:
            unsafe extern "C" fn(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult,
        pub nvrtcGetPTX:
            unsafe extern "C" fn(prog: nvrtcProgram, ptx: *mut ::core::ffi::c_char) -> nvrtcResult,
        pub nvrtcGetPTXSize:
            unsafe extern "C" fn(prog: nvrtcProgram, ptxSizeRet: *mut usize) -> nvrtcResult,
        pub nvrtcGetProgramLog:
            unsafe extern "C" fn(prog: nvrtcProgram, log: *mut ::core::ffi::c_char) -> nvrtcResult,
        pub nvrtcGetProgramLogSize:
            unsafe extern "C" fn(prog: nvrtcProgram, logSizeRet: *mut usize) -> nvrtcResult,
        pub nvrtcGetSupportedArchs:
            unsafe extern "C" fn(supportedArchs: *mut ::core::ffi::c_int) -> nvrtcResult,
        #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
        pub nvrtcSetFlowCallback: unsafe extern "C" fn(
            prog: nvrtcProgram,
            callback: ::core::option::Option<
                unsafe extern "C" fn(
                    arg1: *mut ::core::ffi::c_void,
                    arg2: *mut ::core::ffi::c_void,
                ) -> ::core::ffi::c_int,
            >,
            payload: *mut ::core::ffi::c_void,
        ) -> nvrtcResult,
        #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
        pub nvrtcSetPCHHeapSize: unsafe extern "C" fn(size: usize) -> nvrtcResult,
        pub nvrtcVersion: unsafe extern "C" fn(
            major: *mut ::core::ffi::c_int,
            minor: *mut ::core::ffi::c_int,
        ) -> nvrtcResult,
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
            let nvrtcAddNameExpression = __library
                .get(b"nvrtcAddNameExpression\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcCompileProgram = __library
                .get(b"nvrtcCompileProgram\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcCreateProgram = __library
                .get(b"nvrtcCreateProgram\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcDestroyProgram = __library
                .get(b"nvrtcDestroyProgram\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcGetCUBIN = __library
                .get(b"nvrtcGetCUBIN\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcGetCUBINSize = __library
                .get(b"nvrtcGetCUBINSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcGetErrorString = __library
                .get(b"nvrtcGetErrorString\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
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
            let nvrtcGetLTOIR = __library
                .get(b"nvrtcGetLTOIR\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
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
            let nvrtcGetLTOIRSize = __library
                .get(b"nvrtcGetLTOIRSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcGetLoweredName = __library
                .get(b"nvrtcGetLoweredName\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcGetNVVM = __library
                .get(b"nvrtcGetNVVM\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcGetNVVMSize = __library
                .get(b"nvrtcGetNVVMSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcGetNumSupportedArchs = __library
                .get(b"nvrtcGetNumSupportedArchs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
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
            let nvrtcGetOptiXIR = __library
                .get(b"nvrtcGetOptiXIR\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
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
            let nvrtcGetOptiXIRSize = __library
                .get(b"nvrtcGetOptiXIRSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
            let nvrtcGetPCHCreateStatus = __library
                .get(b"nvrtcGetPCHCreateStatus\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
            let nvrtcGetPCHHeapSize = __library
                .get(b"nvrtcGetPCHHeapSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
            let nvrtcGetPCHHeapSizeRequired = __library
                .get(b"nvrtcGetPCHHeapSizeRequired\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcGetPTX = __library
                .get(b"nvrtcGetPTX\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcGetPTXSize = __library
                .get(b"nvrtcGetPTXSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcGetProgramLog = __library
                .get(b"nvrtcGetProgramLog\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcGetProgramLogSize = __library
                .get(b"nvrtcGetProgramLogSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcGetSupportedArchs = __library
                .get(b"nvrtcGetSupportedArchs\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
            let nvrtcSetFlowCallback = __library
                .get(b"nvrtcSetFlowCallback\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
            let nvrtcSetPCHHeapSize = __library
                .get(b"nvrtcSetPCHHeapSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvrtcVersion = __library
                .get(b"nvrtcVersion\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            Ok(Self {
                __library,
                nvrtcAddNameExpression,
                nvrtcCompileProgram,
                nvrtcCreateProgram,
                nvrtcDestroyProgram,
                nvrtcGetCUBIN,
                nvrtcGetCUBINSize,
                nvrtcGetErrorString,
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
                nvrtcGetLTOIR,
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
                nvrtcGetLTOIRSize,
                nvrtcGetLoweredName,
                nvrtcGetNVVM,
                nvrtcGetNVVMSize,
                nvrtcGetNumSupportedArchs,
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
                nvrtcGetOptiXIR,
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
                nvrtcGetOptiXIRSize,
                #[cfg(any(
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                nvrtcGetPCHCreateStatus,
                #[cfg(any(
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                nvrtcGetPCHHeapSize,
                #[cfg(any(
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                nvrtcGetPCHHeapSizeRequired,
                nvrtcGetPTX,
                nvrtcGetPTXSize,
                nvrtcGetProgramLog,
                nvrtcGetProgramLogSize,
                nvrtcGetSupportedArchs,
                #[cfg(any(
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                nvrtcSetFlowCallback,
                #[cfg(any(
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                nvrtcSetPCHHeapSize,
                nvrtcVersion,
            })
        }
    }
    pub unsafe fn culib() -> &'static Lib {
        static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
        LIB.get_or_init(|| {
            let lib_names = std::vec!["nvrtc"];
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
