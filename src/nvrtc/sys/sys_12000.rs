/* automatically generated by rust-bindgen 0.71.1 */

pub const CUDA_VERSION: u32 = 12000;
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
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct _nvrtcProgram {
    _unused: [u8; 0],
}
pub type nvrtcProgram = *mut _nvrtcProgram;
pub struct Lib {
    __library: ::libloading::Library,
    pub nvrtcGetErrorString: Result<
        unsafe extern "C" fn(result: nvrtcResult) -> *const ::core::ffi::c_char,
        ::libloading::Error,
    >,
    pub nvrtcVersion: Result<
        unsafe extern "C" fn(
            major: *mut ::core::ffi::c_int,
            minor: *mut ::core::ffi::c_int,
        ) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcGetNumSupportedArchs: Result<
        unsafe extern "C" fn(numArchs: *mut ::core::ffi::c_int) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcGetSupportedArchs: Result<
        unsafe extern "C" fn(supportedArchs: *mut ::core::ffi::c_int) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcCreateProgram: Result<
        unsafe extern "C" fn(
            prog: *mut nvrtcProgram,
            src: *const ::core::ffi::c_char,
            name: *const ::core::ffi::c_char,
            numHeaders: ::core::ffi::c_int,
            headers: *const *const ::core::ffi::c_char,
            includeNames: *const *const ::core::ffi::c_char,
        ) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcDestroyProgram:
        Result<unsafe extern "C" fn(prog: *mut nvrtcProgram) -> nvrtcResult, ::libloading::Error>,
    pub nvrtcCompileProgram: Result<
        unsafe extern "C" fn(
            prog: nvrtcProgram,
            numOptions: ::core::ffi::c_int,
            options: *const *const ::core::ffi::c_char,
        ) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcGetPTXSize: Result<
        unsafe extern "C" fn(prog: nvrtcProgram, ptxSizeRet: *mut usize) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcGetPTX: Result<
        unsafe extern "C" fn(prog: nvrtcProgram, ptx: *mut ::core::ffi::c_char) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcGetCUBINSize: Result<
        unsafe extern "C" fn(prog: nvrtcProgram, cubinSizeRet: *mut usize) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcGetCUBIN: Result<
        unsafe extern "C" fn(prog: nvrtcProgram, cubin: *mut ::core::ffi::c_char) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcGetNVVMSize: Result<
        unsafe extern "C" fn(prog: nvrtcProgram, nvvmSizeRet: *mut usize) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcGetNVVM: Result<
        unsafe extern "C" fn(prog: nvrtcProgram, nvvm: *mut ::core::ffi::c_char) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcGetLTOIRSize: Result<
        unsafe extern "C" fn(prog: nvrtcProgram, LTOIRSizeRet: *mut usize) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcGetLTOIR: Result<
        unsafe extern "C" fn(prog: nvrtcProgram, LTOIR: *mut ::core::ffi::c_char) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcGetOptiXIRSize: Result<
        unsafe extern "C" fn(prog: nvrtcProgram, optixirSizeRet: *mut usize) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcGetOptiXIR: Result<
        unsafe extern "C" fn(prog: nvrtcProgram, optixir: *mut ::core::ffi::c_char) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcGetProgramLogSize: Result<
        unsafe extern "C" fn(prog: nvrtcProgram, logSizeRet: *mut usize) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcGetProgramLog: Result<
        unsafe extern "C" fn(prog: nvrtcProgram, log: *mut ::core::ffi::c_char) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcAddNameExpression: Result<
        unsafe extern "C" fn(
            prog: nvrtcProgram,
            name_expression: *const ::core::ffi::c_char,
        ) -> nvrtcResult,
        ::libloading::Error,
    >,
    pub nvrtcGetLoweredName: Result<
        unsafe extern "C" fn(
            prog: nvrtcProgram,
            name_expression: *const ::core::ffi::c_char,
            lowered_name: *mut *const ::core::ffi::c_char,
        ) -> nvrtcResult,
        ::libloading::Error,
    >,
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
        let nvrtcGetErrorString = __library.get(b"nvrtcGetErrorString\0").map(|sym| *sym);
        let nvrtcVersion = __library.get(b"nvrtcVersion\0").map(|sym| *sym);
        let nvrtcGetNumSupportedArchs = __library
            .get(b"nvrtcGetNumSupportedArchs\0")
            .map(|sym| *sym);
        let nvrtcGetSupportedArchs = __library.get(b"nvrtcGetSupportedArchs\0").map(|sym| *sym);
        let nvrtcCreateProgram = __library.get(b"nvrtcCreateProgram\0").map(|sym| *sym);
        let nvrtcDestroyProgram = __library.get(b"nvrtcDestroyProgram\0").map(|sym| *sym);
        let nvrtcCompileProgram = __library.get(b"nvrtcCompileProgram\0").map(|sym| *sym);
        let nvrtcGetPTXSize = __library.get(b"nvrtcGetPTXSize\0").map(|sym| *sym);
        let nvrtcGetPTX = __library.get(b"nvrtcGetPTX\0").map(|sym| *sym);
        let nvrtcGetCUBINSize = __library.get(b"nvrtcGetCUBINSize\0").map(|sym| *sym);
        let nvrtcGetCUBIN = __library.get(b"nvrtcGetCUBIN\0").map(|sym| *sym);
        let nvrtcGetNVVMSize = __library.get(b"nvrtcGetNVVMSize\0").map(|sym| *sym);
        let nvrtcGetNVVM = __library.get(b"nvrtcGetNVVM\0").map(|sym| *sym);
        let nvrtcGetLTOIRSize = __library.get(b"nvrtcGetLTOIRSize\0").map(|sym| *sym);
        let nvrtcGetLTOIR = __library.get(b"nvrtcGetLTOIR\0").map(|sym| *sym);
        let nvrtcGetOptiXIRSize = __library.get(b"nvrtcGetOptiXIRSize\0").map(|sym| *sym);
        let nvrtcGetOptiXIR = __library.get(b"nvrtcGetOptiXIR\0").map(|sym| *sym);
        let nvrtcGetProgramLogSize = __library.get(b"nvrtcGetProgramLogSize\0").map(|sym| *sym);
        let nvrtcGetProgramLog = __library.get(b"nvrtcGetProgramLog\0").map(|sym| *sym);
        let nvrtcAddNameExpression = __library.get(b"nvrtcAddNameExpression\0").map(|sym| *sym);
        let nvrtcGetLoweredName = __library.get(b"nvrtcGetLoweredName\0").map(|sym| *sym);
        Ok(Lib {
            __library,
            nvrtcGetErrorString,
            nvrtcVersion,
            nvrtcGetNumSupportedArchs,
            nvrtcGetSupportedArchs,
            nvrtcCreateProgram,
            nvrtcDestroyProgram,
            nvrtcCompileProgram,
            nvrtcGetPTXSize,
            nvrtcGetPTX,
            nvrtcGetCUBINSize,
            nvrtcGetCUBIN,
            nvrtcGetNVVMSize,
            nvrtcGetNVVM,
            nvrtcGetLTOIRSize,
            nvrtcGetLTOIR,
            nvrtcGetOptiXIRSize,
            nvrtcGetOptiXIR,
            nvrtcGetProgramLogSize,
            nvrtcGetProgramLog,
            nvrtcAddNameExpression,
            nvrtcGetLoweredName,
        })
    }
    pub unsafe fn nvrtcGetErrorString(&self, result: nvrtcResult) -> *const ::core::ffi::c_char {
        (self
            .nvrtcGetErrorString
            .as_ref()
            .expect("Expected function, got error."))(result)
    }
    pub unsafe fn nvrtcVersion(
        &self,
        major: *mut ::core::ffi::c_int,
        minor: *mut ::core::ffi::c_int,
    ) -> nvrtcResult {
        (self
            .nvrtcVersion
            .as_ref()
            .expect("Expected function, got error."))(major, minor)
    }
    pub unsafe fn nvrtcGetNumSupportedArchs(
        &self,
        numArchs: *mut ::core::ffi::c_int,
    ) -> nvrtcResult {
        (self
            .nvrtcGetNumSupportedArchs
            .as_ref()
            .expect("Expected function, got error."))(numArchs)
    }
    pub unsafe fn nvrtcGetSupportedArchs(
        &self,
        supportedArchs: *mut ::core::ffi::c_int,
    ) -> nvrtcResult {
        (self
            .nvrtcGetSupportedArchs
            .as_ref()
            .expect("Expected function, got error."))(supportedArchs)
    }
    pub unsafe fn nvrtcCreateProgram(
        &self,
        prog: *mut nvrtcProgram,
        src: *const ::core::ffi::c_char,
        name: *const ::core::ffi::c_char,
        numHeaders: ::core::ffi::c_int,
        headers: *const *const ::core::ffi::c_char,
        includeNames: *const *const ::core::ffi::c_char,
    ) -> nvrtcResult {
        (self
            .nvrtcCreateProgram
            .as_ref()
            .expect("Expected function, got error."))(
            prog,
            src,
            name,
            numHeaders,
            headers,
            includeNames,
        )
    }
    pub unsafe fn nvrtcDestroyProgram(&self, prog: *mut nvrtcProgram) -> nvrtcResult {
        (self
            .nvrtcDestroyProgram
            .as_ref()
            .expect("Expected function, got error."))(prog)
    }
    pub unsafe fn nvrtcCompileProgram(
        &self,
        prog: nvrtcProgram,
        numOptions: ::core::ffi::c_int,
        options: *const *const ::core::ffi::c_char,
    ) -> nvrtcResult {
        (self
            .nvrtcCompileProgram
            .as_ref()
            .expect("Expected function, got error."))(prog, numOptions, options)
    }
    pub unsafe fn nvrtcGetPTXSize(
        &self,
        prog: nvrtcProgram,
        ptxSizeRet: *mut usize,
    ) -> nvrtcResult {
        (self
            .nvrtcGetPTXSize
            .as_ref()
            .expect("Expected function, got error."))(prog, ptxSizeRet)
    }
    pub unsafe fn nvrtcGetPTX(
        &self,
        prog: nvrtcProgram,
        ptx: *mut ::core::ffi::c_char,
    ) -> nvrtcResult {
        (self
            .nvrtcGetPTX
            .as_ref()
            .expect("Expected function, got error."))(prog, ptx)
    }
    pub unsafe fn nvrtcGetCUBINSize(
        &self,
        prog: nvrtcProgram,
        cubinSizeRet: *mut usize,
    ) -> nvrtcResult {
        (self
            .nvrtcGetCUBINSize
            .as_ref()
            .expect("Expected function, got error."))(prog, cubinSizeRet)
    }
    pub unsafe fn nvrtcGetCUBIN(
        &self,
        prog: nvrtcProgram,
        cubin: *mut ::core::ffi::c_char,
    ) -> nvrtcResult {
        (self
            .nvrtcGetCUBIN
            .as_ref()
            .expect("Expected function, got error."))(prog, cubin)
    }
    pub unsafe fn nvrtcGetNVVMSize(
        &self,
        prog: nvrtcProgram,
        nvvmSizeRet: *mut usize,
    ) -> nvrtcResult {
        (self
            .nvrtcGetNVVMSize
            .as_ref()
            .expect("Expected function, got error."))(prog, nvvmSizeRet)
    }
    pub unsafe fn nvrtcGetNVVM(
        &self,
        prog: nvrtcProgram,
        nvvm: *mut ::core::ffi::c_char,
    ) -> nvrtcResult {
        (self
            .nvrtcGetNVVM
            .as_ref()
            .expect("Expected function, got error."))(prog, nvvm)
    }
    pub unsafe fn nvrtcGetLTOIRSize(
        &self,
        prog: nvrtcProgram,
        LTOIRSizeRet: *mut usize,
    ) -> nvrtcResult {
        (self
            .nvrtcGetLTOIRSize
            .as_ref()
            .expect("Expected function, got error."))(prog, LTOIRSizeRet)
    }
    pub unsafe fn nvrtcGetLTOIR(
        &self,
        prog: nvrtcProgram,
        LTOIR: *mut ::core::ffi::c_char,
    ) -> nvrtcResult {
        (self
            .nvrtcGetLTOIR
            .as_ref()
            .expect("Expected function, got error."))(prog, LTOIR)
    }
    pub unsafe fn nvrtcGetOptiXIRSize(
        &self,
        prog: nvrtcProgram,
        optixirSizeRet: *mut usize,
    ) -> nvrtcResult {
        (self
            .nvrtcGetOptiXIRSize
            .as_ref()
            .expect("Expected function, got error."))(prog, optixirSizeRet)
    }
    pub unsafe fn nvrtcGetOptiXIR(
        &self,
        prog: nvrtcProgram,
        optixir: *mut ::core::ffi::c_char,
    ) -> nvrtcResult {
        (self
            .nvrtcGetOptiXIR
            .as_ref()
            .expect("Expected function, got error."))(prog, optixir)
    }
    pub unsafe fn nvrtcGetProgramLogSize(
        &self,
        prog: nvrtcProgram,
        logSizeRet: *mut usize,
    ) -> nvrtcResult {
        (self
            .nvrtcGetProgramLogSize
            .as_ref()
            .expect("Expected function, got error."))(prog, logSizeRet)
    }
    pub unsafe fn nvrtcGetProgramLog(
        &self,
        prog: nvrtcProgram,
        log: *mut ::core::ffi::c_char,
    ) -> nvrtcResult {
        (self
            .nvrtcGetProgramLog
            .as_ref()
            .expect("Expected function, got error."))(prog, log)
    }
    pub unsafe fn nvrtcAddNameExpression(
        &self,
        prog: nvrtcProgram,
        name_expression: *const ::core::ffi::c_char,
    ) -> nvrtcResult {
        (self
            .nvrtcAddNameExpression
            .as_ref()
            .expect("Expected function, got error."))(prog, name_expression)
    }
    pub unsafe fn nvrtcGetLoweredName(
        &self,
        prog: nvrtcProgram,
        name_expression: *const ::core::ffi::c_char,
        lowered_name: *mut *const ::core::ffi::c_char,
    ) -> nvrtcResult {
        (self
            .nvrtcGetLoweredName
            .as_ref()
            .expect("Expected function, got error."))(prog, name_expression, lowered_name)
    }
}
