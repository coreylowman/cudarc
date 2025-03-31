use super :: * ;pub unsafe fn nvrtcGetErrorString (result : nvrtcResult) -> * const :: core :: ffi :: c_char { unsafe { culib () . nvrtcGetErrorString (result) } }
pub unsafe fn nvrtcVersion (major : * mut :: core :: ffi :: c_int , minor : * mut :: core :: ffi :: c_int ,) -> nvrtcResult { unsafe { culib () . nvrtcVersion (major , minor) } }
pub unsafe fn nvrtcGetNumSupportedArchs (numArchs : * mut :: core :: ffi :: c_int) -> nvrtcResult { unsafe { culib () . nvrtcGetNumSupportedArchs (numArchs) } }
pub unsafe fn nvrtcGetSupportedArchs (supportedArchs : * mut :: core :: ffi :: c_int) -> nvrtcResult { unsafe { culib () . nvrtcGetSupportedArchs (supportedArchs) } }
pub unsafe fn nvrtcCreateProgram (prog : * mut nvrtcProgram , src : * const :: core :: ffi :: c_char , name : * const :: core :: ffi :: c_char , numHeaders : :: core :: ffi :: c_int , headers : * const * const :: core :: ffi :: c_char , includeNames : * const * const :: core :: ffi :: c_char ,) -> nvrtcResult { unsafe { culib () . nvrtcCreateProgram (prog , src , name , numHeaders , headers , includeNames) } }
pub unsafe fn nvrtcDestroyProgram (prog : * mut nvrtcProgram) -> nvrtcResult { unsafe { culib () . nvrtcDestroyProgram (prog) } }
pub unsafe fn nvrtcCompileProgram (prog : nvrtcProgram , numOptions : :: core :: ffi :: c_int , options : * const * const :: core :: ffi :: c_char ,) -> nvrtcResult { unsafe { culib () . nvrtcCompileProgram (prog , numOptions , options) } }
pub unsafe fn nvrtcGetPTXSize (prog : nvrtcProgram , ptxSizeRet : * mut usize) -> nvrtcResult { unsafe { culib () . nvrtcGetPTXSize (prog , ptxSizeRet) } }
pub unsafe fn nvrtcGetPTX (prog : nvrtcProgram , ptx : * mut :: core :: ffi :: c_char) -> nvrtcResult { unsafe { culib () . nvrtcGetPTX (prog , ptx) } }
pub unsafe fn nvrtcGetCUBINSize (prog : nvrtcProgram , cubinSizeRet : * mut usize) -> nvrtcResult { unsafe { culib () . nvrtcGetCUBINSize (prog , cubinSizeRet) } }
pub unsafe fn nvrtcGetCUBIN (prog : nvrtcProgram , cubin : * mut :: core :: ffi :: c_char) -> nvrtcResult { unsafe { culib () . nvrtcGetCUBIN (prog , cubin) } }
pub unsafe fn nvrtcGetNVVMSize (prog : nvrtcProgram , nvvmSizeRet : * mut usize) -> nvrtcResult { unsafe { culib () . nvrtcGetNVVMSize (prog , nvvmSizeRet) } }
pub unsafe fn nvrtcGetNVVM (prog : nvrtcProgram , nvvm : * mut :: core :: ffi :: c_char) -> nvrtcResult { unsafe { culib () . nvrtcGetNVVM (prog , nvvm) } }
pub unsafe fn nvrtcGetLTOIRSize (prog : nvrtcProgram , LTOIRSizeRet : * mut usize) -> nvrtcResult { unsafe { culib () . nvrtcGetLTOIRSize (prog , LTOIRSizeRet) } }
pub unsafe fn nvrtcGetLTOIR (prog : nvrtcProgram , LTOIR : * mut :: core :: ffi :: c_char) -> nvrtcResult { unsafe { culib () . nvrtcGetLTOIR (prog , LTOIR) } }
pub unsafe fn nvrtcGetOptiXIRSize (prog : nvrtcProgram , optixirSizeRet : * mut usize) -> nvrtcResult { unsafe { culib () . nvrtcGetOptiXIRSize (prog , optixirSizeRet) } }
pub unsafe fn nvrtcGetOptiXIR (prog : nvrtcProgram , optixir : * mut :: core :: ffi :: c_char) -> nvrtcResult { unsafe { culib () . nvrtcGetOptiXIR (prog , optixir) } }
pub unsafe fn nvrtcGetProgramLogSize (prog : nvrtcProgram , logSizeRet : * mut usize) -> nvrtcResult { unsafe { culib () . nvrtcGetProgramLogSize (prog , logSizeRet) } }
pub unsafe fn nvrtcGetProgramLog (prog : nvrtcProgram , log : * mut :: core :: ffi :: c_char) -> nvrtcResult { unsafe { culib () . nvrtcGetProgramLog (prog , log) } }
pub unsafe fn nvrtcAddNameExpression (prog : nvrtcProgram , name_expression : * const :: core :: ffi :: c_char ,) -> nvrtcResult { unsafe { culib () . nvrtcAddNameExpression (prog , name_expression) } }
pub unsafe fn nvrtcGetLoweredName (prog : nvrtcProgram , name_expression : * const :: core :: ffi :: c_char , lowered_name : * mut * const :: core :: ffi :: c_char ,) -> nvrtcResult { unsafe { culib () . nvrtcGetLoweredName (prog , name_expression , lowered_name) } }
pub unsafe fn nvrtcGetPCHHeapSize (ret : * mut usize) -> nvrtcResult { unsafe { culib () . nvrtcGetPCHHeapSize (ret) } }
pub unsafe fn nvrtcSetPCHHeapSize (size : usize) -> nvrtcResult { unsafe { culib () . nvrtcSetPCHHeapSize (size) } }
pub unsafe fn nvrtcGetPCHCreateStatus (prog : nvrtcProgram) -> nvrtcResult { unsafe { culib () . nvrtcGetPCHCreateStatus (prog) } }
pub unsafe fn nvrtcGetPCHHeapSizeRequired (prog : nvrtcProgram , size : * mut usize) -> nvrtcResult { unsafe { culib () . nvrtcGetPCHHeapSizeRequired (prog , size) } }
pub unsafe fn nvrtcSetFlowCallback (prog : nvrtcProgram , callback : :: core :: option :: Option < unsafe extern "C" fn (arg1 : * mut :: core :: ffi :: c_void , arg2 : * mut :: core :: ffi :: c_void ,) -> :: core :: ffi :: c_int , > , payload : * mut :: core :: ffi :: c_void ,) -> nvrtcResult { unsafe { culib () . nvrtcSetFlowCallback (prog , callback , payload) } }
