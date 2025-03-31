pub fn nvrtcGetErrorString (result : nvrtcResult) -> * const :: core :: ffi :: c_char { unsafe { lib () . nvrtcGetErrorString (result) } }
pub fn nvrtcVersion (major : * mut :: core :: ffi :: c_int , minor : * mut :: core :: ffi :: c_int ,) -> nvrtcResult { unsafe { lib () . nvrtcVersion (major , minor) } }
pub fn nvrtcGetNumSupportedArchs (numArchs : * mut :: core :: ffi :: c_int) -> nvrtcResult { unsafe { lib () . nvrtcGetNumSupportedArchs (numArchs) } }
pub fn nvrtcGetSupportedArchs (supportedArchs : * mut :: core :: ffi :: c_int) -> nvrtcResult { unsafe { lib () . nvrtcGetSupportedArchs (supportedArchs) } }
pub fn nvrtcCreateProgram (prog : * mut nvrtcProgram , src : * const :: core :: ffi :: c_char , name : * const :: core :: ffi :: c_char , numHeaders : :: core :: ffi :: c_int , headers : * const * const :: core :: ffi :: c_char , includeNames : * const * const :: core :: ffi :: c_char ,) -> nvrtcResult { unsafe { lib () . nvrtcCreateProgram (prog , src , name , numHeaders , headers , includeNames) } }
pub fn nvrtcDestroyProgram (prog : * mut nvrtcProgram) -> nvrtcResult { unsafe { lib () . nvrtcDestroyProgram (prog) } }
pub fn nvrtcCompileProgram (prog : nvrtcProgram , numOptions : :: core :: ffi :: c_int , options : * const * const :: core :: ffi :: c_char ,) -> nvrtcResult { unsafe { lib () . nvrtcCompileProgram (prog , numOptions , options) } }
pub fn nvrtcGetPTXSize (prog : nvrtcProgram , ptxSizeRet : * mut usize) -> nvrtcResult { unsafe { lib () . nvrtcGetPTXSize (prog , ptxSizeRet) } }
pub fn nvrtcGetPTX (prog : nvrtcProgram , ptx : * mut :: core :: ffi :: c_char) -> nvrtcResult { unsafe { lib () . nvrtcGetPTX (prog , ptx) } }
pub fn nvrtcGetCUBINSize (prog : nvrtcProgram , cubinSizeRet : * mut usize) -> nvrtcResult { unsafe { lib () . nvrtcGetCUBINSize (prog , cubinSizeRet) } }
pub fn nvrtcGetCUBIN (prog : nvrtcProgram , cubin : * mut :: core :: ffi :: c_char) -> nvrtcResult { unsafe { lib () . nvrtcGetCUBIN (prog , cubin) } }
pub fn nvrtcGetNVVMSize (prog : nvrtcProgram , nvvmSizeRet : * mut usize) -> nvrtcResult { unsafe { lib () . nvrtcGetNVVMSize (prog , nvvmSizeRet) } }
pub fn nvrtcGetNVVM (prog : nvrtcProgram , nvvm : * mut :: core :: ffi :: c_char) -> nvrtcResult { unsafe { lib () . nvrtcGetNVVM (prog , nvvm) } }
pub fn nvrtcGetLTOIRSize (prog : nvrtcProgram , LTOIRSizeRet : * mut usize) -> nvrtcResult { unsafe { lib () . nvrtcGetLTOIRSize (prog , LTOIRSizeRet) } }
pub fn nvrtcGetLTOIR (prog : nvrtcProgram , LTOIR : * mut :: core :: ffi :: c_char) -> nvrtcResult { unsafe { lib () . nvrtcGetLTOIR (prog , LTOIR) } }
pub fn nvrtcGetOptiXIRSize (prog : nvrtcProgram , optixirSizeRet : * mut usize) -> nvrtcResult { unsafe { lib () . nvrtcGetOptiXIRSize (prog , optixirSizeRet) } }
pub fn nvrtcGetOptiXIR (prog : nvrtcProgram , optixir : * mut :: core :: ffi :: c_char) -> nvrtcResult { unsafe { lib () . nvrtcGetOptiXIR (prog , optixir) } }
pub fn nvrtcGetProgramLogSize (prog : nvrtcProgram , logSizeRet : * mut usize) -> nvrtcResult { unsafe { lib () . nvrtcGetProgramLogSize (prog , logSizeRet) } }
pub fn nvrtcGetProgramLog (prog : nvrtcProgram , log : * mut :: core :: ffi :: c_char) -> nvrtcResult { unsafe { lib () . nvrtcGetProgramLog (prog , log) } }
pub fn nvrtcAddNameExpression (prog : nvrtcProgram , name_expression : * const :: core :: ffi :: c_char ,) -> nvrtcResult { unsafe { lib () . nvrtcAddNameExpression (prog , name_expression) } }
pub fn nvrtcGetLoweredName (prog : nvrtcProgram , name_expression : * const :: core :: ffi :: c_char , lowered_name : * mut * const :: core :: ffi :: c_char ,) -> nvrtcResult { unsafe { lib () . nvrtcGetLoweredName (prog , name_expression , lowered_name) } }
pub fn nvrtcGetPCHHeapSize (ret : * mut usize) -> nvrtcResult { unsafe { lib () . nvrtcGetPCHHeapSize (ret) } }
pub fn nvrtcSetPCHHeapSize (size : usize) -> nvrtcResult { unsafe { lib () . nvrtcSetPCHHeapSize (size) } }
pub fn nvrtcGetPCHCreateStatus (prog : nvrtcProgram) -> nvrtcResult { unsafe { lib () . nvrtcGetPCHCreateStatus (prog) } }
pub fn nvrtcGetPCHHeapSizeRequired (prog : nvrtcProgram , size : * mut usize) -> nvrtcResult { unsafe { lib () . nvrtcGetPCHHeapSizeRequired (prog , size) } }
pub fn nvrtcSetFlowCallback (prog : nvrtcProgram , callback : :: core :: option :: Option < unsafe extern "C" fn (arg1 : * mut :: core :: ffi :: c_void , arg2 : * mut :: core :: ffi :: c_void ,) -> :: core :: ffi :: c_int , > , payload : * mut :: core :: ffi :: c_void ,) -> nvrtcResult { unsafe { lib () . nvrtcSetFlowCallback (prog , callback , payload) } }
