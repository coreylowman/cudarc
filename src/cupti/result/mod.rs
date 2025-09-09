use core::ffi::CStr;

use super::sys;
use crate::driver;

pub mod activity;

/// Wrapper around an erroneous `CUptiResult`. See
/// NVIDIA's [CUDA Runtime API](https://docs.nvidia.com/cupti/api/group__CUPTI__RESULT__API.html?highlight=CUptiResult#_CPPv411CUptiResult)
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct CuptiError(pub sys::CUptiResult);

impl sys::CUptiResult {
    #[inline]
    pub fn result(self) -> Result<(), CuptiError> {
        match self {
            sys::CUptiResult::CUPTI_SUCCESS => Ok(()),
            _ => Err(CuptiError(self)),
        }
    }
}

impl CuptiError {
    /// Gets the error string for this error.
    ///
    /// See [cuptiGetResultString()](https://docs.nvidia.com/cupti/api/group__CUPTI__RESULT__API.html?highlight=cuptiGetErrorMessage#_CPPv420cuptiGetErrorMessage11CUptiResultPPKc)
    pub fn error_string(&self) -> Result<&CStr, CuptiError> {
        let mut err_str = std::ptr::null();
        unsafe {
            sys::cuptiGetResultString(self.0, &mut err_str).result()?;
            Ok(CStr::from_ptr(err_str))
        }
    }
}

impl std::fmt::Debug for CuptiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let err_str = self.error_string().unwrap();
        f.debug_tuple("CuptiError")
            .field(&self.0)
            .field(&err_str)
            .finish()
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CuptiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CuptiError {}

/// Check support for a compute capability.
///
/// See [cuptiComputeCapabilitySupported()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1ga22c5ce610ffbf5940b7c05be54fc813d).
///
/// # Safety
/// Support must exist.
pub unsafe fn compute_capability_supported(
    major: core::ffi::c_int,
    minor: core::ffi::c_int,
    support: *mut core::ffi::c_int,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiComputeCapabilitySupported(major, minor, support) }.result()
}
/// Check support for a compute device.
///
/// See [cuptiDeviceSupported()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1ga2493c952b9ceccf953ade5a6816fefdb).
///
/// # Safety
/// Support must exist.
pub unsafe fn device_supported(
    dev: driver::sys::CUdevice,
    support: *mut core::ffi::c_int,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiDeviceSupported(dev, support) }.result()
}
/// Query the virtualization mode of the device.
///
/// See [cuptiDeviceVirtualizationMode()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1ga395c59b62aeac395e38ced9d40677c76).
///
/// # Safety
/// Mode must exist.
pub unsafe fn device_virtualization_mode(
    dev: driver::sys::CUdevice,
    mode: *mut sys::CUpti_DeviceVirtualizationMode,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiDeviceVirtualizationMode(dev, mode) }.result()
}
/// Detach CUPTI from the running process.
///
/// See [cuptiFinalize()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1gaad1be905ea718ed54246e52e02667e8f).
pub fn finalize() -> Result<(), CuptiError> {
    unsafe { sys::cuptiFinalize() }.result()
}
/// Get auto boost state.
///
/// See [cuptiGetAutoBoostState()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1ga1ac1cce5ce788b9f2c679d13e982384b).
///
/// # Safety
/// State must exist.
pub unsafe fn get_auto_boost_state(
    context: driver::sys::CUcontext,
    state: *mut sys::CUpti_ActivityAutoBoostState,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiGetAutoBoostState(context, state) }.result()
}
/// Get the ID of a context.
///
/// See [cuptiGetContextId()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1ga036dfd802a6c28c7e4239c82ed98df21).
///
/// # Safety
/// Context ID must exist.
pub unsafe fn get_context_id(
    context: driver::sys::CUcontext,
    context_id: *mut u32,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiGetContextId(context, context_id) }.result()
}
/// Get the ID of a device.
///
/// See [cuptiGetDeviceId()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1ga0cc36b42dbf08fffc772e9c932749c77).
///
/// # Safety
/// Device ID must exist.
pub unsafe fn get_device_id(
    context: driver::sys::CUcontext,
    device_id: *mut u32,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiGetDeviceId(context, device_id) }.result()
}
/// Get the unique ID of executable graph.
///
/// See [cuptiGetGraphExecId()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1ga3a3fd5d89e51eeece46635d614624aa3).
///
/// # Safety
/// P ID must exist.
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000"
))]
pub unsafe fn get_graph_exec_id(
    graph_exec: driver::sys::CUgraphExec,
    p_id: *mut u32,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiGetGraphExecId(graph_exec, p_id) }.result()
}
/// Get the unique ID of graph.
///
/// See [cuptiGetGraphId()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1ga4add923efce4731de28c9f0b04e1e3f9).
///
/// # Safety
/// P ID must exist.
pub unsafe fn get_graph_id(graph: driver::sys::CUgraph, p_id: *mut u32) -> Result<(), CuptiError> {
    unsafe { sys::cuptiGetGraphId(graph, p_id) }.result()
}
/// Get the unique ID of a graph node.
///
/// See [cuptiGetGraphNodeId()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1ga22370b53102428305a97cb37fbc14678).
///
/// # Safety
/// Node ID must exist.
pub unsafe fn get_graph_node_id(
    node: driver::sys::CUgraphNode,
    node_id: *mut u64,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiGetGraphNodeId(node, node_id) }.result()
}
/// Returns the last error from a cupti call or callback.
///
/// See [cuptiGetLastError()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1ga0c83719b0248e09ef94390000d3f1035).
pub fn get_last_error() -> Result<(), CuptiError> {
    unsafe { sys::cuptiGetLastError() }.result()
}
/// Get the ID of a stream.
///
/// See [cuptiGetStreamId()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1ga04ece23d24e29e8d98daadba09f1839c).
///
/// # Safety
/// Stream ID must exist.
pub unsafe fn get_stream_id(
    context: driver::sys::CUcontext,
    stream: driver::sys::CUstream,
    stream_id: *mut u32,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiGetStreamId(context, stream, stream_id) }.result()
}
/// Get the ID of a stream.
///
/// See [cuptiGetStreamIdEx()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1ga062d04c62fdfeed9adb8157cecbaaa55).
///
/// # Safety
/// Stream ID must exist.
pub unsafe fn get_stream_id_ex(
    context: driver::sys::CUcontext,
    stream: driver::sys::CUstream,
    per_thread_stream: u8,
    stream_id: *mut u32,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiGetStreamIdEx(context, stream, per_thread_stream, stream_id) }.result()
}
/// Get the thread-id type.
///
/// See [cuptiGetThreadIdType()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1gabc957f426b741e46d6e9a99a43a974b5).
///
/// # Safety
/// Type must exist.
pub unsafe fn get_thread_id_type(
    r#type: *mut sys::CUpti_ActivityThreadIdType,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiGetThreadIdType(r#type) }.result()
}
/// Get the CUPTI timestamp.
///
/// See [cuptiGetTimestamp()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1ga7d8294c686b5293237a6daae8eae3dde).
///
/// # Safety
/// Timestamp must exist.
pub unsafe fn get_timestamp(timestamp: *mut u64) -> Result<(), CuptiError> {
    unsafe { sys::cuptiGetTimestamp(timestamp) }.result()
}
/// Set the thread-id type.
///
/// See [cuptiSetThreadIdType()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#group__cupti__activity__api_1ga1821f090b841d60643ee37d977d9c64a).
pub fn set_thread_id_type(r#type: sys::CUpti_ActivityThreadIdType) -> Result<(), CuptiError> {
    unsafe { sys::cuptiSetThreadIdType(r#type) }.result()
}

/// Get the name of a callback for a specific domain and callback ID.
///
/// See [cuptiGetCallbackName()](https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html?highlight=cuptiGetCallbackName#_CPPv420cuptiGetCallbackName20CUpti_CallbackDomain8uint32_tPPKc)
///
/// # Safety
/// Name pointer must exist.
pub unsafe fn get_callback_name(
    domain: sys::CUpti_CallbackDomain,
    cbid: u32,
    name: *mut *const core::ffi::c_char,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiGetCallbackName(domain, cbid, name) }.result()
}

/// Get the current enabled/disabled state of a callback for a specific domain and function ID.
///
/// See [cuptiGetCallbackState()](https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html?highlight=cuptiGetCallbackName#_CPPv421cuptiGetCallbackStateP8uint32_t22CUpti_SubscriberHandle20CUpti_CallbackDomain16CUpti_CallbackId)
///
/// # Safety
/// Enable must exist.
pub unsafe fn get_callback_state(
    enable: *mut u32,
    subscriber: sys::CUpti_SubscriberHandle,
    domain: sys::CUpti_CallbackDomain,
    cbid: sys::CUpti_CallbackId,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiGetCallbackState(enable, subscriber, domain, cbid) }.result()
}

/// Initialize a callback subscriber with a callback function and user data.
///
/// See [cuptiSubscribe()](https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html?highlight=cuptiSubscribe#_CPPv414cuptiSubscribeP22CUpti_SubscriberHandle18CUpti_CallbackFuncPv)
///
/// # Safety
/// Subscriber handle must exist.
/// Callback function must exist.
pub unsafe fn subscribe(
    subscriber: *mut sys::CUpti_SubscriberHandle,
    callback: sys::CUpti_CallbackFunc,
    userdata: *mut std::os::raw::c_void,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiSubscribe(subscriber, callback, userdata) }.result()
}

/// Initialize a callback subscriber with a callback function and user data.
///
/// See [cuptiSubscribe()](https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html?highlight=cuptiSubscribe#_CPPv414cuptiSubscribeP22CUpti_SubscriberHandle18CUpti_CallbackFuncPv)
///
/// # Safety
/// Subscriber handle must exist.
/// Callback function must exist.
/// Subscriber parameters `p_params` may be null.
#[cfg(feature = "cuda-13000")]
pub unsafe fn subscribe_v2(
    subscriber: *mut sys::CUpti_SubscriberHandle,
    callback: sys::CUpti_CallbackFunc,
    userdata: *mut core::ffi::c_void,
    p_params: *mut sys::CUpti_SubscriberParams,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiSubscribe_v2(subscriber, callback, userdata, p_params) }.result()
}

/// Unregister a callback subscriber.
///
/// See [cuptiUnsubscribe()](https://docs.nvidia.com/cupti/api/group__CUPTI__CALLBACK__API.html?highlight=cuptiSubscribe#group__cupti__callback__api_1ga20b68c9c33f129179b56687a17356682)
///
/// # Safety
/// Subscriber must exist
pub unsafe fn unsubscribe(subscriber: sys::CUpti_SubscriberHandle) -> Result<(), CuptiError> {
    unsafe { sys::cuptiUnsubscribe(subscriber) }.result()
}
