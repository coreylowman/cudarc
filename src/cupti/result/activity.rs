//! Functions of the Activity API

use super::super::{result::CuptiError, sys};

/// Set PC sampling configuration.
///
/// See [cuptiActivityConfigurePCSampling()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityEnableContext#_CPPv432cuptiActivityConfigurePCSampling9CUcontextP30CUpti_ActivityPCSamplingConfig)
///
/// # Safety
/// Context must exist.
/// Config must exist.
pub unsafe fn configure_pc_sampling(
    ctx: sys::CUcontext,
    config: *mut sys::CUpti_ActivityPCSamplingConfig,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityConfigurePCSampling(ctx, config) }.result()
}

/// Set Unified Memory Counter configuration.
///
/// See [cuptiActivityConfigureUnifiedMemoryCounter()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityEnableContext#_CPPv442cuptiActivityConfigureUnifiedMemoryCounterP40CUpti_ActivityUnifiedMemoryCounterConfig8uint32_t)
///
/// # Safety
/// Config must exist.
pub unsafe fn configure_unified_memory_counter(
    config: *mut sys::CUpti_ActivityUnifiedMemoryCounterConfig,
    count: u32,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityConfigureUnifiedMemoryCounter(config, count) }.result()
}

/// Disable collection of a specific kind of activity record.
///
/// See [cuptiActivityDisable()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityEnableContext#_CPPv420cuptiActivityDisable18CUpti_ActivityKind)
pub fn disable(kind: sys::CUpti_ActivityKind) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityDisable(kind) }.result()
}

/// Disable collection of a specific kind of activity record for a context.
///
/// See [cuptiActivityDisableContext()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityEnableContext#_CPPv427cuptiActivityDisableContext9CUcontext18CUpti_ActivityKind)
///
/// # Safety
/// Context must exist.
pub unsafe fn disable_context(
    context: sys::CUcontext,
    kind: sys::CUpti_ActivityKind,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityDisableContext(context, kind) }.result()
}

/// Enable collection of a specific kind of activity record.
///
/// See [cuptiActivityEnable()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityEnable#_CPPv419cuptiActivityEnable18CUpti_ActivityKind)
pub fn enable(kind: sys::CUpti_ActivityKind) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityEnable(kind) }.result()
}

/// Enables collecting records for all synchronization operations.
///
/// See [](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityEnableContext#_CPPv433cuptiActivityEnableAllSyncRecords7uint8_t)
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
pub fn enable_all_sync_records(enable: u8) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityEnableAllSyncRecords(enable) }.result()
}

/// Enables tracking the source library for memory allocation requests.
///
/// See [](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityEnableContext#_CPPv435cuptiActivityEnableAllocationSource7uint8_t)
#[cfg(any(
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000"
))]
pub fn enable_allocation_source(enable: u8) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityEnableAllocationSource(enable) }.result()
}

/// Enable collection of a specific kind of activity record.
///
/// See [cuptiActivityEnableAndDump()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityEnableContext#_CPPv426cuptiActivityEnableAndDump18CUpti_ActivityKind)
#[cfg(any(
    feature = "cuda-11070",
    feature = "cuda-11080",
    feature = "cuda-12000",
    feature = "cuda-12010",
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000"
))]
pub fn enable_and_dump(kind: sys::CUpti_ActivityKind) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityEnableAndDump(kind) }.result()
}

/// Enable collection of a specific kind of activity record for a context.
///
/// See [cuptiActivityEnableContext()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityEnableContext#_CPPv426cuptiActivityEnableContext9CUcontext18CUpti_ActivityKind)
///
/// # Safety
/// Context must exist
pub unsafe fn enable_context(
    context: sys::CUcontext,
    kind: sys::CUpti_ActivityKind,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityEnableContext(context, kind) }.result()
}

/// Enable/Disable collecting device timestamp for CUPTI_ACTIVITY_KIND_CUDA_EVENT record.
///
/// See [cuptiActivityEnableCudaEventDeviceTimestamps()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityEnableContext#_CPPv444cuptiActivityEnableCudaEventDeviceTimestamps7uint8_t)
#[cfg(feature = "cuda-13000")]
pub unsafe fn enable_cuda_event_device_timestamps(enable: u8) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityEnableCudaEventDeviceTimestamps(enable) }.result()
}

/// Controls the collection of records for device launched graphs.
///
/// See [cuptiActivityEnableDeviceGraph()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#_CPPv430cuptiActivityEnableDeviceGraph7uint8_t)
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000"
))]
pub fn enable_device_graph(enable: u8) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityEnableDeviceGraph(enable) }.result()
}

/// Controls the collection of activity records for specific CUDA Driver APIs.
///
/// See [cuptiActivityEnableDriverApi](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#_CPPv428cuptiActivityEnableDriverApi16CUpti_CallbackId7uint8_t)
#[cfg(any(
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000"
))]
pub fn enable_driver_api(cbid: sys::CUpti_CallbackId, enable: u8) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityEnableDriverApi(cbid, enable) }.result()
}

/// Enables the collection of CUDA kernel timestamps through Hardware Event System(HES).
///
/// See [cuptiActivityEnableHWTrace()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#_CPPv426cuptiActivityEnableHWTrace7uint8_t)
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090", feature = "cuda-13000"))]
pub fn enable_hw_trace(enable: u8) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityEnableHWTrace(enable) }.result()
}

/// Controls the collection of queued and submitted timestamps for kernels.
///
/// See [cuptiActivityEnableLatencyTimestamps()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#_CPPv436cuptiActivityEnableLatencyTimestamps7uint8_t)
pub fn enable_latency_timestamps(enable: u8) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityEnableLatencyTimestamps(enable) }.result()
}

/// Controls the collection of launch attributes for kernels.
///
/// See [cuptiActivityEnableLaunchAttributes()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#_CPPv435cuptiActivityEnableLaunchAttributes7uint8_t)
pub fn enable_launch_attributes(enable: u8) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityEnableLaunchAttributes(enable) }.result()
}

/// Controls the collection of activity records for specific CUDA Runtime APIs.
///
/// See [cuptiActivityEnableRuntimeApi](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#_CPPv429cuptiActivityEnableRuntimeApi16CUpti_CallbackId7uint8_t)
#[cfg(any(
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000"
))]
pub fn enable_runtime_api(cbid: sys::CUpti_CallbackId, enable: u8) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityEnableRuntimeApi(cbid, enable) }.result()
}

/// Wait for all activity records to be delivered via the completion callback.
///
/// See [cuptiActivityFlush()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#_CPPv418cuptiActivityFlush9CUcontext8uint32_t8uint32_t)
///
/// # Safety
/// Context must exist if it is not null.
pub unsafe fn flush(context: sys::CUcontext, stream_id: u32, flag: u32) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityFlush(context, stream_id, flag) }.result()
}

/// Request to deliver activity records via the buffer completion callback.
///
/// See [cuptiActivityFlushAll()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityRegisterCallbacks#_CPPv421cuptiActivityFlushAll8uint32_t)
pub fn flush_all(flag: u32) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityFlushAll(flag) }.result()
}

/// Sets the flush period for the worker thread.
///
/// See [cuptiActivityFlushPeriod()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#_CPPv424cuptiActivityFlushPeriod8uint32_t)
pub fn flush_period(time: u32) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityFlushPeriod(time) }.result()
}

/// Read an activity API attribute.
///
/// See [cuptiActivityGetAttribute()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html#_CPPv425cuptiActivityGetAttribute23CUpti_ActivityAttributeP6size_tPv)
///
/// # Safety
/// Value size must exist.
pub unsafe fn get_attribute(
    attr: sys::CUpti_ActivityAttribute,
    value_size: *mut usize,
    value: *mut core::ffi::c_void,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityGetAttribute(attr, value_size, value) }.result()
}

/// Iterate over the activity records in a buffer.
///
/// See [cuptiActivityGetNextRecord()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityEnable#group__cupti__activity__api_1gab397f490a0df4a1633ea7b6e2420294f)
///
/// # Safety
/// Buffer must exist.
/// Record pointer must exist.
pub unsafe fn get_next_record(
    buffer: *mut u8,
    valid_buffer_size_bytes: usize,
    record: *mut *mut sys::CUpti_Activity,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityGetNextRecord(buffer, valid_buffer_size_bytes, record) }.result()
}

/// Get the number of activity records that were dropped of insufficient buffer space.
///
/// See [cuptiActivityGetNumDroppedRecords()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityGetAttribute#_CPPv433cuptiActivityGetNumDroppedRecords9CUcontext8uint32_tP6size_t)
///
/// # Safety
/// Context must exist if not null.
/// Dropped must exist.
pub unsafe fn get_num_dropped_records(
    context: sys::CUcontext,
    stream_id: u32,
    dropped: *mut usize,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityGetNumDroppedRecords(context, stream_id, dropped) }.result()
}

/// Pop an external correlation id for the calling thread.
///
/// See [cuptiActivityPopExternalCorrelationId()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityGetAttribute#_CPPv437cuptiActivityPopExternalCorrelationId29CUpti_ExternalCorrelationKindP8uint64_t)
///
/// # Safety
/// Last ID must exist.
pub unsafe fn pop_external_correlation_id(
    kind: sys::CUpti_ExternalCorrelationKind,
    last_id: *mut u64,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityPopExternalCorrelationId(kind, last_id) }.result()
}

/// Push an external correlation id for the calling thread.
///
/// See [cuptiActivityPopExternalCorrelationId()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityGetAttribute#_CPPv438cuptiActivityPushExternalCorrelationId29CUpti_ExternalCorrelationKind8uint64_t)
pub fn push_external_correlation_id(
    kind: sys::CUpti_ExternalCorrelationKind,
    id: u64,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityPushExternalCorrelationId(kind, id) }.result()
}

/// Registers callback functions with CUPTI for activity buffer handling.
///
/// See [cuptiActivityRegisterCallbacks()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityRegisterCallbacks#_CPPv430cuptiActivityRegisterCallbacks32CUpti_BuffersCallbackRequestFunc33CUpti_BuffersCallbackCompleteFunc)
pub fn register_callbacks(
    func_buffer_requested: sys::CUpti_BuffersCallbackRequestFunc,
    func_buffer_completed: sys::CUpti_BuffersCallbackCompleteFunc,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityRegisterCallbacks(func_buffer_requested, func_buffer_completed) }
        .result()
}

#[cfg(any(
    feature = "cuda-11070",
    feature = "cuda-11080",
    feature = "cuda-12000",
    feature = "cuda-12010",
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090",
    feature = "cuda-13000"
))]
/// Registers callback function with CUPTI for providing timestamp.
///
/// See [cuptiActivityRegisterTimestampCallback()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityGetAttribute#_CPPv438cuptiActivityRegisterTimestampCallback27CUpti_TimestampCallbackFunc)
pub fn register_timestamp_callback(
    func_timestamp: sys::CUpti_TimestampCallbackFunc,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivityRegisterTimestampCallback(func_timestamp) }.result()
}

/// Write an activity API attribute.
///
/// See [cuptiActivitySetAttribute()](https://docs.nvidia.com/cupti/api/group__CUPTI__ACTIVITY__API.html?highlight=cuptiActivityGetAttribute#_CPPv425cuptiActivitySetAttribute23CUpti_ActivityAttributeP6size_tPv)
///
/// # Safety
/// Value size must exist
pub unsafe fn set_attribute(
    attr: sys::CUpti_ActivityAttribute,
    value_size: *mut usize,
    value: *mut core::ffi::c_void,
) -> Result<(), CuptiError> {
    unsafe { sys::cuptiActivitySetAttribute(attr, value_size, value) }.result()
}
