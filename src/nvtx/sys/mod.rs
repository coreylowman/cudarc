#![cfg_attr(feature = "no-std", no_std)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;
pub type nvtxDomainCreateA_impl_fntype = ::core::option::Option<
    unsafe extern "C" fn(message: *const ::core::ffi::c_char) -> nvtxDomainHandle_t,
>;
pub type nvtxDomainCreateW_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(message: *const wchar_t) -> nvtxDomainHandle_t>;
pub type nvtxDomainDestroy_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t)>;
pub type nvtxDomainHandle_t = *mut nvtxDomainRegistration;
pub type nvtxDomainMarkEx_impl_fntype = ::core::option::Option<
    unsafe extern "C" fn(domain: nvtxDomainHandle_t, eventAttrib: *const nvtxEventAttributes_t),
>;
pub type nvtxDomainNameCategoryA_impl_fntype = ::core::option::Option<
    unsafe extern "C" fn(
        domain: nvtxDomainHandle_t,
        category: u32,
        name: *const ::core::ffi::c_char,
    ),
>;
pub type nvtxDomainNameCategoryW_impl_fntype = ::core::option::Option<
    unsafe extern "C" fn(domain: nvtxDomainHandle_t, category: u32, name: *const wchar_t),
>;
pub type nvtxDomainRangeEnd_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t, id: nvtxRangeId_t)>;
pub type nvtxDomainRangePop_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(domain: nvtxDomainHandle_t) -> ::core::ffi::c_int>;
pub type nvtxDomainRangePushEx_impl_fntype = ::core::option::Option<
    unsafe extern "C" fn(
        domain: nvtxDomainHandle_t,
        eventAttrib: *const nvtxEventAttributes_t,
    ) -> ::core::ffi::c_int,
>;
pub type nvtxDomainRangeStartEx_impl_fntype = ::core::option::Option<
    unsafe extern "C" fn(
        domain: nvtxDomainHandle_t,
        eventAttrib: *const nvtxEventAttributes_t,
    ) -> nvtxRangeId_t,
>;
pub type nvtxDomainRegisterStringA_impl_fntype = ::core::option::Option<
    unsafe extern "C" fn(
        domain: nvtxDomainHandle_t,
        string: *const ::core::ffi::c_char,
    ) -> nvtxStringHandle_t,
>;
pub type nvtxDomainRegisterStringW_impl_fntype = ::core::option::Option<
    unsafe extern "C" fn(domain: nvtxDomainHandle_t, string: *const wchar_t) -> nvtxStringHandle_t,
>;
pub type nvtxDomainRegistration = nvtxDomainRegistration_st;
pub type nvtxDomainResourceCreate_impl_fntype = ::core::option::Option<
    unsafe extern "C" fn(
        domain: nvtxDomainHandle_t,
        attribs: *mut nvtxResourceAttributes_t,
    ) -> nvtxResourceHandle_t,
>;
pub type nvtxDomainResourceDestroy_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(resource: nvtxResourceHandle_t)>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000"))]
pub type nvtxDomainSyncUserAcquireFailed_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(handle: nvtx_nvtxSyncUser_t)>;
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
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
))]
pub type nvtxDomainSyncUserAcquireFailed_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(handle: nvtxSyncUser_t)>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000"))]
pub type nvtxDomainSyncUserAcquireStart_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(handle: nvtx_nvtxSyncUser_t)>;
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
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
))]
pub type nvtxDomainSyncUserAcquireStart_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(handle: nvtxSyncUser_t)>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000"))]
pub type nvtxDomainSyncUserAcquireSuccess_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(handle: nvtx_nvtxSyncUser_t)>;
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
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
))]
pub type nvtxDomainSyncUserAcquireSuccess_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(handle: nvtxSyncUser_t)>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000"))]
pub type nvtxDomainSyncUserCreate_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(
        domain: nvtxDomainHandle_t,
        attribs: *const nvtx_nvtxSyncUserAttributes_t,
    ) -> nvtx_nvtxSyncUser_t,
>;
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
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
))]
pub type nvtxDomainSyncUserCreate_impl_fntype = ::core::option::Option<
    unsafe extern "C" fn(
        domain: nvtxDomainHandle_t,
        attribs: *const nvtxSyncUserAttributes_t,
    ) -> nvtxSyncUser_t,
>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000"))]
pub type nvtxDomainSyncUserDestroy_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(handle: nvtx_nvtxSyncUser_t)>;
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
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
))]
pub type nvtxDomainSyncUserDestroy_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(handle: nvtxSyncUser_t)>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000"))]
pub type nvtxDomainSyncUserReleasing_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(handle: nvtx_nvtxSyncUser_t)>;
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
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
))]
pub type nvtxDomainSyncUserReleasing_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(handle: nvtxSyncUser_t)>;
pub type nvtxEventAttributes_t = nvtxEventAttributes_v2;
pub type nvtxInitialize_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(reserved: *const ::core::ffi::c_void)>;
pub type nvtxMarkA_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(message: *const ::core::ffi::c_char)>;
pub type nvtxMarkEx_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(eventAttrib: *const nvtxEventAttributes_t)>;
pub type nvtxMarkW_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(message: *const wchar_t)>;
pub type nvtxNameCategoryA_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(category: u32, name: *const ::core::ffi::c_char)>;
pub type nvtxNameCategoryW_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(category: u32, name: *const wchar_t)>;
pub type nvtxNameClCommandQueueA_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(command_queue: nvtx_cl_command_queue, name: *const ::core::ffi::c_char),
>;
pub type nvtxNameClCommandQueueW_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(command_queue: nvtx_cl_command_queue, name: *const wchar_t),
>;
pub type nvtxNameClContextA_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(context: nvtx_cl_context, name: *const ::core::ffi::c_char),
>;
pub type nvtxNameClContextW_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(context: nvtx_cl_context, name: *const wchar_t)>;
pub type nvtxNameClDeviceA_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(device: nvtx_cl_device_id, name: *const ::core::ffi::c_char),
>;
pub type nvtxNameClDeviceW_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(device: nvtx_cl_device_id, name: *const wchar_t)>;
pub type nvtxNameClEventA_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(evnt: nvtx_cl_event, name: *const ::core::ffi::c_char),
>;
pub type nvtxNameClEventW_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(evnt: nvtx_cl_event, name: *const wchar_t)>;
pub type nvtxNameClMemObjectA_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(memobj: nvtx_cl_mem, name: *const ::core::ffi::c_char),
>;
pub type nvtxNameClMemObjectW_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(memobj: nvtx_cl_mem, name: *const wchar_t)>;
pub type nvtxNameClProgramA_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(program: nvtx_cl_program, name: *const ::core::ffi::c_char),
>;
pub type nvtxNameClProgramW_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(program: nvtx_cl_program, name: *const wchar_t)>;
pub type nvtxNameClSamplerA_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(sampler: nvtx_cl_sampler, name: *const ::core::ffi::c_char),
>;
pub type nvtxNameClSamplerW_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(sampler: nvtx_cl_sampler, name: *const wchar_t)>;
pub type nvtxNameCuContextA_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(context: nvtx_CUcontext, name: *const ::core::ffi::c_char),
>;
pub type nvtxNameCuContextW_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(context: nvtx_CUcontext, name: *const wchar_t)>;
pub type nvtxNameCuDeviceA_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(device: nvtx_CUdevice, name: *const ::core::ffi::c_char),
>;
pub type nvtxNameCuDeviceW_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(device: nvtx_CUdevice, name: *const wchar_t)>;
pub type nvtxNameCuEventA_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(event: nvtx_CUevent, name: *const ::core::ffi::c_char),
>;
pub type nvtxNameCuEventW_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(event: nvtx_CUevent, name: *const wchar_t)>;
pub type nvtxNameCuStreamA_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(stream: nvtx_CUstream, name: *const ::core::ffi::c_char),
>;
pub type nvtxNameCuStreamW_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(stream: nvtx_CUstream, name: *const wchar_t)>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000"))]
pub type nvtxNameCudaDeviceA_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(device: ::core::ffi::c_int, name: *const ::core::ffi::c_char),
>;
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
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
))]
pub type nvtxNameCudaDeviceA_impl_fntype = ::core::option::Option<
    unsafe extern "C" fn(device: ::core::ffi::c_int, name: *const ::core::ffi::c_char),
>;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000"))]
pub type nvtxNameCudaDeviceW_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(device: ::core::ffi::c_int, name: *const wchar_t)>;
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
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
))]
pub type nvtxNameCudaDeviceW_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(device: ::core::ffi::c_int, name: *const wchar_t)>;
pub type nvtxNameCudaEventA_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(event: nvtx_cudaEvent_t, name: *const ::core::ffi::c_char),
>;
pub type nvtxNameCudaEventW_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(event: nvtx_cudaEvent_t, name: *const wchar_t)>;
pub type nvtxNameCudaStreamA_fakeimpl_fntype = ::core::option::Option<
    unsafe extern "C" fn(stream: nvtx_cudaStream_t, name: *const ::core::ffi::c_char),
>;
pub type nvtxNameCudaStreamW_fakeimpl_fntype =
    ::core::option::Option<unsafe extern "C" fn(stream: nvtx_cudaStream_t, name: *const wchar_t)>;
pub type nvtxNameOsThreadA_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(threadId: u32, name: *const ::core::ffi::c_char)>;
pub type nvtxNameOsThreadW_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(threadId: u32, name: *const wchar_t)>;
pub type nvtxRangeEnd_impl_fntype = ::core::option::Option<unsafe extern "C" fn(id: nvtxRangeId_t)>;
pub type nvtxRangeId_t = u64;
pub type nvtxRangePop_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn() -> ::core::ffi::c_int>;
pub type nvtxRangePushA_impl_fntype = ::core::option::Option<
    unsafe extern "C" fn(message: *const ::core::ffi::c_char) -> ::core::ffi::c_int,
>;
pub type nvtxRangePushEx_impl_fntype = ::core::option::Option<
    unsafe extern "C" fn(eventAttrib: *const nvtxEventAttributes_t) -> ::core::ffi::c_int,
>;
pub type nvtxRangePushW_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(message: *const wchar_t) -> ::core::ffi::c_int>;
pub type nvtxRangeStartA_impl_fntype = ::core::option::Option<
    unsafe extern "C" fn(message: *const ::core::ffi::c_char) -> nvtxRangeId_t,
>;
pub type nvtxRangeStartEx_impl_fntype = ::core::option::Option<
    unsafe extern "C" fn(eventAttrib: *const nvtxEventAttributes_t) -> nvtxRangeId_t,
>;
pub type nvtxRangeStartW_impl_fntype =
    ::core::option::Option<unsafe extern "C" fn(message: *const wchar_t) -> nvtxRangeId_t>;
pub type nvtxResourceAttributes_t = nvtxResourceAttributes_v0;
pub type nvtxResourceHandle_t = *mut nvtxResourceHandle;
pub type nvtxStringHandle_t = *mut nvtxStringRegistration;
pub type nvtxStringRegistration = nvtxStringRegistration_st;
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
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
))]
pub type nvtxSyncUserAttributes_t = nvtxSyncUserAttributes_v0;
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
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
))]
pub type nvtxSyncUser_t = *mut nvtxSyncUser;
pub type nvtx_CUcontext = *mut ::core::ffi::c_void;
pub type nvtx_CUdevice = ::core::ffi::c_int;
pub type nvtx_CUevent = *mut ::core::ffi::c_void;
pub type nvtx_CUstream = *mut ::core::ffi::c_void;
pub type nvtx_cl_command_queue = *mut ::core::ffi::c_void;
pub type nvtx_cl_context = *mut ::core::ffi::c_void;
pub type nvtx_cl_device_id = *mut ::core::ffi::c_void;
pub type nvtx_cl_event = *mut ::core::ffi::c_void;
pub type nvtx_cl_kernel = *mut ::core::ffi::c_void;
pub type nvtx_cl_mem = *mut ::core::ffi::c_void;
pub type nvtx_cl_platform_id = *mut ::core::ffi::c_void;
pub type nvtx_cl_program = *mut ::core::ffi::c_void;
pub type nvtx_cl_sampler = *mut ::core::ffi::c_void;
pub type nvtx_cudaEvent_t = *mut ::core::ffi::c_void;
pub type nvtx_cudaStream_t = *mut ::core::ffi::c_void;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000"))]
pub type nvtx_nvtxSyncUserAttributes_t = ::core::ffi::c_void;
#[cfg(any(feature = "cuda-12090", feature = "cuda-13000"))]
pub type nvtx_nvtxSyncUser_t = *mut ::core::ffi::c_void;
pub type wchar_t = ::core::ffi::c_int;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvtxColorType_t {
    NVTX_COLOR_UNKNOWN = 0,
    NVTX_COLOR_ARGB = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvtxMessageType_t {
    NVTX_MESSAGE_UNKNOWN = 0,
    NVTX_MESSAGE_TYPE_ASCII = 1,
    NVTX_MESSAGE_TYPE_UNICODE = 2,
    NVTX_MESSAGE_TYPE_REGISTERED = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvtxPayloadType_t {
    NVTX_PAYLOAD_UNKNOWN = 0,
    NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 = 1,
    NVTX_PAYLOAD_TYPE_INT64 = 2,
    NVTX_PAYLOAD_TYPE_DOUBLE = 3,
    NVTX_PAYLOAD_TYPE_UNSIGNED_INT32 = 4,
    NVTX_PAYLOAD_TYPE_INT32 = 5,
    NVTX_PAYLOAD_TYPE_FLOAT = 6,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum nvtxResourceGenericType_t {
    NVTX_RESOURCE_TYPE_UNKNOWN = 0,
    NVTX_RESOURCE_TYPE_GENERIC_POINTER = 65537,
    NVTX_RESOURCE_TYPE_GENERIC_HANDLE = 65538,
    NVTX_RESOURCE_TYPE_GENERIC_THREAD_NATIVE = 65539,
    NVTX_RESOURCE_TYPE_GENERIC_THREAD_POSIX = 65540,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvtxDomainRegistration_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct nvtxEventAttributes_v2 {
    pub version: u16,
    pub size: u16,
    pub category: u32,
    pub colorType: i32,
    pub color: u32,
    pub payloadType: i32,
    pub reserved0: i32,
    pub payload: nvtxEventAttributes_v2_payload_t,
    pub messageType: i32,
    pub message: nvtxMessageValue_t,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct nvtxResourceAttributes_v0 {
    pub version: u16,
    pub size: u16,
    pub identifierType: i32,
    pub identifier: nvtxResourceAttributes_v0_identifier_t,
    pub messageType: i32,
    pub message: nvtxMessageValue_t,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvtxResourceHandle {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvtxStringRegistration_st {
    _unused: [u8; 0],
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
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
))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvtxSyncUser {
    _unused: [u8; 0],
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
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
))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct nvtxSyncUserAttributes_v0 {
    _unused: [u8; 0],
}
impl Default for nvtxEventAttributes_v2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for nvtxEventAttributes_v2_payload_t {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for nvtxMessageValue_t {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for nvtxResourceAttributes_v0 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for nvtxResourceAttributes_v0_identifier_t {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union nvtxEventAttributes_v2_payload_t {
    pub ullValue: u64,
    pub llValue: i64,
    pub dValue: f64,
    pub uiValue: u32,
    pub iValue: i32,
    pub fValue: f32,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union nvtxMessageValue_t {
    pub ascii: *const ::core::ffi::c_char,
    pub unicode: *const wchar_t,
    pub registered: nvtxStringHandle_t,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union nvtxResourceAttributes_v0_identifier_t {
    pub pValue: *const ::core::ffi::c_void,
    pub ullValue: u64,
}
#[cfg(not(feature = "dynamic-loading"))]
extern "C" {
    pub fn nvtxDomainCreateA(name: *const ::core::ffi::c_char) -> nvtxDomainHandle_t;
    pub fn nvtxDomainCreateW(name: *const wchar_t) -> nvtxDomainHandle_t;
    pub fn nvtxDomainDestroy(domain: nvtxDomainHandle_t);
    pub fn nvtxDomainMarkEx(domain: nvtxDomainHandle_t, eventAttrib: *const nvtxEventAttributes_t);
    pub fn nvtxDomainNameCategoryA(
        domain: nvtxDomainHandle_t,
        category: u32,
        name: *const ::core::ffi::c_char,
    );
    pub fn nvtxDomainNameCategoryW(domain: nvtxDomainHandle_t, category: u32, name: *const wchar_t);
    pub fn nvtxDomainRangeEnd(domain: nvtxDomainHandle_t, id: nvtxRangeId_t);
    pub fn nvtxDomainRangePop(domain: nvtxDomainHandle_t) -> ::core::ffi::c_int;
    pub fn nvtxDomainRangePushEx(
        domain: nvtxDomainHandle_t,
        eventAttrib: *const nvtxEventAttributes_t,
    ) -> ::core::ffi::c_int;
    pub fn nvtxDomainRangeStartEx(
        domain: nvtxDomainHandle_t,
        eventAttrib: *const nvtxEventAttributes_t,
    ) -> nvtxRangeId_t;
    pub fn nvtxDomainRegisterStringA(
        domain: nvtxDomainHandle_t,
        string: *const ::core::ffi::c_char,
    ) -> nvtxStringHandle_t;
    pub fn nvtxDomainRegisterStringW(
        domain: nvtxDomainHandle_t,
        string: *const wchar_t,
    ) -> nvtxStringHandle_t;
    pub fn nvtxDomainResourceCreate(
        domain: nvtxDomainHandle_t,
        attribs: *mut nvtxResourceAttributes_t,
    ) -> nvtxResourceHandle_t;
    pub fn nvtxDomainResourceDestroy(resource: nvtxResourceHandle_t);
    pub fn nvtxMarkA(message: *const ::core::ffi::c_char);
    pub fn nvtxMarkEx(eventAttrib: *const nvtxEventAttributes_t);
    pub fn nvtxMarkW(message: *const wchar_t);
    pub fn nvtxNameCategoryA(category: u32, name: *const ::core::ffi::c_char);
    pub fn nvtxNameCategoryW(category: u32, name: *const wchar_t);
    pub fn nvtxNameOsThreadA(threadId: u32, name: *const ::core::ffi::c_char);
    pub fn nvtxNameOsThreadW(threadId: u32, name: *const wchar_t);
    pub fn nvtxRangeEnd(id: nvtxRangeId_t);
    pub fn nvtxRangePop() -> ::core::ffi::c_int;
    pub fn nvtxRangePushA(message: *const ::core::ffi::c_char) -> ::core::ffi::c_int;
    pub fn nvtxRangePushEx(eventAttrib: *const nvtxEventAttributes_t) -> ::core::ffi::c_int;
    pub fn nvtxRangePushW(message: *const wchar_t) -> ::core::ffi::c_int;
    pub fn nvtxRangeStartA(message: *const ::core::ffi::c_char) -> nvtxRangeId_t;
    pub fn nvtxRangeStartEx(eventAttrib: *const nvtxEventAttributes_t) -> nvtxRangeId_t;
    pub fn nvtxRangeStartW(message: *const wchar_t) -> nvtxRangeId_t;
}
#[cfg(feature = "dynamic-loading")]
mod loaded {
    use super::*;
    pub unsafe fn nvtxDomainCreateA(name: *const ::core::ffi::c_char) -> nvtxDomainHandle_t {
        (culib().nvtxDomainCreateA)(name)
    }
    pub unsafe fn nvtxDomainCreateW(name: *const wchar_t) -> nvtxDomainHandle_t {
        (culib().nvtxDomainCreateW)(name)
    }
    pub unsafe fn nvtxDomainDestroy(domain: nvtxDomainHandle_t) {
        (culib().nvtxDomainDestroy)(domain)
    }
    pub unsafe fn nvtxDomainMarkEx(
        domain: nvtxDomainHandle_t,
        eventAttrib: *const nvtxEventAttributes_t,
    ) {
        (culib().nvtxDomainMarkEx)(domain, eventAttrib)
    }
    pub unsafe fn nvtxDomainNameCategoryA(
        domain: nvtxDomainHandle_t,
        category: u32,
        name: *const ::core::ffi::c_char,
    ) {
        (culib().nvtxDomainNameCategoryA)(domain, category, name)
    }
    pub unsafe fn nvtxDomainNameCategoryW(
        domain: nvtxDomainHandle_t,
        category: u32,
        name: *const wchar_t,
    ) {
        (culib().nvtxDomainNameCategoryW)(domain, category, name)
    }
    pub unsafe fn nvtxDomainRangeEnd(domain: nvtxDomainHandle_t, id: nvtxRangeId_t) {
        (culib().nvtxDomainRangeEnd)(domain, id)
    }
    pub unsafe fn nvtxDomainRangePop(domain: nvtxDomainHandle_t) -> ::core::ffi::c_int {
        (culib().nvtxDomainRangePop)(domain)
    }
    pub unsafe fn nvtxDomainRangePushEx(
        domain: nvtxDomainHandle_t,
        eventAttrib: *const nvtxEventAttributes_t,
    ) -> ::core::ffi::c_int {
        (culib().nvtxDomainRangePushEx)(domain, eventAttrib)
    }
    pub unsafe fn nvtxDomainRangeStartEx(
        domain: nvtxDomainHandle_t,
        eventAttrib: *const nvtxEventAttributes_t,
    ) -> nvtxRangeId_t {
        (culib().nvtxDomainRangeStartEx)(domain, eventAttrib)
    }
    pub unsafe fn nvtxDomainRegisterStringA(
        domain: nvtxDomainHandle_t,
        string: *const ::core::ffi::c_char,
    ) -> nvtxStringHandle_t {
        (culib().nvtxDomainRegisterStringA)(domain, string)
    }
    pub unsafe fn nvtxDomainRegisterStringW(
        domain: nvtxDomainHandle_t,
        string: *const wchar_t,
    ) -> nvtxStringHandle_t {
        (culib().nvtxDomainRegisterStringW)(domain, string)
    }
    pub unsafe fn nvtxDomainResourceCreate(
        domain: nvtxDomainHandle_t,
        attribs: *mut nvtxResourceAttributes_t,
    ) -> nvtxResourceHandle_t {
        (culib().nvtxDomainResourceCreate)(domain, attribs)
    }
    pub unsafe fn nvtxDomainResourceDestroy(resource: nvtxResourceHandle_t) {
        (culib().nvtxDomainResourceDestroy)(resource)
    }
    pub unsafe fn nvtxMarkA(message: *const ::core::ffi::c_char) {
        (culib().nvtxMarkA)(message)
    }
    pub unsafe fn nvtxMarkEx(eventAttrib: *const nvtxEventAttributes_t) {
        (culib().nvtxMarkEx)(eventAttrib)
    }
    pub unsafe fn nvtxMarkW(message: *const wchar_t) {
        (culib().nvtxMarkW)(message)
    }
    pub unsafe fn nvtxNameCategoryA(category: u32, name: *const ::core::ffi::c_char) {
        (culib().nvtxNameCategoryA)(category, name)
    }
    pub unsafe fn nvtxNameCategoryW(category: u32, name: *const wchar_t) {
        (culib().nvtxNameCategoryW)(category, name)
    }
    pub unsafe fn nvtxNameOsThreadA(threadId: u32, name: *const ::core::ffi::c_char) {
        (culib().nvtxNameOsThreadA)(threadId, name)
    }
    pub unsafe fn nvtxNameOsThreadW(threadId: u32, name: *const wchar_t) {
        (culib().nvtxNameOsThreadW)(threadId, name)
    }
    pub unsafe fn nvtxRangeEnd(id: nvtxRangeId_t) {
        (culib().nvtxRangeEnd)(id)
    }
    pub unsafe fn nvtxRangePop() -> ::core::ffi::c_int {
        (culib().nvtxRangePop)()
    }
    pub unsafe fn nvtxRangePushA(message: *const ::core::ffi::c_char) -> ::core::ffi::c_int {
        (culib().nvtxRangePushA)(message)
    }
    pub unsafe fn nvtxRangePushEx(eventAttrib: *const nvtxEventAttributes_t) -> ::core::ffi::c_int {
        (culib().nvtxRangePushEx)(eventAttrib)
    }
    pub unsafe fn nvtxRangePushW(message: *const wchar_t) -> ::core::ffi::c_int {
        (culib().nvtxRangePushW)(message)
    }
    pub unsafe fn nvtxRangeStartA(message: *const ::core::ffi::c_char) -> nvtxRangeId_t {
        (culib().nvtxRangeStartA)(message)
    }
    pub unsafe fn nvtxRangeStartEx(eventAttrib: *const nvtxEventAttributes_t) -> nvtxRangeId_t {
        (culib().nvtxRangeStartEx)(eventAttrib)
    }
    pub unsafe fn nvtxRangeStartW(message: *const wchar_t) -> nvtxRangeId_t {
        (culib().nvtxRangeStartW)(message)
    }
    pub struct Lib {
        __library: ::libloading::Library,
        pub nvtxDomainCreateA:
            unsafe extern "C" fn(name: *const ::core::ffi::c_char) -> nvtxDomainHandle_t,
        pub nvtxDomainCreateW: unsafe extern "C" fn(name: *const wchar_t) -> nvtxDomainHandle_t,
        pub nvtxDomainDestroy: unsafe extern "C" fn(domain: nvtxDomainHandle_t),
        pub nvtxDomainMarkEx: unsafe extern "C" fn(
            domain: nvtxDomainHandle_t,
            eventAttrib: *const nvtxEventAttributes_t,
        ),
        pub nvtxDomainNameCategoryA: unsafe extern "C" fn(
            domain: nvtxDomainHandle_t,
            category: u32,
            name: *const ::core::ffi::c_char,
        ),
        pub nvtxDomainNameCategoryW:
            unsafe extern "C" fn(domain: nvtxDomainHandle_t, category: u32, name: *const wchar_t),
        pub nvtxDomainRangeEnd: unsafe extern "C" fn(domain: nvtxDomainHandle_t, id: nvtxRangeId_t),
        pub nvtxDomainRangePop:
            unsafe extern "C" fn(domain: nvtxDomainHandle_t) -> ::core::ffi::c_int,
        pub nvtxDomainRangePushEx: unsafe extern "C" fn(
            domain: nvtxDomainHandle_t,
            eventAttrib: *const nvtxEventAttributes_t,
        ) -> ::core::ffi::c_int,
        pub nvtxDomainRangeStartEx: unsafe extern "C" fn(
            domain: nvtxDomainHandle_t,
            eventAttrib: *const nvtxEventAttributes_t,
        ) -> nvtxRangeId_t,
        pub nvtxDomainRegisterStringA: unsafe extern "C" fn(
            domain: nvtxDomainHandle_t,
            string: *const ::core::ffi::c_char,
        ) -> nvtxStringHandle_t,
        pub nvtxDomainRegisterStringW: unsafe extern "C" fn(
            domain: nvtxDomainHandle_t,
            string: *const wchar_t,
        ) -> nvtxStringHandle_t,
        pub nvtxDomainResourceCreate: unsafe extern "C" fn(
            domain: nvtxDomainHandle_t,
            attribs: *mut nvtxResourceAttributes_t,
        ) -> nvtxResourceHandle_t,
        pub nvtxDomainResourceDestroy: unsafe extern "C" fn(resource: nvtxResourceHandle_t),
        pub nvtxMarkA: unsafe extern "C" fn(message: *const ::core::ffi::c_char),
        pub nvtxMarkEx: unsafe extern "C" fn(eventAttrib: *const nvtxEventAttributes_t),
        pub nvtxMarkW: unsafe extern "C" fn(message: *const wchar_t),
        pub nvtxNameCategoryA:
            unsafe extern "C" fn(category: u32, name: *const ::core::ffi::c_char),
        pub nvtxNameCategoryW: unsafe extern "C" fn(category: u32, name: *const wchar_t),
        pub nvtxNameOsThreadA:
            unsafe extern "C" fn(threadId: u32, name: *const ::core::ffi::c_char),
        pub nvtxNameOsThreadW: unsafe extern "C" fn(threadId: u32, name: *const wchar_t),
        pub nvtxRangeEnd: unsafe extern "C" fn(id: nvtxRangeId_t),
        pub nvtxRangePop: unsafe extern "C" fn() -> ::core::ffi::c_int,
        pub nvtxRangePushA:
            unsafe extern "C" fn(message: *const ::core::ffi::c_char) -> ::core::ffi::c_int,
        pub nvtxRangePushEx:
            unsafe extern "C" fn(eventAttrib: *const nvtxEventAttributes_t) -> ::core::ffi::c_int,
        pub nvtxRangePushW: unsafe extern "C" fn(message: *const wchar_t) -> ::core::ffi::c_int,
        pub nvtxRangeStartA:
            unsafe extern "C" fn(message: *const ::core::ffi::c_char) -> nvtxRangeId_t,
        pub nvtxRangeStartEx:
            unsafe extern "C" fn(eventAttrib: *const nvtxEventAttributes_t) -> nvtxRangeId_t,
        pub nvtxRangeStartW: unsafe extern "C" fn(message: *const wchar_t) -> nvtxRangeId_t,
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
            let nvtxDomainCreateA = __library
                .get(b"nvtxDomainCreateA\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxDomainCreateW = __library
                .get(b"nvtxDomainCreateW\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxDomainDestroy = __library
                .get(b"nvtxDomainDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxDomainMarkEx = __library
                .get(b"nvtxDomainMarkEx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxDomainNameCategoryA = __library
                .get(b"nvtxDomainNameCategoryA\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxDomainNameCategoryW = __library
                .get(b"nvtxDomainNameCategoryW\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxDomainRangeEnd = __library
                .get(b"nvtxDomainRangeEnd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxDomainRangePop = __library
                .get(b"nvtxDomainRangePop\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxDomainRangePushEx = __library
                .get(b"nvtxDomainRangePushEx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxDomainRangeStartEx = __library
                .get(b"nvtxDomainRangeStartEx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxDomainRegisterStringA = __library
                .get(b"nvtxDomainRegisterStringA\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxDomainRegisterStringW = __library
                .get(b"nvtxDomainRegisterStringW\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxDomainResourceCreate = __library
                .get(b"nvtxDomainResourceCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxDomainResourceDestroy = __library
                .get(b"nvtxDomainResourceDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxMarkA = __library
                .get(b"nvtxMarkA\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxMarkEx = __library
                .get(b"nvtxMarkEx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxMarkW = __library
                .get(b"nvtxMarkW\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxNameCategoryA = __library
                .get(b"nvtxNameCategoryA\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxNameCategoryW = __library
                .get(b"nvtxNameCategoryW\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxNameOsThreadA = __library
                .get(b"nvtxNameOsThreadA\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxNameOsThreadW = __library
                .get(b"nvtxNameOsThreadW\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxRangeEnd = __library
                .get(b"nvtxRangeEnd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxRangePop = __library
                .get(b"nvtxRangePop\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxRangePushA = __library
                .get(b"nvtxRangePushA\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxRangePushEx = __library
                .get(b"nvtxRangePushEx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxRangePushW = __library
                .get(b"nvtxRangePushW\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxRangeStartA = __library
                .get(b"nvtxRangeStartA\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxRangeStartEx = __library
                .get(b"nvtxRangeStartEx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let nvtxRangeStartW = __library
                .get(b"nvtxRangeStartW\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            Ok(Self {
                __library,
                nvtxDomainCreateA,
                nvtxDomainCreateW,
                nvtxDomainDestroy,
                nvtxDomainMarkEx,
                nvtxDomainNameCategoryA,
                nvtxDomainNameCategoryW,
                nvtxDomainRangeEnd,
                nvtxDomainRangePop,
                nvtxDomainRangePushEx,
                nvtxDomainRangeStartEx,
                nvtxDomainRegisterStringA,
                nvtxDomainRegisterStringW,
                nvtxDomainResourceCreate,
                nvtxDomainResourceDestroy,
                nvtxMarkA,
                nvtxMarkEx,
                nvtxMarkW,
                nvtxNameCategoryA,
                nvtxNameCategoryW,
                nvtxNameOsThreadA,
                nvtxNameOsThreadW,
                nvtxRangeEnd,
                nvtxRangePop,
                nvtxRangePushA,
                nvtxRangePushEx,
                nvtxRangePushW,
                nvtxRangeStartA,
                nvtxRangeStartEx,
                nvtxRangeStartW,
            })
        }
    }
    pub unsafe fn is_culib_present() -> bool {
        let lib_names = ["nvToolsExt"];
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
            let lib_names = std::vec!["nvToolsExt"];
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
