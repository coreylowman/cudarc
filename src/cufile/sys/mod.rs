#![cfg_attr(feature = "no-std", no_std)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;
pub use self::CUFILEStatus_enum as CUfileStatus_t;
pub use self::CUfileDriverControlFlags as CUfileDriverControlFlags_t;
pub use self::CUfileDriverStatusFlags as CUfileDriverStatusFlags_t;
pub use self::CUfileFeatureFlags as CUfileFeatureFlags_t;
pub use self::CUfileOpcode as CUfileOpcode_t;
pub use self::cudaError_enum as CUresult;
pub use self::cufileBatchMode as CUfileBatchMode_t;
pub type CUfileBatchHandle_t = *mut ::core::ffi::c_void;
pub type CUfileDrvProps_t = CUfileDrvProps;
pub type CUfileError_t = CUfileError;
pub type CUfileFSOps_t = CUfileFSOps;
pub type CUfileHandle_t = *mut ::core::ffi::c_void;
pub type CUfileIOEvents_t = CUfileIOEvents;
pub type CUfileIOParams_t = CUfileIOParams;
pub type CUstream = *mut CUstream_st;
pub type __loff_t = __off64_t;
pub type __off64_t = ::core::ffi::c_long;
pub type __off_t = ::core::ffi::c_long;
pub type __syscall_slong_t = ::core::ffi::c_long;
pub type __time_t = ::core::ffi::c_long;
pub type cufileRDMAInfo_t = cufileRDMAInfo;
pub type loff_t = __loff_t;
pub type off_t = __off_t;
pub type sa_family_t = ::core::ffi::c_ushort;
pub type sockaddr_t = sockaddr;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUFILEStatus_enum {
    CUFILE_WAITING = 1,
    CUFILE_PENDING = 2,
    CUFILE_INVALID = 4,
    CUFILE_CANCELED = 8,
    CUFILE_COMPLETE = 16,
    CUFILE_TIMEOUT = 32,
    CUFILE_FAILED = 64,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfileDriverControlFlags {
    CU_FILE_USE_POLL_MODE = 0,
    CU_FILE_ALLOW_COMPAT_MODE = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfileDriverStatusFlags {
    CU_FILE_LUSTRE_SUPPORTED = 0,
    CU_FILE_WEKAFS_SUPPORTED = 1,
    CU_FILE_NFS_SUPPORTED = 2,
    CU_FILE_GPFS_SUPPORTED = 3,
    CU_FILE_NVME_SUPPORTED = 4,
    CU_FILE_NVMEOF_SUPPORTED = 5,
    CU_FILE_SCSI_SUPPORTED = 6,
    CU_FILE_SCALEFLUX_CSD_SUPPORTED = 7,
    CU_FILE_NVMESH_SUPPORTED = 8,
    CU_FILE_BEEGFS_SUPPORTED = 9,
    CU_FILE_NVME_P2P_SUPPORTED = 11,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfileFeatureFlags {
    CU_FILE_DYN_ROUTING_SUPPORTED = 0,
    CU_FILE_BATCH_IO_SUPPORTED = 1,
    CU_FILE_STREAMS_SUPPORTED = 2,
    CU_FILE_PARALLEL_IO_SUPPORTED = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfileFileHandleType {
    CU_FILE_HANDLE_TYPE_OPAQUE_FD = 1,
    CU_FILE_HANDLE_TYPE_OPAQUE_WIN32 = 2,
    CU_FILE_HANDLE_TYPE_USERSPACE_FS = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfileOpError {
    CU_FILE_SUCCESS = 0,
    CU_FILE_DRIVER_NOT_INITIALIZED = 5001,
    CU_FILE_DRIVER_INVALID_PROPS = 5002,
    CU_FILE_DRIVER_UNSUPPORTED_LIMIT = 5003,
    CU_FILE_DRIVER_VERSION_MISMATCH = 5004,
    CU_FILE_DRIVER_VERSION_READ_ERROR = 5005,
    CU_FILE_DRIVER_CLOSING = 5006,
    CU_FILE_PLATFORM_NOT_SUPPORTED = 5007,
    CU_FILE_IO_NOT_SUPPORTED = 5008,
    CU_FILE_DEVICE_NOT_SUPPORTED = 5009,
    CU_FILE_NVFS_DRIVER_ERROR = 5010,
    CU_FILE_CUDA_DRIVER_ERROR = 5011,
    CU_FILE_CUDA_POINTER_INVALID = 5012,
    CU_FILE_CUDA_MEMORY_TYPE_INVALID = 5013,
    CU_FILE_CUDA_POINTER_RANGE_ERROR = 5014,
    CU_FILE_CUDA_CONTEXT_MISMATCH = 5015,
    CU_FILE_INVALID_MAPPING_SIZE = 5016,
    CU_FILE_INVALID_MAPPING_RANGE = 5017,
    CU_FILE_INVALID_FILE_TYPE = 5018,
    CU_FILE_INVALID_FILE_OPEN_FLAG = 5019,
    CU_FILE_DIO_NOT_SET = 5020,
    CU_FILE_INVALID_VALUE = 5022,
    CU_FILE_MEMORY_ALREADY_REGISTERED = 5023,
    CU_FILE_MEMORY_NOT_REGISTERED = 5024,
    CU_FILE_PERMISSION_DENIED = 5025,
    CU_FILE_DRIVER_ALREADY_OPEN = 5026,
    CU_FILE_HANDLE_NOT_REGISTERED = 5027,
    CU_FILE_HANDLE_ALREADY_REGISTERED = 5028,
    CU_FILE_DEVICE_NOT_FOUND = 5029,
    CU_FILE_INTERNAL_ERROR = 5030,
    CU_FILE_GETNEWFD_FAILED = 5031,
    CU_FILE_NVFS_SETUP_ERROR = 5033,
    CU_FILE_IO_DISABLED = 5034,
    CU_FILE_BATCH_SUBMIT_FAILED = 5035,
    CU_FILE_GPU_MEMORY_PINNING_FAILED = 5036,
    CU_FILE_BATCH_FULL = 5037,
    CU_FILE_ASYNC_NOT_SUPPORTED = 5038,
    CU_FILE_IO_MAX_ERROR = 5039,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfileOpcode {
    CUFILE_READ = 0,
    CUFILE_WRITE = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError_enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_DEINITIALIZED = 4,
    CUDA_ERROR_PROFILER_DISABLED = 5,
    CUDA_ERROR_PROFILER_NOT_INITIALIZED = 6,
    CUDA_ERROR_PROFILER_ALREADY_STARTED = 7,
    CUDA_ERROR_PROFILER_ALREADY_STOPPED = 8,
    CUDA_ERROR_STUB_LIBRARY = 34,
    CUDA_ERROR_DEVICE_UNAVAILABLE = 46,
    CUDA_ERROR_NO_DEVICE = 100,
    CUDA_ERROR_INVALID_DEVICE = 101,
    CUDA_ERROR_DEVICE_NOT_LICENSED = 102,
    CUDA_ERROR_INVALID_IMAGE = 200,
    CUDA_ERROR_INVALID_CONTEXT = 201,
    CUDA_ERROR_CONTEXT_ALREADY_CURRENT = 202,
    CUDA_ERROR_MAP_FAILED = 205,
    CUDA_ERROR_UNMAP_FAILED = 206,
    CUDA_ERROR_ARRAY_IS_MAPPED = 207,
    CUDA_ERROR_ALREADY_MAPPED = 208,
    CUDA_ERROR_NO_BINARY_FOR_GPU = 209,
    CUDA_ERROR_ALREADY_ACQUIRED = 210,
    CUDA_ERROR_NOT_MAPPED = 211,
    CUDA_ERROR_NOT_MAPPED_AS_ARRAY = 212,
    CUDA_ERROR_NOT_MAPPED_AS_POINTER = 213,
    CUDA_ERROR_ECC_UNCORRECTABLE = 214,
    CUDA_ERROR_UNSUPPORTED_LIMIT = 215,
    CUDA_ERROR_CONTEXT_ALREADY_IN_USE = 216,
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217,
    CUDA_ERROR_INVALID_PTX = 218,
    CUDA_ERROR_INVALID_GRAPHICS_CONTEXT = 219,
    CUDA_ERROR_NVLINK_UNCORRECTABLE = 220,
    CUDA_ERROR_JIT_COMPILER_NOT_FOUND = 221,
    CUDA_ERROR_UNSUPPORTED_PTX_VERSION = 222,
    CUDA_ERROR_JIT_COMPILATION_DISABLED = 223,
    CUDA_ERROR_UNSUPPORTED_EXEC_AFFINITY = 224,
    CUDA_ERROR_UNSUPPORTED_DEVSIDE_SYNC = 225,
    CUDA_ERROR_CONTAINED = 226,
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
    CUDA_ERROR_LOSSY_QUERY = 402,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_ILLEGAL_ADDRESS = 700,
    CUDA_ERROR_LAUNCH_OUT_OF_RESOURCES = 701,
    CUDA_ERROR_LAUNCH_TIMEOUT = 702,
    CUDA_ERROR_LAUNCH_INCOMPATIBLE_TEXTURING = 703,
    CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705,
    CUDA_ERROR_PRIMARY_CONTEXT_ACTIVE = 708,
    CUDA_ERROR_CONTEXT_IS_DESTROYED = 709,
    CUDA_ERROR_ASSERT = 710,
    CUDA_ERROR_TOO_MANY_PEERS = 711,
    CUDA_ERROR_HOST_MEMORY_ALREADY_REGISTERED = 712,
    CUDA_ERROR_HOST_MEMORY_NOT_REGISTERED = 713,
    CUDA_ERROR_HARDWARE_STACK_ERROR = 714,
    CUDA_ERROR_ILLEGAL_INSTRUCTION = 715,
    CUDA_ERROR_MISALIGNED_ADDRESS = 716,
    CUDA_ERROR_INVALID_ADDRESS_SPACE = 717,
    CUDA_ERROR_INVALID_PC = 718,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE = 720,
    CUDA_ERROR_TENSOR_MEMORY_LEAK = 721,
    CUDA_ERROR_NOT_PERMITTED = 800,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_SYSTEM_NOT_READY = 802,
    CUDA_ERROR_SYSTEM_DRIVER_MISMATCH = 803,
    CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE = 804,
    CUDA_ERROR_MPS_CONNECTION_FAILED = 805,
    CUDA_ERROR_MPS_RPC_FAILURE = 806,
    CUDA_ERROR_MPS_SERVER_NOT_READY = 807,
    CUDA_ERROR_MPS_MAX_CLIENTS_REACHED = 808,
    CUDA_ERROR_MPS_MAX_CONNECTIONS_REACHED = 809,
    CUDA_ERROR_MPS_CLIENT_TERMINATED = 810,
    CUDA_ERROR_CDP_NOT_SUPPORTED = 811,
    CUDA_ERROR_CDP_VERSION_MISMATCH = 812,
    CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED = 900,
    CUDA_ERROR_STREAM_CAPTURE_INVALIDATED = 901,
    CUDA_ERROR_STREAM_CAPTURE_MERGE = 902,
    CUDA_ERROR_STREAM_CAPTURE_UNMATCHED = 903,
    CUDA_ERROR_STREAM_CAPTURE_UNJOINED = 904,
    CUDA_ERROR_STREAM_CAPTURE_ISOLATION = 905,
    CUDA_ERROR_STREAM_CAPTURE_IMPLICIT = 906,
    CUDA_ERROR_CAPTURED_EVENT = 907,
    CUDA_ERROR_STREAM_CAPTURE_WRONG_THREAD = 908,
    CUDA_ERROR_TIMEOUT = 909,
    CUDA_ERROR_GRAPH_EXEC_UPDATE_FAILURE = 910,
    CUDA_ERROR_EXTERNAL_DEVICE = 911,
    CUDA_ERROR_INVALID_CLUSTER_SIZE = 912,
    CUDA_ERROR_FUNCTION_NOT_LOADED = 913,
    CUDA_ERROR_INVALID_RESOURCE_TYPE = 914,
    CUDA_ERROR_INVALID_RESOURCE_CONFIGURATION = 915,
    CUDA_ERROR_KEY_ROTATION = 916,
    CUDA_ERROR_UNKNOWN = 999,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cufileBatchMode {
    CUFILE_BATCH = 1,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUfileDescr_t {
    pub type_: CUfileFileHandleType,
    pub handle: CUfileDescr_t__bindgen_ty_1,
    pub fs_ops: *const CUfileFSOps_t,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUfileDrvProps {
    pub nvfs: CUfileDrvProps__bindgen_ty_1,
    pub fflags: ::core::ffi::c_uint,
    pub max_device_cache_size: ::core::ffi::c_uint,
    pub per_buffer_cache_size: ::core::ffi::c_uint,
    pub max_device_pinned_mem_size: ::core::ffi::c_uint,
    pub max_batch_io_size: ::core::ffi::c_uint,
    pub max_batch_io_timeout_msecs: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUfileDrvProps__bindgen_ty_1 {
    pub major_version: ::core::ffi::c_uint,
    pub minor_version: ::core::ffi::c_uint,
    pub poll_thresh_size: usize,
    pub max_direct_io_size: usize,
    pub dstatusflags: ::core::ffi::c_uint,
    pub dcontrolflags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUfileError {
    pub err: CUfileOpError,
    pub cu_err: CUresult,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUfileFSOps {
    pub fs_type: ::core::option::Option<
        unsafe extern "C" fn(
            handle: *mut ::core::ffi::c_void,
        ) -> *const ::core::ffi::c_char,
    >,
    pub getRDMADeviceList: ::core::option::Option<
        unsafe extern "C" fn(
            handle: *mut ::core::ffi::c_void,
            hostaddrs: *mut *mut sockaddr_t,
        ) -> ::core::ffi::c_int,
    >,
    pub getRDMADevicePriority: ::core::option::Option<
        unsafe extern "C" fn(
            handle: *mut ::core::ffi::c_void,
            arg1: *mut ::core::ffi::c_char,
            arg2: usize,
            arg3: loff_t,
            hostaddr: *mut sockaddr_t,
        ) -> ::core::ffi::c_int,
    >,
    pub read: ::core::option::Option<
        unsafe extern "C" fn(
            handle: *mut ::core::ffi::c_void,
            arg1: *mut ::core::ffi::c_char,
            arg2: usize,
            arg3: loff_t,
            arg4: *mut cufileRDMAInfo_t,
        ) -> isize,
    >,
    pub write: ::core::option::Option<
        unsafe extern "C" fn(
            handle: *mut ::core::ffi::c_void,
            arg1: *const ::core::ffi::c_char,
            arg2: usize,
            arg3: loff_t,
            arg4: *mut cufileRDMAInfo_t,
        ) -> isize,
    >,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUfileIOEvents {
    pub cookie: *mut ::core::ffi::c_void,
    pub status: CUfileStatus_t,
    pub ret: usize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUfileIOParams {
    pub mode: CUfileBatchMode_t,
    pub u: CUfileIOParams__bindgen_ty_1,
    pub fh: CUfileHandle_t,
    pub opcode: CUfileOpcode_t,
    pub cookie: *mut ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUfileIOParams__bindgen_ty_1__bindgen_ty_1 {
    pub devPtr_base: *mut ::core::ffi::c_void,
    pub file_offset: off_t,
    pub devPtr_offset: off_t,
    pub size: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cufileRDMAInfo {
    pub version: ::core::ffi::c_int,
    pub desc_len: ::core::ffi::c_int,
    pub desc_str: *const ::core::ffi::c_char,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct sockaddr {
    pub sa_family: sa_family_t,
    pub sa_data: [::core::ffi::c_char; 14usize],
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct timespec {
    pub tv_sec: __time_t,
    pub tv_nsec: __syscall_slong_t,
}
impl Default for CUfileDescr_t {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUfileDescr_t__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUfileError {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUfileIOEvents {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUfileIOParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUfileIOParams__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUfileIOParams__bindgen_ty_1__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cufileRDMAInfo {
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
pub union CUfileDescr_t__bindgen_ty_1 {
    pub fd: ::core::ffi::c_int,
    pub handle: *mut ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUfileIOParams__bindgen_ty_1 {
    pub batch: CUfileIOParams__bindgen_ty_1__bindgen_ty_1,
}
#[cfg(not(feature = "dynamic-loading"))]
extern "C" {
    pub fn cuFileBatchIOCancel(batch_idp: CUfileBatchHandle_t) -> CUfileError_t;
    pub fn cuFileBatchIODestroy(batch_idp: CUfileBatchHandle_t);
    pub fn cuFileBatchIOGetStatus(
        batch_idp: CUfileBatchHandle_t,
        min_nr: ::core::ffi::c_uint,
        nr: *mut ::core::ffi::c_uint,
        iocbp: *mut CUfileIOEvents_t,
        timeout: *mut timespec,
    ) -> CUfileError_t;
    pub fn cuFileBatchIOSetUp(
        batch_idp: *mut CUfileBatchHandle_t,
        nr: ::core::ffi::c_uint,
    ) -> CUfileError_t;
    pub fn cuFileBatchIOSubmit(
        batch_idp: CUfileBatchHandle_t,
        nr: ::core::ffi::c_uint,
        iocbp: *mut CUfileIOParams_t,
        flags: ::core::ffi::c_uint,
    ) -> CUfileError_t;
    pub fn cuFileBufDeregister(bufPtr_base: *const ::core::ffi::c_void) -> CUfileError_t;
    pub fn cuFileBufRegister(
        bufPtr_base: *const ::core::ffi::c_void,
        length: usize,
        flags: ::core::ffi::c_int,
    ) -> CUfileError_t;
    pub fn cuFileDriverClose() -> CUfileError_t;
    pub fn cuFileDriverClose_v2() -> CUfileError_t;
    pub fn cuFileDriverGetProperties(props: *mut CUfileDrvProps_t) -> CUfileError_t;
    pub fn cuFileDriverOpen() -> CUfileError_t;
    pub fn cuFileDriverSetMaxCacheSize(max_cache_size: usize) -> CUfileError_t;
    pub fn cuFileDriverSetMaxDirectIOSize(max_direct_io_size: usize) -> CUfileError_t;
    pub fn cuFileDriverSetMaxPinnedMemSize(max_pinned_size: usize) -> CUfileError_t;
    pub fn cuFileDriverSetPollMode(
        poll: bool,
        poll_threshold_size: usize,
    ) -> CUfileError_t;
    pub fn cuFileGetVersion(version: *mut ::core::ffi::c_int) -> CUfileError_t;
    pub fn cuFileHandleDeregister(fh: CUfileHandle_t);
    pub fn cuFileHandleRegister(
        fh: *mut CUfileHandle_t,
        descr: *mut CUfileDescr_t,
    ) -> CUfileError_t;
    pub fn cuFileRead(
        fh: CUfileHandle_t,
        bufPtr_base: *mut ::core::ffi::c_void,
        size: usize,
        file_offset: off_t,
        bufPtr_offset: off_t,
    ) -> isize;
    pub fn cuFileReadAsync(
        fh: CUfileHandle_t,
        bufPtr_base: *mut ::core::ffi::c_void,
        size_p: *mut usize,
        file_offset_p: *mut off_t,
        bufPtr_offset_p: *mut off_t,
        bytes_read_p: *mut isize,
        stream: CUstream,
    ) -> CUfileError_t;
    pub fn cuFileStreamDeregister(stream: CUstream) -> CUfileError_t;
    pub fn cuFileStreamRegister(
        stream: CUstream,
        flags: ::core::ffi::c_uint,
    ) -> CUfileError_t;
    pub fn cuFileUseCount() -> ::core::ffi::c_long;
    pub fn cuFileWrite(
        fh: CUfileHandle_t,
        bufPtr_base: *const ::core::ffi::c_void,
        size: usize,
        file_offset: off_t,
        bufPtr_offset: off_t,
    ) -> isize;
    pub fn cuFileWriteAsync(
        fh: CUfileHandle_t,
        bufPtr_base: *mut ::core::ffi::c_void,
        size_p: *mut usize,
        file_offset_p: *mut off_t,
        bufPtr_offset_p: *mut off_t,
        bytes_written_p: *mut isize,
        stream: CUstream,
    ) -> CUfileError_t;
}
#[cfg(feature = "dynamic-loading")]
mod loaded {
    use super::*;
    pub unsafe fn cuFileBatchIOCancel(batch_idp: CUfileBatchHandle_t) -> CUfileError_t {
        (culib().cuFileBatchIOCancel)(batch_idp)
    }
    pub unsafe fn cuFileBatchIODestroy(batch_idp: CUfileBatchHandle_t) {
        (culib().cuFileBatchIODestroy)(batch_idp)
    }
    pub unsafe fn cuFileBatchIOGetStatus(
        batch_idp: CUfileBatchHandle_t,
        min_nr: ::core::ffi::c_uint,
        nr: *mut ::core::ffi::c_uint,
        iocbp: *mut CUfileIOEvents_t,
        timeout: *mut timespec,
    ) -> CUfileError_t {
        (culib().cuFileBatchIOGetStatus)(batch_idp, min_nr, nr, iocbp, timeout)
    }
    pub unsafe fn cuFileBatchIOSetUp(
        batch_idp: *mut CUfileBatchHandle_t,
        nr: ::core::ffi::c_uint,
    ) -> CUfileError_t {
        (culib().cuFileBatchIOSetUp)(batch_idp, nr)
    }
    pub unsafe fn cuFileBatchIOSubmit(
        batch_idp: CUfileBatchHandle_t,
        nr: ::core::ffi::c_uint,
        iocbp: *mut CUfileIOParams_t,
        flags: ::core::ffi::c_uint,
    ) -> CUfileError_t {
        (culib().cuFileBatchIOSubmit)(batch_idp, nr, iocbp, flags)
    }
    pub unsafe fn cuFileBufDeregister(
        bufPtr_base: *const ::core::ffi::c_void,
    ) -> CUfileError_t {
        (culib().cuFileBufDeregister)(bufPtr_base)
    }
    pub unsafe fn cuFileBufRegister(
        bufPtr_base: *const ::core::ffi::c_void,
        length: usize,
        flags: ::core::ffi::c_int,
    ) -> CUfileError_t {
        (culib().cuFileBufRegister)(bufPtr_base, length, flags)
    }
    pub unsafe fn cuFileDriverClose() -> CUfileError_t {
        (culib().cuFileDriverClose)()
    }
    pub unsafe fn cuFileDriverClose_v2() -> CUfileError_t {
        (culib().cuFileDriverClose_v2)()
    }
    pub unsafe fn cuFileDriverGetProperties(
        props: *mut CUfileDrvProps_t,
    ) -> CUfileError_t {
        (culib().cuFileDriverGetProperties)(props)
    }
    pub unsafe fn cuFileDriverOpen() -> CUfileError_t {
        (culib().cuFileDriverOpen)()
    }
    pub unsafe fn cuFileDriverSetMaxCacheSize(max_cache_size: usize) -> CUfileError_t {
        (culib().cuFileDriverSetMaxCacheSize)(max_cache_size)
    }
    pub unsafe fn cuFileDriverSetMaxDirectIOSize(
        max_direct_io_size: usize,
    ) -> CUfileError_t {
        (culib().cuFileDriverSetMaxDirectIOSize)(max_direct_io_size)
    }
    pub unsafe fn cuFileDriverSetMaxPinnedMemSize(
        max_pinned_size: usize,
    ) -> CUfileError_t {
        (culib().cuFileDriverSetMaxPinnedMemSize)(max_pinned_size)
    }
    pub unsafe fn cuFileDriverSetPollMode(
        poll: bool,
        poll_threshold_size: usize,
    ) -> CUfileError_t {
        (culib().cuFileDriverSetPollMode)(poll, poll_threshold_size)
    }
    pub unsafe fn cuFileGetVersion(version: *mut ::core::ffi::c_int) -> CUfileError_t {
        (culib().cuFileGetVersion)(version)
    }
    pub unsafe fn cuFileHandleDeregister(fh: CUfileHandle_t) {
        (culib().cuFileHandleDeregister)(fh)
    }
    pub unsafe fn cuFileHandleRegister(
        fh: *mut CUfileHandle_t,
        descr: *mut CUfileDescr_t,
    ) -> CUfileError_t {
        (culib().cuFileHandleRegister)(fh, descr)
    }
    pub unsafe fn cuFileRead(
        fh: CUfileHandle_t,
        bufPtr_base: *mut ::core::ffi::c_void,
        size: usize,
        file_offset: off_t,
        bufPtr_offset: off_t,
    ) -> isize {
        (culib().cuFileRead)(fh, bufPtr_base, size, file_offset, bufPtr_offset)
    }
    pub unsafe fn cuFileReadAsync(
        fh: CUfileHandle_t,
        bufPtr_base: *mut ::core::ffi::c_void,
        size_p: *mut usize,
        file_offset_p: *mut off_t,
        bufPtr_offset_p: *mut off_t,
        bytes_read_p: *mut isize,
        stream: CUstream,
    ) -> CUfileError_t {
        (culib()
            .cuFileReadAsync)(
            fh,
            bufPtr_base,
            size_p,
            file_offset_p,
            bufPtr_offset_p,
            bytes_read_p,
            stream,
        )
    }
    pub unsafe fn cuFileStreamDeregister(stream: CUstream) -> CUfileError_t {
        (culib().cuFileStreamDeregister)(stream)
    }
    pub unsafe fn cuFileStreamRegister(
        stream: CUstream,
        flags: ::core::ffi::c_uint,
    ) -> CUfileError_t {
        (culib().cuFileStreamRegister)(stream, flags)
    }
    pub unsafe fn cuFileUseCount() -> ::core::ffi::c_long {
        (culib().cuFileUseCount)()
    }
    pub unsafe fn cuFileWrite(
        fh: CUfileHandle_t,
        bufPtr_base: *const ::core::ffi::c_void,
        size: usize,
        file_offset: off_t,
        bufPtr_offset: off_t,
    ) -> isize {
        (culib().cuFileWrite)(fh, bufPtr_base, size, file_offset, bufPtr_offset)
    }
    pub unsafe fn cuFileWriteAsync(
        fh: CUfileHandle_t,
        bufPtr_base: *mut ::core::ffi::c_void,
        size_p: *mut usize,
        file_offset_p: *mut off_t,
        bufPtr_offset_p: *mut off_t,
        bytes_written_p: *mut isize,
        stream: CUstream,
    ) -> CUfileError_t {
        (culib()
            .cuFileWriteAsync)(
            fh,
            bufPtr_base,
            size_p,
            file_offset_p,
            bufPtr_offset_p,
            bytes_written_p,
            stream,
        )
    }
    pub struct Lib {
        __library: ::libloading::Library,
        pub cuFileBatchIOCancel: unsafe extern "C" fn(
            batch_idp: CUfileBatchHandle_t,
        ) -> CUfileError_t,
        pub cuFileBatchIODestroy: unsafe extern "C" fn(batch_idp: CUfileBatchHandle_t),
        pub cuFileBatchIOGetStatus: unsafe extern "C" fn(
            batch_idp: CUfileBatchHandle_t,
            min_nr: ::core::ffi::c_uint,
            nr: *mut ::core::ffi::c_uint,
            iocbp: *mut CUfileIOEvents_t,
            timeout: *mut timespec,
        ) -> CUfileError_t,
        pub cuFileBatchIOSetUp: unsafe extern "C" fn(
            batch_idp: *mut CUfileBatchHandle_t,
            nr: ::core::ffi::c_uint,
        ) -> CUfileError_t,
        pub cuFileBatchIOSubmit: unsafe extern "C" fn(
            batch_idp: CUfileBatchHandle_t,
            nr: ::core::ffi::c_uint,
            iocbp: *mut CUfileIOParams_t,
            flags: ::core::ffi::c_uint,
        ) -> CUfileError_t,
        pub cuFileBufDeregister: unsafe extern "C" fn(
            bufPtr_base: *const ::core::ffi::c_void,
        ) -> CUfileError_t,
        pub cuFileBufRegister: unsafe extern "C" fn(
            bufPtr_base: *const ::core::ffi::c_void,
            length: usize,
            flags: ::core::ffi::c_int,
        ) -> CUfileError_t,
        pub cuFileDriverClose: unsafe extern "C" fn() -> CUfileError_t,
        pub cuFileDriverClose_v2: unsafe extern "C" fn() -> CUfileError_t,
        pub cuFileDriverGetProperties: unsafe extern "C" fn(
            props: *mut CUfileDrvProps_t,
        ) -> CUfileError_t,
        pub cuFileDriverOpen: unsafe extern "C" fn() -> CUfileError_t,
        pub cuFileDriverSetMaxCacheSize: unsafe extern "C" fn(
            max_cache_size: usize,
        ) -> CUfileError_t,
        pub cuFileDriverSetMaxDirectIOSize: unsafe extern "C" fn(
            max_direct_io_size: usize,
        ) -> CUfileError_t,
        pub cuFileDriverSetMaxPinnedMemSize: unsafe extern "C" fn(
            max_pinned_size: usize,
        ) -> CUfileError_t,
        pub cuFileDriverSetPollMode: unsafe extern "C" fn(
            poll: bool,
            poll_threshold_size: usize,
        ) -> CUfileError_t,
        pub cuFileGetVersion: unsafe extern "C" fn(
            version: *mut ::core::ffi::c_int,
        ) -> CUfileError_t,
        pub cuFileHandleDeregister: unsafe extern "C" fn(fh: CUfileHandle_t),
        pub cuFileHandleRegister: unsafe extern "C" fn(
            fh: *mut CUfileHandle_t,
            descr: *mut CUfileDescr_t,
        ) -> CUfileError_t,
        pub cuFileRead: unsafe extern "C" fn(
            fh: CUfileHandle_t,
            bufPtr_base: *mut ::core::ffi::c_void,
            size: usize,
            file_offset: off_t,
            bufPtr_offset: off_t,
        ) -> isize,
        pub cuFileReadAsync: unsafe extern "C" fn(
            fh: CUfileHandle_t,
            bufPtr_base: *mut ::core::ffi::c_void,
            size_p: *mut usize,
            file_offset_p: *mut off_t,
            bufPtr_offset_p: *mut off_t,
            bytes_read_p: *mut isize,
            stream: CUstream,
        ) -> CUfileError_t,
        pub cuFileStreamDeregister: unsafe extern "C" fn(
            stream: CUstream,
        ) -> CUfileError_t,
        pub cuFileStreamRegister: unsafe extern "C" fn(
            stream: CUstream,
            flags: ::core::ffi::c_uint,
        ) -> CUfileError_t,
        pub cuFileUseCount: unsafe extern "C" fn() -> ::core::ffi::c_long,
        pub cuFileWrite: unsafe extern "C" fn(
            fh: CUfileHandle_t,
            bufPtr_base: *const ::core::ffi::c_void,
            size: usize,
            file_offset: off_t,
            bufPtr_offset: off_t,
        ) -> isize,
        pub cuFileWriteAsync: unsafe extern "C" fn(
            fh: CUfileHandle_t,
            bufPtr_base: *mut ::core::ffi::c_void,
            size_p: *mut usize,
            file_offset_p: *mut off_t,
            bufPtr_offset_p: *mut off_t,
            bytes_written_p: *mut isize,
            stream: CUstream,
        ) -> CUfileError_t,
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
            let cuFileBatchIOCancel = __library
                .get(b"cuFileBatchIOCancel\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileBatchIODestroy = __library
                .get(b"cuFileBatchIODestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileBatchIOGetStatus = __library
                .get(b"cuFileBatchIOGetStatus\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileBatchIOSetUp = __library
                .get(b"cuFileBatchIOSetUp\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileBatchIOSubmit = __library
                .get(b"cuFileBatchIOSubmit\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileBufDeregister = __library
                .get(b"cuFileBufDeregister\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileBufRegister = __library
                .get(b"cuFileBufRegister\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileDriverClose = __library
                .get(b"cuFileDriverClose\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileDriverClose_v2 = __library
                .get(b"cuFileDriverClose_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileDriverGetProperties = __library
                .get(b"cuFileDriverGetProperties\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileDriverOpen = __library
                .get(b"cuFileDriverOpen\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileDriverSetMaxCacheSize = __library
                .get(b"cuFileDriverSetMaxCacheSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileDriverSetMaxDirectIOSize = __library
                .get(b"cuFileDriverSetMaxDirectIOSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileDriverSetMaxPinnedMemSize = __library
                .get(b"cuFileDriverSetMaxPinnedMemSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileDriverSetPollMode = __library
                .get(b"cuFileDriverSetPollMode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileGetVersion = __library
                .get(b"cuFileGetVersion\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileHandleDeregister = __library
                .get(b"cuFileHandleDeregister\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileHandleRegister = __library
                .get(b"cuFileHandleRegister\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileRead = __library
                .get(b"cuFileRead\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileReadAsync = __library
                .get(b"cuFileReadAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileStreamDeregister = __library
                .get(b"cuFileStreamDeregister\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileStreamRegister = __library
                .get(b"cuFileStreamRegister\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileUseCount = __library
                .get(b"cuFileUseCount\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileWrite = __library
                .get(b"cuFileWrite\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFileWriteAsync = __library
                .get(b"cuFileWriteAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            Ok(Self {
                __library,
                cuFileBatchIOCancel,
                cuFileBatchIODestroy,
                cuFileBatchIOGetStatus,
                cuFileBatchIOSetUp,
                cuFileBatchIOSubmit,
                cuFileBufDeregister,
                cuFileBufRegister,
                cuFileDriverClose,
                cuFileDriverClose_v2,
                cuFileDriverGetProperties,
                cuFileDriverOpen,
                cuFileDriverSetMaxCacheSize,
                cuFileDriverSetMaxDirectIOSize,
                cuFileDriverSetMaxPinnedMemSize,
                cuFileDriverSetPollMode,
                cuFileGetVersion,
                cuFileHandleDeregister,
                cuFileHandleRegister,
                cuFileRead,
                cuFileReadAsync,
                cuFileStreamDeregister,
                cuFileStreamRegister,
                cuFileUseCount,
                cuFileWrite,
                cuFileWriteAsync,
            })
        }
    }
    pub unsafe fn culib() -> &'static Lib {
        static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
        LIB.get_or_init(|| {
            let lib_names = std::vec!["cufile"];
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
