#![cfg_attr(feature = "no-std", no_std)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub use self::cudaAsyncNotificationType_enum as cudaAsyncNotificationType;
pub use self::cudaDataType_t as cudaDataType;
pub use self::cudaError as cudaError_t;
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub use self::cudaGraphDependencyType_enum as cudaGraphDependencyType;
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
))]
pub use self::cudaOutputMode as cudaOutputMode_t;
#[cfg(any(feature = "cuda-11040"))]
pub const CUDART_VERSION: u32 = 11040;
#[cfg(any(feature = "cuda-11050"))]
pub const CUDART_VERSION: u32 = 11050;
#[cfg(any(feature = "cuda-11060"))]
pub const CUDART_VERSION: u32 = 11060;
#[cfg(any(feature = "cuda-11070"))]
pub const CUDART_VERSION: u32 = 11070;
#[cfg(any(feature = "cuda-11080"))]
pub const CUDART_VERSION: u32 = 11080;
#[cfg(any(feature = "cuda-12000"))]
pub const CUDART_VERSION: u32 = 12000;
#[cfg(any(feature = "cuda-12010"))]
pub const CUDART_VERSION: u32 = 12010;
#[cfg(any(feature = "cuda-12020"))]
pub const CUDART_VERSION: u32 = 12020;
#[cfg(any(feature = "cuda-12030"))]
pub const CUDART_VERSION: u32 = 12030;
#[cfg(any(feature = "cuda-12040"))]
pub const CUDART_VERSION: u32 = 12040;
#[cfg(any(feature = "cuda-12050"))]
pub const CUDART_VERSION: u32 = 12050;
#[cfg(any(feature = "cuda-12060"))]
pub const CUDART_VERSION: u32 = 12060;
#[cfg(any(feature = "cuda-12080"))]
pub const CUDART_VERSION: u32 = 12080;
#[cfg(any(feature = "cuda-12090"))]
pub const CUDART_VERSION: u32 = 12090;
pub const CUDA_IPC_HANDLE_SIZE: u32 = 64;
pub const cudaArrayColorAttachment: u32 = 32;
pub const cudaArrayCubemap: u32 = 4;
pub const cudaArrayDefault: u32 = 0;
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub const cudaArrayDeferredMapping: u32 = 128;
pub const cudaArrayLayered: u32 = 1;
pub const cudaArraySparse: u32 = 64;
pub const cudaArraySparsePropertiesSingleMipTail: u32 = 1;
pub const cudaArraySurfaceLoadStore: u32 = 2;
pub const cudaArrayTextureGather: u32 = 8;
pub const cudaCooperativeLaunchMultiDeviceNoPostSync: u32 = 2;
pub const cudaCooperativeLaunchMultiDeviceNoPreSync: u32 = 1;
pub const cudaDeviceBlockingSync: u32 = 4;
pub const cudaDeviceLmemResizeToMax: u32 = 16;
pub const cudaDeviceMapHost: u32 = 8;
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080",
    feature = "cuda-12000"
))]
pub const cudaDeviceMask: u32 = 31;
#[cfg(any(
    feature = "cuda-12010",
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub const cudaDeviceMask: u32 = 255;
pub const cudaDeviceScheduleAuto: u32 = 0;
pub const cudaDeviceScheduleBlockingSync: u32 = 4;
pub const cudaDeviceScheduleMask: u32 = 7;
pub const cudaDeviceScheduleSpin: u32 = 1;
pub const cudaDeviceScheduleYield: u32 = 2;
#[cfg(any(
    feature = "cuda-12010",
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub const cudaDeviceSyncMemops: u32 = 128;
pub const cudaEventBlockingSync: u32 = 1;
pub const cudaEventDefault: u32 = 0;
pub const cudaEventDisableTiming: u32 = 2;
pub const cudaEventInterprocess: u32 = 4;
pub const cudaEventRecordDefault: u32 = 0;
pub const cudaEventRecordExternal: u32 = 1;
pub const cudaEventWaitDefault: u32 = 0;
pub const cudaEventWaitExternal: u32 = 1;
pub const cudaExternalMemoryDedicated: u32 = 1;
pub const cudaExternalSemaphoreSignalSkipNvSciBufMemSync: u32 = 1;
pub const cudaExternalSemaphoreWaitSkipNvSciBufMemSync: u32 = 2;
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub const cudaGraphKernelNodePortDefault: u32 = 0;
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub const cudaGraphKernelNodePortLaunchCompletion: u32 = 2;
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub const cudaGraphKernelNodePortProgrammatic: u32 = 1;
pub const cudaHostAllocDefault: u32 = 0;
pub const cudaHostAllocMapped: u32 = 2;
pub const cudaHostAllocPortable: u32 = 1;
pub const cudaHostAllocWriteCombined: u32 = 4;
pub const cudaHostRegisterDefault: u32 = 0;
pub const cudaHostRegisterIoMemory: u32 = 4;
pub const cudaHostRegisterMapped: u32 = 2;
pub const cudaHostRegisterPortable: u32 = 1;
pub const cudaHostRegisterReadOnly: u32 = 8;
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
pub const cudaInitDeviceFlagsAreValid: u32 = 1;
pub const cudaIpcMemLazyEnablePeerAccess: u32 = 1;
pub const cudaMemAttachGlobal: u32 = 1;
pub const cudaMemAttachHost: u32 = 2;
pub const cudaMemAttachSingle: u32 = 4;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
pub const cudaMemPoolCreateUsageHwDecompress: u32 = 2;
pub const cudaNvSciSyncAttrSignal: u32 = 1;
pub const cudaNvSciSyncAttrWait: u32 = 2;
pub const cudaOccupancyDefault: u32 = 0;
pub const cudaOccupancyDisableCachingOverride: u32 = 1;
pub const cudaPeerAccessDefault: u32 = 0;
pub const cudaStreamDefault: u32 = 0;
pub const cudaStreamNonBlocking: u32 = 1;
pub const cudaSurfaceType1D: u32 = 1;
pub const cudaSurfaceType1DLayered: u32 = 241;
pub const cudaSurfaceType2D: u32 = 2;
pub const cudaSurfaceType2DLayered: u32 = 242;
pub const cudaSurfaceType3D: u32 = 3;
pub const cudaSurfaceTypeCubemap: u32 = 12;
pub const cudaSurfaceTypeCubemapLayered: u32 = 252;
pub const cudaTextureType1D: u32 = 1;
pub const cudaTextureType1DLayered: u32 = 241;
pub const cudaTextureType2D: u32 = 2;
pub const cudaTextureType2DLayered: u32 = 242;
pub const cudaTextureType3D: u32 = 3;
pub const cudaTextureTypeCubemap: u32 = 12;
pub const cudaTextureTypeCubemapLayered: u32 = 252;
pub type cudaArray_const_t = *const cudaArray;
pub type cudaArray_t = *mut cudaArray;
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub type cudaAsyncCallback = ::core::option::Option<
    unsafe extern "C" fn(
        arg1: *mut cudaAsyncNotificationInfo_t,
        arg2: *mut ::core::ffi::c_void,
        arg3: cudaAsyncCallbackHandle_t,
    ),
>;
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub type cudaAsyncCallbackHandle_t = *mut cudaAsyncCallbackEntry;
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub type cudaAsyncNotificationInfo_t = cudaAsyncNotificationInfo;
pub type cudaEvent_t = *mut CUevent_st;
pub type cudaExternalMemory_t = *mut CUexternalMemory_st;
pub type cudaExternalSemaphore_t = *mut CUexternalSemaphore_st;
pub type cudaFunction_t = *mut CUfunc_st;
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub type cudaGraphConditionalHandle = ::core::ffi::c_ulonglong;
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub type cudaGraphDeviceNode_t = *mut CUgraphDeviceUpdatableNode_st;
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub type cudaGraphEdgeData = cudaGraphEdgeData_st;
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
pub type cudaGraphExecUpdateResultInfo = cudaGraphExecUpdateResultInfo_st;
pub type cudaGraphExec_t = *mut CUgraphExec_st;
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
pub type cudaGraphInstantiateParams = cudaGraphInstantiateParams_st;
pub type cudaGraphNode_t = *mut CUgraphNode_st;
pub type cudaGraph_t = *mut CUgraph_st;
pub type cudaGraphicsResource_t = *mut cudaGraphicsResource;
pub type cudaHostFn_t =
    ::core::option::Option<unsafe extern "C" fn(userData: *mut ::core::ffi::c_void)>;
pub type cudaIpcEventHandle_t = cudaIpcEventHandle_st;
pub type cudaIpcMemHandle_t = cudaIpcMemHandle_st;
#[cfg(any(
    feature = "cuda-12010",
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub type cudaKernel_t = *mut CUkern_st;
#[cfg(any(
    feature = "cuda-11080",
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
pub type cudaLaunchAttribute = cudaLaunchAttribute_st;
#[cfg(any(
    feature = "cuda-11080",
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
pub type cudaLaunchConfig_t = cudaLaunchConfig_st;
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
pub type cudaLaunchMemSyncDomainMap = cudaLaunchMemSyncDomainMap_st;
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
pub type cudaLibrary_t = *mut CUlib_st;
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
pub type cudaMemFabricHandle_t = cudaMemFabricHandle_st;
pub type cudaMemPool_t = *mut CUmemPoolHandle_st;
pub type cudaMipmappedArray_const_t = *const cudaMipmappedArray;
pub type cudaMipmappedArray_t = *mut cudaMipmappedArray;
pub type cudaStreamCallback_t = ::core::option::Option<
    unsafe extern "C" fn(
        stream: cudaStream_t,
        status: cudaError_t,
        userData: *mut ::core::ffi::c_void,
    ),
>;
pub type cudaStream_t = *mut CUstream_st;
pub type cudaSurfaceObject_t = ::core::ffi::c_ulonglong;
pub type cudaTextureObject_t = ::core::ffi::c_ulonglong;
pub type cudaUUID_t = CUuuid_st;
pub type cudaUserObject_t = *mut CUuserObject_st;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaAccessProperty {
    cudaAccessPropertyNormal = 0,
    cudaAccessPropertyStreaming = 1,
    cudaAccessPropertyPersisting = 2,
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaAsyncNotificationType_enum {
    cudaAsyncNotificationTypeOverBudget = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaCGScope {
    cudaCGScopeInvalid = 0,
    cudaCGScopeGrid = 1,
    cudaCGScopeMultiGrid = 2,
}
#[cfg(any(feature = "cuda-11040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaChannelFormatKind {
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3,
    cudaChannelFormatKindNV12 = 4,
}
#[cfg(any(
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
    feature = "cuda-12060"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaChannelFormatKind {
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3,
    cudaChannelFormatKindNV12 = 4,
    cudaChannelFormatKindUnsignedNormalized8X1 = 5,
    cudaChannelFormatKindUnsignedNormalized8X2 = 6,
    cudaChannelFormatKindUnsignedNormalized8X4 = 7,
    cudaChannelFormatKindUnsignedNormalized16X1 = 8,
    cudaChannelFormatKindUnsignedNormalized16X2 = 9,
    cudaChannelFormatKindUnsignedNormalized16X4 = 10,
    cudaChannelFormatKindSignedNormalized8X1 = 11,
    cudaChannelFormatKindSignedNormalized8X2 = 12,
    cudaChannelFormatKindSignedNormalized8X4 = 13,
    cudaChannelFormatKindSignedNormalized16X1 = 14,
    cudaChannelFormatKindSignedNormalized16X2 = 15,
    cudaChannelFormatKindSignedNormalized16X4 = 16,
    cudaChannelFormatKindUnsignedBlockCompressed1 = 17,
    cudaChannelFormatKindUnsignedBlockCompressed1SRGB = 18,
    cudaChannelFormatKindUnsignedBlockCompressed2 = 19,
    cudaChannelFormatKindUnsignedBlockCompressed2SRGB = 20,
    cudaChannelFormatKindUnsignedBlockCompressed3 = 21,
    cudaChannelFormatKindUnsignedBlockCompressed3SRGB = 22,
    cudaChannelFormatKindUnsignedBlockCompressed4 = 23,
    cudaChannelFormatKindSignedBlockCompressed4 = 24,
    cudaChannelFormatKindUnsignedBlockCompressed5 = 25,
    cudaChannelFormatKindSignedBlockCompressed5 = 26,
    cudaChannelFormatKindUnsignedBlockCompressed6H = 27,
    cudaChannelFormatKindSignedBlockCompressed6H = 28,
    cudaChannelFormatKindUnsignedBlockCompressed7 = 29,
    cudaChannelFormatKindUnsignedBlockCompressed7SRGB = 30,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaChannelFormatKind {
    cudaChannelFormatKindSigned = 0,
    cudaChannelFormatKindUnsigned = 1,
    cudaChannelFormatKindFloat = 2,
    cudaChannelFormatKindNone = 3,
    cudaChannelFormatKindNV12 = 4,
    cudaChannelFormatKindUnsignedNormalized8X1 = 5,
    cudaChannelFormatKindUnsignedNormalized8X2 = 6,
    cudaChannelFormatKindUnsignedNormalized8X4 = 7,
    cudaChannelFormatKindUnsignedNormalized16X1 = 8,
    cudaChannelFormatKindUnsignedNormalized16X2 = 9,
    cudaChannelFormatKindUnsignedNormalized16X4 = 10,
    cudaChannelFormatKindSignedNormalized8X1 = 11,
    cudaChannelFormatKindSignedNormalized8X2 = 12,
    cudaChannelFormatKindSignedNormalized8X4 = 13,
    cudaChannelFormatKindSignedNormalized16X1 = 14,
    cudaChannelFormatKindSignedNormalized16X2 = 15,
    cudaChannelFormatKindSignedNormalized16X4 = 16,
    cudaChannelFormatKindUnsignedBlockCompressed1 = 17,
    cudaChannelFormatKindUnsignedBlockCompressed1SRGB = 18,
    cudaChannelFormatKindUnsignedBlockCompressed2 = 19,
    cudaChannelFormatKindUnsignedBlockCompressed2SRGB = 20,
    cudaChannelFormatKindUnsignedBlockCompressed3 = 21,
    cudaChannelFormatKindUnsignedBlockCompressed3SRGB = 22,
    cudaChannelFormatKindUnsignedBlockCompressed4 = 23,
    cudaChannelFormatKindSignedBlockCompressed4 = 24,
    cudaChannelFormatKindUnsignedBlockCompressed5 = 25,
    cudaChannelFormatKindSignedBlockCompressed5 = 26,
    cudaChannelFormatKindUnsignedBlockCompressed6H = 27,
    cudaChannelFormatKindSignedBlockCompressed6H = 28,
    cudaChannelFormatKindUnsignedBlockCompressed7 = 29,
    cudaChannelFormatKindUnsignedBlockCompressed7SRGB = 30,
    cudaChannelFormatKindUnsignedNormalized1010102 = 31,
}
#[cfg(any(
    feature = "cuda-11080",
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
pub enum cudaClusterSchedulingPolicy {
    cudaClusterSchedulingPolicyDefault = 0,
    cudaClusterSchedulingPolicySpread = 1,
    cudaClusterSchedulingPolicyLoadBalancing = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaComputeMode {
    cudaComputeModeDefault = 0,
    cudaComputeModeExclusive = 1,
    cudaComputeModeProhibited = 2,
    cudaComputeModeExclusiveProcess = 3,
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070"
))]
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
#[cfg(any(
    feature = "cuda-11080",
    feature = "cuda-12000",
    feature = "cuda-12010",
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060"
))]
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
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
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
#[cfg(any(feature = "cuda-11040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrMaxTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrMax = 120,
}
#[cfg(any(feature = "cuda-11050"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrMax = 120,
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrMax = 122,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrMax = 122,
}
#[cfg(any(feature = "cuda-12000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrMax = 127,
}
#[cfg(any(feature = "cuda-12010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrMax = 133,
}
#[cfg(any(feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrNumaConfig = 130,
    cudaDevAttrNumaId = 131,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrHostNumaId = 134,
    cudaDevAttrMax = 135,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrNumaConfig = 130,
    cudaDevAttrNumaId = 131,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrMpsEnabled = 133,
    cudaDevAttrHostNumaId = 134,
    cudaDevAttrMax = 135,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrNumaConfig = 130,
    cudaDevAttrNumaId = 131,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrMpsEnabled = 133,
    cudaDevAttrHostNumaId = 134,
    cudaDevAttrD3D12CigSupported = 135,
    cudaDevAttrMax = 136,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrNumaConfig = 130,
    cudaDevAttrNumaId = 131,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrMpsEnabled = 133,
    cudaDevAttrHostNumaId = 134,
    cudaDevAttrD3D12CigSupported = 135,
    cudaDevAttrGpuPciDeviceId = 139,
    cudaDevAttrGpuPciSubsystemId = 140,
    cudaDevAttrHostNumaMultinodeIpcSupported = 143,
    cudaDevAttrMax = 144,
}
#[cfg(any(feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock = 1,
    cudaDevAttrMaxBlockDimX = 2,
    cudaDevAttrMaxBlockDimY = 3,
    cudaDevAttrMaxBlockDimZ = 4,
    cudaDevAttrMaxGridDimX = 5,
    cudaDevAttrMaxGridDimY = 6,
    cudaDevAttrMaxGridDimZ = 7,
    cudaDevAttrMaxSharedMemoryPerBlock = 8,
    cudaDevAttrTotalConstantMemory = 9,
    cudaDevAttrWarpSize = 10,
    cudaDevAttrMaxPitch = 11,
    cudaDevAttrMaxRegistersPerBlock = 12,
    cudaDevAttrClockRate = 13,
    cudaDevAttrTextureAlignment = 14,
    cudaDevAttrGpuOverlap = 15,
    cudaDevAttrMultiProcessorCount = 16,
    cudaDevAttrKernelExecTimeout = 17,
    cudaDevAttrIntegrated = 18,
    cudaDevAttrCanMapHostMemory = 19,
    cudaDevAttrComputeMode = 20,
    cudaDevAttrMaxTexture1DWidth = 21,
    cudaDevAttrMaxTexture2DWidth = 22,
    cudaDevAttrMaxTexture2DHeight = 23,
    cudaDevAttrMaxTexture3DWidth = 24,
    cudaDevAttrMaxTexture3DHeight = 25,
    cudaDevAttrMaxTexture3DDepth = 26,
    cudaDevAttrMaxTexture2DLayeredWidth = 27,
    cudaDevAttrMaxTexture2DLayeredHeight = 28,
    cudaDevAttrMaxTexture2DLayeredLayers = 29,
    cudaDevAttrSurfaceAlignment = 30,
    cudaDevAttrConcurrentKernels = 31,
    cudaDevAttrEccEnabled = 32,
    cudaDevAttrPciBusId = 33,
    cudaDevAttrPciDeviceId = 34,
    cudaDevAttrTccDriver = 35,
    cudaDevAttrMemoryClockRate = 36,
    cudaDevAttrGlobalMemoryBusWidth = 37,
    cudaDevAttrL2CacheSize = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor = 39,
    cudaDevAttrAsyncEngineCount = 40,
    cudaDevAttrUnifiedAddressing = 41,
    cudaDevAttrMaxTexture1DLayeredWidth = 42,
    cudaDevAttrMaxTexture1DLayeredLayers = 43,
    cudaDevAttrMaxTexture2DGatherWidth = 45,
    cudaDevAttrMaxTexture2DGatherHeight = 46,
    cudaDevAttrMaxTexture3DWidthAlt = 47,
    cudaDevAttrMaxTexture3DHeightAlt = 48,
    cudaDevAttrMaxTexture3DDepthAlt = 49,
    cudaDevAttrPciDomainId = 50,
    cudaDevAttrTexturePitchAlignment = 51,
    cudaDevAttrMaxTextureCubemapWidth = 52,
    cudaDevAttrMaxTextureCubemapLayeredWidth = 53,
    cudaDevAttrMaxTextureCubemapLayeredLayers = 54,
    cudaDevAttrMaxSurface1DWidth = 55,
    cudaDevAttrMaxSurface2DWidth = 56,
    cudaDevAttrMaxSurface2DHeight = 57,
    cudaDevAttrMaxSurface3DWidth = 58,
    cudaDevAttrMaxSurface3DHeight = 59,
    cudaDevAttrMaxSurface3DDepth = 60,
    cudaDevAttrMaxSurface1DLayeredWidth = 61,
    cudaDevAttrMaxSurface1DLayeredLayers = 62,
    cudaDevAttrMaxSurface2DLayeredWidth = 63,
    cudaDevAttrMaxSurface2DLayeredHeight = 64,
    cudaDevAttrMaxSurface2DLayeredLayers = 65,
    cudaDevAttrMaxSurfaceCubemapWidth = 66,
    cudaDevAttrMaxSurfaceCubemapLayeredWidth = 67,
    cudaDevAttrMaxSurfaceCubemapLayeredLayers = 68,
    cudaDevAttrMaxTexture1DLinearWidth = 69,
    cudaDevAttrMaxTexture2DLinearWidth = 70,
    cudaDevAttrMaxTexture2DLinearHeight = 71,
    cudaDevAttrMaxTexture2DLinearPitch = 72,
    cudaDevAttrMaxTexture2DMipmappedWidth = 73,
    cudaDevAttrMaxTexture2DMipmappedHeight = 74,
    cudaDevAttrComputeCapabilityMajor = 75,
    cudaDevAttrComputeCapabilityMinor = 76,
    cudaDevAttrMaxTexture1DMipmappedWidth = 77,
    cudaDevAttrStreamPrioritiesSupported = 78,
    cudaDevAttrGlobalL1CacheSupported = 79,
    cudaDevAttrLocalL1CacheSupported = 80,
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxRegistersPerMultiprocessor = 82,
    cudaDevAttrManagedMemory = 83,
    cudaDevAttrIsMultiGpuBoard = 84,
    cudaDevAttrMultiGpuBoardGroupID = 85,
    cudaDevAttrHostNativeAtomicSupported = 86,
    cudaDevAttrSingleToDoublePrecisionPerfRatio = 87,
    cudaDevAttrPageableMemoryAccess = 88,
    cudaDevAttrConcurrentManagedAccess = 89,
    cudaDevAttrComputePreemptionSupported = 90,
    cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
    cudaDevAttrReserved92 = 92,
    cudaDevAttrReserved93 = 93,
    cudaDevAttrReserved94 = 94,
    cudaDevAttrCooperativeLaunch = 95,
    cudaDevAttrCooperativeMultiDeviceLaunch = 96,
    cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
    cudaDevAttrCanFlushRemoteWrites = 98,
    cudaDevAttrHostRegisterSupported = 99,
    cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
    cudaDevAttrDirectManagedMemAccessFromHost = 101,
    cudaDevAttrMaxBlocksPerMultiprocessor = 106,
    cudaDevAttrMaxPersistingL2CacheSize = 108,
    cudaDevAttrMaxAccessPolicyWindowSize = 109,
    cudaDevAttrReservedSharedMemoryPerBlock = 111,
    cudaDevAttrSparseCudaArraySupported = 112,
    cudaDevAttrHostRegisterReadOnlySupported = 113,
    cudaDevAttrTimelineSemaphoreInteropSupported = 114,
    cudaDevAttrMemoryPoolsSupported = 115,
    cudaDevAttrGPUDirectRDMASupported = 116,
    cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117,
    cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
    cudaDevAttrMemoryPoolSupportedHandleTypes = 119,
    cudaDevAttrClusterLaunch = 120,
    cudaDevAttrDeferredMappingCudaArraySupported = 121,
    cudaDevAttrReserved122 = 122,
    cudaDevAttrReserved123 = 123,
    cudaDevAttrReserved124 = 124,
    cudaDevAttrIpcEventSupport = 125,
    cudaDevAttrMemSyncDomainCount = 126,
    cudaDevAttrReserved127 = 127,
    cudaDevAttrReserved128 = 128,
    cudaDevAttrReserved129 = 129,
    cudaDevAttrNumaConfig = 130,
    cudaDevAttrNumaId = 131,
    cudaDevAttrReserved132 = 132,
    cudaDevAttrMpsEnabled = 133,
    cudaDevAttrHostNumaId = 134,
    cudaDevAttrD3D12CigSupported = 135,
    cudaDevAttrVulkanCigSupported = 138,
    cudaDevAttrGpuPciDeviceId = 139,
    cudaDevAttrGpuPciSubsystemId = 140,
    cudaDevAttrReserved141 = 141,
    cudaDevAttrHostNumaMemoryPoolsSupported = 142,
    cudaDevAttrHostNumaMultinodeIpcSupported = 143,
    cudaDevAttrMax = 144,
}
#[cfg(any(
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
pub enum cudaDeviceNumaConfig {
    cudaDeviceNumaConfigNone = 0,
    cudaDeviceNumaConfigNumaNode = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaDeviceP2PAttr {
    cudaDevP2PAttrPerformanceRank = 1,
    cudaDevP2PAttrAccessSupported = 2,
    cudaDevP2PAttrNativeAtomicSupported = 3,
    cudaDevP2PAttrCudaArrayAccessSupported = 4,
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
pub enum cudaDriverEntryPointQueryResult {
    cudaDriverEntryPointSuccess = 0,
    cudaDriverEntryPointSymbolNotFound = 1,
    cudaDriverEntryPointVersionNotSufficent = 2,
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[cfg(any(feature = "cuda-12000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorCdpNotSupported = 811,
    cudaErrorCdpVersionMismatch = 812,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorUnsupportedDevSideSync = 225,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorCdpNotSupported = 811,
    cudaErrorCdpVersionMismatch = 812,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040", feature = "cuda-12050"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorUnsupportedDevSideSync = 225,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorLossyQuery = 402,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorCdpNotSupported = 811,
    cudaErrorCdpVersionMismatch = 812,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[cfg(any(feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorUnsupportedDevSideSync = 225,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorLossyQuery = 402,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorCdpNotSupported = 811,
    cudaErrorCdpVersionMismatch = 812,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorFunctionNotLoaded = 913,
    cudaErrorInvalidResourceType = 914,
    cudaErrorInvalidResourceConfiguration = 915,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaError {
    cudaSuccess = 0,
    cudaErrorInvalidValue = 1,
    cudaErrorMemoryAllocation = 2,
    cudaErrorInitializationError = 3,
    cudaErrorCudartUnloading = 4,
    cudaErrorProfilerDisabled = 5,
    cudaErrorProfilerNotInitialized = 6,
    cudaErrorProfilerAlreadyStarted = 7,
    cudaErrorProfilerAlreadyStopped = 8,
    cudaErrorInvalidConfiguration = 9,
    cudaErrorInvalidPitchValue = 12,
    cudaErrorInvalidSymbol = 13,
    cudaErrorInvalidHostPointer = 16,
    cudaErrorInvalidDevicePointer = 17,
    cudaErrorInvalidTexture = 18,
    cudaErrorInvalidTextureBinding = 19,
    cudaErrorInvalidChannelDescriptor = 20,
    cudaErrorInvalidMemcpyDirection = 21,
    cudaErrorAddressOfConstant = 22,
    cudaErrorTextureFetchFailed = 23,
    cudaErrorTextureNotBound = 24,
    cudaErrorSynchronizationError = 25,
    cudaErrorInvalidFilterSetting = 26,
    cudaErrorInvalidNormSetting = 27,
    cudaErrorMixedDeviceExecution = 28,
    cudaErrorNotYetImplemented = 31,
    cudaErrorMemoryValueTooLarge = 32,
    cudaErrorStubLibrary = 34,
    cudaErrorInsufficientDriver = 35,
    cudaErrorCallRequiresNewerDriver = 36,
    cudaErrorInvalidSurface = 37,
    cudaErrorDuplicateVariableName = 43,
    cudaErrorDuplicateTextureName = 44,
    cudaErrorDuplicateSurfaceName = 45,
    cudaErrorDevicesUnavailable = 46,
    cudaErrorIncompatibleDriverContext = 49,
    cudaErrorMissingConfiguration = 52,
    cudaErrorPriorLaunchFailure = 53,
    cudaErrorLaunchMaxDepthExceeded = 65,
    cudaErrorLaunchFileScopedTex = 66,
    cudaErrorLaunchFileScopedSurf = 67,
    cudaErrorSyncDepthExceeded = 68,
    cudaErrorLaunchPendingCountExceeded = 69,
    cudaErrorInvalidDeviceFunction = 98,
    cudaErrorNoDevice = 100,
    cudaErrorInvalidDevice = 101,
    cudaErrorDeviceNotLicensed = 102,
    cudaErrorSoftwareValidityNotEstablished = 103,
    cudaErrorStartupFailure = 127,
    cudaErrorInvalidKernelImage = 200,
    cudaErrorDeviceUninitialized = 201,
    cudaErrorMapBufferObjectFailed = 205,
    cudaErrorUnmapBufferObjectFailed = 206,
    cudaErrorArrayIsMapped = 207,
    cudaErrorAlreadyMapped = 208,
    cudaErrorNoKernelImageForDevice = 209,
    cudaErrorAlreadyAcquired = 210,
    cudaErrorNotMapped = 211,
    cudaErrorNotMappedAsArray = 212,
    cudaErrorNotMappedAsPointer = 213,
    cudaErrorECCUncorrectable = 214,
    cudaErrorUnsupportedLimit = 215,
    cudaErrorDeviceAlreadyInUse = 216,
    cudaErrorPeerAccessUnsupported = 217,
    cudaErrorInvalidPtx = 218,
    cudaErrorInvalidGraphicsContext = 219,
    cudaErrorNvlinkUncorrectable = 220,
    cudaErrorJitCompilerNotFound = 221,
    cudaErrorUnsupportedPtxVersion = 222,
    cudaErrorJitCompilationDisabled = 223,
    cudaErrorUnsupportedExecAffinity = 224,
    cudaErrorUnsupportedDevSideSync = 225,
    cudaErrorContained = 226,
    cudaErrorInvalidSource = 300,
    cudaErrorFileNotFound = 301,
    cudaErrorSharedObjectSymbolNotFound = 302,
    cudaErrorSharedObjectInitFailed = 303,
    cudaErrorOperatingSystem = 304,
    cudaErrorInvalidResourceHandle = 400,
    cudaErrorIllegalState = 401,
    cudaErrorLossyQuery = 402,
    cudaErrorSymbolNotFound = 500,
    cudaErrorNotReady = 600,
    cudaErrorIllegalAddress = 700,
    cudaErrorLaunchOutOfResources = 701,
    cudaErrorLaunchTimeout = 702,
    cudaErrorLaunchIncompatibleTexturing = 703,
    cudaErrorPeerAccessAlreadyEnabled = 704,
    cudaErrorPeerAccessNotEnabled = 705,
    cudaErrorSetOnActiveProcess = 708,
    cudaErrorContextIsDestroyed = 709,
    cudaErrorAssert = 710,
    cudaErrorTooManyPeers = 711,
    cudaErrorHostMemoryAlreadyRegistered = 712,
    cudaErrorHostMemoryNotRegistered = 713,
    cudaErrorHardwareStackError = 714,
    cudaErrorIllegalInstruction = 715,
    cudaErrorMisalignedAddress = 716,
    cudaErrorInvalidAddressSpace = 717,
    cudaErrorInvalidPc = 718,
    cudaErrorLaunchFailure = 719,
    cudaErrorCooperativeLaunchTooLarge = 720,
    cudaErrorTensorMemoryLeak = 721,
    cudaErrorNotPermitted = 800,
    cudaErrorNotSupported = 801,
    cudaErrorSystemNotReady = 802,
    cudaErrorSystemDriverMismatch = 803,
    cudaErrorCompatNotSupportedOnDevice = 804,
    cudaErrorMpsConnectionFailed = 805,
    cudaErrorMpsRpcFailure = 806,
    cudaErrorMpsServerNotReady = 807,
    cudaErrorMpsMaxClientsReached = 808,
    cudaErrorMpsMaxConnectionsReached = 809,
    cudaErrorMpsClientTerminated = 810,
    cudaErrorCdpNotSupported = 811,
    cudaErrorCdpVersionMismatch = 812,
    cudaErrorStreamCaptureUnsupported = 900,
    cudaErrorStreamCaptureInvalidated = 901,
    cudaErrorStreamCaptureMerge = 902,
    cudaErrorStreamCaptureUnmatched = 903,
    cudaErrorStreamCaptureUnjoined = 904,
    cudaErrorStreamCaptureIsolation = 905,
    cudaErrorStreamCaptureImplicit = 906,
    cudaErrorCapturedEvent = 907,
    cudaErrorStreamCaptureWrongThread = 908,
    cudaErrorTimeout = 909,
    cudaErrorGraphExecUpdateFailure = 910,
    cudaErrorExternalDevice = 911,
    cudaErrorInvalidClusterSize = 912,
    cudaErrorFunctionNotLoaded = 913,
    cudaErrorInvalidResourceType = 914,
    cudaErrorInvalidResourceConfiguration = 915,
    cudaErrorUnknown = 999,
    cudaErrorApiFailureBase = 10000,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaExternalMemoryHandleType {
    cudaExternalMemoryHandleTypeOpaqueFd = 1,
    cudaExternalMemoryHandleTypeOpaqueWin32 = 2,
    cudaExternalMemoryHandleTypeOpaqueWin32Kmt = 3,
    cudaExternalMemoryHandleTypeD3D12Heap = 4,
    cudaExternalMemoryHandleTypeD3D12Resource = 5,
    cudaExternalMemoryHandleTypeD3D11Resource = 6,
    cudaExternalMemoryHandleTypeD3D11ResourceKmt = 7,
    cudaExternalMemoryHandleTypeNvSciBuf = 8,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaExternalSemaphoreHandleType {
    cudaExternalSemaphoreHandleTypeOpaqueFd = 1,
    cudaExternalSemaphoreHandleTypeOpaqueWin32 = 2,
    cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt = 3,
    cudaExternalSemaphoreHandleTypeD3D12Fence = 4,
    cudaExternalSemaphoreHandleTypeD3D11Fence = 5,
    cudaExternalSemaphoreHandleTypeNvSciSync = 6,
    cudaExternalSemaphoreHandleTypeKeyedMutex = 7,
    cudaExternalSemaphoreHandleTypeKeyedMutexKmt = 8,
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd = 9,
    cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32 = 10,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaFlushGPUDirectRDMAWritesOptions {
    cudaFlushGPUDirectRDMAWritesOptionHost = 1,
    cudaFlushGPUDirectRDMAWritesOptionMemOps = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaFlushGPUDirectRDMAWritesScope {
    cudaFlushGPUDirectRDMAWritesToOwner = 100,
    cudaFlushGPUDirectRDMAWritesToAllDevices = 200,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaFlushGPUDirectRDMAWritesTarget {
    cudaFlushGPUDirectRDMAWritesTargetCurrentDevice = 0,
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
    cudaFuncAttributePreferredSharedMemoryCarveout = 9,
    cudaFuncAttributeMax = 10,
}
#[cfg(any(
    feature = "cuda-11080",
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
pub enum cudaFuncAttribute {
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
    cudaFuncAttributePreferredSharedMemoryCarveout = 9,
    cudaFuncAttributeClusterDimMustBeSet = 10,
    cudaFuncAttributeRequiredClusterWidth = 11,
    cudaFuncAttributeRequiredClusterHeight = 12,
    cudaFuncAttributeRequiredClusterDepth = 13,
    cudaFuncAttributeNonPortableClusterSizeAllowed = 14,
    cudaFuncAttributeClusterSchedulingPolicyPreference = 15,
    cudaFuncAttributeMax = 16,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaFuncCache {
    cudaFuncCachePreferNone = 0,
    cudaFuncCachePreferShared = 1,
    cudaFuncCachePreferL1 = 2,
    cudaFuncCachePreferEqual = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGPUDirectRDMAWritesOrdering {
    cudaGPUDirectRDMAWritesOrderingNone = 0,
    cudaGPUDirectRDMAWritesOrderingOwner = 100,
    cudaGPUDirectRDMAWritesOrderingAllDevices = 200,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGetDriverEntryPointFlags {
    cudaEnableDefault = 0,
    cudaEnableLegacyStream = 1,
    cudaEnablePerThreadDefaultStream = 2,
}
#[cfg(any(feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphChildGraphNodeOwnership {
    cudaGraphChildGraphOwnershipClone = 0,
    cudaGraphChildGraphOwnershipMove = 1,
}
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphConditionalHandleFlags {
    cudaGraphCondAssignDefault = 1,
}
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphConditionalNodeType {
    cudaGraphCondTypeIf = 0,
    cudaGraphCondTypeWhile = 1,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphConditionalNodeType {
    cudaGraphCondTypeIf = 0,
    cudaGraphCondTypeWhile = 1,
    cudaGraphCondTypeSwitch = 2,
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080",
    feature = "cuda-12000",
    feature = "cuda-12010",
    feature = "cuda-12020"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphDebugDotFlags {
    cudaGraphDebugDotFlagsVerbose = 1,
    cudaGraphDebugDotFlagsKernelNodeParams = 4,
    cudaGraphDebugDotFlagsMemcpyNodeParams = 8,
    cudaGraphDebugDotFlagsMemsetNodeParams = 16,
    cudaGraphDebugDotFlagsHostNodeParams = 32,
    cudaGraphDebugDotFlagsEventNodeParams = 64,
    cudaGraphDebugDotFlagsExtSemasSignalNodeParams = 128,
    cudaGraphDebugDotFlagsExtSemasWaitNodeParams = 256,
    cudaGraphDebugDotFlagsKernelNodeAttributes = 512,
    cudaGraphDebugDotFlagsHandles = 1024,
}
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphDebugDotFlags {
    cudaGraphDebugDotFlagsVerbose = 1,
    cudaGraphDebugDotFlagsKernelNodeParams = 4,
    cudaGraphDebugDotFlagsMemcpyNodeParams = 8,
    cudaGraphDebugDotFlagsMemsetNodeParams = 16,
    cudaGraphDebugDotFlagsHostNodeParams = 32,
    cudaGraphDebugDotFlagsEventNodeParams = 64,
    cudaGraphDebugDotFlagsExtSemasSignalNodeParams = 128,
    cudaGraphDebugDotFlagsExtSemasWaitNodeParams = 256,
    cudaGraphDebugDotFlagsKernelNodeAttributes = 512,
    cudaGraphDebugDotFlagsHandles = 1024,
    cudaGraphDebugDotFlagsConditionalNodeParams = 32768,
}
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphDependencyType_enum {
    cudaGraphDependencyTypeDefault = 0,
    cudaGraphDependencyTypeProgrammatic = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphExecUpdateResult {
    cudaGraphExecUpdateSuccess = 0,
    cudaGraphExecUpdateError = 1,
    cudaGraphExecUpdateErrorTopologyChanged = 2,
    cudaGraphExecUpdateErrorNodeTypeChanged = 3,
    cudaGraphExecUpdateErrorFunctionChanged = 4,
    cudaGraphExecUpdateErrorParametersChanged = 5,
    cudaGraphExecUpdateErrorNotSupported = 6,
    cudaGraphExecUpdateErrorUnsupportedFunctionChange = 7,
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphExecUpdateResult {
    cudaGraphExecUpdateSuccess = 0,
    cudaGraphExecUpdateError = 1,
    cudaGraphExecUpdateErrorTopologyChanged = 2,
    cudaGraphExecUpdateErrorNodeTypeChanged = 3,
    cudaGraphExecUpdateErrorFunctionChanged = 4,
    cudaGraphExecUpdateErrorParametersChanged = 5,
    cudaGraphExecUpdateErrorNotSupported = 6,
    cudaGraphExecUpdateErrorUnsupportedFunctionChange = 7,
    cudaGraphExecUpdateErrorAttributesChanged = 8,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphInstantiateFlags {
    cudaGraphInstantiateFlagAutoFreeOnLaunch = 1,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphInstantiateFlags {
    cudaGraphInstantiateFlagAutoFreeOnLaunch = 1,
    cudaGraphInstantiateFlagUseNodePriority = 8,
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
pub enum cudaGraphInstantiateFlags {
    cudaGraphInstantiateFlagAutoFreeOnLaunch = 1,
    cudaGraphInstantiateFlagUpload = 2,
    cudaGraphInstantiateFlagDeviceLaunch = 4,
    cudaGraphInstantiateFlagUseNodePriority = 8,
}
#[cfg(any(
    feature = "cuda-12000",
    feature = "cuda-12010",
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphInstantiateResult {
    cudaGraphInstantiateSuccess = 0,
    cudaGraphInstantiateError = 1,
    cudaGraphInstantiateInvalidStructure = 2,
    cudaGraphInstantiateNodeOperationNotSupported = 3,
    cudaGraphInstantiateMultipleDevicesNotSupported = 4,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphInstantiateResult {
    cudaGraphInstantiateSuccess = 0,
    cudaGraphInstantiateError = 1,
    cudaGraphInstantiateInvalidStructure = 2,
    cudaGraphInstantiateNodeOperationNotSupported = 3,
    cudaGraphInstantiateMultipleDevicesNotSupported = 4,
    cudaGraphInstantiateConditionalHandleUnused = 5,
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphKernelNodeField {
    cudaGraphKernelNodeFieldInvalid = 0,
    cudaGraphKernelNodeFieldGridDim = 1,
    cudaGraphKernelNodeFieldParam = 2,
    cudaGraphKernelNodeFieldEnabled = 3,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphMemAttributeType {
    cudaGraphMemAttrUsedMemCurrent = 1,
    cudaGraphMemAttrUsedMemHigh = 2,
    cudaGraphMemAttrReservedMemCurrent = 3,
    cudaGraphMemAttrReservedMemHigh = 4,
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphMemAttributeType {
    cudaGraphMemAttrUsedMemCurrent = 0,
    cudaGraphMemAttrUsedMemHigh = 1,
    cudaGraphMemAttrReservedMemCurrent = 2,
    cudaGraphMemAttrReservedMemHigh = 3,
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080",
    feature = "cuda-12000",
    feature = "cuda-12010",
    feature = "cuda-12020"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel = 0,
    cudaGraphNodeTypeMemcpy = 1,
    cudaGraphNodeTypeMemset = 2,
    cudaGraphNodeTypeHost = 3,
    cudaGraphNodeTypeGraph = 4,
    cudaGraphNodeTypeEmpty = 5,
    cudaGraphNodeTypeWaitEvent = 6,
    cudaGraphNodeTypeEventRecord = 7,
    cudaGraphNodeTypeExtSemaphoreSignal = 8,
    cudaGraphNodeTypeExtSemaphoreWait = 9,
    cudaGraphNodeTypeMemAlloc = 10,
    cudaGraphNodeTypeMemFree = 11,
    cudaGraphNodeTypeCount = 12,
}
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphNodeType {
    cudaGraphNodeTypeKernel = 0,
    cudaGraphNodeTypeMemcpy = 1,
    cudaGraphNodeTypeMemset = 2,
    cudaGraphNodeTypeHost = 3,
    cudaGraphNodeTypeGraph = 4,
    cudaGraphNodeTypeEmpty = 5,
    cudaGraphNodeTypeWaitEvent = 6,
    cudaGraphNodeTypeEventRecord = 7,
    cudaGraphNodeTypeExtSemaphoreSignal = 8,
    cudaGraphNodeTypeExtSemaphoreWait = 9,
    cudaGraphNodeTypeMemAlloc = 10,
    cudaGraphNodeTypeMemFree = 11,
    cudaGraphNodeTypeConditional = 13,
    cudaGraphNodeTypeCount = 14,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphicsCubeFace {
    cudaGraphicsCubeFacePositiveX = 0,
    cudaGraphicsCubeFaceNegativeX = 1,
    cudaGraphicsCubeFacePositiveY = 2,
    cudaGraphicsCubeFaceNegativeY = 3,
    cudaGraphicsCubeFacePositiveZ = 4,
    cudaGraphicsCubeFaceNegativeZ = 5,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphicsMapFlags {
    cudaGraphicsMapFlagsNone = 0,
    cudaGraphicsMapFlagsReadOnly = 1,
    cudaGraphicsMapFlagsWriteDiscard = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaGraphicsRegisterFlags {
    cudaGraphicsRegisterFlagsNone = 0,
    cudaGraphicsRegisterFlagsReadOnly = 1,
    cudaGraphicsRegisterFlagsWriteDiscard = 2,
    cudaGraphicsRegisterFlagsSurfaceLoadStore = 4,
    cudaGraphicsRegisterFlagsTextureGather = 8,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaJitOption {
    cudaJitMaxRegisters = 0,
    cudaJitThreadsPerBlock = 1,
    cudaJitWallTime = 2,
    cudaJitInfoLogBuffer = 3,
    cudaJitInfoLogBufferSizeBytes = 4,
    cudaJitErrorLogBuffer = 5,
    cudaJitErrorLogBufferSizeBytes = 6,
    cudaJitOptimizationLevel = 7,
    cudaJitFallbackStrategy = 10,
    cudaJitGenerateDebugInfo = 11,
    cudaJitLogVerbose = 12,
    cudaJitGenerateLineInfo = 13,
    cudaJitCacheMode = 14,
    cudaJitPositionIndependentCode = 30,
    cudaJitMinCtaPerSm = 31,
    cudaJitMaxThreadsPerBlock = 32,
    cudaJitOverrideDirectiveValues = 33,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaJit_CacheMode {
    cudaJitCacheOptionNone = 0,
    cudaJitCacheOptionCG = 1,
    cudaJitCacheOptionCA = 2,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaJit_Fallback {
    cudaPreferPtx = 0,
    cudaPreferBinary = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaKernelNodeAttrID {
    cudaKernelNodeAttributeAccessPolicyWindow = 1,
    cudaKernelNodeAttributeCooperative = 2,
}
#[cfg(any(feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaKernelNodeAttrID {
    cudaKernelNodeAttributeAccessPolicyWindow = 1,
    cudaKernelNodeAttributeCooperative = 2,
    cudaKernelNodeAttributePriority = 8,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
    cudaLaunchAttributeMemSyncDomainMap = 9,
    cudaLaunchAttributeMemSyncDomain = 10,
}
#[cfg(any(feature = "cuda-12030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
    cudaLaunchAttributeMemSyncDomainMap = 9,
    cudaLaunchAttributeMemSyncDomain = 10,
    cudaLaunchAttributeLaunchCompletionEvent = 12,
}
#[cfg(any(feature = "cuda-12040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
    cudaLaunchAttributeMemSyncDomainMap = 9,
    cudaLaunchAttributeMemSyncDomain = 10,
    cudaLaunchAttributeLaunchCompletionEvent = 12,
    cudaLaunchAttributeDeviceUpdatableKernelNode = 13,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
    cudaLaunchAttributeMemSyncDomainMap = 9,
    cudaLaunchAttributeMemSyncDomain = 10,
    cudaLaunchAttributeLaunchCompletionEvent = 12,
    cudaLaunchAttributeDeviceUpdatableKernelNode = 13,
    cudaLaunchAttributePreferredSharedMemoryCarveout = 14,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLaunchAttributeID {
    cudaLaunchAttributeIgnore = 0,
    cudaLaunchAttributeAccessPolicyWindow = 1,
    cudaLaunchAttributeCooperative = 2,
    cudaLaunchAttributeSynchronizationPolicy = 3,
    cudaLaunchAttributeClusterDimension = 4,
    cudaLaunchAttributeClusterSchedulingPolicyPreference = 5,
    cudaLaunchAttributeProgrammaticStreamSerialization = 6,
    cudaLaunchAttributeProgrammaticEvent = 7,
    cudaLaunchAttributePriority = 8,
    cudaLaunchAttributeMemSyncDomainMap = 9,
    cudaLaunchAttributeMemSyncDomain = 10,
    cudaLaunchAttributePreferredClusterDimension = 11,
    cudaLaunchAttributeLaunchCompletionEvent = 12,
    cudaLaunchAttributeDeviceUpdatableKernelNode = 13,
    cudaLaunchAttributePreferredSharedMemoryCarveout = 14,
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
pub enum cudaLaunchMemSyncDomain {
    cudaLaunchMemSyncDomainDefault = 0,
    cudaLaunchMemSyncDomainRemote = 1,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLibraryOption {
    cudaLibraryHostUniversalFunctionAndDataTable = 0,
    cudaLibraryBinaryIsPreserved = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaLimit {
    cudaLimitStackSize = 0,
    cudaLimitPrintfFifoSize = 1,
    cudaLimitMallocHeapSize = 2,
    cudaLimitDevRuntimeSyncDepth = 3,
    cudaLimitDevRuntimePendingLaunchCount = 4,
    cudaLimitMaxL2FetchGranularity = 5,
    cudaLimitPersistingL2CacheSize = 6,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemAccessFlags {
    cudaMemAccessFlagsProtNone = 0,
    cudaMemAccessFlagsProtRead = 1,
    cudaMemAccessFlagsProtReadWrite = 3,
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
    feature = "cuda-12030"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemAllocationHandleType {
    cudaMemHandleTypeNone = 0,
    cudaMemHandleTypePosixFileDescriptor = 1,
    cudaMemHandleTypeWin32 = 2,
    cudaMemHandleTypeWin32Kmt = 4,
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemAllocationHandleType {
    cudaMemHandleTypeNone = 0,
    cudaMemHandleTypePosixFileDescriptor = 1,
    cudaMemHandleTypeWin32 = 2,
    cudaMemHandleTypeWin32Kmt = 4,
    cudaMemHandleTypeFabric = 8,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemAllocationType {
    cudaMemAllocationTypeInvalid = 0,
    cudaMemAllocationTypePinned = 1,
    cudaMemAllocationTypeMax = 2147483647,
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080",
    feature = "cuda-12000",
    feature = "cuda-12010"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemLocationType {
    cudaMemLocationTypeInvalid = 0,
    cudaMemLocationTypeDevice = 1,
}
#[cfg(any(
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
pub enum cudaMemLocationType {
    cudaMemLocationTypeInvalid = 0,
    cudaMemLocationTypeDevice = 1,
    cudaMemLocationTypeHost = 2,
    cudaMemLocationTypeHostNuma = 3,
    cudaMemLocationTypeHostNumaCurrent = 4,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemPoolAttr {
    cudaMemPoolReuseFollowEventDependencies = 1,
    cudaMemPoolReuseAllowOpportunistic = 2,
    cudaMemPoolReuseAllowInternalDependencies = 3,
    cudaMemPoolAttrReleaseThreshold = 4,
    cudaMemPoolAttrReservedMemCurrent = 5,
    cudaMemPoolAttrReservedMemHigh = 6,
    cudaMemPoolAttrUsedMemCurrent = 7,
    cudaMemPoolAttrUsedMemHigh = 8,
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080",
    feature = "cuda-12000",
    feature = "cuda-12010"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemRangeAttribute {
    cudaMemRangeAttributeReadMostly = 1,
    cudaMemRangeAttributePreferredLocation = 2,
    cudaMemRangeAttributeAccessedBy = 3,
    cudaMemRangeAttributeLastPrefetchLocation = 4,
}
#[cfg(any(
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
pub enum cudaMemRangeAttribute {
    cudaMemRangeAttributeReadMostly = 1,
    cudaMemRangeAttributePreferredLocation = 2,
    cudaMemRangeAttributeAccessedBy = 3,
    cudaMemRangeAttributeLastPrefetchLocation = 4,
    cudaMemRangeAttributePreferredLocationType = 5,
    cudaMemRangeAttributePreferredLocationId = 6,
    cudaMemRangeAttributeLastPrefetchLocationType = 7,
    cudaMemRangeAttributeLastPrefetchLocationId = 8,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemcpy3DOperandType {
    cudaMemcpyOperandTypePointer = 1,
    cudaMemcpyOperandTypeArray = 2,
    cudaMemcpyOperandTypeMax = 2147483647,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemcpyFlags {
    cudaMemcpyFlagDefault = 0,
    cudaMemcpyFlagPreferOverlapWithCompute = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemcpySrcAccessOrder {
    cudaMemcpySrcAccessOrderInvalid = 0,
    cudaMemcpySrcAccessOrderStream = 1,
    cudaMemcpySrcAccessOrderDuringApiCall = 2,
    cudaMemcpySrcAccessOrderAny = 3,
    cudaMemcpySrcAccessOrderMax = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemoryAdvise {
    cudaMemAdviseSetReadMostly = 1,
    cudaMemAdviseUnsetReadMostly = 2,
    cudaMemAdviseSetPreferredLocation = 3,
    cudaMemAdviseUnsetPreferredLocation = 4,
    cudaMemAdviseSetAccessedBy = 5,
    cudaMemAdviseUnsetAccessedBy = 6,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaMemoryType {
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3,
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaOutputMode {
    cudaKeyValuePair = 0,
    cudaCSV = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaResourceType {
    cudaResourceTypeArray = 0,
    cudaResourceTypeMipmappedArray = 1,
    cudaResourceTypeLinear = 2,
    cudaResourceTypePitch2D = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaResourceViewFormat {
    cudaResViewFormatNone = 0,
    cudaResViewFormatUnsignedChar1 = 1,
    cudaResViewFormatUnsignedChar2 = 2,
    cudaResViewFormatUnsignedChar4 = 3,
    cudaResViewFormatSignedChar1 = 4,
    cudaResViewFormatSignedChar2 = 5,
    cudaResViewFormatSignedChar4 = 6,
    cudaResViewFormatUnsignedShort1 = 7,
    cudaResViewFormatUnsignedShort2 = 8,
    cudaResViewFormatUnsignedShort4 = 9,
    cudaResViewFormatSignedShort1 = 10,
    cudaResViewFormatSignedShort2 = 11,
    cudaResViewFormatSignedShort4 = 12,
    cudaResViewFormatUnsignedInt1 = 13,
    cudaResViewFormatUnsignedInt2 = 14,
    cudaResViewFormatUnsignedInt4 = 15,
    cudaResViewFormatSignedInt1 = 16,
    cudaResViewFormatSignedInt2 = 17,
    cudaResViewFormatSignedInt4 = 18,
    cudaResViewFormatHalf1 = 19,
    cudaResViewFormatHalf2 = 20,
    cudaResViewFormatHalf4 = 21,
    cudaResViewFormatFloat1 = 22,
    cudaResViewFormatFloat2 = 23,
    cudaResViewFormatFloat4 = 24,
    cudaResViewFormatUnsignedBlockCompressed1 = 25,
    cudaResViewFormatUnsignedBlockCompressed2 = 26,
    cudaResViewFormatUnsignedBlockCompressed3 = 27,
    cudaResViewFormatUnsignedBlockCompressed4 = 28,
    cudaResViewFormatSignedBlockCompressed4 = 29,
    cudaResViewFormatUnsignedBlockCompressed5 = 30,
    cudaResViewFormatSignedBlockCompressed5 = 31,
    cudaResViewFormatUnsignedBlockCompressed6H = 32,
    cudaResViewFormatSignedBlockCompressed6H = 33,
    cudaResViewFormatUnsignedBlockCompressed7 = 34,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaRoundMode {
    cudaRoundNearest = 0,
    cudaRoundZero = 1,
    cudaRoundPosInf = 2,
    cudaRoundMinInf = 3,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaSharedCarveout {
    cudaSharedmemCarveoutDefault = -1,
    cudaSharedmemCarveoutMaxShared = 100,
    cudaSharedmemCarveoutMaxL1 = 0,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaSharedMemConfig {
    cudaSharedMemBankSizeDefault = 0,
    cudaSharedMemBankSizeFourByte = 1,
    cudaSharedMemBankSizeEightByte = 2,
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070"
))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaStreamAttrID {
    cudaStreamAttributeAccessPolicyWindow = 1,
    cudaStreamAttributeSynchronizationPolicy = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaStreamCaptureMode {
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaStreamCaptureStatus {
    cudaStreamCaptureStatusNone = 0,
    cudaStreamCaptureStatusActive = 1,
    cudaStreamCaptureStatusInvalidated = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaStreamUpdateCaptureDependenciesFlags {
    cudaStreamAddCaptureDependencies = 0,
    cudaStreamSetCaptureDependencies = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaSurfaceBoundaryMode {
    cudaBoundaryModeZero = 0,
    cudaBoundaryModeClamp = 1,
    cudaBoundaryModeTrap = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaSurfaceFormatMode {
    cudaFormatModeForced = 0,
    cudaFormatModeAuto = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaSynchronizationPolicy {
    cudaSyncPolicyAuto = 1,
    cudaSyncPolicySpin = 2,
    cudaSyncPolicyYield = 3,
    cudaSyncPolicyBlockingSync = 4,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaTextureAddressMode {
    cudaAddressModeWrap = 0,
    cudaAddressModeClamp = 1,
    cudaAddressModeMirror = 2,
    cudaAddressModeBorder = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaTextureFilterMode {
    cudaFilterModePoint = 0,
    cudaFilterModeLinear = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaTextureReadMode {
    cudaReadModeElementType = 0,
    cudaReadModeNormalizedFloat = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaUserObjectFlags {
    cudaUserObjectNoDestructorSync = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum cudaUserObjectRetainFlags {
    cudaGraphUserObjectMove = 1,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUevent_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUexternalMemory_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUexternalSemaphore_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUfunc_st {
    _unused: [u8; 0],
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraphDeviceUpdatableNode_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraphExec_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraphNode_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraph_st {
    _unused: [u8; 0],
}
#[cfg(any(
    feature = "cuda-12010",
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUkern_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUlib_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUmemPoolHandle_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUuserObject_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUuuid_st {
    pub bytes: [::core::ffi::c_char; 16usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct cudaAccessPolicyWindow {
    pub base_ptr: *mut ::core::ffi::c_void,
    pub num_bytes: usize,
    pub hitRatio: f32,
    pub hitProp: cudaAccessProperty,
    pub missProp: cudaAccessProperty,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaArray {
    _unused: [u8; 0],
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaArrayMemoryRequirements {
    pub size: usize,
    pub alignment: usize,
    pub reserved: [::core::ffi::c_uint; 4usize],
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaArraySparseProperties {
    pub tileExtent: cudaArraySparseProperties__bindgen_ty_1,
    pub miptailFirstLevel: ::core::ffi::c_uint,
    pub miptailSize: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 4usize],
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaArraySparseProperties__bindgen_ty_1 {
    pub width: ::core::ffi::c_uint,
    pub height: ::core::ffi::c_uint,
    pub depth: ::core::ffi::c_uint,
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaAsyncCallbackEntry {
    _unused: [u8; 0],
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaAsyncNotificationInfo {
    pub type_: cudaAsyncNotificationType,
    pub info: cudaAsyncNotificationInfo__bindgen_ty_1,
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaAsyncNotificationInfo__bindgen_ty_1__bindgen_ty_1 {
    pub bytesOverBudget: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaChannelFormatDesc {
    pub x: ::core::ffi::c_int,
    pub y: ::core::ffi::c_int,
    pub z: ::core::ffi::c_int,
    pub w: ::core::ffi::c_int,
    pub f: cudaChannelFormatKind,
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaChildGraphNodeParams {
    pub graph: cudaGraph_t,
}
#[cfg(any(feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaChildGraphNodeParams {
    pub graph: cudaGraph_t,
    pub ownership: cudaGraphChildGraphNodeOwnership,
}
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaConditionalNodeParams {
    pub handle: cudaGraphConditionalHandle,
    pub type_: cudaGraphConditionalNodeType,
    pub size: ::core::ffi::c_uint,
    pub phGraph_out: *mut cudaGraph_t,
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaDeviceProp {
    pub name: [::core::ffi::c_char; 256usize],
    pub uuid: cudaUUID_t,
    pub luid: [::core::ffi::c_char; 8usize],
    pub luidDeviceNodeMask: ::core::ffi::c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: ::core::ffi::c_int,
    pub warpSize: ::core::ffi::c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub maxThreadsDim: [::core::ffi::c_int; 3usize],
    pub maxGridSize: [::core::ffi::c_int; 3usize],
    pub clockRate: ::core::ffi::c_int,
    pub totalConstMem: usize,
    pub major: ::core::ffi::c_int,
    pub minor: ::core::ffi::c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: ::core::ffi::c_int,
    pub multiProcessorCount: ::core::ffi::c_int,
    pub kernelExecTimeoutEnabled: ::core::ffi::c_int,
    pub integrated: ::core::ffi::c_int,
    pub canMapHostMemory: ::core::ffi::c_int,
    pub computeMode: ::core::ffi::c_int,
    pub maxTexture1D: ::core::ffi::c_int,
    pub maxTexture1DMipmap: ::core::ffi::c_int,
    pub maxTexture1DLinear: ::core::ffi::c_int,
    pub maxTexture2D: [::core::ffi::c_int; 2usize],
    pub maxTexture2DMipmap: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLinear: [::core::ffi::c_int; 3usize],
    pub maxTexture2DGather: [::core::ffi::c_int; 2usize],
    pub maxTexture3D: [::core::ffi::c_int; 3usize],
    pub maxTexture3DAlt: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemap: ::core::ffi::c_int,
    pub maxTexture1DLayered: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLayered: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemapLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface1D: ::core::ffi::c_int,
    pub maxSurface2D: [::core::ffi::c_int; 2usize],
    pub maxSurface3D: [::core::ffi::c_int; 3usize],
    pub maxSurface1DLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface2DLayered: [::core::ffi::c_int; 3usize],
    pub maxSurfaceCubemap: ::core::ffi::c_int,
    pub maxSurfaceCubemapLayered: [::core::ffi::c_int; 2usize],
    pub surfaceAlignment: usize,
    pub concurrentKernels: ::core::ffi::c_int,
    pub ECCEnabled: ::core::ffi::c_int,
    pub pciBusID: ::core::ffi::c_int,
    pub pciDeviceID: ::core::ffi::c_int,
    pub pciDomainID: ::core::ffi::c_int,
    pub tccDriver: ::core::ffi::c_int,
    pub asyncEngineCount: ::core::ffi::c_int,
    pub unifiedAddressing: ::core::ffi::c_int,
    pub memoryClockRate: ::core::ffi::c_int,
    pub memoryBusWidth: ::core::ffi::c_int,
    pub l2CacheSize: ::core::ffi::c_int,
    pub persistingL2CacheMaxSize: ::core::ffi::c_int,
    pub maxThreadsPerMultiProcessor: ::core::ffi::c_int,
    pub streamPrioritiesSupported: ::core::ffi::c_int,
    pub globalL1CacheSupported: ::core::ffi::c_int,
    pub localL1CacheSupported: ::core::ffi::c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: ::core::ffi::c_int,
    pub managedMemory: ::core::ffi::c_int,
    pub isMultiGpuBoard: ::core::ffi::c_int,
    pub multiGpuBoardGroupID: ::core::ffi::c_int,
    pub hostNativeAtomicSupported: ::core::ffi::c_int,
    pub singleToDoublePrecisionPerfRatio: ::core::ffi::c_int,
    pub pageableMemoryAccess: ::core::ffi::c_int,
    pub concurrentManagedAccess: ::core::ffi::c_int,
    pub computePreemptionSupported: ::core::ffi::c_int,
    pub canUseHostPointerForRegisteredMem: ::core::ffi::c_int,
    pub cooperativeLaunch: ::core::ffi::c_int,
    pub cooperativeMultiDeviceLaunch: ::core::ffi::c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: ::core::ffi::c_int,
    pub directManagedMemAccessFromHost: ::core::ffi::c_int,
    pub maxBlocksPerMultiProcessor: ::core::ffi::c_int,
    pub accessPolicyMaxWindowSize: ::core::ffi::c_int,
    pub reservedSharedMemPerBlock: usize,
}
#[cfg(any(feature = "cuda-12000"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaDeviceProp {
    pub name: [::core::ffi::c_char; 256usize],
    pub uuid: cudaUUID_t,
    pub luid: [::core::ffi::c_char; 8usize],
    pub luidDeviceNodeMask: ::core::ffi::c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: ::core::ffi::c_int,
    pub warpSize: ::core::ffi::c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub maxThreadsDim: [::core::ffi::c_int; 3usize],
    pub maxGridSize: [::core::ffi::c_int; 3usize],
    pub clockRate: ::core::ffi::c_int,
    pub totalConstMem: usize,
    pub major: ::core::ffi::c_int,
    pub minor: ::core::ffi::c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: ::core::ffi::c_int,
    pub multiProcessorCount: ::core::ffi::c_int,
    pub kernelExecTimeoutEnabled: ::core::ffi::c_int,
    pub integrated: ::core::ffi::c_int,
    pub canMapHostMemory: ::core::ffi::c_int,
    pub computeMode: ::core::ffi::c_int,
    pub maxTexture1D: ::core::ffi::c_int,
    pub maxTexture1DMipmap: ::core::ffi::c_int,
    pub maxTexture1DLinear: ::core::ffi::c_int,
    pub maxTexture2D: [::core::ffi::c_int; 2usize],
    pub maxTexture2DMipmap: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLinear: [::core::ffi::c_int; 3usize],
    pub maxTexture2DGather: [::core::ffi::c_int; 2usize],
    pub maxTexture3D: [::core::ffi::c_int; 3usize],
    pub maxTexture3DAlt: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemap: ::core::ffi::c_int,
    pub maxTexture1DLayered: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLayered: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemapLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface1D: ::core::ffi::c_int,
    pub maxSurface2D: [::core::ffi::c_int; 2usize],
    pub maxSurface3D: [::core::ffi::c_int; 3usize],
    pub maxSurface1DLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface2DLayered: [::core::ffi::c_int; 3usize],
    pub maxSurfaceCubemap: ::core::ffi::c_int,
    pub maxSurfaceCubemapLayered: [::core::ffi::c_int; 2usize],
    pub surfaceAlignment: usize,
    pub concurrentKernels: ::core::ffi::c_int,
    pub ECCEnabled: ::core::ffi::c_int,
    pub pciBusID: ::core::ffi::c_int,
    pub pciDeviceID: ::core::ffi::c_int,
    pub pciDomainID: ::core::ffi::c_int,
    pub tccDriver: ::core::ffi::c_int,
    pub asyncEngineCount: ::core::ffi::c_int,
    pub unifiedAddressing: ::core::ffi::c_int,
    pub memoryClockRate: ::core::ffi::c_int,
    pub memoryBusWidth: ::core::ffi::c_int,
    pub l2CacheSize: ::core::ffi::c_int,
    pub persistingL2CacheMaxSize: ::core::ffi::c_int,
    pub maxThreadsPerMultiProcessor: ::core::ffi::c_int,
    pub streamPrioritiesSupported: ::core::ffi::c_int,
    pub globalL1CacheSupported: ::core::ffi::c_int,
    pub localL1CacheSupported: ::core::ffi::c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: ::core::ffi::c_int,
    pub managedMemory: ::core::ffi::c_int,
    pub isMultiGpuBoard: ::core::ffi::c_int,
    pub multiGpuBoardGroupID: ::core::ffi::c_int,
    pub hostNativeAtomicSupported: ::core::ffi::c_int,
    pub singleToDoublePrecisionPerfRatio: ::core::ffi::c_int,
    pub pageableMemoryAccess: ::core::ffi::c_int,
    pub concurrentManagedAccess: ::core::ffi::c_int,
    pub computePreemptionSupported: ::core::ffi::c_int,
    pub canUseHostPointerForRegisteredMem: ::core::ffi::c_int,
    pub cooperativeLaunch: ::core::ffi::c_int,
    pub cooperativeMultiDeviceLaunch: ::core::ffi::c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: ::core::ffi::c_int,
    pub directManagedMemAccessFromHost: ::core::ffi::c_int,
    pub maxBlocksPerMultiProcessor: ::core::ffi::c_int,
    pub accessPolicyMaxWindowSize: ::core::ffi::c_int,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: ::core::ffi::c_int,
    pub sparseCudaArraySupported: ::core::ffi::c_int,
    pub hostRegisterReadOnlySupported: ::core::ffi::c_int,
    pub timelineSemaphoreInteropSupported: ::core::ffi::c_int,
    pub memoryPoolsSupported: ::core::ffi::c_int,
    pub gpuDirectRDMASupported: ::core::ffi::c_int,
    pub gpuDirectRDMAFlushWritesOptions: ::core::ffi::c_uint,
    pub gpuDirectRDMAWritesOrdering: ::core::ffi::c_int,
    pub memoryPoolSupportedHandleTypes: ::core::ffi::c_uint,
    pub deferredMappingCudaArraySupported: ::core::ffi::c_int,
    pub ipcEventSupported: ::core::ffi::c_int,
    pub clusterLaunch: ::core::ffi::c_int,
    pub unifiedFunctionPointers: ::core::ffi::c_int,
    pub reserved: [::core::ffi::c_int; 63usize],
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaDeviceProp {
    pub name: [::core::ffi::c_char; 256usize],
    pub uuid: cudaUUID_t,
    pub luid: [::core::ffi::c_char; 8usize],
    pub luidDeviceNodeMask: ::core::ffi::c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: ::core::ffi::c_int,
    pub warpSize: ::core::ffi::c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub maxThreadsDim: [::core::ffi::c_int; 3usize],
    pub maxGridSize: [::core::ffi::c_int; 3usize],
    pub clockRate: ::core::ffi::c_int,
    pub totalConstMem: usize,
    pub major: ::core::ffi::c_int,
    pub minor: ::core::ffi::c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: ::core::ffi::c_int,
    pub multiProcessorCount: ::core::ffi::c_int,
    pub kernelExecTimeoutEnabled: ::core::ffi::c_int,
    pub integrated: ::core::ffi::c_int,
    pub canMapHostMemory: ::core::ffi::c_int,
    pub computeMode: ::core::ffi::c_int,
    pub maxTexture1D: ::core::ffi::c_int,
    pub maxTexture1DMipmap: ::core::ffi::c_int,
    pub maxTexture1DLinear: ::core::ffi::c_int,
    pub maxTexture2D: [::core::ffi::c_int; 2usize],
    pub maxTexture2DMipmap: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLinear: [::core::ffi::c_int; 3usize],
    pub maxTexture2DGather: [::core::ffi::c_int; 2usize],
    pub maxTexture3D: [::core::ffi::c_int; 3usize],
    pub maxTexture3DAlt: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemap: ::core::ffi::c_int,
    pub maxTexture1DLayered: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLayered: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemapLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface1D: ::core::ffi::c_int,
    pub maxSurface2D: [::core::ffi::c_int; 2usize],
    pub maxSurface3D: [::core::ffi::c_int; 3usize],
    pub maxSurface1DLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface2DLayered: [::core::ffi::c_int; 3usize],
    pub maxSurfaceCubemap: ::core::ffi::c_int,
    pub maxSurfaceCubemapLayered: [::core::ffi::c_int; 2usize],
    pub surfaceAlignment: usize,
    pub concurrentKernels: ::core::ffi::c_int,
    pub ECCEnabled: ::core::ffi::c_int,
    pub pciBusID: ::core::ffi::c_int,
    pub pciDeviceID: ::core::ffi::c_int,
    pub pciDomainID: ::core::ffi::c_int,
    pub tccDriver: ::core::ffi::c_int,
    pub asyncEngineCount: ::core::ffi::c_int,
    pub unifiedAddressing: ::core::ffi::c_int,
    pub memoryClockRate: ::core::ffi::c_int,
    pub memoryBusWidth: ::core::ffi::c_int,
    pub l2CacheSize: ::core::ffi::c_int,
    pub persistingL2CacheMaxSize: ::core::ffi::c_int,
    pub maxThreadsPerMultiProcessor: ::core::ffi::c_int,
    pub streamPrioritiesSupported: ::core::ffi::c_int,
    pub globalL1CacheSupported: ::core::ffi::c_int,
    pub localL1CacheSupported: ::core::ffi::c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: ::core::ffi::c_int,
    pub managedMemory: ::core::ffi::c_int,
    pub isMultiGpuBoard: ::core::ffi::c_int,
    pub multiGpuBoardGroupID: ::core::ffi::c_int,
    pub hostNativeAtomicSupported: ::core::ffi::c_int,
    pub singleToDoublePrecisionPerfRatio: ::core::ffi::c_int,
    pub pageableMemoryAccess: ::core::ffi::c_int,
    pub concurrentManagedAccess: ::core::ffi::c_int,
    pub computePreemptionSupported: ::core::ffi::c_int,
    pub canUseHostPointerForRegisteredMem: ::core::ffi::c_int,
    pub cooperativeLaunch: ::core::ffi::c_int,
    pub cooperativeMultiDeviceLaunch: ::core::ffi::c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: ::core::ffi::c_int,
    pub directManagedMemAccessFromHost: ::core::ffi::c_int,
    pub maxBlocksPerMultiProcessor: ::core::ffi::c_int,
    pub accessPolicyMaxWindowSize: ::core::ffi::c_int,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: ::core::ffi::c_int,
    pub sparseCudaArraySupported: ::core::ffi::c_int,
    pub hostRegisterReadOnlySupported: ::core::ffi::c_int,
    pub timelineSemaphoreInteropSupported: ::core::ffi::c_int,
    pub memoryPoolsSupported: ::core::ffi::c_int,
    pub gpuDirectRDMASupported: ::core::ffi::c_int,
    pub gpuDirectRDMAFlushWritesOptions: ::core::ffi::c_uint,
    pub gpuDirectRDMAWritesOrdering: ::core::ffi::c_int,
    pub memoryPoolSupportedHandleTypes: ::core::ffi::c_uint,
    pub deferredMappingCudaArraySupported: ::core::ffi::c_int,
    pub ipcEventSupported: ::core::ffi::c_int,
    pub clusterLaunch: ::core::ffi::c_int,
    pub unifiedFunctionPointers: ::core::ffi::c_int,
    pub reserved2: [::core::ffi::c_int; 2usize],
    pub reserved: [::core::ffi::c_int; 61usize],
}
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaDeviceProp {
    pub name: [::core::ffi::c_char; 256usize],
    pub uuid: cudaUUID_t,
    pub luid: [::core::ffi::c_char; 8usize],
    pub luidDeviceNodeMask: ::core::ffi::c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: ::core::ffi::c_int,
    pub warpSize: ::core::ffi::c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub maxThreadsDim: [::core::ffi::c_int; 3usize],
    pub maxGridSize: [::core::ffi::c_int; 3usize],
    pub clockRate: ::core::ffi::c_int,
    pub totalConstMem: usize,
    pub major: ::core::ffi::c_int,
    pub minor: ::core::ffi::c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: ::core::ffi::c_int,
    pub multiProcessorCount: ::core::ffi::c_int,
    pub kernelExecTimeoutEnabled: ::core::ffi::c_int,
    pub integrated: ::core::ffi::c_int,
    pub canMapHostMemory: ::core::ffi::c_int,
    pub computeMode: ::core::ffi::c_int,
    pub maxTexture1D: ::core::ffi::c_int,
    pub maxTexture1DMipmap: ::core::ffi::c_int,
    pub maxTexture1DLinear: ::core::ffi::c_int,
    pub maxTexture2D: [::core::ffi::c_int; 2usize],
    pub maxTexture2DMipmap: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLinear: [::core::ffi::c_int; 3usize],
    pub maxTexture2DGather: [::core::ffi::c_int; 2usize],
    pub maxTexture3D: [::core::ffi::c_int; 3usize],
    pub maxTexture3DAlt: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemap: ::core::ffi::c_int,
    pub maxTexture1DLayered: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLayered: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemapLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface1D: ::core::ffi::c_int,
    pub maxSurface2D: [::core::ffi::c_int; 2usize],
    pub maxSurface3D: [::core::ffi::c_int; 3usize],
    pub maxSurface1DLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface2DLayered: [::core::ffi::c_int; 3usize],
    pub maxSurfaceCubemap: ::core::ffi::c_int,
    pub maxSurfaceCubemapLayered: [::core::ffi::c_int; 2usize],
    pub surfaceAlignment: usize,
    pub concurrentKernels: ::core::ffi::c_int,
    pub ECCEnabled: ::core::ffi::c_int,
    pub pciBusID: ::core::ffi::c_int,
    pub pciDeviceID: ::core::ffi::c_int,
    pub pciDomainID: ::core::ffi::c_int,
    pub tccDriver: ::core::ffi::c_int,
    pub asyncEngineCount: ::core::ffi::c_int,
    pub unifiedAddressing: ::core::ffi::c_int,
    pub memoryClockRate: ::core::ffi::c_int,
    pub memoryBusWidth: ::core::ffi::c_int,
    pub l2CacheSize: ::core::ffi::c_int,
    pub persistingL2CacheMaxSize: ::core::ffi::c_int,
    pub maxThreadsPerMultiProcessor: ::core::ffi::c_int,
    pub streamPrioritiesSupported: ::core::ffi::c_int,
    pub globalL1CacheSupported: ::core::ffi::c_int,
    pub localL1CacheSupported: ::core::ffi::c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: ::core::ffi::c_int,
    pub managedMemory: ::core::ffi::c_int,
    pub isMultiGpuBoard: ::core::ffi::c_int,
    pub multiGpuBoardGroupID: ::core::ffi::c_int,
    pub hostNativeAtomicSupported: ::core::ffi::c_int,
    pub singleToDoublePrecisionPerfRatio: ::core::ffi::c_int,
    pub pageableMemoryAccess: ::core::ffi::c_int,
    pub concurrentManagedAccess: ::core::ffi::c_int,
    pub computePreemptionSupported: ::core::ffi::c_int,
    pub canUseHostPointerForRegisteredMem: ::core::ffi::c_int,
    pub cooperativeLaunch: ::core::ffi::c_int,
    pub cooperativeMultiDeviceLaunch: ::core::ffi::c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: ::core::ffi::c_int,
    pub directManagedMemAccessFromHost: ::core::ffi::c_int,
    pub maxBlocksPerMultiProcessor: ::core::ffi::c_int,
    pub accessPolicyMaxWindowSize: ::core::ffi::c_int,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: ::core::ffi::c_int,
    pub sparseCudaArraySupported: ::core::ffi::c_int,
    pub hostRegisterReadOnlySupported: ::core::ffi::c_int,
    pub timelineSemaphoreInteropSupported: ::core::ffi::c_int,
    pub memoryPoolsSupported: ::core::ffi::c_int,
    pub gpuDirectRDMASupported: ::core::ffi::c_int,
    pub gpuDirectRDMAFlushWritesOptions: ::core::ffi::c_uint,
    pub gpuDirectRDMAWritesOrdering: ::core::ffi::c_int,
    pub memoryPoolSupportedHandleTypes: ::core::ffi::c_uint,
    pub deferredMappingCudaArraySupported: ::core::ffi::c_int,
    pub ipcEventSupported: ::core::ffi::c_int,
    pub clusterLaunch: ::core::ffi::c_int,
    pub unifiedFunctionPointers: ::core::ffi::c_int,
    pub reserved2: [::core::ffi::c_int; 2usize],
    pub reserved1: [::core::ffi::c_int; 1usize],
    pub reserved: [::core::ffi::c_int; 60usize],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaDeviceProp {
    pub name: [::core::ffi::c_char; 256usize],
    pub uuid: cudaUUID_t,
    pub luid: [::core::ffi::c_char; 8usize],
    pub luidDeviceNodeMask: ::core::ffi::c_uint,
    pub totalGlobalMem: usize,
    pub sharedMemPerBlock: usize,
    pub regsPerBlock: ::core::ffi::c_int,
    pub warpSize: ::core::ffi::c_int,
    pub memPitch: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub maxThreadsDim: [::core::ffi::c_int; 3usize],
    pub maxGridSize: [::core::ffi::c_int; 3usize],
    pub clockRate: ::core::ffi::c_int,
    pub totalConstMem: usize,
    pub major: ::core::ffi::c_int,
    pub minor: ::core::ffi::c_int,
    pub textureAlignment: usize,
    pub texturePitchAlignment: usize,
    pub deviceOverlap: ::core::ffi::c_int,
    pub multiProcessorCount: ::core::ffi::c_int,
    pub kernelExecTimeoutEnabled: ::core::ffi::c_int,
    pub integrated: ::core::ffi::c_int,
    pub canMapHostMemory: ::core::ffi::c_int,
    pub computeMode: ::core::ffi::c_int,
    pub maxTexture1D: ::core::ffi::c_int,
    pub maxTexture1DMipmap: ::core::ffi::c_int,
    pub maxTexture1DLinear: ::core::ffi::c_int,
    pub maxTexture2D: [::core::ffi::c_int; 2usize],
    pub maxTexture2DMipmap: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLinear: [::core::ffi::c_int; 3usize],
    pub maxTexture2DGather: [::core::ffi::c_int; 2usize],
    pub maxTexture3D: [::core::ffi::c_int; 3usize],
    pub maxTexture3DAlt: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemap: ::core::ffi::c_int,
    pub maxTexture1DLayered: [::core::ffi::c_int; 2usize],
    pub maxTexture2DLayered: [::core::ffi::c_int; 3usize],
    pub maxTextureCubemapLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface1D: ::core::ffi::c_int,
    pub maxSurface2D: [::core::ffi::c_int; 2usize],
    pub maxSurface3D: [::core::ffi::c_int; 3usize],
    pub maxSurface1DLayered: [::core::ffi::c_int; 2usize],
    pub maxSurface2DLayered: [::core::ffi::c_int; 3usize],
    pub maxSurfaceCubemap: ::core::ffi::c_int,
    pub maxSurfaceCubemapLayered: [::core::ffi::c_int; 2usize],
    pub surfaceAlignment: usize,
    pub concurrentKernels: ::core::ffi::c_int,
    pub ECCEnabled: ::core::ffi::c_int,
    pub pciBusID: ::core::ffi::c_int,
    pub pciDeviceID: ::core::ffi::c_int,
    pub pciDomainID: ::core::ffi::c_int,
    pub tccDriver: ::core::ffi::c_int,
    pub asyncEngineCount: ::core::ffi::c_int,
    pub unifiedAddressing: ::core::ffi::c_int,
    pub memoryClockRate: ::core::ffi::c_int,
    pub memoryBusWidth: ::core::ffi::c_int,
    pub l2CacheSize: ::core::ffi::c_int,
    pub persistingL2CacheMaxSize: ::core::ffi::c_int,
    pub maxThreadsPerMultiProcessor: ::core::ffi::c_int,
    pub streamPrioritiesSupported: ::core::ffi::c_int,
    pub globalL1CacheSupported: ::core::ffi::c_int,
    pub localL1CacheSupported: ::core::ffi::c_int,
    pub sharedMemPerMultiprocessor: usize,
    pub regsPerMultiprocessor: ::core::ffi::c_int,
    pub managedMemory: ::core::ffi::c_int,
    pub isMultiGpuBoard: ::core::ffi::c_int,
    pub multiGpuBoardGroupID: ::core::ffi::c_int,
    pub hostNativeAtomicSupported: ::core::ffi::c_int,
    pub singleToDoublePrecisionPerfRatio: ::core::ffi::c_int,
    pub pageableMemoryAccess: ::core::ffi::c_int,
    pub concurrentManagedAccess: ::core::ffi::c_int,
    pub computePreemptionSupported: ::core::ffi::c_int,
    pub canUseHostPointerForRegisteredMem: ::core::ffi::c_int,
    pub cooperativeLaunch: ::core::ffi::c_int,
    pub cooperativeMultiDeviceLaunch: ::core::ffi::c_int,
    pub sharedMemPerBlockOptin: usize,
    pub pageableMemoryAccessUsesHostPageTables: ::core::ffi::c_int,
    pub directManagedMemAccessFromHost: ::core::ffi::c_int,
    pub maxBlocksPerMultiProcessor: ::core::ffi::c_int,
    pub accessPolicyMaxWindowSize: ::core::ffi::c_int,
    pub reservedSharedMemPerBlock: usize,
    pub hostRegisterSupported: ::core::ffi::c_int,
    pub sparseCudaArraySupported: ::core::ffi::c_int,
    pub hostRegisterReadOnlySupported: ::core::ffi::c_int,
    pub timelineSemaphoreInteropSupported: ::core::ffi::c_int,
    pub memoryPoolsSupported: ::core::ffi::c_int,
    pub gpuDirectRDMASupported: ::core::ffi::c_int,
    pub gpuDirectRDMAFlushWritesOptions: ::core::ffi::c_uint,
    pub gpuDirectRDMAWritesOrdering: ::core::ffi::c_int,
    pub memoryPoolSupportedHandleTypes: ::core::ffi::c_uint,
    pub deferredMappingCudaArraySupported: ::core::ffi::c_int,
    pub ipcEventSupported: ::core::ffi::c_int,
    pub clusterLaunch: ::core::ffi::c_int,
    pub unifiedFunctionPointers: ::core::ffi::c_int,
    pub reserved: [::core::ffi::c_int; 63usize],
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaEventRecordNodeParams {
    pub event: cudaEvent_t,
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaEventWaitNodeParams {
    pub event: cudaEvent_t,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExtent {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalMemoryBufferDesc {
    pub offset: ::core::ffi::c_ulonglong,
    pub size: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalMemoryHandleDesc {
    pub type_: cudaExternalMemoryHandleType,
    pub handle: cudaExternalMemoryHandleDesc__bindgen_ty_1,
    pub size: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalMemoryHandleDesc__bindgen_ty_1__bindgen_ty_1 {
    pub handle: *mut ::core::ffi::c_void,
    pub name: *const ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalMemoryMipmappedArrayDesc {
    pub offset: ::core::ffi::c_ulonglong,
    pub formatDesc: cudaChannelFormatDesc,
    pub extent: cudaExtent,
    pub flags: ::core::ffi::c_uint,
    pub numLevels: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreHandleDesc {
    pub type_: cudaExternalSemaphoreHandleType,
    pub handle: cudaExternalSemaphoreHandleDesc__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreHandleDesc__bindgen_ty_1__bindgen_ty_1 {
    pub handle: *mut ::core::ffi::c_void,
    pub name: *const ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreSignalNodeParams {
    pub extSemArray: *mut cudaExternalSemaphore_t,
    pub paramsArray: *const cudaExternalSemaphoreSignalParams,
    pub numExtSems: ::core::ffi::c_uint,
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreSignalNodeParamsV2 {
    pub extSemArray: *mut cudaExternalSemaphore_t,
    pub paramsArray: *const cudaExternalSemaphoreSignalParams,
    pub numExtSems: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreSignalParams {
    pub params: cudaExternalSemaphoreSignalParams__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreSignalParams__bindgen_ty_1 {
    pub fence: cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSync: cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_2,
    pub keyedMutex: cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_3,
    pub reserved: [::core::ffi::c_uint; 12usize],
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_1 {
    pub value: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_3 {
    pub key: ::core::ffi::c_ulonglong,
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreSignalParams_v1 {
    pub params: cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1 {
    pub fence: cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSync: cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1__bindgen_ty_2,
    pub keyedMutex: cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1__bindgen_ty_3,
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1__bindgen_ty_1 {
    pub value: ::core::ffi::c_ulonglong,
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1__bindgen_ty_3 {
    pub key: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreWaitNodeParams {
    pub extSemArray: *mut cudaExternalSemaphore_t,
    pub paramsArray: *const cudaExternalSemaphoreWaitParams,
    pub numExtSems: ::core::ffi::c_uint,
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreWaitNodeParamsV2 {
    pub extSemArray: *mut cudaExternalSemaphore_t,
    pub paramsArray: *const cudaExternalSemaphoreWaitParams,
    pub numExtSems: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreWaitParams {
    pub params: cudaExternalSemaphoreWaitParams__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreWaitParams__bindgen_ty_1 {
    pub fence: cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSync: cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_2,
    pub keyedMutex: cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_3,
    pub reserved: [::core::ffi::c_uint; 10usize],
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_1 {
    pub value: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_3 {
    pub key: ::core::ffi::c_ulonglong,
    pub timeoutMs: ::core::ffi::c_uint,
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreWaitParams_v1 {
    pub params: cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1 {
    pub fence: cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSync: cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1__bindgen_ty_2,
    pub keyedMutex: cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1__bindgen_ty_3,
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1__bindgen_ty_1 {
    pub value: ::core::ffi::c_ulonglong,
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1__bindgen_ty_3 {
    pub key: ::core::ffi::c_ulonglong,
    pub timeoutMs: ::core::ffi::c_uint,
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaFuncAttributes {
    pub sharedSizeBytes: usize,
    pub constSizeBytes: usize,
    pub localSizeBytes: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub numRegs: ::core::ffi::c_int,
    pub ptxVersion: ::core::ffi::c_int,
    pub binaryVersion: ::core::ffi::c_int,
    pub cacheModeCA: ::core::ffi::c_int,
    pub maxDynamicSharedSizeBytes: ::core::ffi::c_int,
    pub preferredShmemCarveout: ::core::ffi::c_int,
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
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaFuncAttributes {
    pub sharedSizeBytes: usize,
    pub constSizeBytes: usize,
    pub localSizeBytes: usize,
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub numRegs: ::core::ffi::c_int,
    pub ptxVersion: ::core::ffi::c_int,
    pub binaryVersion: ::core::ffi::c_int,
    pub cacheModeCA: ::core::ffi::c_int,
    pub maxDynamicSharedSizeBytes: ::core::ffi::c_int,
    pub preferredShmemCarveout: ::core::ffi::c_int,
    pub clusterDimMustBeSet: ::core::ffi::c_int,
    pub requiredClusterWidth: ::core::ffi::c_int,
    pub requiredClusterHeight: ::core::ffi::c_int,
    pub requiredClusterDepth: ::core::ffi::c_int,
    pub clusterSchedulingPolicyPreference: ::core::ffi::c_int,
    pub nonPortableClusterSizeAllowed: ::core::ffi::c_int,
    pub reserved: [::core::ffi::c_int; 16usize],
}
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaGraphEdgeData_st {
    pub from_port: ::core::ffi::c_uchar,
    pub to_port: ::core::ffi::c_uchar,
    pub type_: ::core::ffi::c_uchar,
    pub reserved: [::core::ffi::c_uchar; 5usize],
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
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaGraphExecUpdateResultInfo_st {
    pub result: cudaGraphExecUpdateResult,
    pub errorNode: cudaGraphNode_t,
    pub errorFromNode: cudaGraphNode_t,
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
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaGraphInstantiateParams_st {
    pub flags: ::core::ffi::c_ulonglong,
    pub uploadStream: cudaStream_t,
    pub errNode_out: cudaGraphNode_t,
    pub result_out: cudaGraphInstantiateResult,
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaGraphKernelNodeUpdate {
    pub node: cudaGraphDeviceNode_t,
    pub field: cudaGraphKernelNodeField,
    pub updateData: cudaGraphKernelNodeUpdate__bindgen_ty_1,
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaGraphKernelNodeUpdate__bindgen_ty_1__bindgen_ty_1 {
    pub pValue: *const ::core::ffi::c_void,
    pub offset: usize,
    pub size: usize,
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaGraphNodeParams {
    pub type_: cudaGraphNodeType,
    pub reserved0: [::core::ffi::c_int; 3usize],
    pub __bindgen_anon_1: cudaGraphNodeParams__bindgen_ty_1,
    pub reserved2: ::core::ffi::c_longlong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaGraphicsResource {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaHostNodeParams {
    pub fn_: cudaHostFn_t,
    pub userData: *mut ::core::ffi::c_void,
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaHostNodeParamsV2 {
    pub fn_: cudaHostFn_t,
    pub userData: *mut ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaIpcEventHandle_st {
    pub reserved: [::core::ffi::c_char; 64usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaIpcMemHandle_st {
    pub reserved: [::core::ffi::c_char; 64usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaKernelNodeParams {
    pub func: *mut ::core::ffi::c_void,
    pub gridDim: dim3,
    pub blockDim: dim3,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub kernelParams: *mut *mut ::core::ffi::c_void,
    pub extra: *mut *mut ::core::ffi::c_void,
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaKernelNodeParamsV2 {
    pub func: *mut ::core::ffi::c_void,
    pub gridDim: dim3,
    pub blockDim: dim3,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub kernelParams: *mut *mut ::core::ffi::c_void,
    pub extra: *mut *mut ::core::ffi::c_void,
}
#[cfg(any(
    feature = "cuda-11080",
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
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchAttributeValue__bindgen_ty_1 {
    pub x: ::core::ffi::c_uint,
    pub y: ::core::ffi::c_uint,
    pub z: ::core::ffi::c_uint,
}
#[cfg(any(
    feature = "cuda-11080",
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
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchAttributeValue__bindgen_ty_2 {
    pub event: cudaEvent_t,
    pub flags: ::core::ffi::c_int,
    pub triggerAtBlockStart: ::core::ffi::c_int,
}
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchAttributeValue__bindgen_ty_3 {
    pub event: cudaEvent_t,
    pub flags: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchAttributeValue__bindgen_ty_3 {
    pub x: ::core::ffi::c_uint,
    pub y: ::core::ffi::c_uint,
    pub z: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchAttributeValue__bindgen_ty_4 {
    pub deviceUpdatable: ::core::ffi::c_int,
    pub devNode: cudaGraphDeviceNode_t,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchAttributeValue__bindgen_ty_4 {
    pub event: cudaEvent_t,
    pub flags: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchAttributeValue__bindgen_ty_5 {
    pub deviceUpdatable: ::core::ffi::c_int,
    pub devNode: cudaGraphDeviceNode_t,
}
#[cfg(any(
    feature = "cuda-11080",
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
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaLaunchAttribute_st {
    pub id: cudaLaunchAttributeID,
    pub pad: [::core::ffi::c_char; 4usize],
    pub val: cudaLaunchAttributeValue,
}
#[cfg(any(
    feature = "cuda-11080",
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
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchConfig_st {
    pub gridDim: dim3,
    pub blockDim: dim3,
    pub dynamicSmemBytes: usize,
    pub stream: cudaStream_t,
    pub attrs: *mut cudaLaunchAttribute,
    pub numAttrs: ::core::ffi::c_uint,
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
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchMemSyncDomainMap_st {
    pub default_: ::core::ffi::c_uchar,
    pub remote: ::core::ffi::c_uchar,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaLaunchParams {
    pub func: *mut ::core::ffi::c_void,
    pub gridDim: dim3,
    pub blockDim: dim3,
    pub args: *mut *mut ::core::ffi::c_void,
    pub sharedMem: usize,
    pub stream: cudaStream_t,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemAccessDesc {
    pub location: cudaMemLocation,
    pub flags: cudaMemAccessFlags,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemAllocNodeParams {
    pub poolProps: cudaMemPoolProps,
    pub accessDescs: *const cudaMemAccessDesc,
    pub accessDescCount: usize,
    pub bytesize: usize,
    pub dptr: *mut ::core::ffi::c_void,
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemAllocNodeParamsV2 {
    pub poolProps: cudaMemPoolProps,
    pub accessDescs: *const cudaMemAccessDesc,
    pub accessDescCount: usize,
    pub bytesize: usize,
    pub dptr: *mut ::core::ffi::c_void,
}
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemFabricHandle_st {
    pub reserved: [::core::ffi::c_char; 64usize],
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemFreeNodeParams {
    pub dptr: *mut ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemLocation {
    pub type_: cudaMemLocationType,
    pub id: ::core::ffi::c_int,
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080",
    feature = "cuda-12000",
    feature = "cuda-12010"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemPoolProps {
    pub allocType: cudaMemAllocationType,
    pub handleTypes: cudaMemAllocationHandleType,
    pub location: cudaMemLocation,
    pub win32SecurityAttributes: *mut ::core::ffi::c_void,
    pub reserved: [::core::ffi::c_uchar; 64usize],
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemPoolProps {
    pub allocType: cudaMemAllocationType,
    pub handleTypes: cudaMemAllocationHandleType,
    pub location: cudaMemLocation,
    pub win32SecurityAttributes: *mut ::core::ffi::c_void,
    pub maxSize: usize,
    pub reserved: [::core::ffi::c_uchar; 56usize],
}
#[cfg(any(feature = "cuda-12060", feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemPoolProps {
    pub allocType: cudaMemAllocationType,
    pub handleTypes: cudaMemAllocationHandleType,
    pub location: cudaMemLocation,
    pub win32SecurityAttributes: *mut ::core::ffi::c_void,
    pub maxSize: usize,
    pub usage: ::core::ffi::c_ushort,
    pub reserved: [::core::ffi::c_uchar; 54usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemPoolPtrExportData {
    pub reserved: [::core::ffi::c_uchar; 64usize],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaMemcpy3DBatchOp {
    pub src: cudaMemcpy3DOperand,
    pub dst: cudaMemcpy3DOperand,
    pub extent: cudaExtent,
    pub srcAccessOrder: cudaMemcpySrcAccessOrder,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaMemcpy3DOperand {
    pub type_: cudaMemcpy3DOperandType,
    pub op: cudaMemcpy3DOperand__bindgen_ty_1,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemcpy3DOperand__bindgen_ty_1__bindgen_ty_1 {
    pub ptr: *mut ::core::ffi::c_void,
    pub rowLength: usize,
    pub layerHeight: usize,
    pub locHint: cudaMemLocation,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemcpy3DOperand__bindgen_ty_1__bindgen_ty_2 {
    pub array: cudaArray_t,
    pub offset: cudaOffset3D,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemcpy3DParms {
    pub srcArray: cudaArray_t,
    pub srcPos: cudaPos,
    pub srcPtr: cudaPitchedPtr,
    pub dstArray: cudaArray_t,
    pub dstPos: cudaPos,
    pub dstPtr: cudaPitchedPtr,
    pub extent: cudaExtent,
    pub kind: cudaMemcpyKind,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemcpy3DPeerParms {
    pub srcArray: cudaArray_t,
    pub srcPos: cudaPos,
    pub srcPtr: cudaPitchedPtr,
    pub srcDevice: ::core::ffi::c_int,
    pub dstArray: cudaArray_t,
    pub dstPos: cudaPos,
    pub dstPtr: cudaPitchedPtr,
    pub dstDevice: ::core::ffi::c_int,
    pub extent: cudaExtent,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemcpyAttributes {
    pub srcAccessOrder: cudaMemcpySrcAccessOrder,
    pub srcLocHint: cudaMemLocation,
    pub dstLocHint: cudaMemLocation,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemcpyNodeParams {
    pub flags: ::core::ffi::c_int,
    pub reserved: [::core::ffi::c_int; 3usize],
    pub copyParams: cudaMemcpy3DParms,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemsetParams {
    pub dst: *mut ::core::ffi::c_void,
    pub pitch: usize,
    pub value: ::core::ffi::c_uint,
    pub elementSize: ::core::ffi::c_uint,
    pub width: usize,
    pub height: usize,
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaMemsetParamsV2 {
    pub dst: *mut ::core::ffi::c_void,
    pub pitch: usize,
    pub value: ::core::ffi::c_uint,
    pub elementSize: ::core::ffi::c_uint,
    pub width: usize,
    pub height: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct cudaMipmappedArray {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaOffset3D {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaPitchedPtr {
    pub ptr: *mut ::core::ffi::c_void,
    pub pitch: usize,
    pub xsize: usize,
    pub ysize: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaPointerAttributes {
    pub type_: cudaMemoryType,
    pub device: ::core::ffi::c_int,
    pub devicePointer: *mut ::core::ffi::c_void,
    pub hostPointer: *mut ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaPos {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct cudaResourceDesc {
    pub resType: cudaResourceType,
    pub res: cudaResourceDesc__bindgen_ty_1,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_1 {
    pub array: cudaArray_t,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_2 {
    pub mipmap: cudaMipmappedArray_t,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_3 {
    pub devPtr: *mut ::core::ffi::c_void,
    pub desc: cudaChannelFormatDesc,
    pub sizeInBytes: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaResourceDesc__bindgen_ty_1__bindgen_ty_4 {
    pub devPtr: *mut ::core::ffi::c_void,
    pub desc: cudaChannelFormatDesc,
    pub width: usize,
    pub height: usize,
    pub pitchInBytes: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudaResourceViewDesc {
    pub format: cudaResourceViewFormat,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub firstMipmapLevel: ::core::ffi::c_uint,
    pub lastMipmapLevel: ::core::ffi::c_uint,
    pub firstLayer: ::core::ffi::c_uint,
    pub lastLayer: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct cudaTextureDesc {
    pub addressMode: [cudaTextureAddressMode; 3usize],
    pub filterMode: cudaTextureFilterMode,
    pub readMode: cudaTextureReadMode,
    pub sRGB: ::core::ffi::c_int,
    pub borderColor: [f32; 4usize],
    pub normalizedCoords: ::core::ffi::c_int,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: cudaTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub disableTrilinearOptimization: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-11060", feature = "cuda-11070"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct cudaTextureDesc {
    pub addressMode: [cudaTextureAddressMode; 3usize],
    pub filterMode: cudaTextureFilterMode,
    pub readMode: cudaTextureReadMode,
    pub sRGB: ::core::ffi::c_int,
    pub borderColor: [f32; 4usize],
    pub normalizedCoords: ::core::ffi::c_int,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: cudaTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub disableTrilinearOptimization: ::core::ffi::c_int,
    pub seamlessCubemap: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct cudaTextureDesc {
    pub addressMode: [cudaTextureAddressMode; 3usize],
    pub filterMode: cudaTextureFilterMode,
    pub readMode: cudaTextureReadMode,
    pub sRGB: ::core::ffi::c_int,
    pub borderColor: [f32; 4usize],
    pub normalizedCoords: ::core::ffi::c_int,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: cudaTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub disableTrilinearOptimization: ::core::ffi::c_int,
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
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct cudaTextureDesc {
    pub addressMode: [cudaTextureAddressMode; 3usize],
    pub filterMode: cudaTextureFilterMode,
    pub readMode: cudaTextureReadMode,
    pub sRGB: ::core::ffi::c_int,
    pub borderColor: [f32; 4usize],
    pub normalizedCoords: ::core::ffi::c_int,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: cudaTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub disableTrilinearOptimization: ::core::ffi::c_int,
    pub seamlessCubemap: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct cudaTextureDesc_v2 {
    pub addressMode: [cudaTextureAddressMode; 3usize],
    pub filterMode: cudaTextureFilterMode,
    pub readMode: cudaTextureReadMode,
    pub sRGB: ::core::ffi::c_int,
    pub borderColor: [f32; 4usize],
    pub normalizedCoords: ::core::ffi::c_int,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: cudaTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub disableTrilinearOptimization: ::core::ffi::c_int,
    pub seamlessCubemap: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct cudalibraryHostUniversalFunctionAndDataTable {
    pub functionTable: *mut ::core::ffi::c_void,
    pub functionWindowSize: usize,
    pub dataTable: *mut ::core::ffi::c_void,
    pub dataWindowSize: usize,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct dim3 {
    pub x: ::core::ffi::c_uint,
    pub y: ::core::ffi::c_uint,
    pub z: ::core::ffi::c_uint,
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct surfaceReference {
    pub channelDesc: cudaChannelFormatDesc,
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
))]
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct textureReference {
    pub normalized: ::core::ffi::c_int,
    pub filterMode: cudaTextureFilterMode,
    pub addressMode: [cudaTextureAddressMode; 3usize],
    pub channelDesc: cudaChannelFormatDesc,
    pub sRGB: ::core::ffi::c_int,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: cudaTextureFilterMode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub disableTrilinearOptimization: ::core::ffi::c_int,
    pub __cudaReserved: [::core::ffi::c_int; 14usize],
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
impl cudaDataType_t {
    pub const CUDA_R_8F_UE4M3: cudaDataType_t = cudaDataType_t::CUDA_R_8F_E4M3;
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl cudaDeviceAttr {
    pub const cudaDevAttrMaxTimelineSemaphoreInteropSupported: cudaDeviceAttr =
        cudaDeviceAttr::cudaDevAttrTimelineSemaphoreInteropSupported;
}
impl Default for cudaAccessPolicyWindow {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaAsyncNotificationInfo {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaAsyncNotificationInfo__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaChannelFormatDesc {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaChildGraphNodeParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaConditionalNodeParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaDeviceProp {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaEventRecordNodeParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaEventWaitNodeParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaExternalMemoryHandleDesc {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaExternalMemoryHandleDesc__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaExternalMemoryHandleDesc__bindgen_ty_1__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaExternalMemoryMipmappedArrayDesc {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaExternalSemaphoreHandleDesc {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaExternalSemaphoreHandleDesc__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaExternalSemaphoreHandleDesc__bindgen_ty_1__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaExternalSemaphoreSignalNodeParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaExternalSemaphoreSignalNodeParamsV2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaExternalSemaphoreSignalParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaExternalSemaphoreSignalParams__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaExternalSemaphoreSignalParams_v1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1__bindgen_ty_2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaExternalSemaphoreWaitNodeParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaExternalSemaphoreWaitNodeParamsV2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaExternalSemaphoreWaitParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaExternalSemaphoreWaitParams__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaExternalSemaphoreWaitParams_v1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1__bindgen_ty_2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
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
impl Default for cudaGraphExecUpdateResultInfo_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
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
impl Default for cudaGraphInstantiateParams_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaGraphKernelNodeUpdate {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaGraphKernelNodeUpdate__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaGraphKernelNodeUpdate__bindgen_ty_1__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaGraphNodeParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaGraphNodeParams__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaHostNodeParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaHostNodeParamsV2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaIpcEventHandle_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaIpcMemHandle_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070"
))]
impl Default for cudaKernelNodeAttrValue {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaKernelNodeParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaKernelNodeParamsV2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-11080",
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
impl Default for cudaLaunchAttributeValue {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-11080",
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
impl Default for cudaLaunchAttributeValue__bindgen_ty_2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060"
))]
impl Default for cudaLaunchAttributeValue__bindgen_ty_3 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaLaunchAttributeValue__bindgen_ty_4 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
impl Default for cudaLaunchAttributeValue__bindgen_ty_5 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-11080",
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
impl Default for cudaLaunchAttribute_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-11080",
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
impl Default for cudaLaunchConfig_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaLaunchParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaMemAccessDesc {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaMemAllocNodeParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaMemAllocNodeParamsV2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaMemFabricHandle_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaMemFreeNodeParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaMemLocation {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaMemPoolProps {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaMemPoolPtrExportData {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
impl Default for cudaMemcpy3DBatchOp {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
impl Default for cudaMemcpy3DOperand {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
impl Default for cudaMemcpy3DOperand__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
impl Default for cudaMemcpy3DOperand__bindgen_ty_1__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
impl Default for cudaMemcpy3DOperand__bindgen_ty_1__bindgen_ty_2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaMemcpy3DParms {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaMemcpy3DPeerParms {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
impl Default for cudaMemcpyAttributes {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaMemcpyNodeParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaMemsetParams {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12020",
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
impl Default for cudaMemsetParamsV2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaPitchedPtr {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaPointerAttributes {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaResourceDesc {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaResourceDesc__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaResourceDesc__bindgen_ty_1__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaResourceDesc__bindgen_ty_1__bindgen_ty_2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaResourceDesc__bindgen_ty_1__bindgen_ty_3 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaResourceDesc__bindgen_ty_1__bindgen_ty_4 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaResourceViewDesc {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070"
))]
impl Default for cudaStreamAttrValue {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for cudaTextureDesc {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-11080"))]
impl Default for cudaTextureDesc_v2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
impl Default for cudalibraryHostUniversalFunctionAndDataTable {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
))]
impl Default for surfaceReference {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070",
    feature = "cuda-11080"
))]
impl Default for textureReference {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaAsyncNotificationInfo__bindgen_ty_1 {
    pub overBudget: cudaAsyncNotificationInfo__bindgen_ty_1__bindgen_ty_1,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaExternalMemoryHandleDesc__bindgen_ty_1 {
    pub fd: ::core::ffi::c_int,
    pub win32: cudaExternalMemoryHandleDesc__bindgen_ty_1__bindgen_ty_1,
    pub nvSciBufObject: *const ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaExternalSemaphoreHandleDesc__bindgen_ty_1 {
    pub fd: ::core::ffi::c_int,
    pub win32: cudaExternalSemaphoreHandleDesc__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSyncObj: *const ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaExternalSemaphoreSignalParams__bindgen_ty_1__bindgen_ty_2 {
    pub fence: *mut ::core::ffi::c_void,
    pub reserved: ::core::ffi::c_ulonglong,
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaExternalSemaphoreSignalParams_v1__bindgen_ty_1__bindgen_ty_2 {
    pub fence: *mut ::core::ffi::c_void,
    pub reserved: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaExternalSemaphoreWaitParams__bindgen_ty_1__bindgen_ty_2 {
    pub fence: *mut ::core::ffi::c_void,
    pub reserved: ::core::ffi::c_ulonglong,
}
#[cfg(any(
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
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaExternalSemaphoreWaitParams_v1__bindgen_ty_1__bindgen_ty_2 {
    pub fence: *mut ::core::ffi::c_void,
    pub reserved: ::core::ffi::c_ulonglong,
}
#[cfg(any(
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaGraphKernelNodeUpdate__bindgen_ty_1 {
    pub gridDim: dim3,
    pub param: cudaGraphKernelNodeUpdate__bindgen_ty_1__bindgen_ty_1,
    pub isEnabled: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12020"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaGraphNodeParams__bindgen_ty_1 {
    pub reserved1: [::core::ffi::c_longlong; 29usize],
    pub kernel: cudaKernelNodeParamsV2,
    pub memcpy: cudaMemcpyNodeParams,
    pub memset: cudaMemsetParamsV2,
    pub host: cudaHostNodeParamsV2,
    pub graph: cudaChildGraphNodeParams,
    pub eventWait: cudaEventWaitNodeParams,
    pub eventRecord: cudaEventRecordNodeParams,
    pub extSemSignal: cudaExternalSemaphoreSignalNodeParamsV2,
    pub extSemWait: cudaExternalSemaphoreWaitNodeParamsV2,
    pub alloc: cudaMemAllocNodeParamsV2,
    pub free: cudaMemFreeNodeParams,
}
#[cfg(any(
    feature = "cuda-12030",
    feature = "cuda-12040",
    feature = "cuda-12050",
    feature = "cuda-12060",
    feature = "cuda-12080",
    feature = "cuda-12090"
))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaGraphNodeParams__bindgen_ty_1 {
    pub reserved1: [::core::ffi::c_longlong; 29usize],
    pub kernel: cudaKernelNodeParamsV2,
    pub memcpy: cudaMemcpyNodeParams,
    pub memset: cudaMemsetParamsV2,
    pub host: cudaHostNodeParamsV2,
    pub graph: cudaChildGraphNodeParams,
    pub eventWait: cudaEventWaitNodeParams,
    pub eventRecord: cudaEventRecordNodeParams,
    pub extSemSignal: cudaExternalSemaphoreSignalNodeParamsV2,
    pub extSemWait: cudaExternalSemaphoreWaitNodeParamsV2,
    pub alloc: cudaMemAllocNodeParamsV2,
    pub free: cudaMemFreeNodeParams,
    pub conditional: cudaConditionalNodeParams,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaKernelNodeAttrValue {
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-11070"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaKernelNodeAttrValue {
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub priority: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaLaunchAttributeValue {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: cudaSynchronizationPolicy,
    pub clusterDim: cudaLaunchAttributeValue__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: cudaClusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: cudaLaunchAttributeValue__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaLaunchAttributeValue {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: cudaSynchronizationPolicy,
    pub clusterDim: cudaLaunchAttributeValue__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: cudaClusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: cudaLaunchAttributeValue__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: cudaLaunchMemSyncDomainMap,
    pub memSyncDomain: cudaLaunchMemSyncDomain,
}
#[cfg(any(feature = "cuda-12030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaLaunchAttributeValue {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: cudaSynchronizationPolicy,
    pub clusterDim: cudaLaunchAttributeValue__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: cudaClusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: cudaLaunchAttributeValue__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: cudaLaunchMemSyncDomainMap,
    pub memSyncDomain: cudaLaunchMemSyncDomain,
    pub launchCompletionEvent: cudaLaunchAttributeValue__bindgen_ty_3,
}
#[cfg(any(feature = "cuda-12040"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaLaunchAttributeValue {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: cudaSynchronizationPolicy,
    pub clusterDim: cudaLaunchAttributeValue__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: cudaClusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: cudaLaunchAttributeValue__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: cudaLaunchMemSyncDomainMap,
    pub memSyncDomain: cudaLaunchMemSyncDomain,
    pub launchCompletionEvent: cudaLaunchAttributeValue__bindgen_ty_3,
    pub deviceUpdatableKernelNode: cudaLaunchAttributeValue__bindgen_ty_4,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaLaunchAttributeValue {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: cudaSynchronizationPolicy,
    pub clusterDim: cudaLaunchAttributeValue__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: cudaClusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: cudaLaunchAttributeValue__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: cudaLaunchMemSyncDomainMap,
    pub memSyncDomain: cudaLaunchMemSyncDomain,
    pub launchCompletionEvent: cudaLaunchAttributeValue__bindgen_ty_3,
    pub deviceUpdatableKernelNode: cudaLaunchAttributeValue__bindgen_ty_4,
    pub sharedMemCarveout: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaLaunchAttributeValue {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: cudaSynchronizationPolicy,
    pub clusterDim: cudaLaunchAttributeValue__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: cudaClusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: cudaLaunchAttributeValue__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: cudaLaunchMemSyncDomainMap,
    pub memSyncDomain: cudaLaunchMemSyncDomain,
    pub preferredClusterDim: cudaLaunchAttributeValue__bindgen_ty_3,
    pub launchCompletionEvent: cudaLaunchAttributeValue__bindgen_ty_4,
    pub deviceUpdatableKernelNode: cudaLaunchAttributeValue__bindgen_ty_5,
    pub sharedMemCarveout: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaMemcpy3DOperand__bindgen_ty_1 {
    pub ptr: cudaMemcpy3DOperand__bindgen_ty_1__bindgen_ty_1,
    pub array: cudaMemcpy3DOperand__bindgen_ty_1__bindgen_ty_2,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaResourceDesc__bindgen_ty_1 {
    pub array: cudaResourceDesc__bindgen_ty_1__bindgen_ty_1,
    pub mipmap: cudaResourceDesc__bindgen_ty_1__bindgen_ty_2,
    pub linear: cudaResourceDesc__bindgen_ty_1__bindgen_ty_3,
    pub pitch2D: cudaResourceDesc__bindgen_ty_1__bindgen_ty_4,
}
#[cfg(any(
    feature = "cuda-11040",
    feature = "cuda-11050",
    feature = "cuda-11060",
    feature = "cuda-11070"
))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union cudaStreamAttrValue {
    pub accessPolicyWindow: cudaAccessPolicyWindow,
    pub syncPolicy: cudaSynchronizationPolicy,
}
#[cfg(not(feature = "dynamic-loading"))]
extern "C" {
    pub fn cudaArrayGetInfo(
        desc: *mut cudaChannelFormatDesc,
        extent: *mut cudaExtent,
        flags: *mut ::core::ffi::c_uint,
        array: cudaArray_t,
    ) -> cudaError_t;
    #[cfg(any(
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
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaArrayGetMemoryRequirements(
        memoryRequirements: *mut cudaArrayMemoryRequirements,
        array: cudaArray_t,
        device: ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaArrayGetPlane(
        pPlaneArray: *mut cudaArray_t,
        hArray: cudaArray_t,
        planeIdx: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaArrayGetSparseProperties(
        sparseProperties: *mut cudaArraySparseProperties,
        array: cudaArray_t,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub fn cudaBindSurfaceToArray(
        surfref: *const surfaceReference,
        array: cudaArray_const_t,
        desc: *const cudaChannelFormatDesc,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub fn cudaBindTexture(
        offset: *mut usize,
        texref: *const textureReference,
        devPtr: *const ::core::ffi::c_void,
        desc: *const cudaChannelFormatDesc,
        size: usize,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub fn cudaBindTexture2D(
        offset: *mut usize,
        texref: *const textureReference,
        devPtr: *const ::core::ffi::c_void,
        desc: *const cudaChannelFormatDesc,
        width: usize,
        height: usize,
        pitch: usize,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub fn cudaBindTextureToArray(
        texref: *const textureReference,
        array: cudaArray_const_t,
        desc: *const cudaChannelFormatDesc,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub fn cudaBindTextureToMipmappedArray(
        texref: *const textureReference,
        mipmappedArray: cudaMipmappedArray_const_t,
        desc: *const cudaChannelFormatDesc,
    ) -> cudaError_t;
    pub fn cudaChooseDevice(
        device: *mut ::core::ffi::c_int,
        prop: *const cudaDeviceProp,
    ) -> cudaError_t;
    pub fn cudaCreateChannelDesc(
        x: ::core::ffi::c_int,
        y: ::core::ffi::c_int,
        z: ::core::ffi::c_int,
        w: ::core::ffi::c_int,
        f: cudaChannelFormatKind,
    ) -> cudaChannelFormatDesc;
    pub fn cudaCreateSurfaceObject(
        pSurfObject: *mut cudaSurfaceObject_t,
        pResDesc: *const cudaResourceDesc,
    ) -> cudaError_t;
    pub fn cudaCreateTextureObject(
        pTexObject: *mut cudaTextureObject_t,
        pResDesc: *const cudaResourceDesc,
        pTexDesc: *const cudaTextureDesc,
        pResViewDesc: *const cudaResourceViewDesc,
    ) -> cudaError_t;
    #[cfg(any(feature = "cuda-11080"))]
    pub fn cudaCreateTextureObject_v2(
        pTexObject: *mut cudaTextureObject_t,
        pResDesc: *const cudaResourceDesc,
        pTexDesc: *const cudaTextureDesc_v2,
        pResViewDesc: *const cudaResourceViewDesc,
    ) -> cudaError_t;
    pub fn cudaCtxResetPersistingL2Cache() -> cudaError_t;
    pub fn cudaDestroyExternalMemory(extMem: cudaExternalMemory_t) -> cudaError_t;
    pub fn cudaDestroyExternalSemaphore(extSem: cudaExternalSemaphore_t) -> cudaError_t;
    pub fn cudaDestroySurfaceObject(surfObject: cudaSurfaceObject_t) -> cudaError_t;
    pub fn cudaDestroyTextureObject(texObject: cudaTextureObject_t) -> cudaError_t;
    pub fn cudaDeviceCanAccessPeer(
        canAccessPeer: *mut ::core::ffi::c_int,
        device: ::core::ffi::c_int,
        peerDevice: ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaDeviceDisablePeerAccess(peerDevice: ::core::ffi::c_int) -> cudaError_t;
    pub fn cudaDeviceEnablePeerAccess(
        peerDevice: ::core::ffi::c_int,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaDeviceFlushGPUDirectRDMAWrites(
        target: cudaFlushGPUDirectRDMAWritesTarget,
        scope: cudaFlushGPUDirectRDMAWritesScope,
    ) -> cudaError_t;
    pub fn cudaDeviceGetAttribute(
        value: *mut ::core::ffi::c_int,
        attr: cudaDeviceAttr,
        device: ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaDeviceGetByPCIBusId(
        device: *mut ::core::ffi::c_int,
        pciBusId: *const ::core::ffi::c_char,
    ) -> cudaError_t;
    pub fn cudaDeviceGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t;
    pub fn cudaDeviceGetDefaultMemPool(
        memPool: *mut cudaMemPool_t,
        device: ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaDeviceGetGraphMemAttribute(
        device: ::core::ffi::c_int,
        attr: cudaGraphMemAttributeType,
        value: *mut ::core::ffi::c_void,
    ) -> cudaError_t;
    pub fn cudaDeviceGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t;
    pub fn cudaDeviceGetMemPool(
        memPool: *mut cudaMemPool_t,
        device: ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaDeviceGetP2PAttribute(
        value: *mut ::core::ffi::c_int,
        attr: cudaDeviceP2PAttr,
        srcDevice: ::core::ffi::c_int,
        dstDevice: ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaDeviceGetPCIBusId(
        pciBusId: *mut ::core::ffi::c_char,
        len: ::core::ffi::c_int,
        device: ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaDeviceGetSharedMemConfig(pConfig: *mut cudaSharedMemConfig) -> cudaError_t;
    pub fn cudaDeviceGetStreamPriorityRange(
        leastPriority: *mut ::core::ffi::c_int,
        greatestPriority: *mut ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaDeviceGetTexture1DLinearMaxWidth(
        maxWidthInElements: *mut usize,
        fmtDesc: *const cudaChannelFormatDesc,
        device: ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaDeviceGraphMemTrim(device: ::core::ffi::c_int) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaDeviceRegisterAsyncNotification(
        device: ::core::ffi::c_int,
        callbackFunc: cudaAsyncCallback,
        userData: *mut ::core::ffi::c_void,
        callback: *mut cudaAsyncCallbackHandle_t,
    ) -> cudaError_t;
    pub fn cudaDeviceReset() -> cudaError_t;
    pub fn cudaDeviceSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t;
    pub fn cudaDeviceSetGraphMemAttribute(
        device: ::core::ffi::c_int,
        attr: cudaGraphMemAttributeType,
        value: *mut ::core::ffi::c_void,
    ) -> cudaError_t;
    pub fn cudaDeviceSetLimit(limit: cudaLimit, value: usize) -> cudaError_t;
    pub fn cudaDeviceSetMemPool(device: ::core::ffi::c_int, memPool: cudaMemPool_t) -> cudaError_t;
    pub fn cudaDeviceSetSharedMemConfig(config: cudaSharedMemConfig) -> cudaError_t;
    pub fn cudaDeviceSynchronize() -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaDeviceUnregisterAsyncNotification(
        device: ::core::ffi::c_int,
        callback: cudaAsyncCallbackHandle_t,
    ) -> cudaError_t;
    pub fn cudaDriverGetVersion(driverVersion: *mut ::core::ffi::c_int) -> cudaError_t;
    pub fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t;
    pub fn cudaEventCreateWithFlags(
        event: *mut cudaEvent_t,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventElapsedTime(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t;
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn cudaEventElapsedTime_v2(
        ms: *mut f32,
        start: cudaEvent_t,
        end: cudaEvent_t,
    ) -> cudaError_t;
    pub fn cudaEventQuery(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaEventRecordWithFlags(
        event: cudaEvent_t,
        stream: cudaStream_t,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t;
    pub fn cudaExternalMemoryGetMappedBuffer(
        devPtr: *mut *mut ::core::ffi::c_void,
        extMem: cudaExternalMemory_t,
        bufferDesc: *const cudaExternalMemoryBufferDesc,
    ) -> cudaError_t;
    pub fn cudaExternalMemoryGetMappedMipmappedArray(
        mipmap: *mut cudaMipmappedArray_t,
        extMem: cudaExternalMemory_t,
        mipmapDesc: *const cudaExternalMemoryMipmappedArrayDesc,
    ) -> cudaError_t;
    pub fn cudaFree(devPtr: *mut ::core::ffi::c_void) -> cudaError_t;
    pub fn cudaFreeArray(array: cudaArray_t) -> cudaError_t;
    pub fn cudaFreeAsync(devPtr: *mut ::core::ffi::c_void, hStream: cudaStream_t) -> cudaError_t;
    pub fn cudaFreeHost(ptr: *mut ::core::ffi::c_void) -> cudaError_t;
    pub fn cudaFreeMipmappedArray(mipmappedArray: cudaMipmappedArray_t) -> cudaError_t;
    pub fn cudaFuncGetAttributes(
        attr: *mut cudaFuncAttributes,
        func: *const ::core::ffi::c_void,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaFuncGetName(
        name: *mut *const ::core::ffi::c_char,
        func: *const ::core::ffi::c_void,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaFuncGetParamInfo(
        func: *const ::core::ffi::c_void,
        paramIndex: usize,
        paramOffset: *mut usize,
        paramSize: *mut usize,
    ) -> cudaError_t;
    pub fn cudaFuncSetAttribute(
        func: *const ::core::ffi::c_void,
        attr: cudaFuncAttribute,
        value: ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaFuncSetCacheConfig(
        func: *const ::core::ffi::c_void,
        cacheConfig: cudaFuncCache,
    ) -> cudaError_t;
    pub fn cudaFuncSetSharedMemConfig(
        func: *const ::core::ffi::c_void,
        config: cudaSharedMemConfig,
    ) -> cudaError_t;
    pub fn cudaGetChannelDesc(
        desc: *mut cudaChannelFormatDesc,
        array: cudaArray_const_t,
    ) -> cudaError_t;
    pub fn cudaGetDevice(device: *mut ::core::ffi::c_int) -> cudaError_t;
    pub fn cudaGetDeviceCount(count: *mut ::core::ffi::c_int) -> cudaError_t;
    pub fn cudaGetDeviceFlags(flags: *mut ::core::ffi::c_uint) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub fn cudaGetDeviceProperties(
        prop: *mut cudaDeviceProp,
        device: ::core::ffi::c_int,
    ) -> cudaError_t;
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
    pub fn cudaGetDeviceProperties_v2(
        prop: *mut cudaDeviceProp,
        device: ::core::ffi::c_int,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub fn cudaGetDriverEntryPoint(
        symbol: *const ::core::ffi::c_char,
        funcPtr: *mut *mut ::core::ffi::c_void,
        flags: ::core::ffi::c_ulonglong,
    ) -> cudaError_t;
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
    pub fn cudaGetDriverEntryPoint(
        symbol: *const ::core::ffi::c_char,
        funcPtr: *mut *mut ::core::ffi::c_void,
        flags: ::core::ffi::c_ulonglong,
        driverStatus: *mut cudaDriverEntryPointQueryResult,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaGetDriverEntryPointByVersion(
        symbol: *const ::core::ffi::c_char,
        funcPtr: *mut *mut ::core::ffi::c_void,
        cudaVersion: ::core::ffi::c_uint,
        flags: ::core::ffi::c_ulonglong,
        driverStatus: *mut cudaDriverEntryPointQueryResult,
    ) -> cudaError_t;
    pub fn cudaGetErrorName(error: cudaError_t) -> *const ::core::ffi::c_char;
    pub fn cudaGetErrorString(error: cudaError_t) -> *const ::core::ffi::c_char;
    pub fn cudaGetExportTable(
        ppExportTable: *mut *const ::core::ffi::c_void,
        pExportTableId: *const cudaUUID_t,
    ) -> cudaError_t;
    pub fn cudaGetFuncBySymbol(
        functionPtr: *mut cudaFunction_t,
        symbolPtr: *const ::core::ffi::c_void,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaGetKernel(
        kernelPtr: *mut cudaKernel_t,
        entryFuncAddr: *const ::core::ffi::c_void,
    ) -> cudaError_t;
    pub fn cudaGetLastError() -> cudaError_t;
    pub fn cudaGetMipmappedArrayLevel(
        levelArray: *mut cudaArray_t,
        mipmappedArray: cudaMipmappedArray_const_t,
        level: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaGetSurfaceObjectResourceDesc(
        pResDesc: *mut cudaResourceDesc,
        surfObject: cudaSurfaceObject_t,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub fn cudaGetSurfaceReference(
        surfref: *mut *const surfaceReference,
        symbol: *const ::core::ffi::c_void,
    ) -> cudaError_t;
    pub fn cudaGetSymbolAddress(
        devPtr: *mut *mut ::core::ffi::c_void,
        symbol: *const ::core::ffi::c_void,
    ) -> cudaError_t;
    pub fn cudaGetSymbolSize(size: *mut usize, symbol: *const ::core::ffi::c_void) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub fn cudaGetTextureAlignmentOffset(
        offset: *mut usize,
        texref: *const textureReference,
    ) -> cudaError_t;
    pub fn cudaGetTextureObjectResourceDesc(
        pResDesc: *mut cudaResourceDesc,
        texObject: cudaTextureObject_t,
    ) -> cudaError_t;
    pub fn cudaGetTextureObjectResourceViewDesc(
        pResViewDesc: *mut cudaResourceViewDesc,
        texObject: cudaTextureObject_t,
    ) -> cudaError_t;
    pub fn cudaGetTextureObjectTextureDesc(
        pTexDesc: *mut cudaTextureDesc,
        texObject: cudaTextureObject_t,
    ) -> cudaError_t;
    #[cfg(any(feature = "cuda-11080"))]
    pub fn cudaGetTextureObjectTextureDesc_v2(
        pTexDesc: *mut cudaTextureDesc_v2,
        texObject: cudaTextureObject_t,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub fn cudaGetTextureReference(
        texref: *mut *const textureReference,
        symbol: *const ::core::ffi::c_void,
    ) -> cudaError_t;
    pub fn cudaGraphAddChildGraphNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        childGraph: cudaGraph_t,
    ) -> cudaError_t;
    pub fn cudaGraphAddDependencies(
        graph: cudaGraph_t,
        from: *const cudaGraphNode_t,
        to: *const cudaGraphNode_t,
        numDependencies: usize,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaGraphAddDependencies_v2(
        graph: cudaGraph_t,
        from: *const cudaGraphNode_t,
        to: *const cudaGraphNode_t,
        edgeData: *const cudaGraphEdgeData,
        numDependencies: usize,
    ) -> cudaError_t;
    pub fn cudaGraphAddEmptyNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
    ) -> cudaError_t;
    pub fn cudaGraphAddEventRecordNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        event: cudaEvent_t,
    ) -> cudaError_t;
    pub fn cudaGraphAddEventWaitNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        event: cudaEvent_t,
    ) -> cudaError_t;
    pub fn cudaGraphAddExternalSemaphoresSignalNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphAddExternalSemaphoresWaitNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphAddHostNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        pNodeParams: *const cudaHostNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphAddKernelNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        pNodeParams: *const cudaKernelNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphAddMemAllocNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        nodeParams: *mut cudaMemAllocNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphAddMemFreeNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        dptr: *mut ::core::ffi::c_void,
    ) -> cudaError_t;
    pub fn cudaGraphAddMemcpyNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        pCopyParams: *const cudaMemcpy3DParms,
    ) -> cudaError_t;
    pub fn cudaGraphAddMemcpyNode1D(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        dst: *mut ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaGraphAddMemcpyNodeFromSymbol(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        dst: *mut ::core::ffi::c_void,
        symbol: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaGraphAddMemcpyNodeToSymbol(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        symbol: *const ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaGraphAddMemsetNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        pMemsetParams: *const cudaMemsetParams,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaGraphAddNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        nodeParams: *mut cudaGraphNodeParams,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaGraphAddNode_v2(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        dependencyData: *const cudaGraphEdgeData,
        numDependencies: usize,
        nodeParams: *mut cudaGraphNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphChildGraphNodeGetGraph(
        node: cudaGraphNode_t,
        pGraph: *mut cudaGraph_t,
    ) -> cudaError_t;
    pub fn cudaGraphClone(pGraphClone: *mut cudaGraph_t, originalGraph: cudaGraph_t)
        -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaGraphConditionalHandleCreate(
        pHandle_out: *mut cudaGraphConditionalHandle,
        graph: cudaGraph_t,
        defaultLaunchValue: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaGraphCreate(pGraph: *mut cudaGraph_t, flags: ::core::ffi::c_uint) -> cudaError_t;
    pub fn cudaGraphDebugDotPrint(
        graph: cudaGraph_t,
        path: *const ::core::ffi::c_char,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t;
    pub fn cudaGraphDestroyNode(node: cudaGraphNode_t) -> cudaError_t;
    pub fn cudaGraphEventRecordNodeGetEvent(
        node: cudaGraphNode_t,
        event_out: *mut cudaEvent_t,
    ) -> cudaError_t;
    pub fn cudaGraphEventRecordNodeSetEvent(
        node: cudaGraphNode_t,
        event: cudaEvent_t,
    ) -> cudaError_t;
    pub fn cudaGraphEventWaitNodeGetEvent(
        node: cudaGraphNode_t,
        event_out: *mut cudaEvent_t,
    ) -> cudaError_t;
    pub fn cudaGraphEventWaitNodeSetEvent(node: cudaGraphNode_t, event: cudaEvent_t)
        -> cudaError_t;
    pub fn cudaGraphExecChildGraphNodeSetParams(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        childGraph: cudaGraph_t,
    ) -> cudaError_t;
    pub fn cudaGraphExecDestroy(graphExec: cudaGraphExec_t) -> cudaError_t;
    pub fn cudaGraphExecEventRecordNodeSetEvent(
        hGraphExec: cudaGraphExec_t,
        hNode: cudaGraphNode_t,
        event: cudaEvent_t,
    ) -> cudaError_t;
    pub fn cudaGraphExecEventWaitNodeSetEvent(
        hGraphExec: cudaGraphExec_t,
        hNode: cudaGraphNode_t,
        event: cudaEvent_t,
    ) -> cudaError_t;
    pub fn cudaGraphExecExternalSemaphoresSignalNodeSetParams(
        hGraphExec: cudaGraphExec_t,
        hNode: cudaGraphNode_t,
        nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphExecExternalSemaphoresWaitNodeSetParams(
        hGraphExec: cudaGraphExec_t,
        hNode: cudaGraphNode_t,
        nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
    ) -> cudaError_t;
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
    pub fn cudaGraphExecGetFlags(
        graphExec: cudaGraphExec_t,
        flags: *mut ::core::ffi::c_ulonglong,
    ) -> cudaError_t;
    pub fn cudaGraphExecHostNodeSetParams(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        pNodeParams: *const cudaHostNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphExecKernelNodeSetParams(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        pNodeParams: *const cudaKernelNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphExecMemcpyNodeSetParams(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        pNodeParams: *const cudaMemcpy3DParms,
    ) -> cudaError_t;
    pub fn cudaGraphExecMemcpyNodeSetParams1D(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        dst: *mut ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaGraphExecMemcpyNodeSetParamsFromSymbol(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        dst: *mut ::core::ffi::c_void,
        symbol: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaGraphExecMemcpyNodeSetParamsToSymbol(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        symbol: *const ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaGraphExecMemsetNodeSetParams(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        pNodeParams: *const cudaMemsetParams,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaGraphExecNodeSetParams(
        graphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        nodeParams: *mut cudaGraphNodeParams,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub fn cudaGraphExecUpdate(
        hGraphExec: cudaGraphExec_t,
        hGraph: cudaGraph_t,
        hErrorNode_out: *mut cudaGraphNode_t,
        updateResult_out: *mut cudaGraphExecUpdateResult,
    ) -> cudaError_t;
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
    pub fn cudaGraphExecUpdate(
        hGraphExec: cudaGraphExec_t,
        hGraph: cudaGraph_t,
        resultInfo: *mut cudaGraphExecUpdateResultInfo,
    ) -> cudaError_t;
    pub fn cudaGraphExternalSemaphoresSignalNodeGetParams(
        hNode: cudaGraphNode_t,
        params_out: *mut cudaExternalSemaphoreSignalNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphExternalSemaphoresSignalNodeSetParams(
        hNode: cudaGraphNode_t,
        nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphExternalSemaphoresWaitNodeGetParams(
        hNode: cudaGraphNode_t,
        params_out: *mut cudaExternalSemaphoreWaitNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphExternalSemaphoresWaitNodeSetParams(
        hNode: cudaGraphNode_t,
        nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphGetEdges(
        graph: cudaGraph_t,
        from: *mut cudaGraphNode_t,
        to: *mut cudaGraphNode_t,
        numEdges: *mut usize,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaGraphGetEdges_v2(
        graph: cudaGraph_t,
        from: *mut cudaGraphNode_t,
        to: *mut cudaGraphNode_t,
        edgeData: *mut cudaGraphEdgeData,
        numEdges: *mut usize,
    ) -> cudaError_t;
    pub fn cudaGraphGetNodes(
        graph: cudaGraph_t,
        nodes: *mut cudaGraphNode_t,
        numNodes: *mut usize,
    ) -> cudaError_t;
    pub fn cudaGraphGetRootNodes(
        graph: cudaGraph_t,
        pRootNodes: *mut cudaGraphNode_t,
        pNumRootNodes: *mut usize,
    ) -> cudaError_t;
    pub fn cudaGraphHostNodeGetParams(
        node: cudaGraphNode_t,
        pNodeParams: *mut cudaHostNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphHostNodeSetParams(
        node: cudaGraphNode_t,
        pNodeParams: *const cudaHostNodeParams,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub fn cudaGraphInstantiate(
        pGraphExec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        pErrorNode: *mut cudaGraphNode_t,
        pLogBuffer: *mut ::core::ffi::c_char,
        bufferSize: usize,
    ) -> cudaError_t;
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
    pub fn cudaGraphInstantiate(
        pGraphExec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        flags: ::core::ffi::c_ulonglong,
    ) -> cudaError_t;
    pub fn cudaGraphInstantiateWithFlags(
        pGraphExec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        flags: ::core::ffi::c_ulonglong,
    ) -> cudaError_t;
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
    pub fn cudaGraphInstantiateWithParams(
        pGraphExec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        instantiateParams: *mut cudaGraphInstantiateParams,
    ) -> cudaError_t;
    pub fn cudaGraphKernelNodeCopyAttributes(
        hSrc: cudaGraphNode_t,
        hDst: cudaGraphNode_t,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    ))]
    pub fn cudaGraphKernelNodeGetAttribute(
        hNode: cudaGraphNode_t,
        attr: cudaKernelNodeAttrID,
        value_out: *mut cudaKernelNodeAttrValue,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11080",
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
    pub fn cudaGraphKernelNodeGetAttribute(
        hNode: cudaGraphNode_t,
        attr: cudaLaunchAttributeID,
        value_out: *mut cudaLaunchAttributeValue,
    ) -> cudaError_t;
    pub fn cudaGraphKernelNodeGetParams(
        node: cudaGraphNode_t,
        pNodeParams: *mut cudaKernelNodeParams,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    ))]
    pub fn cudaGraphKernelNodeSetAttribute(
        hNode: cudaGraphNode_t,
        attr: cudaKernelNodeAttrID,
        value: *const cudaKernelNodeAttrValue,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11080",
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
    pub fn cudaGraphKernelNodeSetAttribute(
        hNode: cudaGraphNode_t,
        attr: cudaLaunchAttributeID,
        value: *const cudaLaunchAttributeValue,
    ) -> cudaError_t;
    pub fn cudaGraphKernelNodeSetParams(
        node: cudaGraphNode_t,
        pNodeParams: *const cudaKernelNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphLaunch(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaGraphMemAllocNodeGetParams(
        node: cudaGraphNode_t,
        params_out: *mut cudaMemAllocNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphMemFreeNodeGetParams(
        node: cudaGraphNode_t,
        dptr_out: *mut ::core::ffi::c_void,
    ) -> cudaError_t;
    pub fn cudaGraphMemcpyNodeGetParams(
        node: cudaGraphNode_t,
        pNodeParams: *mut cudaMemcpy3DParms,
    ) -> cudaError_t;
    pub fn cudaGraphMemcpyNodeSetParams(
        node: cudaGraphNode_t,
        pNodeParams: *const cudaMemcpy3DParms,
    ) -> cudaError_t;
    pub fn cudaGraphMemcpyNodeSetParams1D(
        node: cudaGraphNode_t,
        dst: *mut ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaGraphMemcpyNodeSetParamsFromSymbol(
        node: cudaGraphNode_t,
        dst: *mut ::core::ffi::c_void,
        symbol: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaGraphMemcpyNodeSetParamsToSymbol(
        node: cudaGraphNode_t,
        symbol: *const ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaGraphMemsetNodeGetParams(
        node: cudaGraphNode_t,
        pNodeParams: *mut cudaMemsetParams,
    ) -> cudaError_t;
    pub fn cudaGraphMemsetNodeSetParams(
        node: cudaGraphNode_t,
        pNodeParams: *const cudaMemsetParams,
    ) -> cudaError_t;
    pub fn cudaGraphNodeFindInClone(
        pNode: *mut cudaGraphNode_t,
        originalNode: cudaGraphNode_t,
        clonedGraph: cudaGraph_t,
    ) -> cudaError_t;
    pub fn cudaGraphNodeGetDependencies(
        node: cudaGraphNode_t,
        pDependencies: *mut cudaGraphNode_t,
        pNumDependencies: *mut usize,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaGraphNodeGetDependencies_v2(
        node: cudaGraphNode_t,
        pDependencies: *mut cudaGraphNode_t,
        edgeData: *mut cudaGraphEdgeData,
        pNumDependencies: *mut usize,
    ) -> cudaError_t;
    pub fn cudaGraphNodeGetDependentNodes(
        node: cudaGraphNode_t,
        pDependentNodes: *mut cudaGraphNode_t,
        pNumDependentNodes: *mut usize,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaGraphNodeGetDependentNodes_v2(
        node: cudaGraphNode_t,
        pDependentNodes: *mut cudaGraphNode_t,
        edgeData: *mut cudaGraphEdgeData,
        pNumDependentNodes: *mut usize,
    ) -> cudaError_t;
    #[cfg(any(
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
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaGraphNodeGetEnabled(
        hGraphExec: cudaGraphExec_t,
        hNode: cudaGraphNode_t,
        isEnabled: *mut ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaGraphNodeGetType(
        node: cudaGraphNode_t,
        pType: *mut cudaGraphNodeType,
    ) -> cudaError_t;
    #[cfg(any(
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
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaGraphNodeSetEnabled(
        hGraphExec: cudaGraphExec_t,
        hNode: cudaGraphNode_t,
        isEnabled: ::core::ffi::c_uint,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaGraphNodeSetParams(
        node: cudaGraphNode_t,
        nodeParams: *mut cudaGraphNodeParams,
    ) -> cudaError_t;
    pub fn cudaGraphReleaseUserObject(
        graph: cudaGraph_t,
        object: cudaUserObject_t,
        count: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaGraphRemoveDependencies(
        graph: cudaGraph_t,
        from: *const cudaGraphNode_t,
        to: *const cudaGraphNode_t,
        numDependencies: usize,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaGraphRemoveDependencies_v2(
        graph: cudaGraph_t,
        from: *const cudaGraphNode_t,
        to: *const cudaGraphNode_t,
        edgeData: *const cudaGraphEdgeData,
        numDependencies: usize,
    ) -> cudaError_t;
    pub fn cudaGraphRetainUserObject(
        graph: cudaGraph_t,
        object: cudaUserObject_t,
        count: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaGraphUpload(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t;
    pub fn cudaGraphicsMapResources(
        count: ::core::ffi::c_int,
        resources: *mut cudaGraphicsResource_t,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaGraphicsResourceGetMappedMipmappedArray(
        mipmappedArray: *mut cudaMipmappedArray_t,
        resource: cudaGraphicsResource_t,
    ) -> cudaError_t;
    pub fn cudaGraphicsResourceGetMappedPointer(
        devPtr: *mut *mut ::core::ffi::c_void,
        size: *mut usize,
        resource: cudaGraphicsResource_t,
    ) -> cudaError_t;
    pub fn cudaGraphicsResourceSetMapFlags(
        resource: cudaGraphicsResource_t,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaGraphicsSubResourceGetMappedArray(
        array: *mut cudaArray_t,
        resource: cudaGraphicsResource_t,
        arrayIndex: ::core::ffi::c_uint,
        mipLevel: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaGraphicsUnmapResources(
        count: ::core::ffi::c_int,
        resources: *mut cudaGraphicsResource_t,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaGraphicsUnregisterResource(resource: cudaGraphicsResource_t) -> cudaError_t;
    pub fn cudaHostAlloc(
        pHost: *mut *mut ::core::ffi::c_void,
        size: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaHostGetDevicePointer(
        pDevice: *mut *mut ::core::ffi::c_void,
        pHost: *mut ::core::ffi::c_void,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaHostGetFlags(
        pFlags: *mut ::core::ffi::c_uint,
        pHost: *mut ::core::ffi::c_void,
    ) -> cudaError_t;
    pub fn cudaHostRegister(
        ptr: *mut ::core::ffi::c_void,
        size: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaHostUnregister(ptr: *mut ::core::ffi::c_void) -> cudaError_t;
    pub fn cudaImportExternalMemory(
        extMem_out: *mut cudaExternalMemory_t,
        memHandleDesc: *const cudaExternalMemoryHandleDesc,
    ) -> cudaError_t;
    pub fn cudaImportExternalSemaphore(
        extSem_out: *mut cudaExternalSemaphore_t,
        semHandleDesc: *const cudaExternalSemaphoreHandleDesc,
    ) -> cudaError_t;
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
    pub fn cudaInitDevice(
        device: ::core::ffi::c_int,
        deviceFlags: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaIpcCloseMemHandle(devPtr: *mut ::core::ffi::c_void) -> cudaError_t;
    pub fn cudaIpcGetEventHandle(
        handle: *mut cudaIpcEventHandle_t,
        event: cudaEvent_t,
    ) -> cudaError_t;
    pub fn cudaIpcGetMemHandle(
        handle: *mut cudaIpcMemHandle_t,
        devPtr: *mut ::core::ffi::c_void,
    ) -> cudaError_t;
    pub fn cudaIpcOpenEventHandle(
        event: *mut cudaEvent_t,
        handle: cudaIpcEventHandle_t,
    ) -> cudaError_t;
    pub fn cudaIpcOpenMemHandle(
        devPtr: *mut *mut ::core::ffi::c_void,
        handle: cudaIpcMemHandle_t,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn cudaKernelSetAttributeForDevice(
        kernel: cudaKernel_t,
        attr: cudaFuncAttribute,
        value: ::core::ffi::c_int,
        device: ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaLaunchCooperativeKernel(
        func: *const ::core::ffi::c_void,
        gridDim: dim3,
        blockDim: dim3,
        args: *mut *mut ::core::ffi::c_void,
        sharedMem: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaLaunchCooperativeKernelMultiDevice(
        launchParamsList: *mut cudaLaunchParams,
        numDevices: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaLaunchHostFunc(
        stream: cudaStream_t,
        fn_: cudaHostFn_t,
        userData: *mut ::core::ffi::c_void,
    ) -> cudaError_t;
    pub fn cudaLaunchKernel(
        func: *const ::core::ffi::c_void,
        gridDim: dim3,
        blockDim: dim3,
        args: *mut *mut ::core::ffi::c_void,
        sharedMem: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11080",
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
    pub fn cudaLaunchKernelExC(
        config: *const cudaLaunchConfig_t,
        func: *const ::core::ffi::c_void,
        args: *mut *mut ::core::ffi::c_void,
    ) -> cudaError_t;
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn cudaLibraryEnumerateKernels(
        kernels: *mut cudaKernel_t,
        numKernels: ::core::ffi::c_uint,
        lib: cudaLibrary_t,
    ) -> cudaError_t;
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn cudaLibraryGetGlobal(
        dptr: *mut *mut ::core::ffi::c_void,
        bytes: *mut usize,
        library: cudaLibrary_t,
        name: *const ::core::ffi::c_char,
    ) -> cudaError_t;
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn cudaLibraryGetKernel(
        pKernel: *mut cudaKernel_t,
        library: cudaLibrary_t,
        name: *const ::core::ffi::c_char,
    ) -> cudaError_t;
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn cudaLibraryGetKernelCount(
        count: *mut ::core::ffi::c_uint,
        lib: cudaLibrary_t,
    ) -> cudaError_t;
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn cudaLibraryGetManaged(
        dptr: *mut *mut ::core::ffi::c_void,
        bytes: *mut usize,
        library: cudaLibrary_t,
        name: *const ::core::ffi::c_char,
    ) -> cudaError_t;
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn cudaLibraryGetUnifiedFunction(
        fptr: *mut *mut ::core::ffi::c_void,
        library: cudaLibrary_t,
        symbol: *const ::core::ffi::c_char,
    ) -> cudaError_t;
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn cudaLibraryLoadData(
        library: *mut cudaLibrary_t,
        code: *const ::core::ffi::c_void,
        jitOptions: *mut cudaJitOption,
        jitOptionsValues: *mut *mut ::core::ffi::c_void,
        numJitOptions: ::core::ffi::c_uint,
        libraryOptions: *mut cudaLibraryOption,
        libraryOptionValues: *mut *mut ::core::ffi::c_void,
        numLibraryOptions: ::core::ffi::c_uint,
    ) -> cudaError_t;
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn cudaLibraryLoadFromFile(
        library: *mut cudaLibrary_t,
        fileName: *const ::core::ffi::c_char,
        jitOptions: *mut cudaJitOption,
        jitOptionsValues: *mut *mut ::core::ffi::c_void,
        numJitOptions: ::core::ffi::c_uint,
        libraryOptions: *mut cudaLibraryOption,
        libraryOptionValues: *mut *mut ::core::ffi::c_void,
        numLibraryOptions: ::core::ffi::c_uint,
    ) -> cudaError_t;
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn cudaLibraryUnload(library: cudaLibrary_t) -> cudaError_t;
    pub fn cudaMalloc(devPtr: *mut *mut ::core::ffi::c_void, size: usize) -> cudaError_t;
    pub fn cudaMalloc3D(pitchedDevPtr: *mut cudaPitchedPtr, extent: cudaExtent) -> cudaError_t;
    pub fn cudaMalloc3DArray(
        array: *mut cudaArray_t,
        desc: *const cudaChannelFormatDesc,
        extent: cudaExtent,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaMallocArray(
        array: *mut cudaArray_t,
        desc: *const cudaChannelFormatDesc,
        width: usize,
        height: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaMallocAsync(
        devPtr: *mut *mut ::core::ffi::c_void,
        size: usize,
        hStream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMallocFromPoolAsync(
        ptr: *mut *mut ::core::ffi::c_void,
        size: usize,
        memPool: cudaMemPool_t,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMallocHost(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> cudaError_t;
    pub fn cudaMallocManaged(
        devPtr: *mut *mut ::core::ffi::c_void,
        size: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaMallocMipmappedArray(
        mipmappedArray: *mut cudaMipmappedArray_t,
        desc: *const cudaChannelFormatDesc,
        extent: cudaExtent,
        numLevels: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaMallocPitch(
        devPtr: *mut *mut ::core::ffi::c_void,
        pitch: *mut usize,
        width: usize,
        height: usize,
    ) -> cudaError_t;
    pub fn cudaMemAdvise(
        devPtr: *const ::core::ffi::c_void,
        count: usize,
        advice: cudaMemoryAdvise,
        device: ::core::ffi::c_int,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaMemAdvise_v2(
        devPtr: *const ::core::ffi::c_void,
        count: usize,
        advice: cudaMemoryAdvise,
        location: cudaMemLocation,
    ) -> cudaError_t;
    pub fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t;
    pub fn cudaMemPoolCreate(
        memPool: *mut cudaMemPool_t,
        poolProps: *const cudaMemPoolProps,
    ) -> cudaError_t;
    pub fn cudaMemPoolDestroy(memPool: cudaMemPool_t) -> cudaError_t;
    pub fn cudaMemPoolExportPointer(
        exportData: *mut cudaMemPoolPtrExportData,
        ptr: *mut ::core::ffi::c_void,
    ) -> cudaError_t;
    pub fn cudaMemPoolExportToShareableHandle(
        shareableHandle: *mut ::core::ffi::c_void,
        memPool: cudaMemPool_t,
        handleType: cudaMemAllocationHandleType,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaMemPoolGetAccess(
        flags: *mut cudaMemAccessFlags,
        memPool: cudaMemPool_t,
        location: *mut cudaMemLocation,
    ) -> cudaError_t;
    pub fn cudaMemPoolGetAttribute(
        memPool: cudaMemPool_t,
        attr: cudaMemPoolAttr,
        value: *mut ::core::ffi::c_void,
    ) -> cudaError_t;
    pub fn cudaMemPoolImportFromShareableHandle(
        memPool: *mut cudaMemPool_t,
        shareableHandle: *mut ::core::ffi::c_void,
        handleType: cudaMemAllocationHandleType,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaMemPoolImportPointer(
        ptr: *mut *mut ::core::ffi::c_void,
        memPool: cudaMemPool_t,
        exportData: *mut cudaMemPoolPtrExportData,
    ) -> cudaError_t;
    pub fn cudaMemPoolSetAccess(
        memPool: cudaMemPool_t,
        descList: *const cudaMemAccessDesc,
        count: usize,
    ) -> cudaError_t;
    pub fn cudaMemPoolSetAttribute(
        memPool: cudaMemPool_t,
        attr: cudaMemPoolAttr,
        value: *mut ::core::ffi::c_void,
    ) -> cudaError_t;
    pub fn cudaMemPoolTrimTo(memPool: cudaMemPool_t, minBytesToKeep: usize) -> cudaError_t;
    pub fn cudaMemPrefetchAsync(
        devPtr: *const ::core::ffi::c_void,
        count: usize,
        dstDevice: ::core::ffi::c_int,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaMemPrefetchAsync_v2(
        devPtr: *const ::core::ffi::c_void,
        count: usize,
        location: cudaMemLocation,
        flags: ::core::ffi::c_uint,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemRangeGetAttribute(
        data: *mut ::core::ffi::c_void,
        dataSize: usize,
        attribute: cudaMemRangeAttribute,
        devPtr: *const ::core::ffi::c_void,
        count: usize,
    ) -> cudaError_t;
    pub fn cudaMemRangeGetAttributes(
        data: *mut *mut ::core::ffi::c_void,
        dataSizes: *mut usize,
        attributes: *mut cudaMemRangeAttribute,
        numAttributes: usize,
        devPtr: *const ::core::ffi::c_void,
        count: usize,
    ) -> cudaError_t;
    pub fn cudaMemcpy(
        dst: *mut ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemcpy2D(
        dst: *mut ::core::ffi::c_void,
        dpitch: usize,
        src: *const ::core::ffi::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemcpy2DArrayToArray(
        dst: cudaArray_t,
        wOffsetDst: usize,
        hOffsetDst: usize,
        src: cudaArray_const_t,
        wOffsetSrc: usize,
        hOffsetSrc: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemcpy2DAsync(
        dst: *mut ::core::ffi::c_void,
        dpitch: usize,
        src: *const ::core::ffi::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemcpy2DFromArray(
        dst: *mut ::core::ffi::c_void,
        dpitch: usize,
        src: cudaArray_const_t,
        wOffset: usize,
        hOffset: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemcpy2DFromArrayAsync(
        dst: *mut ::core::ffi::c_void,
        dpitch: usize,
        src: cudaArray_const_t,
        wOffset: usize,
        hOffset: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemcpy2DToArray(
        dst: cudaArray_t,
        wOffset: usize,
        hOffset: usize,
        src: *const ::core::ffi::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemcpy2DToArrayAsync(
        dst: cudaArray_t,
        wOffset: usize,
        hOffset: usize,
        src: *const ::core::ffi::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemcpy3D(p: *const cudaMemcpy3DParms) -> cudaError_t;
    pub fn cudaMemcpy3DAsync(p: *const cudaMemcpy3DParms, stream: cudaStream_t) -> cudaError_t;
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn cudaMemcpy3DBatchAsync(
        numOps: usize,
        opList: *mut cudaMemcpy3DBatchOp,
        failIdx: *mut usize,
        flags: ::core::ffi::c_ulonglong,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemcpy3DPeer(p: *const cudaMemcpy3DPeerParms) -> cudaError_t;
    pub fn cudaMemcpy3DPeerAsync(
        p: *const cudaMemcpy3DPeerParms,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemcpyArrayToArray(
        dst: cudaArray_t,
        wOffsetDst: usize,
        hOffsetDst: usize,
        src: cudaArray_const_t,
        wOffsetSrc: usize,
        hOffsetSrc: usize,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemcpyAsync(
        dst: *mut ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn cudaMemcpyBatchAsync(
        dsts: *mut *mut ::core::ffi::c_void,
        srcs: *mut *mut ::core::ffi::c_void,
        sizes: *mut usize,
        count: usize,
        attrs: *mut cudaMemcpyAttributes,
        attrsIdxs: *mut usize,
        numAttrs: usize,
        failIdx: *mut usize,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemcpyFromArray(
        dst: *mut ::core::ffi::c_void,
        src: cudaArray_const_t,
        wOffset: usize,
        hOffset: usize,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemcpyFromArrayAsync(
        dst: *mut ::core::ffi::c_void,
        src: cudaArray_const_t,
        wOffset: usize,
        hOffset: usize,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemcpyFromSymbol(
        dst: *mut ::core::ffi::c_void,
        symbol: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemcpyFromSymbolAsync(
        dst: *mut ::core::ffi::c_void,
        symbol: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemcpyPeer(
        dst: *mut ::core::ffi::c_void,
        dstDevice: ::core::ffi::c_int,
        src: *const ::core::ffi::c_void,
        srcDevice: ::core::ffi::c_int,
        count: usize,
    ) -> cudaError_t;
    pub fn cudaMemcpyPeerAsync(
        dst: *mut ::core::ffi::c_void,
        dstDevice: ::core::ffi::c_int,
        src: *const ::core::ffi::c_void,
        srcDevice: ::core::ffi::c_int,
        count: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemcpyToArray(
        dst: cudaArray_t,
        wOffset: usize,
        hOffset: usize,
        src: *const ::core::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemcpyToArrayAsync(
        dst: cudaArray_t,
        wOffset: usize,
        hOffset: usize,
        src: *const ::core::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemcpyToSymbol(
        symbol: *const ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t;
    pub fn cudaMemcpyToSymbolAsync(
        symbol: *const ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemset(
        devPtr: *mut ::core::ffi::c_void,
        value: ::core::ffi::c_int,
        count: usize,
    ) -> cudaError_t;
    pub fn cudaMemset2D(
        devPtr: *mut ::core::ffi::c_void,
        pitch: usize,
        value: ::core::ffi::c_int,
        width: usize,
        height: usize,
    ) -> cudaError_t;
    pub fn cudaMemset2DAsync(
        devPtr: *mut ::core::ffi::c_void,
        pitch: usize,
        value: ::core::ffi::c_int,
        width: usize,
        height: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemset3D(
        pitchedDevPtr: cudaPitchedPtr,
        value: ::core::ffi::c_int,
        extent: cudaExtent,
    ) -> cudaError_t;
    pub fn cudaMemset3DAsync(
        pitchedDevPtr: cudaPitchedPtr,
        value: ::core::ffi::c_int,
        extent: cudaExtent,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaMemsetAsync(
        devPtr: *mut ::core::ffi::c_void,
        value: ::core::ffi::c_int,
        count: usize,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[cfg(any(
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
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaMipmappedArrayGetMemoryRequirements(
        memoryRequirements: *mut cudaArrayMemoryRequirements,
        mipmap: cudaMipmappedArray_t,
        device: ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaMipmappedArrayGetSparseProperties(
        sparseProperties: *mut cudaArraySparseProperties,
        mipmap: cudaMipmappedArray_t,
    ) -> cudaError_t;
    pub fn cudaOccupancyAvailableDynamicSMemPerBlock(
        dynamicSmemSize: *mut usize,
        func: *const ::core::ffi::c_void,
        numBlocks: ::core::ffi::c_int,
        blockSize: ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks: *mut ::core::ffi::c_int,
        func: *const ::core::ffi::c_void,
        blockSize: ::core::ffi::c_int,
        dynamicSMemSize: usize,
    ) -> cudaError_t;
    pub fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks: *mut ::core::ffi::c_int,
        func: *const ::core::ffi::c_void,
        blockSize: ::core::ffi::c_int,
        dynamicSMemSize: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11080",
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
    pub fn cudaOccupancyMaxActiveClusters(
        numClusters: *mut ::core::ffi::c_int,
        func: *const ::core::ffi::c_void,
        launchConfig: *const cudaLaunchConfig_t,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11080",
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
    pub fn cudaOccupancyMaxPotentialClusterSize(
        clusterSize: *mut ::core::ffi::c_int,
        func: *const ::core::ffi::c_void,
        launchConfig: *const cudaLaunchConfig_t,
    ) -> cudaError_t;
    pub fn cudaPeekAtLastError() -> cudaError_t;
    pub fn cudaPointerGetAttributes(
        attributes: *mut cudaPointerAttributes,
        ptr: *const ::core::ffi::c_void,
    ) -> cudaError_t;
    pub fn cudaProfilerStop() -> cudaError_t;
    pub fn cudaRuntimeGetVersion(runtimeVersion: *mut ::core::ffi::c_int) -> cudaError_t;
    pub fn cudaSetDevice(device: ::core::ffi::c_int) -> cudaError_t;
    pub fn cudaSetDeviceFlags(flags: ::core::ffi::c_uint) -> cudaError_t;
    pub fn cudaSetDoubleForDevice(d: *mut f64) -> cudaError_t;
    pub fn cudaSetDoubleForHost(d: *mut f64) -> cudaError_t;
    pub fn cudaSetValidDevices(
        device_arr: *mut ::core::ffi::c_int,
        len: ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaSignalExternalSemaphoresAsync_v2(
        extSemArray: *const cudaExternalSemaphore_t,
        paramsArray: *const cudaExternalSemaphoreSignalParams,
        numExtSems: ::core::ffi::c_uint,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub fn cudaStreamAddCallback(
        stream: cudaStream_t,
        callback: cudaStreamCallback_t,
        userData: *mut ::core::ffi::c_void,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaStreamAttachMemAsync(
        stream: cudaStream_t,
        devPtr: *mut ::core::ffi::c_void,
        length: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaStreamBeginCapture(stream: cudaStream_t, mode: cudaStreamCaptureMode)
        -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaStreamBeginCaptureToGraph(
        stream: cudaStream_t,
        graph: cudaGraph_t,
        dependencies: *const cudaGraphNode_t,
        dependencyData: *const cudaGraphEdgeData,
        numDependencies: usize,
        mode: cudaStreamCaptureMode,
    ) -> cudaError_t;
    pub fn cudaStreamCopyAttributes(dst: cudaStream_t, src: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t;
    pub fn cudaStreamCreateWithFlags(
        pStream: *mut cudaStream_t,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaStreamCreateWithPriority(
        pStream: *mut cudaStream_t,
        flags: ::core::ffi::c_uint,
        priority: ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamEndCapture(stream: cudaStream_t, pGraph: *mut cudaGraph_t) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    ))]
    pub fn cudaStreamGetAttribute(
        hStream: cudaStream_t,
        attr: cudaStreamAttrID,
        value_out: *mut cudaStreamAttrValue,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11080",
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
    pub fn cudaStreamGetAttribute(
        hStream: cudaStream_t,
        attr: cudaLaunchAttributeID,
        value_out: *mut cudaLaunchAttributeValue,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub fn cudaStreamGetCaptureInfo(
        stream: cudaStream_t,
        pCaptureStatus: *mut cudaStreamCaptureStatus,
        pId: *mut ::core::ffi::c_ulonglong,
    ) -> cudaError_t;
    pub fn cudaStreamGetCaptureInfo_v2(
        stream: cudaStream_t,
        captureStatus_out: *mut cudaStreamCaptureStatus,
        id_out: *mut ::core::ffi::c_ulonglong,
        graph_out: *mut cudaGraph_t,
        dependencies_out: *mut *const cudaGraphNode_t,
        numDependencies_out: *mut usize,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaStreamGetCaptureInfo_v3(
        stream: cudaStream_t,
        captureStatus_out: *mut cudaStreamCaptureStatus,
        id_out: *mut ::core::ffi::c_ulonglong,
        graph_out: *mut cudaGraph_t,
        dependencies_out: *mut *const cudaGraphNode_t,
        edgeData_out: *mut *const cudaGraphEdgeData,
        numDependencies_out: *mut usize,
    ) -> cudaError_t;
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub fn cudaStreamGetDevice(
        hStream: cudaStream_t,
        device: *mut ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaStreamGetFlags(
        hStream: cudaStream_t,
        flags: *mut ::core::ffi::c_uint,
    ) -> cudaError_t;
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
    pub fn cudaStreamGetId(
        hStream: cudaStream_t,
        streamId: *mut ::core::ffi::c_ulonglong,
    ) -> cudaError_t;
    pub fn cudaStreamGetPriority(
        hStream: cudaStream_t,
        priority: *mut ::core::ffi::c_int,
    ) -> cudaError_t;
    pub fn cudaStreamIsCapturing(
        stream: cudaStream_t,
        pCaptureStatus: *mut cudaStreamCaptureStatus,
    ) -> cudaError_t;
    pub fn cudaStreamQuery(stream: cudaStream_t) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    ))]
    pub fn cudaStreamSetAttribute(
        hStream: cudaStream_t,
        attr: cudaStreamAttrID,
        value: *const cudaStreamAttrValue,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11080",
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
    pub fn cudaStreamSetAttribute(
        hStream: cudaStream_t,
        attr: cudaLaunchAttributeID,
        value: *const cudaLaunchAttributeValue,
    ) -> cudaError_t;
    pub fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t;
    pub fn cudaStreamUpdateCaptureDependencies(
        stream: cudaStream_t,
        dependencies: *mut cudaGraphNode_t,
        numDependencies: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub fn cudaStreamUpdateCaptureDependencies_v2(
        stream: cudaStream_t,
        dependencies: *mut cudaGraphNode_t,
        dependencyData: *const cudaGraphEdgeData,
        numDependencies: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaStreamWaitEvent(
        stream: cudaStream_t,
        event: cudaEvent_t,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaThreadExchangeStreamCaptureMode(mode: *mut cudaStreamCaptureMode) -> cudaError_t;
    pub fn cudaThreadExit() -> cudaError_t;
    pub fn cudaThreadGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t;
    pub fn cudaThreadGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t;
    pub fn cudaThreadSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t;
    pub fn cudaThreadSetLimit(limit: cudaLimit, value: usize) -> cudaError_t;
    pub fn cudaThreadSynchronize() -> cudaError_t;
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub fn cudaUnbindTexture(texref: *const textureReference) -> cudaError_t;
    pub fn cudaUserObjectCreate(
        object_out: *mut cudaUserObject_t,
        ptr: *mut ::core::ffi::c_void,
        destroy: cudaHostFn_t,
        initialRefcount: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaUserObjectRelease(
        object: cudaUserObject_t,
        count: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaUserObjectRetain(
        object: cudaUserObject_t,
        count: ::core::ffi::c_uint,
    ) -> cudaError_t;
    pub fn cudaWaitExternalSemaphoresAsync_v2(
        extSemArray: *const cudaExternalSemaphore_t,
        paramsArray: *const cudaExternalSemaphoreWaitParams,
        numExtSems: ::core::ffi::c_uint,
        stream: cudaStream_t,
    ) -> cudaError_t;
}
#[cfg(feature = "dynamic-loading")]
mod loaded {
    use super::*;
    pub unsafe fn cudaArrayGetInfo(
        desc: *mut cudaChannelFormatDesc,
        extent: *mut cudaExtent,
        flags: *mut ::core::ffi::c_uint,
        array: cudaArray_t,
    ) -> cudaError_t {
        (culib().cudaArrayGetInfo)(desc, extent, flags, array)
    }
    #[cfg(any(
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
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaArrayGetMemoryRequirements(
        memoryRequirements: *mut cudaArrayMemoryRequirements,
        array: cudaArray_t,
        device: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaArrayGetMemoryRequirements)(memoryRequirements, array, device)
    }
    pub unsafe fn cudaArrayGetPlane(
        pPlaneArray: *mut cudaArray_t,
        hArray: cudaArray_t,
        planeIdx: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaArrayGetPlane)(pPlaneArray, hArray, planeIdx)
    }
    pub unsafe fn cudaArrayGetSparseProperties(
        sparseProperties: *mut cudaArraySparseProperties,
        array: cudaArray_t,
    ) -> cudaError_t {
        (culib().cudaArrayGetSparseProperties)(sparseProperties, array)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn cudaBindSurfaceToArray(
        surfref: *const surfaceReference,
        array: cudaArray_const_t,
        desc: *const cudaChannelFormatDesc,
    ) -> cudaError_t {
        (culib().cudaBindSurfaceToArray)(surfref, array, desc)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn cudaBindTexture(
        offset: *mut usize,
        texref: *const textureReference,
        devPtr: *const ::core::ffi::c_void,
        desc: *const cudaChannelFormatDesc,
        size: usize,
    ) -> cudaError_t {
        (culib().cudaBindTexture)(offset, texref, devPtr, desc, size)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn cudaBindTexture2D(
        offset: *mut usize,
        texref: *const textureReference,
        devPtr: *const ::core::ffi::c_void,
        desc: *const cudaChannelFormatDesc,
        width: usize,
        height: usize,
        pitch: usize,
    ) -> cudaError_t {
        (culib().cudaBindTexture2D)(offset, texref, devPtr, desc, width, height, pitch)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn cudaBindTextureToArray(
        texref: *const textureReference,
        array: cudaArray_const_t,
        desc: *const cudaChannelFormatDesc,
    ) -> cudaError_t {
        (culib().cudaBindTextureToArray)(texref, array, desc)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn cudaBindTextureToMipmappedArray(
        texref: *const textureReference,
        mipmappedArray: cudaMipmappedArray_const_t,
        desc: *const cudaChannelFormatDesc,
    ) -> cudaError_t {
        (culib().cudaBindTextureToMipmappedArray)(texref, mipmappedArray, desc)
    }
    pub unsafe fn cudaChooseDevice(
        device: *mut ::core::ffi::c_int,
        prop: *const cudaDeviceProp,
    ) -> cudaError_t {
        (culib().cudaChooseDevice)(device, prop)
    }
    pub unsafe fn cudaCreateChannelDesc(
        x: ::core::ffi::c_int,
        y: ::core::ffi::c_int,
        z: ::core::ffi::c_int,
        w: ::core::ffi::c_int,
        f: cudaChannelFormatKind,
    ) -> cudaChannelFormatDesc {
        (culib().cudaCreateChannelDesc)(x, y, z, w, f)
    }
    pub unsafe fn cudaCreateSurfaceObject(
        pSurfObject: *mut cudaSurfaceObject_t,
        pResDesc: *const cudaResourceDesc,
    ) -> cudaError_t {
        (culib().cudaCreateSurfaceObject)(pSurfObject, pResDesc)
    }
    pub unsafe fn cudaCreateTextureObject(
        pTexObject: *mut cudaTextureObject_t,
        pResDesc: *const cudaResourceDesc,
        pTexDesc: *const cudaTextureDesc,
        pResViewDesc: *const cudaResourceViewDesc,
    ) -> cudaError_t {
        (culib().cudaCreateTextureObject)(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    }
    #[cfg(any(feature = "cuda-11080"))]
    pub unsafe fn cudaCreateTextureObject_v2(
        pTexObject: *mut cudaTextureObject_t,
        pResDesc: *const cudaResourceDesc,
        pTexDesc: *const cudaTextureDesc_v2,
        pResViewDesc: *const cudaResourceViewDesc,
    ) -> cudaError_t {
        (culib().cudaCreateTextureObject_v2)(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    }
    pub unsafe fn cudaCtxResetPersistingL2Cache() -> cudaError_t {
        (culib().cudaCtxResetPersistingL2Cache)()
    }
    pub unsafe fn cudaDestroyExternalMemory(extMem: cudaExternalMemory_t) -> cudaError_t {
        (culib().cudaDestroyExternalMemory)(extMem)
    }
    pub unsafe fn cudaDestroyExternalSemaphore(extSem: cudaExternalSemaphore_t) -> cudaError_t {
        (culib().cudaDestroyExternalSemaphore)(extSem)
    }
    pub unsafe fn cudaDestroySurfaceObject(surfObject: cudaSurfaceObject_t) -> cudaError_t {
        (culib().cudaDestroySurfaceObject)(surfObject)
    }
    pub unsafe fn cudaDestroyTextureObject(texObject: cudaTextureObject_t) -> cudaError_t {
        (culib().cudaDestroyTextureObject)(texObject)
    }
    pub unsafe fn cudaDeviceCanAccessPeer(
        canAccessPeer: *mut ::core::ffi::c_int,
        device: ::core::ffi::c_int,
        peerDevice: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaDeviceCanAccessPeer)(canAccessPeer, device, peerDevice)
    }
    pub unsafe fn cudaDeviceDisablePeerAccess(peerDevice: ::core::ffi::c_int) -> cudaError_t {
        (culib().cudaDeviceDisablePeerAccess)(peerDevice)
    }
    pub unsafe fn cudaDeviceEnablePeerAccess(
        peerDevice: ::core::ffi::c_int,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaDeviceEnablePeerAccess)(peerDevice, flags)
    }
    pub unsafe fn cudaDeviceFlushGPUDirectRDMAWrites(
        target: cudaFlushGPUDirectRDMAWritesTarget,
        scope: cudaFlushGPUDirectRDMAWritesScope,
    ) -> cudaError_t {
        (culib().cudaDeviceFlushGPUDirectRDMAWrites)(target, scope)
    }
    pub unsafe fn cudaDeviceGetAttribute(
        value: *mut ::core::ffi::c_int,
        attr: cudaDeviceAttr,
        device: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaDeviceGetAttribute)(value, attr, device)
    }
    pub unsafe fn cudaDeviceGetByPCIBusId(
        device: *mut ::core::ffi::c_int,
        pciBusId: *const ::core::ffi::c_char,
    ) -> cudaError_t {
        (culib().cudaDeviceGetByPCIBusId)(device, pciBusId)
    }
    pub unsafe fn cudaDeviceGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t {
        (culib().cudaDeviceGetCacheConfig)(pCacheConfig)
    }
    pub unsafe fn cudaDeviceGetDefaultMemPool(
        memPool: *mut cudaMemPool_t,
        device: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaDeviceGetDefaultMemPool)(memPool, device)
    }
    pub unsafe fn cudaDeviceGetGraphMemAttribute(
        device: ::core::ffi::c_int,
        attr: cudaGraphMemAttributeType,
        value: *mut ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaDeviceGetGraphMemAttribute)(device, attr, value)
    }
    pub unsafe fn cudaDeviceGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t {
        (culib().cudaDeviceGetLimit)(pValue, limit)
    }
    pub unsafe fn cudaDeviceGetMemPool(
        memPool: *mut cudaMemPool_t,
        device: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaDeviceGetMemPool)(memPool, device)
    }
    pub unsafe fn cudaDeviceGetP2PAttribute(
        value: *mut ::core::ffi::c_int,
        attr: cudaDeviceP2PAttr,
        srcDevice: ::core::ffi::c_int,
        dstDevice: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaDeviceGetP2PAttribute)(value, attr, srcDevice, dstDevice)
    }
    pub unsafe fn cudaDeviceGetPCIBusId(
        pciBusId: *mut ::core::ffi::c_char,
        len: ::core::ffi::c_int,
        device: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaDeviceGetPCIBusId)(pciBusId, len, device)
    }
    pub unsafe fn cudaDeviceGetSharedMemConfig(pConfig: *mut cudaSharedMemConfig) -> cudaError_t {
        (culib().cudaDeviceGetSharedMemConfig)(pConfig)
    }
    pub unsafe fn cudaDeviceGetStreamPriorityRange(
        leastPriority: *mut ::core::ffi::c_int,
        greatestPriority: *mut ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaDeviceGetStreamPriorityRange)(leastPriority, greatestPriority)
    }
    pub unsafe fn cudaDeviceGetTexture1DLinearMaxWidth(
        maxWidthInElements: *mut usize,
        fmtDesc: *const cudaChannelFormatDesc,
        device: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaDeviceGetTexture1DLinearMaxWidth)(maxWidthInElements, fmtDesc, device)
    }
    pub unsafe fn cudaDeviceGraphMemTrim(device: ::core::ffi::c_int) -> cudaError_t {
        (culib().cudaDeviceGraphMemTrim)(device)
    }
    #[cfg(any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaDeviceRegisterAsyncNotification(
        device: ::core::ffi::c_int,
        callbackFunc: cudaAsyncCallback,
        userData: *mut ::core::ffi::c_void,
        callback: *mut cudaAsyncCallbackHandle_t,
    ) -> cudaError_t {
        (culib().cudaDeviceRegisterAsyncNotification)(device, callbackFunc, userData, callback)
    }
    pub unsafe fn cudaDeviceReset() -> cudaError_t {
        (culib().cudaDeviceReset)()
    }
    pub unsafe fn cudaDeviceSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t {
        (culib().cudaDeviceSetCacheConfig)(cacheConfig)
    }
    pub unsafe fn cudaDeviceSetGraphMemAttribute(
        device: ::core::ffi::c_int,
        attr: cudaGraphMemAttributeType,
        value: *mut ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaDeviceSetGraphMemAttribute)(device, attr, value)
    }
    pub unsafe fn cudaDeviceSetLimit(limit: cudaLimit, value: usize) -> cudaError_t {
        (culib().cudaDeviceSetLimit)(limit, value)
    }
    pub unsafe fn cudaDeviceSetMemPool(
        device: ::core::ffi::c_int,
        memPool: cudaMemPool_t,
    ) -> cudaError_t {
        (culib().cudaDeviceSetMemPool)(device, memPool)
    }
    pub unsafe fn cudaDeviceSetSharedMemConfig(config: cudaSharedMemConfig) -> cudaError_t {
        (culib().cudaDeviceSetSharedMemConfig)(config)
    }
    pub unsafe fn cudaDeviceSynchronize() -> cudaError_t {
        (culib().cudaDeviceSynchronize)()
    }
    #[cfg(any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaDeviceUnregisterAsyncNotification(
        device: ::core::ffi::c_int,
        callback: cudaAsyncCallbackHandle_t,
    ) -> cudaError_t {
        (culib().cudaDeviceUnregisterAsyncNotification)(device, callback)
    }
    pub unsafe fn cudaDriverGetVersion(driverVersion: *mut ::core::ffi::c_int) -> cudaError_t {
        (culib().cudaDriverGetVersion)(driverVersion)
    }
    pub unsafe fn cudaEventCreate(event: *mut cudaEvent_t) -> cudaError_t {
        (culib().cudaEventCreate)(event)
    }
    pub unsafe fn cudaEventCreateWithFlags(
        event: *mut cudaEvent_t,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaEventCreateWithFlags)(event, flags)
    }
    pub unsafe fn cudaEventDestroy(event: cudaEvent_t) -> cudaError_t {
        (culib().cudaEventDestroy)(event)
    }
    pub unsafe fn cudaEventElapsedTime(
        ms: *mut f32,
        start: cudaEvent_t,
        end: cudaEvent_t,
    ) -> cudaError_t {
        (culib().cudaEventElapsedTime)(ms, start, end)
    }
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn cudaEventElapsedTime_v2(
        ms: *mut f32,
        start: cudaEvent_t,
        end: cudaEvent_t,
    ) -> cudaError_t {
        (culib().cudaEventElapsedTime_v2)(ms, start, end)
    }
    pub unsafe fn cudaEventQuery(event: cudaEvent_t) -> cudaError_t {
        (culib().cudaEventQuery)(event)
    }
    pub unsafe fn cudaEventRecord(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t {
        (culib().cudaEventRecord)(event, stream)
    }
    pub unsafe fn cudaEventRecordWithFlags(
        event: cudaEvent_t,
        stream: cudaStream_t,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaEventRecordWithFlags)(event, stream, flags)
    }
    pub unsafe fn cudaEventSynchronize(event: cudaEvent_t) -> cudaError_t {
        (culib().cudaEventSynchronize)(event)
    }
    pub unsafe fn cudaExternalMemoryGetMappedBuffer(
        devPtr: *mut *mut ::core::ffi::c_void,
        extMem: cudaExternalMemory_t,
        bufferDesc: *const cudaExternalMemoryBufferDesc,
    ) -> cudaError_t {
        (culib().cudaExternalMemoryGetMappedBuffer)(devPtr, extMem, bufferDesc)
    }
    pub unsafe fn cudaExternalMemoryGetMappedMipmappedArray(
        mipmap: *mut cudaMipmappedArray_t,
        extMem: cudaExternalMemory_t,
        mipmapDesc: *const cudaExternalMemoryMipmappedArrayDesc,
    ) -> cudaError_t {
        (culib().cudaExternalMemoryGetMappedMipmappedArray)(mipmap, extMem, mipmapDesc)
    }
    pub unsafe fn cudaFree(devPtr: *mut ::core::ffi::c_void) -> cudaError_t {
        (culib().cudaFree)(devPtr)
    }
    pub unsafe fn cudaFreeArray(array: cudaArray_t) -> cudaError_t {
        (culib().cudaFreeArray)(array)
    }
    pub unsafe fn cudaFreeAsync(
        devPtr: *mut ::core::ffi::c_void,
        hStream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaFreeAsync)(devPtr, hStream)
    }
    pub unsafe fn cudaFreeHost(ptr: *mut ::core::ffi::c_void) -> cudaError_t {
        (culib().cudaFreeHost)(ptr)
    }
    pub unsafe fn cudaFreeMipmappedArray(mipmappedArray: cudaMipmappedArray_t) -> cudaError_t {
        (culib().cudaFreeMipmappedArray)(mipmappedArray)
    }
    pub unsafe fn cudaFuncGetAttributes(
        attr: *mut cudaFuncAttributes,
        func: *const ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaFuncGetAttributes)(attr, func)
    }
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaFuncGetName(
        name: *mut *const ::core::ffi::c_char,
        func: *const ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaFuncGetName)(name, func)
    }
    #[cfg(any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaFuncGetParamInfo(
        func: *const ::core::ffi::c_void,
        paramIndex: usize,
        paramOffset: *mut usize,
        paramSize: *mut usize,
    ) -> cudaError_t {
        (culib().cudaFuncGetParamInfo)(func, paramIndex, paramOffset, paramSize)
    }
    pub unsafe fn cudaFuncSetAttribute(
        func: *const ::core::ffi::c_void,
        attr: cudaFuncAttribute,
        value: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaFuncSetAttribute)(func, attr, value)
    }
    pub unsafe fn cudaFuncSetCacheConfig(
        func: *const ::core::ffi::c_void,
        cacheConfig: cudaFuncCache,
    ) -> cudaError_t {
        (culib().cudaFuncSetCacheConfig)(func, cacheConfig)
    }
    pub unsafe fn cudaFuncSetSharedMemConfig(
        func: *const ::core::ffi::c_void,
        config: cudaSharedMemConfig,
    ) -> cudaError_t {
        (culib().cudaFuncSetSharedMemConfig)(func, config)
    }
    pub unsafe fn cudaGetChannelDesc(
        desc: *mut cudaChannelFormatDesc,
        array: cudaArray_const_t,
    ) -> cudaError_t {
        (culib().cudaGetChannelDesc)(desc, array)
    }
    pub unsafe fn cudaGetDevice(device: *mut ::core::ffi::c_int) -> cudaError_t {
        (culib().cudaGetDevice)(device)
    }
    pub unsafe fn cudaGetDeviceCount(count: *mut ::core::ffi::c_int) -> cudaError_t {
        (culib().cudaGetDeviceCount)(count)
    }
    pub unsafe fn cudaGetDeviceFlags(flags: *mut ::core::ffi::c_uint) -> cudaError_t {
        (culib().cudaGetDeviceFlags)(flags)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn cudaGetDeviceProperties(
        prop: *mut cudaDeviceProp,
        device: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaGetDeviceProperties)(prop, device)
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
    pub unsafe fn cudaGetDeviceProperties_v2(
        prop: *mut cudaDeviceProp,
        device: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaGetDeviceProperties_v2)(prop, device)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn cudaGetDriverEntryPoint(
        symbol: *const ::core::ffi::c_char,
        funcPtr: *mut *mut ::core::ffi::c_void,
        flags: ::core::ffi::c_ulonglong,
    ) -> cudaError_t {
        (culib().cudaGetDriverEntryPoint)(symbol, funcPtr, flags)
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
    pub unsafe fn cudaGetDriverEntryPoint(
        symbol: *const ::core::ffi::c_char,
        funcPtr: *mut *mut ::core::ffi::c_void,
        flags: ::core::ffi::c_ulonglong,
        driverStatus: *mut cudaDriverEntryPointQueryResult,
    ) -> cudaError_t {
        (culib().cudaGetDriverEntryPoint)(symbol, funcPtr, flags, driverStatus)
    }
    #[cfg(any(
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaGetDriverEntryPointByVersion(
        symbol: *const ::core::ffi::c_char,
        funcPtr: *mut *mut ::core::ffi::c_void,
        cudaVersion: ::core::ffi::c_uint,
        flags: ::core::ffi::c_ulonglong,
        driverStatus: *mut cudaDriverEntryPointQueryResult,
    ) -> cudaError_t {
        (culib().cudaGetDriverEntryPointByVersion)(
            symbol,
            funcPtr,
            cudaVersion,
            flags,
            driverStatus,
        )
    }
    pub unsafe fn cudaGetErrorName(error: cudaError_t) -> *const ::core::ffi::c_char {
        (culib().cudaGetErrorName)(error)
    }
    pub unsafe fn cudaGetErrorString(error: cudaError_t) -> *const ::core::ffi::c_char {
        (culib().cudaGetErrorString)(error)
    }
    pub unsafe fn cudaGetExportTable(
        ppExportTable: *mut *const ::core::ffi::c_void,
        pExportTableId: *const cudaUUID_t,
    ) -> cudaError_t {
        (culib().cudaGetExportTable)(ppExportTable, pExportTableId)
    }
    pub unsafe fn cudaGetFuncBySymbol(
        functionPtr: *mut cudaFunction_t,
        symbolPtr: *const ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaGetFuncBySymbol)(functionPtr, symbolPtr)
    }
    #[cfg(any(
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaGetKernel(
        kernelPtr: *mut cudaKernel_t,
        entryFuncAddr: *const ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaGetKernel)(kernelPtr, entryFuncAddr)
    }
    pub unsafe fn cudaGetLastError() -> cudaError_t {
        (culib().cudaGetLastError)()
    }
    pub unsafe fn cudaGetMipmappedArrayLevel(
        levelArray: *mut cudaArray_t,
        mipmappedArray: cudaMipmappedArray_const_t,
        level: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaGetMipmappedArrayLevel)(levelArray, mipmappedArray, level)
    }
    pub unsafe fn cudaGetSurfaceObjectResourceDesc(
        pResDesc: *mut cudaResourceDesc,
        surfObject: cudaSurfaceObject_t,
    ) -> cudaError_t {
        (culib().cudaGetSurfaceObjectResourceDesc)(pResDesc, surfObject)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn cudaGetSurfaceReference(
        surfref: *mut *const surfaceReference,
        symbol: *const ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaGetSurfaceReference)(surfref, symbol)
    }
    pub unsafe fn cudaGetSymbolAddress(
        devPtr: *mut *mut ::core::ffi::c_void,
        symbol: *const ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaGetSymbolAddress)(devPtr, symbol)
    }
    pub unsafe fn cudaGetSymbolSize(
        size: *mut usize,
        symbol: *const ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaGetSymbolSize)(size, symbol)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn cudaGetTextureAlignmentOffset(
        offset: *mut usize,
        texref: *const textureReference,
    ) -> cudaError_t {
        (culib().cudaGetTextureAlignmentOffset)(offset, texref)
    }
    pub unsafe fn cudaGetTextureObjectResourceDesc(
        pResDesc: *mut cudaResourceDesc,
        texObject: cudaTextureObject_t,
    ) -> cudaError_t {
        (culib().cudaGetTextureObjectResourceDesc)(pResDesc, texObject)
    }
    pub unsafe fn cudaGetTextureObjectResourceViewDesc(
        pResViewDesc: *mut cudaResourceViewDesc,
        texObject: cudaTextureObject_t,
    ) -> cudaError_t {
        (culib().cudaGetTextureObjectResourceViewDesc)(pResViewDesc, texObject)
    }
    pub unsafe fn cudaGetTextureObjectTextureDesc(
        pTexDesc: *mut cudaTextureDesc,
        texObject: cudaTextureObject_t,
    ) -> cudaError_t {
        (culib().cudaGetTextureObjectTextureDesc)(pTexDesc, texObject)
    }
    #[cfg(any(feature = "cuda-11080"))]
    pub unsafe fn cudaGetTextureObjectTextureDesc_v2(
        pTexDesc: *mut cudaTextureDesc_v2,
        texObject: cudaTextureObject_t,
    ) -> cudaError_t {
        (culib().cudaGetTextureObjectTextureDesc_v2)(pTexDesc, texObject)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn cudaGetTextureReference(
        texref: *mut *const textureReference,
        symbol: *const ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaGetTextureReference)(texref, symbol)
    }
    pub unsafe fn cudaGraphAddChildGraphNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        childGraph: cudaGraph_t,
    ) -> cudaError_t {
        (culib().cudaGraphAddChildGraphNode)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            childGraph,
        )
    }
    pub unsafe fn cudaGraphAddDependencies(
        graph: cudaGraph_t,
        from: *const cudaGraphNode_t,
        to: *const cudaGraphNode_t,
        numDependencies: usize,
    ) -> cudaError_t {
        (culib().cudaGraphAddDependencies)(graph, from, to, numDependencies)
    }
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaGraphAddDependencies_v2(
        graph: cudaGraph_t,
        from: *const cudaGraphNode_t,
        to: *const cudaGraphNode_t,
        edgeData: *const cudaGraphEdgeData,
        numDependencies: usize,
    ) -> cudaError_t {
        (culib().cudaGraphAddDependencies_v2)(graph, from, to, edgeData, numDependencies)
    }
    pub unsafe fn cudaGraphAddEmptyNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
    ) -> cudaError_t {
        (culib().cudaGraphAddEmptyNode)(pGraphNode, graph, pDependencies, numDependencies)
    }
    pub unsafe fn cudaGraphAddEventRecordNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        event: cudaEvent_t,
    ) -> cudaError_t {
        (culib().cudaGraphAddEventRecordNode)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            event,
        )
    }
    pub unsafe fn cudaGraphAddEventWaitNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        event: cudaEvent_t,
    ) -> cudaError_t {
        (culib().cudaGraphAddEventWaitNode)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            event,
        )
    }
    pub unsafe fn cudaGraphAddExternalSemaphoresSignalNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphAddExternalSemaphoresSignalNode)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            nodeParams,
        )
    }
    pub unsafe fn cudaGraphAddExternalSemaphoresWaitNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphAddExternalSemaphoresWaitNode)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            nodeParams,
        )
    }
    pub unsafe fn cudaGraphAddHostNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        pNodeParams: *const cudaHostNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphAddHostNode)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            pNodeParams,
        )
    }
    pub unsafe fn cudaGraphAddKernelNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        pNodeParams: *const cudaKernelNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphAddKernelNode)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            pNodeParams,
        )
    }
    pub unsafe fn cudaGraphAddMemAllocNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        nodeParams: *mut cudaMemAllocNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphAddMemAllocNode)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            nodeParams,
        )
    }
    pub unsafe fn cudaGraphAddMemFreeNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        dptr: *mut ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaGraphAddMemFreeNode)(pGraphNode, graph, pDependencies, numDependencies, dptr)
    }
    pub unsafe fn cudaGraphAddMemcpyNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        pCopyParams: *const cudaMemcpy3DParms,
    ) -> cudaError_t {
        (culib().cudaGraphAddMemcpyNode)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            pCopyParams,
        )
    }
    pub unsafe fn cudaGraphAddMemcpyNode1D(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        dst: *mut ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaGraphAddMemcpyNode1D)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            dst,
            src,
            count,
            kind,
        )
    }
    pub unsafe fn cudaGraphAddMemcpyNodeFromSymbol(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        dst: *mut ::core::ffi::c_void,
        symbol: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaGraphAddMemcpyNodeFromSymbol)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            dst,
            symbol,
            count,
            offset,
            kind,
        )
    }
    pub unsafe fn cudaGraphAddMemcpyNodeToSymbol(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        symbol: *const ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaGraphAddMemcpyNodeToSymbol)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            symbol,
            src,
            count,
            offset,
            kind,
        )
    }
    pub unsafe fn cudaGraphAddMemsetNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        pMemsetParams: *const cudaMemsetParams,
    ) -> cudaError_t {
        (culib().cudaGraphAddMemsetNode)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            pMemsetParams,
        )
    }
    #[cfg(any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaGraphAddNode(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        numDependencies: usize,
        nodeParams: *mut cudaGraphNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphAddNode)(
            pGraphNode,
            graph,
            pDependencies,
            numDependencies,
            nodeParams,
        )
    }
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaGraphAddNode_v2(
        pGraphNode: *mut cudaGraphNode_t,
        graph: cudaGraph_t,
        pDependencies: *const cudaGraphNode_t,
        dependencyData: *const cudaGraphEdgeData,
        numDependencies: usize,
        nodeParams: *mut cudaGraphNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphAddNode_v2)(
            pGraphNode,
            graph,
            pDependencies,
            dependencyData,
            numDependencies,
            nodeParams,
        )
    }
    pub unsafe fn cudaGraphChildGraphNodeGetGraph(
        node: cudaGraphNode_t,
        pGraph: *mut cudaGraph_t,
    ) -> cudaError_t {
        (culib().cudaGraphChildGraphNodeGetGraph)(node, pGraph)
    }
    pub unsafe fn cudaGraphClone(
        pGraphClone: *mut cudaGraph_t,
        originalGraph: cudaGraph_t,
    ) -> cudaError_t {
        (culib().cudaGraphClone)(pGraphClone, originalGraph)
    }
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaGraphConditionalHandleCreate(
        pHandle_out: *mut cudaGraphConditionalHandle,
        graph: cudaGraph_t,
        defaultLaunchValue: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaGraphConditionalHandleCreate)(pHandle_out, graph, defaultLaunchValue, flags)
    }
    pub unsafe fn cudaGraphCreate(
        pGraph: *mut cudaGraph_t,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaGraphCreate)(pGraph, flags)
    }
    pub unsafe fn cudaGraphDebugDotPrint(
        graph: cudaGraph_t,
        path: *const ::core::ffi::c_char,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaGraphDebugDotPrint)(graph, path, flags)
    }
    pub unsafe fn cudaGraphDestroy(graph: cudaGraph_t) -> cudaError_t {
        (culib().cudaGraphDestroy)(graph)
    }
    pub unsafe fn cudaGraphDestroyNode(node: cudaGraphNode_t) -> cudaError_t {
        (culib().cudaGraphDestroyNode)(node)
    }
    pub unsafe fn cudaGraphEventRecordNodeGetEvent(
        node: cudaGraphNode_t,
        event_out: *mut cudaEvent_t,
    ) -> cudaError_t {
        (culib().cudaGraphEventRecordNodeGetEvent)(node, event_out)
    }
    pub unsafe fn cudaGraphEventRecordNodeSetEvent(
        node: cudaGraphNode_t,
        event: cudaEvent_t,
    ) -> cudaError_t {
        (culib().cudaGraphEventRecordNodeSetEvent)(node, event)
    }
    pub unsafe fn cudaGraphEventWaitNodeGetEvent(
        node: cudaGraphNode_t,
        event_out: *mut cudaEvent_t,
    ) -> cudaError_t {
        (culib().cudaGraphEventWaitNodeGetEvent)(node, event_out)
    }
    pub unsafe fn cudaGraphEventWaitNodeSetEvent(
        node: cudaGraphNode_t,
        event: cudaEvent_t,
    ) -> cudaError_t {
        (culib().cudaGraphEventWaitNodeSetEvent)(node, event)
    }
    pub unsafe fn cudaGraphExecChildGraphNodeSetParams(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        childGraph: cudaGraph_t,
    ) -> cudaError_t {
        (culib().cudaGraphExecChildGraphNodeSetParams)(hGraphExec, node, childGraph)
    }
    pub unsafe fn cudaGraphExecDestroy(graphExec: cudaGraphExec_t) -> cudaError_t {
        (culib().cudaGraphExecDestroy)(graphExec)
    }
    pub unsafe fn cudaGraphExecEventRecordNodeSetEvent(
        hGraphExec: cudaGraphExec_t,
        hNode: cudaGraphNode_t,
        event: cudaEvent_t,
    ) -> cudaError_t {
        (culib().cudaGraphExecEventRecordNodeSetEvent)(hGraphExec, hNode, event)
    }
    pub unsafe fn cudaGraphExecEventWaitNodeSetEvent(
        hGraphExec: cudaGraphExec_t,
        hNode: cudaGraphNode_t,
        event: cudaEvent_t,
    ) -> cudaError_t {
        (culib().cudaGraphExecEventWaitNodeSetEvent)(hGraphExec, hNode, event)
    }
    pub unsafe fn cudaGraphExecExternalSemaphoresSignalNodeSetParams(
        hGraphExec: cudaGraphExec_t,
        hNode: cudaGraphNode_t,
        nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphExecExternalSemaphoresSignalNodeSetParams)(hGraphExec, hNode, nodeParams)
    }
    pub unsafe fn cudaGraphExecExternalSemaphoresWaitNodeSetParams(
        hGraphExec: cudaGraphExec_t,
        hNode: cudaGraphNode_t,
        nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphExecExternalSemaphoresWaitNodeSetParams)(hGraphExec, hNode, nodeParams)
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
    pub unsafe fn cudaGraphExecGetFlags(
        graphExec: cudaGraphExec_t,
        flags: *mut ::core::ffi::c_ulonglong,
    ) -> cudaError_t {
        (culib().cudaGraphExecGetFlags)(graphExec, flags)
    }
    pub unsafe fn cudaGraphExecHostNodeSetParams(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        pNodeParams: *const cudaHostNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphExecHostNodeSetParams)(hGraphExec, node, pNodeParams)
    }
    pub unsafe fn cudaGraphExecKernelNodeSetParams(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        pNodeParams: *const cudaKernelNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphExecKernelNodeSetParams)(hGraphExec, node, pNodeParams)
    }
    pub unsafe fn cudaGraphExecMemcpyNodeSetParams(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        pNodeParams: *const cudaMemcpy3DParms,
    ) -> cudaError_t {
        (culib().cudaGraphExecMemcpyNodeSetParams)(hGraphExec, node, pNodeParams)
    }
    pub unsafe fn cudaGraphExecMemcpyNodeSetParams1D(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        dst: *mut ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaGraphExecMemcpyNodeSetParams1D)(hGraphExec, node, dst, src, count, kind)
    }
    pub unsafe fn cudaGraphExecMemcpyNodeSetParamsFromSymbol(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        dst: *mut ::core::ffi::c_void,
        symbol: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaGraphExecMemcpyNodeSetParamsFromSymbol)(
            hGraphExec, node, dst, symbol, count, offset, kind,
        )
    }
    pub unsafe fn cudaGraphExecMemcpyNodeSetParamsToSymbol(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        symbol: *const ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaGraphExecMemcpyNodeSetParamsToSymbol)(
            hGraphExec, node, symbol, src, count, offset, kind,
        )
    }
    pub unsafe fn cudaGraphExecMemsetNodeSetParams(
        hGraphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        pNodeParams: *const cudaMemsetParams,
    ) -> cudaError_t {
        (culib().cudaGraphExecMemsetNodeSetParams)(hGraphExec, node, pNodeParams)
    }
    #[cfg(any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaGraphExecNodeSetParams(
        graphExec: cudaGraphExec_t,
        node: cudaGraphNode_t,
        nodeParams: *mut cudaGraphNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphExecNodeSetParams)(graphExec, node, nodeParams)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn cudaGraphExecUpdate(
        hGraphExec: cudaGraphExec_t,
        hGraph: cudaGraph_t,
        hErrorNode_out: *mut cudaGraphNode_t,
        updateResult_out: *mut cudaGraphExecUpdateResult,
    ) -> cudaError_t {
        (culib().cudaGraphExecUpdate)(hGraphExec, hGraph, hErrorNode_out, updateResult_out)
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
    pub unsafe fn cudaGraphExecUpdate(
        hGraphExec: cudaGraphExec_t,
        hGraph: cudaGraph_t,
        resultInfo: *mut cudaGraphExecUpdateResultInfo,
    ) -> cudaError_t {
        (culib().cudaGraphExecUpdate)(hGraphExec, hGraph, resultInfo)
    }
    pub unsafe fn cudaGraphExternalSemaphoresSignalNodeGetParams(
        hNode: cudaGraphNode_t,
        params_out: *mut cudaExternalSemaphoreSignalNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphExternalSemaphoresSignalNodeGetParams)(hNode, params_out)
    }
    pub unsafe fn cudaGraphExternalSemaphoresSignalNodeSetParams(
        hNode: cudaGraphNode_t,
        nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphExternalSemaphoresSignalNodeSetParams)(hNode, nodeParams)
    }
    pub unsafe fn cudaGraphExternalSemaphoresWaitNodeGetParams(
        hNode: cudaGraphNode_t,
        params_out: *mut cudaExternalSemaphoreWaitNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphExternalSemaphoresWaitNodeGetParams)(hNode, params_out)
    }
    pub unsafe fn cudaGraphExternalSemaphoresWaitNodeSetParams(
        hNode: cudaGraphNode_t,
        nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphExternalSemaphoresWaitNodeSetParams)(hNode, nodeParams)
    }
    pub unsafe fn cudaGraphGetEdges(
        graph: cudaGraph_t,
        from: *mut cudaGraphNode_t,
        to: *mut cudaGraphNode_t,
        numEdges: *mut usize,
    ) -> cudaError_t {
        (culib().cudaGraphGetEdges)(graph, from, to, numEdges)
    }
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaGraphGetEdges_v2(
        graph: cudaGraph_t,
        from: *mut cudaGraphNode_t,
        to: *mut cudaGraphNode_t,
        edgeData: *mut cudaGraphEdgeData,
        numEdges: *mut usize,
    ) -> cudaError_t {
        (culib().cudaGraphGetEdges_v2)(graph, from, to, edgeData, numEdges)
    }
    pub unsafe fn cudaGraphGetNodes(
        graph: cudaGraph_t,
        nodes: *mut cudaGraphNode_t,
        numNodes: *mut usize,
    ) -> cudaError_t {
        (culib().cudaGraphGetNodes)(graph, nodes, numNodes)
    }
    pub unsafe fn cudaGraphGetRootNodes(
        graph: cudaGraph_t,
        pRootNodes: *mut cudaGraphNode_t,
        pNumRootNodes: *mut usize,
    ) -> cudaError_t {
        (culib().cudaGraphGetRootNodes)(graph, pRootNodes, pNumRootNodes)
    }
    pub unsafe fn cudaGraphHostNodeGetParams(
        node: cudaGraphNode_t,
        pNodeParams: *mut cudaHostNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphHostNodeGetParams)(node, pNodeParams)
    }
    pub unsafe fn cudaGraphHostNodeSetParams(
        node: cudaGraphNode_t,
        pNodeParams: *const cudaHostNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphHostNodeSetParams)(node, pNodeParams)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn cudaGraphInstantiate(
        pGraphExec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        pErrorNode: *mut cudaGraphNode_t,
        pLogBuffer: *mut ::core::ffi::c_char,
        bufferSize: usize,
    ) -> cudaError_t {
        (culib().cudaGraphInstantiate)(pGraphExec, graph, pErrorNode, pLogBuffer, bufferSize)
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
    pub unsafe fn cudaGraphInstantiate(
        pGraphExec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        flags: ::core::ffi::c_ulonglong,
    ) -> cudaError_t {
        (culib().cudaGraphInstantiate)(pGraphExec, graph, flags)
    }
    pub unsafe fn cudaGraphInstantiateWithFlags(
        pGraphExec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        flags: ::core::ffi::c_ulonglong,
    ) -> cudaError_t {
        (culib().cudaGraphInstantiateWithFlags)(pGraphExec, graph, flags)
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
    pub unsafe fn cudaGraphInstantiateWithParams(
        pGraphExec: *mut cudaGraphExec_t,
        graph: cudaGraph_t,
        instantiateParams: *mut cudaGraphInstantiateParams,
    ) -> cudaError_t {
        (culib().cudaGraphInstantiateWithParams)(pGraphExec, graph, instantiateParams)
    }
    pub unsafe fn cudaGraphKernelNodeCopyAttributes(
        hSrc: cudaGraphNode_t,
        hDst: cudaGraphNode_t,
    ) -> cudaError_t {
        (culib().cudaGraphKernelNodeCopyAttributes)(hSrc, hDst)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    ))]
    pub unsafe fn cudaGraphKernelNodeGetAttribute(
        hNode: cudaGraphNode_t,
        attr: cudaKernelNodeAttrID,
        value_out: *mut cudaKernelNodeAttrValue,
    ) -> cudaError_t {
        (culib().cudaGraphKernelNodeGetAttribute)(hNode, attr, value_out)
    }
    #[cfg(any(
        feature = "cuda-11080",
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
    pub unsafe fn cudaGraphKernelNodeGetAttribute(
        hNode: cudaGraphNode_t,
        attr: cudaLaunchAttributeID,
        value_out: *mut cudaLaunchAttributeValue,
    ) -> cudaError_t {
        (culib().cudaGraphKernelNodeGetAttribute)(hNode, attr, value_out)
    }
    pub unsafe fn cudaGraphKernelNodeGetParams(
        node: cudaGraphNode_t,
        pNodeParams: *mut cudaKernelNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphKernelNodeGetParams)(node, pNodeParams)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    ))]
    pub unsafe fn cudaGraphKernelNodeSetAttribute(
        hNode: cudaGraphNode_t,
        attr: cudaKernelNodeAttrID,
        value: *const cudaKernelNodeAttrValue,
    ) -> cudaError_t {
        (culib().cudaGraphKernelNodeSetAttribute)(hNode, attr, value)
    }
    #[cfg(any(
        feature = "cuda-11080",
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
    pub unsafe fn cudaGraphKernelNodeSetAttribute(
        hNode: cudaGraphNode_t,
        attr: cudaLaunchAttributeID,
        value: *const cudaLaunchAttributeValue,
    ) -> cudaError_t {
        (culib().cudaGraphKernelNodeSetAttribute)(hNode, attr, value)
    }
    pub unsafe fn cudaGraphKernelNodeSetParams(
        node: cudaGraphNode_t,
        pNodeParams: *const cudaKernelNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphKernelNodeSetParams)(node, pNodeParams)
    }
    pub unsafe fn cudaGraphLaunch(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t {
        (culib().cudaGraphLaunch)(graphExec, stream)
    }
    pub unsafe fn cudaGraphMemAllocNodeGetParams(
        node: cudaGraphNode_t,
        params_out: *mut cudaMemAllocNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphMemAllocNodeGetParams)(node, params_out)
    }
    pub unsafe fn cudaGraphMemFreeNodeGetParams(
        node: cudaGraphNode_t,
        dptr_out: *mut ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaGraphMemFreeNodeGetParams)(node, dptr_out)
    }
    pub unsafe fn cudaGraphMemcpyNodeGetParams(
        node: cudaGraphNode_t,
        pNodeParams: *mut cudaMemcpy3DParms,
    ) -> cudaError_t {
        (culib().cudaGraphMemcpyNodeGetParams)(node, pNodeParams)
    }
    pub unsafe fn cudaGraphMemcpyNodeSetParams(
        node: cudaGraphNode_t,
        pNodeParams: *const cudaMemcpy3DParms,
    ) -> cudaError_t {
        (culib().cudaGraphMemcpyNodeSetParams)(node, pNodeParams)
    }
    pub unsafe fn cudaGraphMemcpyNodeSetParams1D(
        node: cudaGraphNode_t,
        dst: *mut ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaGraphMemcpyNodeSetParams1D)(node, dst, src, count, kind)
    }
    pub unsafe fn cudaGraphMemcpyNodeSetParamsFromSymbol(
        node: cudaGraphNode_t,
        dst: *mut ::core::ffi::c_void,
        symbol: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaGraphMemcpyNodeSetParamsFromSymbol)(node, dst, symbol, count, offset, kind)
    }
    pub unsafe fn cudaGraphMemcpyNodeSetParamsToSymbol(
        node: cudaGraphNode_t,
        symbol: *const ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaGraphMemcpyNodeSetParamsToSymbol)(node, symbol, src, count, offset, kind)
    }
    pub unsafe fn cudaGraphMemsetNodeGetParams(
        node: cudaGraphNode_t,
        pNodeParams: *mut cudaMemsetParams,
    ) -> cudaError_t {
        (culib().cudaGraphMemsetNodeGetParams)(node, pNodeParams)
    }
    pub unsafe fn cudaGraphMemsetNodeSetParams(
        node: cudaGraphNode_t,
        pNodeParams: *const cudaMemsetParams,
    ) -> cudaError_t {
        (culib().cudaGraphMemsetNodeSetParams)(node, pNodeParams)
    }
    pub unsafe fn cudaGraphNodeFindInClone(
        pNode: *mut cudaGraphNode_t,
        originalNode: cudaGraphNode_t,
        clonedGraph: cudaGraph_t,
    ) -> cudaError_t {
        (culib().cudaGraphNodeFindInClone)(pNode, originalNode, clonedGraph)
    }
    pub unsafe fn cudaGraphNodeGetDependencies(
        node: cudaGraphNode_t,
        pDependencies: *mut cudaGraphNode_t,
        pNumDependencies: *mut usize,
    ) -> cudaError_t {
        (culib().cudaGraphNodeGetDependencies)(node, pDependencies, pNumDependencies)
    }
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaGraphNodeGetDependencies_v2(
        node: cudaGraphNode_t,
        pDependencies: *mut cudaGraphNode_t,
        edgeData: *mut cudaGraphEdgeData,
        pNumDependencies: *mut usize,
    ) -> cudaError_t {
        (culib().cudaGraphNodeGetDependencies_v2)(node, pDependencies, edgeData, pNumDependencies)
    }
    pub unsafe fn cudaGraphNodeGetDependentNodes(
        node: cudaGraphNode_t,
        pDependentNodes: *mut cudaGraphNode_t,
        pNumDependentNodes: *mut usize,
    ) -> cudaError_t {
        (culib().cudaGraphNodeGetDependentNodes)(node, pDependentNodes, pNumDependentNodes)
    }
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaGraphNodeGetDependentNodes_v2(
        node: cudaGraphNode_t,
        pDependentNodes: *mut cudaGraphNode_t,
        edgeData: *mut cudaGraphEdgeData,
        pNumDependentNodes: *mut usize,
    ) -> cudaError_t {
        (culib().cudaGraphNodeGetDependentNodes_v2)(
            node,
            pDependentNodes,
            edgeData,
            pNumDependentNodes,
        )
    }
    #[cfg(any(
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
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaGraphNodeGetEnabled(
        hGraphExec: cudaGraphExec_t,
        hNode: cudaGraphNode_t,
        isEnabled: *mut ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaGraphNodeGetEnabled)(hGraphExec, hNode, isEnabled)
    }
    pub unsafe fn cudaGraphNodeGetType(
        node: cudaGraphNode_t,
        pType: *mut cudaGraphNodeType,
    ) -> cudaError_t {
        (culib().cudaGraphNodeGetType)(node, pType)
    }
    #[cfg(any(
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
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaGraphNodeSetEnabled(
        hGraphExec: cudaGraphExec_t,
        hNode: cudaGraphNode_t,
        isEnabled: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaGraphNodeSetEnabled)(hGraphExec, hNode, isEnabled)
    }
    #[cfg(any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaGraphNodeSetParams(
        node: cudaGraphNode_t,
        nodeParams: *mut cudaGraphNodeParams,
    ) -> cudaError_t {
        (culib().cudaGraphNodeSetParams)(node, nodeParams)
    }
    pub unsafe fn cudaGraphReleaseUserObject(
        graph: cudaGraph_t,
        object: cudaUserObject_t,
        count: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaGraphReleaseUserObject)(graph, object, count)
    }
    pub unsafe fn cudaGraphRemoveDependencies(
        graph: cudaGraph_t,
        from: *const cudaGraphNode_t,
        to: *const cudaGraphNode_t,
        numDependencies: usize,
    ) -> cudaError_t {
        (culib().cudaGraphRemoveDependencies)(graph, from, to, numDependencies)
    }
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaGraphRemoveDependencies_v2(
        graph: cudaGraph_t,
        from: *const cudaGraphNode_t,
        to: *const cudaGraphNode_t,
        edgeData: *const cudaGraphEdgeData,
        numDependencies: usize,
    ) -> cudaError_t {
        (culib().cudaGraphRemoveDependencies_v2)(graph, from, to, edgeData, numDependencies)
    }
    pub unsafe fn cudaGraphRetainUserObject(
        graph: cudaGraph_t,
        object: cudaUserObject_t,
        count: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaGraphRetainUserObject)(graph, object, count, flags)
    }
    pub unsafe fn cudaGraphUpload(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t {
        (culib().cudaGraphUpload)(graphExec, stream)
    }
    pub unsafe fn cudaGraphicsMapResources(
        count: ::core::ffi::c_int,
        resources: *mut cudaGraphicsResource_t,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaGraphicsMapResources)(count, resources, stream)
    }
    pub unsafe fn cudaGraphicsResourceGetMappedMipmappedArray(
        mipmappedArray: *mut cudaMipmappedArray_t,
        resource: cudaGraphicsResource_t,
    ) -> cudaError_t {
        (culib().cudaGraphicsResourceGetMappedMipmappedArray)(mipmappedArray, resource)
    }
    pub unsafe fn cudaGraphicsResourceGetMappedPointer(
        devPtr: *mut *mut ::core::ffi::c_void,
        size: *mut usize,
        resource: cudaGraphicsResource_t,
    ) -> cudaError_t {
        (culib().cudaGraphicsResourceGetMappedPointer)(devPtr, size, resource)
    }
    pub unsafe fn cudaGraphicsResourceSetMapFlags(
        resource: cudaGraphicsResource_t,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaGraphicsResourceSetMapFlags)(resource, flags)
    }
    pub unsafe fn cudaGraphicsSubResourceGetMappedArray(
        array: *mut cudaArray_t,
        resource: cudaGraphicsResource_t,
        arrayIndex: ::core::ffi::c_uint,
        mipLevel: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaGraphicsSubResourceGetMappedArray)(array, resource, arrayIndex, mipLevel)
    }
    pub unsafe fn cudaGraphicsUnmapResources(
        count: ::core::ffi::c_int,
        resources: *mut cudaGraphicsResource_t,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaGraphicsUnmapResources)(count, resources, stream)
    }
    pub unsafe fn cudaGraphicsUnregisterResource(resource: cudaGraphicsResource_t) -> cudaError_t {
        (culib().cudaGraphicsUnregisterResource)(resource)
    }
    pub unsafe fn cudaHostAlloc(
        pHost: *mut *mut ::core::ffi::c_void,
        size: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaHostAlloc)(pHost, size, flags)
    }
    pub unsafe fn cudaHostGetDevicePointer(
        pDevice: *mut *mut ::core::ffi::c_void,
        pHost: *mut ::core::ffi::c_void,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaHostGetDevicePointer)(pDevice, pHost, flags)
    }
    pub unsafe fn cudaHostGetFlags(
        pFlags: *mut ::core::ffi::c_uint,
        pHost: *mut ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaHostGetFlags)(pFlags, pHost)
    }
    pub unsafe fn cudaHostRegister(
        ptr: *mut ::core::ffi::c_void,
        size: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaHostRegister)(ptr, size, flags)
    }
    pub unsafe fn cudaHostUnregister(ptr: *mut ::core::ffi::c_void) -> cudaError_t {
        (culib().cudaHostUnregister)(ptr)
    }
    pub unsafe fn cudaImportExternalMemory(
        extMem_out: *mut cudaExternalMemory_t,
        memHandleDesc: *const cudaExternalMemoryHandleDesc,
    ) -> cudaError_t {
        (culib().cudaImportExternalMemory)(extMem_out, memHandleDesc)
    }
    pub unsafe fn cudaImportExternalSemaphore(
        extSem_out: *mut cudaExternalSemaphore_t,
        semHandleDesc: *const cudaExternalSemaphoreHandleDesc,
    ) -> cudaError_t {
        (culib().cudaImportExternalSemaphore)(extSem_out, semHandleDesc)
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
    pub unsafe fn cudaInitDevice(
        device: ::core::ffi::c_int,
        deviceFlags: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaInitDevice)(device, deviceFlags, flags)
    }
    pub unsafe fn cudaIpcCloseMemHandle(devPtr: *mut ::core::ffi::c_void) -> cudaError_t {
        (culib().cudaIpcCloseMemHandle)(devPtr)
    }
    pub unsafe fn cudaIpcGetEventHandle(
        handle: *mut cudaIpcEventHandle_t,
        event: cudaEvent_t,
    ) -> cudaError_t {
        (culib().cudaIpcGetEventHandle)(handle, event)
    }
    pub unsafe fn cudaIpcGetMemHandle(
        handle: *mut cudaIpcMemHandle_t,
        devPtr: *mut ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaIpcGetMemHandle)(handle, devPtr)
    }
    pub unsafe fn cudaIpcOpenEventHandle(
        event: *mut cudaEvent_t,
        handle: cudaIpcEventHandle_t,
    ) -> cudaError_t {
        (culib().cudaIpcOpenEventHandle)(event, handle)
    }
    pub unsafe fn cudaIpcOpenMemHandle(
        devPtr: *mut *mut ::core::ffi::c_void,
        handle: cudaIpcMemHandle_t,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaIpcOpenMemHandle)(devPtr, handle, flags)
    }
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn cudaKernelSetAttributeForDevice(
        kernel: cudaKernel_t,
        attr: cudaFuncAttribute,
        value: ::core::ffi::c_int,
        device: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaKernelSetAttributeForDevice)(kernel, attr, value, device)
    }
    pub unsafe fn cudaLaunchCooperativeKernel(
        func: *const ::core::ffi::c_void,
        gridDim: dim3,
        blockDim: dim3,
        args: *mut *mut ::core::ffi::c_void,
        sharedMem: usize,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaLaunchCooperativeKernel)(func, gridDim, blockDim, args, sharedMem, stream)
    }
    pub unsafe fn cudaLaunchCooperativeKernelMultiDevice(
        launchParamsList: *mut cudaLaunchParams,
        numDevices: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaLaunchCooperativeKernelMultiDevice)(launchParamsList, numDevices, flags)
    }
    pub unsafe fn cudaLaunchHostFunc(
        stream: cudaStream_t,
        fn_: cudaHostFn_t,
        userData: *mut ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaLaunchHostFunc)(stream, fn_, userData)
    }
    pub unsafe fn cudaLaunchKernel(
        func: *const ::core::ffi::c_void,
        gridDim: dim3,
        blockDim: dim3,
        args: *mut *mut ::core::ffi::c_void,
        sharedMem: usize,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaLaunchKernel)(func, gridDim, blockDim, args, sharedMem, stream)
    }
    #[cfg(any(
        feature = "cuda-11080",
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
    pub unsafe fn cudaLaunchKernelExC(
        config: *const cudaLaunchConfig_t,
        func: *const ::core::ffi::c_void,
        args: *mut *mut ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaLaunchKernelExC)(config, func, args)
    }
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn cudaLibraryEnumerateKernels(
        kernels: *mut cudaKernel_t,
        numKernels: ::core::ffi::c_uint,
        lib: cudaLibrary_t,
    ) -> cudaError_t {
        (culib().cudaLibraryEnumerateKernels)(kernels, numKernels, lib)
    }
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn cudaLibraryGetGlobal(
        dptr: *mut *mut ::core::ffi::c_void,
        bytes: *mut usize,
        library: cudaLibrary_t,
        name: *const ::core::ffi::c_char,
    ) -> cudaError_t {
        (culib().cudaLibraryGetGlobal)(dptr, bytes, library, name)
    }
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn cudaLibraryGetKernel(
        pKernel: *mut cudaKernel_t,
        library: cudaLibrary_t,
        name: *const ::core::ffi::c_char,
    ) -> cudaError_t {
        (culib().cudaLibraryGetKernel)(pKernel, library, name)
    }
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn cudaLibraryGetKernelCount(
        count: *mut ::core::ffi::c_uint,
        lib: cudaLibrary_t,
    ) -> cudaError_t {
        (culib().cudaLibraryGetKernelCount)(count, lib)
    }
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn cudaLibraryGetManaged(
        dptr: *mut *mut ::core::ffi::c_void,
        bytes: *mut usize,
        library: cudaLibrary_t,
        name: *const ::core::ffi::c_char,
    ) -> cudaError_t {
        (culib().cudaLibraryGetManaged)(dptr, bytes, library, name)
    }
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn cudaLibraryGetUnifiedFunction(
        fptr: *mut *mut ::core::ffi::c_void,
        library: cudaLibrary_t,
        symbol: *const ::core::ffi::c_char,
    ) -> cudaError_t {
        (culib().cudaLibraryGetUnifiedFunction)(fptr, library, symbol)
    }
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn cudaLibraryLoadData(
        library: *mut cudaLibrary_t,
        code: *const ::core::ffi::c_void,
        jitOptions: *mut cudaJitOption,
        jitOptionsValues: *mut *mut ::core::ffi::c_void,
        numJitOptions: ::core::ffi::c_uint,
        libraryOptions: *mut cudaLibraryOption,
        libraryOptionValues: *mut *mut ::core::ffi::c_void,
        numLibraryOptions: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaLibraryLoadData)(
            library,
            code,
            jitOptions,
            jitOptionsValues,
            numJitOptions,
            libraryOptions,
            libraryOptionValues,
            numLibraryOptions,
        )
    }
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn cudaLibraryLoadFromFile(
        library: *mut cudaLibrary_t,
        fileName: *const ::core::ffi::c_char,
        jitOptions: *mut cudaJitOption,
        jitOptionsValues: *mut *mut ::core::ffi::c_void,
        numJitOptions: ::core::ffi::c_uint,
        libraryOptions: *mut cudaLibraryOption,
        libraryOptionValues: *mut *mut ::core::ffi::c_void,
        numLibraryOptions: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaLibraryLoadFromFile)(
            library,
            fileName,
            jitOptions,
            jitOptionsValues,
            numJitOptions,
            libraryOptions,
            libraryOptionValues,
            numLibraryOptions,
        )
    }
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn cudaLibraryUnload(library: cudaLibrary_t) -> cudaError_t {
        (culib().cudaLibraryUnload)(library)
    }
    pub unsafe fn cudaMalloc(devPtr: *mut *mut ::core::ffi::c_void, size: usize) -> cudaError_t {
        (culib().cudaMalloc)(devPtr, size)
    }
    pub unsafe fn cudaMalloc3D(
        pitchedDevPtr: *mut cudaPitchedPtr,
        extent: cudaExtent,
    ) -> cudaError_t {
        (culib().cudaMalloc3D)(pitchedDevPtr, extent)
    }
    pub unsafe fn cudaMalloc3DArray(
        array: *mut cudaArray_t,
        desc: *const cudaChannelFormatDesc,
        extent: cudaExtent,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaMalloc3DArray)(array, desc, extent, flags)
    }
    pub unsafe fn cudaMallocArray(
        array: *mut cudaArray_t,
        desc: *const cudaChannelFormatDesc,
        width: usize,
        height: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaMallocArray)(array, desc, width, height, flags)
    }
    pub unsafe fn cudaMallocAsync(
        devPtr: *mut *mut ::core::ffi::c_void,
        size: usize,
        hStream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMallocAsync)(devPtr, size, hStream)
    }
    pub unsafe fn cudaMallocFromPoolAsync(
        ptr: *mut *mut ::core::ffi::c_void,
        size: usize,
        memPool: cudaMemPool_t,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMallocFromPoolAsync)(ptr, size, memPool, stream)
    }
    pub unsafe fn cudaMallocHost(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> cudaError_t {
        (culib().cudaMallocHost)(ptr, size)
    }
    pub unsafe fn cudaMallocManaged(
        devPtr: *mut *mut ::core::ffi::c_void,
        size: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaMallocManaged)(devPtr, size, flags)
    }
    pub unsafe fn cudaMallocMipmappedArray(
        mipmappedArray: *mut cudaMipmappedArray_t,
        desc: *const cudaChannelFormatDesc,
        extent: cudaExtent,
        numLevels: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaMallocMipmappedArray)(mipmappedArray, desc, extent, numLevels, flags)
    }
    pub unsafe fn cudaMallocPitch(
        devPtr: *mut *mut ::core::ffi::c_void,
        pitch: *mut usize,
        width: usize,
        height: usize,
    ) -> cudaError_t {
        (culib().cudaMallocPitch)(devPtr, pitch, width, height)
    }
    pub unsafe fn cudaMemAdvise(
        devPtr: *const ::core::ffi::c_void,
        count: usize,
        advice: cudaMemoryAdvise,
        device: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaMemAdvise)(devPtr, count, advice, device)
    }
    #[cfg(any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaMemAdvise_v2(
        devPtr: *const ::core::ffi::c_void,
        count: usize,
        advice: cudaMemoryAdvise,
        location: cudaMemLocation,
    ) -> cudaError_t {
        (culib().cudaMemAdvise_v2)(devPtr, count, advice, location)
    }
    pub unsafe fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> cudaError_t {
        (culib().cudaMemGetInfo)(free, total)
    }
    pub unsafe fn cudaMemPoolCreate(
        memPool: *mut cudaMemPool_t,
        poolProps: *const cudaMemPoolProps,
    ) -> cudaError_t {
        (culib().cudaMemPoolCreate)(memPool, poolProps)
    }
    pub unsafe fn cudaMemPoolDestroy(memPool: cudaMemPool_t) -> cudaError_t {
        (culib().cudaMemPoolDestroy)(memPool)
    }
    pub unsafe fn cudaMemPoolExportPointer(
        exportData: *mut cudaMemPoolPtrExportData,
        ptr: *mut ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaMemPoolExportPointer)(exportData, ptr)
    }
    pub unsafe fn cudaMemPoolExportToShareableHandle(
        shareableHandle: *mut ::core::ffi::c_void,
        memPool: cudaMemPool_t,
        handleType: cudaMemAllocationHandleType,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaMemPoolExportToShareableHandle)(shareableHandle, memPool, handleType, flags)
    }
    pub unsafe fn cudaMemPoolGetAccess(
        flags: *mut cudaMemAccessFlags,
        memPool: cudaMemPool_t,
        location: *mut cudaMemLocation,
    ) -> cudaError_t {
        (culib().cudaMemPoolGetAccess)(flags, memPool, location)
    }
    pub unsafe fn cudaMemPoolGetAttribute(
        memPool: cudaMemPool_t,
        attr: cudaMemPoolAttr,
        value: *mut ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaMemPoolGetAttribute)(memPool, attr, value)
    }
    pub unsafe fn cudaMemPoolImportFromShareableHandle(
        memPool: *mut cudaMemPool_t,
        shareableHandle: *mut ::core::ffi::c_void,
        handleType: cudaMemAllocationHandleType,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaMemPoolImportFromShareableHandle)(memPool, shareableHandle, handleType, flags)
    }
    pub unsafe fn cudaMemPoolImportPointer(
        ptr: *mut *mut ::core::ffi::c_void,
        memPool: cudaMemPool_t,
        exportData: *mut cudaMemPoolPtrExportData,
    ) -> cudaError_t {
        (culib().cudaMemPoolImportPointer)(ptr, memPool, exportData)
    }
    pub unsafe fn cudaMemPoolSetAccess(
        memPool: cudaMemPool_t,
        descList: *const cudaMemAccessDesc,
        count: usize,
    ) -> cudaError_t {
        (culib().cudaMemPoolSetAccess)(memPool, descList, count)
    }
    pub unsafe fn cudaMemPoolSetAttribute(
        memPool: cudaMemPool_t,
        attr: cudaMemPoolAttr,
        value: *mut ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaMemPoolSetAttribute)(memPool, attr, value)
    }
    pub unsafe fn cudaMemPoolTrimTo(memPool: cudaMemPool_t, minBytesToKeep: usize) -> cudaError_t {
        (culib().cudaMemPoolTrimTo)(memPool, minBytesToKeep)
    }
    pub unsafe fn cudaMemPrefetchAsync(
        devPtr: *const ::core::ffi::c_void,
        count: usize,
        dstDevice: ::core::ffi::c_int,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemPrefetchAsync)(devPtr, count, dstDevice, stream)
    }
    #[cfg(any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaMemPrefetchAsync_v2(
        devPtr: *const ::core::ffi::c_void,
        count: usize,
        location: cudaMemLocation,
        flags: ::core::ffi::c_uint,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemPrefetchAsync_v2)(devPtr, count, location, flags, stream)
    }
    pub unsafe fn cudaMemRangeGetAttribute(
        data: *mut ::core::ffi::c_void,
        dataSize: usize,
        attribute: cudaMemRangeAttribute,
        devPtr: *const ::core::ffi::c_void,
        count: usize,
    ) -> cudaError_t {
        (culib().cudaMemRangeGetAttribute)(data, dataSize, attribute, devPtr, count)
    }
    pub unsafe fn cudaMemRangeGetAttributes(
        data: *mut *mut ::core::ffi::c_void,
        dataSizes: *mut usize,
        attributes: *mut cudaMemRangeAttribute,
        numAttributes: usize,
        devPtr: *const ::core::ffi::c_void,
        count: usize,
    ) -> cudaError_t {
        (culib().cudaMemRangeGetAttributes)(
            data,
            dataSizes,
            attributes,
            numAttributes,
            devPtr,
            count,
        )
    }
    pub unsafe fn cudaMemcpy(
        dst: *mut ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaMemcpy)(dst, src, count, kind)
    }
    pub unsafe fn cudaMemcpy2D(
        dst: *mut ::core::ffi::c_void,
        dpitch: usize,
        src: *const ::core::ffi::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaMemcpy2D)(dst, dpitch, src, spitch, width, height, kind)
    }
    pub unsafe fn cudaMemcpy2DArrayToArray(
        dst: cudaArray_t,
        wOffsetDst: usize,
        hOffsetDst: usize,
        src: cudaArray_const_t,
        wOffsetSrc: usize,
        hOffsetSrc: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaMemcpy2DArrayToArray)(
            dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, width, height, kind,
        )
    }
    pub unsafe fn cudaMemcpy2DAsync(
        dst: *mut ::core::ffi::c_void,
        dpitch: usize,
        src: *const ::core::ffi::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemcpy2DAsync)(dst, dpitch, src, spitch, width, height, kind, stream)
    }
    pub unsafe fn cudaMemcpy2DFromArray(
        dst: *mut ::core::ffi::c_void,
        dpitch: usize,
        src: cudaArray_const_t,
        wOffset: usize,
        hOffset: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaMemcpy2DFromArray)(dst, dpitch, src, wOffset, hOffset, width, height, kind)
    }
    pub unsafe fn cudaMemcpy2DFromArrayAsync(
        dst: *mut ::core::ffi::c_void,
        dpitch: usize,
        src: cudaArray_const_t,
        wOffset: usize,
        hOffset: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemcpy2DFromArrayAsync)(
            dst, dpitch, src, wOffset, hOffset, width, height, kind, stream,
        )
    }
    pub unsafe fn cudaMemcpy2DToArray(
        dst: cudaArray_t,
        wOffset: usize,
        hOffset: usize,
        src: *const ::core::ffi::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaMemcpy2DToArray)(dst, wOffset, hOffset, src, spitch, width, height, kind)
    }
    pub unsafe fn cudaMemcpy2DToArrayAsync(
        dst: cudaArray_t,
        wOffset: usize,
        hOffset: usize,
        src: *const ::core::ffi::c_void,
        spitch: usize,
        width: usize,
        height: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemcpy2DToArrayAsync)(
            dst, wOffset, hOffset, src, spitch, width, height, kind, stream,
        )
    }
    pub unsafe fn cudaMemcpy3D(p: *const cudaMemcpy3DParms) -> cudaError_t {
        (culib().cudaMemcpy3D)(p)
    }
    pub unsafe fn cudaMemcpy3DAsync(
        p: *const cudaMemcpy3DParms,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemcpy3DAsync)(p, stream)
    }
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn cudaMemcpy3DBatchAsync(
        numOps: usize,
        opList: *mut cudaMemcpy3DBatchOp,
        failIdx: *mut usize,
        flags: ::core::ffi::c_ulonglong,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemcpy3DBatchAsync)(numOps, opList, failIdx, flags, stream)
    }
    pub unsafe fn cudaMemcpy3DPeer(p: *const cudaMemcpy3DPeerParms) -> cudaError_t {
        (culib().cudaMemcpy3DPeer)(p)
    }
    pub unsafe fn cudaMemcpy3DPeerAsync(
        p: *const cudaMemcpy3DPeerParms,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemcpy3DPeerAsync)(p, stream)
    }
    pub unsafe fn cudaMemcpyArrayToArray(
        dst: cudaArray_t,
        wOffsetDst: usize,
        hOffsetDst: usize,
        src: cudaArray_const_t,
        wOffsetSrc: usize,
        hOffsetSrc: usize,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaMemcpyArrayToArray)(
            dst, wOffsetDst, hOffsetDst, src, wOffsetSrc, hOffsetSrc, count, kind,
        )
    }
    pub unsafe fn cudaMemcpyAsync(
        dst: *mut ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemcpyAsync)(dst, src, count, kind, stream)
    }
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn cudaMemcpyBatchAsync(
        dsts: *mut *mut ::core::ffi::c_void,
        srcs: *mut *mut ::core::ffi::c_void,
        sizes: *mut usize,
        count: usize,
        attrs: *mut cudaMemcpyAttributes,
        attrsIdxs: *mut usize,
        numAttrs: usize,
        failIdx: *mut usize,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemcpyBatchAsync)(
            dsts, srcs, sizes, count, attrs, attrsIdxs, numAttrs, failIdx, stream,
        )
    }
    pub unsafe fn cudaMemcpyFromArray(
        dst: *mut ::core::ffi::c_void,
        src: cudaArray_const_t,
        wOffset: usize,
        hOffset: usize,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaMemcpyFromArray)(dst, src, wOffset, hOffset, count, kind)
    }
    pub unsafe fn cudaMemcpyFromArrayAsync(
        dst: *mut ::core::ffi::c_void,
        src: cudaArray_const_t,
        wOffset: usize,
        hOffset: usize,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemcpyFromArrayAsync)(dst, src, wOffset, hOffset, count, kind, stream)
    }
    pub unsafe fn cudaMemcpyFromSymbol(
        dst: *mut ::core::ffi::c_void,
        symbol: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaMemcpyFromSymbol)(dst, symbol, count, offset, kind)
    }
    pub unsafe fn cudaMemcpyFromSymbolAsync(
        dst: *mut ::core::ffi::c_void,
        symbol: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemcpyFromSymbolAsync)(dst, symbol, count, offset, kind, stream)
    }
    pub unsafe fn cudaMemcpyPeer(
        dst: *mut ::core::ffi::c_void,
        dstDevice: ::core::ffi::c_int,
        src: *const ::core::ffi::c_void,
        srcDevice: ::core::ffi::c_int,
        count: usize,
    ) -> cudaError_t {
        (culib().cudaMemcpyPeer)(dst, dstDevice, src, srcDevice, count)
    }
    pub unsafe fn cudaMemcpyPeerAsync(
        dst: *mut ::core::ffi::c_void,
        dstDevice: ::core::ffi::c_int,
        src: *const ::core::ffi::c_void,
        srcDevice: ::core::ffi::c_int,
        count: usize,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemcpyPeerAsync)(dst, dstDevice, src, srcDevice, count, stream)
    }
    pub unsafe fn cudaMemcpyToArray(
        dst: cudaArray_t,
        wOffset: usize,
        hOffset: usize,
        src: *const ::core::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaMemcpyToArray)(dst, wOffset, hOffset, src, count, kind)
    }
    pub unsafe fn cudaMemcpyToArrayAsync(
        dst: cudaArray_t,
        wOffset: usize,
        hOffset: usize,
        src: *const ::core::ffi::c_void,
        count: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemcpyToArrayAsync)(dst, wOffset, hOffset, src, count, kind, stream)
    }
    pub unsafe fn cudaMemcpyToSymbol(
        symbol: *const ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
    ) -> cudaError_t {
        (culib().cudaMemcpyToSymbol)(symbol, src, count, offset, kind)
    }
    pub unsafe fn cudaMemcpyToSymbolAsync(
        symbol: *const ::core::ffi::c_void,
        src: *const ::core::ffi::c_void,
        count: usize,
        offset: usize,
        kind: cudaMemcpyKind,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemcpyToSymbolAsync)(symbol, src, count, offset, kind, stream)
    }
    pub unsafe fn cudaMemset(
        devPtr: *mut ::core::ffi::c_void,
        value: ::core::ffi::c_int,
        count: usize,
    ) -> cudaError_t {
        (culib().cudaMemset)(devPtr, value, count)
    }
    pub unsafe fn cudaMemset2D(
        devPtr: *mut ::core::ffi::c_void,
        pitch: usize,
        value: ::core::ffi::c_int,
        width: usize,
        height: usize,
    ) -> cudaError_t {
        (culib().cudaMemset2D)(devPtr, pitch, value, width, height)
    }
    pub unsafe fn cudaMemset2DAsync(
        devPtr: *mut ::core::ffi::c_void,
        pitch: usize,
        value: ::core::ffi::c_int,
        width: usize,
        height: usize,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemset2DAsync)(devPtr, pitch, value, width, height, stream)
    }
    pub unsafe fn cudaMemset3D(
        pitchedDevPtr: cudaPitchedPtr,
        value: ::core::ffi::c_int,
        extent: cudaExtent,
    ) -> cudaError_t {
        (culib().cudaMemset3D)(pitchedDevPtr, value, extent)
    }
    pub unsafe fn cudaMemset3DAsync(
        pitchedDevPtr: cudaPitchedPtr,
        value: ::core::ffi::c_int,
        extent: cudaExtent,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemset3DAsync)(pitchedDevPtr, value, extent, stream)
    }
    pub unsafe fn cudaMemsetAsync(
        devPtr: *mut ::core::ffi::c_void,
        value: ::core::ffi::c_int,
        count: usize,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaMemsetAsync)(devPtr, value, count, stream)
    }
    #[cfg(any(
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
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaMipmappedArrayGetMemoryRequirements(
        memoryRequirements: *mut cudaArrayMemoryRequirements,
        mipmap: cudaMipmappedArray_t,
        device: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaMipmappedArrayGetMemoryRequirements)(memoryRequirements, mipmap, device)
    }
    pub unsafe fn cudaMipmappedArrayGetSparseProperties(
        sparseProperties: *mut cudaArraySparseProperties,
        mipmap: cudaMipmappedArray_t,
    ) -> cudaError_t {
        (culib().cudaMipmappedArrayGetSparseProperties)(sparseProperties, mipmap)
    }
    pub unsafe fn cudaOccupancyAvailableDynamicSMemPerBlock(
        dynamicSmemSize: *mut usize,
        func: *const ::core::ffi::c_void,
        numBlocks: ::core::ffi::c_int,
        blockSize: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaOccupancyAvailableDynamicSMemPerBlock)(
            dynamicSmemSize,
            func,
            numBlocks,
            blockSize,
        )
    }
    pub unsafe fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks: *mut ::core::ffi::c_int,
        func: *const ::core::ffi::c_void,
        blockSize: ::core::ffi::c_int,
        dynamicSMemSize: usize,
    ) -> cudaError_t {
        (culib().cudaOccupancyMaxActiveBlocksPerMultiprocessor)(
            numBlocks,
            func,
            blockSize,
            dynamicSMemSize,
        )
    }
    pub unsafe fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks: *mut ::core::ffi::c_int,
        func: *const ::core::ffi::c_void,
        blockSize: ::core::ffi::c_int,
        dynamicSMemSize: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(
            numBlocks,
            func,
            blockSize,
            dynamicSMemSize,
            flags,
        )
    }
    #[cfg(any(
        feature = "cuda-11080",
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
    pub unsafe fn cudaOccupancyMaxActiveClusters(
        numClusters: *mut ::core::ffi::c_int,
        func: *const ::core::ffi::c_void,
        launchConfig: *const cudaLaunchConfig_t,
    ) -> cudaError_t {
        (culib().cudaOccupancyMaxActiveClusters)(numClusters, func, launchConfig)
    }
    #[cfg(any(
        feature = "cuda-11080",
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
    pub unsafe fn cudaOccupancyMaxPotentialClusterSize(
        clusterSize: *mut ::core::ffi::c_int,
        func: *const ::core::ffi::c_void,
        launchConfig: *const cudaLaunchConfig_t,
    ) -> cudaError_t {
        (culib().cudaOccupancyMaxPotentialClusterSize)(clusterSize, func, launchConfig)
    }
    pub unsafe fn cudaPeekAtLastError() -> cudaError_t {
        (culib().cudaPeekAtLastError)()
    }
    pub unsafe fn cudaPointerGetAttributes(
        attributes: *mut cudaPointerAttributes,
        ptr: *const ::core::ffi::c_void,
    ) -> cudaError_t {
        (culib().cudaPointerGetAttributes)(attributes, ptr)
    }
    pub unsafe fn cudaProfilerStop() -> cudaError_t {
        (culib().cudaProfilerStop)()
    }
    pub unsafe fn cudaRuntimeGetVersion(runtimeVersion: *mut ::core::ffi::c_int) -> cudaError_t {
        (culib().cudaRuntimeGetVersion)(runtimeVersion)
    }
    pub unsafe fn cudaSetDevice(device: ::core::ffi::c_int) -> cudaError_t {
        (culib().cudaSetDevice)(device)
    }
    pub unsafe fn cudaSetDeviceFlags(flags: ::core::ffi::c_uint) -> cudaError_t {
        (culib().cudaSetDeviceFlags)(flags)
    }
    pub unsafe fn cudaSetDoubleForDevice(d: *mut f64) -> cudaError_t {
        (culib().cudaSetDoubleForDevice)(d)
    }
    pub unsafe fn cudaSetDoubleForHost(d: *mut f64) -> cudaError_t {
        (culib().cudaSetDoubleForHost)(d)
    }
    pub unsafe fn cudaSetValidDevices(
        device_arr: *mut ::core::ffi::c_int,
        len: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaSetValidDevices)(device_arr, len)
    }
    pub unsafe fn cudaSignalExternalSemaphoresAsync_v2(
        extSemArray: *const cudaExternalSemaphore_t,
        paramsArray: *const cudaExternalSemaphoreSignalParams,
        numExtSems: ::core::ffi::c_uint,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaSignalExternalSemaphoresAsync_v2)(extSemArray, paramsArray, numExtSems, stream)
    }
    pub unsafe fn cudaStreamAddCallback(
        stream: cudaStream_t,
        callback: cudaStreamCallback_t,
        userData: *mut ::core::ffi::c_void,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaStreamAddCallback)(stream, callback, userData, flags)
    }
    pub unsafe fn cudaStreamAttachMemAsync(
        stream: cudaStream_t,
        devPtr: *mut ::core::ffi::c_void,
        length: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaStreamAttachMemAsync)(stream, devPtr, length, flags)
    }
    pub unsafe fn cudaStreamBeginCapture(
        stream: cudaStream_t,
        mode: cudaStreamCaptureMode,
    ) -> cudaError_t {
        (culib().cudaStreamBeginCapture)(stream, mode)
    }
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaStreamBeginCaptureToGraph(
        stream: cudaStream_t,
        graph: cudaGraph_t,
        dependencies: *const cudaGraphNode_t,
        dependencyData: *const cudaGraphEdgeData,
        numDependencies: usize,
        mode: cudaStreamCaptureMode,
    ) -> cudaError_t {
        (culib().cudaStreamBeginCaptureToGraph)(
            stream,
            graph,
            dependencies,
            dependencyData,
            numDependencies,
            mode,
        )
    }
    pub unsafe fn cudaStreamCopyAttributes(dst: cudaStream_t, src: cudaStream_t) -> cudaError_t {
        (culib().cudaStreamCopyAttributes)(dst, src)
    }
    pub unsafe fn cudaStreamCreate(pStream: *mut cudaStream_t) -> cudaError_t {
        (culib().cudaStreamCreate)(pStream)
    }
    pub unsafe fn cudaStreamCreateWithFlags(
        pStream: *mut cudaStream_t,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaStreamCreateWithFlags)(pStream, flags)
    }
    pub unsafe fn cudaStreamCreateWithPriority(
        pStream: *mut cudaStream_t,
        flags: ::core::ffi::c_uint,
        priority: ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaStreamCreateWithPriority)(pStream, flags, priority)
    }
    pub unsafe fn cudaStreamDestroy(stream: cudaStream_t) -> cudaError_t {
        (culib().cudaStreamDestroy)(stream)
    }
    pub unsafe fn cudaStreamEndCapture(
        stream: cudaStream_t,
        pGraph: *mut cudaGraph_t,
    ) -> cudaError_t {
        (culib().cudaStreamEndCapture)(stream, pGraph)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    ))]
    pub unsafe fn cudaStreamGetAttribute(
        hStream: cudaStream_t,
        attr: cudaStreamAttrID,
        value_out: *mut cudaStreamAttrValue,
    ) -> cudaError_t {
        (culib().cudaStreamGetAttribute)(hStream, attr, value_out)
    }
    #[cfg(any(
        feature = "cuda-11080",
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
    pub unsafe fn cudaStreamGetAttribute(
        hStream: cudaStream_t,
        attr: cudaLaunchAttributeID,
        value_out: *mut cudaLaunchAttributeValue,
    ) -> cudaError_t {
        (culib().cudaStreamGetAttribute)(hStream, attr, value_out)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn cudaStreamGetCaptureInfo(
        stream: cudaStream_t,
        pCaptureStatus: *mut cudaStreamCaptureStatus,
        pId: *mut ::core::ffi::c_ulonglong,
    ) -> cudaError_t {
        (culib().cudaStreamGetCaptureInfo)(stream, pCaptureStatus, pId)
    }
    pub unsafe fn cudaStreamGetCaptureInfo_v2(
        stream: cudaStream_t,
        captureStatus_out: *mut cudaStreamCaptureStatus,
        id_out: *mut ::core::ffi::c_ulonglong,
        graph_out: *mut cudaGraph_t,
        dependencies_out: *mut *const cudaGraphNode_t,
        numDependencies_out: *mut usize,
    ) -> cudaError_t {
        (culib().cudaStreamGetCaptureInfo_v2)(
            stream,
            captureStatus_out,
            id_out,
            graph_out,
            dependencies_out,
            numDependencies_out,
        )
    }
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaStreamGetCaptureInfo_v3(
        stream: cudaStream_t,
        captureStatus_out: *mut cudaStreamCaptureStatus,
        id_out: *mut ::core::ffi::c_ulonglong,
        graph_out: *mut cudaGraph_t,
        dependencies_out: *mut *const cudaGraphNode_t,
        edgeData_out: *mut *const cudaGraphEdgeData,
        numDependencies_out: *mut usize,
    ) -> cudaError_t {
        (culib().cudaStreamGetCaptureInfo_v3)(
            stream,
            captureStatus_out,
            id_out,
            graph_out,
            dependencies_out,
            edgeData_out,
            numDependencies_out,
        )
    }
    #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
    pub unsafe fn cudaStreamGetDevice(
        hStream: cudaStream_t,
        device: *mut ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaStreamGetDevice)(hStream, device)
    }
    pub unsafe fn cudaStreamGetFlags(
        hStream: cudaStream_t,
        flags: *mut ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaStreamGetFlags)(hStream, flags)
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
    pub unsafe fn cudaStreamGetId(
        hStream: cudaStream_t,
        streamId: *mut ::core::ffi::c_ulonglong,
    ) -> cudaError_t {
        (culib().cudaStreamGetId)(hStream, streamId)
    }
    pub unsafe fn cudaStreamGetPriority(
        hStream: cudaStream_t,
        priority: *mut ::core::ffi::c_int,
    ) -> cudaError_t {
        (culib().cudaStreamGetPriority)(hStream, priority)
    }
    pub unsafe fn cudaStreamIsCapturing(
        stream: cudaStream_t,
        pCaptureStatus: *mut cudaStreamCaptureStatus,
    ) -> cudaError_t {
        (culib().cudaStreamIsCapturing)(stream, pCaptureStatus)
    }
    pub unsafe fn cudaStreamQuery(stream: cudaStream_t) -> cudaError_t {
        (culib().cudaStreamQuery)(stream)
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    ))]
    pub unsafe fn cudaStreamSetAttribute(
        hStream: cudaStream_t,
        attr: cudaStreamAttrID,
        value: *const cudaStreamAttrValue,
    ) -> cudaError_t {
        (culib().cudaStreamSetAttribute)(hStream, attr, value)
    }
    #[cfg(any(
        feature = "cuda-11080",
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
    pub unsafe fn cudaStreamSetAttribute(
        hStream: cudaStream_t,
        attr: cudaLaunchAttributeID,
        value: *const cudaLaunchAttributeValue,
    ) -> cudaError_t {
        (culib().cudaStreamSetAttribute)(hStream, attr, value)
    }
    pub unsafe fn cudaStreamSynchronize(stream: cudaStream_t) -> cudaError_t {
        (culib().cudaStreamSynchronize)(stream)
    }
    pub unsafe fn cudaStreamUpdateCaptureDependencies(
        stream: cudaStream_t,
        dependencies: *mut cudaGraphNode_t,
        numDependencies: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaStreamUpdateCaptureDependencies)(stream, dependencies, numDependencies, flags)
    }
    #[cfg(any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080",
        feature = "cuda-12090"
    ))]
    pub unsafe fn cudaStreamUpdateCaptureDependencies_v2(
        stream: cudaStream_t,
        dependencies: *mut cudaGraphNode_t,
        dependencyData: *const cudaGraphEdgeData,
        numDependencies: usize,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaStreamUpdateCaptureDependencies_v2)(
            stream,
            dependencies,
            dependencyData,
            numDependencies,
            flags,
        )
    }
    pub unsafe fn cudaStreamWaitEvent(
        stream: cudaStream_t,
        event: cudaEvent_t,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaStreamWaitEvent)(stream, event, flags)
    }
    pub unsafe fn cudaThreadExchangeStreamCaptureMode(
        mode: *mut cudaStreamCaptureMode,
    ) -> cudaError_t {
        (culib().cudaThreadExchangeStreamCaptureMode)(mode)
    }
    pub unsafe fn cudaThreadExit() -> cudaError_t {
        (culib().cudaThreadExit)()
    }
    pub unsafe fn cudaThreadGetCacheConfig(pCacheConfig: *mut cudaFuncCache) -> cudaError_t {
        (culib().cudaThreadGetCacheConfig)(pCacheConfig)
    }
    pub unsafe fn cudaThreadGetLimit(pValue: *mut usize, limit: cudaLimit) -> cudaError_t {
        (culib().cudaThreadGetLimit)(pValue, limit)
    }
    pub unsafe fn cudaThreadSetCacheConfig(cacheConfig: cudaFuncCache) -> cudaError_t {
        (culib().cudaThreadSetCacheConfig)(cacheConfig)
    }
    pub unsafe fn cudaThreadSetLimit(limit: cudaLimit, value: usize) -> cudaError_t {
        (culib().cudaThreadSetLimit)(limit, value)
    }
    pub unsafe fn cudaThreadSynchronize() -> cudaError_t {
        (culib().cudaThreadSynchronize)()
    }
    #[cfg(any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    ))]
    pub unsafe fn cudaUnbindTexture(texref: *const textureReference) -> cudaError_t {
        (culib().cudaUnbindTexture)(texref)
    }
    pub unsafe fn cudaUserObjectCreate(
        object_out: *mut cudaUserObject_t,
        ptr: *mut ::core::ffi::c_void,
        destroy: cudaHostFn_t,
        initialRefcount: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaUserObjectCreate)(object_out, ptr, destroy, initialRefcount, flags)
    }
    pub unsafe fn cudaUserObjectRelease(
        object: cudaUserObject_t,
        count: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaUserObjectRelease)(object, count)
    }
    pub unsafe fn cudaUserObjectRetain(
        object: cudaUserObject_t,
        count: ::core::ffi::c_uint,
    ) -> cudaError_t {
        (culib().cudaUserObjectRetain)(object, count)
    }
    pub unsafe fn cudaWaitExternalSemaphoresAsync_v2(
        extSemArray: *const cudaExternalSemaphore_t,
        paramsArray: *const cudaExternalSemaphoreWaitParams,
        numExtSems: ::core::ffi::c_uint,
        stream: cudaStream_t,
    ) -> cudaError_t {
        (culib().cudaWaitExternalSemaphoresAsync_v2)(extSemArray, paramsArray, numExtSems, stream)
    }
    pub struct Lib {
        __library: ::libloading::Library,
        pub cudaArrayGetInfo: unsafe extern "C" fn(
            desc: *mut cudaChannelFormatDesc,
            extent: *mut cudaExtent,
            flags: *mut ::core::ffi::c_uint,
            array: cudaArray_t,
        ) -> cudaError_t,
        #[cfg(any(
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
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaArrayGetMemoryRequirements: unsafe extern "C" fn(
            memoryRequirements: *mut cudaArrayMemoryRequirements,
            array: cudaArray_t,
            device: ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaArrayGetPlane: unsafe extern "C" fn(
            pPlaneArray: *mut cudaArray_t,
            hArray: cudaArray_t,
            planeIdx: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaArrayGetSparseProperties: unsafe extern "C" fn(
            sparseProperties: *mut cudaArraySparseProperties,
            array: cudaArray_t,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        ))]
        pub cudaBindSurfaceToArray: unsafe extern "C" fn(
            surfref: *const surfaceReference,
            array: cudaArray_const_t,
            desc: *const cudaChannelFormatDesc,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        ))]
        pub cudaBindTexture: unsafe extern "C" fn(
            offset: *mut usize,
            texref: *const textureReference,
            devPtr: *const ::core::ffi::c_void,
            desc: *const cudaChannelFormatDesc,
            size: usize,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        ))]
        pub cudaBindTexture2D: unsafe extern "C" fn(
            offset: *mut usize,
            texref: *const textureReference,
            devPtr: *const ::core::ffi::c_void,
            desc: *const cudaChannelFormatDesc,
            width: usize,
            height: usize,
            pitch: usize,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        ))]
        pub cudaBindTextureToArray: unsafe extern "C" fn(
            texref: *const textureReference,
            array: cudaArray_const_t,
            desc: *const cudaChannelFormatDesc,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        ))]
        pub cudaBindTextureToMipmappedArray: unsafe extern "C" fn(
            texref: *const textureReference,
            mipmappedArray: cudaMipmappedArray_const_t,
            desc: *const cudaChannelFormatDesc,
        ) -> cudaError_t,
        pub cudaChooseDevice: unsafe extern "C" fn(
            device: *mut ::core::ffi::c_int,
            prop: *const cudaDeviceProp,
        ) -> cudaError_t,
        pub cudaCreateChannelDesc: unsafe extern "C" fn(
            x: ::core::ffi::c_int,
            y: ::core::ffi::c_int,
            z: ::core::ffi::c_int,
            w: ::core::ffi::c_int,
            f: cudaChannelFormatKind,
        ) -> cudaChannelFormatDesc,
        pub cudaCreateSurfaceObject: unsafe extern "C" fn(
            pSurfObject: *mut cudaSurfaceObject_t,
            pResDesc: *const cudaResourceDesc,
        ) -> cudaError_t,
        pub cudaCreateTextureObject: unsafe extern "C" fn(
            pTexObject: *mut cudaTextureObject_t,
            pResDesc: *const cudaResourceDesc,
            pTexDesc: *const cudaTextureDesc,
            pResViewDesc: *const cudaResourceViewDesc,
        ) -> cudaError_t,
        #[cfg(any(feature = "cuda-11080"))]
        pub cudaCreateTextureObject_v2: unsafe extern "C" fn(
            pTexObject: *mut cudaTextureObject_t,
            pResDesc: *const cudaResourceDesc,
            pTexDesc: *const cudaTextureDesc_v2,
            pResViewDesc: *const cudaResourceViewDesc,
        ) -> cudaError_t,
        pub cudaCtxResetPersistingL2Cache: unsafe extern "C" fn() -> cudaError_t,
        pub cudaDestroyExternalMemory:
            unsafe extern "C" fn(extMem: cudaExternalMemory_t) -> cudaError_t,
        pub cudaDestroyExternalSemaphore:
            unsafe extern "C" fn(extSem: cudaExternalSemaphore_t) -> cudaError_t,
        pub cudaDestroySurfaceObject:
            unsafe extern "C" fn(surfObject: cudaSurfaceObject_t) -> cudaError_t,
        pub cudaDestroyTextureObject:
            unsafe extern "C" fn(texObject: cudaTextureObject_t) -> cudaError_t,
        pub cudaDeviceCanAccessPeer: unsafe extern "C" fn(
            canAccessPeer: *mut ::core::ffi::c_int,
            device: ::core::ffi::c_int,
            peerDevice: ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaDeviceDisablePeerAccess:
            unsafe extern "C" fn(peerDevice: ::core::ffi::c_int) -> cudaError_t,
        pub cudaDeviceEnablePeerAccess: unsafe extern "C" fn(
            peerDevice: ::core::ffi::c_int,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaDeviceFlushGPUDirectRDMAWrites: unsafe extern "C" fn(
            target: cudaFlushGPUDirectRDMAWritesTarget,
            scope: cudaFlushGPUDirectRDMAWritesScope,
        ) -> cudaError_t,
        pub cudaDeviceGetAttribute: unsafe extern "C" fn(
            value: *mut ::core::ffi::c_int,
            attr: cudaDeviceAttr,
            device: ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaDeviceGetByPCIBusId: unsafe extern "C" fn(
            device: *mut ::core::ffi::c_int,
            pciBusId: *const ::core::ffi::c_char,
        ) -> cudaError_t,
        pub cudaDeviceGetCacheConfig:
            unsafe extern "C" fn(pCacheConfig: *mut cudaFuncCache) -> cudaError_t,
        pub cudaDeviceGetDefaultMemPool: unsafe extern "C" fn(
            memPool: *mut cudaMemPool_t,
            device: ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaDeviceGetGraphMemAttribute: unsafe extern "C" fn(
            device: ::core::ffi::c_int,
            attr: cudaGraphMemAttributeType,
            value: *mut ::core::ffi::c_void,
        ) -> cudaError_t,
        pub cudaDeviceGetLimit:
            unsafe extern "C" fn(pValue: *mut usize, limit: cudaLimit) -> cudaError_t,
        pub cudaDeviceGetMemPool: unsafe extern "C" fn(
            memPool: *mut cudaMemPool_t,
            device: ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaDeviceGetP2PAttribute: unsafe extern "C" fn(
            value: *mut ::core::ffi::c_int,
            attr: cudaDeviceP2PAttr,
            srcDevice: ::core::ffi::c_int,
            dstDevice: ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaDeviceGetPCIBusId: unsafe extern "C" fn(
            pciBusId: *mut ::core::ffi::c_char,
            len: ::core::ffi::c_int,
            device: ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaDeviceGetSharedMemConfig:
            unsafe extern "C" fn(pConfig: *mut cudaSharedMemConfig) -> cudaError_t,
        pub cudaDeviceGetStreamPriorityRange: unsafe extern "C" fn(
            leastPriority: *mut ::core::ffi::c_int,
            greatestPriority: *mut ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaDeviceGetTexture1DLinearMaxWidth: unsafe extern "C" fn(
            maxWidthInElements: *mut usize,
            fmtDesc: *const cudaChannelFormatDesc,
            device: ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaDeviceGraphMemTrim: unsafe extern "C" fn(device: ::core::ffi::c_int) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaDeviceRegisterAsyncNotification: unsafe extern "C" fn(
            device: ::core::ffi::c_int,
            callbackFunc: cudaAsyncCallback,
            userData: *mut ::core::ffi::c_void,
            callback: *mut cudaAsyncCallbackHandle_t,
        ) -> cudaError_t,
        pub cudaDeviceReset: unsafe extern "C" fn() -> cudaError_t,
        pub cudaDeviceSetCacheConfig:
            unsafe extern "C" fn(cacheConfig: cudaFuncCache) -> cudaError_t,
        pub cudaDeviceSetGraphMemAttribute: unsafe extern "C" fn(
            device: ::core::ffi::c_int,
            attr: cudaGraphMemAttributeType,
            value: *mut ::core::ffi::c_void,
        ) -> cudaError_t,
        pub cudaDeviceSetLimit: unsafe extern "C" fn(limit: cudaLimit, value: usize) -> cudaError_t,
        pub cudaDeviceSetMemPool:
            unsafe extern "C" fn(device: ::core::ffi::c_int, memPool: cudaMemPool_t) -> cudaError_t,
        pub cudaDeviceSetSharedMemConfig:
            unsafe extern "C" fn(config: cudaSharedMemConfig) -> cudaError_t,
        pub cudaDeviceSynchronize: unsafe extern "C" fn() -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaDeviceUnregisterAsyncNotification: unsafe extern "C" fn(
            device: ::core::ffi::c_int,
            callback: cudaAsyncCallbackHandle_t,
        ) -> cudaError_t,
        pub cudaDriverGetVersion:
            unsafe extern "C" fn(driverVersion: *mut ::core::ffi::c_int) -> cudaError_t,
        pub cudaEventCreate: unsafe extern "C" fn(event: *mut cudaEvent_t) -> cudaError_t,
        pub cudaEventCreateWithFlags: unsafe extern "C" fn(
            event: *mut cudaEvent_t,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaEventDestroy: unsafe extern "C" fn(event: cudaEvent_t) -> cudaError_t,
        pub cudaEventElapsedTime:
            unsafe extern "C" fn(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t,
        #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
        pub cudaEventElapsedTime_v2:
            unsafe extern "C" fn(ms: *mut f32, start: cudaEvent_t, end: cudaEvent_t) -> cudaError_t,
        pub cudaEventQuery: unsafe extern "C" fn(event: cudaEvent_t) -> cudaError_t,
        pub cudaEventRecord:
            unsafe extern "C" fn(event: cudaEvent_t, stream: cudaStream_t) -> cudaError_t,
        pub cudaEventRecordWithFlags: unsafe extern "C" fn(
            event: cudaEvent_t,
            stream: cudaStream_t,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaEventSynchronize: unsafe extern "C" fn(event: cudaEvent_t) -> cudaError_t,
        pub cudaExternalMemoryGetMappedBuffer: unsafe extern "C" fn(
            devPtr: *mut *mut ::core::ffi::c_void,
            extMem: cudaExternalMemory_t,
            bufferDesc: *const cudaExternalMemoryBufferDesc,
        ) -> cudaError_t,
        pub cudaExternalMemoryGetMappedMipmappedArray: unsafe extern "C" fn(
            mipmap: *mut cudaMipmappedArray_t,
            extMem: cudaExternalMemory_t,
            mipmapDesc: *const cudaExternalMemoryMipmappedArrayDesc,
        ) -> cudaError_t,
        pub cudaFree: unsafe extern "C" fn(devPtr: *mut ::core::ffi::c_void) -> cudaError_t,
        pub cudaFreeArray: unsafe extern "C" fn(array: cudaArray_t) -> cudaError_t,
        pub cudaFreeAsync: unsafe extern "C" fn(
            devPtr: *mut ::core::ffi::c_void,
            hStream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaFreeHost: unsafe extern "C" fn(ptr: *mut ::core::ffi::c_void) -> cudaError_t,
        pub cudaFreeMipmappedArray:
            unsafe extern "C" fn(mipmappedArray: cudaMipmappedArray_t) -> cudaError_t,
        pub cudaFuncGetAttributes: unsafe extern "C" fn(
            attr: *mut cudaFuncAttributes,
            func: *const ::core::ffi::c_void,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaFuncGetName: unsafe extern "C" fn(
            name: *mut *const ::core::ffi::c_char,
            func: *const ::core::ffi::c_void,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaFuncGetParamInfo: unsafe extern "C" fn(
            func: *const ::core::ffi::c_void,
            paramIndex: usize,
            paramOffset: *mut usize,
            paramSize: *mut usize,
        ) -> cudaError_t,
        pub cudaFuncSetAttribute: unsafe extern "C" fn(
            func: *const ::core::ffi::c_void,
            attr: cudaFuncAttribute,
            value: ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaFuncSetCacheConfig: unsafe extern "C" fn(
            func: *const ::core::ffi::c_void,
            cacheConfig: cudaFuncCache,
        ) -> cudaError_t,
        pub cudaFuncSetSharedMemConfig: unsafe extern "C" fn(
            func: *const ::core::ffi::c_void,
            config: cudaSharedMemConfig,
        ) -> cudaError_t,
        pub cudaGetChannelDesc: unsafe extern "C" fn(
            desc: *mut cudaChannelFormatDesc,
            array: cudaArray_const_t,
        ) -> cudaError_t,
        pub cudaGetDevice: unsafe extern "C" fn(device: *mut ::core::ffi::c_int) -> cudaError_t,
        pub cudaGetDeviceCount: unsafe extern "C" fn(count: *mut ::core::ffi::c_int) -> cudaError_t,
        pub cudaGetDeviceFlags:
            unsafe extern "C" fn(flags: *mut ::core::ffi::c_uint) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        ))]
        pub cudaGetDeviceProperties: unsafe extern "C" fn(
            prop: *mut cudaDeviceProp,
            device: ::core::ffi::c_int,
        ) -> cudaError_t,
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
        pub cudaGetDeviceProperties_v2: unsafe extern "C" fn(
            prop: *mut cudaDeviceProp,
            device: ::core::ffi::c_int,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        ))]
        pub cudaGetDriverEntryPoint: unsafe extern "C" fn(
            symbol: *const ::core::ffi::c_char,
            funcPtr: *mut *mut ::core::ffi::c_void,
            flags: ::core::ffi::c_ulonglong,
        ) -> cudaError_t,
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
        pub cudaGetDriverEntryPoint: unsafe extern "C" fn(
            symbol: *const ::core::ffi::c_char,
            funcPtr: *mut *mut ::core::ffi::c_void,
            flags: ::core::ffi::c_ulonglong,
            driverStatus: *mut cudaDriverEntryPointQueryResult,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaGetDriverEntryPointByVersion: unsafe extern "C" fn(
            symbol: *const ::core::ffi::c_char,
            funcPtr: *mut *mut ::core::ffi::c_void,
            cudaVersion: ::core::ffi::c_uint,
            flags: ::core::ffi::c_ulonglong,
            driverStatus: *mut cudaDriverEntryPointQueryResult,
        ) -> cudaError_t,
        pub cudaGetErrorName:
            unsafe extern "C" fn(error: cudaError_t) -> *const ::core::ffi::c_char,
        pub cudaGetErrorString:
            unsafe extern "C" fn(error: cudaError_t) -> *const ::core::ffi::c_char,
        pub cudaGetExportTable: unsafe extern "C" fn(
            ppExportTable: *mut *const ::core::ffi::c_void,
            pExportTableId: *const cudaUUID_t,
        ) -> cudaError_t,
        pub cudaGetFuncBySymbol: unsafe extern "C" fn(
            functionPtr: *mut cudaFunction_t,
            symbolPtr: *const ::core::ffi::c_void,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaGetKernel: unsafe extern "C" fn(
            kernelPtr: *mut cudaKernel_t,
            entryFuncAddr: *const ::core::ffi::c_void,
        ) -> cudaError_t,
        pub cudaGetLastError: unsafe extern "C" fn() -> cudaError_t,
        pub cudaGetMipmappedArrayLevel: unsafe extern "C" fn(
            levelArray: *mut cudaArray_t,
            mipmappedArray: cudaMipmappedArray_const_t,
            level: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaGetSurfaceObjectResourceDesc: unsafe extern "C" fn(
            pResDesc: *mut cudaResourceDesc,
            surfObject: cudaSurfaceObject_t,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        ))]
        pub cudaGetSurfaceReference: unsafe extern "C" fn(
            surfref: *mut *const surfaceReference,
            symbol: *const ::core::ffi::c_void,
        ) -> cudaError_t,
        pub cudaGetSymbolAddress: unsafe extern "C" fn(
            devPtr: *mut *mut ::core::ffi::c_void,
            symbol: *const ::core::ffi::c_void,
        ) -> cudaError_t,
        pub cudaGetSymbolSize: unsafe extern "C" fn(
            size: *mut usize,
            symbol: *const ::core::ffi::c_void,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        ))]
        pub cudaGetTextureAlignmentOffset: unsafe extern "C" fn(
            offset: *mut usize,
            texref: *const textureReference,
        ) -> cudaError_t,
        pub cudaGetTextureObjectResourceDesc: unsafe extern "C" fn(
            pResDesc: *mut cudaResourceDesc,
            texObject: cudaTextureObject_t,
        ) -> cudaError_t,
        pub cudaGetTextureObjectResourceViewDesc: unsafe extern "C" fn(
            pResViewDesc: *mut cudaResourceViewDesc,
            texObject: cudaTextureObject_t,
        ) -> cudaError_t,
        pub cudaGetTextureObjectTextureDesc: unsafe extern "C" fn(
            pTexDesc: *mut cudaTextureDesc,
            texObject: cudaTextureObject_t,
        ) -> cudaError_t,
        #[cfg(any(feature = "cuda-11080"))]
        pub cudaGetTextureObjectTextureDesc_v2: unsafe extern "C" fn(
            pTexDesc: *mut cudaTextureDesc_v2,
            texObject: cudaTextureObject_t,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        ))]
        pub cudaGetTextureReference: unsafe extern "C" fn(
            texref: *mut *const textureReference,
            symbol: *const ::core::ffi::c_void,
        ) -> cudaError_t,
        pub cudaGraphAddChildGraphNode: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
            childGraph: cudaGraph_t,
        ) -> cudaError_t,
        pub cudaGraphAddDependencies: unsafe extern "C" fn(
            graph: cudaGraph_t,
            from: *const cudaGraphNode_t,
            to: *const cudaGraphNode_t,
            numDependencies: usize,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaGraphAddDependencies_v2: unsafe extern "C" fn(
            graph: cudaGraph_t,
            from: *const cudaGraphNode_t,
            to: *const cudaGraphNode_t,
            edgeData: *const cudaGraphEdgeData,
            numDependencies: usize,
        ) -> cudaError_t,
        pub cudaGraphAddEmptyNode: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
        ) -> cudaError_t,
        pub cudaGraphAddEventRecordNode: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
            event: cudaEvent_t,
        ) -> cudaError_t,
        pub cudaGraphAddEventWaitNode: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
            event: cudaEvent_t,
        ) -> cudaError_t,
        pub cudaGraphAddExternalSemaphoresSignalNode: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
            nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
        ) -> cudaError_t,
        pub cudaGraphAddExternalSemaphoresWaitNode: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
            nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
        ) -> cudaError_t,
        pub cudaGraphAddHostNode: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
            pNodeParams: *const cudaHostNodeParams,
        ) -> cudaError_t,
        pub cudaGraphAddKernelNode: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
            pNodeParams: *const cudaKernelNodeParams,
        ) -> cudaError_t,
        pub cudaGraphAddMemAllocNode: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
            nodeParams: *mut cudaMemAllocNodeParams,
        ) -> cudaError_t,
        pub cudaGraphAddMemFreeNode: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
            dptr: *mut ::core::ffi::c_void,
        ) -> cudaError_t,
        pub cudaGraphAddMemcpyNode: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
            pCopyParams: *const cudaMemcpy3DParms,
        ) -> cudaError_t,
        pub cudaGraphAddMemcpyNode1D: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
            dst: *mut ::core::ffi::c_void,
            src: *const ::core::ffi::c_void,
            count: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaGraphAddMemcpyNodeFromSymbol: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
            dst: *mut ::core::ffi::c_void,
            symbol: *const ::core::ffi::c_void,
            count: usize,
            offset: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaGraphAddMemcpyNodeToSymbol: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
            symbol: *const ::core::ffi::c_void,
            src: *const ::core::ffi::c_void,
            count: usize,
            offset: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaGraphAddMemsetNode: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
            pMemsetParams: *const cudaMemsetParams,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaGraphAddNode: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            numDependencies: usize,
            nodeParams: *mut cudaGraphNodeParams,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaGraphAddNode_v2: unsafe extern "C" fn(
            pGraphNode: *mut cudaGraphNode_t,
            graph: cudaGraph_t,
            pDependencies: *const cudaGraphNode_t,
            dependencyData: *const cudaGraphEdgeData,
            numDependencies: usize,
            nodeParams: *mut cudaGraphNodeParams,
        ) -> cudaError_t,
        pub cudaGraphChildGraphNodeGetGraph:
            unsafe extern "C" fn(node: cudaGraphNode_t, pGraph: *mut cudaGraph_t) -> cudaError_t,
        pub cudaGraphClone: unsafe extern "C" fn(
            pGraphClone: *mut cudaGraph_t,
            originalGraph: cudaGraph_t,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaGraphConditionalHandleCreate: unsafe extern "C" fn(
            pHandle_out: *mut cudaGraphConditionalHandle,
            graph: cudaGraph_t,
            defaultLaunchValue: ::core::ffi::c_uint,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaGraphCreate: unsafe extern "C" fn(
            pGraph: *mut cudaGraph_t,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaGraphDebugDotPrint: unsafe extern "C" fn(
            graph: cudaGraph_t,
            path: *const ::core::ffi::c_char,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaGraphDestroy: unsafe extern "C" fn(graph: cudaGraph_t) -> cudaError_t,
        pub cudaGraphDestroyNode: unsafe extern "C" fn(node: cudaGraphNode_t) -> cudaError_t,
        pub cudaGraphEventRecordNodeGetEvent:
            unsafe extern "C" fn(node: cudaGraphNode_t, event_out: *mut cudaEvent_t) -> cudaError_t,
        pub cudaGraphEventRecordNodeSetEvent:
            unsafe extern "C" fn(node: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t,
        pub cudaGraphEventWaitNodeGetEvent:
            unsafe extern "C" fn(node: cudaGraphNode_t, event_out: *mut cudaEvent_t) -> cudaError_t,
        pub cudaGraphEventWaitNodeSetEvent:
            unsafe extern "C" fn(node: cudaGraphNode_t, event: cudaEvent_t) -> cudaError_t,
        pub cudaGraphExecChildGraphNodeSetParams: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            node: cudaGraphNode_t,
            childGraph: cudaGraph_t,
        ) -> cudaError_t,
        pub cudaGraphExecDestroy: unsafe extern "C" fn(graphExec: cudaGraphExec_t) -> cudaError_t,
        pub cudaGraphExecEventRecordNodeSetEvent: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            hNode: cudaGraphNode_t,
            event: cudaEvent_t,
        ) -> cudaError_t,
        pub cudaGraphExecEventWaitNodeSetEvent: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            hNode: cudaGraphNode_t,
            event: cudaEvent_t,
        ) -> cudaError_t,
        pub cudaGraphExecExternalSemaphoresSignalNodeSetParams: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            hNode: cudaGraphNode_t,
            nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
        )
            -> cudaError_t,
        pub cudaGraphExecExternalSemaphoresWaitNodeSetParams: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            hNode: cudaGraphNode_t,
            nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
        )
            -> cudaError_t,
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
        pub cudaGraphExecGetFlags: unsafe extern "C" fn(
            graphExec: cudaGraphExec_t,
            flags: *mut ::core::ffi::c_ulonglong,
        ) -> cudaError_t,
        pub cudaGraphExecHostNodeSetParams: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            node: cudaGraphNode_t,
            pNodeParams: *const cudaHostNodeParams,
        ) -> cudaError_t,
        pub cudaGraphExecKernelNodeSetParams: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            node: cudaGraphNode_t,
            pNodeParams: *const cudaKernelNodeParams,
        ) -> cudaError_t,
        pub cudaGraphExecMemcpyNodeSetParams: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            node: cudaGraphNode_t,
            pNodeParams: *const cudaMemcpy3DParms,
        ) -> cudaError_t,
        pub cudaGraphExecMemcpyNodeSetParams1D: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            node: cudaGraphNode_t,
            dst: *mut ::core::ffi::c_void,
            src: *const ::core::ffi::c_void,
            count: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaGraphExecMemcpyNodeSetParamsFromSymbol: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            node: cudaGraphNode_t,
            dst: *mut ::core::ffi::c_void,
            symbol: *const ::core::ffi::c_void,
            count: usize,
            offset: usize,
            kind: cudaMemcpyKind,
        )
            -> cudaError_t,
        pub cudaGraphExecMemcpyNodeSetParamsToSymbol: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            node: cudaGraphNode_t,
            symbol: *const ::core::ffi::c_void,
            src: *const ::core::ffi::c_void,
            count: usize,
            offset: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaGraphExecMemsetNodeSetParams: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            node: cudaGraphNode_t,
            pNodeParams: *const cudaMemsetParams,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaGraphExecNodeSetParams: unsafe extern "C" fn(
            graphExec: cudaGraphExec_t,
            node: cudaGraphNode_t,
            nodeParams: *mut cudaGraphNodeParams,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        ))]
        pub cudaGraphExecUpdate: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            hGraph: cudaGraph_t,
            hErrorNode_out: *mut cudaGraphNode_t,
            updateResult_out: *mut cudaGraphExecUpdateResult,
        ) -> cudaError_t,
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
        pub cudaGraphExecUpdate: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            hGraph: cudaGraph_t,
            resultInfo: *mut cudaGraphExecUpdateResultInfo,
        ) -> cudaError_t,
        pub cudaGraphExternalSemaphoresSignalNodeGetParams: unsafe extern "C" fn(
            hNode: cudaGraphNode_t,
            params_out: *mut cudaExternalSemaphoreSignalNodeParams,
        )
            -> cudaError_t,
        pub cudaGraphExternalSemaphoresSignalNodeSetParams: unsafe extern "C" fn(
            hNode: cudaGraphNode_t,
            nodeParams: *const cudaExternalSemaphoreSignalNodeParams,
        )
            -> cudaError_t,
        pub cudaGraphExternalSemaphoresWaitNodeGetParams: unsafe extern "C" fn(
            hNode: cudaGraphNode_t,
            params_out: *mut cudaExternalSemaphoreWaitNodeParams,
        )
            -> cudaError_t,
        pub cudaGraphExternalSemaphoresWaitNodeSetParams: unsafe extern "C" fn(
            hNode: cudaGraphNode_t,
            nodeParams: *const cudaExternalSemaphoreWaitNodeParams,
        )
            -> cudaError_t,
        pub cudaGraphGetEdges: unsafe extern "C" fn(
            graph: cudaGraph_t,
            from: *mut cudaGraphNode_t,
            to: *mut cudaGraphNode_t,
            numEdges: *mut usize,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaGraphGetEdges_v2: unsafe extern "C" fn(
            graph: cudaGraph_t,
            from: *mut cudaGraphNode_t,
            to: *mut cudaGraphNode_t,
            edgeData: *mut cudaGraphEdgeData,
            numEdges: *mut usize,
        ) -> cudaError_t,
        pub cudaGraphGetNodes: unsafe extern "C" fn(
            graph: cudaGraph_t,
            nodes: *mut cudaGraphNode_t,
            numNodes: *mut usize,
        ) -> cudaError_t,
        pub cudaGraphGetRootNodes: unsafe extern "C" fn(
            graph: cudaGraph_t,
            pRootNodes: *mut cudaGraphNode_t,
            pNumRootNodes: *mut usize,
        ) -> cudaError_t,
        pub cudaGraphHostNodeGetParams: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            pNodeParams: *mut cudaHostNodeParams,
        ) -> cudaError_t,
        pub cudaGraphHostNodeSetParams: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            pNodeParams: *const cudaHostNodeParams,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        ))]
        pub cudaGraphInstantiate: unsafe extern "C" fn(
            pGraphExec: *mut cudaGraphExec_t,
            graph: cudaGraph_t,
            pErrorNode: *mut cudaGraphNode_t,
            pLogBuffer: *mut ::core::ffi::c_char,
            bufferSize: usize,
        ) -> cudaError_t,
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
        pub cudaGraphInstantiate: unsafe extern "C" fn(
            pGraphExec: *mut cudaGraphExec_t,
            graph: cudaGraph_t,
            flags: ::core::ffi::c_ulonglong,
        ) -> cudaError_t,
        pub cudaGraphInstantiateWithFlags: unsafe extern "C" fn(
            pGraphExec: *mut cudaGraphExec_t,
            graph: cudaGraph_t,
            flags: ::core::ffi::c_ulonglong,
        ) -> cudaError_t,
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
        pub cudaGraphInstantiateWithParams: unsafe extern "C" fn(
            pGraphExec: *mut cudaGraphExec_t,
            graph: cudaGraph_t,
            instantiateParams: *mut cudaGraphInstantiateParams,
        ) -> cudaError_t,
        pub cudaGraphKernelNodeCopyAttributes:
            unsafe extern "C" fn(hSrc: cudaGraphNode_t, hDst: cudaGraphNode_t) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070"
        ))]
        pub cudaGraphKernelNodeGetAttribute: unsafe extern "C" fn(
            hNode: cudaGraphNode_t,
            attr: cudaKernelNodeAttrID,
            value_out: *mut cudaKernelNodeAttrValue,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11080",
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
        pub cudaGraphKernelNodeGetAttribute: unsafe extern "C" fn(
            hNode: cudaGraphNode_t,
            attr: cudaLaunchAttributeID,
            value_out: *mut cudaLaunchAttributeValue,
        ) -> cudaError_t,
        pub cudaGraphKernelNodeGetParams: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            pNodeParams: *mut cudaKernelNodeParams,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070"
        ))]
        pub cudaGraphKernelNodeSetAttribute: unsafe extern "C" fn(
            hNode: cudaGraphNode_t,
            attr: cudaKernelNodeAttrID,
            value: *const cudaKernelNodeAttrValue,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11080",
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
        pub cudaGraphKernelNodeSetAttribute: unsafe extern "C" fn(
            hNode: cudaGraphNode_t,
            attr: cudaLaunchAttributeID,
            value: *const cudaLaunchAttributeValue,
        ) -> cudaError_t,
        pub cudaGraphKernelNodeSetParams: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            pNodeParams: *const cudaKernelNodeParams,
        ) -> cudaError_t,
        pub cudaGraphLaunch:
            unsafe extern "C" fn(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t,
        pub cudaGraphMemAllocNodeGetParams: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            params_out: *mut cudaMemAllocNodeParams,
        ) -> cudaError_t,
        pub cudaGraphMemFreeNodeGetParams: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            dptr_out: *mut ::core::ffi::c_void,
        ) -> cudaError_t,
        pub cudaGraphMemcpyNodeGetParams: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            pNodeParams: *mut cudaMemcpy3DParms,
        ) -> cudaError_t,
        pub cudaGraphMemcpyNodeSetParams: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            pNodeParams: *const cudaMemcpy3DParms,
        ) -> cudaError_t,
        pub cudaGraphMemcpyNodeSetParams1D: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            dst: *mut ::core::ffi::c_void,
            src: *const ::core::ffi::c_void,
            count: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaGraphMemcpyNodeSetParamsFromSymbol: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            dst: *mut ::core::ffi::c_void,
            symbol: *const ::core::ffi::c_void,
            count: usize,
            offset: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaGraphMemcpyNodeSetParamsToSymbol: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            symbol: *const ::core::ffi::c_void,
            src: *const ::core::ffi::c_void,
            count: usize,
            offset: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaGraphMemsetNodeGetParams: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            pNodeParams: *mut cudaMemsetParams,
        ) -> cudaError_t,
        pub cudaGraphMemsetNodeSetParams: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            pNodeParams: *const cudaMemsetParams,
        ) -> cudaError_t,
        pub cudaGraphNodeFindInClone: unsafe extern "C" fn(
            pNode: *mut cudaGraphNode_t,
            originalNode: cudaGraphNode_t,
            clonedGraph: cudaGraph_t,
        ) -> cudaError_t,
        pub cudaGraphNodeGetDependencies: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            pDependencies: *mut cudaGraphNode_t,
            pNumDependencies: *mut usize,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaGraphNodeGetDependencies_v2: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            pDependencies: *mut cudaGraphNode_t,
            edgeData: *mut cudaGraphEdgeData,
            pNumDependencies: *mut usize,
        ) -> cudaError_t,
        pub cudaGraphNodeGetDependentNodes: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            pDependentNodes: *mut cudaGraphNode_t,
            pNumDependentNodes: *mut usize,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaGraphNodeGetDependentNodes_v2: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            pDependentNodes: *mut cudaGraphNode_t,
            edgeData: *mut cudaGraphEdgeData,
            pNumDependentNodes: *mut usize,
        ) -> cudaError_t,
        #[cfg(any(
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
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaGraphNodeGetEnabled: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            hNode: cudaGraphNode_t,
            isEnabled: *mut ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaGraphNodeGetType: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            pType: *mut cudaGraphNodeType,
        ) -> cudaError_t,
        #[cfg(any(
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
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaGraphNodeSetEnabled: unsafe extern "C" fn(
            hGraphExec: cudaGraphExec_t,
            hNode: cudaGraphNode_t,
            isEnabled: ::core::ffi::c_uint,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaGraphNodeSetParams: unsafe extern "C" fn(
            node: cudaGraphNode_t,
            nodeParams: *mut cudaGraphNodeParams,
        ) -> cudaError_t,
        pub cudaGraphReleaseUserObject: unsafe extern "C" fn(
            graph: cudaGraph_t,
            object: cudaUserObject_t,
            count: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaGraphRemoveDependencies: unsafe extern "C" fn(
            graph: cudaGraph_t,
            from: *const cudaGraphNode_t,
            to: *const cudaGraphNode_t,
            numDependencies: usize,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaGraphRemoveDependencies_v2: unsafe extern "C" fn(
            graph: cudaGraph_t,
            from: *const cudaGraphNode_t,
            to: *const cudaGraphNode_t,
            edgeData: *const cudaGraphEdgeData,
            numDependencies: usize,
        ) -> cudaError_t,
        pub cudaGraphRetainUserObject: unsafe extern "C" fn(
            graph: cudaGraph_t,
            object: cudaUserObject_t,
            count: ::core::ffi::c_uint,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaGraphUpload:
            unsafe extern "C" fn(graphExec: cudaGraphExec_t, stream: cudaStream_t) -> cudaError_t,
        pub cudaGraphicsMapResources: unsafe extern "C" fn(
            count: ::core::ffi::c_int,
            resources: *mut cudaGraphicsResource_t,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaGraphicsResourceGetMappedMipmappedArray: unsafe extern "C" fn(
            mipmappedArray: *mut cudaMipmappedArray_t,
            resource: cudaGraphicsResource_t,
        )
            -> cudaError_t,
        pub cudaGraphicsResourceGetMappedPointer: unsafe extern "C" fn(
            devPtr: *mut *mut ::core::ffi::c_void,
            size: *mut usize,
            resource: cudaGraphicsResource_t,
        ) -> cudaError_t,
        pub cudaGraphicsResourceSetMapFlags: unsafe extern "C" fn(
            resource: cudaGraphicsResource_t,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaGraphicsSubResourceGetMappedArray: unsafe extern "C" fn(
            array: *mut cudaArray_t,
            resource: cudaGraphicsResource_t,
            arrayIndex: ::core::ffi::c_uint,
            mipLevel: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaGraphicsUnmapResources: unsafe extern "C" fn(
            count: ::core::ffi::c_int,
            resources: *mut cudaGraphicsResource_t,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaGraphicsUnregisterResource:
            unsafe extern "C" fn(resource: cudaGraphicsResource_t) -> cudaError_t,
        pub cudaHostAlloc: unsafe extern "C" fn(
            pHost: *mut *mut ::core::ffi::c_void,
            size: usize,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaHostGetDevicePointer: unsafe extern "C" fn(
            pDevice: *mut *mut ::core::ffi::c_void,
            pHost: *mut ::core::ffi::c_void,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaHostGetFlags: unsafe extern "C" fn(
            pFlags: *mut ::core::ffi::c_uint,
            pHost: *mut ::core::ffi::c_void,
        ) -> cudaError_t,
        pub cudaHostRegister: unsafe extern "C" fn(
            ptr: *mut ::core::ffi::c_void,
            size: usize,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaHostUnregister: unsafe extern "C" fn(ptr: *mut ::core::ffi::c_void) -> cudaError_t,
        pub cudaImportExternalMemory: unsafe extern "C" fn(
            extMem_out: *mut cudaExternalMemory_t,
            memHandleDesc: *const cudaExternalMemoryHandleDesc,
        ) -> cudaError_t,
        pub cudaImportExternalSemaphore: unsafe extern "C" fn(
            extSem_out: *mut cudaExternalSemaphore_t,
            semHandleDesc: *const cudaExternalSemaphoreHandleDesc,
        ) -> cudaError_t,
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
        pub cudaInitDevice: unsafe extern "C" fn(
            device: ::core::ffi::c_int,
            deviceFlags: ::core::ffi::c_uint,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaIpcCloseMemHandle:
            unsafe extern "C" fn(devPtr: *mut ::core::ffi::c_void) -> cudaError_t,
        pub cudaIpcGetEventHandle: unsafe extern "C" fn(
            handle: *mut cudaIpcEventHandle_t,
            event: cudaEvent_t,
        ) -> cudaError_t,
        pub cudaIpcGetMemHandle: unsafe extern "C" fn(
            handle: *mut cudaIpcMemHandle_t,
            devPtr: *mut ::core::ffi::c_void,
        ) -> cudaError_t,
        pub cudaIpcOpenEventHandle: unsafe extern "C" fn(
            event: *mut cudaEvent_t,
            handle: cudaIpcEventHandle_t,
        ) -> cudaError_t,
        pub cudaIpcOpenMemHandle: unsafe extern "C" fn(
            devPtr: *mut *mut ::core::ffi::c_void,
            handle: cudaIpcMemHandle_t,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
        pub cudaKernelSetAttributeForDevice: unsafe extern "C" fn(
            kernel: cudaKernel_t,
            attr: cudaFuncAttribute,
            value: ::core::ffi::c_int,
            device: ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaLaunchCooperativeKernel: unsafe extern "C" fn(
            func: *const ::core::ffi::c_void,
            gridDim: dim3,
            blockDim: dim3,
            args: *mut *mut ::core::ffi::c_void,
            sharedMem: usize,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaLaunchCooperativeKernelMultiDevice: unsafe extern "C" fn(
            launchParamsList: *mut cudaLaunchParams,
            numDevices: ::core::ffi::c_uint,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaLaunchHostFunc: unsafe extern "C" fn(
            stream: cudaStream_t,
            fn_: cudaHostFn_t,
            userData: *mut ::core::ffi::c_void,
        ) -> cudaError_t,
        pub cudaLaunchKernel: unsafe extern "C" fn(
            func: *const ::core::ffi::c_void,
            gridDim: dim3,
            blockDim: dim3,
            args: *mut *mut ::core::ffi::c_void,
            sharedMem: usize,
            stream: cudaStream_t,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11080",
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
        pub cudaLaunchKernelExC: unsafe extern "C" fn(
            config: *const cudaLaunchConfig_t,
            func: *const ::core::ffi::c_void,
            args: *mut *mut ::core::ffi::c_void,
        ) -> cudaError_t,
        #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
        pub cudaLibraryEnumerateKernels: unsafe extern "C" fn(
            kernels: *mut cudaKernel_t,
            numKernels: ::core::ffi::c_uint,
            lib: cudaLibrary_t,
        ) -> cudaError_t,
        #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
        pub cudaLibraryGetGlobal: unsafe extern "C" fn(
            dptr: *mut *mut ::core::ffi::c_void,
            bytes: *mut usize,
            library: cudaLibrary_t,
            name: *const ::core::ffi::c_char,
        ) -> cudaError_t,
        #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
        pub cudaLibraryGetKernel: unsafe extern "C" fn(
            pKernel: *mut cudaKernel_t,
            library: cudaLibrary_t,
            name: *const ::core::ffi::c_char,
        ) -> cudaError_t,
        #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
        pub cudaLibraryGetKernelCount: unsafe extern "C" fn(
            count: *mut ::core::ffi::c_uint,
            lib: cudaLibrary_t,
        ) -> cudaError_t,
        #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
        pub cudaLibraryGetManaged: unsafe extern "C" fn(
            dptr: *mut *mut ::core::ffi::c_void,
            bytes: *mut usize,
            library: cudaLibrary_t,
            name: *const ::core::ffi::c_char,
        ) -> cudaError_t,
        #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
        pub cudaLibraryGetUnifiedFunction: unsafe extern "C" fn(
            fptr: *mut *mut ::core::ffi::c_void,
            library: cudaLibrary_t,
            symbol: *const ::core::ffi::c_char,
        ) -> cudaError_t,
        #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
        pub cudaLibraryLoadData: unsafe extern "C" fn(
            library: *mut cudaLibrary_t,
            code: *const ::core::ffi::c_void,
            jitOptions: *mut cudaJitOption,
            jitOptionsValues: *mut *mut ::core::ffi::c_void,
            numJitOptions: ::core::ffi::c_uint,
            libraryOptions: *mut cudaLibraryOption,
            libraryOptionValues: *mut *mut ::core::ffi::c_void,
            numLibraryOptions: ::core::ffi::c_uint,
        ) -> cudaError_t,
        #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
        pub cudaLibraryLoadFromFile: unsafe extern "C" fn(
            library: *mut cudaLibrary_t,
            fileName: *const ::core::ffi::c_char,
            jitOptions: *mut cudaJitOption,
            jitOptionsValues: *mut *mut ::core::ffi::c_void,
            numJitOptions: ::core::ffi::c_uint,
            libraryOptions: *mut cudaLibraryOption,
            libraryOptionValues: *mut *mut ::core::ffi::c_void,
            numLibraryOptions: ::core::ffi::c_uint,
        ) -> cudaError_t,
        #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
        pub cudaLibraryUnload: unsafe extern "C" fn(library: cudaLibrary_t) -> cudaError_t,
        pub cudaMalloc:
            unsafe extern "C" fn(devPtr: *mut *mut ::core::ffi::c_void, size: usize) -> cudaError_t,
        pub cudaMalloc3D: unsafe extern "C" fn(
            pitchedDevPtr: *mut cudaPitchedPtr,
            extent: cudaExtent,
        ) -> cudaError_t,
        pub cudaMalloc3DArray: unsafe extern "C" fn(
            array: *mut cudaArray_t,
            desc: *const cudaChannelFormatDesc,
            extent: cudaExtent,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaMallocArray: unsafe extern "C" fn(
            array: *mut cudaArray_t,
            desc: *const cudaChannelFormatDesc,
            width: usize,
            height: usize,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaMallocAsync: unsafe extern "C" fn(
            devPtr: *mut *mut ::core::ffi::c_void,
            size: usize,
            hStream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMallocFromPoolAsync: unsafe extern "C" fn(
            ptr: *mut *mut ::core::ffi::c_void,
            size: usize,
            memPool: cudaMemPool_t,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMallocHost:
            unsafe extern "C" fn(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> cudaError_t,
        pub cudaMallocManaged: unsafe extern "C" fn(
            devPtr: *mut *mut ::core::ffi::c_void,
            size: usize,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaMallocMipmappedArray: unsafe extern "C" fn(
            mipmappedArray: *mut cudaMipmappedArray_t,
            desc: *const cudaChannelFormatDesc,
            extent: cudaExtent,
            numLevels: ::core::ffi::c_uint,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaMallocPitch: unsafe extern "C" fn(
            devPtr: *mut *mut ::core::ffi::c_void,
            pitch: *mut usize,
            width: usize,
            height: usize,
        ) -> cudaError_t,
        pub cudaMemAdvise: unsafe extern "C" fn(
            devPtr: *const ::core::ffi::c_void,
            count: usize,
            advice: cudaMemoryAdvise,
            device: ::core::ffi::c_int,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaMemAdvise_v2: unsafe extern "C" fn(
            devPtr: *const ::core::ffi::c_void,
            count: usize,
            advice: cudaMemoryAdvise,
            location: cudaMemLocation,
        ) -> cudaError_t,
        pub cudaMemGetInfo:
            unsafe extern "C" fn(free: *mut usize, total: *mut usize) -> cudaError_t,
        pub cudaMemPoolCreate: unsafe extern "C" fn(
            memPool: *mut cudaMemPool_t,
            poolProps: *const cudaMemPoolProps,
        ) -> cudaError_t,
        pub cudaMemPoolDestroy: unsafe extern "C" fn(memPool: cudaMemPool_t) -> cudaError_t,
        pub cudaMemPoolExportPointer: unsafe extern "C" fn(
            exportData: *mut cudaMemPoolPtrExportData,
            ptr: *mut ::core::ffi::c_void,
        ) -> cudaError_t,
        pub cudaMemPoolExportToShareableHandle: unsafe extern "C" fn(
            shareableHandle: *mut ::core::ffi::c_void,
            memPool: cudaMemPool_t,
            handleType: cudaMemAllocationHandleType,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaMemPoolGetAccess: unsafe extern "C" fn(
            flags: *mut cudaMemAccessFlags,
            memPool: cudaMemPool_t,
            location: *mut cudaMemLocation,
        ) -> cudaError_t,
        pub cudaMemPoolGetAttribute: unsafe extern "C" fn(
            memPool: cudaMemPool_t,
            attr: cudaMemPoolAttr,
            value: *mut ::core::ffi::c_void,
        ) -> cudaError_t,
        pub cudaMemPoolImportFromShareableHandle: unsafe extern "C" fn(
            memPool: *mut cudaMemPool_t,
            shareableHandle: *mut ::core::ffi::c_void,
            handleType: cudaMemAllocationHandleType,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaMemPoolImportPointer: unsafe extern "C" fn(
            ptr: *mut *mut ::core::ffi::c_void,
            memPool: cudaMemPool_t,
            exportData: *mut cudaMemPoolPtrExportData,
        ) -> cudaError_t,
        pub cudaMemPoolSetAccess: unsafe extern "C" fn(
            memPool: cudaMemPool_t,
            descList: *const cudaMemAccessDesc,
            count: usize,
        ) -> cudaError_t,
        pub cudaMemPoolSetAttribute: unsafe extern "C" fn(
            memPool: cudaMemPool_t,
            attr: cudaMemPoolAttr,
            value: *mut ::core::ffi::c_void,
        ) -> cudaError_t,
        pub cudaMemPoolTrimTo:
            unsafe extern "C" fn(memPool: cudaMemPool_t, minBytesToKeep: usize) -> cudaError_t,
        pub cudaMemPrefetchAsync: unsafe extern "C" fn(
            devPtr: *const ::core::ffi::c_void,
            count: usize,
            dstDevice: ::core::ffi::c_int,
            stream: cudaStream_t,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaMemPrefetchAsync_v2: unsafe extern "C" fn(
            devPtr: *const ::core::ffi::c_void,
            count: usize,
            location: cudaMemLocation,
            flags: ::core::ffi::c_uint,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMemRangeGetAttribute: unsafe extern "C" fn(
            data: *mut ::core::ffi::c_void,
            dataSize: usize,
            attribute: cudaMemRangeAttribute,
            devPtr: *const ::core::ffi::c_void,
            count: usize,
        ) -> cudaError_t,
        pub cudaMemRangeGetAttributes: unsafe extern "C" fn(
            data: *mut *mut ::core::ffi::c_void,
            dataSizes: *mut usize,
            attributes: *mut cudaMemRangeAttribute,
            numAttributes: usize,
            devPtr: *const ::core::ffi::c_void,
            count: usize,
        ) -> cudaError_t,
        pub cudaMemcpy: unsafe extern "C" fn(
            dst: *mut ::core::ffi::c_void,
            src: *const ::core::ffi::c_void,
            count: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaMemcpy2D: unsafe extern "C" fn(
            dst: *mut ::core::ffi::c_void,
            dpitch: usize,
            src: *const ::core::ffi::c_void,
            spitch: usize,
            width: usize,
            height: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaMemcpy2DArrayToArray: unsafe extern "C" fn(
            dst: cudaArray_t,
            wOffsetDst: usize,
            hOffsetDst: usize,
            src: cudaArray_const_t,
            wOffsetSrc: usize,
            hOffsetSrc: usize,
            width: usize,
            height: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaMemcpy2DAsync: unsafe extern "C" fn(
            dst: *mut ::core::ffi::c_void,
            dpitch: usize,
            src: *const ::core::ffi::c_void,
            spitch: usize,
            width: usize,
            height: usize,
            kind: cudaMemcpyKind,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMemcpy2DFromArray: unsafe extern "C" fn(
            dst: *mut ::core::ffi::c_void,
            dpitch: usize,
            src: cudaArray_const_t,
            wOffset: usize,
            hOffset: usize,
            width: usize,
            height: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaMemcpy2DFromArrayAsync: unsafe extern "C" fn(
            dst: *mut ::core::ffi::c_void,
            dpitch: usize,
            src: cudaArray_const_t,
            wOffset: usize,
            hOffset: usize,
            width: usize,
            height: usize,
            kind: cudaMemcpyKind,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMemcpy2DToArray: unsafe extern "C" fn(
            dst: cudaArray_t,
            wOffset: usize,
            hOffset: usize,
            src: *const ::core::ffi::c_void,
            spitch: usize,
            width: usize,
            height: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaMemcpy2DToArrayAsync: unsafe extern "C" fn(
            dst: cudaArray_t,
            wOffset: usize,
            hOffset: usize,
            src: *const ::core::ffi::c_void,
            spitch: usize,
            width: usize,
            height: usize,
            kind: cudaMemcpyKind,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMemcpy3D: unsafe extern "C" fn(p: *const cudaMemcpy3DParms) -> cudaError_t,
        pub cudaMemcpy3DAsync:
            unsafe extern "C" fn(p: *const cudaMemcpy3DParms, stream: cudaStream_t) -> cudaError_t,
        #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
        pub cudaMemcpy3DBatchAsync: unsafe extern "C" fn(
            numOps: usize,
            opList: *mut cudaMemcpy3DBatchOp,
            failIdx: *mut usize,
            flags: ::core::ffi::c_ulonglong,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMemcpy3DPeer: unsafe extern "C" fn(p: *const cudaMemcpy3DPeerParms) -> cudaError_t,
        pub cudaMemcpy3DPeerAsync: unsafe extern "C" fn(
            p: *const cudaMemcpy3DPeerParms,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMemcpyArrayToArray: unsafe extern "C" fn(
            dst: cudaArray_t,
            wOffsetDst: usize,
            hOffsetDst: usize,
            src: cudaArray_const_t,
            wOffsetSrc: usize,
            hOffsetSrc: usize,
            count: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaMemcpyAsync: unsafe extern "C" fn(
            dst: *mut ::core::ffi::c_void,
            src: *const ::core::ffi::c_void,
            count: usize,
            kind: cudaMemcpyKind,
            stream: cudaStream_t,
        ) -> cudaError_t,
        #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
        pub cudaMemcpyBatchAsync: unsafe extern "C" fn(
            dsts: *mut *mut ::core::ffi::c_void,
            srcs: *mut *mut ::core::ffi::c_void,
            sizes: *mut usize,
            count: usize,
            attrs: *mut cudaMemcpyAttributes,
            attrsIdxs: *mut usize,
            numAttrs: usize,
            failIdx: *mut usize,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMemcpyFromArray: unsafe extern "C" fn(
            dst: *mut ::core::ffi::c_void,
            src: cudaArray_const_t,
            wOffset: usize,
            hOffset: usize,
            count: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaMemcpyFromArrayAsync: unsafe extern "C" fn(
            dst: *mut ::core::ffi::c_void,
            src: cudaArray_const_t,
            wOffset: usize,
            hOffset: usize,
            count: usize,
            kind: cudaMemcpyKind,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMemcpyFromSymbol: unsafe extern "C" fn(
            dst: *mut ::core::ffi::c_void,
            symbol: *const ::core::ffi::c_void,
            count: usize,
            offset: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaMemcpyFromSymbolAsync: unsafe extern "C" fn(
            dst: *mut ::core::ffi::c_void,
            symbol: *const ::core::ffi::c_void,
            count: usize,
            offset: usize,
            kind: cudaMemcpyKind,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMemcpyPeer: unsafe extern "C" fn(
            dst: *mut ::core::ffi::c_void,
            dstDevice: ::core::ffi::c_int,
            src: *const ::core::ffi::c_void,
            srcDevice: ::core::ffi::c_int,
            count: usize,
        ) -> cudaError_t,
        pub cudaMemcpyPeerAsync: unsafe extern "C" fn(
            dst: *mut ::core::ffi::c_void,
            dstDevice: ::core::ffi::c_int,
            src: *const ::core::ffi::c_void,
            srcDevice: ::core::ffi::c_int,
            count: usize,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMemcpyToArray: unsafe extern "C" fn(
            dst: cudaArray_t,
            wOffset: usize,
            hOffset: usize,
            src: *const ::core::ffi::c_void,
            count: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaMemcpyToArrayAsync: unsafe extern "C" fn(
            dst: cudaArray_t,
            wOffset: usize,
            hOffset: usize,
            src: *const ::core::ffi::c_void,
            count: usize,
            kind: cudaMemcpyKind,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMemcpyToSymbol: unsafe extern "C" fn(
            symbol: *const ::core::ffi::c_void,
            src: *const ::core::ffi::c_void,
            count: usize,
            offset: usize,
            kind: cudaMemcpyKind,
        ) -> cudaError_t,
        pub cudaMemcpyToSymbolAsync: unsafe extern "C" fn(
            symbol: *const ::core::ffi::c_void,
            src: *const ::core::ffi::c_void,
            count: usize,
            offset: usize,
            kind: cudaMemcpyKind,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMemset: unsafe extern "C" fn(
            devPtr: *mut ::core::ffi::c_void,
            value: ::core::ffi::c_int,
            count: usize,
        ) -> cudaError_t,
        pub cudaMemset2D: unsafe extern "C" fn(
            devPtr: *mut ::core::ffi::c_void,
            pitch: usize,
            value: ::core::ffi::c_int,
            width: usize,
            height: usize,
        ) -> cudaError_t,
        pub cudaMemset2DAsync: unsafe extern "C" fn(
            devPtr: *mut ::core::ffi::c_void,
            pitch: usize,
            value: ::core::ffi::c_int,
            width: usize,
            height: usize,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMemset3D: unsafe extern "C" fn(
            pitchedDevPtr: cudaPitchedPtr,
            value: ::core::ffi::c_int,
            extent: cudaExtent,
        ) -> cudaError_t,
        pub cudaMemset3DAsync: unsafe extern "C" fn(
            pitchedDevPtr: cudaPitchedPtr,
            value: ::core::ffi::c_int,
            extent: cudaExtent,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaMemsetAsync: unsafe extern "C" fn(
            devPtr: *mut ::core::ffi::c_void,
            value: ::core::ffi::c_int,
            count: usize,
            stream: cudaStream_t,
        ) -> cudaError_t,
        #[cfg(any(
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
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaMipmappedArrayGetMemoryRequirements: unsafe extern "C" fn(
            memoryRequirements: *mut cudaArrayMemoryRequirements,
            mipmap: cudaMipmappedArray_t,
            device: ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaMipmappedArrayGetSparseProperties: unsafe extern "C" fn(
            sparseProperties: *mut cudaArraySparseProperties,
            mipmap: cudaMipmappedArray_t,
        ) -> cudaError_t,
        pub cudaOccupancyAvailableDynamicSMemPerBlock: unsafe extern "C" fn(
            dynamicSmemSize: *mut usize,
            func: *const ::core::ffi::c_void,
            numBlocks: ::core::ffi::c_int,
            blockSize: ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaOccupancyMaxActiveBlocksPerMultiprocessor: unsafe extern "C" fn(
            numBlocks: *mut ::core::ffi::c_int,
            func: *const ::core::ffi::c_void,
            blockSize: ::core::ffi::c_int,
            dynamicSMemSize: usize,
        )
            -> cudaError_t,
        pub cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags:
            unsafe extern "C" fn(
                numBlocks: *mut ::core::ffi::c_int,
                func: *const ::core::ffi::c_void,
                blockSize: ::core::ffi::c_int,
                dynamicSMemSize: usize,
                flags: ::core::ffi::c_uint,
            ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11080",
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
        pub cudaOccupancyMaxActiveClusters: unsafe extern "C" fn(
            numClusters: *mut ::core::ffi::c_int,
            func: *const ::core::ffi::c_void,
            launchConfig: *const cudaLaunchConfig_t,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11080",
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
        pub cudaOccupancyMaxPotentialClusterSize: unsafe extern "C" fn(
            clusterSize: *mut ::core::ffi::c_int,
            func: *const ::core::ffi::c_void,
            launchConfig: *const cudaLaunchConfig_t,
        ) -> cudaError_t,
        pub cudaPeekAtLastError: unsafe extern "C" fn() -> cudaError_t,
        pub cudaPointerGetAttributes: unsafe extern "C" fn(
            attributes: *mut cudaPointerAttributes,
            ptr: *const ::core::ffi::c_void,
        ) -> cudaError_t,
        pub cudaProfilerStop: unsafe extern "C" fn() -> cudaError_t,
        pub cudaRuntimeGetVersion:
            unsafe extern "C" fn(runtimeVersion: *mut ::core::ffi::c_int) -> cudaError_t,
        pub cudaSetDevice: unsafe extern "C" fn(device: ::core::ffi::c_int) -> cudaError_t,
        pub cudaSetDeviceFlags: unsafe extern "C" fn(flags: ::core::ffi::c_uint) -> cudaError_t,
        pub cudaSetDoubleForDevice: unsafe extern "C" fn(d: *mut f64) -> cudaError_t,
        pub cudaSetDoubleForHost: unsafe extern "C" fn(d: *mut f64) -> cudaError_t,
        pub cudaSetValidDevices: unsafe extern "C" fn(
            device_arr: *mut ::core::ffi::c_int,
            len: ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaSignalExternalSemaphoresAsync_v2: unsafe extern "C" fn(
            extSemArray: *const cudaExternalSemaphore_t,
            paramsArray: *const cudaExternalSemaphoreSignalParams,
            numExtSems: ::core::ffi::c_uint,
            stream: cudaStream_t,
        ) -> cudaError_t,
        pub cudaStreamAddCallback: unsafe extern "C" fn(
            stream: cudaStream_t,
            callback: cudaStreamCallback_t,
            userData: *mut ::core::ffi::c_void,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaStreamAttachMemAsync: unsafe extern "C" fn(
            stream: cudaStream_t,
            devPtr: *mut ::core::ffi::c_void,
            length: usize,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaStreamBeginCapture:
            unsafe extern "C" fn(stream: cudaStream_t, mode: cudaStreamCaptureMode) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaStreamBeginCaptureToGraph: unsafe extern "C" fn(
            stream: cudaStream_t,
            graph: cudaGraph_t,
            dependencies: *const cudaGraphNode_t,
            dependencyData: *const cudaGraphEdgeData,
            numDependencies: usize,
            mode: cudaStreamCaptureMode,
        ) -> cudaError_t,
        pub cudaStreamCopyAttributes:
            unsafe extern "C" fn(dst: cudaStream_t, src: cudaStream_t) -> cudaError_t,
        pub cudaStreamCreate: unsafe extern "C" fn(pStream: *mut cudaStream_t) -> cudaError_t,
        pub cudaStreamCreateWithFlags: unsafe extern "C" fn(
            pStream: *mut cudaStream_t,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaStreamCreateWithPriority: unsafe extern "C" fn(
            pStream: *mut cudaStream_t,
            flags: ::core::ffi::c_uint,
            priority: ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaStreamDestroy: unsafe extern "C" fn(stream: cudaStream_t) -> cudaError_t,
        pub cudaStreamEndCapture:
            unsafe extern "C" fn(stream: cudaStream_t, pGraph: *mut cudaGraph_t) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070"
        ))]
        pub cudaStreamGetAttribute: unsafe extern "C" fn(
            hStream: cudaStream_t,
            attr: cudaStreamAttrID,
            value_out: *mut cudaStreamAttrValue,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11080",
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
        pub cudaStreamGetAttribute: unsafe extern "C" fn(
            hStream: cudaStream_t,
            attr: cudaLaunchAttributeID,
            value_out: *mut cudaLaunchAttributeValue,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        ))]
        pub cudaStreamGetCaptureInfo: unsafe extern "C" fn(
            stream: cudaStream_t,
            pCaptureStatus: *mut cudaStreamCaptureStatus,
            pId: *mut ::core::ffi::c_ulonglong,
        ) -> cudaError_t,
        pub cudaStreamGetCaptureInfo_v2: unsafe extern "C" fn(
            stream: cudaStream_t,
            captureStatus_out: *mut cudaStreamCaptureStatus,
            id_out: *mut ::core::ffi::c_ulonglong,
            graph_out: *mut cudaGraph_t,
            dependencies_out: *mut *const cudaGraphNode_t,
            numDependencies_out: *mut usize,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaStreamGetCaptureInfo_v3: unsafe extern "C" fn(
            stream: cudaStream_t,
            captureStatus_out: *mut cudaStreamCaptureStatus,
            id_out: *mut ::core::ffi::c_ulonglong,
            graph_out: *mut cudaGraph_t,
            dependencies_out: *mut *const cudaGraphNode_t,
            edgeData_out: *mut *const cudaGraphEdgeData,
            numDependencies_out: *mut usize,
        ) -> cudaError_t,
        #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
        pub cudaStreamGetDevice: unsafe extern "C" fn(
            hStream: cudaStream_t,
            device: *mut ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaStreamGetFlags: unsafe extern "C" fn(
            hStream: cudaStream_t,
            flags: *mut ::core::ffi::c_uint,
        ) -> cudaError_t,
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
        pub cudaStreamGetId: unsafe extern "C" fn(
            hStream: cudaStream_t,
            streamId: *mut ::core::ffi::c_ulonglong,
        ) -> cudaError_t,
        pub cudaStreamGetPriority: unsafe extern "C" fn(
            hStream: cudaStream_t,
            priority: *mut ::core::ffi::c_int,
        ) -> cudaError_t,
        pub cudaStreamIsCapturing: unsafe extern "C" fn(
            stream: cudaStream_t,
            pCaptureStatus: *mut cudaStreamCaptureStatus,
        ) -> cudaError_t,
        pub cudaStreamQuery: unsafe extern "C" fn(stream: cudaStream_t) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070"
        ))]
        pub cudaStreamSetAttribute: unsafe extern "C" fn(
            hStream: cudaStream_t,
            attr: cudaStreamAttrID,
            value: *const cudaStreamAttrValue,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11080",
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
        pub cudaStreamSetAttribute: unsafe extern "C" fn(
            hStream: cudaStream_t,
            attr: cudaLaunchAttributeID,
            value: *const cudaLaunchAttributeValue,
        ) -> cudaError_t,
        pub cudaStreamSynchronize: unsafe extern "C" fn(stream: cudaStream_t) -> cudaError_t,
        pub cudaStreamUpdateCaptureDependencies: unsafe extern "C" fn(
            stream: cudaStream_t,
            dependencies: *mut cudaGraphNode_t,
            numDependencies: usize,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        #[cfg(any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080",
            feature = "cuda-12090"
        ))]
        pub cudaStreamUpdateCaptureDependencies_v2: unsafe extern "C" fn(
            stream: cudaStream_t,
            dependencies: *mut cudaGraphNode_t,
            dependencyData: *const cudaGraphEdgeData,
            numDependencies: usize,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaStreamWaitEvent: unsafe extern "C" fn(
            stream: cudaStream_t,
            event: cudaEvent_t,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaThreadExchangeStreamCaptureMode:
            unsafe extern "C" fn(mode: *mut cudaStreamCaptureMode) -> cudaError_t,
        pub cudaThreadExit: unsafe extern "C" fn() -> cudaError_t,
        pub cudaThreadGetCacheConfig:
            unsafe extern "C" fn(pCacheConfig: *mut cudaFuncCache) -> cudaError_t,
        pub cudaThreadGetLimit:
            unsafe extern "C" fn(pValue: *mut usize, limit: cudaLimit) -> cudaError_t,
        pub cudaThreadSetCacheConfig:
            unsafe extern "C" fn(cacheConfig: cudaFuncCache) -> cudaError_t,
        pub cudaThreadSetLimit: unsafe extern "C" fn(limit: cudaLimit, value: usize) -> cudaError_t,
        pub cudaThreadSynchronize: unsafe extern "C" fn() -> cudaError_t,
        #[cfg(any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        ))]
        pub cudaUnbindTexture: unsafe extern "C" fn(texref: *const textureReference) -> cudaError_t,
        pub cudaUserObjectCreate: unsafe extern "C" fn(
            object_out: *mut cudaUserObject_t,
            ptr: *mut ::core::ffi::c_void,
            destroy: cudaHostFn_t,
            initialRefcount: ::core::ffi::c_uint,
            flags: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaUserObjectRelease: unsafe extern "C" fn(
            object: cudaUserObject_t,
            count: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaUserObjectRetain: unsafe extern "C" fn(
            object: cudaUserObject_t,
            count: ::core::ffi::c_uint,
        ) -> cudaError_t,
        pub cudaWaitExternalSemaphoresAsync_v2: unsafe extern "C" fn(
            extSemArray: *const cudaExternalSemaphore_t,
            paramsArray: *const cudaExternalSemaphoreWaitParams,
            numExtSems: ::core::ffi::c_uint,
            stream: cudaStream_t,
        ) -> cudaError_t,
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
            let cudaArrayGetInfo = __library
                .get(b"cudaArrayGetInfo\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
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
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaArrayGetMemoryRequirements = __library
                .get(b"cudaArrayGetMemoryRequirements\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaArrayGetPlane = __library
                .get(b"cudaArrayGetPlane\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaArrayGetSparseProperties = __library
                .get(b"cudaArrayGetSparseProperties\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            ))]
            let cudaBindSurfaceToArray = __library
                .get(b"cudaBindSurfaceToArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            ))]
            let cudaBindTexture = __library
                .get(b"cudaBindTexture\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            ))]
            let cudaBindTexture2D = __library
                .get(b"cudaBindTexture2D\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            ))]
            let cudaBindTextureToArray = __library
                .get(b"cudaBindTextureToArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            ))]
            let cudaBindTextureToMipmappedArray = __library
                .get(b"cudaBindTextureToMipmappedArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaChooseDevice = __library
                .get(b"cudaChooseDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaCreateChannelDesc = __library
                .get(b"cudaCreateChannelDesc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaCreateSurfaceObject = __library
                .get(b"cudaCreateSurfaceObject\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaCreateTextureObject = __library
                .get(b"cudaCreateTextureObject\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-11080"))]
            let cudaCreateTextureObject_v2 = __library
                .get(b"cudaCreateTextureObject_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaCtxResetPersistingL2Cache = __library
                .get(b"cudaCtxResetPersistingL2Cache\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDestroyExternalMemory = __library
                .get(b"cudaDestroyExternalMemory\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDestroyExternalSemaphore = __library
                .get(b"cudaDestroyExternalSemaphore\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDestroySurfaceObject = __library
                .get(b"cudaDestroySurfaceObject\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDestroyTextureObject = __library
                .get(b"cudaDestroyTextureObject\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceCanAccessPeer = __library
                .get(b"cudaDeviceCanAccessPeer\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceDisablePeerAccess = __library
                .get(b"cudaDeviceDisablePeerAccess\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceEnablePeerAccess = __library
                .get(b"cudaDeviceEnablePeerAccess\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceFlushGPUDirectRDMAWrites = __library
                .get(b"cudaDeviceFlushGPUDirectRDMAWrites\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceGetAttribute = __library
                .get(b"cudaDeviceGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceGetByPCIBusId = __library
                .get(b"cudaDeviceGetByPCIBusId\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceGetCacheConfig = __library
                .get(b"cudaDeviceGetCacheConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceGetDefaultMemPool = __library
                .get(b"cudaDeviceGetDefaultMemPool\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceGetGraphMemAttribute = __library
                .get(b"cudaDeviceGetGraphMemAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceGetLimit = __library
                .get(b"cudaDeviceGetLimit\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceGetMemPool = __library
                .get(b"cudaDeviceGetMemPool\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceGetP2PAttribute = __library
                .get(b"cudaDeviceGetP2PAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceGetPCIBusId = __library
                .get(b"cudaDeviceGetPCIBusId\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceGetSharedMemConfig = __library
                .get(b"cudaDeviceGetSharedMemConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceGetStreamPriorityRange = __library
                .get(b"cudaDeviceGetStreamPriorityRange\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceGetTexture1DLinearMaxWidth = __library
                .get(b"cudaDeviceGetTexture1DLinearMaxWidth\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceGraphMemTrim = __library
                .get(b"cudaDeviceGraphMemTrim\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaDeviceRegisterAsyncNotification = __library
                .get(b"cudaDeviceRegisterAsyncNotification\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceReset = __library
                .get(b"cudaDeviceReset\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceSetCacheConfig = __library
                .get(b"cudaDeviceSetCacheConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceSetGraphMemAttribute = __library
                .get(b"cudaDeviceSetGraphMemAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceSetLimit = __library
                .get(b"cudaDeviceSetLimit\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceSetMemPool = __library
                .get(b"cudaDeviceSetMemPool\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceSetSharedMemConfig = __library
                .get(b"cudaDeviceSetSharedMemConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDeviceSynchronize = __library
                .get(b"cudaDeviceSynchronize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaDeviceUnregisterAsyncNotification = __library
                .get(b"cudaDeviceUnregisterAsyncNotification\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaDriverGetVersion = __library
                .get(b"cudaDriverGetVersion\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaEventCreate = __library
                .get(b"cudaEventCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaEventCreateWithFlags = __library
                .get(b"cudaEventCreateWithFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaEventDestroy = __library
                .get(b"cudaEventDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaEventElapsedTime = __library
                .get(b"cudaEventElapsedTime\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
            let cudaEventElapsedTime_v2 = __library
                .get(b"cudaEventElapsedTime_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaEventQuery = __library
                .get(b"cudaEventQuery\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaEventRecord = __library
                .get(b"cudaEventRecord\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaEventRecordWithFlags = __library
                .get(b"cudaEventRecordWithFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaEventSynchronize = __library
                .get(b"cudaEventSynchronize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaExternalMemoryGetMappedBuffer = __library
                .get(b"cudaExternalMemoryGetMappedBuffer\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaExternalMemoryGetMappedMipmappedArray = __library
                .get(b"cudaExternalMemoryGetMappedMipmappedArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaFree = __library
                .get(b"cudaFree\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaFreeArray = __library
                .get(b"cudaFreeArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaFreeAsync = __library
                .get(b"cudaFreeAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaFreeHost = __library
                .get(b"cudaFreeHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaFreeMipmappedArray = __library
                .get(b"cudaFreeMipmappedArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaFuncGetAttributes = __library
                .get(b"cudaFuncGetAttributes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaFuncGetName = __library
                .get(b"cudaFuncGetName\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaFuncGetParamInfo = __library
                .get(b"cudaFuncGetParamInfo\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaFuncSetAttribute = __library
                .get(b"cudaFuncSetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaFuncSetCacheConfig = __library
                .get(b"cudaFuncSetCacheConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaFuncSetSharedMemConfig = __library
                .get(b"cudaFuncSetSharedMemConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetChannelDesc = __library
                .get(b"cudaGetChannelDesc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetDevice = __library
                .get(b"cudaGetDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetDeviceCount = __library
                .get(b"cudaGetDeviceCount\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetDeviceFlags = __library
                .get(b"cudaGetDeviceFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            ))]
            let cudaGetDeviceProperties = __library
                .get(b"cudaGetDeviceProperties\0")
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
            let cudaGetDeviceProperties_v2 = __library
                .get(b"cudaGetDeviceProperties_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            ))]
            let cudaGetDriverEntryPoint = __library
                .get(b"cudaGetDriverEntryPoint\0")
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
            let cudaGetDriverEntryPoint = __library
                .get(b"cudaGetDriverEntryPoint\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaGetDriverEntryPointByVersion = __library
                .get(b"cudaGetDriverEntryPointByVersion\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetErrorName = __library
                .get(b"cudaGetErrorName\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetErrorString = __library
                .get(b"cudaGetErrorString\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetExportTable = __library
                .get(b"cudaGetExportTable\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetFuncBySymbol = __library
                .get(b"cudaGetFuncBySymbol\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaGetKernel = __library
                .get(b"cudaGetKernel\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetLastError = __library
                .get(b"cudaGetLastError\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetMipmappedArrayLevel = __library
                .get(b"cudaGetMipmappedArrayLevel\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetSurfaceObjectResourceDesc = __library
                .get(b"cudaGetSurfaceObjectResourceDesc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            ))]
            let cudaGetSurfaceReference = __library
                .get(b"cudaGetSurfaceReference\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetSymbolAddress = __library
                .get(b"cudaGetSymbolAddress\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetSymbolSize = __library
                .get(b"cudaGetSymbolSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            ))]
            let cudaGetTextureAlignmentOffset = __library
                .get(b"cudaGetTextureAlignmentOffset\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetTextureObjectResourceDesc = __library
                .get(b"cudaGetTextureObjectResourceDesc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetTextureObjectResourceViewDesc = __library
                .get(b"cudaGetTextureObjectResourceViewDesc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGetTextureObjectTextureDesc = __library
                .get(b"cudaGetTextureObjectTextureDesc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-11080"))]
            let cudaGetTextureObjectTextureDesc_v2 = __library
                .get(b"cudaGetTextureObjectTextureDesc_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            ))]
            let cudaGetTextureReference = __library
                .get(b"cudaGetTextureReference\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddChildGraphNode = __library
                .get(b"cudaGraphAddChildGraphNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddDependencies = __library
                .get(b"cudaGraphAddDependencies\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaGraphAddDependencies_v2 = __library
                .get(b"cudaGraphAddDependencies_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddEmptyNode = __library
                .get(b"cudaGraphAddEmptyNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddEventRecordNode = __library
                .get(b"cudaGraphAddEventRecordNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddEventWaitNode = __library
                .get(b"cudaGraphAddEventWaitNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddExternalSemaphoresSignalNode = __library
                .get(b"cudaGraphAddExternalSemaphoresSignalNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddExternalSemaphoresWaitNode = __library
                .get(b"cudaGraphAddExternalSemaphoresWaitNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddHostNode = __library
                .get(b"cudaGraphAddHostNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddKernelNode = __library
                .get(b"cudaGraphAddKernelNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddMemAllocNode = __library
                .get(b"cudaGraphAddMemAllocNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddMemFreeNode = __library
                .get(b"cudaGraphAddMemFreeNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddMemcpyNode = __library
                .get(b"cudaGraphAddMemcpyNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddMemcpyNode1D = __library
                .get(b"cudaGraphAddMemcpyNode1D\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddMemcpyNodeFromSymbol = __library
                .get(b"cudaGraphAddMemcpyNodeFromSymbol\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddMemcpyNodeToSymbol = __library
                .get(b"cudaGraphAddMemcpyNodeToSymbol\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphAddMemsetNode = __library
                .get(b"cudaGraphAddMemsetNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaGraphAddNode = __library
                .get(b"cudaGraphAddNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaGraphAddNode_v2 = __library
                .get(b"cudaGraphAddNode_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphChildGraphNodeGetGraph = __library
                .get(b"cudaGraphChildGraphNodeGetGraph\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphClone = __library
                .get(b"cudaGraphClone\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaGraphConditionalHandleCreate = __library
                .get(b"cudaGraphConditionalHandleCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphCreate = __library
                .get(b"cudaGraphCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphDebugDotPrint = __library
                .get(b"cudaGraphDebugDotPrint\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphDestroy = __library
                .get(b"cudaGraphDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphDestroyNode = __library
                .get(b"cudaGraphDestroyNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphEventRecordNodeGetEvent = __library
                .get(b"cudaGraphEventRecordNodeGetEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphEventRecordNodeSetEvent = __library
                .get(b"cudaGraphEventRecordNodeSetEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphEventWaitNodeGetEvent = __library
                .get(b"cudaGraphEventWaitNodeGetEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphEventWaitNodeSetEvent = __library
                .get(b"cudaGraphEventWaitNodeSetEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExecChildGraphNodeSetParams = __library
                .get(b"cudaGraphExecChildGraphNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExecDestroy = __library
                .get(b"cudaGraphExecDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExecEventRecordNodeSetEvent = __library
                .get(b"cudaGraphExecEventRecordNodeSetEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExecEventWaitNodeSetEvent = __library
                .get(b"cudaGraphExecEventWaitNodeSetEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExecExternalSemaphoresSignalNodeSetParams = __library
                .get(b"cudaGraphExecExternalSemaphoresSignalNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExecExternalSemaphoresWaitNodeSetParams = __library
                .get(b"cudaGraphExecExternalSemaphoresWaitNodeSetParams\0")
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
            let cudaGraphExecGetFlags = __library
                .get(b"cudaGraphExecGetFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExecHostNodeSetParams = __library
                .get(b"cudaGraphExecHostNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExecKernelNodeSetParams = __library
                .get(b"cudaGraphExecKernelNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExecMemcpyNodeSetParams = __library
                .get(b"cudaGraphExecMemcpyNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExecMemcpyNodeSetParams1D = __library
                .get(b"cudaGraphExecMemcpyNodeSetParams1D\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExecMemcpyNodeSetParamsFromSymbol = __library
                .get(b"cudaGraphExecMemcpyNodeSetParamsFromSymbol\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExecMemcpyNodeSetParamsToSymbol = __library
                .get(b"cudaGraphExecMemcpyNodeSetParamsToSymbol\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExecMemsetNodeSetParams = __library
                .get(b"cudaGraphExecMemsetNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaGraphExecNodeSetParams = __library
                .get(b"cudaGraphExecNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            ))]
            let cudaGraphExecUpdate = __library
                .get(b"cudaGraphExecUpdate\0")
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
            let cudaGraphExecUpdate = __library
                .get(b"cudaGraphExecUpdate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExternalSemaphoresSignalNodeGetParams = __library
                .get(b"cudaGraphExternalSemaphoresSignalNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExternalSemaphoresSignalNodeSetParams = __library
                .get(b"cudaGraphExternalSemaphoresSignalNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExternalSemaphoresWaitNodeGetParams = __library
                .get(b"cudaGraphExternalSemaphoresWaitNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphExternalSemaphoresWaitNodeSetParams = __library
                .get(b"cudaGraphExternalSemaphoresWaitNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphGetEdges = __library
                .get(b"cudaGraphGetEdges\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaGraphGetEdges_v2 = __library
                .get(b"cudaGraphGetEdges_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphGetNodes = __library
                .get(b"cudaGraphGetNodes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphGetRootNodes = __library
                .get(b"cudaGraphGetRootNodes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphHostNodeGetParams = __library
                .get(b"cudaGraphHostNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphHostNodeSetParams = __library
                .get(b"cudaGraphHostNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            ))]
            let cudaGraphInstantiate = __library
                .get(b"cudaGraphInstantiate\0")
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
            let cudaGraphInstantiate = __library
                .get(b"cudaGraphInstantiate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphInstantiateWithFlags = __library
                .get(b"cudaGraphInstantiateWithFlags\0")
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
            let cudaGraphInstantiateWithParams = __library
                .get(b"cudaGraphInstantiateWithParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphKernelNodeCopyAttributes = __library
                .get(b"cudaGraphKernelNodeCopyAttributes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070"
            ))]
            let cudaGraphKernelNodeGetAttribute = __library
                .get(b"cudaGraphKernelNodeGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11080",
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
            let cudaGraphKernelNodeGetAttribute = __library
                .get(b"cudaGraphKernelNodeGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphKernelNodeGetParams = __library
                .get(b"cudaGraphKernelNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070"
            ))]
            let cudaGraphKernelNodeSetAttribute = __library
                .get(b"cudaGraphKernelNodeSetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11080",
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
            let cudaGraphKernelNodeSetAttribute = __library
                .get(b"cudaGraphKernelNodeSetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphKernelNodeSetParams = __library
                .get(b"cudaGraphKernelNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphLaunch = __library
                .get(b"cudaGraphLaunch\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphMemAllocNodeGetParams = __library
                .get(b"cudaGraphMemAllocNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphMemFreeNodeGetParams = __library
                .get(b"cudaGraphMemFreeNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphMemcpyNodeGetParams = __library
                .get(b"cudaGraphMemcpyNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphMemcpyNodeSetParams = __library
                .get(b"cudaGraphMemcpyNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphMemcpyNodeSetParams1D = __library
                .get(b"cudaGraphMemcpyNodeSetParams1D\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphMemcpyNodeSetParamsFromSymbol = __library
                .get(b"cudaGraphMemcpyNodeSetParamsFromSymbol\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphMemcpyNodeSetParamsToSymbol = __library
                .get(b"cudaGraphMemcpyNodeSetParamsToSymbol\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphMemsetNodeGetParams = __library
                .get(b"cudaGraphMemsetNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphMemsetNodeSetParams = __library
                .get(b"cudaGraphMemsetNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphNodeFindInClone = __library
                .get(b"cudaGraphNodeFindInClone\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphNodeGetDependencies = __library
                .get(b"cudaGraphNodeGetDependencies\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaGraphNodeGetDependencies_v2 = __library
                .get(b"cudaGraphNodeGetDependencies_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphNodeGetDependentNodes = __library
                .get(b"cudaGraphNodeGetDependentNodes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaGraphNodeGetDependentNodes_v2 = __library
                .get(b"cudaGraphNodeGetDependentNodes_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
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
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaGraphNodeGetEnabled = __library
                .get(b"cudaGraphNodeGetEnabled\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphNodeGetType = __library
                .get(b"cudaGraphNodeGetType\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
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
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaGraphNodeSetEnabled = __library
                .get(b"cudaGraphNodeSetEnabled\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaGraphNodeSetParams = __library
                .get(b"cudaGraphNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphReleaseUserObject = __library
                .get(b"cudaGraphReleaseUserObject\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphRemoveDependencies = __library
                .get(b"cudaGraphRemoveDependencies\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaGraphRemoveDependencies_v2 = __library
                .get(b"cudaGraphRemoveDependencies_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphRetainUserObject = __library
                .get(b"cudaGraphRetainUserObject\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphUpload = __library
                .get(b"cudaGraphUpload\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphicsMapResources = __library
                .get(b"cudaGraphicsMapResources\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphicsResourceGetMappedMipmappedArray = __library
                .get(b"cudaGraphicsResourceGetMappedMipmappedArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphicsResourceGetMappedPointer = __library
                .get(b"cudaGraphicsResourceGetMappedPointer\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphicsResourceSetMapFlags = __library
                .get(b"cudaGraphicsResourceSetMapFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphicsSubResourceGetMappedArray = __library
                .get(b"cudaGraphicsSubResourceGetMappedArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphicsUnmapResources = __library
                .get(b"cudaGraphicsUnmapResources\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaGraphicsUnregisterResource = __library
                .get(b"cudaGraphicsUnregisterResource\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaHostAlloc = __library
                .get(b"cudaHostAlloc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaHostGetDevicePointer = __library
                .get(b"cudaHostGetDevicePointer\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaHostGetFlags = __library
                .get(b"cudaHostGetFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaHostRegister = __library
                .get(b"cudaHostRegister\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaHostUnregister = __library
                .get(b"cudaHostUnregister\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaImportExternalMemory = __library
                .get(b"cudaImportExternalMemory\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaImportExternalSemaphore = __library
                .get(b"cudaImportExternalSemaphore\0")
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
            let cudaInitDevice = __library
                .get(b"cudaInitDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaIpcCloseMemHandle = __library
                .get(b"cudaIpcCloseMemHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaIpcGetEventHandle = __library
                .get(b"cudaIpcGetEventHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaIpcGetMemHandle = __library
                .get(b"cudaIpcGetMemHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaIpcOpenEventHandle = __library
                .get(b"cudaIpcOpenEventHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaIpcOpenMemHandle = __library
                .get(b"cudaIpcOpenMemHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
            let cudaKernelSetAttributeForDevice = __library
                .get(b"cudaKernelSetAttributeForDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaLaunchCooperativeKernel = __library
                .get(b"cudaLaunchCooperativeKernel\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaLaunchCooperativeKernelMultiDevice = __library
                .get(b"cudaLaunchCooperativeKernelMultiDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaLaunchHostFunc = __library
                .get(b"cudaLaunchHostFunc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaLaunchKernel = __library
                .get(b"cudaLaunchKernel\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11080",
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
            let cudaLaunchKernelExC = __library
                .get(b"cudaLaunchKernelExC\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
            let cudaLibraryEnumerateKernels = __library
                .get(b"cudaLibraryEnumerateKernels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
            let cudaLibraryGetGlobal = __library
                .get(b"cudaLibraryGetGlobal\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
            let cudaLibraryGetKernel = __library
                .get(b"cudaLibraryGetKernel\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
            let cudaLibraryGetKernelCount = __library
                .get(b"cudaLibraryGetKernelCount\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
            let cudaLibraryGetManaged = __library
                .get(b"cudaLibraryGetManaged\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
            let cudaLibraryGetUnifiedFunction = __library
                .get(b"cudaLibraryGetUnifiedFunction\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
            let cudaLibraryLoadData = __library
                .get(b"cudaLibraryLoadData\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
            let cudaLibraryLoadFromFile = __library
                .get(b"cudaLibraryLoadFromFile\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
            let cudaLibraryUnload = __library
                .get(b"cudaLibraryUnload\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMalloc = __library
                .get(b"cudaMalloc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMalloc3D = __library
                .get(b"cudaMalloc3D\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMalloc3DArray = __library
                .get(b"cudaMalloc3DArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMallocArray = __library
                .get(b"cudaMallocArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMallocAsync = __library
                .get(b"cudaMallocAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMallocFromPoolAsync = __library
                .get(b"cudaMallocFromPoolAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMallocHost = __library
                .get(b"cudaMallocHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMallocManaged = __library
                .get(b"cudaMallocManaged\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMallocMipmappedArray = __library
                .get(b"cudaMallocMipmappedArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMallocPitch = __library
                .get(b"cudaMallocPitch\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemAdvise = __library
                .get(b"cudaMemAdvise\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaMemAdvise_v2 = __library
                .get(b"cudaMemAdvise_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemGetInfo = __library
                .get(b"cudaMemGetInfo\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemPoolCreate = __library
                .get(b"cudaMemPoolCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemPoolDestroy = __library
                .get(b"cudaMemPoolDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemPoolExportPointer = __library
                .get(b"cudaMemPoolExportPointer\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemPoolExportToShareableHandle = __library
                .get(b"cudaMemPoolExportToShareableHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemPoolGetAccess = __library
                .get(b"cudaMemPoolGetAccess\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemPoolGetAttribute = __library
                .get(b"cudaMemPoolGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemPoolImportFromShareableHandle = __library
                .get(b"cudaMemPoolImportFromShareableHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemPoolImportPointer = __library
                .get(b"cudaMemPoolImportPointer\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemPoolSetAccess = __library
                .get(b"cudaMemPoolSetAccess\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemPoolSetAttribute = __library
                .get(b"cudaMemPoolSetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemPoolTrimTo = __library
                .get(b"cudaMemPoolTrimTo\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemPrefetchAsync = __library
                .get(b"cudaMemPrefetchAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaMemPrefetchAsync_v2 = __library
                .get(b"cudaMemPrefetchAsync_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemRangeGetAttribute = __library
                .get(b"cudaMemRangeGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemRangeGetAttributes = __library
                .get(b"cudaMemRangeGetAttributes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpy = __library
                .get(b"cudaMemcpy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpy2D = __library
                .get(b"cudaMemcpy2D\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpy2DArrayToArray = __library
                .get(b"cudaMemcpy2DArrayToArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpy2DAsync = __library
                .get(b"cudaMemcpy2DAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpy2DFromArray = __library
                .get(b"cudaMemcpy2DFromArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpy2DFromArrayAsync = __library
                .get(b"cudaMemcpy2DFromArrayAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpy2DToArray = __library
                .get(b"cudaMemcpy2DToArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpy2DToArrayAsync = __library
                .get(b"cudaMemcpy2DToArrayAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpy3D = __library
                .get(b"cudaMemcpy3D\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpy3DAsync = __library
                .get(b"cudaMemcpy3DAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
            let cudaMemcpy3DBatchAsync = __library
                .get(b"cudaMemcpy3DBatchAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpy3DPeer = __library
                .get(b"cudaMemcpy3DPeer\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpy3DPeerAsync = __library
                .get(b"cudaMemcpy3DPeerAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpyArrayToArray = __library
                .get(b"cudaMemcpyArrayToArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpyAsync = __library
                .get(b"cudaMemcpyAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
            let cudaMemcpyBatchAsync = __library
                .get(b"cudaMemcpyBatchAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpyFromArray = __library
                .get(b"cudaMemcpyFromArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpyFromArrayAsync = __library
                .get(b"cudaMemcpyFromArrayAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpyFromSymbol = __library
                .get(b"cudaMemcpyFromSymbol\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpyFromSymbolAsync = __library
                .get(b"cudaMemcpyFromSymbolAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpyPeer = __library
                .get(b"cudaMemcpyPeer\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpyPeerAsync = __library
                .get(b"cudaMemcpyPeerAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpyToArray = __library
                .get(b"cudaMemcpyToArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpyToArrayAsync = __library
                .get(b"cudaMemcpyToArrayAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpyToSymbol = __library
                .get(b"cudaMemcpyToSymbol\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemcpyToSymbolAsync = __library
                .get(b"cudaMemcpyToSymbolAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemset = __library
                .get(b"cudaMemset\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemset2D = __library
                .get(b"cudaMemset2D\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemset2DAsync = __library
                .get(b"cudaMemset2DAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemset3D = __library
                .get(b"cudaMemset3D\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemset3DAsync = __library
                .get(b"cudaMemset3DAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMemsetAsync = __library
                .get(b"cudaMemsetAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
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
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaMipmappedArrayGetMemoryRequirements = __library
                .get(b"cudaMipmappedArrayGetMemoryRequirements\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaMipmappedArrayGetSparseProperties = __library
                .get(b"cudaMipmappedArrayGetSparseProperties\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaOccupancyAvailableDynamicSMemPerBlock = __library
                .get(b"cudaOccupancyAvailableDynamicSMemPerBlock\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaOccupancyMaxActiveBlocksPerMultiprocessor = __library
                .get(b"cudaOccupancyMaxActiveBlocksPerMultiprocessor\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = __library
                .get(b"cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11080",
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
            let cudaOccupancyMaxActiveClusters = __library
                .get(b"cudaOccupancyMaxActiveClusters\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11080",
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
            let cudaOccupancyMaxPotentialClusterSize = __library
                .get(b"cudaOccupancyMaxPotentialClusterSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaPeekAtLastError = __library
                .get(b"cudaPeekAtLastError\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaPointerGetAttributes = __library
                .get(b"cudaPointerGetAttributes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaProfilerStop = __library
                .get(b"cudaProfilerStop\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaRuntimeGetVersion = __library
                .get(b"cudaRuntimeGetVersion\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaSetDevice = __library
                .get(b"cudaSetDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaSetDeviceFlags = __library
                .get(b"cudaSetDeviceFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaSetDoubleForDevice = __library
                .get(b"cudaSetDoubleForDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaSetDoubleForHost = __library
                .get(b"cudaSetDoubleForHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaSetValidDevices = __library
                .get(b"cudaSetValidDevices\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaSignalExternalSemaphoresAsync_v2 = __library
                .get(b"cudaSignalExternalSemaphoresAsync_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamAddCallback = __library
                .get(b"cudaStreamAddCallback\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamAttachMemAsync = __library
                .get(b"cudaStreamAttachMemAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamBeginCapture = __library
                .get(b"cudaStreamBeginCapture\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaStreamBeginCaptureToGraph = __library
                .get(b"cudaStreamBeginCaptureToGraph\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamCopyAttributes = __library
                .get(b"cudaStreamCopyAttributes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamCreate = __library
                .get(b"cudaStreamCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamCreateWithFlags = __library
                .get(b"cudaStreamCreateWithFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamCreateWithPriority = __library
                .get(b"cudaStreamCreateWithPriority\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamDestroy = __library
                .get(b"cudaStreamDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamEndCapture = __library
                .get(b"cudaStreamEndCapture\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070"
            ))]
            let cudaStreamGetAttribute = __library
                .get(b"cudaStreamGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11080",
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
            let cudaStreamGetAttribute = __library
                .get(b"cudaStreamGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            ))]
            let cudaStreamGetCaptureInfo = __library
                .get(b"cudaStreamGetCaptureInfo\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamGetCaptureInfo_v2 = __library
                .get(b"cudaStreamGetCaptureInfo_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaStreamGetCaptureInfo_v3 = __library
                .get(b"cudaStreamGetCaptureInfo_v3\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
            let cudaStreamGetDevice = __library
                .get(b"cudaStreamGetDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamGetFlags = __library
                .get(b"cudaStreamGetFlags\0")
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
            let cudaStreamGetId = __library
                .get(b"cudaStreamGetId\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamGetPriority = __library
                .get(b"cudaStreamGetPriority\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamIsCapturing = __library
                .get(b"cudaStreamIsCapturing\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamQuery = __library
                .get(b"cudaStreamQuery\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070"
            ))]
            let cudaStreamSetAttribute = __library
                .get(b"cudaStreamSetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11080",
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
            let cudaStreamSetAttribute = __library
                .get(b"cudaStreamSetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamSynchronize = __library
                .get(b"cudaStreamSynchronize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamUpdateCaptureDependencies = __library
                .get(b"cudaStreamUpdateCaptureDependencies\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080",
                feature = "cuda-12090"
            ))]
            let cudaStreamUpdateCaptureDependencies_v2 = __library
                .get(b"cudaStreamUpdateCaptureDependencies_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaStreamWaitEvent = __library
                .get(b"cudaStreamWaitEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaThreadExchangeStreamCaptureMode = __library
                .get(b"cudaThreadExchangeStreamCaptureMode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaThreadExit = __library
                .get(b"cudaThreadExit\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaThreadGetCacheConfig = __library
                .get(b"cudaThreadGetCacheConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaThreadGetLimit = __library
                .get(b"cudaThreadGetLimit\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaThreadSetCacheConfig = __library
                .get(b"cudaThreadSetCacheConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaThreadSetLimit = __library
                .get(b"cudaThreadSetLimit\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaThreadSynchronize = __library
                .get(b"cudaThreadSynchronize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            ))]
            let cudaUnbindTexture = __library
                .get(b"cudaUnbindTexture\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaUserObjectCreate = __library
                .get(b"cudaUserObjectCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaUserObjectRelease = __library
                .get(b"cudaUserObjectRelease\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaUserObjectRetain = __library
                .get(b"cudaUserObjectRetain\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cudaWaitExternalSemaphoresAsync_v2 = __library
                .get(b"cudaWaitExternalSemaphoresAsync_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            Ok(Self {
                __library,
                cudaArrayGetInfo,
                #[cfg(any(
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
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaArrayGetMemoryRequirements,
                cudaArrayGetPlane,
                cudaArrayGetSparseProperties,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                ))]
                cudaBindSurfaceToArray,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                ))]
                cudaBindTexture,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                ))]
                cudaBindTexture2D,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                ))]
                cudaBindTextureToArray,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                ))]
                cudaBindTextureToMipmappedArray,
                cudaChooseDevice,
                cudaCreateChannelDesc,
                cudaCreateSurfaceObject,
                cudaCreateTextureObject,
                #[cfg(any(feature = "cuda-11080"))]
                cudaCreateTextureObject_v2,
                cudaCtxResetPersistingL2Cache,
                cudaDestroyExternalMemory,
                cudaDestroyExternalSemaphore,
                cudaDestroySurfaceObject,
                cudaDestroyTextureObject,
                cudaDeviceCanAccessPeer,
                cudaDeviceDisablePeerAccess,
                cudaDeviceEnablePeerAccess,
                cudaDeviceFlushGPUDirectRDMAWrites,
                cudaDeviceGetAttribute,
                cudaDeviceGetByPCIBusId,
                cudaDeviceGetCacheConfig,
                cudaDeviceGetDefaultMemPool,
                cudaDeviceGetGraphMemAttribute,
                cudaDeviceGetLimit,
                cudaDeviceGetMemPool,
                cudaDeviceGetP2PAttribute,
                cudaDeviceGetPCIBusId,
                cudaDeviceGetSharedMemConfig,
                cudaDeviceGetStreamPriorityRange,
                cudaDeviceGetTexture1DLinearMaxWidth,
                cudaDeviceGraphMemTrim,
                #[cfg(any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaDeviceRegisterAsyncNotification,
                cudaDeviceReset,
                cudaDeviceSetCacheConfig,
                cudaDeviceSetGraphMemAttribute,
                cudaDeviceSetLimit,
                cudaDeviceSetMemPool,
                cudaDeviceSetSharedMemConfig,
                cudaDeviceSynchronize,
                #[cfg(any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaDeviceUnregisterAsyncNotification,
                cudaDriverGetVersion,
                cudaEventCreate,
                cudaEventCreateWithFlags,
                cudaEventDestroy,
                cudaEventElapsedTime,
                #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
                cudaEventElapsedTime_v2,
                cudaEventQuery,
                cudaEventRecord,
                cudaEventRecordWithFlags,
                cudaEventSynchronize,
                cudaExternalMemoryGetMappedBuffer,
                cudaExternalMemoryGetMappedMipmappedArray,
                cudaFree,
                cudaFreeArray,
                cudaFreeAsync,
                cudaFreeHost,
                cudaFreeMipmappedArray,
                cudaFuncGetAttributes,
                #[cfg(any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaFuncGetName,
                #[cfg(any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaFuncGetParamInfo,
                cudaFuncSetAttribute,
                cudaFuncSetCacheConfig,
                cudaFuncSetSharedMemConfig,
                cudaGetChannelDesc,
                cudaGetDevice,
                cudaGetDeviceCount,
                cudaGetDeviceFlags,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                ))]
                cudaGetDeviceProperties,
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
                cudaGetDeviceProperties_v2,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                ))]
                cudaGetDriverEntryPoint,
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
                cudaGetDriverEntryPoint,
                #[cfg(any(
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaGetDriverEntryPointByVersion,
                cudaGetErrorName,
                cudaGetErrorString,
                cudaGetExportTable,
                cudaGetFuncBySymbol,
                #[cfg(any(
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaGetKernel,
                cudaGetLastError,
                cudaGetMipmappedArrayLevel,
                cudaGetSurfaceObjectResourceDesc,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                ))]
                cudaGetSurfaceReference,
                cudaGetSymbolAddress,
                cudaGetSymbolSize,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                ))]
                cudaGetTextureAlignmentOffset,
                cudaGetTextureObjectResourceDesc,
                cudaGetTextureObjectResourceViewDesc,
                cudaGetTextureObjectTextureDesc,
                #[cfg(any(feature = "cuda-11080"))]
                cudaGetTextureObjectTextureDesc_v2,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                ))]
                cudaGetTextureReference,
                cudaGraphAddChildGraphNode,
                cudaGraphAddDependencies,
                #[cfg(any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaGraphAddDependencies_v2,
                cudaGraphAddEmptyNode,
                cudaGraphAddEventRecordNode,
                cudaGraphAddEventWaitNode,
                cudaGraphAddExternalSemaphoresSignalNode,
                cudaGraphAddExternalSemaphoresWaitNode,
                cudaGraphAddHostNode,
                cudaGraphAddKernelNode,
                cudaGraphAddMemAllocNode,
                cudaGraphAddMemFreeNode,
                cudaGraphAddMemcpyNode,
                cudaGraphAddMemcpyNode1D,
                cudaGraphAddMemcpyNodeFromSymbol,
                cudaGraphAddMemcpyNodeToSymbol,
                cudaGraphAddMemsetNode,
                #[cfg(any(
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaGraphAddNode,
                #[cfg(any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaGraphAddNode_v2,
                cudaGraphChildGraphNodeGetGraph,
                cudaGraphClone,
                #[cfg(any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaGraphConditionalHandleCreate,
                cudaGraphCreate,
                cudaGraphDebugDotPrint,
                cudaGraphDestroy,
                cudaGraphDestroyNode,
                cudaGraphEventRecordNodeGetEvent,
                cudaGraphEventRecordNodeSetEvent,
                cudaGraphEventWaitNodeGetEvent,
                cudaGraphEventWaitNodeSetEvent,
                cudaGraphExecChildGraphNodeSetParams,
                cudaGraphExecDestroy,
                cudaGraphExecEventRecordNodeSetEvent,
                cudaGraphExecEventWaitNodeSetEvent,
                cudaGraphExecExternalSemaphoresSignalNodeSetParams,
                cudaGraphExecExternalSemaphoresWaitNodeSetParams,
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
                cudaGraphExecGetFlags,
                cudaGraphExecHostNodeSetParams,
                cudaGraphExecKernelNodeSetParams,
                cudaGraphExecMemcpyNodeSetParams,
                cudaGraphExecMemcpyNodeSetParams1D,
                cudaGraphExecMemcpyNodeSetParamsFromSymbol,
                cudaGraphExecMemcpyNodeSetParamsToSymbol,
                cudaGraphExecMemsetNodeSetParams,
                #[cfg(any(
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaGraphExecNodeSetParams,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                ))]
                cudaGraphExecUpdate,
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
                cudaGraphExecUpdate,
                cudaGraphExternalSemaphoresSignalNodeGetParams,
                cudaGraphExternalSemaphoresSignalNodeSetParams,
                cudaGraphExternalSemaphoresWaitNodeGetParams,
                cudaGraphExternalSemaphoresWaitNodeSetParams,
                cudaGraphGetEdges,
                #[cfg(any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaGraphGetEdges_v2,
                cudaGraphGetNodes,
                cudaGraphGetRootNodes,
                cudaGraphHostNodeGetParams,
                cudaGraphHostNodeSetParams,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                ))]
                cudaGraphInstantiate,
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
                cudaGraphInstantiate,
                cudaGraphInstantiateWithFlags,
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
                cudaGraphInstantiateWithParams,
                cudaGraphKernelNodeCopyAttributes,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070"
                ))]
                cudaGraphKernelNodeGetAttribute,
                #[cfg(any(
                    feature = "cuda-11080",
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
                cudaGraphKernelNodeGetAttribute,
                cudaGraphKernelNodeGetParams,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070"
                ))]
                cudaGraphKernelNodeSetAttribute,
                #[cfg(any(
                    feature = "cuda-11080",
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
                cudaGraphKernelNodeSetAttribute,
                cudaGraphKernelNodeSetParams,
                cudaGraphLaunch,
                cudaGraphMemAllocNodeGetParams,
                cudaGraphMemFreeNodeGetParams,
                cudaGraphMemcpyNodeGetParams,
                cudaGraphMemcpyNodeSetParams,
                cudaGraphMemcpyNodeSetParams1D,
                cudaGraphMemcpyNodeSetParamsFromSymbol,
                cudaGraphMemcpyNodeSetParamsToSymbol,
                cudaGraphMemsetNodeGetParams,
                cudaGraphMemsetNodeSetParams,
                cudaGraphNodeFindInClone,
                cudaGraphNodeGetDependencies,
                #[cfg(any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaGraphNodeGetDependencies_v2,
                cudaGraphNodeGetDependentNodes,
                #[cfg(any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaGraphNodeGetDependentNodes_v2,
                #[cfg(any(
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
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaGraphNodeGetEnabled,
                cudaGraphNodeGetType,
                #[cfg(any(
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
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaGraphNodeSetEnabled,
                #[cfg(any(
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaGraphNodeSetParams,
                cudaGraphReleaseUserObject,
                cudaGraphRemoveDependencies,
                #[cfg(any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaGraphRemoveDependencies_v2,
                cudaGraphRetainUserObject,
                cudaGraphUpload,
                cudaGraphicsMapResources,
                cudaGraphicsResourceGetMappedMipmappedArray,
                cudaGraphicsResourceGetMappedPointer,
                cudaGraphicsResourceSetMapFlags,
                cudaGraphicsSubResourceGetMappedArray,
                cudaGraphicsUnmapResources,
                cudaGraphicsUnregisterResource,
                cudaHostAlloc,
                cudaHostGetDevicePointer,
                cudaHostGetFlags,
                cudaHostRegister,
                cudaHostUnregister,
                cudaImportExternalMemory,
                cudaImportExternalSemaphore,
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
                cudaInitDevice,
                cudaIpcCloseMemHandle,
                cudaIpcGetEventHandle,
                cudaIpcGetMemHandle,
                cudaIpcOpenEventHandle,
                cudaIpcOpenMemHandle,
                #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
                cudaKernelSetAttributeForDevice,
                cudaLaunchCooperativeKernel,
                cudaLaunchCooperativeKernelMultiDevice,
                cudaLaunchHostFunc,
                cudaLaunchKernel,
                #[cfg(any(
                    feature = "cuda-11080",
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
                cudaLaunchKernelExC,
                #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
                cudaLibraryEnumerateKernels,
                #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
                cudaLibraryGetGlobal,
                #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
                cudaLibraryGetKernel,
                #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
                cudaLibraryGetKernelCount,
                #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
                cudaLibraryGetManaged,
                #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
                cudaLibraryGetUnifiedFunction,
                #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
                cudaLibraryLoadData,
                #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
                cudaLibraryLoadFromFile,
                #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
                cudaLibraryUnload,
                cudaMalloc,
                cudaMalloc3D,
                cudaMalloc3DArray,
                cudaMallocArray,
                cudaMallocAsync,
                cudaMallocFromPoolAsync,
                cudaMallocHost,
                cudaMallocManaged,
                cudaMallocMipmappedArray,
                cudaMallocPitch,
                cudaMemAdvise,
                #[cfg(any(
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaMemAdvise_v2,
                cudaMemGetInfo,
                cudaMemPoolCreate,
                cudaMemPoolDestroy,
                cudaMemPoolExportPointer,
                cudaMemPoolExportToShareableHandle,
                cudaMemPoolGetAccess,
                cudaMemPoolGetAttribute,
                cudaMemPoolImportFromShareableHandle,
                cudaMemPoolImportPointer,
                cudaMemPoolSetAccess,
                cudaMemPoolSetAttribute,
                cudaMemPoolTrimTo,
                cudaMemPrefetchAsync,
                #[cfg(any(
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaMemPrefetchAsync_v2,
                cudaMemRangeGetAttribute,
                cudaMemRangeGetAttributes,
                cudaMemcpy,
                cudaMemcpy2D,
                cudaMemcpy2DArrayToArray,
                cudaMemcpy2DAsync,
                cudaMemcpy2DFromArray,
                cudaMemcpy2DFromArrayAsync,
                cudaMemcpy2DToArray,
                cudaMemcpy2DToArrayAsync,
                cudaMemcpy3D,
                cudaMemcpy3DAsync,
                #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
                cudaMemcpy3DBatchAsync,
                cudaMemcpy3DPeer,
                cudaMemcpy3DPeerAsync,
                cudaMemcpyArrayToArray,
                cudaMemcpyAsync,
                #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
                cudaMemcpyBatchAsync,
                cudaMemcpyFromArray,
                cudaMemcpyFromArrayAsync,
                cudaMemcpyFromSymbol,
                cudaMemcpyFromSymbolAsync,
                cudaMemcpyPeer,
                cudaMemcpyPeerAsync,
                cudaMemcpyToArray,
                cudaMemcpyToArrayAsync,
                cudaMemcpyToSymbol,
                cudaMemcpyToSymbolAsync,
                cudaMemset,
                cudaMemset2D,
                cudaMemset2DAsync,
                cudaMemset3D,
                cudaMemset3DAsync,
                cudaMemsetAsync,
                #[cfg(any(
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
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaMipmappedArrayGetMemoryRequirements,
                cudaMipmappedArrayGetSparseProperties,
                cudaOccupancyAvailableDynamicSMemPerBlock,
                cudaOccupancyMaxActiveBlocksPerMultiprocessor,
                cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
                #[cfg(any(
                    feature = "cuda-11080",
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
                cudaOccupancyMaxActiveClusters,
                #[cfg(any(
                    feature = "cuda-11080",
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
                cudaOccupancyMaxPotentialClusterSize,
                cudaPeekAtLastError,
                cudaPointerGetAttributes,
                cudaProfilerStop,
                cudaRuntimeGetVersion,
                cudaSetDevice,
                cudaSetDeviceFlags,
                cudaSetDoubleForDevice,
                cudaSetDoubleForHost,
                cudaSetValidDevices,
                cudaSignalExternalSemaphoresAsync_v2,
                cudaStreamAddCallback,
                cudaStreamAttachMemAsync,
                cudaStreamBeginCapture,
                #[cfg(any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaStreamBeginCaptureToGraph,
                cudaStreamCopyAttributes,
                cudaStreamCreate,
                cudaStreamCreateWithFlags,
                cudaStreamCreateWithPriority,
                cudaStreamDestroy,
                cudaStreamEndCapture,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070"
                ))]
                cudaStreamGetAttribute,
                #[cfg(any(
                    feature = "cuda-11080",
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
                cudaStreamGetAttribute,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                ))]
                cudaStreamGetCaptureInfo,
                cudaStreamGetCaptureInfo_v2,
                #[cfg(any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaStreamGetCaptureInfo_v3,
                #[cfg(any(feature = "cuda-12080", feature = "cuda-12090"))]
                cudaStreamGetDevice,
                cudaStreamGetFlags,
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
                cudaStreamGetId,
                cudaStreamGetPriority,
                cudaStreamIsCapturing,
                cudaStreamQuery,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070"
                ))]
                cudaStreamSetAttribute,
                #[cfg(any(
                    feature = "cuda-11080",
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
                cudaStreamSetAttribute,
                cudaStreamSynchronize,
                cudaStreamUpdateCaptureDependencies,
                #[cfg(any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080",
                    feature = "cuda-12090"
                ))]
                cudaStreamUpdateCaptureDependencies_v2,
                cudaStreamWaitEvent,
                cudaThreadExchangeStreamCaptureMode,
                cudaThreadExit,
                cudaThreadGetCacheConfig,
                cudaThreadGetLimit,
                cudaThreadSetCacheConfig,
                cudaThreadSetLimit,
                cudaThreadSynchronize,
                #[cfg(any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                ))]
                cudaUnbindTexture,
                cudaUserObjectCreate,
                cudaUserObjectRelease,
                cudaUserObjectRetain,
                cudaWaitExternalSemaphoresAsync_v2,
            })
        }
    }
    pub unsafe fn culib() -> &'static Lib {
        static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
        LIB.get_or_init(|| {
            let lib_names = std::vec!["cudart"];
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
