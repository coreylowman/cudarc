#![cfg_attr(feature = "no-std", no_std)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;
pub use self::CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum as CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS;
pub use self::CUGPUDirectRDMAWritesOrdering_enum as CUGPUDirectRDMAWritesOrdering;
pub use self::CUaccessProperty_enum as CUaccessProperty;
pub use self::CUaddress_mode_enum as CUaddress_mode;
pub use self::CUarraySparseSubresourceType_enum as CUarraySparseSubresourceType;
pub use self::CUarray_cubemap_face_enum as CUarray_cubemap_face;
pub use self::CUarray_format_enum as CUarray_format;
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUasyncNotificationType_enum as CUasyncNotificationType;
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
pub use self::CUcigDataType_enum as CUcigDataType;
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUclusterSchedulingPolicy_enum as CUclusterSchedulingPolicy;
pub use self::CUcomputemode_enum as CUcomputemode;
#[cfg(
    any(
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUcoredumpSettings_enum as CUcoredumpSettings;
pub use self::CUctx_flags_enum as CUctx_flags;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUdeviceNumaConfig_enum as CUdeviceNumaConfig;
pub use self::CUdevice_P2PAttribute_enum as CUdevice_P2PAttribute;
pub use self::CUdevice_attribute_enum as CUdevice_attribute;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUdriverProcAddressQueryResult_enum as CUdriverProcAddressQueryResult;
pub use self::CUdriverProcAddress_flags_enum as CUdriverProcAddress_flags;
pub use self::CUevent_flags_enum as CUevent_flags;
pub use self::CUevent_record_flags_enum as CUevent_record_flags;
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUevent_sched_flags_enum as CUevent_sched_flags;
pub use self::CUevent_wait_flags_enum as CUevent_wait_flags;
pub use self::CUexecAffinityType_enum as CUexecAffinityType;
pub use self::CUexternalMemoryHandleType_enum as CUexternalMemoryHandleType;
pub use self::CUexternalSemaphoreHandleType_enum as CUexternalSemaphoreHandleType;
pub use self::CUfilter_mode_enum as CUfilter_mode;
pub use self::CUflushGPUDirectRDMAWritesOptions_enum as CUflushGPUDirectRDMAWritesOptions;
pub use self::CUflushGPUDirectRDMAWritesScope_enum as CUflushGPUDirectRDMAWritesScope;
pub use self::CUflushGPUDirectRDMAWritesTarget_enum as CUflushGPUDirectRDMAWritesTarget;
pub use self::CUfunc_cache_enum as CUfunc_cache;
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUfunctionLoadingState_enum as CUfunctionLoadingState;
pub use self::CUfunction_attribute_enum as CUfunction_attribute;
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUgraphConditionalNodeType_enum as CUgraphConditionalNodeType;
pub use self::CUgraphDebugDot_flags_enum as CUgraphDebugDot_flags;
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUgraphDependencyType_enum as CUgraphDependencyType;
pub use self::CUgraphExecUpdateResult_enum as CUgraphExecUpdateResult;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUgraphInstantiateResult_enum as CUgraphInstantiateResult;
pub use self::CUgraphInstantiate_flags_enum as CUgraphInstantiate_flags;
pub use self::CUgraphMem_attribute_enum as CUgraphMem_attribute;
pub use self::CUgraphNodeType_enum as CUgraphNodeType;
pub use self::CUgraphicsMapResourceFlags_enum as CUgraphicsMapResourceFlags;
pub use self::CUgraphicsRegisterFlags_enum as CUgraphicsRegisterFlags;
pub use self::CUipcMem_flags_enum as CUipcMem_flags;
pub use self::CUjitInputType_enum as CUjitInputType;
pub use self::CUjit_cacheMode_enum as CUjit_cacheMode;
pub use self::CUjit_fallback_enum as CUjit_fallback;
pub use self::CUjit_option_enum as CUjit_option;
pub use self::CUjit_target_enum as CUjit_target;
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    )
)]
pub use self::CUkernelNodeAttrID_enum as CUkernelNodeAttrID;
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUlaunchAttributeID as CUkernelNodeAttrID;
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUlaunchAttributeID as CUstreamAttrID;
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUlaunchAttributeID_enum as CUlaunchAttributeID;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUlaunchMemSyncDomain_enum as CUlaunchMemSyncDomain;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUlibraryOption_enum as CUlibraryOption;
pub use self::CUlimit_enum as CUlimit;
pub use self::CUmemAccess_flags_enum as CUmemAccess_flags;
pub use self::CUmemAllocationCompType_enum as CUmemAllocationCompType;
pub use self::CUmemAllocationGranularity_flags_enum as CUmemAllocationGranularity_flags;
pub use self::CUmemAllocationHandleType_enum as CUmemAllocationHandleType;
pub use self::CUmemAllocationType_enum as CUmemAllocationType;
pub use self::CUmemAttach_flags_enum as CUmemAttach_flags;
#[cfg(any(feature = "cuda-12080"))]
pub use self::CUmemDecompressAlgorithm_enum as CUmemDecompressAlgorithm;
pub use self::CUmemHandleType_enum as CUmemHandleType;
pub use self::CUmemLocationType_enum as CUmemLocationType;
pub use self::CUmemOperationType_enum as CUmemOperationType;
pub use self::CUmemPool_attribute_enum as CUmemPool_attribute;
#[cfg(any(feature = "cuda-12080"))]
pub use self::CUmemRangeFlags_enum as CUmemRangeFlags;
#[cfg(
    any(
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
    )
)]
pub use self::CUmemRangeHandleType_enum as CUmemRangeHandleType;
pub use self::CUmem_advise_enum as CUmem_advise;
pub use self::CUmem_range_attribute_enum as CUmem_range_attribute;
#[cfg(any(feature = "cuda-12080"))]
pub use self::CUmemcpy3DOperandType_enum as CUmemcpy3DOperandType;
#[cfg(any(feature = "cuda-12080"))]
pub use self::CUmemcpyFlags_enum as CUmemcpyFlags;
#[cfg(any(feature = "cuda-12080"))]
pub use self::CUmemcpySrcAccessOrder_enum as CUmemcpySrcAccessOrder;
pub use self::CUmemorytype_enum as CUmemorytype;
#[cfg(
    any(
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
    )
)]
pub use self::CUmoduleLoadingMode_enum as CUmoduleLoadingMode;
#[cfg(
    any(
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUmulticastGranularity_flags_enum as CUmulticastGranularity_flags;
pub use self::CUoccupancy_flags_enum as CUoccupancy_flags;
pub use self::CUoutput_mode_enum as CUoutput_mode;
pub use self::CUpointer_attribute_enum as CUpointer_attribute;
#[cfg(any(feature = "cuda-12080"))]
pub use self::CUprocessState_enum as CUprocessState;
pub use self::CUresourceViewFormat_enum as CUresourceViewFormat;
pub use self::CUresourcetype_enum as CUresourcetype;
pub use self::CUshared_carveout_enum as CUshared_carveout;
pub use self::CUsharedconfig_enum as CUsharedconfig;
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    )
)]
pub use self::CUstreamAttrID_enum as CUstreamAttrID;
pub use self::CUstreamBatchMemOpType_enum as CUstreamBatchMemOpType;
pub use self::CUstreamCaptureMode_enum as CUstreamCaptureMode;
pub use self::CUstreamCaptureStatus_enum as CUstreamCaptureStatus;
#[cfg(
    any(
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
    )
)]
pub use self::CUstreamMemoryBarrier_flags_enum as CUstreamMemoryBarrier_flags;
pub use self::CUstreamUpdateCaptureDependencies_flags_enum as CUstreamUpdateCaptureDependencies_flags;
pub use self::CUstreamWaitValue_flags_enum as CUstreamWaitValue_flags;
pub use self::CUstreamWriteValue_flags_enum as CUstreamWriteValue_flags;
pub use self::CUstream_flags_enum as CUstream_flags;
pub use self::CUsynchronizationPolicy_enum as CUsynchronizationPolicy;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUtensorMapDataType_enum as CUtensorMapDataType;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUtensorMapFloatOOBfill_enum as CUtensorMapFloatOOBfill;
#[cfg(any(feature = "cuda-12080"))]
pub use self::CUtensorMapIm2ColWideMode_enum as CUtensorMapIm2ColWideMode;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUtensorMapInterleave_enum as CUtensorMapInterleave;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUtensorMapL2promotion_enum as CUtensorMapL2promotion;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub use self::CUtensorMapSwizzle_enum as CUtensorMapSwizzle;
pub use self::CUuserObjectRetain_flags_enum as CUuserObjectRetain_flags;
pub use self::CUuserObject_flags_enum as CUuserObject_flags;
pub use self::cudaError_enum as CUresult;
pub const CUDA_ARRAY3D_2DARRAY: u32 = 1;
pub const CUDA_ARRAY3D_COLOR_ATTACHMENT: u32 = 32;
pub const CUDA_ARRAY3D_CUBEMAP: u32 = 4;
#[cfg(
    any(
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
    )
)]
pub const CUDA_ARRAY3D_DEFERRED_MAPPING: u32 = 128;
pub const CUDA_ARRAY3D_DEPTH_TEXTURE: u32 = 16;
pub const CUDA_ARRAY3D_LAYERED: u32 = 1;
pub const CUDA_ARRAY3D_SPARSE: u32 = 64;
pub const CUDA_ARRAY3D_SURFACE_LDST: u32 = 2;
pub const CUDA_ARRAY3D_TEXTURE_GATHER: u32 = 8;
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
pub const CUDA_ARRAY3D_VIDEO_ENCODE_DECODE: u32 = 256;
pub const CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_POST_LAUNCH_SYNC: u32 = 2;
pub const CUDA_COOPERATIVE_LAUNCH_MULTI_DEVICE_NO_PRE_LAUNCH_SYNC: u32 = 1;
pub const CUDA_EXTERNAL_MEMORY_DEDICATED: u32 = 1;
pub const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_SKIP_NVSCIBUF_MEMSYNC: u32 = 1;
pub const CUDA_EXTERNAL_SEMAPHORE_WAIT_SKIP_NVSCIBUF_MEMSYNC: u32 = 2;
pub const CUDA_NVSCISYNC_ATTR_SIGNAL: u32 = 1;
pub const CUDA_NVSCISYNC_ATTR_WAIT: u32 = 2;
#[cfg(any(feature = "cuda-11040"))]
pub const CUDA_VERSION: u32 = 11040;
#[cfg(any(feature = "cuda-11050"))]
pub const CUDA_VERSION: u32 = 11050;
#[cfg(any(feature = "cuda-11060"))]
pub const CUDA_VERSION: u32 = 11060;
#[cfg(any(feature = "cuda-11070"))]
pub const CUDA_VERSION: u32 = 11070;
#[cfg(any(feature = "cuda-11080"))]
pub const CUDA_VERSION: u32 = 11080;
#[cfg(any(feature = "cuda-12000"))]
pub const CUDA_VERSION: u32 = 12000;
#[cfg(any(feature = "cuda-12010"))]
pub const CUDA_VERSION: u32 = 12010;
#[cfg(any(feature = "cuda-12020"))]
pub const CUDA_VERSION: u32 = 12020;
#[cfg(any(feature = "cuda-12030"))]
pub const CUDA_VERSION: u32 = 12030;
#[cfg(any(feature = "cuda-12040"))]
pub const CUDA_VERSION: u32 = 12040;
#[cfg(any(feature = "cuda-12050"))]
pub const CUDA_VERSION: u32 = 12050;
#[cfg(any(feature = "cuda-12060"))]
pub const CUDA_VERSION: u32 = 12060;
#[cfg(any(feature = "cuda-12080"))]
pub const CUDA_VERSION: u32 = 12080;
pub const CU_ARRAY_SPARSE_PROPERTIES_SINGLE_MIPTAIL: u32 = 1;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub const CU_COMPUTE_ACCELERATED_TARGET_BASE: u32 = 65536;
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub const CU_GRAPH_COND_ASSIGN_DEFAULT: u32 = 1;
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub const CU_GRAPH_KERNEL_NODE_PORT_DEFAULT: u32 = 0;
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub const CU_GRAPH_KERNEL_NODE_PORT_LAUNCH_ORDER: u32 = 2;
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub const CU_GRAPH_KERNEL_NODE_PORT_PROGRAMMATIC: u32 = 1;
pub const CU_IPC_HANDLE_SIZE: u32 = 64;
#[cfg(
    any(
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
    )
)]
pub const CU_LAUNCH_PARAM_BUFFER_POINTER_AS_INT: u32 = 1;
#[cfg(
    any(
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
    )
)]
pub const CU_LAUNCH_PARAM_BUFFER_SIZE_AS_INT: u32 = 2;
#[cfg(
    any(
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
    )
)]
pub const CU_LAUNCH_PARAM_END_AS_INT: u32 = 0;
pub const CU_MEMHOSTALLOC_DEVICEMAP: u32 = 2;
pub const CU_MEMHOSTALLOC_PORTABLE: u32 = 1;
pub const CU_MEMHOSTALLOC_WRITECOMBINED: u32 = 4;
pub const CU_MEMHOSTREGISTER_DEVICEMAP: u32 = 2;
pub const CU_MEMHOSTREGISTER_IOMEMORY: u32 = 4;
pub const CU_MEMHOSTREGISTER_PORTABLE: u32 = 1;
pub const CU_MEMHOSTREGISTER_READ_ONLY: u32 = 8;
#[cfg(any(feature = "cuda-12080"))]
pub const CU_MEM_CREATE_USAGE_HW_DECOMPRESS: u32 = 2;
pub const CU_MEM_CREATE_USAGE_TILE_POOL: u32 = 1;
#[cfg(any(feature = "cuda-12080"))]
pub const CU_MEM_POOL_CREATE_USAGE_HW_DECOMPRESS: u32 = 2;
pub const CU_PARAM_TR_DEFAULT: i32 = -1;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub const CU_TENSOR_MAP_NUM_QWORDS: u32 = 16;
pub const CU_TRSA_OVERRIDE_FORMAT: u32 = 1;
pub const CU_TRSF_DISABLE_TRILINEAR_OPTIMIZATION: u32 = 32;
pub const CU_TRSF_NORMALIZED_COORDINATES: u32 = 2;
pub const CU_TRSF_READ_AS_INTEGER: u32 = 1;
#[cfg(
    any(
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
    )
)]
pub const CU_TRSF_SEAMLESS_CUBEMAP: u32 = 64;
pub const CU_TRSF_SRGB: u32 = 16;
pub type CUDA_ARRAY3D_DESCRIPTOR = CUDA_ARRAY3D_DESCRIPTOR_v2;
pub type CUDA_ARRAY3D_DESCRIPTOR_v2 = CUDA_ARRAY3D_DESCRIPTOR_st;
pub type CUDA_ARRAY_DESCRIPTOR = CUDA_ARRAY_DESCRIPTOR_v2;
pub type CUDA_ARRAY_DESCRIPTOR_v2 = CUDA_ARRAY_DESCRIPTOR_st;
#[cfg(
    any(
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
    )
)]
pub type CUDA_ARRAY_MEMORY_REQUIREMENTS = CUDA_ARRAY_MEMORY_REQUIREMENTS_v1;
#[cfg(
    any(
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
    )
)]
pub type CUDA_ARRAY_MEMORY_REQUIREMENTS_v1 = CUDA_ARRAY_MEMORY_REQUIREMENTS_st;
pub type CUDA_ARRAY_SPARSE_PROPERTIES = CUDA_ARRAY_SPARSE_PROPERTIES_v1;
pub type CUDA_ARRAY_SPARSE_PROPERTIES_v1 = CUDA_ARRAY_SPARSE_PROPERTIES_st;
#[cfg(
    any(
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010"
    )
)]
pub type CUDA_BATCH_MEM_OP_NODE_PARAMS = CUDA_BATCH_MEM_OP_NODE_PARAMS_st;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_BATCH_MEM_OP_NODE_PARAMS = CUDA_BATCH_MEM_OP_NODE_PARAMS_v1;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_BATCH_MEM_OP_NODE_PARAMS_v1 = CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_BATCH_MEM_OP_NODE_PARAMS_v2 = CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_CHILD_GRAPH_NODE_PARAMS = CUDA_CHILD_GRAPH_NODE_PARAMS_st;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_EVENT_RECORD_NODE_PARAMS = CUDA_EVENT_RECORD_NODE_PARAMS_st;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_EVENT_WAIT_NODE_PARAMS = CUDA_EVENT_WAIT_NODE_PARAMS_st;
pub type CUDA_EXTERNAL_MEMORY_BUFFER_DESC = CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1;
pub type CUDA_EXTERNAL_MEMORY_BUFFER_DESC_v1 = CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st;
pub type CUDA_EXTERNAL_MEMORY_HANDLE_DESC = CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1;
pub type CUDA_EXTERNAL_MEMORY_HANDLE_DESC_v1 = CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st;
pub type CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC = CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1;
pub type CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_v1 = CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st;
pub type CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1;
pub type CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_v1 = CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st;
pub type CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1;
pub type CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_v1 = CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st;
pub type CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1;
pub type CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_v1 = CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st;
pub type CUDA_EXT_SEM_SIGNAL_NODE_PARAMS = CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1;
pub type CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v1 = CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2 = CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st;
pub type CUDA_EXT_SEM_WAIT_NODE_PARAMS = CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1;
pub type CUDA_EXT_SEM_WAIT_NODE_PARAMS_v1 = CUDA_EXT_SEM_WAIT_NODE_PARAMS_st;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2 = CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_GRAPH_INSTANTIATE_PARAMS = CUDA_GRAPH_INSTANTIATE_PARAMS_st;
pub type CUDA_HOST_NODE_PARAMS = CUDA_HOST_NODE_PARAMS_v1;
pub type CUDA_HOST_NODE_PARAMS_v1 = CUDA_HOST_NODE_PARAMS_st;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_HOST_NODE_PARAMS_v2 = CUDA_HOST_NODE_PARAMS_v2_st;
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080"
    )
)]
pub type CUDA_KERNEL_NODE_PARAMS = CUDA_KERNEL_NODE_PARAMS_v1;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_KERNEL_NODE_PARAMS = CUDA_KERNEL_NODE_PARAMS_v2;
pub type CUDA_KERNEL_NODE_PARAMS_v1 = CUDA_KERNEL_NODE_PARAMS_st;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_KERNEL_NODE_PARAMS_v2 = CUDA_KERNEL_NODE_PARAMS_v2_st;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_KERNEL_NODE_PARAMS_v3 = CUDA_KERNEL_NODE_PARAMS_v3_st;
pub type CUDA_LAUNCH_PARAMS = CUDA_LAUNCH_PARAMS_v1;
pub type CUDA_LAUNCH_PARAMS_v1 = CUDA_LAUNCH_PARAMS_st;
pub type CUDA_MEMCPY2D = CUDA_MEMCPY2D_v2;
pub type CUDA_MEMCPY2D_v2 = CUDA_MEMCPY2D_st;
pub type CUDA_MEMCPY3D = CUDA_MEMCPY3D_v2;
#[cfg(any(feature = "cuda-12080"))]
pub type CUDA_MEMCPY3D_BATCH_OP = CUDA_MEMCPY3D_BATCH_OP_v1;
#[cfg(any(feature = "cuda-12080"))]
pub type CUDA_MEMCPY3D_BATCH_OP_v1 = CUDA_MEMCPY3D_BATCH_OP_st;
pub type CUDA_MEMCPY3D_PEER = CUDA_MEMCPY3D_PEER_v1;
pub type CUDA_MEMCPY3D_PEER_v1 = CUDA_MEMCPY3D_PEER_st;
pub type CUDA_MEMCPY3D_v2 = CUDA_MEMCPY3D_st;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_MEMCPY_NODE_PARAMS = CUDA_MEMCPY_NODE_PARAMS_st;
pub type CUDA_MEMSET_NODE_PARAMS = CUDA_MEMSET_NODE_PARAMS_v1;
pub type CUDA_MEMSET_NODE_PARAMS_v1 = CUDA_MEMSET_NODE_PARAMS_st;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_MEMSET_NODE_PARAMS_v2 = CUDA_MEMSET_NODE_PARAMS_v2_st;
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010"
    )
)]
pub type CUDA_MEM_ALLOC_NODE_PARAMS = CUDA_MEM_ALLOC_NODE_PARAMS_st;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_MEM_ALLOC_NODE_PARAMS = CUDA_MEM_ALLOC_NODE_PARAMS_v1;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_MEM_ALLOC_NODE_PARAMS_v1 = CUDA_MEM_ALLOC_NODE_PARAMS_v1_st;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_MEM_ALLOC_NODE_PARAMS_v2 = CUDA_MEM_ALLOC_NODE_PARAMS_v2_st;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUDA_MEM_FREE_NODE_PARAMS = CUDA_MEM_FREE_NODE_PARAMS_st;
pub type CUDA_POINTER_ATTRIBUTE_P2P_TOKENS = CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1;
pub type CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_v1 = CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st;
pub type CUDA_RESOURCE_DESC = CUDA_RESOURCE_DESC_v1;
pub type CUDA_RESOURCE_DESC_v1 = CUDA_RESOURCE_DESC_st;
pub type CUDA_RESOURCE_VIEW_DESC = CUDA_RESOURCE_VIEW_DESC_v1;
pub type CUDA_RESOURCE_VIEW_DESC_v1 = CUDA_RESOURCE_VIEW_DESC_st;
pub type CUDA_TEXTURE_DESC = CUDA_TEXTURE_DESC_v1;
pub type CUDA_TEXTURE_DESC_v1 = CUDA_TEXTURE_DESC_st;
pub type CUaccessPolicyWindow = CUaccessPolicyWindow_v1;
pub type CUaccessPolicyWindow_v1 = CUaccessPolicyWindow_st;
pub type CUarray = *mut CUarray_st;
pub type CUarrayMapInfo = CUarrayMapInfo_v1;
pub type CUarrayMapInfo_v1 = CUarrayMapInfo_st;
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUasyncCallback = ::core::option::Option<
    unsafe extern "C" fn(
        info: *mut CUasyncNotificationInfo,
        userData: *mut ::core::ffi::c_void,
        callback: CUasyncCallbackHandle,
    ),
>;
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUasyncCallbackHandle = *mut CUasyncCallbackEntry_st;
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUasyncNotificationInfo = CUasyncNotificationInfo_st;
#[cfg(any(feature = "cuda-12080"))]
pub type CUcheckpointCheckpointArgs = CUcheckpointCheckpointArgs_st;
#[cfg(any(feature = "cuda-12080"))]
pub type CUcheckpointLockArgs = CUcheckpointLockArgs_st;
#[cfg(any(feature = "cuda-12080"))]
pub type CUcheckpointRestoreArgs = CUcheckpointRestoreArgs_st;
#[cfg(any(feature = "cuda-12080"))]
pub type CUcheckpointUnlockArgs = CUcheckpointUnlockArgs_st;
pub type CUcontext = *mut CUctx_st;
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
pub type CUctxCigParam = CUctxCigParam_st;
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
pub type CUctxCreateParams = CUctxCreateParams_st;
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUdevResource = CUdevResource_v1;
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUdevResourceDesc = *mut CUdevResourceDesc_st;
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUdevResource_v1 = CUdevResource_st;
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUdevSmResource = CUdevSmResource_st;
pub type CUdevice = CUdevice_v1;
pub type CUdevice_v1 = ::core::ffi::c_int;
pub type CUdeviceptr = CUdeviceptr_v2;
pub type CUdeviceptr_v2 = ::core::ffi::c_ulonglong;
pub type CUdevprop = CUdevprop_v1;
pub type CUdevprop_v1 = CUdevprop_st;
pub type CUevent = *mut CUevent_st;
pub type CUexecAffinityParam = CUexecAffinityParam_v1;
pub type CUexecAffinityParam_v1 = CUexecAffinityParam_st;
pub type CUexecAffinitySmCount = CUexecAffinitySmCount_v1;
pub type CUexecAffinitySmCount_v1 = CUexecAffinitySmCount_st;
#[cfg(any(feature = "cuda-12080"))]
pub type CUextent3D = CUextent3D_v1;
#[cfg(any(feature = "cuda-12080"))]
pub type CUextent3D_v1 = CUextent3D_st;
pub type CUexternalMemory = *mut CUextMemory_st;
pub type CUexternalSemaphore = *mut CUextSemaphore_st;
pub type CUfunction = *mut CUfunc_st;
pub type CUgraph = *mut CUgraph_st;
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUgraphConditionalHandle = cuuint64_t;
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUgraphDeviceNode = *mut CUgraphDeviceUpdatableNode_st;
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUgraphEdgeData = CUgraphEdgeData_st;
pub type CUgraphExec = *mut CUgraphExec_st;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUgraphExecUpdateResultInfo = CUgraphExecUpdateResultInfo_v1;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUgraphExecUpdateResultInfo_v1 = CUgraphExecUpdateResultInfo_st;
pub type CUgraphNode = *mut CUgraphNode_st;
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUgraphNodeParams = CUgraphNodeParams_st;
pub type CUgraphicsResource = *mut CUgraphicsResource_st;
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUgreenCtx = *mut CUgreenCtx_st;
pub type CUhostFn = ::core::option::Option<
    unsafe extern "C" fn(userData: *mut ::core::ffi::c_void),
>;
pub type CUipcEventHandle = CUipcEventHandle_v1;
pub type CUipcEventHandle_v1 = CUipcEventHandle_st;
pub type CUipcMemHandle = CUipcMemHandle_v1;
pub type CUipcMemHandle_v1 = CUipcMemHandle_st;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUkernel = *mut CUkern_st;
pub type CUkernelNodeAttrValue = CUkernelNodeAttrValue_v1;
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    )
)]
pub type CUkernelNodeAttrValue_v1 = CUkernelNodeAttrValue_union;
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUkernelNodeAttrValue_v1 = CUlaunchAttributeValue;
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUlaunchAttribute = CUlaunchAttribute_st;
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUlaunchAttributeValue = CUlaunchAttributeValue_union;
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUlaunchConfig = CUlaunchConfig_st;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUlaunchMemSyncDomainMap = CUlaunchMemSyncDomainMap_st;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUlibrary = *mut CUlib_st;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUlibraryHostUniversalFunctionAndDataTable = CUlibraryHostUniversalFunctionAndDataTable_st;
pub type CUlinkState = *mut CUlinkState_st;
pub type CUmemAccessDesc = CUmemAccessDesc_v1;
pub type CUmemAccessDesc_v1 = CUmemAccessDesc_st;
pub type CUmemAllocationProp = CUmemAllocationProp_v1;
pub type CUmemAllocationProp_v1 = CUmemAllocationProp_st;
#[cfg(any(feature = "cuda-12080"))]
pub type CUmemDecompressParams = CUmemDecompressParams_st;
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUmemFabricHandle = CUmemFabricHandle_v1;
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUmemFabricHandle_v1 = CUmemFabricHandle_st;
pub type CUmemGenericAllocationHandle = CUmemGenericAllocationHandle_v1;
pub type CUmemGenericAllocationHandle_v1 = ::core::ffi::c_ulonglong;
pub type CUmemLocation = CUmemLocation_v1;
pub type CUmemLocation_v1 = CUmemLocation_st;
pub type CUmemPoolProps = CUmemPoolProps_v1;
pub type CUmemPoolProps_v1 = CUmemPoolProps_st;
pub type CUmemPoolPtrExportData = CUmemPoolPtrExportData_v1;
pub type CUmemPoolPtrExportData_v1 = CUmemPoolPtrExportData_st;
#[cfg(any(feature = "cuda-12080"))]
pub type CUmemcpy3DOperand = CUmemcpy3DOperand_v1;
#[cfg(any(feature = "cuda-12080"))]
pub type CUmemcpy3DOperand_v1 = CUmemcpy3DOperand_st;
#[cfg(any(feature = "cuda-12080"))]
pub type CUmemcpyAttributes = CUmemcpyAttributes_v1;
#[cfg(any(feature = "cuda-12080"))]
pub type CUmemcpyAttributes_v1 = CUmemcpyAttributes_st;
pub type CUmemoryPool = *mut CUmemPoolHandle_st;
pub type CUmipmappedArray = *mut CUmipmappedArray_st;
pub type CUmodule = *mut CUmod_st;
#[cfg(
    any(
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUmulticastObjectProp = CUmulticastObjectProp_v1;
#[cfg(
    any(
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUmulticastObjectProp_v1 = CUmulticastObjectProp_st;
pub type CUoccupancyB2DSize = ::core::option::Option<
    unsafe extern "C" fn(blockSize: ::core::ffi::c_int) -> usize,
>;
#[cfg(any(feature = "cuda-12080"))]
pub type CUoffset3D = CUoffset3D_v1;
#[cfg(any(feature = "cuda-12080"))]
pub type CUoffset3D_v1 = CUoffset3D_st;
pub type CUstream = *mut CUstream_st;
pub type CUstreamAttrValue = CUstreamAttrValue_v1;
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    )
)]
pub type CUstreamAttrValue_v1 = CUstreamAttrValue_union;
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUstreamAttrValue_v1 = CUlaunchAttributeValue;
pub type CUstreamBatchMemOpParams = CUstreamBatchMemOpParams_v1;
pub type CUstreamBatchMemOpParams_v1 = CUstreamBatchMemOpParams_union;
pub type CUstreamCallback = ::core::option::Option<
    unsafe extern "C" fn(
        hStream: CUstream,
        status: CUresult,
        userData: *mut ::core::ffi::c_void,
    ),
>;
pub type CUsurfObject = CUsurfObject_v1;
pub type CUsurfObject_v1 = ::core::ffi::c_ulonglong;
pub type CUsurfref = *mut CUsurfref_st;
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
pub type CUtensorMap = CUtensorMap_st;
pub type CUtexObject = CUtexObject_v1;
pub type CUtexObject_v1 = ::core::ffi::c_ulonglong;
pub type CUtexref = *mut CUtexref_st;
pub type CUuserObject = *mut CUuserObject_st;
pub type CUuuid = CUuuid_st;
pub type cuuint32_t = u32;
pub type cuuint64_t = u64;
#[cfg(any(feature = "cuda-12050"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUCoredumpGenerationFlags {
    CU_COREDUMP_DEFAULT_FLAGS = 0,
    CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES = 1,
    CU_COREDUMP_SKIP_GLOBAL_MEMORY = 2,
    CU_COREDUMP_SKIP_SHARED_MEMORY = 4,
    CU_COREDUMP_SKIP_LOCAL_MEMORY = 8,
    CU_COREDUMP_SKIP_ABORT = 16,
    CU_COREDUMP_LIGHTWEIGHT_FLAGS = 15,
}
#[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUCoredumpGenerationFlags {
    CU_COREDUMP_DEFAULT_FLAGS = 0,
    CU_COREDUMP_SKIP_NONRELOCATED_ELF_IMAGES = 1,
    CU_COREDUMP_SKIP_GLOBAL_MEMORY = 2,
    CU_COREDUMP_SKIP_SHARED_MEMORY = 4,
    CU_COREDUMP_SKIP_LOCAL_MEMORY = 8,
    CU_COREDUMP_SKIP_ABORT = 16,
    CU_COREDUMP_SKIP_CONSTBANK_MEMORY = 32,
    CU_COREDUMP_LIGHTWEIGHT_FLAGS = 47,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS_enum {
    CU_POINTER_ATTRIBUTE_ACCESS_FLAG_NONE = 0,
    CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READ = 1,
    CU_POINTER_ATTRIBUTE_ACCESS_FLAG_READWRITE = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUGPUDirectRDMAWritesOrdering_enum {
    CU_GPU_DIRECT_RDMA_WRITES_ORDERING_NONE = 0,
    CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER = 100,
    CU_GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES = 200,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUaccessProperty_enum {
    CU_ACCESS_PROPERTY_NORMAL = 0,
    CU_ACCESS_PROPERTY_STREAMING = 1,
    CU_ACCESS_PROPERTY_PERSISTING = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUaddress_mode_enum {
    CU_TR_ADDRESS_MODE_WRAP = 0,
    CU_TR_ADDRESS_MODE_CLAMP = 1,
    CU_TR_ADDRESS_MODE_MIRROR = 2,
    CU_TR_ADDRESS_MODE_BORDER = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUarraySparseSubresourceType_enum {
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_SPARSE_LEVEL = 0,
    CU_ARRAY_SPARSE_SUBRESOURCE_TYPE_MIPTAIL = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUarray_cubemap_face_enum {
    CU_CUBEMAP_FACE_POSITIVE_X = 0,
    CU_CUBEMAP_FACE_NEGATIVE_X = 1,
    CU_CUBEMAP_FACE_POSITIVE_Y = 2,
    CU_CUBEMAP_FACE_NEGATIVE_Y = 3,
    CU_CUBEMAP_FACE_POSITIVE_Z = 4,
    CU_CUBEMAP_FACE_NEGATIVE_Z = 5,
}
#[cfg(any(feature = "cuda-11040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUarray_format_enum {
    CU_AD_FORMAT_UNSIGNED_INT8 = 1,
    CU_AD_FORMAT_UNSIGNED_INT16 = 2,
    CU_AD_FORMAT_UNSIGNED_INT32 = 3,
    CU_AD_FORMAT_SIGNED_INT8 = 8,
    CU_AD_FORMAT_SIGNED_INT16 = 9,
    CU_AD_FORMAT_SIGNED_INT32 = 10,
    CU_AD_FORMAT_HALF = 16,
    CU_AD_FORMAT_FLOAT = 32,
    CU_AD_FORMAT_NV12 = 176,
}
#[cfg(
    any(
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUarray_format_enum {
    CU_AD_FORMAT_UNSIGNED_INT8 = 1,
    CU_AD_FORMAT_UNSIGNED_INT16 = 2,
    CU_AD_FORMAT_UNSIGNED_INT32 = 3,
    CU_AD_FORMAT_SIGNED_INT8 = 8,
    CU_AD_FORMAT_SIGNED_INT16 = 9,
    CU_AD_FORMAT_SIGNED_INT32 = 10,
    CU_AD_FORMAT_HALF = 16,
    CU_AD_FORMAT_FLOAT = 32,
    CU_AD_FORMAT_NV12 = 176,
    CU_AD_FORMAT_UNORM_INT8X1 = 192,
    CU_AD_FORMAT_UNORM_INT8X2 = 193,
    CU_AD_FORMAT_UNORM_INT8X4 = 194,
    CU_AD_FORMAT_UNORM_INT16X1 = 195,
    CU_AD_FORMAT_UNORM_INT16X2 = 196,
    CU_AD_FORMAT_UNORM_INT16X4 = 197,
    CU_AD_FORMAT_SNORM_INT8X1 = 198,
    CU_AD_FORMAT_SNORM_INT8X2 = 199,
    CU_AD_FORMAT_SNORM_INT8X4 = 200,
    CU_AD_FORMAT_SNORM_INT16X1 = 201,
    CU_AD_FORMAT_SNORM_INT16X2 = 202,
    CU_AD_FORMAT_SNORM_INT16X4 = 203,
    CU_AD_FORMAT_BC1_UNORM = 145,
    CU_AD_FORMAT_BC1_UNORM_SRGB = 146,
    CU_AD_FORMAT_BC2_UNORM = 147,
    CU_AD_FORMAT_BC2_UNORM_SRGB = 148,
    CU_AD_FORMAT_BC3_UNORM = 149,
    CU_AD_FORMAT_BC3_UNORM_SRGB = 150,
    CU_AD_FORMAT_BC4_UNORM = 151,
    CU_AD_FORMAT_BC4_SNORM = 152,
    CU_AD_FORMAT_BC5_UNORM = 153,
    CU_AD_FORMAT_BC5_SNORM = 154,
    CU_AD_FORMAT_BC6H_UF16 = 155,
    CU_AD_FORMAT_BC6H_SF16 = 156,
    CU_AD_FORMAT_BC7_UNORM = 157,
    CU_AD_FORMAT_BC7_UNORM_SRGB = 158,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUarray_format_enum {
    CU_AD_FORMAT_UNSIGNED_INT8 = 1,
    CU_AD_FORMAT_UNSIGNED_INT16 = 2,
    CU_AD_FORMAT_UNSIGNED_INT32 = 3,
    CU_AD_FORMAT_SIGNED_INT8 = 8,
    CU_AD_FORMAT_SIGNED_INT16 = 9,
    CU_AD_FORMAT_SIGNED_INT32 = 10,
    CU_AD_FORMAT_HALF = 16,
    CU_AD_FORMAT_FLOAT = 32,
    CU_AD_FORMAT_NV12 = 176,
    CU_AD_FORMAT_UNORM_INT8X1 = 192,
    CU_AD_FORMAT_UNORM_INT8X2 = 193,
    CU_AD_FORMAT_UNORM_INT8X4 = 194,
    CU_AD_FORMAT_UNORM_INT16X1 = 195,
    CU_AD_FORMAT_UNORM_INT16X2 = 196,
    CU_AD_FORMAT_UNORM_INT16X4 = 197,
    CU_AD_FORMAT_SNORM_INT8X1 = 198,
    CU_AD_FORMAT_SNORM_INT8X2 = 199,
    CU_AD_FORMAT_SNORM_INT8X4 = 200,
    CU_AD_FORMAT_SNORM_INT16X1 = 201,
    CU_AD_FORMAT_SNORM_INT16X2 = 202,
    CU_AD_FORMAT_SNORM_INT16X4 = 203,
    CU_AD_FORMAT_BC1_UNORM = 145,
    CU_AD_FORMAT_BC1_UNORM_SRGB = 146,
    CU_AD_FORMAT_BC2_UNORM = 147,
    CU_AD_FORMAT_BC2_UNORM_SRGB = 148,
    CU_AD_FORMAT_BC3_UNORM = 149,
    CU_AD_FORMAT_BC3_UNORM_SRGB = 150,
    CU_AD_FORMAT_BC4_UNORM = 151,
    CU_AD_FORMAT_BC4_SNORM = 152,
    CU_AD_FORMAT_BC5_UNORM = 153,
    CU_AD_FORMAT_BC5_SNORM = 154,
    CU_AD_FORMAT_BC6H_UF16 = 155,
    CU_AD_FORMAT_BC6H_SF16 = 156,
    CU_AD_FORMAT_BC7_UNORM = 157,
    CU_AD_FORMAT_BC7_UNORM_SRGB = 158,
    CU_AD_FORMAT_P010 = 159,
    CU_AD_FORMAT_P016 = 161,
    CU_AD_FORMAT_NV16 = 162,
    CU_AD_FORMAT_P210 = 163,
    CU_AD_FORMAT_P216 = 164,
    CU_AD_FORMAT_YUY2 = 165,
    CU_AD_FORMAT_Y210 = 166,
    CU_AD_FORMAT_Y216 = 167,
    CU_AD_FORMAT_AYUV = 168,
    CU_AD_FORMAT_Y410 = 169,
    CU_AD_FORMAT_Y416 = 177,
    CU_AD_FORMAT_Y444_PLANAR8 = 178,
    CU_AD_FORMAT_Y444_PLANAR10 = 179,
    CU_AD_FORMAT_MAX = 2147483647,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUarray_format_enum {
    CU_AD_FORMAT_UNSIGNED_INT8 = 1,
    CU_AD_FORMAT_UNSIGNED_INT16 = 2,
    CU_AD_FORMAT_UNSIGNED_INT32 = 3,
    CU_AD_FORMAT_SIGNED_INT8 = 8,
    CU_AD_FORMAT_SIGNED_INT16 = 9,
    CU_AD_FORMAT_SIGNED_INT32 = 10,
    CU_AD_FORMAT_HALF = 16,
    CU_AD_FORMAT_FLOAT = 32,
    CU_AD_FORMAT_NV12 = 176,
    CU_AD_FORMAT_UNORM_INT8X1 = 192,
    CU_AD_FORMAT_UNORM_INT8X2 = 193,
    CU_AD_FORMAT_UNORM_INT8X4 = 194,
    CU_AD_FORMAT_UNORM_INT16X1 = 195,
    CU_AD_FORMAT_UNORM_INT16X2 = 196,
    CU_AD_FORMAT_UNORM_INT16X4 = 197,
    CU_AD_FORMAT_SNORM_INT8X1 = 198,
    CU_AD_FORMAT_SNORM_INT8X2 = 199,
    CU_AD_FORMAT_SNORM_INT8X4 = 200,
    CU_AD_FORMAT_SNORM_INT16X1 = 201,
    CU_AD_FORMAT_SNORM_INT16X2 = 202,
    CU_AD_FORMAT_SNORM_INT16X4 = 203,
    CU_AD_FORMAT_BC1_UNORM = 145,
    CU_AD_FORMAT_BC1_UNORM_SRGB = 146,
    CU_AD_FORMAT_BC2_UNORM = 147,
    CU_AD_FORMAT_BC2_UNORM_SRGB = 148,
    CU_AD_FORMAT_BC3_UNORM = 149,
    CU_AD_FORMAT_BC3_UNORM_SRGB = 150,
    CU_AD_FORMAT_BC4_UNORM = 151,
    CU_AD_FORMAT_BC4_SNORM = 152,
    CU_AD_FORMAT_BC5_UNORM = 153,
    CU_AD_FORMAT_BC5_SNORM = 154,
    CU_AD_FORMAT_BC6H_UF16 = 155,
    CU_AD_FORMAT_BC6H_SF16 = 156,
    CU_AD_FORMAT_BC7_UNORM = 157,
    CU_AD_FORMAT_BC7_UNORM_SRGB = 158,
    CU_AD_FORMAT_P010 = 159,
    CU_AD_FORMAT_P016 = 161,
    CU_AD_FORMAT_NV16 = 162,
    CU_AD_FORMAT_P210 = 163,
    CU_AD_FORMAT_P216 = 164,
    CU_AD_FORMAT_YUY2 = 165,
    CU_AD_FORMAT_Y210 = 166,
    CU_AD_FORMAT_Y216 = 167,
    CU_AD_FORMAT_AYUV = 168,
    CU_AD_FORMAT_Y410 = 169,
    CU_AD_FORMAT_Y416 = 177,
    CU_AD_FORMAT_Y444_PLANAR8 = 178,
    CU_AD_FORMAT_Y444_PLANAR10 = 179,
    CU_AD_FORMAT_YUV444_8bit_SemiPlanar = 180,
    CU_AD_FORMAT_YUV444_16bit_SemiPlanar = 181,
    CU_AD_FORMAT_UNORM_INT_101010_2 = 80,
    CU_AD_FORMAT_MAX = 2147483647,
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUasyncNotificationType_enum {
    CU_ASYNC_NOTIFICATION_TYPE_OVER_BUDGET = 1,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUcigDataType_enum {
    CIG_DATA_TYPE_D3D12_COMMAND_QUEUE = 1,
}
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUclusterSchedulingPolicy_enum {
    CU_CLUSTER_SCHEDULING_POLICY_DEFAULT = 0,
    CU_CLUSTER_SCHEDULING_POLICY_SPREAD = 1,
    CU_CLUSTER_SCHEDULING_POLICY_LOAD_BALANCING = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUcomputemode_enum {
    CU_COMPUTEMODE_DEFAULT = 0,
    CU_COMPUTEMODE_PROHIBITED = 2,
    CU_COMPUTEMODE_EXCLUSIVE_PROCESS = 3,
}
#[cfg(
    any(
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUcoredumpSettings_enum {
    CU_COREDUMP_ENABLE_ON_EXCEPTION = 1,
    CU_COREDUMP_TRIGGER_HOST = 2,
    CU_COREDUMP_LIGHTWEIGHT = 3,
    CU_COREDUMP_ENABLE_USER_TRIGGER = 4,
    CU_COREDUMP_FILE = 5,
    CU_COREDUMP_PIPE = 6,
    CU_COREDUMP_MAX = 7,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUcoredumpSettings_enum {
    CU_COREDUMP_ENABLE_ON_EXCEPTION = 1,
    CU_COREDUMP_TRIGGER_HOST = 2,
    CU_COREDUMP_LIGHTWEIGHT = 3,
    CU_COREDUMP_ENABLE_USER_TRIGGER = 4,
    CU_COREDUMP_FILE = 5,
    CU_COREDUMP_PIPE = 6,
    CU_COREDUMP_GENERATION_FLAGS = 7,
    CU_COREDUMP_MAX = 8,
}
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUctx_flags_enum {
    CU_CTX_SCHED_AUTO = 0,
    CU_CTX_SCHED_SPIN = 1,
    CU_CTX_SCHED_YIELD = 2,
    CU_CTX_SCHED_BLOCKING_SYNC = 4,
    CU_CTX_SCHED_MASK = 7,
    CU_CTX_MAP_HOST = 8,
    CU_CTX_LMEM_RESIZE_TO_MAX = 16,
    CU_CTX_FLAGS_MASK = 31,
}
#[cfg(
    any(
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUctx_flags_enum {
    CU_CTX_SCHED_AUTO = 0,
    CU_CTX_SCHED_SPIN = 1,
    CU_CTX_SCHED_YIELD = 2,
    CU_CTX_SCHED_BLOCKING_SYNC = 4,
    CU_CTX_SCHED_MASK = 7,
    CU_CTX_MAP_HOST = 8,
    CU_CTX_LMEM_RESIZE_TO_MAX = 16,
    CU_CTX_COREDUMP_ENABLE = 32,
    CU_CTX_USER_COREDUMP_ENABLE = 64,
    CU_CTX_SYNC_MEMOPS = 128,
    CU_CTX_FLAGS_MASK = 255,
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevResourceType {
    CU_DEV_RESOURCE_TYPE_INVALID = 0,
    CU_DEV_RESOURCE_TYPE_SM = 1,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevSmResourceSplit_flags {
    CU_DEV_SM_RESOURCE_SPLIT_IGNORE_SM_COSCHEDULING = 1,
    CU_DEV_SM_RESOURCE_SPLIT_MAX_POTENTIAL_CLUSTER_SIZE = 2,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdeviceNumaConfig_enum {
    CU_DEVICE_NUMA_CONFIG_NONE = 0,
    CU_DEVICE_NUMA_CONFIG_NUMA_NODE = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_P2PAttribute_enum {
    CU_DEVICE_P2P_ATTRIBUTE_PERFORMANCE_RANK = 1,
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_SUPPORTED = 2,
    CU_DEVICE_P2P_ATTRIBUTE_NATIVE_ATOMIC_SUPPORTED = 3,
    CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED = 4,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_MAX = 120,
}
#[cfg(any(feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_MAX = 122,
}
#[cfg(any(feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V2 = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2 = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_MAX = 125,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V2 = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V2 = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_MAX = 125,
}
#[cfg(any(feature = "cuda-12000"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_MAX = 130,
}
#[cfg(any(feature = "cuda-12010"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_MAX = 133,
}
#[cfg(any(feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134,
    CU_DEVICE_ATTRIBUTE_MAX = 135,
}
#[cfg(any(feature = "cuda-12030", feature = "cuda-12040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134,
    CU_DEVICE_ATTRIBUTE_MAX = 135,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134,
    CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED = 135,
    CU_DEVICE_ATTRIBUTE_MAX = 136,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdevice_attribute_enum {
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 1,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X = 2,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y = 3,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z = 4,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X = 5,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y = 6,
    CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z = 7,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY = 9,
    CU_DEVICE_ATTRIBUTE_WARP_SIZE = 10,
    CU_DEVICE_ATTRIBUTE_MAX_PITCH = 11,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK = 12,
    CU_DEVICE_ATTRIBUTE_CLOCK_RATE = 13,
    CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT = 14,
    CU_DEVICE_ATTRIBUTE_GPU_OVERLAP = 15,
    CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT = 16,
    CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT = 17,
    CU_DEVICE_ATTRIBUTE_INTEGRATED = 18,
    CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY = 19,
    CU_DEVICE_ATTRIBUTE_COMPUTE_MODE = 20,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH = 21,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH = 22,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT = 23,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH = 24,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT = 25,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH = 26,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH = 27,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT = 28,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS = 29,
    CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT = 30,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS = 31,
    CU_DEVICE_ATTRIBUTE_ECC_ENABLED = 32,
    CU_DEVICE_ATTRIBUTE_PCI_BUS_ID = 33,
    CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID = 34,
    CU_DEVICE_ATTRIBUTE_TCC_DRIVER = 35,
    CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE = 36,
    CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH = 37,
    CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE = 38,
    CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR = 39,
    CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT = 40,
    CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING = 41,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH = 42,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS = 43,
    CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER = 44,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH = 45,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT = 46,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE = 47,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE = 48,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE = 49,
    CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID = 50,
    CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT = 51,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH = 52,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH = 53,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS = 54,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH = 55,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH = 56,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT = 57,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH = 58,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT = 59,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH = 60,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH = 61,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS = 62,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH = 63,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT = 64,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS = 65,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH = 66,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH = 67,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS = 68,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH = 69,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH = 70,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT = 71,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH = 72,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH = 73,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT = 74,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76,
    CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH = 77,
    CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED = 78,
    CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED = 79,
    CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED = 80,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR = 81,
    CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR = 82,
    CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY = 83,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD = 84,
    CU_DEVICE_ATTRIBUTE_MULTI_GPU_BOARD_GROUP_ID = 85,
    CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED = 86,
    CU_DEVICE_ATTRIBUTE_SINGLE_TO_DOUBLE_PRECISION_PERF_RATIO = 87,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS = 88,
    CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS = 89,
    CU_DEVICE_ATTRIBUTE_COMPUTE_PREEMPTION_SUPPORTED = 90,
    CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM = 91,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS_V1 = 92,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS_V1 = 93,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR_V1 = 94,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH = 95,
    CU_DEVICE_ATTRIBUTE_COOPERATIVE_MULTI_DEVICE_LAUNCH = 96,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES = 98,
    CU_DEVICE_ATTRIBUTE_HOST_REGISTER_SUPPORTED = 99,
    CU_DEVICE_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES = 100,
    CU_DEVICE_ATTRIBUTE_DIRECT_MANAGED_MEM_ACCESS_FROM_HOST = 101,
    CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED = 102,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR_SUPPORTED = 103,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_HANDLE_SUPPORTED = 104,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_WIN32_KMT_HANDLE_SUPPORTED = 105,
    CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR = 106,
    CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED = 107,
    CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE = 108,
    CU_DEVICE_ATTRIBUTE_MAX_ACCESS_POLICY_WINDOW_SIZE = 109,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WITH_CUDA_VMM_SUPPORTED = 110,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 111,
    CU_DEVICE_ATTRIBUTE_SPARSE_CUDA_ARRAY_SUPPORTED = 112,
    CU_DEVICE_ATTRIBUTE_READ_ONLY_HOST_REGISTER_SUPPORTED = 113,
    CU_DEVICE_ATTRIBUTE_TIMELINE_SEMAPHORE_INTEROP_SUPPORTED = 114,
    CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED = 115,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED = 116,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS = 117,
    CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING = 118,
    CU_DEVICE_ATTRIBUTE_MEMPOOL_SUPPORTED_HANDLE_TYPES = 119,
    CU_DEVICE_ATTRIBUTE_CLUSTER_LAUNCH = 120,
    CU_DEVICE_ATTRIBUTE_DEFERRED_MAPPING_CUDA_ARRAY_SUPPORTED = 121,
    CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 122,
    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_WAIT_VALUE_NOR = 123,
    CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED = 124,
    CU_DEVICE_ATTRIBUTE_IPC_EVENT_SUPPORTED = 125,
    CU_DEVICE_ATTRIBUTE_MEM_SYNC_DOMAIN_COUNT = 126,
    CU_DEVICE_ATTRIBUTE_TENSOR_MAP_ACCESS_SUPPORTED = 127,
    CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED = 128,
    CU_DEVICE_ATTRIBUTE_UNIFIED_FUNCTION_POINTERS = 129,
    CU_DEVICE_ATTRIBUTE_NUMA_CONFIG = 130,
    CU_DEVICE_ATTRIBUTE_NUMA_ID = 131,
    CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED = 132,
    CU_DEVICE_ATTRIBUTE_MPS_ENABLED = 133,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID = 134,
    CU_DEVICE_ATTRIBUTE_D3D12_CIG_SUPPORTED = 135,
    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_ALGORITHM_MASK = 136,
    CU_DEVICE_ATTRIBUTE_MEM_DECOMPRESS_MAXIMUM_LENGTH = 137,
    CU_DEVICE_ATTRIBUTE_GPU_PCI_DEVICE_ID = 139,
    CU_DEVICE_ATTRIBUTE_GPU_PCI_SUBSYSTEM_ID = 140,
    CU_DEVICE_ATTRIBUTE_HOST_NUMA_MULTINODE_IPC_SUPPORTED = 143,
    CU_DEVICE_ATTRIBUTE_MAX = 144,
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdriverProcAddressQueryResult_enum {
    CU_GET_PROC_ADDRESS_SUCCESS = 0,
    CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND = 1,
    CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUdriverProcAddress_flags_enum {
    CU_GET_PROC_ADDRESS_DEFAULT = 0,
    CU_GET_PROC_ADDRESS_LEGACY_STREAM = 1,
    CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUevent_flags_enum {
    CU_EVENT_DEFAULT = 0,
    CU_EVENT_BLOCKING_SYNC = 1,
    CU_EVENT_DISABLE_TIMING = 2,
    CU_EVENT_INTERPROCESS = 4,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUevent_record_flags_enum {
    CU_EVENT_RECORD_DEFAULT = 0,
    CU_EVENT_RECORD_EXTERNAL = 1,
}
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUevent_sched_flags_enum {
    CU_EVENT_SCHED_AUTO = 0,
    CU_EVENT_SCHED_SPIN = 1,
    CU_EVENT_SCHED_YIELD = 2,
    CU_EVENT_SCHED_BLOCKING_SYNC = 4,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUevent_wait_flags_enum {
    CU_EVENT_WAIT_DEFAULT = 0,
    CU_EVENT_WAIT_EXTERNAL = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUexecAffinityType_enum {
    CU_EXEC_AFFINITY_TYPE_SM_COUNT = 0,
    CU_EXEC_AFFINITY_TYPE_MAX = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUexternalMemoryHandleType_enum {
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD = 1,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32 = 2,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_HEAP = 4,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D12_RESOURCE = 5,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE = 6,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_D3D11_RESOURCE_KMT = 7,
    CU_EXTERNAL_MEMORY_HANDLE_TYPE_NVSCIBUF = 8,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUexternalSemaphoreHandleType_enum {
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD = 1,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32 = 2,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT = 3,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D12_FENCE = 4,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_FENCE = 5,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_NVSCISYNC = 6,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX = 7,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_D3D11_KEYED_MUTEX_KMT = 8,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD = 9,
    CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_WIN32 = 10,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfilter_mode_enum {
    CU_TR_FILTER_MODE_POINT = 0,
    CU_TR_FILTER_MODE_LINEAR = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUflushGPUDirectRDMAWritesOptions_enum {
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST = 1,
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUflushGPUDirectRDMAWritesScope_enum {
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER = 100,
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_ALL_DEVICES = 200,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUflushGPUDirectRDMAWritesTarget_enum {
    CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX = 0,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfunc_cache_enum {
    CU_FUNC_CACHE_PREFER_NONE = 0,
    CU_FUNC_CACHE_PREFER_SHARED = 1,
    CU_FUNC_CACHE_PREFER_L1 = 2,
    CU_FUNC_CACHE_PREFER_EQUAL = 3,
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfunctionLoadingState_enum {
    CU_FUNCTION_LOADING_STATE_UNLOADED = 0,
    CU_FUNCTION_LOADING_STATE_LOADED = 1,
    CU_FUNCTION_LOADING_STATE_MAX = 2,
}
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfunction_attribute_enum {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    CU_FUNC_ATTRIBUTE_MAX = 10,
}
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUfunction_attribute_enum {
    CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK = 0,
    CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES = 1,
    CU_FUNC_ATTRIBUTE_CONST_SIZE_BYTES = 2,
    CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES = 3,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4,
    CU_FUNC_ATTRIBUTE_PTX_VERSION = 5,
    CU_FUNC_ATTRIBUTE_BINARY_VERSION = 6,
    CU_FUNC_ATTRIBUTE_CACHE_MODE_CA = 7,
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 9,
    CU_FUNC_ATTRIBUTE_CLUSTER_SIZE_MUST_BE_SET = 10,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_WIDTH = 11,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_HEIGHT = 12,
    CU_FUNC_ATTRIBUTE_REQUIRED_CLUSTER_DEPTH = 13,
    CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED = 14,
    CU_FUNC_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 15,
    CU_FUNC_ATTRIBUTE_MAX = 16,
}
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphConditionalNodeType_enum {
    CU_GRAPH_COND_TYPE_IF = 0,
    CU_GRAPH_COND_TYPE_WHILE = 1,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphConditionalNodeType_enum {
    CU_GRAPH_COND_TYPE_IF = 0,
    CU_GRAPH_COND_TYPE_WHILE = 1,
    CU_GRAPH_COND_TYPE_SWITCH = 2,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphDebugDot_flags_enum {
    CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = 1,
    CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = 2,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = 4,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = 8,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = 16,
    CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = 32,
    CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = 64,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = 128,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = 256,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = 512,
    CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES = 1024,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS = 2048,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS = 4096,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphDebugDot_flags_enum {
    CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = 1,
    CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = 2,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = 4,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = 8,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = 16,
    CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = 32,
    CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = 64,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = 128,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = 256,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = 512,
    CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES = 1024,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS = 2048,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS = 4096,
    CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS = 8192,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphDebugDot_flags_enum {
    CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = 1,
    CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = 2,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = 4,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = 8,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = 16,
    CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = 32,
    CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = 64,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = 128,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = 256,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = 512,
    CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES = 1024,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS = 2048,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS = 4096,
    CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS = 8192,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO = 16384,
}
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphDebugDot_flags_enum {
    CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE = 1,
    CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = 2,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = 4,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = 8,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = 16,
    CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = 32,
    CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = 64,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = 128,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = 256,
    CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = 512,
    CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES = 1024,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_ALLOC_NODE_PARAMS = 2048,
    CU_GRAPH_DEBUG_DOT_FLAGS_MEM_FREE_NODE_PARAMS = 4096,
    CU_GRAPH_DEBUG_DOT_FLAGS_BATCH_MEM_OP_NODE_PARAMS = 8192,
    CU_GRAPH_DEBUG_DOT_FLAGS_EXTRA_TOPO_INFO = 16384,
    CU_GRAPH_DEBUG_DOT_FLAGS_CONDITIONAL_NODE_PARAMS = 32768,
}
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphDependencyType_enum {
    CU_GRAPH_DEPENDENCY_TYPE_DEFAULT = 0,
    CU_GRAPH_DEPENDENCY_TYPE_PROGRAMMATIC = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphExecUpdateResult_enum {
    CU_GRAPH_EXEC_UPDATE_SUCCESS = 0,
    CU_GRAPH_EXEC_UPDATE_ERROR = 1,
    CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 2,
    CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 3,
    CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 4,
    CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 5,
    CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 6,
    CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = 7,
}
#[cfg(
    any(
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
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphExecUpdateResult_enum {
    CU_GRAPH_EXEC_UPDATE_SUCCESS = 0,
    CU_GRAPH_EXEC_UPDATE_ERROR = 1,
    CU_GRAPH_EXEC_UPDATE_ERROR_TOPOLOGY_CHANGED = 2,
    CU_GRAPH_EXEC_UPDATE_ERROR_NODE_TYPE_CHANGED = 3,
    CU_GRAPH_EXEC_UPDATE_ERROR_FUNCTION_CHANGED = 4,
    CU_GRAPH_EXEC_UPDATE_ERROR_PARAMETERS_CHANGED = 5,
    CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED = 6,
    CU_GRAPH_EXEC_UPDATE_ERROR_UNSUPPORTED_FUNCTION_CHANGE = 7,
    CU_GRAPH_EXEC_UPDATE_ERROR_ATTRIBUTES_CHANGED = 8,
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphInstantiateResult_enum {
    CUDA_GRAPH_INSTANTIATE_SUCCESS = 0,
    CUDA_GRAPH_INSTANTIATE_ERROR = 1,
    CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE = 2,
    CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED = 3,
    CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED = 4,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphInstantiateResult_enum {
    CUDA_GRAPH_INSTANTIATE_SUCCESS = 0,
    CUDA_GRAPH_INSTANTIATE_ERROR = 1,
    CUDA_GRAPH_INSTANTIATE_INVALID_STRUCTURE = 2,
    CUDA_GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED = 3,
    CUDA_GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED = 4,
    CUDA_GRAPH_INSTANTIATE_CONDITIONAL_HANDLE_UNUSED = 5,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphInstantiate_flags_enum {
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphInstantiate_flags_enum {
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1,
    CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 8,
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphInstantiate_flags_enum {
    CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH = 1,
    CUDA_GRAPH_INSTANTIATE_FLAG_UPLOAD = 2,
    CUDA_GRAPH_INSTANTIATE_FLAG_DEVICE_LAUNCH = 4,
    CUDA_GRAPH_INSTANTIATE_FLAG_USE_NODE_PRIORITY = 8,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphMem_attribute_enum {
    CU_GRAPH_MEM_ATTR_USED_MEM_CURRENT = 0,
    CU_GRAPH_MEM_ATTR_USED_MEM_HIGH = 1,
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_CURRENT = 2,
    CU_GRAPH_MEM_ATTR_RESERVED_MEM_HIGH = 3,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphNodeType_enum {
    CU_GRAPH_NODE_TYPE_KERNEL = 0,
    CU_GRAPH_NODE_TYPE_MEMCPY = 1,
    CU_GRAPH_NODE_TYPE_MEMSET = 2,
    CU_GRAPH_NODE_TYPE_HOST = 3,
    CU_GRAPH_NODE_TYPE_GRAPH = 4,
    CU_GRAPH_NODE_TYPE_EMPTY = 5,
    CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6,
    CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9,
    CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10,
    CU_GRAPH_NODE_TYPE_MEM_FREE = 11,
}
#[cfg(
    any(
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphNodeType_enum {
    CU_GRAPH_NODE_TYPE_KERNEL = 0,
    CU_GRAPH_NODE_TYPE_MEMCPY = 1,
    CU_GRAPH_NODE_TYPE_MEMSET = 2,
    CU_GRAPH_NODE_TYPE_HOST = 3,
    CU_GRAPH_NODE_TYPE_GRAPH = 4,
    CU_GRAPH_NODE_TYPE_EMPTY = 5,
    CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6,
    CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9,
    CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10,
    CU_GRAPH_NODE_TYPE_MEM_FREE = 11,
    CU_GRAPH_NODE_TYPE_BATCH_MEM_OP = 12,
}
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphNodeType_enum {
    CU_GRAPH_NODE_TYPE_KERNEL = 0,
    CU_GRAPH_NODE_TYPE_MEMCPY = 1,
    CU_GRAPH_NODE_TYPE_MEMSET = 2,
    CU_GRAPH_NODE_TYPE_HOST = 3,
    CU_GRAPH_NODE_TYPE_GRAPH = 4,
    CU_GRAPH_NODE_TYPE_EMPTY = 5,
    CU_GRAPH_NODE_TYPE_WAIT_EVENT = 6,
    CU_GRAPH_NODE_TYPE_EVENT_RECORD = 7,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL = 8,
    CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT = 9,
    CU_GRAPH_NODE_TYPE_MEM_ALLOC = 10,
    CU_GRAPH_NODE_TYPE_MEM_FREE = 11,
    CU_GRAPH_NODE_TYPE_BATCH_MEM_OP = 12,
    CU_GRAPH_NODE_TYPE_CONDITIONAL = 13,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphicsMapResourceFlags_enum {
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_NONE = 0,
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_READ_ONLY = 1,
    CU_GRAPHICS_MAP_RESOURCE_FLAGS_WRITE_DISCARD = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgraphicsRegisterFlags_enum {
    CU_GRAPHICS_REGISTER_FLAGS_NONE = 0,
    CU_GRAPHICS_REGISTER_FLAGS_READ_ONLY = 1,
    CU_GRAPHICS_REGISTER_FLAGS_WRITE_DISCARD = 2,
    CU_GRAPHICS_REGISTER_FLAGS_SURFACE_LDST = 4,
    CU_GRAPHICS_REGISTER_FLAGS_TEXTURE_GATHER = 8,
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUgreenCtxCreate_flags {
    CU_GREEN_CTX_DEFAULT_STREAM = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUipcMem_flags_enum {
    CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjitInputType_enum {
    CU_JIT_INPUT_CUBIN = 0,
    CU_JIT_INPUT_PTX = 1,
    CU_JIT_INPUT_FATBINARY = 2,
    CU_JIT_INPUT_OBJECT = 3,
    CU_JIT_INPUT_LIBRARY = 4,
    CU_JIT_INPUT_NVVM = 5,
    CU_JIT_NUM_INPUT_TYPES = 6,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_cacheMode_enum {
    CU_JIT_CACHE_OPTION_NONE = 0,
    CU_JIT_CACHE_OPTION_CG = 1,
    CU_JIT_CACHE_OPTION_CA = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_fallback_enum {
    CU_PREFER_PTX = 0,
    CU_PREFER_BINARY = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_LTO = 20,
    CU_JIT_FTZ = 21,
    CU_JIT_PREC_DIV = 22,
    CU_JIT_PREC_SQRT = 23,
    CU_JIT_FMA = 24,
    CU_JIT_NUM_OPTIONS = 25,
}
#[cfg(any(feature = "cuda-11070", feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_LTO = 20,
    CU_JIT_FTZ = 21,
    CU_JIT_PREC_DIV = 22,
    CU_JIT_PREC_SQRT = 23,
    CU_JIT_FMA = 24,
    CU_JIT_REFERENCED_KERNEL_NAMES = 25,
    CU_JIT_REFERENCED_KERNEL_COUNT = 26,
    CU_JIT_REFERENCED_VARIABLE_NAMES = 27,
    CU_JIT_REFERENCED_VARIABLE_COUNT = 28,
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29,
    CU_JIT_NUM_OPTIONS = 30,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_LTO = 20,
    CU_JIT_FTZ = 21,
    CU_JIT_PREC_DIV = 22,
    CU_JIT_PREC_SQRT = 23,
    CU_JIT_FMA = 24,
    CU_JIT_REFERENCED_KERNEL_NAMES = 25,
    CU_JIT_REFERENCED_KERNEL_COUNT = 26,
    CU_JIT_REFERENCED_VARIABLE_NAMES = 27,
    CU_JIT_REFERENCED_VARIABLE_COUNT = 28,
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29,
    CU_JIT_POSITION_INDEPENDENT_CODE = 30,
    CU_JIT_NUM_OPTIONS = 31,
}
#[cfg(any(feature = "cuda-12030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_LTO = 20,
    CU_JIT_FTZ = 21,
    CU_JIT_PREC_DIV = 22,
    CU_JIT_PREC_SQRT = 23,
    CU_JIT_FMA = 24,
    CU_JIT_REFERENCED_KERNEL_NAMES = 25,
    CU_JIT_REFERENCED_KERNEL_COUNT = 26,
    CU_JIT_REFERENCED_VARIABLE_NAMES = 27,
    CU_JIT_REFERENCED_VARIABLE_COUNT = 28,
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29,
    CU_JIT_POSITION_INDEPENDENT_CODE = 30,
    CU_JIT_MIN_CTA_PER_SM = 31,
    CU_JIT_NUM_OPTIONS = 32,
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_option_enum {
    CU_JIT_MAX_REGISTERS = 0,
    CU_JIT_THREADS_PER_BLOCK = 1,
    CU_JIT_WALL_TIME = 2,
    CU_JIT_INFO_LOG_BUFFER = 3,
    CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES = 4,
    CU_JIT_ERROR_LOG_BUFFER = 5,
    CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES = 6,
    CU_JIT_OPTIMIZATION_LEVEL = 7,
    CU_JIT_TARGET_FROM_CUCONTEXT = 8,
    CU_JIT_TARGET = 9,
    CU_JIT_FALLBACK_STRATEGY = 10,
    CU_JIT_GENERATE_DEBUG_INFO = 11,
    CU_JIT_LOG_VERBOSE = 12,
    CU_JIT_GENERATE_LINE_INFO = 13,
    CU_JIT_CACHE_MODE = 14,
    CU_JIT_NEW_SM3X_OPT = 15,
    CU_JIT_FAST_COMPILE = 16,
    CU_JIT_GLOBAL_SYMBOL_NAMES = 17,
    CU_JIT_GLOBAL_SYMBOL_ADDRESSES = 18,
    CU_JIT_GLOBAL_SYMBOL_COUNT = 19,
    CU_JIT_LTO = 20,
    CU_JIT_FTZ = 21,
    CU_JIT_PREC_DIV = 22,
    CU_JIT_PREC_SQRT = 23,
    CU_JIT_FMA = 24,
    CU_JIT_REFERENCED_KERNEL_NAMES = 25,
    CU_JIT_REFERENCED_KERNEL_COUNT = 26,
    CU_JIT_REFERENCED_VARIABLE_NAMES = 27,
    CU_JIT_REFERENCED_VARIABLE_COUNT = 28,
    CU_JIT_OPTIMIZE_UNUSED_DEVICE_VARIABLES = 29,
    CU_JIT_POSITION_INDEPENDENT_CODE = 30,
    CU_JIT_MIN_CTA_PER_SM = 31,
    CU_JIT_MAX_THREADS_PER_BLOCK = 32,
    CU_JIT_OVERRIDE_DIRECTIVE_VALUES = 33,
    CU_JIT_NUM_OPTIONS = 34,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_target_enum {
    CU_TARGET_COMPUTE_20 = 20,
    CU_TARGET_COMPUTE_21 = 21,
    CU_TARGET_COMPUTE_30 = 30,
    CU_TARGET_COMPUTE_32 = 32,
    CU_TARGET_COMPUTE_35 = 35,
    CU_TARGET_COMPUTE_37 = 37,
    CU_TARGET_COMPUTE_50 = 50,
    CU_TARGET_COMPUTE_52 = 52,
    CU_TARGET_COMPUTE_53 = 53,
    CU_TARGET_COMPUTE_60 = 60,
    CU_TARGET_COMPUTE_61 = 61,
    CU_TARGET_COMPUTE_62 = 62,
    CU_TARGET_COMPUTE_70 = 70,
    CU_TARGET_COMPUTE_72 = 72,
    CU_TARGET_COMPUTE_75 = 75,
    CU_TARGET_COMPUTE_80 = 80,
    CU_TARGET_COMPUTE_86 = 86,
}
#[cfg(any(feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_target_enum {
    CU_TARGET_COMPUTE_20 = 20,
    CU_TARGET_COMPUTE_21 = 21,
    CU_TARGET_COMPUTE_30 = 30,
    CU_TARGET_COMPUTE_32 = 32,
    CU_TARGET_COMPUTE_35 = 35,
    CU_TARGET_COMPUTE_37 = 37,
    CU_TARGET_COMPUTE_50 = 50,
    CU_TARGET_COMPUTE_52 = 52,
    CU_TARGET_COMPUTE_53 = 53,
    CU_TARGET_COMPUTE_60 = 60,
    CU_TARGET_COMPUTE_61 = 61,
    CU_TARGET_COMPUTE_62 = 62,
    CU_TARGET_COMPUTE_70 = 70,
    CU_TARGET_COMPUTE_72 = 72,
    CU_TARGET_COMPUTE_75 = 75,
    CU_TARGET_COMPUTE_80 = 80,
    CU_TARGET_COMPUTE_86 = 86,
    CU_TARGET_COMPUTE_87 = 87,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_target_enum {
    CU_TARGET_COMPUTE_20 = 20,
    CU_TARGET_COMPUTE_21 = 21,
    CU_TARGET_COMPUTE_30 = 30,
    CU_TARGET_COMPUTE_32 = 32,
    CU_TARGET_COMPUTE_35 = 35,
    CU_TARGET_COMPUTE_37 = 37,
    CU_TARGET_COMPUTE_50 = 50,
    CU_TARGET_COMPUTE_52 = 52,
    CU_TARGET_COMPUTE_53 = 53,
    CU_TARGET_COMPUTE_60 = 60,
    CU_TARGET_COMPUTE_61 = 61,
    CU_TARGET_COMPUTE_62 = 62,
    CU_TARGET_COMPUTE_70 = 70,
    CU_TARGET_COMPUTE_72 = 72,
    CU_TARGET_COMPUTE_75 = 75,
    CU_TARGET_COMPUTE_80 = 80,
    CU_TARGET_COMPUTE_86 = 86,
    CU_TARGET_COMPUTE_87 = 87,
    CU_TARGET_COMPUTE_89 = 89,
    CU_TARGET_COMPUTE_90 = 90,
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_target_enum {
    CU_TARGET_COMPUTE_30 = 30,
    CU_TARGET_COMPUTE_32 = 32,
    CU_TARGET_COMPUTE_35 = 35,
    CU_TARGET_COMPUTE_37 = 37,
    CU_TARGET_COMPUTE_50 = 50,
    CU_TARGET_COMPUTE_52 = 52,
    CU_TARGET_COMPUTE_53 = 53,
    CU_TARGET_COMPUTE_60 = 60,
    CU_TARGET_COMPUTE_61 = 61,
    CU_TARGET_COMPUTE_62 = 62,
    CU_TARGET_COMPUTE_70 = 70,
    CU_TARGET_COMPUTE_72 = 72,
    CU_TARGET_COMPUTE_75 = 75,
    CU_TARGET_COMPUTE_80 = 80,
    CU_TARGET_COMPUTE_86 = 86,
    CU_TARGET_COMPUTE_87 = 87,
    CU_TARGET_COMPUTE_89 = 89,
    CU_TARGET_COMPUTE_90 = 90,
    CU_TARGET_COMPUTE_90A = 65626,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUjit_target_enum {
    CU_TARGET_COMPUTE_30 = 30,
    CU_TARGET_COMPUTE_32 = 32,
    CU_TARGET_COMPUTE_35 = 35,
    CU_TARGET_COMPUTE_37 = 37,
    CU_TARGET_COMPUTE_50 = 50,
    CU_TARGET_COMPUTE_52 = 52,
    CU_TARGET_COMPUTE_53 = 53,
    CU_TARGET_COMPUTE_60 = 60,
    CU_TARGET_COMPUTE_61 = 61,
    CU_TARGET_COMPUTE_62 = 62,
    CU_TARGET_COMPUTE_70 = 70,
    CU_TARGET_COMPUTE_72 = 72,
    CU_TARGET_COMPUTE_75 = 75,
    CU_TARGET_COMPUTE_80 = 80,
    CU_TARGET_COMPUTE_86 = 86,
    CU_TARGET_COMPUTE_87 = 87,
    CU_TARGET_COMPUTE_89 = 89,
    CU_TARGET_COMPUTE_90 = 90,
    CU_TARGET_COMPUTE_100 = 100,
    CU_TARGET_COMPUTE_101 = 101,
    CU_TARGET_COMPUTE_120 = 120,
    CU_TARGET_COMPUTE_90A = 65626,
    CU_TARGET_COMPUTE_100A = 65636,
    CU_TARGET_COMPUTE_101A = 65637,
    CU_TARGET_COMPUTE_120A = 65656,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUkernelNodeAttrID_enum {
    CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = 2,
}
#[cfg(any(feature = "cuda-11070"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUkernelNodeAttrID_enum {
    CU_KERNEL_NODE_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_KERNEL_NODE_ATTRIBUTE_COOPERATIVE = 2,
    CU_KERNEL_NODE_ATTRIBUTE_PRIORITY = 8,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10,
}
#[cfg(any(feature = "cuda-12030"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10,
    CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT = 12,
}
#[cfg(any(feature = "cuda-12040"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10,
    CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT = 12,
    CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE = 13,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10,
    CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT = 12,
    CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE = 13,
    CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 14,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchAttributeID_enum {
    CU_LAUNCH_ATTRIBUTE_IGNORE = 0,
    CU_LAUNCH_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_LAUNCH_ATTRIBUTE_COOPERATIVE = 2,
    CU_LAUNCH_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION = 4,
    CU_LAUNCH_ATTRIBUTE_CLUSTER_SCHEDULING_POLICY_PREFERENCE = 5,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION = 6,
    CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_EVENT = 7,
    CU_LAUNCH_ATTRIBUTE_PRIORITY = 8,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN_MAP = 9,
    CU_LAUNCH_ATTRIBUTE_MEM_SYNC_DOMAIN = 10,
    CU_LAUNCH_ATTRIBUTE_PREFERRED_CLUSTER_DIMENSION = 11,
    CU_LAUNCH_ATTRIBUTE_LAUNCH_COMPLETION_EVENT = 12,
    CU_LAUNCH_ATTRIBUTE_DEVICE_UPDATABLE_KERNEL_NODE = 13,
    CU_LAUNCH_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT = 14,
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlaunchMemSyncDomain_enum {
    CU_LAUNCH_MEM_SYNC_DOMAIN_DEFAULT = 0,
    CU_LAUNCH_MEM_SYNC_DOMAIN_REMOTE = 1,
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlibraryOption_enum {
    CU_LIBRARY_HOST_UNIVERSAL_FUNCTION_AND_DATA_TABLE = 0,
    CU_LIBRARY_BINARY_IS_PRESERVED = 1,
    CU_LIBRARY_NUM_OPTIONS = 2,
}
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlimit_enum {
    CU_LIMIT_STACK_SIZE = 0,
    CU_LIMIT_PRINTF_FIFO_SIZE = 1,
    CU_LIMIT_MALLOC_HEAP_SIZE = 2,
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3,
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4,
    CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 5,
    CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 6,
    CU_LIMIT_MAX = 7,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUlimit_enum {
    CU_LIMIT_STACK_SIZE = 0,
    CU_LIMIT_PRINTF_FIFO_SIZE = 1,
    CU_LIMIT_MALLOC_HEAP_SIZE = 2,
    CU_LIMIT_DEV_RUNTIME_SYNC_DEPTH = 3,
    CU_LIMIT_DEV_RUNTIME_PENDING_LAUNCH_COUNT = 4,
    CU_LIMIT_MAX_L2_FETCH_GRANULARITY = 5,
    CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 6,
    CU_LIMIT_SHMEM_SIZE = 7,
    CU_LIMIT_CIG_ENABLED = 8,
    CU_LIMIT_CIG_SHMEM_FALLBACK_ENABLED = 9,
    CU_LIMIT_MAX = 10,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemAccess_flags_enum {
    CU_MEM_ACCESS_FLAGS_PROT_NONE = 0,
    CU_MEM_ACCESS_FLAGS_PROT_READ = 1,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3,
    CU_MEM_ACCESS_FLAGS_PROT_MAX = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemAllocationCompType_enum {
    CU_MEM_ALLOCATION_COMP_NONE = 0,
    CU_MEM_ALLOCATION_COMP_GENERIC = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemAllocationGranularity_flags_enum {
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0,
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 1,
}
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemAllocationHandleType_enum {
    CU_MEM_HANDLE_TYPE_NONE = 0,
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1,
    CU_MEM_HANDLE_TYPE_WIN32 = 2,
    CU_MEM_HANDLE_TYPE_WIN32_KMT = 4,
    CU_MEM_HANDLE_TYPE_MAX = 2147483647,
}
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemAllocationHandleType_enum {
    CU_MEM_HANDLE_TYPE_NONE = 0,
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 1,
    CU_MEM_HANDLE_TYPE_WIN32 = 2,
    CU_MEM_HANDLE_TYPE_WIN32_KMT = 4,
    CU_MEM_HANDLE_TYPE_FABRIC = 8,
    CU_MEM_HANDLE_TYPE_MAX = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemAllocationType_enum {
    CU_MEM_ALLOCATION_TYPE_INVALID = 0,
    CU_MEM_ALLOCATION_TYPE_PINNED = 1,
    CU_MEM_ALLOCATION_TYPE_MAX = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemAttach_flags_enum {
    CU_MEM_ATTACH_GLOBAL = 1,
    CU_MEM_ATTACH_HOST = 2,
    CU_MEM_ATTACH_SINGLE = 4,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemDecompressAlgorithm_enum {
    CU_MEM_DECOMPRESS_UNSUPPORTED = 0,
    CU_MEM_DECOMPRESS_ALGORITHM_DEFLATE = 1,
    CU_MEM_DECOMPRESS_ALGORITHM_SNAPPY = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemHandleType_enum {
    CU_MEM_HANDLE_TYPE_GENERIC = 0,
}
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemLocationType_enum {
    CU_MEM_LOCATION_TYPE_INVALID = 0,
    CU_MEM_LOCATION_TYPE_DEVICE = 1,
    CU_MEM_LOCATION_TYPE_MAX = 2147483647,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemLocationType_enum {
    CU_MEM_LOCATION_TYPE_INVALID = 0,
    CU_MEM_LOCATION_TYPE_DEVICE = 1,
    CU_MEM_LOCATION_TYPE_HOST = 2,
    CU_MEM_LOCATION_TYPE_HOST_NUMA = 3,
    CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT = 4,
    CU_MEM_LOCATION_TYPE_MAX = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemOperationType_enum {
    CU_MEM_OPERATION_TYPE_MAP = 1,
    CU_MEM_OPERATION_TYPE_UNMAP = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemPool_attribute_enum {
    CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES = 1,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC = 2,
    CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES = 3,
    CU_MEMPOOL_ATTR_RELEASE_THRESHOLD = 4,
    CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT = 5,
    CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH = 6,
    CU_MEMPOOL_ATTR_USED_MEM_CURRENT = 7,
    CU_MEMPOOL_ATTR_USED_MEM_HIGH = 8,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemRangeFlags_enum {
    CU_MEM_RANGE_FLAG_DMA_BUF_MAPPING_TYPE_PCIE = 1,
}
#[cfg(
    any(
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
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemRangeHandleType_enum {
    CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD = 1,
    CU_MEM_RANGE_HANDLE_TYPE_MAX = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmem_advise_enum {
    CU_MEM_ADVISE_SET_READ_MOSTLY = 1,
    CU_MEM_ADVISE_UNSET_READ_MOSTLY = 2,
    CU_MEM_ADVISE_SET_PREFERRED_LOCATION = 3,
    CU_MEM_ADVISE_UNSET_PREFERRED_LOCATION = 4,
    CU_MEM_ADVISE_SET_ACCESSED_BY = 5,
    CU_MEM_ADVISE_UNSET_ACCESSED_BY = 6,
}
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmem_range_attribute_enum {
    CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1,
    CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2,
    CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3,
    CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmem_range_attribute_enum {
    CU_MEM_RANGE_ATTRIBUTE_READ_MOSTLY = 1,
    CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION = 2,
    CU_MEM_RANGE_ATTRIBUTE_ACCESSED_BY = 3,
    CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION = 4,
    CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_TYPE = 5,
    CU_MEM_RANGE_ATTRIBUTE_PREFERRED_LOCATION_ID = 6,
    CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_TYPE = 7,
    CU_MEM_RANGE_ATTRIBUTE_LAST_PREFETCH_LOCATION_ID = 8,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemcpy3DOperandType_enum {
    CU_MEMCPY_OPERAND_TYPE_POINTER = 1,
    CU_MEMCPY_OPERAND_TYPE_ARRAY = 2,
    CU_MEMCPY_OPERAND_TYPE_MAX = 2147483647,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemcpyFlags_enum {
    CU_MEMCPY_FLAG_DEFAULT = 0,
    CU_MEMCPY_FLAG_PREFER_OVERLAP_WITH_COMPUTE = 1,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemcpySrcAccessOrder_enum {
    CU_MEMCPY_SRC_ACCESS_ORDER_INVALID = 0,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM = 1,
    CU_MEMCPY_SRC_ACCESS_ORDER_DURING_API_CALL = 2,
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY = 3,
    CU_MEMCPY_SRC_ACCESS_ORDER_MAX = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmemorytype_enum {
    CU_MEMORYTYPE_HOST = 1,
    CU_MEMORYTYPE_DEVICE = 2,
    CU_MEMORYTYPE_ARRAY = 3,
    CU_MEMORYTYPE_UNIFIED = 4,
}
#[cfg(
    any(
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
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmoduleLoadingMode_enum {
    CU_MODULE_EAGER_LOADING = 1,
    CU_MODULE_LAZY_LOADING = 2,
}
#[cfg(
    any(
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUmulticastGranularity_flags_enum {
    CU_MULTICAST_GRANULARITY_MINIMUM = 0,
    CU_MULTICAST_GRANULARITY_RECOMMENDED = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUoccupancy_flags_enum {
    CU_OCCUPANCY_DEFAULT = 0,
    CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUoutput_mode_enum {
    CU_OUT_KEY_VALUE_PAIR = 0,
    CU_OUT_CSV = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
    CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
    CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
    CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
    CU_POINTER_ATTRIBUTE_MAPPED = 13,
    CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
    CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
    CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17,
}
#[cfg(
    any(
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
    CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
    CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
    CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
    CU_POINTER_ATTRIBUTE_MAPPED = 13,
    CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
    CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
    CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17,
    CU_POINTER_ATTRIBUTE_MAPPING_SIZE = 18,
    CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR = 19,
    CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = 20,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUpointer_attribute_enum {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2,
    CU_POINTER_ATTRIBUTE_DEVICE_POINTER = 3,
    CU_POINTER_ATTRIBUTE_HOST_POINTER = 4,
    CU_POINTER_ATTRIBUTE_P2P_TOKENS = 5,
    CU_POINTER_ATTRIBUTE_SYNC_MEMOPS = 6,
    CU_POINTER_ATTRIBUTE_BUFFER_ID = 7,
    CU_POINTER_ATTRIBUTE_IS_MANAGED = 8,
    CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL = 9,
    CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE = 10,
    CU_POINTER_ATTRIBUTE_RANGE_START_ADDR = 11,
    CU_POINTER_ATTRIBUTE_RANGE_SIZE = 12,
    CU_POINTER_ATTRIBUTE_MAPPED = 13,
    CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES = 14,
    CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE = 15,
    CU_POINTER_ATTRIBUTE_ACCESS_FLAGS = 16,
    CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE = 17,
    CU_POINTER_ATTRIBUTE_MAPPING_SIZE = 18,
    CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR = 19,
    CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID = 20,
    CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE = 21,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUprocessState_enum {
    CU_PROCESS_STATE_RUNNING = 0,
    CU_PROCESS_STATE_LOCKED = 1,
    CU_PROCESS_STATE_CHECKPOINTED = 2,
    CU_PROCESS_STATE_FAILED = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUresourceViewFormat_enum {
    CU_RES_VIEW_FORMAT_NONE = 0,
    CU_RES_VIEW_FORMAT_UINT_1X8 = 1,
    CU_RES_VIEW_FORMAT_UINT_2X8 = 2,
    CU_RES_VIEW_FORMAT_UINT_4X8 = 3,
    CU_RES_VIEW_FORMAT_SINT_1X8 = 4,
    CU_RES_VIEW_FORMAT_SINT_2X8 = 5,
    CU_RES_VIEW_FORMAT_SINT_4X8 = 6,
    CU_RES_VIEW_FORMAT_UINT_1X16 = 7,
    CU_RES_VIEW_FORMAT_UINT_2X16 = 8,
    CU_RES_VIEW_FORMAT_UINT_4X16 = 9,
    CU_RES_VIEW_FORMAT_SINT_1X16 = 10,
    CU_RES_VIEW_FORMAT_SINT_2X16 = 11,
    CU_RES_VIEW_FORMAT_SINT_4X16 = 12,
    CU_RES_VIEW_FORMAT_UINT_1X32 = 13,
    CU_RES_VIEW_FORMAT_UINT_2X32 = 14,
    CU_RES_VIEW_FORMAT_UINT_4X32 = 15,
    CU_RES_VIEW_FORMAT_SINT_1X32 = 16,
    CU_RES_VIEW_FORMAT_SINT_2X32 = 17,
    CU_RES_VIEW_FORMAT_SINT_4X32 = 18,
    CU_RES_VIEW_FORMAT_FLOAT_1X16 = 19,
    CU_RES_VIEW_FORMAT_FLOAT_2X16 = 20,
    CU_RES_VIEW_FORMAT_FLOAT_4X16 = 21,
    CU_RES_VIEW_FORMAT_FLOAT_1X32 = 22,
    CU_RES_VIEW_FORMAT_FLOAT_2X32 = 23,
    CU_RES_VIEW_FORMAT_FLOAT_4X32 = 24,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC1 = 25,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC2 = 26,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC3 = 27,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC4 = 28,
    CU_RES_VIEW_FORMAT_SIGNED_BC4 = 29,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC5 = 30,
    CU_RES_VIEW_FORMAT_SIGNED_BC5 = 31,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC6H = 32,
    CU_RES_VIEW_FORMAT_SIGNED_BC6H = 33,
    CU_RES_VIEW_FORMAT_UNSIGNED_BC7 = 34,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUresourcetype_enum {
    CU_RESOURCE_TYPE_ARRAY = 0,
    CU_RESOURCE_TYPE_MIPMAPPED_ARRAY = 1,
    CU_RESOURCE_TYPE_LINEAR = 2,
    CU_RESOURCE_TYPE_PITCH2D = 3,
}
#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUshared_carveout_enum {
    CU_SHAREDMEM_CARVEOUT_DEFAULT = -1,
    CU_SHAREDMEM_CARVEOUT_MAX_SHARED = 100,
    CU_SHAREDMEM_CARVEOUT_MAX_L1 = 0,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUsharedconfig_enum {
    CU_SHARED_MEM_CONFIG_DEFAULT_BANK_SIZE = 0,
    CU_SHARED_MEM_CONFIG_FOUR_BYTE_BANK_SIZE = 1,
    CU_SHARED_MEM_CONFIG_EIGHT_BYTE_BANK_SIZE = 2,
}
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamAttrID_enum {
    CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW = 1,
    CU_STREAM_ATTRIBUTE_SYNCHRONIZATION_POLICY = 3,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamBatchMemOpType_enum {
    CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1,
    CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,
    CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4,
    CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5,
    CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3,
}
#[cfg(
    any(
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
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamBatchMemOpType_enum {
    CU_STREAM_MEM_OP_WAIT_VALUE_32 = 1,
    CU_STREAM_MEM_OP_WRITE_VALUE_32 = 2,
    CU_STREAM_MEM_OP_WAIT_VALUE_64 = 4,
    CU_STREAM_MEM_OP_WRITE_VALUE_64 = 5,
    CU_STREAM_MEM_OP_BARRIER = 6,
    CU_STREAM_MEM_OP_FLUSH_REMOTE_WRITES = 3,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamCaptureMode_enum {
    CU_STREAM_CAPTURE_MODE_GLOBAL = 0,
    CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1,
    CU_STREAM_CAPTURE_MODE_RELAXED = 2,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamCaptureStatus_enum {
    CU_STREAM_CAPTURE_STATUS_NONE = 0,
    CU_STREAM_CAPTURE_STATUS_ACTIVE = 1,
    CU_STREAM_CAPTURE_STATUS_INVALIDATED = 2,
}
#[cfg(
    any(
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
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamMemoryBarrier_flags_enum {
    CU_STREAM_MEMORY_BARRIER_TYPE_SYS = 0,
    CU_STREAM_MEMORY_BARRIER_TYPE_GPU = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamUpdateCaptureDependencies_flags_enum {
    CU_STREAM_ADD_CAPTURE_DEPENDENCIES = 0,
    CU_STREAM_SET_CAPTURE_DEPENDENCIES = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamWaitValue_flags_enum {
    CU_STREAM_WAIT_VALUE_GEQ = 0,
    CU_STREAM_WAIT_VALUE_EQ = 1,
    CU_STREAM_WAIT_VALUE_AND = 2,
    CU_STREAM_WAIT_VALUE_NOR = 3,
    CU_STREAM_WAIT_VALUE_FLUSH = 1073741824,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstreamWriteValue_flags_enum {
    CU_STREAM_WRITE_VALUE_DEFAULT = 0,
    CU_STREAM_WRITE_VALUE_NO_MEMORY_BARRIER = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUstream_flags_enum {
    CU_STREAM_DEFAULT = 0,
    CU_STREAM_NON_BLOCKING = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUsynchronizationPolicy_enum {
    CU_SYNC_POLICY_AUTO = 1,
    CU_SYNC_POLICY_SPIN = 2,
    CU_SYNC_POLICY_YIELD = 3,
    CU_SYNC_POLICY_BLOCKING_SYNC = 4,
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapDataType_enum {
    CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
    CU_TENSOR_MAP_DATA_TYPE_UINT16 = 1,
    CU_TENSOR_MAP_DATA_TYPE_UINT32 = 2,
    CU_TENSOR_MAP_DATA_TYPE_INT32 = 3,
    CU_TENSOR_MAP_DATA_TYPE_UINT64 = 4,
    CU_TENSOR_MAP_DATA_TYPE_INT64 = 5,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16 = 6,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT64 = 8,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 9,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ = 10,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 = 11,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ = 12,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapDataType_enum {
    CU_TENSOR_MAP_DATA_TYPE_UINT8 = 0,
    CU_TENSOR_MAP_DATA_TYPE_UINT16 = 1,
    CU_TENSOR_MAP_DATA_TYPE_UINT32 = 2,
    CU_TENSOR_MAP_DATA_TYPE_INT32 = 3,
    CU_TENSOR_MAP_DATA_TYPE_UINT64 = 4,
    CU_TENSOR_MAP_DATA_TYPE_INT64 = 5,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT16 = 6,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32 = 7,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT64 = 8,
    CU_TENSOR_MAP_DATA_TYPE_BFLOAT16 = 9,
    CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ = 10,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32 = 11,
    CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ = 12,
    CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B = 13,
    CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B = 14,
    CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B = 15,
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapFloatOOBfill_enum {
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE = 0,
    CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA = 1,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapIm2ColWideMode_enum {
    CU_TENSOR_MAP_IM2COL_WIDE_MODE_W = 0,
    CU_TENSOR_MAP_IM2COL_WIDE_MODE_W128 = 1,
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapInterleave_enum {
    CU_TENSOR_MAP_INTERLEAVE_NONE = 0,
    CU_TENSOR_MAP_INTERLEAVE_16B = 1,
    CU_TENSOR_MAP_INTERLEAVE_32B = 2,
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapL2promotion_enum {
    CU_TENSOR_MAP_L2_PROMOTION_NONE = 0,
    CU_TENSOR_MAP_L2_PROMOTION_L2_64B = 1,
    CU_TENSOR_MAP_L2_PROMOTION_L2_128B = 2,
    CU_TENSOR_MAP_L2_PROMOTION_L2_256B = 3,
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060"
    )
)]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapSwizzle_enum {
    CU_TENSOR_MAP_SWIZZLE_NONE = 0,
    CU_TENSOR_MAP_SWIZZLE_32B = 1,
    CU_TENSOR_MAP_SWIZZLE_64B = 2,
    CU_TENSOR_MAP_SWIZZLE_128B = 3,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUtensorMapSwizzle_enum {
    CU_TENSOR_MAP_SWIZZLE_NONE = 0,
    CU_TENSOR_MAP_SWIZZLE_32B = 1,
    CU_TENSOR_MAP_SWIZZLE_64B = 2,
    CU_TENSOR_MAP_SWIZZLE_128B = 3,
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B = 4,
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B = 5,
    CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B = 6,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUuserObjectRetain_flags_enum {
    CU_GRAPH_USER_OBJECT_MOVE = 1,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum CUuserObject_flags_enum {
    CU_USER_OBJECT_NO_DESTRUCTOR_SYNC = 1,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
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
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
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
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-11070"))]
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
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
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
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-11080"))]
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
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
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
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-12000"))]
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
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
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
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-12010", feature = "cuda-12020"))]
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
    CUDA_ERROR_INVALID_SOURCE = 300,
    CUDA_ERROR_FILE_NOT_FOUND = 301,
    CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND = 302,
    CUDA_ERROR_SHARED_OBJECT_INIT_FAILED = 303,
    CUDA_ERROR_OPERATING_SYSTEM = 304,
    CUDA_ERROR_INVALID_HANDLE = 400,
    CUDA_ERROR_ILLEGAL_STATE = 401,
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
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-12030"))]
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
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
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
    CUDA_ERROR_UNKNOWN = 999,
}
#[cfg(any(feature = "cuda-12080"))]
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
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_ARRAY3D_DESCRIPTOR_st {
    pub Width: usize,
    pub Height: usize,
    pub Depth: usize,
    pub Format: CUarray_format,
    pub NumChannels: ::core::ffi::c_uint,
    pub Flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_ARRAY_DESCRIPTOR_st {
    pub Width: usize,
    pub Height: usize,
    pub Format: CUarray_format,
    pub NumChannels: ::core::ffi::c_uint,
}
#[cfg(
    any(
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
    )
)]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_ARRAY_MEMORY_REQUIREMENTS_st {
    pub size: usize,
    pub alignment: usize,
    pub reserved: [::core::ffi::c_uint; 4usize],
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_ARRAY_SPARSE_PROPERTIES_st {
    pub tileExtent: CUDA_ARRAY_SPARSE_PROPERTIES_st__bindgen_ty_1,
    pub miptailFirstLevel: ::core::ffi::c_uint,
    pub miptailSize: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 4usize],
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_ARRAY_SPARSE_PROPERTIES_st__bindgen_ty_1 {
    pub width: ::core::ffi::c_uint,
    pub height: ::core::ffi::c_uint,
    pub depth: ::core::ffi::c_uint,
}
#[cfg(
    any(
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_BATCH_MEM_OP_NODE_PARAMS_st {
    pub ctx: CUcontext,
    pub count: ::core::ffi::c_uint,
    pub paramArray: *mut CUstreamBatchMemOpParams,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st {
    pub ctx: CUcontext,
    pub count: ::core::ffi::c_uint,
    pub paramArray: *mut CUstreamBatchMemOpParams,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st {
    pub ctx: CUcontext,
    pub count: ::core::ffi::c_uint,
    pub paramArray: *mut CUstreamBatchMemOpParams,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_CHILD_GRAPH_NODE_PARAMS_st {
    pub graph: CUgraph,
}
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_CONDITIONAL_NODE_PARAMS {
    pub handle: CUgraphConditionalHandle,
    pub type_: CUgraphConditionalNodeType,
    pub size: ::core::ffi::c_uint,
    pub phGraph_out: *mut CUgraph,
    pub ctx: CUcontext,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EVENT_RECORD_NODE_PARAMS_st {
    pub event: CUevent,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EVENT_WAIT_NODE_PARAMS_st {
    pub event: CUevent,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_MEMORY_BUFFER_DESC_st {
    pub offset: ::core::ffi::c_ulonglong,
    pub size: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st {
    pub type_: CUexternalMemoryHandleType,
    pub handle: CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1,
    pub size: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
    pub handle: *mut ::core::ffi::c_void,
    pub name: *const ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st {
    pub offset: ::core::ffi::c_ulonglong,
    pub arrayDesc: CUDA_ARRAY3D_DESCRIPTOR,
    pub numLevels: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st {
    pub type_: CUexternalSemaphoreHandleType,
    pub handle: CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
    pub handle: *mut ::core::ffi::c_void,
    pub name: *const ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st {
    pub params: CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1 {
    pub fence: CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSync: CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_2,
    pub keyedMutex: CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_3,
    pub reserved: [::core::ffi::c_uint; 12usize],
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_1 {
    pub value: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_3 {
    pub key: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st {
    pub params: CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1 {
    pub fence: CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSync: CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_2,
    pub keyedMutex: CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_3,
    pub reserved: [::core::ffi::c_uint; 10usize],
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_1 {
    pub value: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_3 {
    pub key: ::core::ffi::c_ulonglong,
    pub timeoutMs: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st {
    pub extSemArray: *mut CUexternalSemaphore,
    pub paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
    pub numExtSems: ::core::ffi::c_uint,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st {
    pub extSemArray: *mut CUexternalSemaphore,
    pub paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
    pub numExtSems: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXT_SEM_WAIT_NODE_PARAMS_st {
    pub extSemArray: *mut CUexternalSemaphore,
    pub paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
    pub numExtSems: ::core::ffi::c_uint,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st {
    pub extSemArray: *mut CUexternalSemaphore,
    pub paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
    pub numExtSems: ::core::ffi::c_uint,
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_GRAPH_INSTANTIATE_PARAMS_st {
    pub flags: cuuint64_t,
    pub hUploadStream: CUstream,
    pub hErrNode_out: CUgraphNode,
    pub result_out: CUgraphInstantiateResult,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_HOST_NODE_PARAMS_st {
    pub fn_: CUhostFn,
    pub userData: *mut ::core::ffi::c_void,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_HOST_NODE_PARAMS_v2_st {
    pub fn_: CUhostFn,
    pub userData: *mut ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_KERNEL_NODE_PARAMS_st {
    pub func: CUfunction,
    pub gridDimX: ::core::ffi::c_uint,
    pub gridDimY: ::core::ffi::c_uint,
    pub gridDimZ: ::core::ffi::c_uint,
    pub blockDimX: ::core::ffi::c_uint,
    pub blockDimY: ::core::ffi::c_uint,
    pub blockDimZ: ::core::ffi::c_uint,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub kernelParams: *mut *mut ::core::ffi::c_void,
    pub extra: *mut *mut ::core::ffi::c_void,
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_KERNEL_NODE_PARAMS_v2_st {
    pub func: CUfunction,
    pub gridDimX: ::core::ffi::c_uint,
    pub gridDimY: ::core::ffi::c_uint,
    pub gridDimZ: ::core::ffi::c_uint,
    pub blockDimX: ::core::ffi::c_uint,
    pub blockDimY: ::core::ffi::c_uint,
    pub blockDimZ: ::core::ffi::c_uint,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub kernelParams: *mut *mut ::core::ffi::c_void,
    pub extra: *mut *mut ::core::ffi::c_void,
    pub kern: CUkernel,
    pub ctx: CUcontext,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_KERNEL_NODE_PARAMS_v3_st {
    pub func: CUfunction,
    pub gridDimX: ::core::ffi::c_uint,
    pub gridDimY: ::core::ffi::c_uint,
    pub gridDimZ: ::core::ffi::c_uint,
    pub blockDimX: ::core::ffi::c_uint,
    pub blockDimY: ::core::ffi::c_uint,
    pub blockDimZ: ::core::ffi::c_uint,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub kernelParams: *mut *mut ::core::ffi::c_void,
    pub extra: *mut *mut ::core::ffi::c_void,
    pub kern: CUkernel,
    pub ctx: CUcontext,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_LAUNCH_PARAMS_st {
    pub function: CUfunction,
    pub gridDimX: ::core::ffi::c_uint,
    pub gridDimY: ::core::ffi::c_uint,
    pub gridDimZ: ::core::ffi::c_uint,
    pub blockDimX: ::core::ffi::c_uint,
    pub blockDimY: ::core::ffi::c_uint,
    pub blockDimZ: ::core::ffi::c_uint,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub hStream: CUstream,
    pub kernelParams: *mut *mut ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEMCPY2D_st {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcMemoryType: CUmemorytype,
    pub srcHost: *const ::core::ffi::c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub srcPitch: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstMemoryType: CUmemorytype,
    pub dstHost: *mut ::core::ffi::c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub dstPitch: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_MEMCPY3D_BATCH_OP_st {
    pub src: CUmemcpy3DOperand,
    pub dst: CUmemcpy3DOperand,
    pub extent: CUextent3D,
    pub srcAccessOrder: CUmemcpySrcAccessOrder,
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEMCPY3D_PEER_st {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcZ: usize,
    pub srcLOD: usize,
    pub srcMemoryType: CUmemorytype,
    pub srcHost: *const ::core::ffi::c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub srcContext: CUcontext,
    pub srcPitch: usize,
    pub srcHeight: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstZ: usize,
    pub dstLOD: usize,
    pub dstMemoryType: CUmemorytype,
    pub dstHost: *mut ::core::ffi::c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub dstContext: CUcontext,
    pub dstPitch: usize,
    pub dstHeight: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
    pub Depth: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEMCPY3D_st {
    pub srcXInBytes: usize,
    pub srcY: usize,
    pub srcZ: usize,
    pub srcLOD: usize,
    pub srcMemoryType: CUmemorytype,
    pub srcHost: *const ::core::ffi::c_void,
    pub srcDevice: CUdeviceptr,
    pub srcArray: CUarray,
    pub reserved0: *mut ::core::ffi::c_void,
    pub srcPitch: usize,
    pub srcHeight: usize,
    pub dstXInBytes: usize,
    pub dstY: usize,
    pub dstZ: usize,
    pub dstLOD: usize,
    pub dstMemoryType: CUmemorytype,
    pub dstHost: *mut ::core::ffi::c_void,
    pub dstDevice: CUdeviceptr,
    pub dstArray: CUarray,
    pub reserved1: *mut ::core::ffi::c_void,
    pub dstPitch: usize,
    pub dstHeight: usize,
    pub WidthInBytes: usize,
    pub Height: usize,
    pub Depth: usize,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEMCPY_NODE_PARAMS_st {
    pub flags: ::core::ffi::c_int,
    pub reserved: ::core::ffi::c_int,
    pub copyCtx: CUcontext,
    pub copyParams: CUDA_MEMCPY3D,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEMSET_NODE_PARAMS_st {
    pub dst: CUdeviceptr,
    pub pitch: usize,
    pub value: ::core::ffi::c_uint,
    pub elementSize: ::core::ffi::c_uint,
    pub width: usize,
    pub height: usize,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEMSET_NODE_PARAMS_v2_st {
    pub dst: CUdeviceptr,
    pub pitch: usize,
    pub value: ::core::ffi::c_uint,
    pub elementSize: ::core::ffi::c_uint,
    pub width: usize,
    pub height: usize,
    pub ctx: CUcontext,
}
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEM_ALLOC_NODE_PARAMS_st {
    pub poolProps: CUmemPoolProps,
    pub accessDescs: *const CUmemAccessDesc,
    pub accessDescCount: usize,
    pub bytesize: usize,
    pub dptr: CUdeviceptr,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEM_ALLOC_NODE_PARAMS_v1_st {
    pub poolProps: CUmemPoolProps,
    pub accessDescs: *const CUmemAccessDesc,
    pub accessDescCount: usize,
    pub bytesize: usize,
    pub dptr: CUdeviceptr,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEM_ALLOC_NODE_PARAMS_v2_st {
    pub poolProps: CUmemPoolProps,
    pub accessDescs: *const CUmemAccessDesc,
    pub accessDescCount: usize,
    pub bytesize: usize,
    pub dptr: CUdeviceptr,
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_MEM_FREE_NODE_PARAMS_st {
    pub dptr: CUdeviceptr,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_POINTER_ATTRIBUTE_P2P_TOKENS_st {
    pub p2pToken: ::core::ffi::c_ulonglong,
    pub vaSpaceToken: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUDA_RESOURCE_DESC_st {
    pub resType: CUresourcetype,
    pub res: CUDA_RESOURCE_DESC_st__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
    pub hArray: CUarray,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2 {
    pub hMipmappedArray: CUmipmappedArray,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3 {
    pub devPtr: CUdeviceptr,
    pub format: CUarray_format,
    pub numChannels: ::core::ffi::c_uint,
    pub sizeInBytes: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4 {
    pub devPtr: CUdeviceptr,
    pub format: CUarray_format,
    pub numChannels: ::core::ffi::c_uint,
    pub width: usize,
    pub height: usize,
    pub pitchInBytes: usize,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5 {
    pub reserved: [::core::ffi::c_int; 32usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUDA_RESOURCE_VIEW_DESC_st {
    pub format: CUresourceViewFormat,
    pub width: usize,
    pub height: usize,
    pub depth: usize,
    pub firstMipmapLevel: ::core::ffi::c_uint,
    pub lastMipmapLevel: ::core::ffi::c_uint,
    pub firstLayer: ::core::ffi::c_uint,
    pub lastLayer: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 16usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct CUDA_TEXTURE_DESC_st {
    pub addressMode: [CUaddress_mode; 3usize],
    pub filterMode: CUfilter_mode,
    pub flags: ::core::ffi::c_uint,
    pub maxAnisotropy: ::core::ffi::c_uint,
    pub mipmapFilterMode: CUfilter_mode,
    pub mipmapLevelBias: f32,
    pub minMipmapLevelClamp: f32,
    pub maxMipmapLevelClamp: f32,
    pub borderColor: [f32; 4usize],
    pub reserved: [::core::ffi::c_int; 12usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialOrd, PartialEq)]
pub struct CUaccessPolicyWindow_st {
    pub base_ptr: *mut ::core::ffi::c_void,
    pub num_bytes: usize,
    pub hitRatio: f32,
    pub hitProp: CUaccessProperty,
    pub missProp: CUaccessProperty,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUarrayMapInfo_st {
    pub resourceType: CUresourcetype,
    pub resource: CUarrayMapInfo_st__bindgen_ty_1,
    pub subresourceType: CUarraySparseSubresourceType,
    pub subresource: CUarrayMapInfo_st__bindgen_ty_2,
    pub memOperationType: CUmemOperationType,
    pub memHandleType: CUmemHandleType,
    pub memHandle: CUarrayMapInfo_st__bindgen_ty_3,
    pub offset: ::core::ffi::c_ulonglong,
    pub deviceBitMask: ::core::ffi::c_uint,
    pub flags: ::core::ffi::c_uint,
    pub reserved: [::core::ffi::c_uint; 2usize],
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUarrayMapInfo_st__bindgen_ty_2__bindgen_ty_1 {
    pub level: ::core::ffi::c_uint,
    pub layer: ::core::ffi::c_uint,
    pub offsetX: ::core::ffi::c_uint,
    pub offsetY: ::core::ffi::c_uint,
    pub offsetZ: ::core::ffi::c_uint,
    pub extentWidth: ::core::ffi::c_uint,
    pub extentHeight: ::core::ffi::c_uint,
    pub extentDepth: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUarrayMapInfo_st__bindgen_ty_2__bindgen_ty_2 {
    pub layer: ::core::ffi::c_uint,
    pub offset: ::core::ffi::c_ulonglong,
    pub size: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUarray_st {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUasyncCallbackEntry_st {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUasyncNotificationInfo_st {
    pub type_: CUasyncNotificationType,
    pub info: CUasyncNotificationInfo_st__bindgen_ty_1,
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUasyncNotificationInfo_st__bindgen_ty_1__bindgen_ty_1 {
    pub bytesOverBudget: ::core::ffi::c_ulonglong,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUcheckpointCheckpointArgs_st {
    pub reserved: [cuuint64_t; 8usize],
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUcheckpointLockArgs_st {
    pub timeoutMs: ::core::ffi::c_uint,
    pub reserved0: ::core::ffi::c_uint,
    pub reserved1: [cuuint64_t; 7usize],
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUcheckpointRestoreArgs_st {
    pub reserved: [cuuint64_t; 8usize],
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUcheckpointUnlockArgs_st {
    pub reserved: [cuuint64_t; 8usize],
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUctxCigParam_st {
    pub sharedDataType: CUcigDataType,
    pub sharedData: *mut ::core::ffi::c_void,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUctxCreateParams_st {
    pub execAffinityParams: *mut CUexecAffinityParam,
    pub numExecAffinityParams: ::core::ffi::c_int,
    pub cigParams: *mut CUctxCigParam,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUctx_st {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUdevResourceDesc_st {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUdevResource_st {
    pub type_: CUdevResourceType,
    pub _internal_padding: [::core::ffi::c_uchar; 92usize],
    pub __bindgen_anon_1: CUdevResource_st__bindgen_ty_1,
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUdevSmResource_st {
    pub smCount: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUdevprop_st {
    pub maxThreadsPerBlock: ::core::ffi::c_int,
    pub maxThreadsDim: [::core::ffi::c_int; 3usize],
    pub maxGridSize: [::core::ffi::c_int; 3usize],
    pub sharedMemPerBlock: ::core::ffi::c_int,
    pub totalConstantMemory: ::core::ffi::c_int,
    pub SIMDWidth: ::core::ffi::c_int,
    pub memPitch: ::core::ffi::c_int,
    pub regsPerBlock: ::core::ffi::c_int,
    pub clockRate: ::core::ffi::c_int,
    pub textureAlign: ::core::ffi::c_int,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUevent_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUexecAffinityParam_st {
    pub type_: CUexecAffinityType,
    pub param: CUexecAffinityParam_st__bindgen_ty_1,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUexecAffinitySmCount_st {
    pub val: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUextMemory_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUextSemaphore_st {
    _unused: [u8; 0],
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUextent3D_st {
    pub width: usize,
    pub height: usize,
    pub depth: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUfunc_st {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraphDeviceUpdatableNode_st {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUgraphEdgeData_st {
    pub from_port: ::core::ffi::c_uchar,
    pub to_port: ::core::ffi::c_uchar,
    pub type_: ::core::ffi::c_uchar,
    pub reserved: [::core::ffi::c_uchar; 5usize],
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUgraphExecUpdateResultInfo_st {
    pub result: CUgraphExecUpdateResult,
    pub errorNode: CUgraphNode,
    pub errorFromNode: CUgraphNode,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraphExec_st {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUgraphNodeParams_st {
    pub type_: CUgraphNodeType,
    pub reserved0: [::core::ffi::c_int; 3usize],
    pub __bindgen_anon_1: CUgraphNodeParams_st__bindgen_ty_1,
    pub reserved2: ::core::ffi::c_longlong,
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
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgraphicsResource_st {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUgreenCtx_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUipcEventHandle_st {
    pub reserved: [::core::ffi::c_char; 64usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUipcMemHandle_st {
    pub reserved: [::core::ffi::c_char; 64usize],
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUkern_st {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchAttributeValue_union__bindgen_ty_1 {
    pub x: ::core::ffi::c_uint,
    pub y: ::core::ffi::c_uint,
    pub z: ::core::ffi::c_uint,
}
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchAttributeValue_union__bindgen_ty_2 {
    pub event: CUevent,
    pub flags: ::core::ffi::c_int,
    pub triggerAtBlockStart: ::core::ffi::c_int,
}
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchAttributeValue_union__bindgen_ty_3 {
    pub event: CUevent,
    pub flags: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchAttributeValue_union__bindgen_ty_4 {
    pub deviceUpdatable: ::core::ffi::c_int,
    pub devNode: CUgraphDeviceNode,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchAttributeValue_union__bindgen_ty_4 {
    pub x: ::core::ffi::c_uint,
    pub y: ::core::ffi::c_uint,
    pub z: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchAttributeValue_union__bindgen_ty_5 {
    pub deviceUpdatable: ::core::ffi::c_int,
    pub devNode: CUgraphDeviceNode,
}
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUlaunchAttribute_st {
    pub id: CUlaunchAttributeID,
    pub pad: [::core::ffi::c_char; 4usize],
    pub value: CUlaunchAttributeValue,
}
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchConfig_st {
    pub gridDimX: ::core::ffi::c_uint,
    pub gridDimY: ::core::ffi::c_uint,
    pub gridDimZ: ::core::ffi::c_uint,
    pub blockDimX: ::core::ffi::c_uint,
    pub blockDimY: ::core::ffi::c_uint,
    pub blockDimZ: ::core::ffi::c_uint,
    pub sharedMemBytes: ::core::ffi::c_uint,
    pub hStream: CUstream,
    pub attrs: *mut CUlaunchAttribute,
    pub numAttrs: ::core::ffi::c_uint,
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlaunchMemSyncDomainMap_st {
    pub default_: ::core::ffi::c_uchar,
    pub remote: ::core::ffi::c_uchar,
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUlib_st {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUlibraryHostUniversalFunctionAndDataTable_st {
    pub functionTable: *mut ::core::ffi::c_void,
    pub functionWindowSize: usize,
    pub dataTable: *mut ::core::ffi::c_void,
    pub dataWindowSize: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUlinkState_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemAccessDesc_st {
    pub location: CUmemLocation,
    pub flags: CUmemAccess_flags,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemAllocationProp_st {
    pub type_: CUmemAllocationType,
    pub requestedHandleTypes: CUmemAllocationHandleType,
    pub location: CUmemLocation,
    pub win32HandleMetaData: *mut ::core::ffi::c_void,
    pub allocFlags: CUmemAllocationProp_st__bindgen_ty_1,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemAllocationProp_st__bindgen_ty_1 {
    pub compressionType: ::core::ffi::c_uchar,
    pub gpuDirectRDMACapable: ::core::ffi::c_uchar,
    pub usage: ::core::ffi::c_ushort,
    pub reserved: [::core::ffi::c_uchar; 4usize],
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemDecompressParams_st {
    pub srcNumBytes: usize,
    pub dstNumBytes: usize,
    pub dstActBytes: *mut cuuint32_t,
    pub src: *const ::core::ffi::c_void,
    pub dst: *mut ::core::ffi::c_void,
    pub algo: CUmemDecompressAlgorithm,
    pub padding: [::core::ffi::c_uchar; 20usize],
}
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemFabricHandle_st {
    pub data: [::core::ffi::c_uchar; 64usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemLocation_st {
    pub type_: CUmemLocationType,
    pub id: ::core::ffi::c_int,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUmemPoolHandle_st {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemPoolProps_st {
    pub allocType: CUmemAllocationType,
    pub handleTypes: CUmemAllocationHandleType,
    pub location: CUmemLocation,
    pub win32SecurityAttributes: *mut ::core::ffi::c_void,
    pub reserved: [::core::ffi::c_uchar; 64usize],
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050"
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemPoolProps_st {
    pub allocType: CUmemAllocationType,
    pub handleTypes: CUmemAllocationHandleType,
    pub location: CUmemLocation,
    pub win32SecurityAttributes: *mut ::core::ffi::c_void,
    pub maxSize: usize,
    pub reserved: [::core::ffi::c_uchar; 56usize],
}
#[cfg(any(feature = "cuda-12060", feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemPoolProps_st {
    pub allocType: CUmemAllocationType,
    pub handleTypes: CUmemAllocationHandleType,
    pub location: CUmemLocation,
    pub win32SecurityAttributes: *mut ::core::ffi::c_void,
    pub maxSize: usize,
    pub usage: ::core::ffi::c_ushort,
    pub reserved: [::core::ffi::c_uchar; 54usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemPoolPtrExportData_st {
    pub reserved: [::core::ffi::c_uchar; 64usize],
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUmemcpy3DOperand_st {
    pub type_: CUmemcpy3DOperandType,
    pub op: CUmemcpy3DOperand_st__bindgen_ty_1,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemcpy3DOperand_st__bindgen_ty_1__bindgen_ty_1 {
    pub ptr: CUdeviceptr,
    pub rowLength: usize,
    pub layerHeight: usize,
    pub locHint: CUmemLocation,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemcpy3DOperand_st__bindgen_ty_1__bindgen_ty_2 {
    pub array: CUarray,
    pub offset: CUoffset3D,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmemcpyAttributes_st {
    pub srcAccessOrder: CUmemcpySrcAccessOrder,
    pub srcLocHint: CUmemLocation,
    pub dstLocHint: CUmemLocation,
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUmipmappedArray_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUmod_st {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUmulticastObjectProp_st {
    pub numDevices: ::core::ffi::c_uint,
    pub size: usize,
    pub handleTypes: ::core::ffi::c_ulonglong,
    pub flags: ::core::ffi::c_ulonglong,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUoffset3D_st {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUstreamBatchMemOpParams_union_CUstreamMemOpFlushRemoteWritesParams_st {
    pub operation: CUstreamBatchMemOpType,
    pub flags: ::core::ffi::c_uint,
}
#[cfg(
    any(
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
    )
)]
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUstreamBatchMemOpParams_union_CUstreamMemOpMemoryBarrierParams_st {
    pub operation: CUstreamBatchMemOpType,
    pub flags: ::core::ffi::c_uint,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st {
    pub operation: CUstreamBatchMemOpType,
    pub address: CUdeviceptr,
    pub __bindgen_anon_1: CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub alias: CUdeviceptr,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st {
    pub operation: CUstreamBatchMemOpType,
    pub address: CUdeviceptr,
    pub __bindgen_anon_1: CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st__bindgen_ty_1,
    pub flags: ::core::ffi::c_uint,
    pub alias: CUdeviceptr,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUsurfref_st {
    _unused: [u8; 0],
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[repr(align(64))]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct CUtensorMap_st {
    pub opaque: [cuuint64_t; 16usize],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUtexref_st {
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
impl CUctx_flags_enum {
    pub const CU_CTX_BLOCKING_SYNC: CUctx_flags_enum = CUctx_flags_enum::CU_CTX_SCHED_BLOCKING_SYNC;
}
impl CUdevice_P2PAttribute_enum {
    pub const CU_DEVICE_P2P_ATTRIBUTE_CUDA_ARRAY_ACCESS_SUPPORTED: CUdevice_P2PAttribute_enum = CUdevice_P2PAttribute_enum::CU_DEVICE_P2P_ATTRIBUTE_ACCESS_ACCESS_SUPPORTED;
}
impl CUdevice_attribute_enum {
    pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT: CUdevice_attribute_enum = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_HEIGHT;
}
impl CUdevice_attribute_enum {
    pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES: CUdevice_attribute_enum = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_LAYERS;
}
impl CUdevice_attribute_enum {
    pub const CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH: CUdevice_attribute_enum = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LAYERED_WIDTH;
}
impl CUdevice_attribute_enum {
    pub const CU_DEVICE_ATTRIBUTE_REGISTERS_PER_BLOCK: CUdevice_attribute_enum = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK;
}
impl CUdevice_attribute_enum {
    pub const CU_DEVICE_ATTRIBUTE_SHARED_MEMORY_PER_BLOCK: CUdevice_attribute_enum = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK;
}
impl CUdevice_attribute_enum {
    pub const CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED: CUdevice_attribute_enum = CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED;
}
impl Default for CUDA_ARRAY3D_DESCRIPTOR_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_ARRAY_DESCRIPTOR_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010"
    )
)]
impl Default for CUDA_BATCH_MEM_OP_NODE_PARAMS_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_BATCH_MEM_OP_NODE_PARAMS_v1_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_BATCH_MEM_OP_NODE_PARAMS_v2_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_CHILD_GRAPH_NODE_PARAMS_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_CONDITIONAL_NODE_PARAMS {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_EVENT_RECORD_NODE_PARAMS_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_EVENT_WAIT_NODE_PARAMS_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_EXT_SEM_WAIT_NODE_PARAMS_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_GRAPH_INSTANTIATE_PARAMS_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_HOST_NODE_PARAMS_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_HOST_NODE_PARAMS_v2_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_KERNEL_NODE_PARAMS_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_KERNEL_NODE_PARAMS_v2_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_KERNEL_NODE_PARAMS_v3_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_LAUNCH_PARAMS_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_MEMCPY2D_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080"))]
impl Default for CUDA_MEMCPY3D_BATCH_OP_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_MEMCPY3D_PEER_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_MEMCPY3D_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_MEMCPY_NODE_PARAMS_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_MEMSET_NODE_PARAMS_v2_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070",
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010"
    )
)]
impl Default for CUDA_MEM_ALLOC_NODE_PARAMS_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_MEM_ALLOC_NODE_PARAMS_v1_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUDA_MEM_ALLOC_NODE_PARAMS_v2_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_RESOURCE_DESC_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_RESOURCE_VIEW_DESC_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUDA_TEXTURE_DESC_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUaccessPolicyWindow_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUarrayMapInfo_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUarrayMapInfo_st__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUarrayMapInfo_st__bindgen_ty_2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUarrayMapInfo_st__bindgen_ty_3 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUasyncNotificationInfo_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUasyncNotificationInfo_st__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
impl Default for CUctxCigParam_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
impl Default for CUctxCreateParams_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUdevResource_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUdevResource_st__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUexecAffinityParam_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUexecAffinityParam_st__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUgraphExecUpdateResultInfo_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUgraphNodeParams_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUgraphNodeParams_st__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUipcEventHandle_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUipcMemHandle_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    )
)]
impl Default for CUkernelNodeAttrValue_union {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUlaunchAttributeValue_union {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUlaunchAttributeValue_union__bindgen_ty_2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUlaunchAttributeValue_union__bindgen_ty_3 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12040", feature = "cuda-12050", feature = "cuda-12060"))]
impl Default for CUlaunchAttributeValue_union__bindgen_ty_4 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080"))]
impl Default for CUlaunchAttributeValue_union__bindgen_ty_5 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUlaunchAttribute_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-11080",
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUlaunchConfig_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUlibraryHostUniversalFunctionAndDataTable_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUmemAccessDesc_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUmemAllocationProp_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080"))]
impl Default for CUmemDecompressParams_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUmemFabricHandle_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUmemLocation_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUmemPoolProps_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUmemPoolPtrExportData_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080"))]
impl Default for CUmemcpy3DOperand_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080"))]
impl Default for CUmemcpy3DOperand_st__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080"))]
impl Default for CUmemcpy3DOperand_st__bindgen_ty_1__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080"))]
impl Default for CUmemcpy3DOperand_st__bindgen_ty_1__bindgen_ty_2 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(any(feature = "cuda-12080"))]
impl Default for CUmemcpyAttributes_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    )
)]
impl Default for CUstreamAttrValue_union {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUstreamBatchMemOpParams_union {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUstreamBatchMemOpParams_union_CUstreamMemOpFlushRemoteWritesParams_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
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
    )
)]
impl Default for CUstreamBatchMemOpParams_union_CUstreamMemOpMemoryBarrierParams_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default
for CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default
for CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st__bindgen_ty_1 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(
    any(
        feature = "cuda-12000",
        feature = "cuda-12010",
        feature = "cuda-12020",
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
impl Default for CUtensorMap_st {
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
pub union CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1 {
    pub fd: ::core::ffi::c_int,
    pub win32: CUDA_EXTERNAL_MEMORY_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1,
    pub nvSciBufObject: *const ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1 {
    pub fd: ::core::ffi::c_int,
    pub win32: CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC_st__bindgen_ty_1__bindgen_ty_1,
    pub nvSciSyncObj: *const ::core::ffi::c_void,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS_st__bindgen_ty_1__bindgen_ty_2 {
    pub fence: *mut ::core::ffi::c_void,
    pub reserved: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS_st__bindgen_ty_1__bindgen_ty_2 {
    pub fence: *mut ::core::ffi::c_void,
    pub reserved: ::core::ffi::c_ulonglong,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUDA_RESOURCE_DESC_st__bindgen_ty_1 {
    pub array: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_1,
    pub mipmap: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_2,
    pub linear: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_3,
    pub pitch2D: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_4,
    pub reserved: CUDA_RESOURCE_DESC_st__bindgen_ty_1__bindgen_ty_5,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUarrayMapInfo_st__bindgen_ty_1 {
    pub mipmap: CUmipmappedArray,
    pub array: CUarray,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUarrayMapInfo_st__bindgen_ty_2 {
    pub sparseLevel: CUarrayMapInfo_st__bindgen_ty_2__bindgen_ty_1,
    pub miptail: CUarrayMapInfo_st__bindgen_ty_2__bindgen_ty_2,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUarrayMapInfo_st__bindgen_ty_3 {
    pub memHandle: CUmemGenericAllocationHandle,
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUasyncNotificationInfo_st__bindgen_ty_1 {
    pub overBudget: CUasyncNotificationInfo_st__bindgen_ty_1__bindgen_ty_1,
}
#[cfg(
    any(
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUdevResource_st__bindgen_ty_1 {
    pub sm: CUdevSmResource,
    pub _oversize: [::core::ffi::c_uchar; 48usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUexecAffinityParam_st__bindgen_ty_1 {
    pub smCount: CUexecAffinitySmCount,
}
#[cfg(any(feature = "cuda-12020"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUgraphNodeParams_st__bindgen_ty_1 {
    pub reserved1: [::core::ffi::c_longlong; 29usize],
    pub kernel: CUDA_KERNEL_NODE_PARAMS_v3,
    pub memcpy: CUDA_MEMCPY_NODE_PARAMS,
    pub memset: CUDA_MEMSET_NODE_PARAMS_v2,
    pub host: CUDA_HOST_NODE_PARAMS_v2,
    pub graph: CUDA_CHILD_GRAPH_NODE_PARAMS,
    pub eventWait: CUDA_EVENT_WAIT_NODE_PARAMS,
    pub eventRecord: CUDA_EVENT_RECORD_NODE_PARAMS,
    pub extSemSignal: CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2,
    pub extSemWait: CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2,
    pub alloc: CUDA_MEM_ALLOC_NODE_PARAMS_v2,
    pub free: CUDA_MEM_FREE_NODE_PARAMS,
    pub memOp: CUDA_BATCH_MEM_OP_NODE_PARAMS_v2,
}
#[cfg(
    any(
        feature = "cuda-12030",
        feature = "cuda-12040",
        feature = "cuda-12050",
        feature = "cuda-12060",
        feature = "cuda-12080"
    )
)]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUgraphNodeParams_st__bindgen_ty_1 {
    pub reserved1: [::core::ffi::c_longlong; 29usize],
    pub kernel: CUDA_KERNEL_NODE_PARAMS_v3,
    pub memcpy: CUDA_MEMCPY_NODE_PARAMS,
    pub memset: CUDA_MEMSET_NODE_PARAMS_v2,
    pub host: CUDA_HOST_NODE_PARAMS_v2,
    pub graph: CUDA_CHILD_GRAPH_NODE_PARAMS,
    pub eventWait: CUDA_EVENT_WAIT_NODE_PARAMS,
    pub eventRecord: CUDA_EVENT_RECORD_NODE_PARAMS,
    pub extSemSignal: CUDA_EXT_SEM_SIGNAL_NODE_PARAMS_v2,
    pub extSemWait: CUDA_EXT_SEM_WAIT_NODE_PARAMS_v2,
    pub alloc: CUDA_MEM_ALLOC_NODE_PARAMS_v2,
    pub free: CUDA_MEM_FREE_NODE_PARAMS,
    pub memOp: CUDA_BATCH_MEM_OP_NODE_PARAMS_v2,
    pub conditional: CUDA_CONDITIONAL_NODE_PARAMS,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUkernelNodeAttrValue_union {
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-11070"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUkernelNodeAttrValue_union {
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub priority: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-11080"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUlaunchAttributeValue_union {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: CUsynchronizationPolicy,
    pub clusterDim: CUlaunchAttributeValue_union__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: CUclusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: CUlaunchAttributeValue_union__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
}
#[cfg(any(feature = "cuda-12000", feature = "cuda-12010", feature = "cuda-12020"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUlaunchAttributeValue_union {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: CUsynchronizationPolicy,
    pub clusterDim: CUlaunchAttributeValue_union__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: CUclusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: CUlaunchAttributeValue_union__bindgen_ty_2,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: CUlaunchMemSyncDomainMap,
    pub memSyncDomain: CUlaunchMemSyncDomain,
}
#[cfg(any(feature = "cuda-12030"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUlaunchAttributeValue_union {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: CUsynchronizationPolicy,
    pub clusterDim: CUlaunchAttributeValue_union__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: CUclusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: CUlaunchAttributeValue_union__bindgen_ty_2,
    pub launchCompletionEvent: CUlaunchAttributeValue_union__bindgen_ty_3,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: CUlaunchMemSyncDomainMap,
    pub memSyncDomain: CUlaunchMemSyncDomain,
}
#[cfg(any(feature = "cuda-12040"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUlaunchAttributeValue_union {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: CUsynchronizationPolicy,
    pub clusterDim: CUlaunchAttributeValue_union__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: CUclusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: CUlaunchAttributeValue_union__bindgen_ty_2,
    pub launchCompletionEvent: CUlaunchAttributeValue_union__bindgen_ty_3,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: CUlaunchMemSyncDomainMap,
    pub memSyncDomain: CUlaunchMemSyncDomain,
    pub deviceUpdatableKernelNode: CUlaunchAttributeValue_union__bindgen_ty_4,
}
#[cfg(any(feature = "cuda-12050", feature = "cuda-12060"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUlaunchAttributeValue_union {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: CUsynchronizationPolicy,
    pub clusterDim: CUlaunchAttributeValue_union__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: CUclusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: CUlaunchAttributeValue_union__bindgen_ty_2,
    pub launchCompletionEvent: CUlaunchAttributeValue_union__bindgen_ty_3,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: CUlaunchMemSyncDomainMap,
    pub memSyncDomain: CUlaunchMemSyncDomain,
    pub deviceUpdatableKernelNode: CUlaunchAttributeValue_union__bindgen_ty_4,
    pub sharedMemCarveout: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUlaunchAttributeValue_union {
    pub pad: [::core::ffi::c_char; 64usize],
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub cooperative: ::core::ffi::c_int,
    pub syncPolicy: CUsynchronizationPolicy,
    pub clusterDim: CUlaunchAttributeValue_union__bindgen_ty_1,
    pub clusterSchedulingPolicyPreference: CUclusterSchedulingPolicy,
    pub programmaticStreamSerializationAllowed: ::core::ffi::c_int,
    pub programmaticEvent: CUlaunchAttributeValue_union__bindgen_ty_2,
    pub launchCompletionEvent: CUlaunchAttributeValue_union__bindgen_ty_3,
    pub priority: ::core::ffi::c_int,
    pub memSyncDomainMap: CUlaunchMemSyncDomainMap,
    pub memSyncDomain: CUlaunchMemSyncDomain,
    pub preferredClusterDim: CUlaunchAttributeValue_union__bindgen_ty_4,
    pub deviceUpdatableKernelNode: CUlaunchAttributeValue_union__bindgen_ty_5,
    pub sharedMemCarveout: ::core::ffi::c_uint,
}
#[cfg(any(feature = "cuda-12080"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUmemcpy3DOperand_st__bindgen_ty_1 {
    pub ptr: CUmemcpy3DOperand_st__bindgen_ty_1__bindgen_ty_1,
    pub array: CUmemcpy3DOperand_st__bindgen_ty_1__bindgen_ty_2,
}
#[cfg(
    any(
        feature = "cuda-11040",
        feature = "cuda-11050",
        feature = "cuda-11060",
        feature = "cuda-11070"
    )
)]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUstreamAttrValue_union {
    pub accessPolicyWindow: CUaccessPolicyWindow,
    pub syncPolicy: CUsynchronizationPolicy,
}
#[cfg(any(feature = "cuda-11040", feature = "cuda-11050", feature = "cuda-11060"))]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUstreamBatchMemOpParams_union {
    pub operation: CUstreamBatchMemOpType,
    pub waitValue: CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st,
    pub writeValue: CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st,
    pub flushRemoteWrites: CUstreamBatchMemOpParams_union_CUstreamMemOpFlushRemoteWritesParams_st,
    pub pad: [cuuint64_t; 6usize],
}
#[cfg(
    any(
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
    )
)]
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUstreamBatchMemOpParams_union {
    pub operation: CUstreamBatchMemOpType,
    pub waitValue: CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st,
    pub writeValue: CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st,
    pub flushRemoteWrites: CUstreamBatchMemOpParams_union_CUstreamMemOpFlushRemoteWritesParams_st,
    pub memoryBarrier: CUstreamBatchMemOpParams_union_CUstreamMemOpMemoryBarrierParams_st,
    pub pad: [cuuint64_t; 6usize],
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUstreamBatchMemOpParams_union_CUstreamMemOpWaitValueParams_st__bindgen_ty_1 {
    pub value: cuuint32_t,
    pub value64: cuuint64_t,
}
#[repr(C)]
#[derive(Copy, Clone)]
pub union CUstreamBatchMemOpParams_union_CUstreamMemOpWriteValueParams_st__bindgen_ty_1 {
    pub value: cuuint32_t,
    pub value64: cuuint64_t,
}
#[cfg(not(feature = "dynamic-loading"))]
extern "C" {
    pub fn cuArray3DCreate_v2(
        pHandle: *mut CUarray,
        pAllocateArray: *const CUDA_ARRAY3D_DESCRIPTOR,
    ) -> CUresult;
    pub fn cuArray3DGetDescriptor_v2(
        pArrayDescriptor: *mut CUDA_ARRAY3D_DESCRIPTOR,
        hArray: CUarray,
    ) -> CUresult;
    pub fn cuArrayCreate_v2(
        pHandle: *mut CUarray,
        pAllocateArray: *const CUDA_ARRAY_DESCRIPTOR,
    ) -> CUresult;
    pub fn cuArrayDestroy(hArray: CUarray) -> CUresult;
    pub fn cuArrayGetDescriptor_v2(
        pArrayDescriptor: *mut CUDA_ARRAY_DESCRIPTOR,
        hArray: CUarray,
    ) -> CUresult;
    #[cfg(
        any(
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
        )
    )]
    pub fn cuArrayGetMemoryRequirements(
        memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS,
        array: CUarray,
        device: CUdevice,
    ) -> CUresult;
    pub fn cuArrayGetPlane(
        pPlaneArray: *mut CUarray,
        hArray: CUarray,
        planeIdx: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuArrayGetSparseProperties(
        sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES,
        array: CUarray,
    ) -> CUresult;
    #[cfg(any(feature = "cuda-12080"))]
    pub fn cuCheckpointProcessCheckpoint(
        pid: ::core::ffi::c_int,
        args: *mut CUcheckpointCheckpointArgs,
    ) -> CUresult;
    #[cfg(any(feature = "cuda-12080"))]
    pub fn cuCheckpointProcessGetRestoreThreadId(
        pid: ::core::ffi::c_int,
        tid: *mut ::core::ffi::c_int,
    ) -> CUresult;
    #[cfg(any(feature = "cuda-12080"))]
    pub fn cuCheckpointProcessGetState(
        pid: ::core::ffi::c_int,
        state: *mut CUprocessState,
    ) -> CUresult;
    #[cfg(any(feature = "cuda-12080"))]
    pub fn cuCheckpointProcessLock(
        pid: ::core::ffi::c_int,
        args: *mut CUcheckpointLockArgs,
    ) -> CUresult;
    #[cfg(any(feature = "cuda-12080"))]
    pub fn cuCheckpointProcessRestore(
        pid: ::core::ffi::c_int,
        args: *mut CUcheckpointRestoreArgs,
    ) -> CUresult;
    #[cfg(any(feature = "cuda-12080"))]
    pub fn cuCheckpointProcessUnlock(
        pid: ::core::ffi::c_int,
        args: *mut CUcheckpointUnlockArgs,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuCoredumpGetAttribute(
        attrib: CUcoredumpSettings,
        value: *mut ::core::ffi::c_void,
        size: *mut usize,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuCoredumpGetAttributeGlobal(
        attrib: CUcoredumpSettings,
        value: *mut ::core::ffi::c_void,
        size: *mut usize,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuCoredumpSetAttribute(
        attrib: CUcoredumpSettings,
        value: *mut ::core::ffi::c_void,
        size: *mut usize,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuCoredumpSetAttributeGlobal(
        attrib: CUcoredumpSettings,
        value: *mut ::core::ffi::c_void,
        size: *mut usize,
    ) -> CUresult;
    pub fn cuCtxAttach(pctx: *mut CUcontext, flags: ::core::ffi::c_uint) -> CUresult;
    pub fn cuCtxCreate_v2(
        pctx: *mut CUcontext,
        flags: ::core::ffi::c_uint,
        dev: CUdevice,
    ) -> CUresult;
    pub fn cuCtxCreate_v3(
        pctx: *mut CUcontext,
        paramsArray: *mut CUexecAffinityParam,
        numParams: ::core::ffi::c_int,
        flags: ::core::ffi::c_uint,
        dev: CUdevice,
    ) -> CUresult;
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub fn cuCtxCreate_v4(
        pctx: *mut CUcontext,
        ctxCreateParams: *mut CUctxCreateParams,
        flags: ::core::ffi::c_uint,
        dev: CUdevice,
    ) -> CUresult;
    pub fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult;
    pub fn cuCtxDetach(ctx: CUcontext) -> CUresult;
    pub fn cuCtxDisablePeerAccess(peerContext: CUcontext) -> CUresult;
    pub fn cuCtxEnablePeerAccess(
        peerContext: CUcontext,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuCtxFromGreenCtx(pContext: *mut CUcontext, hCtx: CUgreenCtx) -> CUresult;
    pub fn cuCtxGetApiVersion(
        ctx: CUcontext,
        version: *mut ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuCtxGetCacheConfig(pconfig: *mut CUfunc_cache) -> CUresult;
    pub fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuCtxGetDevResource(
        hCtx: CUcontext,
        resource: *mut CUdevResource,
        type_: CUdevResourceType,
    ) -> CUresult;
    pub fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult;
    pub fn cuCtxGetExecAffinity(
        pExecAffinity: *mut CUexecAffinityParam,
        type_: CUexecAffinityType,
    ) -> CUresult;
    pub fn cuCtxGetFlags(flags: *mut ::core::ffi::c_uint) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuCtxGetId(ctx: CUcontext, ctxId: *mut ::core::ffi::c_ulonglong) -> CUresult;
    pub fn cuCtxGetLimit(pvalue: *mut usize, limit: CUlimit) -> CUresult;
    pub fn cuCtxGetSharedMemConfig(pConfig: *mut CUsharedconfig) -> CUresult;
    pub fn cuCtxGetStreamPriorityRange(
        leastPriority: *mut ::core::ffi::c_int,
        greatestPriority: *mut ::core::ffi::c_int,
    ) -> CUresult;
    pub fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult;
    pub fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult;
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub fn cuCtxRecordEvent(hCtx: CUcontext, hEvent: CUevent) -> CUresult;
    pub fn cuCtxResetPersistingL2Cache() -> CUresult;
    pub fn cuCtxSetCacheConfig(config: CUfunc_cache) -> CUresult;
    pub fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuCtxSetFlags(flags: ::core::ffi::c_uint) -> CUresult;
    pub fn cuCtxSetLimit(limit: CUlimit, value: usize) -> CUresult;
    pub fn cuCtxSetSharedMemConfig(config: CUsharedconfig) -> CUresult;
    pub fn cuCtxSynchronize() -> CUresult;
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub fn cuCtxWaitEvent(hCtx: CUcontext, hEvent: CUevent) -> CUresult;
    pub fn cuDestroyExternalMemory(extMem: CUexternalMemory) -> CUresult;
    pub fn cuDestroyExternalSemaphore(extSem: CUexternalSemaphore) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuDevResourceGenerateDesc(
        phDesc: *mut CUdevResourceDesc,
        resources: *mut CUdevResource,
        nbResources: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuDevSmResourceSplitByCount(
        result: *mut CUdevResource,
        nbGroups: *mut ::core::ffi::c_uint,
        input: *const CUdevResource,
        remaining: *mut CUdevResource,
        useFlags: ::core::ffi::c_uint,
        minCount: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuDeviceCanAccessPeer(
        canAccessPeer: *mut ::core::ffi::c_int,
        dev: CUdevice,
        peerDev: CUdevice,
    ) -> CUresult;
    pub fn cuDeviceComputeCapability(
        major: *mut ::core::ffi::c_int,
        minor: *mut ::core::ffi::c_int,
        dev: CUdevice,
    ) -> CUresult;
    pub fn cuDeviceGet(device: *mut CUdevice, ordinal: ::core::ffi::c_int) -> CUresult;
    pub fn cuDeviceGetAttribute(
        pi: *mut ::core::ffi::c_int,
        attrib: CUdevice_attribute,
        dev: CUdevice,
    ) -> CUresult;
    pub fn cuDeviceGetByPCIBusId(
        dev: *mut CUdevice,
        pciBusId: *const ::core::ffi::c_char,
    ) -> CUresult;
    pub fn cuDeviceGetCount(count: *mut ::core::ffi::c_int) -> CUresult;
    pub fn cuDeviceGetDefaultMemPool(
        pool_out: *mut CUmemoryPool,
        dev: CUdevice,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuDeviceGetDevResource(
        device: CUdevice,
        resource: *mut CUdevResource,
        type_: CUdevResourceType,
    ) -> CUresult;
    pub fn cuDeviceGetExecAffinitySupport(
        pi: *mut ::core::ffi::c_int,
        type_: CUexecAffinityType,
        dev: CUdevice,
    ) -> CUresult;
    pub fn cuDeviceGetGraphMemAttribute(
        device: CUdevice,
        attr: CUgraphMem_attribute,
        value: *mut ::core::ffi::c_void,
    ) -> CUresult;
    pub fn cuDeviceGetLuid(
        luid: *mut ::core::ffi::c_char,
        deviceNodeMask: *mut ::core::ffi::c_uint,
        dev: CUdevice,
    ) -> CUresult;
    pub fn cuDeviceGetMemPool(pool: *mut CUmemoryPool, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetName(
        name: *mut ::core::ffi::c_char,
        len: ::core::ffi::c_int,
        dev: CUdevice,
    ) -> CUresult;
    pub fn cuDeviceGetNvSciSyncAttributes(
        nvSciSyncAttrList: *mut ::core::ffi::c_void,
        dev: CUdevice,
        flags: ::core::ffi::c_int,
    ) -> CUresult;
    pub fn cuDeviceGetP2PAttribute(
        value: *mut ::core::ffi::c_int,
        attrib: CUdevice_P2PAttribute,
        srcDevice: CUdevice,
        dstDevice: CUdevice,
    ) -> CUresult;
    pub fn cuDeviceGetPCIBusId(
        pciBusId: *mut ::core::ffi::c_char,
        len: ::core::ffi::c_int,
        dev: CUdevice,
    ) -> CUresult;
    pub fn cuDeviceGetProperties(prop: *mut CUdevprop, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetTexture1DLinearMaxWidth(
        maxWidthInElements: *mut usize,
        format: CUarray_format,
        numChannels: ::core::ffi::c_uint,
        dev: CUdevice,
    ) -> CUresult;
    pub fn cuDeviceGetUuid(uuid: *mut CUuuid, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGetUuid_v2(uuid: *mut CUuuid, dev: CUdevice) -> CUresult;
    pub fn cuDeviceGraphMemTrim(device: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxGetState(
        dev: CUdevice,
        flags: *mut ::core::ffi::c_uint,
        active: *mut ::core::ffi::c_int,
    ) -> CUresult;
    pub fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxReset_v2(dev: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxRetain(pctx: *mut CUcontext, dev: CUdevice) -> CUresult;
    pub fn cuDevicePrimaryCtxSetFlags_v2(
        dev: CUdevice,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuDeviceRegisterAsyncNotification(
        device: CUdevice,
        callbackFunc: CUasyncCallback,
        userData: *mut ::core::ffi::c_void,
        callback: *mut CUasyncCallbackHandle,
    ) -> CUresult;
    pub fn cuDeviceSetGraphMemAttribute(
        device: CUdevice,
        attr: CUgraphMem_attribute,
        value: *mut ::core::ffi::c_void,
    ) -> CUresult;
    pub fn cuDeviceSetMemPool(dev: CUdevice, pool: CUmemoryPool) -> CUresult;
    pub fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuDeviceUnregisterAsyncNotification(
        device: CUdevice,
        callback: CUasyncCallbackHandle,
    ) -> CUresult;
    pub fn cuDriverGetVersion(driverVersion: *mut ::core::ffi::c_int) -> CUresult;
    pub fn cuEventCreate(phEvent: *mut CUevent, Flags: ::core::ffi::c_uint) -> CUresult;
    pub fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult;
    pub fn cuEventElapsedTime(
        pMilliseconds: *mut f32,
        hStart: CUevent,
        hEnd: CUevent,
    ) -> CUresult;
    #[cfg(any(feature = "cuda-12080"))]
    pub fn cuEventElapsedTime_v2(
        pMilliseconds: *mut f32,
        hStart: CUevent,
        hEnd: CUevent,
    ) -> CUresult;
    pub fn cuEventQuery(hEvent: CUevent) -> CUresult;
    pub fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult;
    pub fn cuEventRecordWithFlags(
        hEvent: CUevent,
        hStream: CUstream,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuEventSynchronize(hEvent: CUevent) -> CUresult;
    pub fn cuExternalMemoryGetMappedBuffer(
        devPtr: *mut CUdeviceptr,
        extMem: CUexternalMemory,
        bufferDesc: *const CUDA_EXTERNAL_MEMORY_BUFFER_DESC,
    ) -> CUresult;
    pub fn cuExternalMemoryGetMappedMipmappedArray(
        mipmap: *mut CUmipmappedArray,
        extMem: CUexternalMemory,
        mipmapDesc: *const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC,
    ) -> CUresult;
    pub fn cuFlushGPUDirectRDMAWrites(
        target: CUflushGPUDirectRDMAWritesTarget,
        scope: CUflushGPUDirectRDMAWritesScope,
    ) -> CUresult;
    pub fn cuFuncGetAttribute(
        pi: *mut ::core::ffi::c_int,
        attrib: CUfunction_attribute,
        hfunc: CUfunction,
    ) -> CUresult;
    pub fn cuFuncGetModule(hmod: *mut CUmodule, hfunc: CUfunction) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuFuncGetName(
        name: *mut *const ::core::ffi::c_char,
        hfunc: CUfunction,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuFuncGetParamInfo(
        func: CUfunction,
        paramIndex: usize,
        paramOffset: *mut usize,
        paramSize: *mut usize,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuFuncIsLoaded(
        state: *mut CUfunctionLoadingState,
        function: CUfunction,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuFuncLoad(function: CUfunction) -> CUresult;
    pub fn cuFuncSetAttribute(
        hfunc: CUfunction,
        attrib: CUfunction_attribute,
        value: ::core::ffi::c_int,
    ) -> CUresult;
    pub fn cuFuncSetBlockShape(
        hfunc: CUfunction,
        x: ::core::ffi::c_int,
        y: ::core::ffi::c_int,
        z: ::core::ffi::c_int,
    ) -> CUresult;
    pub fn cuFuncSetCacheConfig(hfunc: CUfunction, config: CUfunc_cache) -> CUresult;
    pub fn cuFuncSetSharedMemConfig(
        hfunc: CUfunction,
        config: CUsharedconfig,
    ) -> CUresult;
    pub fn cuFuncSetSharedSize(
        hfunc: CUfunction,
        bytes: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuGetErrorName(
        error: CUresult,
        pStr: *mut *const ::core::ffi::c_char,
    ) -> CUresult;
    pub fn cuGetErrorString(
        error: CUresult,
        pStr: *mut *const ::core::ffi::c_char,
    ) -> CUresult;
    pub fn cuGetExportTable(
        ppExportTable: *mut *const ::core::ffi::c_void,
        pExportTableId: *const CUuuid,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub fn cuGetProcAddress(
        symbol: *const ::core::ffi::c_char,
        pfn: *mut *mut ::core::ffi::c_void,
        cudaVersion: ::core::ffi::c_int,
        flags: cuuint64_t,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGetProcAddress_v2(
        symbol: *const ::core::ffi::c_char,
        pfn: *mut *mut ::core::ffi::c_void,
        cudaVersion: ::core::ffi::c_int,
        flags: cuuint64_t,
        symbolStatus: *mut CUdriverProcAddressQueryResult,
    ) -> CUresult;
    #[cfg(
        any(
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
        )
    )]
    pub fn cuGraphAddBatchMemOpNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphAddChildGraphNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        childGraph: CUgraph,
    ) -> CUresult;
    pub fn cuGraphAddDependencies(
        hGraph: CUgraph,
        from: *const CUgraphNode,
        to: *const CUgraphNode,
        numDependencies: usize,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphAddDependencies_v2(
        hGraph: CUgraph,
        from: *const CUgraphNode,
        to: *const CUgraphNode,
        edgeData: *const CUgraphEdgeData,
        numDependencies: usize,
    ) -> CUresult;
    pub fn cuGraphAddEmptyNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
    ) -> CUresult;
    pub fn cuGraphAddEventRecordNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        event: CUevent,
    ) -> CUresult;
    pub fn cuGraphAddEventWaitNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        event: CUevent,
    ) -> CUresult;
    pub fn cuGraphAddExternalSemaphoresSignalNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphAddExternalSemaphoresWaitNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphAddHostNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *const CUDA_HOST_NODE_PARAMS,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub fn cuGraphAddKernelNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphAddKernelNode_v2(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphAddMemAllocNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *mut CUDA_MEM_ALLOC_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphAddMemFreeNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        dptr: CUdeviceptr,
    ) -> CUresult;
    pub fn cuGraphAddMemcpyNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        copyParams: *const CUDA_MEMCPY3D,
        ctx: CUcontext,
    ) -> CUresult;
    pub fn cuGraphAddMemsetNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        memsetParams: *const CUDA_MEMSET_NODE_PARAMS,
        ctx: CUcontext,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphAddNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *mut CUgraphNodeParams,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphAddNode_v2(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        dependencyData: *const CUgraphEdgeData,
        numDependencies: usize,
        nodeParams: *mut CUgraphNodeParams,
    ) -> CUresult;
    #[cfg(
        any(
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
        )
    )]
    pub fn cuGraphBatchMemOpNodeGetParams(
        hNode: CUgraphNode,
        nodeParams_out: *mut CUDA_BATCH_MEM_OP_NODE_PARAMS,
    ) -> CUresult;
    #[cfg(
        any(
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
        )
    )]
    pub fn cuGraphBatchMemOpNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphChildGraphNodeGetGraph(
        hNode: CUgraphNode,
        phGraph: *mut CUgraph,
    ) -> CUresult;
    pub fn cuGraphClone(phGraphClone: *mut CUgraph, originalGraph: CUgraph) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphConditionalHandleCreate(
        pHandle_out: *mut CUgraphConditionalHandle,
        hGraph: CUgraph,
        ctx: CUcontext,
        defaultLaunchValue: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuGraphCreate(phGraph: *mut CUgraph, flags: ::core::ffi::c_uint) -> CUresult;
    pub fn cuGraphDebugDotPrint(
        hGraph: CUgraph,
        path: *const ::core::ffi::c_char,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuGraphDestroy(hGraph: CUgraph) -> CUresult;
    pub fn cuGraphDestroyNode(hNode: CUgraphNode) -> CUresult;
    pub fn cuGraphEventRecordNodeGetEvent(
        hNode: CUgraphNode,
        event_out: *mut CUevent,
    ) -> CUresult;
    pub fn cuGraphEventRecordNodeSetEvent(
        hNode: CUgraphNode,
        event: CUevent,
    ) -> CUresult;
    pub fn cuGraphEventWaitNodeGetEvent(
        hNode: CUgraphNode,
        event_out: *mut CUevent,
    ) -> CUresult;
    pub fn cuGraphEventWaitNodeSetEvent(hNode: CUgraphNode, event: CUevent) -> CUresult;
    #[cfg(
        any(
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
        )
    )]
    pub fn cuGraphExecBatchMemOpNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphExecChildGraphNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        childGraph: CUgraph,
    ) -> CUresult;
    pub fn cuGraphExecDestroy(hGraphExec: CUgraphExec) -> CUresult;
    pub fn cuGraphExecEventRecordNodeSetEvent(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        event: CUevent,
    ) -> CUresult;
    pub fn cuGraphExecEventWaitNodeSetEvent(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        event: CUevent,
    ) -> CUresult;
    pub fn cuGraphExecExternalSemaphoresSignalNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphExecExternalSemaphoresWaitNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphExecGetFlags(
        hGraphExec: CUgraphExec,
        flags: *mut cuuint64_t,
    ) -> CUresult;
    pub fn cuGraphExecHostNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        nodeParams: *const CUDA_HOST_NODE_PARAMS,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub fn cuGraphExecKernelNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphExecKernelNodeSetParams_v2(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphExecMemcpyNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        copyParams: *const CUDA_MEMCPY3D,
        ctx: CUcontext,
    ) -> CUresult;
    pub fn cuGraphExecMemsetNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        memsetParams: *const CUDA_MEMSET_NODE_PARAMS,
        ctx: CUcontext,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphExecNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        nodeParams: *mut CUgraphNodeParams,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub fn cuGraphExecUpdate(
        hGraphExec: CUgraphExec,
        hGraph: CUgraph,
        hErrorNode_out: *mut CUgraphNode,
        updateResult_out: *mut CUgraphExecUpdateResult,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphExecUpdate_v2(
        hGraphExec: CUgraphExec,
        hGraph: CUgraph,
        resultInfo: *mut CUgraphExecUpdateResultInfo,
    ) -> CUresult;
    pub fn cuGraphExternalSemaphoresSignalNodeGetParams(
        hNode: CUgraphNode,
        params_out: *mut CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphExternalSemaphoresSignalNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphExternalSemaphoresWaitNodeGetParams(
        hNode: CUgraphNode,
        params_out: *mut CUDA_EXT_SEM_WAIT_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphExternalSemaphoresWaitNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphGetEdges(
        hGraph: CUgraph,
        from: *mut CUgraphNode,
        to: *mut CUgraphNode,
        numEdges: *mut usize,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphGetEdges_v2(
        hGraph: CUgraph,
        from: *mut CUgraphNode,
        to: *mut CUgraphNode,
        edgeData: *mut CUgraphEdgeData,
        numEdges: *mut usize,
    ) -> CUresult;
    pub fn cuGraphGetNodes(
        hGraph: CUgraph,
        nodes: *mut CUgraphNode,
        numNodes: *mut usize,
    ) -> CUresult;
    pub fn cuGraphGetRootNodes(
        hGraph: CUgraph,
        rootNodes: *mut CUgraphNode,
        numRootNodes: *mut usize,
    ) -> CUresult;
    pub fn cuGraphHostNodeGetParams(
        hNode: CUgraphNode,
        nodeParams: *mut CUDA_HOST_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphHostNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_HOST_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphInstantiateWithFlags(
        phGraphExec: *mut CUgraphExec,
        hGraph: CUgraph,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphInstantiateWithParams(
        phGraphExec: *mut CUgraphExec,
        hGraph: CUgraph,
        instantiateParams: *mut CUDA_GRAPH_INSTANTIATE_PARAMS,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub fn cuGraphInstantiate_v2(
        phGraphExec: *mut CUgraphExec,
        hGraph: CUgraph,
        phErrorNode: *mut CUgraphNode,
        logBuffer: *mut ::core::ffi::c_char,
        bufferSize: usize,
    ) -> CUresult;
    pub fn cuGraphKernelNodeCopyAttributes(
        dst: CUgraphNode,
        src: CUgraphNode,
    ) -> CUresult;
    pub fn cuGraphKernelNodeGetAttribute(
        hNode: CUgraphNode,
        attr: CUkernelNodeAttrID,
        value_out: *mut CUkernelNodeAttrValue,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub fn cuGraphKernelNodeGetParams(
        hNode: CUgraphNode,
        nodeParams: *mut CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphKernelNodeGetParams_v2(
        hNode: CUgraphNode,
        nodeParams: *mut CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphKernelNodeSetAttribute(
        hNode: CUgraphNode,
        attr: CUkernelNodeAttrID,
        value: *const CUkernelNodeAttrValue,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub fn cuGraphKernelNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphKernelNodeSetParams_v2(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphLaunch(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult;
    pub fn cuGraphMemAllocNodeGetParams(
        hNode: CUgraphNode,
        params_out: *mut CUDA_MEM_ALLOC_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphMemFreeNodeGetParams(
        hNode: CUgraphNode,
        dptr_out: *mut CUdeviceptr,
    ) -> CUresult;
    pub fn cuGraphMemcpyNodeGetParams(
        hNode: CUgraphNode,
        nodeParams: *mut CUDA_MEMCPY3D,
    ) -> CUresult;
    pub fn cuGraphMemcpyNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_MEMCPY3D,
    ) -> CUresult;
    pub fn cuGraphMemsetNodeGetParams(
        hNode: CUgraphNode,
        nodeParams: *mut CUDA_MEMSET_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphMemsetNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_MEMSET_NODE_PARAMS,
    ) -> CUresult;
    pub fn cuGraphNodeFindInClone(
        phNode: *mut CUgraphNode,
        hOriginalNode: CUgraphNode,
        hClonedGraph: CUgraph,
    ) -> CUresult;
    pub fn cuGraphNodeGetDependencies(
        hNode: CUgraphNode,
        dependencies: *mut CUgraphNode,
        numDependencies: *mut usize,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphNodeGetDependencies_v2(
        hNode: CUgraphNode,
        dependencies: *mut CUgraphNode,
        edgeData: *mut CUgraphEdgeData,
        numDependencies: *mut usize,
    ) -> CUresult;
    pub fn cuGraphNodeGetDependentNodes(
        hNode: CUgraphNode,
        dependentNodes: *mut CUgraphNode,
        numDependentNodes: *mut usize,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphNodeGetDependentNodes_v2(
        hNode: CUgraphNode,
        dependentNodes: *mut CUgraphNode,
        edgeData: *mut CUgraphEdgeData,
        numDependentNodes: *mut usize,
    ) -> CUresult;
    #[cfg(
        any(
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
        )
    )]
    pub fn cuGraphNodeGetEnabled(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        isEnabled: *mut ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuGraphNodeGetType(
        hNode: CUgraphNode,
        type_: *mut CUgraphNodeType,
    ) -> CUresult;
    #[cfg(
        any(
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
        )
    )]
    pub fn cuGraphNodeSetEnabled(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        isEnabled: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *mut CUgraphNodeParams,
    ) -> CUresult;
    pub fn cuGraphReleaseUserObject(
        graph: CUgraph,
        object: CUuserObject,
        count: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuGraphRemoveDependencies(
        hGraph: CUgraph,
        from: *const CUgraphNode,
        to: *const CUgraphNode,
        numDependencies: usize,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGraphRemoveDependencies_v2(
        hGraph: CUgraph,
        from: *const CUgraphNode,
        to: *const CUgraphNode,
        edgeData: *const CUgraphEdgeData,
        numDependencies: usize,
    ) -> CUresult;
    pub fn cuGraphRetainUserObject(
        graph: CUgraph,
        object: CUuserObject,
        count: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuGraphUpload(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult;
    pub fn cuGraphicsMapResources(
        count: ::core::ffi::c_uint,
        resources: *mut CUgraphicsResource,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuGraphicsResourceGetMappedMipmappedArray(
        pMipmappedArray: *mut CUmipmappedArray,
        resource: CUgraphicsResource,
    ) -> CUresult;
    pub fn cuGraphicsResourceGetMappedPointer_v2(
        pDevPtr: *mut CUdeviceptr,
        pSize: *mut usize,
        resource: CUgraphicsResource,
    ) -> CUresult;
    pub fn cuGraphicsResourceSetMapFlags_v2(
        resource: CUgraphicsResource,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuGraphicsSubResourceGetMappedArray(
        pArray: *mut CUarray,
        resource: CUgraphicsResource,
        arrayIndex: ::core::ffi::c_uint,
        mipLevel: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuGraphicsUnmapResources(
        count: ::core::ffi::c_uint,
        resources: *mut CUgraphicsResource,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuGraphicsUnregisterResource(resource: CUgraphicsResource) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGreenCtxCreate(
        phCtx: *mut CUgreenCtx,
        desc: CUdevResourceDesc,
        dev: CUdevice,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGreenCtxDestroy(hCtx: CUgreenCtx) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGreenCtxGetDevResource(
        hCtx: CUgreenCtx,
        resource: *mut CUdevResource,
        type_: CUdevResourceType,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGreenCtxRecordEvent(hCtx: CUgreenCtx, hEvent: CUevent) -> CUresult;
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub fn cuGreenCtxStreamCreate(
        phStream: *mut CUstream,
        greenCtx: CUgreenCtx,
        flags: ::core::ffi::c_uint,
        priority: ::core::ffi::c_int,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuGreenCtxWaitEvent(hCtx: CUgreenCtx, hEvent: CUevent) -> CUresult;
    pub fn cuImportExternalMemory(
        extMem_out: *mut CUexternalMemory,
        memHandleDesc: *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
    ) -> CUresult;
    pub fn cuImportExternalSemaphore(
        extSem_out: *mut CUexternalSemaphore,
        semHandleDesc: *const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC,
    ) -> CUresult;
    pub fn cuInit(Flags: ::core::ffi::c_uint) -> CUresult;
    pub fn cuIpcCloseMemHandle(dptr: CUdeviceptr) -> CUresult;
    pub fn cuIpcGetEventHandle(
        pHandle: *mut CUipcEventHandle,
        event: CUevent,
    ) -> CUresult;
    pub fn cuIpcGetMemHandle(
        pHandle: *mut CUipcMemHandle,
        dptr: CUdeviceptr,
    ) -> CUresult;
    pub fn cuIpcOpenEventHandle(
        phEvent: *mut CUevent,
        handle: CUipcEventHandle,
    ) -> CUresult;
    pub fn cuIpcOpenMemHandle_v2(
        pdptr: *mut CUdeviceptr,
        handle: CUipcMemHandle,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuKernelGetAttribute(
        pi: *mut ::core::ffi::c_int,
        attrib: CUfunction_attribute,
        kernel: CUkernel,
        dev: CUdevice,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuKernelGetFunction(pFunc: *mut CUfunction, kernel: CUkernel) -> CUresult;
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub fn cuKernelGetLibrary(pLib: *mut CUlibrary, kernel: CUkernel) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuKernelGetName(
        name: *mut *const ::core::ffi::c_char,
        hfunc: CUkernel,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuKernelGetParamInfo(
        kernel: CUkernel,
        paramIndex: usize,
        paramOffset: *mut usize,
        paramSize: *mut usize,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuKernelSetAttribute(
        attrib: CUfunction_attribute,
        val: ::core::ffi::c_int,
        kernel: CUkernel,
        dev: CUdevice,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuKernelSetCacheConfig(
        kernel: CUkernel,
        config: CUfunc_cache,
        dev: CUdevice,
    ) -> CUresult;
    pub fn cuLaunch(f: CUfunction) -> CUresult;
    pub fn cuLaunchCooperativeKernel(
        f: CUfunction,
        gridDimX: ::core::ffi::c_uint,
        gridDimY: ::core::ffi::c_uint,
        gridDimZ: ::core::ffi::c_uint,
        blockDimX: ::core::ffi::c_uint,
        blockDimY: ::core::ffi::c_uint,
        blockDimZ: ::core::ffi::c_uint,
        sharedMemBytes: ::core::ffi::c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut ::core::ffi::c_void,
    ) -> CUresult;
    pub fn cuLaunchCooperativeKernelMultiDevice(
        launchParamsList: *mut CUDA_LAUNCH_PARAMS,
        numDevices: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuLaunchGrid(
        f: CUfunction,
        grid_width: ::core::ffi::c_int,
        grid_height: ::core::ffi::c_int,
    ) -> CUresult;
    pub fn cuLaunchGridAsync(
        f: CUfunction,
        grid_width: ::core::ffi::c_int,
        grid_height: ::core::ffi::c_int,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuLaunchHostFunc(
        hStream: CUstream,
        fn_: CUhostFn,
        userData: *mut ::core::ffi::c_void,
    ) -> CUresult;
    pub fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: ::core::ffi::c_uint,
        gridDimY: ::core::ffi::c_uint,
        gridDimZ: ::core::ffi::c_uint,
        blockDimX: ::core::ffi::c_uint,
        blockDimY: ::core::ffi::c_uint,
        blockDimZ: ::core::ffi::c_uint,
        sharedMemBytes: ::core::ffi::c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut ::core::ffi::c_void,
        extra: *mut *mut ::core::ffi::c_void,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuLaunchKernelEx(
        config: *const CUlaunchConfig,
        f: CUfunction,
        kernelParams: *mut *mut ::core::ffi::c_void,
        extra: *mut *mut ::core::ffi::c_void,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuLibraryEnumerateKernels(
        kernels: *mut CUkernel,
        numKernels: ::core::ffi::c_uint,
        lib: CUlibrary,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuLibraryGetGlobal(
        dptr: *mut CUdeviceptr,
        bytes: *mut usize,
        library: CUlibrary,
        name: *const ::core::ffi::c_char,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuLibraryGetKernel(
        pKernel: *mut CUkernel,
        library: CUlibrary,
        name: *const ::core::ffi::c_char,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuLibraryGetKernelCount(
        count: *mut ::core::ffi::c_uint,
        lib: CUlibrary,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuLibraryGetManaged(
        dptr: *mut CUdeviceptr,
        bytes: *mut usize,
        library: CUlibrary,
        name: *const ::core::ffi::c_char,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuLibraryGetModule(pMod: *mut CUmodule, library: CUlibrary) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuLibraryGetUnifiedFunction(
        fptr: *mut *mut ::core::ffi::c_void,
        library: CUlibrary,
        symbol: *const ::core::ffi::c_char,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuLibraryLoadData(
        library: *mut CUlibrary,
        code: *const ::core::ffi::c_void,
        jitOptions: *mut CUjit_option,
        jitOptionsValues: *mut *mut ::core::ffi::c_void,
        numJitOptions: ::core::ffi::c_uint,
        libraryOptions: *mut CUlibraryOption,
        libraryOptionValues: *mut *mut ::core::ffi::c_void,
        numLibraryOptions: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuLibraryLoadFromFile(
        library: *mut CUlibrary,
        fileName: *const ::core::ffi::c_char,
        jitOptions: *mut CUjit_option,
        jitOptionsValues: *mut *mut ::core::ffi::c_void,
        numJitOptions: ::core::ffi::c_uint,
        libraryOptions: *mut CUlibraryOption,
        libraryOptionValues: *mut *mut ::core::ffi::c_void,
        numLibraryOptions: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuLibraryUnload(library: CUlibrary) -> CUresult;
    pub fn cuLinkAddData_v2(
        state: CUlinkState,
        type_: CUjitInputType,
        data: *mut ::core::ffi::c_void,
        size: usize,
        name: *const ::core::ffi::c_char,
        numOptions: ::core::ffi::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::core::ffi::c_void,
    ) -> CUresult;
    pub fn cuLinkAddFile_v2(
        state: CUlinkState,
        type_: CUjitInputType,
        path: *const ::core::ffi::c_char,
        numOptions: ::core::ffi::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::core::ffi::c_void,
    ) -> CUresult;
    pub fn cuLinkComplete(
        state: CUlinkState,
        cubinOut: *mut *mut ::core::ffi::c_void,
        sizeOut: *mut usize,
    ) -> CUresult;
    pub fn cuLinkCreate_v2(
        numOptions: ::core::ffi::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::core::ffi::c_void,
        stateOut: *mut CUlinkState,
    ) -> CUresult;
    pub fn cuLinkDestroy(state: CUlinkState) -> CUresult;
    pub fn cuMemAddressFree(ptr: CUdeviceptr, size: usize) -> CUresult;
    pub fn cuMemAddressReserve(
        ptr: *mut CUdeviceptr,
        size: usize,
        alignment: usize,
        addr: CUdeviceptr,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult;
    pub fn cuMemAdvise(
        devPtr: CUdeviceptr,
        count: usize,
        advice: CUmem_advise,
        device: CUdevice,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuMemAdvise_v2(
        devPtr: CUdeviceptr,
        count: usize,
        advice: CUmem_advise,
        location: CUmemLocation,
    ) -> CUresult;
    pub fn cuMemAllocAsync(
        dptr: *mut CUdeviceptr,
        bytesize: usize,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemAllocFromPoolAsync(
        dptr: *mut CUdeviceptr,
        bytesize: usize,
        pool: CUmemoryPool,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemAllocHost_v2(
        pp: *mut *mut ::core::ffi::c_void,
        bytesize: usize,
    ) -> CUresult;
    pub fn cuMemAllocManaged(
        dptr: *mut CUdeviceptr,
        bytesize: usize,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuMemAllocPitch_v2(
        dptr: *mut CUdeviceptr,
        pPitch: *mut usize,
        WidthInBytes: usize,
        Height: usize,
        ElementSizeBytes: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult;
    #[cfg(any(feature = "cuda-12080"))]
    pub fn cuMemBatchDecompressAsync(
        paramsArray: *mut CUmemDecompressParams,
        count: usize,
        flags: ::core::ffi::c_uint,
        errorIndex: *mut usize,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuMemCreate(
        handle: *mut CUmemGenericAllocationHandle,
        size: usize,
        prop: *const CUmemAllocationProp,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult;
    pub fn cuMemExportToShareableHandle(
        shareableHandle: *mut ::core::ffi::c_void,
        handle: CUmemGenericAllocationHandle,
        handleType: CUmemAllocationHandleType,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult;
    pub fn cuMemFreeAsync(dptr: CUdeviceptr, hStream: CUstream) -> CUresult;
    pub fn cuMemFreeHost(p: *mut ::core::ffi::c_void) -> CUresult;
    pub fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult;
    pub fn cuMemGetAccess(
        flags: *mut ::core::ffi::c_ulonglong,
        location: *const CUmemLocation,
        ptr: CUdeviceptr,
    ) -> CUresult;
    pub fn cuMemGetAddressRange_v2(
        pbase: *mut CUdeviceptr,
        psize: *mut usize,
        dptr: CUdeviceptr,
    ) -> CUresult;
    pub fn cuMemGetAllocationGranularity(
        granularity: *mut usize,
        prop: *const CUmemAllocationProp,
        option: CUmemAllocationGranularity_flags,
    ) -> CUresult;
    pub fn cuMemGetAllocationPropertiesFromHandle(
        prop: *mut CUmemAllocationProp,
        handle: CUmemGenericAllocationHandle,
    ) -> CUresult;
    #[cfg(
        any(
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
        )
    )]
    pub fn cuMemGetHandleForAddressRange(
        handle: *mut ::core::ffi::c_void,
        dptr: CUdeviceptr,
        size: usize,
        handleType: CUmemRangeHandleType,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult;
    pub fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult;
    pub fn cuMemHostAlloc(
        pp: *mut *mut ::core::ffi::c_void,
        bytesize: usize,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuMemHostGetDevicePointer_v2(
        pdptr: *mut CUdeviceptr,
        p: *mut ::core::ffi::c_void,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuMemHostGetFlags(
        pFlags: *mut ::core::ffi::c_uint,
        p: *mut ::core::ffi::c_void,
    ) -> CUresult;
    pub fn cuMemHostRegister_v2(
        p: *mut ::core::ffi::c_void,
        bytesize: usize,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuMemHostUnregister(p: *mut ::core::ffi::c_void) -> CUresult;
    pub fn cuMemImportFromShareableHandle(
        handle: *mut CUmemGenericAllocationHandle,
        osHandle: *mut ::core::ffi::c_void,
        shHandleType: CUmemAllocationHandleType,
    ) -> CUresult;
    pub fn cuMemMap(
        ptr: CUdeviceptr,
        size: usize,
        offset: usize,
        handle: CUmemGenericAllocationHandle,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult;
    pub fn cuMemMapArrayAsync(
        mapInfoList: *mut CUarrayMapInfo,
        count: ::core::ffi::c_uint,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemPoolCreate(
        pool: *mut CUmemoryPool,
        poolProps: *const CUmemPoolProps,
    ) -> CUresult;
    pub fn cuMemPoolDestroy(pool: CUmemoryPool) -> CUresult;
    pub fn cuMemPoolExportPointer(
        shareData_out: *mut CUmemPoolPtrExportData,
        ptr: CUdeviceptr,
    ) -> CUresult;
    pub fn cuMemPoolExportToShareableHandle(
        handle_out: *mut ::core::ffi::c_void,
        pool: CUmemoryPool,
        handleType: CUmemAllocationHandleType,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult;
    pub fn cuMemPoolGetAccess(
        flags: *mut CUmemAccess_flags,
        memPool: CUmemoryPool,
        location: *mut CUmemLocation,
    ) -> CUresult;
    pub fn cuMemPoolGetAttribute(
        pool: CUmemoryPool,
        attr: CUmemPool_attribute,
        value: *mut ::core::ffi::c_void,
    ) -> CUresult;
    pub fn cuMemPoolImportFromShareableHandle(
        pool_out: *mut CUmemoryPool,
        handle: *mut ::core::ffi::c_void,
        handleType: CUmemAllocationHandleType,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult;
    pub fn cuMemPoolImportPointer(
        ptr_out: *mut CUdeviceptr,
        pool: CUmemoryPool,
        shareData: *mut CUmemPoolPtrExportData,
    ) -> CUresult;
    pub fn cuMemPoolSetAccess(
        pool: CUmemoryPool,
        map: *const CUmemAccessDesc,
        count: usize,
    ) -> CUresult;
    pub fn cuMemPoolSetAttribute(
        pool: CUmemoryPool,
        attr: CUmemPool_attribute,
        value: *mut ::core::ffi::c_void,
    ) -> CUresult;
    pub fn cuMemPoolTrimTo(pool: CUmemoryPool, minBytesToKeep: usize) -> CUresult;
    pub fn cuMemPrefetchAsync(
        devPtr: CUdeviceptr,
        count: usize,
        dstDevice: CUdevice,
        hStream: CUstream,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuMemPrefetchAsync_v2(
        devPtr: CUdeviceptr,
        count: usize,
        location: CUmemLocation,
        flags: ::core::ffi::c_uint,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemRangeGetAttribute(
        data: *mut ::core::ffi::c_void,
        dataSize: usize,
        attribute: CUmem_range_attribute,
        devPtr: CUdeviceptr,
        count: usize,
    ) -> CUresult;
    pub fn cuMemRangeGetAttributes(
        data: *mut *mut ::core::ffi::c_void,
        dataSizes: *mut usize,
        attributes: *mut CUmem_range_attribute,
        numAttributes: usize,
        devPtr: CUdeviceptr,
        count: usize,
    ) -> CUresult;
    pub fn cuMemRelease(handle: CUmemGenericAllocationHandle) -> CUresult;
    pub fn cuMemRetainAllocationHandle(
        handle: *mut CUmemGenericAllocationHandle,
        addr: *mut ::core::ffi::c_void,
    ) -> CUresult;
    pub fn cuMemSetAccess(
        ptr: CUdeviceptr,
        size: usize,
        desc: *const CUmemAccessDesc,
        count: usize,
    ) -> CUresult;
    pub fn cuMemUnmap(ptr: CUdeviceptr, size: usize) -> CUresult;
    pub fn cuMemcpy(dst: CUdeviceptr, src: CUdeviceptr, ByteCount: usize) -> CUresult;
    pub fn cuMemcpy2DAsync_v2(
        pCopy: *const CUDA_MEMCPY2D,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpy2DUnaligned_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult;
    pub fn cuMemcpy2D_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult;
    pub fn cuMemcpy3DAsync_v2(
        pCopy: *const CUDA_MEMCPY3D,
        hStream: CUstream,
    ) -> CUresult;
    #[cfg(any(feature = "cuda-12080"))]
    pub fn cuMemcpy3DBatchAsync(
        numOps: usize,
        opList: *mut CUDA_MEMCPY3D_BATCH_OP,
        failIdx: *mut usize,
        flags: ::core::ffi::c_ulonglong,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpy3DPeer(pCopy: *const CUDA_MEMCPY3D_PEER) -> CUresult;
    pub fn cuMemcpy3DPeerAsync(
        pCopy: *const CUDA_MEMCPY3D_PEER,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpy3D_v2(pCopy: *const CUDA_MEMCPY3D) -> CUresult;
    pub fn cuMemcpyAsync(
        dst: CUdeviceptr,
        src: CUdeviceptr,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyAtoA_v2(
        dstArray: CUarray,
        dstOffset: usize,
        srcArray: CUarray,
        srcOffset: usize,
        ByteCount: usize,
    ) -> CUresult;
    pub fn cuMemcpyAtoD_v2(
        dstDevice: CUdeviceptr,
        srcArray: CUarray,
        srcOffset: usize,
        ByteCount: usize,
    ) -> CUresult;
    pub fn cuMemcpyAtoHAsync_v2(
        dstHost: *mut ::core::ffi::c_void,
        srcArray: CUarray,
        srcOffset: usize,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyAtoH_v2(
        dstHost: *mut ::core::ffi::c_void,
        srcArray: CUarray,
        srcOffset: usize,
        ByteCount: usize,
    ) -> CUresult;
    #[cfg(any(feature = "cuda-12080"))]
    pub fn cuMemcpyBatchAsync(
        dsts: *mut CUdeviceptr,
        srcs: *mut CUdeviceptr,
        sizes: *mut usize,
        count: usize,
        attrs: *mut CUmemcpyAttributes,
        attrsIdxs: *mut usize,
        numAttrs: usize,
        failIdx: *mut usize,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyDtoA_v2(
        dstArray: CUarray,
        dstOffset: usize,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
    ) -> CUresult;
    pub fn cuMemcpyDtoDAsync_v2(
        dstDevice: CUdeviceptr,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyDtoD_v2(
        dstDevice: CUdeviceptr,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
    ) -> CUresult;
    pub fn cuMemcpyDtoHAsync_v2(
        dstHost: *mut ::core::ffi::c_void,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyDtoH_v2(
        dstHost: *mut ::core::ffi::c_void,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
    ) -> CUresult;
    pub fn cuMemcpyHtoAAsync_v2(
        dstArray: CUarray,
        dstOffset: usize,
        srcHost: *const ::core::ffi::c_void,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyHtoA_v2(
        dstArray: CUarray,
        dstOffset: usize,
        srcHost: *const ::core::ffi::c_void,
        ByteCount: usize,
    ) -> CUresult;
    pub fn cuMemcpyHtoDAsync_v2(
        dstDevice: CUdeviceptr,
        srcHost: *const ::core::ffi::c_void,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemcpyHtoD_v2(
        dstDevice: CUdeviceptr,
        srcHost: *const ::core::ffi::c_void,
        ByteCount: usize,
    ) -> CUresult;
    pub fn cuMemcpyPeer(
        dstDevice: CUdeviceptr,
        dstContext: CUcontext,
        srcDevice: CUdeviceptr,
        srcContext: CUcontext,
        ByteCount: usize,
    ) -> CUresult;
    pub fn cuMemcpyPeerAsync(
        dstDevice: CUdeviceptr,
        dstContext: CUcontext,
        srcDevice: CUdeviceptr,
        srcContext: CUcontext,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemsetD16Async(
        dstDevice: CUdeviceptr,
        us: ::core::ffi::c_ushort,
        N: usize,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemsetD16_v2(
        dstDevice: CUdeviceptr,
        us: ::core::ffi::c_ushort,
        N: usize,
    ) -> CUresult;
    pub fn cuMemsetD2D16Async(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        us: ::core::ffi::c_ushort,
        Width: usize,
        Height: usize,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemsetD2D16_v2(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        us: ::core::ffi::c_ushort,
        Width: usize,
        Height: usize,
    ) -> CUresult;
    pub fn cuMemsetD2D32Async(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        ui: ::core::ffi::c_uint,
        Width: usize,
        Height: usize,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemsetD2D32_v2(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        ui: ::core::ffi::c_uint,
        Width: usize,
        Height: usize,
    ) -> CUresult;
    pub fn cuMemsetD2D8Async(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        uc: ::core::ffi::c_uchar,
        Width: usize,
        Height: usize,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemsetD2D8_v2(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        uc: ::core::ffi::c_uchar,
        Width: usize,
        Height: usize,
    ) -> CUresult;
    pub fn cuMemsetD32Async(
        dstDevice: CUdeviceptr,
        ui: ::core::ffi::c_uint,
        N: usize,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemsetD32_v2(
        dstDevice: CUdeviceptr,
        ui: ::core::ffi::c_uint,
        N: usize,
    ) -> CUresult;
    pub fn cuMemsetD8Async(
        dstDevice: CUdeviceptr,
        uc: ::core::ffi::c_uchar,
        N: usize,
        hStream: CUstream,
    ) -> CUresult;
    pub fn cuMemsetD8_v2(
        dstDevice: CUdeviceptr,
        uc: ::core::ffi::c_uchar,
        N: usize,
    ) -> CUresult;
    pub fn cuMipmappedArrayCreate(
        pHandle: *mut CUmipmappedArray,
        pMipmappedArrayDesc: *const CUDA_ARRAY3D_DESCRIPTOR,
        numMipmapLevels: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuMipmappedArrayDestroy(hMipmappedArray: CUmipmappedArray) -> CUresult;
    pub fn cuMipmappedArrayGetLevel(
        pLevelArray: *mut CUarray,
        hMipmappedArray: CUmipmappedArray,
        level: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
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
        )
    )]
    pub fn cuMipmappedArrayGetMemoryRequirements(
        memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS,
        mipmap: CUmipmappedArray,
        device: CUdevice,
    ) -> CUresult;
    pub fn cuMipmappedArrayGetSparseProperties(
        sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES,
        mipmap: CUmipmappedArray,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuModuleEnumerateFunctions(
        functions: *mut CUfunction,
        numFunctions: ::core::ffi::c_uint,
        mod_: CUmodule,
    ) -> CUresult;
    pub fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const ::core::ffi::c_char,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuModuleGetFunctionCount(
        count: *mut ::core::ffi::c_uint,
        mod_: CUmodule,
    ) -> CUresult;
    pub fn cuModuleGetGlobal_v2(
        dptr: *mut CUdeviceptr,
        bytes: *mut usize,
        hmod: CUmodule,
        name: *const ::core::ffi::c_char,
    ) -> CUresult;
    #[cfg(
        any(
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
        )
    )]
    pub fn cuModuleGetLoadingMode(mode: *mut CUmoduleLoadingMode) -> CUresult;
    pub fn cuModuleGetSurfRef(
        pSurfRef: *mut CUsurfref,
        hmod: CUmodule,
        name: *const ::core::ffi::c_char,
    ) -> CUresult;
    pub fn cuModuleGetTexRef(
        pTexRef: *mut CUtexref,
        hmod: CUmodule,
        name: *const ::core::ffi::c_char,
    ) -> CUresult;
    pub fn cuModuleLoad(
        module: *mut CUmodule,
        fname: *const ::core::ffi::c_char,
    ) -> CUresult;
    pub fn cuModuleLoadData(
        module: *mut CUmodule,
        image: *const ::core::ffi::c_void,
    ) -> CUresult;
    pub fn cuModuleLoadDataEx(
        module: *mut CUmodule,
        image: *const ::core::ffi::c_void,
        numOptions: ::core::ffi::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::core::ffi::c_void,
    ) -> CUresult;
    pub fn cuModuleLoadFatBinary(
        module: *mut CUmodule,
        fatCubin: *const ::core::ffi::c_void,
    ) -> CUresult;
    pub fn cuModuleUnload(hmod: CUmodule) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuMulticastAddDevice(
        mcHandle: CUmemGenericAllocationHandle,
        dev: CUdevice,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuMulticastBindAddr(
        mcHandle: CUmemGenericAllocationHandle,
        mcOffset: usize,
        memptr: CUdeviceptr,
        size: usize,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuMulticastBindMem(
        mcHandle: CUmemGenericAllocationHandle,
        mcOffset: usize,
        memHandle: CUmemGenericAllocationHandle,
        memOffset: usize,
        size: usize,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuMulticastCreate(
        mcHandle: *mut CUmemGenericAllocationHandle,
        prop: *const CUmulticastObjectProp,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuMulticastGetGranularity(
        granularity: *mut usize,
        prop: *const CUmulticastObjectProp,
        option: CUmulticastGranularity_flags,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuMulticastUnbind(
        mcHandle: CUmemGenericAllocationHandle,
        dev: CUdevice,
        mcOffset: usize,
        size: usize,
    ) -> CUresult;
    pub fn cuOccupancyAvailableDynamicSMemPerBlock(
        dynamicSmemSize: *mut usize,
        func: CUfunction,
        numBlocks: ::core::ffi::c_int,
        blockSize: ::core::ffi::c_int,
    ) -> CUresult;
    pub fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks: *mut ::core::ffi::c_int,
        func: CUfunction,
        blockSize: ::core::ffi::c_int,
        dynamicSMemSize: usize,
    ) -> CUresult;
    pub fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks: *mut ::core::ffi::c_int,
        func: CUfunction,
        blockSize: ::core::ffi::c_int,
        dynamicSMemSize: usize,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuOccupancyMaxActiveClusters(
        numClusters: *mut ::core::ffi::c_int,
        func: CUfunction,
        config: *const CUlaunchConfig,
    ) -> CUresult;
    pub fn cuOccupancyMaxPotentialBlockSize(
        minGridSize: *mut ::core::ffi::c_int,
        blockSize: *mut ::core::ffi::c_int,
        func: CUfunction,
        blockSizeToDynamicSMemSize: CUoccupancyB2DSize,
        dynamicSMemSize: usize,
        blockSizeLimit: ::core::ffi::c_int,
    ) -> CUresult;
    pub fn cuOccupancyMaxPotentialBlockSizeWithFlags(
        minGridSize: *mut ::core::ffi::c_int,
        blockSize: *mut ::core::ffi::c_int,
        func: CUfunction,
        blockSizeToDynamicSMemSize: CUoccupancyB2DSize,
        dynamicSMemSize: usize,
        blockSizeLimit: ::core::ffi::c_int,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuOccupancyMaxPotentialClusterSize(
        clusterSize: *mut ::core::ffi::c_int,
        func: CUfunction,
        config: *const CUlaunchConfig,
    ) -> CUresult;
    pub fn cuParamSetSize(hfunc: CUfunction, numbytes: ::core::ffi::c_uint) -> CUresult;
    pub fn cuParamSetTexRef(
        hfunc: CUfunction,
        texunit: ::core::ffi::c_int,
        hTexRef: CUtexref,
    ) -> CUresult;
    pub fn cuParamSetf(
        hfunc: CUfunction,
        offset: ::core::ffi::c_int,
        value: f32,
    ) -> CUresult;
    pub fn cuParamSeti(
        hfunc: CUfunction,
        offset: ::core::ffi::c_int,
        value: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuParamSetv(
        hfunc: CUfunction,
        offset: ::core::ffi::c_int,
        ptr: *mut ::core::ffi::c_void,
        numbytes: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuPointerGetAttribute(
        data: *mut ::core::ffi::c_void,
        attribute: CUpointer_attribute,
        ptr: CUdeviceptr,
    ) -> CUresult;
    pub fn cuPointerGetAttributes(
        numAttributes: ::core::ffi::c_uint,
        attributes: *mut CUpointer_attribute,
        data: *mut *mut ::core::ffi::c_void,
        ptr: CUdeviceptr,
    ) -> CUresult;
    pub fn cuPointerSetAttribute(
        value: *const ::core::ffi::c_void,
        attribute: CUpointer_attribute,
        ptr: CUdeviceptr,
    ) -> CUresult;
    pub fn cuProfilerInitialize(
        configFile: *const ::core::ffi::c_char,
        outputFile: *const ::core::ffi::c_char,
        outputMode: CUoutput_mode,
    ) -> CUresult;
    pub fn cuProfilerStart() -> CUresult;
    pub fn cuProfilerStop() -> CUresult;
    pub fn cuSignalExternalSemaphoresAsync(
        extSemArray: *const CUexternalSemaphore,
        paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
        numExtSems: ::core::ffi::c_uint,
        stream: CUstream,
    ) -> CUresult;
    pub fn cuStreamAddCallback(
        hStream: CUstream,
        callback: CUstreamCallback,
        userData: *mut ::core::ffi::c_void,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuStreamAttachMemAsync(
        hStream: CUstream,
        dptr: CUdeviceptr,
        length: usize,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub fn cuStreamBatchMemOp(
        stream: CUstream,
        count: ::core::ffi::c_uint,
        paramArray: *mut CUstreamBatchMemOpParams,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
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
        )
    )]
    pub fn cuStreamBatchMemOp_v2(
        stream: CUstream,
        count: ::core::ffi::c_uint,
        paramArray: *mut CUstreamBatchMemOpParams,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuStreamBeginCaptureToGraph(
        hStream: CUstream,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        dependencyData: *const CUgraphEdgeData,
        numDependencies: usize,
        mode: CUstreamCaptureMode,
    ) -> CUresult;
    pub fn cuStreamBeginCapture_v2(
        hStream: CUstream,
        mode: CUstreamCaptureMode,
    ) -> CUresult;
    pub fn cuStreamCopyAttributes(dst: CUstream, src: CUstream) -> CUresult;
    pub fn cuStreamCreate(
        phStream: *mut CUstream,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuStreamCreateWithPriority(
        phStream: *mut CUstream,
        flags: ::core::ffi::c_uint,
        priority: ::core::ffi::c_int,
    ) -> CUresult;
    pub fn cuStreamDestroy_v2(hStream: CUstream) -> CUresult;
    pub fn cuStreamEndCapture(hStream: CUstream, phGraph: *mut CUgraph) -> CUresult;
    pub fn cuStreamGetAttribute(
        hStream: CUstream,
        attr: CUstreamAttrID,
        value_out: *mut CUstreamAttrValue,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub fn cuStreamGetCaptureInfo(
        hStream: CUstream,
        captureStatus_out: *mut CUstreamCaptureStatus,
        id_out: *mut cuuint64_t,
    ) -> CUresult;
    pub fn cuStreamGetCaptureInfo_v2(
        hStream: CUstream,
        captureStatus_out: *mut CUstreamCaptureStatus,
        id_out: *mut cuuint64_t,
        graph_out: *mut CUgraph,
        dependencies_out: *mut *const CUgraphNode,
        numDependencies_out: *mut usize,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuStreamGetCaptureInfo_v3(
        hStream: CUstream,
        captureStatus_out: *mut CUstreamCaptureStatus,
        id_out: *mut cuuint64_t,
        graph_out: *mut CUgraph,
        dependencies_out: *mut *const CUgraphNode,
        edgeData_out: *mut *const CUgraphEdgeData,
        numDependencies_out: *mut usize,
    ) -> CUresult;
    pub fn cuStreamGetCtx(hStream: CUstream, pctx: *mut CUcontext) -> CUresult;
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub fn cuStreamGetCtx_v2(
        hStream: CUstream,
        pCtx: *mut CUcontext,
        pGreenCtx: *mut CUgreenCtx,
    ) -> CUresult;
    #[cfg(any(feature = "cuda-12080"))]
    pub fn cuStreamGetDevice(hStream: CUstream, device: *mut CUdevice) -> CUresult;
    pub fn cuStreamGetFlags(
        hStream: CUstream,
        flags: *mut ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuStreamGetGreenCtx(hStream: CUstream, phCtx: *mut CUgreenCtx) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuStreamGetId(
        hStream: CUstream,
        streamId: *mut ::core::ffi::c_ulonglong,
    ) -> CUresult;
    pub fn cuStreamGetPriority(
        hStream: CUstream,
        priority: *mut ::core::ffi::c_int,
    ) -> CUresult;
    pub fn cuStreamIsCapturing(
        hStream: CUstream,
        captureStatus: *mut CUstreamCaptureStatus,
    ) -> CUresult;
    pub fn cuStreamQuery(hStream: CUstream) -> CUresult;
    pub fn cuStreamSetAttribute(
        hStream: CUstream,
        attr: CUstreamAttrID,
        value: *const CUstreamAttrValue,
    ) -> CUresult;
    pub fn cuStreamSynchronize(hStream: CUstream) -> CUresult;
    pub fn cuStreamUpdateCaptureDependencies(
        hStream: CUstream,
        dependencies: *mut CUgraphNode,
        numDependencies: usize,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuStreamUpdateCaptureDependencies_v2(
        hStream: CUstream,
        dependencies: *mut CUgraphNode,
        dependencyData: *const CUgraphEdgeData,
        numDependencies: usize,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuStreamWaitEvent(
        hStream: CUstream,
        hEvent: CUevent,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub fn cuStreamWaitValue32(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint32_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
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
        )
    )]
    pub fn cuStreamWaitValue32_v2(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint32_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub fn cuStreamWaitValue64(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint64_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
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
        )
    )]
    pub fn cuStreamWaitValue64_v2(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint64_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub fn cuStreamWriteValue32(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint32_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
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
        )
    )]
    pub fn cuStreamWriteValue32_v2(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint32_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub fn cuStreamWriteValue64(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint64_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
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
        )
    )]
    pub fn cuStreamWriteValue64_v2(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint64_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuSurfObjectCreate(
        pSurfObject: *mut CUsurfObject,
        pResDesc: *const CUDA_RESOURCE_DESC,
    ) -> CUresult;
    pub fn cuSurfObjectDestroy(surfObject: CUsurfObject) -> CUresult;
    pub fn cuSurfObjectGetResourceDesc(
        pResDesc: *mut CUDA_RESOURCE_DESC,
        surfObject: CUsurfObject,
    ) -> CUresult;
    pub fn cuSurfRefGetArray(phArray: *mut CUarray, hSurfRef: CUsurfref) -> CUresult;
    pub fn cuSurfRefSetArray(
        hSurfRef: CUsurfref,
        hArray: CUarray,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuTensorMapEncodeIm2col(
        tensorMap: *mut CUtensorMap,
        tensorDataType: CUtensorMapDataType,
        tensorRank: cuuint32_t,
        globalAddress: *mut ::core::ffi::c_void,
        globalDim: *const cuuint64_t,
        globalStrides: *const cuuint64_t,
        pixelBoxLowerCorner: *const ::core::ffi::c_int,
        pixelBoxUpperCorner: *const ::core::ffi::c_int,
        channelsPerPixel: cuuint32_t,
        pixelsPerColumn: cuuint32_t,
        elementStrides: *const cuuint32_t,
        interleave: CUtensorMapInterleave,
        swizzle: CUtensorMapSwizzle,
        l2Promotion: CUtensorMapL2promotion,
        oobFill: CUtensorMapFloatOOBfill,
    ) -> CUresult;
    #[cfg(any(feature = "cuda-12080"))]
    pub fn cuTensorMapEncodeIm2colWide(
        tensorMap: *mut CUtensorMap,
        tensorDataType: CUtensorMapDataType,
        tensorRank: cuuint32_t,
        globalAddress: *mut ::core::ffi::c_void,
        globalDim: *const cuuint64_t,
        globalStrides: *const cuuint64_t,
        pixelBoxLowerCornerWidth: ::core::ffi::c_int,
        pixelBoxUpperCornerWidth: ::core::ffi::c_int,
        channelsPerPixel: cuuint32_t,
        pixelsPerColumn: cuuint32_t,
        elementStrides: *const cuuint32_t,
        interleave: CUtensorMapInterleave,
        mode: CUtensorMapIm2ColWideMode,
        swizzle: CUtensorMapSwizzle,
        l2Promotion: CUtensorMapL2promotion,
        oobFill: CUtensorMapFloatOOBfill,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuTensorMapEncodeTiled(
        tensorMap: *mut CUtensorMap,
        tensorDataType: CUtensorMapDataType,
        tensorRank: cuuint32_t,
        globalAddress: *mut ::core::ffi::c_void,
        globalDim: *const cuuint64_t,
        globalStrides: *const cuuint64_t,
        boxDim: *const cuuint32_t,
        elementStrides: *const cuuint32_t,
        interleave: CUtensorMapInterleave,
        swizzle: CUtensorMapSwizzle,
        l2Promotion: CUtensorMapL2promotion,
        oobFill: CUtensorMapFloatOOBfill,
    ) -> CUresult;
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub fn cuTensorMapReplaceAddress(
        tensorMap: *mut CUtensorMap,
        globalAddress: *mut ::core::ffi::c_void,
    ) -> CUresult;
    pub fn cuTexObjectCreate(
        pTexObject: *mut CUtexObject,
        pResDesc: *const CUDA_RESOURCE_DESC,
        pTexDesc: *const CUDA_TEXTURE_DESC,
        pResViewDesc: *const CUDA_RESOURCE_VIEW_DESC,
    ) -> CUresult;
    pub fn cuTexObjectDestroy(texObject: CUtexObject) -> CUresult;
    pub fn cuTexObjectGetResourceDesc(
        pResDesc: *mut CUDA_RESOURCE_DESC,
        texObject: CUtexObject,
    ) -> CUresult;
    pub fn cuTexObjectGetResourceViewDesc(
        pResViewDesc: *mut CUDA_RESOURCE_VIEW_DESC,
        texObject: CUtexObject,
    ) -> CUresult;
    pub fn cuTexObjectGetTextureDesc(
        pTexDesc: *mut CUDA_TEXTURE_DESC,
        texObject: CUtexObject,
    ) -> CUresult;
    pub fn cuTexRefCreate(pTexRef: *mut CUtexref) -> CUresult;
    pub fn cuTexRefDestroy(hTexRef: CUtexref) -> CUresult;
    pub fn cuTexRefGetAddressMode(
        pam: *mut CUaddress_mode,
        hTexRef: CUtexref,
        dim: ::core::ffi::c_int,
    ) -> CUresult;
    pub fn cuTexRefGetAddress_v2(pdptr: *mut CUdeviceptr, hTexRef: CUtexref) -> CUresult;
    pub fn cuTexRefGetArray(phArray: *mut CUarray, hTexRef: CUtexref) -> CUresult;
    pub fn cuTexRefGetBorderColor(pBorderColor: *mut f32, hTexRef: CUtexref) -> CUresult;
    pub fn cuTexRefGetFilterMode(pfm: *mut CUfilter_mode, hTexRef: CUtexref) -> CUresult;
    pub fn cuTexRefGetFlags(
        pFlags: *mut ::core::ffi::c_uint,
        hTexRef: CUtexref,
    ) -> CUresult;
    pub fn cuTexRefGetFormat(
        pFormat: *mut CUarray_format,
        pNumChannels: *mut ::core::ffi::c_int,
        hTexRef: CUtexref,
    ) -> CUresult;
    pub fn cuTexRefGetMaxAnisotropy(
        pmaxAniso: *mut ::core::ffi::c_int,
        hTexRef: CUtexref,
    ) -> CUresult;
    pub fn cuTexRefGetMipmapFilterMode(
        pfm: *mut CUfilter_mode,
        hTexRef: CUtexref,
    ) -> CUresult;
    pub fn cuTexRefGetMipmapLevelBias(pbias: *mut f32, hTexRef: CUtexref) -> CUresult;
    pub fn cuTexRefGetMipmapLevelClamp(
        pminMipmapLevelClamp: *mut f32,
        pmaxMipmapLevelClamp: *mut f32,
        hTexRef: CUtexref,
    ) -> CUresult;
    pub fn cuTexRefGetMipmappedArray(
        phMipmappedArray: *mut CUmipmappedArray,
        hTexRef: CUtexref,
    ) -> CUresult;
    pub fn cuTexRefSetAddress2D_v3(
        hTexRef: CUtexref,
        desc: *const CUDA_ARRAY_DESCRIPTOR,
        dptr: CUdeviceptr,
        Pitch: usize,
    ) -> CUresult;
    pub fn cuTexRefSetAddressMode(
        hTexRef: CUtexref,
        dim: ::core::ffi::c_int,
        am: CUaddress_mode,
    ) -> CUresult;
    pub fn cuTexRefSetAddress_v2(
        ByteOffset: *mut usize,
        hTexRef: CUtexref,
        dptr: CUdeviceptr,
        bytes: usize,
    ) -> CUresult;
    pub fn cuTexRefSetArray(
        hTexRef: CUtexref,
        hArray: CUarray,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuTexRefSetBorderColor(hTexRef: CUtexref, pBorderColor: *mut f32) -> CUresult;
    pub fn cuTexRefSetFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) -> CUresult;
    pub fn cuTexRefSetFlags(hTexRef: CUtexref, Flags: ::core::ffi::c_uint) -> CUresult;
    pub fn cuTexRefSetFormat(
        hTexRef: CUtexref,
        fmt: CUarray_format,
        NumPackedComponents: ::core::ffi::c_int,
    ) -> CUresult;
    pub fn cuTexRefSetMaxAnisotropy(
        hTexRef: CUtexref,
        maxAniso: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuTexRefSetMipmapFilterMode(hTexRef: CUtexref, fm: CUfilter_mode) -> CUresult;
    pub fn cuTexRefSetMipmapLevelBias(hTexRef: CUtexref, bias: f32) -> CUresult;
    pub fn cuTexRefSetMipmapLevelClamp(
        hTexRef: CUtexref,
        minMipmapLevelClamp: f32,
        maxMipmapLevelClamp: f32,
    ) -> CUresult;
    pub fn cuTexRefSetMipmappedArray(
        hTexRef: CUtexref,
        hMipmappedArray: CUmipmappedArray,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuThreadExchangeStreamCaptureMode(mode: *mut CUstreamCaptureMode) -> CUresult;
    pub fn cuUserObjectCreate(
        object_out: *mut CUuserObject,
        ptr: *mut ::core::ffi::c_void,
        destroy: CUhostFn,
        initialRefcount: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuUserObjectRelease(
        object: CUuserObject,
        count: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuUserObjectRetain(
        object: CUuserObject,
        count: ::core::ffi::c_uint,
    ) -> CUresult;
    pub fn cuWaitExternalSemaphoresAsync(
        extSemArray: *const CUexternalSemaphore,
        paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
        numExtSems: ::core::ffi::c_uint,
        stream: CUstream,
    ) -> CUresult;
}
#[cfg(feature = "dynamic-loading")]
mod loaded {
    use super::*;
    pub unsafe fn cuArray3DCreate_v2(
        pHandle: *mut CUarray,
        pAllocateArray: *const CUDA_ARRAY3D_DESCRIPTOR,
    ) -> CUresult {
        (culib().cuArray3DCreate_v2)(pHandle, pAllocateArray)
    }
    pub unsafe fn cuArray3DGetDescriptor_v2(
        pArrayDescriptor: *mut CUDA_ARRAY3D_DESCRIPTOR,
        hArray: CUarray,
    ) -> CUresult {
        (culib().cuArray3DGetDescriptor_v2)(pArrayDescriptor, hArray)
    }
    pub unsafe fn cuArrayCreate_v2(
        pHandle: *mut CUarray,
        pAllocateArray: *const CUDA_ARRAY_DESCRIPTOR,
    ) -> CUresult {
        (culib().cuArrayCreate_v2)(pHandle, pAllocateArray)
    }
    pub unsafe fn cuArrayDestroy(hArray: CUarray) -> CUresult {
        (culib().cuArrayDestroy)(hArray)
    }
    pub unsafe fn cuArrayGetDescriptor_v2(
        pArrayDescriptor: *mut CUDA_ARRAY_DESCRIPTOR,
        hArray: CUarray,
    ) -> CUresult {
        (culib().cuArrayGetDescriptor_v2)(pArrayDescriptor, hArray)
    }
    #[cfg(
        any(
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
        )
    )]
    pub unsafe fn cuArrayGetMemoryRequirements(
        memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS,
        array: CUarray,
        device: CUdevice,
    ) -> CUresult {
        (culib().cuArrayGetMemoryRequirements)(memoryRequirements, array, device)
    }
    pub unsafe fn cuArrayGetPlane(
        pPlaneArray: *mut CUarray,
        hArray: CUarray,
        planeIdx: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuArrayGetPlane)(pPlaneArray, hArray, planeIdx)
    }
    pub unsafe fn cuArrayGetSparseProperties(
        sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES,
        array: CUarray,
    ) -> CUresult {
        (culib().cuArrayGetSparseProperties)(sparseProperties, array)
    }
    #[cfg(any(feature = "cuda-12080"))]
    pub unsafe fn cuCheckpointProcessCheckpoint(
        pid: ::core::ffi::c_int,
        args: *mut CUcheckpointCheckpointArgs,
    ) -> CUresult {
        (culib().cuCheckpointProcessCheckpoint)(pid, args)
    }
    #[cfg(any(feature = "cuda-12080"))]
    pub unsafe fn cuCheckpointProcessGetRestoreThreadId(
        pid: ::core::ffi::c_int,
        tid: *mut ::core::ffi::c_int,
    ) -> CUresult {
        (culib().cuCheckpointProcessGetRestoreThreadId)(pid, tid)
    }
    #[cfg(any(feature = "cuda-12080"))]
    pub unsafe fn cuCheckpointProcessGetState(
        pid: ::core::ffi::c_int,
        state: *mut CUprocessState,
    ) -> CUresult {
        (culib().cuCheckpointProcessGetState)(pid, state)
    }
    #[cfg(any(feature = "cuda-12080"))]
    pub unsafe fn cuCheckpointProcessLock(
        pid: ::core::ffi::c_int,
        args: *mut CUcheckpointLockArgs,
    ) -> CUresult {
        (culib().cuCheckpointProcessLock)(pid, args)
    }
    #[cfg(any(feature = "cuda-12080"))]
    pub unsafe fn cuCheckpointProcessRestore(
        pid: ::core::ffi::c_int,
        args: *mut CUcheckpointRestoreArgs,
    ) -> CUresult {
        (culib().cuCheckpointProcessRestore)(pid, args)
    }
    #[cfg(any(feature = "cuda-12080"))]
    pub unsafe fn cuCheckpointProcessUnlock(
        pid: ::core::ffi::c_int,
        args: *mut CUcheckpointUnlockArgs,
    ) -> CUresult {
        (culib().cuCheckpointProcessUnlock)(pid, args)
    }
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuCoredumpGetAttribute(
        attrib: CUcoredumpSettings,
        value: *mut ::core::ffi::c_void,
        size: *mut usize,
    ) -> CUresult {
        (culib().cuCoredumpGetAttribute)(attrib, value, size)
    }
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuCoredumpGetAttributeGlobal(
        attrib: CUcoredumpSettings,
        value: *mut ::core::ffi::c_void,
        size: *mut usize,
    ) -> CUresult {
        (culib().cuCoredumpGetAttributeGlobal)(attrib, value, size)
    }
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuCoredumpSetAttribute(
        attrib: CUcoredumpSettings,
        value: *mut ::core::ffi::c_void,
        size: *mut usize,
    ) -> CUresult {
        (culib().cuCoredumpSetAttribute)(attrib, value, size)
    }
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuCoredumpSetAttributeGlobal(
        attrib: CUcoredumpSettings,
        value: *mut ::core::ffi::c_void,
        size: *mut usize,
    ) -> CUresult {
        (culib().cuCoredumpSetAttributeGlobal)(attrib, value, size)
    }
    pub unsafe fn cuCtxAttach(
        pctx: *mut CUcontext,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuCtxAttach)(pctx, flags)
    }
    pub unsafe fn cuCtxCreate_v2(
        pctx: *mut CUcontext,
        flags: ::core::ffi::c_uint,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuCtxCreate_v2)(pctx, flags, dev)
    }
    pub unsafe fn cuCtxCreate_v3(
        pctx: *mut CUcontext,
        paramsArray: *mut CUexecAffinityParam,
        numParams: ::core::ffi::c_int,
        flags: ::core::ffi::c_uint,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuCtxCreate_v3)(pctx, paramsArray, numParams, flags, dev)
    }
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub unsafe fn cuCtxCreate_v4(
        pctx: *mut CUcontext,
        ctxCreateParams: *mut CUctxCreateParams,
        flags: ::core::ffi::c_uint,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuCtxCreate_v4)(pctx, ctxCreateParams, flags, dev)
    }
    pub unsafe fn cuCtxDestroy_v2(ctx: CUcontext) -> CUresult {
        (culib().cuCtxDestroy_v2)(ctx)
    }
    pub unsafe fn cuCtxDetach(ctx: CUcontext) -> CUresult {
        (culib().cuCtxDetach)(ctx)
    }
    pub unsafe fn cuCtxDisablePeerAccess(peerContext: CUcontext) -> CUresult {
        (culib().cuCtxDisablePeerAccess)(peerContext)
    }
    pub unsafe fn cuCtxEnablePeerAccess(
        peerContext: CUcontext,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuCtxEnablePeerAccess)(peerContext, Flags)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuCtxFromGreenCtx(
        pContext: *mut CUcontext,
        hCtx: CUgreenCtx,
    ) -> CUresult {
        (culib().cuCtxFromGreenCtx)(pContext, hCtx)
    }
    pub unsafe fn cuCtxGetApiVersion(
        ctx: CUcontext,
        version: *mut ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuCtxGetApiVersion)(ctx, version)
    }
    pub unsafe fn cuCtxGetCacheConfig(pconfig: *mut CUfunc_cache) -> CUresult {
        (culib().cuCtxGetCacheConfig)(pconfig)
    }
    pub unsafe fn cuCtxGetCurrent(pctx: *mut CUcontext) -> CUresult {
        (culib().cuCtxGetCurrent)(pctx)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuCtxGetDevResource(
        hCtx: CUcontext,
        resource: *mut CUdevResource,
        type_: CUdevResourceType,
    ) -> CUresult {
        (culib().cuCtxGetDevResource)(hCtx, resource, type_)
    }
    pub unsafe fn cuCtxGetDevice(device: *mut CUdevice) -> CUresult {
        (culib().cuCtxGetDevice)(device)
    }
    pub unsafe fn cuCtxGetExecAffinity(
        pExecAffinity: *mut CUexecAffinityParam,
        type_: CUexecAffinityType,
    ) -> CUresult {
        (culib().cuCtxGetExecAffinity)(pExecAffinity, type_)
    }
    pub unsafe fn cuCtxGetFlags(flags: *mut ::core::ffi::c_uint) -> CUresult {
        (culib().cuCtxGetFlags)(flags)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuCtxGetId(
        ctx: CUcontext,
        ctxId: *mut ::core::ffi::c_ulonglong,
    ) -> CUresult {
        (culib().cuCtxGetId)(ctx, ctxId)
    }
    pub unsafe fn cuCtxGetLimit(pvalue: *mut usize, limit: CUlimit) -> CUresult {
        (culib().cuCtxGetLimit)(pvalue, limit)
    }
    pub unsafe fn cuCtxGetSharedMemConfig(pConfig: *mut CUsharedconfig) -> CUresult {
        (culib().cuCtxGetSharedMemConfig)(pConfig)
    }
    pub unsafe fn cuCtxGetStreamPriorityRange(
        leastPriority: *mut ::core::ffi::c_int,
        greatestPriority: *mut ::core::ffi::c_int,
    ) -> CUresult {
        (culib().cuCtxGetStreamPriorityRange)(leastPriority, greatestPriority)
    }
    pub unsafe fn cuCtxPopCurrent_v2(pctx: *mut CUcontext) -> CUresult {
        (culib().cuCtxPopCurrent_v2)(pctx)
    }
    pub unsafe fn cuCtxPushCurrent_v2(ctx: CUcontext) -> CUresult {
        (culib().cuCtxPushCurrent_v2)(ctx)
    }
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub unsafe fn cuCtxRecordEvent(hCtx: CUcontext, hEvent: CUevent) -> CUresult {
        (culib().cuCtxRecordEvent)(hCtx, hEvent)
    }
    pub unsafe fn cuCtxResetPersistingL2Cache() -> CUresult {
        (culib().cuCtxResetPersistingL2Cache)()
    }
    pub unsafe fn cuCtxSetCacheConfig(config: CUfunc_cache) -> CUresult {
        (culib().cuCtxSetCacheConfig)(config)
    }
    pub unsafe fn cuCtxSetCurrent(ctx: CUcontext) -> CUresult {
        (culib().cuCtxSetCurrent)(ctx)
    }
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuCtxSetFlags(flags: ::core::ffi::c_uint) -> CUresult {
        (culib().cuCtxSetFlags)(flags)
    }
    pub unsafe fn cuCtxSetLimit(limit: CUlimit, value: usize) -> CUresult {
        (culib().cuCtxSetLimit)(limit, value)
    }
    pub unsafe fn cuCtxSetSharedMemConfig(config: CUsharedconfig) -> CUresult {
        (culib().cuCtxSetSharedMemConfig)(config)
    }
    pub unsafe fn cuCtxSynchronize() -> CUresult {
        (culib().cuCtxSynchronize)()
    }
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub unsafe fn cuCtxWaitEvent(hCtx: CUcontext, hEvent: CUevent) -> CUresult {
        (culib().cuCtxWaitEvent)(hCtx, hEvent)
    }
    pub unsafe fn cuDestroyExternalMemory(extMem: CUexternalMemory) -> CUresult {
        (culib().cuDestroyExternalMemory)(extMem)
    }
    pub unsafe fn cuDestroyExternalSemaphore(extSem: CUexternalSemaphore) -> CUresult {
        (culib().cuDestroyExternalSemaphore)(extSem)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuDevResourceGenerateDesc(
        phDesc: *mut CUdevResourceDesc,
        resources: *mut CUdevResource,
        nbResources: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuDevResourceGenerateDesc)(phDesc, resources, nbResources)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuDevSmResourceSplitByCount(
        result: *mut CUdevResource,
        nbGroups: *mut ::core::ffi::c_uint,
        input: *const CUdevResource,
        remaining: *mut CUdevResource,
        useFlags: ::core::ffi::c_uint,
        minCount: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib()
            .cuDevSmResourceSplitByCount)(
            result,
            nbGroups,
            input,
            remaining,
            useFlags,
            minCount,
        )
    }
    pub unsafe fn cuDeviceCanAccessPeer(
        canAccessPeer: *mut ::core::ffi::c_int,
        dev: CUdevice,
        peerDev: CUdevice,
    ) -> CUresult {
        (culib().cuDeviceCanAccessPeer)(canAccessPeer, dev, peerDev)
    }
    pub unsafe fn cuDeviceComputeCapability(
        major: *mut ::core::ffi::c_int,
        minor: *mut ::core::ffi::c_int,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuDeviceComputeCapability)(major, minor, dev)
    }
    pub unsafe fn cuDeviceGet(
        device: *mut CUdevice,
        ordinal: ::core::ffi::c_int,
    ) -> CUresult {
        (culib().cuDeviceGet)(device, ordinal)
    }
    pub unsafe fn cuDeviceGetAttribute(
        pi: *mut ::core::ffi::c_int,
        attrib: CUdevice_attribute,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuDeviceGetAttribute)(pi, attrib, dev)
    }
    pub unsafe fn cuDeviceGetByPCIBusId(
        dev: *mut CUdevice,
        pciBusId: *const ::core::ffi::c_char,
    ) -> CUresult {
        (culib().cuDeviceGetByPCIBusId)(dev, pciBusId)
    }
    pub unsafe fn cuDeviceGetCount(count: *mut ::core::ffi::c_int) -> CUresult {
        (culib().cuDeviceGetCount)(count)
    }
    pub unsafe fn cuDeviceGetDefaultMemPool(
        pool_out: *mut CUmemoryPool,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuDeviceGetDefaultMemPool)(pool_out, dev)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuDeviceGetDevResource(
        device: CUdevice,
        resource: *mut CUdevResource,
        type_: CUdevResourceType,
    ) -> CUresult {
        (culib().cuDeviceGetDevResource)(device, resource, type_)
    }
    pub unsafe fn cuDeviceGetExecAffinitySupport(
        pi: *mut ::core::ffi::c_int,
        type_: CUexecAffinityType,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuDeviceGetExecAffinitySupport)(pi, type_, dev)
    }
    pub unsafe fn cuDeviceGetGraphMemAttribute(
        device: CUdevice,
        attr: CUgraphMem_attribute,
        value: *mut ::core::ffi::c_void,
    ) -> CUresult {
        (culib().cuDeviceGetGraphMemAttribute)(device, attr, value)
    }
    pub unsafe fn cuDeviceGetLuid(
        luid: *mut ::core::ffi::c_char,
        deviceNodeMask: *mut ::core::ffi::c_uint,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuDeviceGetLuid)(luid, deviceNodeMask, dev)
    }
    pub unsafe fn cuDeviceGetMemPool(
        pool: *mut CUmemoryPool,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuDeviceGetMemPool)(pool, dev)
    }
    pub unsafe fn cuDeviceGetName(
        name: *mut ::core::ffi::c_char,
        len: ::core::ffi::c_int,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuDeviceGetName)(name, len, dev)
    }
    pub unsafe fn cuDeviceGetNvSciSyncAttributes(
        nvSciSyncAttrList: *mut ::core::ffi::c_void,
        dev: CUdevice,
        flags: ::core::ffi::c_int,
    ) -> CUresult {
        (culib().cuDeviceGetNvSciSyncAttributes)(nvSciSyncAttrList, dev, flags)
    }
    pub unsafe fn cuDeviceGetP2PAttribute(
        value: *mut ::core::ffi::c_int,
        attrib: CUdevice_P2PAttribute,
        srcDevice: CUdevice,
        dstDevice: CUdevice,
    ) -> CUresult {
        (culib().cuDeviceGetP2PAttribute)(value, attrib, srcDevice, dstDevice)
    }
    pub unsafe fn cuDeviceGetPCIBusId(
        pciBusId: *mut ::core::ffi::c_char,
        len: ::core::ffi::c_int,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuDeviceGetPCIBusId)(pciBusId, len, dev)
    }
    pub unsafe fn cuDeviceGetProperties(
        prop: *mut CUdevprop,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuDeviceGetProperties)(prop, dev)
    }
    pub unsafe fn cuDeviceGetTexture1DLinearMaxWidth(
        maxWidthInElements: *mut usize,
        format: CUarray_format,
        numChannels: ::core::ffi::c_uint,
        dev: CUdevice,
    ) -> CUresult {
        (culib()
            .cuDeviceGetTexture1DLinearMaxWidth)(
            maxWidthInElements,
            format,
            numChannels,
            dev,
        )
    }
    pub unsafe fn cuDeviceGetUuid(uuid: *mut CUuuid, dev: CUdevice) -> CUresult {
        (culib().cuDeviceGetUuid)(uuid, dev)
    }
    pub unsafe fn cuDeviceGetUuid_v2(uuid: *mut CUuuid, dev: CUdevice) -> CUresult {
        (culib().cuDeviceGetUuid_v2)(uuid, dev)
    }
    pub unsafe fn cuDeviceGraphMemTrim(device: CUdevice) -> CUresult {
        (culib().cuDeviceGraphMemTrim)(device)
    }
    pub unsafe fn cuDevicePrimaryCtxGetState(
        dev: CUdevice,
        flags: *mut ::core::ffi::c_uint,
        active: *mut ::core::ffi::c_int,
    ) -> CUresult {
        (culib().cuDevicePrimaryCtxGetState)(dev, flags, active)
    }
    pub unsafe fn cuDevicePrimaryCtxRelease_v2(dev: CUdevice) -> CUresult {
        (culib().cuDevicePrimaryCtxRelease_v2)(dev)
    }
    pub unsafe fn cuDevicePrimaryCtxReset_v2(dev: CUdevice) -> CUresult {
        (culib().cuDevicePrimaryCtxReset_v2)(dev)
    }
    pub unsafe fn cuDevicePrimaryCtxRetain(
        pctx: *mut CUcontext,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuDevicePrimaryCtxRetain)(pctx, dev)
    }
    pub unsafe fn cuDevicePrimaryCtxSetFlags_v2(
        dev: CUdevice,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuDevicePrimaryCtxSetFlags_v2)(dev, flags)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuDeviceRegisterAsyncNotification(
        device: CUdevice,
        callbackFunc: CUasyncCallback,
        userData: *mut ::core::ffi::c_void,
        callback: *mut CUasyncCallbackHandle,
    ) -> CUresult {
        (culib()
            .cuDeviceRegisterAsyncNotification)(device, callbackFunc, userData, callback)
    }
    pub unsafe fn cuDeviceSetGraphMemAttribute(
        device: CUdevice,
        attr: CUgraphMem_attribute,
        value: *mut ::core::ffi::c_void,
    ) -> CUresult {
        (culib().cuDeviceSetGraphMemAttribute)(device, attr, value)
    }
    pub unsafe fn cuDeviceSetMemPool(dev: CUdevice, pool: CUmemoryPool) -> CUresult {
        (culib().cuDeviceSetMemPool)(dev, pool)
    }
    pub unsafe fn cuDeviceTotalMem_v2(bytes: *mut usize, dev: CUdevice) -> CUresult {
        (culib().cuDeviceTotalMem_v2)(bytes, dev)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuDeviceUnregisterAsyncNotification(
        device: CUdevice,
        callback: CUasyncCallbackHandle,
    ) -> CUresult {
        (culib().cuDeviceUnregisterAsyncNotification)(device, callback)
    }
    pub unsafe fn cuDriverGetVersion(
        driverVersion: *mut ::core::ffi::c_int,
    ) -> CUresult {
        (culib().cuDriverGetVersion)(driverVersion)
    }
    pub unsafe fn cuEventCreate(
        phEvent: *mut CUevent,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuEventCreate)(phEvent, Flags)
    }
    pub unsafe fn cuEventDestroy_v2(hEvent: CUevent) -> CUresult {
        (culib().cuEventDestroy_v2)(hEvent)
    }
    pub unsafe fn cuEventElapsedTime(
        pMilliseconds: *mut f32,
        hStart: CUevent,
        hEnd: CUevent,
    ) -> CUresult {
        (culib().cuEventElapsedTime)(pMilliseconds, hStart, hEnd)
    }
    #[cfg(any(feature = "cuda-12080"))]
    pub unsafe fn cuEventElapsedTime_v2(
        pMilliseconds: *mut f32,
        hStart: CUevent,
        hEnd: CUevent,
    ) -> CUresult {
        (culib().cuEventElapsedTime_v2)(pMilliseconds, hStart, hEnd)
    }
    pub unsafe fn cuEventQuery(hEvent: CUevent) -> CUresult {
        (culib().cuEventQuery)(hEvent)
    }
    pub unsafe fn cuEventRecord(hEvent: CUevent, hStream: CUstream) -> CUresult {
        (culib().cuEventRecord)(hEvent, hStream)
    }
    pub unsafe fn cuEventRecordWithFlags(
        hEvent: CUevent,
        hStream: CUstream,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuEventRecordWithFlags)(hEvent, hStream, flags)
    }
    pub unsafe fn cuEventSynchronize(hEvent: CUevent) -> CUresult {
        (culib().cuEventSynchronize)(hEvent)
    }
    pub unsafe fn cuExternalMemoryGetMappedBuffer(
        devPtr: *mut CUdeviceptr,
        extMem: CUexternalMemory,
        bufferDesc: *const CUDA_EXTERNAL_MEMORY_BUFFER_DESC,
    ) -> CUresult {
        (culib().cuExternalMemoryGetMappedBuffer)(devPtr, extMem, bufferDesc)
    }
    pub unsafe fn cuExternalMemoryGetMappedMipmappedArray(
        mipmap: *mut CUmipmappedArray,
        extMem: CUexternalMemory,
        mipmapDesc: *const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC,
    ) -> CUresult {
        (culib().cuExternalMemoryGetMappedMipmappedArray)(mipmap, extMem, mipmapDesc)
    }
    pub unsafe fn cuFlushGPUDirectRDMAWrites(
        target: CUflushGPUDirectRDMAWritesTarget,
        scope: CUflushGPUDirectRDMAWritesScope,
    ) -> CUresult {
        (culib().cuFlushGPUDirectRDMAWrites)(target, scope)
    }
    pub unsafe fn cuFuncGetAttribute(
        pi: *mut ::core::ffi::c_int,
        attrib: CUfunction_attribute,
        hfunc: CUfunction,
    ) -> CUresult {
        (culib().cuFuncGetAttribute)(pi, attrib, hfunc)
    }
    pub unsafe fn cuFuncGetModule(hmod: *mut CUmodule, hfunc: CUfunction) -> CUresult {
        (culib().cuFuncGetModule)(hmod, hfunc)
    }
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuFuncGetName(
        name: *mut *const ::core::ffi::c_char,
        hfunc: CUfunction,
    ) -> CUresult {
        (culib().cuFuncGetName)(name, hfunc)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuFuncGetParamInfo(
        func: CUfunction,
        paramIndex: usize,
        paramOffset: *mut usize,
        paramSize: *mut usize,
    ) -> CUresult {
        (culib().cuFuncGetParamInfo)(func, paramIndex, paramOffset, paramSize)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuFuncIsLoaded(
        state: *mut CUfunctionLoadingState,
        function: CUfunction,
    ) -> CUresult {
        (culib().cuFuncIsLoaded)(state, function)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuFuncLoad(function: CUfunction) -> CUresult {
        (culib().cuFuncLoad)(function)
    }
    pub unsafe fn cuFuncSetAttribute(
        hfunc: CUfunction,
        attrib: CUfunction_attribute,
        value: ::core::ffi::c_int,
    ) -> CUresult {
        (culib().cuFuncSetAttribute)(hfunc, attrib, value)
    }
    pub unsafe fn cuFuncSetBlockShape(
        hfunc: CUfunction,
        x: ::core::ffi::c_int,
        y: ::core::ffi::c_int,
        z: ::core::ffi::c_int,
    ) -> CUresult {
        (culib().cuFuncSetBlockShape)(hfunc, x, y, z)
    }
    pub unsafe fn cuFuncSetCacheConfig(
        hfunc: CUfunction,
        config: CUfunc_cache,
    ) -> CUresult {
        (culib().cuFuncSetCacheConfig)(hfunc, config)
    }
    pub unsafe fn cuFuncSetSharedMemConfig(
        hfunc: CUfunction,
        config: CUsharedconfig,
    ) -> CUresult {
        (culib().cuFuncSetSharedMemConfig)(hfunc, config)
    }
    pub unsafe fn cuFuncSetSharedSize(
        hfunc: CUfunction,
        bytes: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuFuncSetSharedSize)(hfunc, bytes)
    }
    pub unsafe fn cuGetErrorName(
        error: CUresult,
        pStr: *mut *const ::core::ffi::c_char,
    ) -> CUresult {
        (culib().cuGetErrorName)(error, pStr)
    }
    pub unsafe fn cuGetErrorString(
        error: CUresult,
        pStr: *mut *const ::core::ffi::c_char,
    ) -> CUresult {
        (culib().cuGetErrorString)(error, pStr)
    }
    pub unsafe fn cuGetExportTable(
        ppExportTable: *mut *const ::core::ffi::c_void,
        pExportTableId: *const CUuuid,
    ) -> CUresult {
        (culib().cuGetExportTable)(ppExportTable, pExportTableId)
    }
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub unsafe fn cuGetProcAddress(
        symbol: *const ::core::ffi::c_char,
        pfn: *mut *mut ::core::ffi::c_void,
        cudaVersion: ::core::ffi::c_int,
        flags: cuuint64_t,
    ) -> CUresult {
        (culib().cuGetProcAddress)(symbol, pfn, cudaVersion, flags)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGetProcAddress_v2(
        symbol: *const ::core::ffi::c_char,
        pfn: *mut *mut ::core::ffi::c_void,
        cudaVersion: ::core::ffi::c_int,
        flags: cuuint64_t,
        symbolStatus: *mut CUdriverProcAddressQueryResult,
    ) -> CUresult {
        (culib().cuGetProcAddress_v2)(symbol, pfn, cudaVersion, flags, symbolStatus)
    }
    #[cfg(
        any(
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
        )
    )]
    pub unsafe fn cuGraphAddBatchMemOpNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
    ) -> CUresult {
        (culib()
            .cuGraphAddBatchMemOpNode)(
            phGraphNode,
            hGraph,
            dependencies,
            numDependencies,
            nodeParams,
        )
    }
    pub unsafe fn cuGraphAddChildGraphNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        childGraph: CUgraph,
    ) -> CUresult {
        (culib()
            .cuGraphAddChildGraphNode)(
            phGraphNode,
            hGraph,
            dependencies,
            numDependencies,
            childGraph,
        )
    }
    pub unsafe fn cuGraphAddDependencies(
        hGraph: CUgraph,
        from: *const CUgraphNode,
        to: *const CUgraphNode,
        numDependencies: usize,
    ) -> CUresult {
        (culib().cuGraphAddDependencies)(hGraph, from, to, numDependencies)
    }
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphAddDependencies_v2(
        hGraph: CUgraph,
        from: *const CUgraphNode,
        to: *const CUgraphNode,
        edgeData: *const CUgraphEdgeData,
        numDependencies: usize,
    ) -> CUresult {
        (culib().cuGraphAddDependencies_v2)(hGraph, from, to, edgeData, numDependencies)
    }
    pub unsafe fn cuGraphAddEmptyNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
    ) -> CUresult {
        (culib().cuGraphAddEmptyNode)(phGraphNode, hGraph, dependencies, numDependencies)
    }
    pub unsafe fn cuGraphAddEventRecordNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        event: CUevent,
    ) -> CUresult {
        (culib()
            .cuGraphAddEventRecordNode)(
            phGraphNode,
            hGraph,
            dependencies,
            numDependencies,
            event,
        )
    }
    pub unsafe fn cuGraphAddEventWaitNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        event: CUevent,
    ) -> CUresult {
        (culib()
            .cuGraphAddEventWaitNode)(
            phGraphNode,
            hGraph,
            dependencies,
            numDependencies,
            event,
        )
    }
    pub unsafe fn cuGraphAddExternalSemaphoresSignalNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
    ) -> CUresult {
        (culib()
            .cuGraphAddExternalSemaphoresSignalNode)(
            phGraphNode,
            hGraph,
            dependencies,
            numDependencies,
            nodeParams,
        )
    }
    pub unsafe fn cuGraphAddExternalSemaphoresWaitNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS,
    ) -> CUresult {
        (culib()
            .cuGraphAddExternalSemaphoresWaitNode)(
            phGraphNode,
            hGraph,
            dependencies,
            numDependencies,
            nodeParams,
        )
    }
    pub unsafe fn cuGraphAddHostNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *const CUDA_HOST_NODE_PARAMS,
    ) -> CUresult {
        (culib()
            .cuGraphAddHostNode)(
            phGraphNode,
            hGraph,
            dependencies,
            numDependencies,
            nodeParams,
        )
    }
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub unsafe fn cuGraphAddKernelNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult {
        (culib()
            .cuGraphAddKernelNode)(
            phGraphNode,
            hGraph,
            dependencies,
            numDependencies,
            nodeParams,
        )
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphAddKernelNode_v2(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult {
        (culib()
            .cuGraphAddKernelNode_v2)(
            phGraphNode,
            hGraph,
            dependencies,
            numDependencies,
            nodeParams,
        )
    }
    pub unsafe fn cuGraphAddMemAllocNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *mut CUDA_MEM_ALLOC_NODE_PARAMS,
    ) -> CUresult {
        (culib()
            .cuGraphAddMemAllocNode)(
            phGraphNode,
            hGraph,
            dependencies,
            numDependencies,
            nodeParams,
        )
    }
    pub unsafe fn cuGraphAddMemFreeNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        dptr: CUdeviceptr,
    ) -> CUresult {
        (culib()
            .cuGraphAddMemFreeNode)(
            phGraphNode,
            hGraph,
            dependencies,
            numDependencies,
            dptr,
        )
    }
    pub unsafe fn cuGraphAddMemcpyNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        copyParams: *const CUDA_MEMCPY3D,
        ctx: CUcontext,
    ) -> CUresult {
        (culib()
            .cuGraphAddMemcpyNode)(
            phGraphNode,
            hGraph,
            dependencies,
            numDependencies,
            copyParams,
            ctx,
        )
    }
    pub unsafe fn cuGraphAddMemsetNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        memsetParams: *const CUDA_MEMSET_NODE_PARAMS,
        ctx: CUcontext,
    ) -> CUresult {
        (culib()
            .cuGraphAddMemsetNode)(
            phGraphNode,
            hGraph,
            dependencies,
            numDependencies,
            memsetParams,
            ctx,
        )
    }
    #[cfg(
        any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphAddNode(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        numDependencies: usize,
        nodeParams: *mut CUgraphNodeParams,
    ) -> CUresult {
        (culib()
            .cuGraphAddNode)(
            phGraphNode,
            hGraph,
            dependencies,
            numDependencies,
            nodeParams,
        )
    }
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphAddNode_v2(
        phGraphNode: *mut CUgraphNode,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        dependencyData: *const CUgraphEdgeData,
        numDependencies: usize,
        nodeParams: *mut CUgraphNodeParams,
    ) -> CUresult {
        (culib()
            .cuGraphAddNode_v2)(
            phGraphNode,
            hGraph,
            dependencies,
            dependencyData,
            numDependencies,
            nodeParams,
        )
    }
    #[cfg(
        any(
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
        )
    )]
    pub unsafe fn cuGraphBatchMemOpNodeGetParams(
        hNode: CUgraphNode,
        nodeParams_out: *mut CUDA_BATCH_MEM_OP_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphBatchMemOpNodeGetParams)(hNode, nodeParams_out)
    }
    #[cfg(
        any(
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
        )
    )]
    pub unsafe fn cuGraphBatchMemOpNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphBatchMemOpNodeSetParams)(hNode, nodeParams)
    }
    pub unsafe fn cuGraphChildGraphNodeGetGraph(
        hNode: CUgraphNode,
        phGraph: *mut CUgraph,
    ) -> CUresult {
        (culib().cuGraphChildGraphNodeGetGraph)(hNode, phGraph)
    }
    pub unsafe fn cuGraphClone(
        phGraphClone: *mut CUgraph,
        originalGraph: CUgraph,
    ) -> CUresult {
        (culib().cuGraphClone)(phGraphClone, originalGraph)
    }
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphConditionalHandleCreate(
        pHandle_out: *mut CUgraphConditionalHandle,
        hGraph: CUgraph,
        ctx: CUcontext,
        defaultLaunchValue: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib()
            .cuGraphConditionalHandleCreate)(
            pHandle_out,
            hGraph,
            ctx,
            defaultLaunchValue,
            flags,
        )
    }
    pub unsafe fn cuGraphCreate(
        phGraph: *mut CUgraph,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuGraphCreate)(phGraph, flags)
    }
    pub unsafe fn cuGraphDebugDotPrint(
        hGraph: CUgraph,
        path: *const ::core::ffi::c_char,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuGraphDebugDotPrint)(hGraph, path, flags)
    }
    pub unsafe fn cuGraphDestroy(hGraph: CUgraph) -> CUresult {
        (culib().cuGraphDestroy)(hGraph)
    }
    pub unsafe fn cuGraphDestroyNode(hNode: CUgraphNode) -> CUresult {
        (culib().cuGraphDestroyNode)(hNode)
    }
    pub unsafe fn cuGraphEventRecordNodeGetEvent(
        hNode: CUgraphNode,
        event_out: *mut CUevent,
    ) -> CUresult {
        (culib().cuGraphEventRecordNodeGetEvent)(hNode, event_out)
    }
    pub unsafe fn cuGraphEventRecordNodeSetEvent(
        hNode: CUgraphNode,
        event: CUevent,
    ) -> CUresult {
        (culib().cuGraphEventRecordNodeSetEvent)(hNode, event)
    }
    pub unsafe fn cuGraphEventWaitNodeGetEvent(
        hNode: CUgraphNode,
        event_out: *mut CUevent,
    ) -> CUresult {
        (culib().cuGraphEventWaitNodeGetEvent)(hNode, event_out)
    }
    pub unsafe fn cuGraphEventWaitNodeSetEvent(
        hNode: CUgraphNode,
        event: CUevent,
    ) -> CUresult {
        (culib().cuGraphEventWaitNodeSetEvent)(hNode, event)
    }
    #[cfg(
        any(
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
        )
    )]
    pub unsafe fn cuGraphExecBatchMemOpNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphExecBatchMemOpNodeSetParams)(hGraphExec, hNode, nodeParams)
    }
    pub unsafe fn cuGraphExecChildGraphNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        childGraph: CUgraph,
    ) -> CUresult {
        (culib().cuGraphExecChildGraphNodeSetParams)(hGraphExec, hNode, childGraph)
    }
    pub unsafe fn cuGraphExecDestroy(hGraphExec: CUgraphExec) -> CUresult {
        (culib().cuGraphExecDestroy)(hGraphExec)
    }
    pub unsafe fn cuGraphExecEventRecordNodeSetEvent(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        event: CUevent,
    ) -> CUresult {
        (culib().cuGraphExecEventRecordNodeSetEvent)(hGraphExec, hNode, event)
    }
    pub unsafe fn cuGraphExecEventWaitNodeSetEvent(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        event: CUevent,
    ) -> CUresult {
        (culib().cuGraphExecEventWaitNodeSetEvent)(hGraphExec, hNode, event)
    }
    pub unsafe fn cuGraphExecExternalSemaphoresSignalNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
    ) -> CUresult {
        (culib()
            .cuGraphExecExternalSemaphoresSignalNodeSetParams)(
            hGraphExec,
            hNode,
            nodeParams,
        )
    }
    pub unsafe fn cuGraphExecExternalSemaphoresWaitNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS,
    ) -> CUresult {
        (culib()
            .cuGraphExecExternalSemaphoresWaitNodeSetParams)(
            hGraphExec,
            hNode,
            nodeParams,
        )
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphExecGetFlags(
        hGraphExec: CUgraphExec,
        flags: *mut cuuint64_t,
    ) -> CUresult {
        (culib().cuGraphExecGetFlags)(hGraphExec, flags)
    }
    pub unsafe fn cuGraphExecHostNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        nodeParams: *const CUDA_HOST_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphExecHostNodeSetParams)(hGraphExec, hNode, nodeParams)
    }
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub unsafe fn cuGraphExecKernelNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphExecKernelNodeSetParams)(hGraphExec, hNode, nodeParams)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphExecKernelNodeSetParams_v2(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphExecKernelNodeSetParams_v2)(hGraphExec, hNode, nodeParams)
    }
    pub unsafe fn cuGraphExecMemcpyNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        copyParams: *const CUDA_MEMCPY3D,
        ctx: CUcontext,
    ) -> CUresult {
        (culib().cuGraphExecMemcpyNodeSetParams)(hGraphExec, hNode, copyParams, ctx)
    }
    pub unsafe fn cuGraphExecMemsetNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        memsetParams: *const CUDA_MEMSET_NODE_PARAMS,
        ctx: CUcontext,
    ) -> CUresult {
        (culib().cuGraphExecMemsetNodeSetParams)(hGraphExec, hNode, memsetParams, ctx)
    }
    #[cfg(
        any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphExecNodeSetParams(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        nodeParams: *mut CUgraphNodeParams,
    ) -> CUresult {
        (culib().cuGraphExecNodeSetParams)(hGraphExec, hNode, nodeParams)
    }
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub unsafe fn cuGraphExecUpdate(
        hGraphExec: CUgraphExec,
        hGraph: CUgraph,
        hErrorNode_out: *mut CUgraphNode,
        updateResult_out: *mut CUgraphExecUpdateResult,
    ) -> CUresult {
        (culib().cuGraphExecUpdate)(hGraphExec, hGraph, hErrorNode_out, updateResult_out)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphExecUpdate_v2(
        hGraphExec: CUgraphExec,
        hGraph: CUgraph,
        resultInfo: *mut CUgraphExecUpdateResultInfo,
    ) -> CUresult {
        (culib().cuGraphExecUpdate_v2)(hGraphExec, hGraph, resultInfo)
    }
    pub unsafe fn cuGraphExternalSemaphoresSignalNodeGetParams(
        hNode: CUgraphNode,
        params_out: *mut CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphExternalSemaphoresSignalNodeGetParams)(hNode, params_out)
    }
    pub unsafe fn cuGraphExternalSemaphoresSignalNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphExternalSemaphoresSignalNodeSetParams)(hNode, nodeParams)
    }
    pub unsafe fn cuGraphExternalSemaphoresWaitNodeGetParams(
        hNode: CUgraphNode,
        params_out: *mut CUDA_EXT_SEM_WAIT_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphExternalSemaphoresWaitNodeGetParams)(hNode, params_out)
    }
    pub unsafe fn cuGraphExternalSemaphoresWaitNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphExternalSemaphoresWaitNodeSetParams)(hNode, nodeParams)
    }
    pub unsafe fn cuGraphGetEdges(
        hGraph: CUgraph,
        from: *mut CUgraphNode,
        to: *mut CUgraphNode,
        numEdges: *mut usize,
    ) -> CUresult {
        (culib().cuGraphGetEdges)(hGraph, from, to, numEdges)
    }
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphGetEdges_v2(
        hGraph: CUgraph,
        from: *mut CUgraphNode,
        to: *mut CUgraphNode,
        edgeData: *mut CUgraphEdgeData,
        numEdges: *mut usize,
    ) -> CUresult {
        (culib().cuGraphGetEdges_v2)(hGraph, from, to, edgeData, numEdges)
    }
    pub unsafe fn cuGraphGetNodes(
        hGraph: CUgraph,
        nodes: *mut CUgraphNode,
        numNodes: *mut usize,
    ) -> CUresult {
        (culib().cuGraphGetNodes)(hGraph, nodes, numNodes)
    }
    pub unsafe fn cuGraphGetRootNodes(
        hGraph: CUgraph,
        rootNodes: *mut CUgraphNode,
        numRootNodes: *mut usize,
    ) -> CUresult {
        (culib().cuGraphGetRootNodes)(hGraph, rootNodes, numRootNodes)
    }
    pub unsafe fn cuGraphHostNodeGetParams(
        hNode: CUgraphNode,
        nodeParams: *mut CUDA_HOST_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphHostNodeGetParams)(hNode, nodeParams)
    }
    pub unsafe fn cuGraphHostNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_HOST_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphHostNodeSetParams)(hNode, nodeParams)
    }
    pub unsafe fn cuGraphInstantiateWithFlags(
        phGraphExec: *mut CUgraphExec,
        hGraph: CUgraph,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult {
        (culib().cuGraphInstantiateWithFlags)(phGraphExec, hGraph, flags)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphInstantiateWithParams(
        phGraphExec: *mut CUgraphExec,
        hGraph: CUgraph,
        instantiateParams: *mut CUDA_GRAPH_INSTANTIATE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphInstantiateWithParams)(phGraphExec, hGraph, instantiateParams)
    }
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub unsafe fn cuGraphInstantiate_v2(
        phGraphExec: *mut CUgraphExec,
        hGraph: CUgraph,
        phErrorNode: *mut CUgraphNode,
        logBuffer: *mut ::core::ffi::c_char,
        bufferSize: usize,
    ) -> CUresult {
        (culib()
            .cuGraphInstantiate_v2)(
            phGraphExec,
            hGraph,
            phErrorNode,
            logBuffer,
            bufferSize,
        )
    }
    pub unsafe fn cuGraphKernelNodeCopyAttributes(
        dst: CUgraphNode,
        src: CUgraphNode,
    ) -> CUresult {
        (culib().cuGraphKernelNodeCopyAttributes)(dst, src)
    }
    pub unsafe fn cuGraphKernelNodeGetAttribute(
        hNode: CUgraphNode,
        attr: CUkernelNodeAttrID,
        value_out: *mut CUkernelNodeAttrValue,
    ) -> CUresult {
        (culib().cuGraphKernelNodeGetAttribute)(hNode, attr, value_out)
    }
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub unsafe fn cuGraphKernelNodeGetParams(
        hNode: CUgraphNode,
        nodeParams: *mut CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphKernelNodeGetParams)(hNode, nodeParams)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphKernelNodeGetParams_v2(
        hNode: CUgraphNode,
        nodeParams: *mut CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphKernelNodeGetParams_v2)(hNode, nodeParams)
    }
    pub unsafe fn cuGraphKernelNodeSetAttribute(
        hNode: CUgraphNode,
        attr: CUkernelNodeAttrID,
        value: *const CUkernelNodeAttrValue,
    ) -> CUresult {
        (culib().cuGraphKernelNodeSetAttribute)(hNode, attr, value)
    }
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub unsafe fn cuGraphKernelNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphKernelNodeSetParams)(hNode, nodeParams)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphKernelNodeSetParams_v2(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphKernelNodeSetParams_v2)(hNode, nodeParams)
    }
    pub unsafe fn cuGraphLaunch(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult {
        (culib().cuGraphLaunch)(hGraphExec, hStream)
    }
    pub unsafe fn cuGraphMemAllocNodeGetParams(
        hNode: CUgraphNode,
        params_out: *mut CUDA_MEM_ALLOC_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphMemAllocNodeGetParams)(hNode, params_out)
    }
    pub unsafe fn cuGraphMemFreeNodeGetParams(
        hNode: CUgraphNode,
        dptr_out: *mut CUdeviceptr,
    ) -> CUresult {
        (culib().cuGraphMemFreeNodeGetParams)(hNode, dptr_out)
    }
    pub unsafe fn cuGraphMemcpyNodeGetParams(
        hNode: CUgraphNode,
        nodeParams: *mut CUDA_MEMCPY3D,
    ) -> CUresult {
        (culib().cuGraphMemcpyNodeGetParams)(hNode, nodeParams)
    }
    pub unsafe fn cuGraphMemcpyNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_MEMCPY3D,
    ) -> CUresult {
        (culib().cuGraphMemcpyNodeSetParams)(hNode, nodeParams)
    }
    pub unsafe fn cuGraphMemsetNodeGetParams(
        hNode: CUgraphNode,
        nodeParams: *mut CUDA_MEMSET_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphMemsetNodeGetParams)(hNode, nodeParams)
    }
    pub unsafe fn cuGraphMemsetNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *const CUDA_MEMSET_NODE_PARAMS,
    ) -> CUresult {
        (culib().cuGraphMemsetNodeSetParams)(hNode, nodeParams)
    }
    pub unsafe fn cuGraphNodeFindInClone(
        phNode: *mut CUgraphNode,
        hOriginalNode: CUgraphNode,
        hClonedGraph: CUgraph,
    ) -> CUresult {
        (culib().cuGraphNodeFindInClone)(phNode, hOriginalNode, hClonedGraph)
    }
    pub unsafe fn cuGraphNodeGetDependencies(
        hNode: CUgraphNode,
        dependencies: *mut CUgraphNode,
        numDependencies: *mut usize,
    ) -> CUresult {
        (culib().cuGraphNodeGetDependencies)(hNode, dependencies, numDependencies)
    }
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphNodeGetDependencies_v2(
        hNode: CUgraphNode,
        dependencies: *mut CUgraphNode,
        edgeData: *mut CUgraphEdgeData,
        numDependencies: *mut usize,
    ) -> CUresult {
        (culib()
            .cuGraphNodeGetDependencies_v2)(
            hNode,
            dependencies,
            edgeData,
            numDependencies,
        )
    }
    pub unsafe fn cuGraphNodeGetDependentNodes(
        hNode: CUgraphNode,
        dependentNodes: *mut CUgraphNode,
        numDependentNodes: *mut usize,
    ) -> CUresult {
        (culib().cuGraphNodeGetDependentNodes)(hNode, dependentNodes, numDependentNodes)
    }
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphNodeGetDependentNodes_v2(
        hNode: CUgraphNode,
        dependentNodes: *mut CUgraphNode,
        edgeData: *mut CUgraphEdgeData,
        numDependentNodes: *mut usize,
    ) -> CUresult {
        (culib()
            .cuGraphNodeGetDependentNodes_v2)(
            hNode,
            dependentNodes,
            edgeData,
            numDependentNodes,
        )
    }
    #[cfg(
        any(
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
        )
    )]
    pub unsafe fn cuGraphNodeGetEnabled(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        isEnabled: *mut ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuGraphNodeGetEnabled)(hGraphExec, hNode, isEnabled)
    }
    pub unsafe fn cuGraphNodeGetType(
        hNode: CUgraphNode,
        type_: *mut CUgraphNodeType,
    ) -> CUresult {
        (culib().cuGraphNodeGetType)(hNode, type_)
    }
    #[cfg(
        any(
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
        )
    )]
    pub unsafe fn cuGraphNodeSetEnabled(
        hGraphExec: CUgraphExec,
        hNode: CUgraphNode,
        isEnabled: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuGraphNodeSetEnabled)(hGraphExec, hNode, isEnabled)
    }
    #[cfg(
        any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphNodeSetParams(
        hNode: CUgraphNode,
        nodeParams: *mut CUgraphNodeParams,
    ) -> CUresult {
        (culib().cuGraphNodeSetParams)(hNode, nodeParams)
    }
    pub unsafe fn cuGraphReleaseUserObject(
        graph: CUgraph,
        object: CUuserObject,
        count: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuGraphReleaseUserObject)(graph, object, count)
    }
    pub unsafe fn cuGraphRemoveDependencies(
        hGraph: CUgraph,
        from: *const CUgraphNode,
        to: *const CUgraphNode,
        numDependencies: usize,
    ) -> CUresult {
        (culib().cuGraphRemoveDependencies)(hGraph, from, to, numDependencies)
    }
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGraphRemoveDependencies_v2(
        hGraph: CUgraph,
        from: *const CUgraphNode,
        to: *const CUgraphNode,
        edgeData: *const CUgraphEdgeData,
        numDependencies: usize,
    ) -> CUresult {
        (culib()
            .cuGraphRemoveDependencies_v2)(hGraph, from, to, edgeData, numDependencies)
    }
    pub unsafe fn cuGraphRetainUserObject(
        graph: CUgraph,
        object: CUuserObject,
        count: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuGraphRetainUserObject)(graph, object, count, flags)
    }
    pub unsafe fn cuGraphUpload(hGraphExec: CUgraphExec, hStream: CUstream) -> CUresult {
        (culib().cuGraphUpload)(hGraphExec, hStream)
    }
    pub unsafe fn cuGraphicsMapResources(
        count: ::core::ffi::c_uint,
        resources: *mut CUgraphicsResource,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuGraphicsMapResources)(count, resources, hStream)
    }
    pub unsafe fn cuGraphicsResourceGetMappedMipmappedArray(
        pMipmappedArray: *mut CUmipmappedArray,
        resource: CUgraphicsResource,
    ) -> CUresult {
        (culib().cuGraphicsResourceGetMappedMipmappedArray)(pMipmappedArray, resource)
    }
    pub unsafe fn cuGraphicsResourceGetMappedPointer_v2(
        pDevPtr: *mut CUdeviceptr,
        pSize: *mut usize,
        resource: CUgraphicsResource,
    ) -> CUresult {
        (culib().cuGraphicsResourceGetMappedPointer_v2)(pDevPtr, pSize, resource)
    }
    pub unsafe fn cuGraphicsResourceSetMapFlags_v2(
        resource: CUgraphicsResource,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuGraphicsResourceSetMapFlags_v2)(resource, flags)
    }
    pub unsafe fn cuGraphicsSubResourceGetMappedArray(
        pArray: *mut CUarray,
        resource: CUgraphicsResource,
        arrayIndex: ::core::ffi::c_uint,
        mipLevel: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib()
            .cuGraphicsSubResourceGetMappedArray)(pArray, resource, arrayIndex, mipLevel)
    }
    pub unsafe fn cuGraphicsUnmapResources(
        count: ::core::ffi::c_uint,
        resources: *mut CUgraphicsResource,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuGraphicsUnmapResources)(count, resources, hStream)
    }
    pub unsafe fn cuGraphicsUnregisterResource(
        resource: CUgraphicsResource,
    ) -> CUresult {
        (culib().cuGraphicsUnregisterResource)(resource)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGreenCtxCreate(
        phCtx: *mut CUgreenCtx,
        desc: CUdevResourceDesc,
        dev: CUdevice,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuGreenCtxCreate)(phCtx, desc, dev, flags)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGreenCtxDestroy(hCtx: CUgreenCtx) -> CUresult {
        (culib().cuGreenCtxDestroy)(hCtx)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGreenCtxGetDevResource(
        hCtx: CUgreenCtx,
        resource: *mut CUdevResource,
        type_: CUdevResourceType,
    ) -> CUresult {
        (culib().cuGreenCtxGetDevResource)(hCtx, resource, type_)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGreenCtxRecordEvent(hCtx: CUgreenCtx, hEvent: CUevent) -> CUresult {
        (culib().cuGreenCtxRecordEvent)(hCtx, hEvent)
    }
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub unsafe fn cuGreenCtxStreamCreate(
        phStream: *mut CUstream,
        greenCtx: CUgreenCtx,
        flags: ::core::ffi::c_uint,
        priority: ::core::ffi::c_int,
    ) -> CUresult {
        (culib().cuGreenCtxStreamCreate)(phStream, greenCtx, flags, priority)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuGreenCtxWaitEvent(hCtx: CUgreenCtx, hEvent: CUevent) -> CUresult {
        (culib().cuGreenCtxWaitEvent)(hCtx, hEvent)
    }
    pub unsafe fn cuImportExternalMemory(
        extMem_out: *mut CUexternalMemory,
        memHandleDesc: *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
    ) -> CUresult {
        (culib().cuImportExternalMemory)(extMem_out, memHandleDesc)
    }
    pub unsafe fn cuImportExternalSemaphore(
        extSem_out: *mut CUexternalSemaphore,
        semHandleDesc: *const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC,
    ) -> CUresult {
        (culib().cuImportExternalSemaphore)(extSem_out, semHandleDesc)
    }
    pub unsafe fn cuInit(Flags: ::core::ffi::c_uint) -> CUresult {
        (culib().cuInit)(Flags)
    }
    pub unsafe fn cuIpcCloseMemHandle(dptr: CUdeviceptr) -> CUresult {
        (culib().cuIpcCloseMemHandle)(dptr)
    }
    pub unsafe fn cuIpcGetEventHandle(
        pHandle: *mut CUipcEventHandle,
        event: CUevent,
    ) -> CUresult {
        (culib().cuIpcGetEventHandle)(pHandle, event)
    }
    pub unsafe fn cuIpcGetMemHandle(
        pHandle: *mut CUipcMemHandle,
        dptr: CUdeviceptr,
    ) -> CUresult {
        (culib().cuIpcGetMemHandle)(pHandle, dptr)
    }
    pub unsafe fn cuIpcOpenEventHandle(
        phEvent: *mut CUevent,
        handle: CUipcEventHandle,
    ) -> CUresult {
        (culib().cuIpcOpenEventHandle)(phEvent, handle)
    }
    pub unsafe fn cuIpcOpenMemHandle_v2(
        pdptr: *mut CUdeviceptr,
        handle: CUipcMemHandle,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuIpcOpenMemHandle_v2)(pdptr, handle, Flags)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuKernelGetAttribute(
        pi: *mut ::core::ffi::c_int,
        attrib: CUfunction_attribute,
        kernel: CUkernel,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuKernelGetAttribute)(pi, attrib, kernel, dev)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuKernelGetFunction(
        pFunc: *mut CUfunction,
        kernel: CUkernel,
    ) -> CUresult {
        (culib().cuKernelGetFunction)(pFunc, kernel)
    }
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub unsafe fn cuKernelGetLibrary(
        pLib: *mut CUlibrary,
        kernel: CUkernel,
    ) -> CUresult {
        (culib().cuKernelGetLibrary)(pLib, kernel)
    }
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuKernelGetName(
        name: *mut *const ::core::ffi::c_char,
        hfunc: CUkernel,
    ) -> CUresult {
        (culib().cuKernelGetName)(name, hfunc)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuKernelGetParamInfo(
        kernel: CUkernel,
        paramIndex: usize,
        paramOffset: *mut usize,
        paramSize: *mut usize,
    ) -> CUresult {
        (culib().cuKernelGetParamInfo)(kernel, paramIndex, paramOffset, paramSize)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuKernelSetAttribute(
        attrib: CUfunction_attribute,
        val: ::core::ffi::c_int,
        kernel: CUkernel,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuKernelSetAttribute)(attrib, val, kernel, dev)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuKernelSetCacheConfig(
        kernel: CUkernel,
        config: CUfunc_cache,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuKernelSetCacheConfig)(kernel, config, dev)
    }
    pub unsafe fn cuLaunch(f: CUfunction) -> CUresult {
        (culib().cuLaunch)(f)
    }
    pub unsafe fn cuLaunchCooperativeKernel(
        f: CUfunction,
        gridDimX: ::core::ffi::c_uint,
        gridDimY: ::core::ffi::c_uint,
        gridDimZ: ::core::ffi::c_uint,
        blockDimX: ::core::ffi::c_uint,
        blockDimY: ::core::ffi::c_uint,
        blockDimZ: ::core::ffi::c_uint,
        sharedMemBytes: ::core::ffi::c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut ::core::ffi::c_void,
    ) -> CUresult {
        (culib()
            .cuLaunchCooperativeKernel)(
            f,
            gridDimX,
            gridDimY,
            gridDimZ,
            blockDimX,
            blockDimY,
            blockDimZ,
            sharedMemBytes,
            hStream,
            kernelParams,
        )
    }
    pub unsafe fn cuLaunchCooperativeKernelMultiDevice(
        launchParamsList: *mut CUDA_LAUNCH_PARAMS,
        numDevices: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib()
            .cuLaunchCooperativeKernelMultiDevice)(launchParamsList, numDevices, flags)
    }
    pub unsafe fn cuLaunchGrid(
        f: CUfunction,
        grid_width: ::core::ffi::c_int,
        grid_height: ::core::ffi::c_int,
    ) -> CUresult {
        (culib().cuLaunchGrid)(f, grid_width, grid_height)
    }
    pub unsafe fn cuLaunchGridAsync(
        f: CUfunction,
        grid_width: ::core::ffi::c_int,
        grid_height: ::core::ffi::c_int,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuLaunchGridAsync)(f, grid_width, grid_height, hStream)
    }
    pub unsafe fn cuLaunchHostFunc(
        hStream: CUstream,
        fn_: CUhostFn,
        userData: *mut ::core::ffi::c_void,
    ) -> CUresult {
        (culib().cuLaunchHostFunc)(hStream, fn_, userData)
    }
    pub unsafe fn cuLaunchKernel(
        f: CUfunction,
        gridDimX: ::core::ffi::c_uint,
        gridDimY: ::core::ffi::c_uint,
        gridDimZ: ::core::ffi::c_uint,
        blockDimX: ::core::ffi::c_uint,
        blockDimY: ::core::ffi::c_uint,
        blockDimZ: ::core::ffi::c_uint,
        sharedMemBytes: ::core::ffi::c_uint,
        hStream: CUstream,
        kernelParams: *mut *mut ::core::ffi::c_void,
        extra: *mut *mut ::core::ffi::c_void,
    ) -> CUresult {
        (culib()
            .cuLaunchKernel)(
            f,
            gridDimX,
            gridDimY,
            gridDimZ,
            blockDimX,
            blockDimY,
            blockDimZ,
            sharedMemBytes,
            hStream,
            kernelParams,
            extra,
        )
    }
    #[cfg(
        any(
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuLaunchKernelEx(
        config: *const CUlaunchConfig,
        f: CUfunction,
        kernelParams: *mut *mut ::core::ffi::c_void,
        extra: *mut *mut ::core::ffi::c_void,
    ) -> CUresult {
        (culib().cuLaunchKernelEx)(config, f, kernelParams, extra)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuLibraryEnumerateKernels(
        kernels: *mut CUkernel,
        numKernels: ::core::ffi::c_uint,
        lib: CUlibrary,
    ) -> CUresult {
        (culib().cuLibraryEnumerateKernels)(kernels, numKernels, lib)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuLibraryGetGlobal(
        dptr: *mut CUdeviceptr,
        bytes: *mut usize,
        library: CUlibrary,
        name: *const ::core::ffi::c_char,
    ) -> CUresult {
        (culib().cuLibraryGetGlobal)(dptr, bytes, library, name)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuLibraryGetKernel(
        pKernel: *mut CUkernel,
        library: CUlibrary,
        name: *const ::core::ffi::c_char,
    ) -> CUresult {
        (culib().cuLibraryGetKernel)(pKernel, library, name)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuLibraryGetKernelCount(
        count: *mut ::core::ffi::c_uint,
        lib: CUlibrary,
    ) -> CUresult {
        (culib().cuLibraryGetKernelCount)(count, lib)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuLibraryGetManaged(
        dptr: *mut CUdeviceptr,
        bytes: *mut usize,
        library: CUlibrary,
        name: *const ::core::ffi::c_char,
    ) -> CUresult {
        (culib().cuLibraryGetManaged)(dptr, bytes, library, name)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuLibraryGetModule(
        pMod: *mut CUmodule,
        library: CUlibrary,
    ) -> CUresult {
        (culib().cuLibraryGetModule)(pMod, library)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuLibraryGetUnifiedFunction(
        fptr: *mut *mut ::core::ffi::c_void,
        library: CUlibrary,
        symbol: *const ::core::ffi::c_char,
    ) -> CUresult {
        (culib().cuLibraryGetUnifiedFunction)(fptr, library, symbol)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuLibraryLoadData(
        library: *mut CUlibrary,
        code: *const ::core::ffi::c_void,
        jitOptions: *mut CUjit_option,
        jitOptionsValues: *mut *mut ::core::ffi::c_void,
        numJitOptions: ::core::ffi::c_uint,
        libraryOptions: *mut CUlibraryOption,
        libraryOptionValues: *mut *mut ::core::ffi::c_void,
        numLibraryOptions: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib()
            .cuLibraryLoadData)(
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
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuLibraryLoadFromFile(
        library: *mut CUlibrary,
        fileName: *const ::core::ffi::c_char,
        jitOptions: *mut CUjit_option,
        jitOptionsValues: *mut *mut ::core::ffi::c_void,
        numJitOptions: ::core::ffi::c_uint,
        libraryOptions: *mut CUlibraryOption,
        libraryOptionValues: *mut *mut ::core::ffi::c_void,
        numLibraryOptions: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib()
            .cuLibraryLoadFromFile)(
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
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuLibraryUnload(library: CUlibrary) -> CUresult {
        (culib().cuLibraryUnload)(library)
    }
    pub unsafe fn cuLinkAddData_v2(
        state: CUlinkState,
        type_: CUjitInputType,
        data: *mut ::core::ffi::c_void,
        size: usize,
        name: *const ::core::ffi::c_char,
        numOptions: ::core::ffi::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::core::ffi::c_void,
    ) -> CUresult {
        (culib()
            .cuLinkAddData_v2)(
            state,
            type_,
            data,
            size,
            name,
            numOptions,
            options,
            optionValues,
        )
    }
    pub unsafe fn cuLinkAddFile_v2(
        state: CUlinkState,
        type_: CUjitInputType,
        path: *const ::core::ffi::c_char,
        numOptions: ::core::ffi::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::core::ffi::c_void,
    ) -> CUresult {
        (culib().cuLinkAddFile_v2)(state, type_, path, numOptions, options, optionValues)
    }
    pub unsafe fn cuLinkComplete(
        state: CUlinkState,
        cubinOut: *mut *mut ::core::ffi::c_void,
        sizeOut: *mut usize,
    ) -> CUresult {
        (culib().cuLinkComplete)(state, cubinOut, sizeOut)
    }
    pub unsafe fn cuLinkCreate_v2(
        numOptions: ::core::ffi::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::core::ffi::c_void,
        stateOut: *mut CUlinkState,
    ) -> CUresult {
        (culib().cuLinkCreate_v2)(numOptions, options, optionValues, stateOut)
    }
    pub unsafe fn cuLinkDestroy(state: CUlinkState) -> CUresult {
        (culib().cuLinkDestroy)(state)
    }
    pub unsafe fn cuMemAddressFree(ptr: CUdeviceptr, size: usize) -> CUresult {
        (culib().cuMemAddressFree)(ptr, size)
    }
    pub unsafe fn cuMemAddressReserve(
        ptr: *mut CUdeviceptr,
        size: usize,
        alignment: usize,
        addr: CUdeviceptr,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult {
        (culib().cuMemAddressReserve)(ptr, size, alignment, addr, flags)
    }
    pub unsafe fn cuMemAdvise(
        devPtr: CUdeviceptr,
        count: usize,
        advice: CUmem_advise,
        device: CUdevice,
    ) -> CUresult {
        (culib().cuMemAdvise)(devPtr, count, advice, device)
    }
    #[cfg(
        any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuMemAdvise_v2(
        devPtr: CUdeviceptr,
        count: usize,
        advice: CUmem_advise,
        location: CUmemLocation,
    ) -> CUresult {
        (culib().cuMemAdvise_v2)(devPtr, count, advice, location)
    }
    pub unsafe fn cuMemAllocAsync(
        dptr: *mut CUdeviceptr,
        bytesize: usize,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemAllocAsync)(dptr, bytesize, hStream)
    }
    pub unsafe fn cuMemAllocFromPoolAsync(
        dptr: *mut CUdeviceptr,
        bytesize: usize,
        pool: CUmemoryPool,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemAllocFromPoolAsync)(dptr, bytesize, pool, hStream)
    }
    pub unsafe fn cuMemAllocHost_v2(
        pp: *mut *mut ::core::ffi::c_void,
        bytesize: usize,
    ) -> CUresult {
        (culib().cuMemAllocHost_v2)(pp, bytesize)
    }
    pub unsafe fn cuMemAllocManaged(
        dptr: *mut CUdeviceptr,
        bytesize: usize,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuMemAllocManaged)(dptr, bytesize, flags)
    }
    pub unsafe fn cuMemAllocPitch_v2(
        dptr: *mut CUdeviceptr,
        pPitch: *mut usize,
        WidthInBytes: usize,
        Height: usize,
        ElementSizeBytes: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib()
            .cuMemAllocPitch_v2)(dptr, pPitch, WidthInBytes, Height, ElementSizeBytes)
    }
    pub unsafe fn cuMemAlloc_v2(dptr: *mut CUdeviceptr, bytesize: usize) -> CUresult {
        (culib().cuMemAlloc_v2)(dptr, bytesize)
    }
    #[cfg(any(feature = "cuda-12080"))]
    pub unsafe fn cuMemBatchDecompressAsync(
        paramsArray: *mut CUmemDecompressParams,
        count: usize,
        flags: ::core::ffi::c_uint,
        errorIndex: *mut usize,
        stream: CUstream,
    ) -> CUresult {
        (culib()
            .cuMemBatchDecompressAsync)(paramsArray, count, flags, errorIndex, stream)
    }
    pub unsafe fn cuMemCreate(
        handle: *mut CUmemGenericAllocationHandle,
        size: usize,
        prop: *const CUmemAllocationProp,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult {
        (culib().cuMemCreate)(handle, size, prop, flags)
    }
    pub unsafe fn cuMemExportToShareableHandle(
        shareableHandle: *mut ::core::ffi::c_void,
        handle: CUmemGenericAllocationHandle,
        handleType: CUmemAllocationHandleType,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult {
        (culib()
            .cuMemExportToShareableHandle)(shareableHandle, handle, handleType, flags)
    }
    pub unsafe fn cuMemFreeAsync(dptr: CUdeviceptr, hStream: CUstream) -> CUresult {
        (culib().cuMemFreeAsync)(dptr, hStream)
    }
    pub unsafe fn cuMemFreeHost(p: *mut ::core::ffi::c_void) -> CUresult {
        (culib().cuMemFreeHost)(p)
    }
    pub unsafe fn cuMemFree_v2(dptr: CUdeviceptr) -> CUresult {
        (culib().cuMemFree_v2)(dptr)
    }
    pub unsafe fn cuMemGetAccess(
        flags: *mut ::core::ffi::c_ulonglong,
        location: *const CUmemLocation,
        ptr: CUdeviceptr,
    ) -> CUresult {
        (culib().cuMemGetAccess)(flags, location, ptr)
    }
    pub unsafe fn cuMemGetAddressRange_v2(
        pbase: *mut CUdeviceptr,
        psize: *mut usize,
        dptr: CUdeviceptr,
    ) -> CUresult {
        (culib().cuMemGetAddressRange_v2)(pbase, psize, dptr)
    }
    pub unsafe fn cuMemGetAllocationGranularity(
        granularity: *mut usize,
        prop: *const CUmemAllocationProp,
        option: CUmemAllocationGranularity_flags,
    ) -> CUresult {
        (culib().cuMemGetAllocationGranularity)(granularity, prop, option)
    }
    pub unsafe fn cuMemGetAllocationPropertiesFromHandle(
        prop: *mut CUmemAllocationProp,
        handle: CUmemGenericAllocationHandle,
    ) -> CUresult {
        (culib().cuMemGetAllocationPropertiesFromHandle)(prop, handle)
    }
    #[cfg(
        any(
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
        )
    )]
    pub unsafe fn cuMemGetHandleForAddressRange(
        handle: *mut ::core::ffi::c_void,
        dptr: CUdeviceptr,
        size: usize,
        handleType: CUmemRangeHandleType,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult {
        (culib().cuMemGetHandleForAddressRange)(handle, dptr, size, handleType, flags)
    }
    pub unsafe fn cuMemGetInfo_v2(free: *mut usize, total: *mut usize) -> CUresult {
        (culib().cuMemGetInfo_v2)(free, total)
    }
    pub unsafe fn cuMemHostAlloc(
        pp: *mut *mut ::core::ffi::c_void,
        bytesize: usize,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuMemHostAlloc)(pp, bytesize, Flags)
    }
    pub unsafe fn cuMemHostGetDevicePointer_v2(
        pdptr: *mut CUdeviceptr,
        p: *mut ::core::ffi::c_void,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuMemHostGetDevicePointer_v2)(pdptr, p, Flags)
    }
    pub unsafe fn cuMemHostGetFlags(
        pFlags: *mut ::core::ffi::c_uint,
        p: *mut ::core::ffi::c_void,
    ) -> CUresult {
        (culib().cuMemHostGetFlags)(pFlags, p)
    }
    pub unsafe fn cuMemHostRegister_v2(
        p: *mut ::core::ffi::c_void,
        bytesize: usize,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuMemHostRegister_v2)(p, bytesize, Flags)
    }
    pub unsafe fn cuMemHostUnregister(p: *mut ::core::ffi::c_void) -> CUresult {
        (culib().cuMemHostUnregister)(p)
    }
    pub unsafe fn cuMemImportFromShareableHandle(
        handle: *mut CUmemGenericAllocationHandle,
        osHandle: *mut ::core::ffi::c_void,
        shHandleType: CUmemAllocationHandleType,
    ) -> CUresult {
        (culib().cuMemImportFromShareableHandle)(handle, osHandle, shHandleType)
    }
    pub unsafe fn cuMemMap(
        ptr: CUdeviceptr,
        size: usize,
        offset: usize,
        handle: CUmemGenericAllocationHandle,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult {
        (culib().cuMemMap)(ptr, size, offset, handle, flags)
    }
    pub unsafe fn cuMemMapArrayAsync(
        mapInfoList: *mut CUarrayMapInfo,
        count: ::core::ffi::c_uint,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemMapArrayAsync)(mapInfoList, count, hStream)
    }
    pub unsafe fn cuMemPoolCreate(
        pool: *mut CUmemoryPool,
        poolProps: *const CUmemPoolProps,
    ) -> CUresult {
        (culib().cuMemPoolCreate)(pool, poolProps)
    }
    pub unsafe fn cuMemPoolDestroy(pool: CUmemoryPool) -> CUresult {
        (culib().cuMemPoolDestroy)(pool)
    }
    pub unsafe fn cuMemPoolExportPointer(
        shareData_out: *mut CUmemPoolPtrExportData,
        ptr: CUdeviceptr,
    ) -> CUresult {
        (culib().cuMemPoolExportPointer)(shareData_out, ptr)
    }
    pub unsafe fn cuMemPoolExportToShareableHandle(
        handle_out: *mut ::core::ffi::c_void,
        pool: CUmemoryPool,
        handleType: CUmemAllocationHandleType,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult {
        (culib().cuMemPoolExportToShareableHandle)(handle_out, pool, handleType, flags)
    }
    pub unsafe fn cuMemPoolGetAccess(
        flags: *mut CUmemAccess_flags,
        memPool: CUmemoryPool,
        location: *mut CUmemLocation,
    ) -> CUresult {
        (culib().cuMemPoolGetAccess)(flags, memPool, location)
    }
    pub unsafe fn cuMemPoolGetAttribute(
        pool: CUmemoryPool,
        attr: CUmemPool_attribute,
        value: *mut ::core::ffi::c_void,
    ) -> CUresult {
        (culib().cuMemPoolGetAttribute)(pool, attr, value)
    }
    pub unsafe fn cuMemPoolImportFromShareableHandle(
        pool_out: *mut CUmemoryPool,
        handle: *mut ::core::ffi::c_void,
        handleType: CUmemAllocationHandleType,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult {
        (culib().cuMemPoolImportFromShareableHandle)(pool_out, handle, handleType, flags)
    }
    pub unsafe fn cuMemPoolImportPointer(
        ptr_out: *mut CUdeviceptr,
        pool: CUmemoryPool,
        shareData: *mut CUmemPoolPtrExportData,
    ) -> CUresult {
        (culib().cuMemPoolImportPointer)(ptr_out, pool, shareData)
    }
    pub unsafe fn cuMemPoolSetAccess(
        pool: CUmemoryPool,
        map: *const CUmemAccessDesc,
        count: usize,
    ) -> CUresult {
        (culib().cuMemPoolSetAccess)(pool, map, count)
    }
    pub unsafe fn cuMemPoolSetAttribute(
        pool: CUmemoryPool,
        attr: CUmemPool_attribute,
        value: *mut ::core::ffi::c_void,
    ) -> CUresult {
        (culib().cuMemPoolSetAttribute)(pool, attr, value)
    }
    pub unsafe fn cuMemPoolTrimTo(
        pool: CUmemoryPool,
        minBytesToKeep: usize,
    ) -> CUresult {
        (culib().cuMemPoolTrimTo)(pool, minBytesToKeep)
    }
    pub unsafe fn cuMemPrefetchAsync(
        devPtr: CUdeviceptr,
        count: usize,
        dstDevice: CUdevice,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemPrefetchAsync)(devPtr, count, dstDevice, hStream)
    }
    #[cfg(
        any(
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuMemPrefetchAsync_v2(
        devPtr: CUdeviceptr,
        count: usize,
        location: CUmemLocation,
        flags: ::core::ffi::c_uint,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemPrefetchAsync_v2)(devPtr, count, location, flags, hStream)
    }
    pub unsafe fn cuMemRangeGetAttribute(
        data: *mut ::core::ffi::c_void,
        dataSize: usize,
        attribute: CUmem_range_attribute,
        devPtr: CUdeviceptr,
        count: usize,
    ) -> CUresult {
        (culib().cuMemRangeGetAttribute)(data, dataSize, attribute, devPtr, count)
    }
    pub unsafe fn cuMemRangeGetAttributes(
        data: *mut *mut ::core::ffi::c_void,
        dataSizes: *mut usize,
        attributes: *mut CUmem_range_attribute,
        numAttributes: usize,
        devPtr: CUdeviceptr,
        count: usize,
    ) -> CUresult {
        (culib()
            .cuMemRangeGetAttributes)(
            data,
            dataSizes,
            attributes,
            numAttributes,
            devPtr,
            count,
        )
    }
    pub unsafe fn cuMemRelease(handle: CUmemGenericAllocationHandle) -> CUresult {
        (culib().cuMemRelease)(handle)
    }
    pub unsafe fn cuMemRetainAllocationHandle(
        handle: *mut CUmemGenericAllocationHandle,
        addr: *mut ::core::ffi::c_void,
    ) -> CUresult {
        (culib().cuMemRetainAllocationHandle)(handle, addr)
    }
    pub unsafe fn cuMemSetAccess(
        ptr: CUdeviceptr,
        size: usize,
        desc: *const CUmemAccessDesc,
        count: usize,
    ) -> CUresult {
        (culib().cuMemSetAccess)(ptr, size, desc, count)
    }
    pub unsafe fn cuMemUnmap(ptr: CUdeviceptr, size: usize) -> CUresult {
        (culib().cuMemUnmap)(ptr, size)
    }
    pub unsafe fn cuMemcpy(
        dst: CUdeviceptr,
        src: CUdeviceptr,
        ByteCount: usize,
    ) -> CUresult {
        (culib().cuMemcpy)(dst, src, ByteCount)
    }
    pub unsafe fn cuMemcpy2DAsync_v2(
        pCopy: *const CUDA_MEMCPY2D,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemcpy2DAsync_v2)(pCopy, hStream)
    }
    pub unsafe fn cuMemcpy2DUnaligned_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult {
        (culib().cuMemcpy2DUnaligned_v2)(pCopy)
    }
    pub unsafe fn cuMemcpy2D_v2(pCopy: *const CUDA_MEMCPY2D) -> CUresult {
        (culib().cuMemcpy2D_v2)(pCopy)
    }
    pub unsafe fn cuMemcpy3DAsync_v2(
        pCopy: *const CUDA_MEMCPY3D,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemcpy3DAsync_v2)(pCopy, hStream)
    }
    #[cfg(any(feature = "cuda-12080"))]
    pub unsafe fn cuMemcpy3DBatchAsync(
        numOps: usize,
        opList: *mut CUDA_MEMCPY3D_BATCH_OP,
        failIdx: *mut usize,
        flags: ::core::ffi::c_ulonglong,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemcpy3DBatchAsync)(numOps, opList, failIdx, flags, hStream)
    }
    pub unsafe fn cuMemcpy3DPeer(pCopy: *const CUDA_MEMCPY3D_PEER) -> CUresult {
        (culib().cuMemcpy3DPeer)(pCopy)
    }
    pub unsafe fn cuMemcpy3DPeerAsync(
        pCopy: *const CUDA_MEMCPY3D_PEER,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemcpy3DPeerAsync)(pCopy, hStream)
    }
    pub unsafe fn cuMemcpy3D_v2(pCopy: *const CUDA_MEMCPY3D) -> CUresult {
        (culib().cuMemcpy3D_v2)(pCopy)
    }
    pub unsafe fn cuMemcpyAsync(
        dst: CUdeviceptr,
        src: CUdeviceptr,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemcpyAsync)(dst, src, ByteCount, hStream)
    }
    pub unsafe fn cuMemcpyAtoA_v2(
        dstArray: CUarray,
        dstOffset: usize,
        srcArray: CUarray,
        srcOffset: usize,
        ByteCount: usize,
    ) -> CUresult {
        (culib().cuMemcpyAtoA_v2)(dstArray, dstOffset, srcArray, srcOffset, ByteCount)
    }
    pub unsafe fn cuMemcpyAtoD_v2(
        dstDevice: CUdeviceptr,
        srcArray: CUarray,
        srcOffset: usize,
        ByteCount: usize,
    ) -> CUresult {
        (culib().cuMemcpyAtoD_v2)(dstDevice, srcArray, srcOffset, ByteCount)
    }
    pub unsafe fn cuMemcpyAtoHAsync_v2(
        dstHost: *mut ::core::ffi::c_void,
        srcArray: CUarray,
        srcOffset: usize,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemcpyAtoHAsync_v2)(dstHost, srcArray, srcOffset, ByteCount, hStream)
    }
    pub unsafe fn cuMemcpyAtoH_v2(
        dstHost: *mut ::core::ffi::c_void,
        srcArray: CUarray,
        srcOffset: usize,
        ByteCount: usize,
    ) -> CUresult {
        (culib().cuMemcpyAtoH_v2)(dstHost, srcArray, srcOffset, ByteCount)
    }
    #[cfg(any(feature = "cuda-12080"))]
    pub unsafe fn cuMemcpyBatchAsync(
        dsts: *mut CUdeviceptr,
        srcs: *mut CUdeviceptr,
        sizes: *mut usize,
        count: usize,
        attrs: *mut CUmemcpyAttributes,
        attrsIdxs: *mut usize,
        numAttrs: usize,
        failIdx: *mut usize,
        hStream: CUstream,
    ) -> CUresult {
        (culib()
            .cuMemcpyBatchAsync)(
            dsts,
            srcs,
            sizes,
            count,
            attrs,
            attrsIdxs,
            numAttrs,
            failIdx,
            hStream,
        )
    }
    pub unsafe fn cuMemcpyDtoA_v2(
        dstArray: CUarray,
        dstOffset: usize,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
    ) -> CUresult {
        (culib().cuMemcpyDtoA_v2)(dstArray, dstOffset, srcDevice, ByteCount)
    }
    pub unsafe fn cuMemcpyDtoDAsync_v2(
        dstDevice: CUdeviceptr,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemcpyDtoDAsync_v2)(dstDevice, srcDevice, ByteCount, hStream)
    }
    pub unsafe fn cuMemcpyDtoD_v2(
        dstDevice: CUdeviceptr,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
    ) -> CUresult {
        (culib().cuMemcpyDtoD_v2)(dstDevice, srcDevice, ByteCount)
    }
    pub unsafe fn cuMemcpyDtoHAsync_v2(
        dstHost: *mut ::core::ffi::c_void,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemcpyDtoHAsync_v2)(dstHost, srcDevice, ByteCount, hStream)
    }
    pub unsafe fn cuMemcpyDtoH_v2(
        dstHost: *mut ::core::ffi::c_void,
        srcDevice: CUdeviceptr,
        ByteCount: usize,
    ) -> CUresult {
        (culib().cuMemcpyDtoH_v2)(dstHost, srcDevice, ByteCount)
    }
    pub unsafe fn cuMemcpyHtoAAsync_v2(
        dstArray: CUarray,
        dstOffset: usize,
        srcHost: *const ::core::ffi::c_void,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemcpyHtoAAsync_v2)(dstArray, dstOffset, srcHost, ByteCount, hStream)
    }
    pub unsafe fn cuMemcpyHtoA_v2(
        dstArray: CUarray,
        dstOffset: usize,
        srcHost: *const ::core::ffi::c_void,
        ByteCount: usize,
    ) -> CUresult {
        (culib().cuMemcpyHtoA_v2)(dstArray, dstOffset, srcHost, ByteCount)
    }
    pub unsafe fn cuMemcpyHtoDAsync_v2(
        dstDevice: CUdeviceptr,
        srcHost: *const ::core::ffi::c_void,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemcpyHtoDAsync_v2)(dstDevice, srcHost, ByteCount, hStream)
    }
    pub unsafe fn cuMemcpyHtoD_v2(
        dstDevice: CUdeviceptr,
        srcHost: *const ::core::ffi::c_void,
        ByteCount: usize,
    ) -> CUresult {
        (culib().cuMemcpyHtoD_v2)(dstDevice, srcHost, ByteCount)
    }
    pub unsafe fn cuMemcpyPeer(
        dstDevice: CUdeviceptr,
        dstContext: CUcontext,
        srcDevice: CUdeviceptr,
        srcContext: CUcontext,
        ByteCount: usize,
    ) -> CUresult {
        (culib().cuMemcpyPeer)(dstDevice, dstContext, srcDevice, srcContext, ByteCount)
    }
    pub unsafe fn cuMemcpyPeerAsync(
        dstDevice: CUdeviceptr,
        dstContext: CUcontext,
        srcDevice: CUdeviceptr,
        srcContext: CUcontext,
        ByteCount: usize,
        hStream: CUstream,
    ) -> CUresult {
        (culib()
            .cuMemcpyPeerAsync)(
            dstDevice,
            dstContext,
            srcDevice,
            srcContext,
            ByteCount,
            hStream,
        )
    }
    pub unsafe fn cuMemsetD16Async(
        dstDevice: CUdeviceptr,
        us: ::core::ffi::c_ushort,
        N: usize,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemsetD16Async)(dstDevice, us, N, hStream)
    }
    pub unsafe fn cuMemsetD16_v2(
        dstDevice: CUdeviceptr,
        us: ::core::ffi::c_ushort,
        N: usize,
    ) -> CUresult {
        (culib().cuMemsetD16_v2)(dstDevice, us, N)
    }
    pub unsafe fn cuMemsetD2D16Async(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        us: ::core::ffi::c_ushort,
        Width: usize,
        Height: usize,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemsetD2D16Async)(dstDevice, dstPitch, us, Width, Height, hStream)
    }
    pub unsafe fn cuMemsetD2D16_v2(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        us: ::core::ffi::c_ushort,
        Width: usize,
        Height: usize,
    ) -> CUresult {
        (culib().cuMemsetD2D16_v2)(dstDevice, dstPitch, us, Width, Height)
    }
    pub unsafe fn cuMemsetD2D32Async(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        ui: ::core::ffi::c_uint,
        Width: usize,
        Height: usize,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemsetD2D32Async)(dstDevice, dstPitch, ui, Width, Height, hStream)
    }
    pub unsafe fn cuMemsetD2D32_v2(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        ui: ::core::ffi::c_uint,
        Width: usize,
        Height: usize,
    ) -> CUresult {
        (culib().cuMemsetD2D32_v2)(dstDevice, dstPitch, ui, Width, Height)
    }
    pub unsafe fn cuMemsetD2D8Async(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        uc: ::core::ffi::c_uchar,
        Width: usize,
        Height: usize,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemsetD2D8Async)(dstDevice, dstPitch, uc, Width, Height, hStream)
    }
    pub unsafe fn cuMemsetD2D8_v2(
        dstDevice: CUdeviceptr,
        dstPitch: usize,
        uc: ::core::ffi::c_uchar,
        Width: usize,
        Height: usize,
    ) -> CUresult {
        (culib().cuMemsetD2D8_v2)(dstDevice, dstPitch, uc, Width, Height)
    }
    pub unsafe fn cuMemsetD32Async(
        dstDevice: CUdeviceptr,
        ui: ::core::ffi::c_uint,
        N: usize,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemsetD32Async)(dstDevice, ui, N, hStream)
    }
    pub unsafe fn cuMemsetD32_v2(
        dstDevice: CUdeviceptr,
        ui: ::core::ffi::c_uint,
        N: usize,
    ) -> CUresult {
        (culib().cuMemsetD32_v2)(dstDevice, ui, N)
    }
    pub unsafe fn cuMemsetD8Async(
        dstDevice: CUdeviceptr,
        uc: ::core::ffi::c_uchar,
        N: usize,
        hStream: CUstream,
    ) -> CUresult {
        (culib().cuMemsetD8Async)(dstDevice, uc, N, hStream)
    }
    pub unsafe fn cuMemsetD8_v2(
        dstDevice: CUdeviceptr,
        uc: ::core::ffi::c_uchar,
        N: usize,
    ) -> CUresult {
        (culib().cuMemsetD8_v2)(dstDevice, uc, N)
    }
    pub unsafe fn cuMipmappedArrayCreate(
        pHandle: *mut CUmipmappedArray,
        pMipmappedArrayDesc: *const CUDA_ARRAY3D_DESCRIPTOR,
        numMipmapLevels: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuMipmappedArrayCreate)(pHandle, pMipmappedArrayDesc, numMipmapLevels)
    }
    pub unsafe fn cuMipmappedArrayDestroy(
        hMipmappedArray: CUmipmappedArray,
    ) -> CUresult {
        (culib().cuMipmappedArrayDestroy)(hMipmappedArray)
    }
    pub unsafe fn cuMipmappedArrayGetLevel(
        pLevelArray: *mut CUarray,
        hMipmappedArray: CUmipmappedArray,
        level: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuMipmappedArrayGetLevel)(pLevelArray, hMipmappedArray, level)
    }
    #[cfg(
        any(
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
        )
    )]
    pub unsafe fn cuMipmappedArrayGetMemoryRequirements(
        memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS,
        mipmap: CUmipmappedArray,
        device: CUdevice,
    ) -> CUresult {
        (culib()
            .cuMipmappedArrayGetMemoryRequirements)(memoryRequirements, mipmap, device)
    }
    pub unsafe fn cuMipmappedArrayGetSparseProperties(
        sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES,
        mipmap: CUmipmappedArray,
    ) -> CUresult {
        (culib().cuMipmappedArrayGetSparseProperties)(sparseProperties, mipmap)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuModuleEnumerateFunctions(
        functions: *mut CUfunction,
        numFunctions: ::core::ffi::c_uint,
        mod_: CUmodule,
    ) -> CUresult {
        (culib().cuModuleEnumerateFunctions)(functions, numFunctions, mod_)
    }
    pub unsafe fn cuModuleGetFunction(
        hfunc: *mut CUfunction,
        hmod: CUmodule,
        name: *const ::core::ffi::c_char,
    ) -> CUresult {
        (culib().cuModuleGetFunction)(hfunc, hmod, name)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuModuleGetFunctionCount(
        count: *mut ::core::ffi::c_uint,
        mod_: CUmodule,
    ) -> CUresult {
        (culib().cuModuleGetFunctionCount)(count, mod_)
    }
    pub unsafe fn cuModuleGetGlobal_v2(
        dptr: *mut CUdeviceptr,
        bytes: *mut usize,
        hmod: CUmodule,
        name: *const ::core::ffi::c_char,
    ) -> CUresult {
        (culib().cuModuleGetGlobal_v2)(dptr, bytes, hmod, name)
    }
    #[cfg(
        any(
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
        )
    )]
    pub unsafe fn cuModuleGetLoadingMode(mode: *mut CUmoduleLoadingMode) -> CUresult {
        (culib().cuModuleGetLoadingMode)(mode)
    }
    pub unsafe fn cuModuleGetSurfRef(
        pSurfRef: *mut CUsurfref,
        hmod: CUmodule,
        name: *const ::core::ffi::c_char,
    ) -> CUresult {
        (culib().cuModuleGetSurfRef)(pSurfRef, hmod, name)
    }
    pub unsafe fn cuModuleGetTexRef(
        pTexRef: *mut CUtexref,
        hmod: CUmodule,
        name: *const ::core::ffi::c_char,
    ) -> CUresult {
        (culib().cuModuleGetTexRef)(pTexRef, hmod, name)
    }
    pub unsafe fn cuModuleLoad(
        module: *mut CUmodule,
        fname: *const ::core::ffi::c_char,
    ) -> CUresult {
        (culib().cuModuleLoad)(module, fname)
    }
    pub unsafe fn cuModuleLoadData(
        module: *mut CUmodule,
        image: *const ::core::ffi::c_void,
    ) -> CUresult {
        (culib().cuModuleLoadData)(module, image)
    }
    pub unsafe fn cuModuleLoadDataEx(
        module: *mut CUmodule,
        image: *const ::core::ffi::c_void,
        numOptions: ::core::ffi::c_uint,
        options: *mut CUjit_option,
        optionValues: *mut *mut ::core::ffi::c_void,
    ) -> CUresult {
        (culib().cuModuleLoadDataEx)(module, image, numOptions, options, optionValues)
    }
    pub unsafe fn cuModuleLoadFatBinary(
        module: *mut CUmodule,
        fatCubin: *const ::core::ffi::c_void,
    ) -> CUresult {
        (culib().cuModuleLoadFatBinary)(module, fatCubin)
    }
    pub unsafe fn cuModuleUnload(hmod: CUmodule) -> CUresult {
        (culib().cuModuleUnload)(hmod)
    }
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuMulticastAddDevice(
        mcHandle: CUmemGenericAllocationHandle,
        dev: CUdevice,
    ) -> CUresult {
        (culib().cuMulticastAddDevice)(mcHandle, dev)
    }
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuMulticastBindAddr(
        mcHandle: CUmemGenericAllocationHandle,
        mcOffset: usize,
        memptr: CUdeviceptr,
        size: usize,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult {
        (culib().cuMulticastBindAddr)(mcHandle, mcOffset, memptr, size, flags)
    }
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuMulticastBindMem(
        mcHandle: CUmemGenericAllocationHandle,
        mcOffset: usize,
        memHandle: CUmemGenericAllocationHandle,
        memOffset: usize,
        size: usize,
        flags: ::core::ffi::c_ulonglong,
    ) -> CUresult {
        (culib()
            .cuMulticastBindMem)(mcHandle, mcOffset, memHandle, memOffset, size, flags)
    }
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuMulticastCreate(
        mcHandle: *mut CUmemGenericAllocationHandle,
        prop: *const CUmulticastObjectProp,
    ) -> CUresult {
        (culib().cuMulticastCreate)(mcHandle, prop)
    }
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuMulticastGetGranularity(
        granularity: *mut usize,
        prop: *const CUmulticastObjectProp,
        option: CUmulticastGranularity_flags,
    ) -> CUresult {
        (culib().cuMulticastGetGranularity)(granularity, prop, option)
    }
    #[cfg(
        any(
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuMulticastUnbind(
        mcHandle: CUmemGenericAllocationHandle,
        dev: CUdevice,
        mcOffset: usize,
        size: usize,
    ) -> CUresult {
        (culib().cuMulticastUnbind)(mcHandle, dev, mcOffset, size)
    }
    pub unsafe fn cuOccupancyAvailableDynamicSMemPerBlock(
        dynamicSmemSize: *mut usize,
        func: CUfunction,
        numBlocks: ::core::ffi::c_int,
        blockSize: ::core::ffi::c_int,
    ) -> CUresult {
        (culib()
            .cuOccupancyAvailableDynamicSMemPerBlock)(
            dynamicSmemSize,
            func,
            numBlocks,
            blockSize,
        )
    }
    pub unsafe fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
        numBlocks: *mut ::core::ffi::c_int,
        func: CUfunction,
        blockSize: ::core::ffi::c_int,
        dynamicSMemSize: usize,
    ) -> CUresult {
        (culib()
            .cuOccupancyMaxActiveBlocksPerMultiprocessor)(
            numBlocks,
            func,
            blockSize,
            dynamicSMemSize,
        )
    }
    pub unsafe fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
        numBlocks: *mut ::core::ffi::c_int,
        func: CUfunction,
        blockSize: ::core::ffi::c_int,
        dynamicSMemSize: usize,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib()
            .cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags)(
            numBlocks,
            func,
            blockSize,
            dynamicSMemSize,
            flags,
        )
    }
    #[cfg(
        any(
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuOccupancyMaxActiveClusters(
        numClusters: *mut ::core::ffi::c_int,
        func: CUfunction,
        config: *const CUlaunchConfig,
    ) -> CUresult {
        (culib().cuOccupancyMaxActiveClusters)(numClusters, func, config)
    }
    pub unsafe fn cuOccupancyMaxPotentialBlockSize(
        minGridSize: *mut ::core::ffi::c_int,
        blockSize: *mut ::core::ffi::c_int,
        func: CUfunction,
        blockSizeToDynamicSMemSize: CUoccupancyB2DSize,
        dynamicSMemSize: usize,
        blockSizeLimit: ::core::ffi::c_int,
    ) -> CUresult {
        (culib()
            .cuOccupancyMaxPotentialBlockSize)(
            minGridSize,
            blockSize,
            func,
            blockSizeToDynamicSMemSize,
            dynamicSMemSize,
            blockSizeLimit,
        )
    }
    pub unsafe fn cuOccupancyMaxPotentialBlockSizeWithFlags(
        minGridSize: *mut ::core::ffi::c_int,
        blockSize: *mut ::core::ffi::c_int,
        func: CUfunction,
        blockSizeToDynamicSMemSize: CUoccupancyB2DSize,
        dynamicSMemSize: usize,
        blockSizeLimit: ::core::ffi::c_int,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib()
            .cuOccupancyMaxPotentialBlockSizeWithFlags)(
            minGridSize,
            blockSize,
            func,
            blockSizeToDynamicSMemSize,
            dynamicSMemSize,
            blockSizeLimit,
            flags,
        )
    }
    #[cfg(
        any(
            feature = "cuda-11080",
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuOccupancyMaxPotentialClusterSize(
        clusterSize: *mut ::core::ffi::c_int,
        func: CUfunction,
        config: *const CUlaunchConfig,
    ) -> CUresult {
        (culib().cuOccupancyMaxPotentialClusterSize)(clusterSize, func, config)
    }
    pub unsafe fn cuParamSetSize(
        hfunc: CUfunction,
        numbytes: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuParamSetSize)(hfunc, numbytes)
    }
    pub unsafe fn cuParamSetTexRef(
        hfunc: CUfunction,
        texunit: ::core::ffi::c_int,
        hTexRef: CUtexref,
    ) -> CUresult {
        (culib().cuParamSetTexRef)(hfunc, texunit, hTexRef)
    }
    pub unsafe fn cuParamSetf(
        hfunc: CUfunction,
        offset: ::core::ffi::c_int,
        value: f32,
    ) -> CUresult {
        (culib().cuParamSetf)(hfunc, offset, value)
    }
    pub unsafe fn cuParamSeti(
        hfunc: CUfunction,
        offset: ::core::ffi::c_int,
        value: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuParamSeti)(hfunc, offset, value)
    }
    pub unsafe fn cuParamSetv(
        hfunc: CUfunction,
        offset: ::core::ffi::c_int,
        ptr: *mut ::core::ffi::c_void,
        numbytes: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuParamSetv)(hfunc, offset, ptr, numbytes)
    }
    pub unsafe fn cuPointerGetAttribute(
        data: *mut ::core::ffi::c_void,
        attribute: CUpointer_attribute,
        ptr: CUdeviceptr,
    ) -> CUresult {
        (culib().cuPointerGetAttribute)(data, attribute, ptr)
    }
    pub unsafe fn cuPointerGetAttributes(
        numAttributes: ::core::ffi::c_uint,
        attributes: *mut CUpointer_attribute,
        data: *mut *mut ::core::ffi::c_void,
        ptr: CUdeviceptr,
    ) -> CUresult {
        (culib().cuPointerGetAttributes)(numAttributes, attributes, data, ptr)
    }
    pub unsafe fn cuPointerSetAttribute(
        value: *const ::core::ffi::c_void,
        attribute: CUpointer_attribute,
        ptr: CUdeviceptr,
    ) -> CUresult {
        (culib().cuPointerSetAttribute)(value, attribute, ptr)
    }
    pub unsafe fn cuProfilerInitialize(
        configFile: *const ::core::ffi::c_char,
        outputFile: *const ::core::ffi::c_char,
        outputMode: CUoutput_mode,
    ) -> CUresult {
        (culib().cuProfilerInitialize)(configFile, outputFile, outputMode)
    }
    pub unsafe fn cuProfilerStart() -> CUresult {
        (culib().cuProfilerStart)()
    }
    pub unsafe fn cuProfilerStop() -> CUresult {
        (culib().cuProfilerStop)()
    }
    pub unsafe fn cuSignalExternalSemaphoresAsync(
        extSemArray: *const CUexternalSemaphore,
        paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
        numExtSems: ::core::ffi::c_uint,
        stream: CUstream,
    ) -> CUresult {
        (culib()
            .cuSignalExternalSemaphoresAsync)(
            extSemArray,
            paramsArray,
            numExtSems,
            stream,
        )
    }
    pub unsafe fn cuStreamAddCallback(
        hStream: CUstream,
        callback: CUstreamCallback,
        userData: *mut ::core::ffi::c_void,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuStreamAddCallback)(hStream, callback, userData, flags)
    }
    pub unsafe fn cuStreamAttachMemAsync(
        hStream: CUstream,
        dptr: CUdeviceptr,
        length: usize,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuStreamAttachMemAsync)(hStream, dptr, length, flags)
    }
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub unsafe fn cuStreamBatchMemOp(
        stream: CUstream,
        count: ::core::ffi::c_uint,
        paramArray: *mut CUstreamBatchMemOpParams,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuStreamBatchMemOp)(stream, count, paramArray, flags)
    }
    #[cfg(
        any(
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
        )
    )]
    pub unsafe fn cuStreamBatchMemOp_v2(
        stream: CUstream,
        count: ::core::ffi::c_uint,
        paramArray: *mut CUstreamBatchMemOpParams,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuStreamBatchMemOp_v2)(stream, count, paramArray, flags)
    }
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuStreamBeginCaptureToGraph(
        hStream: CUstream,
        hGraph: CUgraph,
        dependencies: *const CUgraphNode,
        dependencyData: *const CUgraphEdgeData,
        numDependencies: usize,
        mode: CUstreamCaptureMode,
    ) -> CUresult {
        (culib()
            .cuStreamBeginCaptureToGraph)(
            hStream,
            hGraph,
            dependencies,
            dependencyData,
            numDependencies,
            mode,
        )
    }
    pub unsafe fn cuStreamBeginCapture_v2(
        hStream: CUstream,
        mode: CUstreamCaptureMode,
    ) -> CUresult {
        (culib().cuStreamBeginCapture_v2)(hStream, mode)
    }
    pub unsafe fn cuStreamCopyAttributes(dst: CUstream, src: CUstream) -> CUresult {
        (culib().cuStreamCopyAttributes)(dst, src)
    }
    pub unsafe fn cuStreamCreate(
        phStream: *mut CUstream,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuStreamCreate)(phStream, Flags)
    }
    pub unsafe fn cuStreamCreateWithPriority(
        phStream: *mut CUstream,
        flags: ::core::ffi::c_uint,
        priority: ::core::ffi::c_int,
    ) -> CUresult {
        (culib().cuStreamCreateWithPriority)(phStream, flags, priority)
    }
    pub unsafe fn cuStreamDestroy_v2(hStream: CUstream) -> CUresult {
        (culib().cuStreamDestroy_v2)(hStream)
    }
    pub unsafe fn cuStreamEndCapture(
        hStream: CUstream,
        phGraph: *mut CUgraph,
    ) -> CUresult {
        (culib().cuStreamEndCapture)(hStream, phGraph)
    }
    pub unsafe fn cuStreamGetAttribute(
        hStream: CUstream,
        attr: CUstreamAttrID,
        value_out: *mut CUstreamAttrValue,
    ) -> CUresult {
        (culib().cuStreamGetAttribute)(hStream, attr, value_out)
    }
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub unsafe fn cuStreamGetCaptureInfo(
        hStream: CUstream,
        captureStatus_out: *mut CUstreamCaptureStatus,
        id_out: *mut cuuint64_t,
    ) -> CUresult {
        (culib().cuStreamGetCaptureInfo)(hStream, captureStatus_out, id_out)
    }
    pub unsafe fn cuStreamGetCaptureInfo_v2(
        hStream: CUstream,
        captureStatus_out: *mut CUstreamCaptureStatus,
        id_out: *mut cuuint64_t,
        graph_out: *mut CUgraph,
        dependencies_out: *mut *const CUgraphNode,
        numDependencies_out: *mut usize,
    ) -> CUresult {
        (culib()
            .cuStreamGetCaptureInfo_v2)(
            hStream,
            captureStatus_out,
            id_out,
            graph_out,
            dependencies_out,
            numDependencies_out,
        )
    }
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuStreamGetCaptureInfo_v3(
        hStream: CUstream,
        captureStatus_out: *mut CUstreamCaptureStatus,
        id_out: *mut cuuint64_t,
        graph_out: *mut CUgraph,
        dependencies_out: *mut *const CUgraphNode,
        edgeData_out: *mut *const CUgraphEdgeData,
        numDependencies_out: *mut usize,
    ) -> CUresult {
        (culib()
            .cuStreamGetCaptureInfo_v3)(
            hStream,
            captureStatus_out,
            id_out,
            graph_out,
            dependencies_out,
            edgeData_out,
            numDependencies_out,
        )
    }
    pub unsafe fn cuStreamGetCtx(hStream: CUstream, pctx: *mut CUcontext) -> CUresult {
        (culib().cuStreamGetCtx)(hStream, pctx)
    }
    #[cfg(any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080"))]
    pub unsafe fn cuStreamGetCtx_v2(
        hStream: CUstream,
        pCtx: *mut CUcontext,
        pGreenCtx: *mut CUgreenCtx,
    ) -> CUresult {
        (culib().cuStreamGetCtx_v2)(hStream, pCtx, pGreenCtx)
    }
    #[cfg(any(feature = "cuda-12080"))]
    pub unsafe fn cuStreamGetDevice(
        hStream: CUstream,
        device: *mut CUdevice,
    ) -> CUresult {
        (culib().cuStreamGetDevice)(hStream, device)
    }
    pub unsafe fn cuStreamGetFlags(
        hStream: CUstream,
        flags: *mut ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuStreamGetFlags)(hStream, flags)
    }
    #[cfg(
        any(
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuStreamGetGreenCtx(
        hStream: CUstream,
        phCtx: *mut CUgreenCtx,
    ) -> CUresult {
        (culib().cuStreamGetGreenCtx)(hStream, phCtx)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuStreamGetId(
        hStream: CUstream,
        streamId: *mut ::core::ffi::c_ulonglong,
    ) -> CUresult {
        (culib().cuStreamGetId)(hStream, streamId)
    }
    pub unsafe fn cuStreamGetPriority(
        hStream: CUstream,
        priority: *mut ::core::ffi::c_int,
    ) -> CUresult {
        (culib().cuStreamGetPriority)(hStream, priority)
    }
    pub unsafe fn cuStreamIsCapturing(
        hStream: CUstream,
        captureStatus: *mut CUstreamCaptureStatus,
    ) -> CUresult {
        (culib().cuStreamIsCapturing)(hStream, captureStatus)
    }
    pub unsafe fn cuStreamQuery(hStream: CUstream) -> CUresult {
        (culib().cuStreamQuery)(hStream)
    }
    pub unsafe fn cuStreamSetAttribute(
        hStream: CUstream,
        attr: CUstreamAttrID,
        value: *const CUstreamAttrValue,
    ) -> CUresult {
        (culib().cuStreamSetAttribute)(hStream, attr, value)
    }
    pub unsafe fn cuStreamSynchronize(hStream: CUstream) -> CUresult {
        (culib().cuStreamSynchronize)(hStream)
    }
    pub unsafe fn cuStreamUpdateCaptureDependencies(
        hStream: CUstream,
        dependencies: *mut CUgraphNode,
        numDependencies: usize,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib()
            .cuStreamUpdateCaptureDependencies)(
            hStream,
            dependencies,
            numDependencies,
            flags,
        )
    }
    #[cfg(
        any(
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuStreamUpdateCaptureDependencies_v2(
        hStream: CUstream,
        dependencies: *mut CUgraphNode,
        dependencyData: *const CUgraphEdgeData,
        numDependencies: usize,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib()
            .cuStreamUpdateCaptureDependencies_v2)(
            hStream,
            dependencies,
            dependencyData,
            numDependencies,
            flags,
        )
    }
    pub unsafe fn cuStreamWaitEvent(
        hStream: CUstream,
        hEvent: CUevent,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuStreamWaitEvent)(hStream, hEvent, Flags)
    }
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub unsafe fn cuStreamWaitValue32(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint32_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuStreamWaitValue32)(stream, addr, value, flags)
    }
    #[cfg(
        any(
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
        )
    )]
    pub unsafe fn cuStreamWaitValue32_v2(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint32_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuStreamWaitValue32_v2)(stream, addr, value, flags)
    }
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub unsafe fn cuStreamWaitValue64(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint64_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuStreamWaitValue64)(stream, addr, value, flags)
    }
    #[cfg(
        any(
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
        )
    )]
    pub unsafe fn cuStreamWaitValue64_v2(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint64_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuStreamWaitValue64_v2)(stream, addr, value, flags)
    }
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub unsafe fn cuStreamWriteValue32(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint32_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuStreamWriteValue32)(stream, addr, value, flags)
    }
    #[cfg(
        any(
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
        )
    )]
    pub unsafe fn cuStreamWriteValue32_v2(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint32_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuStreamWriteValue32_v2)(stream, addr, value, flags)
    }
    #[cfg(
        any(
            feature = "cuda-11040",
            feature = "cuda-11050",
            feature = "cuda-11060",
            feature = "cuda-11070",
            feature = "cuda-11080"
        )
    )]
    pub unsafe fn cuStreamWriteValue64(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint64_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuStreamWriteValue64)(stream, addr, value, flags)
    }
    #[cfg(
        any(
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
        )
    )]
    pub unsafe fn cuStreamWriteValue64_v2(
        stream: CUstream,
        addr: CUdeviceptr,
        value: cuuint64_t,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuStreamWriteValue64_v2)(stream, addr, value, flags)
    }
    pub unsafe fn cuSurfObjectCreate(
        pSurfObject: *mut CUsurfObject,
        pResDesc: *const CUDA_RESOURCE_DESC,
    ) -> CUresult {
        (culib().cuSurfObjectCreate)(pSurfObject, pResDesc)
    }
    pub unsafe fn cuSurfObjectDestroy(surfObject: CUsurfObject) -> CUresult {
        (culib().cuSurfObjectDestroy)(surfObject)
    }
    pub unsafe fn cuSurfObjectGetResourceDesc(
        pResDesc: *mut CUDA_RESOURCE_DESC,
        surfObject: CUsurfObject,
    ) -> CUresult {
        (culib().cuSurfObjectGetResourceDesc)(pResDesc, surfObject)
    }
    pub unsafe fn cuSurfRefGetArray(
        phArray: *mut CUarray,
        hSurfRef: CUsurfref,
    ) -> CUresult {
        (culib().cuSurfRefGetArray)(phArray, hSurfRef)
    }
    pub unsafe fn cuSurfRefSetArray(
        hSurfRef: CUsurfref,
        hArray: CUarray,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuSurfRefSetArray)(hSurfRef, hArray, Flags)
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuTensorMapEncodeIm2col(
        tensorMap: *mut CUtensorMap,
        tensorDataType: CUtensorMapDataType,
        tensorRank: cuuint32_t,
        globalAddress: *mut ::core::ffi::c_void,
        globalDim: *const cuuint64_t,
        globalStrides: *const cuuint64_t,
        pixelBoxLowerCorner: *const ::core::ffi::c_int,
        pixelBoxUpperCorner: *const ::core::ffi::c_int,
        channelsPerPixel: cuuint32_t,
        pixelsPerColumn: cuuint32_t,
        elementStrides: *const cuuint32_t,
        interleave: CUtensorMapInterleave,
        swizzle: CUtensorMapSwizzle,
        l2Promotion: CUtensorMapL2promotion,
        oobFill: CUtensorMapFloatOOBfill,
    ) -> CUresult {
        (culib()
            .cuTensorMapEncodeIm2col)(
            tensorMap,
            tensorDataType,
            tensorRank,
            globalAddress,
            globalDim,
            globalStrides,
            pixelBoxLowerCorner,
            pixelBoxUpperCorner,
            channelsPerPixel,
            pixelsPerColumn,
            elementStrides,
            interleave,
            swizzle,
            l2Promotion,
            oobFill,
        )
    }
    #[cfg(any(feature = "cuda-12080"))]
    pub unsafe fn cuTensorMapEncodeIm2colWide(
        tensorMap: *mut CUtensorMap,
        tensorDataType: CUtensorMapDataType,
        tensorRank: cuuint32_t,
        globalAddress: *mut ::core::ffi::c_void,
        globalDim: *const cuuint64_t,
        globalStrides: *const cuuint64_t,
        pixelBoxLowerCornerWidth: ::core::ffi::c_int,
        pixelBoxUpperCornerWidth: ::core::ffi::c_int,
        channelsPerPixel: cuuint32_t,
        pixelsPerColumn: cuuint32_t,
        elementStrides: *const cuuint32_t,
        interleave: CUtensorMapInterleave,
        mode: CUtensorMapIm2ColWideMode,
        swizzle: CUtensorMapSwizzle,
        l2Promotion: CUtensorMapL2promotion,
        oobFill: CUtensorMapFloatOOBfill,
    ) -> CUresult {
        (culib()
            .cuTensorMapEncodeIm2colWide)(
            tensorMap,
            tensorDataType,
            tensorRank,
            globalAddress,
            globalDim,
            globalStrides,
            pixelBoxLowerCornerWidth,
            pixelBoxUpperCornerWidth,
            channelsPerPixel,
            pixelsPerColumn,
            elementStrides,
            interleave,
            mode,
            swizzle,
            l2Promotion,
            oobFill,
        )
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuTensorMapEncodeTiled(
        tensorMap: *mut CUtensorMap,
        tensorDataType: CUtensorMapDataType,
        tensorRank: cuuint32_t,
        globalAddress: *mut ::core::ffi::c_void,
        globalDim: *const cuuint64_t,
        globalStrides: *const cuuint64_t,
        boxDim: *const cuuint32_t,
        elementStrides: *const cuuint32_t,
        interleave: CUtensorMapInterleave,
        swizzle: CUtensorMapSwizzle,
        l2Promotion: CUtensorMapL2promotion,
        oobFill: CUtensorMapFloatOOBfill,
    ) -> CUresult {
        (culib()
            .cuTensorMapEncodeTiled)(
            tensorMap,
            tensorDataType,
            tensorRank,
            globalAddress,
            globalDim,
            globalStrides,
            boxDim,
            elementStrides,
            interleave,
            swizzle,
            l2Promotion,
            oobFill,
        )
    }
    #[cfg(
        any(
            feature = "cuda-12000",
            feature = "cuda-12010",
            feature = "cuda-12020",
            feature = "cuda-12030",
            feature = "cuda-12040",
            feature = "cuda-12050",
            feature = "cuda-12060",
            feature = "cuda-12080"
        )
    )]
    pub unsafe fn cuTensorMapReplaceAddress(
        tensorMap: *mut CUtensorMap,
        globalAddress: *mut ::core::ffi::c_void,
    ) -> CUresult {
        (culib().cuTensorMapReplaceAddress)(tensorMap, globalAddress)
    }
    pub unsafe fn cuTexObjectCreate(
        pTexObject: *mut CUtexObject,
        pResDesc: *const CUDA_RESOURCE_DESC,
        pTexDesc: *const CUDA_TEXTURE_DESC,
        pResViewDesc: *const CUDA_RESOURCE_VIEW_DESC,
    ) -> CUresult {
        (culib().cuTexObjectCreate)(pTexObject, pResDesc, pTexDesc, pResViewDesc)
    }
    pub unsafe fn cuTexObjectDestroy(texObject: CUtexObject) -> CUresult {
        (culib().cuTexObjectDestroy)(texObject)
    }
    pub unsafe fn cuTexObjectGetResourceDesc(
        pResDesc: *mut CUDA_RESOURCE_DESC,
        texObject: CUtexObject,
    ) -> CUresult {
        (culib().cuTexObjectGetResourceDesc)(pResDesc, texObject)
    }
    pub unsafe fn cuTexObjectGetResourceViewDesc(
        pResViewDesc: *mut CUDA_RESOURCE_VIEW_DESC,
        texObject: CUtexObject,
    ) -> CUresult {
        (culib().cuTexObjectGetResourceViewDesc)(pResViewDesc, texObject)
    }
    pub unsafe fn cuTexObjectGetTextureDesc(
        pTexDesc: *mut CUDA_TEXTURE_DESC,
        texObject: CUtexObject,
    ) -> CUresult {
        (culib().cuTexObjectGetTextureDesc)(pTexDesc, texObject)
    }
    pub unsafe fn cuTexRefCreate(pTexRef: *mut CUtexref) -> CUresult {
        (culib().cuTexRefCreate)(pTexRef)
    }
    pub unsafe fn cuTexRefDestroy(hTexRef: CUtexref) -> CUresult {
        (culib().cuTexRefDestroy)(hTexRef)
    }
    pub unsafe fn cuTexRefGetAddressMode(
        pam: *mut CUaddress_mode,
        hTexRef: CUtexref,
        dim: ::core::ffi::c_int,
    ) -> CUresult {
        (culib().cuTexRefGetAddressMode)(pam, hTexRef, dim)
    }
    pub unsafe fn cuTexRefGetAddress_v2(
        pdptr: *mut CUdeviceptr,
        hTexRef: CUtexref,
    ) -> CUresult {
        (culib().cuTexRefGetAddress_v2)(pdptr, hTexRef)
    }
    pub unsafe fn cuTexRefGetArray(
        phArray: *mut CUarray,
        hTexRef: CUtexref,
    ) -> CUresult {
        (culib().cuTexRefGetArray)(phArray, hTexRef)
    }
    pub unsafe fn cuTexRefGetBorderColor(
        pBorderColor: *mut f32,
        hTexRef: CUtexref,
    ) -> CUresult {
        (culib().cuTexRefGetBorderColor)(pBorderColor, hTexRef)
    }
    pub unsafe fn cuTexRefGetFilterMode(
        pfm: *mut CUfilter_mode,
        hTexRef: CUtexref,
    ) -> CUresult {
        (culib().cuTexRefGetFilterMode)(pfm, hTexRef)
    }
    pub unsafe fn cuTexRefGetFlags(
        pFlags: *mut ::core::ffi::c_uint,
        hTexRef: CUtexref,
    ) -> CUresult {
        (culib().cuTexRefGetFlags)(pFlags, hTexRef)
    }
    pub unsafe fn cuTexRefGetFormat(
        pFormat: *mut CUarray_format,
        pNumChannels: *mut ::core::ffi::c_int,
        hTexRef: CUtexref,
    ) -> CUresult {
        (culib().cuTexRefGetFormat)(pFormat, pNumChannels, hTexRef)
    }
    pub unsafe fn cuTexRefGetMaxAnisotropy(
        pmaxAniso: *mut ::core::ffi::c_int,
        hTexRef: CUtexref,
    ) -> CUresult {
        (culib().cuTexRefGetMaxAnisotropy)(pmaxAniso, hTexRef)
    }
    pub unsafe fn cuTexRefGetMipmapFilterMode(
        pfm: *mut CUfilter_mode,
        hTexRef: CUtexref,
    ) -> CUresult {
        (culib().cuTexRefGetMipmapFilterMode)(pfm, hTexRef)
    }
    pub unsafe fn cuTexRefGetMipmapLevelBias(
        pbias: *mut f32,
        hTexRef: CUtexref,
    ) -> CUresult {
        (culib().cuTexRefGetMipmapLevelBias)(pbias, hTexRef)
    }
    pub unsafe fn cuTexRefGetMipmapLevelClamp(
        pminMipmapLevelClamp: *mut f32,
        pmaxMipmapLevelClamp: *mut f32,
        hTexRef: CUtexref,
    ) -> CUresult {
        (culib()
            .cuTexRefGetMipmapLevelClamp)(
            pminMipmapLevelClamp,
            pmaxMipmapLevelClamp,
            hTexRef,
        )
    }
    pub unsafe fn cuTexRefGetMipmappedArray(
        phMipmappedArray: *mut CUmipmappedArray,
        hTexRef: CUtexref,
    ) -> CUresult {
        (culib().cuTexRefGetMipmappedArray)(phMipmappedArray, hTexRef)
    }
    pub unsafe fn cuTexRefSetAddress2D_v3(
        hTexRef: CUtexref,
        desc: *const CUDA_ARRAY_DESCRIPTOR,
        dptr: CUdeviceptr,
        Pitch: usize,
    ) -> CUresult {
        (culib().cuTexRefSetAddress2D_v3)(hTexRef, desc, dptr, Pitch)
    }
    pub unsafe fn cuTexRefSetAddressMode(
        hTexRef: CUtexref,
        dim: ::core::ffi::c_int,
        am: CUaddress_mode,
    ) -> CUresult {
        (culib().cuTexRefSetAddressMode)(hTexRef, dim, am)
    }
    pub unsafe fn cuTexRefSetAddress_v2(
        ByteOffset: *mut usize,
        hTexRef: CUtexref,
        dptr: CUdeviceptr,
        bytes: usize,
    ) -> CUresult {
        (culib().cuTexRefSetAddress_v2)(ByteOffset, hTexRef, dptr, bytes)
    }
    pub unsafe fn cuTexRefSetArray(
        hTexRef: CUtexref,
        hArray: CUarray,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuTexRefSetArray)(hTexRef, hArray, Flags)
    }
    pub unsafe fn cuTexRefSetBorderColor(
        hTexRef: CUtexref,
        pBorderColor: *mut f32,
    ) -> CUresult {
        (culib().cuTexRefSetBorderColor)(hTexRef, pBorderColor)
    }
    pub unsafe fn cuTexRefSetFilterMode(
        hTexRef: CUtexref,
        fm: CUfilter_mode,
    ) -> CUresult {
        (culib().cuTexRefSetFilterMode)(hTexRef, fm)
    }
    pub unsafe fn cuTexRefSetFlags(
        hTexRef: CUtexref,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuTexRefSetFlags)(hTexRef, Flags)
    }
    pub unsafe fn cuTexRefSetFormat(
        hTexRef: CUtexref,
        fmt: CUarray_format,
        NumPackedComponents: ::core::ffi::c_int,
    ) -> CUresult {
        (culib().cuTexRefSetFormat)(hTexRef, fmt, NumPackedComponents)
    }
    pub unsafe fn cuTexRefSetMaxAnisotropy(
        hTexRef: CUtexref,
        maxAniso: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuTexRefSetMaxAnisotropy)(hTexRef, maxAniso)
    }
    pub unsafe fn cuTexRefSetMipmapFilterMode(
        hTexRef: CUtexref,
        fm: CUfilter_mode,
    ) -> CUresult {
        (culib().cuTexRefSetMipmapFilterMode)(hTexRef, fm)
    }
    pub unsafe fn cuTexRefSetMipmapLevelBias(hTexRef: CUtexref, bias: f32) -> CUresult {
        (culib().cuTexRefSetMipmapLevelBias)(hTexRef, bias)
    }
    pub unsafe fn cuTexRefSetMipmapLevelClamp(
        hTexRef: CUtexref,
        minMipmapLevelClamp: f32,
        maxMipmapLevelClamp: f32,
    ) -> CUresult {
        (culib()
            .cuTexRefSetMipmapLevelClamp)(
            hTexRef,
            minMipmapLevelClamp,
            maxMipmapLevelClamp,
        )
    }
    pub unsafe fn cuTexRefSetMipmappedArray(
        hTexRef: CUtexref,
        hMipmappedArray: CUmipmappedArray,
        Flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuTexRefSetMipmappedArray)(hTexRef, hMipmappedArray, Flags)
    }
    pub unsafe fn cuThreadExchangeStreamCaptureMode(
        mode: *mut CUstreamCaptureMode,
    ) -> CUresult {
        (culib().cuThreadExchangeStreamCaptureMode)(mode)
    }
    pub unsafe fn cuUserObjectCreate(
        object_out: *mut CUuserObject,
        ptr: *mut ::core::ffi::c_void,
        destroy: CUhostFn,
        initialRefcount: ::core::ffi::c_uint,
        flags: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuUserObjectCreate)(object_out, ptr, destroy, initialRefcount, flags)
    }
    pub unsafe fn cuUserObjectRelease(
        object: CUuserObject,
        count: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuUserObjectRelease)(object, count)
    }
    pub unsafe fn cuUserObjectRetain(
        object: CUuserObject,
        count: ::core::ffi::c_uint,
    ) -> CUresult {
        (culib().cuUserObjectRetain)(object, count)
    }
    pub unsafe fn cuWaitExternalSemaphoresAsync(
        extSemArray: *const CUexternalSemaphore,
        paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
        numExtSems: ::core::ffi::c_uint,
        stream: CUstream,
    ) -> CUresult {
        (culib()
            .cuWaitExternalSemaphoresAsync)(extSemArray, paramsArray, numExtSems, stream)
    }
    pub struct Lib {
        __library: ::libloading::Library,
        pub cuArray3DCreate_v2: unsafe extern "C" fn(
            pHandle: *mut CUarray,
            pAllocateArray: *const CUDA_ARRAY3D_DESCRIPTOR,
        ) -> CUresult,
        pub cuArray3DGetDescriptor_v2: unsafe extern "C" fn(
            pArrayDescriptor: *mut CUDA_ARRAY3D_DESCRIPTOR,
            hArray: CUarray,
        ) -> CUresult,
        pub cuArrayCreate_v2: unsafe extern "C" fn(
            pHandle: *mut CUarray,
            pAllocateArray: *const CUDA_ARRAY_DESCRIPTOR,
        ) -> CUresult,
        pub cuArrayDestroy: unsafe extern "C" fn(hArray: CUarray) -> CUresult,
        pub cuArrayGetDescriptor_v2: unsafe extern "C" fn(
            pArrayDescriptor: *mut CUDA_ARRAY_DESCRIPTOR,
            hArray: CUarray,
        ) -> CUresult,
        #[cfg(
            any(
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
            )
        )]
        pub cuArrayGetMemoryRequirements: unsafe extern "C" fn(
            memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS,
            array: CUarray,
            device: CUdevice,
        ) -> CUresult,
        pub cuArrayGetPlane: unsafe extern "C" fn(
            pPlaneArray: *mut CUarray,
            hArray: CUarray,
            planeIdx: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuArrayGetSparseProperties: unsafe extern "C" fn(
            sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES,
            array: CUarray,
        ) -> CUresult,
        #[cfg(any(feature = "cuda-12080"))]
        pub cuCheckpointProcessCheckpoint: unsafe extern "C" fn(
            pid: ::core::ffi::c_int,
            args: *mut CUcheckpointCheckpointArgs,
        ) -> CUresult,
        #[cfg(any(feature = "cuda-12080"))]
        pub cuCheckpointProcessGetRestoreThreadId: unsafe extern "C" fn(
            pid: ::core::ffi::c_int,
            tid: *mut ::core::ffi::c_int,
        ) -> CUresult,
        #[cfg(any(feature = "cuda-12080"))]
        pub cuCheckpointProcessGetState: unsafe extern "C" fn(
            pid: ::core::ffi::c_int,
            state: *mut CUprocessState,
        ) -> CUresult,
        #[cfg(any(feature = "cuda-12080"))]
        pub cuCheckpointProcessLock: unsafe extern "C" fn(
            pid: ::core::ffi::c_int,
            args: *mut CUcheckpointLockArgs,
        ) -> CUresult,
        #[cfg(any(feature = "cuda-12080"))]
        pub cuCheckpointProcessRestore: unsafe extern "C" fn(
            pid: ::core::ffi::c_int,
            args: *mut CUcheckpointRestoreArgs,
        ) -> CUresult,
        #[cfg(any(feature = "cuda-12080"))]
        pub cuCheckpointProcessUnlock: unsafe extern "C" fn(
            pid: ::core::ffi::c_int,
            args: *mut CUcheckpointUnlockArgs,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuCoredumpGetAttribute: unsafe extern "C" fn(
            attrib: CUcoredumpSettings,
            value: *mut ::core::ffi::c_void,
            size: *mut usize,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuCoredumpGetAttributeGlobal: unsafe extern "C" fn(
            attrib: CUcoredumpSettings,
            value: *mut ::core::ffi::c_void,
            size: *mut usize,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuCoredumpSetAttribute: unsafe extern "C" fn(
            attrib: CUcoredumpSettings,
            value: *mut ::core::ffi::c_void,
            size: *mut usize,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuCoredumpSetAttributeGlobal: unsafe extern "C" fn(
            attrib: CUcoredumpSettings,
            value: *mut ::core::ffi::c_void,
            size: *mut usize,
        ) -> CUresult,
        pub cuCtxAttach: unsafe extern "C" fn(
            pctx: *mut CUcontext,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuCtxCreate_v2: unsafe extern "C" fn(
            pctx: *mut CUcontext,
            flags: ::core::ffi::c_uint,
            dev: CUdevice,
        ) -> CUresult,
        pub cuCtxCreate_v3: unsafe extern "C" fn(
            pctx: *mut CUcontext,
            paramsArray: *mut CUexecAffinityParam,
            numParams: ::core::ffi::c_int,
            flags: ::core::ffi::c_uint,
            dev: CUdevice,
        ) -> CUresult,
        #[cfg(
            any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080")
        )]
        pub cuCtxCreate_v4: unsafe extern "C" fn(
            pctx: *mut CUcontext,
            ctxCreateParams: *mut CUctxCreateParams,
            flags: ::core::ffi::c_uint,
            dev: CUdevice,
        ) -> CUresult,
        pub cuCtxDestroy_v2: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
        pub cuCtxDetach: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
        pub cuCtxDisablePeerAccess: unsafe extern "C" fn(
            peerContext: CUcontext,
        ) -> CUresult,
        pub cuCtxEnablePeerAccess: unsafe extern "C" fn(
            peerContext: CUcontext,
            Flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuCtxFromGreenCtx: unsafe extern "C" fn(
            pContext: *mut CUcontext,
            hCtx: CUgreenCtx,
        ) -> CUresult,
        pub cuCtxGetApiVersion: unsafe extern "C" fn(
            ctx: CUcontext,
            version: *mut ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuCtxGetCacheConfig: unsafe extern "C" fn(
            pconfig: *mut CUfunc_cache,
        ) -> CUresult,
        pub cuCtxGetCurrent: unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuCtxGetDevResource: unsafe extern "C" fn(
            hCtx: CUcontext,
            resource: *mut CUdevResource,
            type_: CUdevResourceType,
        ) -> CUresult,
        pub cuCtxGetDevice: unsafe extern "C" fn(device: *mut CUdevice) -> CUresult,
        pub cuCtxGetExecAffinity: unsafe extern "C" fn(
            pExecAffinity: *mut CUexecAffinityParam,
            type_: CUexecAffinityType,
        ) -> CUresult,
        pub cuCtxGetFlags: unsafe extern "C" fn(
            flags: *mut ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuCtxGetId: unsafe extern "C" fn(
            ctx: CUcontext,
            ctxId: *mut ::core::ffi::c_ulonglong,
        ) -> CUresult,
        pub cuCtxGetLimit: unsafe extern "C" fn(
            pvalue: *mut usize,
            limit: CUlimit,
        ) -> CUresult,
        pub cuCtxGetSharedMemConfig: unsafe extern "C" fn(
            pConfig: *mut CUsharedconfig,
        ) -> CUresult,
        pub cuCtxGetStreamPriorityRange: unsafe extern "C" fn(
            leastPriority: *mut ::core::ffi::c_int,
            greatestPriority: *mut ::core::ffi::c_int,
        ) -> CUresult,
        pub cuCtxPopCurrent_v2: unsafe extern "C" fn(pctx: *mut CUcontext) -> CUresult,
        pub cuCtxPushCurrent_v2: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
        #[cfg(
            any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080")
        )]
        pub cuCtxRecordEvent: unsafe extern "C" fn(
            hCtx: CUcontext,
            hEvent: CUevent,
        ) -> CUresult,
        pub cuCtxResetPersistingL2Cache: unsafe extern "C" fn() -> CUresult,
        pub cuCtxSetCacheConfig: unsafe extern "C" fn(config: CUfunc_cache) -> CUresult,
        pub cuCtxSetCurrent: unsafe extern "C" fn(ctx: CUcontext) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuCtxSetFlags: unsafe extern "C" fn(flags: ::core::ffi::c_uint) -> CUresult,
        pub cuCtxSetLimit: unsafe extern "C" fn(
            limit: CUlimit,
            value: usize,
        ) -> CUresult,
        pub cuCtxSetSharedMemConfig: unsafe extern "C" fn(
            config: CUsharedconfig,
        ) -> CUresult,
        pub cuCtxSynchronize: unsafe extern "C" fn() -> CUresult,
        #[cfg(
            any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080")
        )]
        pub cuCtxWaitEvent: unsafe extern "C" fn(
            hCtx: CUcontext,
            hEvent: CUevent,
        ) -> CUresult,
        pub cuDestroyExternalMemory: unsafe extern "C" fn(
            extMem: CUexternalMemory,
        ) -> CUresult,
        pub cuDestroyExternalSemaphore: unsafe extern "C" fn(
            extSem: CUexternalSemaphore,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuDevResourceGenerateDesc: unsafe extern "C" fn(
            phDesc: *mut CUdevResourceDesc,
            resources: *mut CUdevResource,
            nbResources: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuDevSmResourceSplitByCount: unsafe extern "C" fn(
            result: *mut CUdevResource,
            nbGroups: *mut ::core::ffi::c_uint,
            input: *const CUdevResource,
            remaining: *mut CUdevResource,
            useFlags: ::core::ffi::c_uint,
            minCount: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuDeviceCanAccessPeer: unsafe extern "C" fn(
            canAccessPeer: *mut ::core::ffi::c_int,
            dev: CUdevice,
            peerDev: CUdevice,
        ) -> CUresult,
        pub cuDeviceComputeCapability: unsafe extern "C" fn(
            major: *mut ::core::ffi::c_int,
            minor: *mut ::core::ffi::c_int,
            dev: CUdevice,
        ) -> CUresult,
        pub cuDeviceGet: unsafe extern "C" fn(
            device: *mut CUdevice,
            ordinal: ::core::ffi::c_int,
        ) -> CUresult,
        pub cuDeviceGetAttribute: unsafe extern "C" fn(
            pi: *mut ::core::ffi::c_int,
            attrib: CUdevice_attribute,
            dev: CUdevice,
        ) -> CUresult,
        pub cuDeviceGetByPCIBusId: unsafe extern "C" fn(
            dev: *mut CUdevice,
            pciBusId: *const ::core::ffi::c_char,
        ) -> CUresult,
        pub cuDeviceGetCount: unsafe extern "C" fn(
            count: *mut ::core::ffi::c_int,
        ) -> CUresult,
        pub cuDeviceGetDefaultMemPool: unsafe extern "C" fn(
            pool_out: *mut CUmemoryPool,
            dev: CUdevice,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuDeviceGetDevResource: unsafe extern "C" fn(
            device: CUdevice,
            resource: *mut CUdevResource,
            type_: CUdevResourceType,
        ) -> CUresult,
        pub cuDeviceGetExecAffinitySupport: unsafe extern "C" fn(
            pi: *mut ::core::ffi::c_int,
            type_: CUexecAffinityType,
            dev: CUdevice,
        ) -> CUresult,
        pub cuDeviceGetGraphMemAttribute: unsafe extern "C" fn(
            device: CUdevice,
            attr: CUgraphMem_attribute,
            value: *mut ::core::ffi::c_void,
        ) -> CUresult,
        pub cuDeviceGetLuid: unsafe extern "C" fn(
            luid: *mut ::core::ffi::c_char,
            deviceNodeMask: *mut ::core::ffi::c_uint,
            dev: CUdevice,
        ) -> CUresult,
        pub cuDeviceGetMemPool: unsafe extern "C" fn(
            pool: *mut CUmemoryPool,
            dev: CUdevice,
        ) -> CUresult,
        pub cuDeviceGetName: unsafe extern "C" fn(
            name: *mut ::core::ffi::c_char,
            len: ::core::ffi::c_int,
            dev: CUdevice,
        ) -> CUresult,
        pub cuDeviceGetNvSciSyncAttributes: unsafe extern "C" fn(
            nvSciSyncAttrList: *mut ::core::ffi::c_void,
            dev: CUdevice,
            flags: ::core::ffi::c_int,
        ) -> CUresult,
        pub cuDeviceGetP2PAttribute: unsafe extern "C" fn(
            value: *mut ::core::ffi::c_int,
            attrib: CUdevice_P2PAttribute,
            srcDevice: CUdevice,
            dstDevice: CUdevice,
        ) -> CUresult,
        pub cuDeviceGetPCIBusId: unsafe extern "C" fn(
            pciBusId: *mut ::core::ffi::c_char,
            len: ::core::ffi::c_int,
            dev: CUdevice,
        ) -> CUresult,
        pub cuDeviceGetProperties: unsafe extern "C" fn(
            prop: *mut CUdevprop,
            dev: CUdevice,
        ) -> CUresult,
        pub cuDeviceGetTexture1DLinearMaxWidth: unsafe extern "C" fn(
            maxWidthInElements: *mut usize,
            format: CUarray_format,
            numChannels: ::core::ffi::c_uint,
            dev: CUdevice,
        ) -> CUresult,
        pub cuDeviceGetUuid: unsafe extern "C" fn(
            uuid: *mut CUuuid,
            dev: CUdevice,
        ) -> CUresult,
        pub cuDeviceGetUuid_v2: unsafe extern "C" fn(
            uuid: *mut CUuuid,
            dev: CUdevice,
        ) -> CUresult,
        pub cuDeviceGraphMemTrim: unsafe extern "C" fn(device: CUdevice) -> CUresult,
        pub cuDevicePrimaryCtxGetState: unsafe extern "C" fn(
            dev: CUdevice,
            flags: *mut ::core::ffi::c_uint,
            active: *mut ::core::ffi::c_int,
        ) -> CUresult,
        pub cuDevicePrimaryCtxRelease_v2: unsafe extern "C" fn(
            dev: CUdevice,
        ) -> CUresult,
        pub cuDevicePrimaryCtxReset_v2: unsafe extern "C" fn(dev: CUdevice) -> CUresult,
        pub cuDevicePrimaryCtxRetain: unsafe extern "C" fn(
            pctx: *mut CUcontext,
            dev: CUdevice,
        ) -> CUresult,
        pub cuDevicePrimaryCtxSetFlags_v2: unsafe extern "C" fn(
            dev: CUdevice,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuDeviceRegisterAsyncNotification: unsafe extern "C" fn(
            device: CUdevice,
            callbackFunc: CUasyncCallback,
            userData: *mut ::core::ffi::c_void,
            callback: *mut CUasyncCallbackHandle,
        ) -> CUresult,
        pub cuDeviceSetGraphMemAttribute: unsafe extern "C" fn(
            device: CUdevice,
            attr: CUgraphMem_attribute,
            value: *mut ::core::ffi::c_void,
        ) -> CUresult,
        pub cuDeviceSetMemPool: unsafe extern "C" fn(
            dev: CUdevice,
            pool: CUmemoryPool,
        ) -> CUresult,
        pub cuDeviceTotalMem_v2: unsafe extern "C" fn(
            bytes: *mut usize,
            dev: CUdevice,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuDeviceUnregisterAsyncNotification: unsafe extern "C" fn(
            device: CUdevice,
            callback: CUasyncCallbackHandle,
        ) -> CUresult,
        pub cuDriverGetVersion: unsafe extern "C" fn(
            driverVersion: *mut ::core::ffi::c_int,
        ) -> CUresult,
        pub cuEventCreate: unsafe extern "C" fn(
            phEvent: *mut CUevent,
            Flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuEventDestroy_v2: unsafe extern "C" fn(hEvent: CUevent) -> CUresult,
        pub cuEventElapsedTime: unsafe extern "C" fn(
            pMilliseconds: *mut f32,
            hStart: CUevent,
            hEnd: CUevent,
        ) -> CUresult,
        #[cfg(any(feature = "cuda-12080"))]
        pub cuEventElapsedTime_v2: unsafe extern "C" fn(
            pMilliseconds: *mut f32,
            hStart: CUevent,
            hEnd: CUevent,
        ) -> CUresult,
        pub cuEventQuery: unsafe extern "C" fn(hEvent: CUevent) -> CUresult,
        pub cuEventRecord: unsafe extern "C" fn(
            hEvent: CUevent,
            hStream: CUstream,
        ) -> CUresult,
        pub cuEventRecordWithFlags: unsafe extern "C" fn(
            hEvent: CUevent,
            hStream: CUstream,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuEventSynchronize: unsafe extern "C" fn(hEvent: CUevent) -> CUresult,
        pub cuExternalMemoryGetMappedBuffer: unsafe extern "C" fn(
            devPtr: *mut CUdeviceptr,
            extMem: CUexternalMemory,
            bufferDesc: *const CUDA_EXTERNAL_MEMORY_BUFFER_DESC,
        ) -> CUresult,
        pub cuExternalMemoryGetMappedMipmappedArray: unsafe extern "C" fn(
            mipmap: *mut CUmipmappedArray,
            extMem: CUexternalMemory,
            mipmapDesc: *const CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC,
        ) -> CUresult,
        pub cuFlushGPUDirectRDMAWrites: unsafe extern "C" fn(
            target: CUflushGPUDirectRDMAWritesTarget,
            scope: CUflushGPUDirectRDMAWritesScope,
        ) -> CUresult,
        pub cuFuncGetAttribute: unsafe extern "C" fn(
            pi: *mut ::core::ffi::c_int,
            attrib: CUfunction_attribute,
            hfunc: CUfunction,
        ) -> CUresult,
        pub cuFuncGetModule: unsafe extern "C" fn(
            hmod: *mut CUmodule,
            hfunc: CUfunction,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuFuncGetName: unsafe extern "C" fn(
            name: *mut *const ::core::ffi::c_char,
            hfunc: CUfunction,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuFuncGetParamInfo: unsafe extern "C" fn(
            func: CUfunction,
            paramIndex: usize,
            paramOffset: *mut usize,
            paramSize: *mut usize,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuFuncIsLoaded: unsafe extern "C" fn(
            state: *mut CUfunctionLoadingState,
            function: CUfunction,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuFuncLoad: unsafe extern "C" fn(function: CUfunction) -> CUresult,
        pub cuFuncSetAttribute: unsafe extern "C" fn(
            hfunc: CUfunction,
            attrib: CUfunction_attribute,
            value: ::core::ffi::c_int,
        ) -> CUresult,
        pub cuFuncSetBlockShape: unsafe extern "C" fn(
            hfunc: CUfunction,
            x: ::core::ffi::c_int,
            y: ::core::ffi::c_int,
            z: ::core::ffi::c_int,
        ) -> CUresult,
        pub cuFuncSetCacheConfig: unsafe extern "C" fn(
            hfunc: CUfunction,
            config: CUfunc_cache,
        ) -> CUresult,
        pub cuFuncSetSharedMemConfig: unsafe extern "C" fn(
            hfunc: CUfunction,
            config: CUsharedconfig,
        ) -> CUresult,
        pub cuFuncSetSharedSize: unsafe extern "C" fn(
            hfunc: CUfunction,
            bytes: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuGetErrorName: unsafe extern "C" fn(
            error: CUresult,
            pStr: *mut *const ::core::ffi::c_char,
        ) -> CUresult,
        pub cuGetErrorString: unsafe extern "C" fn(
            error: CUresult,
            pStr: *mut *const ::core::ffi::c_char,
        ) -> CUresult,
        pub cuGetExportTable: unsafe extern "C" fn(
            ppExportTable: *mut *const ::core::ffi::c_void,
            pExportTableId: *const CUuuid,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            )
        )]
        pub cuGetProcAddress: unsafe extern "C" fn(
            symbol: *const ::core::ffi::c_char,
            pfn: *mut *mut ::core::ffi::c_void,
            cudaVersion: ::core::ffi::c_int,
            flags: cuuint64_t,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGetProcAddress_v2: unsafe extern "C" fn(
            symbol: *const ::core::ffi::c_char,
            pfn: *mut *mut ::core::ffi::c_void,
            cudaVersion: ::core::ffi::c_int,
            flags: cuuint64_t,
            symbolStatus: *mut CUdriverProcAddressQueryResult,
        ) -> CUresult,
        #[cfg(
            any(
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
            )
        )]
        pub cuGraphAddBatchMemOpNode: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            numDependencies: usize,
            nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphAddChildGraphNode: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            numDependencies: usize,
            childGraph: CUgraph,
        ) -> CUresult,
        pub cuGraphAddDependencies: unsafe extern "C" fn(
            hGraph: CUgraph,
            from: *const CUgraphNode,
            to: *const CUgraphNode,
            numDependencies: usize,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphAddDependencies_v2: unsafe extern "C" fn(
            hGraph: CUgraph,
            from: *const CUgraphNode,
            to: *const CUgraphNode,
            edgeData: *const CUgraphEdgeData,
            numDependencies: usize,
        ) -> CUresult,
        pub cuGraphAddEmptyNode: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            numDependencies: usize,
        ) -> CUresult,
        pub cuGraphAddEventRecordNode: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            numDependencies: usize,
            event: CUevent,
        ) -> CUresult,
        pub cuGraphAddEventWaitNode: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            numDependencies: usize,
            event: CUevent,
        ) -> CUresult,
        pub cuGraphAddExternalSemaphoresSignalNode: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            numDependencies: usize,
            nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphAddExternalSemaphoresWaitNode: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            numDependencies: usize,
            nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphAddHostNode: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            numDependencies: usize,
            nodeParams: *const CUDA_HOST_NODE_PARAMS,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            )
        )]
        pub cuGraphAddKernelNode: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            numDependencies: usize,
            nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphAddKernelNode_v2: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            numDependencies: usize,
            nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphAddMemAllocNode: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            numDependencies: usize,
            nodeParams: *mut CUDA_MEM_ALLOC_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphAddMemFreeNode: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            numDependencies: usize,
            dptr: CUdeviceptr,
        ) -> CUresult,
        pub cuGraphAddMemcpyNode: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            numDependencies: usize,
            copyParams: *const CUDA_MEMCPY3D,
            ctx: CUcontext,
        ) -> CUresult,
        pub cuGraphAddMemsetNode: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            numDependencies: usize,
            memsetParams: *const CUDA_MEMSET_NODE_PARAMS,
            ctx: CUcontext,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphAddNode: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            numDependencies: usize,
            nodeParams: *mut CUgraphNodeParams,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphAddNode_v2: unsafe extern "C" fn(
            phGraphNode: *mut CUgraphNode,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            dependencyData: *const CUgraphEdgeData,
            numDependencies: usize,
            nodeParams: *mut CUgraphNodeParams,
        ) -> CUresult,
        #[cfg(
            any(
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
            )
        )]
        pub cuGraphBatchMemOpNodeGetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            nodeParams_out: *mut CUDA_BATCH_MEM_OP_NODE_PARAMS,
        ) -> CUresult,
        #[cfg(
            any(
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
            )
        )]
        pub cuGraphBatchMemOpNodeSetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphChildGraphNodeGetGraph: unsafe extern "C" fn(
            hNode: CUgraphNode,
            phGraph: *mut CUgraph,
        ) -> CUresult,
        pub cuGraphClone: unsafe extern "C" fn(
            phGraphClone: *mut CUgraph,
            originalGraph: CUgraph,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphConditionalHandleCreate: unsafe extern "C" fn(
            pHandle_out: *mut CUgraphConditionalHandle,
            hGraph: CUgraph,
            ctx: CUcontext,
            defaultLaunchValue: ::core::ffi::c_uint,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuGraphCreate: unsafe extern "C" fn(
            phGraph: *mut CUgraph,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuGraphDebugDotPrint: unsafe extern "C" fn(
            hGraph: CUgraph,
            path: *const ::core::ffi::c_char,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuGraphDestroy: unsafe extern "C" fn(hGraph: CUgraph) -> CUresult,
        pub cuGraphDestroyNode: unsafe extern "C" fn(hNode: CUgraphNode) -> CUresult,
        pub cuGraphEventRecordNodeGetEvent: unsafe extern "C" fn(
            hNode: CUgraphNode,
            event_out: *mut CUevent,
        ) -> CUresult,
        pub cuGraphEventRecordNodeSetEvent: unsafe extern "C" fn(
            hNode: CUgraphNode,
            event: CUevent,
        ) -> CUresult,
        pub cuGraphEventWaitNodeGetEvent: unsafe extern "C" fn(
            hNode: CUgraphNode,
            event_out: *mut CUevent,
        ) -> CUresult,
        pub cuGraphEventWaitNodeSetEvent: unsafe extern "C" fn(
            hNode: CUgraphNode,
            event: CUevent,
        ) -> CUresult,
        #[cfg(
            any(
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
            )
        )]
        pub cuGraphExecBatchMemOpNodeSetParams: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hNode: CUgraphNode,
            nodeParams: *const CUDA_BATCH_MEM_OP_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphExecChildGraphNodeSetParams: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hNode: CUgraphNode,
            childGraph: CUgraph,
        ) -> CUresult,
        pub cuGraphExecDestroy: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
        ) -> CUresult,
        pub cuGraphExecEventRecordNodeSetEvent: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hNode: CUgraphNode,
            event: CUevent,
        ) -> CUresult,
        pub cuGraphExecEventWaitNodeSetEvent: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hNode: CUgraphNode,
            event: CUevent,
        ) -> CUresult,
        pub cuGraphExecExternalSemaphoresSignalNodeSetParams: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hNode: CUgraphNode,
            nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphExecExternalSemaphoresWaitNodeSetParams: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hNode: CUgraphNode,
            nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphExecGetFlags: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            flags: *mut cuuint64_t,
        ) -> CUresult,
        pub cuGraphExecHostNodeSetParams: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hNode: CUgraphNode,
            nodeParams: *const CUDA_HOST_NODE_PARAMS,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            )
        )]
        pub cuGraphExecKernelNodeSetParams: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hNode: CUgraphNode,
            nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphExecKernelNodeSetParams_v2: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hNode: CUgraphNode,
            nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphExecMemcpyNodeSetParams: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hNode: CUgraphNode,
            copyParams: *const CUDA_MEMCPY3D,
            ctx: CUcontext,
        ) -> CUresult,
        pub cuGraphExecMemsetNodeSetParams: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hNode: CUgraphNode,
            memsetParams: *const CUDA_MEMSET_NODE_PARAMS,
            ctx: CUcontext,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphExecNodeSetParams: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hNode: CUgraphNode,
            nodeParams: *mut CUgraphNodeParams,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            )
        )]
        pub cuGraphExecUpdate: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hGraph: CUgraph,
            hErrorNode_out: *mut CUgraphNode,
            updateResult_out: *mut CUgraphExecUpdateResult,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphExecUpdate_v2: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hGraph: CUgraph,
            resultInfo: *mut CUgraphExecUpdateResultInfo,
        ) -> CUresult,
        pub cuGraphExternalSemaphoresSignalNodeGetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            params_out: *mut CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphExternalSemaphoresSignalNodeSetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            nodeParams: *const CUDA_EXT_SEM_SIGNAL_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphExternalSemaphoresWaitNodeGetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            params_out: *mut CUDA_EXT_SEM_WAIT_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphExternalSemaphoresWaitNodeSetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            nodeParams: *const CUDA_EXT_SEM_WAIT_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphGetEdges: unsafe extern "C" fn(
            hGraph: CUgraph,
            from: *mut CUgraphNode,
            to: *mut CUgraphNode,
            numEdges: *mut usize,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphGetEdges_v2: unsafe extern "C" fn(
            hGraph: CUgraph,
            from: *mut CUgraphNode,
            to: *mut CUgraphNode,
            edgeData: *mut CUgraphEdgeData,
            numEdges: *mut usize,
        ) -> CUresult,
        pub cuGraphGetNodes: unsafe extern "C" fn(
            hGraph: CUgraph,
            nodes: *mut CUgraphNode,
            numNodes: *mut usize,
        ) -> CUresult,
        pub cuGraphGetRootNodes: unsafe extern "C" fn(
            hGraph: CUgraph,
            rootNodes: *mut CUgraphNode,
            numRootNodes: *mut usize,
        ) -> CUresult,
        pub cuGraphHostNodeGetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            nodeParams: *mut CUDA_HOST_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphHostNodeSetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            nodeParams: *const CUDA_HOST_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphInstantiateWithFlags: unsafe extern "C" fn(
            phGraphExec: *mut CUgraphExec,
            hGraph: CUgraph,
            flags: ::core::ffi::c_ulonglong,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphInstantiateWithParams: unsafe extern "C" fn(
            phGraphExec: *mut CUgraphExec,
            hGraph: CUgraph,
            instantiateParams: *mut CUDA_GRAPH_INSTANTIATE_PARAMS,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            )
        )]
        pub cuGraphInstantiate_v2: unsafe extern "C" fn(
            phGraphExec: *mut CUgraphExec,
            hGraph: CUgraph,
            phErrorNode: *mut CUgraphNode,
            logBuffer: *mut ::core::ffi::c_char,
            bufferSize: usize,
        ) -> CUresult,
        pub cuGraphKernelNodeCopyAttributes: unsafe extern "C" fn(
            dst: CUgraphNode,
            src: CUgraphNode,
        ) -> CUresult,
        pub cuGraphKernelNodeGetAttribute: unsafe extern "C" fn(
            hNode: CUgraphNode,
            attr: CUkernelNodeAttrID,
            value_out: *mut CUkernelNodeAttrValue,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            )
        )]
        pub cuGraphKernelNodeGetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            nodeParams: *mut CUDA_KERNEL_NODE_PARAMS,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphKernelNodeGetParams_v2: unsafe extern "C" fn(
            hNode: CUgraphNode,
            nodeParams: *mut CUDA_KERNEL_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphKernelNodeSetAttribute: unsafe extern "C" fn(
            hNode: CUgraphNode,
            attr: CUkernelNodeAttrID,
            value: *const CUkernelNodeAttrValue,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            )
        )]
        pub cuGraphKernelNodeSetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphKernelNodeSetParams_v2: unsafe extern "C" fn(
            hNode: CUgraphNode,
            nodeParams: *const CUDA_KERNEL_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphLaunch: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hStream: CUstream,
        ) -> CUresult,
        pub cuGraphMemAllocNodeGetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            params_out: *mut CUDA_MEM_ALLOC_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphMemFreeNodeGetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            dptr_out: *mut CUdeviceptr,
        ) -> CUresult,
        pub cuGraphMemcpyNodeGetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            nodeParams: *mut CUDA_MEMCPY3D,
        ) -> CUresult,
        pub cuGraphMemcpyNodeSetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            nodeParams: *const CUDA_MEMCPY3D,
        ) -> CUresult,
        pub cuGraphMemsetNodeGetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            nodeParams: *mut CUDA_MEMSET_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphMemsetNodeSetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            nodeParams: *const CUDA_MEMSET_NODE_PARAMS,
        ) -> CUresult,
        pub cuGraphNodeFindInClone: unsafe extern "C" fn(
            phNode: *mut CUgraphNode,
            hOriginalNode: CUgraphNode,
            hClonedGraph: CUgraph,
        ) -> CUresult,
        pub cuGraphNodeGetDependencies: unsafe extern "C" fn(
            hNode: CUgraphNode,
            dependencies: *mut CUgraphNode,
            numDependencies: *mut usize,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphNodeGetDependencies_v2: unsafe extern "C" fn(
            hNode: CUgraphNode,
            dependencies: *mut CUgraphNode,
            edgeData: *mut CUgraphEdgeData,
            numDependencies: *mut usize,
        ) -> CUresult,
        pub cuGraphNodeGetDependentNodes: unsafe extern "C" fn(
            hNode: CUgraphNode,
            dependentNodes: *mut CUgraphNode,
            numDependentNodes: *mut usize,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphNodeGetDependentNodes_v2: unsafe extern "C" fn(
            hNode: CUgraphNode,
            dependentNodes: *mut CUgraphNode,
            edgeData: *mut CUgraphEdgeData,
            numDependentNodes: *mut usize,
        ) -> CUresult,
        #[cfg(
            any(
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
            )
        )]
        pub cuGraphNodeGetEnabled: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hNode: CUgraphNode,
            isEnabled: *mut ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuGraphNodeGetType: unsafe extern "C" fn(
            hNode: CUgraphNode,
            type_: *mut CUgraphNodeType,
        ) -> CUresult,
        #[cfg(
            any(
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
            )
        )]
        pub cuGraphNodeSetEnabled: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hNode: CUgraphNode,
            isEnabled: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphNodeSetParams: unsafe extern "C" fn(
            hNode: CUgraphNode,
            nodeParams: *mut CUgraphNodeParams,
        ) -> CUresult,
        pub cuGraphReleaseUserObject: unsafe extern "C" fn(
            graph: CUgraph,
            object: CUuserObject,
            count: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuGraphRemoveDependencies: unsafe extern "C" fn(
            hGraph: CUgraph,
            from: *const CUgraphNode,
            to: *const CUgraphNode,
            numDependencies: usize,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGraphRemoveDependencies_v2: unsafe extern "C" fn(
            hGraph: CUgraph,
            from: *const CUgraphNode,
            to: *const CUgraphNode,
            edgeData: *const CUgraphEdgeData,
            numDependencies: usize,
        ) -> CUresult,
        pub cuGraphRetainUserObject: unsafe extern "C" fn(
            graph: CUgraph,
            object: CUuserObject,
            count: ::core::ffi::c_uint,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuGraphUpload: unsafe extern "C" fn(
            hGraphExec: CUgraphExec,
            hStream: CUstream,
        ) -> CUresult,
        pub cuGraphicsMapResources: unsafe extern "C" fn(
            count: ::core::ffi::c_uint,
            resources: *mut CUgraphicsResource,
            hStream: CUstream,
        ) -> CUresult,
        pub cuGraphicsResourceGetMappedMipmappedArray: unsafe extern "C" fn(
            pMipmappedArray: *mut CUmipmappedArray,
            resource: CUgraphicsResource,
        ) -> CUresult,
        pub cuGraphicsResourceGetMappedPointer_v2: unsafe extern "C" fn(
            pDevPtr: *mut CUdeviceptr,
            pSize: *mut usize,
            resource: CUgraphicsResource,
        ) -> CUresult,
        pub cuGraphicsResourceSetMapFlags_v2: unsafe extern "C" fn(
            resource: CUgraphicsResource,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuGraphicsSubResourceGetMappedArray: unsafe extern "C" fn(
            pArray: *mut CUarray,
            resource: CUgraphicsResource,
            arrayIndex: ::core::ffi::c_uint,
            mipLevel: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuGraphicsUnmapResources: unsafe extern "C" fn(
            count: ::core::ffi::c_uint,
            resources: *mut CUgraphicsResource,
            hStream: CUstream,
        ) -> CUresult,
        pub cuGraphicsUnregisterResource: unsafe extern "C" fn(
            resource: CUgraphicsResource,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGreenCtxCreate: unsafe extern "C" fn(
            phCtx: *mut CUgreenCtx,
            desc: CUdevResourceDesc,
            dev: CUdevice,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGreenCtxDestroy: unsafe extern "C" fn(hCtx: CUgreenCtx) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGreenCtxGetDevResource: unsafe extern "C" fn(
            hCtx: CUgreenCtx,
            resource: *mut CUdevResource,
            type_: CUdevResourceType,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGreenCtxRecordEvent: unsafe extern "C" fn(
            hCtx: CUgreenCtx,
            hEvent: CUevent,
        ) -> CUresult,
        #[cfg(
            any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080")
        )]
        pub cuGreenCtxStreamCreate: unsafe extern "C" fn(
            phStream: *mut CUstream,
            greenCtx: CUgreenCtx,
            flags: ::core::ffi::c_uint,
            priority: ::core::ffi::c_int,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuGreenCtxWaitEvent: unsafe extern "C" fn(
            hCtx: CUgreenCtx,
            hEvent: CUevent,
        ) -> CUresult,
        pub cuImportExternalMemory: unsafe extern "C" fn(
            extMem_out: *mut CUexternalMemory,
            memHandleDesc: *const CUDA_EXTERNAL_MEMORY_HANDLE_DESC,
        ) -> CUresult,
        pub cuImportExternalSemaphore: unsafe extern "C" fn(
            extSem_out: *mut CUexternalSemaphore,
            semHandleDesc: *const CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC,
        ) -> CUresult,
        pub cuInit: unsafe extern "C" fn(Flags: ::core::ffi::c_uint) -> CUresult,
        pub cuIpcCloseMemHandle: unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult,
        pub cuIpcGetEventHandle: unsafe extern "C" fn(
            pHandle: *mut CUipcEventHandle,
            event: CUevent,
        ) -> CUresult,
        pub cuIpcGetMemHandle: unsafe extern "C" fn(
            pHandle: *mut CUipcMemHandle,
            dptr: CUdeviceptr,
        ) -> CUresult,
        pub cuIpcOpenEventHandle: unsafe extern "C" fn(
            phEvent: *mut CUevent,
            handle: CUipcEventHandle,
        ) -> CUresult,
        pub cuIpcOpenMemHandle_v2: unsafe extern "C" fn(
            pdptr: *mut CUdeviceptr,
            handle: CUipcMemHandle,
            Flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuKernelGetAttribute: unsafe extern "C" fn(
            pi: *mut ::core::ffi::c_int,
            attrib: CUfunction_attribute,
            kernel: CUkernel,
            dev: CUdevice,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuKernelGetFunction: unsafe extern "C" fn(
            pFunc: *mut CUfunction,
            kernel: CUkernel,
        ) -> CUresult,
        #[cfg(
            any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080")
        )]
        pub cuKernelGetLibrary: unsafe extern "C" fn(
            pLib: *mut CUlibrary,
            kernel: CUkernel,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuKernelGetName: unsafe extern "C" fn(
            name: *mut *const ::core::ffi::c_char,
            hfunc: CUkernel,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuKernelGetParamInfo: unsafe extern "C" fn(
            kernel: CUkernel,
            paramIndex: usize,
            paramOffset: *mut usize,
            paramSize: *mut usize,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuKernelSetAttribute: unsafe extern "C" fn(
            attrib: CUfunction_attribute,
            val: ::core::ffi::c_int,
            kernel: CUkernel,
            dev: CUdevice,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuKernelSetCacheConfig: unsafe extern "C" fn(
            kernel: CUkernel,
            config: CUfunc_cache,
            dev: CUdevice,
        ) -> CUresult,
        pub cuLaunch: unsafe extern "C" fn(f: CUfunction) -> CUresult,
        pub cuLaunchCooperativeKernel: unsafe extern "C" fn(
            f: CUfunction,
            gridDimX: ::core::ffi::c_uint,
            gridDimY: ::core::ffi::c_uint,
            gridDimZ: ::core::ffi::c_uint,
            blockDimX: ::core::ffi::c_uint,
            blockDimY: ::core::ffi::c_uint,
            blockDimZ: ::core::ffi::c_uint,
            sharedMemBytes: ::core::ffi::c_uint,
            hStream: CUstream,
            kernelParams: *mut *mut ::core::ffi::c_void,
        ) -> CUresult,
        pub cuLaunchCooperativeKernelMultiDevice: unsafe extern "C" fn(
            launchParamsList: *mut CUDA_LAUNCH_PARAMS,
            numDevices: ::core::ffi::c_uint,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuLaunchGrid: unsafe extern "C" fn(
            f: CUfunction,
            grid_width: ::core::ffi::c_int,
            grid_height: ::core::ffi::c_int,
        ) -> CUresult,
        pub cuLaunchGridAsync: unsafe extern "C" fn(
            f: CUfunction,
            grid_width: ::core::ffi::c_int,
            grid_height: ::core::ffi::c_int,
            hStream: CUstream,
        ) -> CUresult,
        pub cuLaunchHostFunc: unsafe extern "C" fn(
            hStream: CUstream,
            fn_: CUhostFn,
            userData: *mut ::core::ffi::c_void,
        ) -> CUresult,
        pub cuLaunchKernel: unsafe extern "C" fn(
            f: CUfunction,
            gridDimX: ::core::ffi::c_uint,
            gridDimY: ::core::ffi::c_uint,
            gridDimZ: ::core::ffi::c_uint,
            blockDimX: ::core::ffi::c_uint,
            blockDimY: ::core::ffi::c_uint,
            blockDimZ: ::core::ffi::c_uint,
            sharedMemBytes: ::core::ffi::c_uint,
            hStream: CUstream,
            kernelParams: *mut *mut ::core::ffi::c_void,
            extra: *mut *mut ::core::ffi::c_void,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11080",
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuLaunchKernelEx: unsafe extern "C" fn(
            config: *const CUlaunchConfig,
            f: CUfunction,
            kernelParams: *mut *mut ::core::ffi::c_void,
            extra: *mut *mut ::core::ffi::c_void,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuLibraryEnumerateKernels: unsafe extern "C" fn(
            kernels: *mut CUkernel,
            numKernels: ::core::ffi::c_uint,
            lib: CUlibrary,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuLibraryGetGlobal: unsafe extern "C" fn(
            dptr: *mut CUdeviceptr,
            bytes: *mut usize,
            library: CUlibrary,
            name: *const ::core::ffi::c_char,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuLibraryGetKernel: unsafe extern "C" fn(
            pKernel: *mut CUkernel,
            library: CUlibrary,
            name: *const ::core::ffi::c_char,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuLibraryGetKernelCount: unsafe extern "C" fn(
            count: *mut ::core::ffi::c_uint,
            lib: CUlibrary,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuLibraryGetManaged: unsafe extern "C" fn(
            dptr: *mut CUdeviceptr,
            bytes: *mut usize,
            library: CUlibrary,
            name: *const ::core::ffi::c_char,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuLibraryGetModule: unsafe extern "C" fn(
            pMod: *mut CUmodule,
            library: CUlibrary,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuLibraryGetUnifiedFunction: unsafe extern "C" fn(
            fptr: *mut *mut ::core::ffi::c_void,
            library: CUlibrary,
            symbol: *const ::core::ffi::c_char,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuLibraryLoadData: unsafe extern "C" fn(
            library: *mut CUlibrary,
            code: *const ::core::ffi::c_void,
            jitOptions: *mut CUjit_option,
            jitOptionsValues: *mut *mut ::core::ffi::c_void,
            numJitOptions: ::core::ffi::c_uint,
            libraryOptions: *mut CUlibraryOption,
            libraryOptionValues: *mut *mut ::core::ffi::c_void,
            numLibraryOptions: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuLibraryLoadFromFile: unsafe extern "C" fn(
            library: *mut CUlibrary,
            fileName: *const ::core::ffi::c_char,
            jitOptions: *mut CUjit_option,
            jitOptionsValues: *mut *mut ::core::ffi::c_void,
            numJitOptions: ::core::ffi::c_uint,
            libraryOptions: *mut CUlibraryOption,
            libraryOptionValues: *mut *mut ::core::ffi::c_void,
            numLibraryOptions: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuLibraryUnload: unsafe extern "C" fn(library: CUlibrary) -> CUresult,
        pub cuLinkAddData_v2: unsafe extern "C" fn(
            state: CUlinkState,
            type_: CUjitInputType,
            data: *mut ::core::ffi::c_void,
            size: usize,
            name: *const ::core::ffi::c_char,
            numOptions: ::core::ffi::c_uint,
            options: *mut CUjit_option,
            optionValues: *mut *mut ::core::ffi::c_void,
        ) -> CUresult,
        pub cuLinkAddFile_v2: unsafe extern "C" fn(
            state: CUlinkState,
            type_: CUjitInputType,
            path: *const ::core::ffi::c_char,
            numOptions: ::core::ffi::c_uint,
            options: *mut CUjit_option,
            optionValues: *mut *mut ::core::ffi::c_void,
        ) -> CUresult,
        pub cuLinkComplete: unsafe extern "C" fn(
            state: CUlinkState,
            cubinOut: *mut *mut ::core::ffi::c_void,
            sizeOut: *mut usize,
        ) -> CUresult,
        pub cuLinkCreate_v2: unsafe extern "C" fn(
            numOptions: ::core::ffi::c_uint,
            options: *mut CUjit_option,
            optionValues: *mut *mut ::core::ffi::c_void,
            stateOut: *mut CUlinkState,
        ) -> CUresult,
        pub cuLinkDestroy: unsafe extern "C" fn(state: CUlinkState) -> CUresult,
        pub cuMemAddressFree: unsafe extern "C" fn(
            ptr: CUdeviceptr,
            size: usize,
        ) -> CUresult,
        pub cuMemAddressReserve: unsafe extern "C" fn(
            ptr: *mut CUdeviceptr,
            size: usize,
            alignment: usize,
            addr: CUdeviceptr,
            flags: ::core::ffi::c_ulonglong,
        ) -> CUresult,
        pub cuMemAdvise: unsafe extern "C" fn(
            devPtr: CUdeviceptr,
            count: usize,
            advice: CUmem_advise,
            device: CUdevice,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuMemAdvise_v2: unsafe extern "C" fn(
            devPtr: CUdeviceptr,
            count: usize,
            advice: CUmem_advise,
            location: CUmemLocation,
        ) -> CUresult,
        pub cuMemAllocAsync: unsafe extern "C" fn(
            dptr: *mut CUdeviceptr,
            bytesize: usize,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemAllocFromPoolAsync: unsafe extern "C" fn(
            dptr: *mut CUdeviceptr,
            bytesize: usize,
            pool: CUmemoryPool,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemAllocHost_v2: unsafe extern "C" fn(
            pp: *mut *mut ::core::ffi::c_void,
            bytesize: usize,
        ) -> CUresult,
        pub cuMemAllocManaged: unsafe extern "C" fn(
            dptr: *mut CUdeviceptr,
            bytesize: usize,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuMemAllocPitch_v2: unsafe extern "C" fn(
            dptr: *mut CUdeviceptr,
            pPitch: *mut usize,
            WidthInBytes: usize,
            Height: usize,
            ElementSizeBytes: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuMemAlloc_v2: unsafe extern "C" fn(
            dptr: *mut CUdeviceptr,
            bytesize: usize,
        ) -> CUresult,
        #[cfg(any(feature = "cuda-12080"))]
        pub cuMemBatchDecompressAsync: unsafe extern "C" fn(
            paramsArray: *mut CUmemDecompressParams,
            count: usize,
            flags: ::core::ffi::c_uint,
            errorIndex: *mut usize,
            stream: CUstream,
        ) -> CUresult,
        pub cuMemCreate: unsafe extern "C" fn(
            handle: *mut CUmemGenericAllocationHandle,
            size: usize,
            prop: *const CUmemAllocationProp,
            flags: ::core::ffi::c_ulonglong,
        ) -> CUresult,
        pub cuMemExportToShareableHandle: unsafe extern "C" fn(
            shareableHandle: *mut ::core::ffi::c_void,
            handle: CUmemGenericAllocationHandle,
            handleType: CUmemAllocationHandleType,
            flags: ::core::ffi::c_ulonglong,
        ) -> CUresult,
        pub cuMemFreeAsync: unsafe extern "C" fn(
            dptr: CUdeviceptr,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemFreeHost: unsafe extern "C" fn(p: *mut ::core::ffi::c_void) -> CUresult,
        pub cuMemFree_v2: unsafe extern "C" fn(dptr: CUdeviceptr) -> CUresult,
        pub cuMemGetAccess: unsafe extern "C" fn(
            flags: *mut ::core::ffi::c_ulonglong,
            location: *const CUmemLocation,
            ptr: CUdeviceptr,
        ) -> CUresult,
        pub cuMemGetAddressRange_v2: unsafe extern "C" fn(
            pbase: *mut CUdeviceptr,
            psize: *mut usize,
            dptr: CUdeviceptr,
        ) -> CUresult,
        pub cuMemGetAllocationGranularity: unsafe extern "C" fn(
            granularity: *mut usize,
            prop: *const CUmemAllocationProp,
            option: CUmemAllocationGranularity_flags,
        ) -> CUresult,
        pub cuMemGetAllocationPropertiesFromHandle: unsafe extern "C" fn(
            prop: *mut CUmemAllocationProp,
            handle: CUmemGenericAllocationHandle,
        ) -> CUresult,
        #[cfg(
            any(
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
            )
        )]
        pub cuMemGetHandleForAddressRange: unsafe extern "C" fn(
            handle: *mut ::core::ffi::c_void,
            dptr: CUdeviceptr,
            size: usize,
            handleType: CUmemRangeHandleType,
            flags: ::core::ffi::c_ulonglong,
        ) -> CUresult,
        pub cuMemGetInfo_v2: unsafe extern "C" fn(
            free: *mut usize,
            total: *mut usize,
        ) -> CUresult,
        pub cuMemHostAlloc: unsafe extern "C" fn(
            pp: *mut *mut ::core::ffi::c_void,
            bytesize: usize,
            Flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuMemHostGetDevicePointer_v2: unsafe extern "C" fn(
            pdptr: *mut CUdeviceptr,
            p: *mut ::core::ffi::c_void,
            Flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuMemHostGetFlags: unsafe extern "C" fn(
            pFlags: *mut ::core::ffi::c_uint,
            p: *mut ::core::ffi::c_void,
        ) -> CUresult,
        pub cuMemHostRegister_v2: unsafe extern "C" fn(
            p: *mut ::core::ffi::c_void,
            bytesize: usize,
            Flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuMemHostUnregister: unsafe extern "C" fn(
            p: *mut ::core::ffi::c_void,
        ) -> CUresult,
        pub cuMemImportFromShareableHandle: unsafe extern "C" fn(
            handle: *mut CUmemGenericAllocationHandle,
            osHandle: *mut ::core::ffi::c_void,
            shHandleType: CUmemAllocationHandleType,
        ) -> CUresult,
        pub cuMemMap: unsafe extern "C" fn(
            ptr: CUdeviceptr,
            size: usize,
            offset: usize,
            handle: CUmemGenericAllocationHandle,
            flags: ::core::ffi::c_ulonglong,
        ) -> CUresult,
        pub cuMemMapArrayAsync: unsafe extern "C" fn(
            mapInfoList: *mut CUarrayMapInfo,
            count: ::core::ffi::c_uint,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemPoolCreate: unsafe extern "C" fn(
            pool: *mut CUmemoryPool,
            poolProps: *const CUmemPoolProps,
        ) -> CUresult,
        pub cuMemPoolDestroy: unsafe extern "C" fn(pool: CUmemoryPool) -> CUresult,
        pub cuMemPoolExportPointer: unsafe extern "C" fn(
            shareData_out: *mut CUmemPoolPtrExportData,
            ptr: CUdeviceptr,
        ) -> CUresult,
        pub cuMemPoolExportToShareableHandle: unsafe extern "C" fn(
            handle_out: *mut ::core::ffi::c_void,
            pool: CUmemoryPool,
            handleType: CUmemAllocationHandleType,
            flags: ::core::ffi::c_ulonglong,
        ) -> CUresult,
        pub cuMemPoolGetAccess: unsafe extern "C" fn(
            flags: *mut CUmemAccess_flags,
            memPool: CUmemoryPool,
            location: *mut CUmemLocation,
        ) -> CUresult,
        pub cuMemPoolGetAttribute: unsafe extern "C" fn(
            pool: CUmemoryPool,
            attr: CUmemPool_attribute,
            value: *mut ::core::ffi::c_void,
        ) -> CUresult,
        pub cuMemPoolImportFromShareableHandle: unsafe extern "C" fn(
            pool_out: *mut CUmemoryPool,
            handle: *mut ::core::ffi::c_void,
            handleType: CUmemAllocationHandleType,
            flags: ::core::ffi::c_ulonglong,
        ) -> CUresult,
        pub cuMemPoolImportPointer: unsafe extern "C" fn(
            ptr_out: *mut CUdeviceptr,
            pool: CUmemoryPool,
            shareData: *mut CUmemPoolPtrExportData,
        ) -> CUresult,
        pub cuMemPoolSetAccess: unsafe extern "C" fn(
            pool: CUmemoryPool,
            map: *const CUmemAccessDesc,
            count: usize,
        ) -> CUresult,
        pub cuMemPoolSetAttribute: unsafe extern "C" fn(
            pool: CUmemoryPool,
            attr: CUmemPool_attribute,
            value: *mut ::core::ffi::c_void,
        ) -> CUresult,
        pub cuMemPoolTrimTo: unsafe extern "C" fn(
            pool: CUmemoryPool,
            minBytesToKeep: usize,
        ) -> CUresult,
        pub cuMemPrefetchAsync: unsafe extern "C" fn(
            devPtr: CUdeviceptr,
            count: usize,
            dstDevice: CUdevice,
            hStream: CUstream,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuMemPrefetchAsync_v2: unsafe extern "C" fn(
            devPtr: CUdeviceptr,
            count: usize,
            location: CUmemLocation,
            flags: ::core::ffi::c_uint,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemRangeGetAttribute: unsafe extern "C" fn(
            data: *mut ::core::ffi::c_void,
            dataSize: usize,
            attribute: CUmem_range_attribute,
            devPtr: CUdeviceptr,
            count: usize,
        ) -> CUresult,
        pub cuMemRangeGetAttributes: unsafe extern "C" fn(
            data: *mut *mut ::core::ffi::c_void,
            dataSizes: *mut usize,
            attributes: *mut CUmem_range_attribute,
            numAttributes: usize,
            devPtr: CUdeviceptr,
            count: usize,
        ) -> CUresult,
        pub cuMemRelease: unsafe extern "C" fn(
            handle: CUmemGenericAllocationHandle,
        ) -> CUresult,
        pub cuMemRetainAllocationHandle: unsafe extern "C" fn(
            handle: *mut CUmemGenericAllocationHandle,
            addr: *mut ::core::ffi::c_void,
        ) -> CUresult,
        pub cuMemSetAccess: unsafe extern "C" fn(
            ptr: CUdeviceptr,
            size: usize,
            desc: *const CUmemAccessDesc,
            count: usize,
        ) -> CUresult,
        pub cuMemUnmap: unsafe extern "C" fn(ptr: CUdeviceptr, size: usize) -> CUresult,
        pub cuMemcpy: unsafe extern "C" fn(
            dst: CUdeviceptr,
            src: CUdeviceptr,
            ByteCount: usize,
        ) -> CUresult,
        pub cuMemcpy2DAsync_v2: unsafe extern "C" fn(
            pCopy: *const CUDA_MEMCPY2D,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemcpy2DUnaligned_v2: unsafe extern "C" fn(
            pCopy: *const CUDA_MEMCPY2D,
        ) -> CUresult,
        pub cuMemcpy2D_v2: unsafe extern "C" fn(pCopy: *const CUDA_MEMCPY2D) -> CUresult,
        pub cuMemcpy3DAsync_v2: unsafe extern "C" fn(
            pCopy: *const CUDA_MEMCPY3D,
            hStream: CUstream,
        ) -> CUresult,
        #[cfg(any(feature = "cuda-12080"))]
        pub cuMemcpy3DBatchAsync: unsafe extern "C" fn(
            numOps: usize,
            opList: *mut CUDA_MEMCPY3D_BATCH_OP,
            failIdx: *mut usize,
            flags: ::core::ffi::c_ulonglong,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemcpy3DPeer: unsafe extern "C" fn(
            pCopy: *const CUDA_MEMCPY3D_PEER,
        ) -> CUresult,
        pub cuMemcpy3DPeerAsync: unsafe extern "C" fn(
            pCopy: *const CUDA_MEMCPY3D_PEER,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemcpy3D_v2: unsafe extern "C" fn(pCopy: *const CUDA_MEMCPY3D) -> CUresult,
        pub cuMemcpyAsync: unsafe extern "C" fn(
            dst: CUdeviceptr,
            src: CUdeviceptr,
            ByteCount: usize,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemcpyAtoA_v2: unsafe extern "C" fn(
            dstArray: CUarray,
            dstOffset: usize,
            srcArray: CUarray,
            srcOffset: usize,
            ByteCount: usize,
        ) -> CUresult,
        pub cuMemcpyAtoD_v2: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            srcArray: CUarray,
            srcOffset: usize,
            ByteCount: usize,
        ) -> CUresult,
        pub cuMemcpyAtoHAsync_v2: unsafe extern "C" fn(
            dstHost: *mut ::core::ffi::c_void,
            srcArray: CUarray,
            srcOffset: usize,
            ByteCount: usize,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemcpyAtoH_v2: unsafe extern "C" fn(
            dstHost: *mut ::core::ffi::c_void,
            srcArray: CUarray,
            srcOffset: usize,
            ByteCount: usize,
        ) -> CUresult,
        #[cfg(any(feature = "cuda-12080"))]
        pub cuMemcpyBatchAsync: unsafe extern "C" fn(
            dsts: *mut CUdeviceptr,
            srcs: *mut CUdeviceptr,
            sizes: *mut usize,
            count: usize,
            attrs: *mut CUmemcpyAttributes,
            attrsIdxs: *mut usize,
            numAttrs: usize,
            failIdx: *mut usize,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemcpyDtoA_v2: unsafe extern "C" fn(
            dstArray: CUarray,
            dstOffset: usize,
            srcDevice: CUdeviceptr,
            ByteCount: usize,
        ) -> CUresult,
        pub cuMemcpyDtoDAsync_v2: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            srcDevice: CUdeviceptr,
            ByteCount: usize,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemcpyDtoD_v2: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            srcDevice: CUdeviceptr,
            ByteCount: usize,
        ) -> CUresult,
        pub cuMemcpyDtoHAsync_v2: unsafe extern "C" fn(
            dstHost: *mut ::core::ffi::c_void,
            srcDevice: CUdeviceptr,
            ByteCount: usize,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemcpyDtoH_v2: unsafe extern "C" fn(
            dstHost: *mut ::core::ffi::c_void,
            srcDevice: CUdeviceptr,
            ByteCount: usize,
        ) -> CUresult,
        pub cuMemcpyHtoAAsync_v2: unsafe extern "C" fn(
            dstArray: CUarray,
            dstOffset: usize,
            srcHost: *const ::core::ffi::c_void,
            ByteCount: usize,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemcpyHtoA_v2: unsafe extern "C" fn(
            dstArray: CUarray,
            dstOffset: usize,
            srcHost: *const ::core::ffi::c_void,
            ByteCount: usize,
        ) -> CUresult,
        pub cuMemcpyHtoDAsync_v2: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            srcHost: *const ::core::ffi::c_void,
            ByteCount: usize,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemcpyHtoD_v2: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            srcHost: *const ::core::ffi::c_void,
            ByteCount: usize,
        ) -> CUresult,
        pub cuMemcpyPeer: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            dstContext: CUcontext,
            srcDevice: CUdeviceptr,
            srcContext: CUcontext,
            ByteCount: usize,
        ) -> CUresult,
        pub cuMemcpyPeerAsync: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            dstContext: CUcontext,
            srcDevice: CUdeviceptr,
            srcContext: CUcontext,
            ByteCount: usize,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemsetD16Async: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            us: ::core::ffi::c_ushort,
            N: usize,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemsetD16_v2: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            us: ::core::ffi::c_ushort,
            N: usize,
        ) -> CUresult,
        pub cuMemsetD2D16Async: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            dstPitch: usize,
            us: ::core::ffi::c_ushort,
            Width: usize,
            Height: usize,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemsetD2D16_v2: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            dstPitch: usize,
            us: ::core::ffi::c_ushort,
            Width: usize,
            Height: usize,
        ) -> CUresult,
        pub cuMemsetD2D32Async: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            dstPitch: usize,
            ui: ::core::ffi::c_uint,
            Width: usize,
            Height: usize,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemsetD2D32_v2: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            dstPitch: usize,
            ui: ::core::ffi::c_uint,
            Width: usize,
            Height: usize,
        ) -> CUresult,
        pub cuMemsetD2D8Async: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            dstPitch: usize,
            uc: ::core::ffi::c_uchar,
            Width: usize,
            Height: usize,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemsetD2D8_v2: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            dstPitch: usize,
            uc: ::core::ffi::c_uchar,
            Width: usize,
            Height: usize,
        ) -> CUresult,
        pub cuMemsetD32Async: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            ui: ::core::ffi::c_uint,
            N: usize,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemsetD32_v2: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            ui: ::core::ffi::c_uint,
            N: usize,
        ) -> CUresult,
        pub cuMemsetD8Async: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            uc: ::core::ffi::c_uchar,
            N: usize,
            hStream: CUstream,
        ) -> CUresult,
        pub cuMemsetD8_v2: unsafe extern "C" fn(
            dstDevice: CUdeviceptr,
            uc: ::core::ffi::c_uchar,
            N: usize,
        ) -> CUresult,
        pub cuMipmappedArrayCreate: unsafe extern "C" fn(
            pHandle: *mut CUmipmappedArray,
            pMipmappedArrayDesc: *const CUDA_ARRAY3D_DESCRIPTOR,
            numMipmapLevels: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuMipmappedArrayDestroy: unsafe extern "C" fn(
            hMipmappedArray: CUmipmappedArray,
        ) -> CUresult,
        pub cuMipmappedArrayGetLevel: unsafe extern "C" fn(
            pLevelArray: *mut CUarray,
            hMipmappedArray: CUmipmappedArray,
            level: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
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
            )
        )]
        pub cuMipmappedArrayGetMemoryRequirements: unsafe extern "C" fn(
            memoryRequirements: *mut CUDA_ARRAY_MEMORY_REQUIREMENTS,
            mipmap: CUmipmappedArray,
            device: CUdevice,
        ) -> CUresult,
        pub cuMipmappedArrayGetSparseProperties: unsafe extern "C" fn(
            sparseProperties: *mut CUDA_ARRAY_SPARSE_PROPERTIES,
            mipmap: CUmipmappedArray,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuModuleEnumerateFunctions: unsafe extern "C" fn(
            functions: *mut CUfunction,
            numFunctions: ::core::ffi::c_uint,
            mod_: CUmodule,
        ) -> CUresult,
        pub cuModuleGetFunction: unsafe extern "C" fn(
            hfunc: *mut CUfunction,
            hmod: CUmodule,
            name: *const ::core::ffi::c_char,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuModuleGetFunctionCount: unsafe extern "C" fn(
            count: *mut ::core::ffi::c_uint,
            mod_: CUmodule,
        ) -> CUresult,
        pub cuModuleGetGlobal_v2: unsafe extern "C" fn(
            dptr: *mut CUdeviceptr,
            bytes: *mut usize,
            hmod: CUmodule,
            name: *const ::core::ffi::c_char,
        ) -> CUresult,
        #[cfg(
            any(
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
            )
        )]
        pub cuModuleGetLoadingMode: unsafe extern "C" fn(
            mode: *mut CUmoduleLoadingMode,
        ) -> CUresult,
        pub cuModuleGetSurfRef: unsafe extern "C" fn(
            pSurfRef: *mut CUsurfref,
            hmod: CUmodule,
            name: *const ::core::ffi::c_char,
        ) -> CUresult,
        pub cuModuleGetTexRef: unsafe extern "C" fn(
            pTexRef: *mut CUtexref,
            hmod: CUmodule,
            name: *const ::core::ffi::c_char,
        ) -> CUresult,
        pub cuModuleLoad: unsafe extern "C" fn(
            module: *mut CUmodule,
            fname: *const ::core::ffi::c_char,
        ) -> CUresult,
        pub cuModuleLoadData: unsafe extern "C" fn(
            module: *mut CUmodule,
            image: *const ::core::ffi::c_void,
        ) -> CUresult,
        pub cuModuleLoadDataEx: unsafe extern "C" fn(
            module: *mut CUmodule,
            image: *const ::core::ffi::c_void,
            numOptions: ::core::ffi::c_uint,
            options: *mut CUjit_option,
            optionValues: *mut *mut ::core::ffi::c_void,
        ) -> CUresult,
        pub cuModuleLoadFatBinary: unsafe extern "C" fn(
            module: *mut CUmodule,
            fatCubin: *const ::core::ffi::c_void,
        ) -> CUresult,
        pub cuModuleUnload: unsafe extern "C" fn(hmod: CUmodule) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuMulticastAddDevice: unsafe extern "C" fn(
            mcHandle: CUmemGenericAllocationHandle,
            dev: CUdevice,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuMulticastBindAddr: unsafe extern "C" fn(
            mcHandle: CUmemGenericAllocationHandle,
            mcOffset: usize,
            memptr: CUdeviceptr,
            size: usize,
            flags: ::core::ffi::c_ulonglong,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuMulticastBindMem: unsafe extern "C" fn(
            mcHandle: CUmemGenericAllocationHandle,
            mcOffset: usize,
            memHandle: CUmemGenericAllocationHandle,
            memOffset: usize,
            size: usize,
            flags: ::core::ffi::c_ulonglong,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuMulticastCreate: unsafe extern "C" fn(
            mcHandle: *mut CUmemGenericAllocationHandle,
            prop: *const CUmulticastObjectProp,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuMulticastGetGranularity: unsafe extern "C" fn(
            granularity: *mut usize,
            prop: *const CUmulticastObjectProp,
            option: CUmulticastGranularity_flags,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuMulticastUnbind: unsafe extern "C" fn(
            mcHandle: CUmemGenericAllocationHandle,
            dev: CUdevice,
            mcOffset: usize,
            size: usize,
        ) -> CUresult,
        pub cuOccupancyAvailableDynamicSMemPerBlock: unsafe extern "C" fn(
            dynamicSmemSize: *mut usize,
            func: CUfunction,
            numBlocks: ::core::ffi::c_int,
            blockSize: ::core::ffi::c_int,
        ) -> CUresult,
        pub cuOccupancyMaxActiveBlocksPerMultiprocessor: unsafe extern "C" fn(
            numBlocks: *mut ::core::ffi::c_int,
            func: CUfunction,
            blockSize: ::core::ffi::c_int,
            dynamicSMemSize: usize,
        ) -> CUresult,
        pub cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags: unsafe extern "C" fn(
            numBlocks: *mut ::core::ffi::c_int,
            func: CUfunction,
            blockSize: ::core::ffi::c_int,
            dynamicSMemSize: usize,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11080",
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuOccupancyMaxActiveClusters: unsafe extern "C" fn(
            numClusters: *mut ::core::ffi::c_int,
            func: CUfunction,
            config: *const CUlaunchConfig,
        ) -> CUresult,
        pub cuOccupancyMaxPotentialBlockSize: unsafe extern "C" fn(
            minGridSize: *mut ::core::ffi::c_int,
            blockSize: *mut ::core::ffi::c_int,
            func: CUfunction,
            blockSizeToDynamicSMemSize: CUoccupancyB2DSize,
            dynamicSMemSize: usize,
            blockSizeLimit: ::core::ffi::c_int,
        ) -> CUresult,
        pub cuOccupancyMaxPotentialBlockSizeWithFlags: unsafe extern "C" fn(
            minGridSize: *mut ::core::ffi::c_int,
            blockSize: *mut ::core::ffi::c_int,
            func: CUfunction,
            blockSizeToDynamicSMemSize: CUoccupancyB2DSize,
            dynamicSMemSize: usize,
            blockSizeLimit: ::core::ffi::c_int,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11080",
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuOccupancyMaxPotentialClusterSize: unsafe extern "C" fn(
            clusterSize: *mut ::core::ffi::c_int,
            func: CUfunction,
            config: *const CUlaunchConfig,
        ) -> CUresult,
        pub cuParamSetSize: unsafe extern "C" fn(
            hfunc: CUfunction,
            numbytes: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuParamSetTexRef: unsafe extern "C" fn(
            hfunc: CUfunction,
            texunit: ::core::ffi::c_int,
            hTexRef: CUtexref,
        ) -> CUresult,
        pub cuParamSetf: unsafe extern "C" fn(
            hfunc: CUfunction,
            offset: ::core::ffi::c_int,
            value: f32,
        ) -> CUresult,
        pub cuParamSeti: unsafe extern "C" fn(
            hfunc: CUfunction,
            offset: ::core::ffi::c_int,
            value: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuParamSetv: unsafe extern "C" fn(
            hfunc: CUfunction,
            offset: ::core::ffi::c_int,
            ptr: *mut ::core::ffi::c_void,
            numbytes: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuPointerGetAttribute: unsafe extern "C" fn(
            data: *mut ::core::ffi::c_void,
            attribute: CUpointer_attribute,
            ptr: CUdeviceptr,
        ) -> CUresult,
        pub cuPointerGetAttributes: unsafe extern "C" fn(
            numAttributes: ::core::ffi::c_uint,
            attributes: *mut CUpointer_attribute,
            data: *mut *mut ::core::ffi::c_void,
            ptr: CUdeviceptr,
        ) -> CUresult,
        pub cuPointerSetAttribute: unsafe extern "C" fn(
            value: *const ::core::ffi::c_void,
            attribute: CUpointer_attribute,
            ptr: CUdeviceptr,
        ) -> CUresult,
        pub cuProfilerInitialize: unsafe extern "C" fn(
            configFile: *const ::core::ffi::c_char,
            outputFile: *const ::core::ffi::c_char,
            outputMode: CUoutput_mode,
        ) -> CUresult,
        pub cuProfilerStart: unsafe extern "C" fn() -> CUresult,
        pub cuProfilerStop: unsafe extern "C" fn() -> CUresult,
        pub cuSignalExternalSemaphoresAsync: unsafe extern "C" fn(
            extSemArray: *const CUexternalSemaphore,
            paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS,
            numExtSems: ::core::ffi::c_uint,
            stream: CUstream,
        ) -> CUresult,
        pub cuStreamAddCallback: unsafe extern "C" fn(
            hStream: CUstream,
            callback: CUstreamCallback,
            userData: *mut ::core::ffi::c_void,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuStreamAttachMemAsync: unsafe extern "C" fn(
            hStream: CUstream,
            dptr: CUdeviceptr,
            length: usize,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            )
        )]
        pub cuStreamBatchMemOp: unsafe extern "C" fn(
            stream: CUstream,
            count: ::core::ffi::c_uint,
            paramArray: *mut CUstreamBatchMemOpParams,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
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
            )
        )]
        pub cuStreamBatchMemOp_v2: unsafe extern "C" fn(
            stream: CUstream,
            count: ::core::ffi::c_uint,
            paramArray: *mut CUstreamBatchMemOpParams,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuStreamBeginCaptureToGraph: unsafe extern "C" fn(
            hStream: CUstream,
            hGraph: CUgraph,
            dependencies: *const CUgraphNode,
            dependencyData: *const CUgraphEdgeData,
            numDependencies: usize,
            mode: CUstreamCaptureMode,
        ) -> CUresult,
        pub cuStreamBeginCapture_v2: unsafe extern "C" fn(
            hStream: CUstream,
            mode: CUstreamCaptureMode,
        ) -> CUresult,
        pub cuStreamCopyAttributes: unsafe extern "C" fn(
            dst: CUstream,
            src: CUstream,
        ) -> CUresult,
        pub cuStreamCreate: unsafe extern "C" fn(
            phStream: *mut CUstream,
            Flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuStreamCreateWithPriority: unsafe extern "C" fn(
            phStream: *mut CUstream,
            flags: ::core::ffi::c_uint,
            priority: ::core::ffi::c_int,
        ) -> CUresult,
        pub cuStreamDestroy_v2: unsafe extern "C" fn(hStream: CUstream) -> CUresult,
        pub cuStreamEndCapture: unsafe extern "C" fn(
            hStream: CUstream,
            phGraph: *mut CUgraph,
        ) -> CUresult,
        pub cuStreamGetAttribute: unsafe extern "C" fn(
            hStream: CUstream,
            attr: CUstreamAttrID,
            value_out: *mut CUstreamAttrValue,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            )
        )]
        pub cuStreamGetCaptureInfo: unsafe extern "C" fn(
            hStream: CUstream,
            captureStatus_out: *mut CUstreamCaptureStatus,
            id_out: *mut cuuint64_t,
        ) -> CUresult,
        pub cuStreamGetCaptureInfo_v2: unsafe extern "C" fn(
            hStream: CUstream,
            captureStatus_out: *mut CUstreamCaptureStatus,
            id_out: *mut cuuint64_t,
            graph_out: *mut CUgraph,
            dependencies_out: *mut *const CUgraphNode,
            numDependencies_out: *mut usize,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuStreamGetCaptureInfo_v3: unsafe extern "C" fn(
            hStream: CUstream,
            captureStatus_out: *mut CUstreamCaptureStatus,
            id_out: *mut cuuint64_t,
            graph_out: *mut CUgraph,
            dependencies_out: *mut *const CUgraphNode,
            edgeData_out: *mut *const CUgraphEdgeData,
            numDependencies_out: *mut usize,
        ) -> CUresult,
        pub cuStreamGetCtx: unsafe extern "C" fn(
            hStream: CUstream,
            pctx: *mut CUcontext,
        ) -> CUresult,
        #[cfg(
            any(feature = "cuda-12050", feature = "cuda-12060", feature = "cuda-12080")
        )]
        pub cuStreamGetCtx_v2: unsafe extern "C" fn(
            hStream: CUstream,
            pCtx: *mut CUcontext,
            pGreenCtx: *mut CUgreenCtx,
        ) -> CUresult,
        #[cfg(any(feature = "cuda-12080"))]
        pub cuStreamGetDevice: unsafe extern "C" fn(
            hStream: CUstream,
            device: *mut CUdevice,
        ) -> CUresult,
        pub cuStreamGetFlags: unsafe extern "C" fn(
            hStream: CUstream,
            flags: *mut ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuStreamGetGreenCtx: unsafe extern "C" fn(
            hStream: CUstream,
            phCtx: *mut CUgreenCtx,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuStreamGetId: unsafe extern "C" fn(
            hStream: CUstream,
            streamId: *mut ::core::ffi::c_ulonglong,
        ) -> CUresult,
        pub cuStreamGetPriority: unsafe extern "C" fn(
            hStream: CUstream,
            priority: *mut ::core::ffi::c_int,
        ) -> CUresult,
        pub cuStreamIsCapturing: unsafe extern "C" fn(
            hStream: CUstream,
            captureStatus: *mut CUstreamCaptureStatus,
        ) -> CUresult,
        pub cuStreamQuery: unsafe extern "C" fn(hStream: CUstream) -> CUresult,
        pub cuStreamSetAttribute: unsafe extern "C" fn(
            hStream: CUstream,
            attr: CUstreamAttrID,
            value: *const CUstreamAttrValue,
        ) -> CUresult,
        pub cuStreamSynchronize: unsafe extern "C" fn(hStream: CUstream) -> CUresult,
        pub cuStreamUpdateCaptureDependencies: unsafe extern "C" fn(
            hStream: CUstream,
            dependencies: *mut CUgraphNode,
            numDependencies: usize,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuStreamUpdateCaptureDependencies_v2: unsafe extern "C" fn(
            hStream: CUstream,
            dependencies: *mut CUgraphNode,
            dependencyData: *const CUgraphEdgeData,
            numDependencies: usize,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuStreamWaitEvent: unsafe extern "C" fn(
            hStream: CUstream,
            hEvent: CUevent,
            Flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            )
        )]
        pub cuStreamWaitValue32: unsafe extern "C" fn(
            stream: CUstream,
            addr: CUdeviceptr,
            value: cuuint32_t,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
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
            )
        )]
        pub cuStreamWaitValue32_v2: unsafe extern "C" fn(
            stream: CUstream,
            addr: CUdeviceptr,
            value: cuuint32_t,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            )
        )]
        pub cuStreamWaitValue64: unsafe extern "C" fn(
            stream: CUstream,
            addr: CUdeviceptr,
            value: cuuint64_t,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
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
            )
        )]
        pub cuStreamWaitValue64_v2: unsafe extern "C" fn(
            stream: CUstream,
            addr: CUdeviceptr,
            value: cuuint64_t,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            )
        )]
        pub cuStreamWriteValue32: unsafe extern "C" fn(
            stream: CUstream,
            addr: CUdeviceptr,
            value: cuuint32_t,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
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
            )
        )]
        pub cuStreamWriteValue32_v2: unsafe extern "C" fn(
            stream: CUstream,
            addr: CUdeviceptr,
            value: cuuint32_t,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-11040",
                feature = "cuda-11050",
                feature = "cuda-11060",
                feature = "cuda-11070",
                feature = "cuda-11080"
            )
        )]
        pub cuStreamWriteValue64: unsafe extern "C" fn(
            stream: CUstream,
            addr: CUdeviceptr,
            value: cuuint64_t,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
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
            )
        )]
        pub cuStreamWriteValue64_v2: unsafe extern "C" fn(
            stream: CUstream,
            addr: CUdeviceptr,
            value: cuuint64_t,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuSurfObjectCreate: unsafe extern "C" fn(
            pSurfObject: *mut CUsurfObject,
            pResDesc: *const CUDA_RESOURCE_DESC,
        ) -> CUresult,
        pub cuSurfObjectDestroy: unsafe extern "C" fn(
            surfObject: CUsurfObject,
        ) -> CUresult,
        pub cuSurfObjectGetResourceDesc: unsafe extern "C" fn(
            pResDesc: *mut CUDA_RESOURCE_DESC,
            surfObject: CUsurfObject,
        ) -> CUresult,
        pub cuSurfRefGetArray: unsafe extern "C" fn(
            phArray: *mut CUarray,
            hSurfRef: CUsurfref,
        ) -> CUresult,
        pub cuSurfRefSetArray: unsafe extern "C" fn(
            hSurfRef: CUsurfref,
            hArray: CUarray,
            Flags: ::core::ffi::c_uint,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuTensorMapEncodeIm2col: unsafe extern "C" fn(
            tensorMap: *mut CUtensorMap,
            tensorDataType: CUtensorMapDataType,
            tensorRank: cuuint32_t,
            globalAddress: *mut ::core::ffi::c_void,
            globalDim: *const cuuint64_t,
            globalStrides: *const cuuint64_t,
            pixelBoxLowerCorner: *const ::core::ffi::c_int,
            pixelBoxUpperCorner: *const ::core::ffi::c_int,
            channelsPerPixel: cuuint32_t,
            pixelsPerColumn: cuuint32_t,
            elementStrides: *const cuuint32_t,
            interleave: CUtensorMapInterleave,
            swizzle: CUtensorMapSwizzle,
            l2Promotion: CUtensorMapL2promotion,
            oobFill: CUtensorMapFloatOOBfill,
        ) -> CUresult,
        #[cfg(any(feature = "cuda-12080"))]
        pub cuTensorMapEncodeIm2colWide: unsafe extern "C" fn(
            tensorMap: *mut CUtensorMap,
            tensorDataType: CUtensorMapDataType,
            tensorRank: cuuint32_t,
            globalAddress: *mut ::core::ffi::c_void,
            globalDim: *const cuuint64_t,
            globalStrides: *const cuuint64_t,
            pixelBoxLowerCornerWidth: ::core::ffi::c_int,
            pixelBoxUpperCornerWidth: ::core::ffi::c_int,
            channelsPerPixel: cuuint32_t,
            pixelsPerColumn: cuuint32_t,
            elementStrides: *const cuuint32_t,
            interleave: CUtensorMapInterleave,
            mode: CUtensorMapIm2ColWideMode,
            swizzle: CUtensorMapSwizzle,
            l2Promotion: CUtensorMapL2promotion,
            oobFill: CUtensorMapFloatOOBfill,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuTensorMapEncodeTiled: unsafe extern "C" fn(
            tensorMap: *mut CUtensorMap,
            tensorDataType: CUtensorMapDataType,
            tensorRank: cuuint32_t,
            globalAddress: *mut ::core::ffi::c_void,
            globalDim: *const cuuint64_t,
            globalStrides: *const cuuint64_t,
            boxDim: *const cuuint32_t,
            elementStrides: *const cuuint32_t,
            interleave: CUtensorMapInterleave,
            swizzle: CUtensorMapSwizzle,
            l2Promotion: CUtensorMapL2promotion,
            oobFill: CUtensorMapFloatOOBfill,
        ) -> CUresult,
        #[cfg(
            any(
                feature = "cuda-12000",
                feature = "cuda-12010",
                feature = "cuda-12020",
                feature = "cuda-12030",
                feature = "cuda-12040",
                feature = "cuda-12050",
                feature = "cuda-12060",
                feature = "cuda-12080"
            )
        )]
        pub cuTensorMapReplaceAddress: unsafe extern "C" fn(
            tensorMap: *mut CUtensorMap,
            globalAddress: *mut ::core::ffi::c_void,
        ) -> CUresult,
        pub cuTexObjectCreate: unsafe extern "C" fn(
            pTexObject: *mut CUtexObject,
            pResDesc: *const CUDA_RESOURCE_DESC,
            pTexDesc: *const CUDA_TEXTURE_DESC,
            pResViewDesc: *const CUDA_RESOURCE_VIEW_DESC,
        ) -> CUresult,
        pub cuTexObjectDestroy: unsafe extern "C" fn(texObject: CUtexObject) -> CUresult,
        pub cuTexObjectGetResourceDesc: unsafe extern "C" fn(
            pResDesc: *mut CUDA_RESOURCE_DESC,
            texObject: CUtexObject,
        ) -> CUresult,
        pub cuTexObjectGetResourceViewDesc: unsafe extern "C" fn(
            pResViewDesc: *mut CUDA_RESOURCE_VIEW_DESC,
            texObject: CUtexObject,
        ) -> CUresult,
        pub cuTexObjectGetTextureDesc: unsafe extern "C" fn(
            pTexDesc: *mut CUDA_TEXTURE_DESC,
            texObject: CUtexObject,
        ) -> CUresult,
        pub cuTexRefCreate: unsafe extern "C" fn(pTexRef: *mut CUtexref) -> CUresult,
        pub cuTexRefDestroy: unsafe extern "C" fn(hTexRef: CUtexref) -> CUresult,
        pub cuTexRefGetAddressMode: unsafe extern "C" fn(
            pam: *mut CUaddress_mode,
            hTexRef: CUtexref,
            dim: ::core::ffi::c_int,
        ) -> CUresult,
        pub cuTexRefGetAddress_v2: unsafe extern "C" fn(
            pdptr: *mut CUdeviceptr,
            hTexRef: CUtexref,
        ) -> CUresult,
        pub cuTexRefGetArray: unsafe extern "C" fn(
            phArray: *mut CUarray,
            hTexRef: CUtexref,
        ) -> CUresult,
        pub cuTexRefGetBorderColor: unsafe extern "C" fn(
            pBorderColor: *mut f32,
            hTexRef: CUtexref,
        ) -> CUresult,
        pub cuTexRefGetFilterMode: unsafe extern "C" fn(
            pfm: *mut CUfilter_mode,
            hTexRef: CUtexref,
        ) -> CUresult,
        pub cuTexRefGetFlags: unsafe extern "C" fn(
            pFlags: *mut ::core::ffi::c_uint,
            hTexRef: CUtexref,
        ) -> CUresult,
        pub cuTexRefGetFormat: unsafe extern "C" fn(
            pFormat: *mut CUarray_format,
            pNumChannels: *mut ::core::ffi::c_int,
            hTexRef: CUtexref,
        ) -> CUresult,
        pub cuTexRefGetMaxAnisotropy: unsafe extern "C" fn(
            pmaxAniso: *mut ::core::ffi::c_int,
            hTexRef: CUtexref,
        ) -> CUresult,
        pub cuTexRefGetMipmapFilterMode: unsafe extern "C" fn(
            pfm: *mut CUfilter_mode,
            hTexRef: CUtexref,
        ) -> CUresult,
        pub cuTexRefGetMipmapLevelBias: unsafe extern "C" fn(
            pbias: *mut f32,
            hTexRef: CUtexref,
        ) -> CUresult,
        pub cuTexRefGetMipmapLevelClamp: unsafe extern "C" fn(
            pminMipmapLevelClamp: *mut f32,
            pmaxMipmapLevelClamp: *mut f32,
            hTexRef: CUtexref,
        ) -> CUresult,
        pub cuTexRefGetMipmappedArray: unsafe extern "C" fn(
            phMipmappedArray: *mut CUmipmappedArray,
            hTexRef: CUtexref,
        ) -> CUresult,
        pub cuTexRefSetAddress2D_v3: unsafe extern "C" fn(
            hTexRef: CUtexref,
            desc: *const CUDA_ARRAY_DESCRIPTOR,
            dptr: CUdeviceptr,
            Pitch: usize,
        ) -> CUresult,
        pub cuTexRefSetAddressMode: unsafe extern "C" fn(
            hTexRef: CUtexref,
            dim: ::core::ffi::c_int,
            am: CUaddress_mode,
        ) -> CUresult,
        pub cuTexRefSetAddress_v2: unsafe extern "C" fn(
            ByteOffset: *mut usize,
            hTexRef: CUtexref,
            dptr: CUdeviceptr,
            bytes: usize,
        ) -> CUresult,
        pub cuTexRefSetArray: unsafe extern "C" fn(
            hTexRef: CUtexref,
            hArray: CUarray,
            Flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuTexRefSetBorderColor: unsafe extern "C" fn(
            hTexRef: CUtexref,
            pBorderColor: *mut f32,
        ) -> CUresult,
        pub cuTexRefSetFilterMode: unsafe extern "C" fn(
            hTexRef: CUtexref,
            fm: CUfilter_mode,
        ) -> CUresult,
        pub cuTexRefSetFlags: unsafe extern "C" fn(
            hTexRef: CUtexref,
            Flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuTexRefSetFormat: unsafe extern "C" fn(
            hTexRef: CUtexref,
            fmt: CUarray_format,
            NumPackedComponents: ::core::ffi::c_int,
        ) -> CUresult,
        pub cuTexRefSetMaxAnisotropy: unsafe extern "C" fn(
            hTexRef: CUtexref,
            maxAniso: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuTexRefSetMipmapFilterMode: unsafe extern "C" fn(
            hTexRef: CUtexref,
            fm: CUfilter_mode,
        ) -> CUresult,
        pub cuTexRefSetMipmapLevelBias: unsafe extern "C" fn(
            hTexRef: CUtexref,
            bias: f32,
        ) -> CUresult,
        pub cuTexRefSetMipmapLevelClamp: unsafe extern "C" fn(
            hTexRef: CUtexref,
            minMipmapLevelClamp: f32,
            maxMipmapLevelClamp: f32,
        ) -> CUresult,
        pub cuTexRefSetMipmappedArray: unsafe extern "C" fn(
            hTexRef: CUtexref,
            hMipmappedArray: CUmipmappedArray,
            Flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuThreadExchangeStreamCaptureMode: unsafe extern "C" fn(
            mode: *mut CUstreamCaptureMode,
        ) -> CUresult,
        pub cuUserObjectCreate: unsafe extern "C" fn(
            object_out: *mut CUuserObject,
            ptr: *mut ::core::ffi::c_void,
            destroy: CUhostFn,
            initialRefcount: ::core::ffi::c_uint,
            flags: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuUserObjectRelease: unsafe extern "C" fn(
            object: CUuserObject,
            count: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuUserObjectRetain: unsafe extern "C" fn(
            object: CUuserObject,
            count: ::core::ffi::c_uint,
        ) -> CUresult,
        pub cuWaitExternalSemaphoresAsync: unsafe extern "C" fn(
            extSemArray: *const CUexternalSemaphore,
            paramsArray: *const CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS,
            numExtSems: ::core::ffi::c_uint,
            stream: CUstream,
        ) -> CUresult,
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
            let cuArray3DCreate_v2 = __library
                .get(b"cuArray3DCreate_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuArray3DGetDescriptor_v2 = __library
                .get(b"cuArray3DGetDescriptor_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuArrayCreate_v2 = __library
                .get(b"cuArrayCreate_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuArrayDestroy = __library
                .get(b"cuArrayDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuArrayGetDescriptor_v2 = __library
                .get(b"cuArrayGetDescriptor_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
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
                )
            )]
            let cuArrayGetMemoryRequirements = __library
                .get(b"cuArrayGetMemoryRequirements\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuArrayGetPlane = __library
                .get(b"cuArrayGetPlane\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuArrayGetSparseProperties = __library
                .get(b"cuArrayGetSparseProperties\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080"))]
            let cuCheckpointProcessCheckpoint = __library
                .get(b"cuCheckpointProcessCheckpoint\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080"))]
            let cuCheckpointProcessGetRestoreThreadId = __library
                .get(b"cuCheckpointProcessGetRestoreThreadId\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080"))]
            let cuCheckpointProcessGetState = __library
                .get(b"cuCheckpointProcessGetState\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080"))]
            let cuCheckpointProcessLock = __library
                .get(b"cuCheckpointProcessLock\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080"))]
            let cuCheckpointProcessRestore = __library
                .get(b"cuCheckpointProcessRestore\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080"))]
            let cuCheckpointProcessUnlock = __library
                .get(b"cuCheckpointProcessUnlock\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuCoredumpGetAttribute = __library
                .get(b"cuCoredumpGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuCoredumpGetAttributeGlobal = __library
                .get(b"cuCoredumpGetAttributeGlobal\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuCoredumpSetAttribute = __library
                .get(b"cuCoredumpSetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuCoredumpSetAttributeGlobal = __library
                .get(b"cuCoredumpSetAttributeGlobal\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxAttach = __library
                .get(b"cuCtxAttach\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxCreate_v2 = __library
                .get(b"cuCtxCreate_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxCreate_v3 = __library
                .get(b"cuCtxCreate_v3\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuCtxCreate_v4 = __library
                .get(b"cuCtxCreate_v4\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxDestroy_v2 = __library
                .get(b"cuCtxDestroy_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxDetach = __library
                .get(b"cuCtxDetach\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxDisablePeerAccess = __library
                .get(b"cuCtxDisablePeerAccess\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxEnablePeerAccess = __library
                .get(b"cuCtxEnablePeerAccess\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuCtxFromGreenCtx = __library
                .get(b"cuCtxFromGreenCtx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxGetApiVersion = __library
                .get(b"cuCtxGetApiVersion\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxGetCacheConfig = __library
                .get(b"cuCtxGetCacheConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxGetCurrent = __library
                .get(b"cuCtxGetCurrent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuCtxGetDevResource = __library
                .get(b"cuCtxGetDevResource\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxGetDevice = __library
                .get(b"cuCtxGetDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxGetExecAffinity = __library
                .get(b"cuCtxGetExecAffinity\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxGetFlags = __library
                .get(b"cuCtxGetFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuCtxGetId = __library
                .get(b"cuCtxGetId\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxGetLimit = __library
                .get(b"cuCtxGetLimit\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxGetSharedMemConfig = __library
                .get(b"cuCtxGetSharedMemConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxGetStreamPriorityRange = __library
                .get(b"cuCtxGetStreamPriorityRange\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxPopCurrent_v2 = __library
                .get(b"cuCtxPopCurrent_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxPushCurrent_v2 = __library
                .get(b"cuCtxPushCurrent_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuCtxRecordEvent = __library
                .get(b"cuCtxRecordEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxResetPersistingL2Cache = __library
                .get(b"cuCtxResetPersistingL2Cache\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxSetCacheConfig = __library
                .get(b"cuCtxSetCacheConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxSetCurrent = __library
                .get(b"cuCtxSetCurrent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuCtxSetFlags = __library
                .get(b"cuCtxSetFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxSetLimit = __library
                .get(b"cuCtxSetLimit\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxSetSharedMemConfig = __library
                .get(b"cuCtxSetSharedMemConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuCtxSynchronize = __library
                .get(b"cuCtxSynchronize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuCtxWaitEvent = __library
                .get(b"cuCtxWaitEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDestroyExternalMemory = __library
                .get(b"cuDestroyExternalMemory\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDestroyExternalSemaphore = __library
                .get(b"cuDestroyExternalSemaphore\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuDevResourceGenerateDesc = __library
                .get(b"cuDevResourceGenerateDesc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuDevSmResourceSplitByCount = __library
                .get(b"cuDevSmResourceSplitByCount\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceCanAccessPeer = __library
                .get(b"cuDeviceCanAccessPeer\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceComputeCapability = __library
                .get(b"cuDeviceComputeCapability\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGet = __library
                .get(b"cuDeviceGet\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetAttribute = __library
                .get(b"cuDeviceGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetByPCIBusId = __library
                .get(b"cuDeviceGetByPCIBusId\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetCount = __library
                .get(b"cuDeviceGetCount\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetDefaultMemPool = __library
                .get(b"cuDeviceGetDefaultMemPool\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuDeviceGetDevResource = __library
                .get(b"cuDeviceGetDevResource\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetExecAffinitySupport = __library
                .get(b"cuDeviceGetExecAffinitySupport\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetGraphMemAttribute = __library
                .get(b"cuDeviceGetGraphMemAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetLuid = __library
                .get(b"cuDeviceGetLuid\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetMemPool = __library
                .get(b"cuDeviceGetMemPool\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetName = __library
                .get(b"cuDeviceGetName\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetNvSciSyncAttributes = __library
                .get(b"cuDeviceGetNvSciSyncAttributes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetP2PAttribute = __library
                .get(b"cuDeviceGetP2PAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetPCIBusId = __library
                .get(b"cuDeviceGetPCIBusId\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetProperties = __library
                .get(b"cuDeviceGetProperties\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetTexture1DLinearMaxWidth = __library
                .get(b"cuDeviceGetTexture1DLinearMaxWidth\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetUuid = __library
                .get(b"cuDeviceGetUuid\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGetUuid_v2 = __library
                .get(b"cuDeviceGetUuid_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceGraphMemTrim = __library
                .get(b"cuDeviceGraphMemTrim\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDevicePrimaryCtxGetState = __library
                .get(b"cuDevicePrimaryCtxGetState\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDevicePrimaryCtxRelease_v2 = __library
                .get(b"cuDevicePrimaryCtxRelease_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDevicePrimaryCtxReset_v2 = __library
                .get(b"cuDevicePrimaryCtxReset_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDevicePrimaryCtxRetain = __library
                .get(b"cuDevicePrimaryCtxRetain\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDevicePrimaryCtxSetFlags_v2 = __library
                .get(b"cuDevicePrimaryCtxSetFlags_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuDeviceRegisterAsyncNotification = __library
                .get(b"cuDeviceRegisterAsyncNotification\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceSetGraphMemAttribute = __library
                .get(b"cuDeviceSetGraphMemAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceSetMemPool = __library
                .get(b"cuDeviceSetMemPool\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDeviceTotalMem_v2 = __library
                .get(b"cuDeviceTotalMem_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuDeviceUnregisterAsyncNotification = __library
                .get(b"cuDeviceUnregisterAsyncNotification\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuDriverGetVersion = __library
                .get(b"cuDriverGetVersion\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuEventCreate = __library
                .get(b"cuEventCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuEventDestroy_v2 = __library
                .get(b"cuEventDestroy_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuEventElapsedTime = __library
                .get(b"cuEventElapsedTime\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080"))]
            let cuEventElapsedTime_v2 = __library
                .get(b"cuEventElapsedTime_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuEventQuery = __library
                .get(b"cuEventQuery\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuEventRecord = __library
                .get(b"cuEventRecord\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuEventRecordWithFlags = __library
                .get(b"cuEventRecordWithFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuEventSynchronize = __library
                .get(b"cuEventSynchronize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuExternalMemoryGetMappedBuffer = __library
                .get(b"cuExternalMemoryGetMappedBuffer\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuExternalMemoryGetMappedMipmappedArray = __library
                .get(b"cuExternalMemoryGetMappedMipmappedArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFlushGPUDirectRDMAWrites = __library
                .get(b"cuFlushGPUDirectRDMAWrites\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFuncGetAttribute = __library
                .get(b"cuFuncGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFuncGetModule = __library
                .get(b"cuFuncGetModule\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuFuncGetName = __library
                .get(b"cuFuncGetName\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuFuncGetParamInfo = __library
                .get(b"cuFuncGetParamInfo\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuFuncIsLoaded = __library
                .get(b"cuFuncIsLoaded\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuFuncLoad = __library
                .get(b"cuFuncLoad\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFuncSetAttribute = __library
                .get(b"cuFuncSetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFuncSetBlockShape = __library
                .get(b"cuFuncSetBlockShape\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFuncSetCacheConfig = __library
                .get(b"cuFuncSetCacheConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFuncSetSharedMemConfig = __library
                .get(b"cuFuncSetSharedMemConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuFuncSetSharedSize = __library
                .get(b"cuFuncSetSharedSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGetErrorName = __library
                .get(b"cuGetErrorName\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGetErrorString = __library
                .get(b"cuGetErrorString\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGetExportTable = __library
                .get(b"cuGetExportTable\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                )
            )]
            let cuGetProcAddress = __library
                .get(b"cuGetProcAddress\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGetProcAddress_v2 = __library
                .get(b"cuGetProcAddress_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
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
                )
            )]
            let cuGraphAddBatchMemOpNode = __library
                .get(b"cuGraphAddBatchMemOpNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphAddChildGraphNode = __library
                .get(b"cuGraphAddChildGraphNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphAddDependencies = __library
                .get(b"cuGraphAddDependencies\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphAddDependencies_v2 = __library
                .get(b"cuGraphAddDependencies_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphAddEmptyNode = __library
                .get(b"cuGraphAddEmptyNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphAddEventRecordNode = __library
                .get(b"cuGraphAddEventRecordNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphAddEventWaitNode = __library
                .get(b"cuGraphAddEventWaitNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphAddExternalSemaphoresSignalNode = __library
                .get(b"cuGraphAddExternalSemaphoresSignalNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphAddExternalSemaphoresWaitNode = __library
                .get(b"cuGraphAddExternalSemaphoresWaitNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphAddHostNode = __library
                .get(b"cuGraphAddHostNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                )
            )]
            let cuGraphAddKernelNode = __library
                .get(b"cuGraphAddKernelNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphAddKernelNode_v2 = __library
                .get(b"cuGraphAddKernelNode_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphAddMemAllocNode = __library
                .get(b"cuGraphAddMemAllocNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphAddMemFreeNode = __library
                .get(b"cuGraphAddMemFreeNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphAddMemcpyNode = __library
                .get(b"cuGraphAddMemcpyNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphAddMemsetNode = __library
                .get(b"cuGraphAddMemsetNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphAddNode = __library
                .get(b"cuGraphAddNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphAddNode_v2 = __library
                .get(b"cuGraphAddNode_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
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
                )
            )]
            let cuGraphBatchMemOpNodeGetParams = __library
                .get(b"cuGraphBatchMemOpNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
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
                )
            )]
            let cuGraphBatchMemOpNodeSetParams = __library
                .get(b"cuGraphBatchMemOpNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphChildGraphNodeGetGraph = __library
                .get(b"cuGraphChildGraphNodeGetGraph\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphClone = __library
                .get(b"cuGraphClone\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphConditionalHandleCreate = __library
                .get(b"cuGraphConditionalHandleCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphCreate = __library
                .get(b"cuGraphCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphDebugDotPrint = __library
                .get(b"cuGraphDebugDotPrint\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphDestroy = __library
                .get(b"cuGraphDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphDestroyNode = __library
                .get(b"cuGraphDestroyNode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphEventRecordNodeGetEvent = __library
                .get(b"cuGraphEventRecordNodeGetEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphEventRecordNodeSetEvent = __library
                .get(b"cuGraphEventRecordNodeSetEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphEventWaitNodeGetEvent = __library
                .get(b"cuGraphEventWaitNodeGetEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphEventWaitNodeSetEvent = __library
                .get(b"cuGraphEventWaitNodeSetEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
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
                )
            )]
            let cuGraphExecBatchMemOpNodeSetParams = __library
                .get(b"cuGraphExecBatchMemOpNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphExecChildGraphNodeSetParams = __library
                .get(b"cuGraphExecChildGraphNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphExecDestroy = __library
                .get(b"cuGraphExecDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphExecEventRecordNodeSetEvent = __library
                .get(b"cuGraphExecEventRecordNodeSetEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphExecEventWaitNodeSetEvent = __library
                .get(b"cuGraphExecEventWaitNodeSetEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphExecExternalSemaphoresSignalNodeSetParams = __library
                .get(b"cuGraphExecExternalSemaphoresSignalNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphExecExternalSemaphoresWaitNodeSetParams = __library
                .get(b"cuGraphExecExternalSemaphoresWaitNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphExecGetFlags = __library
                .get(b"cuGraphExecGetFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphExecHostNodeSetParams = __library
                .get(b"cuGraphExecHostNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                )
            )]
            let cuGraphExecKernelNodeSetParams = __library
                .get(b"cuGraphExecKernelNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphExecKernelNodeSetParams_v2 = __library
                .get(b"cuGraphExecKernelNodeSetParams_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphExecMemcpyNodeSetParams = __library
                .get(b"cuGraphExecMemcpyNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphExecMemsetNodeSetParams = __library
                .get(b"cuGraphExecMemsetNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphExecNodeSetParams = __library
                .get(b"cuGraphExecNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                )
            )]
            let cuGraphExecUpdate = __library
                .get(b"cuGraphExecUpdate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphExecUpdate_v2 = __library
                .get(b"cuGraphExecUpdate_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphExternalSemaphoresSignalNodeGetParams = __library
                .get(b"cuGraphExternalSemaphoresSignalNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphExternalSemaphoresSignalNodeSetParams = __library
                .get(b"cuGraphExternalSemaphoresSignalNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphExternalSemaphoresWaitNodeGetParams = __library
                .get(b"cuGraphExternalSemaphoresWaitNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphExternalSemaphoresWaitNodeSetParams = __library
                .get(b"cuGraphExternalSemaphoresWaitNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphGetEdges = __library
                .get(b"cuGraphGetEdges\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphGetEdges_v2 = __library
                .get(b"cuGraphGetEdges_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphGetNodes = __library
                .get(b"cuGraphGetNodes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphGetRootNodes = __library
                .get(b"cuGraphGetRootNodes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphHostNodeGetParams = __library
                .get(b"cuGraphHostNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphHostNodeSetParams = __library
                .get(b"cuGraphHostNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphInstantiateWithFlags = __library
                .get(b"cuGraphInstantiateWithFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphInstantiateWithParams = __library
                .get(b"cuGraphInstantiateWithParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                )
            )]
            let cuGraphInstantiate_v2 = __library
                .get(b"cuGraphInstantiate_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphKernelNodeCopyAttributes = __library
                .get(b"cuGraphKernelNodeCopyAttributes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphKernelNodeGetAttribute = __library
                .get(b"cuGraphKernelNodeGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                )
            )]
            let cuGraphKernelNodeGetParams = __library
                .get(b"cuGraphKernelNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphKernelNodeGetParams_v2 = __library
                .get(b"cuGraphKernelNodeGetParams_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphKernelNodeSetAttribute = __library
                .get(b"cuGraphKernelNodeSetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                )
            )]
            let cuGraphKernelNodeSetParams = __library
                .get(b"cuGraphKernelNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphKernelNodeSetParams_v2 = __library
                .get(b"cuGraphKernelNodeSetParams_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphLaunch = __library
                .get(b"cuGraphLaunch\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphMemAllocNodeGetParams = __library
                .get(b"cuGraphMemAllocNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphMemFreeNodeGetParams = __library
                .get(b"cuGraphMemFreeNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphMemcpyNodeGetParams = __library
                .get(b"cuGraphMemcpyNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphMemcpyNodeSetParams = __library
                .get(b"cuGraphMemcpyNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphMemsetNodeGetParams = __library
                .get(b"cuGraphMemsetNodeGetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphMemsetNodeSetParams = __library
                .get(b"cuGraphMemsetNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphNodeFindInClone = __library
                .get(b"cuGraphNodeFindInClone\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphNodeGetDependencies = __library
                .get(b"cuGraphNodeGetDependencies\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphNodeGetDependencies_v2 = __library
                .get(b"cuGraphNodeGetDependencies_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphNodeGetDependentNodes = __library
                .get(b"cuGraphNodeGetDependentNodes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphNodeGetDependentNodes_v2 = __library
                .get(b"cuGraphNodeGetDependentNodes_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
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
                )
            )]
            let cuGraphNodeGetEnabled = __library
                .get(b"cuGraphNodeGetEnabled\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphNodeGetType = __library
                .get(b"cuGraphNodeGetType\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
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
                )
            )]
            let cuGraphNodeSetEnabled = __library
                .get(b"cuGraphNodeSetEnabled\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphNodeSetParams = __library
                .get(b"cuGraphNodeSetParams\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphReleaseUserObject = __library
                .get(b"cuGraphReleaseUserObject\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphRemoveDependencies = __library
                .get(b"cuGraphRemoveDependencies\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGraphRemoveDependencies_v2 = __library
                .get(b"cuGraphRemoveDependencies_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphRetainUserObject = __library
                .get(b"cuGraphRetainUserObject\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphUpload = __library
                .get(b"cuGraphUpload\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphicsMapResources = __library
                .get(b"cuGraphicsMapResources\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphicsResourceGetMappedMipmappedArray = __library
                .get(b"cuGraphicsResourceGetMappedMipmappedArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphicsResourceGetMappedPointer_v2 = __library
                .get(b"cuGraphicsResourceGetMappedPointer_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphicsResourceSetMapFlags_v2 = __library
                .get(b"cuGraphicsResourceSetMapFlags_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphicsSubResourceGetMappedArray = __library
                .get(b"cuGraphicsSubResourceGetMappedArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphicsUnmapResources = __library
                .get(b"cuGraphicsUnmapResources\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuGraphicsUnregisterResource = __library
                .get(b"cuGraphicsUnregisterResource\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGreenCtxCreate = __library
                .get(b"cuGreenCtxCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGreenCtxDestroy = __library
                .get(b"cuGreenCtxDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGreenCtxGetDevResource = __library
                .get(b"cuGreenCtxGetDevResource\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGreenCtxRecordEvent = __library
                .get(b"cuGreenCtxRecordEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGreenCtxStreamCreate = __library
                .get(b"cuGreenCtxStreamCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuGreenCtxWaitEvent = __library
                .get(b"cuGreenCtxWaitEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuImportExternalMemory = __library
                .get(b"cuImportExternalMemory\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuImportExternalSemaphore = __library
                .get(b"cuImportExternalSemaphore\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuInit = __library
                .get(b"cuInit\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuIpcCloseMemHandle = __library
                .get(b"cuIpcCloseMemHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuIpcGetEventHandle = __library
                .get(b"cuIpcGetEventHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuIpcGetMemHandle = __library
                .get(b"cuIpcGetMemHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuIpcOpenEventHandle = __library
                .get(b"cuIpcOpenEventHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuIpcOpenMemHandle_v2 = __library
                .get(b"cuIpcOpenMemHandle_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuKernelGetAttribute = __library
                .get(b"cuKernelGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuKernelGetFunction = __library
                .get(b"cuKernelGetFunction\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuKernelGetLibrary = __library
                .get(b"cuKernelGetLibrary\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuKernelGetName = __library
                .get(b"cuKernelGetName\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuKernelGetParamInfo = __library
                .get(b"cuKernelGetParamInfo\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuKernelSetAttribute = __library
                .get(b"cuKernelSetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuKernelSetCacheConfig = __library
                .get(b"cuKernelSetCacheConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuLaunch = __library
                .get(b"cuLaunch\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuLaunchCooperativeKernel = __library
                .get(b"cuLaunchCooperativeKernel\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuLaunchCooperativeKernelMultiDevice = __library
                .get(b"cuLaunchCooperativeKernelMultiDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuLaunchGrid = __library
                .get(b"cuLaunchGrid\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuLaunchGridAsync = __library
                .get(b"cuLaunchGridAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuLaunchHostFunc = __library
                .get(b"cuLaunchHostFunc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuLaunchKernel = __library
                .get(b"cuLaunchKernel\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11080",
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuLaunchKernelEx = __library
                .get(b"cuLaunchKernelEx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuLibraryEnumerateKernels = __library
                .get(b"cuLibraryEnumerateKernels\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuLibraryGetGlobal = __library
                .get(b"cuLibraryGetGlobal\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuLibraryGetKernel = __library
                .get(b"cuLibraryGetKernel\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuLibraryGetKernelCount = __library
                .get(b"cuLibraryGetKernelCount\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuLibraryGetManaged = __library
                .get(b"cuLibraryGetManaged\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuLibraryGetModule = __library
                .get(b"cuLibraryGetModule\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuLibraryGetUnifiedFunction = __library
                .get(b"cuLibraryGetUnifiedFunction\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuLibraryLoadData = __library
                .get(b"cuLibraryLoadData\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuLibraryLoadFromFile = __library
                .get(b"cuLibraryLoadFromFile\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuLibraryUnload = __library
                .get(b"cuLibraryUnload\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuLinkAddData_v2 = __library
                .get(b"cuLinkAddData_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuLinkAddFile_v2 = __library
                .get(b"cuLinkAddFile_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuLinkComplete = __library
                .get(b"cuLinkComplete\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuLinkCreate_v2 = __library
                .get(b"cuLinkCreate_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuLinkDestroy = __library
                .get(b"cuLinkDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemAddressFree = __library
                .get(b"cuMemAddressFree\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemAddressReserve = __library
                .get(b"cuMemAddressReserve\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemAdvise = __library
                .get(b"cuMemAdvise\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuMemAdvise_v2 = __library
                .get(b"cuMemAdvise_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemAllocAsync = __library
                .get(b"cuMemAllocAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemAllocFromPoolAsync = __library
                .get(b"cuMemAllocFromPoolAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemAllocHost_v2 = __library
                .get(b"cuMemAllocHost_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemAllocManaged = __library
                .get(b"cuMemAllocManaged\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemAllocPitch_v2 = __library
                .get(b"cuMemAllocPitch_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemAlloc_v2 = __library
                .get(b"cuMemAlloc_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080"))]
            let cuMemBatchDecompressAsync = __library
                .get(b"cuMemBatchDecompressAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemCreate = __library
                .get(b"cuMemCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemExportToShareableHandle = __library
                .get(b"cuMemExportToShareableHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemFreeAsync = __library
                .get(b"cuMemFreeAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemFreeHost = __library
                .get(b"cuMemFreeHost\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemFree_v2 = __library
                .get(b"cuMemFree_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemGetAccess = __library
                .get(b"cuMemGetAccess\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemGetAddressRange_v2 = __library
                .get(b"cuMemGetAddressRange_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemGetAllocationGranularity = __library
                .get(b"cuMemGetAllocationGranularity\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemGetAllocationPropertiesFromHandle = __library
                .get(b"cuMemGetAllocationPropertiesFromHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
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
                )
            )]
            let cuMemGetHandleForAddressRange = __library
                .get(b"cuMemGetHandleForAddressRange\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemGetInfo_v2 = __library
                .get(b"cuMemGetInfo_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemHostAlloc = __library
                .get(b"cuMemHostAlloc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemHostGetDevicePointer_v2 = __library
                .get(b"cuMemHostGetDevicePointer_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemHostGetFlags = __library
                .get(b"cuMemHostGetFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemHostRegister_v2 = __library
                .get(b"cuMemHostRegister_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemHostUnregister = __library
                .get(b"cuMemHostUnregister\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemImportFromShareableHandle = __library
                .get(b"cuMemImportFromShareableHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemMap = __library
                .get(b"cuMemMap\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemMapArrayAsync = __library
                .get(b"cuMemMapArrayAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemPoolCreate = __library
                .get(b"cuMemPoolCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemPoolDestroy = __library
                .get(b"cuMemPoolDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemPoolExportPointer = __library
                .get(b"cuMemPoolExportPointer\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemPoolExportToShareableHandle = __library
                .get(b"cuMemPoolExportToShareableHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemPoolGetAccess = __library
                .get(b"cuMemPoolGetAccess\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemPoolGetAttribute = __library
                .get(b"cuMemPoolGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemPoolImportFromShareableHandle = __library
                .get(b"cuMemPoolImportFromShareableHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemPoolImportPointer = __library
                .get(b"cuMemPoolImportPointer\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemPoolSetAccess = __library
                .get(b"cuMemPoolSetAccess\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemPoolSetAttribute = __library
                .get(b"cuMemPoolSetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemPoolTrimTo = __library
                .get(b"cuMemPoolTrimTo\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemPrefetchAsync = __library
                .get(b"cuMemPrefetchAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuMemPrefetchAsync_v2 = __library
                .get(b"cuMemPrefetchAsync_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemRangeGetAttribute = __library
                .get(b"cuMemRangeGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemRangeGetAttributes = __library
                .get(b"cuMemRangeGetAttributes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemRelease = __library
                .get(b"cuMemRelease\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemRetainAllocationHandle = __library
                .get(b"cuMemRetainAllocationHandle\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemSetAccess = __library
                .get(b"cuMemSetAccess\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemUnmap = __library
                .get(b"cuMemUnmap\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpy = __library
                .get(b"cuMemcpy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpy2DAsync_v2 = __library
                .get(b"cuMemcpy2DAsync_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpy2DUnaligned_v2 = __library
                .get(b"cuMemcpy2DUnaligned_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpy2D_v2 = __library
                .get(b"cuMemcpy2D_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpy3DAsync_v2 = __library
                .get(b"cuMemcpy3DAsync_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080"))]
            let cuMemcpy3DBatchAsync = __library
                .get(b"cuMemcpy3DBatchAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpy3DPeer = __library
                .get(b"cuMemcpy3DPeer\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpy3DPeerAsync = __library
                .get(b"cuMemcpy3DPeerAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpy3D_v2 = __library
                .get(b"cuMemcpy3D_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyAsync = __library
                .get(b"cuMemcpyAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyAtoA_v2 = __library
                .get(b"cuMemcpyAtoA_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyAtoD_v2 = __library
                .get(b"cuMemcpyAtoD_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyAtoHAsync_v2 = __library
                .get(b"cuMemcpyAtoHAsync_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyAtoH_v2 = __library
                .get(b"cuMemcpyAtoH_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080"))]
            let cuMemcpyBatchAsync = __library
                .get(b"cuMemcpyBatchAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyDtoA_v2 = __library
                .get(b"cuMemcpyDtoA_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyDtoDAsync_v2 = __library
                .get(b"cuMemcpyDtoDAsync_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyDtoD_v2 = __library
                .get(b"cuMemcpyDtoD_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyDtoHAsync_v2 = __library
                .get(b"cuMemcpyDtoHAsync_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyDtoH_v2 = __library
                .get(b"cuMemcpyDtoH_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyHtoAAsync_v2 = __library
                .get(b"cuMemcpyHtoAAsync_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyHtoA_v2 = __library
                .get(b"cuMemcpyHtoA_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyHtoDAsync_v2 = __library
                .get(b"cuMemcpyHtoDAsync_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyHtoD_v2 = __library
                .get(b"cuMemcpyHtoD_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyPeer = __library
                .get(b"cuMemcpyPeer\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemcpyPeerAsync = __library
                .get(b"cuMemcpyPeerAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemsetD16Async = __library
                .get(b"cuMemsetD16Async\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemsetD16_v2 = __library
                .get(b"cuMemsetD16_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemsetD2D16Async = __library
                .get(b"cuMemsetD2D16Async\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemsetD2D16_v2 = __library
                .get(b"cuMemsetD2D16_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemsetD2D32Async = __library
                .get(b"cuMemsetD2D32Async\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemsetD2D32_v2 = __library
                .get(b"cuMemsetD2D32_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemsetD2D8Async = __library
                .get(b"cuMemsetD2D8Async\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemsetD2D8_v2 = __library
                .get(b"cuMemsetD2D8_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemsetD32Async = __library
                .get(b"cuMemsetD32Async\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemsetD32_v2 = __library
                .get(b"cuMemsetD32_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemsetD8Async = __library
                .get(b"cuMemsetD8Async\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMemsetD8_v2 = __library
                .get(b"cuMemsetD8_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMipmappedArrayCreate = __library
                .get(b"cuMipmappedArrayCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMipmappedArrayDestroy = __library
                .get(b"cuMipmappedArrayDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMipmappedArrayGetLevel = __library
                .get(b"cuMipmappedArrayGetLevel\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
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
                )
            )]
            let cuMipmappedArrayGetMemoryRequirements = __library
                .get(b"cuMipmappedArrayGetMemoryRequirements\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuMipmappedArrayGetSparseProperties = __library
                .get(b"cuMipmappedArrayGetSparseProperties\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuModuleEnumerateFunctions = __library
                .get(b"cuModuleEnumerateFunctions\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuModuleGetFunction = __library
                .get(b"cuModuleGetFunction\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuModuleGetFunctionCount = __library
                .get(b"cuModuleGetFunctionCount\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuModuleGetGlobal_v2 = __library
                .get(b"cuModuleGetGlobal_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
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
                )
            )]
            let cuModuleGetLoadingMode = __library
                .get(b"cuModuleGetLoadingMode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuModuleGetSurfRef = __library
                .get(b"cuModuleGetSurfRef\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuModuleGetTexRef = __library
                .get(b"cuModuleGetTexRef\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuModuleLoad = __library
                .get(b"cuModuleLoad\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuModuleLoadData = __library
                .get(b"cuModuleLoadData\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuModuleLoadDataEx = __library
                .get(b"cuModuleLoadDataEx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuModuleLoadFatBinary = __library
                .get(b"cuModuleLoadFatBinary\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuModuleUnload = __library
                .get(b"cuModuleUnload\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuMulticastAddDevice = __library
                .get(b"cuMulticastAddDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuMulticastBindAddr = __library
                .get(b"cuMulticastBindAddr\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuMulticastBindMem = __library
                .get(b"cuMulticastBindMem\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuMulticastCreate = __library
                .get(b"cuMulticastCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuMulticastGetGranularity = __library
                .get(b"cuMulticastGetGranularity\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuMulticastUnbind = __library
                .get(b"cuMulticastUnbind\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuOccupancyAvailableDynamicSMemPerBlock = __library
                .get(b"cuOccupancyAvailableDynamicSMemPerBlock\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuOccupancyMaxActiveBlocksPerMultiprocessor = __library
                .get(b"cuOccupancyMaxActiveBlocksPerMultiprocessor\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags = __library
                .get(b"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11080",
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuOccupancyMaxActiveClusters = __library
                .get(b"cuOccupancyMaxActiveClusters\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuOccupancyMaxPotentialBlockSize = __library
                .get(b"cuOccupancyMaxPotentialBlockSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuOccupancyMaxPotentialBlockSizeWithFlags = __library
                .get(b"cuOccupancyMaxPotentialBlockSizeWithFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11080",
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuOccupancyMaxPotentialClusterSize = __library
                .get(b"cuOccupancyMaxPotentialClusterSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuParamSetSize = __library
                .get(b"cuParamSetSize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuParamSetTexRef = __library
                .get(b"cuParamSetTexRef\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuParamSetf = __library
                .get(b"cuParamSetf\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuParamSeti = __library
                .get(b"cuParamSeti\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuParamSetv = __library
                .get(b"cuParamSetv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuPointerGetAttribute = __library
                .get(b"cuPointerGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuPointerGetAttributes = __library
                .get(b"cuPointerGetAttributes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuPointerSetAttribute = __library
                .get(b"cuPointerSetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuProfilerInitialize = __library
                .get(b"cuProfilerInitialize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuProfilerStart = __library
                .get(b"cuProfilerStart\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuProfilerStop = __library
                .get(b"cuProfilerStop\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuSignalExternalSemaphoresAsync = __library
                .get(b"cuSignalExternalSemaphoresAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamAddCallback = __library
                .get(b"cuStreamAddCallback\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamAttachMemAsync = __library
                .get(b"cuStreamAttachMemAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                )
            )]
            let cuStreamBatchMemOp = __library
                .get(b"cuStreamBatchMemOp\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
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
                )
            )]
            let cuStreamBatchMemOp_v2 = __library
                .get(b"cuStreamBatchMemOp_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuStreamBeginCaptureToGraph = __library
                .get(b"cuStreamBeginCaptureToGraph\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamBeginCapture_v2 = __library
                .get(b"cuStreamBeginCapture_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamCopyAttributes = __library
                .get(b"cuStreamCopyAttributes\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamCreate = __library
                .get(b"cuStreamCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamCreateWithPriority = __library
                .get(b"cuStreamCreateWithPriority\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamDestroy_v2 = __library
                .get(b"cuStreamDestroy_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamEndCapture = __library
                .get(b"cuStreamEndCapture\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamGetAttribute = __library
                .get(b"cuStreamGetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                )
            )]
            let cuStreamGetCaptureInfo = __library
                .get(b"cuStreamGetCaptureInfo\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamGetCaptureInfo_v2 = __library
                .get(b"cuStreamGetCaptureInfo_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuStreamGetCaptureInfo_v3 = __library
                .get(b"cuStreamGetCaptureInfo_v3\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamGetCtx = __library
                .get(b"cuStreamGetCtx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuStreamGetCtx_v2 = __library
                .get(b"cuStreamGetCtx_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080"))]
            let cuStreamGetDevice = __library
                .get(b"cuStreamGetDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamGetFlags = __library
                .get(b"cuStreamGetFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuStreamGetGreenCtx = __library
                .get(b"cuStreamGetGreenCtx\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuStreamGetId = __library
                .get(b"cuStreamGetId\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamGetPriority = __library
                .get(b"cuStreamGetPriority\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamIsCapturing = __library
                .get(b"cuStreamIsCapturing\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamQuery = __library
                .get(b"cuStreamQuery\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamSetAttribute = __library
                .get(b"cuStreamSetAttribute\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamSynchronize = __library
                .get(b"cuStreamSynchronize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamUpdateCaptureDependencies = __library
                .get(b"cuStreamUpdateCaptureDependencies\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuStreamUpdateCaptureDependencies_v2 = __library
                .get(b"cuStreamUpdateCaptureDependencies_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuStreamWaitEvent = __library
                .get(b"cuStreamWaitEvent\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                )
            )]
            let cuStreamWaitValue32 = __library
                .get(b"cuStreamWaitValue32\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
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
                )
            )]
            let cuStreamWaitValue32_v2 = __library
                .get(b"cuStreamWaitValue32_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                )
            )]
            let cuStreamWaitValue64 = __library
                .get(b"cuStreamWaitValue64\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
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
                )
            )]
            let cuStreamWaitValue64_v2 = __library
                .get(b"cuStreamWaitValue64_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                )
            )]
            let cuStreamWriteValue32 = __library
                .get(b"cuStreamWriteValue32\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
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
                )
            )]
            let cuStreamWriteValue32_v2 = __library
                .get(b"cuStreamWriteValue32_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-11040",
                    feature = "cuda-11050",
                    feature = "cuda-11060",
                    feature = "cuda-11070",
                    feature = "cuda-11080"
                )
            )]
            let cuStreamWriteValue64 = __library
                .get(b"cuStreamWriteValue64\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
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
                )
            )]
            let cuStreamWriteValue64_v2 = __library
                .get(b"cuStreamWriteValue64_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuSurfObjectCreate = __library
                .get(b"cuSurfObjectCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuSurfObjectDestroy = __library
                .get(b"cuSurfObjectDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuSurfObjectGetResourceDesc = __library
                .get(b"cuSurfObjectGetResourceDesc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuSurfRefGetArray = __library
                .get(b"cuSurfRefGetArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuSurfRefSetArray = __library
                .get(b"cuSurfRefSetArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuTensorMapEncodeIm2col = __library
                .get(b"cuTensorMapEncodeIm2col\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(any(feature = "cuda-12080"))]
            let cuTensorMapEncodeIm2colWide = __library
                .get(b"cuTensorMapEncodeIm2colWide\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuTensorMapEncodeTiled = __library
                .get(b"cuTensorMapEncodeTiled\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            #[cfg(
                any(
                    feature = "cuda-12000",
                    feature = "cuda-12010",
                    feature = "cuda-12020",
                    feature = "cuda-12030",
                    feature = "cuda-12040",
                    feature = "cuda-12050",
                    feature = "cuda-12060",
                    feature = "cuda-12080"
                )
            )]
            let cuTensorMapReplaceAddress = __library
                .get(b"cuTensorMapReplaceAddress\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexObjectCreate = __library
                .get(b"cuTexObjectCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexObjectDestroy = __library
                .get(b"cuTexObjectDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexObjectGetResourceDesc = __library
                .get(b"cuTexObjectGetResourceDesc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexObjectGetResourceViewDesc = __library
                .get(b"cuTexObjectGetResourceViewDesc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexObjectGetTextureDesc = __library
                .get(b"cuTexObjectGetTextureDesc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefCreate = __library
                .get(b"cuTexRefCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefDestroy = __library
                .get(b"cuTexRefDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefGetAddressMode = __library
                .get(b"cuTexRefGetAddressMode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefGetAddress_v2 = __library
                .get(b"cuTexRefGetAddress_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefGetArray = __library
                .get(b"cuTexRefGetArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefGetBorderColor = __library
                .get(b"cuTexRefGetBorderColor\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefGetFilterMode = __library
                .get(b"cuTexRefGetFilterMode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefGetFlags = __library
                .get(b"cuTexRefGetFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefGetFormat = __library
                .get(b"cuTexRefGetFormat\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefGetMaxAnisotropy = __library
                .get(b"cuTexRefGetMaxAnisotropy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefGetMipmapFilterMode = __library
                .get(b"cuTexRefGetMipmapFilterMode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefGetMipmapLevelBias = __library
                .get(b"cuTexRefGetMipmapLevelBias\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefGetMipmapLevelClamp = __library
                .get(b"cuTexRefGetMipmapLevelClamp\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefGetMipmappedArray = __library
                .get(b"cuTexRefGetMipmappedArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefSetAddress2D_v3 = __library
                .get(b"cuTexRefSetAddress2D_v3\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefSetAddressMode = __library
                .get(b"cuTexRefSetAddressMode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefSetAddress_v2 = __library
                .get(b"cuTexRefSetAddress_v2\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefSetArray = __library
                .get(b"cuTexRefSetArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefSetBorderColor = __library
                .get(b"cuTexRefSetBorderColor\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefSetFilterMode = __library
                .get(b"cuTexRefSetFilterMode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefSetFlags = __library
                .get(b"cuTexRefSetFlags\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefSetFormat = __library
                .get(b"cuTexRefSetFormat\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefSetMaxAnisotropy = __library
                .get(b"cuTexRefSetMaxAnisotropy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefSetMipmapFilterMode = __library
                .get(b"cuTexRefSetMipmapFilterMode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefSetMipmapLevelBias = __library
                .get(b"cuTexRefSetMipmapLevelBias\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefSetMipmapLevelClamp = __library
                .get(b"cuTexRefSetMipmapLevelClamp\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuTexRefSetMipmappedArray = __library
                .get(b"cuTexRefSetMipmappedArray\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuThreadExchangeStreamCaptureMode = __library
                .get(b"cuThreadExchangeStreamCaptureMode\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuUserObjectCreate = __library
                .get(b"cuUserObjectCreate\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuUserObjectRelease = __library
                .get(b"cuUserObjectRelease\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuUserObjectRetain = __library
                .get(b"cuUserObjectRetain\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let cuWaitExternalSemaphoresAsync = __library
                .get(b"cuWaitExternalSemaphoresAsync\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            Ok(Self {
                __library,
                cuArray3DCreate_v2,
                cuArray3DGetDescriptor_v2,
                cuArrayCreate_v2,
                cuArrayDestroy,
                cuArrayGetDescriptor_v2,
                #[cfg(
                    any(
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
                    )
                )]
                cuArrayGetMemoryRequirements,
                cuArrayGetPlane,
                cuArrayGetSparseProperties,
                #[cfg(any(feature = "cuda-12080"))]
                cuCheckpointProcessCheckpoint,
                #[cfg(any(feature = "cuda-12080"))]
                cuCheckpointProcessGetRestoreThreadId,
                #[cfg(any(feature = "cuda-12080"))]
                cuCheckpointProcessGetState,
                #[cfg(any(feature = "cuda-12080"))]
                cuCheckpointProcessLock,
                #[cfg(any(feature = "cuda-12080"))]
                cuCheckpointProcessRestore,
                #[cfg(any(feature = "cuda-12080"))]
                cuCheckpointProcessUnlock,
                #[cfg(
                    any(
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuCoredumpGetAttribute,
                #[cfg(
                    any(
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuCoredumpGetAttributeGlobal,
                #[cfg(
                    any(
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuCoredumpSetAttribute,
                #[cfg(
                    any(
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuCoredumpSetAttributeGlobal,
                cuCtxAttach,
                cuCtxCreate_v2,
                cuCtxCreate_v3,
                #[cfg(
                    any(
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuCtxCreate_v4,
                cuCtxDestroy_v2,
                cuCtxDetach,
                cuCtxDisablePeerAccess,
                cuCtxEnablePeerAccess,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuCtxFromGreenCtx,
                cuCtxGetApiVersion,
                cuCtxGetCacheConfig,
                cuCtxGetCurrent,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuCtxGetDevResource,
                cuCtxGetDevice,
                cuCtxGetExecAffinity,
                cuCtxGetFlags,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuCtxGetId,
                cuCtxGetLimit,
                cuCtxGetSharedMemConfig,
                cuCtxGetStreamPriorityRange,
                cuCtxPopCurrent_v2,
                cuCtxPushCurrent_v2,
                #[cfg(
                    any(
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuCtxRecordEvent,
                cuCtxResetPersistingL2Cache,
                cuCtxSetCacheConfig,
                cuCtxSetCurrent,
                #[cfg(
                    any(
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuCtxSetFlags,
                cuCtxSetLimit,
                cuCtxSetSharedMemConfig,
                cuCtxSynchronize,
                #[cfg(
                    any(
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuCtxWaitEvent,
                cuDestroyExternalMemory,
                cuDestroyExternalSemaphore,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuDevResourceGenerateDesc,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuDevSmResourceSplitByCount,
                cuDeviceCanAccessPeer,
                cuDeviceComputeCapability,
                cuDeviceGet,
                cuDeviceGetAttribute,
                cuDeviceGetByPCIBusId,
                cuDeviceGetCount,
                cuDeviceGetDefaultMemPool,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuDeviceGetDevResource,
                cuDeviceGetExecAffinitySupport,
                cuDeviceGetGraphMemAttribute,
                cuDeviceGetLuid,
                cuDeviceGetMemPool,
                cuDeviceGetName,
                cuDeviceGetNvSciSyncAttributes,
                cuDeviceGetP2PAttribute,
                cuDeviceGetPCIBusId,
                cuDeviceGetProperties,
                cuDeviceGetTexture1DLinearMaxWidth,
                cuDeviceGetUuid,
                cuDeviceGetUuid_v2,
                cuDeviceGraphMemTrim,
                cuDevicePrimaryCtxGetState,
                cuDevicePrimaryCtxRelease_v2,
                cuDevicePrimaryCtxReset_v2,
                cuDevicePrimaryCtxRetain,
                cuDevicePrimaryCtxSetFlags_v2,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuDeviceRegisterAsyncNotification,
                cuDeviceSetGraphMemAttribute,
                cuDeviceSetMemPool,
                cuDeviceTotalMem_v2,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuDeviceUnregisterAsyncNotification,
                cuDriverGetVersion,
                cuEventCreate,
                cuEventDestroy_v2,
                cuEventElapsedTime,
                #[cfg(any(feature = "cuda-12080"))]
                cuEventElapsedTime_v2,
                cuEventQuery,
                cuEventRecord,
                cuEventRecordWithFlags,
                cuEventSynchronize,
                cuExternalMemoryGetMappedBuffer,
                cuExternalMemoryGetMappedMipmappedArray,
                cuFlushGPUDirectRDMAWrites,
                cuFuncGetAttribute,
                cuFuncGetModule,
                #[cfg(
                    any(
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuFuncGetName,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuFuncGetParamInfo,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuFuncIsLoaded,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuFuncLoad,
                cuFuncSetAttribute,
                cuFuncSetBlockShape,
                cuFuncSetCacheConfig,
                cuFuncSetSharedMemConfig,
                cuFuncSetSharedSize,
                cuGetErrorName,
                cuGetErrorString,
                cuGetExportTable,
                #[cfg(
                    any(
                        feature = "cuda-11040",
                        feature = "cuda-11050",
                        feature = "cuda-11060",
                        feature = "cuda-11070",
                        feature = "cuda-11080"
                    )
                )]
                cuGetProcAddress,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGetProcAddress_v2,
                #[cfg(
                    any(
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
                    )
                )]
                cuGraphAddBatchMemOpNode,
                cuGraphAddChildGraphNode,
                cuGraphAddDependencies,
                #[cfg(
                    any(
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphAddDependencies_v2,
                cuGraphAddEmptyNode,
                cuGraphAddEventRecordNode,
                cuGraphAddEventWaitNode,
                cuGraphAddExternalSemaphoresSignalNode,
                cuGraphAddExternalSemaphoresWaitNode,
                cuGraphAddHostNode,
                #[cfg(
                    any(
                        feature = "cuda-11040",
                        feature = "cuda-11050",
                        feature = "cuda-11060",
                        feature = "cuda-11070",
                        feature = "cuda-11080"
                    )
                )]
                cuGraphAddKernelNode,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphAddKernelNode_v2,
                cuGraphAddMemAllocNode,
                cuGraphAddMemFreeNode,
                cuGraphAddMemcpyNode,
                cuGraphAddMemsetNode,
                #[cfg(
                    any(
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphAddNode,
                #[cfg(
                    any(
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphAddNode_v2,
                #[cfg(
                    any(
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
                    )
                )]
                cuGraphBatchMemOpNodeGetParams,
                #[cfg(
                    any(
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
                    )
                )]
                cuGraphBatchMemOpNodeSetParams,
                cuGraphChildGraphNodeGetGraph,
                cuGraphClone,
                #[cfg(
                    any(
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphConditionalHandleCreate,
                cuGraphCreate,
                cuGraphDebugDotPrint,
                cuGraphDestroy,
                cuGraphDestroyNode,
                cuGraphEventRecordNodeGetEvent,
                cuGraphEventRecordNodeSetEvent,
                cuGraphEventWaitNodeGetEvent,
                cuGraphEventWaitNodeSetEvent,
                #[cfg(
                    any(
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
                    )
                )]
                cuGraphExecBatchMemOpNodeSetParams,
                cuGraphExecChildGraphNodeSetParams,
                cuGraphExecDestroy,
                cuGraphExecEventRecordNodeSetEvent,
                cuGraphExecEventWaitNodeSetEvent,
                cuGraphExecExternalSemaphoresSignalNodeSetParams,
                cuGraphExecExternalSemaphoresWaitNodeSetParams,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphExecGetFlags,
                cuGraphExecHostNodeSetParams,
                #[cfg(
                    any(
                        feature = "cuda-11040",
                        feature = "cuda-11050",
                        feature = "cuda-11060",
                        feature = "cuda-11070",
                        feature = "cuda-11080"
                    )
                )]
                cuGraphExecKernelNodeSetParams,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphExecKernelNodeSetParams_v2,
                cuGraphExecMemcpyNodeSetParams,
                cuGraphExecMemsetNodeSetParams,
                #[cfg(
                    any(
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphExecNodeSetParams,
                #[cfg(
                    any(
                        feature = "cuda-11040",
                        feature = "cuda-11050",
                        feature = "cuda-11060",
                        feature = "cuda-11070",
                        feature = "cuda-11080"
                    )
                )]
                cuGraphExecUpdate,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphExecUpdate_v2,
                cuGraphExternalSemaphoresSignalNodeGetParams,
                cuGraphExternalSemaphoresSignalNodeSetParams,
                cuGraphExternalSemaphoresWaitNodeGetParams,
                cuGraphExternalSemaphoresWaitNodeSetParams,
                cuGraphGetEdges,
                #[cfg(
                    any(
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphGetEdges_v2,
                cuGraphGetNodes,
                cuGraphGetRootNodes,
                cuGraphHostNodeGetParams,
                cuGraphHostNodeSetParams,
                cuGraphInstantiateWithFlags,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphInstantiateWithParams,
                #[cfg(
                    any(
                        feature = "cuda-11040",
                        feature = "cuda-11050",
                        feature = "cuda-11060",
                        feature = "cuda-11070",
                        feature = "cuda-11080"
                    )
                )]
                cuGraphInstantiate_v2,
                cuGraphKernelNodeCopyAttributes,
                cuGraphKernelNodeGetAttribute,
                #[cfg(
                    any(
                        feature = "cuda-11040",
                        feature = "cuda-11050",
                        feature = "cuda-11060",
                        feature = "cuda-11070",
                        feature = "cuda-11080"
                    )
                )]
                cuGraphKernelNodeGetParams,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphKernelNodeGetParams_v2,
                cuGraphKernelNodeSetAttribute,
                #[cfg(
                    any(
                        feature = "cuda-11040",
                        feature = "cuda-11050",
                        feature = "cuda-11060",
                        feature = "cuda-11070",
                        feature = "cuda-11080"
                    )
                )]
                cuGraphKernelNodeSetParams,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphKernelNodeSetParams_v2,
                cuGraphLaunch,
                cuGraphMemAllocNodeGetParams,
                cuGraphMemFreeNodeGetParams,
                cuGraphMemcpyNodeGetParams,
                cuGraphMemcpyNodeSetParams,
                cuGraphMemsetNodeGetParams,
                cuGraphMemsetNodeSetParams,
                cuGraphNodeFindInClone,
                cuGraphNodeGetDependencies,
                #[cfg(
                    any(
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphNodeGetDependencies_v2,
                cuGraphNodeGetDependentNodes,
                #[cfg(
                    any(
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphNodeGetDependentNodes_v2,
                #[cfg(
                    any(
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
                    )
                )]
                cuGraphNodeGetEnabled,
                cuGraphNodeGetType,
                #[cfg(
                    any(
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
                    )
                )]
                cuGraphNodeSetEnabled,
                #[cfg(
                    any(
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphNodeSetParams,
                cuGraphReleaseUserObject,
                cuGraphRemoveDependencies,
                #[cfg(
                    any(
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGraphRemoveDependencies_v2,
                cuGraphRetainUserObject,
                cuGraphUpload,
                cuGraphicsMapResources,
                cuGraphicsResourceGetMappedMipmappedArray,
                cuGraphicsResourceGetMappedPointer_v2,
                cuGraphicsResourceSetMapFlags_v2,
                cuGraphicsSubResourceGetMappedArray,
                cuGraphicsUnmapResources,
                cuGraphicsUnregisterResource,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGreenCtxCreate,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGreenCtxDestroy,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGreenCtxGetDevResource,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGreenCtxRecordEvent,
                #[cfg(
                    any(
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGreenCtxStreamCreate,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuGreenCtxWaitEvent,
                cuImportExternalMemory,
                cuImportExternalSemaphore,
                cuInit,
                cuIpcCloseMemHandle,
                cuIpcGetEventHandle,
                cuIpcGetMemHandle,
                cuIpcOpenEventHandle,
                cuIpcOpenMemHandle_v2,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuKernelGetAttribute,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuKernelGetFunction,
                #[cfg(
                    any(
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuKernelGetLibrary,
                #[cfg(
                    any(
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuKernelGetName,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuKernelGetParamInfo,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuKernelSetAttribute,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuKernelSetCacheConfig,
                cuLaunch,
                cuLaunchCooperativeKernel,
                cuLaunchCooperativeKernelMultiDevice,
                cuLaunchGrid,
                cuLaunchGridAsync,
                cuLaunchHostFunc,
                cuLaunchKernel,
                #[cfg(
                    any(
                        feature = "cuda-11080",
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuLaunchKernelEx,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuLibraryEnumerateKernels,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuLibraryGetGlobal,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuLibraryGetKernel,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuLibraryGetKernelCount,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuLibraryGetManaged,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuLibraryGetModule,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuLibraryGetUnifiedFunction,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuLibraryLoadData,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuLibraryLoadFromFile,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuLibraryUnload,
                cuLinkAddData_v2,
                cuLinkAddFile_v2,
                cuLinkComplete,
                cuLinkCreate_v2,
                cuLinkDestroy,
                cuMemAddressFree,
                cuMemAddressReserve,
                cuMemAdvise,
                #[cfg(
                    any(
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuMemAdvise_v2,
                cuMemAllocAsync,
                cuMemAllocFromPoolAsync,
                cuMemAllocHost_v2,
                cuMemAllocManaged,
                cuMemAllocPitch_v2,
                cuMemAlloc_v2,
                #[cfg(any(feature = "cuda-12080"))]
                cuMemBatchDecompressAsync,
                cuMemCreate,
                cuMemExportToShareableHandle,
                cuMemFreeAsync,
                cuMemFreeHost,
                cuMemFree_v2,
                cuMemGetAccess,
                cuMemGetAddressRange_v2,
                cuMemGetAllocationGranularity,
                cuMemGetAllocationPropertiesFromHandle,
                #[cfg(
                    any(
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
                    )
                )]
                cuMemGetHandleForAddressRange,
                cuMemGetInfo_v2,
                cuMemHostAlloc,
                cuMemHostGetDevicePointer_v2,
                cuMemHostGetFlags,
                cuMemHostRegister_v2,
                cuMemHostUnregister,
                cuMemImportFromShareableHandle,
                cuMemMap,
                cuMemMapArrayAsync,
                cuMemPoolCreate,
                cuMemPoolDestroy,
                cuMemPoolExportPointer,
                cuMemPoolExportToShareableHandle,
                cuMemPoolGetAccess,
                cuMemPoolGetAttribute,
                cuMemPoolImportFromShareableHandle,
                cuMemPoolImportPointer,
                cuMemPoolSetAccess,
                cuMemPoolSetAttribute,
                cuMemPoolTrimTo,
                cuMemPrefetchAsync,
                #[cfg(
                    any(
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuMemPrefetchAsync_v2,
                cuMemRangeGetAttribute,
                cuMemRangeGetAttributes,
                cuMemRelease,
                cuMemRetainAllocationHandle,
                cuMemSetAccess,
                cuMemUnmap,
                cuMemcpy,
                cuMemcpy2DAsync_v2,
                cuMemcpy2DUnaligned_v2,
                cuMemcpy2D_v2,
                cuMemcpy3DAsync_v2,
                #[cfg(any(feature = "cuda-12080"))]
                cuMemcpy3DBatchAsync,
                cuMemcpy3DPeer,
                cuMemcpy3DPeerAsync,
                cuMemcpy3D_v2,
                cuMemcpyAsync,
                cuMemcpyAtoA_v2,
                cuMemcpyAtoD_v2,
                cuMemcpyAtoHAsync_v2,
                cuMemcpyAtoH_v2,
                #[cfg(any(feature = "cuda-12080"))]
                cuMemcpyBatchAsync,
                cuMemcpyDtoA_v2,
                cuMemcpyDtoDAsync_v2,
                cuMemcpyDtoD_v2,
                cuMemcpyDtoHAsync_v2,
                cuMemcpyDtoH_v2,
                cuMemcpyHtoAAsync_v2,
                cuMemcpyHtoA_v2,
                cuMemcpyHtoDAsync_v2,
                cuMemcpyHtoD_v2,
                cuMemcpyPeer,
                cuMemcpyPeerAsync,
                cuMemsetD16Async,
                cuMemsetD16_v2,
                cuMemsetD2D16Async,
                cuMemsetD2D16_v2,
                cuMemsetD2D32Async,
                cuMemsetD2D32_v2,
                cuMemsetD2D8Async,
                cuMemsetD2D8_v2,
                cuMemsetD32Async,
                cuMemsetD32_v2,
                cuMemsetD8Async,
                cuMemsetD8_v2,
                cuMipmappedArrayCreate,
                cuMipmappedArrayDestroy,
                cuMipmappedArrayGetLevel,
                #[cfg(
                    any(
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
                    )
                )]
                cuMipmappedArrayGetMemoryRequirements,
                cuMipmappedArrayGetSparseProperties,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuModuleEnumerateFunctions,
                cuModuleGetFunction,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuModuleGetFunctionCount,
                cuModuleGetGlobal_v2,
                #[cfg(
                    any(
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
                    )
                )]
                cuModuleGetLoadingMode,
                cuModuleGetSurfRef,
                cuModuleGetTexRef,
                cuModuleLoad,
                cuModuleLoadData,
                cuModuleLoadDataEx,
                cuModuleLoadFatBinary,
                cuModuleUnload,
                #[cfg(
                    any(
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuMulticastAddDevice,
                #[cfg(
                    any(
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuMulticastBindAddr,
                #[cfg(
                    any(
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuMulticastBindMem,
                #[cfg(
                    any(
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuMulticastCreate,
                #[cfg(
                    any(
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuMulticastGetGranularity,
                #[cfg(
                    any(
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuMulticastUnbind,
                cuOccupancyAvailableDynamicSMemPerBlock,
                cuOccupancyMaxActiveBlocksPerMultiprocessor,
                cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
                #[cfg(
                    any(
                        feature = "cuda-11080",
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuOccupancyMaxActiveClusters,
                cuOccupancyMaxPotentialBlockSize,
                cuOccupancyMaxPotentialBlockSizeWithFlags,
                #[cfg(
                    any(
                        feature = "cuda-11080",
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuOccupancyMaxPotentialClusterSize,
                cuParamSetSize,
                cuParamSetTexRef,
                cuParamSetf,
                cuParamSeti,
                cuParamSetv,
                cuPointerGetAttribute,
                cuPointerGetAttributes,
                cuPointerSetAttribute,
                cuProfilerInitialize,
                cuProfilerStart,
                cuProfilerStop,
                cuSignalExternalSemaphoresAsync,
                cuStreamAddCallback,
                cuStreamAttachMemAsync,
                #[cfg(
                    any(
                        feature = "cuda-11040",
                        feature = "cuda-11050",
                        feature = "cuda-11060",
                        feature = "cuda-11070",
                        feature = "cuda-11080"
                    )
                )]
                cuStreamBatchMemOp,
                #[cfg(
                    any(
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
                    )
                )]
                cuStreamBatchMemOp_v2,
                #[cfg(
                    any(
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuStreamBeginCaptureToGraph,
                cuStreamBeginCapture_v2,
                cuStreamCopyAttributes,
                cuStreamCreate,
                cuStreamCreateWithPriority,
                cuStreamDestroy_v2,
                cuStreamEndCapture,
                cuStreamGetAttribute,
                #[cfg(
                    any(
                        feature = "cuda-11040",
                        feature = "cuda-11050",
                        feature = "cuda-11060",
                        feature = "cuda-11070",
                        feature = "cuda-11080"
                    )
                )]
                cuStreamGetCaptureInfo,
                cuStreamGetCaptureInfo_v2,
                #[cfg(
                    any(
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuStreamGetCaptureInfo_v3,
                cuStreamGetCtx,
                #[cfg(
                    any(
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuStreamGetCtx_v2,
                #[cfg(any(feature = "cuda-12080"))]
                cuStreamGetDevice,
                cuStreamGetFlags,
                #[cfg(
                    any(
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuStreamGetGreenCtx,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuStreamGetId,
                cuStreamGetPriority,
                cuStreamIsCapturing,
                cuStreamQuery,
                cuStreamSetAttribute,
                cuStreamSynchronize,
                cuStreamUpdateCaptureDependencies,
                #[cfg(
                    any(
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuStreamUpdateCaptureDependencies_v2,
                cuStreamWaitEvent,
                #[cfg(
                    any(
                        feature = "cuda-11040",
                        feature = "cuda-11050",
                        feature = "cuda-11060",
                        feature = "cuda-11070",
                        feature = "cuda-11080"
                    )
                )]
                cuStreamWaitValue32,
                #[cfg(
                    any(
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
                    )
                )]
                cuStreamWaitValue32_v2,
                #[cfg(
                    any(
                        feature = "cuda-11040",
                        feature = "cuda-11050",
                        feature = "cuda-11060",
                        feature = "cuda-11070",
                        feature = "cuda-11080"
                    )
                )]
                cuStreamWaitValue64,
                #[cfg(
                    any(
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
                    )
                )]
                cuStreamWaitValue64_v2,
                #[cfg(
                    any(
                        feature = "cuda-11040",
                        feature = "cuda-11050",
                        feature = "cuda-11060",
                        feature = "cuda-11070",
                        feature = "cuda-11080"
                    )
                )]
                cuStreamWriteValue32,
                #[cfg(
                    any(
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
                    )
                )]
                cuStreamWriteValue32_v2,
                #[cfg(
                    any(
                        feature = "cuda-11040",
                        feature = "cuda-11050",
                        feature = "cuda-11060",
                        feature = "cuda-11070",
                        feature = "cuda-11080"
                    )
                )]
                cuStreamWriteValue64,
                #[cfg(
                    any(
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
                    )
                )]
                cuStreamWriteValue64_v2,
                cuSurfObjectCreate,
                cuSurfObjectDestroy,
                cuSurfObjectGetResourceDesc,
                cuSurfRefGetArray,
                cuSurfRefSetArray,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuTensorMapEncodeIm2col,
                #[cfg(any(feature = "cuda-12080"))]
                cuTensorMapEncodeIm2colWide,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuTensorMapEncodeTiled,
                #[cfg(
                    any(
                        feature = "cuda-12000",
                        feature = "cuda-12010",
                        feature = "cuda-12020",
                        feature = "cuda-12030",
                        feature = "cuda-12040",
                        feature = "cuda-12050",
                        feature = "cuda-12060",
                        feature = "cuda-12080"
                    )
                )]
                cuTensorMapReplaceAddress,
                cuTexObjectCreate,
                cuTexObjectDestroy,
                cuTexObjectGetResourceDesc,
                cuTexObjectGetResourceViewDesc,
                cuTexObjectGetTextureDesc,
                cuTexRefCreate,
                cuTexRefDestroy,
                cuTexRefGetAddressMode,
                cuTexRefGetAddress_v2,
                cuTexRefGetArray,
                cuTexRefGetBorderColor,
                cuTexRefGetFilterMode,
                cuTexRefGetFlags,
                cuTexRefGetFormat,
                cuTexRefGetMaxAnisotropy,
                cuTexRefGetMipmapFilterMode,
                cuTexRefGetMipmapLevelBias,
                cuTexRefGetMipmapLevelClamp,
                cuTexRefGetMipmappedArray,
                cuTexRefSetAddress2D_v3,
                cuTexRefSetAddressMode,
                cuTexRefSetAddress_v2,
                cuTexRefSetArray,
                cuTexRefSetBorderColor,
                cuTexRefSetFilterMode,
                cuTexRefSetFlags,
                cuTexRefSetFormat,
                cuTexRefSetMaxAnisotropy,
                cuTexRefSetMipmapFilterMode,
                cuTexRefSetMipmapLevelBias,
                cuTexRefSetMipmapLevelClamp,
                cuTexRefSetMipmappedArray,
                cuThreadExchangeStreamCaptureMode,
                cuUserObjectCreate,
                cuUserObjectRelease,
                cuUserObjectRetain,
                cuWaitExternalSemaphoresAsync,
            })
        }
    }
    pub unsafe fn culib() -> &'static Lib {
        static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
        LIB.get_or_init(|| {
            let lib_names = std::vec!["cuda", "nvcuda"];
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
