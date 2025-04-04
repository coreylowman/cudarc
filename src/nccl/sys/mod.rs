#![cfg_attr(feature = "no-std", no_std)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#[cfg(feature = "no-std")]
extern crate alloc;
#[cfg(feature = "no-std")]
extern crate no_std_compat as std;
pub type cudaStream_t = *mut CUstream_st;
pub type ncclComm_t = *mut ncclComm;
pub type ncclConfig_t = ncclConfig_v21700;
pub type ncclSimInfo_t = ncclSimInfo_v22200;
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum ncclDataType_t {
    ncclInt8 = 0,
    ncclUint8 = 1,
    ncclInt32 = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6,
    ncclFloat32 = 7,
    ncclFloat64 = 8,
    ncclBfloat16 = 9,
    ncclFloat8e4m3 = 10,
    ncclFloat8e5m2 = 11,
    ncclNumTypes = 12,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum ncclRedOp_dummy_t {
    ncclNumOps_dummy = 5,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum ncclRedOp_t {
    ncclSum = 0,
    ncclProd = 1,
    ncclMax = 2,
    ncclMin = 3,
    ncclAvg = 4,
    ncclNumOps = 5,
    ncclMaxRedOp = 2147483647,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum ncclResult_t {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7,
    ncclNumResults = 8,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum ncclScalarResidence_t {
    ncclScalarDevice = 0,
    ncclScalarHostImmediate = 1,
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ncclComm {
    _unused: [u8; 0],
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct ncclConfig_v21700 {
    pub size: usize,
    pub magic: ::core::ffi::c_uint,
    pub version: ::core::ffi::c_uint,
    pub blocking: ::core::ffi::c_int,
    pub cgaClusterSize: ::core::ffi::c_int,
    pub minCTAs: ::core::ffi::c_int,
    pub maxCTAs: ::core::ffi::c_int,
    pub netName: *const ::core::ffi::c_char,
    pub splitShare: ::core::ffi::c_int,
    pub trafficClass: ::core::ffi::c_int,
}
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialOrd, PartialEq)]
pub struct ncclSimInfo_v22200 {
    pub size: usize,
    pub magic: ::core::ffi::c_uint,
    pub version: ::core::ffi::c_uint,
    pub estimatedTime: f32,
}
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct ncclUniqueId {
    pub internal: [::core::ffi::c_char; 128usize],
}
impl ncclDataType_t {
    pub const ncclChar: ncclDataType_t = ncclDataType_t::ncclInt8;
}
impl ncclDataType_t {
    pub const ncclDouble: ncclDataType_t = ncclDataType_t::ncclFloat64;
}
impl ncclDataType_t {
    pub const ncclFloat: ncclDataType_t = ncclDataType_t::ncclFloat32;
}
impl ncclDataType_t {
    pub const ncclHalf: ncclDataType_t = ncclDataType_t::ncclFloat16;
}
impl ncclDataType_t {
    pub const ncclInt: ncclDataType_t = ncclDataType_t::ncclInt32;
}
impl Default for ncclConfig_v21700 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
impl Default for ncclUniqueId {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
#[cfg(not(feature = "dynamic-loading"))]
extern "C" {
    pub fn ncclAllGather(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        sendcount: usize,
        datatype: ncclDataType_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclAllReduce(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclBcast(
        buff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclBroadcast(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclCommAbort(comm: ncclComm_t) -> ncclResult_t;
    pub fn ncclCommCount(comm: ncclComm_t, count: *mut ::core::ffi::c_int) -> ncclResult_t;
    pub fn ncclCommCuDevice(comm: ncclComm_t, device: *mut ::core::ffi::c_int) -> ncclResult_t;
    pub fn ncclCommDeregister(comm: ncclComm_t, handle: *mut ::core::ffi::c_void) -> ncclResult_t;
    pub fn ncclCommDestroy(comm: ncclComm_t) -> ncclResult_t;
    pub fn ncclCommFinalize(comm: ncclComm_t) -> ncclResult_t;
    pub fn ncclCommGetAsyncError(comm: ncclComm_t, asyncError: *mut ncclResult_t) -> ncclResult_t;
    pub fn ncclCommInitAll(
        comm: *mut ncclComm_t,
        ndev: ::core::ffi::c_int,
        devlist: *const ::core::ffi::c_int,
    ) -> ncclResult_t;
    pub fn ncclCommInitRank(
        comm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        commId: ncclUniqueId,
        rank: ::core::ffi::c_int,
    ) -> ncclResult_t;
    pub fn ncclCommInitRankConfig(
        comm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        commId: ncclUniqueId,
        rank: ::core::ffi::c_int,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t;
    pub fn ncclCommInitRankScalable(
        newcomm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        myrank: ::core::ffi::c_int,
        nId: ::core::ffi::c_int,
        commIds: *mut ncclUniqueId,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t;
    pub fn ncclCommRegister(
        comm: ncclComm_t,
        buff: *mut ::core::ffi::c_void,
        size: usize,
        handle: *mut *mut ::core::ffi::c_void,
    ) -> ncclResult_t;
    pub fn ncclCommSplit(
        comm: ncclComm_t,
        color: ::core::ffi::c_int,
        key: ::core::ffi::c_int,
        newcomm: *mut ncclComm_t,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t;
    pub fn ncclCommUserRank(comm: ncclComm_t, rank: *mut ::core::ffi::c_int) -> ncclResult_t;
    pub fn ncclGetErrorString(result: ncclResult_t) -> *const ::core::ffi::c_char;
    pub fn ncclGetLastError(comm: ncclComm_t) -> *const ::core::ffi::c_char;
    pub fn ncclGetUniqueId(uniqueId: *mut ncclUniqueId) -> ncclResult_t;
    pub fn ncclGetVersion(version: *mut ::core::ffi::c_int) -> ncclResult_t;
    pub fn ncclGroupEnd() -> ncclResult_t;
    pub fn ncclGroupSimulateEnd(simInfo: *mut ncclSimInfo_t) -> ncclResult_t;
    pub fn ncclGroupStart() -> ncclResult_t;
    pub fn ncclMemAlloc(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> ncclResult_t;
    pub fn ncclMemFree(ptr: *mut ::core::ffi::c_void) -> ncclResult_t;
    pub fn ncclRecv(
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclRedOpCreatePreMulSum(
        op: *mut ncclRedOp_t,
        scalar: *mut ::core::ffi::c_void,
        datatype: ncclDataType_t,
        residence: ncclScalarResidence_t,
        comm: ncclComm_t,
    ) -> ncclResult_t;
    pub fn ncclRedOpDestroy(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t;
    pub fn ncclReduce(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclReduceScatter(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        recvcount: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
    pub fn ncclResetDebugInit();
    pub fn ncclSend(
        sendbuff: *const ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
}
#[cfg(feature = "dynamic-loading")]
mod loaded {
    use super::*;
    pub unsafe fn ncclAllGather(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        sendcount: usize,
        datatype: ncclDataType_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclAllGather)(sendbuff, recvbuff, sendcount, datatype, comm, stream)
    }
    pub unsafe fn ncclAllReduce(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclAllReduce)(sendbuff, recvbuff, count, datatype, op, comm, stream)
    }
    pub unsafe fn ncclBcast(
        buff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclBcast)(buff, count, datatype, root, comm, stream)
    }
    pub unsafe fn ncclBroadcast(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclBroadcast)(sendbuff, recvbuff, count, datatype, root, comm, stream)
    }
    pub unsafe fn ncclCommAbort(comm: ncclComm_t) -> ncclResult_t {
        (culib().ncclCommAbort)(comm)
    }
    pub unsafe fn ncclCommCount(comm: ncclComm_t, count: *mut ::core::ffi::c_int) -> ncclResult_t {
        (culib().ncclCommCount)(comm, count)
    }
    pub unsafe fn ncclCommCuDevice(
        comm: ncclComm_t,
        device: *mut ::core::ffi::c_int,
    ) -> ncclResult_t {
        (culib().ncclCommCuDevice)(comm, device)
    }
    pub unsafe fn ncclCommDeregister(
        comm: ncclComm_t,
        handle: *mut ::core::ffi::c_void,
    ) -> ncclResult_t {
        (culib().ncclCommDeregister)(comm, handle)
    }
    pub unsafe fn ncclCommDestroy(comm: ncclComm_t) -> ncclResult_t {
        (culib().ncclCommDestroy)(comm)
    }
    pub unsafe fn ncclCommFinalize(comm: ncclComm_t) -> ncclResult_t {
        (culib().ncclCommFinalize)(comm)
    }
    pub unsafe fn ncclCommGetAsyncError(
        comm: ncclComm_t,
        asyncError: *mut ncclResult_t,
    ) -> ncclResult_t {
        (culib().ncclCommGetAsyncError)(comm, asyncError)
    }
    pub unsafe fn ncclCommInitAll(
        comm: *mut ncclComm_t,
        ndev: ::core::ffi::c_int,
        devlist: *const ::core::ffi::c_int,
    ) -> ncclResult_t {
        (culib().ncclCommInitAll)(comm, ndev, devlist)
    }
    pub unsafe fn ncclCommInitRank(
        comm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        commId: ncclUniqueId,
        rank: ::core::ffi::c_int,
    ) -> ncclResult_t {
        (culib().ncclCommInitRank)(comm, nranks, commId, rank)
    }
    pub unsafe fn ncclCommInitRankConfig(
        comm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        commId: ncclUniqueId,
        rank: ::core::ffi::c_int,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t {
        (culib().ncclCommInitRankConfig)(comm, nranks, commId, rank, config)
    }
    pub unsafe fn ncclCommInitRankScalable(
        newcomm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        myrank: ::core::ffi::c_int,
        nId: ::core::ffi::c_int,
        commIds: *mut ncclUniqueId,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t {
        (culib().ncclCommInitRankScalable)(newcomm, nranks, myrank, nId, commIds, config)
    }
    pub unsafe fn ncclCommRegister(
        comm: ncclComm_t,
        buff: *mut ::core::ffi::c_void,
        size: usize,
        handle: *mut *mut ::core::ffi::c_void,
    ) -> ncclResult_t {
        (culib().ncclCommRegister)(comm, buff, size, handle)
    }
    pub unsafe fn ncclCommSplit(
        comm: ncclComm_t,
        color: ::core::ffi::c_int,
        key: ::core::ffi::c_int,
        newcomm: *mut ncclComm_t,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t {
        (culib().ncclCommSplit)(comm, color, key, newcomm, config)
    }
    pub unsafe fn ncclCommUserRank(
        comm: ncclComm_t,
        rank: *mut ::core::ffi::c_int,
    ) -> ncclResult_t {
        (culib().ncclCommUserRank)(comm, rank)
    }
    pub unsafe fn ncclGetErrorString(result: ncclResult_t) -> *const ::core::ffi::c_char {
        (culib().ncclGetErrorString)(result)
    }
    pub unsafe fn ncclGetLastError(comm: ncclComm_t) -> *const ::core::ffi::c_char {
        (culib().ncclGetLastError)(comm)
    }
    pub unsafe fn ncclGetUniqueId(uniqueId: *mut ncclUniqueId) -> ncclResult_t {
        (culib().ncclGetUniqueId)(uniqueId)
    }
    pub unsafe fn ncclGetVersion(version: *mut ::core::ffi::c_int) -> ncclResult_t {
        (culib().ncclGetVersion)(version)
    }
    pub unsafe fn ncclGroupEnd() -> ncclResult_t {
        (culib().ncclGroupEnd)()
    }
    pub unsafe fn ncclGroupSimulateEnd(simInfo: *mut ncclSimInfo_t) -> ncclResult_t {
        (culib().ncclGroupSimulateEnd)(simInfo)
    }
    pub unsafe fn ncclGroupStart() -> ncclResult_t {
        (culib().ncclGroupStart)()
    }
    pub unsafe fn ncclMemAlloc(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> ncclResult_t {
        (culib().ncclMemAlloc)(ptr, size)
    }
    pub unsafe fn ncclMemFree(ptr: *mut ::core::ffi::c_void) -> ncclResult_t {
        (culib().ncclMemFree)(ptr)
    }
    pub unsafe fn ncclRecv(
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclRecv)(recvbuff, count, datatype, peer, comm, stream)
    }
    pub unsafe fn ncclRedOpCreatePreMulSum(
        op: *mut ncclRedOp_t,
        scalar: *mut ::core::ffi::c_void,
        datatype: ncclDataType_t,
        residence: ncclScalarResidence_t,
        comm: ncclComm_t,
    ) -> ncclResult_t {
        (culib().ncclRedOpCreatePreMulSum)(op, scalar, datatype, residence, comm)
    }
    pub unsafe fn ncclRedOpDestroy(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t {
        (culib().ncclRedOpDestroy)(op, comm)
    }
    pub unsafe fn ncclReduce(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclReduce)(sendbuff, recvbuff, count, datatype, op, root, comm, stream)
    }
    pub unsafe fn ncclReduceScatter(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        recvcount: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclReduceScatter)(sendbuff, recvbuff, recvcount, datatype, op, comm, stream)
    }
    pub unsafe fn ncclResetDebugInit() {
        (culib().ncclResetDebugInit)()
    }
    pub unsafe fn ncclSend(
        sendbuff: *const ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (culib().ncclSend)(sendbuff, count, datatype, peer, comm, stream)
    }
    pub struct Lib {
        __library: ::libloading::Library,
        pub ncclAllGather: unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            sendcount: usize,
            datatype: ncclDataType_t,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclAllReduce: unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            op: ncclRedOp_t,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclBcast: unsafe extern "C" fn(
            buff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            root: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclBroadcast: unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            root: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclCommAbort: unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t,
        pub ncclCommCount:
            unsafe extern "C" fn(comm: ncclComm_t, count: *mut ::core::ffi::c_int) -> ncclResult_t,
        pub ncclCommCuDevice:
            unsafe extern "C" fn(comm: ncclComm_t, device: *mut ::core::ffi::c_int) -> ncclResult_t,
        pub ncclCommDeregister: unsafe extern "C" fn(
            comm: ncclComm_t,
            handle: *mut ::core::ffi::c_void,
        ) -> ncclResult_t,
        pub ncclCommDestroy: unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t,
        pub ncclCommFinalize: unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t,
        pub ncclCommGetAsyncError:
            unsafe extern "C" fn(comm: ncclComm_t, asyncError: *mut ncclResult_t) -> ncclResult_t,
        pub ncclCommInitAll: unsafe extern "C" fn(
            comm: *mut ncclComm_t,
            ndev: ::core::ffi::c_int,
            devlist: *const ::core::ffi::c_int,
        ) -> ncclResult_t,
        pub ncclCommInitRank: unsafe extern "C" fn(
            comm: *mut ncclComm_t,
            nranks: ::core::ffi::c_int,
            commId: ncclUniqueId,
            rank: ::core::ffi::c_int,
        ) -> ncclResult_t,
        pub ncclCommInitRankConfig: unsafe extern "C" fn(
            comm: *mut ncclComm_t,
            nranks: ::core::ffi::c_int,
            commId: ncclUniqueId,
            rank: ::core::ffi::c_int,
            config: *mut ncclConfig_t,
        ) -> ncclResult_t,
        pub ncclCommInitRankScalable: unsafe extern "C" fn(
            newcomm: *mut ncclComm_t,
            nranks: ::core::ffi::c_int,
            myrank: ::core::ffi::c_int,
            nId: ::core::ffi::c_int,
            commIds: *mut ncclUniqueId,
            config: *mut ncclConfig_t,
        ) -> ncclResult_t,
        pub ncclCommRegister: unsafe extern "C" fn(
            comm: ncclComm_t,
            buff: *mut ::core::ffi::c_void,
            size: usize,
            handle: *mut *mut ::core::ffi::c_void,
        ) -> ncclResult_t,
        pub ncclCommSplit: unsafe extern "C" fn(
            comm: ncclComm_t,
            color: ::core::ffi::c_int,
            key: ::core::ffi::c_int,
            newcomm: *mut ncclComm_t,
            config: *mut ncclConfig_t,
        ) -> ncclResult_t,
        pub ncclCommUserRank:
            unsafe extern "C" fn(comm: ncclComm_t, rank: *mut ::core::ffi::c_int) -> ncclResult_t,
        pub ncclGetErrorString:
            unsafe extern "C" fn(result: ncclResult_t) -> *const ::core::ffi::c_char,
        pub ncclGetLastError: unsafe extern "C" fn(comm: ncclComm_t) -> *const ::core::ffi::c_char,
        pub ncclGetUniqueId: unsafe extern "C" fn(uniqueId: *mut ncclUniqueId) -> ncclResult_t,
        pub ncclGetVersion: unsafe extern "C" fn(version: *mut ::core::ffi::c_int) -> ncclResult_t,
        pub ncclGroupEnd: unsafe extern "C" fn() -> ncclResult_t,
        pub ncclGroupSimulateEnd: unsafe extern "C" fn(simInfo: *mut ncclSimInfo_t) -> ncclResult_t,
        pub ncclGroupStart: unsafe extern "C" fn() -> ncclResult_t,
        pub ncclMemAlloc:
            unsafe extern "C" fn(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> ncclResult_t,
        pub ncclMemFree: unsafe extern "C" fn(ptr: *mut ::core::ffi::c_void) -> ncclResult_t,
        pub ncclRecv: unsafe extern "C" fn(
            recvbuff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            peer: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclRedOpCreatePreMulSum: unsafe extern "C" fn(
            op: *mut ncclRedOp_t,
            scalar: *mut ::core::ffi::c_void,
            datatype: ncclDataType_t,
            residence: ncclScalarResidence_t,
            comm: ncclComm_t,
        ) -> ncclResult_t,
        pub ncclRedOpDestroy:
            unsafe extern "C" fn(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t,
        pub ncclReduce: unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            op: ncclRedOp_t,
            root: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclReduceScatter: unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            recvcount: usize,
            datatype: ncclDataType_t,
            op: ncclRedOp_t,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        pub ncclResetDebugInit: unsafe extern "C" fn(),
        pub ncclSend: unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            peer: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
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
            let ncclAllGather = __library
                .get(b"ncclAllGather\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclAllReduce = __library
                .get(b"ncclAllReduce\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclBcast = __library
                .get(b"ncclBcast\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclBroadcast = __library
                .get(b"ncclBroadcast\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommAbort = __library
                .get(b"ncclCommAbort\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommCount = __library
                .get(b"ncclCommCount\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommCuDevice = __library
                .get(b"ncclCommCuDevice\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommDeregister = __library
                .get(b"ncclCommDeregister\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommDestroy = __library
                .get(b"ncclCommDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommFinalize = __library
                .get(b"ncclCommFinalize\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommGetAsyncError = __library
                .get(b"ncclCommGetAsyncError\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommInitAll = __library
                .get(b"ncclCommInitAll\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommInitRank = __library
                .get(b"ncclCommInitRank\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommInitRankConfig = __library
                .get(b"ncclCommInitRankConfig\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommInitRankScalable = __library
                .get(b"ncclCommInitRankScalable\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommRegister = __library
                .get(b"ncclCommRegister\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommSplit = __library
                .get(b"ncclCommSplit\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclCommUserRank = __library
                .get(b"ncclCommUserRank\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclGetErrorString = __library
                .get(b"ncclGetErrorString\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclGetLastError = __library
                .get(b"ncclGetLastError\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclGetUniqueId = __library
                .get(b"ncclGetUniqueId\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclGetVersion = __library
                .get(b"ncclGetVersion\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclGroupEnd = __library
                .get(b"ncclGroupEnd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclGroupSimulateEnd = __library
                .get(b"ncclGroupSimulateEnd\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclGroupStart = __library
                .get(b"ncclGroupStart\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclMemAlloc = __library
                .get(b"ncclMemAlloc\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclMemFree = __library
                .get(b"ncclMemFree\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclRecv = __library
                .get(b"ncclRecv\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclRedOpCreatePreMulSum = __library
                .get(b"ncclRedOpCreatePreMulSum\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclRedOpDestroy = __library
                .get(b"ncclRedOpDestroy\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclReduce = __library
                .get(b"ncclReduce\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclReduceScatter = __library
                .get(b"ncclReduceScatter\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclResetDebugInit = __library
                .get(b"ncclResetDebugInit\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            let ncclSend = __library
                .get(b"ncclSend\0")
                .map(|sym| *sym)
                .expect("Expected symbol in library");
            Ok(Self {
                __library,
                ncclAllGather,
                ncclAllReduce,
                ncclBcast,
                ncclBroadcast,
                ncclCommAbort,
                ncclCommCount,
                ncclCommCuDevice,
                ncclCommDeregister,
                ncclCommDestroy,
                ncclCommFinalize,
                ncclCommGetAsyncError,
                ncclCommInitAll,
                ncclCommInitRank,
                ncclCommInitRankConfig,
                ncclCommInitRankScalable,
                ncclCommRegister,
                ncclCommSplit,
                ncclCommUserRank,
                ncclGetErrorString,
                ncclGetLastError,
                ncclGetUniqueId,
                ncclGetVersion,
                ncclGroupEnd,
                ncclGroupSimulateEnd,
                ncclGroupStart,
                ncclMemAlloc,
                ncclMemFree,
                ncclRecv,
                ncclRedOpCreatePreMulSum,
                ncclRedOpDestroy,
                ncclReduce,
                ncclReduceScatter,
                ncclResetDebugInit,
                ncclSend,
            })
        }
    }
    pub unsafe fn culib() -> &'static Lib {
        static LIB: std::sync::OnceLock<Lib> = std::sync::OnceLock::new();
        LIB.get_or_init(|| {
            let lib_names = std::vec!["nccl"];
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
