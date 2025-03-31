/* automatically generated by rust-bindgen 0.71.1 */

#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CUstream_st {
    _unused: [u8; 0],
}
pub type cudaStream_t = *mut CUstream_st;
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct ncclComm {
    _unused: [u8; 0],
}
pub type ncclComm_t = *mut ncclComm;
#[repr(C)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub struct ncclUniqueId {
    pub internal: [::core::ffi::c_char; 128usize],
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of ncclUniqueId"][::core::mem::size_of::<ncclUniqueId>() - 128usize];
    ["Alignment of ncclUniqueId"][::core::mem::align_of::<ncclUniqueId>() - 1usize];
    ["Offset of field: ncclUniqueId::internal"]
        [::core::mem::offset_of!(ncclUniqueId, internal) - 0usize];
};
impl Default for ncclUniqueId {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
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
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of ncclConfig_v21700"][::core::mem::size_of::<ncclConfig_v21700>() - 48usize];
    ["Alignment of ncclConfig_v21700"][::core::mem::align_of::<ncclConfig_v21700>() - 8usize];
    ["Offset of field: ncclConfig_v21700::size"]
        [::core::mem::offset_of!(ncclConfig_v21700, size) - 0usize];
    ["Offset of field: ncclConfig_v21700::magic"]
        [::core::mem::offset_of!(ncclConfig_v21700, magic) - 8usize];
    ["Offset of field: ncclConfig_v21700::version"]
        [::core::mem::offset_of!(ncclConfig_v21700, version) - 12usize];
    ["Offset of field: ncclConfig_v21700::blocking"]
        [::core::mem::offset_of!(ncclConfig_v21700, blocking) - 16usize];
    ["Offset of field: ncclConfig_v21700::cgaClusterSize"]
        [::core::mem::offset_of!(ncclConfig_v21700, cgaClusterSize) - 20usize];
    ["Offset of field: ncclConfig_v21700::minCTAs"]
        [::core::mem::offset_of!(ncclConfig_v21700, minCTAs) - 24usize];
    ["Offset of field: ncclConfig_v21700::maxCTAs"]
        [::core::mem::offset_of!(ncclConfig_v21700, maxCTAs) - 28usize];
    ["Offset of field: ncclConfig_v21700::netName"]
        [::core::mem::offset_of!(ncclConfig_v21700, netName) - 32usize];
    ["Offset of field: ncclConfig_v21700::splitShare"]
        [::core::mem::offset_of!(ncclConfig_v21700, splitShare) - 40usize];
    ["Offset of field: ncclConfig_v21700::trafficClass"]
        [::core::mem::offset_of!(ncclConfig_v21700, trafficClass) - 44usize];
};
impl Default for ncclConfig_v21700 {
    fn default() -> Self {
        let mut s = ::core::mem::MaybeUninit::<Self>::uninit();
        unsafe {
            ::core::ptr::write_bytes(s.as_mut_ptr(), 0, 1);
            s.assume_init()
        }
    }
}
pub type ncclConfig_t = ncclConfig_v21700;
#[repr(C)]
#[derive(Debug, Default, Copy, Clone, PartialOrd, PartialEq)]
pub struct ncclSimInfo_v22200 {
    pub size: usize,
    pub magic: ::core::ffi::c_uint,
    pub version: ::core::ffi::c_uint,
    pub estimatedTime: f32,
}
#[allow(clippy::unnecessary_operation, clippy::identity_op)]
const _: () = {
    ["Size of ncclSimInfo_v22200"][::core::mem::size_of::<ncclSimInfo_v22200>() - 24usize];
    ["Alignment of ncclSimInfo_v22200"][::core::mem::align_of::<ncclSimInfo_v22200>() - 8usize];
    ["Offset of field: ncclSimInfo_v22200::size"]
        [::core::mem::offset_of!(ncclSimInfo_v22200, size) - 0usize];
    ["Offset of field: ncclSimInfo_v22200::magic"]
        [::core::mem::offset_of!(ncclSimInfo_v22200, magic) - 8usize];
    ["Offset of field: ncclSimInfo_v22200::version"]
        [::core::mem::offset_of!(ncclSimInfo_v22200, version) - 12usize];
    ["Offset of field: ncclSimInfo_v22200::estimatedTime"]
        [::core::mem::offset_of!(ncclSimInfo_v22200, estimatedTime) - 16usize];
};
pub type ncclSimInfo_t = ncclSimInfo_v22200;
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
impl ncclDataType_t {
    pub const ncclChar: ncclDataType_t = ncclDataType_t::ncclInt8;
}
impl ncclDataType_t {
    pub const ncclInt: ncclDataType_t = ncclDataType_t::ncclInt32;
}
impl ncclDataType_t {
    pub const ncclHalf: ncclDataType_t = ncclDataType_t::ncclFloat16;
}
impl ncclDataType_t {
    pub const ncclFloat: ncclDataType_t = ncclDataType_t::ncclFloat32;
}
impl ncclDataType_t {
    pub const ncclDouble: ncclDataType_t = ncclDataType_t::ncclFloat64;
}
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
pub enum ncclScalarResidence_t {
    ncclScalarDevice = 0,
    ncclScalarHostImmediate = 1,
}
pub struct Lib {
    __library: ::libloading::Library,
    pub ncclMemAlloc: Result<
        unsafe extern "C" fn(ptr: *mut *mut ::core::ffi::c_void, size: usize) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclMemFree: Result<
        unsafe extern "C" fn(ptr: *mut ::core::ffi::c_void) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclGetVersion: Result<
        unsafe extern "C" fn(version: *mut ::core::ffi::c_int) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclGetUniqueId: Result<
        unsafe extern "C" fn(uniqueId: *mut ncclUniqueId) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclCommInitRankConfig: Result<
        unsafe extern "C" fn(
            comm: *mut ncclComm_t,
            nranks: ::core::ffi::c_int,
            commId: ncclUniqueId,
            rank: ::core::ffi::c_int,
            config: *mut ncclConfig_t,
        ) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclCommInitRank: Result<
        unsafe extern "C" fn(
            comm: *mut ncclComm_t,
            nranks: ::core::ffi::c_int,
            commId: ncclUniqueId,
            rank: ::core::ffi::c_int,
        ) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclCommInitAll: Result<
        unsafe extern "C" fn(
            comm: *mut ncclComm_t,
            ndev: ::core::ffi::c_int,
            devlist: *const ::core::ffi::c_int,
        ) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclCommFinalize:
        Result<unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t, ::libloading::Error>,
    pub ncclCommDestroy:
        Result<unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t, ::libloading::Error>,
    pub ncclCommAbort:
        Result<unsafe extern "C" fn(comm: ncclComm_t) -> ncclResult_t, ::libloading::Error>,
    pub ncclCommSplit: Result<
        unsafe extern "C" fn(
            comm: ncclComm_t,
            color: ::core::ffi::c_int,
            key: ::core::ffi::c_int,
            newcomm: *mut ncclComm_t,
            config: *mut ncclConfig_t,
        ) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclCommInitRankScalable: Result<
        unsafe extern "C" fn(
            newcomm: *mut ncclComm_t,
            nranks: ::core::ffi::c_int,
            myrank: ::core::ffi::c_int,
            nId: ::core::ffi::c_int,
            commIds: *mut ncclUniqueId,
            config: *mut ncclConfig_t,
        ) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclGetErrorString: Result<
        unsafe extern "C" fn(result: ncclResult_t) -> *const ::core::ffi::c_char,
        ::libloading::Error,
    >,
    pub ncclGetLastError: Result<
        unsafe extern "C" fn(comm: ncclComm_t) -> *const ::core::ffi::c_char,
        ::libloading::Error,
    >,
    pub ncclResetDebugInit: Result<unsafe extern "C" fn(), ::libloading::Error>,
    pub ncclCommGetAsyncError: Result<
        unsafe extern "C" fn(comm: ncclComm_t, asyncError: *mut ncclResult_t) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclCommCount: Result<
        unsafe extern "C" fn(comm: ncclComm_t, count: *mut ::core::ffi::c_int) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclCommCuDevice: Result<
        unsafe extern "C" fn(comm: ncclComm_t, device: *mut ::core::ffi::c_int) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclCommUserRank: Result<
        unsafe extern "C" fn(comm: ncclComm_t, rank: *mut ::core::ffi::c_int) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclCommRegister: Result<
        unsafe extern "C" fn(
            comm: ncclComm_t,
            buff: *mut ::core::ffi::c_void,
            size: usize,
            handle: *mut *mut ::core::ffi::c_void,
        ) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclCommDeregister: Result<
        unsafe extern "C" fn(comm: ncclComm_t, handle: *mut ::core::ffi::c_void) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclRedOpCreatePreMulSum: Result<
        unsafe extern "C" fn(
            op: *mut ncclRedOp_t,
            scalar: *mut ::core::ffi::c_void,
            datatype: ncclDataType_t,
            residence: ncclScalarResidence_t,
            comm: ncclComm_t,
        ) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclRedOpDestroy: Result<
        unsafe extern "C" fn(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclReduce: Result<
        unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            op: ncclRedOp_t,
            root: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclBcast: Result<
        unsafe extern "C" fn(
            buff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            root: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclBroadcast: Result<
        unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            root: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclAllReduce: Result<
        unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            op: ncclRedOp_t,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclReduceScatter: Result<
        unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            recvcount: usize,
            datatype: ncclDataType_t,
            op: ncclRedOp_t,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclAllGather: Result<
        unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            recvbuff: *mut ::core::ffi::c_void,
            sendcount: usize,
            datatype: ncclDataType_t,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclSend: Result<
        unsafe extern "C" fn(
            sendbuff: *const ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            peer: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclRecv: Result<
        unsafe extern "C" fn(
            recvbuff: *mut ::core::ffi::c_void,
            count: usize,
            datatype: ncclDataType_t,
            peer: ::core::ffi::c_int,
            comm: ncclComm_t,
            stream: cudaStream_t,
        ) -> ncclResult_t,
        ::libloading::Error,
    >,
    pub ncclGroupStart: Result<unsafe extern "C" fn() -> ncclResult_t, ::libloading::Error>,
    pub ncclGroupEnd: Result<unsafe extern "C" fn() -> ncclResult_t, ::libloading::Error>,
    pub ncclGroupSimulateEnd: Result<
        unsafe extern "C" fn(simInfo: *mut ncclSimInfo_t) -> ncclResult_t,
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
        let ncclMemAlloc = __library.get(b"ncclMemAlloc\0").map(|sym| *sym);
        let ncclMemFree = __library.get(b"ncclMemFree\0").map(|sym| *sym);
        let ncclGetVersion = __library.get(b"ncclGetVersion\0").map(|sym| *sym);
        let ncclGetUniqueId = __library.get(b"ncclGetUniqueId\0").map(|sym| *sym);
        let ncclCommInitRankConfig = __library.get(b"ncclCommInitRankConfig\0").map(|sym| *sym);
        let ncclCommInitRank = __library.get(b"ncclCommInitRank\0").map(|sym| *sym);
        let ncclCommInitAll = __library.get(b"ncclCommInitAll\0").map(|sym| *sym);
        let ncclCommFinalize = __library.get(b"ncclCommFinalize\0").map(|sym| *sym);
        let ncclCommDestroy = __library.get(b"ncclCommDestroy\0").map(|sym| *sym);
        let ncclCommAbort = __library.get(b"ncclCommAbort\0").map(|sym| *sym);
        let ncclCommSplit = __library.get(b"ncclCommSplit\0").map(|sym| *sym);
        let ncclCommInitRankScalable = __library.get(b"ncclCommInitRankScalable\0").map(|sym| *sym);
        let ncclGetErrorString = __library.get(b"ncclGetErrorString\0").map(|sym| *sym);
        let ncclGetLastError = __library.get(b"ncclGetLastError\0").map(|sym| *sym);
        let ncclResetDebugInit = __library.get(b"ncclResetDebugInit\0").map(|sym| *sym);
        let ncclCommGetAsyncError = __library.get(b"ncclCommGetAsyncError\0").map(|sym| *sym);
        let ncclCommCount = __library.get(b"ncclCommCount\0").map(|sym| *sym);
        let ncclCommCuDevice = __library.get(b"ncclCommCuDevice\0").map(|sym| *sym);
        let ncclCommUserRank = __library.get(b"ncclCommUserRank\0").map(|sym| *sym);
        let ncclCommRegister = __library.get(b"ncclCommRegister\0").map(|sym| *sym);
        let ncclCommDeregister = __library.get(b"ncclCommDeregister\0").map(|sym| *sym);
        let ncclRedOpCreatePreMulSum = __library.get(b"ncclRedOpCreatePreMulSum\0").map(|sym| *sym);
        let ncclRedOpDestroy = __library.get(b"ncclRedOpDestroy\0").map(|sym| *sym);
        let ncclReduce = __library.get(b"ncclReduce\0").map(|sym| *sym);
        let ncclBcast = __library.get(b"ncclBcast\0").map(|sym| *sym);
        let ncclBroadcast = __library.get(b"ncclBroadcast\0").map(|sym| *sym);
        let ncclAllReduce = __library.get(b"ncclAllReduce\0").map(|sym| *sym);
        let ncclReduceScatter = __library.get(b"ncclReduceScatter\0").map(|sym| *sym);
        let ncclAllGather = __library.get(b"ncclAllGather\0").map(|sym| *sym);
        let ncclSend = __library.get(b"ncclSend\0").map(|sym| *sym);
        let ncclRecv = __library.get(b"ncclRecv\0").map(|sym| *sym);
        let ncclGroupStart = __library.get(b"ncclGroupStart\0").map(|sym| *sym);
        let ncclGroupEnd = __library.get(b"ncclGroupEnd\0").map(|sym| *sym);
        let ncclGroupSimulateEnd = __library.get(b"ncclGroupSimulateEnd\0").map(|sym| *sym);
        Ok(Lib {
            __library,
            ncclMemAlloc,
            ncclMemFree,
            ncclGetVersion,
            ncclGetUniqueId,
            ncclCommInitRankConfig,
            ncclCommInitRank,
            ncclCommInitAll,
            ncclCommFinalize,
            ncclCommDestroy,
            ncclCommAbort,
            ncclCommSplit,
            ncclCommInitRankScalable,
            ncclGetErrorString,
            ncclGetLastError,
            ncclResetDebugInit,
            ncclCommGetAsyncError,
            ncclCommCount,
            ncclCommCuDevice,
            ncclCommUserRank,
            ncclCommRegister,
            ncclCommDeregister,
            ncclRedOpCreatePreMulSum,
            ncclRedOpDestroy,
            ncclReduce,
            ncclBcast,
            ncclBroadcast,
            ncclAllReduce,
            ncclReduceScatter,
            ncclAllGather,
            ncclSend,
            ncclRecv,
            ncclGroupStart,
            ncclGroupEnd,
            ncclGroupSimulateEnd,
        })
    }
    pub unsafe fn ncclMemAlloc(
        &self,
        ptr: *mut *mut ::core::ffi::c_void,
        size: usize,
    ) -> ncclResult_t {
        (self
            .ncclMemAlloc
            .as_ref()
            .expect("Expected function, got error."))(ptr, size)
    }
    pub unsafe fn ncclMemFree(&self, ptr: *mut ::core::ffi::c_void) -> ncclResult_t {
        (self
            .ncclMemFree
            .as_ref()
            .expect("Expected function, got error."))(ptr)
    }
    pub unsafe fn ncclGetVersion(&self, version: *mut ::core::ffi::c_int) -> ncclResult_t {
        (self
            .ncclGetVersion
            .as_ref()
            .expect("Expected function, got error."))(version)
    }
    pub unsafe fn ncclGetUniqueId(&self, uniqueId: *mut ncclUniqueId) -> ncclResult_t {
        (self
            .ncclGetUniqueId
            .as_ref()
            .expect("Expected function, got error."))(uniqueId)
    }
    pub unsafe fn ncclCommInitRankConfig(
        &self,
        comm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        commId: ncclUniqueId,
        rank: ::core::ffi::c_int,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t {
        (self
            .ncclCommInitRankConfig
            .as_ref()
            .expect("Expected function, got error."))(comm, nranks, commId, rank, config)
    }
    pub unsafe fn ncclCommInitRank(
        &self,
        comm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        commId: ncclUniqueId,
        rank: ::core::ffi::c_int,
    ) -> ncclResult_t {
        (self
            .ncclCommInitRank
            .as_ref()
            .expect("Expected function, got error."))(comm, nranks, commId, rank)
    }
    pub unsafe fn ncclCommInitAll(
        &self,
        comm: *mut ncclComm_t,
        ndev: ::core::ffi::c_int,
        devlist: *const ::core::ffi::c_int,
    ) -> ncclResult_t {
        (self
            .ncclCommInitAll
            .as_ref()
            .expect("Expected function, got error."))(comm, ndev, devlist)
    }
    pub unsafe fn ncclCommFinalize(&self, comm: ncclComm_t) -> ncclResult_t {
        (self
            .ncclCommFinalize
            .as_ref()
            .expect("Expected function, got error."))(comm)
    }
    pub unsafe fn ncclCommDestroy(&self, comm: ncclComm_t) -> ncclResult_t {
        (self
            .ncclCommDestroy
            .as_ref()
            .expect("Expected function, got error."))(comm)
    }
    pub unsafe fn ncclCommAbort(&self, comm: ncclComm_t) -> ncclResult_t {
        (self
            .ncclCommAbort
            .as_ref()
            .expect("Expected function, got error."))(comm)
    }
    pub unsafe fn ncclCommSplit(
        &self,
        comm: ncclComm_t,
        color: ::core::ffi::c_int,
        key: ::core::ffi::c_int,
        newcomm: *mut ncclComm_t,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t {
        (self
            .ncclCommSplit
            .as_ref()
            .expect("Expected function, got error."))(comm, color, key, newcomm, config)
    }
    pub unsafe fn ncclCommInitRankScalable(
        &self,
        newcomm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        myrank: ::core::ffi::c_int,
        nId: ::core::ffi::c_int,
        commIds: *mut ncclUniqueId,
        config: *mut ncclConfig_t,
    ) -> ncclResult_t {
        (self
            .ncclCommInitRankScalable
            .as_ref()
            .expect("Expected function, got error."))(
            newcomm, nranks, myrank, nId, commIds, config
        )
    }
    pub unsafe fn ncclGetErrorString(&self, result: ncclResult_t) -> *const ::core::ffi::c_char {
        (self
            .ncclGetErrorString
            .as_ref()
            .expect("Expected function, got error."))(result)
    }
    pub unsafe fn ncclGetLastError(&self, comm: ncclComm_t) -> *const ::core::ffi::c_char {
        (self
            .ncclGetLastError
            .as_ref()
            .expect("Expected function, got error."))(comm)
    }
    pub unsafe fn ncclResetDebugInit(&self) {
        (self
            .ncclResetDebugInit
            .as_ref()
            .expect("Expected function, got error."))()
    }
    pub unsafe fn ncclCommGetAsyncError(
        &self,
        comm: ncclComm_t,
        asyncError: *mut ncclResult_t,
    ) -> ncclResult_t {
        (self
            .ncclCommGetAsyncError
            .as_ref()
            .expect("Expected function, got error."))(comm, asyncError)
    }
    pub unsafe fn ncclCommCount(
        &self,
        comm: ncclComm_t,
        count: *mut ::core::ffi::c_int,
    ) -> ncclResult_t {
        (self
            .ncclCommCount
            .as_ref()
            .expect("Expected function, got error."))(comm, count)
    }
    pub unsafe fn ncclCommCuDevice(
        &self,
        comm: ncclComm_t,
        device: *mut ::core::ffi::c_int,
    ) -> ncclResult_t {
        (self
            .ncclCommCuDevice
            .as_ref()
            .expect("Expected function, got error."))(comm, device)
    }
    pub unsafe fn ncclCommUserRank(
        &self,
        comm: ncclComm_t,
        rank: *mut ::core::ffi::c_int,
    ) -> ncclResult_t {
        (self
            .ncclCommUserRank
            .as_ref()
            .expect("Expected function, got error."))(comm, rank)
    }
    pub unsafe fn ncclCommRegister(
        &self,
        comm: ncclComm_t,
        buff: *mut ::core::ffi::c_void,
        size: usize,
        handle: *mut *mut ::core::ffi::c_void,
    ) -> ncclResult_t {
        (self
            .ncclCommRegister
            .as_ref()
            .expect("Expected function, got error."))(comm, buff, size, handle)
    }
    pub unsafe fn ncclCommDeregister(
        &self,
        comm: ncclComm_t,
        handle: *mut ::core::ffi::c_void,
    ) -> ncclResult_t {
        (self
            .ncclCommDeregister
            .as_ref()
            .expect("Expected function, got error."))(comm, handle)
    }
    pub unsafe fn ncclRedOpCreatePreMulSum(
        &self,
        op: *mut ncclRedOp_t,
        scalar: *mut ::core::ffi::c_void,
        datatype: ncclDataType_t,
        residence: ncclScalarResidence_t,
        comm: ncclComm_t,
    ) -> ncclResult_t {
        (self
            .ncclRedOpCreatePreMulSum
            .as_ref()
            .expect("Expected function, got error."))(op, scalar, datatype, residence, comm)
    }
    pub unsafe fn ncclRedOpDestroy(&self, op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t {
        (self
            .ncclRedOpDestroy
            .as_ref()
            .expect("Expected function, got error."))(op, comm)
    }
    pub unsafe fn ncclReduce(
        &self,
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (self
            .ncclReduce
            .as_ref()
            .expect("Expected function, got error."))(
            sendbuff, recvbuff, count, datatype, op, root, comm, stream,
        )
    }
    pub unsafe fn ncclBcast(
        &self,
        buff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (self
            .ncclBcast
            .as_ref()
            .expect("Expected function, got error."))(
            buff, count, datatype, root, comm, stream
        )
    }
    pub unsafe fn ncclBroadcast(
        &self,
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (self
            .ncclBroadcast
            .as_ref()
            .expect("Expected function, got error."))(
            sendbuff, recvbuff, count, datatype, root, comm, stream,
        )
    }
    pub unsafe fn ncclAllReduce(
        &self,
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (self
            .ncclAllReduce
            .as_ref()
            .expect("Expected function, got error."))(
            sendbuff, recvbuff, count, datatype, op, comm, stream,
        )
    }
    pub unsafe fn ncclReduceScatter(
        &self,
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        recvcount: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (self
            .ncclReduceScatter
            .as_ref()
            .expect("Expected function, got error."))(
            sendbuff, recvbuff, recvcount, datatype, op, comm, stream,
        )
    }
    pub unsafe fn ncclAllGather(
        &self,
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        sendcount: usize,
        datatype: ncclDataType_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (self
            .ncclAllGather
            .as_ref()
            .expect("Expected function, got error."))(
            sendbuff, recvbuff, sendcount, datatype, comm, stream,
        )
    }
    pub unsafe fn ncclSend(
        &self,
        sendbuff: *const ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (self
            .ncclSend
            .as_ref()
            .expect("Expected function, got error."))(
            sendbuff, count, datatype, peer, comm, stream
        )
    }
    pub unsafe fn ncclRecv(
        &self,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t {
        (self
            .ncclRecv
            .as_ref()
            .expect("Expected function, got error."))(
            recvbuff, count, datatype, peer, comm, stream
        )
    }
    pub unsafe fn ncclGroupStart(&self) -> ncclResult_t {
        (self
            .ncclGroupStart
            .as_ref()
            .expect("Expected function, got error."))()
    }
    pub unsafe fn ncclGroupEnd(&self) -> ncclResult_t {
        (self
            .ncclGroupEnd
            .as_ref()
            .expect("Expected function, got error."))()
    }
    pub unsafe fn ncclGroupSimulateEnd(&self, simInfo: *mut ncclSimInfo_t) -> ncclResult_t {
        (self
            .ncclGroupSimulateEnd
            .as_ref()
            .expect("Expected function, got error."))(simInfo)
    }
}
