/* automatically generated by rust-bindgen 0.69.4 */

pub const CUDA_VERSION: u32 = 11070;
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
#[test]
fn bindgen_test_layout_ncclUniqueId() {
    const UNINIT: ::core::mem::MaybeUninit<ncclUniqueId> = ::core::mem::MaybeUninit::uninit();
    let ptr = UNINIT.as_ptr();
    assert_eq!(
        ::core::mem::size_of::<ncclUniqueId>(),
        128usize,
        concat!("Size of: ", stringify!(ncclUniqueId))
    );
    assert_eq!(
        ::core::mem::align_of::<ncclUniqueId>(),
        1usize,
        concat!("Alignment of ", stringify!(ncclUniqueId))
    );
    assert_eq!(
        unsafe { ::core::ptr::addr_of!((*ptr).internal) as usize - ptr as usize },
        0usize,
        concat!(
            "Offset of field: ",
            stringify!(ncclUniqueId),
            "::",
            stringify!(internal)
        )
    );
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
    ncclNumResults = 7,
}
extern "C" {
    pub fn ncclGetVersion(version: *mut ::core::ffi::c_int) -> ncclResult_t;
}
extern "C" {
    pub fn ncclGetUniqueId(uniqueId: *mut ncclUniqueId) -> ncclResult_t;
}
extern "C" {
    pub fn ncclCommInitRank(
        comm: *mut ncclComm_t,
        nranks: ::core::ffi::c_int,
        commId: ncclUniqueId,
        rank: ::core::ffi::c_int,
    ) -> ncclResult_t;
}
extern "C" {
    pub fn ncclCommInitAll(
        comm: *mut ncclComm_t,
        ndev: ::core::ffi::c_int,
        devlist: *const ::core::ffi::c_int,
    ) -> ncclResult_t;
}
extern "C" {
    pub fn ncclCommDestroy(comm: ncclComm_t) -> ncclResult_t;
}
extern "C" {
    pub fn ncclCommAbort(comm: ncclComm_t) -> ncclResult_t;
}
extern "C" {
    pub fn ncclGetErrorString(result: ncclResult_t) -> *const ::core::ffi::c_char;
}
extern "C" {
    pub fn ncclGetLastError(comm: ncclComm_t) -> *const ::core::ffi::c_char;
}
extern "C" {
    pub fn ncclCommGetAsyncError(comm: ncclComm_t, asyncError: *mut ncclResult_t) -> ncclResult_t;
}
extern "C" {
    pub fn ncclCommCount(comm: ncclComm_t, count: *mut ::core::ffi::c_int) -> ncclResult_t;
}
extern "C" {
    pub fn ncclCommCuDevice(comm: ncclComm_t, device: *mut ::core::ffi::c_int) -> ncclResult_t;
}
extern "C" {
    pub fn ncclCommUserRank(comm: ncclComm_t, rank: *mut ::core::ffi::c_int) -> ncclResult_t;
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
    ncclNumTypes = 10,
}
#[repr(u32)]
#[derive(Debug, Copy, Clone, Hash, PartialOrd, Ord, PartialEq, Eq)]
pub enum ncclScalarResidence_t {
    ncclScalarDevice = 0,
    ncclScalarHostImmediate = 1,
}
extern "C" {
    pub fn ncclRedOpCreatePreMulSum(
        op: *mut ncclRedOp_t,
        scalar: *mut ::core::ffi::c_void,
        datatype: ncclDataType_t,
        residence: ncclScalarResidence_t,
        comm: ncclComm_t,
    ) -> ncclResult_t;
}
extern "C" {
    pub fn ncclRedOpDestroy(op: ncclRedOp_t, comm: ncclComm_t) -> ncclResult_t;
}
extern "C" {
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
}
extern "C" {
    pub fn ncclBcast(
        buff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
}
extern "C" {
    pub fn ncclBroadcast(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        root: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
}
extern "C" {
    pub fn ncclAllReduce(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
}
extern "C" {
    pub fn ncclReduceScatter(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        recvcount: usize,
        datatype: ncclDataType_t,
        op: ncclRedOp_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
}
extern "C" {
    pub fn ncclAllGather(
        sendbuff: *const ::core::ffi::c_void,
        recvbuff: *mut ::core::ffi::c_void,
        sendcount: usize,
        datatype: ncclDataType_t,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
}
extern "C" {
    pub fn ncclSend(
        sendbuff: *const ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
}
extern "C" {
    pub fn ncclRecv(
        recvbuff: *mut ::core::ffi::c_void,
        count: usize,
        datatype: ncclDataType_t,
        peer: ::core::ffi::c_int,
        comm: ncclComm_t,
        stream: cudaStream_t,
    ) -> ncclResult_t;
}
extern "C" {
    pub fn ncclGroupStart() -> ncclResult_t;
}
extern "C" {
    pub fn ncclGroupEnd() -> ncclResult_t;
}
