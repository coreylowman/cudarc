//! A thin wrapper around [sys] providing [Result]s with [NcclError].

use super::sys::{self, ncclCommSplit, ncclGetVersion, ncclRedOpCreatePreMulSum, ncclRedOpDestroy};
use std::mem::MaybeUninit;

/// Wrapper around [sys::ncclResult_t].
/// See [NCCL docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/types.html?highlight=ncclresult_t#ncclresult-t)
#[derive(Clone, PartialEq, Eq)]
pub struct NcclError(pub sys::ncclResult_t);

// #[cfg(feature = "std")]
impl std::fmt::Debug for NcclError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NcclError")
    }
}

#[derive(Clone, PartialEq, Eq)]
pub enum NcclStatus {
    Success,
    InProgress,
    NumResults,
}

impl sys::ncclResult_t {
    /// Transforms into a [Result] of [NcclError]
    pub fn result(self) -> Result<NcclStatus, NcclError> {
        match self {
            sys::ncclResult_t::ncclSuccess => Ok(NcclStatus::Success),
            sys::ncclResult_t::ncclInProgress => Ok(NcclStatus::InProgress),
            sys::ncclResult_t::ncclNumResults => Ok(NcclStatus::NumResults),
            _ => Err(NcclError(self)),
        }
    }
}

// #[cfg(feature = "std")]
// impl std::fmt::Display for NcclError {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         write!(f, "{self:?}")
//     }
// }

pub unsafe fn comm_finalize(comm: sys::ncclComm_t) -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclCommFinalize(comm).result() }
}

pub unsafe fn comm_destry(comm: sys::ncclComm_t) -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclCommDestroy(comm).result() }
}

pub unsafe fn comm_abort(comm: sys::ncclComm_t) -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclCommAbort(comm).result() }
}

pub unsafe fn get_nccl_version() -> Result<::core::ffi::c_int, NcclError> {
    unsafe {
        let mut version: ::core::ffi::c_int = 0;
        ncclGetVersion(&mut version).result()?;
        return Ok(version);
    }
}

pub unsafe fn get_uniqueid() -> Result<sys::ncclUniqueId, NcclError> {
    unsafe {
        let mut uniqueid = MaybeUninit::uninit();
        sys::ncclGetUniqueId(uniqueid.as_mut_ptr()).result()?;
        return Ok(uniqueid.assume_init());
    }
}

pub unsafe fn comm_init_rank_config(
    comm: *mut sys::ncclComm_t,
    nranks: ::core::ffi::c_int,
    comm_id: sys::ncclUniqueId,
    rank: ::core::ffi::c_int,
    config: *mut sys::ncclConfig_t,
) -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclCommInitRankConfig(comm, nranks, comm_id, rank, config).result() }
}

pub unsafe fn comm_init_rank(
    comm: *mut sys::ncclComm_t,
    nranks: ::core::ffi::c_int,
    comm_id: sys::ncclUniqueId,
    rank: ::core::ffi::c_int,
) -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclCommInitRank(comm, nranks, comm_id, rank).result() }
}

pub unsafe fn comm_init_all(
    comm: *mut sys::ncclComm_t,
    ndev: ::core::ffi::c_int,
    devlist: *const ::core::ffi::c_int,
) -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclCommInitAll(comm, ndev, devlist).result() }
}

pub unsafe fn comm_split(
    comm: sys::ncclComm_t,
    color: ::core::ffi::c_int,
    key: ::core::ffi::c_int,
    newcomm: *mut sys::ncclComm_t,
    config: *mut sys::ncclConfig_t,
) -> Result<NcclStatus, NcclError> {
    unsafe { ncclCommSplit(comm, color, key, newcomm, config).result() }
}

pub unsafe fn comm_count(comm: sys::ncclComm_t) -> Result<::core::ffi::c_int, NcclError> {
    unsafe {
        let mut count = 0;
        sys::ncclCommCount(comm, &mut count).result()?;
        Ok(count)
    }
}

pub unsafe fn comm_cu_device(comm: sys::ncclComm_t) -> Result<::core::ffi::c_int, NcclError> {
    unsafe {
        let mut device = 0;
        sys::ncclCommCuDevice(comm, &mut device).result()?;
        Ok(device)
    }
}

pub unsafe fn comm_user_rank(comm: sys::ncclComm_t) -> Result<::core::ffi::c_int, NcclError> {
    unsafe {
        let mut rank = 0;
        sys::ncclCommUserRank(comm, &mut rank).result()?;
        Ok(rank)
    }
}

pub unsafe fn reduce_op_create_pre_mul_sum(
    op: *mut sys::ncclRedOp_t,
    scalar: *mut ::core::ffi::c_void,
    datatype: sys::ncclDataType_t,
    residence: sys::ncclScalarResidence_t,
    comm: sys::ncclComm_t,
) -> Result<NcclStatus, NcclError> {
    unsafe { ncclRedOpCreatePreMulSum(op, scalar, datatype, residence, comm).result() }
}

pub unsafe fn reduce_op_destory(
    op: sys::ncclRedOp_t,
    comm: sys::ncclComm_t,
) -> Result<NcclStatus, NcclError> {
    unsafe { ncclRedOpDestroy(op, comm).result() }
}

pub unsafe fn reduce(
    sendbuff: *const ::core::ffi::c_void,
    recvbuff: *mut ::core::ffi::c_void,
    count: usize,
    datatype: sys::ncclDataType_t,
    op: sys::ncclRedOp_t,
    root: ::core::ffi::c_int,
    comm: sys::ncclComm_t,
    stream: sys::cudaStream_t,
) -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclReduce(sendbuff, recvbuff, count, datatype, op, root, comm, stream).result() }
}

pub unsafe fn bcast(
    buff: *mut ::core::ffi::c_void,
    count: usize,
    datatype: sys::ncclDataType_t,
    root: ::core::ffi::c_int,
    comm: sys::ncclComm_t,
    stream: sys::cudaStream_t,
) -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclBcast(buff, count, datatype, root, comm, stream).result() }
}

pub unsafe fn broadcast(
    sendbuff: *const ::core::ffi::c_void,
    recvbuff: *mut ::core::ffi::c_void,
    count: usize,
    datatype: sys::ncclDataType_t,
    root: ::core::ffi::c_int,
    comm: sys::ncclComm_t,
    stream: sys::cudaStream_t,
) -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclBroadcast(sendbuff, recvbuff, count, datatype, root, comm, stream).result() }
}

pub unsafe fn all_reduce(
    sendbuff: *const ::core::ffi::c_void,
    recvbuff: *mut ::core::ffi::c_void,
    count: usize,
    datatype: sys::ncclDataType_t,
    op: sys::ncclRedOp_t,
    comm: sys::ncclComm_t,
    stream: sys::cudaStream_t,
) -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclAllReduce(sendbuff, recvbuff, count, datatype, op, comm, stream).result() }
}

pub unsafe fn reduce_scatter(
    sendbuff: *const ::core::ffi::c_void,
    recvbuff: *mut ::core::ffi::c_void,
    recvcount: usize,
    datatype: sys::ncclDataType_t,
    op: sys::ncclRedOp_t,
    comm: sys::ncclComm_t,
    stream: sys::cudaStream_t,
) -> Result<NcclStatus, NcclError> {
    unsafe {
        sys::ncclReduceScatter(sendbuff, recvbuff, recvcount, datatype, op, comm, stream).result()
    }
}

pub unsafe fn all_gather(
    sendbuff: *const ::core::ffi::c_void,
    recvbuff: *mut ::core::ffi::c_void,
    sendcount: usize,
    datatype: sys::ncclDataType_t,
    comm: sys::ncclComm_t,
    stream: sys::cudaStream_t,
) -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclAllGather(sendbuff, recvbuff, sendcount, datatype, comm, stream).result() }
}

pub unsafe fn send(
    sendbuff: *const ::core::ffi::c_void,
    count: usize,
    datatype: sys::ncclDataType_t,
    peer: ::core::ffi::c_int,
    comm: sys::ncclComm_t,
    stream: sys::cudaStream_t,
) -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclSend(sendbuff, count, datatype, peer, comm, stream).result() }
}

pub unsafe fn recv(
    recvbuff: *mut ::core::ffi::c_void,
    count: usize,
    datatype: sys::ncclDataType_t,
    peer: ::core::ffi::c_int,
    comm: sys::ncclComm_t,
    stream: sys::cudaStream_t,
) -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclRecv(recvbuff, count, datatype, peer, comm, stream).result() }
}
pub unsafe fn group_end() -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclGroupEnd().result() }
}

pub unsafe fn group_start() -> Result<NcclStatus, NcclError> {
    unsafe { sys::ncclGroupStart().result() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::driver::CudaDevice;

    #[test]
    fn single_thread() {
        let n_devices = CudaDevice::count().unwrap() as usize;
        let n = 2;

        let mut devs = vec![];
        let mut sendslices = vec![];
        let mut recvslices = vec![];
        for i in 0..n_devices {
            let dev = CudaDevice::new(i).unwrap();
            let slice = dev.htod_copy(vec![(i + 1) as f32 * 1.0; n]).unwrap();
            sendslices.push(slice);
            let slice = dev.alloc_zeros::<f32>(n).unwrap();
            recvslices.push(slice);
            devs.push(dev);
        }
        // let devs: Vec<_> = (0..n_devices)
        //     .map(|i| CudaDevice::new(i).unwrap())
        //     .collect();
        // let sendslices: Vec<_> = devs
        //     .iter()
        //     .enumerate()
        //     .map(|(i, dev)| {
        //         let slice = dev.htod_copy(vec![(i + 1) as f32 * 1.0; n]).unwrap();
        //         slice
        //     })
        //     .collect();
        // let mut recvslices: Vec<_> = devs
        //     .iter()
        //     .map(|dev| {
        //         let slice = dev.alloc_zeros::<f32>(n).unwrap();
        //         slice
        //     })
        //     .collect();
        // let streams: Vec<crate::driver::safe::CudaStream> = devs
        //     .iter()
        //     .map(|dev| dev.fork_default_stream().unwrap())
        //     .collect();
        let mut comms = vec![std::ptr::null_mut(); n_devices];
        let ordinals: Vec<_> = devs.iter().map(|d| d.ordinal as i32).collect();
        // todo!("Comms {comms:?}");
        unsafe {
            comm_init_all(comms.as_mut_ptr(), n_devices as i32, ordinals.as_ptr()).unwrap();

            use std::ffi::c_void;
            group_start().unwrap();
            for i in 0..n_devices {
                let dev = CudaDevice::new(i).unwrap();
                println!(
                    "Addr {:?} - {:?}",
                    sendslices[i].cu_device_ptr, recvslices[i].cu_device_ptr
                );
                all_reduce(
                    sendslices[i].cu_device_ptr as *const c_void,
                    recvslices[i].cu_device_ptr as *mut c_void,
                    n,
                    sys::ncclDataType_t::ncclFloat32,
                    sys::ncclRedOp_t::ncclSum,
                    comms[i],
                    dev.stream as sys::cudaStream_t,
                    // streams[i].stream as sys::cudaStream_t,
                )
                .unwrap();
            }
            group_end().unwrap();
            devs[0].synchronize().unwrap();
        }
        // devs[0].synchronize();
        // recvslices[0].device.synchronize().unwrap();
        for (i, (recv, dev)) in recvslices.iter().zip(devs.iter()).enumerate() {
            let dev = CudaDevice::new(i).unwrap();
            let out = dev.dtoh_sync_copy(recv).unwrap();
            assert_eq!(out, vec![(n_devices * (n_devices + 1)) as f32 / 2.0; n]);
        }
    }
}
