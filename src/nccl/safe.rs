use super::{result, sys};
use crate::driver::{CudaContext, CudaStream, DevicePtr, DevicePtrMut};
use std::{mem::MaybeUninit, sync::Arc, vec, vec::Vec};

pub use result::{group_end, group_start};

#[derive(Debug)]
pub struct Comm {
    comm: sys::ncclComm_t,
    stream: Arc<CudaStream>,
    rank: usize,
    world_size: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Id {
    id: sys::ncclUniqueId,
}

impl Id {
    pub fn new() -> Result<Self, result::NcclError> {
        let id = result::get_uniqueid()?;
        Ok(Self { id })
    }

    pub fn uninit(internal: [::core::ffi::c_char; 128usize]) -> Self {
        let id = sys::ncclUniqueId { internal };
        Self { id }
    }

    pub fn internal(&self) -> &[::core::ffi::c_char; 128usize] {
        &self.id.internal
    }
}

pub enum ReduceOp {
    Sum,
    Prod,
    Max,
    Min,
    Avg,
}

fn convert_to_nccl_reduce_op(op: &ReduceOp) -> sys::ncclRedOp_t {
    match op {
        ReduceOp::Sum => sys::ncclRedOp_t::ncclSum,
        ReduceOp::Prod => sys::ncclRedOp_t::ncclProd,
        ReduceOp::Max => sys::ncclRedOp_t::ncclMax,
        ReduceOp::Min => sys::ncclRedOp_t::ncclMin,
        ReduceOp::Avg => sys::ncclRedOp_t::ncclAvg,
    }
}

impl Drop for Comm {
    fn drop(&mut self) {
        // TODO(thenerdstation): Shoule we instead do finalize then destory?
        unsafe {
            result::comm_abort(self.comm).expect("Error when aborting Comm.");
        }
    }
}

pub trait NcclType {
    fn as_nccl_type() -> sys::ncclDataType_t;
}

macro_rules! define_nccl_type {
    ($t:ty, $nccl_type:expr) => {
        impl NcclType for $t {
            fn as_nccl_type() -> sys::ncclDataType_t {
                $nccl_type
            }
        }
    };
}

define_nccl_type!(f32, sys::ncclDataType_t::ncclFloat32);
define_nccl_type!(f64, sys::ncclDataType_t::ncclFloat64);
define_nccl_type!(i8, sys::ncclDataType_t::ncclInt8);
define_nccl_type!(i32, sys::ncclDataType_t::ncclInt32);
define_nccl_type!(i64, sys::ncclDataType_t::ncclInt64);
define_nccl_type!(u8, sys::ncclDataType_t::ncclUint8);
define_nccl_type!(u32, sys::ncclDataType_t::ncclUint32);
define_nccl_type!(u64, sys::ncclDataType_t::ncclUint64);
define_nccl_type!(char, sys::ncclDataType_t::ncclUint8);
#[cfg(feature = "f16")]
define_nccl_type!(half::f16, sys::ncclDataType_t::ncclFloat16);
#[cfg(feature = "f16")]
define_nccl_type!(half::bf16, sys::ncclDataType_t::ncclBfloat16);
impl Comm {
    /// Primitive to create new communication link on a single thread.
    /// WARNING: You are likely to get limited throughput using a single core
    /// to control multiple GPUs (see issue #169).
    /// ```
    /// # use cudarc::driver::safe::{CudaDevice};
    /// # use cudarc::nccl::safe::{Comm, ReduceOp, group_start, group_end};
    /// let n = 2;
    /// let n_devices = CudaDevice::count().unwrap() as usize;
    /// let devices : Vec<_> = (0..n_devices).flat_map(CudaDevice::new).collect();
    /// let comms = Comm::from_devices(devices).unwrap();
    /// group_start().unwrap();
    /// (0..n_devices).map(|i| {
    ///     let comm = &comms[i];
    ///     let dev = comm.device();
    ///     let slice = dev.htod_copy(vec![(i + 1) as f32 * 1.0; n]).unwrap();
    ///     let mut slice_receive = dev.alloc_zeros::<f32>(n).unwrap();
    ///     comm.all_reduce(&slice, &mut slice_receive, &ReduceOp::Sum)
    ///         .unwrap();
    /// });
    /// group_start().unwrap();
    /// ```
    pub fn from_devices(streams: Vec<Arc<CudaStream>>) -> Result<Vec<Self>, result::NcclError> {
        let n_streams = streams.len();
        let mut comms = vec![std::ptr::null_mut(); n_streams];
        let ordinals: Vec<_> = streams
            .iter()
            .map(|d| d.context().ordinal() as i32)
            .collect();
        unsafe {
            result::comm_init_all(comms.as_mut_ptr(), n_streams as i32, ordinals.as_ptr())?;
        }

        let comms: Vec<Self> = comms
            .into_iter()
            .zip(streams.iter().cloned())
            .enumerate()
            .map(|(rank, (comm, stream))| Self {
                comm,
                stream,
                rank,
                world_size: n_streams,
            })
            .collect();

        Ok(comms)
    }

    pub fn stream(&self) -> Arc<CudaStream> {
        self.stream.clone()
    }

    pub fn context(&self) -> &Arc<CudaContext> {
        self.stream.context()
    }

    pub fn ordinal(&self) -> usize {
        self.stream.ctx.ordinal
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Primitive to create new communication link on each process (threads are possible but not
    /// recommended).
    ///
    /// WARNING: If using threads, uou are likely to get limited throughput using a single core
    /// to control multiple GPUs. Cuda drivers effectively use a global mutex thrashing
    /// performance on multi threaded multi GPU (see issue #169).
    /// ```
    /// # use cudarc::driver::safe::{CudaDevice};
    /// # use cudarc::nccl::safe::{Comm, Id, ReduceOp};
    /// let n = 2;
    /// let n_devices = 1; // This is to simplify this example.
    /// // Spawn this only on rank 0
    /// let id = Id::new().unwrap();
    /// // Send id.internal() to other ranks
    /// // let id = Id::uninit(id.internal().clone()); on other ranks
    ///
    /// let rank = 0;
    /// let dev = CudaDevice::new(rank).unwrap();
    /// let comm = Comm::from_rank(dev.clone(), rank, n_devices, id).unwrap();
    /// let slice = dev.htod_copy(vec![(rank + 1) as f32 * 1.0; n]).unwrap();
    /// let mut slice_receive = dev.alloc_zeros::<f32>(n).unwrap();
    /// comm.all_reduce(&slice, &mut slice_receive, &ReduceOp::Sum)
    ///     .unwrap();
    /// let out = dev.dtoh_sync_copy(&slice_receive).unwrap();
    /// assert_eq!(out, vec![(n_devices * (n_devices + 1)) as f32 / 2.0; n]);
    /// ```
    pub fn from_rank(
        stream: Arc<CudaStream>,
        rank: usize,
        world_size: usize,
        id: Id,
    ) -> Result<Self, result::NcclError> {
        let mut comm = MaybeUninit::uninit();

        let comm = unsafe {
            result::comm_init_rank(
                comm.as_mut_ptr(),
                world_size
                    .try_into()
                    .expect("World_size cannot be casted to i32"),
                id.id,
                rank.try_into().expect("Rank cannot be cast to i32"),
            )?;
            comm.assume_init()
        };
        Ok(Self {
            comm,
            stream,
            rank,
            world_size,
        })
    }
}

impl Comm {
    /// Send data to one peer, see [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclsend)
    pub fn send<S: DevicePtr<T>, T: NcclType>(
        &self,
        data: &S,
        peer: i32,
    ) -> Result<(), result::NcclError> {
        let (src, _record_src) = data.device_ptr(&self.stream);
        unsafe {
            result::send(
                src as _,
                data.len(),
                T::as_nccl_type(),
                peer,
                self.comm,
                self.stream.cu_stream as _,
            )
        }?;
        Ok(())
    }

    /// Receive data from one peer, see [cuda docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/api/p2p.html#ncclrecv)
    pub fn recv<R: DevicePtrMut<T>, T: NcclType>(
        &self,
        buff: &mut R,
        peer: i32,
    ) -> Result<result::NcclStatus, result::NcclError> {
        let count = buff.len();
        let (dst, _record_dst) = buff.device_ptr_mut(&self.stream);
        unsafe {
            result::recv(
                dst as _,
                count,
                T::as_nccl_type(),
                peer,
                self.comm,
                self.stream.cu_stream as _,
            )
        }
    }

    /// Broadcasts a value from `root` rank to every other ranks `recvbuff`.
    /// sendbuff is ignored on ranks other than `root`, so you can pass `None`
    /// on non-root ranks.
    ///
    /// sendbuff must be Some on root rank!
    ///
    /// See [nccl docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#broadcast)
    pub fn broadcast<S: DevicePtr<T>, R: DevicePtrMut<T>, T: NcclType>(
        &self,
        sendbuff: Option<&S>,
        recvbuff: &mut R,
        root: i32,
    ) -> Result<result::NcclStatus, result::NcclError> {
        debug_assert!(sendbuff.is_some() || self.rank != root as usize);
        let count = recvbuff.len();
        let (src, _record_src) = sendbuff.map(|b| b.device_ptr(&self.stream)).unzip();
        let (dst, _record_dst) = recvbuff.device_ptr_mut(&self.stream);
        unsafe {
            result::broadcast(
                src.map(|ptr| ptr as _).unwrap_or(std::ptr::null()),
                dst as _,
                count,
                T::as_nccl_type(),
                root,
                self.comm,
                self.stream.cu_stream as _,
            )
        }
    }

    /// In place version of [Comm::broadcast()].
    /// See [nccl docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#broadcast)
    pub fn broadcast_in_place<R: DevicePtrMut<T>, T: NcclType>(
        &self,
        recvbuff: &mut R,
        root: i32,
    ) -> Result<result::NcclStatus, result::NcclError> {
        let count = recvbuff.len();
        let (dst, _record_dst) = recvbuff.device_ptr_mut(&self.stream);
        unsafe {
            result::broadcast(
                dst as _,
                dst as _,
                count,
                T::as_nccl_type(),
                root,
                self.comm,
                self.stream.cu_stream as _,
            )
        }
    }

    /// See [nccl docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allgather)
    pub fn all_gather<S: DevicePtr<T>, R: DevicePtrMut<T>, T: NcclType>(
        &self,
        sendbuff: &S,
        recvbuff: &mut R,
    ) -> Result<result::NcclStatus, result::NcclError> {
        let (src, _record_src) = sendbuff.device_ptr(&self.stream);
        let (dst, _record_dst) = recvbuff.device_ptr_mut(&self.stream);
        unsafe {
            result::all_gather(
                src as _,
                dst as _,
                sendbuff.len(),
                T::as_nccl_type(),
                self.comm,
                self.stream.cu_stream as _,
            )
        }
    }

    /// See [nccl docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce)
    pub fn all_reduce<S: DevicePtr<T>, R: DevicePtrMut<T>, T: NcclType>(
        &self,
        sendbuff: &S,
        recvbuff: &mut R,
        reduce_op: &ReduceOp,
    ) -> Result<result::NcclStatus, result::NcclError> {
        let (src, _record_src) = sendbuff.device_ptr(&self.stream);
        let (dst, _record_dst) = recvbuff.device_ptr_mut(&self.stream);
        unsafe {
            result::all_reduce(
                src as _,
                dst as _,
                sendbuff.len(),
                T::as_nccl_type(),
                convert_to_nccl_reduce_op(reduce_op),
                self.comm,
                self.stream.cu_stream as _,
            )
        }
    }

    /// In place version of [Comm::all_reduce()].
    /// See [nccl docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#allreduce)
    pub fn all_reduce_in_place<R: DevicePtrMut<T>, T: NcclType>(
        &self,
        buff: &mut R,
        reduce_op: &ReduceOp,
    ) -> Result<result::NcclStatus, result::NcclError> {
        let count = buff.len();
        let (dst, _record_dst) = buff.device_ptr_mut(&self.stream);
        unsafe {
            result::all_reduce(
                dst as _,
                dst as _,
                count,
                T::as_nccl_type(),
                convert_to_nccl_reduce_op(reduce_op),
                self.comm,
                self.stream.cu_stream as _,
            )
        }
    }

    /// Reduces the sendbuff from all ranks into the recvbuff on the
    /// `root` rank.
    ///
    /// recvbuff must be Some on the root rank!
    ///
    /// See [nccl docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reduce)
    pub fn reduce<S: DevicePtr<T>, R: DevicePtrMut<T>, T: NcclType>(
        &self,
        sendbuff: &S,
        recvbuff: Option<&mut R>,
        reduce_op: &ReduceOp,
        root: i32,
    ) -> Result<result::NcclStatus, result::NcclError> {
        debug_assert!(recvbuff.is_some() || self.rank != root as usize);

        let (src, _record_src) = sendbuff.device_ptr(&self.stream);
        let (dst, _record_dst) = recvbuff.map(|b| b.device_ptr_mut(&self.stream)).unzip();
        unsafe {
            result::reduce(
                src as _,
                dst.map(|ptr| ptr as _).unwrap_or(std::ptr::null_mut()),
                sendbuff.len(),
                T::as_nccl_type(),
                convert_to_nccl_reduce_op(reduce_op),
                root,
                self.comm,
                self.stream.cu_stream as _,
            )
        }
    }

    /// In place version of [Comm::reduce()].
    /// See [nccl docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reduce)
    pub fn reduce_in_place<R: DevicePtrMut<T>, T: NcclType>(
        &self,
        recvbuff: &mut R,
        reduce_op: &ReduceOp,
        root: i32,
    ) -> Result<result::NcclStatus, result::NcclError> {
        let count = recvbuff.len();
        let (dst, _record_dst) = recvbuff.device_ptr_mut(&self.stream);
        unsafe {
            result::reduce(
                dst as _,
                dst as _,
                count,
                T::as_nccl_type(),
                convert_to_nccl_reduce_op(reduce_op),
                root,
                self.comm,
                self.stream.cu_stream as _,
            )
        }
    }

    /// See [nccl docs](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/collectives.html#reducescatter)
    pub fn reduce_scatter<S: DevicePtr<T>, R: DevicePtrMut<T>, T: NcclType>(
        &self,
        sendbuff: &S,
        recvbuff: &mut R,
        reduce_op: &ReduceOp,
    ) -> Result<result::NcclStatus, result::NcclError> {
        let count = recvbuff.len();
        let (src, _record_src) = sendbuff.device_ptr(&self.stream);
        let (dst, _record_dst) = recvbuff.device_ptr_mut(&self.stream);
        unsafe {
            result::reduce_scatter(
                src as _,
                dst as _,
                count,
                T::as_nccl_type(),
                convert_to_nccl_reduce_op(reduce_op),
                self.comm,
                self.stream.cu_stream as _,
            )
        }
    }
}

#[macro_export]
macro_rules! group {
    ($x:block) => {
        unsafe {
            result::group_start().unwrap();
        }
        $x
        unsafe {
            result::group_end().unwrap();
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    #[cfg(feature = "no-std")]
    use no_std_compat::println;

    #[test]
    fn test_all_reduce() {
        let n = 2;
        let n_devices = CudaContext::device_count().unwrap() as usize;
        let id = Id::new().unwrap();
        let threads: Vec<_> = (0..n_devices)
            .map(|i| {
                println!("III {i}");
                std::thread::spawn(move || {
                    println!("Within thread {i}");
                    let ctx = CudaContext::new(i).unwrap();
                    let stream = ctx.default_stream();
                    let comm = Comm::from_rank(stream.clone(), i, n_devices, id).unwrap();
                    let slice = stream.memcpy_stod(&vec![(i + 1) as f32 * 1.0; n]).unwrap();
                    let mut slice_receive = stream.alloc_zeros::<f32>(n).unwrap();
                    comm.all_reduce(&slice, &mut slice_receive, &ReduceOp::Sum)
                        .unwrap();

                    let out = stream.memcpy_dtov(&slice_receive).unwrap();

                    assert_eq!(out, vec![(n_devices * (n_devices + 1)) as f32 / 2.0; n]);
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap()
        }
    }

    #[test]
    fn test_all_reduce_views() {
        let n = 2;
        let n_devices = CudaContext::device_count().unwrap() as usize;
        let id = Id::new().unwrap();
        let threads: Vec<_> = (0..n_devices)
            .map(|i| {
                println!("III {i}");
                std::thread::spawn(move || {
                    println!("Within thread {i}");
                    let ctx = CudaContext::new(i).unwrap();
                    let stream = ctx.default_stream();
                    let comm = Comm::from_rank(stream.clone(), i, n_devices, id).unwrap();
                    let slice = stream.memcpy_stod(&vec![(i + 1) as f32 * 1.0; n]).unwrap();
                    let mut slice_receive = stream.alloc_zeros::<f32>(n).unwrap();
                    let slice_view = slice.slice(..);
                    let mut slice_receive_view = slice_receive.slice_mut(..);

                    comm.all_reduce(&slice_view, &mut slice_receive_view, &ReduceOp::Sum)
                        .unwrap();

                    let out = stream.memcpy_dtov(&slice_receive).unwrap();

                    assert_eq!(out, vec![(n_devices * (n_devices + 1)) as f32 / 2.0; n]);
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap()
        }
    }
}
