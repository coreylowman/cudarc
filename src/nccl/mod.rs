//! [Comm] wraps around the [NCCL API](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html), via:
//! 
//! 1. Instantiate with [Comm::from_devices()] or [Comm::from_rank()]
//! 2. Peer to peer with [Comm::send()]/[Comm::recv()]
//! 3. Broadcast [Comm::broadcast()]/[Comm::broadcast_in_place()]
//! 4. Reduce: [Comm::reduce()]/[Comm::reduce_in_place()]
//! 5. Gather & Reduce [Comm::all_gather()]/[Comm::all_reduce()]/[Comm::all_reduce_in_place()]
//! 
//! Note that all above apis work with [crate::driver::DevicePtr]/[crate::driver::DevicePtrMut], so they
//! accept [crate::driver::CudaSlice], [crate::driver::CudaView], and [crate::driver::CudaViewMut].

pub mod result;
pub mod safe;
#[allow(warnings)]
pub mod sys;

pub use safe::*;
