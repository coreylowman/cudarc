use super::{result, sys};
use crate::driver::{CudaDevice, CudaSlice};
use std::ptr;
use std::sync::Arc;

pub struct Comm {
    comm: sys::ncclComm_t,
    pub device: Arc<CudaDevice>,
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

pub fn init_device_comms(devices: Vec<Arc<CudaDevice>>) -> Result<Vec<Comm>, result::NcclError> {
    let mut comms = Vec::<sys::ncclComm_t>::with_capacity(devices.len());
    let device_ordinals = devices
        .iter()
        .map(|x| x.ordinal as i32)
        .collect::<Vec<i32>>();
    unsafe {
        result::comm_init_all(
            comms.as_mut_ptr(),
            devices.len() as i32,
            device_ordinals.as_ptr(),
        )?;
    }
    let zipped = comms.iter().zip(devices.iter());
    Ok(zipped
        .map(|(comm, device)| Comm {
            comm: *comm,
            device: device.clone(),
        })
        .collect())
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

impl Comm {
    pub fn send<T: NcclType>(
        &self,
        data: &CudaSlice<T>,
        peer: i32,
    ) -> Result<(), result::NcclError> {
        unsafe {
            result::send(
                data.cu_device_ptr as *mut _,
                data.len,
                T::as_nccl_type(),
                peer,
                self.comm,
                self.device.stream as *mut _,
            )?;
        }
        Ok(())
    }

    pub fn recv<T: NcclType>(
        &self,
        buff: &mut CudaSlice<T>,
        peer: i32,
    ) -> Result<result::NcclStatus, result::NcclError> {
        unsafe {
            result::recv(
                buff.cu_device_ptr as *mut _,
                buff.len,
                T::as_nccl_type(),
                peer,
                self.comm,
                self.device.stream as *mut _,
            )
        }
    }

    pub fn broadcast<T: NcclType>(
        &self,
        sendbuff: &Option<CudaSlice<T>>,
        recvbuff: &mut CudaSlice<T>,
        root: i32,
    ) -> Result<result::NcclStatus, result::NcclError> {
        unsafe {
            let send_ptr = match sendbuff {
                Some(buffer) => buffer.cu_device_ptr as *mut _,
                None => ptr::null(),
            };
            result::broadcast(
                send_ptr,
                recvbuff.cu_device_ptr as *mut _,
                recvbuff.len,
                T::as_nccl_type(),
                root,
                self.comm,
                self.device.stream as *mut _,
            )
        }
    }

    pub fn all_gather<T: NcclType>(
        &self,
        sendbuff: &CudaSlice<T>,
        recvbuff: &mut CudaSlice<T>,
    ) -> Result<result::NcclStatus, result::NcclError> {
        unsafe {
            result::all_gather(
                sendbuff.cu_device_ptr as *mut _,
                recvbuff.cu_device_ptr as *mut _,
                sendbuff.len,
                T::as_nccl_type(),
                self.comm,
                self.device.stream as *mut _,
            )
        }
    }

    pub fn all_reduce<T: NcclType>(
        &self,
        sendbuff: &CudaSlice<T>,
        recvbuff: &mut CudaSlice<T>,
        reduce_op: &ReduceOp,
    ) -> Result<result::NcclStatus, result::NcclError> {
        unsafe {
            result::all_reduce(
                sendbuff.cu_device_ptr as *mut _,
                recvbuff.cu_device_ptr as *mut _,
                sendbuff.len,
                T::as_nccl_type(),
                convert_to_nccl_reduce_op(&reduce_op),
                self.comm,
                self.device.stream as *mut _,
            )
        }
    }

    pub fn reduce<T: NcclType>(
        &self,
        sendbuff: &CudaSlice<T>,
        recvbuff: &mut CudaSlice<T>,
        reduce_op: &ReduceOp,
        root: i32
    ) -> Result<result::NcclStatus, result::NcclError> {
        unsafe {
            result::reduce(
                sendbuff.cu_device_ptr as *mut _,
                recvbuff.cu_device_ptr as *mut _,
                sendbuff.len,
                T::as_nccl_type(),
                convert_to_nccl_reduce_op(&reduce_op),
                root,
                self.comm,
                self.device.stream as *mut _,
            )
        }
    }

    pub fn reduce_scatter<T: NcclType>(
        &self,
        sendbuff: &CudaSlice<T>,
        recvbuff: &mut CudaSlice<T>,
        reduce_op: &ReduceOp,
    ) -> Result<result::NcclStatus, result::NcclError> {
        unsafe {
            result::reduce_scatter(
                sendbuff.cu_device_ptr as *mut _,
                recvbuff.cu_device_ptr as *mut _,
                recvbuff.len,
                T::as_nccl_type(),
                convert_to_nccl_reduce_op(&reduce_op),
                self.comm,
                self.device.stream as *mut _,
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
