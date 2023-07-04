use super::{result, sys};
use crate::driver::{CudaDevice, CudaSlice};
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::Arc;

#[derive(Debug)]
pub struct Comm {
    comm: sys::ncclComm_t,
    pub device: Arc<CudaDevice>,
}

#[derive(Debug)]
pub struct Id {
    id: sys::ncclUniqueId,
}

impl Id {
    pub fn new() -> Result<Self, result::NcclError> {
        let id = unsafe { result::get_uniqueid()? };
        Ok(Self { id })
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

pub fn init_device_comms(devices: Vec<Arc<CudaDevice>>) -> Result<Vec<Comm>, result::NcclError> {
    let mut comms = vec![std::ptr::null_mut() as *mut sys::ncclComm; devices.len()];
    let device_ordinals = devices
        .iter()
        .map(|x| x.ordinal as i32)
        .collect::<Vec<i32>>();
    let comms: Vec<_> = unsafe {
        result::comm_init_all(
            comms.as_mut_ptr(),
            devices.len() as i32,
            device_ordinals.as_ptr(),
        )?;
        comms
    };
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
    fn from_rank(device: Arc<CudaDevice>, ranks: usize, id: Id) -> Result<Self, result::NcclError> {
        let mut comm = MaybeUninit::uninit();
        unsafe {
            // result::group_start()?;
        }

        let rank = device.ordinal;
        println!("{rank}/{ranks} {id:?} {:?}", comm);

        let comm = unsafe {
            result::comm_init_rank(
                comm.as_mut_ptr(),
                ranks.try_into().expect("Ranks cannot be casted to i32"),
                id.id,
                rank.try_into().expect("Rank cannot be cast to i32"),
            )?;
            comm.assume_init()
        };
        println!("Comm {:?}", comm);
        unsafe {
            // result::group_end()?;
        }
        Ok(Self { comm, device })
    }
}

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
        root: i32,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_reduce() {
        let mut threads = vec![];
        let n = 2;
        println!("Entering test");
        let n_devices = CudaDevice::count().unwrap() as usize;
        for i in 0..n_devices {
            println!("Created id");
            threads.push(std::thread::spawn(move || {
                let id = Id::new().unwrap();
                let dev = CudaDevice::new(i).unwrap();

                println!("Creating comm {i}");
                let comm = Comm::from_rank(dev.clone(), n_devices, id).unwrap();
                println!("Created comm {i}");
                let slice = dev.htod_copy(vec![(i as f32) * 1.0; n]).unwrap();
                let mut slice_receive = dev.alloc_zeros::<f32>(n).unwrap();
                comm.all_reduce(&slice, &mut slice_receive, &ReduceOp::Sum)
                    .unwrap();

                let out = dev.dtoh_sync_copy(&slice_receive).unwrap();

                assert_eq!(out, &[3.0, 3.0]);
            }));
        }
        for t in threads {
            t.join().unwrap()
        }
    }

    // #[test]
    // fn test_all_reduce2() {
    //     let n_devices = 2;
    //     let devs: Vec<_> = (0..n_devices)
    //         .map(|i| CudaDevice::new(i).unwrap())
    //         .collect();
    //     let comms = init_device_comms(devs.clone()).unwrap();
    //     for i in 0..n_devices {
    //         // let dev = &devs[i];
    //         let dev = CudaDevice::new(i).unwrap();
    //         let comm = &comms[i];
    //         let slice = dev
    //             .htod_copy(vec![(i as f64) * 1.0, (i as f64) * 1.0])
    //             .unwrap();
    //         let mut slice_receive = dev.alloc_zeros::<f64>(2).unwrap();
    //         comm.all_reduce(&slice, &mut slice_receive, &ReduceOp::Sum)
    //             .unwrap();

    //         let out = dev.dtoh_sync_copy(&slice_receive).unwrap();

    //         assert_eq!(out, &[3.0, 3.0]);
    //     }
    // }
}
