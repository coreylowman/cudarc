#![feature(generic_associated_types)]

use cudas::cuda::{
    refcount::{CudaDevice, CudaRc},
    result::CudaError,
};
use std::rc::Rc;

trait DeviceAlloc<T> {
    type Device;
    fn device(&self) -> &Rc<Self::Device>;
}

trait ToDevice<T, D>: DeviceAlloc<T> {
    type Err;
    type DeviceAlloc: DeviceAlloc<T, Device = D>;
    fn to(&self, device: &Rc<D>) -> Result<Self::DeviceAlloc, Self::Err>;
}

pub struct Cpu;

pub struct CpuAlloc<T> {
    data: Rc<T>,
    device: Rc<Cpu>,
}

impl<T> DeviceAlloc<T> for CpuAlloc<T> {
    type Device = Cpu;
    fn device(&self) -> &Rc<Self::Device> {
        &self.device
    }
}

impl<T> ToDevice<T, CudaDevice> for CpuAlloc<T> {
    type Err = CudaError;
    type DeviceAlloc = CudaRc<T>;
    fn to(&self, device: &Rc<CudaDevice>) -> Result<Self::DeviceAlloc, Self::Err> {
        let mut alloc = device.alloc()?;
        device.clone_from_host(&mut alloc, self.data.as_ref())?;
        Ok(alloc)
    }
}

impl<T> DeviceAlloc<T> for CudaRc<T> {
    type Device = CudaDevice;
    fn device(&self) -> &Rc<Self::Device> {
        &self.device
    }
}

struct Tensor1D<const M: usize, D: DeviceAlloc<[f32; M]> = CpuAlloc<[f32; M]>> {
    data: Rc<D>,
}

fn main() {
    let cpu = Rc::new(Cpu);
    let gpu = CudaDevice::new(0).unwrap();

    let t = Tensor1D {
        data: Rc::new(CpuAlloc {
            data: Box::new([1.0f32, 2.0, 3.0]),
            device: cpu,
        }),
    };

    let mut q = Rc::new(gpu.alloc::<[f32; 3]>().unwrap());
    // let r = q.clone();
    let t = Rc::make_mut(&mut q);

    // let t_gpu = t.gpu(&gpu).unwrap();

    // let t_cpu = t_gpu.cpu(&cpu).unwrap();
    // println!("{:?}", t_cpu.data);
}
