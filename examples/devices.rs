#![feature(generic_associated_types)]

use cudarc::prelude::*;
use std::rc::Rc;

pub struct Cpu;

pub struct CpuRc<T> {
    data: Rc<T>,
    device: Rc<Cpu>,
}

trait DeviceRc<T> {
    type Device;
    fn device_ref(&self) -> &Rc<Self::Device>;
}

impl<T> DeviceRc<T> for CpuRc<T> {
    type Device = Cpu;
    fn device_ref(&self) -> &Rc<Self::Device> {
        &self.device
    }
}

impl<T> DeviceRc<T> for CudaRc<T> {
    type Device = CudaDevice;
    fn device_ref(&self) -> &Rc<Self::Device> {
        self.device()
    }
}

trait Device {
    type DeviceRc<T>: DeviceRc<T>;
}

impl Device for Cpu {
    type DeviceRc<T> = CpuRc<T>;
}

impl Device for CudaDevice {
    type DeviceRc<T> = CudaRc<T>;
}

trait CloneToDevice<T: Clone, D: Device> {
    type Err;
    fn to(&self, device: &Rc<D>) -> Result<D::DeviceRc<T>, Self::Err>;
}

impl<T: Clone> CloneToDevice<T, CudaDevice> for CpuRc<T> {
    type Err = CudaError;
    fn to(&self, device: &Rc<CudaDevice>) -> Result<CudaRc<T>, Self::Err> {
        device.take(self.data.clone())
    }
}

impl<T: Clone> CloneToDevice<T, Cpu> for CudaRc<T> {
    type Err = CudaError;
    fn to(&self, device: &Rc<Cpu>) -> Result<CpuRc<T>, Self::Err> {
        let data = self.clone().into_host()?;
        Ok(CpuRc {
            data,
            device: device.clone(),
        })
    }
}

struct Tensor1D<const M: usize, D: Device = Cpu> {
    rc: D::DeviceRc<[f32; M]>,
}

impl<const M: usize, Src: Device, Dst: Device> CloneToDevice<[f32; M], Dst> for Tensor1D<M, Src>
where
    Src::DeviceRc<[f32; M]>: CloneToDevice<[f32; M], Dst>,
{
    type Err = <Src::DeviceRc<[f32; M]> as CloneToDevice<[f32; M], Dst>>::Err;
    fn to(&self, device: &Rc<Dst>) -> Result<<Dst as Device>::DeviceRc<[f32; M]>, Self::Err> {
        self.rc.to(device)
    }
}

fn main() {
    let cpu = Rc::new(Cpu);
    let gpu = CudaDeviceBuilder::new(0).build().unwrap();

    let t: Tensor1D<3> = Tensor1D {
        rc: CpuRc {
            data: Rc::new([1.0, 2.0, 3.0]),
            device: cpu.clone(),
        },
    };
    let t_gpu = t.to(&gpu).unwrap();
    let t_cpu = t_gpu.to(&cpu).unwrap();
    println!("{:?}", t_cpu.data);
}
