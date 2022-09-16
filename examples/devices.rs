#![feature(generic_associated_types)]

use cudas::cuda::{
    borrow::{CudaDevice, InCudaMemory},
    result::CudaError,
};

trait Device {
    type Err;
    type InDeviceMemory<'device, T>
    where
        Self: 'device;

    fn take<T>(&self, on_host: Box<T>) -> Result<Self::InDeviceMemory<'_, T>, Self::Err>;
    fn release<T>(&self, on_device: Self::InDeviceMemory<'_, T>) -> Result<Box<T>, Self::Err>;
}

impl Device for CudaDevice {
    type Err = CudaError;
    type InDeviceMemory<'device, T> = InCudaMemory<'device, T>;

    fn take<T>(&self, on_host: Box<T>) -> Result<Self::InDeviceMemory<'_, T>, Self::Err> {
        self.take(on_host)
    }

    fn release<T>(&self, on_device: Self::InDeviceMemory<'_, T>) -> Result<Box<T>, Self::Err> {
        self.release(on_device)
    }
}

struct Cpu;

impl Device for Cpu {
    type Err = ();
    type InDeviceMemory<'device, T> = Box<T>;

    fn take<T>(&self, on_host: Box<T>) -> Result<Self::InDeviceMemory<'_, T>, Self::Err> {
        Ok(on_host)
    }

    fn release<T>(&self, on_device: Self::InDeviceMemory<'_, T>) -> Result<Box<T>, Self::Err> {
        Ok(on_device)
    }
}

struct Tensor1D<'device, const M: usize, D: 'device + Device = Cpu> {
    data: D::InDeviceMemory<'device, [f32; M]>,
    device: &'device D,
}

impl<'device, const M: usize> Tensor1D<'device, M, Cpu> {
    fn gpu(
        self,
        device: &'device CudaDevice,
    ) -> Result<Tensor1D<'device, M, CudaDevice>, CudaError> {
        let data = device.take(self.data)?;
        Ok(Tensor1D { data, device })
    }
}

impl<'device, const M: usize> Tensor1D<'device, M, CudaDevice> {
    fn cpu(self, device: &'device Cpu) -> Result<Tensor1D<'device, M, Cpu>, CudaError> {
        let data = self.device.release(self.data)?;
        Ok(Tensor1D { data, device })
    }
}

fn main() {
    let cpu = Cpu;
    let gpu = CudaDevice::new(0).unwrap();

    let t = Tensor1D {
        data: Box::new([1.0f32, 2.0, 3.0]),
        device: &cpu,
    };

    let t_gpu = t.gpu(&gpu).unwrap();

    let t_cpu = t_gpu.cpu(&cpu).unwrap();
    println!("{:?}", t_cpu.data);
}
