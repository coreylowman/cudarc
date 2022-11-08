use core::mem::MaybeUninit;

use alloc::rc::Rc;

use crate::{prelude::*, driver::sys::{CUdeviceptr, cuMemAllocAsync, cuMemFreeAsync}};

pub trait RequiresAlgorithmWithWorkspace<A> {
    fn get_algorithm(&self) -> CudnnResult<A>;
    fn get_workspace_size(&self, algorithm: &A) -> CudnnResult<usize>;
    fn execute(&mut self, algorithm: &A, workspace_allocation: CUdeviceptr, workspace_size: usize) -> CudnnResult<()>;
}
pub struct AlgorithmWithWorkspace<A, T> {
    pub(crate) data: T,
    pub(crate) algorithm: A,
    pub(crate) workspace_allocation: CUdeviceptr,
    pub(crate) workspace_size: usize,
    pub(crate) device: Rc<CudaDevice>
}
impl<A, T: RequiresAlgorithmWithWorkspace<A>> AlgorithmWithWorkspace<A, T> {
    pub fn create(data: T, device: Rc<CudaDevice>) -> CudnnResult<Self> {
        let algorithm = data.get_algorithm()?;
        let workspace_size = data.get_workspace_size(&algorithm)?;
        let workspace_allocation = unsafe {
            let mut dev_ptr = MaybeUninit::uninit();
            // TODO other return type
            cuMemAllocAsync(dev_ptr.as_mut_ptr(), workspace_size, device.cu_stream).result().unwrap();
            dev_ptr.assume_init()
        };
        Ok(Self {data,algorithm, workspace_size, workspace_allocation, device})
    }

    pub fn execute(&mut self) -> CudnnResult<()> {
        self.data.execute(&self.algorithm, self.workspace_allocation, self.workspace_size)
    }
}
impl<A, T> Drop for AlgorithmWithWorkspace<A, T> {
    fn drop(&mut self) {
        unsafe { cuMemFreeAsync(self.workspace_allocation, self.device.cu_stream).result().unwrap() }
    }
}