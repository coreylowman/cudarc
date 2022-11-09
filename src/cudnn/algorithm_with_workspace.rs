use alloc::rc::Rc;

use crate::driver::sys::CUdeviceptr;
use crate::prelude::*;

pub trait RequiresAlgorithmWithWorkspace<A> {
    fn get_algorithm(&self) -> CudnnResult<A>;
    fn get_workspace_size(&self, algorithm: &A) -> CudnnResult<usize>;
    fn execute(
        &mut self,
        algorithm: &A,
        workspace_allocation: CUdeviceptr,
        workspace_size: usize,
    ) -> CudnnResult<()>;
}
pub struct AlgorithmWithWorkspace<A, T> {
    pub(crate) data:      T,
    pub(crate) algorithm: A,
}
impl<A, T: RequiresAlgorithmWithWorkspace<A>> AlgorithmWithWorkspace<A, T> {
    pub fn create(data: T, device: Rc<CudaDevice>) -> CudnnResult<WithWorkspace<Self>> {
        WithWorkspace::create(
            Self {
                algorithm: data.get_algorithm()?,
                data,
            },
            device,
        )
    }
}
impl<A, T: RequiresAlgorithmWithWorkspace<A>> RequiresWorkspace for AlgorithmWithWorkspace<A, T> {
    fn get_workspace_size(&self) -> CudnnResult<usize> {
        self.data.get_workspace_size(&self.algorithm)
    }

    fn execute(
        &mut self,
        workspace_allocation: CUdeviceptr,
        workspace_size: usize,
    ) -> CudnnResult<()> {
        self.data
            .execute(&self.algorithm, workspace_allocation, workspace_size)
    }
}
