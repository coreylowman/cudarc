use core::mem::MaybeUninit;

use alloc::rc::Rc;

use crate::driver::sys::{cuMemAllocAsync, cuMemFreeAsync, CUdeviceptr};
use crate::prelude::*;

/// A trait to get an algorithm `A` and allocate a workspace for a convolution
/// operation (forward, backward, backward filter).
pub trait RequiresAlgorithmWithWorkspace<A> {
    type InputA;
    type InputB;
    type Output;

    fn get_algorithm(&self, cudnn_handle: &CudnnHandle) -> CudaCudnnResult<A>;
    fn get_workspace_size(
        &self,
        cudnn_handle: &CudnnHandle,
        algorithm: &A,
    ) -> CudaCudnnResult<usize>;
    #[allow(clippy::too_many_arguments)]
    /// Executes the convolution operation with the algorithm `A` from
    /// `get_algorithm` and a workspace of size `get_workspace_size`.
    fn execute(
        &mut self,
        cudnn_handle: &CudnnHandle,
        algorithm: &A,
        workspace_allocation: CUdeviceptr,
        workspace_size: usize,
        input_a: &Self::InputA,
        input_b: &Self::InputB,
        output: &mut Self::Output,
    ) -> CudaCudnnResult<()>;
}
/// A wrapper around a type `T` implementing `RequiresAlgorithmWithWorkspace`.
///
/// This frees the workspace memory when this is dropped.
pub struct AlgorithmWithWorkspace<A, T> {
    data: T,
    algorithm: A,
    workspace_allocation: CUdeviceptr,
    workspace_size: usize,
    device: Rc<CudaDevice>,
}
impl<A, T: RequiresAlgorithmWithWorkspace<A>> AlgorithmWithWorkspace<A, T> {
    /// Creates a new [AlgorithmWithWorkspace]. This allocates enough workspace
    /// for later use.
    pub fn create(
        cudnn_handle: &CudnnHandle,
        data: T,
        device: Rc<CudaDevice>,
    ) -> CudaCudnnResult<Self> {
        let algorithm = data.get_algorithm(cudnn_handle)?;
        let workspace_size = data.get_workspace_size(cudnn_handle, &algorithm)?;
        let workspace_allocation = unsafe {
            let mut dev_ptr = MaybeUninit::uninit();
            cuMemAllocAsync(dev_ptr.as_mut_ptr(), workspace_size, device.cu_stream).result()?;
            dev_ptr.assume_init()
        };
        Ok(Self {
            algorithm,
            data,
            workspace_size,
            workspace_allocation,
            device,
        })
    }

    /// Executes the convolution operation with the algorithm `A` from
    /// `get_algorithm` and a workspace of size `get_workspace_size`.
    pub fn execute(
        &mut self,
        cudnn_handle: &CudnnHandle,
        input_a: &T::InputA,
        input_b: &T::InputB,
        output: &mut T::Output,
    ) -> CudaCudnnResult<()> {
        self.data.execute(
            cudnn_handle,
            &self.algorithm,
            self.workspace_allocation,
            self.workspace_size,
            input_a,
            input_b,
            output,
        )
    }
}
impl<A, T> Drop for AlgorithmWithWorkspace<A, T> {
    fn drop(&mut self) {
        unsafe {
            cuMemFreeAsync(self.workspace_allocation, self.device.cu_stream)
                .result()
                .unwrap()
        }
    }
}
