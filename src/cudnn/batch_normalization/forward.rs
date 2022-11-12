use alloc::rc::Rc;

use crate::cudnn::sys::*;
use crate::prelude::*;

pub struct BatchNormalizationForward<
    T,
    const N: usize,
    const C: usize,
    const H: usize,
    const W: usize,
> {
    bias: Tensor4D<T, 1, C, 1, 1>,
    scale: Tensor4D<T, 1, C, 1, 1>,
    saved_mean: Tensor4D<T, 1, C, 1, 1>,
    saved_variance: Tensor4D<T, 1, C, 1, 1>,
    running_mean: Tensor4D<T, 1, C, 1, 1>,
    running_variance: Tensor4D<T, 1, C, 1, 1>,
}
impl<T: TensorDataType, const N: usize, const C: usize, const H: usize, const W: usize>
    BatchNormalizationForward<T, N, C, H, W>
{
    pub fn create(device: &Rc<CudaDevice>, cudnn_handle: &CudnnHandle) -> CudaCudnnResult<Self> {
        let bias = Tensor4D::alloc_all_same(device, cudnn_handle, &T::ZERO)?;
        let scale = Tensor4D::alloc_all_same(device, cudnn_handle, &T::ONE)?;
        let saved_mean = unsafe { Tensor4D::alloc_uninit(device) }?;
        let saved_variance = unsafe { Tensor4D::alloc_uninit(device) }?;
        let running_mean = Tensor4D::alloc_all_same(device, cudnn_handle, &T::ZERO)?;
        let running_variance = Tensor4D::alloc_all_same(device, cudnn_handle, &T::ONE)?;
        Ok(Self {
            bias,
            scale,
            saved_mean,
            saved_variance,
            running_mean,
            running_variance,
        })
    }

    pub fn get_backward(&self) -> BatchNormalizationBackward<T, N, C, H, W> {
        BatchNormalizationBackward::create(
            self.scale.clone(),
            self.saved_mean.clone(),
            self.saved_variance.clone(),
        )
    }

    pub fn execute(
        &self,
        cudnn_handle: &CudnnHandle,
        x: &Tensor4D<T, N, C, H, W>,
        y: &mut Tensor4D<T, N, C, H, W>,
        exponential_average_factor: f64,
    ) -> CudaCudnnResult<()> {
        unsafe {
            cudnnBatchNormalizationForwardTraining(
                cudnn_handle.get_handle(),
                MODE,
                &T::ONE as *const _ as *const _,
                &T::ZERO as *const _ as *const _,
                x.get_descriptor(),
                x.get_data_ptr(),
                y.get_descriptor(),
                y.get_data_ptr_mut(),
                self.scale.get_descriptor(),
                self.scale.get_data_ptr(),
                self.bias.get_data_ptr(),
                exponential_average_factor,
                self.running_mean.get_data_ptr_mut(),
                self.running_variance.get_data_ptr_mut(),
                CUDNN_BN_MIN_EPSILON,
                self.saved_mean.get_data_ptr_mut(),
                self.saved_variance.get_data_ptr_mut(),
            )
            .result()
        }
    }
}
