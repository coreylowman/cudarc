use alloc::rc::Rc;

use crate::cudnn::sys::*;
use crate::prelude::*;

/// Uses per image (after conv2d) normalization.
pub type BatchNormalizationForwardPerImage<
    T,
    const N: usize,
    const C: usize,
    const H: usize,
    const W: usize,
> = BatchNormalizationForward<T, N, C, H, W, true>;
/// Uses per activation normalization.
pub type BatchNormalizationForwardPerActivation<
    T,
    const N: usize,
    const C: usize,
    const H: usize,
    const W: usize,
> = BatchNormalizationForward<T, N, C, H, W, false>;

pub struct BatchNormalizationForward<
    T,
    const N: usize,
    const C: usize,
    const H: usize,
    const W: usize,
    const PER_IMAGE: bool,
> {
    bias: Tensor4D<T, 1, C, 1, 1>,
    scale: Tensor4D<T, 1, C, 1, 1>,
    saved_mean: Tensor4D<T, 1, C, 1, 1>,
    saved_variance: Tensor4D<T, 1, C, 1, 1>,
    running_mean: Tensor4D<T, 1, C, 1, 1>,
    running_variance: Tensor4D<T, 1, C, 1, 1>,
    forward_passes: usize,
}
impl<
        T: TensorDataType,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
        const PER_IMAGE: bool,
    > BatchNormalizationForward<T, N, C, H, W, PER_IMAGE>
{
    pub fn create(device: &Rc<CudaDevice>, cudnn_handle: &CudnnHandle) -> CudaCudnnResult<Self> {
        let bias = Tensor4D::alloc_all_same(device, cudnn_handle, &T::ZERO)?;
        let scale = Tensor4D::alloc_all_same(device, cudnn_handle, &T::ONE)?;
        let saved_mean = unsafe { Tensor4D::alloc_uninit(device) }?;
        let saved_variance = unsafe { Tensor4D::alloc_uninit(device) }?;
        // change this if bias and scale get overlapping tensors
        let running_mean = bias.clone_into_new(device)?;
        let running_variance = scale.clone_into_new(device)?;
        Ok(Self {
            bias,
            scale,
            saved_mean,
            saved_variance,
            running_mean,
            running_variance,
            forward_passes: 0,
        })
    }

    /// Creates a [BatchNormalizationBackward] that shares the `saved_mean` and
    /// `saved_variance`.
    pub fn get_backward(&self) -> BatchNormalizationBackward<T, N, C, H, W, PER_IMAGE> {
        BatchNormalizationBackward::create(
            self.scale.clone(),
            self.saved_mean.clone(),
            self.saved_variance.clone(),
        )
    }

    /// Batch normalization for training.
    pub fn train(
        &mut self,
        cudnn_handle: &CudnnHandle,
        x: &Tensor4D<T, N, C, H, W>,
        y: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()> {
        self.forward_passes += 1;
        unsafe {
            cudnnBatchNormalizationForwardTraining(
                cudnn_handle.get_handle(),
                if PER_IMAGE {
                    cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL_PERSISTENT
                } else {
                    cudnnBatchNormMode_t::CUDNN_BATCHNORM_PER_ACTIVATION
                },
                &T::ONE as *const _ as *const _,
                &T::ZERO as *const _ as *const _,
                x.get_descriptor(),
                x.get_data_ptr(),
                y.get_descriptor(),
                y.get_data_ptr_mut(),
                self.scale.get_descriptor(),
                self.scale.get_data_ptr(),
                self.bias.get_data_ptr(),
                (self.forward_passes as f64).recip(),
                self.running_mean.get_data_ptr_mut(),
                self.running_variance.get_data_ptr_mut(),
                CUDNN_BN_MIN_EPSILON,
                self.saved_mean.get_data_ptr_mut(),
                self.saved_variance.get_data_ptr_mut(),
            )
            .result()
        }
    }

    /// Batch normalization for inference.
    pub fn inference(
        &self,
        cudnn_handle: &CudnnHandle,
        x: &Tensor4D<T, N, C, H, W>,
        y: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()> {
        unsafe {
            cudnnBatchNormalizationForwardInference(
                cudnn_handle.get_handle(),
                if PER_IMAGE {
                    cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL_PERSISTENT
                } else {
                    cudnnBatchNormMode_t::CUDNN_BATCHNORM_PER_ACTIVATION
                },
                &T::ONE as *const _ as *const _,
                &T::ZERO as *const _ as *const _,
                x.get_descriptor(),
                x.get_data_ptr(),
                y.get_descriptor(),
                y.get_data_ptr_mut(),
                self.scale.get_descriptor(),
                self.scale.get_data_ptr(),
                self.bias.get_data_ptr(),
                self.running_mean.get_data_ptr(),
                self.running_variance.get_data_ptr(),
                CUDNN_BN_MIN_EPSILON,
            )
            .result()
        }
    }
}
