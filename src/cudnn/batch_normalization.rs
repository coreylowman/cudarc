use alloc::rc::Rc;

use crate::cudnn::sys::*;
use crate::prelude::*;

/// Uses per image (after conv2d) normalization.
pub type BatchNormalizationPerImage<
    T,
    const N: usize,
    const C: usize,
    const H: usize,
    const W: usize,
> = BatchNormalization<T, N, C, H, W, true>;
/// Uses per activation normalization.
pub type BatchNormalizationPerActivation<
    T,
    const N: usize,
    const C: usize,
    const H: usize,
    const W: usize,
> = BatchNormalization<T, N, C, H, W, false>;

pub struct BatchNormalization<
    T,
    const N: usize,
    const C: usize,
    const H: usize,
    const W: usize,
    const PER_IMAGE: bool,
> {
    bias: Tensor4D<T, 1, C, 1, 1>,
    scale: Tensor4D<T, 1, C, 1, 1>,
    dummy_diff: Tensor4D<T, 1, C, 1, 1>,
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
    > BatchNormalization<T, N, C, H, W, PER_IMAGE>
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
            dummy_diff: scale.clone(),
            scale,
            saved_mean,
            saved_variance,
            running_mean,
            running_variance,
            forward_passes: 0,
        })
    }

    /// Batch normalization for training.
    pub fn forward_train(
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
    pub fn forward_inference(
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

    // removing this mut is ok if dummy_diff casts the immutable device pointer to a
    // mutable one
    pub fn backward(
        &mut self,
        cudnn_handle: &CudnnHandle,
        x: &Tensor4D<T, N, C, H, W>,
        dy: &Tensor4D<T, N, C, H, W>,
        dx: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()> {
        unsafe {
            cudnnBatchNormalizationBackward(
                cudnn_handle.get_handle(),
                if PER_IMAGE {
                    cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL_PERSISTENT
                } else {
                    cudnnBatchNormMode_t::CUDNN_BATCHNORM_PER_ACTIVATION
                },
                &T::ONE as *const _ as *const _,
                &T::ZERO as *const _ as *const _,
                &T::ZERO as *const _ as *const _,
                &T::ONE as *const _ as *const _,
                x.get_descriptor(),
                x.get_data_ptr(),
                dy.get_descriptor(),
                dy.get_data_ptr(),
                dx.get_descriptor(),
                dx.get_data_ptr_mut(),
                self.dummy_diff.get_descriptor(),
                self.dummy_diff.get_data_ptr(),
                self.dummy_diff.get_data_ptr_mut(),
                self.dummy_diff.get_data_ptr_mut(),
                CUDNN_BN_MIN_EPSILON,
                self.saved_mean.get_data_ptr(),
                self.saved_variance.get_data_ptr(),
            )
            .result()
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_batch_normalization() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let cudnn_handle = CudnnHandle::create(&device).unwrap();

        let x = Tensor4D::alloc_with(&device, [[[[2.0, 0.0]], [[-2.0, -6.0]]]]).unwrap();
        let mut dx = unsafe { Tensor4D::alloc_uninit(&device).unwrap() };
        let mut y = unsafe { Tensor4D::alloc_uninit(&device).unwrap() };

        let mut batch_norm = BatchNormalizationPerImage::create(&device, &cudnn_handle).unwrap();

        batch_norm.forward_train(&cudnn_handle, &x, &mut y).unwrap();

        assert_eq!(
            &*y.get_data().as_host().unwrap(),
            &[[[[1.0, -1.0]], [[1.0, -1.0]]]]
        );

        batch_norm.backward(&cudnn_handle, &x, &y, &mut dx).unwrap();

        // TODO check if this is really right
        assert_eq!(
            &*dx.get_data().as_host().unwrap(),
            &[[[[0.0, 0.0]], [[0.0, 0.0]]]]
        );
    }
}
