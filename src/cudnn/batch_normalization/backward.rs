use crate::cudnn::sys::*;
use crate::prelude::*;

pub struct BatchNormalizationBackward<
    T,
    const N: usize,
    const C: usize,
    const H: usize,
    const W: usize,
    const PER_IMAGE: bool,
> {
    dummy_diff:     Tensor4D<T, 1, C, 1, 1>,
    saved_mean:     Tensor4D<T, 1, C, 1, 1>,
    saved_variance: Tensor4D<T, 1, C, 1, 1>,
}
impl<
        T: TensorDataType,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
        const PER_IMAGE: bool,
    > BatchNormalizationBackward<T, N, C, H, W, PER_IMAGE>
{
    pub fn create(
        dummy_diff: Tensor4D<T, 1, C, 1, 1>,
        saved_mean: Tensor4D<T, 1, C, 1, 1>,
        saved_variance: Tensor4D<T, 1, C, 1, 1>,
    ) -> Self {
        Self {
            dummy_diff,
            saved_mean,
            saved_variance,
        }
    }

    pub fn execute(
        &self,
        cudnn_handle: &CudnnHandle,
        x: &Tensor4D<T, N, C, H, W>,
        dx: &mut Tensor4D<T, N, C, H, W>,
        dy: &Tensor4D<T, N, C, H, W>,
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
