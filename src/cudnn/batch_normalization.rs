use alloc::rc::Rc;

use super::sys::*;
use crate::prelude::*;

const MODE: cudnnBatchNormMode_t = cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

pub struct BatchNormalizationBackward<
    T,
    const N: usize,
    const C: usize,
    const H: usize,
    const W: usize,
> {
    dummy_diff:     Tensor4D<T, 1, C, 1, 1>,
    saved_mean:     Tensor4D<T, 1, C, 1, 1>,
    saved_variance: Tensor4D<T, 1, C, 1, 1>,
}
impl<T: TensorDataType, const N: usize, const C: usize, const H: usize, const W: usize>
    BatchNormalizationBackward<T, N, C, H, W>
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
                MODE,
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

        let forward = BatchNormalizationForward::create(&device, &cudnn_handle).unwrap();
        let backward = forward.get_backward();

        forward.execute(&cudnn_handle, &x, &mut y, 1.0).unwrap();

        assert_eq!(&*y.get_data().unwrap(), &[[[[1.0, -1.0]], [[1.0, -1.0]]]]);

        backward.execute(&cudnn_handle, &x, &mut dx, &y).unwrap();

        // TODO check if this is really right
        assert_eq!(&*dx.get_data().unwrap(), &[[[[0.0, 0.0]], [[0.0, 0.0]]]]);
    }
}
