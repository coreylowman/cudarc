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
    ) -> CudnnResult<()> {
        unsafe {
            cudnnBatchNormalizationBackward(
                cudnn_handle.0,
                MODE,
                &T::ONE as *const _ as *const _,
                &T::ZERO as *const _ as *const _,
                &T::ZERO as *const _ as *const _,
                &T::ONE as *const _ as *const _,
                x.descriptor.0,
                x.data.t_cuda.cu_device_ptr as *const _,
                dy.descriptor.0,
                dy.data.t_cuda.cu_device_ptr as *const _,
                dx.descriptor.0,
                dx.data.t_cuda.cu_device_ptr as *mut _,
                self.dummy_diff.descriptor.0,
                self.dummy_diff.data.t_cuda.cu_device_ptr as *const _,
                self.dummy_diff.data.t_cuda.cu_device_ptr as *mut _,
                self.dummy_diff.data.t_cuda.cu_device_ptr as *mut _,
                CUDNN_BN_MIN_EPSILON,
                self.saved_mean.data.t_cuda.cu_device_ptr as *const _,
                self.saved_variance.data.t_cuda.cu_device_ptr as *const _,
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
    ) -> CudnnResult<()> {
        unsafe {
            cudnnBatchNormalizationForwardTraining(
                cudnn_handle.0,
                MODE,
                &T::ONE as *const _ as *const _,
                &T::ZERO as *const _ as *const _,
                x.descriptor.0,
                x.data.t_cuda.cu_device_ptr as *const _,
                y.descriptor.0,
                y.data.t_cuda.cu_device_ptr as *mut _,
                self.scale.descriptor.0,
                self.scale.data.t_cuda.cu_device_ptr as *const _,
                self.bias.data.t_cuda.cu_device_ptr as *const _,
                exponential_average_factor,
                self.running_mean.data.t_cuda.cu_device_ptr as *mut _,
                self.running_variance.data.t_cuda.cu_device_ptr as *mut _,
                CUDNN_BN_MIN_EPSILON,
                self.saved_mean.data.t_cuda.cu_device_ptr as *mut _,
                self.saved_variance.data.t_cuda.cu_device_ptr as *mut _,
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

        assert_eq!(&*y.data.clone().into_host().unwrap(), &[[[[1.0, -1.0]], [
            [1.0, -1.0]
        ]]]);

        backward.execute(&cudnn_handle, &x, &mut dx, &y).unwrap();

        // TODO seems to be wrong
        std::println!("{:?}", dx.data.into_host());
    }
}
