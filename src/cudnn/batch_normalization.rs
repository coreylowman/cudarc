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
    cudnn_handle: Rc<CudnnHandle>,
    x: Tensor4D<T, N, C, H, W>,
    dx: Tensor4D<T, N, C, H, W>,
    dy: Tensor4D<T, N, C, H, W>,
    dummy_diff: Tensor4D<T, 1, C, 1, 1>,
    saved_mean: Tensor4D<T, 1, C, 1, 1>,
    saved_variance: Tensor4D<T, 1, C, 1, 1>,
}
impl<T: TensorDataType, const N: usize, const C: usize, const H: usize, const W: usize>
    BatchNormalizationBackward<T, N, C, H, W>
{
    pub fn create(
        cudnn_handle: Rc<CudnnHandle>,
        x: Tensor4D<T, N, C, H, W>,
        dx: Tensor4D<T, N, C, H, W>,
        dy: Tensor4D<T, N, C, H, W>,
        dummy_diff: Tensor4D<T, 1, C, 1, 1>,
        saved_mean: Tensor4D<T, 1, C, 1, 1>,
        saved_variance: Tensor4D<T, 1, C, 1, 1>,
    ) -> Self {
        Self {
            cudnn_handle,
            x,
            dx,
            dy,
            dummy_diff,
            saved_mean,
            saved_variance,
        }
    }

    pub fn execute(&self) -> CudnnResult<()> {
        unsafe {
            cudnnBatchNormalizationBackward(
                self.cudnn_handle.0,
                MODE,
                &T::ONE as *const _ as *const _,
                &T::ZERO as *const _ as *const _,
                &T::ZERO as *const _ as *const _,
                &T::ONE as *const _ as *const _,
                self.x.descriptor.0,
                self.x.data.t_cuda.cu_device_ptr as *const _,
                self.dy.descriptor.0,
                self.dy.data.t_cuda.cu_device_ptr as *const _,
                self.dx.descriptor.0,
                self.dx.data.t_cuda.cu_device_ptr as *mut _,
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
    cudnn_handle: Rc<CudnnHandle>,
    x: Tensor4D<T, N, C, H, W>,
    y: Tensor4D<T, N, C, H, W>,
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
    pub fn create(
        device: &Rc<CudaDevice>,
        cudnn_handle: Rc<CudnnHandle>,
        x: Tensor4D<T, N, C, H, W>,
        y: Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<Self> {
        let bias = Tensor4D::alloc_all_same(device, &cudnn_handle, &T::ZERO)?;
        let scale = Tensor4D::alloc_all_same(device, &cudnn_handle, &T::ONE)?;
        let saved_mean = unsafe { Tensor4D::alloc_uninit(device) }?;
        let saved_variance = unsafe { Tensor4D::alloc_uninit(device) }?;
        let running_mean = Tensor4D::alloc_all_same(device, &cudnn_handle, &T::ZERO)?;
        let running_variance = Tensor4D::alloc_all_same(device, &cudnn_handle, &T::ONE)?;
        Ok(Self {
            cudnn_handle,
            x,
            y,
            bias,
            scale,
            saved_mean,
            saved_variance,
            running_mean,
            running_variance,
        })
    }

    pub fn get_backward(
        &self,
        dx: Tensor4D<T, N, C, H, W>,
        dy: Tensor4D<T, N, C, H, W>,
    ) -> BatchNormalizationBackward<T, N, C, H, W> {
        BatchNormalizationBackward::create(
            Rc::clone(&self.cudnn_handle),
            self.x.clone(),
            dx,
            dy,
            self.scale.clone(),
            self.saved_mean.clone(),
            self.saved_variance.clone(),
        )
    }

    pub fn execute(&self, exponential_average_factor: f64) -> CudnnResult<()> {
        unsafe {
            cudnnBatchNormalizationForwardTraining(
                self.cudnn_handle.0,
                MODE,
                &T::ONE as *const _ as *const _,
                &T::ZERO as *const _ as *const _,
                self.x.descriptor.0,
                self.x.data.t_cuda.cu_device_ptr as *const _,
                self.y.descriptor.0,
                self.y.data.t_cuda.cu_device_ptr as *mut _,
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
    use alloc::rc::Rc;

    use crate::prelude::*;

    #[test]
    fn test_batch_normalization() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let cudnn_handle = Rc::new(CudnnHandle::create(&device).unwrap());

        let x = Tensor4D::alloc_with(&device, [[[[2.0, 0.0]], [[-2.0, -6.0]]]]).unwrap();
        let dx = unsafe { Tensor4D::alloc_uninit(&device).unwrap() };
        let y = unsafe { Tensor4D::alloc_uninit(&device).unwrap() };

        let forward =
            BatchNormalizationForward::create(&device, cudnn_handle, x.clone(), y.clone()).unwrap();
        let backward = forward.get_backward(dx.clone(), y.clone());

        forward.execute(1.0).unwrap();

        assert_eq!(&*y.data.into_host().unwrap(), &[[[[1.0, -1.0]], [[
            1.0, -1.0
        ]]]]);

        backward.execute().unwrap();

        // TODO seems to be wrong
        std::println!("{:?}", dx.data.into_host());
    }
}
