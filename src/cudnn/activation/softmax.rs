use crate::cudnn::sys::*;
use crate::prelude::*;

/// This does the softmax activation per image.
pub struct Softmax;

impl Softmax {
    pub fn forward<
        T: TensorDataType,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
    >(
        &self,
        cudnn_handle: &CudnnHandle,
        x: &Tensor4D<T, N, C, H, W>,
        y: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()> {
        unsafe {
            cudnnSoftmaxForward(
                cudnn_handle.get_handle(),
                cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST,
                cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
                &T::ONE as *const _ as *const _,
                x.get_descriptor(),
                x.get_data_ptr(),
                &T::ZERO as *const _ as *const _,
                y.get_descriptor(),
                y.get_data_ptr_mut(),
            )
        }
        .result()
    }

    pub fn backward<
        T: TensorDataType,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
    >(
        &self,
        cudnn_handle: &CudnnHandle,
        y: &Tensor4D<T, N, C, H, W>,
        dy: &Tensor4D<T, N, C, H, W>,
        dx: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()> {
        unsafe {
            cudnnSoftmaxBackward(
                cudnn_handle.get_handle(),
                cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST,
                cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_CHANNEL,
                &T::ONE as *const _ as *const _,
                y.get_descriptor(),
                y.get_data_ptr(),
                dy.get_descriptor(),
                dy.get_data_ptr(),
                &T::ZERO as *const _ as *const _,
                dx.get_descriptor(),
                dx.get_data_ptr_mut(),
            )
        }
        .result()
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_softmax() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let cudnn_handle = CudnnHandle::create(&device).unwrap();

        let mut input_allocation = device.alloc_zeros().unwrap();
        CudaRng::new(0, device.clone())
            .unwrap()
            .fill_with_normal(&mut input_allocation, 0.0, 1.0)
            .unwrap();
        let input = Tensor4D::<f64, 2, 5, 2, 1>::create_with(input_allocation).unwrap();
        let mut output = unsafe { Tensor4D::alloc_uninit(&device) }.unwrap();

        Softmax.forward(&cudnn_handle, &input, &mut output).unwrap();
        let out = output.get_data().as_host().unwrap();
        for channel in out.into_iter() {
            assert!(
                (channel
                    .into_iter()
                    .flatten()
                    .flatten()
                    .reduce(|a, b| a + b)
                    .unwrap()
                    - 1.0)
                    .abs()
                    < 0.0001
            );
        }
    }
}
