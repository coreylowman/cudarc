const fn get_mode<const IS_MAX: bool>() -> cudnnPoolingMode_t {
    if IS_MAX {
        cudnnPoolingMode_t::CUDNN_POOLING_MAX
    } else {
        cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
    }
}

use core::mem::MaybeUninit;

use crate::cudnn::sys::*;
use crate::prelude::*;

pub struct Pooling2D<
    const F_H: usize,
    const F_W: usize,
    const P_H: usize,
    const P_W: usize,
    const S_H: usize,
    const S_W: usize,
    const IS_MAX: bool,
>(cudnnPoolingDescriptor_t);

impl<
        const F_H: usize,
        const F_W: usize,
        const P_H: usize,
        const P_W: usize,
        const S_H: usize,
        const S_W: usize,
        const IS_MAX: bool,
    > Pooling2D<F_H, F_W, P_H, P_W, S_H, S_W, IS_MAX>
{
    pub fn create() -> CudaCudnnResult<Self> {
        Ok(Self(unsafe {
            let mut descriptor = MaybeUninit::uninit();
            cudnnCreatePoolingDescriptor(descriptor.as_mut_ptr()).result()?;
            let descriptor = descriptor.assume_init();
            cudnnSetPooling2dDescriptor(
                descriptor,
                get_mode::<IS_MAX>(),
                cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
                F_H as _,
                F_W as _,
                P_H as _,
                P_W as _,
                S_H as _,
                S_W as _,
            )
            .result()?;
            descriptor
        }))
    }

    pub fn get_descriptor(&self) -> cudnnPoolingDescriptor_t {
        self.0
    }

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
        y: &mut Tensor4D<
            T,
            N,
            C,
            { ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE },
            { ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE },
        >,
    ) -> CudaCudnnResult<()>
    where
        [(); ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE]:,
        [(); ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE]:,
    {
        unsafe {
            cudnnPoolingForward(
                cudnn_handle.get_handle(),
                self.0,
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
        x: &Tensor4D<T, N, C, H, W>,
        y: &Tensor4D<
            T,
            N,
            C,
            { ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE },
            { ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE },
        >,
        dy: &Tensor4D<
            T,
            N,
            C,
            { ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE },
            { ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE },
        >,
        dx: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudaCudnnResult<()>
    where
        [(); ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE]:,
        [(); ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE]:,
    {
        unsafe {
            cudnnPoolingBackward(
                cudnn_handle.get_handle(),
                self.0,
                &T::ONE as *const _ as *const _,
                y.get_descriptor(),
                y.get_data_ptr(),
                dy.get_descriptor(),
                dy.get_data_ptr(),
                x.get_descriptor(),
                x.get_data_ptr(),
                &T::ZERO as *const _ as *const _,
                dx.get_descriptor(),
                dx.get_data_ptr_mut(),
            )
        }
        .result()
    }
}
impl<
        const F_H: usize,
        const F_W: usize,
        const P_H: usize,
        const P_W: usize,
        const S_H: usize,
        const S_W: usize,
        const IS_MAX: bool,
    > Drop for Pooling2D<F_H, F_W, P_H, P_W, S_H, S_W, IS_MAX>
{
    fn drop(&mut self) {
        unsafe { cudnnDestroyPoolingDescriptor(self.0) }
            .result()
            .unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_pooling() {
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let cudnn_handle = CudnnHandle::create(&device).unwrap();

        let input = Tensor4D::alloc_with(&device, [[[[1.0f64, 2.0, 3.0], [-1.0, 0.0, -1.0], [
            0.0, 3.0, 5.0,
        ]]]])
        .unwrap();
        let mut output = unsafe { Tensor4D::alloc_uninit(&device) }.unwrap();
        let mut dx = unsafe { Tensor4D::alloc_uninit(&device) }.unwrap();

        let pooling2d = Pooling2D::<2, 2, 0, 0, 1, 1, true>::create().unwrap();

        pooling2d
            .forward(&cudnn_handle, &input, &mut output)
            .unwrap();

        assert_eq!(output.get_data().as_host().unwrap()[0][0], [[2.0, 3.0], [
            3.0, 5.0
        ]]);

        pooling2d
            .backward(&cudnn_handle, &input, &output, &output, &mut dx)
            .unwrap();

        assert_eq!(dx.get_data().as_host().unwrap()[0][0], [
            [0.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 3.0, 5.0]
        ]);
    }
}
