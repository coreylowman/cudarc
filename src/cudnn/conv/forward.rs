use core::mem::MaybeUninit;

use alloc::rc::Rc;

use super::super::sys::*;
use crate::prelude::*;

/// A struct that holds all the data to calculate `y` by the filter and `x`.
pub struct Convolution2DForward<
    T,
    const H: usize,
    const W: usize,
    const P_H: usize,
    const P_W: usize,
    const V_S: usize,
    const H_S: usize,
    const N: usize,
    const C_IN: usize,
    const C_OUT: usize,
    const F_H: usize,
    const F_W: usize,
    const S_H: usize,
    const S_W: usize,
> where
    [(); ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE]:,
    [(); ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE]:,
{
    descriptor: Rc<ConvolutionDescriptor>,
    x: Tensor4D<T, N, C_IN, H, W>,
    filter: Filter<T, C_OUT, C_IN, F_H, F_W>,
    y: Tensor4D<
        T,
        N,
        C_OUT,
        { ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE },
        { ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE },
    >,
    cudnn_handle: Rc<CudnnHandle>,
}
impl<
        T: TensorDataType,
        const H: usize,
        const W: usize,
        const P_H: usize,
        const P_W: usize,
        const V_S: usize,
        const H_S: usize,
        const N: usize,
        const C_IN: usize,
        const C_OUT: usize,
        const F_H: usize,
        const F_W: usize,
        const S_H: usize,
        const S_W: usize,
    > Convolution2DForward<T, H, W, P_H, P_W, V_S, H_S, N, C_IN, C_OUT, F_H, F_W, S_H, S_W>
where
    [(); ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE]:,
    [(); ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE]:,
{
    pub fn create(
        cudnn_handle: Rc<CudnnHandle>,
        x: Tensor4D<T, N, C_IN, H, W>,
        filter: Filter<T, C_OUT, C_IN, F_H, F_W>,
        y: Tensor4D<
            T,
            N,
            C_OUT,
            { ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE },
            { ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE },
        >,
    ) -> CudnnResult<Self> {
        let descriptor = Rc::new(ConvolutionDescriptor::create()?);
        unsafe {
            cudnnSetConvolution2dDescriptor(
                descriptor.0,
                P_H as _,
                P_W as _,
                S_H as _,
                S_W as _,
                1,
                1,
                cudnnConvolutionMode_t::CUDNN_CONVOLUTION,
                T::get_data_type(),
            )
        }
        .result()?;
        Ok(Self {
            cudnn_handle,
            descriptor,
            x,
            filter,
            y,
        })
    }

    pub fn get_backward(
        &self,
        dy: Tensor4D<
            T,
            N,
            C_OUT,
            { ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE },
            { ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE },
        >,
        dx: Tensor4D<T, N, C_IN, H, W>,
    ) -> Convolution2DBackward<T, H, W, P_H, P_W, V_S, H_S, N, C_IN, C_OUT, F_H, F_W, S_H, S_W>
    {
        Convolution2DBackward::create(
            self.cudnn_handle.clone(),
            self.descriptor.clone(),
            dy,
            self.filter.clone(),
            dx,
        )
    }

    pub fn get_filter_backward(
        &self,
        dy: Tensor4D<
            T,
            N,
            C_OUT,
            { ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE },
            { ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE },
        >,
        dw: Filter<T, C_OUT, C_IN, F_H, F_W>,
    ) -> Convolution2DBackwardFilter<T, H, W, P_H, P_W, V_S, H_S, N, C_IN, C_OUT, F_H, F_W, S_H, S_W>
    {
        Convolution2DBackwardFilter::create(
            self.cudnn_handle.clone(),
            self.descriptor.clone(),
            self.filter.clone(),
            self.x.clone(),
            self.y.clone(),
            dy,
            dw,
        )
    }
}
impl<
        T: TensorDataType,
        const H: usize,
        const W: usize,
        const P_H: usize,
        const P_W: usize,
        const V_S: usize,
        const H_S: usize,
        const N: usize,
        const C_IN: usize,
        const C_OUT: usize,
        const F_H: usize,
        const F_W: usize,
        const S_H: usize,
        const S_W: usize,
    > RequiresAlgorithmWithWorkspace<cudnnConvolutionFwdAlgoPerf_t>
    for Convolution2DForward<T, H, W, P_H, P_W, V_S, H_S, N, C_IN, C_OUT, F_H, F_W, S_H, S_W>
where
    [(); ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE]:,
    [(); ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE]:,
{
    fn get_workspace_size(&self, algorithm: &cudnnConvolutionFwdAlgoPerf_t) -> CudnnResult<usize> {
        let mut workspace_size = MaybeUninit::uninit();
        unsafe {
            cudnnGetConvolutionForwardWorkspaceSize(
                self.cudnn_handle.0,
                self.x.descriptor.0,
                self.filter.descriptor.0,
                self.descriptor.0,
                self.y.descriptor.0,
                algorithm.algo,
                workspace_size.as_mut_ptr(),
            )
            .result()?;
            Ok(workspace_size.assume_init())
        }
    }

    fn get_algorithm(&self) -> CudnnResult<cudnnConvolutionFwdAlgoPerf_t> {
        let mut output_amount = MaybeUninit::uninit();
        let mut algorithm = MaybeUninit::uninit();
        unsafe {
            cudnnGetConvolutionForwardAlgorithm_v7(
                self.cudnn_handle.0,
                self.x.descriptor.0,
                self.filter.descriptor.0,
                self.descriptor.0,
                self.y.descriptor.0,
                1,
                output_amount.as_mut_ptr(),
                algorithm.as_mut_ptr(),
            )
            .result()?;
            assert_eq!(
                output_amount.assume_init(),
                1,
                "cudnnGetConvolutionForwardAlgorithm_v7 returned 0 algorithms"
            );
            Ok(algorithm.assume_init())
        }
    }

    fn execute(
        &mut self,
        algorithm: &cudnnConvolutionFwdAlgoPerf_t,
        workspace_allocation: crate::driver::sys::CUdeviceptr,
        workspace_size: usize,
    ) -> CudnnResult<()> {
        unsafe {
            cudnnConvolutionForward(
                self.cudnn_handle.0,
                &T::ONE as *const _ as *const _,
                self.x.descriptor.0,
                self.x.data.t_cuda.cu_device_ptr as *const _,
                self.filter.descriptor.0,
                self.filter.data.t_cuda.cu_device_ptr as *const _,
                self.descriptor.0,
                algorithm.algo,
                workspace_allocation as *mut _,
                workspace_size,
                &T::ZERO as *const _ as *const _,
                self.y.descriptor.0,
                self.y.data.t_cuda.cu_device_ptr as *mut _,
            )
        }
        .result()
    }
}
