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
    x: Rc<TensorDescriptor<T, N, C_IN, H, W>>,
    filter: Rc<FilterDescriptor<T, C_OUT, C_IN, F_H, F_W>>,
    y: Rc<
        TensorDescriptor<
            T,
            N,
            C_OUT,
            { ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE },
            { ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE },
        >,
    >,
}
impl<
        T: TensorDataType,
        const H: usize,
        const W: usize,
        const P_H: usize,
        const P_W: usize,
        const N: usize,
        const C_IN: usize,
        const C_OUT: usize,
        const F_H: usize,
        const F_W: usize,
        const S_H: usize,
        const S_W: usize,
    > Convolution2DForward<T, H, W, P_H, P_W, N, C_IN, C_OUT, F_H, F_W, S_H, S_W>
where
    [(); ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE]:,
    [(); ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE]:,
{
    /// Creates a new [Convlution2DForward] by a [TensorDescriptor] of `x` and
    /// `y` and the [FilterDescriptor] of `filter`.
    ///
    /// # See also
    /// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnSetConvolution2dDescriptor>
    pub fn create(
        x: Rc<TensorDescriptor<T, N, C_IN, H, W>>,
        filter: Rc<FilterDescriptor<T, C_OUT, C_IN, F_H, F_W>>,
        y: Rc<
            TensorDescriptor<
                T,
                N,
                C_OUT,
                { ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE },
                { ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE },
            >,
        >,
    ) -> CudaCudnnResult<Self> {
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
            descriptor,
            x,
            filter,
            y,
        })
    }

    /// Creates a [Convolution2DBackward] of the same [ConvolutionDescriptor],
    /// requiring the [TensorDescriptor] of `dy` and `dx`.
    pub fn get_backward(
        &self,
        dy: Rc<
            TensorDescriptor<
                T,
                N,
                C_OUT,
                { ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE },
                { ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE },
            >,
        >,
        dx: Rc<TensorDescriptor<T, N, C_IN, H, W>>,
    ) -> Convolution2DBackward<T, H, W, P_H, P_W, N, C_IN, C_OUT, F_H, F_W, S_H, S_W> {
        Convolution2DBackward::create(self.descriptor.clone(), dy, Rc::clone(&self.filter), dx)
    }

    /// Creates a [Convolution2DBackwardFilter] of the same
    /// [ConvolutionDescriptor], requiring the [TensorDescriptor] of `dy`.
    pub fn get_filter_backward(
        &self,
        dy: Rc<
            TensorDescriptor<
                T,
                N,
                C_OUT,
                { ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE },
                { ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE },
            >,
        >,
    ) -> Convolution2DBackwardFilter<T, H, W, P_H, P_W, N, C_IN, C_OUT, F_H, F_W, S_H, S_W> {
        Convolution2DBackwardFilter::create(
            self.descriptor.clone(),
            Rc::clone(&self.filter),
            Rc::clone(&self.x),
            dy,
        )
    }
}
impl<
        T: TensorDataType,
        const H: usize,
        const W: usize,
        const P_H: usize,
        const P_W: usize,
        const N: usize,
        const C_IN: usize,
        const C_OUT: usize,
        const F_H: usize,
        const F_W: usize,
        const S_H: usize,
        const S_W: usize,
    > RequiresAlgorithmWithWorkspace<cudnnConvolutionFwdAlgoPerf_t>
    for Convolution2DForward<T, H, W, P_H, P_W, N, C_IN, C_OUT, F_H, F_W, S_H, S_W>
where
    [(); ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE]:,
    [(); ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE]:,
    [(); F_W * F_H * C_IN * C_OUT]:,
{
    type InputA = Tensor4DData<T, N, C_IN, H, W>;
    type InputB = Tensor4DData<T, C_OUT, C_IN, F_H, F_W>;
    type Output = Tensor4DData<
        T,
        N,
        C_OUT,
        { ConvolutionOutput::<H, P_H, F_H, S_H>::SIZE },
        { ConvolutionOutput::<W, P_W, F_W, S_W>::SIZE },
    >;

    fn get_workspace_size(
        &self,
        cudnn_handle: &CudnnHandle,
        algorithm: &cudnnConvolutionFwdAlgoPerf_t,
    ) -> CudaCudnnResult<usize> {
        let mut workspace_size = MaybeUninit::uninit();
        unsafe {
            cudnnGetConvolutionForwardWorkspaceSize(
                cudnn_handle.get_handle(),
                self.x.get_descriptor(),
                self.filter.get_descriptor(),
                self.descriptor.0,
                self.y.get_descriptor(),
                algorithm.algo,
                workspace_size.as_mut_ptr(),
            )
            .result()?;
            Ok(workspace_size.assume_init())
        }
    }

    fn get_algorithm(
        &self,
        cudnn_handle: &CudnnHandle,
    ) -> CudaCudnnResult<cudnnConvolutionFwdAlgoPerf_t> {
        let mut output_amount = MaybeUninit::uninit();
        let mut algorithm = MaybeUninit::uninit();
        unsafe {
            cudnnGetConvolutionForwardAlgorithm_v7(
                cudnn_handle.get_handle(),
                self.x.get_descriptor(),
                self.filter.get_descriptor(),
                self.descriptor.0,
                self.y.get_descriptor(),
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
        cudnn_handle: &CudnnHandle,
        algorithm: &cudnnConvolutionFwdAlgoPerf_t,
        workspace_allocation: crate::driver::sys::CUdeviceptr,
        workspace_size: usize,
        x: &Self::InputA,
        filter: &Self::InputB,
        y: &mut Self::Output,
    ) -> CudaCudnnResult<()> {
        unsafe {
            cudnnConvolutionForward(
                cudnn_handle.get_handle(),
                &T::ONE as *const _ as *const _,
                self.x.get_descriptor(),
                x.get_data_ptr(),
                self.filter.get_descriptor(),
                filter.get_data_ptr(),
                self.descriptor.0,
                algorithm.algo,
                workspace_allocation as *mut _,
                workspace_size,
                &T::ZERO as *const _ as *const _,
                self.y.get_descriptor(),
                y.get_data_ptr_mut(),
            )
        }
        .result()
    }
}
