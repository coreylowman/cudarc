use core::marker::PhantomData;
use core::mem::zeroed;

use super::sys::*;
use crate::prelude::*;

mod filter;

pub use filter::*;

pub struct ConvolutionDescriptor(pub(crate) cudnnConvolutionDescriptor_t);
impl ConvolutionDescriptor {
    pub fn create() -> CudnnResult<Self> {
        let mut descriptor: Self = unsafe { std::mem::zeroed() };
        unsafe { cudnnCreateConvolutionDescriptor(&mut descriptor.0 as *mut _) }.result()?;
        Ok(descriptor)
    }
}
impl Drop for ConvolutionDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyConvolutionDescriptor(self.0) }
            .result()
            .unwrap();
    }
}

pub struct Assert<const A: usize, const B: usize>;
pub trait ConstEq {}
impl<const A: usize> ConstEq for Assert<A, A> {}

/// `U` and `V` must be >0
pub struct Convolution2D<T, const P_H: usize, const P_W: usize, const V_S: usize, const H_S: usize>
{
    descriptor: ConvolutionDescriptor,
    data_type:  PhantomData<T>,
}
impl<T: TensorDataType, const P_H: usize, const P_W: usize, const V_S: usize, const H_S: usize>
    Convolution2D<T, P_H, P_W, V_S, H_S>
{
    pub fn create() -> CudnnResult<Self> {
        let descriptor = ConvolutionDescriptor::create()?;
        unsafe {
            cudnnSetConvolution2dDescriptor(
                descriptor.0,
                P_H as _,
                P_W as _,
                V_S as _,
                H_S as _,
                1,
                1,
                cudnnConvolutionMode_t::CUDNN_CONVOLUTION,
                T::get_data_type(),
            )
        }
        .result()?;
        Ok(Self {
            descriptor,
            data_type: PhantomData,
        })
    }

    pub fn forward<
        const C_IN: usize,
        const C_OUT: usize,
        const H_IN: usize,
        const W_IN: usize,
        const H_F: usize,
        const W_F: usize,
        const N: usize,
    >(
        &self,
        cudnn_handle: CudnnHandle,
        input: &Tensor4D<T, N, C_IN, H_IN, W_IN>,
        filter: &Filter<T, C_OUT, C_IN, H_F, W_F>,
        output: &mut Tensor4D<
            T,
            N,
            C_OUT,
            { H_IN - (H_F - 1) + 2 * P_H },
            { W_IN - (W_F - 1) + 2 * P_W },
        >,
        workspace: CudaRc<[u8; 10000]>,
    ) -> CudnnResult<()> {
        unsafe {
            // let mut n = zeroed();
            // let mut c = zeroed();
            // let mut h = zeroed();
            // let mut w = zeroed();
            // cudnnGetConvolution2dForwardOutputDim(self.descriptor.0,
            // input.descriptor.descriptor.0, filter.descriptor.0, &mut n, &mut c, &mut h,
            // &mut w).result().unwrap(); std::println!("{n}|{c}|{h}|{w}");
            // std::println!("{}|{}|{}|{}", N, C_OUT, H_IN - (H_F - 1) + 2 * P_H, W_IN -
            // (W_F - 1) + 2 * P_W);
            let mut output_amount: i32 = zeroed();
            let mut algorithm: cudnnConvolutionFwdAlgoPerf_t = zeroed();
            cudnnGetConvolutionForwardAlgorithm_v7(
                cudnn_handle.0,
                input.descriptor.descriptor.0,
                filter.descriptor.0,
                self.descriptor.0,
                output.descriptor.descriptor.0,
                1,
                &mut output_amount,
                &mut algorithm,
            )
            .result()?;
            assert_eq!(
                output_amount, 1,
                "cudnnGetConvolutionForwardAlgorithm_v7 returned 0 algorithms"
            );
            let mut workspace_size: usize = zeroed();
            cudnnGetConvolutionForwardWorkspaceSize(
                cudnn_handle.0,
                input.descriptor.descriptor.0,
                filter.descriptor.0,
                self.descriptor.0,
                output.descriptor.descriptor.0,
                algorithm.algo,
                &mut workspace_size,
            )
            .result()?;
            cudnnConvolutionForward(
                cudnn_handle.0,
                &T::ONE as *const _ as *const _,
                input.descriptor.descriptor.0,
                input.data.t_cuda.cu_device_ptr as *const _,
                filter.descriptor.0,
                filter.data.t_cuda.cu_device_ptr as *const _,
                self.descriptor.0,
                algorithm.algo,
                workspace.t_cuda.cu_device_ptr as *mut _,
                workspace_size,
                &T::ZERO as *const _ as *const _,
                output.descriptor.descriptor.0,
                output.data.t_cuda.cu_device_ptr as *mut _,
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
    fn test_create_descriptor() {
        let _descriptor = Convolution2D::<f64, 0, 0, 1, 1>::create().unwrap();
    }

    #[test]
    fn test_simple_convolution() {
        let cudnn_handle = CudnnHandle::create().unwrap();
        let cuda = CudaDeviceBuilder::new(0).build().unwrap();

        let input_allocation = cuda.alloc_zeros().unwrap();
        let filter_allocation = cuda.alloc_zeros().unwrap();
        let output_allocation = cuda.take(Rc::new(unsafe { std::mem::zeroed() })).unwrap();
        let workspace_allocation: CudaRc<[u8; 10000]> = cuda.alloc_zeros().unwrap();

        let mut input_data = [[0.0; 5]; 5];
        for y in 0..5 {
            for x in 0..5 {
                input_data[y][x] = (y * 5 + x) as f64;
            }
        }
        let input =
            Tensor4D::create_async(input_allocation.clone(), &[[input_data, input_data]]).unwrap();
        let filter = Filter::<f64, 1, 2, 2, 2>::create_async(
            filter_allocation.clone(),
            &[[[[1.0f64; 2]; 2]; 2]; 1],
        )
        .unwrap();
        let mut output = unsafe { Tensor4D::uninit(output_allocation.clone()) }.unwrap();

        let convolution = Convolution2D::<f64, 0, 0, 1, 1>::create().unwrap();
        convolution
            .forward(
                cudnn_handle,
                &input,
                &filter,
                &mut output,
                workspace_allocation.clone(),
            )
            .unwrap();

        let output = output_allocation.sync_release().unwrap().unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let expected = 24 + y * 40 + x * 8;
                let actual = output[0][0][y][x];
                assert!(
                    (actual - expected as f64) < f64::EPSILON,
                    "Output data {output:?} is at index {x}|{y} not {expected}, but {actual}."
                );
            }
        }
    }
}
