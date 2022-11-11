use core::mem::MaybeUninit;

use super::sys::*;
use crate::prelude::*;

mod backward;
mod filter;
mod filter_backward;
mod forward;

pub use backward::*;
pub use filter::*;
pub use filter_backward::*;
pub use forward::*;

/// A convolution descriptor, may be reused if the same convolution settings
/// (padding, stride, ...) are used.
///
/// # See also
/// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnConvolutionDescriptor_t>
pub struct ConvolutionDescriptor(pub(crate) cudnnConvolutionDescriptor_t);
impl ConvolutionDescriptor {
    /// Creates a new [ConvolutionDescriptor].
    ///
    /// # See also
    /// <https://docs.nvidia.com/deeplearning/cudnn/api/index.html#cudnnCreateConvolutionDescriptor>
    pub fn create() -> CudaCudnnResult<Self> {
        let mut descriptor = MaybeUninit::uninit();
        unsafe {
            cudnnCreateConvolutionDescriptor(descriptor.as_mut_ptr()).result()?;
            Ok(Self(descriptor.assume_init()))
        }
    }
}
impl Drop for ConvolutionDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyConvolutionDescriptor(self.0) }
            .result()
            .unwrap();
    }
}

/// A trait wrapper for [ConvolutionOutput].
pub trait ConvolutionOutputTrait {
    const SIZE: usize;
}
/// [ConvolutionOutputTrait::SIZE] is the output dimension, calculated by:
///     - `D`: the input dimension (width or height)
///     - `P`: the padding, equal on both sides
///     - `K`: the kernel size (or filter size)
///     - `S`: the stride
pub struct ConvolutionOutput<const H: usize, const P: usize, const K: usize, const S: usize>;
impl<const D: usize, const P: usize, const K: usize, const S: usize> ConvolutionOutputTrait
    for ConvolutionOutput<D, P, K, S>
{
    const SIZE: usize = (D + 2 * P - K) / S + 1;
}

#[cfg(test)]
mod tests {
    use alloc::rc::Rc;

    use crate::prelude::*;

    #[test]
    fn test_create_descriptor() {
        let _descriptor = ConvolutionDescriptor::create().unwrap();
    }

    #[test]
    fn test_simple_convolution() {
        let cuda = CudaDeviceBuilder::new(0).build().unwrap();
        let cudnn_handle = Rc::new(CudnnHandle::create(&cuda).unwrap());

        let mut input_data = [[0.0; 5]; 5];
        for y in 0..5 {
            for x in 0..5 {
                input_data[y][x] = (y * 5 + x) as f64;
            }
        }

        let x = Tensor4D::alloc_with(&cuda, [[input_data, input_data]]).unwrap();
        let filter =
            Filter::<f64, 1, 2, 2, 2>::alloc_with(&cuda, [[[[1.0f64; 2]; 2]; 2]; 1]).unwrap();
        let y = unsafe { Tensor4D::alloc_uninit(&cuda) }.unwrap();
        let dx = unsafe { Tensor4D::alloc_uninit(&cuda) }.unwrap();
        let dw = unsafe { Filter::alloc_uninit(&cuda) }.unwrap();

        let convolution_forward =
            Convolution2DForward::<f64, 5, 5, 0, 0, 1, 1, 1, 2, 1, 2, 2, 1, 1>::create(
                cudnn_handle.clone(),
                x,
                filter,
                y.clone(),
            )
            .unwrap();
        let convolution_backward = convolution_forward.get_backward(y.clone(), dx.clone());
        let convolution_backward_filter =
            convolution_forward.get_filter_backward(y.clone(), dw.clone());
        let mut convolution =
            AlgorithmWithWorkspace::create(convolution_forward, cuda.clone()).unwrap();
        convolution.execute().unwrap();

        let output = y.data.into_host().unwrap();
        for y in 0..4 {
            for x in 0..4 {
                let expected = 24 + y * 40 + x * 8;
                let actual = output[0][0][y][x];
                assert!(
                    (actual - expected as f64) < f64::EPSILON,
                    "Output data {y:?} is at index {x}|{y} not {expected}, but {actual}."
                );
            }
        }

        let mut convolution =
            AlgorithmWithWorkspace::create(convolution_backward, cuda.clone()).unwrap();
        convolution.execute().unwrap();

        // TODO maybe check if this is right?
        assert_eq!(&*dx.data.into_host().unwrap(), &[[
            [
                [24.0, 56.0, 72.0, 88.0, 48.0],
                [88.0, 192.0, 224.0, 256.0, 136.0],
                [168.0, 352.0, 384.0, 416.0, 216.0],
                [248.0, 512.0, 544.0, 576.0, 296.0],
                [144.0, 296.0, 312.0, 328.0, 168.0]
            ],
            [
                [24.0, 56.0, 72.0, 88.0, 48.0],
                [88.0, 192.0, 224.0, 256.0, 136.0],
                [168.0, 352.0, 384.0, 416.0, 216.0],
                [248.0, 512.0, 544.0, 576.0, 296.0],
                [144.0, 296.0, 312.0, 328.0, 168.0]
            ]
        ]]);

        let mut convolution =
            AlgorithmWithWorkspace::create(convolution_backward_filter, cuda.clone()).unwrap();
        convolution.execute().unwrap();

        // TODO maybe check if this is right?
        assert_eq!(&*dw.data.into_host().unwrap(), &[[
            [[27200.0, 25664.0], [19520.0, 17984.0]],
            [[27200.0, 25664.0], [19520.0, 17984.0]]
        ]]);
    }
}
