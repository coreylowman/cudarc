//! Safe wrappers around cuDNN.
//!
//! # Convolutions
//!
//! 1. Allocate tensor descriptors with [`Cudnn::create_4d_tensor()`]
//! 2. Allocate filter descriptors with [`Cudnn::create_4d_filter()`]
//! 3. Allocate conv descriptors with [`Cudnn::create_conv2d()`]
//! 4. Instantiate one of the following algorithms with the descriptors:
//!    a. [`Conv2dForward`]
//!    b. [`Conv2dBackwardData`] for computing gradient of image
//!    c. [`Conv2dBackwardFilter`] for computing gradient of filters
//! 5. Call the `pick_algorithm` method of the struct. Specify the number of options to compare with a const generic.
//! 6. Call the `get_workspace_size` method of the struct.
//! 7. Re-allocate the workspace to the appropriate size.
//! 8. Call the `launch` method of the struct.
//!
//! # Reductions

mod activation;
mod conv;
mod core;
mod pooling;
mod reduce;
mod softmax;

#[allow(deprecated)]
pub use self::conv::{
    // Deprecated APIs
    Conv2dBackwardData,
    Conv2dBackwardFilter,
    Conv2dDescriptor,
    Conv2dForward,
    // Current APIs
    ConvBackwardData,
    ConvBackwardFilter,
    ConvBiasActivationForward,
    ConvDescriptor,
    ConvForward,
    FilterDescriptor,
};
pub use self::core::{Cudnn, CudnnDataType, TensorDescriptor};
pub use self::pooling::{PoolingDescriptor, PoolingForward};
pub use self::reduce::{FlatIndices, NoIndices, ReduceTensor, ReductionDescriptor};
pub use super::result::CudnnError;
pub use activation::{ActivationDescriptor, ActivationForward};
pub use softmax::{Softmax, SoftmaxForward};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cudnn::safe::softmax::SoftmaxForward;
    use crate::{cudnn, driver::CudaContext};
    #[cfg(feature = "no-std")]
    use no_std_compat::vec;

    #[test]
    fn test_create_descriptors() -> Result<(), CudnnError> {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cudnn = Cudnn::new(stream)?;
        let _ = cudnn.create_4d_tensor_ex::<f32>([1, 2, 3, 4], [24, 12, 4, 1])?;
        let _ = cudnn.create_nd_tensor::<f64>(&[1, 2, 3, 4, 5, 6], &[720, 360, 120, 30, 6, 1])?;
        let _ = cudnn.create_4d_filter::<f32>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [3, 3, 3, 3],
        )?;
        let _ = cudnn.create_reduction_flat_indices::<f32>(
            cudnn::sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD,
            cudnn::sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
        )?;
        let _ = cudnn.create_reduction_no_indices::<f32>(
            cudnn::sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD,
            cudnn::sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
        )?;
        Ok(())
    }

    #[test]
    fn test_conv2d_pick_algorithms() -> Result<(), CudnnError> {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cudnn = Cudnn::new(stream)?;

        let conv = cudnn.create_conv2d::<f32>(
            [0; 2],
            [1; 2],
            [1; 2],
            cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )?;
        let x = cudnn.create_4d_tensor::<f32>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [100, 128, 224, 224],
        )?;
        let filter = cudnn.create_4d_filter::<f32>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [256, 128, 3, 3],
        )?;
        let y = cudnn.create_4d_tensor::<f32>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [100, 256, 222, 222],
        )?;

        {
            let op = ConvForward {
                conv: &conv,
                x: &x,
                w: &filter,
                y: &y,
            };
            let algo = op.pick_algorithm()?;
            assert_eq!(
                algo,
                cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
            );
        }

        {
            let op = ConvBackwardData {
                conv: &conv,
                dx: &x,
                w: &filter,
                dy: &y,
            };
            let algo = op.pick_algorithm()?;
            assert_eq!(
                algo,
                cudnn::sys::cudnnConvolutionBwdDataAlgo_t::CUDNN_CONVOLUTION_BWD_DATA_ALGO_1
            );
        }

        {
            let op = ConvBackwardFilter {
                conv: &conv,
                x: &x,
                dw: &filter,
                dy: &y,
            };
            let algo = op.pick_algorithm()?;
            assert_eq!(
                algo,
                cudnn::sys::cudnnConvolutionBwdFilterAlgo_t::CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
            );
        }

        Ok(())
    }

    #[test]
    fn test_conv1d() -> Result<(), CudnnError> {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cudnn = Cudnn::new(stream.clone())?;

        let conv = cudnn.create_convnd::<f32>(
            &[0; 2],
            &[1; 2],
            &[1; 2],
            cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )?;
        // With less than 4 dimensions, 4D tensors should be used with 1 set for unused
        // dimensions

        // Create input, filter and output tensors
        let x = stream.clone_htod(&vec![1.0f32; 100 * 128 * 32]).unwrap();
        let x_desc = cudnn.create_4d_tensor::<f32>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [100, 128, 32, 1],
        )?;
        let filter = stream.clone_htod(&vec![1.0f32; 256 * 128 * 3]).unwrap();
        let filter_desc = cudnn.create_nd_filter::<f32>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            &[256, 128, 3, 1],
        )?;
        let mut y = stream.alloc_zeros::<f32>(100 * 256 * 30).unwrap();
        let y_desc = cudnn.create_4d_tensor::<f32>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            [100, 256, 30, 1],
        )?;

        {
            let op = ConvForward {
                conv: &conv,
                x: &x_desc,
                w: &filter_desc,
                y: &y_desc,
            };

            // Pick algorithm
            // Note that the number of dimensions in the filter and input
            // must match. Hence the similarity with Conv2D operation.
            let algo = op.pick_algorithm()?;

            // Get workspace size
            let workspace_size = op.get_workspace_size(algo)?;
            let mut workspace = stream.alloc_zeros::<u8>(workspace_size).unwrap();

            // Launch conv operation
            unsafe {
                op.launch(algo, Some(&mut workspace), (1.0, 0.0), &x, &filter, &mut y)?;
            }

            let y_host = stream.clone_dtoh(&y).unwrap();
            assert_eq!(y_host.len(), 100 * 256 * 30);
            assert_eq!(y_host[0], 128.0 * 3.0);
        }

        Ok(())
    }

    #[test]
    fn test_conv3d() -> Result<(), CudnnError> {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cudnn = Cudnn::new(stream.clone())?;

        let conv = cudnn.create_convnd::<f32>(
            &[0; 3],
            &[1; 3],
            &[1; 3],
            cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )?;

        // Create input, filter and output tensors
        let x = stream
            .clone_htod(&vec![1.0f32; 32 * 3 * 64 * 64 * 64])
            .unwrap();
        let x_desc = cudnn.create_nd_tensor::<f32>(
            &[32, 3, 64, 64, 64],
            &[3 * 64 * 64 * 64, 64 * 64 * 64, 64 * 64, 64, 1],
        )?;
        let filter = stream
            .clone_htod(&vec![1.0f32; 32 * 3 * 4 * 4 * 4])
            .unwrap();
        let filter_desc = cudnn.create_nd_filter::<f32>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            &[32, 3, 4, 4, 4],
        )?;
        let mut y = stream.alloc_zeros::<f32>(32 * 32 * 61 * 61 * 61).unwrap();
        let y_desc = cudnn.create_nd_tensor::<f32>(
            &[32, 32, 61, 61, 61],
            &[32 * 61 * 61 * 61, 61 * 61 * 61, 61 * 61, 61, 1],
        )?;

        {
            let op = ConvForward {
                conv: &conv,
                x: &x_desc,
                w: &filter_desc,
                y: &y_desc,
            };

            // Pick algorithm
            let algo = op.pick_algorithm()?;

            // Get workspace size
            let workspace_size = op.get_workspace_size(algo)?;
            let mut workspace = stream.alloc_zeros::<u8>(workspace_size).unwrap();

            // Launch conv operation
            unsafe {
                op.launch(algo, Some(&mut workspace), (1.0, 0.0), &x, &filter, &mut y)?;
            }

            let y_host = stream.clone_dtoh(&y).unwrap();
            assert_eq!(y_host.len(), 32 * 32 * 61 * 61 * 61);
            assert_eq!(y_host[0], 3.0 * 4.0 * 4.0 * 4.0);
        }

        Ok(())
    }

    #[test]
    fn test_reduction() {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cudnn = Cudnn::new(stream.clone()).unwrap();

        let a = stream
            .clone_htod(&std::vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])
            .unwrap();
        let mut c = stream.alloc_zeros::<f32>(1).unwrap();

        let reduce = cudnn
            .create_reduction_no_indices::<f32>(
                cudnn::sys::cudnnReduceTensorOp_t::CUDNN_REDUCE_TENSOR_ADD,
                cudnn::sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
            )
            .unwrap();
        let a_desc = cudnn
            .create_nd_tensor::<f32>(&[1, 1, 2, 3], &[0, 6, 3, 1])
            .unwrap();
        let c_desc = cudnn
            .create_nd_tensor::<f32>(&[1, 1, 1, 1], &[0, 0, 0, 1])
            .unwrap();
        let op = ReduceTensor {
            reduce: &reduce,
            a: &a_desc,
            c: &c_desc,
        };

        let workspace_size = op.get_workspace_size().unwrap();
        let mut workspace = stream.alloc_zeros::<u8>(workspace_size).unwrap();

        unsafe { op.launch(&mut workspace, (1.0, 0.0), &a, &mut c) }.unwrap();

        let c_host = stream.clone_dtoh(&c).unwrap();
        assert_eq!(c_host.len(), 1);
        assert_eq!(c_host[0], 21.0);
    }

    #[test]
    fn test_conv_bias_activation() -> Result<(), CudnnError> {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cudnn = Cudnn::new(stream.clone())?;

        let conv = cudnn.create_convnd::<f32>(
            &[0; 3],
            &[1; 3],
            &[1; 3],
            cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )?;

        // Create input, filter and output tensors
        let x = stream
            .clone_htod(&vec![1.0f32; 32 * 3 * 64 * 64 * 64])
            .unwrap();
        let x_desc = cudnn.create_nd_tensor::<f32>(
            &[32, 3, 64, 64, 64],
            &[3 * 64 * 64 * 64, 64 * 64 * 64, 64 * 64, 64, 1],
        )?;
        let filter = stream
            .clone_htod(&vec![1.0f32; 32 * 3 * 4 * 4 * 4])
            .unwrap();
        let filter_desc = cudnn.create_nd_filter::<f32>(
            cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            &[32, 3, 4, 4, 4],
        )?;
        let bias = stream.clone_htod(&[1.0f32; 32]).unwrap();
        let bias_desc = cudnn.create_nd_tensor::<f32>(&[1, 32, 1, 1, 1], &[32, 1, 1, 1, 1])?;
        let activation_desc = cudnn.create_activation::<f32>(
            cudnn::sys::cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
            cudnn::sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
            f64::MAX,
        )?;
        let z = stream
            .clone_htod(&vec![0.0f32; 32 * 32 * 61 * 61 * 61])
            .unwrap();
        let z_desc = cudnn.create_nd_tensor::<f32>(
            &[32, 32, 61, 61, 61],
            &[32 * 61 * 61 * 61, 61 * 61 * 61, 61 * 61, 61, 1],
        )?;
        let mut y = stream.alloc_zeros::<f32>(32 * 32 * 61 * 61 * 61).unwrap();
        let y_desc = cudnn.create_nd_tensor::<f32>(
            &[32, 32, 61, 61, 61],
            &[32 * 61 * 61 * 61, 61 * 61 * 61, 61 * 61, 61, 1],
        )?;

        {
            let op = ConvBiasActivationForward {
                conv: &conv,
                act: &activation_desc,
                x: &x_desc,
                w: &filter_desc,
                y: &y_desc,
                z: &z_desc,
                bias: &bias_desc,
            };

            // Pick algorithm
            let algo = op.pick_algorithm()?;

            // Get workspace size
            let workspace_size = op.get_workspace_size(algo)?;
            let mut workspace = stream.alloc_zeros::<u8>(workspace_size).unwrap();

            // Launch conv operation
            unsafe {
                op.launch(
                    algo,
                    Some(&mut workspace),
                    (1.0, 0.0),
                    &x,
                    &filter,
                    &z,
                    &bias,
                    &mut y,
                )?;
            }

            let y_host = stream.clone_dtoh(&y).unwrap();
            assert_eq!(y_host.len(), 32 * 32 * 61 * 61 * 61);
            assert_eq!(y_host[0], 3.0 * 4.0 * 4.0 * 4.0 + 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_pooling() -> Result<(), CudnnError> {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cudnn = Cudnn::new(stream.clone())?;

        let pooling = cudnn.create_poolingnd::<f32>(
            &[2, 2],
            &[0, 0],
            &[2, 2],
            cudnn::sys::cudnnPoolingMode_t::CUDNN_POOLING_MAX,
            cudnn::sys::cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
        )?;

        // Create input, filter and output tensors
        let x = stream
            .clone_htod(&[
                1.0, 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0,
            ])
            .unwrap();
        let x_desc = cudnn.create_nd_tensor::<f32>(&[32, 3, 4, 4], &[32 * 3 * 4, 3 * 4, 4, 1])?;
        let mut y = stream.alloc_zeros::<f32>(32 * 3 * 2 * 2).unwrap();
        let y_desc = cudnn.create_nd_tensor::<f32>(&[32, 3, 2, 2], &[3 * 2 * 2, 2 * 2, 2, 1])?;

        {
            let op = PoolingForward {
                pooling: &pooling,
                x: &x_desc,
                y: &y_desc,
            };

            // Launch conv operation
            unsafe {
                op.launch((1.0, 0.0), &x, &mut y)?;
            }

            let y_host = stream.clone_dtoh(&y).unwrap();
            assert_eq!(y_host.len(), 32 * 3 * 2 * 2);
            assert_eq!(y_host[0], 6.0);
        }

        Ok(())
    }

    #[test]
    fn test_activation() -> Result<(), CudnnError> {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cudnn = Cudnn::new(stream.clone())?;

        let act = cudnn.create_activation::<f32>(
            cudnn::sys::cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
            cudnn::sys::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
            f64::MAX,
        )?;

        // Create input, filter and output tensors
        let x = stream.clone_htod(&[-1.0, 2.0, -3.0, 100.0]).unwrap();
        let x_desc = cudnn.create_nd_tensor::<f32>(&[1, 1, 2, 2], &[2 * 2, 2 * 2, 2, 1])?;
        let mut y = stream.alloc_zeros::<f32>(4).unwrap();
        let y_desc = cudnn.create_nd_tensor::<f32>(&[1, 1, 2, 2], &[2 * 2, 2 * 2, 2, 1])?;

        {
            let op = ActivationForward {
                act: &act,
                x: &x_desc,
                y: &y_desc,
            };

            // Launch conv operation
            unsafe {
                op.launch((1.0, 0.0), &x, &mut y)?;
            }

            let y_host = stream.clone_dtoh(&y).unwrap();
            assert_eq!(y_host.len(), 2 * 2);
            assert_eq!(y_host[0], 0.0);
            assert_eq!(y_host[1], 2.0);
            assert_eq!(y_host[2], 0.0);
            assert_eq!(y_host[3], 100.0);
        }

        Ok(())
    }

    #[test]
    fn test_softmax() -> Result<(), CudnnError> {
        let ctx = CudaContext::new(0).unwrap();
        let stream = ctx.default_stream();
        let cudnn = Cudnn::new(stream.clone())?;

        let softmax = cudnn
            .create_softmax::<f32>(cudnn::sys::cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE)?;

        // Create input, filter and output tensors.
        let x = stream.clone_htod(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let x_desc = cudnn.create_nd_tensor::<f32>(&[1, 1, 2, 2], &[2 * 2, 2 * 2, 2, 1])?;
        let mut y = stream.alloc_zeros::<f32>(4).unwrap();
        let y_desc = cudnn.create_nd_tensor::<f32>(&[1, 1, 2, 2], &[2 * 2, 2 * 2, 2, 1])?;

        {
            let op = SoftmaxForward {
                softmax: &softmax,
                x: &x_desc,
                y: &y_desc,
            };

            unsafe {
                op.launch(
                    (1.0, 0.0),
                    cudnn::sys::cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_FAST,
                    &x,
                    &mut y,
                )?;
            }

            let y_host = stream.clone_dtoh(&y).unwrap();
            assert_eq!(y_host.len(), 2 * 2);
            assert_eq!(y_host[0], 0.0320586);
            assert_eq!(y_host[1], 0.08714432);
            assert_eq!(y_host[2], 0.23688282);
            assert_eq!(y_host[3], 0.6439142);
        }

        Ok(())
    }
}
