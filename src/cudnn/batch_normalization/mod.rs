mod backward;
mod forward;


use crate::cudnn::sys::cudnnBatchNormMode_t;
// TODO allow other batch norm
pub(crate) const MODE: cudnnBatchNormMode_t =
    cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

pub use backward::*;
pub use forward::*;

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

        assert_eq!(&*y.get_data().as_host().unwrap(), &[[[[1.0, -1.0]], [[
            1.0, -1.0
        ]]]]);

        backward.execute(&cudnn_handle, &x, &mut dx, &y).unwrap();

        // TODO check if this is really right
        assert_eq!(&*dx.get_data().as_host().unwrap(), &[[[[0.0, 0.0]], [[
            0.0, 0.0
        ]]]]);
    }
}
