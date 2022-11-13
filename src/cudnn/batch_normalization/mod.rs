mod backward;
mod forward;

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

        let mut forward =
            BatchNormalizationForwardPerImage::create(&device, &cudnn_handle).unwrap();
        let mut backward = forward.get_backward();

        forward.train(&cudnn_handle, &x, &mut y).unwrap();

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
