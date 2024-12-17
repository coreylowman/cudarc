use crate::{
    cudnn::{result::CudnnError, sys},
    driver::{DevicePtr, DevicePtrMut},
};

use crate::cudnn::{result, Cudnn, CudnnDataType, TensorDescriptor};
use std::{marker::PhantomData, sync::Arc};

/// A descriptor of the window for pooling operation. Create with [`Cudnn::create_poolingnd()`]
pub struct PoolingDescriptor<T> {
    desc: sys::cudnnPoolingDescriptor_t,
    #[allow(unused)]
    handle: Arc<Cudnn>,
    marker: PhantomData<T>,
}

impl Cudnn {
    /// Create a window nd descriptor.
    pub fn create_poolingnd<T: CudnnDataType>(
        self: &Arc<Cudnn>,
        window: &[std::ffi::c_int],
        pads: &[std::ffi::c_int],
        strides: &[std::ffi::c_int],
        mode: sys::cudnnPoolingMode_t,
        nan_propagation: sys::cudnnNanPropagation_t,
    ) -> Result<PoolingDescriptor<T>, CudnnError> {
        let desc = result::create_pooling_descriptor()?;
        let desc = PoolingDescriptor {
            desc,
            handle: self.clone(),
            marker: PhantomData,
        };

        result::set_pooling_descriptor(
            desc.desc,
            mode,
            nan_propagation,
            window.len() as std::ffi::c_int,
            window,
            pads,
            strides,
        )?;

        Ok(desc)
    }
}

/// The pooling forward operation. Pass in references to descriptors
/// directly, and then call [`PoolingForward::launch()`].
pub struct PoolingForward<'a, P, X, Y> {
    pub pooling: &'a PoolingDescriptor<P>,
    pub x: &'a TensorDescriptor<X>,
    pub y: &'a TensorDescriptor<Y>,
}

impl<'a, P, X, Y> PoolingForward<'a, P, X, Y>
where
    P: CudnnDataType,
    X: CudnnDataType,
    Y: CudnnDataType,
{
    /// Launches the operation.
    ///
    /// - `src` is the input tensor
    /// - `y` is the output
    ///
    /// # Safety
    /// The arguments must match the data type/layout specified in the
    /// descriptors in `self.
    pub unsafe fn launch<Src, Dst>(
        &self,
        (alpha, beta): (Y, Y),
        src: &Src,
        y: &mut Dst,
    ) -> Result<(), CudnnError>
    where
        Src: DevicePtr<X>,
        Dst: DevicePtrMut<Y>,
    {
        let alpha = alpha.into_scaling_parameter();
        let beta = beta.into_scaling_parameter();
        result::pooling_forward(
            self.pooling.handle.handle,
            self.pooling.desc,
            (&alpha) as *const Y::Scalar as *const std::ffi::c_void,
            self.x.desc,
            *src.device_ptr() as *const X as *const std::ffi::c_void,
            (&beta) as *const Y::Scalar as *const std::ffi::c_void,
            self.y.desc,
            *y.device_ptr_mut() as *mut Y as *mut std::ffi::c_void,
        )
    }
}
