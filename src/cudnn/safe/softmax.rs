use crate::cudnn::{result, sys, Cudnn, CudnnDataType, CudnnError, TensorDescriptor};
use crate::driver::{DevicePtr, DevicePtrMut};
use alloc::sync::Arc;
use core::marker::PhantomData;

/// A handle for the Softmax operation. Create with [`Cudnn::create_softmax()`]
#[derive(Debug)]
pub struct Softmax<T> {
    pub(crate) handle: Arc<Cudnn>,
    pub(crate) mode: sys::cudnnSoftmaxMode_t,
    pub(crate) marker: PhantomData<T>,
}

impl Cudnn {
    pub fn create_softmax<T: CudnnDataType>(
        self: &Arc<Cudnn>,
        mode: sys::cudnnSoftmaxMode_t,
    ) -> Result<Softmax<T>, CudnnError> {
        Ok(Softmax {
            handle: self.clone(),
            mode,
            marker: PhantomData,
        })
    }
}

/// The Softmax forward operation. Pass in references to descriptors
/// directly, and then call [`SoftmaxForward::launch()`] .
pub struct SoftmaxForward<'a, A: CudnnDataType, X: CudnnDataType, Y: CudnnDataType> {
    pub softmax: &'a Softmax<A>,
    pub x: &'a TensorDescriptor<X>,
    pub y: &'a TensorDescriptor<Y>,
}

impl<A, X, Y> SoftmaxForward<'_, A, X, Y>
where
    A: CudnnDataType,
    X: CudnnDataType,
    Y: CudnnDataType,
{
    /// Launches the operation.
    ///
    /// - `x` is the input tensor
    /// - `y` is the output
    ///
    /// # Safety
    /// The arguments must match the data type/layout specified in the
    /// descriptors in `self.
    pub unsafe fn launch<Src, Dst>(
        &self,
        (alpha, beta): (Y, Y),
        algo: sys::cudnnSoftmaxAlgorithm_t,
        x: &Src,
        y: &mut Dst,
    ) -> Result<(), CudnnError>
    where
        Src: DevicePtr<A>,
        Dst: DevicePtrMut<A>,
    {
        let stream = &self.x.handle.stream;
        let alpha = alpha.into_scaling_parameter();
        let beta = beta.into_scaling_parameter();
        let (x, _record_src) = x.device_ptr(stream);
        let (y, _record_y) = y.device_ptr_mut(stream);
        result::softmax_forward(
            self.softmax.handle.handle,
            algo,
            self.softmax.mode,
            (&alpha) as *const Y::Scalar as *const std::ffi::c_void,
            self.x.desc,
            x as *const X as *const std::ffi::c_void,
            (&beta) as *const Y::Scalar as *const std::ffi::c_void,
            self.y.desc,
            y as *mut Y as *mut std::ffi::c_void,
        )
    }
}
