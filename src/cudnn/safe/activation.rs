use crate::cudnn::{result, sys, Cudnn, CudnnDataType, CudnnError, TensorDescriptor};
use crate::driver::{DevicePtr, DevicePtrMut};
use core::marker::PhantomData;
use std::sync::Arc;

#[derive(Debug)]
pub struct ActivationDescriptor<T> {
    pub(crate) desc: sys::cudnnActivationDescriptor_t,
    #[allow(unused)]
    pub(crate) handle: Arc<Cudnn>,
    pub(crate) marker: PhantomData<T>,
}

impl Cudnn {
    pub fn create_activation<T: CudnnDataType>(
        self: &Arc<Cudnn>,
        mode: sys::cudnnActivationMode_t,
        nan_propagation: sys::cudnnNanPropagation_t,
        coef: f64,
    ) -> Result<ActivationDescriptor<T>, CudnnError> {
        let desc = result::create_activation_descriptor()?;
        let desc = ActivationDescriptor {
            desc,
            handle: self.clone(),
            marker: PhantomData,
        };
        result::set_activation_descriptor(desc.desc, mode, nan_propagation, coef)?;
        Ok(desc)
    }
}

pub struct ActivationForward<'a, A: CudnnDataType, X: CudnnDataType, Y: CudnnDataType> {
    /// Activation function.
    pub act: &'a ActivationDescriptor<A>,
    pub x: &'a TensorDescriptor<X>,
    pub y: &'a TensorDescriptor<Y>,
}

impl<'a, A, X, Y> ActivationForward<'a, A, X, Y>
where
    A: CudnnDataType,
    X: CudnnDataType,
    Y: CudnnDataType,
{
    pub fn launch<Src, Dst>(
        &self,
        (alpha, beta): (Y, Y),
        x: &Src,
        y: &mut Dst,
    ) -> Result<(), CudnnError>
    where
        Src: DevicePtr<A>,
        Dst: DevicePtrMut<A>,
    {
        let alpha = alpha.into_scaling_parameter();
        let beta = beta.into_scaling_parameter();
        result::activation_forward(
            self.act.handle.handle,
            self.act.desc,
            (&alpha) as *const Y::Scalar as *const std::ffi::c_void,
            self.x.desc,
            *x.device_ptr() as *const X as *const std::ffi::c_void,
            (&beta) as *const Y::Scalar as *const std::ffi::c_void,
            self.y.desc,
            *y.device_ptr_mut() as *mut Y as *mut std::ffi::c_void,
        )
    }
}
