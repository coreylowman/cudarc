use crate::cudnn::{result, sys, Cudnn, CudnnDataType, CudnnError};
use crate::driver::{DevicePtr, DevicePtrMut};
use core::marker::PhantomData;
use std::sync::Arc;

pub struct ActivationForward<'a, A: CudnnDataType> {
    /// Activation function.
    pub act: &'a ActivationDescriptor<A>,
}

impl<'a, T> ActivationForward<'a, T>
where
    T: CudnnDataType,
{
    pub fn launch<Src, Dst>(
        &self,
        (alpha, beta): (T, T),
        x_desc: sys::cudnnTensorDescriptor_t,
        x: &Src,
        y_desc: sys::cudnnTensorDescriptor_t,
        y: &mut Dst,
    ) -> Result<(), CudnnError>
    where
        Src: DevicePtr<T>,
        Dst: DevicePtrMut<T>,
    {
        let alpha = alpha.into_scaling_parameter();
        let beta = beta.into_scaling_parameter();
        result::activation_forward(
            self.act.handle.handle,
            self.act.desc,
            (&alpha) as *const T::Scalar as *const std::ffi::c_void,
            x_desc,
            *x.device_ptr() as *const T as *const std::ffi::c_void,
            (&beta) as *const T::Scalar as *const std::ffi::c_void,
            y_desc,
            *y.device_ptr_mut() as *mut T as *mut std::ffi::c_void,
        )
    }
}

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
