use crate::{
    cudnn::{result::CudnnError, sys},
    driver::{DevicePtr, DevicePtrMut},
};

use crate::cudnn::{result, Cudnn, CudnnDataType, TensorDescriptor};
use std::{marker::PhantomData, sync::Arc};

pub struct PoolingDescriptor<T> {
    desc: sys::cudnnPoolingDescriptor_t,
    #[allow(unused)]
    handle: Arc<Cudnn>,
    marker: PhantomData<T>,
}

impl Cudnn {
    pub fn create_poolingnd<T: CudnnDataType>(
        self: &Arc<Cudnn>,
        input: &[std::ffi::c_int],
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
            input.len() as std::ffi::c_int,
            input,
            pads,
            strides,
        )?;

        Ok(desc)
    }
}

pub struct PoolingForward<'a, P, X, Y> {
    pooling: &'a PoolingDescriptor<P>,
    x: &'a TensorDescriptor<X>,
    y: &'a TensorDescriptor<Y>,
}

impl<'a, P, X, Y> PoolingForward<'a, P, X, Y>
where
    P: CudnnDataType,
    X: CudnnDataType,
    Y: CudnnDataType,
{
    pub fn launch<Input, Output>(
        &self,
        (alpha, beta): (Y, Y),
        input: &Input,
        output: &mut Output,
    ) -> Result<(), CudnnError>
    where
        Input: DevicePtr<X>,
        Output: DevicePtrMut<Y>,
    {
        let alpha = alpha.into_scaling_parameter();
        let beta = beta.into_scaling_parameter();
        result::pooling_forward(
            self.pooling.handle.handle,
            self.pooling.desc,
            (&alpha) as *const Y::Scalar as *const std::ffi::c_void,
            self.x.desc,
            *input.device_ptr() as *const X as *const std::ffi::c_void,
            (&beta) as *const Y::Scalar as *const std::ffi::c_void,
            self.y.desc,
            *output.device_ptr_mut() as *mut Y as *mut std::ffi::c_void,
        )
    }
}
