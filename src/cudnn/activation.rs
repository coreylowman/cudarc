use core::marker::PhantomData;

use super::sys::*;
use crate::prelude::*;

pub struct Activation<T> {
    descriptor:      ActivationDescriptor,
    activation_mode: PhantomData<T>,
}
impl<A: ActivationMode> Activation<A> {
    pub fn create() -> CudnnResult<Self> {
        let descriptor = ActivationDescriptor::create()?;
        unsafe {
            cudnnSetActivationDescriptor(
                descriptor.0,
                A::get_activation_mode(),
                cudnnNanPropagation_t::CUDNN_PROPAGATE_NAN,
                A::get_additional_parameter(),
            )
        }
        .result()?;
        Ok(Self {
            descriptor,
            activation_mode: PhantomData,
        })
    }

    pub fn forward<
        T: TensorDataType,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
    >(
        &self,
        cudnn_handle: CudnnHandle,
        input: &Tensor4D<T, N, C, H, W>,
        output: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudnnResult<()> {
        unsafe {
            cudnnActivationForward(
                cudnn_handle.0,
                self.descriptor.0,
                &T::ONE as *const _ as *const _,
                input.descriptor.descriptor.0,
                input.data.t_cuda.cu_device_ptr as *const _,
                &T::ZERO as *const _ as *const _,
                output.descriptor.descriptor.0,
                output.data.t_cuda.cu_device_ptr as *mut _,
            )
        }
        .result()
    }

    pub fn forward_inplace<
        T: TensorDataType,
        const N: usize,
        const C: usize,
        const H: usize,
        const W: usize,
    >(
        &self,
        cudnn_handle: CudnnHandle,
        input: &mut Tensor4D<T, N, C, H, W>,
    ) -> CudnnResult<()> {
        unsafe {
            cudnnActivationForward(
                cudnn_handle.0,
                self.descriptor.0,
                &T::ONE as *const _ as *const _,
                input.descriptor.descriptor.0,
                input.data.t_cuda.cu_device_ptr as *const _,
                &T::ZERO as *const _ as *const _,
                input.descriptor.descriptor.0,
                input.data.t_cuda.cu_device_ptr as *mut _,
            )
        }
        .result()
    }
}

pub struct ActivationDescriptor(pub(crate) cudnnActivationDescriptor_t);
impl ActivationDescriptor {
    pub fn create() -> CudnnResult<Self> {
        let mut descriptor: Self = unsafe { std::mem::zeroed() };
        unsafe { cudnnCreateActivationDescriptor(&mut descriptor.0 as *mut _) }.result()?;
        Ok(descriptor)
    }
}
impl Drop for ActivationDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyActivationDescriptor(self.0) }
            .result()
            .unwrap();
    }
}

#[cfg(test)]
mod tests {
    use crate::driver::sys::cuMemcpyDtoH_v2;
    use crate::prelude::*;

    #[test]
    fn test_create_descriptor() {
        let _descriptor = ActivationDescriptor::create().unwrap();
    }

    #[test]
    fn test_relu_activation_f32() {
        let cudnn_handle = CudnnHandle::create().unwrap();
        let cuda = CudaDeviceBuilder::new(0).build().unwrap();
        let allocation_in = cuda.alloc_zeros().unwrap();
        let tensor_in =
            Tensor2D::create(allocation_in, [[[[f32::NAN, 2.0]]], [[[-1.0, 0.0]]]]).unwrap();
        let allocation_out = cuda.alloc_zeros().unwrap();
        let mut tensor_out = unsafe { Tensor2D::uninit(allocation_out.clone()) }.unwrap();
        let activation = Activation::<Relu>::create().unwrap();
        activation
            .forward(cudnn_handle, &tensor_in, &mut tensor_out)
            .unwrap();
        let mut out = [[[[0.0f32; 2]; 1]; 1]; 2];
        unsafe {
            cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut _,
                allocation_out.t_cuda.cu_device_ptr,
                tensor_out.size(),
            )
        }
        .result()
        .unwrap();

        assert!(out[0][0][0][0].is_nan());
        assert!((out[0][0][0][1] - 2.0).abs() < f32::EPSILON);
        assert!(out[1][0][0][0].abs() < f32::EPSILON);
        assert!(out[1][0][0][1].abs() < f32::EPSILON);
    }

    #[test]
    fn test_relu_activation_f64_inplace() {
        let cudnn_handle = CudnnHandle::create().unwrap();
        let cuda = CudaDeviceBuilder::new(0).build().unwrap();
        let allocation_in = cuda.alloc_zeros().unwrap();
        let mut tensor_in = Tensor2D::create(allocation_in.clone(), [[[[f64::NAN, 2.0]]], [[[
            -1.0, 0.0,
        ]]]])
        .unwrap();
        let activation = Activation::<Relu>::create().unwrap();
        activation
            .forward_inplace(cudnn_handle, &mut tensor_in)
            .unwrap();
        let mut out = [[[[0.0f64; 2]; 1]; 1]; 2];
        unsafe {
            cuMemcpyDtoH_v2(
                out.as_mut_ptr() as *mut _,
                allocation_in.t_cuda.cu_device_ptr,
                tensor_in.size(),
            )
        }
        .result()
        .unwrap();

        assert!(out[0][0][0][0].is_nan());
        assert!((out[0][0][0][1] - 2.0).abs() < f64::EPSILON);
        assert!(out[1][0][0][0].abs() < f64::EPSILON);
        assert!(out[1][0][0][1].abs() < f64::EPSILON);
    }
}
