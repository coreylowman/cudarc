use core::{marker::PhantomData, mem::MaybeUninit};

use crate::device::CudaSlice;

use super::{
    sys::{
        cudnnCreateTensorDescriptor, cudnnDestroyTensorDescriptor, cudnnSetTensorNdDescriptorEx,
        cudnnTensorDescriptor_t, cudnnTensorFormat_t,
    },
    CudnnResult, TensorDataType,
};

/// `N` must be greater than 4 and smaller than CUDNN_DIM_MAX
pub struct TensorNdDescriptor<const N: usize, T: TensorDataType> {
    descriptor: TensorDescriptor,
    data_type: PhantomData<T>,
}
impl<const N: usize, T: TensorDataType> TensorNdDescriptor<N, T> {
    pub fn new(dimensions: [i32; N]) -> CudnnResult<Self> {
        let descriptor = TensorDescriptor::new()?;
        unsafe {
            cudnnSetTensorNdDescriptorEx(
                descriptor.0,
                cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                T::get_data_type(),
                N as i32,
                dimensions.as_ptr(),
            )
            .result()?;
        }
        Ok(Self {
            descriptor,
            data_type: PhantomData,
        })
    }

    pub fn descriptor(&self) -> cudnnTensorDescriptor_t {
        self.descriptor.0
    }
}
pub struct TensorDescriptor(cudnnTensorDescriptor_t);
impl TensorDescriptor {
    pub fn new() -> CudnnResult<Self> {
        let mut descriptor = MaybeUninit::uninit();
        unsafe {
            cudnnCreateTensorDescriptor(descriptor.as_mut_ptr()).result()?;
            Ok(Self(descriptor.assume_init()))
        }
    }
}
impl Drop for TensorDescriptor {
    fn drop(&mut self) {
        unsafe { cudnnDestroyTensorDescriptor(self.0).result().unwrap() }
    }
}

pub struct TensorNd<const N: usize, T: TensorDataType> {
    descriptor: TensorNdDescriptor<N, T>,
    data: CudaSlice<T>,
}
impl<const N: usize, T: TensorDataType> TensorNd<N, T> {
    pub fn new(dimensions: [i32; N], data: CudaSlice<T>) -> CudnnResult<Self> {
        assert!(N >= 4, "`N` must be at least 4");
        assert_eq!(dimensions.iter().product::<i32>(), data.len() as i32, "The size of the CudaSlice must match the total dimension size.");
        let descriptor = TensorNdDescriptor::new(dimensions)?;
        Ok(Self { descriptor, data })
    }

    pub fn descriptor(&self) -> cudnnTensorDescriptor_t {
        self.descriptor.descriptor()
    }

    pub fn data(&self) -> &CudaSlice<T> {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use crate::device::CudaDeviceBuilder;

    use super::TensorNd;

    #[test]
    fn create_and_drop() {
        let _tensor = TensorNd::<4, f32>::new(
            [1, 2, 3, 4],
            CudaDeviceBuilder::new(0)
                .build()
                .unwrap()
                .alloc_zeros_async(1 * 2 * 3 * 4)
                .unwrap(),
        );
    }

    #[test]
    #[should_panic]
    fn not_enough_dimensions() {
        let _tensor = TensorNd::<3, f32>::new(
            [1, 2, 3],
            CudaDeviceBuilder::new(0)
                .build()
                .unwrap()
                .alloc_zeros_async(1 * 2 * 3)
                .unwrap(),
        );
    }

    #[test]
    #[should_panic]
    fn dimension_mismatch() {
        let _tensor = TensorNd::<4, f32>::new(
            [1, 2, 3, 4],
            CudaDeviceBuilder::new(0)
                .build()
                .unwrap()
                .alloc_zeros_async(1 * 2 * 3)
                .unwrap(),
        );
    }
}
