use core::mem::size_of;
use core::ops::{Deref, DerefMut};

use crate::prelude::*;

use super::sys::*;

/// The base trait for CublasTensors. Currently, the following are implemented:
/// - [CublasVector]: a tensor of rank 1
/// - [CublasMatrix]: a tensor of rank 2
pub trait CublasTensor: Sized + Deref<Target = CudaRc<Self::Value>> + DerefMut {
    /// This type represents the type the host has for this [CublasTensor].
    type Value;

    /// Creates a new tensor by the given `allocation` and `value`.
    fn new(allocation: CudaRc<Self::Value>, value: &Self::Value) -> CublasResult<Self> {
        let mut s = unsafe { Self::uninit(allocation) };
        s.copy_from(value)?;
        Ok(s)
    }

    /// Creates [CublasTensor] of a [CudaRc].
    ///
    /// # Safety
    /// This allocation must be have been initialized or has
    /// to be initialized with [CublasTensor::set] before using it.
    unsafe fn uninit(allocation: CudaRc<Self::Value>) -> Self;

    /// Copies the data `value` from the host to the device.
    fn copy_from(&mut self, value: &Self::Value) -> CublasResult<()>;
    /// Copies the data from the [CublasTensor] to `out`.
    fn copy_to(&self, out: &mut Self::Value) -> CublasResult<()>;

    /// Returns a pointer to the first element of the [CublasTensor] on the
    /// device.
    fn get_device_pointer(&self) -> *const std::ffi::c_void {
        self.deref().t_cuda.cu_device_ptr as *const _
    }
    /// Returns a mutable pointer to the first element of the [CublasTensor] on
    /// the device.
    fn get_device_pointer_mut(&mut self) -> *mut std::ffi::c_void {
        self.deref_mut().t_cuda.cu_device_ptr as *mut _
    }
}

/// A cublas Vector with `S` elements of type `T`.
pub struct CublasVector<T, const S: usize>(CudaRc<[T; S]>);

/// A cublas Matrix with `R` rows and `C` columns in row-major format.
/// Unlike the Matrix in Cuda itself (<https://docs.nvidia.com/cuda/cublas/index.html#data-layout>),
/// this is actually row-major by always "applying" the transpose operation
/// (<https://docs.nvidia.com/cuda/cublas/index.html#cublasoperation_t>) instead
/// of the non-transpose one. To use a [CublasMatrix] with the transpose
/// operation, use [CublasMatrixTransposed] instead.
pub struct CublasMatrix<T, const R: usize, const C: usize>(CudaRc<[[T; C]; R]>);
/// The transposed version of [CublasMatrix]. This still uses row-major format,
/// but all the operations on this matrix will NOT use the transpose operation.
///
/// Read more on [CublasMatrix]
pub struct CublasMatrixTransposed<T, const R: usize, const C: usize>(CublasMatrix<T, R, C>);
impl<T, const R: usize, const C: usize> CublasMatrix<T, R, C> {
    /// Transposes the [CublasMatrix] to a [CublasMatrixTransposed].
    /// This does nothing besides changing the type.
    ///
    /// Read more on [CublasMatrix]
    pub fn transposed(self) -> CublasMatrixTransposed<T, R, C> {
        CublasMatrixTransposed(self)
    }
}
impl<T, const R: usize, const C: usize> CublasMatrixTransposed<T, R, C> {
    /// Transposes the [CublasMatrixTransposed] to a [CublasMatrix].
    /// This does nothing besides changing the type.
    ///
    /// Read more on [CublasMatrix]
    pub fn transposed(self) -> CublasMatrix<T, R, C> {
        self.0
    }
}
impl<T, const R: usize, const C: usize> CublasMatrixTransposed<T, R, C> {
    /// Returns a pointer to the first element of a [CublasMatrixTransposed] on
    /// the device.
    pub fn get_device_pointer(&self) -> *const std::ffi::c_void {
        self.0.get_device_pointer()
    }

    /// Returns a mutable pointer to the first element of a
    /// [CublasMatrixTransposed] on the device.
    pub fn get_device_pointer_mut(&mut self) -> *mut std::ffi::c_void {
        self.0.get_device_pointer_mut()
    }
}

/// Implements [CublasTensor] for [CublasVector] and [CublasMatrix] with as few
/// repetitions as possible.
macro_rules! impl_tensor {
    (@impl_get_set: $fn:ident, ($($const:tt),+), $from:expr, $to:expr, $stride:tt) => {
        unsafe {
            $fn(
                $($const as _),+,
                size_of::<T>() as _,
                // not casting here to prevent accidently wrong target casts
                $from,
                $stride as _,
                $to,
                $stride as _,
            )
        }.result()
    };

    ($target:ident<$($const:tt),+>, $type:ty, $set:ident, $get:ident, $stride:tt) => {
        impl<T, $(const $const: usize),+> Deref for $target<T, $($const),+> {
            type Target = CudaRc<$type>;

            fn deref(&self) -> &Self::Target {
                &self.0
            }
        }
        impl<T, $(const $const: usize),+> DerefMut for $target<T, $($const),+> {
            fn deref_mut(&mut self) -> &mut Self::Target {
                &mut self.0
            }
        }

        impl<T, $(const $const: usize),+> CublasTensor for $target<T, $($const),+> {
            type Value = $type;

            unsafe fn uninit(allocation: CudaRc<Self::Value>) -> Self {
                Self(allocation)
            }

            fn copy_from(&mut self, value: &Self::Value) -> CublasResult<()> {
                impl_tensor!(@impl_get_set: $set, ($($const),+), value.as_ptr() as *const _, self.get_device_pointer_mut(), $stride)
            }

            fn copy_to(&self, out: &mut Self::Value) -> CublasResult<()> {
                impl_tensor!(@impl_get_set: $get, ($($const),+), self.get_device_pointer(), out.as_mut_ptr() as *mut _, $stride)
            }
        }
    };
}
impl_tensor!(CublasVector<S>, [T; S], cublasSetVector, cublasGetVector, 1);
impl_tensor!(CublasMatrix<R, C>, [[T; C]; R], cublasSetMatrix, cublasGetMatrix, R);

#[cfg(test)]
mod tests {
    use core::any::Any;

    use crate::prelude::*;

    #[test]
    fn test_create_vector() {
        let h_vector = [1.0f32, 2.0, 3.0];
        let mut h_out = [0.0; 3];
        unsafe {
            let device = CudaDeviceBuilder::new(0).build().unwrap();
            let ptr = device.alloc().unwrap();
            let d_vector = CublasVector::new(ptr, &h_vector).unwrap();
            d_vector.copy_to(&mut h_out).unwrap();
        }
        assert_eq!(h_vector, h_out);
    }

    #[test]
    fn test_create_matrix() {
        let h_matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut h_out = [[0.0; 3]; 2];
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let ptr = unsafe { device.alloc() }.unwrap();
        let d_matrix = CublasMatrix::new(ptr, &h_matrix).unwrap();
        d_matrix.copy_to(&mut h_out).unwrap();
        assert_eq!(h_matrix, h_out);
    }

    #[test]
    fn test_double_transpose_matrix() {
        let h_matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let device = CudaDeviceBuilder::new(0).build().unwrap();
        let ptr1 = unsafe { device.alloc() }.unwrap();
        let ptr2 = unsafe { device.alloc() }.unwrap();
        let d_matrix1 = CublasMatrix::new(ptr1, &h_matrix).unwrap();
        let d_matrix2 = CublasMatrix::new(ptr2, &h_matrix).unwrap();
        assert_eq!(
            d_matrix1.type_id(),
            d_matrix2.transposed().transposed().type_id()
        );
    }
}
