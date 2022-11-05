use core::mem::size_of;
use core::ops::{Deref, DerefMut};
use core::ptr::null_mut;

use super::sys::*;

use crate::prelude::CudaRc;

/// Wrapper around [sys::cublasStatus_t].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct CublasError(pub cublasStatus_t);

pub type CublasResult<T> = Result<T, CublasError>;
impl cublasStatus_t {
    /// Transforms into a [Result] of [CublasError]
    pub fn result(self) -> CublasResult<()> {
        match self {
            cublasStatus_t::CUBLAS_STATUS_SUCCESS => Ok(()),
            _ => Err(CublasError(self)),
        }
    }
}

#[cfg(feature = "std")]
impl std::fmt::Display for CublasError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

#[cfg(feature = "std")]
impl std::error::Error for CublasError {}

pub trait CublasTensor<T>: Sized + Deref<Target=CudaRc<T>> + DerefMut {
    type Value;

    fn new(allocation: CudaRc<Self::Value>, value: &Self::Value) -> CublasResult<Self> {
        let mut s = unsafe { Self::uninit(allocation) };
        s.set(value)?;
        Ok(s)
    }

    /// Creates [CublasTensor] of a [CudaRc].
    ///
    /// # Safety
    /// This allocation must be have been initialized or has
    /// to be initialized with [CublasTensor::set] before using it.
    unsafe fn uninit(allocation: CudaRc<Self::Value>) -> Self;
    fn set(&mut self, value: &Self::Value) -> CublasResult<()>;
    fn get(&self, out: &mut Self::Value) -> CublasResult<()>;
    fn get_device_pointer(&self) -> *const std::ffi::c_void {
        self.deref().t_cuda.cu_device_ptr as *const _
    }
    fn get_device_pointer_mut(&mut self) -> *mut std::ffi::c_void {
        self.deref_mut().t_cuda.cu_device_ptr as *mut _
    }
}
impl<T, const S: usize> Deref for CublasVector<T, S> {
    type Target = CudaRc<[T; S]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T, const S: usize> DerefMut for CublasVector<T, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub struct CublasVector<T, const S: usize>(CudaRc<[T; S]>);
impl<T, const S: usize> CublasTensor<[T; S]> for CublasVector<T, S> {
    type Value = [T; S];

    unsafe fn uninit(allocation: CudaRc<Self::Value>) -> Self {
        Self(allocation)
    }

    fn set(&mut self, value: &Self::Value) -> CublasResult<()> {
        unsafe {
            cublasSetVector(
                S as _,
                size_of::<T>() as _,
                value.as_ptr() as *const _,
                1,
                self.get_device_pointer_mut(),
                1,
            )
        }
        .result()
    }

    fn get(&self, out: &mut Self::Value) -> CublasResult<()> {
        unsafe {
            cublasGetVector(
                S as _,
                size_of::<T>() as _,
                self.get_device_pointer(),
                1,
                out.as_mut_ptr() as *mut _,
                1,
            )
        }
        .result()
    }
}
pub trait Gemv<T, M, const S: usize, const C: usize>: Sized {
    type InputVector;

    fn gemv(
        &mut self,
        cublas_handle: &CublasHandle,
        matrix: &M,
        vector: &Self::InputVector,
        add_to_output: bool,
    ) -> CublasResult<()>;
}
macro_rules! impl_gemv {
    (
        $type:ty,
        $struct:ident,
        $op:expr,
        $row:ident,
        $col:ident,
        $out:ident,
        $in:ident,
        $zero:literal,
        $one:literal,
        $cublas_fn:ident
    ) => {
        impl<const $row: usize, const $col: usize> Gemv<$type, $struct<$type, $row, $col>, $row, $col>
            for CublasVector<$type, $out>
        {
            type InputVector = CublasVector<$type, $in>;

            fn gemv(
                &mut self,
                cublas_handle: &CublasHandle,
                matrix: &$struct<$type, $row, $col>,
                vector: &Self::InputVector,
                add_to_output: bool,
            ) -> CublasResult<()> {
                unsafe {
                    $cublas_fn(
                        cublas_handle.0,
                        $op,
                        $row as _,
                        $col as _,
                        &$one as *const _,
                        matrix.get_device_pointer() as *const _,
                        $row as _,
                        vector.get_device_pointer() as *const _,
                        1,
                        &(if add_to_output { $one } else { $zero }) as *const _,
                        self.get_device_pointer_mut() as *mut _,
                        1,
                    )
                }
                .result()
            }
        }
    };
    ($type:ty : $zero:literal, $one:literal, $cublas_fn:ident) => {
        impl_gemv!(
            $type,
            CublasMatrix,
            cublasOperation_t::CUBLAS_OP_N,
            R,
            C,
            R,
            C,
            $zero,
            $one,
            $cublas_fn
        );
        impl_gemv!(
            $type,
            TransposedCublasMatrix,
            cublasOperation_t::CUBLAS_OP_T,
            R,
            C,
            C,
            R,
            $zero,
            $one,
            $cublas_fn
        );
    };
}
impl_gemv!(f32: 0.0f32, 1.0f32, cublasSgemv_v2);
impl_gemv!(f64: 0.0f64, 1.0f64, cublasDgemv_v2);

type TransposedCublasMatrix<T, const R: usize, const C: usize> = Transposed<CublasMatrix<T, R, C>>;
impl<T, const R: usize, const C: usize> Transposed<CublasMatrix<T, R, C>> {
    pub(crate) fn get_device_pointer(&self) -> *const std::ffi::c_void {
        self.0.get_device_pointer()
    }
    pub(crate) fn get_device_pointer_mut(&mut self) -> *mut std::ffi::c_void {
        self.0.get_device_pointer_mut()
    }
}
        
/// A cublas Matrix with `R` rows and `C` columns in COLUMN-MAJOR!!! format.
pub struct CublasMatrix<T, const R: usize, const C: usize>(CudaRc<[[T; R]; C]>);
impl<T, const R: usize, const C: usize> Deref for CublasMatrix<T, R, C> {
    type Target = CudaRc<[[T; R]; C]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T, const R: usize, const C: usize> DerefMut for CublasMatrix<T, R, C> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

pub struct Transposed<T>(T);
pub trait TransposeCublasMatrix {
    type Transposed;

    fn transposed(self) -> Self::Transposed;
}
impl<T, const R: usize, const C: usize> TransposeCublasMatrix for CublasMatrix<T, R, C> {
    type Transposed = Transposed<Self>;

    fn transposed(self) -> Self::Transposed {
        Transposed(self)
    }
}
impl<T, const R: usize, const C: usize> TransposeCublasMatrix for Transposed<CublasMatrix<T, R, C>> {
    type Transposed = CublasMatrix<T, R, C>;

    fn transposed(self) -> Self::Transposed {
        self.0
    }
}
impl<T, const R: usize, const C: usize> CublasTensor<[[T; R]; C]> for CublasMatrix<T, R, C> {
    type Value = [[T; R]; C];

    unsafe fn uninit(allocation: CudaRc<Self::Value>) -> Self {
        Self(allocation)
    }

    fn set(&mut self, matrix: &Self::Value) -> CublasResult<()> {
        unsafe {
            cublasSetMatrix(
                R as _,
                C as _,
                size_of::<T>() as _,
                matrix.as_ptr() as *const _,
                R as _,
                self.get_device_pointer_mut(),
                R as _,
            )
        }
        .result()
    }

    fn get(&self, out: &mut Self::Value) -> CublasResult<()> {
        unsafe {
            cublasGetMatrix(
                R as _,
                C as _,
                size_of::<T>() as _,
                self.get_device_pointer(),
                R as _,
                out.as_mut_ptr() as *mut _,
                R as _,
            )
        }
        .result()
    }
}

pub struct CublasHandle(cublasHandle_t);
impl Drop for CublasHandle {
    fn drop(&mut self) {
        unsafe {
            cublasDestroy_v2(self.0);
        }
    }
}

impl CublasHandle {
    pub fn create() -> CublasResult<Self> {
        let mut handle = Self(null_mut());
        unsafe { cublasCreate_v2(&mut handle.0 as *mut _) }.result()?;
        Ok(handle)
    }
}

#[cfg(test)]
mod tests {
    use crate::prelude::CudaDeviceBuilder;

    use super::*;

    #[test]
    fn test_create_vector() {
        let h_vector = [1.0f32, 2.0, 3.0];
        let mut h_out = [0.0; 3];
        unsafe {
            let device = CudaDeviceBuilder::new(0).build().unwrap();
            let ptr = device.alloc().unwrap();
            let d_vector = CublasVector::new(ptr, &h_vector).unwrap();
            d_vector.get(&mut h_out).unwrap();
        }
        assert_eq!(h_vector, h_out);
    }

    #[test]
    fn test_create_matrix() {
        let h_matrix = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let mut h_out = [[0.0; 3]; 2];
        unsafe {
            let device = CudaDeviceBuilder::new(0).build().unwrap();
            let ptr = device.alloc().unwrap();
            let d_matrix = CublasMatrix::new(ptr, &h_matrix).unwrap();
            d_matrix.get(&mut h_out).unwrap();
        }
        assert_eq!(h_matrix, h_out);
    }

    #[test]
    fn test_sgemv() {
        let h_vector = [1.0, 2.0, 3.0];
        let h_matrix = [[1.0, 2.0], [0.5, 1.0], [1.0 / 3.0, 1.0]];
        let h_expected = [3.0, 7.0];

        // type-hinting `f32` in `h_vector` causes inference bugs
        let mut h_out = [0.0f32; 2];
        unsafe {
            let device = CudaDeviceBuilder::new(0).build().unwrap();
            let cublas = CublasHandle::create().unwrap();

            let ptr_m = device.alloc().unwrap();
            let d_matrix = CublasMatrix::new(ptr_m, &h_matrix).unwrap();

            let ptr_v = device.alloc().unwrap();
            let d_vector = CublasVector::new(ptr_v, &h_vector).unwrap();

            let ptr_o = device.alloc().unwrap();
            let mut d_out = CublasVector::uninit(ptr_o);
            d_out.gemv(&cublas, &d_matrix, &d_vector, false).unwrap();

            d_out.get(&mut h_out).unwrap();
        }
        assert_eq!(h_out, h_expected);
    }

    #[test]
    fn test_dgemv_transposed() {
        let h_vector = [1.0, 2.0, 3.0];
        let h_matrix = [[1.0, 0.5, 1.0 / 3.0], [2.0, 1.0, 1.0]];
        let h_expected = [3.0, 7.0];

        let mut h_out = [0.0; 2];
        unsafe {
            let device = CudaDeviceBuilder::new(0).build().unwrap();
            let cublas = CublasHandle::create().unwrap();

            let ptr_m = device.alloc().unwrap();
            let d_matrix = CublasMatrix::new(ptr_m, &h_matrix).unwrap();

            let ptr_v = device.alloc().unwrap();
            let d_vector = CublasVector::new(ptr_v, &h_vector).unwrap();

            let ptr_o = device.alloc().unwrap();
            let mut d_out = CublasVector::uninit(ptr_o);
            d_out
                .gemv(&cublas, &d_matrix.transposed(), &d_vector, false)
                .unwrap();

            d_out.get(&mut h_out).unwrap();
        }
        assert_eq!(h_out, h_expected);
    }
}
